import hashlib
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from trimesh_utils import IcoSphereRef, asSpherical


def _build_sphere_grid(sphere_node_type, sphere_rank):
    """构建二十面体球面采样网格，返回形状为 [1, L, 1, 2] 的 grid_sample 采样网格。"""
    icosphere_ref = IcoSphereRef(sphere_node_type)
    normals = icosphere_ref.get_normals(rank=sphere_rank)
    normals_rphitheta = asSpherical(normals)
    normals_wh = np.stack(
        (
            normals_rphitheta[:, 2] / 180,
            normals_rphitheta[:, 1] / 180 * 2 - 1,
        ),
        axis=1,
    ).astype(np.float32)
    return torch.from_numpy(normals_wh).reshape(1, -1, 1, 2)


def _sample_erp_periodic(tensor, sphere_grid_tensor, mode="bilinear"):
    """Sample ERP data with circular longitude and clamped latitude."""
    width = tensor.shape[-1]
    padded = F.pad(tensor, (1, 1, 0, 0), mode="circular")
    periodic_grid = sphere_grid_tensor.clone()
    periodic_grid[..., 0] *= width / (width + 2.0)
    return F.grid_sample(
        padded,
        periodic_grid,
        mode=mode,
        padding_mode="border",
        align_corners=False,
    )


def _sample_to_sphere(sphere_grid_tensor, rgb, saliency, fixation):
    """将 ERP 格式的 RGB / 显著图 / 注视图采样到球面节点上。"""
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().unsqueeze(0)
    rgb_sampled = _sample_erp_periodic(
        rgb_tensor, sphere_grid_tensor
    ).squeeze(0).squeeze(-1).transpose(0, 1).numpy()

    def sample_grayscale(image):
        tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).unsqueeze(0).float()
        return _sample_erp_periodic(
            tensor, sphere_grid_tensor
        ).reshape(-1).numpy().astype(np.float32)

    fixation_tensor = (
        torch.from_numpy(np.ascontiguousarray(fixation))
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
    )
    sphere_fixation = _sample_erp_periodic(
        fixation_tensor,
        sphere_grid_tensor,
        mode="nearest",
    ).reshape(-1).numpy().astype(np.float32)
    return rgb_sampled, sample_grayscale(saliency), sphere_fixation


class SaliencyDataset(Dataset):
    """Load 360-degree video clips and project ERP data onto an icosphere."""

    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")

    def __init__(
        self,
        dataname,
        root_dir,
        video_id,
        dataset_kwargs,
        data_type="train",
        include_partial=False,
    ):
        self.dataname = dataname
        self.root_dir = Path(root_dir)
        self.video_ids = [str(item) for item in video_id]
        self.seq_length = dataset_kwargs["seq_length"]
        self.sphere_rank = dataset_kwargs["sphere_rank"]
        self.sphere_node_type = dataset_kwargs["sphere_node_type"]
        self.data_type = data_type
        self.include_partial = include_partial

        if self.seq_length <= 0:
            raise ValueError("seq_length must be positive")
        if data_type not in {"train", "test"}:
            raise ValueError(f"Unsupported data_type: {data_type}")

        self.sphere_grid_tensor = _build_sphere_grid(self.sphere_node_type, self.sphere_rank)
        self.video_frame_pts = {}

        self.norm_mean = torch.tensor([0.485, 0.456, 0.406])
        self.norm_std = torch.tensor([0.229, 0.224, 0.225])

        if self.dataname in {"SVGC_AVA", "Sports-360", "VR-EyeTracking"}:
            # These datasets keep all videos and annotations in shared directories;
            # train/test lists determine the logical split.
            self.video_base_dir = self.root_dir / "videos"
            self.annotation_base_dir = self.root_dir
        else:
            split_name = "training" if data_type == "train" else "testing"
            self.video_base_dir = self.root_dir / "videos" / data_type
            self.annotation_base_dir = self.root_dir / split_name
        self.data_list = self._build_video_data_list()

        if not self.data_list:
            raise RuntimeError(
                f"No valid samples found for {self.dataname} under {self.root_dir} "
                f"(split={self.data_type})"
            )
        print(
            f"Loaded {len(self.data_list)} {self.data_type} clips from "
            f"{len(self.video_ids)} videos (sphere={self.sphere_node_type}:{self.sphere_rank})"
        )

    def _find_video_file(self, video_id):
        exact_candidate = self.video_base_dir / video_id
        if (
            exact_candidate.is_file()
            and exact_candidate.suffix.lower() in self.VIDEO_EXTENSIONS
        ):
            return exact_candidate
        for extension in self.VIDEO_EXTENSIONS:
            candidate = self.video_base_dir / f"{video_id}{extension}"
            if candidate.is_file():
                return candidate
            candidate = self.video_base_dir / f"{video_id}{extension.upper()}"
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _indexed_annotations(directory):
        files = {}
        for path in Path(directory).iterdir():
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            try:
                frame_number = int(path.stem)
            except ValueError:
                continue
            files[frame_number] = path
        return files

    def _build_video_data_list(self):
        data_list = []
        for video_id in self.video_ids:
            video_path = self._find_video_file(video_id)
            if self.dataname in {"SVGC_AVA", "Sports-360", "VR-EyeTracking"}:
                annotation_dir = self.annotation_base_dir
                maps_dir = annotation_dir / "maps" / video_id
                fixation_dir = annotation_dir / "fixation" / video_id
            else:
                annotation_dir = self.annotation_base_dir / video_id
                maps_dir = annotation_dir / "maps"
                fixation_dir = annotation_dir / "fixation"
            if video_path is None:
                print(f"Warning: video not found for {video_id} in {self.video_base_dir}")
                continue
            if not maps_dir.is_dir() or not fixation_dir.is_dir():
                print(f"Warning: annotations not found for {video_id} in {annotation_dir}")
                continue

            saliency_files = self._indexed_annotations(maps_dir)
            fixation_files = self._indexed_annotations(fixation_dir)
            paired_frame_numbers = set(saliency_files) & set(fixation_files)
            if not paired_frame_numbers:
                print(
                    f"Warning: skipping {video_id}; no paired map/fixation annotations found "
                    f"(maps={len(saliency_files)}, fixation={len(fixation_files)})"
                )
                continue

            capture = cv2.VideoCapture(str(video_path))
            total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if self.dataname == "VR-EyeTracking":
                # OpenCV's frame-number seek can land on another decoded frame
                # when presentation timestamps are reordered (e.g. 102.mp4).
                # Record timestamps in sequential decode order so every clip
                # can verify its first frame after seeking.
                frame_pts = self._indexed_video_pts(video_path)
                self.video_frame_pts[str(video_path)] = frame_pts
                total_video_frames = len(frame_pts)

            # VR-EyeTracking annotations name the video frame by its zero-based
            # index. A missing 0000.png must not shift every later match by one.
            if self.dataname == "VR-EyeTracking" or 0 in paired_frame_numbers:
                label_start_index = 0
            elif total_video_frames in paired_frame_numbers:
                label_start_index = 1
            else:
                print(
                    f"Warning: cannot confirm annotation index base for {video_id}; "
                    "assuming 1-based labels"
                )
                label_start_index = 1

            missing_clips = 0
            for start in range(0, total_video_frames, self.seq_length):
                valid_length = min(self.seq_length, total_video_frames - start)
                if valid_length < self.seq_length and not self.include_partial:
                    break
                frame_indices = list(range(start, start + valid_length))
                if valid_length < self.seq_length:
                    frame_indices.extend([frame_indices[-1]] * (self.seq_length - valid_length))
                label_numbers = [index + label_start_index for index in frame_indices]
                if not all(
                    number in saliency_files and number in fixation_files
                    for number in label_numbers
                ):
                    missing_clips += 1
                    continue
                data_list.append(
                    {
                        "video_path": str(video_path),
                        "frame_indices": frame_indices,
                        "sal_seq": [str(saliency_files[number]) for number in label_numbers],
                        "fix_seq": [str(fixation_files[number]) for number in label_numbers],
                        "videoID": video_id,
                        "valid_length": valid_length,
                    }
                )
            if missing_clips:
                print(
                    f"Warning: skipped {missing_clips} clips from {video_id} "
                    "because map or fixation frames are missing"
                )
        return data_list

    @staticmethod
    def _indexed_video_pts(video_path):
        video_stat = video_path.stat()
        cache_key = hashlib.sha256(
            f"{video_path.resolve()}:{video_stat.st_size}:"
            f"{video_stat.st_mtime_ns}:{cv2.__version__}".encode()
        ).hexdigest()
        cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        cache_dir = cache_root / "uahs" / "vr_frame_pts"
        cache_path = cache_dir / f"{cache_key}.npy"
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if cached.ndim == 1 and cached.dtype == np.int64 and len(cached):
                return cached
        except (OSError, ValueError, EOFError):
            pass

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        timestamps = []
        try:
            while capture.grab():
                milliseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
                if not np.isfinite(milliseconds):
                    raise RuntimeError(f"Invalid frame timestamp in {video_path}")
                timestamps.append(round(milliseconds * 1000))
        finally:
            capture.release()
        if not timestamps:
            raise RuntimeError(f"No decodable frames in {video_path}")
        frame_pts = np.asarray(timestamps, dtype=np.int64)

        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npy", delete=False) as tmp:
                temporary_path = Path(tmp.name)
                np.save(tmp, frame_pts)
            os.replace(temporary_path, cache_path)
        except OSError as exc:
            print(f"Warning: could not cache VR frame index for {video_path}: {exc}")
            if "temporary_path" in locals():
                temporary_path.unlink(missing_ok=True)
        return frame_pts

    def __len__(self):
        return len(self.data_list)

    def iter_expected_frames(self):
        """Yield only frames in valid clips, retaining their source label paths."""
        for item in self.data_list:
            for sal_path, fix_path in zip(
                item["sal_seq"][:item["valid_length"]],
                item["fix_seq"][:item["valid_length"]],
            ):
                yield (
                    item["videoID"],
                    Path(sal_path).stem,
                    Path(sal_path),
                    Path(fix_path),
                )

    def _iter_video_frames(self, video_path, frame_indices):
        if self.dataname == "VR-EyeTracking":
            yield from self._iter_vr_video_frames(video_path, frame_indices)
            return

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
        previous_index = frame_indices[0] - 1
        previous_frame = None
        try:
            for frame_index in frame_indices:
                if frame_index == previous_index:
                    yield previous_frame
                    continue
                if frame_index != previous_index + 1:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to decode frame {frame_index} from {video_path}")
                previous_index = frame_index
                previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield previous_frame
        finally:
            capture.release()

    def _iter_vr_video_frames(self, video_path, frame_indices):
        target = frame_indices[0]
        frame_pts = self.video_frame_pts[video_path]
        expected_pts = frame_pts[target]
        seek_start = max(0, target - 16)
        # Duplicate timestamps cannot identify a frame uniquely; decode from
        # the beginning in that rare case instead of risking a silent shift.
        needs_sequential = np.count_nonzero(frame_pts == expected_pts) != 1
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        try:
            first_frame = None
            if seek_start and not needs_sequential:
                capture.set(cv2.CAP_PROP_POS_FRAMES, seek_start)
                for _ in range(target - seek_start + 64):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    actual_pts = round(capture.get(cv2.CAP_PROP_POS_MSEC) * 1000)
                    if actual_pts == expected_pts:
                        first_frame = frame
                        break
            using_sequential = first_frame is None
            if using_sequential:
                capture.release()
                capture = cv2.VideoCapture(video_path)
                if not capture.isOpened():
                    raise RuntimeError(f"Failed to open video: {video_path}")
                for index in range(target + 1):
                    ok, first_frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Failed to decode frame {index} from {video_path}")

            previous_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            yield previous_frame
            previous_index = target
            for frame_index in frame_indices[1:]:
                if frame_index == previous_index:
                    yield previous_frame
                    continue
                if frame_index != previous_index + 1:
                    raise RuntimeError(f"Non-consecutive VR clip in {video_path}")
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to decode frame {frame_index} from {video_path}")
                if (
                    not using_sequential
                    and round(capture.get(cv2.CAP_PROP_POS_MSEC) * 1000)
                    != frame_pts[frame_index]
                ):
                    # A seek may also disturb the following frames. Keep the
                    # already-verified prefix and decode the remainder from
                    # frame zero rather than silently mixing timelines.
                    capture.release()
                    capture = cv2.VideoCapture(video_path)
                    if not capture.isOpened():
                        raise RuntimeError(f"Failed to open video: {video_path}")
                    for index in range(frame_index + 1):
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError(
                                f"Failed to decode frame {index} from {video_path}"
                            )
                    using_sequential = True
                previous_index = frame_index
                previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield previous_frame
        finally:
            capture.release()

    def __getitem__(self, idx):
        item = self.data_list[idx]
        rgb_source = self._iter_video_frames(item["video_path"], item["frame_indices"])

        sphere_rgbs = []
        sphere_sals = []
        sphere_fixations = []
        sals = []
        for rgb, sal_path, fix_path in zip(rgb_source, item["sal_seq"], item["fix_seq"]):
            saliency = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
            fixation = cv2.imread(fix_path, cv2.IMREAD_GRAYSCALE)
            if saliency is None or fixation is None:
                raise RuntimeError(f"Failed to read annotations: {sal_path}, {fix_path}")
            sphere_rgb, sphere_sal, sphere_fix = self._convert_to_sphere(
                rgb, saliency, fixation
            )
            sphere_rgbs.append(sphere_rgb)
            sphere_sals.append(sphere_sal)
            sphere_fixations.append(sphere_fix)
            sals.append(saliency)

        if len(sphere_rgbs) != self.seq_length:
            raise RuntimeError(
                f"Expected {self.seq_length} frames but decoded {len(sphere_rgbs)} "
                f"from {item['video_path']}"
            )

        sphere_rgb = torch.from_numpy(np.stack(sphere_rgbs)).float().div_(255.0)
        sphere_sal = torch.from_numpy(np.stack(sphere_sals)).float()
        sphere_fix = torch.from_numpy(np.stack(sphere_fixations)).float()
        return {
            "sphere_rgb": sphere_rgb,
            "sphere_sal": sphere_sal,
            "sphere_fix": sphere_fix,
            "erp_sal": torch.from_numpy(np.stack(sals)).float().div_(255.0),
            "normalized_sphere_rgb": (sphere_rgb - self.norm_mean) / self.norm_std,
            "normalized_sphere_sal": sphere_sal / 255.0,
            "normalized_sphere_fix": sphere_fix / 255.0,
            "seq_path": item["sal_seq"],
            "videoID": item["videoID"],
            "valid_length": item["valid_length"],
        }

    def _convert_to_sphere(self, rgb, saliency, fixation):
        return _sample_to_sphere(self.sphere_grid_tensor, rgb, saliency, fixation)


class AVSSaliencyDataset(Dataset):
    """AVS-ODV 数据集加载器（帧图像布局）。

    目录结构:
        root_dir/
            frames/<video_id>/0001.jpg        # 视频帧
            maps/<video_id>/0001_e.jpg        # 显著图（文件名 = 帧名 + "_e"）
            fixation/<video_id>/0001_efix.png # 注视图（文件名 = 帧名 + "_efix"）

    每个样本为同一视频内连续的 seq_length 帧，帧与真值按文件名一一对应。
    """

    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

    def __init__(self, dataname, root_dir, video_id, dataset_kwargs):
        self.dataname = dataname
        self.root_dir = Path(root_dir)
        self.video_ids = [str(item) for item in video_id]
        self.seq_length = dataset_kwargs["seq_length"]
        self.sphere_rank = dataset_kwargs["sphere_rank"]
        self.sphere_node_type = dataset_kwargs["sphere_node_type"]

        if self.seq_length <= 0:
            raise ValueError("seq_length must be positive")

        self.sphere_grid_tensor = _build_sphere_grid(self.sphere_node_type, self.sphere_rank)

        self.norm_mean = torch.tensor([0.485, 0.456, 0.406])
        self.norm_std = torch.tensor([0.229, 0.224, 0.225])

        self.data_list = self._build_data_list()
        if not self.data_list:
            raise RuntimeError(
                f"No valid samples found for {self.dataname} under {self.root_dir}"
            )
        print(
            f"Loaded {len(self.data_list)} clips from {len(self.video_ids)} videos "
            f"(sphere={self.sphere_node_type}:{self.sphere_rank})"
        )

    def _build_data_list(self):
        data_list = []
        for video_id in self.video_ids:
            frames_dir = self.root_dir / "frames" / video_id
            maps_dir = self.root_dir / "maps" / video_id
            fixation_dir = self.root_dir / "fixation" / video_id
            if not (frames_dir.is_dir() and maps_dir.is_dir() and fixation_dir.is_dir()):
                print(f"Warning: skipping {video_id}; frames/maps/fixation not found under {self.root_dir}")
                continue

                        
            frame_files = sorted(
                p for p in frames_dir.iterdir()
                if p.suffix.lower() in self.IMAGE_EXTENSIONS
            )

                                                    
            for start in range(0, len(frame_files) - self.seq_length + 1, self.seq_length):
                rgb_seq = []
                sal_seq = []
                fix_seq = []
                for frame_path in frame_files[start:start + self.seq_length]:
                    base_name = frame_path.stem
                    sal_path = maps_dir / f"{base_name}_e.jpg"
                    fix_path = fixation_dir / f"{base_name}_efix.png"
                    if sal_path.is_file() and fix_path.is_file():
                        rgb_seq.append(str(frame_path))
                        sal_seq.append(str(sal_path))
                        fix_seq.append(str(fix_path))
                    else:
                        print(f"Warning: annotations missing for {video_id}/{base_name}, drop this clip")
                if len(rgb_seq) == self.seq_length:
                    data_list.append(
                        {
                            "rgb_seq": rgb_seq,
                            "sal_seq": sal_seq,
                            "fix_seq": fix_seq,
                            "videoID": video_id,
                        }
                    )
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        sals = []
        sphere_rgbs = []
        sphere_sals = []
        sphere_fixs = []
        for rgb_path, sal_path, fix_path in zip(item["rgb_seq"], item["sal_seq"], item["fix_seq"]):
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
            fix = cv2.imread(fix_path, cv2.IMREAD_GRAYSCALE)
            if rgb is None or sal is None or fix is None:
                raise RuntimeError(f"Corrupted data: {rgb_path}, {sal_path}, {fix_path}")

            sphere_rgb, sphere_sal, sphere_fix = _sample_to_sphere(
                self.sphere_grid_tensor,
                rgb,
                sal,
                fix,
            )
            sphere_rgbs.append(sphere_rgb)
            sphere_sals.append(sphere_sal)
            sphere_fixs.append(sphere_fix)
            sals.append(sal)

        sphere_rgb = torch.from_numpy(np.stack(sphere_rgbs)).float().div_(255.0)
        sphere_sal = torch.from_numpy(np.stack(sphere_sals)).float()
        sphere_fix = torch.from_numpy(np.stack(sphere_fixs)).float()
        return {
            "sphere_rgb": sphere_rgb,
            "sphere_sal": sphere_sal,
            "sphere_fix": sphere_fix,
            "erp_sal": torch.from_numpy(np.stack(sals)).float().div_(255.0),
            "normalized_sphere_rgb": (sphere_rgb - self.norm_mean) / self.norm_std,
            "normalized_sphere_sal": sphere_sal / 255.0,
            "normalized_sphere_fix": sphere_fix / 255.0,
            "seq_path": item["sal_seq"],
            "videoID": item["videoID"],
            "valid_length": self.seq_length,
        }
