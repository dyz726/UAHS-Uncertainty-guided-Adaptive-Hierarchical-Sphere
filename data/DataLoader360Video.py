import random
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


def _sample_to_sphere(sphere_grid_tensor, rgb, saliency, fixation):
    """将 ERP 格式的 RGB / 显著图 / 注视图采样到球面节点上。"""
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().unsqueeze(0)
    rgb_sampled = F.grid_sample(
        rgb_tensor, sphere_grid_tensor, padding_mode="border", align_corners=False
    ).squeeze(0).squeeze(-1).transpose(0, 1).numpy()

    def sample_grayscale(image):
        tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).unsqueeze(0).float()
        return F.grid_sample(
            tensor, sphere_grid_tensor, padding_mode="border", align_corners=False
        ).reshape(-1).numpy().astype(np.float32)

    return rgb_sampled, sample_grayscale(saliency), sample_grayscale(fixation)


class SaliencyDataset(Dataset):
    """Load 360-degree video clips and project ERP data onto an icosphere."""

    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")

    def __init__(
        self,
        dataname,
        root_dir,
        video_id,
        dataset_kwargs,
        augmentation_kwargs=None,
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

        self.norm_mean = torch.tensor([0.485, 0.456, 0.406])
        self.norm_std = torch.tensor([0.229, 0.224, 0.225])
        self.augmentation_kwargs = augmentation_kwargs or {}

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
            frame_numbers = sorted(set(saliency_files) & set(fixation_files))

            capture = cv2.VideoCapture(str(video_path))
            total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            zero_based_numbers = list(range(total_video_frames))
            one_based_numbers = list(range(1, total_video_frames + 1))
            if frame_numbers == zero_based_numbers:
                label_start_index = 0
            elif frame_numbers == one_based_numbers:
                label_start_index = 1
            else:
                print(
                    f"Warning: skipping {video_id}; video/annotation frames are not aligned "
                    f"(video={total_video_frames}, maps={len(saliency_files)}, "
                    f"fixation={len(fixation_files)})"
                )
                continue

            for start in range(0, total_video_frames, self.seq_length):
                valid_length = min(self.seq_length, total_video_frames - start)
                if valid_length < self.seq_length and not self.include_partial:
                    break
                frame_indices = list(range(start, start + valid_length))
                if valid_length < self.seq_length:
                    frame_indices.extend([frame_indices[-1]] * (self.seq_length - valid_length))
                label_numbers = [index + label_start_index for index in frame_indices]
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
        return data_list

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def _iter_video_frames(video_path, frame_indices):
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

    def __init__(self, dataname, root_dir, video_id, dataset_kwargs, augmentation_kwargs=None):
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

        augmentation_kwargs = augmentation_kwargs or {}
        self.color_augmentation = augmentation_kwargs.get("color_augmentation", False)
        self.lr_flip_augmentation = augmentation_kwargs.get("lr_flip_augmentation", False)
        self.yaw_rotation_augmentation = augmentation_kwargs.get("yaw_rotation_augmentation", False)

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

                                     
        yaw_rotat = self.yaw_rotation_augmentation and random.random() > 0.5
        roll_ratio = random.randint(0, 100) / 100.0
        lr_flip = self.lr_flip_augmentation and random.random() > 0.5

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

            if yaw_rotat:
                shift = int(rgb.shape[1] * roll_ratio)
                rgb = np.roll(rgb, shift, axis=1)
                sal = np.roll(sal, shift, axis=1)
                fix = np.roll(fix, shift, axis=1)
            if lr_flip:
                rgb = np.fliplr(rgb)
                sal = np.fliplr(sal)
                fix = np.fliplr(fix)

            sphere_rgb, sphere_sal, sphere_fix = _sample_to_sphere(
                self.sphere_grid_tensor, rgb, sal, fix
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
