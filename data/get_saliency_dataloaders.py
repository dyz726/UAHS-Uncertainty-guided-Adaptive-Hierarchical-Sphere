import os
import random
from glob import glob
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader

from .DataLoader360Video import AVSSaliencyDataset, SaliencyDataset


NO_AUGMENTATION = {
    "color_augmentation": False,
    "lr_flip_augmentation": False,
    "yaw_rotation_augmentation": False,
}


def _video_ids(dataset_root_dir, data_type):
    annotation_split = "training" if data_type == "train" else "testing"
    annotation_dir = os.path.join(dataset_root_dir, annotation_split)
    video_dir = os.path.join(dataset_root_dir, "videos", data_type)
    if not os.path.isdir(annotation_dir) or not os.path.isdir(video_dir):
        raise FileNotFoundError(
            f"Video dataset split is incomplete: expected {annotation_dir} and {video_dir}"
        )

    video_ids = []
    for annotation_path in glob(os.path.join(annotation_dir, "*")):
        video_id = os.path.basename(annotation_path)
        has_annotations = (
            os.path.isdir(os.path.join(annotation_path, "maps"))
            and os.path.isdir(os.path.join(annotation_path, "fixation"))
        )
        has_video = any(
            os.path.isfile(os.path.join(video_dir, video_id + extension))
            or os.path.isfile(os.path.join(video_dir, video_id + extension.upper()))
            for extension in SaliencyDataset.VIDEO_EXTENSIONS
        )
        if has_annotations and has_video:
            video_ids.append(video_id)

    return sorted(
        video_ids,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )


def _avs_video_ids(dataset_root_dir, data_type, dataset_split):
    """读取 AVS-ODV 的划分文件（train_list_N.txt / test_list_N.txt），返回视频ID列表。"""
    list_file = os.path.join(dataset_root_dir, f"{data_type}_list_{dataset_split}.txt")
    if not os.path.isfile(list_file):
        raise FileNotFoundError(f"AVS-ODV split file not found: {list_file}")
    with open(list_file, "r") as f:
        return [line.strip().split()[0] for line in f if line.strip()]


def get_dataloaders(
    is_test: bool,
    dataset_name: str,
    dataset_root_dir: Optional[str],
    dataset_kwargs: Dict[str, Any],
    augmentation_kwargs: Dict[str, Any],
    train_batch_size: int,
    val_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    dataset_split: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    if dataset_root_dir is None:
        raise ValueError("dataset_root_dir is required")

    if dataset_name == "AVS-ODV":
                                                                       
        if is_test:
            test_videos = _avs_video_ids(dataset_root_dir, "test", dataset_split)
            if not test_videos:
                raise RuntimeError(f"No {dataset_name} test videos were found")
            print(f"{dataset_name} (split {dataset_split}) test videos: {len(test_videos)}")
            dataset_train = dataset_val = AVSSaliencyDataset(
                dataname=dataset_name,
                root_dir=dataset_root_dir,
                video_id=test_videos,
                dataset_kwargs=dataset_kwargs,
                augmentation_kwargs=NO_AUGMENTATION,
            )
        else:
            video_names = _avs_video_ids(dataset_root_dir, "train", dataset_split)
            if len(video_names) < 2:
                raise RuntimeError(f"At least two {dataset_name} training videos are required")

            rng = random.Random(33)
            rng.shuffle(video_names)
            split_idx = int(0.8 * len(video_names))
            train_videos = video_names[:split_idx]
            val_videos = video_names[split_idx:]
            print(
                f"{dataset_name} (split {dataset_split}) video split: "
                f"{len(train_videos)} train, {len(val_videos)} validation"
            )
            dataset_train = AVSSaliencyDataset(
                dataname=dataset_name,
                root_dir=dataset_root_dir,
                video_id=train_videos,
                dataset_kwargs=dataset_kwargs,
                augmentation_kwargs=augmentation_kwargs,
            )
            dataset_val = AVSSaliencyDataset(
                dataname=dataset_name,
                root_dir=dataset_root_dir,
                video_id=val_videos,
                dataset_kwargs=dataset_kwargs,
                augmentation_kwargs=NO_AUGMENTATION,
            )
    elif dataset_name in {"Sports-360", "SVGC_AVA"}:
                                                              
        if is_test:
            test_videos = _video_ids(dataset_root_dir, "test")
            if not test_videos:
                raise RuntimeError(f"No {dataset_name} test videos were found")
            print(f"{dataset_name} test videos: {len(test_videos)}")
            dataset_train = dataset_val = SaliencyDataset(
                dataname=dataset_name,
                root_dir=dataset_root_dir,
                video_id=test_videos,
                dataset_kwargs=dataset_kwargs,
                augmentation_kwargs=NO_AUGMENTATION,
                data_type="test",
                include_partial=False,
            )
        else:
            video_names = _video_ids(dataset_root_dir, "train")
            if len(video_names) < 2:
                raise RuntimeError(f"At least two {dataset_name} training videos are required")

            rng = random.Random(33)
            rng.shuffle(video_names)
            split_idx = int(0.8 * len(video_names))
            train_videos = video_names[:split_idx]
            val_videos = video_names[split_idx:]
            print(
                f"{dataset_name} video split: {len(train_videos)} train, "
                f"{len(val_videos)} validation"
            )
            dataset_train = SaliencyDataset(
                dataname=dataset_name,
                root_dir=dataset_root_dir,
                video_id=train_videos,
                dataset_kwargs=dataset_kwargs,
                augmentation_kwargs=augmentation_kwargs,
                data_type="train",
            )
            dataset_val = SaliencyDataset(
                dataname=dataset_name,
                root_dir=dataset_root_dir,
                video_id=val_videos,
                dataset_kwargs=dataset_kwargs,
                augmentation_kwargs=NO_AUGMENTATION,
                data_type="train",
            )
    else:
        raise ValueError(
            f"Unsupported dataset_name: {dataset_name} "
            f"(expected one of: Sports-360, AVS-ODV, SVGC_AVA)"
        )

    loader_train = DataLoader(
        dataset_train,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=not is_test,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return loader_train, loader_val
