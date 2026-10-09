import os
import random
from glob import glob
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader

from .DataLoader360Video import AVSSaliencyDataset, SaliencyDataset


def resolve_dataset_split(dataset_name, dataset_split):
    """Return the split supported by the selected dataset.

    AVS-ODV provides splits 1--3, whereas list-based video datasets provide fewer splits.
    Keep ``--dataset_split 3`` usable for shared command lines by falling back to
    the default split 1 for list-based datasets.
    """
    if dataset_name == "VR-EyeTracking" and dataset_split != 1:
        raise ValueError("VR-EyeTracking provides only dataset_split=1")
    if dataset_name in {"SVGC_AVA", "Sports-360"} and dataset_split == 3:
        return 1
    return dataset_split


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


def _split_video_ids(dataset_root_dir, data_type, dataset_split, dataset_name):
    """读取 train_list_N.txt / test_list_N.txt，返回视频 ID 列表。"""
    split_file = (
        f"{data_type}_list.txt" if dataset_name == "VR-EyeTracking"
        else f"{data_type}_list_{dataset_split}.txt"
    )
    list_file = os.path.join(dataset_root_dir, split_file)
    if not os.path.isfile(list_file):
        raise FileNotFoundError(f"{dataset_name} split file not found: {list_file}")
    with open(list_file, "r", encoding="utf-8") as f:
        return [line.strip().split()[0] for line in f if line.strip()]


def get_dataloaders(
    is_test: bool,
    dataset_name: str,
    dataset_root_dir: Optional[str],
    dataset_kwargs: Dict[str, Any],
    train_batch_size: int,
    val_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    dataset_split: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    if dataset_root_dir is None:
        raise ValueError("dataset_root_dir is required")

    dataset_split = resolve_dataset_split(dataset_name, dataset_split)

    if dataset_name == "AVS-ODV":
                                                                       
        if is_test:
            test_videos = _split_video_ids(
                dataset_root_dir, "test", dataset_split, dataset_name
            )
            if not test_videos:
                raise RuntimeError(f"No {dataset_name} test videos were found")
            print(f"{dataset_name} (split {dataset_split}) test videos: {len(test_videos)}")
            dataset_train = dataset_val = AVSSaliencyDataset(
                dataname=dataset_name,
                root_dir=dataset_root_dir,
                video_id=test_videos,
                dataset_kwargs=dataset_kwargs,
            )
        else:
            video_names = _split_video_ids(
                dataset_root_dir, "train", dataset_split, dataset_name
            )
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
            )
            dataset_val = AVSSaliencyDataset(
                dataname=dataset_name,
                root_dir=dataset_root_dir,
                video_id=val_videos,
                dataset_kwargs=dataset_kwargs,
            )
    elif dataset_name in {"SVGC_AVA", "Sports-360", "VR-EyeTracking"}:
        train_videos = _split_video_ids(
            dataset_root_dir, "train", dataset_split, dataset_name
        )
        test_videos = _split_video_ids(
            dataset_root_dir, "test", dataset_split, dataset_name
        )
        overlap = sorted(set(train_videos) & set(test_videos))
        if overlap:
            raise RuntimeError(
                f"{dataset_name} split {dataset_split} contains videos in both "
                f"train and test lists: {overlap}"
            )
        if not train_videos or not test_videos:
            raise RuntimeError(
                f"{dataset_name} split {dataset_split} requires non-empty train and test lists"
            )

        if is_test:
            print(
                f"{dataset_name} (split {dataset_split}) test videos: "
                f"{len(test_videos)}"
            )
            dataset_train = dataset_val = SaliencyDataset(
                dataname=dataset_name,
                root_dir=dataset_root_dir,
                video_id=test_videos,
                dataset_kwargs=dataset_kwargs,
                data_type="test",
                include_partial=dataset_name == "VR-EyeTracking",
            )
        else:
            rng = random.Random(33)
            rng.shuffle(train_videos)
            split_idx = int(0.8 * len(train_videos))
            val_videos = train_videos[split_idx:]
            train_videos = train_videos[:split_idx]
            if not train_videos or not val_videos:
                raise RuntimeError(
                    f"At least two {dataset_name} training videos are required"
                )
            print(
                f"{dataset_name} (split {dataset_split}) video split: "
                f"{len(train_videos)} train, {len(val_videos)} validation; "
                f"{len(test_videos)} test videos remain held out"
            )
            dataset_train = SaliencyDataset(
                dataname=dataset_name,
                root_dir=dataset_root_dir,
                video_id=train_videos,
                dataset_kwargs=dataset_kwargs,
                data_type="train",
            )
            dataset_val = SaliencyDataset(
                dataname=dataset_name,
                root_dir=dataset_root_dir,
                video_id=val_videos,
                dataset_kwargs=dataset_kwargs,
                data_type="train",
            )
    else:
        raise ValueError(
            f"Unsupported dataset_name: {dataset_name} "
            f"(expected one of: Sports-360, AVS-ODV, SVGC_AVA, VR-EyeTracking)"
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
