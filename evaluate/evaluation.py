from pathlib import Path

import numpy as np

from loadpath import (
    load_ground_fix_from_png,
    load_ground_map_from_png,
    load_saliency_map_from_png,
)
from utils_metrics import AUC_Judd, CC, KLD, NSS, SIM


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _image_files(directory):
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _test_video_ids(ground_truth_root, dataset_split, dataset_name):
    split_name = (
        "test_list.txt" if dataset_name == "VR-EyeTracking"
        else f"test_list_{dataset_split}.txt"
    )
    split_path = Path(ground_truth_root) / split_name
    if not split_path.is_file():
        raise FileNotFoundError(f"{dataset_name} split file not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as split_file:
        return [
            line.strip().split()[0]
            for line in split_file
            if line.strip()
        ]


def _expected_erp_frames(ground_truth_root, dataset_name, dataset_split):
    """Return expected prediction and annotation paths for a test split."""
    root = Path(ground_truth_root)
    expected = []

    if dataset_name == "AVS-ODV":
        video_ids = _test_video_ids(root, dataset_split, dataset_name)
        for video_id in video_ids:
            for ground_map_path in _image_files(root / "maps" / video_id):
                stem = ground_map_path.stem
                expected.append(
                    (
                        video_id,
                        stem,
                        ground_map_path,
                        root / "fixation" / video_id / f"{stem}fix.png",
                    )
                )
        return expected

    if dataset_name in {"SVGC_AVA", "Sports-360", "VR-EyeTracking"}:
        video_ids = _test_video_ids(root, dataset_split, dataset_name)
        for video_id in video_ids:
            for ground_map_path in _image_files(root / "maps" / video_id):
                stem = ground_map_path.stem
                expected.append(
                    (
                        video_id,
                        stem,
                        ground_map_path,
                        root / "fixation" / video_id / f"{stem}.png",
                    )
                )
        return expected

    if dataset_name != "Sports-360":
        raise ValueError(f"Unsupported dataset for ERP evaluation: {dataset_name}")
    if not root.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {root}")

    for video_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for ground_map_path in _image_files(video_dir / "maps"):
            stem = ground_map_path.stem
            expected.append(
                (
                    video_dir.name,
                    stem,
                    ground_map_path,
                    video_dir / "fixation" / f"{stem}.png",
                )
            )
    return expected


def evaluate_saliency_maps_in_folder(
        saliency_folder,
        ground_truth_root,
        DatasetName,
        dataset_split=1,
        expected_frames=None,
):
    """Evaluate ERP predictions and report test-set coverage statistics."""
    if expected_frames is None:
        expected_frames = _expected_erp_frames(
            ground_truth_root,
            DatasetName,
            dataset_split,
        )
    metric_values = {
        "AUC-J": [],
        "NSS": [],
        "KL Divergence": [],
        "SIM": [],
        "CC": [],
    }
    matched_frames = 0
    missing_predictions = 0
    skipped_abnormal = 0
    evaluated_frames = 0

    for video_id, stem, ground_map_path, ground_fix_path in expected_frames:
        prediction_path = Path(saliency_folder) / video_id / f"{stem}.png"
        if not prediction_path.is_file():
            missing_predictions += 1
            continue

        if not ground_fix_path.is_file():
            skipped_abnormal += 1
            print(f"Missing fixation map, skipping: {ground_fix_path}")
            continue

        matched_frames += 1
        try:
            sal_map = load_saliency_map_from_png(str(prediction_path))
            ground_map = load_ground_map_from_png(str(ground_map_path))
            ground_fix = load_ground_fix_from_png(str(ground_fix_path))
            if not np.any(sal_map):
                raise ValueError("prediction is all zero")

            frame_metrics = {
                "AUC-J": AUC_Judd(sal_map, ground_fix),
                "NSS": NSS(sal_map, ground_fix),
                "KL Divergence": KLD(sal_map, ground_map),
                "SIM": SIM(sal_map, ground_map),
                "CC": CC(sal_map, ground_map),
            }
            if not all(np.isfinite(value) for value in frame_metrics.values()):
                raise ValueError(f"non-finite metrics: {frame_metrics}")
        except Exception as error:
            skipped_abnormal += 1
            print(f"Abnormal sample skipped ({prediction_path}): {error}")
            continue

        for name, value in frame_metrics.items():
            metric_values[name].append(value)
        evaluated_frames += 1

    print("\n========== ERP Evaluation Coverage ==========")
    print(f"应有帧数: {len(expected_frames)}")
    print(f"实际匹配帧数: {matched_frames}")
    print(f"缺失预测帧数: {missing_predictions}")
    print(f"跳过异常样本数: {skipped_abnormal}")
    print(f"成功计算指标帧数: {evaluated_frames}")

    results = {}
    if evaluated_frames:
        for name, values in metric_values.items():
            results[name] = float(np.mean(values))
            print(f"Average {name}: {results[name]}")
    else:
        print("No valid results found.")

    results["counts"] = {
        "expected_frames": len(expected_frames),
        "matched_frames": matched_frames,
        "missing_predictions": missing_predictions,
        "skipped_abnormal": skipped_abnormal,
        "evaluated_frames": evaluated_frames,
    }
    return results


if __name__ == "__main__":
    dataset_name = "AVS-ODV"
    model = "SphereUformer-split-2"
    saliency_root = (
        "/home/dyz/PythonProject/DataSet_Output/"
        + dataset_name
        + "/Results/Results_Oth/Saliency/"
        + model
        + "/saliency_png"
    )
    ground_truth_folder = "/home/dyz/PythonProject/Dataset/" + dataset_name
    evaluate_saliency_maps_in_folder(
        saliency_root,
        ground_truth_folder,
        dataset_name,
    )
