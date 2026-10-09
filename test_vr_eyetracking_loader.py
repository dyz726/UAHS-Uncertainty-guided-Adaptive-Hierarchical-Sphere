"""End-to-end frame/label alignment checks for VR-EyeTracking clips."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data._utils.collate import default_collate

from data.DataLoader360Video import SaliencyDataset
from data.get_saliency_dataloaders import get_dataloaders, resolve_dataset_split
from inference import InferenceRunner


class VREyeTrackingLoaderTest(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_dir.cleanup)
        self.root = Path(self.temporary_dir.name)
        train_ids = ["train_a.avi", "train_b.avi"]
        test_id = "test.avi"
        (self.root / "train_list.txt").write_text("\n".join(train_ids) + "\n")
        (self.root / "test_list.txt").write_text(test_id + "\n")
        for video_id in train_ids + [test_id]:
            self._make_video(video_id, missing=(0, 25) if video_id == test_id else ())

    def _make_video(self, video_id, missing):
        video_dir = self.root / "videos"
        map_dir = self.root / "maps" / video_id
        fixation_dir = self.root / "fixation" / video_id
        video_dir.mkdir(exist_ok=True)
        map_dir.mkdir(parents=True)
        fixation_dir.mkdir(parents=True)
        writer = cv2.VideoWriter(
            str(video_dir / video_id), cv2.VideoWriter_fourcc(*"MJPG"),
            12.0, (32, 16),
        )
        self.assertTrue(writer.isOpened())
        try:
            for frame_idx in range(37):
                writer.write(np.full((16, 32, 3), frame_idx * 5, dtype=np.uint8))
                annotation = np.full((32, 64), frame_idx + 1, dtype=np.uint8)
                if not missing or frame_idx != missing[0]:
                    cv2.imwrite(str(map_dir / f"{frame_idx:04d}.png"), annotation)
                if not missing or frame_idx != missing[1]:
                    cv2.imwrite(str(fixation_dir / f"{frame_idx:04d}.png"), annotation)
        finally:
            writer.release()

    def _loaders(self, is_test):
        return get_dataloaders(
            is_test=is_test,
            dataset_name="VR-EyeTracking",
            dataset_root_dir=str(self.root),
            dataset_kwargs={
                "sphere_rank": 1,
                "sphere_node_type": "vertex",
                "seq_length": 12,
            },
            train_batch_size=1,
            val_batch_size=1,
            num_workers=0,
            pin_memory=False,
        )

    def test_training_and_test_clips_stay_aligned_across_missing_labels(self):
        train_loader, val_loader = self._loaders(is_test=False)
        self.assertEqual(len(train_loader.dataset), 3)
        self.assertEqual(len(val_loader.dataset), 3)

        _, test_loader = self._loaders(is_test=True)
        dataset = test_loader.dataset
        self.assertEqual([item["frame_indices"][0] for item in dataset.data_list], [12, 36])
        self.assertEqual([item["valid_length"] for item in dataset.data_list], [12, 1])
        self.assertEqual(
            [int(Path(item["sal_seq"][0]).stem) for item in dataset.data_list],
            [12, 36],
        )
        self.assertEqual(
            [int(Path(item["fix_seq"][0]).stem) for item in dataset.data_list],
            [12, 36],
        )

        sample = dataset[0]
        self.assertEqual(tuple(sample["normalized_sphere_rgb"].shape), (12, 42, 3))
        for offset in range(12):
            frame_idx = 12 + offset
            self.assertAlmostEqual(
                float(sample["sphere_rgb"][offset].mean()),
                frame_idx * 5 / 255,
                delta=0.02,
            )
            self.assertAlmostEqual(
                float(sample["normalized_sphere_sal"][offset].mean()),
                (frame_idx + 1) / 255,
                delta=1e-5,
            )
            self.assertAlmostEqual(
                float(sample["normalized_sphere_fix"][offset].mean()),
                (frame_idx + 1) / 255,
                delta=1e-5,
            )

        expected = list(dataset.iter_expected_frames())
        self.assertEqual(len(expected), 13)
        self.assertEqual([int(item[1]) for item in expected], list(range(12, 24)) + [36])

    def test_only_the_declared_split_is_supported(self):
        with self.assertRaisesRegex(ValueError, "dataset_split=1"):
            resolve_dataset_split("VR-EyeTracking", 2)

    def test_seek_is_corrected_by_sequential_frame_timestamp(self):
        timestamps = np.arange(40, dtype=np.int64)

        class OffsetCapture:
            def __init__(self, _path):
                self.position = 0
                self.last = -1

            def isOpened(self):
                return True

            def set(self, _property, index):
                self.position = int(index) + 2
                return True

            def read(self):
                self.last = self.position
                self.position += 1
                return True, np.full((1, 1, 3), self.last, dtype=np.uint8)

            def get(self, _property):
                return float(timestamps[self.last])

            def release(self):
                pass

        dataset = SaliencyDataset.__new__(SaliencyDataset)
        dataset.dataname = "VR-EyeTracking"
        dataset.video_frame_pts = {"fake.mp4": timestamps * 1000}
        with mock.patch("data.DataLoader360Video.cv2.VideoCapture", OffsetCapture):
            frames = list(dataset._iter_video_frames("fake.mp4", [20, 21, 22, 23]))
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [20, 21, 22, 23])

        timestamps[19] = timestamps[20]
        dataset.video_frame_pts = {"fake.mp4": timestamps * 1000}
        with mock.patch("data.DataLoader360Video.cv2.VideoCapture", OffsetCapture):
            frames = list(dataset._iter_video_frames("fake.mp4", [20, 21]))
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [20, 21])

        timestamps[19] = 19

        class SkipAfterSeekCapture(OffsetCapture):
            def __init__(self, path):
                super().__init__(path)
                self.was_seeked = False

            def set(self, property_id, index):
                self.was_seeked = True
                return super().set(property_id, index)

            def read(self):
                if self.was_seeked and self.last == 20:
                    self.position += 1
                return super().read()

        dataset.video_frame_pts = {"fake.mp4": timestamps * 1000}
        with mock.patch("data.DataLoader360Video.cv2.VideoCapture", SkipAfterSeekCapture):
            frames = list(dataset._iter_video_frames("fake.mp4", [20, 21, 22]))
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [20, 21, 22])

    def test_inference_names_and_sizes_only_valid_source_frames(self):
        _, test_loader = self._loaders(is_test=True)
        runner = InferenceRunner.__new__(InferenceRunner)
        output_dir = self.root / "predictions"
        runner.args = SimpleNamespace(
            dataset_name="VR-EyeTracking", output_dir=str(output_dir)
        )
        runner.reconstruct_erp = lambda _values, height, width: torch.arange(
            height * width, dtype=torch.float32
        ).reshape(height, width)
        for sample in test_loader.dataset:
            batch = default_collate([sample])
            predictions = torch.zeros((1, 12, 42))
            runner._save_predictions(batch, predictions)

        video_dir = output_dir / "test.avi"
        filenames = sorted(path.name for path in video_dir.glob("*.png"))
        self.assertEqual(
            filenames,
            [f"{idx:04d}.png" for idx in list(range(12, 24)) + [36]],
        )
        self.assertEqual(
            cv2.imread(str(video_dir / "0036.png"), cv2.IMREAD_GRAYSCALE).shape,
            (128, 256),
        )
        with mock.patch.dict("sys.modules", {"hdf5storage": None}):
            runner.save_mat_results()
        contents = loadmat(str(self.root / "saliency_mat" / "test.avi.mat"))
        self.assertEqual(
            contents["frame_indices"].reshape(-1).tolist(),
            list(range(12, 24)) + [36],
        )


    def test_trainer_test_excludes_padding_and_weights_real_frames(self):
        from train_salient import Trainer

        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(dataset_name="VR-EyeTracking")
        trainer.device = torch.device("cpu")
        trainer.writer = None
        trainer.model = torch.nn.Identity()
        trainer.loader_val = [{
            "normalized_sphere_rgb": torch.zeros(2, 12, 42),
            "normalized_sphere_sal": torch.zeros(2, 12, 42),
            "normalized_sphere_fix": torch.zeros(2, 12, 42),
            "valid_length": torch.tensor([12, 1]),
        }]
        lengths = []

        def metrics(pred, sal, fix, device):
            lengths.append(pred.shape[1])
            self.assertEqual(pred.shape, sal.shape)
            self.assertEqual(pred.shape, fix.shape)
            value = 2.0 if pred.shape[1] == 12 else 4.0
            return {key: torch.tensor(value) for key in ("AUC", "NSS", "CC", "SIM", "KL")}

        with mock.patch("train_salient.batch_compute_metrics", side_effect=metrics):
            results = trainer.test()
        self.assertEqual(lengths, [12, 1])
        for value in results.values():
            self.assertAlmostEqual(value, 28.0 / 13)

    def test_standalone_erp_evaluation_uses_valid_clip_coverage(self):
        from evaluation import main

        with mock.patch("evaluation.evaluate_saliency_maps_in_folder") as evaluate:
            main([
                "--dataset_name", "VR-EyeTracking",
                "--dataset_root_dir", str(self.root),
                "--saliency_folder", str(self.root / "predictions"),
                "--seq_length", "12",
            ])
        expected = evaluate.call_args.kwargs["expected_frames"]
        self.assertEqual([int(frame[1]) for frame in expected], list(range(12, 24)) + [36])


if __name__ == "__main__":
    unittest.main()
