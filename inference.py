import os
import sys

import cv2
import numpy as np
import torch
import tqdm

EVALUATE_DIR = os.path.join(os.path.dirname(__file__), "evaluate")
if EVALUATE_DIR not in sys.path:
    sys.path.insert(0, EVALUATE_DIR)

from data.get_saliency_dataloaders import get_dataloaders
from evaluation import evaluate_saliency_maps_in_folder
from network.sphere_model import build_saliency_model
from Sphere_SalientScore_torch import batch_compute_metrics
from train import parser
from trimesh_utils import IcoSphereRef, asSpherical


class InferenceRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
        )

        _, self.loader_test = get_dataloaders(
            is_test=True,
            dataset_name=args.dataset_name,
            dataset_root_dir=args.dataset_root_dir,
            dataset_kwargs={
                "sphere_rank": args.img_rank,
                "grid_width": args.img_width,
                "sphere_node_type": args.mode,
                "seq_length": args.seq_length,
            },
            augmentation_kwargs={},
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available() and args.use_gpu,
            dataset_split=args.avs_split,
        )

        # Inference accepts both the unchanged baseline and adaptive checkpoints.
        args.use_checkpoint = False
        self.model = build_saliency_model(args)
        self._load_weights(args.base_model_weights)
        self.model.to(self.device).eval()

        sphere_ref = IcoSphereRef(args.mode)
        spherical = asSpherical(sphere_ref.get_normals(rank=args.img_rank))
        normals_wh = np.stack(
            (spherical[:, 2] / 180, spherical[:, 1] / 180 * 2 - 1), axis=1
        ).astype(np.float32)
        self.normals_wh = torch.from_numpy(normals_wh).to(self.device)
        self._reconstruction_cache = {}

        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Inference device: {self.device}")
        print(f"Predictions will be written to: {args.output_dir}")

    def _load_weights(self, path):
        if not path:
            raise ValueError("Inference requires --base_model_weights")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model weights not found: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        model_state = self.model.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and value.shape == model_state[key].shape
        }
        if not compatible:
            raise RuntimeError(f"No compatible parameters found in {path}")
        model_state.update(compatible)
        self.model.load_state_dict(model_state)
        print(f"Loaded {len(compatible)}/{len(model_state)} model tensors from {path}")
        if len(compatible) != len(model_state):
            print(
                "Warning: the checkpoint does not fully match this model; unmatched "
                "layers keep their random initialization, so metrics are not reliable."
            )

    def _to_device(self, batch):
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value) and key.startswith("normalized_sphere_")
        }

    def reconstruct_erp(self, sphere_values, height, width):
        cache_key = (height, width)
        if cache_key not in self._reconstruction_cache:
            scale = torch.tensor(
                [(width - 1) / 2, (height - 1) / 2],
                device=self.device,
                dtype=torch.float32,
            )
            pixel_coords = (self.normals_wh + 1) * scale
            x, y = pixel_coords[:, 0], pixel_coords[:, 1]
            x0 = x.floor().long().clamp(0, width - 1)
            y0 = y.floor().long().clamp(0, height - 1)
            x1 = (x0 + 1).clamp(0, width - 1)
            y1 = (y0 + 1).clamp(0, height - 1)
            dx, dy = x - x0.float(), y - y0.float()
            self._reconstruction_cache[cache_key] = (
                x0,
                y0,
                x1,
                y1,
                (1 - dx) * (1 - dy),
                dx * (1 - dy),
                (1 - dx) * dy,
                dx * dy,
            )

        x0, y0, x1, y1, w00, w10, w01, w11 = self._reconstruction_cache[cache_key]
        result = torch.zeros(
            (height, width), device=self.device, dtype=sphere_values.dtype
        )
        weights = torch.zeros_like(result)
        for xs, ys, interpolation_weights in (
            (x0, y0, w00),
            (x1, y0, w10),
            (x0, y1, w01),
            (x1, y1, w11),
        ):
            result.index_put_(
                (ys, xs), interpolation_weights * sphere_values, accumulate=True
            )
            weights.index_put_((ys, xs), interpolation_weights, accumulate=True)
        return result / weights.clamp_min(1e-8)

    def _save_predictions(self, batch, predictions):
        height, width = 128, 256
        valid_lengths = batch["valid_length"].tolist()
        paths_by_time = batch["seq_path"]
        for sample_idx, valid_length in enumerate(valid_lengths):
            for frame_idx in range(valid_length):
                source_path = paths_by_time[frame_idx][sample_idx]
                if self.args.dataset_name == "AVS-ODV":
                    filelist = source_path.split('/')
                    video_id = filelist[-2]
                else:
                    video_id = os.path.basename(os.path.dirname(os.path.dirname(source_path)))
                output_dir = os.path.join(self.args.output_dir, video_id)
                os.makedirs(output_dir, exist_ok=True)
                prediction = self.reconstruct_erp(
                    predictions[sample_idx, frame_idx], height, width
                )
                prediction = prediction.detach().cpu().numpy()
                prediction = (prediction - prediction.min()) / (
                    prediction.max() - prediction.min() + 1e-8
                )
                output_path = os.path.join(output_dir, (os.path.basename(source_path)).split(".")[0]+".png")
                if not cv2.imwrite(output_path, (prediction * 255).astype(np.uint8)):
                    raise RuntimeError(f"Failed to write prediction: {output_path}")

    def save_mat_results(self):
        """Combine the saved per-frame PNGs of each video into one .mat file,
        written to a saliency_mat folder next to saliency_png."""
        try:
            import hdf5storage

            savemat = hdf5storage.savemat
        except ImportError:
            from scipy.io import savemat

            print("hdf5storage not installed, falling back to scipy.io.savemat")

        mat_dir = os.path.join(os.path.dirname(self.args.output_dir), "saliency_mat")
        os.makedirs(mat_dir, exist_ok=True)
        video_ids = sorted(
            name
            for name in os.listdir(self.args.output_dir)
            if os.path.isdir(os.path.join(self.args.output_dir, name))
        )
        for video_id in tqdm.tqdm(video_ids, desc="Writing .mat results"):
            video_dir = os.path.join(self.args.output_dir, video_id)
            frame_names = [
                name
                for name in os.listdir(video_dir)
                if os.path.isfile(os.path.join(video_dir, name))
            ]
            frame_names.sort(
                key=lambda name: (0, int(os.path.splitext(name)[0]))
                if os.path.splitext(name)[0].isdigit()
                else (1, name)
            )
            if not frame_names:
                continue
            first = cv2.imread(
                os.path.join(video_dir, frame_names[0]), cv2.IMREAD_GRAYSCALE
            )
            height, width = first.shape
            pred_mat = np.zeros((len(frame_names), height, width, 1), dtype=np.uint8)
            for idx, frame_name in enumerate(frame_names):
                sal = cv2.imread(
                    os.path.join(video_dir, frame_name), cv2.IMREAD_GRAYSCALE
                )
                pred_mat[idx, :, :, 0] = sal.astype(np.uint8)
            savemat(os.path.join(mat_dir, video_id + ".mat"), {"salmap": pred_mat})
        print(f".mat results written to: {mat_dir}")

    @torch.no_grad()
    def run(self):
        metric_totals = {name: 0.0 for name in ("AUC", "NSS", "CC", "SIM", "KL")}
        metric_frames = {name: 0 for name in metric_totals}
        progress = tqdm.tqdm(
            self.loader_test, desc=f"{self.args.dataset_name} inference"
        )
        for batch_idx, batch in enumerate(progress):
            device_batch = self._to_device(batch)
            predictions = self.model(device_batch["normalized_sphere_rgb"])
            if isinstance(predictions, dict):
                predictions = predictions["saliency"]
            valid_lengths = batch["valid_length"].tolist()
            for sample_idx, valid_length in enumerate(valid_lengths):
                metrics = batch_compute_metrics(
                    predictions[sample_idx : sample_idx + 1, :valid_length],
                    device_batch["normalized_sphere_sal"][
                        sample_idx : sample_idx + 1, :valid_length
                    ],
                    device_batch["normalized_sphere_fix"][
                        sample_idx : sample_idx + 1, :valid_length
                    ],
                    self.device,
                )
                for name, value in metrics.items():
                    metric_totals[name] += value.item() * valid_length
                    metric_frames[name] += valid_length
            self._save_predictions(batch, predictions)
            if self.args.max_batches and batch_idx + 1 >= self.args.max_batches:
                break

        if self.args.save_mat:
            self.save_mat_results()

        results = {
            name: total / metric_frames[name]
            for name, total in metric_totals.items()
            if metric_frames[name]
        }
        print(
            "Test metrics:",
            " ".join(f"{name}={value:.6f}" for name, value in results.items()),
        )
        return results


def main():
    parser.description = "SphereUFormer video inference and validation"
    parser.add_argument(
        "--output_dir",
        default="/home/dyz/PythonProject/DataSet_Output/Sports-360",
        help="root output directory",
    )
    parser.add_argument(
        "--method_name",
        default="SphereUformer",
        help="method name used in the output path",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="optional batch limit for smoke tests; 0 processes the full test set",
    )
    parser.add_argument(
        "--save_mat",
        action="store_true",
        help="also save per-video .mat results in a saliency_mat folder next to saliency_png",
    )
    args = parser.parse_args()
    args.task = "salient"
    args.test = True
                                          
    if args.dataset_root_dir is None:
        args.dataset_root_dir = os.path.join(
            "/home/dyz/PythonProject/Dataset", args.dataset_name
        )
    args.output_dir = os.path.join(
        args.output_dir,
        "Results",
        "Results_Oth",
        "Saliency",
        args.method_name,
        "saliency_png",
    )
    runner = InferenceRunner(args)
    sphere_metrics = runner.run()

    print("\n========== ERP Metrics ==========")
    if args.dataset_name == "AVS-ODV":
        ground_truth_dir = args.dataset_root_dir
    else:
        ground_truth_dir = os.path.join(args.dataset_root_dir, "testing")
    try:
        evaluate_saliency_maps_in_folder(
            args.output_dir,
            ground_truth_dir,
            args.dataset_name,
        )
    except Exception as error:
        print(f"ERP evaluation failed: {error}")

    print("\n========== Sphere Metrics (for comparison) ==========")
    for name in ("AUC", "NSS", "CC", "SIM", "KL"):
        if name in sphere_metrics:
            print(f"{name}: {sphere_metrics[name]:.6f}")


if __name__ == "__main__":
    main()
