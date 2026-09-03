import json
import os
import sys

import cv2
import numpy as np
import torch
import tqdm

EVALUATE_DIR = os.path.join(os.path.dirname(__file__), "evaluate")
if EVALUATE_DIR not in sys.path:
    sys.path.insert(0, EVALUATE_DIR)

from adaptive_diagnostics import UAHSDiagnosticsAccumulator
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
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available() and args.use_gpu,
            dataset_split=args.avs_split,
        )

        # Inference accepts the unchanged baseline and final UAHS checkpoints.
        args.use_checkpoint = False
        self.model = build_saliency_model(args)
        self._load_weights(args.base_model_weights)
        self.model.to(self.device).eval()
        if args.uahs_diagnostics and args.model_type != "uahs":
            raise ValueError("--uahs_diagnostics requires --model_type uahs")
        if args.model_type == "uahs" and not (
                0 <= args.target_refine_ratio_l2
                <= args.target_refine_ratio_l1 <= 1
        ):
            raise ValueError(
                "UAHS requires 0 <= target_refine_ratio_l2 "
                "<= target_refine_ratio_l1 <= 1"
            )
        if args.uahs_diagnostics:
            self.uahs_diagnostics = UAHSDiagnosticsAccumulator(
                args.target_refine_ratio_l1,
                args.target_refine_ratio_l2,
            )
        else:
            self.uahs_diagnostics = None

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
        source_keys = set(state_dict)
        has_budget_heads = any(
            key.startswith("budget_head_") for key in source_keys
        )
        has_old_l5_head = any(
            key.startswith("budget_head_l4.mlp.") for key in source_keys
        )
        allowed_missing_prefixes = ()
        allowed_unexpected_prefixes = ()
        if not has_budget_heads:
            allowed_missing_prefixes = ("budget_head_l4.", "budget_head_l5.")
        elif has_old_l5_head:
            allowed_missing_prefixes = ("budget_head_l4.",)
            allowed_unexpected_prefixes = ("budget_head_l4.mlp.",)

        shape_mismatches = [
            key for key, value in state_dict.items()
            if key in model_state and value.shape != model_state[key].shape
        ]
        compatible_state = {
            key: value for key, value in state_dict.items()
            if key in model_state and value.shape == model_state[key].shape
        }
        missing = [
            key for key in model_state
            if key not in compatible_state
            and not key.startswith(allowed_missing_prefixes)
        ]
        unexpected = [
            key for key in state_dict
            if key not in model_state
            and not key.startswith(allowed_unexpected_prefixes)
        ]
        if missing or unexpected or shape_mismatches:
            raise RuntimeError(
                "Checkpoint does not match the selected model: "
                f"missing={missing}, unexpected={unexpected}, "
                f"shape_mismatches={shape_mismatches}"
            )
        model_state.update(compatible_state)
        self.model.load_state_dict(model_state, strict=True)
        if not has_budget_heads:
            print(
                "Checkpoint predates dynamic budgets; initialized budget heads "
                "at the configured 0.25 / 0.125 operating point."
            )
        elif has_old_l5_head:
            print(
                "Checkpoint uses the former L5 mean/std budget head; its "
                "weights are ignored and the new DeepSets head retains its "
                "safe 0.25 initialization."
            )
        print(f"Loaded {len(compatible_state)} model tensors from {path}")

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
    def _update_sphere_metrics(
            self,
            predictions,
            device_batch,
            valid_lengths,
            metric_totals,
            metric_frames,
    ):
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

    @staticmethod
    def _mean_metrics(metric_totals, metric_frames):
        return {
            name: total / metric_frames[name]
            for name, total in metric_totals.items()
            if metric_frames[name]
        }

    def _write_uahs_diagnostics(self, results):
        report = self.uahs_diagnostics.summary()
        report["metrics"] = results
        report["configuration"] = {
            "checkpoint": self.args.base_model_weights,
            "dataset": self.args.dataset_name,
            "model_type": self.args.model_type,
            "seq_length": self.args.seq_length,
            "temporal_window_radius": self.args.temporal_window_radius,
            "img_rank": self.model.img_rank,
            "fine_rank": self.model.fine_rank,
            "coarse_rank": self.model.coarse_rank,
            "embed_dim": self.model.embed_dim,
            "num_heads": self.args.enc_num_heads[0],
            "local_motion_blocks": 2,
            "global_content_blocks": 1,
            "global_query_chunk_size": self.args.global_query_chunk_size,
            "initial_budget_l5": self.args.target_refine_ratio_l1,
            "initial_budget_l6": self.args.target_refine_ratio_l2,
            "budget_l5_min": self.args.budget_l5_min,
            "budget_l5_max": self.args.budget_l5_max,
            "budget_error_threshold_l4": self.args.budget_error_threshold_l4,
            "budget_error_threshold_l5": self.args.budget_error_threshold_l5,
            "budget_error_temperature_l4": self.args.budget_error_temperature_l4,
            "budget_error_temperature_l5": self.args.budget_error_temperature_l5,
            "max_batches": self.args.max_batches,
            "routing": "uncertainty_only",
        }
        if hasattr(self.model, "middle_rank"):
            report["configuration"]["middle_rank"] = self.model.middle_rank

        output_path = self.args.diagnostics_output
        if not output_path:
            output_path = os.path.join(
                os.path.dirname(self.args.output_dir),
                "uahs_diagnostics.json",
            )
        output_parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2, ensure_ascii=False)
        print("\n========== UAHS Diagnostics ==========")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"UAHS diagnostics written to: {output_path}")

    @torch.no_grad()
    def run(self):
        metric_totals = {name: 0.0 for name in ("AUC", "NSS", "CC", "SIM", "KL")}
        metric_frames = {name: 0 for name in metric_totals}
        progress = tqdm.tqdm(
            self.loader_test, desc=f"{self.args.dataset_name} inference"
        )
        for batch_idx, batch in enumerate(progress):
            device_batch = self._to_device(batch)
            valid_lengths = batch["valid_length"].tolist()
            if self.uahs_diagnostics is not None:
                uahs_outputs = self.model(
                    device_batch["normalized_sphere_rgb"], return_aux=True
                )
                predictions = uahs_outputs["saliency"]
                self.uahs_diagnostics.update(
                    uahs_outputs,
                    device_batch["normalized_sphere_sal"],
                    valid_lengths,
                    self.model,
                )
            else:
                predictions = self.model(device_batch["normalized_sphere_rgb"])
                if isinstance(predictions, dict):
                    predictions = predictions["saliency"]
            self._update_sphere_metrics(
                predictions,
                device_batch,
                valid_lengths,
                metric_totals,
                metric_frames,
            )
            if not self.args.metrics_only:
                self._save_predictions(batch, predictions)
            if self.args.max_batches and batch_idx + 1 >= self.args.max_batches:
                break

        if self.args.save_mat and not self.args.metrics_only:
            self.save_mat_results()

        results = self._mean_metrics(metric_totals, metric_frames)
        if self.uahs_diagnostics is not None:
            self._write_uahs_diagnostics(results)
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
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="compute metrics/diagnostics without writing predictions or ERP outputs",
    )
    parser.add_argument(
        "--uahs_diagnostics",
        action="store_true",
        help="collect UAHS uncertainty, selector, hierarchy, and efficiency statistics",
    )
    parser.add_argument(
        "--diagnostics_output",
        default=None,
        help="optional JSON path for UAHS diagnostics",
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

    if args.metrics_only:
        print("\nERP evaluation skipped in --metrics_only mode")
    else:
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

    print("\n========== Final Sphere Metrics ==========")
    for name in ("AUC", "NSS", "CC", "SIM", "KL"):
        if name in sphere_metrics:
            print(f"{name}: {sphere_metrics[name]:.6f}")


if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=3 python /home/dyz/PythonProject/Test_Codes/Sampling_test/inference.py \
    --model_type uahs \
    --dataset_name Sports-360 \
    --base_model_weights /home/dyz/PythonProject/Test_Codes/Sampling_test/log/uahs-v3-sports360/models/Epoch_30model.pth \
    --output_dir /home/dyz/PythonProject/DataSet_Output/Sports-360 \
    --method_name UAHS \
    --mode vertex \
    --img_rank 6 \
    --seq_length 12 \
    --temporal_window_radius none \
    --coarse_pool_type mean_max \
    --target_refine_ratio_l1 0.25 \
    --target_refine_ratio_l2 0.125 \
    --global_query_chunk_size 128 \
    --hard_selection_warmup_epochs 0 \
    --val_batch_size 1 \
    --num_workers 8

Epoch_30model.pth
Average AUC-J:  0.9387995677737113
Average NSS:  4.352345161239625
Average KL Divergence:  1.6327730351383458
Average SIM:  0.4899660684485185
Average CC:  0.670838757243907
========== Sphere Metrics (for comparison) ==========
AUC: 0.922776
NSS: 3.646490
CC: 0.663930
SIM: 0.490483
KL: 0.955341
"""
