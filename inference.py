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

from adaptive_diagnostics import (
    AdaptiveDiagnosticsAccumulator,
    AdaptiveDiagnosticsAccumulatorV2,
)
from adaptive_objectives import build_fixed_area_target
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
        adaptive_model_types = {
            "adaptive_sphere_uformer",
            "adaptive_sphere_uformer_v2",
        }
        if args.adaptive_diagnostics and args.model_type not in adaptive_model_types:
            raise ValueError(
                "--adaptive_diagnostics requires an adaptive model_type"
            )
        if (
                args.compare_uniform_gate or args.compare_gate_baselines
        ) and not args.use_adaptive_refinement:
            raise ValueError(
                "--compare_uniform_gate cannot be combined with "
                "--disable_adaptive_refinement"
            )
        if args.compare_gate_baselines and (
                args.model_type != "adaptive_sphere_uformer_v2"
        ):
            raise ValueError(
                "--compare_gate_baselines currently requires UAHS-V2"
            )
        if args.model_type == "adaptive_sphere_uformer_v2" and not (
                0 <= args.target_refine_ratio_l2
                <= args.target_refine_ratio_l1 <= 1
        ):
            raise ValueError(
                "UAHS-V2 requires 0 <= target_refine_ratio_l2 "
                "<= target_refine_ratio_l1 <= 1"
            )
        if not args.adaptive_diagnostics:
            self.adaptive_diagnostics = None
        elif args.model_type == "adaptive_sphere_uformer_v2":
            self.adaptive_diagnostics = AdaptiveDiagnosticsAccumulatorV2(
                args.target_refine_ratio_l1,
                args.target_refine_ratio_l2,
            )
        else:
            self.adaptive_diagnostics = AdaptiveDiagnosticsAccumulator(
                args.target_refine_ratio
            )
        self.comparison_modes = []
        if args.compare_uniform_gate:
            self.comparison_modes.append("full_fine")
        if args.compare_gate_baselines:
            self.comparison_modes.extend((
                "full_fine",
                "uniform_same_budget",
                "random_same_budget",
                "oracle_error_selection",
            ))
        self.comparison_modes = list(dict.fromkeys(self.comparison_modes))

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

    def _build_v2_gate_overrides(
            self,
            mode,
            adaptive_outputs,
            ground_truth,
            batch_index,
            rgb,
    ):
        reference_l4 = adaptive_outputs["gate_l4_parent"]
        reference_l5 = adaptive_outputs["gate_l5_to_l6_local"]
        ratio_l1 = self.args.target_refine_ratio_l1
        ratio_l2 = self.args.target_refine_ratio_l2
        if mode == "full_fine":
            return {
                "l4": torch.ones_like(reference_l4),
                "l5_local": torch.ones_like(reference_l5),
            }
        if mode == "uniform_same_budget":
            local_ratio = ratio_l2 / ratio_l1 if ratio_l1 > 0 else 0.0
            return {
                "l4": torch.full_like(reference_l4, ratio_l1),
                "l5_local": torch.full_like(reference_l5, local_ratio),
            }
        if mode == "oracle_error_selection":
            target_l4 = self.model.aggregate_img_values_to_l4_faces(ground_truth)
            selection_l4 = build_fixed_area_target(
                (target_l4 - adaptive_outputs["saliency_l4"]).abs(),
                self.model.hierarchy_l4_l5.coarse_face_areas,
                ratio_l1,
            )
            # S5 depends on the L4 gate. Recompute it under the oracle parent
            # selection before ranking L5 children; L5's own gate is downstream
            # of S5 and can be temporarily set to one.
            provisional = self.model(
                rgb,
                return_aux=True,
                gate_overrides={
                    "l4": selection_l4,
                    "l5_local": torch.ones_like(reference_l5),
                },
            )
            target_l5 = self.model.aggregate_img_values_to_l5_faces(ground_truth)
            eligible_l5 = self.model.hierarchy_l4_l5.propagate_coarse_face_values(
                selection_l4
            ).bool()
            selection_l5 = build_fixed_area_target(
                (target_l5 - provisional["saliency_l5"]).abs(),
                self.model.hierarchy_l5_l6.coarse_face_areas,
                ratio_l2,
                eligible_mask=eligible_l5,
            )
            return {
                "l4": selection_l4,
                "l5_local": selection_l5,
            }
        if mode == "random_same_budget":
            generator = torch.Generator(device=reference_l4.device)
            generator.manual_seed(self.args.diagnostic_random_seed + batch_index)
            random_l4 = torch.rand(
                reference_l4.shape,
                device=reference_l4.device,
                dtype=reference_l4.dtype,
                generator=generator,
            )
            selection_l4 = build_fixed_area_target(
                random_l4,
                self.model.hierarchy_l4_l5.coarse_face_areas,
                ratio_l1,
            )
            eligible_l5 = self.model.hierarchy_l4_l5.propagate_coarse_face_values(
                selection_l4
            ).bool()
            random_l5 = torch.rand(
                reference_l5.shape,
                device=reference_l5.device,
                dtype=reference_l5.dtype,
                generator=generator,
            )
            selection_l5 = build_fixed_area_target(
                random_l5,
                self.model.hierarchy_l5_l6.coarse_face_areas,
                ratio_l2,
                eligible_mask=eligible_l5,
            )
            return {"l4": selection_l4, "l5_local": selection_l5}
        raise ValueError(f"Unknown gate comparison mode: {mode}")

    def _write_adaptive_diagnostics(self, results, comparison_results):
        report = self.adaptive_diagnostics.summary()
        report["configuration"] = {
            "checkpoint": self.args.base_model_weights,
            "dataset": self.args.dataset_name,
            "img_rank": self.model.img_rank,
            "fine_rank": self.model.fine_rank,
            "coarse_rank": self.model.coarse_rank,
            "target_refine_ratio": self.args.target_refine_ratio,
            "target_refine_ratio_l1": self.args.target_refine_ratio_l1,
            "target_refine_ratio_l2": self.args.target_refine_ratio_l2,
            "max_batches": self.args.max_batches,
        }
        if hasattr(self.model, "middle_rank"):
            report["configuration"]["middle_rank"] = self.model.middle_rank
        if comparison_results:
            higher_is_better = {"AUC", "NSS", "CC", "SIM"}
            report["gate_comparisons"] = {
                "definitions": {
                    "full_fine": "all hierarchy gates are one",
                    "uniform_same_budget": (
                        "constant soft gates with effective areas equal to budgets"
                    ),
                    "random_same_budget": (
                        "random binary hierarchical selection at equal areas"
                    ),
                    "oracle_error_selection": (
                        "GT-error top-area hierarchical selection"
                    ),
                },
                "adaptive_metrics": results,
                "baselines": {},
            }
            for mode, baseline_results in comparison_results.items():
                adaptive_gain = {
                    name: (
                        results[name] - baseline_results[name]
                        if name in higher_is_better
                        else baseline_results[name] - results[name]
                    )
                    for name in results
                    if name in baseline_results
                }
                report["gate_comparisons"]["baselines"][mode] = {
                    "metrics": baseline_results,
                    "adaptive_gain_positive_is_better": adaptive_gain,
                }

        output_path = self.args.diagnostics_output
        if not output_path:
            output_path = os.path.join(
                os.path.dirname(self.args.output_dir),
                "adaptive_diagnostics.json",
            )
        output_parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2, ensure_ascii=False)
        print("\n========== Adaptive Diagnostics ==========")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Adaptive diagnostics written to: {output_path}")

    @torch.no_grad()
    def run(self):
        metric_totals = {name: 0.0 for name in ("AUC", "NSS", "CC", "SIM", "KL")}
        metric_frames = {name: 0 for name in metric_totals}
        comparison_metric_totals = {
            mode: {name: 0.0 for name in metric_totals}
            for mode in self.comparison_modes
        }
        comparison_metric_frames = {
            mode: {name: 0 for name in metric_totals}
            for mode in self.comparison_modes
        }
        progress = tqdm.tqdm(
            self.loader_test, desc=f"{self.args.dataset_name} inference"
        )
        for batch_idx, batch in enumerate(progress):
            device_batch = self._to_device(batch)
            valid_lengths = batch["valid_length"].tolist()
            if self.adaptive_diagnostics is not None:
                adaptive_outputs = self.model(
                    device_batch["normalized_sphere_rgb"], return_aux=True
                )
                predictions = adaptive_outputs["saliency"]
                self.adaptive_diagnostics.update(
                    adaptive_outputs,
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

            for mode in self.comparison_modes:
                if self.args.model_type == "adaptive_sphere_uformer_v2":
                    gate_overrides = self._build_v2_gate_overrides(
                        mode,
                        adaptive_outputs,
                        device_batch["normalized_sphere_sal"],
                        batch_idx,
                        device_batch["normalized_sphere_rgb"],
                    )
                    comparison_predictions = self.model(
                        device_batch["normalized_sphere_rgb"],
                        return_aux=False,
                        gate_overrides=gate_overrides,
                    )
                else:
                    adaptive_enabled = self.model.use_adaptive_refinement
                    try:
                        self.model.use_adaptive_refinement = False
                        comparison_predictions = self.model(
                            device_batch["normalized_sphere_rgb"],
                            return_aux=False,
                        )
                    finally:
                        self.model.use_adaptive_refinement = adaptive_enabled
                self._update_sphere_metrics(
                    comparison_predictions,
                    device_batch,
                    valid_lengths,
                    comparison_metric_totals[mode],
                    comparison_metric_frames[mode],
                )
            self._save_predictions(batch, predictions)
            if self.args.max_batches and batch_idx + 1 >= self.args.max_batches:
                break

        if self.args.save_mat:
            self.save_mat_results()

        results = self._mean_metrics(metric_totals, metric_frames)
        print(
            "Test metrics:",
            " ".join(f"{name}={value:.6f}" for name, value in results.items()),
        )
        comparison_results = {}
        for mode in self.comparison_modes:
            comparison_results[mode] = self._mean_metrics(
                comparison_metric_totals[mode],
                comparison_metric_frames[mode],
            )
            print(
                f"{mode} metrics:",
                " ".join(
                    f"{name}={value:.6f}"
                    for name, value in comparison_results[mode].items()
                ),
            )
        if self.adaptive_diagnostics is not None:
            self._write_adaptive_diagnostics(results, comparison_results)
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
        "--adaptive_diagnostics",
        action="store_true",
        help="collect UAHS gate, uncertainty, budget, and coarse-rank statistics",
    )
    parser.add_argument(
        "--compare_uniform_gate",
        action="store_true",
        help="evaluate the same adaptive checkpoint in full_fine gate=1 mode",
    )
    parser.add_argument(
        "--compare_gate_baselines",
        action="store_true",
        help=(
            "for UAHS-V2 evaluate full_fine, uniform/random same-budget, "
            "and oracle-error gates"
        ),
    )
    parser.add_argument(
        "--diagnostic_random_seed",
        type=int,
        default=0,
        help="reproducible random_same_budget selection seed",
    )
    parser.add_argument(
        "--diagnostics_output",
        default=None,
        help="optional JSON path for adaptive diagnostics",
    )
    args = parser.parse_args()
    if args.compare_uniform_gate or args.compare_gate_baselines:
        args.adaptive_diagnostics = True
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

"""
CUDA_VISIBLE_DEVICES=3 python /home/dyz/PythonProject/Test_Codes/Sampling_test/inference.py \
    --model_type adaptive_sphere_uformer \
    --dataset_name Sports-360 \
    --base_model_weights /home/dyz/PythonProject/Test_Codes/Sampling_test/log/models/Epoch_32model.pth \
    --output_dir /home/dyz/PythonProject/DataSet_Output/Sports-360 \
    --method_name SphereUformer-adp \
    --adaptive_diagnostics \
    --compare_uniform_gate \
    --diagnostics_output /home/dyz/PythonProject/Test_Codes/Sampling_test/log/adaptive_diagnostics.json \
    --seq_length 30 \
    --val_batch_size 1 \
    --num_workers 8
"""
