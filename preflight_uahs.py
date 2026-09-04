"""Real-configuration GPU training preflight for final sparse UAHS.

The script uses the defaults from ``train.py`` and runs three complete FP32
training iterations at B=1, T=12, rank 4->5->6. It is deliberately synthetic:
the purpose is to isolate model, objective, gradient, memory, and timing behavior
from data-loader I/O without reducing the formal network configuration.
"""

import argparse
import json
import os
import time

import torch

from network.sphere_model import build_saliency_model
from train import parser as training_parser
from train_salient import Trainer


GIB = 1024 ** 3


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device):
    if device.type != "cuda":
        return {}
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / GIB,
        "reserved_gb": torch.cuda.memory_reserved(device) / GIB,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / GIB,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / GIB,
    }


class StageProfiler:
    """Record allocator state at V3 stage boundaries without saving tensors."""

    def __init__(self, model, device):
        self.device = device
        self.phase = "idle"
        self.last_stage = None
        self.events = []
        self.handles = []
        stages = {
            "raw_rank6_projection": model.input_proj_l6,
            "raw_rank5_region_pool": model.raw_region_pool_l5,
            "rank4_region_pool": model.coarse_region_pool,
            "rank4_local_motion_encoder": model.coarse_local_encoder,
            "rank4_global_content_block": model.coarse_global_block,
            "rank4_uncertainty": model.uncertainty_head_l4,
            "rank4_dynamic_budget": model.budget_head_l4,
            "rank5_sparse_refiner": model.sparse_refiner_l5,
            "rank5_uncertainty": model.uncertainty_head_l5,
            "rank5_dynamic_budget": model.budget_head_l5,
            "rank6_sparse_refiner": model.sparse_refiner_l6,
            "final_saliency_head": model.output_proj,
        }
        for name, module in stages.items():
            self.handles.append(module.register_forward_pre_hook(
                lambda _module, _inputs, stage=name: self._record(stage, "enter")
            ))
            self.handles.append(module.register_forward_hook(
                lambda _module, _inputs, _output, stage=name:
                self._record(stage, "exit")
            ))

    def _record(self, stage, event):
        self.last_stage = stage
        self.events.append({
            "phase": self.phase,
            "stage": stage,
            "event": event,
            **memory_snapshot(self.device),
        })

    def close(self):
        for handle in self.handles:
            handle.remove()


def formal_args(sequence_length):
    args = training_parser.parse_args([])
    args.model_type = "uahs"
    args.img_rank = 6
    args.seq_length = sequence_length
    args.temporal_window_radius = None
    args.return_aux = False
    args.debug_uahs = False
    return args


def loss_helper(model, args):
    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.args = args
    trainer.loss_kl = Trainer.loss_kl.__get__(trainer, Trainer)
    trainer.area_weighted_mean = Trainer.area_weighted_mean
    return trainer


def complete_loss(trainer, target, outputs):
    batch_size, time_steps, vertices = target.shape
    loss_final = trainer.loss_kl(
        outputs["saliency"].reshape(batch_size * time_steps, vertices),
        target.reshape(batch_size * time_steps, vertices),
        gt_fix=None,
    ) / (batch_size * time_steps)
    auxiliary = trainer.compute_uahs_losses(target, outputs)
    args = trainer.args
    total = (
        loss_final
        + args.lambda_saliency_l4 * auxiliary["loss_saliency_l4"]
        + args.lambda_saliency_l5 * auxiliary["loss_saliency_l5"]
        + args.lambda_uncertainty_l4 * auxiliary["loss_uncertainty_l4"]
        + args.lambda_uncertainty_l5 * auxiliary["loss_uncertainty_l5"]
        + args.lambda_budget_l5 * auxiliary["loss_budget_l5"]
        + args.lambda_budget_l6 * auxiliary["loss_budget_l6"]
    )
    return {"loss_final": loss_final, **auxiliary, "loss_total": total}


def gradient_report(module):
    gradients = [
        parameter.grad.detach()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return {"exists": False, "finite": False, "norm": None}
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    norm = torch.sqrt(sum(gradient.float().square().sum() for gradient in gradients))
    return {"exists": True, "finite": finite, "norm": float(norm)}


def tensor_statistics(tensor):
    values = tensor.detach().float()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def temporal_mask_iou(mask):
    """Mean face IoU between consecutive hard masks."""
    mask = mask.detach().bool()
    if mask.shape[1] <= 1:
        return None
    intersection = (mask[:, 1:] & mask[:, :-1]).sum(dim=-1).float()
    union = (mask[:, 1:] | mask[:, :-1]).sum(dim=-1).float()
    return float((intersection / union.clamp_min(1)).mean())


def output_report(model, outputs, batch_size, time_steps):
    expected = {
        "saliency": (batch_size, time_steps, model.hierarchy_l5_l6.fine_vertex_count),
        "saliency_l4": (batch_size, time_steps, model.hierarchy_l4_l5.coarse_face_count),
        "uncertainty_l4": (batch_size, time_steps, model.hierarchy_l4_l5.coarse_face_count),
        "refinement_risk_l4": (
            batch_size, time_steps, model.hierarchy_l4_l5.coarse_face_count
        ),
        "budget_l5_pred": (batch_size, time_steps),
        "hard_face_mask_l4": (batch_size, time_steps, model.hierarchy_l4_l5.coarse_face_count),
        "saliency_l5": (batch_size, time_steps, model.hierarchy_l5_l6.coarse_face_count),
        "uncertainty_l5": (batch_size, time_steps, model.hierarchy_l5_l6.coarse_face_count),
        "refinement_risk_l5": (
            batch_size, time_steps, model.hierarchy_l5_l6.coarse_face_count
        ),
        "budget_l6_pred": (batch_size, time_steps),
        "hard_face_mask_l5_effective": (
            batch_size, time_steps, model.hierarchy_l5_l6.coarse_face_count
        ),
        "exit_level": (batch_size, time_steps, model.hierarchy_l5_l6.fine_vertex_count),
    }
    report = {}
    for name, shape in expected.items():
        tensor = outputs[name]
        report[name] = {
            "shape": list(tensor.shape),
            "expected_shape": list(shape),
            "shape_ok": tuple(tensor.shape) == shape,
            "device": str(tensor.device),
            "finite": bool(torch.isfinite(tensor.float()).all()),
        }
    return report


def hierarchy_report(model):
    """Verify the exact face subdivision used by both sparse stages."""
    counts_l4_l5 = torch.bincount(
        model.hierarchy_l4_l5.fine_face_to_coarse_face.cpu(),
        minlength=model.hierarchy_l4_l5.coarse_face_count,
    )
    counts_l5_l6 = torch.bincount(
        model.hierarchy_l5_l6.fine_face_to_coarse_face.cpu(),
        minlength=model.hierarchy_l5_l6.coarse_face_count,
    )
    counts_l4_l6 = torch.bincount(
        model.hierarchy_l4_l6.fine_face_to_coarse_face.cpu(),
        minlength=model.hierarchy_l4_l6.coarse_face_count,
    )
    return {
        "l4_to_l5_exactly_4": bool((counts_l4_l5 == 4).all()),
        "l5_to_l6_exactly_4": bool((counts_l5_l6 == 4).all()),
        "l4_to_l6_exactly_16": bool((counts_l4_l6 == 16).all()),
        "coverage_l4_l5": int(counts_l4_l5.sum()),
        "coverage_l5_l6": int(counts_l5_l6.sum()),
        "coverage_l4_l6": int(counts_l4_l6.sum()),
    }


def routing_report(model, outputs):
    eligible_l5 = model.hierarchy_l4_l5.propagate_coarse_face_values(
        outputs["hard_face_mask_l4"]
    ).bool()
    selected_l5 = outputs["hard_face_mask_l5_effective"].bool()
    tolerance_l1 = float(
        model.hierarchy_l4_l5.coarse_face_areas.max()
        / model.hierarchy_l4_l5.coarse_face_areas.sum()
    ) + 1e-6
    tolerance_l2 = float(
        model.hierarchy_l5_l6.coarse_face_areas.max()
        / model.hierarchy_l5_l6.coarse_face_areas.sum()
    ) + 1e-6
    queries_l5 = int(outputs["selected_spatial_queries_l5"].sum())
    queries_l6 = int(outputs["selected_spatial_queries_l6"].sum())
    dense_l5 = int(outputs["dense_spatial_queries_l5"].sum())
    dense_l6 = int(outputs["dense_spatial_queries_l6"].sum())
    return {
        "parent_constraint": not bool((selected_l5 & ~eligible_l5).any()),
        "area_budget_l1": bool((
            (outputs["selected_area_l1"] - outputs["budget_l5_pred"]).abs()
            <= tolerance_l1
        ).all()),
        "area_budget_l2": bool((
            (outputs["selected_area_l2"] - outputs["budget_l6_pred"]).abs()
            <= tolerance_l2
        ).all()),
        "hierarchical_budget": bool((
            outputs["budget_l6_pred"] <= outputs["selected_area_l1"]
        ).all()),
        "l5_selected_queries_only": (
            0 < queries_l5 < dense_l5
            and model.sparse_refiner_l5.attention.last_query_count == queries_l5
        ),
        "l6_selected_queries_only": (
            0 < queries_l6 < dense_l6
            and model.sparse_refiner_l6.attention.last_query_count == queries_l6
        ),
    }


def run_preflight(sequence_length=12, iterations=3):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the final UAHS GPU preflight")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    args = formal_args(sequence_length)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    model = build_saliency_model(args).to(device).train()
    model.set_epoch(0)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    trainer = loss_helper(model, args)
    vertex_count = model.hierarchy_l5_l6.fine_vertex_count
    inputs = torch.randn(1, sequence_length, vertex_count, 3, device=device)
    target = torch.rand(1, sequence_length, vertex_count, device=device)
    profiler = StageProfiler(model, device)
    report = {
        "configuration": {
            "gpu": torch.cuda.get_device_name(device),
            "total_vram_gb": torch.cuda.get_device_properties(device).total_memory / GIB,
            "batch_size": 1,
            "time_steps": sequence_length,
            "img_rank": 6,
            "levels": [4, 5, 6],
            "embed_dim": model.embed_dim,
            "heads": args.enc_num_heads[0],
            "local_depth": 2,
            "global_depth": 1,
            "sparse_depth_l5": 1,
            "sparse_depth_l6": 1,
            "temporal_window_radius": args.temporal_window_radius,
            "precision": "fp32",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "memory_before": memory_snapshot(device),
        "hierarchy": hierarchy_report(model),
        "refinement_heads_absent": not any(
            "refinement_head" in name or "refine_" in name
            for name, _parameter in model.named_parameters()
        ),
        "iterations": [],
    }
    try:
        for iteration in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            iteration_start = time.perf_counter()
            profiler.phase = f"iteration_{iteration + 1}_forward"
            forward_start = time.perf_counter()
            outputs = model(inputs, return_aux=True)
            synchronize(device)
            forward_seconds = time.perf_counter() - forward_start
            after_forward = memory_snapshot(device)

            loss_start = time.perf_counter()
            losses = complete_loss(trainer, target, outputs)
            synchronize(device)
            loss_seconds = time.perf_counter() - loss_start

            profiler.phase = f"iteration_{iteration + 1}_backward"
            backward_start = time.perf_counter()
            losses["loss_total"].backward()
            synchronize(device)
            backward_seconds = time.perf_counter() - backward_start
            after_backward = memory_snapshot(device)

            gradients = {
                name: gradient_report(module)
                for name, module in {
                    "coarse_local_encoder": model.coarse_local_encoder,
                    "coarse_global_block": model.coarse_global_block,
                    "sparse_refiner_l5": model.sparse_refiner_l5,
                    "sparse_refiner_l6": model.sparse_refiner_l6,
                    "uncertainty_head_l4": model.uncertainty_head_l4,
                    "uncertainty_head_l5": model.uncertainty_head_l5,
                    "budget_head_l4": model.budget_head_l4,
                    "budget_head_l5": model.budget_head_l5,
                    "saliency_head_l4": model.saliency_head_l4,
                    "saliency_head_l5": model.saliency_head_l5,
                    "final_saliency_head": model.output_proj,
                }.items()
            }
            optimizer.step()
            synchronize(device)
            total_seconds = time.perf_counter() - iteration_start
            iteration_report = {
                "iteration": iteration + 1,
                "timing_seconds": {
                    "forward": forward_seconds,
                    "loss": loss_seconds,
                    "backward": backward_seconds,
                    "total_with_optimizer": total_seconds,
                },
                "memory_after_forward": after_forward,
                "memory_after_backward": after_backward,
                "memory_after_step": memory_snapshot(device),
                "losses": {
                    name: float(value.detach())
                    for name, value in losses.items()
                    if name.startswith("loss_")
                },
                "all_losses_finite": all(
                    bool(torch.isfinite(value))
                    for name, value in losses.items()
                    if name.startswith("loss_")
                ),
                "uncertainty_l4": tensor_statistics(outputs["uncertainty_l4"]),
                "uncertainty_l5": tensor_statistics(outputs["uncertainty_l5"]),
                "budget_l5_pred": tensor_statistics(outputs["budget_l5_pred"]),
                "budget_l5_target": float(losses["budget_l5_target"]),
                "budget_l6_pred": tensor_statistics(outputs["budget_l6_pred"]),
                "budget_l6_target": float(losses["budget_l6_target"]),
                "budget_l6_alpha_pred": float(losses["budget_l6_alpha_pred"]),
                "budget_l6_alpha_target": float(losses["budget_l6_alpha_target"]),
                "temporal_mask_iou_l4": temporal_mask_iou(
                    outputs["hard_face_mask_l4"]
                ),
                "temporal_mask_iou_l5": temporal_mask_iou(
                    outputs["hard_face_mask_l5_effective"]
                ),
                "selected_area_l1": float(outputs["selected_area_l1"].mean()),
                "selected_area_l2": float(outputs["selected_area_l2"].mean()),
                "selected_vertex_ratio_l5": float(
                    outputs["selected_vertex_ratio_l5"].mean()
                ),
                "selected_vertex_ratio_l6": float(
                    outputs["selected_vertex_ratio_l6"].mean()
                ),
                "selected_queries_l5": int(
                    outputs["selected_spatial_queries_l5"].sum()
                ),
                "selected_queries_l6": int(
                    outputs["selected_spatial_queries_l6"].sum()
                ),
                "active_faces_l5": int(
                    outputs["active_refinement_faces_l5"].sum()
                ),
                "active_faces_l6": int(
                    outputs["active_refinement_faces_l6"].sum()
                ),
                "spatial_query_reduction_l5": float(
                    outputs["spatial_query_reduction_l5"].mean()
                ),
                "spatial_query_reduction_l6": float(
                    outputs["spatial_query_reduction_l6"].mean()
                ),
                "estimated_sparse_attention_flops_l5": int(
                    outputs["estimated_sparse_attention_flops_l5"].sum()
                ),
                "estimated_sparse_attention_flops_l6": int(
                    outputs["estimated_sparse_attention_flops_l6"].sum()
                ),
                "estimated_dense_attention_flops_l5": int(
                    outputs["estimated_dense_attention_flops_l5"].sum()
                ),
                "estimated_dense_attention_flops_l6": int(
                    outputs["estimated_dense_attention_flops_l6"].sum()
                ),
                "gradients": gradients,
                "routing": routing_report(model, outputs),
            }
            if iteration == 0:
                iteration_report["outputs"] = output_report(
                    model, outputs, 1, sequence_length
                )
            report["iterations"].append(iteration_report)
            del outputs, losses

        model.eval()
        inference_times = []
        inference_output = None
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            # One allocator/kernel warm-up followed by five synchronized runs.
            inference_output = model(inputs, return_aux=True)
            synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(5):
                synchronize(device)
                inference_start = time.perf_counter()
                inference_output = model(inputs, return_aux=True)
                synchronize(device)
                inference_times.append(time.perf_counter() - inference_start)
        report["inference"] = {
            "runs": len(inference_times),
            "mean_seconds": sum(inference_times) / len(inference_times),
            "min_seconds": min(inference_times),
            "max_seconds": max(inference_times),
            "memory": memory_snapshot(device),
            "saliency_finite": bool(torch.isfinite(
                inference_output["saliency"]
            ).all()),
        }
        del inference_output
        report["stage_memory_events"] = profiler.events
        hierarchy_ok = all(
            value for key, value in report["hierarchy"].items()
            if key.endswith(("exactly_4", "exactly_16"))
        )
        report["pass"] = (
            hierarchy_ok
            and report["refinement_heads_absent"]
            and all(
                iteration["all_losses_finite"]
                and all(
                    output["shape_ok"] and output["finite"]
                    for output in iteration.get("outputs", {}).values()
                )
                and all(iteration["routing"].values())
                and all(
                    gradient["exists"] and gradient["finite"]
                    for gradient in iteration["gradients"].values()
                )
                for iteration in report["iterations"]
            )
        )
        return report
    except torch.cuda.OutOfMemoryError as error:
        report["pass"] = False
        report["oom"] = {
            "phase": profiler.phase,
            "last_stage": profiler.last_stage,
            "memory": memory_snapshot(device),
            "message": str(error),
        }
        return report
    finally:
        profiler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence_length", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", default="log/uahs_uncertainty_only_preflight.json")
    arguments = parser.parse_args()
    report = run_preflight(arguments.sequence_length, arguments.iterations)
    output_parent = os.path.dirname(os.path.abspath(arguments.output))
    os.makedirs(output_parent, exist_ok=True)
    with open(arguments.output, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
    console_report = {
        "configuration": report["configuration"],
        "iterations": report["iterations"],
        "inference": report.get("inference"),
        "pass": report["pass"],
        "oom": report.get("oom"),
        "detailed_report": arguments.output,
    }
    print(json.dumps(console_report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
