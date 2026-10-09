"""Streaming evaluation diagnostics for final sparse UAHS."""

import math

import torch

from adaptive_objectives import (
    area_weighted_binary_metrics,
    build_fixed_area_target,
    per_frame_spearman,
)

class RunningMoments:
    """Accumulate scalar moments without retaining per-frame tensors."""

    def __init__(self):
        self.count = 0
        self.nonfinite_count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values):
        flat = values.detach().reshape(-1).to(dtype=torch.float64)
        finite = torch.isfinite(flat)
        self.nonfinite_count += int((~finite).sum().item())
        flat = flat[finite]
        if flat.numel() == 0:
            return
        self.count += flat.numel()
        self.total += float(flat.sum().item())
        self.total_square += float(flat.square().sum().item())
        self.minimum = min(self.minimum, float(flat.min().item()))
        self.maximum = max(self.maximum, float(flat.max().item()))

    def summary(self):
        if self.count == 0:
            return {
                "count": 0,
                "nonfinite_count": self.nonfinite_count,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "nonfinite_count": self.nonfinite_count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


class RunningPairMoments:
    """Accumulate moments required for a Pearson correlation."""

    def __init__(self):
        self.count = 0
        self.total_x = 0.0
        self.total_y = 0.0
        self.total_x_square = 0.0
        self.total_y_square = 0.0
        self.total_xy = 0.0

    def update(self, x, y):
        x = x.detach().reshape(-1).to(dtype=torch.float64)
        y = y.detach().reshape(-1).to(dtype=torch.float64)
        finite = torch.isfinite(x) & torch.isfinite(y)
        x, y = x[finite], y[finite]
        if x.numel() == 0:
            return
        self.count += x.numel()
        self.total_x += float(x.sum().item())
        self.total_y += float(y.sum().item())
        self.total_x_square += float(x.square().sum().item())
        self.total_y_square += float(y.square().sum().item())
        self.total_xy += float((x * y).sum().item())

    def correlation(self):
        if self.count < 2:
            return None
        mean_x = self.total_x / self.count
        mean_y = self.total_y / self.count
        variance_x = self.total_x_square / self.count - mean_x * mean_x
        variance_y = self.total_y_square / self.count - mean_y * mean_y
        if variance_x <= 0 or variance_y <= 0:
            return None
        covariance = self.total_xy / self.count - mean_x * mean_y
        correlation = covariance / math.sqrt(variance_x * variance_y)
        return max(-1.0, min(1.0, correlation))


class UncertaintyCalibrationBins:
    """Accumulate mean predicted scale and mean absolute error in fixed bins."""

    def __init__(self, edges=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)):
        self.edges = tuple(float(value) for value in edges)
        self.counts = [0 for _ in range(len(self.edges))]
        self.uncertainty_totals = [0.0 for _ in self.counts]
        self.error_totals = [0.0 for _ in self.counts]

    def update(self, uncertainty, error):
        uncertainty = uncertainty.detach().reshape(-1).to(dtype=torch.float64)
        error = error.detach().reshape(-1).to(dtype=torch.float64)
        finite = torch.isfinite(uncertainty) & torch.isfinite(error)
        uncertainty, error = uncertainty[finite], error[finite]
        if uncertainty.numel() == 0:
            return
        boundaries = uncertainty.new_tensor(self.edges[1:])
        bins = torch.bucketize(uncertainty, boundaries)
        for index in range(len(self.counts)):
            selected = bins == index
            if not bool(selected.any()):
                continue
            self.counts[index] += int(selected.sum().item())
            self.uncertainty_totals[index] += float(uncertainty[selected].sum())
            self.error_totals[index] += float(error[selected].sum())

    def summary(self):
        output = []
        for index, lower in enumerate(self.edges):
            upper = self.edges[index + 1] if index + 1 < len(self.edges) else None
            count = self.counts[index]
            output.append({
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_uncertainty": (
                    self.uncertainty_totals[index] / count if count else None
                ),
                "mean_absolute_error": (
                    self.error_totals[index] / count if count else None
                ),
            })
        return output


def per_frame_kl(prediction, target, epsilon=2.2204e-16):
    """Return KLD(target || prediction) independently for every frame."""
    prediction = prediction.float()
    target = target.float()
    prediction = prediction / (
        prediction.sum(dim=-1, keepdim=True) + epsilon
    )
    target = target / (target.sum(dim=-1, keepdim=True) + epsilon)
    return (
        target * torch.log(target / (prediction + epsilon) + epsilon)
    ).sum(dim=-1)


def build_uahs_selection_targets(
        outputs,
        ground_truth,
        model,
        target_ratio_l1=None,
        target_ratio_l2=None,
):
    """Build error-ranked oracle masks at the model's predicted budgets."""
    target_l4 = model.aggregate_img_values_to_l4_faces(ground_truth)
    target_l5 = model.aggregate_img_values_to_l5_faces(ground_truth)
    error_l4 = (target_l4 - outputs["saliency_l4"]).abs()
    error_l5 = (target_l5 - outputs["saliency_l5"]).abs()
    if target_ratio_l1 is None:
        target_ratio_l1 = outputs["budget_l5_pred"]
    if target_ratio_l2 is None:
        target_ratio_l2 = outputs["budget_l6_pred"]
    selection_l4 = build_fixed_area_target(
        error_l4,
        model.hierarchy_l4_l5.coarse_face_areas,
        target_ratio_l1,
    )
    eligible_l5 = outputs.get("eligible_face_mask_l5")
    if eligible_l5 is None:
        eligible_l5 = model.hierarchy_l4_l5.propagate_coarse_face_values(
            outputs["hard_face_mask_l4"]
        ).bool()
    selection_l5 = build_fixed_area_target(
        error_l5,
        model.hierarchy_l5_l6.coarse_face_areas,
        target_ratio_l2,
        eligible_mask=eligible_l5,
    )
    return {
        "target_l4": target_l4,
        "target_l5": target_l5,
        "error_l4": error_l4,
        "error_l5": error_l5,
        "selection_l4": selection_l4,
        "selection_l5": selection_l5,
    }


class HierarchicalLevelDiagnostics:
    """Streaming uncertainty and hard-selection evidence for one level."""

    def __init__(self, target_ratio):
        self.initial_target_ratio = float(target_ratio)
        self.uncertainty = RunningMoments()
        self.error = RunningMoments()
        self.hard_mask = RunningMoments()
        self.area_ratio = RunningMoments()
        self.predicted_budget = RunningMoments()
        self.area_minus_budget = RunningMoments()
        self.uncertainty_error = RunningPairMoments()
        self.spearman = RunningMoments()
        self.selection_iou = RunningMoments()
        self.selection_precision = RunningMoments()
        self.selection_recall = RunningMoments()
        self.calibration = UncertaintyCalibrationBins()

    def update(
            self,
            uncertainty,
            error,
            hard_mask,
            area_ratio,
            predicted_budget,
            predicted_selection,
            target_selection,
            face_areas,
    ):
        self.uncertainty.update(uncertainty)
        self.error.update(error)
        self.hard_mask.update(hard_mask)
        self.area_ratio.update(area_ratio)
        self.predicted_budget.update(predicted_budget)
        self.area_minus_budget.update(area_ratio - predicted_budget)
        self.uncertainty_error.update(uncertainty, error)
        self.calibration.update(uncertainty, error)
        self.spearman.update(per_frame_spearman(uncertainty, error))
        selection = area_weighted_binary_metrics(
            predicted_selection, target_selection, face_areas
        )
        self.selection_iou.update(selection["iou"])
        self.selection_precision.update(selection["precision"])
        self.selection_recall.update(selection["recall"])

    def summary(self):
        uncertainty = self.uncertainty.summary()
        error = self.error.summary()
        area_ratio = self.area_ratio.summary()
        spearman = self.spearman.summary()
        return {
            "uncertainty": {
                **uncertainty,
                "error": error,
                "pearson": self.uncertainty_error.correlation(),
                "spearman_per_frame": {
                    **spearman,
                    "undefined_count": spearman["nonfinite_count"],
                },
                "mean_scale_to_error_ratio": (
                    None
                    if uncertainty["mean"] is None or error["mean"] in (None, 0)
                    else uncertainty["mean"] / error["mean"]
                ),
                "calibration_bins": self.calibration.summary(),
            },
            "hard_mask": self.hard_mask.summary(),
            "actual_refinement_area": {
                **area_ratio,
                "initial_legacy_budget": self.initial_target_ratio,
                "predicted_budget": self.predicted_budget.summary(),
                "actual_minus_predicted": self.area_minus_budget.summary(),
            },
            "selection": {
                "iou": self.selection_iou.summary(),
                "precision": self.selection_precision.summary(),
                "recall": self.selection_recall.summary(),
            },
        }


class UAHSDiagnosticsAccumulator:
    """Streaming diagnostics for hard sparse rank-(r-2)->(r-1)->r UAHS."""

    def __init__(self, target_ratio_l1, target_ratio_l2):
        self.target_ratio_l1 = float(target_ratio_l1)
        self.target_ratio_l2 = float(target_ratio_l2)
        self.level_l4 = HierarchicalLevelDiagnostics(target_ratio_l1)
        self.level_l5 = HierarchicalLevelDiagnostics(target_ratio_l2)
        self.saliency_l4_kl = RunningMoments()
        self.saliency_l5_kl = RunningMoments()
        self.final_saliency_kl = RunningMoments()
        self.selected_vertex_ratio_l5 = RunningMoments()
        self.selected_vertex_ratio_l6 = RunningMoments()
        self.selected_queries_l5 = RunningMoments()
        self.selected_queries_l6 = RunningMoments()
        self.active_faces_l5 = RunningMoments()
        self.active_faces_l6 = RunningMoments()
        self.query_reduction_l5 = RunningMoments()
        self.query_reduction_l6 = RunningMoments()
        self.sparse_attention_flops_l5 = RunningMoments()
        self.sparse_attention_flops_l6 = RunningMoments()
        self.dense_attention_flops_l5 = RunningMoments()
        self.dense_attention_flops_l6 = RunningMoments()
        self.frame_count = 0

    def update(self, outputs, ground_truth, valid_lengths, model):
        required = {
            "saliency",
            "saliency_l4",
            "uncertainty_l4",
            "budget_l5_pred",
            "hard_face_mask_l4",
            "selected_area_l1",
            "saliency_l5",
            "uncertainty_l5",
            "budget_l6_pred",
            "hard_face_mask_l5_effective",
            "selected_area_l2",
            "selected_vertex_ratio_l5",
            "selected_vertex_ratio_l6",
            "selected_spatial_queries_l5",
            "selected_spatial_queries_l6",
            "active_refinement_faces_l5",
            "active_refinement_faces_l6",
            "spatial_query_reduction_l5",
            "spatial_query_reduction_l6",
            "estimated_sparse_attention_flops_l5",
            "estimated_sparse_attention_flops_l6",
            "estimated_dense_attention_flops_l5",
            "estimated_dense_attention_flops_l6",
        }
        missing = required.difference(outputs)
        if missing:
            raise KeyError(f"UAHS outputs are missing: {sorted(missing)}")
        targets = build_uahs_selection_targets(
            outputs,
            ground_truth,
            model,
        )
        predicted_l4 = outputs["hard_face_mask_l4"]
        predicted_l5 = outputs["hard_face_mask_l5_effective"]
        saliency_l4_up = model.upsample_l4_values_to_img(outputs["saliency_l4"])
        saliency_l5_up = model.upsample_l5_values_to_img(outputs["saliency_l5"])

        batch_size, time_steps = ground_truth.shape[:2]
        if len(valid_lengths) != batch_size:
            raise ValueError("valid_lengths must have one entry per sample")
        for sample_index, valid_length in enumerate(valid_lengths):
            valid_length = int(valid_length)
            if not 0 <= valid_length <= time_steps:
                raise ValueError("A valid_length is outside the sequence range")
            if valid_length == 0:
                continue
            valid = (sample_index, slice(0, valid_length))
            self.level_l4.update(
                outputs["uncertainty_l4"][valid],
                targets["error_l4"][valid],
                outputs["hard_face_mask_l4"][valid],
                outputs["selected_area_l1"][valid],
                outputs["budget_l5_pred"][valid],
                predicted_l4[valid],
                targets["selection_l4"][valid],
                model.hierarchy_l4_l5.coarse_face_areas,
            )
            self.level_l5.update(
                outputs["uncertainty_l5"][valid],
                targets["error_l5"][valid],
                outputs["hard_face_mask_l5_effective"][valid],
                outputs["selected_area_l2"][valid],
                outputs["budget_l6_pred"][valid],
                predicted_l5[valid],
                targets["selection_l5"][valid],
                model.hierarchy_l5_l6.coarse_face_areas,
            )
            target = ground_truth[valid]
            self.saliency_l4_kl.update(per_frame_kl(saliency_l4_up[valid], target))
            self.saliency_l5_kl.update(per_frame_kl(saliency_l5_up[valid], target))
            self.final_saliency_kl.update(
                per_frame_kl(outputs["saliency"][valid], target)
            )
            self.selected_vertex_ratio_l5.update(
                outputs["selected_vertex_ratio_l5"][valid]
            )
            self.selected_vertex_ratio_l6.update(
                outputs["selected_vertex_ratio_l6"][valid]
            )
            self.selected_queries_l5.update(
                outputs["selected_spatial_queries_l5"][valid]
            )
            self.selected_queries_l6.update(
                outputs["selected_spatial_queries_l6"][valid]
            )
            self.active_faces_l5.update(
                outputs["active_refinement_faces_l5"][valid]
            )
            self.active_faces_l6.update(
                outputs["active_refinement_faces_l6"][valid]
            )
            self.query_reduction_l5.update(
                outputs["spatial_query_reduction_l5"][valid]
            )
            self.query_reduction_l6.update(
                outputs["spatial_query_reduction_l6"][valid]
            )
            self.sparse_attention_flops_l5.update(
                outputs["estimated_sparse_attention_flops_l5"][valid]
            )
            self.sparse_attention_flops_l6.update(
                outputs["estimated_sparse_attention_flops_l6"][valid]
            )
            self.dense_attention_flops_l5.update(
                outputs["estimated_dense_attention_flops_l5"][valid]
            )
            self.dense_attention_flops_l6.update(
                outputs["estimated_dense_attention_flops_l6"][valid]
            )
            self.frame_count += valid_length

    def summary(self):
        return {
            "valid_frames": self.frame_count,
            "level_l4": self.level_l4.summary(),
            "level_l5": self.level_l5.summary(),
            "hierarchical_saliency_kl": {
                "saliency_l4_upsampled": self.saliency_l4_kl.summary(),
                "saliency_l5_upsampled": self.saliency_l5_kl.summary(),
                "final_saliency": self.final_saliency_kl.summary(),
            },
            "sparse_efficiency": {
                "selected_vertex_ratio_l5": self.selected_vertex_ratio_l5.summary(),
                "selected_vertex_ratio_l6": self.selected_vertex_ratio_l6.summary(),
                "selected_queries_l5": self.selected_queries_l5.summary(),
                "selected_queries_l6": self.selected_queries_l6.summary(),
                "active_faces_l5": self.active_faces_l5.summary(),
                "active_faces_l6": self.active_faces_l6.summary(),
                "spatial_query_reduction_l5": self.query_reduction_l5.summary(),
                "spatial_query_reduction_l6": self.query_reduction_l6.summary(),
                "estimated_sparse_attention_flops_l5": (
                    self.sparse_attention_flops_l5.summary()
                ),
                "estimated_sparse_attention_flops_l6": (
                    self.sparse_attention_flops_l6.summary()
                ),
                "estimated_dense_attention_flops_l5": (
                    self.dense_attention_flops_l5.summary()
                ),
                "estimated_dense_attention_flops_l6": (
                    self.dense_attention_flops_l6.summary()
                ),
            },
        }
