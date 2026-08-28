"""Streaming evaluation diagnostics for the dense UAHS prototype."""

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


class AdaptiveDiagnosticsAccumulator:
    """Aggregate valid-frame UAHS statistics and calibration evidence."""

    def __init__(self, target_refine_ratio):
        self.target_refine_ratio = float(target_refine_ratio)
        self.uncertainty = RunningMoments()
        self.coarse_absolute_error = RunningMoments()
        self.refine_score = RunningMoments()
        self.fine_face_gate = RunningMoments()
        self.area_refine_ratio = RunningMoments()
        self.coarse_upsampled_kl = RunningMoments()
        self.final_saliency_kl = RunningMoments()
        self.uncertainty_error = RunningPairMoments()
        self.gate_below_005 = 0
        self.gate_above_095 = 0
        self.gate_count = 0
        self.frame_count = 0

    def update(self, outputs, ground_truth, valid_lengths, model):
        required = {
            "saliency",
            "coarse_saliency",
            "uncertainty",
            "refine_score",
            "fine_face_gate",
            "area_refine_ratio",
        }
        missing = required.difference(outputs)
        if missing:
            raise KeyError(f"Adaptive outputs are missing: {sorted(missing)}")

        coarse_target = model.aggregate_img_values_to_coarse_faces(ground_truth)
        coarse_error = (coarse_target - outputs["coarse_saliency"]).abs()
        coarse_upsampled = model.upsample_coarse_values_to_img(
            outputs["coarse_saliency"]
        )
        batch_size, time_steps = ground_truth.shape[:2]
        if len(valid_lengths) != batch_size:
            raise ValueError("valid_lengths must have one entry per sample")

        for sample_index, valid_length in enumerate(valid_lengths):
            valid_length = int(valid_length)
            if not 0 <= valid_length <= time_steps:
                raise ValueError("A valid_length is outside the sequence range")
            if valid_length == 0:
                continue
            time_slice = (sample_index, slice(0, valid_length))
            uncertainty = outputs["uncertainty"][time_slice]
            refine_score = outputs["refine_score"][time_slice]
            fine_face_gate = outputs["fine_face_gate"][time_slice]
            area_ratio = outputs["area_refine_ratio"][time_slice]
            error = coarse_error[time_slice]

            self.uncertainty.update(uncertainty)
            self.coarse_absolute_error.update(error)
            self.refine_score.update(refine_score)
            self.fine_face_gate.update(fine_face_gate)
            self.area_refine_ratio.update(area_ratio)
            self.uncertainty_error.update(uncertainty, error)
            finite_gate = fine_face_gate[torch.isfinite(fine_face_gate)]
            self.gate_below_005 += int((finite_gate < 0.05).sum().item())
            self.gate_above_095 += int((finite_gate > 0.95).sum().item())
            self.gate_count += finite_gate.numel()

            target = ground_truth[time_slice]
            self.coarse_upsampled_kl.update(
                per_frame_kl(coarse_upsampled[time_slice], target)
            )
            self.final_saliency_kl.update(
                per_frame_kl(outputs["saliency"][time_slice], target)
            )
            self.frame_count += valid_length

    def summary(self):
        uncertainty_summary = self.uncertainty.summary()
        error_summary = self.coarse_absolute_error.summary()
        area_summary = self.area_refine_ratio.summary()
        coarse_kl = self.coarse_upsampled_kl.summary()
        final_kl = self.final_saliency_kl.summary()
        area_mean = area_summary["mean"]
        coarse_mean = coarse_kl["mean"]
        final_mean = final_kl["mean"]
        kl_reduction = (
            None if coarse_mean is None or final_mean is None
            else coarse_mean - final_mean
        )
        relative_kl_reduction = (
            None if kl_reduction is None or coarse_mean == 0
            else kl_reduction / coarse_mean
        )
        return {
            "valid_frames": self.frame_count,
            "uncertainty": {
                **uncertainty_summary,
                "coarse_absolute_error": error_summary,
                "coarse_error_pearson": self.uncertainty_error.correlation(),
                "mean_scale_to_error_ratio": (
                    None
                    if uncertainty_summary["mean"] is None
                    or error_summary["mean"] in (None, 0)
                    else uncertainty_summary["mean"] / error_summary["mean"]
                ),
            },
            "refine_score": self.refine_score.summary(),
            "fine_face_gate": {
                **self.fine_face_gate.summary(),
                "fraction_below_0.05": (
                    self.gate_below_005 / self.gate_count
                    if self.gate_count else None
                ),
                "fraction_above_0.95": (
                    self.gate_above_095 / self.gate_count
                    if self.gate_count else None
                ),
            },
            "area_refine_ratio": {
                **area_summary,
                "target": self.target_refine_ratio,
                "mean_minus_target": (
                    None if area_mean is None
                    else area_mean - self.target_refine_ratio
                ),
            },
            "coarse_rank_evidence": {
                "coarse_upsampled_kl": coarse_kl,
                "final_saliency_kl": final_kl,
                "absolute_kl_reduction": kl_reduction,
                "relative_kl_reduction": relative_kl_reduction,
            },
        }


def build_v2_selection_targets(
        outputs,
        ground_truth,
        model,
        target_ratio_l1,
        target_ratio_l2,
):
    """Build oracle fixed-area targets used only by loss/evaluation."""
    target_l4 = model.aggregate_img_values_to_l4_faces(ground_truth)
    target_l5 = model.aggregate_img_values_to_l5_faces(ground_truth)
    error_l4 = (target_l4 - outputs["saliency_l4"]).abs()
    error_l5 = (target_l5 - outputs["saliency_l5"]).abs()
    selection_l4 = build_fixed_area_target(
        error_l4,
        model.hierarchy_l4_l5.coarse_face_areas,
        target_ratio_l1,
    )
    eligible_l5 = model.hierarchy_l4_l5.propagate_coarse_face_values(
        selection_l4
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
    """Streaming uncertainty, gate, and selection evidence for one level."""

    def __init__(self, target_ratio):
        self.target_ratio = float(target_ratio)
        self.uncertainty = RunningMoments()
        self.error = RunningMoments()
        self.refine_score = RunningMoments()
        self.gate = RunningMoments()
        self.area_ratio = RunningMoments()
        self.uncertainty_error = RunningPairMoments()
        self.spearman = RunningMoments()
        self.selection_iou = RunningMoments()
        self.selection_precision = RunningMoments()
        self.selection_recall = RunningMoments()

    def update(
            self,
            uncertainty,
            error,
            refine_score,
            gate,
            area_ratio,
            predicted_selection,
            target_selection,
            face_areas,
    ):
        self.uncertainty.update(uncertainty)
        self.error.update(error)
        self.refine_score.update(refine_score)
        self.gate.update(gate)
        self.area_ratio.update(area_ratio)
        self.uncertainty_error.update(uncertainty, error)
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
            },
            "refine_score": self.refine_score.summary(),
            "gate": self.gate.summary(),
            "actual_refinement_area": {
                **area_ratio,
                "target": self.target_ratio,
                "mean_minus_target": (
                    None if area_ratio["mean"] is None
                    else area_ratio["mean"] - self.target_ratio
                ),
            },
            "selection": {
                "iou": self.selection_iou.summary(),
                "precision": self.selection_precision.summary(),
                "recall": self.selection_recall.summary(),
            },
        }


class AdaptiveDiagnosticsAccumulatorV2:
    """Streaming diagnostics for recursive rank-(r-2)->(r-1)->r UAHS."""

    def __init__(self, target_ratio_l1, target_ratio_l2):
        self.target_ratio_l1 = float(target_ratio_l1)
        self.target_ratio_l2 = float(target_ratio_l2)
        self.level_l4 = HierarchicalLevelDiagnostics(target_ratio_l1)
        self.level_l5 = HierarchicalLevelDiagnostics(target_ratio_l2)
        self.saliency_l4_kl = RunningMoments()
        self.saliency_l5_kl = RunningMoments()
        self.final_saliency_kl = RunningMoments()
        self.frame_count = 0

    def update(self, outputs, ground_truth, valid_lengths, model):
        required = {
            "saliency",
            "saliency_l4",
            "uncertainty_l4",
            "refine_score_l4",
            "gate_l4_parent",
            "area_ratio_l1",
            "saliency_l5",
            "uncertainty_l5",
            "refine_score_l5",
            "gate_l5_to_l6_effective",
            "area_ratio_l2",
        }
        missing = required.difference(outputs)
        if missing:
            raise KeyError(f"UAHS-V2 outputs are missing: {sorted(missing)}")
        targets = build_v2_selection_targets(
            outputs,
            ground_truth,
            model,
            self.target_ratio_l1,
            self.target_ratio_l2,
        )
        predicted_l4 = build_fixed_area_target(
            outputs["refine_score_l4"],
            model.hierarchy_l4_l5.coarse_face_areas,
            self.target_ratio_l1,
        )
        predicted_parent_l5 = model.hierarchy_l4_l5.propagate_coarse_face_values(
            predicted_l4
        ).bool()
        predicted_l5 = build_fixed_area_target(
            outputs["refine_score_l5"],
            model.hierarchy_l5_l6.coarse_face_areas,
            self.target_ratio_l2,
            eligible_mask=predicted_parent_l5,
        )
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
                outputs["refine_score_l4"][valid],
                outputs["gate_l4_parent"][valid],
                outputs["area_ratio_l1"][valid],
                predicted_l4[valid],
                targets["selection_l4"][valid],
                model.hierarchy_l4_l5.coarse_face_areas,
            )
            self.level_l5.update(
                outputs["uncertainty_l5"][valid],
                targets["error_l5"][valid],
                outputs["refine_score_l5"][valid],
                outputs["gate_l5_to_l6_effective"][valid],
                outputs["area_ratio_l2"][valid],
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
        }
