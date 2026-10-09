"""Geometry-aware targets and metrics shared by UAHS training/evaluation."""

import torch


@torch.no_grad()
def build_error_supervised_risk(
        target,
        prediction,
        error_threshold,
        error_temperature,
):
    """Build per-face soft refinement demand from detached prediction error."""
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have identical shapes")
    if error_temperature <= 0:
        raise ValueError("error_temperature must be positive")
    residual = (target - prediction.detach()).abs()
    return torch.sigmoid(
        (residual - float(error_threshold)) / float(error_temperature)
    )


@torch.no_grad()
def build_error_supervised_budget(
        target,
        prediction,
        face_areas,
        error_threshold,
        error_temperature,
        eligible_mask=None,
):
    """Build an area-weighted soft refinement demand from prediction error.

    The denominator is always the complete sphere. When ``eligible_mask`` is
    supplied, only eligible faces contribute to the numerator; consequently a
    child-stage target cannot request area outside its parent-stage region.
    """
    if target.shape[-1] != face_areas.numel():
        raise ValueError("target and face_areas have different face counts")
    demand = build_error_supervised_risk(
        target,
        prediction,
        error_threshold,
        error_temperature,
    )
    if eligible_mask is not None:
        if eligible_mask.shape != demand.shape:
            raise ValueError("eligible_mask must have the same shape as target")
        demand = demand * eligible_mask.detach().to(demand.dtype)
    areas = face_areas.to(device=demand.device, dtype=demand.dtype)
    return (demand * areas).sum(dim=-1) / areas.sum()


@torch.no_grad()
def area_weighted_calibration_error(
        prediction,
        target,
        face_areas,
        eligible_mask=None,
        num_bins=10,
        epsilon=1e-8,
):
    """Return spherical-area-weighted ECE for local refinement risk."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.shape[-1] != face_areas.numel():
        raise ValueError("prediction and face_areas have different face counts")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    prediction = prediction.detach()
    target = target.detach().to(
        device=prediction.device, dtype=prediction.dtype
    )
    areas = face_areas.to(device=prediction.device, dtype=prediction.dtype)
    weights = areas.reshape(*([1] * (prediction.ndim - 1)), -1)
    weights = weights.expand_as(prediction)
    if eligible_mask is not None:
        if eligible_mask.shape != prediction.shape:
            raise ValueError("eligible_mask must match prediction")
        weights = weights * eligible_mask.detach().to(weights.dtype)
    flat_prediction = prediction.reshape(-1)
    flat_target = target.reshape(-1)
    flat_weights = weights.reshape(-1)
    bin_indices = torch.clamp(
        (prediction.clamp(0, 1) * num_bins).long(),
        max=num_bins - 1,
    ).reshape(-1)
    bin_weights = prediction.new_zeros(num_bins).scatter_add_(
        0, bin_indices, flat_weights
    )
    prediction_totals = prediction.new_zeros(num_bins).scatter_add_(
        0, bin_indices, flat_prediction * flat_weights
    )
    target_totals = prediction.new_zeros(num_bins).scatter_add_(
        0, bin_indices, flat_target * flat_weights
    )
    denominator = bin_weights.clamp_min(epsilon)
    bin_error = (
        prediction_totals / denominator - target_totals / denominator
    ).abs()
    total_weight = bin_weights.sum().clamp_min(epsilon)
    return (bin_weights / total_weight * bin_error).sum()


@torch.no_grad()
def build_fixed_area_target(
        scores,
        face_areas,
        target_ratio,
        eligible_mask=None,
):
    """Select highest-score faces whose global spherical area is nearest a ratio.

    Args:
        scores: Tensor shaped ``[..., F]``. Higher values are selected first.
        face_areas: Positive spherical areas shaped ``[F]``.
        target_ratio: Desired fraction of the complete sphere, not merely the
            eligible subset. May be a scalar or a tensor broadcastable to the
            leading dimensions of ``scores`` for per-frame matched budgets.
        eligible_mask: Optional boolean tensor shaped like ``scores``. Faces
            outside this mask can never be selected.
    """
    if scores.shape[-1] != face_areas.numel():
        raise ValueError("scores and face_areas have different face counts")
    if bool((face_areas <= 0).any()):
        raise ValueError("face_areas must be positive")

    ratio = torch.as_tensor(
        target_ratio, device=scores.device, dtype=scores.dtype
    ).detach()
    try:
        ratio = torch.ones(
            scores.shape[:-1], device=scores.device, dtype=scores.dtype
        ) * ratio
    except RuntimeError as error:
        raise ValueError(
            "target_ratio must be scalar or broadcastable to scores.shape[:-1]"
        ) from error
    if bool(((ratio < 0) | (ratio > 1)).any()):
        raise ValueError("target_ratio must be between zero and one")

    flat_scores = scores.detach().reshape(-1, scores.shape[-1])
    if eligible_mask is None:
        flat_eligible = torch.ones_like(flat_scores, dtype=torch.bool)
    else:
        if eligible_mask.shape != scores.shape:
            raise ValueError("eligible_mask must have the same shape as scores")
        flat_eligible = eligible_mask.detach().reshape_as(flat_scores).bool()
    masked_scores = flat_scores.masked_fill(~flat_eligible, float("-inf"))
    order = torch.argsort(masked_scores, dim=-1, descending=True)
    ordered_eligible = torch.gather(flat_eligible, -1, order)
    areas = face_areas.to(device=scores.device, dtype=scores.dtype)
    ordered_areas = areas[order] * ordered_eligible.to(dtype=scores.dtype)
    cumulative_area = ordered_areas.cumsum(dim=-1)
    area_options = torch.cat(
        (cumulative_area.new_zeros((cumulative_area.shape[0], 1)), cumulative_area),
        dim=-1,
    )
    target_area = ratio.reshape(-1, 1) * areas.sum()
    selected_count = (area_options - target_area).abs().argmin(dim=-1)
    positions = torch.arange(
        flat_scores.shape[-1], device=scores.device
    ).reshape(1, -1)
    selected_in_order = (
        positions < selected_count.reshape(-1, 1)
    ) & ordered_eligible
    target = torch.zeros_like(flat_scores)
    target.scatter_(
        -1, order, selected_in_order.to(dtype=target.dtype)
    )
    return target.reshape_as(scores)


def area_weighted_binary_metrics(prediction, target, face_areas, epsilon=1e-8):
    """Return area-weighted IoU, precision, and recall per leading sample."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.shape[-1] != face_areas.numel():
        raise ValueError("Unexpected face count")
    prediction = prediction.bool()
    target = target.bool()
    areas = face_areas.to(device=prediction.device, dtype=torch.float32)
    intersection = (prediction & target).to(areas.dtype) * areas
    union = (prediction | target).to(areas.dtype) * areas
    predicted_area = prediction.to(areas.dtype) * areas
    target_area = target.to(areas.dtype) * areas
    intersection = intersection.sum(dim=-1)
    return {
        "iou": intersection / (union.sum(dim=-1) + epsilon),
        "precision": intersection / (predicted_area.sum(dim=-1) + epsilon),
        "recall": intersection / (target_area.sum(dim=-1) + epsilon),
    }


def _average_tied_ranks(values):
    """Return zero-based ranks, assigning each tied group its average rank."""
    original_shape = values.shape
    flat = values.reshape(-1, original_shape[-1])
    sorted_values, order = torch.sort(flat, dim=-1)
    item_count = flat.shape[-1]
    positions = torch.arange(
        item_count, device=values.device, dtype=torch.float32
    ).reshape(1, -1).expand_as(sorted_values)

    group_start = torch.ones_like(sorted_values, dtype=torch.bool)
    if item_count > 1:
        group_start[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    group_ids = group_start.cumsum(dim=-1) - 1
    rank_sums = torch.zeros_like(positions)
    group_counts = torch.zeros_like(positions)
    rank_sums.scatter_add_(-1, group_ids, positions)
    group_counts.scatter_add_(-1, group_ids, torch.ones_like(positions))
    average_by_group = rank_sums / group_counts.clamp_min(1)
    sorted_ranks = torch.gather(average_by_group, -1, group_ids)
    ranks = torch.zeros_like(sorted_ranks)
    ranks.scatter_(-1, order, sorted_ranks)
    return ranks.reshape(original_shape)


def per_frame_spearman(first, second, epsilon=1e-8):
    """Compute tie-aware Spearman correlation along the final dimension.

    A frame containing non-finite values or zero rank variance is undefined and
    is returned as NaN. Diagnostic accumulators exclude and count these frames.
    """
    if first.shape != second.shape:
        raise ValueError("Spearman inputs must have identical shapes")
    if first.ndim == 0:
        raise ValueError("Spearman inputs require a final observation dimension")
    if first.shape[-1] < 2:
        return torch.full(
            first.shape[:-1], float("nan"), device=first.device,
            dtype=torch.float32,
        )
    finite = torch.isfinite(first).all(dim=-1) & torch.isfinite(second).all(dim=-1)
    first_rank = _average_tied_ranks(first)
    second_rank = _average_tied_ranks(second)
    first_rank = first_rank - first_rank.mean(dim=-1, keepdim=True)
    second_rank = second_rank - second_rank.mean(dim=-1, keepdim=True)
    numerator = (first_rank * second_rank).sum(dim=-1)
    first_variation = first_rank.square().sum(dim=-1)
    second_variation = second_rank.square().sum(dim=-1)
    denominator = torch.sqrt(first_variation * second_variation)
    correlation = numerator / denominator.clamp_min(epsilon)
    defined = finite & (first_variation > 0) & (second_variation > 0)
    return torch.where(
        defined, correlation.clamp(-1, 1), torch.full_like(correlation, float("nan"))
    )


@torch.no_grad()
def budget_regression_metrics(prediction, target, epsilon=1e-12):
    """Summarize paired per-frame budget predictions and targets."""
    if prediction.numel() != target.numel():
        raise ValueError("prediction and target must contain the same number of values")
    prediction = prediction.detach().reshape(-1).to(dtype=torch.float64)
    target = target.detach().reshape(-1).to(
        device=prediction.device, dtype=torch.float64
    )
    finite = torch.isfinite(prediction) & torch.isfinite(target)
    prediction, target = prediction[finite], target[finite]
    if prediction.numel() == 0:
        raise ValueError("budget diagnostics require at least one finite pair")

    difference = prediction - target
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    pearson_denominator = torch.sqrt(
        centered_prediction.square().sum() * centered_target.square().sum()
    )
    if float(pearson_denominator) <= epsilon:
        pearson = float("nan")
    else:
        pearson = float(
            (centered_prediction * centered_target).sum()
            / pearson_denominator
        )
    spearman = float(per_frame_spearman(
        prediction.unsqueeze(0), target.unsqueeze(0), epsilon=epsilon
    ).squeeze(0))
    quantiles = prediction.new_tensor((0.1, 0.5, 0.9))
    prediction_quantiles = torch.quantile(prediction, quantiles)
    target_quantiles = torch.quantile(target, quantiles)

    return {
        "pred_mean": float(prediction.mean()),
        "pred_std": float(prediction.std(unbiased=False)),
        "target_mean": float(target.mean()),
        "target_std": float(target.std(unbiased=False)),
        "mae": float(difference.abs().mean()),
        "rmse": float(difference.square().mean().sqrt()),
        "pearson": pearson,
        "spearman": spearman,
        "under_budget_ratio": float((prediction < target).to(torch.float64).mean()),
        "pred_p10": float(prediction_quantiles[0]),
        "pred_p50": float(prediction_quantiles[1]),
        "pred_p90": float(prediction_quantiles[2]),
        "target_p10": float(target_quantiles[0]),
        "target_p50": float(target_quantiles[1]),
        "target_p90": float(target_quantiles[2]),
        "finite_count": int(prediction.numel()),
        "nonfinite_count": int((~finite).sum()),
    }
