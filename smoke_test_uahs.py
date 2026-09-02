"""Geometry, sparsity, gradient, and regression tests for final UAHS."""

from types import SimpleNamespace

import torch

from adaptive_diagnostics import UAHSDiagnosticsAccumulator
from adaptive_objectives import (
    build_error_supervised_budget,
    build_fixed_area_target,
    per_frame_spearman,
)
from network.sphere_model import build_saliency_model
from network.sphere_PSA import (
    GlobalSphereSelfAttention,
    SparseSphereSelfAttention,
    SphereSelfAttention,
)
from train_salient import Trainer
from trimesh_utils import IcoSphereHierarchy, IcoSphereRef


def model_args(model_type="uahs", img_rank=3):
    return SimpleNamespace(
        model_type=model_type,
        mode="vertex",
        img_rank=img_rank,
        scale_factor=2,
        num_scales=1,
        scale_depth=1,
        win_size_coef=2,
        temporal_window_radius=1,
        d_head_coef=1,
        enc_num_heads=[2],
        bottleneck_num_heads=2,
        dec_num_heads=[2],
        abs_pos_enc_in=True,
        abs_pos_enc=True,
        rel_pos_bias=True,
        rel_pos_bias_size=7,
        rel_pos_init_variance=0.0,
        downsample="center",
        upsample="interpolate",
        dr=0.0,
        dpr=0.0,
        adr=0.0,
        aodr=0.0,
        posdr=0.0,
        debug_skip_attn=False,
        append_self=False,
        use_checkpoint=False,
        coarse_pool_type="mean_max",
        target_refine_ratio_l1=0.25,
        target_refine_ratio_l2=0.125,
        budget_l5_min=0.05,
        budget_l5_max=0.50,
        budget_error_threshold_l4=0.05,
        budget_error_threshold_l5=0.05,
        budget_error_temperature_l4=0.02,
        budget_error_temperature_l5=0.02,
        global_query_chunk_size=32,
        hard_selection_warmup_epochs=0,
        return_aux=False,
        debug_uahs=False,
        lambda_saliency_l4=0.15,
        lambda_saliency_l5=0.15,
        lambda_uncertainty_l4=0.05,
        lambda_uncertainty_l5=0.05,
        lambda_budget_l5=1.0,
        lambda_budget_l6=1.0,
    )


def assert_finite(name, tensor):
    assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"


def assert_gradient(name, module):
    gradients = [parameter.grad for parameter in module.parameters()]
    assert any(gradient is not None for gradient in gradients), f"{name}: no gradient"
    for gradient in gradients:
        if gradient is not None:
            assert_finite(f"{name} gradient", gradient)


def tie_aware_spearman_test():
    tied = torch.tensor([[1.0, 1.0, 2.0]])
    increasing = torch.tensor([[1.0, 2.0, 3.0]])
    correlation = per_frame_spearman(tied, increasing)
    assert torch.allclose(correlation, torch.tensor([3 ** 0.5 / 2]), atol=1e-6)
    undefined = per_frame_spearman(torch.ones_like(tied), increasing)
    assert bool(torch.isnan(undefined).all())
    print("tie-aware Spearman and zero-variance handling: OK")


def hierarchy_and_budget_test():
    ref = IcoSphereRef("vertex")
    for coarse_rank, fine_rank, expected in (
            (4, 5, 4),
            (5, 6, 4),
            (4, 6, 16),
    ):
        hierarchy = IcoSphereHierarchy(coarse_rank, fine_rank, ref)
        counts = torch.bincount(
            hierarchy.fine_face_to_coarse_face,
            minlength=hierarchy.coarse_face_count,
        )
        assert bool((counts == expected).all())
        assert counts.sum().item() == hierarchy.fine_face_count

    l4_l5 = IcoSphereHierarchy(4, 5, ref)
    l5_l6 = IcoSphereHierarchy(5, 6, ref)
    score_l4 = torch.randn(1, 2, l4_l5.coarse_face_count)
    mask_l4 = build_fixed_area_target(
        score_l4, l4_l5.coarse_face_areas, 0.25
    )
    parent_l5 = l4_l5.propagate_coarse_face_values(mask_l4).bool()
    score_l5 = torch.randn(1, 2, l5_l6.coarse_face_count)
    mask_l5 = build_fixed_area_target(
        score_l5,
        l5_l6.coarse_face_areas,
        0.125,
        eligible_mask=parent_l5,
    )
    area_l4 = (
        mask_l4 * l4_l5.coarse_face_areas
    ).sum(-1) / l4_l5.coarse_face_areas.sum()
    area_l5 = (
        mask_l5 * l5_l6.coarse_face_areas
    ).sum(-1) / l5_l6.coarse_face_areas.sum()
    tolerance_l4 = float(l4_l5.coarse_face_areas.max() / l4_l5.coarse_face_areas.sum())
    tolerance_l5 = float(l5_l6.coarse_face_areas.max() / l5_l6.coarse_face_areas.sum())
    assert bool(((area_l4 - 0.25).abs() <= tolerance_l4).all())
    assert bool(((area_l5 - 0.125).abs() <= tolerance_l5).all())
    assert not bool((mask_l5.bool() & ~parent_l5).any())
    dynamic_mask = build_fixed_area_target(
        score_l4,
        l4_l5.coarse_face_areas,
        torch.tensor([[0.10, 0.35]]),
    )
    dynamic_area = (
        dynamic_mask * l4_l5.coarse_face_areas
    ).sum(-1) / l4_l5.coarse_face_areas.sum()
    assert float(dynamic_area[0, 0]) < float(dynamic_area[0, 1])
    residual_target = build_error_supervised_budget(
        torch.ones_like(score_l5),
        torch.zeros_like(score_l5),
        l5_l6.coarse_face_areas,
        error_threshold=0.05,
        error_temperature=0.02,
        eligible_mask=parent_l5,
    )
    eligible_area = (
        parent_l5 * l5_l6.coarse_face_areas
    ).sum(-1) / l5_l6.coarse_face_areas.sum()
    assert bool((residual_target <= eligible_area + 1e-6).all())
    print("hierarchy, tensor budgets, error oracle, and parent constraint: OK")


def sparse_attention_equivalence_test():
    torch.manual_seed(0)
    ref = IcoSphereRef("vertex")
    arguments = dict(
        rank=2,
        icosphere_ref=ref,
        win_size_coef=2,
        num_heads=2,
        d_model=8,
        d_head_coef=1,
        qkv_bias=True,
        attn_drop=0.0,
        out_drop=0.0,
        abs_pos_enc=True,
        abs_pos_enc_size=32,
        rel_pos_bias=True,
        rel_pos_bias_size=7,
        rel_pos_init_variance=0.0,
        append_self=False,
    )
    dense = SphereSelfAttention(**arguments)
    sparse = SparseSphereSelfAttention(**arguments)
    sparse.load_state_dict(dense.state_dict(), strict=True)
    dense_input = torch.randn(2, 162, 8, requires_grad=True)
    sparse_input = dense_input.detach().clone().requires_grad_(True)
    position = torch.randn(1, 162, 32)
    selected = torch.zeros(2, 162, dtype=torch.bool)
    selected[0, ::7] = True
    selected[1, 3::11] = True
    dense_output = dense(dense_input, position)
    sparse_output, pairs = sparse(sparse_input, selected, position)
    dense_selected = dense_output[pairs[:, 0], pairs[:, 1]]
    forward_difference = (dense_selected - sparse_output).abs().max()
    assert float(forward_difference) < 1e-5
    dense_selected.square().sum().backward()
    sparse_output.square().sum().backward()
    input_gradient_difference = (dense_input.grad - sparse_input.grad).abs().max()
    assert float(input_gradient_difference) < 1e-5
    sparse_parameters = dict(sparse.named_parameters())
    parameter_gradient_difference = max(
        float((dense_parameter.grad - sparse_parameters[name].grad).abs().max())
        for name, dense_parameter in dense.named_parameters()
    )
    assert parameter_gradient_difference < 1e-5
    assert sparse.last_query_count == int(selected.sum())
    print(
        "sparse/dense selected-query equivalence: OK",
        f"forward={float(forward_difference):.3e}",
        f"input_grad={float(input_gradient_difference):.3e}",
        f"parameter_grad={parameter_gradient_difference:.3e}",
    )


def global_attention_test():
    attention = GlobalSphereSelfAttention(
        d_model=8, num_heads=2, query_chunk_size=7
    )
    inputs = torch.randn(1, 42, 8, requires_grad=True)
    output = attention(inputs)
    output[:, 0].sum().backward()
    assert attention.last_key_count == 42
    assert attention.last_query_count == 42
    # A single query has a differentiable path through V from every sphere token.
    assert bool((inputs.grad[0].abs().sum(dim=-1) > 0).all())
    assert_finite("global attention", output)
    assert_finite("global attention gradient", inputs.grad)
    print("global full-sphere connectivity and gradient: OK")


def warmup_randomness_test():
    model = build_saliency_model(model_args("uahs", img_rank=3))
    uncertainty = torch.zeros(1, 2, model.hierarchy_l4_l5.coarse_face_count)
    saliency = torch.zeros_like(uncertainty)

    model.train()
    model.hard_selection_warmup_epochs = 1
    model.set_epoch(1)
    warmup_first = model._selection_scores(
        "uncertainty_only", uncertainty, saliency, seed=0
    )
    warmup_second = model._selection_scores(
        "uncertainty_only", uncertainty, saliency, seed=0
    )
    assert not torch.equal(warmup_first, warmup_second)

    model.eval()
    diagnostic_first = model._selection_scores(
        "random_same_budget", uncertainty, saliency, seed=17
    )
    diagnostic_second = model._selection_scores(
        "random_same_budget", uncertainty, saliency, seed=17
    )
    assert torch.equal(diagnostic_first, diagnostic_second)
    print("warm-up RNG advances; diagnostic random selector is reproducible: OK")


def legacy_budget_checkpoint_test():
    """Missing dynamic heads must retain the legacy operating point."""
    source = build_saliency_model(model_args("uahs", img_rank=3))
    legacy_state = {
        name: value
        for name, value in source.state_dict().items()
        if not name.startswith(("budget_head_l4.", "budget_head_l5."))
    }
    restored = build_saliency_model(model_args("uahs", img_rank=3)).eval()
    incompatible = restored.load_state_dict(legacy_state, strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        name.startswith(("budget_head_l4.", "budget_head_l5."))
        for name in incompatible.missing_keys
    )
    with torch.no_grad():
        outputs = restored(torch.randn(1, 1, 642, 3), return_aux=True)
    assert torch.allclose(
        outputs["budget_l5_pred"], torch.tensor([[0.25]]), atol=1e-6
    )
    assert torch.allclose(outputs["budget_l6_alpha"], torch.tensor([[0.5]]))
    assert torch.allclose(
        outputs["budget_l6_pred"], outputs["selected_area_l1"] * 0.5
    )
    print("legacy checkpoint dynamic-budget initialization: OK")


def sparse_scatter_reconstruction_test():
    torch.manual_seed(29)
    model = build_saliency_model(model_args("uahs", img_rank=3))
    base = torch.randn(2, 9, model.embed_dim)
    query_pairs = torch.tensor([[0, 2], [0, 7], [1, 4]])
    selected_candidate = torch.randn(3, model.embed_dim)
    weight = torch.ones(2, 9)
    reconstructed = model._scatter_refinement(
        base,
        selected_candidate,
        query_pairs,
        weight,
        model.fusion_norm_l5,
    )
    base_only = model.fusion_norm_l5(base)
    selected_mask = torch.zeros(2, 9, dtype=torch.bool)
    selected_mask[query_pairs[:, 0], query_pairs[:, 1]] = True
    assert torch.equal(reconstructed[~selected_mask], base_only[~selected_mask])
    assert bool((reconstructed[~selected_mask].abs().sum(dim=-1) > 0).all())
    assert not torch.equal(reconstructed[selected_mask], base_only[selected_mask])
    print("sparse scatter preserves non-selected base features: OK")


def evaluation_ablation_test():
    """Verify the no-L6 evaluation ablation without changing routing."""
    torch.manual_seed(31)
    model = build_saliency_model(model_args("uahs", img_rank=3)).eval()
    inputs = torch.randn(1, 2, 642, 3)
    with torch.no_grad():
        uncertainty_routed = model(inputs, return_aux=True)
        no_l6 = model(
            inputs,
            return_aux=True,
            hard_mask_overrides={
                "l4": uncertainty_routed["hard_face_mask_l4"],
                "l5": uncertainty_routed["hard_face_mask_l5_effective"],
            },
            disable_l6_refinement=True,
        )

    for mask_name in (
            "hard_face_mask_l4",
            "hard_face_mask_l5_effective",
    ):
        assert torch.equal(uncertainty_routed[mask_name], no_l6[mask_name])
    assert int(no_l6["selected_spatial_queries_l6"].sum()) == 0
    assert torch.equal(uncertainty_routed["saliency_l4"], no_l6["saliency_l4"])
    assert torch.equal(uncertainty_routed["saliency_l5"], no_l6["saliency_l5"])
    assert_finite("no-L6 final saliency", no_l6["saliency"])

    model.train()
    try:
        model(inputs, disable_l6_refinement=True)
    except ValueError as error:
        assert "evaluation-only" in str(error)
    else:
        raise AssertionError("Training unexpectedly accepted the no-L6 ablation")
    print("no-L6 same-head uncertainty-routing ablation: OK")


def uahs_training_and_no_gt_leak_test():
    torch.manual_seed(1)
    args = model_args("uahs", img_rank=3)
    model = build_saliency_model(args)
    inputs = torch.randn(1, 2, 642, 3)
    model.eval()
    first = model(inputs, return_aux=True)
    second = model(inputs, return_aux=True)
    inference_without_gt = model(inputs)
    assert tuple(inference_without_gt.shape) == (1, 2, 642)
    assert_finite("inference without GT", inference_without_gt)
    forward_keys = (
        "saliency",
        "uncertainty_l4",
        "uncertainty_l5",
        "hard_face_mask_l4",
        "hard_face_mask_l5_effective",
    )
    for key in forward_keys:
        assert torch.equal(first[key], second[key]), f"{key} is not deterministic"
    assert not any(
        name.startswith(("refine_", "motion_")) for name in first
    )
    assert not any(
        "refinement_head" in name for name, _module in model.named_modules()
    )

    for mode in model.SELECTOR_MODES:
        selected = model(inputs, return_aux=True, selector_mode=mode)
        tolerance_l1 = float(
            model.hierarchy_l4_l5.coarse_face_areas.max()
            / model.hierarchy_l4_l5.coarse_face_areas.sum()
        )
        tolerance_l2 = float(
            model.hierarchy_l5_l6.coarse_face_areas.max()
            / model.hierarchy_l5_l6.coarse_face_areas.sum()
        )
        assert bool((
            (selected["selected_area_l1"] - selected["budget_l5_pred"]).abs()
            <= tolerance_l1
        ).all())
        assert bool((
            (selected["selected_area_l2"] - selected["budget_l6_pred"]).abs()
            <= tolerance_l2
        ).all())
        assert bool((
            selected["budget_l6_pred"] <= selected["selected_area_l1"]
        ).all())
        eligible = model.hierarchy_l4_l5.propagate_coarse_face_values(
            selected["hard_face_mask_l4"]
        ).bool()
        assert not bool((
            selected["hard_face_mask_l5_effective"].bool() & ~eligible
        ).any())

    expected_shapes = {
        "saliency": (1, 2, 642),
        "saliency_l4": (1, 2, 80),
        "uncertainty_l4": (1, 2, 80),
        "budget_l5_pred": (1, 2),
        "hard_face_mask_l4": (1, 2, 80),
        "saliency_l5": (1, 2, 320),
        "uncertainty_l5": (1, 2, 320),
        "budget_l6_alpha": (1, 2),
        "budget_l6_pred": (1, 2),
        "hard_face_mask_l5_effective": (1, 2, 320),
        "exit_level": (1, 2, 642),
    }
    for key, shape in expected_shapes.items():
        assert tuple(first[key].shape) == shape
        assert_finite(key, first[key].float())
    assert set(first["exit_level"].unique().tolist()).issubset({4, 5, 6})
    assert not bool((
        first["hard_face_mask_l5_effective"].bool()
        & ~model.hierarchy_l4_l5.propagate_coarse_face_values(
            first["hard_face_mask_l4"]
        ).bool()
    ).any())
    assert int(first["selected_spatial_queries_l5"].sum()) < 2 * 162
    assert int(first["selected_spatial_queries_l6"].sum()) < 2 * 642

    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.args = args
    trainer.loss_kl = Trainer.loss_kl.__get__(trainer, Trainer)
    trainer.area_weighted_mean = Trainer.area_weighted_mean
    ground_truth_a = torch.rand(1, 2, 642)
    ground_truth_b = torch.rand(1, 2, 642)
    # Labels only change loss values; forward routing was already fixed.
    losses_a = trainer.compute_uahs_losses(ground_truth_a, first)
    losses_b = trainer.compute_uahs_losses(ground_truth_b, first)
    assert set(losses_a) == {
        "loss_saliency_l4",
        "loss_saliency_l5",
        "loss_uncertainty_l4",
        "loss_uncertainty_l5",
        "loss_budget_l5",
        "loss_budget_l6",
        "budget_l5_pred",
        "budget_l5_raw_target",
        "budget_l5_target",
        "budget_l6_pred",
        "budget_l6_target",
        "budget_l6_alpha_pred",
        "budget_l6_alpha_target",
        "budget_l5_selected_area",
        "budget_l6_selected_area",
    }
    assert not torch.equal(losses_a["loss_uncertainty_l4"], losses_b["loss_uncertainty_l4"])
    for losses_for_target in (losses_a, losses_b):
        assert model.budget_l5_min <= float(losses_for_target["budget_l5_target"])
        assert float(losses_for_target["budget_l5_target"]) <= model.budget_l5_max
    saturated_losses = trainer.compute_uahs_losses(
        torch.ones_like(ground_truth_a), first
    )
    assert float(saturated_losses["budget_l5_raw_target"]) > model.budget_l5_max
    assert torch.allclose(
        saturated_losses["budget_l5_target"],
        torch.tensor(
            model.budget_l5_max,
            dtype=saturated_losses["budget_l5_target"].dtype,
        ),
    ), "an unreachable B5 target was not clamped to the prediction range"

    eligible = first["eligible_face_mask_l5"]
    outside_changed = dict(first)
    outside_changed["uncertainty_l5"] = torch.where(
        eligible,
        first["uncertainty_l5"],
        first["uncertainty_l5"] * 10 + 1,
    )
    outside_losses = trainer.compute_uahs_losses(ground_truth_a, outside_changed)
    assert torch.allclose(
        losses_a["loss_uncertainty_l5"],
        outside_losses["loss_uncertainty_l5"],
        atol=1e-7,
    ), "ineligible L5 uncertainty changed the masked uncertainty loss"
    empty_mask_loss = Trainer.area_weighted_masked_mean(
        first["uncertainty_l5"],
        model.hierarchy_l5_l6.coarse_face_areas,
        torch.zeros_like(first["uncertainty_l5"], dtype=torch.bool),
    )
    assert_finite("empty eligible uncertainty loss", empty_mask_loss)
    assert float(empty_mask_loss) == 0.0

    model.train()
    outputs = model(inputs, return_aux=True)
    losses = trainer.compute_uahs_losses(ground_truth_a, outputs)
    assert torch.allclose(
        outputs["budget_l5_pred"], torch.full((1, 2), 0.25), atol=1e-6
    )
    assert torch.allclose(
        outputs["budget_l6_alpha"], torch.full((1, 2), 0.5), atol=1e-6
    )
    assert torch.allclose(
        outputs["budget_l6_pred"], outputs["selected_area_l1"] * 0.5,
        atol=1e-6,
    )
    assert bool((
        outputs["budget_l6_pred"] <= outputs["selected_area_l1"]
    ).all())

    model.zero_grad(set_to_none=True)
    losses["loss_budget_l6"].backward(retain_graph=True)
    assert all(
        parameter.grad is None
        for parameter in model.budget_head_l4.parameters()
    ), "L6 budget loss unexpectedly updated the B5 budget head"
    assert_gradient("budget L5 alpha", model.budget_head_l5)
    assert all(
        parameter.grad is None
        for module in (model.uncertainty_head_l4, model.uncertainty_head_l5)
        for parameter in module.parameters()
    ), "L6 budget loss unexpectedly updated an uncertainty head"

    model.zero_grad(set_to_none=True)
    (losses["loss_budget_l5"] + losses["loss_budget_l6"]).backward(
        retain_graph=True
    )
    assert_gradient("budget L4", model.budget_head_l4)
    assert_gradient("budget L5", model.budget_head_l5)
    assert all(
        parameter.grad is None
        for module in (model.uncertainty_head_l4, model.uncertainty_head_l5)
        for parameter in module.parameters()
    ), "budget loss unexpectedly updated an uncertainty head"
    model.zero_grad(set_to_none=True)
    total_loss = outputs["saliency"].mean()
    for name, value in losses.items():
        if name.startswith("loss_"):
            assert_finite(name, value)
            total_loss = total_loss + value
    total_loss.backward()
    for name, module in (
            ("coarse local", model.coarse_local_encoder),
            ("coarse global", model.coarse_global_block),
            ("sparse L5", model.sparse_refiner_l5),
            ("sparse L6", model.sparse_refiner_l6),
            ("uncertainty L4", model.uncertainty_head_l4),
            ("uncertainty L5", model.uncertainty_head_l5),
            ("budget L4", model.budget_head_l4),
            ("budget L5", model.budget_head_l5),
            ("saliency L4", model.saliency_head_l4),
            ("saliency L5", model.saliency_head_l5),
            ("final saliency", model.output_proj),
    ):
        assert_gradient(name, module)

    diagnostics = UAHSDiagnosticsAccumulator(0.25, 0.125)
    diagnostics.update(outputs, ground_truth_a, [2], model)
    summary = diagnostics.summary()
    assert summary["valid_frames"] == 2
    assert summary["level_l4"]["uncertainty"]["calibration_bins"]
    print("UAHS hard sparse forward/loss/backward/no-GT-leak: OK")


def baseline_regression_test():
    model = build_saliency_model(model_args("sphere_uformer", img_rank=3))
    inputs = torch.randn(1, 2, 642, 3)
    output = model(inputs)
    assert tuple(output.shape) == (1, 2, 642)
    assert_finite("baseline output", output)
    output.mean().backward()
    assert_gradient("SphereUFormer baseline", model)
    print("SphereUFormer baseline forward/backward: OK")


def main():
    tie_aware_spearman_test()
    hierarchy_and_budget_test()
    sparse_attention_equivalence_test()
    global_attention_test()
    warmup_randomness_test()
    legacy_budget_checkpoint_test()
    sparse_scatter_reconstruction_test()
    evaluation_ablation_test()
    uahs_training_and_no_gt_leak_test()
    baseline_regression_test()
    print("UAHS smoke test: PASS")


if __name__ == "__main__":
    main()
