"""Geometry, sparsity, gradient, and regression tests for final UAHS."""

from types import SimpleNamespace

import torch

from adaptive_diagnostics import UAHSDiagnosticsAccumulator
from adaptive_objectives import build_fixed_area_target, per_frame_spearman
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
        win_size_coef=1,
        temporal_window_radius=1,
        d_head_coef=1,
        enc_num_heads=[2],
        bottleneck_num_heads=2,
        dec_num_heads=[2],
        abs_pos_enc_in=True,
        abs_pos_enc=True,
        rel_pos_bias=True,
        rel_pos_bias_size=3,
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
        global_query_chunk_size=32,
        hard_selection_warmup_epochs=0,
        use_uncertainty_refinement=True,
        use_motion_refinement=True,
        return_aux=False,
        debug_uahs=False,
        lambda_saliency_l4=0.15,
        lambda_saliency_l5=0.15,
        lambda_uncertainty_l4=0.05,
        lambda_uncertainty_l5=0.05,
        lambda_refine_l4=0.1,
        lambda_refine_l5=0.1,
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
    print("hierarchy 1->4/1->16, hard budgets, and parent constraint: OK")


def sparse_attention_equivalence_test():
    torch.manual_seed(0)
    ref = IcoSphereRef("vertex")
    arguments = dict(
        rank=2,
        icosphere_ref=ref,
        win_size_coef=1,
        num_heads=2,
        d_model=8,
        d_head_coef=1,
        qkv_bias=True,
        attn_drop=0.0,
        out_drop=0.0,
        abs_pos_enc=False,
        abs_pos_enc_size=0,
        rel_pos_bias=True,
        rel_pos_bias_size=3,
        rel_pos_init_variance=0.0,
        append_self=False,
    )
    dense = SphereSelfAttention(**arguments)
    sparse = SparseSphereSelfAttention(**arguments)
    sparse.load_state_dict(dense.state_dict(), strict=True)
    dense_input = torch.randn(2, 162, 8, requires_grad=True)
    sparse_input = dense_input.detach().clone().requires_grad_(True)
    selected = torch.zeros(2, 162, dtype=torch.bool)
    selected[0, ::7] = True
    selected[1, 3::11] = True
    dense_output = dense(dense_input, None)
    sparse_output, pairs = sparse(sparse_input, selected, None)
    dense_selected = dense_output[pairs[:, 0], pairs[:, 1]]
    forward_difference = (dense_selected - sparse_output).abs().max()
    assert float(forward_difference) < 1e-5
    dense_selected.square().sum().backward()
    sparse_output.square().sum().backward()
    input_gradient_difference = (dense_input.grad - sparse_input.grad).abs().max()
    assert float(input_gradient_difference) < 1e-5
    parameter_gradient_difference = max(
        float((dense_parameter.grad - sparse_parameter.grad).abs().max())
        for (_, dense_parameter), (_, sparse_parameter)
        in zip(dense.named_parameters(), sparse.named_parameters())
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
    logits = torch.zeros(1, 2, model.hierarchy_l4_l5.coarse_face_count)
    uncertainty = torch.zeros_like(logits)
    saliency = torch.zeros_like(logits)

    model.train()
    model.hard_selection_warmup_epochs = 1
    model.set_epoch(1)
    warmup_first = model._selection_scores(
        "learned_refinement_score", logits, uncertainty, saliency, seed=0
    )
    warmup_second = model._selection_scores(
        "learned_refinement_score", logits, uncertainty, saliency, seed=0
    )
    assert not torch.equal(warmup_first, warmup_second)

    model.eval()
    diagnostic_first = model._selection_scores(
        "random_same_budget", logits, uncertainty, saliency, seed=17
    )
    diagnostic_second = model._selection_scores(
        "random_same_budget", logits, uncertainty, saliency, seed=17
    )
    assert torch.equal(diagnostic_first, diagnostic_second)
    print("warm-up RNG advances; diagnostic random selector is reproducible: OK")


def uahs_training_and_no_gt_leak_test():
    torch.manual_seed(1)
    args = model_args("uahs", img_rank=3)
    model = build_saliency_model(args)
    inputs = torch.randn(1, 2, 642, 3)
    model.eval()
    first = model(inputs, return_aux=True)
    second = model(inputs, return_aux=True)
    forward_keys = (
        "saliency",
        "uncertainty_l4",
        "uncertainty_l5",
        "refine_score_l4",
        "refine_score_l5",
        "hard_face_mask_l4",
        "hard_face_mask_l5_effective",
    )
    for key in forward_keys:
        assert torch.equal(first[key], second[key]), f"{key} is not deterministic"

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
            (selected["selected_area_l1"] - 0.25).abs() <= tolerance_l1
        ).all())
        assert bool((
            (selected["selected_area_l2"] - 0.125).abs() <= tolerance_l2
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
        "refine_logits_l4": (1, 2, 80),
        "hard_face_mask_l4": (1, 2, 80),
        "saliency_l5": (1, 2, 320),
        "uncertainty_l5": (1, 2, 320),
        "refine_logits_l5": (1, 2, 320),
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
    trainer.loss_kl_cc = Trainer.loss_kl_cc.__get__(trainer, Trainer)
    trainer.area_weighted_mean = Trainer.area_weighted_mean
    trainer.area_weighted_masked_mean = Trainer.area_weighted_masked_mean
    ground_truth_a = torch.rand(1, 2, 642)
    ground_truth_b = torch.rand(1, 2, 642)
    # Labels only change targets/losses; forward tensors were already fixed.
    losses_a = trainer.compute_uahs_losses(ground_truth_a, first)
    losses_b = trainer.compute_uahs_losses(ground_truth_b, first)
    assert not torch.equal(
        losses_a["selection_target_l4"], losses_b["selection_target_l4"]
    )

    model.train()
    outputs = model(inputs, return_aux=True)
    losses = trainer.compute_uahs_losses(ground_truth_a, outputs)
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
            ("refinement L4", model.refinement_head_l4),
            ("refinement L5", model.refinement_head_l5),
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
    uahs_training_and_no_gt_leak_test()
    baseline_regression_test()
    print("UAHS smoke test: PASS")


if __name__ == "__main__":
    main()
