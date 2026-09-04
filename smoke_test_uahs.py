"""Geometry, sparsity, gradient, and regression tests for final UAHS."""

import tempfile
from types import SimpleNamespace

import torch

from adaptive_diagnostics import UAHSDiagnosticsAccumulator
from adaptive_objectives import (
    budget_regression_metrics,
    build_error_supervised_budget,
    build_fixed_area_target,
    per_frame_spearman,
)
from inference import InferenceRunner
from network.sphere_model import build_saliency_model
from network.uahs import CalibratedLaplaceRiskHead
from network.sphere_PSA import (
    GlobalSphereSelfAttention,
    SparseSphereSelfAttention,
    SphereSelfAttention,
)
from train_salient import BUDGET_FRAME_KEYS, Trainer
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


def budget_regression_metrics_test():
    prediction = torch.tensor([0.1, 0.2, 0.3])
    target = torch.tensor([0.1, 0.3, 0.2])
    metrics = budget_regression_metrics(prediction, target)
    assert abs(metrics["pearson"] - 0.5) < 1e-6
    assert abs(metrics["spearman"] - 0.5) < 1e-6
    assert abs(metrics["under_budget_ratio"] - 1 / 3) < 1e-6
    assert metrics["finite_count"] == 3
    assert metrics["nonfinite_count"] == 0
    assert abs(metrics["pred_p50"] - 0.2) < 1e-6
    print("per-frame budget regression diagnostics: OK")


def calibrated_laplace_risk_head_test():
    """Verify Laplace risk, area integration, monotonicity, and detach."""
    torch.manual_seed(7)
    head = CalibratedLaplaceRiskHead(
        initial_output=0.25,
        error_threshold=0.05,
        output_min=0.05,
        output_max=0.50,
    )
    initial_uncertainty = torch.full(
        (2, 4), float(torch.log(torch.tensor(2.0))) + head.epsilon
    )
    initial_budget, _ = head(initial_uncertainty, torch.ones(4))
    assert torch.allclose(
        initial_budget, torch.full((2,), 0.25), atol=1e-6
    )
    head.reset_identity()
    uncertainty = torch.tensor([
        [0.01, 0.03, 0.06, 0.12],
        [0.02, 0.04, 0.08, 0.16],
    ], requires_grad=True)
    areas = torch.tensor([0.5, 1.0, 1.5, 2.0])
    budget, risk = head(uncertainty, areas)

    weights = areas.reshape(1, -1)
    denominator = weights.sum(dim=-1)
    expected_risk = torch.exp(
        -head.error_threshold / (uncertainty.detach() + head.epsilon)
    )
    expected_budget = (
        (expected_risk * weights).sum(dim=-1) / denominator
    ).clamp(head.output_min, head.output_max)
    assert torch.allclose(head.temperature, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(risk, expected_risk, atol=1e-6)
    assert torch.allclose(budget, expected_budget, atol=1e-6)
    assert bool((risk[:, 1:] > risk[:, :-1]).all())

    permutation = torch.tensor([2, 0, 3, 1])
    permuted_budget, permuted_risk = head(
        uncertainty[:, permutation], areas[permutation]
    )
    assert torch.allclose(budget, permuted_budget, atol=1e-7)
    assert torch.allclose(risk[:, permutation], permuted_risk, atol=1e-7)

    eligible = torch.tensor([
        [True, False, True, False],
        [False, True, True, True],
    ])
    conditional_head = CalibratedLaplaceRiskHead(
        initial_output=0.5,
        error_threshold=0.05,
        output_min=0.0,
        output_max=1.0,
    )
    conditional_head.reset_identity()
    conditional_budget, conditional_risk = conditional_head(
        uncertainty, areas, eligible_mask=eligible
    )
    eligible_weights = weights * eligible
    expected_conditional = (
        (expected_risk * eligible_weights).sum(dim=-1)
        / eligible_weights.sum(dim=-1)
    )
    assert torch.allclose(conditional_risk, expected_risk, atol=1e-6)
    assert torch.allclose(conditional_budget, expected_conditional, atol=1e-6)

    (budget.sum() + risk.mean()).backward()
    assert uncertainty.grad is None
    assert_gradient("calibrated Laplace risk head", head)
    assert sum(parameter.numel() for parameter in head.parameters()) == 2
    print("calibrated Laplace local risk and spherical integration: OK")


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
    """Missing dynamic heads must retain their safe initial calibration."""
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
        expected_budget, expected_risk = restored.budget_head_l4(
            outputs["uncertainty_l4"],
            restored.hierarchy_l4_l5.coarse_face_areas,
        )
    assert torch.allclose(outputs["budget_l5_pred"], expected_budget)
    assert torch.allclose(
        outputs["refinement_risk_l4"], expected_risk
    )
    assert torch.allclose(
        restored.budget_head_l4.temperature, torch.tensor(1.0), atol=1e-6
    )
    expected_alpha, expected_risk_l5 = restored.budget_head_l5(
        outputs["uncertainty_l5"],
        restored.hierarchy_l5_l6.coarse_face_areas,
        eligible_mask=outputs["eligible_face_mask_l5"],
    )
    assert torch.allclose(outputs["budget_l6_alpha"], expected_alpha)
    assert torch.allclose(
        outputs["refinement_risk_l5"], expected_risk_l5
    )
    assert torch.allclose(
        outputs["budget_l6_pred"], outputs["selected_area_l1"] * expected_alpha
    )
    assert abs(float(outputs["budget_l5_pred"].mean()) - 0.25) < 0.01
    assert abs(float(outputs["budget_l6_alpha"].mean()) - 0.5) < 0.01
    assert abs(float(outputs["budget_l6_pred"].mean()) - 0.125) < 0.01
    print("configured 0.25/0.125 startup budgets: OK")


def calibrated_risk_checkpoint_test():
    """Earlier budget heads are replaced while all other weights load."""
    def assert_identity_calibration(model):
        for head in (model.budget_head_l4, model.budget_head_l5):
            assert torch.allclose(head.beta, torch.tensor(0.0))
            assert torch.allclose(
                head.temperature, torch.tensor(1.0), atol=1e-6
            )

    torch.manual_seed(23)
    source = build_saliency_model(model_args("uahs", img_rank=3)).eval()
    with torch.no_grad():
        source.budget_head_l4.beta.fill_(0.2)
        source.budget_head_l5.beta.fill_(-0.3)
    old_dynamic_state = {
        name: value
        for name, value in source.state_dict().items()
        if not name.startswith(("budget_head_l4.", "budget_head_l5."))
    }
    old_dynamic_state["budget_head_l4.face_mlp.0.weight"] = torch.randn(4, 1)
    old_dynamic_state["budget_head_l4.global_mlp.2.bias"] = torch.randn(1)
    old_dynamic_state["budget_head_l5.mlp.0.weight"] = torch.randn(8, 2)
    old_dynamic_state["budget_head_l5.mlp.2.bias"] = torch.randn(1)
    restored = build_saliency_model(model_args("uahs", img_rank=3)).eval()
    with tempfile.NamedTemporaryFile(suffix=".pth") as checkpoint:
        torch.save(old_dynamic_state, checkpoint.name)
        Trainer.load_pretrained(
            Trainer.__new__(Trainer), restored, checkpoint.name
        )
        assert_identity_calibration(restored)
        inference_model = build_saliency_model(
            model_args("uahs", img_rank=3)
        ).eval()
        inference_runner = InferenceRunner.__new__(InferenceRunner)
        inference_runner.model = inference_model
        inference_runner._load_weights(checkpoint.name)
        assert_identity_calibration(inference_model)

        torch.save(source.state_dict(), checkpoint.name)
        inference_runner._load_weights(checkpoint.name)
        for name, value in inference_model.state_dict().items():
            assert torch.equal(value, source.state_dict()[name])

        partial_state = dict(source.state_dict())
        del partial_state["budget_head_l4.raw_temperature"]
        torch.save(partial_state, checkpoint.name)
        partial_training_model = build_saliency_model(
            model_args("uahs", img_rank=3)
        )
        try:
            Trainer.load_pretrained(
                Trainer.__new__(Trainer),
                partial_training_model,
                checkpoint.name,
            )
        except RuntimeError as error:
            assert "Incomplete" in str(error)
        else:
            raise AssertionError("Training accepted an incomplete risk head")
        partial_inference = InferenceRunner.__new__(InferenceRunner)
        partial_inference.model = build_saliency_model(
            model_args("uahs", img_rank=3)
        )
        try:
            partial_inference._load_weights(checkpoint.name)
        except RuntimeError as error:
            assert "Incomplete" in str(error)
        else:
            raise AssertionError("Inference accepted an incomplete risk head")

    with torch.no_grad():
        prediction, risk = restored.budget_head_l4(
            torch.rand(1, 2, 80), torch.ones(80)
        )
    assert tuple(prediction.shape) == (1, 2)
    assert tuple(risk.shape) == (1, 2, 80)
    incompatible = restored.load_state_dict(source.state_dict(), strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    print("legacy migration and incomplete risk checkpoint rejection: OK")


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
    assert isinstance(model.budget_head_l4, CalibratedLaplaceRiskHead)
    assert isinstance(model.budget_head_l5, CalibratedLaplaceRiskHead)
    assert not hasattr(model.budget_head_l4, "mlp")
    assert not hasattr(model.budget_head_l5, "mlp")
    assert sum(
        parameter.numel() for parameter in model.budget_head_l4.parameters()
    ) == 2
    assert sum(
        parameter.numel() for parameter in model.budget_head_l5.parameters()
    ) == 2
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
        "refinement_risk_l4",
        "uncertainty_l5",
        "refinement_risk_l5",
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
        "refinement_risk_l4": (1, 2, 80),
        "budget_l5_pred": (1, 2),
        "hard_face_mask_l4": (1, 2, 80),
        "saliency_l5": (1, 2, 320),
        "uncertainty_l5": (1, 2, 320),
        "refinement_risk_l5": (1, 2, 320),
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
    processed_outputs, processed_losses = trainer.process_batch({
        "normalized_sphere_rgb": inputs,
        "normalized_sphere_sal": ground_truth_a,
        "normalized_sphere_fix": torch.zeros_like(ground_truth_a),
    })
    assert_finite("processed total loss", processed_losses["loss"])
    for prediction_key, target_key in BUDGET_FRAME_KEYS.values():
        assert tuple(processed_outputs[prediction_key].shape) == (1, 2)
        assert tuple(processed_outputs[target_key].shape) == (1, 2)
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
        "budget_l5_pred_per_frame",
        "budget_l5_target_per_frame",
        "budget_l6_pred_per_frame",
        "budget_l6_target_per_frame",
        "risk_l5_pred",
        "risk_l5_target",
        "risk_l5_brier",
        "risk_l5_ece",
        "risk_l5_beta",
        "risk_l5_temperature",
        "budget_l5_consistency",
        "risk_l6_pred",
        "risk_l6_target",
        "risk_l6_brier",
        "risk_l6_ece",
        "risk_l6_beta",
        "risk_l6_temperature",
        "budget_l6_consistency",
    }
    assert not torch.equal(losses_a["loss_uncertainty_l4"], losses_b["loss_uncertainty_l4"])
    for name in (
            "budget_l5_pred_per_frame",
            "budget_l5_target_per_frame",
            "budget_l6_pred_per_frame",
            "budget_l6_target_per_frame",
    ):
        assert tuple(losses_a[name].shape) == (1, 2)
    diagnostic_buffer = Trainer.new_budget_diagnostic_buffer()
    Trainer.update_budget_diagnostic_buffer(
        diagnostic_buffer, processed_outputs
    )
    budget_summary = Trainer.summarize_budget_diagnostic_buffer(
        diagnostic_buffer
    )
    for level in ("l5", "l6"):
        for name in (
                "pred_mean", "pred_std", "target_mean", "target_std",
                "mae", "rmse", "pearson", "spearman",
                "under_budget_ratio", "pred_p10", "pred_p50", "pred_p90",
                "target_p10", "target_p50", "target_p90",
        ):
            assert f"budget_{level}_{name}" in budget_summary
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
    expected_budget, expected_risk = model.budget_head_l4(
        outputs["uncertainty_l4"],
        model.hierarchy_l4_l5.coarse_face_areas,
    )
    assert torch.allclose(outputs["budget_l5_pred"], expected_budget)
    assert torch.allclose(outputs["refinement_risk_l4"], expected_risk)
    expected_alpha, expected_risk_l6 = model.budget_head_l5(
        outputs["uncertainty_l5"],
        model.hierarchy_l5_l6.coarse_face_areas,
        eligible_mask=outputs["eligible_face_mask_l5"],
    )
    assert torch.allclose(outputs["budget_l6_alpha"], expected_alpha)
    assert torch.allclose(outputs["refinement_risk_l5"], expected_risk_l6)
    assert torch.allclose(
        outputs["budget_l6_pred"],
        outputs["selected_area_l1"] * expected_alpha,
    )
    areas_l5 = model.hierarchy_l5_l6.coarse_face_areas
    eligible_weights = (
        outputs["eligible_face_mask_l5"].to(areas_l5.dtype)
        * areas_l5.reshape(1, 1, -1)
    )
    eligible_area = eligible_weights.sum(dim=-1) / areas_l5.sum()
    direct_risk_area = (
        outputs["refinement_risk_l5"] * eligible_weights
    ).sum(dim=-1) / areas_l5.sum()
    assert torch.allclose(
        outputs["selected_area_l1"], eligible_area, atol=1e-6
    )
    assert torch.allclose(
        outputs["budget_l6_pred"], direct_risk_area, atol=1e-6
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
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith(("budget_head_l4.", "budget_head_l5."))
    ), "budget losses unexpectedly updated a non-budget module"
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
    budget_regression_metrics_test()
    calibrated_laplace_risk_head_test()
    hierarchy_and_budget_test()
    sparse_attention_equivalence_test()
    global_attention_test()
    warmup_randomness_test()
    legacy_budget_checkpoint_test()
    calibrated_risk_checkpoint_test()
    sparse_scatter_reconstruction_test()
    evaluation_ablation_test()
    uahs_training_and_no_gt_leak_test()
    baseline_regression_test()
    print("UAHS smoke test: PASS")


if __name__ == "__main__":
    main()
