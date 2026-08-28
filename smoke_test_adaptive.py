"""Low-rank training-path smoke tests for baseline and audited UAHS."""

from types import SimpleNamespace

import torch

from adaptive_diagnostics import (
    AdaptiveDiagnosticsAccumulator,
    AdaptiveDiagnosticsAccumulatorV2,
)
from adaptive_objectives import per_frame_spearman
from network.sphere_model import build_saliency_model
from train_salient import Trainer
from trimesh_utils import IcoSphereHierarchy, IcoSphereRef


def model_args(model_type):
    return SimpleNamespace(
        model_type=model_type,
        mode="vertex",
        img_rank=3,
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
        coarse_rank_offset=2,
        adaptive_coarse_depth=1,
        adaptive_middle_depth=1,
        adaptive_fine_depth=1,
        adaptive_temperature=1.0,
        adaptive_region_type="face",
        coarse_pool_type="mean_max",
        face_to_vertex_reduce="mean",
        use_adaptive_refinement=True,
        use_uncertainty_refinement=True,
        use_motion_refinement=True,
        return_aux=False,
        debug_adaptive=False,
    )


def training_args(model_type="adaptive_sphere_uformer"):
    return SimpleNamespace(
        model_type=model_type,
        use_adaptive_refinement=True,
        use_budget_regularization=True,
        target_refine_ratio=0.25,
        target_refine_ratio_l1=0.25,
        target_refine_ratio_l2=0.125,
        lambda_coarse=0.3,
        lambda_uncertainty=0.1,
        lambda_refine=0.2,
        lambda_budget=0.05,
    )


def assert_finite(name, value):
    assert torch.isfinite(value).all(), f"{name} contains non-finite values"


def assert_module_gradient(name, module):
    gradients = [parameter.grad for parameter in module.parameters()]
    assert gradients, f"{name} has no parameters"
    assert any(gradient is not None for gradient in gradients), f"{name} has no gradient"
    has_nonzero_gradient = False
    for gradient in gradients:
        if gradient is not None:
            assert_finite(f"{name} gradient", gradient)
            has_nonzero_gradient |= bool((gradient != 0).any())
    assert has_nonzero_gradient, f"{name} gradients are all zero"


def hierarchy_test():
    ref = IcoSphereRef("vertex")
    for fine_rank, expected in ((1, 4), (2, 16)):
        hierarchy = IcoSphereHierarchy(0, fine_rank, ref)
        counts = torch.bincount(
            hierarchy.fine_face_to_coarse_face,
            minlength=hierarchy.coarse_face_count,
        )
        assert bool((counts == expected).all())
        assert hierarchy.fine_face_to_coarse_face.numel() == hierarchy.fine_face_count
    print("face hierarchy 1->4 and 1->16: OK")


def spearman_diagnostic_test():
    first = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0, 1.0],
    ])
    second = torch.tensor([
        [4.0, 3.0, 2.0, 1.0],
        [1.0, 2.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
    ])
    correlation = per_frame_spearman(first, second)
    assert torch.allclose(correlation[:1], torch.tensor([-1.0]))
    assert torch.allclose(correlation[1:2], torch.tensor([5.0 / 6.0]))
    assert torch.isnan(correlation[2])
    assert torch.isnan(
        per_frame_spearman(torch.ones(1, 4), torch.ones(1, 4))[0]
    )
    assert torch.isnan(
        per_frame_spearman(torch.ones(1, 1), torch.ones(1, 1))[0]
    )
    print("tie-aware Spearman and zero-variance handling: OK")


def v2_real_rank_hierarchy_test():
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
        assert counts.sum() == hierarchy.fine_face_count
    print("V2 real hierarchy 4->5, 5->6, and 4->6: OK")


def adaptive_training_test(inputs):
    adaptive = build_saliency_model(model_args("adaptive_sphere_uformer"))

    # Forward uses RGB only; labels are deliberately created afterwards.
    default_output = adaptive(inputs)
    assert isinstance(default_output, torch.Tensor)
    aux = adaptive(inputs, return_aux=True)
    expected_shapes = {
        "saliency": (1, 2, 642),
        "coarse_saliency": (1, 2, 20),
        "uncertainty": (1, 2, 20),
        "refine_score": (1, 2, 20),
        "fine_refine_score": (1, 2, 320),
        "fine_face_gate": (1, 2, 320),
        "fine_vertex_gate": (1, 2, 162),
        "area_refine_ratio": (1, 2),
    }
    for name, shape in expected_shapes.items():
        assert tuple(aux[name].shape) == shape, (name, aux[name].shape)
        assert_finite(name, aux[name])
    assert bool((aux["uncertainty"] > 0).all())

    synthetic_saliency = torch.rand(1, 2, 642)
    diagnostics = AdaptiveDiagnosticsAccumulator(target_refine_ratio=0.25)
    diagnostics.update(aux, synthetic_saliency, [2], adaptive)
    diagnostic_summary = diagnostics.summary()
    for name in (
            "uncertainty",
            "refine_score",
            "fine_face_gate",
            "area_refine_ratio",
    ):
        assert diagnostic_summary[name]["mean"] is not None
        assert diagnostic_summary[name]["std"] is not None
        assert diagnostic_summary[name]["nonfinite_count"] == 0
    assert diagnostic_summary["valid_frames"] == 2
    assert (
        diagnostic_summary["coarse_rank_evidence"]["absolute_kl_reduction"]
        is not None
    )

    trainer = Trainer.__new__(Trainer)
    trainer.args = training_args()
    trainer.model = adaptive
    synthetic_inputs = {
        "normalized_sphere_rgb": inputs,
        "normalized_sphere_sal": synthetic_saliency,
        "normalized_sphere_fix": (torch.rand(1, 2, 642) > 0.9).float(),
    }
    _, losses = trainer.process_batch(synthetic_inputs)
    expected_losses = (
        "loss_saliency",
        "loss_coarse",
        "loss_uncertainty",
        "loss_refine",
        "loss_budget",
        "loss",
    )
    for name in expected_losses:
        assert_finite(name, losses[name])
    losses["loss"].backward()

    assert_module_gradient("coarse encoder", adaptive.coarse_encoder)
    assert_module_gradient("fine encoder", adaptive.fine_encoder)
    assert_module_gradient("uncertainty head", adaptive.uncertainty_head)
    assert_module_gradient("refinement head", adaptive.refinement_head)
    assert_module_gradient("coarse saliency head", adaptive.coarse_saliency_head)
    assert_module_gradient("final saliency head", adaptive.output_proj)

    print("adaptive aux shapes:", expected_shapes)
    print(
        "training losses:",
        {name: round(float(losses[name].detach()), 6) for name in expected_losses},
    )
    print("adaptive training gradients: OK")
    print(
        "adaptive diagnostics: OK",
        {
            name: {
                "mean": round(diagnostic_summary[name]["mean"], 6),
                "std": round(diagnostic_summary[name]["std"], 6),
            }
            for name in (
                "uncertainty",
                "refine_score",
                "fine_face_gate",
                "area_refine_ratio",
            )
        },
    )


def v2_training_test():
    args = model_args("adaptive_sphere_uformer_v2")
    args.img_rank = 2
    model = build_saliency_model(args)
    inputs = torch.randn(1, 2, 162, 3)
    default_output = model(inputs)
    assert tuple(default_output.shape) == (1, 2, 162)
    aux = model(inputs, return_aux=True)
    expected_shapes = {
        "saliency": (1, 2, 162),
        "saliency_l4": (1, 2, 20),
        "uncertainty_l4": (1, 2, 20),
        "refine_score_l4": (1, 2, 20),
        "gate_l4_to_l5": (1, 2, 80),
        "area_ratio_l1": (1, 2),
        "saliency_l5": (1, 2, 80),
        "uncertainty_l5": (1, 2, 80),
        "refine_score_l5": (1, 2, 80),
        "gate_l5_to_l6_local": (1, 2, 80),
        "gate_l5_to_l6_effective": (1, 2, 320),
        "area_ratio_l2": (1, 2),
    }
    for name, shape in expected_shapes.items():
        assert tuple(aux[name].shape) == shape, (name, aux[name].shape)
        assert_finite(f"V2 {name}", aux[name])

    parent_zero = torch.zeros_like(aux["gate_l4_parent"])
    local_one = torch.ones_like(aux["gate_l5_to_l6_local"])
    parent_test = model(
        inputs,
        return_aux=True,
        gate_overrides={"l4": parent_zero, "l5_local": local_one},
    )
    assert torch.equal(
        parent_test["gate_l5_to_l6_effective"],
        torch.zeros_like(parent_test["gate_l5_to_l6_effective"]),
    )

    synthetic_saliency = torch.rand(1, 2, 162)
    trainer = Trainer.__new__(Trainer)
    trainer.args = training_args("adaptive_sphere_uformer_v2")
    trainer.model = model
    target_losses = trainer.compute_v2_losses(synthetic_saliency, aux)
    selection_l4 = target_losses["selection_target_l4"]
    selection_l5 = target_losses["selection_target_l5"]
    eligible_l5 = model.hierarchy_l4_l5.propagate_coarse_face_values(
        selection_l4
    ).bool()
    assert not bool((selection_l5.bool() & ~eligible_l5).any())
    target_area_l1 = (
        selection_l4 * model.hierarchy_l4_l5.coarse_face_areas
    ).sum(dim=-1) / model.hierarchy_l4_l5.coarse_face_areas.sum()
    target_area_l2 = (
        selection_l5 * model.hierarchy_l5_l6.coarse_face_areas
    ).sum(dim=-1) / model.hierarchy_l5_l6.coarse_face_areas.sum()
    assert bool((target_area_l1 - 0.25).abs().max() < 0.04)
    assert bool((target_area_l2 - 0.125).abs().max() < 0.04)
    inputs_with_targets = {
        "normalized_sphere_rgb": inputs,
        "normalized_sphere_sal": synthetic_saliency,
        "normalized_sphere_fix": (torch.rand(1, 2, 162) > 0.9).float(),
    }
    _, losses = trainer.process_batch(inputs_with_targets)
    expected_losses = (
        "loss_saliency",
        "loss_saliency_l4",
        "loss_saliency_l5",
        "loss_uncertainty_l4",
        "loss_uncertainty_l5",
        "loss_refine_l4",
        "loss_refine_l5",
        "loss_budget_l1",
        "loss_budget_l2",
        "loss",
    )
    for name in expected_losses:
        assert_finite(name, losses[name])
    losses["loss"].backward()
    for name, module in (
            ("V2 coarse encoder", model.coarse_encoder),
            ("V2 rank5 encoder", model.rank5_encoder),
            ("V2 rank6 encoder", model.rank6_encoder),
            ("V2 uncertainty L4", model.uncertainty_head_l4),
            ("V2 uncertainty L5", model.uncertainty_head_l5),
            ("V2 refinement L4", model.refinement_head_l4),
            ("V2 refinement L5", model.refinement_head_l5),
            ("V2 saliency L4", model.saliency_head_l4),
            ("V2 saliency L5", model.saliency_head_l5),
            ("V2 final saliency", model.output_proj),
    ):
        assert_module_gradient(name, module)

    diagnostics = AdaptiveDiagnosticsAccumulatorV2(0.25, 0.125)
    diagnostics.update(aux, synthetic_saliency, [2], model)
    summary = diagnostics.summary()
    assert summary["valid_frames"] == 2
    for level in ("level_l4", "level_l5"):
        assert summary[level]["uncertainty"]["pearson"] is not None
        assert summary[level]["uncertainty"]["spearman_per_frame"]["mean"] is not None
        assert summary[level]["selection"]["iou"]["mean"] is not None

    from inference import InferenceRunner
    runner = InferenceRunner.__new__(InferenceRunner)
    runner.model = model
    runner.args = SimpleNamespace(
        target_refine_ratio_l1=0.25,
        target_refine_ratio_l2=0.125,
        diagnostic_random_seed=0,
    )
    comparison_ratios = {}
    for mode in (
            "full_fine",
            "uniform_same_budget",
            "random_same_budget",
            "oracle_error_selection",
    ):
        overrides = runner._build_v2_gate_overrides(
            mode, aux, synthetic_saliency, batch_index=0, rgb=inputs
        )
        comparison = model(inputs, return_aux=True, gate_overrides=overrides)
        comparison_ratios[mode] = (
            float(comparison["area_ratio_l1"].mean()),
            float(comparison["area_ratio_l2"].mean()),
        )
    assert comparison_ratios["full_fine"] == (1.0, 1.0)
    assert abs(comparison_ratios["uniform_same_budget"][0] - 0.25) < 1e-6
    assert abs(comparison_ratios["uniform_same_budget"][1] - 0.125) < 1e-6
    for mode in ("random_same_budget", "oracle_error_selection"):
        assert abs(comparison_ratios[mode][0] - 0.25) < 0.04
        assert abs(comparison_ratios[mode][1] - 0.125) < 0.04
    print("V2 aux shapes:", expected_shapes)
    print(
        "V2 training losses:",
        {name: round(float(losses[name].detach()), 6) for name in expected_losses},
    )
    print("V2 gate comparison area ratios:", comparison_ratios)
    print("V2 parent-gate, diagnostics, comparisons, and gradients: OK")


def baseline_regression_test(inputs):
    baseline = build_saliency_model(model_args("sphere_uformer"))
    output = baseline(inputs)
    assert tuple(output.shape) == (1, 2, 642)
    assert_finite("baseline saliency", output)
    output.mean().backward()
    assert_module_gradient("baseline", baseline)
    print("baseline forward/backward: OK", tuple(output.shape))


def optional_gpu_test():
    if not torch.cuda.is_available():
        print("CUDA unavailable: GPU device test SKIPPED")
        return
    device = torch.device("cuda")
    model = build_saliency_model(model_args("adaptive_sphere_uformer")).to(device)
    inputs = torch.randn(1, 2, 642, 3, device=device)
    outputs = model(inputs, return_aux=True)
    for name, value in outputs.items():
        assert value.device == inputs.device, f"{name} is on {value.device}"
        assert_finite(name, value)
    diagnostics = AdaptiveDiagnosticsAccumulator(target_refine_ratio=0.25)
    diagnostics.update(outputs, torch.rand(1, 2, 642, device=device), [2], model)
    assert diagnostics.summary()["uncertainty"]["nonfinite_count"] == 0
    model.use_adaptive_refinement = False
    uniform_output = model(inputs, return_aux=False)
    model.use_adaptive_refinement = True
    assert_finite("CUDA uniform-gate saliency", uniform_output)
    v2_args = model_args("adaptive_sphere_uformer_v2")
    v2_args.img_rank = 2
    v2 = build_saliency_model(v2_args).to(device)
    v2_inputs = torch.randn(1, 2, 162, 3, device=device)
    v2_outputs = v2(v2_inputs, return_aux=True)
    for name, value in v2_outputs.items():
        assert value.device == v2_inputs.device, f"V2 {name} is on {value.device}"
        assert_finite(f"CUDA V2 {name}", value)
    print("CUDA V1/V2 diagnostics and gate forwards: OK")


def main():
    torch.manual_seed(0)
    inputs = torch.randn(1, 2, 642, 3)
    spearman_diagnostic_test()
    hierarchy_test()
    v2_real_rank_hierarchy_test()
    adaptive_training_test(inputs)
    v2_training_test()
    baseline_regression_test(inputs)
    optional_gpu_test()
    print("smoke test: PASS")


if __name__ == "__main__":
    main()
