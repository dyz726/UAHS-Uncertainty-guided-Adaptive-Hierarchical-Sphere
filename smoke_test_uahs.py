"""Geometry, fixed-budget, sparse-attention, and gradient tests for UAHS."""

import tempfile
from types import SimpleNamespace

import torch

from adaptive_objectives import build_fixed_area_target, per_frame_spearman
from Sphere_SalientScore_torch import batch_compute_metrics
from inference import InferenceRunner
from network.sphere_model import build_saliency_model
from network.sphere_PSA import (
    GlobalSphereSelfAttention,
    SparseLocalRefinementBlock,
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
        global_query_chunk_size=32,
        hard_selection_warmup_epochs=0,
        return_aux=False,
        debug_uahs=False,
        lambda_saliency_l4=0.15,
        lambda_saliency_l5=0.15,
        lambda_uncertainty_l4=0.05,
        lambda_uncertainty_l5=0.05,
    )


def assert_finite(name, tensor):
    assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"


def assert_gradient(name, module):
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
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


def hierarchy_and_fixed_budget_test():
    ref = IcoSphereRef("vertex")
    for coarse_rank, fine_rank, expected in ((2, 3, 4), (1, 3, 16)):
        hierarchy = IcoSphereHierarchy(coarse_rank, fine_rank, ref)
        counts = torch.bincount(
            hierarchy.fine_face_to_coarse_face,
            minlength=hierarchy.coarse_face_count,
        )
        assert bool((counts == expected).all())

    l1_l2 = IcoSphereHierarchy(1, 2, ref)
    l2_l3 = IcoSphereHierarchy(2, 3, ref)
    score_l1 = torch.randn(1, 3, l1_l2.coarse_face_count)
    mask_l1 = build_fixed_area_target(
        score_l1, l1_l2.coarse_face_areas, target_ratio=0.25
    )
    eligible_l2 = l1_l2.propagate_coarse_face_values(mask_l1).bool()
    score_l2 = torch.randn(1, 3, l2_l3.coarse_face_count)
    mask_l2 = build_fixed_area_target(
        score_l2,
        l2_l3.coarse_face_areas,
        target_ratio=0.125,
        eligible_mask=eligible_l2,
    )
    area_l1 = (mask_l1 * l1_l2.coarse_face_areas).sum(-1)
    area_l1 = area_l1 / l1_l2.coarse_face_areas.sum()
    area_l2 = (mask_l2 * l2_l3.coarse_face_areas).sum(-1)
    area_l2 = area_l2 / l2_l3.coarse_face_areas.sum()
    tolerance_l1 = float(
        l1_l2.coarse_face_areas.max() / l1_l2.coarse_face_areas.sum()
    )
    tolerance_l2 = float(
        l2_l3.coarse_face_areas.max() / l2_l3.coarse_face_areas.sum()
    )
    assert bool(((area_l1 - 0.25).abs() <= tolerance_l1).all())
    assert bool(((area_l2 - 0.125).abs() <= tolerance_l2).all())
    assert not bool((mask_l2.bool() & ~eligible_l2).any())
    print("hierarchy, fixed spherical-area budgets, and parent constraint: OK")


def sparse_spatial_attention_equivalence_test():
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
    assert float((dense_selected - sparse_output).abs().max()) < 1e-5
    dense_selected.square().sum().backward()
    sparse_output.square().sum().backward()
    assert float((dense_input.grad - sparse_input.grad).abs().max()) < 1e-5
    print("sparse/dense selected-query spatial equivalence: OK")


def sparse_spatial_refinement_test():
    """The fine-level block must refine only selected spatial queries."""
    torch.manual_seed(4)
    ref = IcoSphereRef("vertex")
    block = SparseLocalRefinementBlock(
        rank=1,
        icosphere_ref=ref,
        dim=8,
        num_heads=2,
        d_head_coef=1,
        win_size_coef=2,
        qkv_bias=True,
        attn_drop=0.0,
        out_drop=0.0,
        drop_path=0.0,
        abs_pos_enc=True,
        rel_pos_bias=True,
        rel_pos_bias_size=7,
        rel_pos_init_variance=0.0,
        append_self=False,
    ).eval()
    dense_features = torch.randn(3, 42, 8)
    selected_queries = torch.zeros(3, 42, dtype=torch.bool)
    selected_queries[:, ::7] = True

    normalized = block.norm1(dense_features)
    pos = block.abs_pos_enc(normalized)
    spatial_delta, spatial_pairs = block.attention(
        normalized,
        selected_queries,
        pos,
    )
    expected = dense_features[spatial_pairs[:, 0], spatial_pairs[:, 1]]
    expected = expected + spatial_delta
    expected = expected + block.mlp(block.norm2(expected))

    actual, actual_pairs = block(dense_features, selected_queries)
    assert torch.equal(actual_pairs, spatial_pairs)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert block.attention.last_query_count == int(selected_queries.sum())
    print("selected-query spatial-only refinement: OK")


def global_attention_test():
    attention = GlobalSphereSelfAttention(
        d_model=8, num_heads=2, query_chunk_size=7
    )
    inputs = torch.randn(1, 42, 8, requires_grad=True)
    output = attention(inputs)
    output[:, 0].sum().backward()
    assert attention.last_key_count == 42
    assert attention.last_query_count == 42
    assert bool((inputs.grad[0].abs().sum(dim=-1) > 0).all())
    print("global full-sphere connectivity and gradient: OK")


def warmup_randomness_test():
    model = build_saliency_model(model_args("uahs", img_rank=3))
    uncertainty = torch.zeros(1, 2, model.hierarchy_l4_l5.coarse_face_count)
    saliency = torch.zeros_like(uncertainty)
    model.train()
    model.hard_selection_warmup_epochs = 1
    model.set_epoch(1)
    first = model._selection_scores(
        "uncertainty_only", uncertainty, saliency, seed=0
    )
    second = model._selection_scores(
        "uncertainty_only", uncertainty, saliency, seed=0
    )
    assert not torch.equal(first, second)
    model.eval()
    first = model._selection_scores(
        "random_same_budget", uncertainty, saliency, seed=17
    )
    second = model._selection_scores(
        "random_same_budget", uncertainty, saliency, seed=17
    )
    assert torch.equal(first, second)
    print("warm-up RNG advances; diagnostic random selector is reproducible: OK")


def checkpoint_compatibility_test():
    args = model_args("uahs", img_rank=3)
    source = build_saliency_model(args)
    old_state = dict(source.state_dict())
    position_keys = ("abs_pos_l5.1.weight", "abs_pos_l6.1.weight")
    for key in position_keys:
        old_state.pop(key)
    old_state["sparse_refiner_l5.temporal_attention.q_proj.weight"] = (
        torch.empty(8, 8)
    )
    restored = build_saliency_model(args)
    initialized_positions = {
        key: restored.state_dict()[key].clone() for key in position_keys
    }
    with tempfile.NamedTemporaryFile(suffix=".pth") as checkpoint:
        torch.save(old_state, checkpoint.name)
        Trainer.load_pretrained(
            Trainer.__new__(Trainer), restored, checkpoint.name
        )
        for key in position_keys:
            assert torch.equal(restored.state_dict()[key], initialized_positions[key])
        assert torch.equal(
            restored.output_proj.proj[0].weight,
            source.output_proj.proj[0].weight,
        )
        inference_runner = InferenceRunner.__new__(InferenceRunner)
        inference_runner.model = restored
        try:
            inference_runner._load_weights(checkpoint.name)
        except RuntimeError as error:
            assert "spatial-only sparse refinement" in str(error)
        else:
            raise AssertionError("Inference accepted an old architecture checkpoint")

        torch.save(restored.state_dict(), checkpoint.name)
        inference_runner._load_weights(checkpoint.name)
    print("partial pretraining and strict current inference checkpoints: OK")


def model_forward_and_gradient_test():
    torch.manual_seed(1)
    args = model_args("uahs", img_rank=3)
    model = build_saliency_model(args)
    assert not any("budget_head" in name for name, _ in model.named_modules())
    assert hasattr(model, "abs_pos_l4")
    assert hasattr(model, "abs_pos_l5")
    assert hasattr(model, "abs_pos_l6")
    assert model.sparse_refiner_l5.abs_pos_enc is not None
    assert model.sparse_refiner_l6.abs_pos_enc is not None
    inputs = torch.randn(1, 3, 642, 3)
    model.eval()
    first = model(inputs, return_aux=True)
    second = model(inputs, return_aux=True)
    for key in (
            "saliency",
            "uncertainty_l4",
            "uncertainty_l5",
            "hard_face_mask_l4",
            "hard_face_mask_l5_effective",
    ):
        assert torch.equal(first[key], second[key]), f"{key} is not deterministic"

    expected_shapes = {
        "saliency": (1, 3, 642),
        "saliency_l4": (1, 3, 80),
        "uncertainty_l4": (1, 3, 80),
        "hard_face_mask_l4": (1, 3, 80),
        "saliency_l5": (1, 3, 320),
        "uncertainty_l5": (1, 3, 320),
        "hard_face_mask_l5_effective": (1, 3, 320),
        "exit_level": (1, 3, 642),
    }
    for key, shape in expected_shapes.items():
        assert tuple(first[key].shape) == shape
        assert_finite(key, first[key].float())

    for mode in model.SELECTOR_MODES:
        routed = model(inputs, return_aux=True, selector_mode=mode)
        tolerance_l1 = float(
            model.hierarchy_l4_l5.coarse_face_areas.max()
            / model.hierarchy_l4_l5.coarse_face_areas.sum()
        )
        tolerance_l2 = float(
            model.hierarchy_l5_l6.coarse_face_areas.max()
            / model.hierarchy_l5_l6.coarse_face_areas.sum()
        )
        assert bool((
            (routed["selected_area_l1"] - args.target_refine_ratio_l1).abs()
            <= tolerance_l1
        ).all())
        assert bool((
            (routed["selected_area_l2"] - args.target_refine_ratio_l2).abs()
            <= tolerance_l2
        ).all())
        eligible = model.hierarchy_l4_l5.propagate_coarse_face_values(
            routed["hard_face_mask_l4"]
        ).bool()
        assert not bool((
            routed["hard_face_mask_l5_effective"].bool() & ~eligible
        ).any())

    query_count_l5 = int(routed["selected_spatial_queries_l5"].sum())
    query_count_l6 = int(routed["selected_spatial_queries_l6"].sum())
    assert model.sparse_refiner_l5.attention.last_query_count == query_count_l5
    assert model.sparse_refiner_l6.attention.last_query_count == query_count_l6

    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.args = args
    trainer.loss_kl = Trainer.loss_kl.__get__(trainer, Trainer)
    trainer.area_weighted_mean = Trainer.area_weighted_mean
    target = torch.rand(1, 3, 642)
    fixation = torch.zeros_like(target)
    fixation[..., ::53] = 1
    metrics = batch_compute_metrics(
        first["saliency"].detach(), target, fixation, torch.device("cpu")
    )
    for name, value in metrics.items():
        assert_finite(f"saliency metric {name}", value)
    model.train()
    outputs = model(inputs, return_aux=True)
    auxiliary = trainer.compute_uahs_losses(target, outputs)
    assert set(auxiliary) == {
        "loss_saliency_l4",
        "loss_saliency_l5",
        "loss_uncertainty_l4",
        "loss_uncertainty_l5",
    }
    total = outputs["saliency"].mean() + sum(auxiliary.values())
    assert_finite("total loss", total)
    total.backward()
    for name, module in (
            ("L4 input position", model.abs_pos_l4),
            ("L5 input position", model.abs_pos_l5),
            ("L6 input position", model.abs_pos_l6),
            ("L5 spatial refinement", model.sparse_refiner_l5.attention),
            ("L6 spatial refinement", model.sparse_refiner_l6.attention),
            ("L4 uncertainty", model.uncertainty_head_l4),
            ("L5 uncertainty", model.uncertainty_head_l5),
            ("final saliency", model.output_proj),
    ):
        assert_gradient(name, module)
    for module in (model.abs_pos_l5, model.abs_pos_l6):
        assert any(
            parameter.grad is not None and bool((parameter.grad != 0).any())
            for parameter in module.parameters()
        ), "Fine input position encoding has no nonzero gradient"

    model.eval()
    with torch.no_grad():
        no_l6 = model(inputs, return_aux=True, disable_l6_refinement=True)
    assert int(no_l6["selected_spatial_queries_l6"].sum()) == 0
    print("fixed-budget UAHS forward, spatial refinement, losses, and gradients: OK")


def baseline_regression_test():
    model = build_saliency_model(model_args("sphere_uformer", img_rank=2)).eval()
    inputs = torch.randn(1, 2, 162, 3)
    with torch.no_grad():
        output = model(inputs)
    assert tuple(output.shape) == (1, 2, 162)
    assert_finite("baseline output", output)
    print("published SphereUFormer baseline forward: OK")


def main():
    tie_aware_spearman_test()
    hierarchy_and_fixed_budget_test()
    sparse_spatial_attention_equivalence_test()
    sparse_spatial_refinement_test()
    global_attention_test()
    warmup_randomness_test()
    checkpoint_compatibility_test()
    model_forward_and_gradient_test()
    baseline_regression_test()
    print("UAHS smoke test: PASS")


if __name__ == "__main__":
    main()
