"""Low-rank training-path smoke tests for baseline and audited UAHS."""

from types import SimpleNamespace

import torch

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


def training_args():
    return SimpleNamespace(
        model_type="adaptive_sphere_uformer",
        use_adaptive_refinement=True,
        use_budget_regularization=True,
        target_refine_ratio=0.25,
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
    for gradient in gradients:
        if gradient is not None:
            assert_finite(f"{name} gradient", gradient)


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

    trainer = Trainer.__new__(Trainer)
    trainer.args = training_args()
    trainer.model = adaptive
    synthetic_inputs = {
        "normalized_sphere_rgb": inputs,
        "normalized_sphere_sal": torch.rand(1, 2, 642),
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
    print("CUDA adaptive forward: OK")


def main():
    torch.manual_seed(0)
    inputs = torch.randn(1, 2, 642, 3)
    hierarchy_test()
    adaptive_training_test(inputs)
    baseline_regression_test(inputs)
    optional_gpu_test()
    print("smoke test: PASS")


if __name__ == "__main__":
    main()
