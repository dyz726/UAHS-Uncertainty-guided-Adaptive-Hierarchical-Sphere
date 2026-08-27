"""Low-rank CPU smoke test for baseline and adaptive saliency models."""

import torch

from network.sphere_model import AdaptiveSphereUFormer, SphereUFormer


def make_adaptive():
    return AdaptiveSphereUFormer(
        img_rank=3,
        node_type="vertex",
        embed_dim=8,
        in_scale_factor=2,
        num_heads=2,
        coarse_rank_offset=2,
        adaptive_coarse_depth=1,
        adaptive_fine_depth=1,
        win_size_coef=1,
        abs_pos_enc_in=False,
        abs_pos_enc=False,
        rel_pos_bias=False,
        rel_pos_bias_size=3,
        use_checkpoint=False,
    )


def make_baseline():
    return SphereUFormer(
        img_rank=3,
        node_type="vertex",
        embed_dim=8,
        in_scale_factor=2,
        num_scales=1,
        enc_depths=1,
        dec_depths=1,
        bottleneck_depth=1,
        d_head_coef=1,
        enc_num_heads=(2,),
        bottleneck_num_heads=2,
        dec_num_heads=(2,),
        win_size_coef=1,
        abs_pos_enc_in=False,
        abs_pos_enc=False,
        rel_pos_bias=False,
        rel_pos_bias_size=3,
        use_checkpoint=False,
        upsample="interpolate",
    )


def assert_finite(name, value):
    assert torch.isfinite(value).all(), f"{name} contains non-finite values"


def main():
    torch.manual_seed(0)
    batch_size, time_steps = 1, 2
    # A rank-3 vertex icosphere contains 10 * 4**3 + 2 = 642 nodes.
    inputs = torch.randn(batch_size, time_steps, 642, 3)

    adaptive = make_adaptive()
    default_output = adaptive(inputs)
    assert isinstance(default_output, torch.Tensor)
    assert tuple(default_output.shape) == (1, 2, 642)
    aux = adaptive(inputs, return_aux=True)
    expected_shapes = {
        "saliency": (1, 2, 642),
        "coarse_saliency": (1, 2, 12),
        "refine_score": (1, 2, 12),
        "fine_refine_score": (1, 2, 162),
    }
    for name, shape in expected_shapes.items():
        assert tuple(aux[name].shape) == shape, (name, aux[name].shape)
        assert_finite(name, aux[name])
    for name in ("gate_mean", "gate_max", "gate_min"):
        assert_finite(name, aux[name])

    loss = aux["saliency"].mean()
    loss.backward()
    refinement_parameters = list(adaptive.refinement_head.parameters())
    assert refinement_parameters
    assert all(parameter.grad is not None for parameter in refinement_parameters)

    baseline = make_baseline()
    baseline_output = baseline(inputs)
    assert tuple(baseline_output.shape) == expected_shapes["saliency"]
    assert_finite("baseline saliency", baseline_output)
    baseline_output.mean().backward()

    print("adaptive saliency:", tuple(aux["saliency"].shape))
    print("coarse/refine/fine-refine:", *(expected_shapes[key] for key in (
        "coarse_saliency", "refine_score", "fine_refine_score"
    )))
    print("refinement gradients: OK")
    print("baseline saliency:", tuple(baseline_output.shape))
    print("smoke test: PASS")


if __name__ == "__main__":
    main()
