"""Smoke tests for the L4-only attention ablation."""

from types import SimpleNamespace

import torch

from network.sphere_model import build_saliency_model


def model_args():
    """Use smaller ranks while preserving the two-level ablation topology."""
    return SimpleNamespace(
        model_type="uahs",
        mode="vertex",
        img_rank=4,
        scale_factor=2,
        win_size_coef=1,
        temporal_window_radius=1,
        d_head_coef=1,
        abs_pos_enc_in=True,
        abs_pos_enc=True,
        rel_pos_bias=True,
        rel_pos_bias_size=7,
        rel_pos_init_variance=0.0,
        dr=0.0,
        dpr=0.0,
        adr=0.0,
        aodr=0.0,
        posdr=0.0,
        debug_skip_attn=False,
        append_self=False,
        use_checkpoint=False,
        enc_num_heads=[2, 4, 8, 16],
        coarse_pool_type="mean_max",
        target_refine_ratio_l1=0.25,
        target_refine_ratio_l2=0.125,
        budget_l5_min=0.05,
        budget_l5_max=0.50,
        budget_error_threshold_l4=0.05,
        budget_error_threshold_l5=0.05,
        global_query_chunk_size=32,
        hard_selection_warmup_epochs=0,
        return_aux=False,
        debug_uahs=False,
    )


def architecture_test(model):
    assert len(model.coarse_local_encoder.blocks) == 2
    assert model.coarse_local_encoder.rank == model.coarse_rank
    forbidden = (
        "saliency_head_l4",
        "saliency_head_l5",
        "uncertainty_head",
        "budget_head",
        "sparse_refiner",
        "context_projection",
        "fusion_norm",
    )
    state_keys = tuple(model.state_dict())
    for name in forbidden:
        assert not any(name in key for key in state_keys), name
    assert hasattr(model, "output_proj")
    assert not hasattr(model, "output_proj_l4")


def forward_and_gradient_test(model):
    batch_size, time_steps = 1, 2
    vertices_l6 = model.hierarchy_l4_l6.fine_vertex_count
    seen_output_vertices = []

    def capture_output_input(_module, inputs):
        seen_output_vertices.append(inputs[0].shape[1])

    hook = model.output_proj.register_forward_pre_hook(capture_output_input)
    inputs = torch.randn(batch_size, time_steps, vertices_l6, 3)
    prediction = model(inputs)
    hook.remove()

    assert prediction.shape == (batch_size, time_steps, vertices_l6)
    assert seen_output_vertices == [vertices_l6]
    assert torch.isfinite(prediction).all()
    assert bool(((prediction >= 0) & (prediction <= 1)).all())

    prediction.mean().backward()
    for module in (
            model.input_proj_l6,
            model.coarse_region_pool,
            model.coarse_local_encoder,
            model.coarse_global_block,
            model.output_proj,
    ):
        gradients = [
            parameter.grad for parameter in module.parameters()
            if parameter.requires_grad
        ]
        assert gradients and any(gradient is not None for gradient in gradients)
        assert all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in gradients
        )

    auxiliary = model(inputs.detach(), return_aux=True)
    assert set(auxiliary) == {"saliency"}
    assert auxiliary["saliency"].shape == prediction.shape


def main():
    torch.manual_seed(0)
    model = build_saliency_model(model_args())
    architecture_test(model)
    forward_and_gradient_test(model)
    print("L4-only attention architecture: OK")
    print("L4 feature upsampling and L6 output head: OK")
    print("rank-6 saliency shape and gradients: OK")


if __name__ == "__main__":
    main()
