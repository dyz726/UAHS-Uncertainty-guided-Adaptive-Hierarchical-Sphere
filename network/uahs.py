"""Rank-4 local/global attention ablation for spherical saliency."""

from typing import Optional

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch import Tensor

from trimesh_utils import IcoSphereHierarchy, IcoSphereRef

from .position_encoding import GlobalVerticalPositionEnconding
from .sphere_model import (
    HierarchicalRegionPool,
    InputProj,
    InterpolateUpsample,
    OutputProj,
    SphereUFormerModule,
)
from .sphere_PSA import GlobalContentSpatioTemporalBlock


class UAHS(nn.Module):
    """L4-only attention ablation followed by feature upsampling.

    Rank-6 RGB observations are pooled directly to rank 4. Two local attention
    blocks and one global attention block process the rank-4 features. The
    features are then interpolated to rank 6, where the only saliency head
    generates the final prediction. This ablation contains no L4 prediction
    head, rank-5/rank-6 attention, uncertainty head, routing, or budget head.
    """

    def __init__(
            self,
            img_rank: int,
            node_type: str,
            in_channels: int = 3,
            out_channels: int = 1,
            embed_dim: int = 32,
            in_scale_factor: int = 2,
            d_head_coef: int = 1,
            num_heads: int = 2,
            win_size_coef: int = 1,
            temporal_window_radius: Optional[int] = 5,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_scale=None,
            attn_drop_rate: float = 0.0,
            attn_out_drop_rate: float = 0.0,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            pos_drop_rate: float = 0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            use_checkpoint: bool = False,
            abs_pos_enc_in: bool = True,
            abs_pos_enc: bool = True,
            rel_pos_bias: bool = True,
            rel_pos_bias_size: int = 7,
            rel_pos_init_variance: float = 0.0,
            debug_skip_attn: bool = False,
            append_self: bool = False,
            coarse_pool_type: str = "mean_max",
            global_query_chunk_size: int = 128,
            return_aux: bool = False,
            debug_uahs: bool = False,
            **unused_kwargs,
    ):
        super().__init__()
        del in_scale_factor, qk_scale, debug_skip_attn, unused_kwargs
        if img_rank < 2:
            raise ValueError("The L4 ablation requires img_rank >= 2")
        if node_type != "vertex":
            raise ValueError("The L4 ablation uses vertex features")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.img_rank = img_rank
        self.fine_rank = img_rank
        self.middle_rank = img_rank - 1
        self.coarse_rank = img_rank - 2
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.return_aux = return_aux
        self.debug_uahs = debug_uahs
        self.icosphere_ref = IcoSphereRef(node_type="vertex")

        self.hierarchy_l4_l6 = IcoSphereHierarchy(
            self.coarse_rank, self.fine_rank, self.icosphere_ref
        )
        self.upsample_l4_l5 = InterpolateUpsample(
            self.coarse_rank, self.middle_rank, self.icosphere_ref
        )
        self.upsample_l5_l6 = InterpolateUpsample(
            self.middle_rank, self.fine_rank, self.icosphere_ref
        )

        self.input_proj_l6 = InputProj(in_channels, embed_dim, act_layer=act_layer)
        self.coarse_region_pool = HierarchicalRegionPool(
            embed_dim, pool_type=coarse_pool_type
        )
        self.apply_abs_pos_enc_in = abs_pos_enc_in
        if abs_pos_enc_in:
            self.abs_pos_l4 = self._input_position_encoding(
                self.coarse_rank, embed_dim
            )
        self.pos_drop = nn.Dropout(pos_drop_rate)

        # SphereUFormerModule(depth=2) is exactly the two retained local blocks.
        self.coarse_local_encoder = SphereUFormerModule(
            rank=self.coarse_rank,
            icosphere_ref=self.icosphere_ref,
            dim=embed_dim,
            depth=2,
            num_heads=num_heads,
            d_head_coef=d_head_coef,
            win_size_coef=win_size_coef,
            temporal_window_radius=temporal_window_radius,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_rate,
            attn_out_drop=attn_out_drop_rate,
            mlp_drop=drop_rate,
            drop_path=drop_path_rate,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            abs_pos_enc=abs_pos_enc,
            rel_pos_bias=rel_pos_bias,
            rel_pos_bias_size=rel_pos_bias_size,
            rel_pos_init_variance=rel_pos_init_variance,
            append_self=append_self,
        )
        self.coarse_global_block = GlobalContentSpatioTemporalBlock(
            dim=embed_dim,
            num_heads=num_heads,
            temporal_window_radius=temporal_window_radius,
            d_head_coef=d_head_coef,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_rate,
            out_drop=attn_out_drop_rate,
            drop_path=drop_path_rate,
            query_chunk_size=global_query_chunk_size,
            use_checkpoint=use_checkpoint,
        )

        # The only prediction head is applied after L4 features reach L6.
        self.output_proj = OutputProj(embed_dim, out_channels)
        self.final_sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _input_position_encoding(self, rank, embed_dim):
        return nn.Sequential(
            GlobalVerticalPositionEnconding(
                rank=rank,
                icosphere_ref=self.icosphere_ref,
                mode="phi",
                num_pos_feats=16,
                max_frequency=10000,
                min_frequency=1,
            ),
            nn.Linear(32, embed_dim, bias=False),
        )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: Tensor, return_aux: Optional[bool] = None, **kwargs):
        """Return the L6 saliency map with shape ``[B, T, V_l6]``."""
        del kwargs
        if x.ndim != 4:
            raise ValueError(f"Expected [B,T,V,C], got {tuple(x.shape)}")
        batch_size, time_steps, vertices_l6, channels = x.shape
        if vertices_l6 != self.hierarchy_l4_l6.fine_vertex_count:
            raise ValueError("Input vertex count does not match img_rank")
        flat_img = x.reshape(batch_size * time_steps, vertices_l6, channels)

        features_l6 = self.input_proj_l6(flat_img)
        features_l4 = self.coarse_region_pool(
            features_l6, self.hierarchy_l4_l6
        )
        if self.apply_abs_pos_enc_in:
            features_l4 = features_l4 + self.abs_pos_l4(features_l4)
        features_l4 = self.coarse_local_encoder(
            self.pos_drop(features_l4), time_steps=time_steps
        )
        features_l4 = self.coarse_global_block(
            features_l4, time_steps=time_steps
        )

        features_l5 = self.upsample_l4_l5(features_l4)
        features_l6 = self.upsample_l5_l6(features_l5)
        final_logits = self.output_proj(features_l6)
        saliency = self.final_sigmoid(final_logits)
        if self.out_channels == 1:
            saliency = saliency.squeeze(-1)
        saliency = saliency.reshape(
            batch_size, time_steps, *saliency.shape[1:]
        )

        if self.debug_uahs:
            print(
                "L4 attention ablation:",
                f"ranks={self.coarse_rank}->{self.fine_rank}",
                f"output={tuple(saliency.shape)}",
            )
        return_aux = self.return_aux if return_aux is None else return_aux
        if return_aux:
            return {"saliency": saliency}
        return saliency
