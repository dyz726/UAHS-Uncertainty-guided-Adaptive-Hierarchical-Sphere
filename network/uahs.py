"""Final Uncertainty-guided Adaptive Hierarchical Sphere architecture."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch import Tensor

from adaptive_objectives import build_fixed_area_target
from trimesh_utils import IcoSphereHierarchy, IcoSphereRef

from .position_encoding import GlobalVerticalPositionEnconding
from .sphere_model import (
    HierarchicalRegionPool,
    InputProj,
    InterpolateUpsample,
    OutputProj,
    RefinementHead,
    SphereUFormerModule,
    SphericalUncertaintyHead,
    VertexToFaceAggregation,
)
from .sphere_PSA import (
    GlobalContentSpatioTemporalBlock,
    SparseLocalRefinementBlock,
)


class HardAreaSelector(nn.Module):
    """Non-differentiable, spherical-area-aware hard face selector."""

    def __init__(self, target_ratio: float):
        super().__init__()
        if not 0 <= target_ratio <= 1:
            raise ValueError("target_ratio must be between zero and one")
        self.target_ratio = float(target_ratio)

    @torch.no_grad()
    def forward(
            self,
            scores: Tensor,
            face_areas: Tensor,
            eligible_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return build_fixed_area_target(
            scores,
            face_areas,
            self.target_ratio,
            eligible_mask=eligible_mask,
        )


class UAHS(nn.Module):
    """Global-aware coarse modeling with truly sparse hierarchical refinement.

    Expensive rank-5/rank-6 spatial attention and FFNs are evaluated only for
    vertices incident to hard-selected child faces. Dense fine tensors are kept
    solely as the reconstruction canvas and as K/V feature sources.
    """

    SELECTOR_MODES = {
        "learned_refinement_score",
        "uncertainty_only",
        "saliency_score",
        "random_same_budget",
    }

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
            target_refine_ratio_l1: float = 0.25,
            target_refine_ratio_l2: float = 0.125,
            global_query_chunk_size: int = 128,
            hard_selection_warmup_epochs: int = 0,
            use_uncertainty_refinement: bool = True,
            use_motion_refinement: bool = True,
            return_aux: bool = False,
            debug_uahs: bool = False,
    ):
        super().__init__()
        del in_scale_factor, qk_scale, debug_skip_attn
        if img_rank < 2:
            raise ValueError("UAHS requires img_rank >= 2")
        if node_type != "vertex":
            raise ValueError("UAHS uses a vertex backbone and face regions")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0 <= target_refine_ratio_l2 <= target_refine_ratio_l1 <= 1:
            raise ValueError("Expected 0 <= L2 budget <= L1 budget <= 1")
        if hard_selection_warmup_epochs < 0:
            raise ValueError("hard_selection_warmup_epochs must be non-negative")

        self.img_rank = img_rank
        self.fine_rank = img_rank
        self.middle_rank = img_rank - 1
        self.coarse_rank = img_rank - 2
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.target_refine_ratio_l1 = float(target_refine_ratio_l1)
        self.target_refine_ratio_l2 = float(target_refine_ratio_l2)
        self.hard_selection_warmup_epochs = hard_selection_warmup_epochs
        self.current_epoch = 0
        self.return_aux = return_aux
        self.debug_uahs = debug_uahs
        self.icosphere_ref = IcoSphereRef(node_type="vertex")

        self.hierarchy_l4_l5 = IcoSphereHierarchy(
            self.coarse_rank, self.middle_rank, self.icosphere_ref
        )
        self.hierarchy_l5_l6 = IcoSphereHierarchy(
            self.middle_rank, self.fine_rank, self.icosphere_ref
        )
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
        self.raw_region_pool_l5 = HierarchicalRegionPool(
            embed_dim, pool_type=coarse_pool_type
        )
        self.coarse_region_pool = HierarchicalRegionPool(
            embed_dim, pool_type=coarse_pool_type
        )
        self.context_projection_l5 = nn.Linear(embed_dim, embed_dim)
        self.context_projection_l6 = nn.Linear(embed_dim, embed_dim)
        self.apply_abs_pos_enc_in = abs_pos_enc_in
        if abs_pos_enc_in:
            self.abs_pos_l4 = self._input_position_encoding(
                self.coarse_rank, embed_dim
            )
            self.abs_pos_l5 = self._input_position_encoding(
                self.middle_rank, embed_dim
            )
            self.abs_pos_l6 = self._input_position_encoding(
                self.fine_rank, embed_dim
            )
        self.pos_drop = nn.Dropout(pos_drop_rate)

        local_args = dict(
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
        # Exactly two unchanged local motion-aware blocks.
        self.coarse_local_encoder = SphereUFormerModule(
            rank=self.coarse_rank, **local_args
        )
        # One content-aware temporal + true full-sphere spatial block.
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

        sparse_args = dict(
            icosphere_ref=self.icosphere_ref,
            dim=embed_dim,
            num_heads=num_heads,
            d_head_coef=d_head_coef,
            win_size_coef=win_size_coef,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_rate,
            out_drop=attn_out_drop_rate,
            drop_path=drop_path_rate,
            abs_pos_enc=abs_pos_enc,
            rel_pos_bias=rel_pos_bias,
            rel_pos_bias_size=rel_pos_bias_size,
            rel_pos_init_variance=rel_pos_init_variance,
            append_self=append_self,
        )
        self.sparse_refiner_l5 = SparseLocalRefinementBlock(
            rank=self.middle_rank, **sparse_args
        )
        self.sparse_refiner_l6 = SparseLocalRefinementBlock(
            rank=self.fine_rank, **sparse_args
        )

        self.vertex_to_face_l4 = VertexToFaceAggregation(embed_dim)
        self.vertex_to_face_l5 = VertexToFaceAggregation(embed_dim)
        self.saliency_head_l4 = nn.Linear(embed_dim, out_channels)
        self.saliency_head_l5 = nn.Linear(embed_dim, out_channels)
        self.uncertainty_head_l4 = SphericalUncertaintyHead(embed_dim)
        self.uncertainty_head_l5 = SphericalUncertaintyHead(embed_dim)
        self.refinement_head_l4 = RefinementHead(
            embed_dim,
            use_uncertainty=use_uncertainty_refinement,
            use_motion=use_motion_refinement,
        )
        self.refinement_head_l5 = RefinementHead(
            embed_dim,
            use_uncertainty=use_uncertainty_refinement,
            use_motion=use_motion_refinement,
        )
        self.hard_selector_l4 = HardAreaSelector(target_refine_ratio_l1)
        self.hard_selector_l5 = HardAreaSelector(target_refine_ratio_l2)
        self.fusion_norm_l5 = norm_layer(embed_dim)
        self.fusion_norm_l6 = norm_layer(embed_dim)
        self.output_proj = OutputProj(embed_dim, out_channels)
        self.final_sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

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

    @staticmethod
    def _validate_override(mask, reference, name):
        if mask.shape != reference.shape:
            raise ValueError(
                f"{name} must have shape {tuple(reference.shape)}, "
                f"got {tuple(mask.shape)}"
            )
        return mask.to(device=reference.device).bool().to(reference.dtype)

    @staticmethod
    def _area_ratio(mask, areas):
        areas = areas.to(device=mask.device, dtype=mask.dtype)
        return (mask * areas).sum(dim=-1) / areas.sum()

    @staticmethod
    def _vertex_mask(hierarchy, face_mask):
        return hierarchy.fine_face_values_to_vertices(face_mask) > 0

    @staticmethod
    def _scatter_refinement(base, selected_candidate, query_pairs, weight, norm):
        if query_pairs.shape[0] == 0:
            return norm(base)
        selected_base = base[query_pairs[:, 0], query_pairs[:, 1]]
        selected_weight = weight[
            query_pairs[:, 0], query_pairs[:, 1]
        ].unsqueeze(-1)
        residual = selected_weight * (selected_candidate - selected_base)
        dense_residual = torch.zeros_like(base).index_put(
            (query_pairs[:, 0], query_pairs[:, 1]), residual
        )
        return norm(base + dense_residual)

    def _selection_scores(
            self,
            mode,
            refine_logits,
            uncertainty,
            saliency,
            seed,
    ):
        if mode not in self.SELECTOR_MODES:
            raise ValueError(f"Unknown UAHS selector mode: {mode}")
        warmup = (
            self.training
            and self.hard_selection_warmup_epochs > 0
            and self.current_epoch <= self.hard_selection_warmup_epochs
        )
        if warmup:
            # Training warm-up must explore different regions on successive
            # batches. Use the process RNG instead of recreating a fixed-seed
            # generator on every forward.
            return torch.rand_like(refine_logits)
        if mode == "random_same_budget":
            # Evaluation baselines remain reproducible for a requested seed.
            generator = torch.Generator(device=refine_logits.device)
            generator.manual_seed(int(seed or 0))
            return torch.rand(
                refine_logits.shape,
                device=refine_logits.device,
                dtype=refine_logits.dtype,
                generator=generator,
            )
        if mode == "uncertainty_only":
            return uncertainty.detach()
        if mode == "saliency_score":
            return saliency.detach()
        return refine_logits.detach()

    def aggregate_img_values_to_l5_faces(self, values: Tensor) -> Tensor:
        rank6_faces = self.hierarchy_l5_l6.vertex_values_to_faces(
            values, level="fine"
        )
        return self.hierarchy_l5_l6.aggregate_fine_face_values(
            rank6_faces, area_weighted=True
        )

    def aggregate_img_values_to_l4_faces(self, values: Tensor) -> Tensor:
        rank5_faces = self.aggregate_img_values_to_l5_faces(values)
        return self.hierarchy_l4_l5.aggregate_fine_face_values(
            rank5_faces, area_weighted=True
        )

    def aggregate_img_values_to_coarse_faces(self, values: Tensor) -> Tensor:
        return self.aggregate_img_values_to_l4_faces(values)

    def upsample_l5_values_to_img(self, values: Tensor) -> Tensor:
        rank6_faces = self.hierarchy_l5_l6.propagate_coarse_face_values(values)
        return self.hierarchy_l5_l6.fine_face_values_to_vertices(rank6_faces)

    def upsample_l4_values_to_img(self, values: Tensor) -> Tensor:
        rank5_faces = self.hierarchy_l4_l5.propagate_coarse_face_values(values)
        return self.upsample_l5_values_to_img(rank5_faces)

    def upsample_coarse_values_to_img(self, values: Tensor) -> Tensor:
        return self.upsample_l4_values_to_img(values)

    def forward(
            self,
            x: Tensor,
            return_aux: Optional[bool] = None,
            selector_mode: str = "learned_refinement_score",
            selector_seed: int = 0,
            hard_mask_overrides: Optional[dict] = None,
    ):
        """Predict saliency without labels; overrides are evaluation-only masks."""
        if x.ndim != 4:
            raise ValueError(f"Expected [B,T,V,C], got {tuple(x.shape)}")
        batch_size, time_steps, vertices_l6, channels = x.shape
        if vertices_l6 != self.hierarchy_l5_l6.fine_vertex_count:
            raise ValueError("Input vertex count does not match img_rank")
        batch_frames = batch_size * time_steps
        flat_img = x.reshape(batch_frames, vertices_l6, channels)

        raw_l6 = self.input_proj_l6(flat_img)
        # Preserve rank-6 observations inside every rank-5 region;
        # interpolation is reserved for context, never used as detail.
        raw_l5 = self.raw_region_pool_l5(raw_l6, self.hierarchy_l5_l6)
        features_l4 = self.coarse_region_pool(raw_l6, self.hierarchy_l4_l6)
        if self.apply_abs_pos_enc_in:
            features_l4 = features_l4 + self.abs_pos_l4(features_l4)
            raw_l5 = raw_l5 + self.abs_pos_l5(raw_l5)
            raw_l6 = raw_l6 + self.abs_pos_l6(raw_l6)
        features_l4 = self.coarse_local_encoder(
            self.pos_drop(features_l4), time_steps=time_steps
        )
        features_l4 = self.coarse_global_block(features_l4, time_steps=time_steps)

        face_l4 = self.vertex_to_face_l4(
            features_l4, self.hierarchy_l4_l5
        ).reshape(
            batch_size,
            time_steps,
            self.hierarchy_l4_l5.coarse_face_count,
            self.embed_dim,
        )
        saliency_l4 = self.final_sigmoid(
            self.saliency_head_l4(face_l4).squeeze(-1)
        )
        uncertainty_l4 = self.uncertainty_head_l4(face_l4)
        refine_logits_l4, refine_score_l4 = self.refinement_head_l4(
            face_l4, uncertainty_l4
        )
        score_l4 = self._selection_scores(
            selector_mode,
            refine_logits_l4,
            uncertainty_l4,
            saliency_l4,
            selector_seed,
        )
        hard_l4 = self.hard_selector_l4(
            score_l4, self.hierarchy_l4_l5.coarse_face_areas
        )
        if hard_mask_overrides and "l4" in hard_mask_overrides:
            hard_l4 = self._validate_override(
                hard_mask_overrides["l4"], hard_l4, "hard_mask_overrides['l4']"
            )
        child_faces_l5 = self.hierarchy_l4_l5.propagate_coarse_face_values(
            hard_l4
        )
        selected_vertices_l5 = self._vertex_mask(
            self.hierarchy_l4_l5, child_faces_l5
        )
        weight_faces_l5 = child_faces_l5 * self.hierarchy_l4_l5.propagate_coarse_face_values(
            torch.sigmoid(refine_logits_l4)
        )
        weight_vertices_l5 = self.hierarchy_l4_l5.fine_face_values_to_vertices(
            weight_faces_l5
        )

        base_l5 = self.upsample_l4_l5(features_l4)
        candidate_input_l5 = self.pos_drop(
            raw_l5 + self.context_projection_l5(base_l5)
        )
        candidate_l5, query_pairs_l5 = self.sparse_refiner_l5(
            candidate_input_l5,
            selected_vertices_l5.reshape(batch_frames, -1),
        )
        features_l5 = self._scatter_refinement(
            base_l5,
            candidate_l5,
            query_pairs_l5,
            weight_vertices_l5.reshape(batch_frames, -1),
            self.fusion_norm_l5,
        )

        face_l5 = self.vertex_to_face_l5(
            features_l5, self.hierarchy_l5_l6
        ).reshape(
            batch_size,
            time_steps,
            self.hierarchy_l5_l6.coarse_face_count,
            self.embed_dim,
        )
        saliency_l5 = self.final_sigmoid(
            self.saliency_head_l5(face_l5).squeeze(-1)
        )
        uncertainty_l5 = self.uncertainty_head_l5(face_l5)
        refine_logits_l5, refine_score_l5 = self.refinement_head_l5(
            face_l5, uncertainty_l5
        )
        eligible_l5 = child_faces_l5.bool()
        score_l5 = self._selection_scores(
            selector_mode,
            refine_logits_l5,
            uncertainty_l5,
            saliency_l5,
            selector_seed + 1,
        )
        hard_l5_local = self.hard_selector_l5(
            score_l5,
            self.hierarchy_l5_l6.coarse_face_areas,
            eligible_mask=eligible_l5,
        )
        if hard_mask_overrides and "l5" in hard_mask_overrides:
            hard_l5_local = self._validate_override(
                hard_mask_overrides["l5"],
                hard_l5_local,
                "hard_mask_overrides['l5']",
            )
        hard_l5_effective = hard_l5_local * eligible_l5.to(hard_l5_local.dtype)
        child_faces_l6 = self.hierarchy_l5_l6.propagate_coarse_face_values(
            hard_l5_effective
        )
        selected_vertices_l6 = self._vertex_mask(
            self.hierarchy_l5_l6, child_faces_l6
        )
        weight_faces_l6 = child_faces_l6 * self.hierarchy_l5_l6.propagate_coarse_face_values(
            torch.sigmoid(refine_logits_l5)
        )
        weight_vertices_l6 = self.hierarchy_l5_l6.fine_face_values_to_vertices(
            weight_faces_l6
        )

        base_l6 = self.upsample_l5_l6(features_l5)
        candidate_input_l6 = self.pos_drop(
            raw_l6 + self.context_projection_l6(base_l6)
        )
        candidate_l6, query_pairs_l6 = self.sparse_refiner_l6(
            candidate_input_l6,
            selected_vertices_l6.reshape(batch_frames, -1),
        )
        features_l6 = self._scatter_refinement(
            base_l6,
            candidate_l6,
            query_pairs_l6,
            weight_vertices_l6.reshape(batch_frames, -1),
            self.fusion_norm_l6,
        )
        final_logits = self.output_proj(features_l6)
        if self.out_channels == 1:
            saliency = self.final_sigmoid(
                final_logits.squeeze(-1).reshape(
                    batch_size, time_steps, vertices_l6
                )
            )
        else:
            saliency = self.final_sigmoid(final_logits.reshape(
                batch_size, time_steps, vertices_l6, self.out_channels
            ))

        selected_area_l1 = self._area_ratio(
            hard_l4, self.hierarchy_l4_l5.coarse_face_areas
        )
        selected_area_l2 = self._area_ratio(
            hard_l5_effective, self.hierarchy_l5_l6.coarse_face_areas
        )
        selected_vertex_ratio_l5 = selected_vertices_l5.float().mean(dim=-1)
        selected_vertex_ratio_l6 = selected_vertices_l6.float().mean(dim=-1)

        entered_l5_faces_l6 = self.hierarchy_l5_l6.propagate_coarse_face_values(
            child_faces_l5
        )
        entered_l5_vertices_l6 = self._vertex_mask(
            self.hierarchy_l5_l6, entered_l5_faces_l6
        )
        exit_level = (
            4
            + entered_l5_vertices_l6.to(torch.int64)
            + selected_vertices_l6.to(torch.int64)
        )

        active_faces_l5 = child_faces_l5.sum(dim=-1)
        active_faces_l6 = child_faces_l6.sum(dim=-1)
        query_count_l5 = selected_vertices_l5.sum(dim=-1)
        query_count_l6 = selected_vertices_l6.sum(dim=-1)
        dense_query_count_l5 = torch.full_like(
            query_count_l5, self.hierarchy_l4_l5.fine_vertex_count
        )
        dense_query_count_l6 = torch.full_like(
            query_count_l6, self.hierarchy_l5_l6.fine_vertex_count
        )
        key_count_l5 = self.sparse_refiner_l5.attention.num_keys
        key_count_l6 = self.sparse_refiner_l6.attention.num_keys
        # QK and attention-value interactions, counting multiply and add as two
        # FLOPs each. Projection/FFN work also scales with selected queries but
        # is intentionally excluded from this transparent estimate.
        sparse_flops_l5 = query_count_l5 * key_count_l5 * self.embed_dim * 4
        sparse_flops_l6 = query_count_l6 * key_count_l6 * self.embed_dim * 4
        dense_flops_l5 = dense_query_count_l5 * key_count_l5 * self.embed_dim * 4
        dense_flops_l6 = dense_query_count_l6 * key_count_l6 * self.embed_dim * 4
        query_reduction_l5 = 1 - query_count_l5.float() / dense_query_count_l5
        query_reduction_l6 = 1 - query_count_l6.float() / dense_query_count_l6

        if self.debug_uahs:
            print(
                "UAHS:",
                f"ranks={self.coarse_rank}->{self.middle_rank}->{self.fine_rank}",
                f"area={selected_area_l1.mean().item():.4f}/"
                f"{selected_area_l2.mean().item():.4f}",
                f"vertex_ratio={selected_vertex_ratio_l5.mean().item():.4f}/"
                f"{selected_vertex_ratio_l6.mean().item():.4f}",
                f"queries={int(query_count_l5.sum())}/{int(query_count_l6.sum())}",
            )

        return_aux = self.return_aux if return_aux is None else return_aux
        if not return_aux:
            return saliency
        return {
            "saliency": saliency,
            "saliency_l4": saliency_l4,
            "uncertainty_l4": uncertainty_l4,
            "refine_logits_l4": refine_logits_l4,
            "refine_score_l4": refine_score_l4,
            "hard_face_mask_l4": hard_l4,
            "selected_area_l1": selected_area_l1,
            "saliency_l5": saliency_l5,
            "uncertainty_l5": uncertainty_l5,
            "refine_logits_l5": refine_logits_l5,
            "refine_score_l5": refine_score_l5,
            "hard_face_mask_l5_local": hard_l5_local,
            "hard_face_mask_l5_effective": hard_l5_effective,
            "selected_area_l2": selected_area_l2,
            "selected_vertex_ratio_l5": selected_vertex_ratio_l5,
            "selected_vertex_ratio_l6": selected_vertex_ratio_l6,
            "active_refinement_faces_l5": active_faces_l5,
            "active_refinement_faces_l6": active_faces_l6,
            "selected_spatial_queries_l5": query_count_l5,
            "selected_spatial_queries_l6": query_count_l6,
            "dense_spatial_queries_l5": dense_query_count_l5,
            "dense_spatial_queries_l6": dense_query_count_l6,
            "spatial_query_reduction_l5": query_reduction_l5,
            "spatial_query_reduction_l6": query_reduction_l6,
            "estimated_sparse_attention_flops_l5": sparse_flops_l5,
            "estimated_sparse_attention_flops_l6": sparse_flops_l6,
            "estimated_dense_attention_flops_l5": dense_flops_l5,
            "estimated_dense_attention_flops_l6": dense_flops_l6,
            "exit_level": exit_level,
        }
