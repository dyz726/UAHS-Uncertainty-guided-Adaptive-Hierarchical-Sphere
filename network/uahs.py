"""Final Uncertainty-guided Adaptive Hierarchical Sphere architecture."""

import math
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
            target_ratio: Optional[Tensor] = None,
    ) -> Tensor:
        return build_fixed_area_target(
            scores,
            face_areas,
            self.target_ratio if target_ratio is None else target_ratio,
            eligible_mask=eligible_mask,
        )


class DynamicBudgetHead(nn.Module):
    """Predict a per-frame budget from detached uncertainty statistics."""

    def __init__(
            self,
            initial_output: float,
            output_min: float = 0.0,
            output_max: float = 1.0,
            hidden_dim: int = 8,
    ):
        super().__init__()
        if not 0 <= output_min < output_max <= 1:
            raise ValueError("Expected 0 <= output_min < output_max <= 1")
        if not output_min <= initial_output <= output_max:
            raise ValueError("initial_output must be inside the output range")
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.initial_output = float(initial_output)
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def reset_output(self):
        """Start as a constant legacy budget while retaining trainability."""
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        normalized = (
            (self.initial_output - self.output_min)
            / (self.output_max - self.output_min)
        )
        normalized = min(max(normalized, 1e-4), 1 - 1e-4)
        nn.init.constant_(final.bias, math.log(normalized / (1 - normalized)))

    def forward(
            self,
            uncertainty: Tensor,
            face_areas: Tensor,
            eligible_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if uncertainty.shape[-1] != face_areas.numel():
            raise ValueError("uncertainty and face_areas have different counts")
        # Budget supervision must never reshape the uncertainty representation.
        uncertainty = uncertainty.detach()
        areas = face_areas.to(
            device=uncertainty.device, dtype=uncertainty.dtype
        )
        weights = areas.reshape(*([1] * (uncertainty.ndim - 1)), -1)
        if eligible_mask is not None:
            if eligible_mask.shape != uncertainty.shape:
                raise ValueError("eligible_mask must match uncertainty")
            weights = weights * eligible_mask.detach().to(weights.dtype)
        denominator = weights.sum(dim=-1).clamp_min(
            torch.finfo(uncertainty.dtype).eps
        )
        mean = (uncertainty * weights).sum(dim=-1) / denominator
        variance = (
            (uncertainty - mean.unsqueeze(-1)).square() * weights
        ).sum(dim=-1) / denominator
        statistics = torch.stack((mean, variance.clamp_min(0).sqrt()), dim=-1)
        unit_budget = torch.sigmoid(self.mlp(statistics).squeeze(-1))
        return self.output_min + (
            self.output_max - self.output_min
        ) * unit_budget


class UAHS(nn.Module):
    """Global-aware coarse modeling with truly sparse hierarchical refinement.

    Expensive rank-5/rank-6 spatial attention and FFNs are evaluated only for
    vertices incident to hard-selected child faces. Dense fine tensors are kept
    solely as the reconstruction canvas and as K/V feature sources.
    """

    SELECTOR_MODES = {
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
            budget_l5_min: float = 0.05,
            budget_l5_max: float = 0.50,
            global_query_chunk_size: int = 128,
            hard_selection_warmup_epochs: int = 0,
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
        if not 0 <= budget_l5_min < budget_l5_max <= 1:
            raise ValueError("Expected 0 <= budget_l5_min < budget_l5_max <= 1")
        if not budget_l5_min <= target_refine_ratio_l1 <= budget_l5_max:
            raise ValueError("Initial L5 budget must be inside its output range")
        if target_refine_ratio_l1 <= 0:
            raise ValueError("Initial L5 budget must be positive")
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
        self.budget_l5_min = float(budget_l5_min)
        self.budget_l5_max = float(budget_l5_max)
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
        self.budget_head_l4 = DynamicBudgetHead(
            initial_output=target_refine_ratio_l1,
            output_min=budget_l5_min,
            output_max=budget_l5_max,
        )
        self.budget_head_l5 = DynamicBudgetHead(
            initial_output=target_refine_ratio_l2 / target_refine_ratio_l1,
        )
        self.hard_selector_l4 = HardAreaSelector(target_refine_ratio_l1)
        self.hard_selector_l5 = HardAreaSelector(target_refine_ratio_l2)
        self.fusion_norm_l5 = norm_layer(embed_dim)
        self.fusion_norm_l6 = norm_layer(embed_dim)
        self.output_proj = OutputProj(embed_dim, out_channels)
        self.final_sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)
        # ``apply`` initializes every Linear first; restore legacy budget biases
        # last so old and freshly initialized models start at 0.25 / 0.125.
        self.budget_head_l4.reset_output()
        self.budget_head_l5.reset_output()

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
            return torch.rand_like(uncertainty)
        if mode == "random_same_budget":
            # Evaluation baselines remain reproducible for a requested seed.
            generator = torch.Generator(device=uncertainty.device)
            generator.manual_seed(int(seed or 0))
            return torch.rand(
                uncertainty.shape,
                device=uncertainty.device,
                dtype=uncertainty.dtype,
                generator=generator,
            )
        if mode == "uncertainty_only":
            return uncertainty.detach()
        if mode == "saliency_score":
            return saliency.detach()
        raise AssertionError("Unreachable selector mode")

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
            selector_mode: str = "uncertainty_only",
            selector_seed: int = 0,
            hard_mask_overrides: Optional[dict] = None,
            disable_l6_refinement: bool = False,
    ):
        """Predict saliency without labels; overrides are evaluation-only masks.

        ``disable_l6_refinement`` is an evaluation-only ablation. It preserves
        the uncertainty-based L4/L5 routing, dense L5 representation, L5->L6 base
        reconstruction, fusion normalization, and final output head, while
        skipping only the selected-query rank-6 residual computation.
        """
        if disable_l6_refinement and self.training:
            raise ValueError(
                "disable_l6_refinement is an evaluation-only ablation"
            )
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
        budget_l5_pred = self.budget_head_l4(
            uncertainty_l4, self.hierarchy_l4_l5.coarse_face_areas
        )
        score_l4 = self._selection_scores(
            selector_mode,
            uncertainty_l4,
            saliency_l4,
            selector_seed,
        )
        hard_l4 = self.hard_selector_l4(
            score_l4,
            self.hierarchy_l4_l5.coarse_face_areas,
            target_ratio=budget_l5_pred,
        )
        if hard_mask_overrides and "l4" in hard_mask_overrides:
            hard_l4 = self._validate_override(
                hard_mask_overrides["l4"], hard_l4, "hard_mask_overrides['l4']"
            )
        selected_area_l1 = self._area_ratio(
            hard_l4, self.hierarchy_l4_l5.coarse_face_areas
        )
        child_faces_l5 = self.hierarchy_l4_l5.propagate_coarse_face_values(
            hard_l4
        )
        selected_vertices_l5 = self._vertex_mask(
            self.hierarchy_l4_l5, child_faces_l5
        )
        # The hard mask decides compute and reconstruction. Face-to-vertex mean
        # reduction naturally gives fractional weights only at region borders.
        weight_faces_l5 = child_faces_l5
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
        eligible_l5 = child_faces_l5.bool()
        budget_l6_alpha = self.budget_head_l5(
            uncertainty_l5,
            self.hierarchy_l5_l6.coarse_face_areas,
            eligible_mask=eligible_l5,
        )
        # Base the child budget on the area that was actually routed into L5.
        # The hard selected area is detached/non-differentiable, so L6 budget
        # supervision cannot update the head that predicts B5.
        budget_l6_pred = selected_area_l1.detach() * budget_l6_alpha
        score_l5 = self._selection_scores(
            selector_mode,
            uncertainty_l5,
            saliency_l5,
            selector_seed + 1,
        )
        hard_l5_local = self.hard_selector_l5(
            score_l5,
            self.hierarchy_l5_l6.coarse_face_areas,
            eligible_mask=eligible_l5,
            target_ratio=budget_l6_pred,
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
        weight_faces_l6 = child_faces_l6
        weight_vertices_l6 = self.hierarchy_l5_l6.fine_face_values_to_vertices(
            weight_faces_l6
        )

        base_l6 = self.upsample_l5_l6(features_l5)
        if disable_l6_refinement:
            # This is exactly the no-query branch of _scatter_refinement and
            # therefore keeps the reconstruction/output path comparable.
            features_l6 = self.fusion_norm_l6(base_l6)
        else:
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
        query_count_l6 = (
            torch.zeros_like(selected_vertices_l6.sum(dim=-1))
            if disable_l6_refinement
            else selected_vertices_l6.sum(dim=-1)
        )
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
                f"budget={budget_l5_pred.mean().item():.4f}/"
                f"{budget_l6_pred.mean().item():.4f}",
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
            "budget_l5_pred": budget_l5_pred,
            "hard_face_mask_l4": hard_l4,
            "selected_area_l1": selected_area_l1,
            "saliency_l5": saliency_l5,
            "uncertainty_l5": uncertainty_l5,
            "budget_l6_alpha": budget_l6_alpha,
            "budget_l6_pred": budget_l6_pred,
            "eligible_face_mask_l5": eligible_l5,
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
