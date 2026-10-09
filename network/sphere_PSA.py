from .position_encoding import RelativePositionBias
from trimesh_utils import IcoSphereRef

import math
from typing import Optional,List
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from timm.models.layers import DropPath


class SphereSpatioTemporalAttention(nn.Module):
    """
    球形局部时空注意力模块
    输入: [N, D, C] 其中 N = B*T, D=球形采样点数, C=特征维度
    """

    def __init__(
            self,
                     
            rank: int,
            icosphere_ref,
            win_size_coef: int,
                     
            temporal_window_radius: Optional[int] = 5,
                  
            num_heads: int = 8,
            d_model: int = 512,
            d_head_coef: int = 1,
            qkv_bias: bool = True,
            attn_drop: float = 0.1,
            out_drop: float = 0.1,
                  
            abs_pos_enc: bool = False,
            abs_pos_enc_size: int = 0,
            rel_pos_bias: bool = True,
            rel_pos_bias_size: int = 7,
            rel_pos_init_variance: float = 0.0,
            append_self: bool = False,
    ):
        super().__init__()
        self.batch_size = None
        self.time_steps = None
                   
        self.spatial_attention = SphereSelfAttention(
            rank=rank,
            icosphere_ref=icosphere_ref,
            win_size_coef=win_size_coef,
            num_heads=num_heads,
            d_model=d_model,
            d_head_coef=d_head_coef,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            out_drop=out_drop,
            abs_pos_enc=abs_pos_enc,
            abs_pos_enc_size=abs_pos_enc_size,
            rel_pos_bias=rel_pos_bias,
            rel_pos_bias_size=rel_pos_bias_size,
            rel_pos_init_variance=rel_pos_init_variance,
            append_self = append_self,
        )

               
        self.temporal_attention_diff = Diff_TemporalAttention(
            d_model=d_model,
            num_heads=num_heads,
            temporal_window_radius=temporal_window_radius,
            d_head_coef=d_head_coef,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            out_drop=out_drop,
        )


    def forward(
            self,
            x: Tensor,                        
            time_steps: int,                     
            pos: Optional[Tensor] = None,                       
    ):
        """
        前向传播
        x: 输入张量 [B*T, Vertices, Channels]
        """
        N, D, C = x.shape
        batch_size = N // time_steps
        assert N == batch_size * time_steps, f"输入形状不匹配: N={N}, B*T={batch_size}*{time_steps}={batch_size * time_steps}"
                       
                                   
        x_reshaped = x.view(batch_size, time_steps, D, C)
                 
        out_diff = self.temporal_attention_diff(x_reshaped)                
                             
        temporal_out = out_diff.reshape(batch_size * time_steps, D, C)
        temporal_out = temporal_out + x

                           
        spatial_out = self.spatial_attention(temporal_out, pos)               

        output = spatial_out + temporal_out

        return output

class Diff_TemporalAttention(nn.Module):
    """
    时间注意力模块
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_head_coef: int = 1,
            temporal_window_radius: Optional[int] = 5,
            qkv_bias: bool = True,
            attn_drop: float = 0.1,
            out_drop: float = 0.1,
                      
            use_frame_diff: bool = True,              
    ):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = (d_model // num_heads) * d_head_coef
        self.scale = self.d_head ** -0.5

                
        self.use_frame_diff = use_frame_diff
        if temporal_window_radius is not None and temporal_window_radius < 0:
            raise ValueError("temporal_window_radius must be non-negative or None")
        self.temporal_window_radius = temporal_window_radius

               
        self.q_proj = nn.Linear(d_model, self.d_head * num_heads, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, self.d_head * num_heads, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, self.d_head * num_heads, bias=qkv_bias)

              
        self.out_proj = nn.Linear(self.d_head * num_heads, d_model)

                 
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_drop = nn.Dropout(out_drop)
                
        self.register_buffer('_temporal_mask_cache', None, persistent=False)
        self._cached_temporal_size = None
    def create_temporal_mask(self, T, device):
        if self.temporal_window_radius is None:
            return None
        if (self._temporal_mask_cache is not None and
                self._cached_temporal_size == T):
            return self._temporal_mask_cache

        mask = torch.full((T, T), float('-inf'), device=device)

        for i in range(T):
                                                    
            start = max(0, i - self.temporal_window_radius)
            end = min(T, i + self.temporal_window_radius + 1)
            mask[i, start:end] = 0

        self._temporal_mask_cache = mask
        self._cached_temporal_size = T
        return mask

    def compute_frame_difference(self, x: Tensor) -> Tensor:
        """
        x: [B, T, D, C]
        返回: [B, T, D, C] 帧差特征
        """
        B, T, D, C = x.shape

        frame_diff = torch.zeros_like(x)

        if T == 0:
            return frame_diff

               
        frame_diff[:, 0] = 0

                     
        if T > 1:
            frame_diff[:, 1:] = x[:, 1:] - x[:, :-1]

        return frame_diff


    def forward(self, x: Tensor):
        """前向传播
        x: [B, T, D, C] 
        输出: [B, T, D, C]
        """
        B, T, D, C = x.shape

        frame_diff = x
                
        if self.use_frame_diff:
            frame_diff = self.compute_frame_difference(x)

                                   
        diff_reshaped = frame_diff.permute(0, 2, 1, 3).contiguous()                
        diff_reshaped = diff_reshaped.view(B * D, T, C)               
               
        q = self.q_proj(diff_reshaped).view(B * D, T, self.num_heads, self.d_head).permute(0, 2, 1, 3)                    
        k = self.k_proj(diff_reshaped).view(B * D, T, self.num_heads, self.d_head).permute(0, 2, 1, 3)                    
                
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * D, T, C)
        v = self.v_proj(x_reshaped).view(B * D, T, self.num_heads, self.d_head).permute(0, 2, 1, 3)                    

                                
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn * self.scale

                  
        temporal_mask = self.create_temporal_mask(T, x.device)
        if temporal_mask is not None:
            attn = attn + temporal_mask.unsqueeze(0).unsqueeze(0)

                         
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

                               
        out = torch.matmul(attn, v)

                           
        out = out.permute(0, 2, 1, 3).contiguous().view(B * D, T, -1)                   
        output = self.out_proj(out)               

        out = output.reshape(B, D, T, C).permute(0, 2, 1, 3).contiguous()               
        output = self.out_drop(out)


        return output

class SphereSelfAttention(nn.Module):
    LOGIT_SCALE_PRE_REL_BIAS: bool = True

    def __init__(
            self, *,
            rank: int,
            icosphere_ref: IcoSphereRef,
            win_size_coef,
            num_heads,
            d_model,
            d_head_coef,
            qkv_bias,
                         
            attn_drop=0.,
            out_drop=0.,
            abs_pos_enc: bool = False,
            abs_pos_enc_size: int = 0,
            rel_pos_bias: bool = False,
            rel_pos_bias_size: int = 0,
            rel_pos_init_variance: float = 0.0,
            append_self: bool = False,
    ):
        """
        :param num_heads: number of self attention head
        :param d_model: dimension of model
        :param dropout:
        :param num_keys: number of keys
        """

        super().__init__()

        self.rank = rank
        self.icosphere_ref = icosphere_ref
        self.win_size_coef = win_size_coef

        self.apply_rel_pos_bias = rel_pos_bias
        self.rel_pos_bias = RelativePositionBias(rank, icosphere_ref, win_size_coef, rel_pos_bias_size=rel_pos_bias_size, num_heads=num_heads, init_variance=rel_pos_init_variance)
        self.num_keys = self.rel_pos_bias.num_keys

                                                                     
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = (d_model // num_heads) * d_head_coef
        assert self.d_head == int(self.d_head)

        self.append_self = append_self

        self.apply_abs_pos_enc = abs_pos_enc
        if self.apply_abs_pos_enc:
            self.q_abs_pos_proj = nn.Linear(abs_pos_enc_size, self.d_head * num_heads, bias=False)
            self.k_abs_pos_proj = nn.Linear(abs_pos_enc_size, self.d_head * num_heads, bias=False)

        self.q_proj = nn.Linear(d_model, self.d_head * num_heads, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, self.d_head * num_heads, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, self.d_head * (num_heads + append_self), bias=qkv_bias)

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
                                                                  

        self.out_proj = nn.Linear(self.d_head * (num_heads + append_self), d_model)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_drop = nn.Dropout(out_drop)

    def forward(
            self,
            x: Tensor,
            pos: Optional[Tensor],
            query_mask: Tensor = None,
            key_masks: Optional[Tensor] = None,

    ):
        """
        :param x: B, D, C
        :param query_mask:
        :param key_masks:
        :return:
        """
                               
                                            

        metadata = None

        N, D, C = x.shape
        H = self.num_heads
        K = self.num_keys
        C_H = self.d_head

                                
        q = self.q_proj(x).view(N, D, H, C_H).permute(0,2,1,3)
        k = self.k_proj(x).view(N, D, H, C_H).permute(0,2,1,3)
        v = self.v_proj(x).view(N, D, H + self.append_self, C_H).permute(0,2,1,3)
        if self.append_self:
            v, v_self = torch.split(v, (H, 1), dim=1)
            v_self = v_self.squeeze(1)

        if self.apply_abs_pos_enc:
            assert D == pos.shape[1]
            q_pos = self.q_abs_pos_proj(pos).view(1, D, H, C_H).permute(0,2,1,3)
            k_pos = self.k_abs_pos_proj(pos).view(1, D, H, C_H).permute(0,2,1,3)
            q = q + q_pos
            k = k + k_pos

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

                                     
                                                                                   
                                                                                  
        expanded_idx = self.rel_pos_bias.idx.expand(N, H, -1, -1, C_H)
        expanded_idx_mask = self.rel_pos_bias.idx_mask.expand(N, H, -1, -1)
        expanded_k = k[:, :, :, None, :].expand(-1, -1, -1, K, -1)
        expanded_v = v[:, :, :, None, :].expand(-1, -1, -1, K, -1)

        aligned_k = torch.gather(expanded_k, dim=2, index=expanded_idx)
        aligned_v = torch.gather(expanded_v, dim=2, index=expanded_idx)

                                                                                  
        attn = (q[:, :, :, None, :] * aligned_k).sum(-1)

        if self.LOGIT_SCALE_PRE_REL_BIAS:
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
            attn = attn * logit_scale

        if self.apply_rel_pos_bias:
            rel_coords, rel_bias = self.rel_pos_bias(aligned_k)
            attn = attn + rel_bias

        if not self.LOGIT_SCALE_PRE_REL_BIAS:
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
            attn = attn * logit_scale

                                                                  
        attn = torch.masked_fill(attn, mask=~expanded_idx_mask, value=float('-inf'))

                    
        attn = F.softmax(attn, dim=-1)

                           
        if query_mask is not None:
            raise NotImplementedError("No support for query_mask")
                           
            query_mask_ = query_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            attn = torch.masked_fill(attn, query_mask_.expand_as(attn), 0.0)

        attn = self.attn_drop(attn)

                                                                                    
        out = (attn.unsqueeze(-1) * aligned_v).sum(-2)
                                                        
        out = einops.rearrange(out, "N H D C_H -> N D (H C_H)")
        if self.append_self:
            out = torch.cat((out, v_self), dim=-1)
        out = self.out_proj(out)
        out = self.out_drop(out)

        return out


class GlobalSphereSelfAttention(nn.Module):
    """True full-sphere content attention with optional query chunking.

    Chunking partitions only the query axis. Every query still attends to all
    vertices, so it is mathematically identical to an unchunked attention
    matrix (up to floating-point execution order).
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_head_coef: int = 1,
            qkv_bias: bool = True,
            attn_drop: float = 0.0,
            out_drop: float = 0.0,
            query_chunk_size: int = 128,
            use_checkpoint: bool = False,
    ):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = (d_model // num_heads) * d_head_coef
        self.scale = self.d_head ** -0.5
        self.query_chunk_size = query_chunk_size
        self.use_checkpoint = use_checkpoint
        projected_dim = self.num_heads * self.d_head
        self.q_proj = nn.Linear(d_model, projected_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, projected_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, projected_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(projected_dim, d_model)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_drop = nn.Dropout(out_drop)
        self.last_query_count = 0
        self.last_key_count = 0

    def _attend_chunk(self, query, key, value):
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(F.softmax(attention, dim=-1))
        return torch.matmul(attention, value)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Global attention expects [N, V, C]")
        batch_frames, vertices, _ = x.shape
        projected_shape = (batch_frames, vertices, self.num_heads, self.d_head)
        query = self.q_proj(x).reshape(projected_shape).permute(0, 2, 1, 3)
        key = self.k_proj(x).reshape(projected_shape).permute(0, 2, 1, 3)
        value = self.v_proj(x).reshape(projected_shape).permute(0, 2, 1, 3)
        chunks = []
        for start in range(0, vertices, self.query_chunk_size):
            query_chunk = query[:, :, start:start + self.query_chunk_size]
            if self.use_checkpoint and self.training:
                output_chunk = checkpoint.checkpoint(
                    self._attend_chunk, query_chunk, key, value
                )
            else:
                output_chunk = self._attend_chunk(query_chunk, key, value)
            chunks.append(output_chunk)
        output = torch.cat(chunks, dim=2).permute(0, 2, 1, 3).reshape(
            batch_frames, vertices, -1
        )
        self.last_query_count = batch_frames * vertices
        self.last_key_count = vertices
        return self.out_drop(self.out_proj(output))


class GlobalContentSpatioTemporalBlock(nn.Module):
    """Rank-4 content-aware temporal attention plus full-sphere attention."""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            temporal_window_radius: Optional[int] = 5,
            d_head_coef: int = 1,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            attn_drop: float = 0.0,
            out_drop: float = 0.0,
            drop_path: float = 0.0,
            query_chunk_size: int = 128,
            use_checkpoint: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.temporal_attention = Diff_TemporalAttention(
            d_model=dim,
            num_heads=num_heads,
            d_head_coef=d_head_coef,
            temporal_window_radius=temporal_window_radius,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            out_drop=out_drop,
            use_frame_diff=False,
        )
        self.spatial_attention = GlobalSphereSelfAttention(
            d_model=dim,
            num_heads=num_heads,
            d_head_coef=d_head_coef,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            out_drop=out_drop,
            query_chunk_size=query_chunk_size,
            use_checkpoint=use_checkpoint,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(out_drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(out_drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: Tensor, time_steps: int) -> Tensor:
        batch_frames, vertices, channels = x.shape
        if batch_frames % time_steps:
            raise ValueError("batch_frames must be divisible by time_steps")
        batch_size = batch_frames // time_steps
        normalized = self.norm1(x)
        content = normalized.reshape(batch_size, time_steps, vertices, channels)
        temporal_delta = self.temporal_attention(content).reshape_as(x)
        content_with_time = normalized + temporal_delta
        global_delta = self.spatial_attention(content_with_time)
        output = x + self.drop_path(temporal_delta + global_delta)
        return output + self.drop_path(self.mlp(self.norm2(output)))


class SparseSphereSelfAttention(nn.Module):
    """Local spherical attention evaluated only at selected query vertices.

    Keys and values are gathered from the exact fixed neighborhood used by
    :class:`SphereSelfAttention`. No dense high-resolution attention output is
    formed before selection.
    """

    LOGIT_SCALE_PRE_REL_BIAS: bool = True

    def __init__(
            self, *,
            rank: int,
            icosphere_ref: IcoSphereRef,
            win_size_coef: int,
            num_heads: int,
            d_model: int,
            d_head_coef: int,
            qkv_bias: bool,
            attn_drop: float = 0.0,
            out_drop: float = 0.0,
            abs_pos_enc: bool = False,
            abs_pos_enc_size: int = 0,
            rel_pos_bias: bool = False,
            rel_pos_bias_size: int = 0,
            rel_pos_init_variance: float = 0.0,
            append_self: bool = False,
    ):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.rank = rank
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = (d_model // num_heads) * d_head_coef
        self.append_self = append_self
        self.apply_abs_pos_enc = abs_pos_enc
        self.apply_rel_pos_bias = rel_pos_bias
        self.rel_pos_bias = RelativePositionBias(
            rank,
            icosphere_ref,
            win_size_coef,
            rel_pos_bias_size=rel_pos_bias_size,
            num_heads=num_heads,
            init_variance=rel_pos_init_variance,
        )
        self.num_keys = self.rel_pos_bias.num_keys
        projected_dim = self.d_head * num_heads
        self.q_proj = nn.Linear(d_model, projected_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, projected_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(
            d_model, self.d_head * (num_heads + int(append_self)), bias=qkv_bias
        )
        if abs_pos_enc:
            self.q_abs_pos_proj = nn.Linear(
                abs_pos_enc_size, projected_dim, bias=False
            )
            self.k_abs_pos_proj = nn.Linear(
                abs_pos_enc_size, projected_dim, bias=False
            )
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )
        self.out_proj = nn.Linear(
            self.d_head * (num_heads + int(append_self)), d_model
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_drop = nn.Dropout(out_drop)
        self.last_query_count = 0
        self.last_attention_pair_count = 0

    def _selected_relative_bias(self, query_vertices, dtype):
        coordinates = self.rel_pos_bias.relative_coords
        normalized = coordinates / (coordinates.abs().max() + 1e-8)
        grid = normalized[query_vertices].to(dtype=dtype).unsqueeze(0)
        bias = F.grid_sample(
            self.rel_pos_bias.bias_grid,
            grid,
            align_corners=True,
        )
        return bias.squeeze(0).permute(1, 0, 2)

    def forward(self, x: Tensor, selected_queries: Tensor, pos=None):
        if x.ndim != 3 or selected_queries.shape != x.shape[:2]:
            raise ValueError("Expected x [N,V,C] and selected_queries [N,V]")
        selected_queries = selected_queries.bool()
        query_pairs = selected_queries.nonzero(as_tuple=False)
        query_count = query_pairs.shape[0]
        self.last_query_count = int(query_count)
        self.last_attention_pair_count = int(query_count * self.num_keys)
        if query_count == 0:
            return x.new_empty((0, self.d_model)), query_pairs

        batch_indices = query_pairs[:, 0]
        vertex_indices = query_pairs[:, 1]
        neighbor_indices = self.rel_pos_bias.idx[0, 0, :, :, 0][vertex_indices]
        neighbor_mask = self.rel_pos_bias.idx_mask[0, 0][vertex_indices]
        query_features = x[batch_indices, vertex_indices]
        neighbor_features = x[batch_indices.unsqueeze(1), neighbor_indices]

        query = self.q_proj(query_features).reshape(
            query_count, self.num_heads, self.d_head
        )
        key = self.k_proj(neighbor_features).reshape(
            query_count, self.num_keys, self.num_heads, self.d_head
        ).permute(0, 2, 1, 3)
        value_all = self.v_proj(neighbor_features).reshape(
            query_count,
            self.num_keys,
            self.num_heads + int(self.append_self),
            self.d_head,
        ).permute(0, 2, 1, 3)
        if self.append_self:
            value, _ = torch.split(value_all, (self.num_heads, 1), dim=1)
            self_value = self.v_proj(query_features).reshape(
                query_count,
                self.num_heads + 1,
                self.d_head,
            )[:, -1]
        else:
            value = value_all

        if self.apply_abs_pos_enc:
            if pos is None or pos.shape[1] != x.shape[1]:
                raise ValueError("Absolute position encoding has an invalid shape")
            query_pos = pos[0, vertex_indices]
            neighbor_pos = pos[0, neighbor_indices]
            query = query + self.q_abs_pos_proj(query_pos).reshape(
                query_count, self.num_heads, self.d_head
            )
            key = key + self.k_abs_pos_proj(neighbor_pos).reshape(
                query_count, self.num_keys, self.num_heads, self.d_head
            ).permute(0, 2, 1, 3)

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = (query.unsqueeze(2) * key).sum(dim=-1)
        logit_scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        logit_scale = logit_scale[:, 0, 0].reshape(1, self.num_heads, 1)
        if self.LOGIT_SCALE_PRE_REL_BIAS:
            attention = attention * logit_scale
        if self.apply_rel_pos_bias:
            attention = attention + self._selected_relative_bias(
                vertex_indices, attention.dtype
            )
        if not self.LOGIT_SCALE_PRE_REL_BIAS:
            attention = attention * logit_scale
        attention = attention.masked_fill(
            ~neighbor_mask.unsqueeze(1), float("-inf")
        )
        attention = self.attn_drop(F.softmax(attention, dim=-1))
        output = (attention.unsqueeze(-1) * value).sum(dim=2).reshape(
            query_count, -1
        )
        if self.append_self:
            output = torch.cat((output, self_value), dim=-1)
        return self.out_drop(self.out_proj(output)), query_pairs


class SparseLocalRefinementBlock(nn.Module):
    """Selected-query spatial attention and FFN for a fine sphere level."""

    def __init__(
            self, *,
            rank: int,
            icosphere_ref: IcoSphereRef,
            dim: int,
            num_heads: int,
            d_head_coef: int,
            win_size_coef: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            attn_drop: float = 0.0,
            out_drop: float = 0.0,
            drop_path: float = 0.0,
            abs_pos_enc: bool = False,
            rel_pos_bias: bool = False,
            rel_pos_bias_size: int = 0,
            rel_pos_init_variance: float = 0.0,
            append_self: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if abs_pos_enc:
            abs_pos_size = 32
            from .position_encoding import GlobalVerticalPositionEnconding
            self.abs_pos_enc = GlobalVerticalPositionEnconding(
                rank=rank,
                icosphere_ref=icosphere_ref,
                mode="phi",
                num_pos_feats=abs_pos_size // 2,
                max_frequency=10000,
                min_frequency=1,
            )
        else:
            abs_pos_size = 0
            self.abs_pos_enc = None
        self.attention = SparseSphereSelfAttention(
            rank=rank,
            icosphere_ref=icosphere_ref,
            win_size_coef=win_size_coef,
            num_heads=num_heads,
            d_model=dim,
            d_head_coef=d_head_coef,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            out_drop=out_drop,
            abs_pos_enc=abs_pos_enc,
            abs_pos_enc_size=abs_pos_size,
            rel_pos_bias=rel_pos_bias,
            rel_pos_bias_size=rel_pos_bias_size,
            rel_pos_init_variance=rel_pos_init_variance,
            append_self=append_self,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(out_drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(out_drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, dense_features: Tensor, selected_queries: Tensor):
        normalized = self.norm1(dense_features)
        pos = self.abs_pos_enc(normalized) if self.abs_pos_enc is not None else None
        attention, query_pairs = self.attention(
            normalized, selected_queries, pos
        )
        if query_pairs.shape[0] == 0:
            return attention, query_pairs
        selected = dense_features[query_pairs[:, 0], query_pairs[:, 1]]
        selected = selected + self.drop_path(attention)
        selected = selected + self.drop_path(self.mlp(self.norm2(selected)))
        return selected, query_pairs
