from .position_encoding import RelativePositionBias
from trimesh_utils import IcoSphereRef

from typing import Optional,List
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
                     
            temporal_window_radius: int = 5,                             
                  
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
            temporal_window_radius: int = 5,
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
