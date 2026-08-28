import random
import warnings
from functools import partial
from typing import Dict, Union, List, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum, Tensor
from trimesh import Trimesh

from .position_encoding import GlobalVerticalPositionEnconding
from .sphere_PSA import SphereSelfAttention,SphereSpatioTemporalAttention
from trimesh_utils import (
    get_icosphere,
    IcoSphereHierarchy,
    IcoSphereRef,
    asSpherical,
)



class MLP(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, out_dim=32, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_layer(),
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, out_dim))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


                                         
                  
class MaxDownsample(nn.Module):
    def __init__(self, in_rank: int, out_rank: int, ref: IcoSphereRef):
        super().__init__()
        assert ref.node_type == "face"
        downscale = in_rank - out_rank
        self.swap_dims = True                                            
        self.pool = nn.MaxPool1d(4 ** downscale, 4 ** downscale)

    def forward(self, x: Tensor):
        if self.swap_dims:
            x = einops.rearrange(x, "n c d -> n d c")
        x = self.pool(x)
        if self.swap_dims:
            x = einops.rearrange(x, "n d c -> n c d")
        return x


class AvgDownsample(nn.Module):
    def __init__(self, in_rank: int, out_rank: int, ref: IcoSphereRef):
        super().__init__()
        assert ref.node_type == "face"
        downscale = in_rank - out_rank
        self.swap_dims = True                                            
        self.pool = nn.AvgPool1d(4 ** downscale, 4 ** downscale)

    def forward(self, x: Tensor):
        if self.swap_dims:
            x = einops.rearrange(x, "n c d -> n d c")
        x = self.pool(x)
        if self.swap_dims:
            x = einops.rearrange(x, "n d c -> n c d")
        return x


class CenterDownsample(nn.Module):
    def __init__(self, in_rank: int, out_rank: int, ref: IcoSphereRef):
        super().__init__()

        self.downscale = in_rank - out_rank

        in_normals = ref.get_normals(in_rank)
        out_normals = ref.get_normals(out_rank)

        if in_rank < 7:
            cosine_similarity = in_normals @ out_normals.T
            center_idx = cosine_similarity.argmax(0).tolist()
        else:
            if True:                                
                warnings.warn("RISKY CenterDownsample")
                                                                                              
                if ref.node_type == "vertex":
                    center_idx = list(range(out_normals.shape[0]))
                elif ref.node_type == "face":
                    center_idx = list(range(3, in_normals.shape[0], 4))
                else:
                    raise ValueError(ref.node_type)
            else:
                print(f"IN SHAPE: {in_normals.shape[0]} - OUT SHAPE: {out_normals.shape[0]}")
                center_idx = []
                K = 5000
                for i in range(0, out_normals.shape[0], K):
                    out_normals_i = out_normals[i:i+K]
                                                
                    cosine_similarity = in_normals @ out_normals_i.T
                    center_idx_i = cosine_similarity.argmax(0).tolist()
                    center_idx.extend(center_idx_i)

        assert len(center_idx) == out_normals.shape[0],  f"{len(center_idx)} == {out_normals.shape[0]}"
        self.center_idx = center_idx

    def forward(self, x: Tensor):
        return x[:, self.center_idx, :]


                                         
                
class Upsample(nn.Module):
    def __init__(self, in_rank: int, out_rank: int, ref: IcoSphereRef):
        super().__init__()
        self.upscale = out_rank - in_rank
        self.unpool = lambda x: einops.repeat(x, "n d c -> n (d k) c", k=4**self.upscale)

    def forward(self, x: Tensor):
        x = self.unpool(x)
        return x


class NearestUpsample(nn.Module):
    def __init__(self, in_rank: int, out_rank: int, ref: IcoSphereRef):
        super().__init__()

        self.upscale = out_rank - in_rank

        in_normals = ref.get_normals(in_rank)
        out_normals = ref.get_normals(out_rank)

        if out_rank < 7:
            cosine_similarity = in_normals @ out_normals.T
            center_idx = cosine_similarity.argmax(0).tolist()
        else:
            warnings.warn("BAD NearestUpsample")
            center_idx = [random.choice(range(in_normals.shape[0])) for _ in range(out_normals.shape[0])]

        assert len(center_idx) == out_normals.shape[0]
        self.center_idx = center_idx

    def forward(self, x: Tensor):
        return x[:, self.center_idx]


class InterpolateUpsample(nn.Module):
    def __init__(self, in_rank: int, out_rank: int, ref: IcoSphereRef):
        super().__init__()
        assert ref.node_type == "vertex"

        self.upscale = out_rank - in_rank
        assert self.upscale == 1

        in_ico = ref.get_icosphere(in_rank, refine=True)
        out_ico = ref.get_icosphere(out_rank, refine=True)

        in_size = in_ico.vertices.shape[0]
        out_size = out_ico.vertices.shape[0]
        self.left_idx = list(range(out_size))
        self.right_idx = list(range(out_size))

        for i in range(in_size, out_size):
            indices = [_ for _ in out_ico.vertex_neighbors[i] if _ < in_size]
            self.left_idx[i], self.right_idx[i] = indices

    def forward(self, x: Tensor):
        return (x[:, self.left_idx] + x[:, self.right_idx]) / 2


                                         
                 
class InputProj(nn.Module):
    def __init__(self, in_channel, out_channel, *, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channel, out_channel),
        )
        if act_layer is not None:
            self.proj.add_module(str(len(self.proj)), act_layer())
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class OutputProj(nn.Module):
    def __init__(self, in_channel, out_channel, *, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channel, out_channel),
        )
        if act_layer is not None:
            self.proj.add_module(str(len(self.proj)), act_layer())
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


                                         
class SphereUFormerBlock(nn.Module):
    def __init__(
            self, *,
            rank: int,
            icosphere_ref: IcoSphereRef,
            dim, num_heads, d_head_coef, win_size_coef,
            temporal_window_radius: int = 5,
            mlp_ratio=4.,
            qkv_bias=True, qk_scale=None,
            attn_drop=0., attn_out_drop=0., mlp_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            abs_pos_enc: bool = False,
            abs_pos_enc_size: int = 0,
            rel_pos_bias: bool = False,
            rel_pos_bias_size: int = 0,
            rel_pos_init_variance: float = 0.0,
            debug_skip_attn: bool = False,
            append_self: bool = False,
    ):
        super().__init__()
        self.debug_skip_attn = debug_skip_attn

        self.rank = rank
        self.icosphere_ref = icosphere_ref

        self.dim = dim
        self.num_heads = num_heads
        self.win_size_coef = win_size_coef

        self.mlp_ratio = mlp_ratio

        self.attn = SphereSpatioTemporalAttention(
            rank=rank,
            icosphere_ref=icosphere_ref,
            win_size_coef=win_size_coef,
            temporal_window_radius=temporal_window_radius,
            num_heads=num_heads,
            d_head_coef=d_head_coef,
            d_model=dim,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            out_drop=attn_out_drop,
            abs_pos_enc=abs_pos_enc,
            abs_pos_enc_size=abs_pos_enc_size,
            rel_pos_bias=rel_pos_bias,
            rel_pos_bias_size=rel_pos_bias_size,
            rel_pos_init_variance=rel_pos_init_variance,
            append_self=append_self,
        )

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.drop_path = DropPath(drop_path)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, dim, act_layer=act_layer, drop=mlp_drop)

    def extra_repr(self) -> str:
        return f"!!rank={self.rank}, dim={self.dim}, num_heads={self.num_heads}, win_size={self.win_size_coef}, mlp_ratio={self.mlp_ratio}!!"

    def forward(self, x: Tensor, pos: Optional[Tensor], time_steps: int):
        N, D, C = x.shape

        bus = x

              
        if not self.debug_skip_attn:
            x_ = self.norm1(bus)
            x_ = self.attn(x_, time_steps, pos)
            bus = bus + self.drop_path(x_)

             
        x_ = self.norm2(bus)
        x_ = self.mlp(x_)
        bus = bus + self.drop_path(x_)

        y = bus

        return y


                                                   
class SphereUFormerModule(nn.Module):
    def __init__(
            self, *,
            rank: int,
            icosphere_ref: IcoSphereRef,
            dim,
            depth,
            num_heads,
            d_head_coef,
            win_size_coef,
            temporal_window_radius: int = 5,
            mlp_ratio=4.,
            qkv_bias=True, qk_scale=None,
            attn_drop: float = 0., attn_out_drop: float = 0.,
            mlp_drop: float = 0., drop_path: Union[List[float], float] = 0.1,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            abs_pos_enc: bool = False,
            rel_pos_bias: bool = False,
            rel_pos_bias_size: int = 0,
            rel_pos_init_variance: float = 0.0,
            debug_skip_attn: bool = False,
            append_self: bool = False,
    ):

        super().__init__()
        self.rank = rank
        self.icosphere_ref = icosphere_ref

        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if abs_pos_enc:
            abs_pos_enc_size = 32
            self.abs_pos_enc = GlobalVerticalPositionEnconding(
                rank=rank,
                icosphere_ref=icosphere_ref,
                mode="phi",
                num_pos_feats=abs_pos_enc_size//2,
                max_frequency=10000,
                min_frequency=1,
            )
        else:
            self.abs_pos_enc = None
            abs_pos_enc_size = 0


                      
        self.blocks = nn.ModuleList([
            SphereUFormerBlock(rank=rank, icosphere_ref=icosphere_ref,
                                  dim=dim, num_heads=num_heads, d_head_coef=d_head_coef,
                                  win_size_coef=win_size_coef,
                                  temporal_window_radius=temporal_window_radius,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, attn_out_drop=attn_out_drop, mlp_drop=mlp_drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  act_layer=act_layer, norm_layer=norm_layer,
                                  abs_pos_enc=abs_pos_enc,
                                  abs_pos_enc_size=abs_pos_enc_size,
                                  rel_pos_bias=rel_pos_bias,
                                  rel_pos_bias_size=rel_pos_bias_size,
                                  rel_pos_init_variance=rel_pos_init_variance,
                                  debug_skip_attn=debug_skip_attn,
                                  append_self=append_self,
                                  )
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"!!rank={self.rank}, dim={self.dim}, depth={self.depth}!!"

    def forward(self, x, time_steps: int):

        if self.abs_pos_enc is not None:
            pos = self.abs_pos_enc(x)
        else:
            pos = None

        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                                                                    
                x = checkpoint.checkpoint(partial(blk, time_steps=time_steps), x, pos)
            else:
                x = blk(x, pos, time_steps)
        return x






                                    
class SphereUFormer(nn.Module):
    def __init__(
            self,
            img_rank: int,
            node_type: str,
            in_channels=3,
            out_channels=1,
            embed_dim=32,
            num_scales=4,
            in_scale_factor: int = 2,
            enc_depths=(2, 2, 2, 2),
            bottleneck_depth=2,
            dec_depths=(2, 2, 2, 2),
            d_head_coef: int = 1,
            enc_num_heads=(2, 4, 8, 16),
            bottleneck_num_heads=None,
            dec_num_heads=(16, 16, 8, 4),
            win_size_coef: int = 1,
            temporal_window_radius: int = 5,
            mlp_ratio=4., qkv_bias=True, qk_scale=None,
            attn_drop_rate=0., attn_out_drop_rate=0., drop_rate=0., drop_path_rate=0., pos_drop_rate=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            downsample: str = "center",
            upsample: str = "nearest",
            abs_pos_enc_in: bool = True,
            abs_pos_enc: bool = True,
            rel_pos_bias: bool = True,
            rel_pos_bias_size: int = 7,
            rel_pos_init_variance: float = 0.0,
            debug_skip_attn: bool = False,
            append_self: bool = False,
            Spherical_prior: bool = False,
    ):

        super().__init__()

        enc_num_heads = enc_num_heads or (1, 2, 4, 8, 16, 16)
        dec_num_heads = dec_num_heads or (16, 16, 16, 8, 4, 2)

        if isinstance(enc_depths, int):
            enc_depths = [enc_depths] * num_scales
        if isinstance(dec_depths, int):
            dec_depths = [dec_depths] * num_scales

        enc_depths = enc_depths[:num_scales]
        enc_num_heads = enc_num_heads[:num_scales]
        dec_depths = dec_depths[len(dec_depths)-num_scales:]
        dec_num_heads = dec_num_heads[len(dec_depths)-num_scales:]

        self.img_rank = img_rank
        self.proj_rank = proj_rank = img_rank - int(math.log2(in_scale_factor))
        self.embed_dim = embed_dim
        self.num_enc_layers = len(enc_depths)
        self.num_dec_layers = len(dec_depths)

        self.mlp_ratio = mlp_ratio
        self.win_size_coef = win_size_coef

                                           
        print("Generating sphere refs")
        self.icosphere_ref = IcoSphereRef(node_type=node_type)

        self.pos_drop = nn.Dropout(p=pos_drop_rate)

                          
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        bottleneck_dpr = [drop_path_rate] * bottleneck_depth
        dec_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))][::-1]


        self.apply_abs_pos_enc_in = abs_pos_enc_in
        if self.apply_abs_pos_enc_in:
            abs_pos_enc_size = 32
            self.abs_pos_enc_in = nn.Sequential(
                GlobalVerticalPositionEnconding(
                    rank=proj_rank,
                    icosphere_ref=self.icosphere_ref,
                    mode="phi",
                    num_pos_feats=abs_pos_enc_size//2,
                    max_frequency=10000,
                    min_frequency=1,
                ),
                nn.Linear(abs_pos_enc_size, embed_dim, bias=False),
            )

                                                                                  

                                              
        self.input_proj = InputProj(in_channel=in_channels, out_channel=embed_dim, act_layer=nn.GELU)
        self.output_proj = OutputProj(in_channel=embed_dim, out_channel=out_channels)

        if in_scale_factor > 1:
            self.input_proj = nn.Sequential(
                CenterDownsample(img_rank, proj_rank, ref=self.icosphere_ref),
                self.input_proj,
            )
            self.output_proj = nn.Sequential(
                InterpolateUpsample(proj_rank, img_rank, ref=self.icosphere_ref),
                self.output_proj,
            )

        downsample_layer = {
            "max": MaxDownsample,
            "avg": AvgDownsample,
            "center": CenterDownsample,
        }[downsample]

        upsample_layer = {
            "nearest": NearestUpsample,
            "interpolate": InterpolateUpsample,
        }[upsample]

                 
        print("Building encoder")
        self.enc_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        for i in range(self.num_enc_layers):
            self.enc_blocks.append(
                nn.Sequential(
                    SphereUFormerModule(
                        rank=proj_rank-i,
                        icosphere_ref=self.icosphere_ref,
                        dim=embed_dim * (2 ** i),
                        depth=enc_depths[i],
                        num_heads=enc_num_heads[i],
                        d_head_coef=d_head_coef,
                        win_size_coef=win_size_coef,
                        temporal_window_radius=temporal_window_radius,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop_rate, attn_out_drop=attn_out_drop_rate, mlp_drop=drop_rate,
                        drop_path=enc_dpr[int(sum(enc_depths[:i])):int(sum(enc_depths[:(i+1)]))],
                        act_layer=act_layer, norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        abs_pos_enc=abs_pos_enc,
                        rel_pos_bias=rel_pos_bias,
                        rel_pos_bias_size=rel_pos_bias_size,
                        rel_pos_init_variance=rel_pos_init_variance,
                        debug_skip_attn=debug_skip_attn,
                        append_self=append_self,
                    ),
                )
            )

            self.downsample_blocks.append(
                nn.Sequential(
                    downsample_layer(proj_rank-i, proj_rank-i-1, self.icosphere_ref),
                    norm_layer(embed_dim * (2 ** i)),
                    nn.Linear(in_features=embed_dim * (2 ** i), out_features=embed_dim * (2 ** i) * 2),
                                
                                                           
                )
            )

                    
        I = self.num_enc_layers
        self.bottleneck = SphereUFormerModule(
            rank=proj_rank-I,
            icosphere_ref=self.icosphere_ref,
            dim=embed_dim * (2 ** I),
            depth=bottleneck_depth,
            num_heads=bottleneck_num_heads or dec_num_heads[0],
            d_head_coef=d_head_coef,
            win_size_coef=win_size_coef,
            temporal_window_radius=temporal_window_radius,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            mlp_drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=bottleneck_dpr,
            act_layer=act_layer, norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            abs_pos_enc=abs_pos_enc,
            rel_pos_bias=rel_pos_bias,
            rel_pos_bias_size=rel_pos_bias_size,
            rel_pos_init_variance=rel_pos_init_variance,
            debug_skip_attn=debug_skip_attn,
            append_self=append_self,
        )

                 
        print("Building decoder")
        self.dec_blocks = nn.ModuleList()
        self.dec_norm_layers1 = nn.ModuleList()
        self.dec_norm_layers2 = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i in range(self.num_dec_layers):
            reverse_i = I-i-1

            self.upsample_blocks.append(
                nn.Sequential(
                    norm_layer(embed_dim * (2 ** reverse_i) * 2),
                    nn.Linear(in_features=embed_dim * (2 ** reverse_i) * 2, out_features=embed_dim * (2 ** reverse_i)),
                                
                                                               
                    upsample_layer(proj_rank-reverse_i-1, proj_rank-reverse_i, ref=self.icosphere_ref),
                )
            )

            self.dec_norm_layers1.append(
                norm_layer(embed_dim * (2 ** reverse_i)),
            )
            self.dec_norm_layers2.append(
                norm_layer(embed_dim * (2 ** reverse_i)),
            )
            self.dec_blocks.append(
                nn.Sequential(
                    nn.Linear(in_features=embed_dim * (2 ** reverse_i) * 2, out_features=embed_dim * (2 ** reverse_i)),
                                
                                                               
                    SphereUFormerModule(
                        rank=proj_rank-reverse_i,
                        icosphere_ref=self.icosphere_ref,
                        dim=embed_dim * (2 ** reverse_i),
                        depth=dec_depths[i],
                        num_heads=dec_num_heads[i],
                        d_head_coef=d_head_coef,
                        win_size_coef=win_size_coef,
                        temporal_window_radius=temporal_window_radius,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        mlp_drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dec_dpr[int(sum(dec_depths[:i])):int(sum(dec_depths[:(i + 1)]))],
                        act_layer=act_layer, norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        abs_pos_enc=abs_pos_enc,
                        rel_pos_bias=rel_pos_bias,
                        rel_pos_bias_size=rel_pos_bias_size,
                        rel_pos_init_variance=rel_pos_init_variance,
                        debug_skip_attn=debug_skip_attn,
                        append_self=append_self,
                    )
                )
            )


        self.final_sigmoid = nn.Sigmoid()
        print("Initializing weights")
        self.apply(self._init_weights)

        self.out_channels = out_channels

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"!!img_rank={self.img_rank}, embed_dim={self.embed_dim}, token_mlp={self.mlp}, win_size={self.win_size_coef}!!"

    def forward(self, x):
                          
                           
        B, T, L, C = x.shape
                                                
        x = x.reshape(B * T, L, C)

        y = self.input_proj(x)

        if self.apply_abs_pos_enc_in:
            pos = self.abs_pos_enc_in(y)
            y = y + pos

                                                                                       

        y = self.pos_drop(y)

                 
        enc_outs = []
        for i in range(len(self.enc_blocks)):
            conv_i = self.enc_blocks[i][0](y, time_steps=T)
            enc_outs.append(conv_i)
            y = self.downsample_blocks[i](conv_i)

                    
        y = self.bottleneck(y, time_steps=T)

                 
        for i in range(len(self.dec_blocks)):
            y = self.upsample_blocks[i](y)
            if True:                   
                y = torch.cat([self.dec_norm_layers1[i](y), self.dec_norm_layers2[i](enc_outs[self.num_dec_layers-1-i])], dim=-1)
            else:
                y = self.dec_norm_layers1[i](y)
                y = torch.cat([y, y], dim=-1)
            y = self.dec_blocks[i][0](y)          
            y = self.dec_blocks[i][1](y, time_steps=T)                       


                           
        y = self.output_proj(y)

                   
        if self.out_channels == 1:
            y = y.squeeze(-1)                           
            y = y.reshape(B, T, L)             
        else:
            y = y.reshape(B, T, L, self.out_channels)                           

        sal_map = self.final_sigmoid(y)
        return sal_map


class HierarchicalRegionPool(nn.Module):
    """Pool fine vertex features through exact face descendants."""

    def __init__(
            self,
            dim: int,
            pool_type: str = "mean_max",
    ):
        super().__init__()
        if pool_type not in {"center", "mean", "mean_max"}:
            raise ValueError(f"Unsupported coarse_pool_type: {pool_type}")
        self.pool_type = pool_type
        if pool_type != "center":
            input_dim = dim if pool_type == "mean" else 2 * dim
            self.projection = nn.Linear(input_dim, dim)

    def forward(
            self,
            fine_vertex_features: Tensor,
            hierarchy: IcoSphereHierarchy,
    ) -> Tensor:
        if self.pool_type == "center":
            return hierarchy.center_downsample_vertices(fine_vertex_features)
        fine_face_mean, _ = hierarchy.vertex_feature_stats(
            fine_vertex_features, level="fine"
        )
        region_mean, region_max = hierarchy.fine_face_feature_stats(
            fine_face_mean
        )
        if self.pool_type == "mean":
            coarse_face_features = self.projection(region_mean)
        else:
            coarse_face_features = self.projection(
                torch.cat((region_mean, region_max), dim=-1)
            )
        return hierarchy.face_features_to_vertices(
            coarse_face_features, level="coarse"
        )


class VertexToFaceAggregation(nn.Module):
    """Aggregate three coarse vertex features into one triangular region."""

    def __init__(self, dim: int):
        super().__init__()
        self.projection = nn.Linear(2 * dim, dim)

    def forward(
            self,
            coarse_vertex_features: Tensor,
            hierarchy: IcoSphereHierarchy,
    ) -> Tensor:
        mean, maximum = hierarchy.vertex_feature_stats(
            coarse_vertex_features, level="coarse"
        )
        return self.projection(torch.cat((mean, maximum), dim=-1))


class SphericalUncertaintyHead(nn.Module):
    """Predict positive heteroscedastic Laplace scale without using labels."""

    def __init__(self, dim: int, epsilon: float = 1e-6):
        super().__init__()
        hidden_dim = max(1, dim // 2)
        self.epsilon = epsilon
        self.scale_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, face_features: Tensor) -> Tensor:
        raw_scale = self.scale_mlp(face_features).squeeze(-1)
        return F.softplus(raw_scale) + self.epsilon


class AdaptiveRefinementHead(nn.Module):
    """Predict face refinement from content, uncertainty, and motion."""

    def __init__(
            self,
            dim: int,
            use_uncertainty: bool = True,
            use_motion: bool = True,
    ):
        super().__init__()
        hidden_dim = max(1, dim // 2)
        self.use_uncertainty = use_uncertainty
        self.use_motion = use_motion
        self.refinement_mlp = nn.Sequential(
            nn.Linear(dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
            self,
            face_features: Tensor,
            uncertainty_scale: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return logits and scores for [B, T, Fcoarse, C] features."""
        if self.use_motion and face_features.shape[1] > 1:
            delta = (face_features[:, 1:] - face_features[:, :-1]).abs().mean(
                dim=-1, keepdim=True
            )
            motion = torch.cat((torch.zeros_like(delta[:, :1]), delta), dim=1)
        else:
            motion = torch.zeros_like(face_features[..., :1])
        if self.use_uncertainty:
            # Calibration is governed by L_uncertainty, not by gate gradients.
            uncertainty = uncertainty_scale.detach().unsqueeze(-1)
        else:
            uncertainty = torch.zeros_like(face_features[..., :1])
        inputs = torch.cat((face_features, uncertainty, motion), dim=-1)
        logits = self.refinement_mlp(inputs).squeeze(-1)
        return logits, torch.sigmoid(logits)


class AdaptiveRegionSelector(nn.Module):
    """Soft selection interface reserved for a future top-k implementation."""

    def __init__(self, temperature: float = 1.0, mode: str = "soft"):
        super().__init__()
        if temperature <= 0:
            raise ValueError("adaptive_temperature must be positive")
        if mode != "soft":
            raise ValueError("The first prototype only supports mode='soft'")
        self.temperature = temperature
        self.mode = mode

    def forward(self, logits: Tensor, enabled: bool = True) -> Tensor:
        if not enabled:
            return torch.ones_like(logits)
        return torch.sigmoid(logits / self.temperature)


class AdaptiveSphereUFormer(nn.Module):
    """Dense vertex backbone with face-guided adaptive refinement.

    The fine encoder still processes every fine vertex. The exact face hierarchy
    gives refinement regions a geometric meaning, but does not reduce FLOPs.
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
            temporal_window_radius: int = 5,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_scale=None,
            attn_drop_rate: float = 0.,
            attn_out_drop_rate: float = 0.,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            pos_drop_rate: float = 0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            use_checkpoint: bool = False,
            abs_pos_enc_in: bool = True,
            abs_pos_enc: bool = True,
            rel_pos_bias: bool = True,
            rel_pos_bias_size: int = 7,
            rel_pos_init_variance: float = 0.,
            debug_skip_attn: bool = False,
            append_self: bool = False,
            coarse_rank_offset: int = 2,
            adaptive_coarse_depth: int = 2,
            adaptive_fine_depth: int = 1,
            adaptive_temperature: float = 1.0,
            adaptive_region_type: str = "face",
            coarse_pool_type: str = "mean_max",
            face_to_vertex_reduce: str = "mean",
            use_adaptive_refinement: bool = True,
            use_uncertainty_refinement: bool = True,
            use_motion_refinement: bool = True,
            return_aux: bool = False,
            debug_adaptive: bool = False,
    ):
        super().__init__()
        if in_scale_factor < 1 or in_scale_factor & (in_scale_factor - 1):
            raise ValueError("in_scale_factor must be a positive power of two")
        if coarse_rank_offset < 1:
            raise ValueError("coarse_rank_offset must be at least 1")
        if adaptive_coarse_depth < 1 or adaptive_fine_depth < 1:
            raise ValueError("Adaptive encoder depths must be positive")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if node_type != "vertex":
            raise ValueError(
                "AdaptiveSphereUFormer keeps a vertex backbone; use --mode vertex"
            )
        if adaptive_region_type != "face":
            raise ValueError("The audited adaptive hierarchy only supports face regions")
        if face_to_vertex_reduce != "mean":
            raise ValueError("Only face_to_vertex_reduce='mean' is implemented")

        self.img_rank = img_rank
        self.proj_rank = img_rank - int(math.log2(in_scale_factor))
        self.fine_rank = self.proj_rank
        self.coarse_rank = self.fine_rank - coarse_rank_offset
        if self.coarse_rank < 0:
            raise ValueError(
                f"coarse_rank={self.coarse_rank} is invalid; reduce "
                "coarse_rank_offset or in_scale_factor"
            )

        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.adaptive_region_type = adaptive_region_type
        self.coarse_pool_type = coarse_pool_type
        self.face_to_vertex_reduce = face_to_vertex_reduce
        self.use_adaptive_refinement = use_adaptive_refinement
        self.return_aux = return_aux
        self.debug_adaptive = debug_adaptive
        self.icosphere_ref = IcoSphereRef(node_type=node_type)

        # Topology is derived and validated once; all tensors are registered buffers.
        self.coarse_fine_hierarchy = IcoSphereHierarchy(
            self.coarse_rank, self.fine_rank, self.icosphere_ref
        )
        self.fine_img_hierarchy = IcoSphereHierarchy(
            self.fine_rank, self.img_rank, self.icosphere_ref
        )
        self.coarse_to_fine_upsample = nn.Sequential(*[
            InterpolateUpsample(rank, rank + 1, self.icosphere_ref)
            for rank in range(self.coarse_rank, self.fine_rank)
        ])
        self.output_upsample = nn.Sequential(*[
            InterpolateUpsample(rank, rank + 1, self.icosphere_ref)
            for rank in range(self.fine_rank, self.img_rank)
        ])

        self.fine_input_proj = InputProj(
            in_channels, embed_dim, act_layer=act_layer
        )
        self.coarse_region_pool = HierarchicalRegionPool(
            embed_dim,
            pool_type=coarse_pool_type,
        )
        self.apply_abs_pos_enc_in = abs_pos_enc_in
        if abs_pos_enc_in:
            self.fine_abs_pos_in = nn.Sequential(
                GlobalVerticalPositionEnconding(
                    rank=self.fine_rank,
                    icosphere_ref=self.icosphere_ref,
                    mode="phi",
                    num_pos_feats=16,
                    max_frequency=10000,
                    min_frequency=1,
                ),
                nn.Linear(32, embed_dim, bias=False),
            )
            self.coarse_abs_pos_in = nn.Sequential(
                GlobalVerticalPositionEnconding(
                    rank=self.coarse_rank,
                    icosphere_ref=self.icosphere_ref,
                    mode="phi",
                    num_pos_feats=16,
                    max_frequency=10000,
                    min_frequency=1,
                ),
                nn.Linear(32, embed_dim, bias=False),
            )
        self.pos_drop = nn.Dropout(pos_drop_rate)

        common_module_args = dict(
            icosphere_ref=self.icosphere_ref,
            dim=embed_dim,
            num_heads=num_heads,
            d_head_coef=d_head_coef,
            win_size_coef=win_size_coef,
            temporal_window_radius=temporal_window_radius,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
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
            debug_skip_attn=debug_skip_attn,
            append_self=append_self,
        )
        self.coarse_encoder = SphereUFormerModule(
            rank=self.coarse_rank,
            depth=adaptive_coarse_depth,
            **common_module_args,
        )
        self.fine_encoder = SphereUFormerModule(
            rank=self.fine_rank,
            depth=adaptive_fine_depth,
            **common_module_args,
        )

        self.coarse_vertex_to_face = VertexToFaceAggregation(embed_dim)
        self.coarse_saliency_head = nn.Linear(embed_dim, out_channels)
        self.uncertainty_head = SphericalUncertaintyHead(embed_dim)
        self.refinement_head = AdaptiveRefinementHead(
            embed_dim,
            use_uncertainty=use_uncertainty_refinement,
            use_motion=use_motion_refinement,
        )
        self.region_selector = AdaptiveRegionSelector(adaptive_temperature)
        self.fine_projection = nn.Linear(embed_dim, embed_dim)
        self.fusion_norm = norm_layer(embed_dim)
        self.output_proj = OutputProj(embed_dim, out_channels)
        self.final_sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def upsample_coarse_values_to_img(self, values: Tensor) -> Tensor:
        """Upsample coarse face values to image-rank vertices for visualization."""
        fine_faces = self.coarse_fine_hierarchy.propagate_coarse_face_values(values)
        fine_vertices = self.coarse_fine_hierarchy.fine_face_values_to_vertices(
            fine_faces
        )
        shape = fine_vertices.shape
        img = self.output_upsample(fine_vertices.reshape(-1, shape[-1], 1))
        return img.squeeze(-1).reshape(*shape[:-1], img.shape[1])

    def aggregate_img_values_to_coarse_faces(self, values: Tensor) -> Tensor:
        """Area-weight image-rank vertex targets into exact coarse faces."""
        img_faces = self.fine_img_hierarchy.vertex_values_to_faces(
            values, level="fine"
        )
        fine_faces = self.fine_img_hierarchy.aggregate_fine_face_values(
            img_faces, area_weighted=True
        )
        return self.coarse_fine_hierarchy.aggregate_fine_face_values(
            fine_faces, area_weighted=True
        )

    def aggregate_img_values_to_coarse(self, values: Tensor) -> Tensor:
        """Backward-compatible alias returning coarse face targets."""
        return self.aggregate_img_values_to_coarse_faces(values)

    def forward(self, x: Tensor, return_aux: Optional[bool] = None):
        """Predict saliency from input [B, T, Limg, Cin]."""
        if x.ndim != 4:
            raise ValueError(f"Expected [B, T, L, C] input, got {tuple(x.shape)}")
        B, T, L_img, C = x.shape
        if L_img != self.fine_img_hierarchy.fine_vertex_count:
            raise ValueError(
                f"img_rank={self.img_rank} expects "
                f"{self.fine_img_hierarchy.fine_vertex_count} nodes, got {L_img}"
            )
        flat_img = x.reshape(B * T, L_img, C)

        # Fine vertices are retained subdivision vertices. Coarse content uses
        # mean/max pooling over exact descendant triangular regions.
        fine_rgb = self.fine_img_hierarchy.center_downsample_vertices(flat_img)
        fine_content = self.fine_input_proj(fine_rgb)
        coarse_content = self.coarse_region_pool(
            fine_content, self.coarse_fine_hierarchy
        )
        fine_features = fine_content
        coarse_features = coarse_content
        if self.apply_abs_pos_enc_in:
            fine_features = fine_features + self.fine_abs_pos_in(fine_features)
            coarse_features = coarse_features + self.coarse_abs_pos_in(coarse_features)
        fine_features = self.pos_drop(fine_features)
        coarse_features = self.pos_drop(coarse_features)

        coarse_features = self.coarse_encoder(coarse_features, time_steps=T)
        coarse_face_features = self.coarse_vertex_to_face(
            coarse_features, self.coarse_fine_hierarchy
        )
        coarse_face_bt = coarse_face_features.reshape(
            B,
            T,
            self.coarse_fine_hierarchy.coarse_face_count,
            self.embed_dim,
        )
        coarse_logits = self.coarse_saliency_head(coarse_face_features)
        coarse_saliency = self.final_sigmoid(coarse_logits)
        uncertainty_scale = self.uncertainty_head(coarse_face_bt)

        refine_logits, refine_score = self.refinement_head(
            coarse_face_bt, uncertainty_scale
        )
        coarse_face_gate = self.region_selector(
            refine_logits, enabled=self.use_adaptive_refinement
        )
        fine_refine_score = self.coarse_fine_hierarchy.propagate_coarse_face_values(
            refine_score
        )
        fine_face_gate = self.coarse_fine_hierarchy.propagate_coarse_face_values(
            coarse_face_gate
        )
        fine_vertex_gate = self.coarse_fine_hierarchy.fine_face_values_to_vertices(
            fine_face_gate
        )
        area_refine_ratio = (
            self.coarse_fine_hierarchy.area_weighted_fine_face_ratio(
                fine_face_gate
            )
        )

        fine_features = self.fine_encoder(fine_features, time_steps=T)
        coarse_up = self.coarse_to_fine_upsample(coarse_features)
        fine_residual = self.fine_projection(fine_features)
        fused = self.fusion_norm(
            coarse_up + fine_vertex_gate.reshape(B * T, -1, 1) * fine_residual
        )
        img_features = self.output_upsample(fused)
        final_logits = self.output_proj(img_features)

        if self.out_channels == 1:
            saliency = self.final_sigmoid(
                final_logits.squeeze(-1).reshape(B, T, L_img)
            )
            coarse_saliency = coarse_saliency.squeeze(-1).reshape(
                B, T, self.coarse_fine_hierarchy.coarse_face_count
            )
        else:
            saliency = self.final_sigmoid(
                final_logits.reshape(B, T, L_img, self.out_channels)
            )
            coarse_saliency = coarse_saliency.reshape(
                B,
                T,
                self.coarse_fine_hierarchy.coarse_face_count,
                self.out_channels,
            )

        if self.debug_adaptive:
            print(
                "AdaptiveSphereUFormer:",
                f"coarse_rank={self.coarse_rank}",
                f"fine_rank={self.fine_rank}",
                f"coarse_vertices={self.coarse_fine_hierarchy.coarse_vertex_count}",
                f"fine_vertices={self.coarse_fine_hierarchy.fine_vertex_count}",
                f"coarse_faces={self.coarse_fine_hierarchy.coarse_face_count}",
                f"fine_faces={self.coarse_fine_hierarchy.fine_face_count}",
                f"uncertainty_mean={uncertainty_scale.mean().item():.4f}",
                f"uncertainty_max={uncertainty_scale.max().item():.4f}",
                f"uncertainty_min={uncertainty_scale.min().item():.4f}",
                f"refine_mean={refine_score.mean().item():.4f}",
                f"refine_max={refine_score.max().item():.4f}",
                f"refine_min={refine_score.min().item():.4f}",
                f"face_gate_mean={fine_face_gate.mean().item():.4f}",
                f"vertex_gate_mean={fine_vertex_gate.mean().item():.4f}",
                f"area_refine_ratio={area_refine_ratio.mean().item():.4f}",
            )

        return_aux = self.return_aux if return_aux is None else return_aux
        if not return_aux:
            return saliency
        return {
            "saliency": saliency,
            "coarse_saliency": coarse_saliency,
            "uncertainty": uncertainty_scale,
            "refine_score": refine_score,
            "fine_refine_score": fine_refine_score,
            "fine_face_gate": fine_face_gate,
            "fine_vertex_gate": fine_vertex_gate,
            "area_refine_ratio": area_refine_ratio,
            "gate_mean": area_refine_ratio.mean(),
            "gate_max": fine_face_gate.max(),
            "gate_min": fine_face_gate.min(),
        }


class AdaptiveSphereUFormerV2(nn.Module):
    """Dense rank-(r-2)->(r-1)->r uncertainty-guided hierarchy.

    Every refinement candidate reads raw RGB sampled directly from the original
    rank-r input. Upsampled lower-rank features provide context only. The dense
    rank-(r-1) and rank-r encoders remain active regardless of gate values, so
    this class does not claim sparse FLOP reduction.
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
            temporal_window_radius: int = 5,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_scale=None,
            attn_drop_rate: float = 0.,
            attn_out_drop_rate: float = 0.,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            pos_drop_rate: float = 0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            use_checkpoint: bool = False,
            abs_pos_enc_in: bool = True,
            abs_pos_enc: bool = True,
            rel_pos_bias: bool = True,
            rel_pos_bias_size: int = 7,
            rel_pos_init_variance: float = 0.,
            debug_skip_attn: bool = False,
            append_self: bool = False,
            adaptive_coarse_depth: int = 2,
            adaptive_middle_depth: int = 1,
            adaptive_fine_depth: int = 1,
            adaptive_temperature: float = 1.0,
            adaptive_region_type: str = "face",
            coarse_pool_type: str = "mean_max",
            face_to_vertex_reduce: str = "mean",
            use_adaptive_refinement: bool = True,
            use_uncertainty_refinement: bool = True,
            use_motion_refinement: bool = True,
            return_aux: bool = False,
            debug_adaptive: bool = False,
    ):
        super().__init__()
        del in_scale_factor  # V2 always consumes raw image-rank features.
        if img_rank < 2:
            raise ValueError("UAHS-V2 requires img_rank >= 2")
        if min(
                adaptive_coarse_depth,
                adaptive_middle_depth,
                adaptive_fine_depth,
        ) < 1:
            raise ValueError("All UAHS-V2 encoder depths must be positive")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if node_type != "vertex":
            raise ValueError("UAHS-V2 keeps a vertex backbone; use --mode vertex")
        if adaptive_region_type != "face":
            raise ValueError("UAHS-V2 only supports face-guided regions")
        if face_to_vertex_reduce != "mean":
            raise ValueError("Only face_to_vertex_reduce='mean' is implemented")

        self.img_rank = img_rank
        self.fine_rank = img_rank
        self.middle_rank = img_rank - 1
        self.coarse_rank = img_rank - 2
        self.proj_rank = self.middle_rank
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.use_adaptive_refinement = use_adaptive_refinement
        self.return_aux = return_aux
        self.debug_adaptive = debug_adaptive
        self.icosphere_ref = IcoSphereRef(node_type=node_type)

        # Exact 1-to-4 adjacent hierarchies plus a direct 1-to-16 mapping used
        # only to pool the raw rank-r input into the coarse encoder.
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

        self.input_proj_l5 = InputProj(
            in_channels, embed_dim, act_layer=act_layer
        )
        self.input_proj_l6 = InputProj(
            in_channels, embed_dim, act_layer=act_layer
        )
        self.coarse_region_pool = HierarchicalRegionPool(
            embed_dim, pool_type=coarse_pool_type
        )
        self.context_projection_l5 = nn.Linear(embed_dim, embed_dim)
        self.context_projection_l6 = nn.Linear(embed_dim, embed_dim)
        self.apply_abs_pos_enc_in = abs_pos_enc_in
        if abs_pos_enc_in:
            self.abs_pos_l4 = self._build_input_position_encoding(
                self.coarse_rank, embed_dim
            )
            self.abs_pos_l5 = self._build_input_position_encoding(
                self.middle_rank, embed_dim
            )
            self.abs_pos_l6 = self._build_input_position_encoding(
                self.fine_rank, embed_dim
            )
        self.pos_drop = nn.Dropout(pos_drop_rate)

        common_module_args = dict(
            icosphere_ref=self.icosphere_ref,
            dim=embed_dim,
            num_heads=num_heads,
            d_head_coef=d_head_coef,
            win_size_coef=win_size_coef,
            temporal_window_radius=temporal_window_radius,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
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
            debug_skip_attn=debug_skip_attn,
            append_self=append_self,
        )
        self.coarse_encoder = SphereUFormerModule(
            rank=self.coarse_rank,
            depth=adaptive_coarse_depth,
            **common_module_args,
        )
        self.rank5_encoder = SphereUFormerModule(
            rank=self.middle_rank,
            depth=adaptive_middle_depth,
            **common_module_args,
        )
        self.rank6_encoder = SphereUFormerModule(
            rank=self.fine_rank,
            depth=adaptive_fine_depth,
            **common_module_args,
        )

        self.vertex_to_face_l4 = VertexToFaceAggregation(embed_dim)
        self.vertex_to_face_l5 = VertexToFaceAggregation(embed_dim)
        self.saliency_head_l4 = nn.Linear(embed_dim, out_channels)
        self.saliency_head_l5 = nn.Linear(embed_dim, out_channels)
        self.uncertainty_head_l4 = SphericalUncertaintyHead(embed_dim)
        self.uncertainty_head_l5 = SphericalUncertaintyHead(embed_dim)
        self.refinement_head_l4 = AdaptiveRefinementHead(
            embed_dim,
            use_uncertainty=use_uncertainty_refinement,
            use_motion=use_motion_refinement,
        )
        self.refinement_head_l5 = AdaptiveRefinementHead(
            embed_dim,
            use_uncertainty=use_uncertainty_refinement,
            use_motion=use_motion_refinement,
        )
        self.region_selector_l4 = AdaptiveRegionSelector(adaptive_temperature)
        self.region_selector_l5 = AdaptiveRegionSelector(adaptive_temperature)
        self.fusion_norm_l5 = norm_layer(embed_dim)
        self.fusion_norm_l6 = norm_layer(embed_dim)
        self.output_proj = OutputProj(embed_dim, out_channels)
        self.final_sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _build_input_position_encoding(self, rank, embed_dim):
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
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @staticmethod
    def _validate_gate_override(gate, reference, name):
        if gate.shape != reference.shape:
            raise ValueError(
                f"{name} must have shape {tuple(reference.shape)}, "
                f"got {tuple(gate.shape)}"
            )
        return gate.to(device=reference.device, dtype=reference.dtype).clamp(0, 1)

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
            gate_overrides: Optional[dict] = None,
    ):
        """Predict saliency from rank-r RGB without using labels.

        ``gate_overrides`` is an evaluation-only interface containing optional
        rank-4 ``l4`` and rank-5 ``l5_local`` face gates. It never derives gates
        from ground truth inside the model.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected [B, T, L, C], got {tuple(x.shape)}")
        B, T, L_img, channels = x.shape
        if L_img != self.hierarchy_l5_l6.fine_vertex_count:
            raise ValueError(
                f"img_rank={self.img_rank} expects "
                f"{self.hierarchy_l5_l6.fine_vertex_count} vertices, got {L_img}"
            )
        flat_img = x.reshape(B * T, L_img, channels)

        # Both detail streams originate from the original rank-r RGB.
        raw_rgb_l5 = self.hierarchy_l5_l6.center_downsample_vertices(flat_img)
        raw_features_l5 = self.input_proj_l5(raw_rgb_l5)
        raw_features_l6 = self.input_proj_l6(flat_img)
        coarse_features = self.coarse_region_pool(
            raw_features_l6, self.hierarchy_l4_l6
        )
        if self.apply_abs_pos_enc_in:
            coarse_features = coarse_features + self.abs_pos_l4(coarse_features)
            raw_features_l5 = raw_features_l5 + self.abs_pos_l5(raw_features_l5)
            raw_features_l6 = raw_features_l6 + self.abs_pos_l6(raw_features_l6)
        coarse_features = self.coarse_encoder(
            self.pos_drop(coarse_features), time_steps=T
        )

        face_features_l4 = self.vertex_to_face_l4(
            coarse_features, self.hierarchy_l4_l5
        )
        face_features_l4_bt = face_features_l4.reshape(
            B, T, self.hierarchy_l4_l5.coarse_face_count, self.embed_dim
        )
        saliency_l4 = self.final_sigmoid(
            self.saliency_head_l4(face_features_l4)
        ).squeeze(-1).reshape(B, T, -1)
        uncertainty_l4 = self.uncertainty_head_l4(face_features_l4_bt)
        refine_logits_l4, refine_score_l4 = self.refinement_head_l4(
            face_features_l4_bt, uncertainty_l4
        )
        gate_l4_parent = self.region_selector_l4(
            refine_logits_l4, enabled=self.use_adaptive_refinement
        )
        if gate_overrides and "l4" in gate_overrides:
            gate_l4_parent = self._validate_gate_override(
                gate_overrides["l4"], gate_l4_parent, "gate_overrides['l4']"
            )
        gate_l4_to_l5 = self.hierarchy_l4_l5.propagate_coarse_face_values(
            gate_l4_parent
        )
        vertex_gate_l5 = self.hierarchy_l4_l5.fine_face_values_to_vertices(
            gate_l4_to_l5
        )

        context_l5 = self.upsample_l4_l5(coarse_features)
        candidate_l5 = self.rank5_encoder(
            self.pos_drop(
                raw_features_l5 + self.context_projection_l5(context_l5)
            ),
            time_steps=T,
        )
        fused_l5 = self.fusion_norm_l5(
            context_l5
            + vertex_gate_l5.reshape(B * T, -1, 1)
            * (candidate_l5 - context_l5)
        )

        face_features_l5 = self.vertex_to_face_l5(
            fused_l5, self.hierarchy_l5_l6
        )
        face_features_l5_bt = face_features_l5.reshape(
            B, T, self.hierarchy_l5_l6.coarse_face_count, self.embed_dim
        )
        saliency_l5 = self.final_sigmoid(
            self.saliency_head_l5(face_features_l5)
        ).squeeze(-1).reshape(B, T, -1)
        uncertainty_l5 = self.uncertainty_head_l5(face_features_l5_bt)
        refine_logits_l5, refine_score_l5 = self.refinement_head_l5(
            face_features_l5_bt, uncertainty_l5
        )
        gate_l5_local = self.region_selector_l5(
            refine_logits_l5, enabled=self.use_adaptive_refinement
        )
        if gate_overrides and "l5_local" in gate_overrides:
            gate_l5_local = self._validate_gate_override(
                gate_overrides["l5_local"],
                gate_l5_local,
                "gate_overrides['l5_local']",
            )

        # A rank-5 child can only refine when its exact rank-4 parent is active.
        gate_l5_effective_parent = gate_l4_to_l5 * gate_l5_local
        gate_l5_local_fine = self.hierarchy_l5_l6.propagate_coarse_face_values(
            gate_l5_local
        )
        gate_l5_to_l6_effective = (
            self.hierarchy_l5_l6.propagate_coarse_face_values(
                gate_l5_effective_parent
            )
        )
        vertex_gate_l6 = self.hierarchy_l5_l6.fine_face_values_to_vertices(
            gate_l5_to_l6_effective
        )

        context_l6 = self.upsample_l5_l6(fused_l5)
        candidate_l6 = self.rank6_encoder(
            self.pos_drop(
                raw_features_l6 + self.context_projection_l6(context_l6)
            ),
            time_steps=T,
        )
        fused_l6 = self.fusion_norm_l6(
            context_l6
            + vertex_gate_l6.reshape(B * T, -1, 1)
            * (candidate_l6 - context_l6)
        )
        final_logits = self.output_proj(fused_l6)
        if self.out_channels == 1:
            saliency = self.final_sigmoid(
                final_logits.squeeze(-1).reshape(B, T, L_img)
            )
        else:
            saliency = self.final_sigmoid(
                final_logits.reshape(B, T, L_img, self.out_channels)
            )

        area_ratio_l1 = self.hierarchy_l4_l5.area_weighted_fine_face_ratio(
            gate_l4_to_l5
        )
        area_ratio_l2 = self.hierarchy_l5_l6.area_weighted_fine_face_ratio(
            gate_l5_to_l6_effective
        )
        if self.debug_adaptive:
            print(
                "AdaptiveSphereUFormerV2:",
                f"ranks={self.coarse_rank}->{self.middle_rank}->{self.fine_rank}",
                f"vertices={self.hierarchy_l4_l5.coarse_vertex_count}->"
                f"{self.hierarchy_l4_l5.fine_vertex_count}->"
                f"{self.hierarchy_l5_l6.fine_vertex_count}",
                f"faces={self.hierarchy_l4_l5.coarse_face_count}->"
                f"{self.hierarchy_l4_l5.fine_face_count}->"
                f"{self.hierarchy_l5_l6.fine_face_count}",
                f"uncertainty_l4={uncertainty_l4.mean().item():.4f}",
                f"uncertainty_l5={uncertainty_l5.mean().item():.4f}",
                f"area_l1={area_ratio_l1.mean().item():.4f}",
                f"area_l2={area_ratio_l2.mean().item():.4f}",
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
            "gate_l4_parent": gate_l4_parent,
            "gate_l4_to_l5": gate_l4_to_l5,
            "vertex_gate_l5": vertex_gate_l5,
            "area_ratio_l1": area_ratio_l1,
            "saliency_l5": saliency_l5,
            "uncertainty_l5": uncertainty_l5,
            "refine_logits_l5": refine_logits_l5,
            "refine_score_l5": refine_score_l5,
            "gate_l5_to_l6_local": gate_l5_local,
            "gate_l5_to_l6_local_fine": gate_l5_local_fine,
            "gate_l5_to_l6_effective": gate_l5_to_l6_effective,
            "vertex_gate_l6": vertex_gate_l6,
            "area_ratio_l2": area_ratio_l2,
        }


def build_saliency_model(args, node_type: Optional[str] = None) -> nn.Module:
    """Build the configured baseline or adaptive saliency model."""
    node_type = node_type or args.mode
    common = dict(
        img_rank=args.img_rank,
        node_type=node_type,
        in_channels=3,
        out_channels=1,
        in_scale_factor=args.scale_factor,
        win_size_coef=args.win_size_coef,
        temporal_window_radius=args.temporal_window_radius,
        d_head_coef=args.d_head_coef,
        abs_pos_enc_in=args.abs_pos_enc_in,
        abs_pos_enc=args.abs_pos_enc,
        rel_pos_bias=args.rel_pos_bias,
        rel_pos_bias_size=args.rel_pos_bias_size,
        rel_pos_init_variance=args.rel_pos_init_variance,
        drop_rate=args.dr,
        drop_path_rate=args.dpr,
        attn_drop_rate=args.adr,
        attn_out_drop_rate=args.aodr,
        pos_drop_rate=args.posdr,
        debug_skip_attn=args.debug_skip_attn,
        append_self=args.append_self,
        use_checkpoint=args.use_checkpoint,
    )
    model_type = getattr(args, "model_type", "sphere_uformer")
    if model_type == "sphere_uformer":
        return SphereUFormer(
            num_scales=args.num_scales,
            enc_depths=args.scale_depth,
            dec_depths=args.scale_depth,
            bottleneck_depth=args.scale_depth,
            enc_num_heads=args.enc_num_heads,
            bottleneck_num_heads=args.bottleneck_num_heads,
            dec_num_heads=args.dec_num_heads,
            downsample=args.downsample,
            upsample=args.upsample,
            **common,
        )
    if model_type == "adaptive_sphere_uformer_v2":
        return AdaptiveSphereUFormerV2(
            num_heads=args.enc_num_heads[0],
            adaptive_coarse_depth=args.adaptive_coarse_depth,
            adaptive_middle_depth=getattr(args, "adaptive_middle_depth", 1),
            adaptive_fine_depth=args.adaptive_fine_depth,
            adaptive_temperature=args.adaptive_temperature,
            adaptive_region_type=getattr(args, "adaptive_region_type", "face"),
            coarse_pool_type=getattr(args, "coarse_pool_type", "mean_max"),
            face_to_vertex_reduce=getattr(args, "face_to_vertex_reduce", "mean"),
            use_adaptive_refinement=args.use_adaptive_refinement,
            use_uncertainty_refinement=getattr(
                args, "use_uncertainty_refinement", True
            ),
            use_motion_refinement=args.use_motion_refinement,
            return_aux=args.return_aux,
            debug_adaptive=args.debug_adaptive,
            **common,
        )
    if model_type != "adaptive_sphere_uformer":
        raise ValueError(f"Unsupported model_type: {model_type}")
    return AdaptiveSphereUFormer(
        num_heads=args.enc_num_heads[0],
        coarse_rank_offset=args.coarse_rank_offset,
        adaptive_coarse_depth=args.adaptive_coarse_depth,
        adaptive_fine_depth=args.adaptive_fine_depth,
        adaptive_temperature=args.adaptive_temperature,
        adaptive_region_type=getattr(args, "adaptive_region_type", "face"),
        coarse_pool_type=getattr(args, "coarse_pool_type", "mean_max"),
        face_to_vertex_reduce=getattr(args, "face_to_vertex_reduce", "mean"),
        use_adaptive_refinement=args.use_adaptive_refinement,
        use_uncertainty_refinement=getattr(
            args, "use_uncertainty_refinement", True
        ),
        use_motion_refinement=args.use_motion_refinement,
        return_aux=args.return_aux,
        debug_adaptive=args.debug_adaptive,
        **common,
    )
