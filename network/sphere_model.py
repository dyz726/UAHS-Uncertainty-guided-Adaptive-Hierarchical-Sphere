import random
import warnings
from functools import partial
from typing import Dict, Union, List, Optional

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
            temporal_window_radius: Optional[int] = 5,
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
            temporal_window_radius: Optional[int] = 5,
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
            temporal_window_radius: Optional[int] = 5,
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


def build_saliency_model(args, node_type: Optional[str] = None) -> nn.Module:
    """Build the published baseline or final UAHS model."""
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
    if model_type != "uahs":
        raise ValueError(f"Unsupported model_type: {model_type}")
    # Imported lazily to keep shared baseline primitives in this module without
    # creating an import cycle when network.uahs reuses them.
    from .uahs import UAHS
    return UAHS(
        num_heads=args.enc_num_heads[0],
        coarse_pool_type=getattr(args, "coarse_pool_type", "mean_max"),
        target_refine_ratio_l1=args.target_refine_ratio_l1,
        target_refine_ratio_l2=args.target_refine_ratio_l2,
        budget_l5_min=getattr(args, "budget_l5_min", 0.05),
        budget_l5_max=getattr(args, "budget_l5_max", 0.50),
        global_query_chunk_size=args.global_query_chunk_size,
        hard_selection_warmup_epochs=args.hard_selection_warmup_epochs,
        return_aux=args.return_aux,
        debug_uahs=args.debug_uahs,
        **common,
    )
