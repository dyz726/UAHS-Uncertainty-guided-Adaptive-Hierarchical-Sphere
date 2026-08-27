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
from trimesh_utils import get_icosphere, IcoSphereRef, asSpherical



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


def _nearest_spherical_indices(
        source_normals: np.ndarray,
        target_normals: np.ndarray,
        chunk_size: int = 4096,
) -> torch.Tensor:
    """Map each source normal to its nearest target without a full similarity matrix."""
    target = target_normals.astype(np.float32, copy=False)
    indices = []
    for start in range(0, len(source_normals), chunk_size):
        source = source_normals[start:start + chunk_size].astype(np.float32, copy=False)
        indices.append(np.argmax(source @ target.T, axis=1))
    return torch.from_numpy(np.concatenate(indices).astype(np.int64, copy=False))


def _vertex_ancestor_indices(
        coarse_rank: int,
        fine_rank: int,
        icosphere_ref: IcoSphereRef,
) -> torch.Tensor:
    """Follow trimesh midpoint parents to one deterministic coarse ancestor."""
    ancestor = torch.arange(
        len(icosphere_ref.get_normals(coarse_rank)), dtype=torch.long
    )
    for rank in range(coarse_rank + 1, fine_rank + 1):
        mesh = icosphere_ref.get_icosphere(rank, refine=True)
        previous_size = ancestor.numel()
        current = torch.empty(len(mesh.vertices), dtype=torch.long)
        current[:previous_size] = ancestor
        for node_idx in range(previous_size, len(mesh.vertices)):
            parents = [
                neighbor for neighbor in mesh.vertex_neighbors[node_idx]
                if neighbor < previous_size
            ]
            if len(parents) != 2:
                raise RuntimeError(
                    f"Vertex {node_idx} at rank {rank} has {len(parents)} parents"
                )
            current[node_idx] = ancestor[min(parents)]
        ancestor = current
    return ancestor


class SphericalHierarchyMapping(nn.Module):
    """Reusable nearest-region mapping between two cached icosphere ranks.

    The mapping is built once during initialization and stored as buffers.  It is
    deliberately dense indexing, not sparse adaptive computation.
    """

    def __init__(
            self,
            coarse_rank: int,
            fine_rank: int,
            icosphere_ref: IcoSphereRef,
    ):
        super().__init__()
        if coarse_rank < 0 or fine_rank < coarse_rank:
            raise ValueError(
                f"Expected 0 <= coarse_rank <= fine_rank, got "
                f"{coarse_rank} and {fine_rank}"
            )

        self.coarse_rank = coarse_rank
        self.fine_rank = fine_rank
        coarse_normals = icosphere_ref.get_normals(coarse_rank)
        fine_normals = icosphere_ref.get_normals(fine_rank)

        if coarse_rank == fine_rank:
            fine_to_coarse = torch.arange(len(fine_normals), dtype=torch.long)
            coarse_center = fine_to_coarse.clone()
        elif icosphere_ref.node_type == "vertex":
            # Subdivision retains old vertices and appends edge midpoints.
            fine_to_coarse = _vertex_ancestor_indices(
                coarse_rank, fine_rank, icosphere_ref
            )
            coarse_center = torch.arange(len(coarse_normals), dtype=torch.long)
        else:
            # Trimesh emits four consecutive children for every parent face;
            # this is also the ordering assumed by the existing face pooling.
            fine_to_coarse = torch.arange(
                len(coarse_normals), dtype=torch.long
            ).repeat_interleave(4 ** (fine_rank - coarse_rank))
            coarse_center = _nearest_spherical_indices(
                coarse_normals, fine_normals
            )

        child_count = torch.bincount(
            fine_to_coarse, minlength=len(coarse_normals)
        ).clamp_min(1)
        self.register_buffer("fine_to_coarse_idx", fine_to_coarse)
        self.register_buffer("coarse_center_idx", coarse_center)
        self.register_buffer("child_count", child_count)

    @property
    def coarse_size(self) -> int:
        return int(self.child_count.numel())

    @property
    def fine_size(self) -> int:
        return int(self.fine_to_coarse_idx.numel())

    def upsample(self, x: Tensor, node_dim: int) -> Tensor:
        """Nearest-region upsample along ``node_dim``."""
        return torch.index_select(x, node_dim, self.fine_to_coarse_idx)

    def center_downsample(self, x: Tensor, node_dim: int) -> Tensor:
        """Select the fine node nearest each coarse node."""
        return torch.index_select(x, node_dim, self.coarse_center_idx)

    def aggregate_mean(self, values: Tensor) -> Tensor:
        """Mean fine scalar values into coarse regions; node dimension is last."""
        if values.shape[-1] != self.fine_size:
            raise ValueError(
                f"Expected {self.fine_size} fine nodes, got {values.shape[-1]}"
            )
        flat = values.reshape(-1, self.fine_size)
        index = self.fine_to_coarse_idx.unsqueeze(0).expand(flat.shape[0], -1)
        sums = flat.new_zeros((flat.shape[0], self.coarse_size))
        sums.scatter_add_(1, index, flat)
        means = sums / self.child_count.to(dtype=flat.dtype).unsqueeze(0)
        return means.reshape(*values.shape[:-1], self.coarse_size)


class AdaptiveRefinementHead(nn.Module):
    """Predict region difficulty from coarse content and temporal change."""

    def __init__(self, dim: int, use_motion: bool = True):
        super().__init__()
        hidden_dim = max(1, dim // 2)
        self.use_motion = use_motion
        self.refinement_mlp = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Return logits and scores for features shaped [B, T, Lc, C]."""
        if self.use_motion and features.shape[1] > 1:
            delta = (features[:, 1:] - features[:, :-1]).abs().mean(
                dim=-1, keepdim=True
            )
            first = torch.zeros_like(delta[:, :1])
            motion = torch.cat((first, delta), dim=1)
        else:
            motion = torch.zeros_like(features[..., :1])
        logits = self.refinement_mlp(torch.cat((features, motion), dim=-1)).squeeze(-1)
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
    """Differentiable coarse/fine saliency prototype with a soft adaptive gate.

    The fine encoder still processes every fine node.  The gate tests adaptive
    multi-resolution refinement, but does not yet reduce sparse FLOPs.
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
            use_adaptive_refinement: bool = True,
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
        self.use_adaptive_refinement = use_adaptive_refinement
        self.return_aux = return_aux
        self.debug_adaptive = debug_adaptive
        self.icosphere_ref = IcoSphereRef(node_type=node_type)

        # All hierarchy work is performed once here and then moved with the model.
        self.coarse_fine_mapping = SphericalHierarchyMapping(
            self.coarse_rank, self.fine_rank, self.icosphere_ref
        )
        self.fine_img_mapping = SphericalHierarchyMapping(
            self.fine_rank, self.img_rank, self.icosphere_ref
        )
        if node_type == "vertex":
            self.output_upsample = nn.Sequential(*[
                InterpolateUpsample(rank, rank + 1, self.icosphere_ref)
                for rank in range(self.fine_rank, self.img_rank)
            ])
        else:
            # InterpolateUpsample only supports vertex nodes in the baseline.
            self.output_upsample = None
        img_to_coarse = self.coarse_fine_mapping.fine_to_coarse_idx[
            self.fine_img_mapping.fine_to_coarse_idx
        ]
        img_child_count = torch.bincount(
            img_to_coarse,
            minlength=self.coarse_fine_mapping.coarse_size,
        ).clamp_min(1)
        self.register_buffer("img_to_coarse_idx", img_to_coarse)
        self.register_buffer("img_child_count", img_child_count)

        self.fine_input_proj = InputProj(
            in_channels, embed_dim, act_layer=act_layer
        )
        self.coarse_input_proj = InputProj(
            in_channels, embed_dim, act_layer=act_layer
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

        self.coarse_saliency_head = nn.Linear(embed_dim, out_channels)
        self.refinement_head = AdaptiveRefinementHead(
            embed_dim, use_motion=use_motion_refinement
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
        """Upsample [B, T, Lc] values to [B, T, Limg] for auxiliary loss."""
        fine = self.coarse_fine_mapping.upsample(values, node_dim=-1)
        if self.output_upsample is None:
            return self.fine_img_mapping.upsample(fine, node_dim=-1)
        shape = fine.shape
        img = self.output_upsample(fine.reshape(-1, shape[-1], 1))
        return img.squeeze(-1).reshape(*shape[:-1], img.shape[1])

    def aggregate_img_values_to_coarse(self, values: Tensor) -> Tensor:
        """Aggregate [B, T, Limg] scalar targets to nearest coarse regions."""
        if values.shape[-1] != self.fine_img_mapping.fine_size:
            raise ValueError(
                f"Expected {self.fine_img_mapping.fine_size} image nodes, "
                f"got {values.shape[-1]}"
            )
        flat = values.reshape(-1, values.shape[-1])
        index = self.img_to_coarse_idx.unsqueeze(0).expand(flat.shape[0], -1)
        sums = flat.new_zeros((flat.shape[0], self.coarse_fine_mapping.coarse_size))
        sums.scatter_add_(1, index, flat)
        means = sums / self.img_child_count.to(flat.dtype).unsqueeze(0)
        return means.reshape(
            *values.shape[:-1], self.coarse_fine_mapping.coarse_size
        )

    def forward(self, x: Tensor, return_aux: Optional[bool] = None):
        """Predict saliency from input [B, T, Limg, Cin]."""
        if x.ndim != 4:
            raise ValueError(f"Expected [B, T, L, C] input, got {tuple(x.shape)}")
        B, T, L_img, C = x.shape
        if L_img != self.fine_img_mapping.fine_size:
            raise ValueError(
                f"img_rank={self.img_rank} expects "
                f"{self.fine_img_mapping.fine_size} nodes, got {L_img}"
            )
        flat_img = x.reshape(B * T, L_img, C)

        # Fine RGB is center-sampled from img_rank; coarse RGB from fine_rank.
        fine_rgb = self.fine_img_mapping.center_downsample(flat_img, node_dim=1)
        coarse_rgb = self.coarse_fine_mapping.center_downsample(fine_rgb, node_dim=1)
        fine_features = self.fine_input_proj(fine_rgb)
        coarse_features = self.coarse_input_proj(coarse_rgb)
        if self.apply_abs_pos_enc_in:
            fine_features = fine_features + self.fine_abs_pos_in(fine_features)
            coarse_features = coarse_features + self.coarse_abs_pos_in(coarse_features)
        fine_features = self.pos_drop(fine_features)
        coarse_features = self.pos_drop(coarse_features)

        coarse_features = self.coarse_encoder(coarse_features, time_steps=T)
        coarse_bt = coarse_features.reshape(
            B, T, self.coarse_fine_mapping.coarse_size, self.embed_dim
        )
        coarse_logits = self.coarse_saliency_head(coarse_features)
        coarse_saliency = self.final_sigmoid(coarse_logits)

        refine_logits, refine_score = self.refinement_head(coarse_bt)
        gate_coarse = self.region_selector(
            refine_logits, enabled=self.use_adaptive_refinement
        )
        fine_refine_score = self.coarse_fine_mapping.upsample(
            refine_score, node_dim=-1
        )
        fine_gate = self.coarse_fine_mapping.upsample(gate_coarse, node_dim=-1)

        fine_features = self.fine_encoder(fine_features, time_steps=T)
        coarse_up = self.coarse_fine_mapping.upsample(coarse_features, node_dim=1)
        fine_residual = self.fine_projection(fine_features)
        fused = self.fusion_norm(
            coarse_up + fine_gate.reshape(B * T, -1, 1) * fine_residual
        )
        if self.output_upsample is None:
            img_features = self.fine_img_mapping.upsample(fused, node_dim=1)
        else:
            img_features = self.output_upsample(fused)
        final_logits = self.output_proj(img_features)

        if self.out_channels == 1:
            saliency = self.final_sigmoid(
                final_logits.squeeze(-1).reshape(B, T, L_img)
            )
            coarse_saliency = coarse_saliency.squeeze(-1).reshape(
                B, T, self.coarse_fine_mapping.coarse_size
            )
        else:
            saliency = self.final_sigmoid(
                final_logits.reshape(B, T, L_img, self.out_channels)
            )
            coarse_saliency = coarse_saliency.reshape(
                B, T, self.coarse_fine_mapping.coarse_size, self.out_channels
            )

        if self.debug_adaptive:
            print(
                "AdaptiveSphereUFormer:",
                f"coarse_rank={self.coarse_rank}",
                f"fine_rank={self.fine_rank}",
                f"Lc={self.coarse_fine_mapping.coarse_size}",
                f"Lf={self.coarse_fine_mapping.fine_size}",
                f"refine_mean={refine_score.mean().item():.4f}",
                f"refine_max={refine_score.max().item():.4f}",
                f"refine_min={refine_score.min().item():.4f}",
                f"gate_mean={fine_gate.mean().item():.4f}",
            )

        return_aux = self.return_aux if return_aux is None else return_aux
        if not return_aux:
            return saliency
        return {
            "saliency": saliency,
            "coarse_saliency": coarse_saliency,
            "refine_score": refine_score,
            "fine_refine_score": fine_refine_score,
            "gate_mean": fine_gate.mean(),
            "gate_max": fine_gate.max(),
            "gate_min": fine_gate.min(),
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
    if model_type != "adaptive_sphere_uformer":
        raise ValueError(f"Unsupported model_type: {model_type}")
    return AdaptiveSphereUFormer(
        num_heads=args.enc_num_heads[0],
        coarse_rank_offset=args.coarse_rank_offset,
        adaptive_coarse_depth=args.adaptive_coarse_depth,
        adaptive_fine_depth=args.adaptive_fine_depth,
        adaptive_temperature=args.adaptive_temperature,
        use_adaptive_refinement=args.use_adaptive_refinement,
        use_motion_refinement=args.use_motion_refinement,
        return_aux=args.return_aux,
        debug_adaptive=args.debug_adaptive,
        **common,
    )
