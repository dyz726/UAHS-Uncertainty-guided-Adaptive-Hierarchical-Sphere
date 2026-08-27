import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from tqdm import tqdm

from trimesh_utils import IcoSphereRef, asSpherical


def get_rotation_matrices(rphitheta: np.ndarray) -> np.ndarray:
    r, p, t = rphitheta.T
    p = (90 - p) / 180 * np.pi
    t = t / 180 * np.pi
    zeros, ones = np.zeros_like(t), np.ones_like(t)
    mat_t = np.array([
        [np.cos(-t), -np.sin(-t), zeros],
        [np.sin(-t), np.cos(-t), zeros],
        [zeros, zeros, ones],
    ]).transpose(2,0,1)
    mat_p = np.array([
        [np.cos(p), zeros, np.sin(p)],
        [zeros, ones, zeros],
        [-np.sin(p), zeros, np.cos(p)],
    ]).transpose(2,0,1)
    return mat_p @ mat_t



class GlobalVerticalPositionEnconding(nn.Module):
    def __init__(self, rank: int, icosphere_ref: IcoSphereRef, mode: str, num_pos_feats,
                 max_frequency, min_frequency=1, scale=None):
        super().__init__()

        self.mode = mode
        self.num_pos_feats = num_pos_feats
        self.max_frequency = max_frequency
        self.min_frequency = min_frequency

        scale = math.pi
        self.scale = scale

        normals = icosphere_ref.get_normals(rank)
        spherical_coords = asSpherical(normals)

                   
        dim_t = torch.arange(0, num_pos_feats // 2, dtype=torch.float32)
        dim_t = self.max_frequency ** (2 * dim_t / (num_pos_feats * 2))

        pos = None

                    
                  
        phi = torch.tensor(spherical_coords[:, 1].copy(), dtype=torch.float32).view(-1, 1)
        phi_norm = (phi - 90) / 180
        phi_enc = phi_norm * self.scale / dim_t          

                  
        theta = torch.tensor(spherical_coords[:, 2].copy(), dtype=torch.float32).view(-1, 1)
        theta_rad = theta / 180       
                      
        theta_scale =  self.scale
        theta_enc = theta_rad * theta_scale / dim_t          

                 
        phi_enc = torch.stack((phi_enc.sin(), phi_enc.cos()), dim=1).flatten(start_dim=1)
        theta_enc = torch.stack((theta_enc.sin(), theta_enc.cos()), dim=1).flatten(start_dim=1)
        pos = torch.cat([phi_enc, theta_enc], dim=-1)

        self.register_buffer("pos", pos, persistent=False)

    def forward(self, x: Tensor):
        pos = self.pos.unsqueeze(0)
        return pos


class RelativePositionBias(nn.Module):
    def __init__(self, rank: int, icosphere_ref: IcoSphereRef, win_size_coef: int, rel_pos_bias_size: int, num_heads: int, init_variance: float = 10):
        assert rel_pos_bias_size > 0

        super().__init__()

        self.rank = rank

        normals = icosphere_ref.get_normals(rank)

        normals_rphitheta = asSpherical(normals)
        rot_mat = get_rotation_matrices(normals_rphitheta)

        mapping = icosphere_ref.get_neighbor_mapping(rank=rank, depth=win_size_coef)

        self.num_nodes = len(mapping)
        self.num_keys = max(len(_) for _ in mapping)

                                                      
        idx = torch.arange(0, self.num_nodes).unsqueeze(1).expand(-1, self.num_keys).clone()                                     
        idx_mask = torch.zeros(self.num_nodes, self.num_keys).bool()
        for i, keys in tqdm(enumerate(mapping), desc=f"RelativePositionBias - index mapping {rank}"):
            idx[i, :len(keys)] = torch.tensor(list(keys))
            idx_mask[i, :len(keys)] = 1

                                                  
        self.register_buffer("idx", idx[None, None, :, :, None], persistent=False)
        self.register_buffer("idx_mask", idx_mask[None, None, :, :], persistent=False)

        expanded_normals = torch.tensor(normals.copy(), dtype=torch.float64).unsqueeze(1).expand(-1, self.num_keys, -1)
        expanded_idx = idx.unsqueeze(2).expand(-1, -1, 3)
        aligned_neighbors = torch.gather(expanded_normals, dim=0, index=expanded_idx).numpy()

        rotated_neighbors = (rot_mat @ aligned_neighbors.transpose(0,2,1)).transpose(0,2,1)
        rotated_neighbors_flat = rotated_neighbors.reshape(self.num_nodes*self.num_keys, 3)
        relative_coords_flat = (asSpherical(rotated_neighbors_flat)[:, 1:] - np.array([[90, 0]]))
        relative_coords = relative_coords_flat.reshape(self.num_nodes, self.num_keys, 2)

        self.register_buffer("relative_coords", torch.tensor(relative_coords).float(), persistent=False)
        self.bias_grid = nn.Parameter(init_variance * torch.randn(1, num_heads, rel_pos_bias_size, rel_pos_bias_size), requires_grad=True)

    def get_neighbor_idx(self):
        return self.idx, self.idx_mask

    def forward(self, keys: Tensor):
        N, H, D, K, C_H = keys.shape
        assert D == self.relative_coords.shape[0]
        assert K == self.relative_coords.shape[1]
                                                  
                                               
                                                                                        

        rel_coords = self.relative_coords.unsqueeze(0)                          
        rel_coords_normalized = rel_coords / (rel_coords.abs().max() + 1e-8)
        rel_bias = F.grid_sample(self.bias_grid, grid=rel_coords_normalized, align_corners=True)
        return rel_coords, rel_bias
