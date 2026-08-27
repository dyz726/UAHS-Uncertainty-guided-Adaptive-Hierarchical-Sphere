from copy import deepcopy
from typing import List, Union, Tuple, Dict, Set

import numpy as np
import trimesh
import trimesh.creation
from trimesh import Trimesh


def asCartesian(rphitheta: np.ndarray) -> np.ndarray:
    """球坐标(r,φ,θ)转笛卡尔坐标(x,y,z)
    r: 半径
    φ: 天顶角 [0°,180°] (从+z轴开始)
    θ: 方位角 [-180°,180°] (从+x轴开始)
    """
    r = rphitheta[:, 0]
    phi = np.deg2rad(rphitheta[:, 1])       
    theta = np.deg2rad(rphitheta[:, 2])

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def asSpherical(xyz: np.ndarray) -> np.ndarray:
    """笛卡尔坐标(x,y,z)转球坐标(r,φ,θ)
    返回: [r, φ°, θ°] 数组
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.rad2deg(np.arccos(z / r))             
    theta = np.rad2deg(np.arctan2(y, x))                
    return np.stack([r, phi, theta], axis=1)


def bilinear_interpolate_numpy(im, x, y):
    """双线性插值实现 (numpy版)"""
              
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1

          
    x0, x1 = np.clip(x0, 0, im.shape[1] - 1), np.clip(x1, 0, im.shape[1] - 1)
    y0, y1 = np.clip(y0, 0, im.shape[0] - 1), np.clip(y1, 0, im.shape[0] - 1)

             
    Ia, Ib = im[y0, x0], im[y1, x0]
    Ic, Id = im[y0, x1], im[y1, x1]

          
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

          
    return (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T


def get_icosphere(subdivisions: int, refine=True, radius=1.0, **kwargs):
    """创建细分级别的二十面体球体"""

    def refine_spherical(_mesh):
        """将顶点精确投影到球面上"""
        vectors = _mesh.vertices
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit_vectors = vectors / norms
                      
        offsets = radius - norms.squeeze()
        _mesh.vertices += unit_vectors * offsets[:, np.newaxis]

              
    ico = trimesh.creation.icosahedron()
    ico._validate = False          

          
    for _ in range(subdivisions):
        ico = ico.subdivide()        
        if refine:
            refine_spherical(ico)        

              
    return trimesh.Trimesh(
        vertices=ico.vertices,
        faces=ico.faces,
        metadata={'shape': 'sphere', 'radius': radius},
        process=False,          
        **kwargs
    )


def find_face_neighbors(mesh: Trimesh, depth: int) -> List[Set[int]]:
    """查找面的邻居关系 (广度优先搜索)"""
    num_faces = len(mesh.faces)

             
    old_neighbors = [set() for _ in range(num_faces)]
    neighbors = [{i} for i in range(num_faces)]          

                       
    first_order = [set() for _ in range(num_faces)]
    for idx, face in enumerate(mesh.faces):
                              
        for vertex in face:
            for adj_face in mesh.vertex_faces[vertex]:
                if adj_face != -1 and adj_face != idx:
                    first_order[idx].add(adj_face)

            
    for _ in range(depth):
        new_neighbors = deepcopy(neighbors)
        for i in range(num_faces):
                        
            for n in neighbors[i] - old_neighbors[i]:
                new_neighbors[i] |= first_order[n]            
        old_neighbors = neighbors
        neighbors = new_neighbors

    return neighbors


def find_vertex_neighbors(mesh: Trimesh, depth: int) -> List[Set[int]]:
    """查找顶点的邻居关系 (广度优先搜索)"""
    num_vertices = len(mesh.vertices)

             
    old_neighbors = [set() for _ in range(num_vertices)]
    neighbors = [{i} for i in range(num_vertices)]          

                     
    first_order = [set(mesh.vertex_neighbors[i]) for i in range(num_vertices)]

            
    for _ in range(depth):
        new_neighbors = deepcopy(neighbors)
        for i in range(num_vertices):
            for n in neighbors[i] - old_neighbors[i]:
                new_neighbors[i] |= first_order[n]            
        old_neighbors = neighbors
        neighbors = new_neighbors

    return neighbors


class IcoSphereRef:
    """二十面体球参考系管理器"""

    def __init__(self, node_type: str):
        assert node_type in ("face", "vertex")              
        self.node_type = node_type
        self.icospheres = {}             
        self.neighbor_maps = {}          

    def get_icosphere(self, rank: int, refine: bool) -> Trimesh:
        """获取或创建二十面体球体 (带缓存)"""
        key = (rank, refine)
        if key not in self.icospheres:
            self.icospheres[key] = get_icosphere(rank, refine)
        return self.icospheres[key]

    def get_neighbor_mapping(self, rank: int, depth: int) -> List[Set[int]]:
        """获取邻居映射 (带缓存)"""
        key = (rank, depth)
        if key not in self.neighbor_maps:
            ico = self.get_icosphere(rank, True)
            print(f"Building neighbor mapping {rank}-{depth}")
                            
            if self.node_type == "face":
                self.neighbor_maps[key] = find_face_neighbors(ico, depth)
            else:
                self.neighbor_maps[key] = find_vertex_neighbors(ico, depth)
            print("Building complete")
        return self.neighbor_maps[key]

    def get_normals(self, rank: int) -> np.ndarray:
        """获取法向量"""
        ico = self.get_icosphere(rank, True)
                                  
        return ico.face_normals if self.node_type == "face" else ico.vertices