from copy import deepcopy
from typing import List, Union, Tuple, Dict, Set

import numpy as np
import torch
import torch.nn as nn
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


def spherical_triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return exact spherical areas for unit-sphere triangular faces."""
    triangles = vertices[faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    numerator = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    denominator = (
        1.0
        + np.einsum("ij,ij->i", a, b)
        + np.einsum("ij,ij->i", b, c)
        + np.einsum("ij,ij->i", c, a)
    )
    return 2.0 * np.arctan2(numerator, denominator)


def _coordinate_key(vertex: np.ndarray) -> Tuple[float, float, float]:
    return tuple(np.round(vertex, decimals=10))


def build_adjacent_face_hierarchy(
        coarse_mesh: Trimesh,
        fine_mesh: Trimesh,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an exact one-step 1-to-4 face mapping from mesh topology.

    The implementation reconstructs every edge midpoint and looks up child
    triangles by their vertex sets. It therefore validates rather than assumes
    trimesh's face ordering.
    """
    if len(fine_mesh.faces) != 4 * len(coarse_mesh.faces):
        raise ValueError("Adjacent icosphere ranks must have a 1-to-4 face ratio")

    fine_vertex_by_coordinate = {
        _coordinate_key(vertex): index
        for index, vertex in enumerate(fine_mesh.vertices)
    }
    fine_face_by_vertices = {
        tuple(sorted(map(int, face))): index
        for index, face in enumerate(fine_mesh.faces)
    }
    fine_to_coarse = np.full(len(fine_mesh.faces), -1, dtype=np.int64)
    coarse_to_fine = np.empty((len(coarse_mesh.faces), 4), dtype=np.int64)

    for coarse_index, (a, b, c) in enumerate(coarse_mesh.faces):
        midpoints = []
        for left, right in ((a, b), (b, c), (c, a)):
            midpoint = coarse_mesh.vertices[left] + coarse_mesh.vertices[right]
            midpoint /= np.linalg.norm(midpoint)
            key = _coordinate_key(midpoint)
            if key not in fine_vertex_by_coordinate:
                raise RuntimeError("Subdivision midpoint is absent from the fine mesh")
            midpoints.append(fine_vertex_by_coordinate[key])

        ab, bc, ca = midpoints
        expected_children = (
            (a, ab, ca),
            (ab, b, bc),
            (ca, bc, c),
            (ab, bc, ca),
        )
        for child_offset, child_vertices in enumerate(expected_children):
            key = tuple(sorted(map(int, child_vertices)))
            if key not in fine_face_by_vertices:
                raise RuntimeError("Expected subdivision child face is absent")
            child_index = fine_face_by_vertices[key]
            if fine_to_coarse[child_index] != -1:
                raise RuntimeError("A fine face was assigned to multiple parents")
            fine_to_coarse[child_index] = coarse_index
            coarse_to_fine[coarse_index, child_offset] = child_index

    if np.any(fine_to_coarse < 0):
        raise RuntimeError("Face hierarchy does not cover every fine face")
    return fine_to_coarse, coarse_to_fine


def build_face_hierarchy(
        icosphere_ref: IcoSphereRef,
        coarse_rank: int,
        fine_rank: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build topology-derived face ancestry across one or more ranks."""
    if coarse_rank < 0 or fine_rank < coarse_rank:
        raise ValueError("Expected 0 <= coarse_rank <= fine_rank")
    coarse_face_count = len(icosphere_ref.get_icosphere(coarse_rank, True).faces)
    ancestor = np.arange(coarse_face_count, dtype=np.int64)
    for rank in range(coarse_rank + 1, fine_rank + 1):
        adjacent_parent, _ = build_adjacent_face_hierarchy(
            icosphere_ref.get_icosphere(rank - 1, True),
            icosphere_ref.get_icosphere(rank, True),
        )
        ancestor = ancestor[adjacent_parent]

    descendants_per_face = 4 ** (fine_rank - coarse_rank)
    counts = np.bincount(ancestor, minlength=coarse_face_count)
    if not np.all(counts == descendants_per_face):
        raise RuntimeError("Face hierarchy has an invalid descendant count")
    order = np.argsort(ancestor, kind="stable")
    coarse_to_fine = order.reshape(coarse_face_count, descendants_per_face)
    return ancestor, coarse_to_fine


class IcoSphereHierarchy(nn.Module):
    """Fixed, topology-derived face hierarchy for a vertex feature backbone."""

    def __init__(
            self,
            coarse_rank: int,
            fine_rank: int,
            icosphere_ref: IcoSphereRef,
    ):
        super().__init__()
        coarse_mesh = icosphere_ref.get_icosphere(coarse_rank, True)
        fine_mesh = icosphere_ref.get_icosphere(fine_rank, True)
        fine_to_coarse, coarse_to_fine = build_face_hierarchy(
            icosphere_ref, coarse_rank, fine_rank
        )

        self.coarse_rank = coarse_rank
        self.fine_rank = fine_rank
        self.coarse_vertex_count = len(coarse_mesh.vertices)
        self.fine_vertex_count = len(fine_mesh.vertices)
        self.coarse_face_count = len(coarse_mesh.faces)
        self.fine_face_count = len(fine_mesh.faces)

        coarse_vertex_faces = np.asarray(coarse_mesh.vertex_faces, dtype=np.int64)
        fine_vertex_faces = np.asarray(fine_mesh.vertex_faces, dtype=np.int64)
        self.register_buffer(
            "coarse_face_vertices",
            torch.from_numpy(np.asarray(coarse_mesh.faces, dtype=np.int64)),
        )
        self.register_buffer(
            "fine_face_vertices",
            torch.from_numpy(np.asarray(fine_mesh.faces, dtype=np.int64)),
        )
        self.register_buffer(
            "fine_face_to_coarse_face", torch.from_numpy(fine_to_coarse)
        )
        self.register_buffer(
            "coarse_face_to_fine_faces", torch.from_numpy(coarse_to_fine)
        )
        self.register_buffer(
            "coarse_vertex_to_incident_faces",
            torch.from_numpy(np.maximum(coarse_vertex_faces, 0)),
        )
        self.register_buffer(
            "coarse_vertex_incident_mask",
            torch.from_numpy(coarse_vertex_faces >= 0),
        )
        self.register_buffer(
            "fine_vertex_to_incident_faces",
            torch.from_numpy(np.maximum(fine_vertex_faces, 0)),
        )
        self.register_buffer(
            "fine_vertex_incident_mask",
            torch.from_numpy(fine_vertex_faces >= 0),
        )
        self.register_buffer(
            "coarse_vertex_face_count",
            torch.from_numpy((coarse_vertex_faces >= 0).sum(axis=1)).float(),
        )
        self.register_buffer(
            "fine_vertex_face_count",
            torch.from_numpy((fine_vertex_faces >= 0).sum(axis=1)).float(),
        )
        self.register_buffer(
            "coarse_face_areas",
            torch.from_numpy(
                spherical_triangle_areas(coarse_mesh.vertices, coarse_mesh.faces)
            ).float(),
        )
        self.register_buffer(
            "fine_face_areas",
            torch.from_numpy(
                spherical_triangle_areas(fine_mesh.vertices, fine_mesh.faces)
            ).float(),
        )
        self.register_buffer(
            "coarse_vertex_indices",
            torch.arange(self.coarse_vertex_count, dtype=torch.long),
        )

    def center_downsample_vertices(self, features: torch.Tensor) -> torch.Tensor:
        """Restrict fine vertex samples to the retained coarse vertices."""
        if features.shape[1] != self.fine_vertex_count:
            raise ValueError("Unexpected fine vertex count")
        return torch.index_select(features, 1, self.coarse_vertex_indices)

    def vertex_feature_stats(
            self,
            features: torch.Tensor,
            level: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean and max of the three incident vertex features per face."""
        faces = (
            self.coarse_face_vertices if level == "coarse"
            else self.fine_face_vertices
        )
        expected_vertices = (
            self.coarse_vertex_count if level == "coarse"
            else self.fine_vertex_count
        )
        if features.ndim != 3 or features.shape[1] != expected_vertices:
            raise ValueError("Expected vertex features shaped [N, V, C]")
        first = torch.index_select(features, 1, faces[:, 0])
        second = torch.index_select(features, 1, faces[:, 1])
        third = torch.index_select(features, 1, faces[:, 2])
        mean = (first + second + third) / 3.0
        maximum = torch.maximum(first, torch.maximum(second, third))
        return mean, maximum

    def vertex_values_to_faces(
            self,
            values: torch.Tensor,
            level: str,
    ) -> torch.Tensor:
        """Average scalar vertex values onto faces; vertex dimension is last."""
        faces = (
            self.coarse_face_vertices if level == "coarse"
            else self.fine_face_vertices
        )
        expected_vertices = (
            self.coarse_vertex_count if level == "coarse"
            else self.fine_vertex_count
        )
        if values.shape[-1] != expected_vertices:
            raise ValueError("Unexpected vertex value count")
        return (
            torch.index_select(values, -1, faces[:, 0])
            + torch.index_select(values, -1, faces[:, 1])
            + torch.index_select(values, -1, faces[:, 2])
        ) / 3.0

    def fine_face_feature_stats(
            self,
            fine_face_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool descendant fine face features into each coarse face."""
        if fine_face_features.ndim != 3:
            raise ValueError("Expected face features shaped [N, F, C]")
        descendants = fine_face_features[:, self.coarse_face_to_fine_faces, :]
        return descendants.mean(dim=2), descendants.amax(dim=2)

    def aggregate_fine_face_values(
            self,
            fine_face_values: torch.Tensor,
            area_weighted: bool = True,
    ) -> torch.Tensor:
        """Aggregate scalar fine face values to coarse faces."""
        if fine_face_values.shape[-1] != self.fine_face_count:
            raise ValueError("Unexpected fine face value count")
        descendants = torch.index_select(
            fine_face_values, -1, self.coarse_face_to_fine_faces.reshape(-1)
        ).reshape(
            *fine_face_values.shape[:-1],
            self.coarse_face_count,
            self.coarse_face_to_fine_faces.shape[1],
        )
        if not area_weighted:
            return descendants.mean(dim=-1)
        areas = self.fine_face_areas[self.coarse_face_to_fine_faces]
        weights = areas / areas.sum(dim=-1, keepdim=True)
        return (descendants * weights).sum(dim=-1)

    def propagate_coarse_face_values(
            self,
            coarse_face_values: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate coarse face values to all exact descendant fine faces."""
        if coarse_face_values.shape[-1] != self.coarse_face_count:
            raise ValueError("Unexpected coarse face value count")
        return torch.index_select(
            coarse_face_values, -1, self.fine_face_to_coarse_face
        )

    def face_features_to_vertices(
            self,
            face_features: torch.Tensor,
            level: str,
    ) -> torch.Tensor:
        """Mean incident face features at every vertex."""
        faces = (
            self.coarse_face_vertices if level == "coarse"
            else self.fine_face_vertices
        )
        vertex_count = (
            self.coarse_vertex_count if level == "coarse"
            else self.fine_vertex_count
        )
        if face_features.ndim != 3 or face_features.shape[1] != len(faces):
            raise ValueError("Expected face features shaped [N, F, C]")
        output = face_features.new_zeros(
            (face_features.shape[0], vertex_count, face_features.shape[2])
        )
        for corner in range(3):
            index = faces[:, corner].reshape(1, -1, 1).expand_as(face_features)
            output.scatter_add_(1, index, face_features)
        counts = (
            self.coarse_vertex_face_count if level == "coarse"
            else self.fine_vertex_face_count
        ).to(dtype=face_features.dtype)
        return output / counts.reshape(1, -1, 1)

    def fine_face_values_to_vertices(
            self,
            fine_face_values: torch.Tensor,
    ) -> torch.Tensor:
        """Mean all incident fine face gate values at every fine vertex."""
        if fine_face_values.shape[-1] != self.fine_face_count:
            raise ValueError("Unexpected fine face value count")
        flat = fine_face_values.reshape(-1, self.fine_face_count)
        output = flat.new_zeros((flat.shape[0], self.fine_vertex_count))
        for corner in range(3):
            index = self.fine_face_vertices[:, corner].reshape(1, -1)
            output.scatter_add_(1, index.expand(flat.shape[0], -1), flat)
        counts = self.fine_vertex_face_count.to(dtype=flat.dtype)
        output = output / counts.reshape(1, -1)
        return output.reshape(*fine_face_values.shape[:-1], self.fine_vertex_count)

    def area_weighted_fine_face_ratio(
            self,
            fine_face_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Return refined spherical area ratio for every leading sample."""
        if fine_face_gate.shape[-1] != self.fine_face_count:
            raise ValueError("Unexpected fine face gate count")
        areas = self.fine_face_areas.to(dtype=fine_face_gate.dtype)
        return (fine_face_gate * areas).sum(dim=-1) / areas.sum()
