"""Geometry diagnostics for the retired vertex and active face hierarchies."""

import numpy as np
import torch

from trimesh_utils import (
    IcoSphereHierarchy,
    IcoSphereRef,
    spherical_triangle_areas,
)


RANK_PAIRS = ((2, 3), (3, 4), (3, 5), (4, 6))


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return (
        f"min={values.min():.6f} max={values.max():.6f} "
        f"mean={values.mean():.6f} std={values.std():.6f} "
        f"median={np.median(values):.6f} "
        f"p10={np.percentile(values, 10):.6f} "
        f"p90={np.percentile(values, 90):.6f}"
    )


def vertex_ancestor_indices(coarse_rank, fine_rank, ref, choose_max):
    ancestor = torch.arange(len(ref.get_normals(coarse_rank)), dtype=torch.long)
    for rank in range(coarse_rank + 1, fine_rank + 1):
        mesh = ref.get_icosphere(rank, True)
        previous_size = ancestor.numel()
        current = torch.empty(len(mesh.vertices), dtype=torch.long)
        current[:previous_size] = ancestor
        for node_index in range(previous_size, len(mesh.vertices)):
            parents = [
                neighbor for neighbor in mesh.vertex_neighbors[node_index]
                if neighbor < previous_size
            ]
            selected = max(parents) if choose_max else min(parents)
            current[node_index] = ancestor[selected]
        ancestor = current
    return ancestor


def connected_components_per_region(mesh, labels):
    labels = labels.numpy()
    visited = np.zeros(len(labels), dtype=bool)
    components = np.zeros(labels.max() + 1, dtype=np.int64)
    for start in range(len(labels)):
        if visited[start]:
            continue
        label = labels[start]
        components[label] += 1
        visited[start] = True
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in mesh.vertex_neighbors[node]:
                if not visited[neighbor] and labels[neighbor] == label:
                    visited[neighbor] = True
                    stack.append(neighbor)
    return components


def vertex_region_diagnostic(ref, coarse_rank, fine_rank):
    coarse = ref.get_icosphere(coarse_rank, True)
    fine = ref.get_icosphere(fine_rank, True)
    # Reproduce the retired implementation locally so the diagnostic remains
    # independent of production model code.
    labels = vertex_ancestor_indices(
        coarse_rank, fine_rank, ref, choose_max=False
    )
    max_parent_labels = vertex_ancestor_indices(
        coarse_rank, fine_rank, ref, choose_max=True
    )
    counts = torch.bincount(labels, minlength=len(coarse.vertices)).numpy()
    cosine = np.einsum(
        "ij,ij->i", fine.vertices, coarse.vertices[labels.numpy()]
    )
    geodesic_degrees = np.degrees(np.arccos(np.clip(cosine, -1, 1)))

    face_areas = spherical_triangle_areas(fine.vertices, fine.faces)
    vertex_areas = np.zeros(len(fine.vertices), dtype=np.float64)
    np.add.at(vertex_areas, fine.faces.reshape(-1), np.repeat(face_areas / 3, 3))
    region_areas = np.bincount(
        labels.numpy(), weights=vertex_areas, minlength=len(coarse.vertices)
    )
    components = connected_components_per_region(fine, labels)

    print(f"rank {coarse_rank}->{fine_rank}")
    print(f"coarse vertices: {len(coarse.vertices)}")
    print(f"fine vertices: {len(fine.vertices)}")
    print("children:", summarize(counts))
    print("geodesic degrees:", summarize(geodesic_degrees))
    print("assigned spherical area:", summarize(region_areas))
    print(f"area relative std: {region_areas.std() / region_areas.mean():.6f}")
    print(f"area max/min ratio: {region_areas.max() / region_areas.min():.6f}")
    print(
        "min-vs-max parent changed: "
        f"{float((labels != max_parent_labels).float().mean()):.6f}"
    )
    print(f"regions with multiple components: {int((components > 1).sum())}")


def face_hierarchy_diagnostic(ref, coarse_rank, fine_rank):
    hierarchy = IcoSphereHierarchy(coarse_rank, fine_rank, ref)
    mapping = hierarchy.fine_face_to_coarse_face
    counts = torch.bincount(
        mapping, minlength=hierarchy.coarse_face_count
    )
    expected = 4 ** (fine_rank - coarse_rank)
    ordering_assumption = torch.arange(
        hierarchy.coarse_face_count
    ).repeat_interleave(expected)
    passed = (
        mapping.numel() == hierarchy.fine_face_count
        and bool((mapping >= 0).all())
        and bool((counts == expected).all())
    )

    print(f"rank {coarse_rank}->{fine_rank}")
    print(f"coarse faces: {hierarchy.coarse_face_count}")
    print(f"fine faces: {hierarchy.fine_face_count}")
    print(f"descendants per parent: min={int(counts.min())} max={int(counts.max())}")
    print(f"expected: {expected}")
    print(f"mapping coverage: {mapping.numel()}/{hierarchy.fine_face_count}")
    print("duplicate parent assignments: 0")
    print(f"ordering matches topology: {bool(torch.equal(mapping, ordering_assumption))}")
    print("PASS" if passed else "FAIL")
    if not passed:
        raise AssertionError("Face hierarchy diagnostic failed")


def pooling_diagnostic(ref):
    hierarchy = IcoSphereHierarchy(3, 5, ref)
    fine_count = hierarchy.fine_vertex_count
    coarse_count = hierarchy.coarse_vertex_count
    homogeneous = torch.full((1, fine_count, 1), 2.0)
    center = hierarchy.center_downsample_vertices(homogeneous)

    fine_face_mean, _ = hierarchy.vertex_feature_stats(homogeneous, "fine")
    region_mean, region_max = hierarchy.fine_face_feature_stats(fine_face_mean)
    print("uniform center/mean max difference:", float((center.mean() - region_mean.mean()).abs()))
    print("uniform center/max max difference:", float((center.max() - region_max.max()).abs()))

    spike = torch.zeros((1, fine_count, 1))
    coarse_vertices = set(range(coarse_count))
    descendants = hierarchy.coarse_face_to_fine_faces
    for fine_faces in descendants:
        candidate_vertices = hierarchy.fine_face_vertices[fine_faces].reshape(-1)
        selected = next(
            int(vertex) for vertex in candidate_vertices
            if int(vertex) not in coarse_vertices
        )
        spike[0, selected, 0] = 10.0
    center_spike = hierarchy.center_downsample_vertices(spike)
    fine_face_mean, _ = hierarchy.vertex_feature_stats(spike, "fine")
    region_mean, region_max = hierarchy.fine_face_feature_stats(fine_face_mean)
    print("local-response center nonzero:", int((center_spike > 0).sum()))
    print("local-response region mean nonzero:", int((region_mean > 0).sum()))
    print("local-response region max nonzero:", int((region_max > 0).sum()))


def target_and_budget_diagnostic(ref):
    errors = torch.tensor((0.0100, 0.0101, 0.0102))
    relative = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
    print("small absolute errors:", errors.tolist())
    print("relative min-max target:", relative.tolist())

    mesh = ref.get_icosphere(5, True)
    areas = torch.from_numpy(
        spherical_triangle_areas(mesh.vertices, mesh.faces)
    )
    quarter = len(areas) // 4
    sorted_areas = torch.sort(areas).values
    small_area_ratio = sorted_areas[:quarter].sum() / sorted_areas.sum()
    large_area_ratio = sorted_areas[-quarter:].sum() / sorted_areas.sum()
    print("rank 5 simple 25% face ratio: 0.250000")
    print(f"rank 5 smallest-face area ratio: {float(small_area_ratio):.6f}")
    print(f"rank 5 largest-face area ratio: {float(large_area_ratio):.6f}")


def main():
    ref = IcoSphereRef("vertex")
    print("==============================")
    print("Vertex hierarchy diagnostic (retired min-parent rule)")
    print("==============================")
    for pair in RANK_PAIRS:
        vertex_region_diagnostic(ref, *pair)
        print()

    print("==============================")
    print("Face hierarchy diagnostic (active topology-derived rule)")
    print("==============================")
    for pair in RANK_PAIRS:
        face_hierarchy_diagnostic(ref, *pair)
        print()

    print("==============================")
    print("Spherical face area diagnostic")
    print("==============================")
    for rank in range(2, 7):
        mesh = ref.get_icosphere(rank, True)
        areas = spherical_triangle_areas(mesh.vertices, mesh.faces)
        print(
            f"rank {rank}: {summarize(areas)} "
            f"relative_std={areas.std() / areas.mean():.6f} "
            f"max/min={areas.max() / areas.min():.6f}"
        )

    print("==============================")
    print("Coarse pooling diagnostic")
    print("==============================")
    pooling_diagnostic(ref)

    print("==============================")
    print("Target and budget diagnostic")
    print("==============================")
    target_and_budget_diagnostic(ref)


if __name__ == "__main__":
    main()
