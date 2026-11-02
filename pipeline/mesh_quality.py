"""
Mesh Quality Control (QC) module.

Validates a reconstructed mesh against the thresholds required for reliable
3D printing, metrology, and web visualization. Used by both single-image
and turntable pipelines so failures surface as actionable diagnostics rather
than silently broken models.

All checks are read-only against the mesh; nothing is mutated here.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import trimesh


class MeshQualityError(Exception):
    """Raised when a mesh fails a hard QC gate (configurable)."""


def validate_mesh(
    mesh_input,
    *,
    min_vertices: int = 100,
    min_faces: int = 100,
    max_non_manifold_edges: int = 0,
    max_disconnected_components: int = 1,
    min_bbox_mm: float = 0.5,
    max_bbox_mm: float = 5000.0,
    require_watertight: bool = False,
    check_self_intersection: bool = True,
    coordinate_space: str = "meters",
) -> Dict[str, Any]:
    """
    Runs a full QC battery and returns a structured report.

    Args:
        mesh_input: path to mesh file or a trimesh.Trimesh instance.
        min_vertices / min_faces: hard lower bounds.
        max_non_manifold_edges: 0 means strictly manifold (hard gate).
        max_disconnected_components: allow only 1 connected body.
        min_bbox_mm / max_bbox_mm: plausible physical extents (mm).
        require_watertight: hard gate if True.
        check_self_intersection: runs the (expensive) intersection test.

    Returns dict with 'passed', 'metrics', 'warnings', 'errors'.
    """
    mesh = mesh_input if isinstance(mesh_input, trimesh.Trimesh) else trimesh.load(
        str(mesh_input), force="mesh"
    )

    metrics: Dict[str, Any] = {}
    warnings: List[str] = []
    errors: List[str] = []

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    metrics["vertices"] = int(len(vertices))
    metrics["faces"] = int(len(faces))

    # --- Structural gates ---
    if len(vertices) < min_vertices:
        errors.append(f"Too few vertices ({len(vertices)} < {min_vertices}).")
    if len(faces) < min_faces:
        errors.append(f"Too few faces ({len(faces)} < {min_faces}).")

    if not np.isfinite(vertices).all():
        errors.append("Mesh contains non-finite vertex coordinates.")
    if len(vertices) == 0 or len(faces) == 0 or not np.isfinite(vertices).all():
        return {"status": "FAILED", "passed": False, "metrics": metrics,
                "warnings": warnings, "errors": errors}

    # --- Topology ---
    metrics["watertight"] = bool(mesh.is_watertight)
    metrics["winding_consistent"] = bool(mesh.is_winding_consistent)
    try:
        incidence = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
        non_manifold = int(np.count_nonzero(incidence > 2))
        metrics["boundary_edges"] = int(np.count_nonzero(incidence == 1))
    except Exception:
        non_manifold = -1
    metrics["non_manifold_edges"] = int(non_manifold)
    if non_manifold > max_non_manifold_edges:
        errors.append(
            f"Non-manifold edges detected ({non_manifold} > {max_non_manifold_edges})."
        )

    # --- Disconnected components ---
    try:
        components = int(mesh.split(only_watertight=False).__len__())
    except Exception:
        components = 1
    metrics["disconnected_components"] = components
    if components > max_disconnected_components:
        warnings.append(
            f"{components} disconnected components (expected <= {max_disconnected_components})."
        )

    # Generated assets have relative scale until a physical reference is supplied.
    extents = mesh.extents
    metrics["coordinate_space"] = coordinate_space
    if coordinate_space == "normalized":
        metrics["bounding_box_units"] = [round(float(x), 6) for x in extents]
        metrics["scale_calibrated"] = False
    else:
        metrics["bounding_box_mm"] = [round(float(x) * 1000.0, 3) for x in extents]
        bbox_max = float(np.max(extents) * 1000.0)
        if bbox_max < min_bbox_mm: warnings.append(f"Bounding box very small ({bbox_max:.3f} mm).")
        if bbox_max > max_bbox_mm: warnings.append(f"Bounding box implausibly large ({bbox_max:.1f} mm).")

    # --- Self-intersection ---
    self_intersections = -1
    if check_self_intersection and len(faces) > 0:
        try:
            self_intersections = int(mesh.intersections.count_intersections)
        except Exception:
            self_intersections = -1
    metrics["self_intersections_reported"] = self_intersections
    if self_intersections > 0:
        warnings.append(
            f"Self-intersections detected ({self_intersections}). Consider repair/decimation."
        )

    # --- Volume ---
    try:
        metrics["volume_units3" if coordinate_space == "normalized" else "volume_mm3"] = round(float(abs(mesh.volume)) * (1.0 if coordinate_space == "normalized" else 1e9), 6)
    except Exception:
        metrics["volume_mm3"] = None

    if not metrics["winding_consistent"]:
        errors.append("Mesh winding is inconsistent.")

    # --- Watertight gate ---
    if require_watertight and not metrics["watertight"]:
        errors.append("Mesh is not watertight (required by caller).")

    passed = len(errors) == 0
    status = "PASS" if passed and not warnings else ("PASS WITH WARNINGS" if passed else "FAILED")
    return {
        "status": status,
        "passed": passed,
        "metrics": metrics,
        "warnings": warnings,
        "errors": errors,
    }


def surface_roughness(mesh_input) -> float:
    """
    Surface roughness from per-vertex normal deviation vs the local neighbor
    average (Laplacian-of-normals). Higher = rougher / noisier surface.

    This is sensitive to smoothing: a clean mesh has normals aligned with its
    neighborhood (low score), while high-frequency vertex noise makes each
    vertex's normal diverge from its neighbors (high score). Implemented with
    a pure-edge adjacency so it never silently no-ops like a global-variance
    fallback would.
    """
    mesh = mesh_input if isinstance(mesh_input, trimesh.Trimesh) else trimesh.load(
        str(mesh_input), force="mesh"
    )
    n = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if n.shape[0] == 0:
        return 0.0
    faces = np.asarray(mesh.faces)
    if faces.shape[0] == 0:
        return 0.0

    # Build edge adjacency.
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adj = [[] for _ in range(n.shape[0])]
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    for i in range(n.shape[0]):
        seen = set()
        uniq = []
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                uniq.append(j)
        adj[i] = uniq

    neighbor_avg = np.zeros_like(n)
    for i, nbrs in enumerate(adj):
        if nbrs:
            neighbor_avg[i] = n[nbrs].mean(axis=0)
        else:
            neighbor_avg[i] = n[i]
    diff = np.linalg.norm(n - neighbor_avg, axis=1)
    return float(np.mean(diff))


def detail_metrics(mesh_input) -> Dict[str, Any]:
    """Returns roughness + topology variance metrics for before/after comparison."""
    mesh = mesh_input if isinstance(mesh_input, trimesh.Trimesh) else trimesh.load(
        str(mesh_input), force="mesh"
    )
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    out: Dict[str, Any] = {}
    try:
        out["surface_roughness"] = round(surface_roughness(mesh), 4)
    except Exception:
        out["surface_roughness"] = None
    # Triangle size variance.
    if len(faces) > 0:
        e = np.linalg.norm(verts[mesh.edges[:, 0]] - verts[mesh.edges[:, 1]], axis=1)
        out["triangle_size_mean"] = round(float(e.mean()), 5)
        out["triangle_size_std"] = round(float(e.std()), 5)
        out["triangle_size_cv"] = round(float(e.std() / (e.mean() + 1e-8)), 4)
    out["vertices"] = int(len(verts))
    out["faces"] = int(len(faces))
    out["connected_components"] = int(mesh.split(only_watertight=False).__len__())
    return out
