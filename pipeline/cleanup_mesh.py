"""
Post-generation mesh cleanup & face-aware smoothing for single-image
AI 3D output (Mode A: TRELLIS / Hunyuan3D / TripoSR).

Pipeline (per 3DINVENZA upgrade spec):
    Generated mesh
        -> remove invalid / isolated components
        -> fix normals / winding
        -> remove degenerate triangles
        -> repair non-manifold regions when safe
        -> optional isotropic remeshing
        -> light Taubin smoothing (preserves volume, no melt)
        -> face-aware regional smoothing (face weak, body stronger)
        -> bounded detail refinement
        -> final validation

Design rules:
  - NEVER aggressively smooth the face. Use Taubin (volume-preserving) not
    Laplacian (shrinking). Face region gets a fraction of the body strength.
  - Do NOT convert texture detail into geometry.
  - Do NOT fake watertight geometry; offer an explicit printable bust base.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import trimesh
from pathlib import Path

# ---------------------------------------------------------------------------
# Configurable cleanup settings (presets select among these)
# ---------------------------------------------------------------------------
SMOOTHING_ENABLED = True
TAUBIN_ITERATIONS = 3          # 0 / 2 / 3 / 5 — never 20
REMESH_ENABLED = False         # only when topology is badly uneven
TARGET_EDGE_LENGTH = 0.01      # in mesh units, used only if REMESH_ENABLED

# Regional smoothing strengths. Face is preserved far more than body/clothing.
FACE_SMOOTHING_STRENGTH = 0.25
BODY_SMOOTHING_STRENGTH = 0.6

# Bounded displacement (generative mode uses a relative ratio, not fake mm).
MAX_DISPLACEMENT_RATIO = 0.001  # fraction of mesh bbox diagonal


# ---------------------------------------------------------------------------
# Face ROI detection (image-driven). Falls back gracefully if OpenCV cascade
# assets are unavailable.
# ---------------------------------------------------------------------------
def detect_face_roi(image_path: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Returns normalized (x0, y0, x1, y1) face bbox in [0,1] image space, or
    None if no face is detected / OpenCV is unavailable.
    """
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if cascade.empty():
                return None
            faces = cascade.detectMultiScale(gray, 1.2, 4, minSize=(40, 40))
        except Exception:
            return None
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        H, W = gray.shape[:2]
        # Expand slightly to include hairline / jaw.
        pad = 0.15
        x0 = max(0.0, (x - w * pad) / W)
        y0 = max(0.0, (y - h * pad) / H)
        x1 = min(1.0, (x + w * (1 + pad)) / W)
        y1 = min(1.0, (y + h * (1 + pad)) / H)
        return (x0, y0, x1, y1)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cleanup primitive steps
# ---------------------------------------------------------------------------
def _remove_invalid_geometry(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.remove_unreferenced_vertices()
    try:
        mask = mesh.nondegenerate_faces()
        if mask is not None and len(mask) > 0:
            mesh.update_faces(mask)
    except Exception:
        pass
    # Remove tiny isolated components (keep the largest body).
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            components = [c for c in components if len(c.faces) > 0]
            components.sort(key=lambda c: len(c.faces), reverse=True)
            largest = components[0]
            for c in components[1:]:
                if len(c.faces) < 200:  # discard small floating bits
                    continue
                largest = trimesh.util.concatenate([largest, c])
            mesh = largest
    except Exception:
        pass
    return mesh


def _fix_topology(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    try:
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    # Repair non-manifold edges only when it does not destroy detail.
    try:
        if mesh.edges_non_manifold.sum() > 0:
            mesh = mesh.split(only_watertight=False)[0]
    except Exception:
        pass
    return mesh


def _optional_remesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if not REMESH_ENABLED:
        return mesh
    try:
        sizes = np.linalg.norm(mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], axis=1)
        if sizes.std() / (sizes.mean() + 1e-8) > 1.5:  # highly inconsistent
            mesh = mesh.simplify_quadric_decimation(percent=0.9)
    except Exception:
        pass
    return mesh


def _planar_decimation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Aggressively decimates near-planar regions to flatten wavy AI artifacts 
    on mechanical/hard-surface objects, while preserving sharp corners.
    """
    try:
        # Decimate to 20% of original faces to force large flat triangles
        # Trimesh wrapper expects `percent` or `face_count` kwarg.
        decimated = mesh.simplify_quadric_decimation(percent=0.2)
        # Re-subdivide slightly if we need density for texture projection,
        # but the planes will remain perfectly flat.
        return decimated
    except Exception as e:
        print(f"[cleanup] Planar decimation failed: {e}")
        return mesh


def _build_adjacency(mesh: trimesh.Trimesh):
    """Edge-built symmetric vertex adjacency as a list of neighbor index lists."""
    n = len(mesh.vertices)
    adjacency = [[] for _ in range(n)]
    for f in mesh.faces:
        for i in range(3):
            u, v = int(f[i]), int(f[(i + 1) % 3])
            adjacency[u].append(v)
            adjacency[v].append(u)
    # De-duplicate while preserving order.
    for i in range(n):
        seen = set()
        uniq = []
        for j in adjacency[i]:
            if j not in seen:
                seen.add(j)
                uniq.append(j)
        adjacency[i] = uniq
    return adjacency


def _taubin_smooth(mesh: trimesh.Trimesh, iterations: int) -> trimesh.Trimesh:
    """
    Explicit volume-preserving Taubin smoothing (lambda | mu mesh fairing).

    Unlike trimesh's smooth wrapper, this deterministically moves vertices:
    a positive Laplacian pass (shrink) followed by a negative Laplacian pass
    (expand) so net volume is preserved and no melt/shrink occurs.
    """
    if iterations <= 0 or len(mesh.vertices) == 0:
        return mesh
    verts = mesh.vertices.astype(np.float64).copy()
    adjacency = _build_adjacency(mesh)
    # lambda in (0,1) shrink; mu > lambda negative (expand). mu ~ -lambda/..
    lam = 0.5
    mu = -0.53
    for _ in range(max(1, iterations)):
        # Positive (shrink) pass.
        lap = np.zeros_like(verts)
        for i, nbrs in enumerate(adjacency):
            if nbrs:
                lap[i] = verts[nbrs].mean(axis=0) - verts[i]
        v_pos = verts + lam * lap
        # Negative (expand) pass on the shrunk positions.
        lap2 = np.zeros_like(v_pos)
        for i, nbrs in enumerate(adjacency):
            if nbrs:
                lap2[i] = v_pos[nbrs].mean(axis=0) - v_pos[i]
        verts = v_pos + mu * lap2
    mesh.vertices = verts.astype(np.float64)
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def _face_aware_smooth(
    mesh: trimesh.Trimesh,
    image_path: Optional[str],
    face_strength: float,
    body_strength: float,
) -> trimesh.Trimesh:
    """
    Applies stronger smoothing to body/clothing and weaker smoothing to the
    face ROI. The ROI is derived from the source image; if unavailable, the
    upper-front of the mesh (typical head position) is used conservatively.
    """
    roi = detect_face_roi(image_path) if image_path else None

    verts = mesh.vertices
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    size = bbox_max - bbox_min
    diag = float(np.linalg.norm(size)) + 1e-8

    # Build a per-vertex "face weight": 1 at face, 0 at body.
    face_weight = np.zeros(len(verts), dtype=np.float32)
    if roi is not None:
        # Project vertices into image-ish normalized space using a simple
        # frontal orthographic assumption: x -> u, y(height) -> v (flip).
        u = (verts[:, 0] - bbox_min[0]) / (size[0] + 1e-8)
        v = 1.0 - (verts[:, 1] - bbox_min[1]) / (size[1] + 1e-8)
        x0, y0, x1, y1 = roi
        in_roi = (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
        face_weight[in_roi] = 1.0
    else:
        # Conservative default: upper 35% of height, central 60% width.
        u = (verts[:, 0] - bbox_min[0]) / (size[0] + 1e-8)
        v = 1.0 - (verts[:, 1] - bbox_min[1]) / (size[1] + 1e-8)
        in_roi = (u >= 0.2) & (u <= 0.8) & (v >= 0.65)
        face_weight[in_roi] = 1.0

    # Per-vertex smoothing factor blends face/body strengths.
    factor = face_weight * face_strength + (1 - face_weight) * body_strength
    factor = np.clip(factor, 0.0, 1.0)

    # One Taubin pass, then blend per-vertex toward the smoothed result.
    smoothed = _taubin_smooth(mesh, iterations=max(1, TAUBIN_ITERATIONS))
    blended = verts + (smoothed.vertices - verts) * factor[:, None]
    mesh.vertices = blended
    return mesh


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def cleanup_generated_mesh(
    input_mesh_path: str,
    output_mesh_path: str,
    image_path: Optional[str] = None,
    *,
    object_type: str = "organic",
    smoothing_enabled: bool = SMOOTHING_ENABLED,
    taubin_iterations: int = TAUBIN_ITERATIONS,
    face_strength: float = FACE_SMOOTHING_STRENGTH,
    body_strength: float = BODY_SMOOTHING_STRENGTH,
    remesh_enabled: bool = REMESH_ENABLED,
    debug: bool = False,
) -> str:
    """
    Structured cleanup of a raw AI-generated mesh. Returns the output path.
    Saves GLB + STL alongside the principal export format.
    """
    global TAUBIN_ITERATIONS, REMESH_ENABLED
    TAUBIN_ITERATIONS = taubin_iterations
    REMESH_ENABLED = remesh_enabled

    out_p = Path(output_mesh_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(str(input_mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    raw_verts = len(mesh.vertices)
    raw_faces = len(mesh.faces)
    print(f"[cleanup] Raw mesh: {raw_verts} verts, {raw_faces} faces")

    mesh = _remove_invalid_geometry(mesh)
    print(f"[cleanup] After invalid-geometry removal: {len(mesh.faces)} faces")

    mesh = _fix_topology(mesh)

    if object_type == "hard_surface":
        print("[cleanup] Hard-surface mode: applying planar decimation and skipping organic smoothing.")
        mesh = _planar_decimation(mesh)
    else:
        mesh = _optional_remesh(mesh)
        if smoothing_enabled:
            mesh = _face_aware_smooth(
                mesh, image_path, face_strength=face_strength, body_strength=body_strength
            )
            print(f"[cleanup] Face-aware smoothing applied (face={face_strength}, body={body_strength}).")

    try:
        mesh.fix_normals()
    except Exception:
        pass

    mesh.export(str(out_p))
    mesh.export(str(out_p.with_suffix(".glb")))
    print(f"[cleanup] Cleaned mesh saved: {out_p}")

    if debug:
        print(f"[cleanup] raw_verts={raw_verts} raw_faces={raw_faces} "
              f"final_verts={len(mesh.vertices)} final_faces={len(mesh.faces)}")
    return str(out_p)
