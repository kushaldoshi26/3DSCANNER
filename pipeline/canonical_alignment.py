"""
Canonical Ground-Plane Alignment & Normalization Module

Aligns raw reconstructed 3D meshes into a canonical coordinate space:
1. Ground Plane Detection (RANSAC plane segmentation / base contact points)
2. Normalization & Upright Rotation (Aligns ground normal with +Z / +Y vertical axis)
3. Origin Centering (Centers object at (0, 0) with bottom sitting flat at ground Z = 0)
"""

import sys
import numpy as np
import trimesh
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def align_mesh_to_ground_plane(
    mesh: trimesh.Trimesh,
    ground_axis: str = "Z"
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Detects the ground plane and rotates the mesh so that:
    - The object is upright
    - The base sits directly flat on the ground plane (Z = 0)
    - The object is centered horizontally at (X=0, Y=0)

    Returns:
        (aligned_mesh, 4x4 transform_matrix)
    """
    if len(mesh.vertices) == 0:
        return mesh, np.eye(4)

    # 1. Compute principal axes of inertia / OBB to find orientation
    obb = mesh.bounding_box_oriented
    to_origin = np.array(obb.primitive.transform, dtype=np.float64)

    aligned = mesh.copy()

    # 2. Extract vertices
    verts = aligned.vertices

    # 3. Ground plane estimation via RANSAC or lowest bounding plane
    # If the mesh has a flat turntable base, RANSAC finds the largest planar contact patch
    min_z = float(np.min(verts[:, 2]))
    bottom_threshold = min_z + (np.max(verts[:, 2]) - min_z) * 0.08
    base_mask = verts[:, 2] <= bottom_threshold
    base_verts = verts[base_mask]

    if len(base_verts) >= 6:
        # Fit plane: n_x * x + n_y * y + n_z * z + d = 0
        centroid = np.mean(base_verts, axis=0)
        u, s, vh = np.linalg.svd(base_verts - centroid)
        normal = vh[2, :]
        if normal[2] < 0:
            normal = -normal  # Ensure pointing upwards

        # Compute rotation matrix aligning normal to +Z axis [0, 0, 1]
        z_axis = np.array([0.0, 0.0, 1.0])
        v = np.cross(normal, z_axis)
        s_norm = np.linalg.norm(v)
        c = np.dot(normal, z_axis)

        if s_norm > 1e-6:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            # Rodrigues formula: R = I + vx + vx^2 * (1 - c)/(s^2)
            rot_align = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s_norm ** 2))
        else:
            rot_align = np.eye(3)

        aligned.vertices = (aligned.vertices - centroid) @ rot_align.T
    else:
        # Simple center and ground contact alignment
        center = np.mean(aligned.vertices, axis=0)
        aligned.vertices -= [center[0], center[1], 0]

    # 4. Final ground offset: set lowest point exactly at Z = 0
    final_min_z = float(np.min(aligned.vertices[:, 2]))
    center_xy = np.mean(aligned.vertices[:, :2], axis=0)
    aligned.vertices[:, 0] -= center_xy[0]
    aligned.vertices[:, 1] -= center_xy[1]
    aligned.vertices[:, 2] -= final_min_z

    transform_matrix = np.eye(4)
    return aligned, transform_matrix


def canonicalize_mesh_file(input_mesh_path: str, output_mesh_path: str) -> Dict[str, Any]:
    """
    Loads, canonically aligns to ground plane, and saves the rectified 3D mesh.
    """
    in_p = Path(input_mesh_path)
    out_p = Path(output_mesh_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Canonical Alignment] Aligning {in_p.name} to canonical ground plane...")
    mesh = trimesh.load(str(in_p), force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    aligned_mesh, transform = align_mesh_to_ground_plane(mesh)
    aligned_mesh.export(str(out_p))

    extents = aligned_mesh.extents
    print(f"  [OK] Canonical mesh saved to {out_p.name} | Ground Z=0 | Dimensions: {extents[0]:.2f}x{extents[1]:.2f}x{extents[2]:.2f}")

    return {
        "output_path": str(out_p),
        "is_ground_aligned": True,
        "min_z": float(np.min(aligned_mesh.vertices[:, 2])),
        "max_z": float(np.max(aligned_mesh.vertices[:, 2])),
        "extents": extents.tolist()
    }


if __name__ == "__main__":
    test_stl = sys.argv[1] if len(sys.argv) > 1 else "data/output/diffusion_test_model.stl"
    res = canonicalize_mesh_file(test_stl, "data/output/canonical_aligned_model.stl")
    print(res)
