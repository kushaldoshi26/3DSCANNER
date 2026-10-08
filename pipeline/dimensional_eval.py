"""
Oriented Bounding Box (OBB) Physical Dimensional Measurement Module

Computes rotation-invariant physical dimensions (Length, Width, Height), Volume,
Surface Area, and Aspect Ratio of 3D meshes using Trimesh Oriented Bounding Box algorithms.
Independent of the object's spatial rotation in 3D space.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import trimesh

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def auto_measure_mesh(
    mesh_input: Union[str, Path, trimesh.Trimesh],
    aruco_reference_mm: Optional[float] = 50.0,
    aruco_measured_length: Optional[float] = None
) -> Dict[str, Any]:
    """
    Loads a 3D mesh and calculates its exact physical Length, Width, Height,
    and Volume using an Oriented Bounding Box (OBB).

    Args:
        mesh_input: Path to mesh file (.stl, .obj, .ply, .glb) or trimesh.Trimesh object
        aruco_reference_mm: The known physical length of the reference marker in mm (default 50.0)
        aruco_measured_length: The measured length of the marker in the 3D space to compute scale. If None, scale is 1.0.

    Returns:
        Dictionary containing OBB physical metrics and dimensions.
    """
    if isinstance(mesh_input, (str, Path)):
        p = Path(mesh_input)
        if not p.exists():
            raise FileNotFoundError(f"3D Mesh file not found: {p}")
        mesh_name = p.name
        mesh = trimesh.load(str(p), force='mesh')
    else:
        mesh_name = "in_memory_mesh"
        mesh = mesh_input

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    if len(mesh.vertices) == 0:
        raise ValueError("Cannot measure empty mesh with 0 vertices.")

    # 1. Compute exact tightest fitting Oriented Bounding Box (OBB)
    try:
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        raw_extents = np.array(extents, dtype=np.float64)
    except Exception:
        obb = mesh.bounding_box_oriented
        raw_extents = np.array(obb.extents, dtype=np.float64)
    
    # Sort extents so:
    # dimensions[2] = Length  (Longest principal side)
    # dimensions[1] = Width   (Intermediate side)
    # dimensions[0] = Height  (Shortest principal side)
    sorted_extents = np.sort(raw_extents)
    
    length_raw = float(sorted_extents[2])
    width_raw = float(sorted_extents[1])
    height_raw = float(sorted_extents[0])

    if aruco_measured_length is not None and aruco_measured_length > 0:
        unit_scale = aruco_reference_mm / aruco_measured_length
    else:
        unit_scale = 1.0

    # Convert to scaled millimeter units
    length_mm = round(length_raw * unit_scale, 2)
    width_mm = round(width_raw * unit_scale, 2)
    height_mm = round(height_raw * unit_scale, 2)

    # 2. Standard Axis-Aligned Bounding Box (AABB) for comparison
    aabb_extents = np.sort(mesh.extents)
    aabb_length_mm = round(float(aabb_extents[2]) * unit_scale, 2)
    aabb_width_mm = round(float(aabb_extents[1]) * unit_scale, 2)
    aabb_height_mm = round(float(aabb_extents[0]) * unit_scale, 2)

    # 3. Volume and Surface Area
    obb_volume_mm3 = round(float(length_mm * width_mm * height_mm), 2)
    
    mesh_volume_exact = bool(mesh.is_watertight)
    try:
        raw_vol = abs(float(mesh.volume)) * (unit_scale ** 3)
        mesh_volume_mm3 = round(raw_vol, 2)
    except Exception:
        mesh_volume_mm3 = obb_volume_mm3 * 0.55  # Approximation for non-watertight

    surface_area_mm2 = round(float(mesh.area) * (unit_scale ** 2), 2)

    # 4. Aspect Ratios & Inertia Principal Moments
    aspect_ratio_lw = round(length_mm / max(width_mm, 1e-4), 2)
    aspect_ratio_lh = round(length_mm / max(height_mm, 1e-4), 2)

    return {
        "filename": mesh_name,
        "length_mm": length_mm,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "dimensions_obb": {
            "length": length_mm,
            "width": width_mm,
            "height": height_mm
        },
        "dimensions_aabb": {
            "length": aabb_length_mm,
            "width": aabb_width_mm,
            "height": aabb_height_mm
        },
        "obb_volume_mm3": obb_volume_mm3,
        "mesh_volume_mm3": mesh_volume_mm3,
        "is_volume_exact": mesh_volume_exact,
        "surface_area_mm2": surface_area_mm2,
        "aspect_ratios": {
            "length_to_width": aspect_ratio_lw,
            "length_to_height": aspect_ratio_lh
        },
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "is_watertight": bool(mesh.is_watertight)
    }


def format_dimensional_report(report: Dict[str, Any]) -> str:
    """Formats OBB measurement dictionary into a clean CLI report."""
    lines = [
        "=" * 60,
        f"  3D AUTOMATED DIMENSIONAL & OBB MEASUREMENT REPORT",
        "=" * 60,
        f"Model: {report['filename']}",
        f"Vertices: {report['vertices']:,} | Faces: {report['faces']:,} | Watertight: {report['is_watertight']}",
        "-" * 60,
        f"  * Length (Longest side):  {report['length_mm']:>8.2f} mm",
        f"  * Width  (Middle side):   {report['width_mm']:>8.2f} mm",
        f"  * Height (Shortest side): {report['height_mm']:>8.2f} mm",
        "-" * 60,
        f"  * Exact Mesh Volume:      {report['mesh_volume_mm3']:>8.2f} mm3 {'(Exact)' if report['is_volume_exact'] else '(Estimated)'}",
        f"  * OBB Bounding Volume:    {report['obb_volume_mm3']:>8.2f} mm3",
        f"  * Total Surface Area:     {report['surface_area_mm2']:>8.2f} mm2",
        f"  * Aspect Ratio (L:W:H):   {report['aspect_ratios']['length_to_width']:.2f} : 1.00 : {report['height_mm']/max(report['width_mm'], 1e-4):.2f}",
        "=" * 60
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) > 1 else "data/output/ai_trained_model.stl"
    res = auto_measure_mesh(test_file)
    print(format_dimensional_report(res))
