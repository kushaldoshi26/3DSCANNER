import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import trimesh

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.dimensional_eval import auto_measure_mesh


def analyze_mesh_topology(mesh_path: str) -> Dict[str, Any]:
    """
    Analyzes 3D mesh topology, manifold health, physical OBB dimensions, and 3D printing suitability.
    """
    p = Path(mesh_path)
    if not p.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mesh = trimesh.load_mesh(str(p))
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    vertices_count = len(mesh.vertices)
    faces_count = len(mesh.faces)

    # Attempt repair before evaluating watertightness
    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_normals(mesh)
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass

    is_watertight = bool(mesh.is_watertight)
    is_winding_consistent = bool(mesh.is_winding_consistent)

    # Compute exact OBB physical measurements
    dim_report = auto_measure_mesh(mesh)

    # Printability scoring
    score = 100
    issues = []

    if not is_watertight:
        score -= 30
        issues.append("Mesh is not watertight (contains open holes or gaps).")

    if not is_winding_consistent:
        score -= 20
        issues.append("Inconsistent face normal orientations.")

    if faces_count > 150000:
        score -= 10
        issues.append("High polygon count (>150k faces). Recommend decimation before slicing.")

    if faces_count < 100:
        score -= 30
        issues.append("Low resolution geometry (<100 faces).")

    score = max(0, min(100, score))

    return {
        "filename": p.name,
        "vertices": vertices_count,
        "faces": faces_count,
        "is_watertight": is_watertight,
        "is_winding_consistent": is_winding_consistent,
        "dimensions_mm": {
            "width": dim_report["width_mm"],
            "height": dim_report["height_mm"],
            "depth": dim_report["length_mm"]
        },
        "dimensions_obb": {
            "length": dim_report["length_mm"],
            "width": dim_report["width_mm"],
            "height": dim_report["height_mm"]
        },
        "length_mm": dim_report["length_mm"],
        "width_mm": dim_report["width_mm"],
        "height_mm": dim_report["height_mm"],
        "obb_volume_mm3": dim_report["obb_volume_mm3"],
        "volume_mm3": dim_report["mesh_volume_mm3"],
        "volume_exact": dim_report["is_volume_exact"],
        "surface_area_mm2": dim_report["surface_area_mm2"],
        "aspect_ratios": dim_report["aspect_ratios"],
        "printability_score": score,
        "issues": issues
    }


def repair_mesh_topology(input_mesh_path: str, output_mesh_path: str) -> str:
    """
    Repairs mesh topology for 3D printing (hole filling, normal fixing, manifold creation).
    """
    in_p = Path(input_mesh_path)
    out_p = Path(output_mesh_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"Repairing mesh topology for 3D printing: {in_p.name}...")
    mesh = trimesh.load_mesh(str(in_p))
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    # Fill holes & fix face orientations
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    mesh.remove_unreferenced_vertices()

    mesh.export(str(out_p))
    print(f"Repaired 3D printing mesh saved to {out_p}")
    return str(out_p)


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "../data/output/ai_trained_model.stl"
    analysis = analyze_mesh_topology(inp)
    print("Mesh Topology & OBB Dimensional Analysis:")
    for k, v in analysis.items():
        print(f"  {k}: {v}")
