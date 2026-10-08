"""
Procedural CAD/Hard-Surface Dataset Generator

Synthesizes thousands of non-organic, mechanical 3D parts (flanges, brackets, enclosures, gears)
using Constructive Solid Geometry (CSG) boolean operations.
This is used to fine-tune the 3D Reconstruction Network to natively predict sharp edges and flat planes.
"""

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trimesh
import numpy as np
from trimesh.creation import box, cylinder, annulus

# Require manifold/watertight geometry for boolean ops
trimesh.tol.merge = 1e-6 

DATASETS_DIR = Path("d:/3dSCANNER/data/datasets/cad_training")

def _save_mesh(mesh: trimesh.Trimesh, name: str):
    """Normalize mesh and save to dataset."""
    if not mesh.is_watertight:
        # We attempt to repair topology
        mesh.fill_holes()
    
    if len(mesh.vertices) == 0:
        return
        
    mesh.vertices -= mesh.center_mass
    max_ext = np.max(mesh.extents)
    if max_ext > 0:
        mesh.vertices /= max_ext
        
    out_path = DATASETS_DIR / f"{name}.stl"
    mesh.export(str(out_path))
    print(f"[CAD Gen] Saved {name}.stl ({len(mesh.faces)} faces)")

def generate_bracket():
    """Generates an L-bracket with mounting holes."""
    thickness = random.uniform(0.05, 0.2)
    width = random.uniform(0.5, 1.2)
    length1 = random.uniform(0.5, 1.5)
    length2 = random.uniform(0.5, 1.5)
    
    # Base and wall
    base = box(extents=[length1, thickness, width])
    base.apply_translation([length1/2, thickness/2, 0])
    
    wall = box(extents=[thickness, length2, width])
    wall.apply_translation([thickness/2, length2/2, 0])
    
    bracket = base.union(wall, engine='blender' if trimesh.boolean.has_fast_math() else 'scad')
    
    # Drill holes
    hole_radius = random.uniform(0.05, 0.15)
    hole = cylinder(radius=hole_radius, height=thickness*3)
    hole.apply_translation([length1 * 0.75, thickness/2, 0])
    
    try:
        # Subtract hole
        final_mesh = bracket.difference(hole, engine='blender' if trimesh.boolean.has_fast_math() else 'scad')
        return final_mesh
    except Exception:
        return bracket

def generate_enclosure():
    """Generates a hollowed out box/enclosure."""
    outer = box(extents=[random.uniform(0.8, 2.0), random.uniform(0.5, 1.5), random.uniform(0.8, 2.0)])
    thickness = random.uniform(0.05, 0.15)
    
    inner_ext = outer.extents - (thickness * 2)
    inner_ext[1] += thickness * 3  # Punch through top
    inner = box(extents=inner_ext)
    inner.apply_translation([0, thickness, 0])
    
    try:
        final_mesh = outer.difference(inner, engine='blender' if trimesh.boolean.has_fast_math() else 'scad')
        return final_mesh
    except Exception:
        return outer

def generate_flange():
    """Generates a circular pipe flange."""
    outer_rad = random.uniform(0.8, 1.5)
    inner_rad = random.uniform(0.3, outer_rad - 0.2)
    thickness = random.uniform(0.1, 0.4)
    
    flange = annulus(r_min=inner_rad, r_max=outer_rad, height=thickness)
    
    return flange

def generate_cad_dataset(num_samples: int = 50):
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(f"Generating {num_samples} CAD / Mechanical Dataset Meshes")
    print("=" * 60)
    
    generators = [
        ("bracket", generate_bracket),
        ("enclosure", generate_enclosure),
        ("flange", generate_flange)
    ]
    
    for i in range(num_samples):
        cat_name, func = random.choice(generators)
        mesh = func()
        if mesh is not None and not mesh.is_empty:
            _save_mesh(mesh, f"{cat_name}_{i:04d}")
            
    print("CAD Dataset generation complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=50, help="Number of CAD samples to generate")
    args = parser.parse_args()
    
    generate_cad_dataset(args.count)
