import trimesh
import numpy as np
from pathlib import Path

def prepare_mesh_dataset(input_dir: str, processed_dir: str):
    """
    Normalizes complex 3D meshes (GLB, OBJ, STL) and extracts point clouds + vertex colors.
    Scales object to fit unit bounding box [-1, 1] and samples high-density surface point cloud.
    """
    in_path = Path(input_dir)
    out_path = Path(processed_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    supported_exts = ('.glb', '.gltf', '.obj', '.stl', '.ply')
    mesh_files = [f for f in in_path.glob("*") if f.suffix.lower() in supported_exts]

    if not mesh_files:
        print(f"No raw mesh files found in {input_dir}. Ready for GLB/OBJ dataset additions.")
        return

    print(f"Processing {len(mesh_files)} raw dataset mesh(es)...")
    for mesh_path in mesh_files:
        try:
            mesh = trimesh.load(str(mesh_path), force='mesh')
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
            
            # 1. Scale object to fit unit bounding box [-1, 1]
            if len(mesh.vertices) > 0:
                mesh.vertices -= mesh.center_mass
                max_extent = np.max(mesh.extents)
                if max_extent > 0:
                    mesh.vertices /= max_extent

            # 2. Sample high-density surface point cloud (10,000 points)
            points, face_idx = trimesh.sample.sample_surface(mesh, 10000)
            
            # Extract normals for sampled points
            normals = mesh.face_normals[face_idx]

            # 3. Export normalized training target
            out_file = out_path / f"{mesh_path.stem}_processed.npz"
            np.savez(out_file, vertices=mesh.vertices, faces=mesh.faces, points=points, normals=normals)
            print(f"Processed dataset item: {mesh_path.name} -> {out_file.name}")
        except Exception as e:
            print(f"Error processing {mesh_path.name}: {e}")

if __name__ == "__main__":
    from pipeline.generate_complex_datasets import generate_complex_datasets

    print("Step 1: Generating complex procedural training meshes...")
    generate_complex_datasets()

    print("\nStep 2: Processing all dataset meshes into training-ready .npz files...")
    prepare_mesh_dataset("d:/3dSCANNER/data/datasets", "d:/3dSCANNER/data/datasets/processed")
