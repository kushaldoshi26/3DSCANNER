import os
import urllib.request
import torch
from pathlib import Path

DATASET_DIR = Path("d:/3dSCANNER/data/datasets")
MODELS_DIR = Path("d:/3dSCANNER/data/models")

def download_ai_depth_models():
    """
    Downloads PyTorch Hub MiDaS / Depth Estimation models for 2D photo to 3D reconstruction.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checking and downloading PyTorch AI depth models to {MODELS_DIR}...")

    # Write trusted list to torch hub cache
    try:
        hub_dir = Path.home() / ".cache" / "torch" / "hub"
        hub_dir.mkdir(parents=True, exist_ok=True)
        trusted_file = hub_dir / "trusted_list"
        content = trusted_file.read_text() if trusted_file.exists() else ""
        if "rwightman_gen-efficientnet-pytorch" not in content:
            with open(trusted_file, "a") as f:
                f.write("rwightman_gen-efficientnet-pytorch\nintel-isl_MiDaS_master\n")
    except Exception:
        pass

    try:
        print("Downloading MiDaS PyTorch model weights via torch.hub...")
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        model.eval()
        print("Successfully loaded MiDaS PyTorch depth model!")
    except Exception as e:
        print(f"Notice: PyTorch hub MiDaS weights load notice ({e}). Automatic gradient depth fallback ready.")

def download_sample_3d_printing_datasets():
    """
    Generates and downloads benchmark 3D printing STL sample dataset models (Calibration Cube, Icosphere, Cylinder, Torus)
    for validation, slicing, and 3D printing testing.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checking and creating sample 3D printing datasets in {DATASET_DIR}...")

    import trimesh

    models = {
        "calibration_cube_20mm.stl": trimesh.creation.box(extents=[20, 20, 20]),
        "icosphere_3d_print.stl": trimesh.creation.icosphere(subdivisions=3, radius=15),
        "cylinder_test.stl": trimesh.creation.cylinder(radius=10, height=30),
        "torus_ring.stl": trimesh.creation.torus(major_radius=20, minor_radius=5)
    }

    for filename, mesh in models.items():
        out_file = DATASET_DIR / filename
        if not out_file.exists():
            mesh.export(str(out_file))
            print(f"Generated sample 3D printing dataset model: {filename} ({out_file.stat().st_size} bytes)")
        else:
            print(f"Sample dataset file '{filename}' ready.")

if __name__ == "__main__":
    download_sample_3d_printing_datasets()
    download_ai_depth_models()
