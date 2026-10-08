"""
Organic Sculpture & Statue Dataset Downloader and Multi-View Renderer

Downloads high-resolution 3D sculpture models from the Stanford 3D Scanning Repository
and common 3D benchmarks:
- Stanford Bunny (classic organic shape)
- Stanford Dragon (intricate scales and folds)
- Armadillo (complex anatomical geometry)
- Queen Nefertiti Bust (detailed facial contours and headdress)
- Spot the Cow (organic quadrupeds)
- Max Planck Bust (detailed head and facial features)

Renders 360-degree multi-view (2D RGB -> 3D Depth) image pairs for training
neural 3D reconstruction networks to understand complex organic contours.
"""

import sys
import math
import urllib.request
from pathlib import Path
import numpy as np
import cv2
import trimesh

DATASETS_DIR = Path("d:/3dSCANNER/data/datasets")
SCULPTURE_DIR = DATASETS_DIR / "sculptures"
RENDER_DIR = DATASETS_DIR / "sculpture_renders"
PROCESSED_DIR = DATASETS_DIR / "processed"

SCULPTURE_URLS = {
    "stanford_bunny": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj",
    "stanford_dragon": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/xyzrgb_dragon.obj",
    "armadillo": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.obj",
    "nefertiti_bust": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/nefertiti.obj",
    "spot_cow": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/spot.obj",
    "max_planck": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/max-planck.obj"
}


def download_sculpture_models() -> list:
    """Downloads benchmark 3D sculpture models if not already present."""
    SCULPTURE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_paths = []

    print("=" * 60)
    print("Downloading High-Resolution 3D Sculpture Datasets")
    print("=" * 60)

    for name, url in SCULPTURE_URLS.items():
        out_file = SCULPTURE_DIR / f"{name}.obj"
        stl_file = DATASETS_DIR / f"{name}.stl"

        if stl_file.exists():
            print(f"  [EXISTS] {name}.stl ({round(stl_file.stat().st_size / 1024, 1)} KB)")
            downloaded_paths.append(stl_file)
            continue

        if not out_file.exists():
            print(f"  Downloading {name} from repository...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as response, open(out_file, "wb") as f:
                    f.write(response.read())
                print(f"  [OK] Saved {out_file.name} ({round(out_file.stat().st_size / 1024, 1)} KB)")
            except Exception as e:
                print(f"  [WARN] Failed to download {name}: {e}")
                continue

        # Load, normalize, and export as STL for training dataset
        try:
            mesh = trimesh.load(str(out_file), force='mesh')
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())

            # Normalize to unit bounding box
            mesh.vertices -= mesh.center_mass
            max_ext = np.max(mesh.extents)
            if max_ext > 0:
                mesh.vertices /= max_ext

            mesh.export(str(stl_file))
            print(f"  [CONVERTED] Exported {stl_file.name} ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
            downloaded_paths.append(stl_file)
        except Exception as e:
            print(f"  [WARN] Could not convert {name}: {e}")

    return downloaded_paths


def render_sculpture_multiview(mesh_path: Path, output_dir: Path, num_views: int = 24, resolution: int = 128):
    """
    Renders 360-degree orbital views (RGB and Depth maps) for a 3D sculpture.
    Uses pure NumPy camera projection rasterization for cross-platform reliability.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(str(mesh_path), force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    # Unit normalization
    mesh.vertices -= mesh.center_mass
    max_ext = np.max(mesh.extents)
    if max_ext > 0:
        mesh.vertices /= max_ext

    # Sample dense surface points and normals
    points, face_idx = trimesh.sample.sample_surface(mesh, 25000)
    normals = mesh.face_normals[face_idx]

    rendered_count = 0
    for i in range(num_views):
        angle_rad = (2.0 * math.pi / num_views) * i
        
        # Rotation matrix around Y axis (turntable)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rot_y = np.array([
            [cos_a,  0, sin_a],
            [0,      1, 0],
            [-sin_a, 0, cos_a]
        ])

        rot_pts = points @ rot_y.T
        rot_norms = normals @ rot_y.T

        # Camera view projection (orthographic + directional shading)
        img_rgb = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        img_depth = np.zeros((resolution, resolution), dtype=np.float32)

        # Light direction (front-top-right)
        light_dir = np.array([0.4, 0.6, 0.7])
        light_dir /= np.linalg.norm(light_dir)

        # Shading: diffuse Lambertian
        dots = np.clip(np.sum(rot_norms * light_dir, axis=1), 0.15, 1.0)

        # Sort points by Z (depth buffer from back to front)
        order = np.argsort(rot_pts[:, 2])

        cx, cy = resolution / 2.0, resolution / 2.0
        scale = (resolution * 0.42)

        for idx in order:
            pt = rot_pts[idx]
            shading = dots[idx]

            px = int(cx + pt[0] * scale)
            py = int(cy - pt[1] * scale)  # Invert Y for image space
            z_val = float(pt[2])

            if 0 <= px < resolution and 0 <= py < resolution:
                # Sculpture stone color palette with shading
                color = np.array([195, 205, 215], dtype=np.float32) * shading
                img_rgb[py, px] = np.clip(color, 0, 255).astype(np.uint8)
                img_depth[py, px] = z_val

        # Save RGB view
        view_name = f"{mesh_path.stem}_view_{i:03d}"
        cv2.imwrite(str(output_dir / f"{view_name}_rgb.png"), img_rgb)
        
        # Normalize and save depth map visual
        depth_vis = ((img_depth - img_depth.min()) / (img_depth.max() - img_depth.min() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{view_name}_depth.png"), depth_vis)
        rendered_count += 1

    print(f"  [RENDERED] {rendered_count} views for {mesh_path.name} -> {output_dir.name}/")


def prepare_sculpture_pipeline():
    """Executes the full sculpture download, multi-view rendering, and dataset prep."""
    # 1. Download famous sculpture models
    models = download_sculpture_models()

    # 2. Render multi-view pairs for each sculpture
    print("\n" + "=" * 60)
    print("Rendering Multi-View 2D/3D Training Pairs for Sculptures")
    print("=" * 60)
    for model_path in models:
        out_renders = RENDER_DIR / model_path.stem
        render_sculpture_multiview(model_path, out_renders, num_views=24, resolution=128)

    # 3. Process into NPZ training files
    print("\n" + "=" * 60)
    print("Generating High-Resolution Training NPZ Targets")
    print("=" * 60)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for model_path in models:
        try:
            mesh = trimesh.load(str(model_path), force='mesh')
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
            points, face_idx = trimesh.sample.sample_surface(mesh, 20000)
            normals = mesh.face_normals[face_idx]
            out_npz = PROCESSED_DIR / f"{model_path.stem}_processed.npz"
            np.savez(out_npz, vertices=mesh.vertices, faces=mesh.faces, points=points, normals=normals)
            print(f"  [SAVED] {out_npz.name} (20,000 sampled surface points)")
        except Exception as e:
            print(f"  [ERROR] {model_path.name}: {e}")

    total_datasets = len(list(DATASETS_DIR.glob("*.stl")))
    print("\n" + "=" * 60)
    print(f"Sculpture Dataset Pipeline Complete! Total STL datasets: {total_datasets}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_sculpture_pipeline()
