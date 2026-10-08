import cv2
import numpy as np
import open3d as o3d
import trimesh
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.preprocess import process_single_image
from pipeline.ai_refine import refine_mesh
from pipeline.texture_projector import project_texture_to_mesh
from pipeline.ai_depth_3d import compute_depth_normals

def _get_foundation_depth(img_bgr: np.ndarray) -> np.ndarray:
    """
    Uses Depth-Anything-V2 (trained on millions of real photos) for depth estimation.
    Falls back to MiDAS, then gradient-based depth.
    This produces FAR better facial depth maps than the custom model
    which was only trained on 42 synthetic meshes.
    """
    import torch
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Depth-Anything-V2 (best for real photos/faces)
    try:
        from transformers import pipeline as hf_pipeline
        from PIL import Image
        print("   Using Depth-Anything-V2 foundation model (optimized for real faces)...")
        device_id = 0 if torch.cuda.is_available() else -1
        pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Base-hf",
            device=device_id
        )
        pil_img = Image.fromarray(img_rgb)
        result = pipe(pil_img)
        depth = np.array(result["depth"], dtype=np.float32)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth
    except Exception as e:
        print(f"   Depth-Anything-V2 unavailable ({e}), trying Small variant...")

    try:
        from transformers import pipeline as hf_pipeline
        from PIL import Image
        device_id = 0 if torch.cuda.is_available() else -1
        pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device_id
        )
        pil_img = Image.fromarray(img_rgb)
        result = pipe(pil_img)
        depth = np.array(result["depth"], dtype=np.float32)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth
    except Exception as e:
        print(f"   Depth-Anything-V2-Small unavailable ({e}), trying MiDAS...")

    # 2. MiDAS fallback
    try:
        midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        midas_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        midas_model.to(device)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = midas_transforms.small_transform
        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze()
        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth
    except Exception as e:
        print(f"   MiDAS unavailable ({e}), using gradient fallback...")

    # 3. Gradient fallback
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) / 255.0
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(dx**2 + dy**2)
    depth = 1.0 - (blur * 0.7 + grad * 0.3)
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)


def single_image_to_3d_mesh(input_image_path: str, output_stl_path: str) -> str:
    """
    Converts a single 2D photo into a high-detail 3D mesh model.
    
    Uses Depth-Anything-V2 foundation model for depth (trained on millions of real
    images including faces), bilateral filtering for smooth surfaces, and high-resolution
    Poisson reconstruction with aggressive Taubin smoothing.
    """
    in_p = Path(input_image_path)
    if not in_p.is_absolute():
        in_p = PROJECT_ROOT / in_p
    
    out_p = Path(output_stl_path)
    if not out_p.is_absolute():
        out_p = PROJECT_ROOT / out_p
    
    # Export to data/exports/
    export_dir = PROJECT_ROOT / "data" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    out_p = export_dir / out_p.name
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"1. Single-Image 3D: Isolating subject from {in_p.name}...")
    temp_rgba_path = out_p.parent / f"temp_{in_p.stem}_rgba.png"
    process_single_image(str(in_p), str(temp_rgba_path))

    rgba = cv2.imread(str(temp_rgba_path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[2] < 4:
        raise ValueError("Failed to obtain 4-channel RGBA image for 3D generation.")

    # Keep very high resolution for facial/surface detail (max 2048)
    max_dim = 2048
    h_orig, w_orig = rgba.shape[:2]
    if max(h_orig, w_orig) > max_dim:
        scale_img = max_dim / max(h_orig, w_orig)
        new_w, new_h = int(w_orig * scale_img), int(h_orig * scale_img)
        rgba = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    b, g, r, alpha = cv2.split(rgba)
    img_bgr = cv2.merge([b, g, r])
    rgb = cv2.merge([r, g, b]) / 255.0
    mask = alpha > 15

    # ============================================================
    # DEPTH-ANYTHING-V2 for real photo depth (NOT custom model)
    # The custom model was trained on synthetic meshes and produces
    # noisy, bumpy depth on real human faces.
    # ============================================================
    print("2. Predicting depth via Depth-Anything-V2 foundation model...")
    ai_depth = _get_foundation_depth(img_bgr)
    
    # ============================================================
    # BILATERAL FILTER — removes high-freq noise while preserving edges
    # This eliminates the "crocodile skin" texture on smooth surfaces
    # ============================================================
    print("   Applying bilateral filter for smooth facial surfaces...")
    ai_depth_f32 = ai_depth.astype(np.float32)
    ai_depth_smooth = cv2.bilateralFilter(ai_depth_f32, d=9, sigmaColor=0.05, sigmaSpace=7)
    # Apply a second pass for extra smoothness on skin
    ai_depth_smooth = cv2.bilateralFilter(ai_depth_smooth, d=9, sigmaColor=0.03, sigmaSpace=5)
    
    # Edge feathering
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    edge_feather = np.clip(dist / 6.0, 0.0, 1.0)
    
    final_depth = ai_depth_smooth * edge_feather
    final_depth[~mask] = 0.0
    
    # Compute normals from the SMOOTHED depth map (not raw)
    normal_map = compute_depth_normals(final_depth, scale=3.0)
    normal_map[~mask] = 0.0

    h, w = alpha.shape
    cx, cy = w / 2.0, h / 2.0
    scale_factor = 2.0 / max(h, w)

    # 3. Generate high-density 3D point cloud
    y_idx, x_idx = np.where(mask)
    if len(y_idx) == 0:
        raise ValueError("No valid foreground points extracted from image.")

    # Increased point limit for higher detail point cloud
    max_pts = 1000000
    if len(y_idx) > max_pts:
        step = int(np.ceil(len(y_idx) / max_pts))
        y_idx = y_idx[::step]
        x_idx = x_idx[::step]

    z_front = (final_depth[y_idx, x_idx] * 0.9).astype(np.float64)
    x_3d = ((x_idx - cx) * scale_factor).astype(np.float64)
    y_3d = (-(y_idx - cy) * scale_factor).astype(np.float64)
    front_pts = np.stack([x_3d, y_3d, z_front], axis=1)
    front_normals = normal_map[y_idx, x_idx].astype(np.float64)
    front_colors = rgb[y_idx, x_idx].astype(np.float64)

    taper = edge_feather[y_idx, x_idx].astype(np.float64)
    z_back = (-z_front * 0.4 * taper).astype(np.float64)
    back_pts = np.stack([x_3d, y_3d, z_back], axis=1)
    back_normals = np.stack([-front_normals[:, 0] * 0.5, -front_normals[:, 1] * 0.5, -np.ones_like(z_front)], axis=1)
    norm_lens = np.linalg.norm(back_normals, axis=1, keepdims=True) + 1e-8
    back_normals = back_normals / norm_lens
    back_colors = front_colors * 0.75

    points = np.vstack([front_pts, back_pts])
    normals = np.vstack([front_normals, back_normals])
    colors = np.vstack([front_colors, back_colors])

    print(f"3. Building Open3D point cloud with {len(points)} points...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ============================================================
    # HIGH-RESOLUTION POISSON (depth=12) + AGGRESSIVE TAUBIN SMOOTHING
    # ============================================================
    print("4. Running Poisson surface reconstruction (depth=12)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=12, scale=1.1, linear_fit=True
    )

    # Trim low-density boundary artifacts
    densities_arr = np.asarray(densities)
    if len(densities_arr) > 0:
        density_threshold = np.quantile(densities_arr, 0.05)
        vertices_to_remove = densities_arr < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

    if len(mesh.triangles) == 0:
        print("Poisson failed, falling back to Alpha Shapes...")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.08)

    # Aggressive Taubin smoothing — 50 iterations for genuinely smooth skin
    print("   Applying Taubin smoothing (50 iterations) for smooth facial texture...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=50)
    mesh.compute_vertex_normals()
    print(f"   Poisson mesh: {len(mesh.vertices)} vertices, {len(np.asarray(mesh.triangles))} faces")
    # Transfer colors from the original point cloud to the reconstructed mesh
    print("   Transferring perfect vertex colors from original point cloud...")
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    mesh_colors = []
    pcd_colors = np.asarray(pcd.colors)
    mesh_vertices = np.asarray(mesh.vertices)
    for v in mesh_vertices:
        _, idx, _ = kdtree.search_knn_vector_3d(v, 1)
        idx_arr = np.asarray(idx).flatten()
        if len(idx_arr) > 0:
            mesh_colors.append(pcd_colors[idx_arr[0]])
        else:
            mesh_colors.append([0.5, 0.5, 0.5])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh_colors))

    # Save initial mesh with vertex colors
    temp_ply = out_p.parent / f"temp_{in_p.stem}_initial.ply"
    o3d.io.write_triangle_mesh(str(temp_ply), mesh)

    # Calculate decimation ratio to hit ~50k faces
    current_faces = len(np.asarray(mesh.triangles))
    target_faces = 50000
    dec_ratio = min(1.0, float(target_faces) / float(current_faces)) if current_faces > 0 else 1.0

    print(f"5. Gentle mesh refinement (Artifact removal, Laplacian, Decimation targeting {target_faces} faces)...")
    out_ply = out_p.with_suffix(".ply")
    refine_mesh(
        str(temp_ply), str(out_ply), 
        reference_images=None,  # No AI displacement for real photos
        smoothing_iterations=2, 
        decimation_ratio=dec_ratio
    )

    # The refine_mesh function automatically generates the final .glb
    glb_p = out_ply.with_suffix(".glb")

    # Clean temporary files
    if temp_rgba_path.exists(): temp_rgba_path.unlink()
    if temp_ply.exists(): temp_ply.unlink()

    print(f"Successfully generated textured 3D model: PLY={out_ply}, GLB={glb_p}")
    return str(out_ply)

if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "data" / "raw" / "frame_000.jpg")
    outp = sys.argv[2] if len(sys.argv) > 2 else str(PROJECT_ROOT / "data" / "exports" / "single_image_model.ply")
    single_image_to_3d_mesh(inp, outp)


