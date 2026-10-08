import cv2
import numpy as np
import torch
import open3d as o3d
import trimesh
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preprocess import process_single_image
from pipeline.ai_refine import refine_mesh
from pipeline.depth_normal_refine import estimate_surface_normals, image_to_detailed_pointcloud
from pipeline.texture_projector import project_texture_to_mesh

DEPTH_ANYTHING_PIPE = None
MIDAS_MODEL = None
MIDAS_TRANSFORM = None

def load_ai_depth_model():
    """Loads Depth-Anything-V2 foundation model on GPU (with MiDaS and gradient fallbacks)."""
    global DEPTH_ANYTHING_PIPE, MIDAS_MODEL, MIDAS_TRANSFORM
    if DEPTH_ANYTHING_PIPE is not None or MIDAS_MODEL is not None:
        return

    # 1. Try Depth-Anything-V2-Base via transformers on GPU
    try:
        from transformers import pipeline
        from PIL import Image
        print("Loading Depth-Anything-V2-Base AI foundation model on GPU...")
        device_id = 0 if torch.cuda.is_available() else -1
        DEPTH_ANYTHING_PIPE = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Base-hf",
            device=device_id
        )
        print("Depth-Anything-V2-Base loaded successfully on GPU!")
        return
    except Exception as e:
        print(f"Depth-Anything-V2-Base notice ({e}). Trying Small model fallback...")
        try:
            from transformers import pipeline
            device_id = 0 if torch.cuda.is_available() else -1
            DEPTH_ANYTHING_PIPE = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=device_id
            )
            print("Depth-Anything-V2-Small loaded successfully on GPU!")
            return
        except Exception as e2:
            print(f"Depth-Anything-V2-Small notice ({e2}). Trying MiDaS fallback...")

    # 2. Try PyTorch MiDaS
    try:
        print("Loading PyTorch MiDaS AI Depth Estimation Model...")
        MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        MIDAS_MODEL.eval()
        if torch.cuda.is_available():
            MIDAS_MODEL.to("cuda")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        MIDAS_TRANSFORM = midas_transforms.small_transform
        print("PyTorch MiDaS model loaded successfully.")
    except Exception as e:
        print(f"Notice: PyTorch MiDaS load skipped ({e}). Using gradient depth fallback.")
        MIDAS_MODEL = False


def estimate_ai_depth(img_bgr: np.ndarray, model_type: str = "foundation") -> np.ndarray:
    """
    Estimates high-precision relative 3D depth map from 2D RGB image.
    Supports:
      - 'custom': Newly trained PyTorch ResNet 3D AI neural network (best_3d_ai_model.pth)
      - 'foundation': Depth-Anything-V2 / MiDaS foundation models
    """
    # If custom trained model requested, use predict_custom_neural_depth_and_normals
    if model_type == "custom":
        try:
            from pipeline.train_ai_3d import predict_custom_neural_depth_and_normals
            depth, normals, mask = predict_custom_neural_depth_and_normals(img_bgr)
            print("   Using Custom Trained PyTorch 3D AI Model (best_3d_ai_model.pth)...")
            return depth
        except Exception as e:
            print(f"   Custom model inference notice ({e}). Falling back to foundation model.")

    load_ai_depth_model()
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Depth-Anything-V2 inference
    if DEPTH_ANYTHING_PIPE is not None:
        try:
            from PIL import Image
            pil_img = Image.fromarray(img_rgb)
            result = DEPTH_ANYTHING_PIPE(pil_img)
            depth_img = result["depth"]
            depth = np.array(depth_img, dtype=np.float32)
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            return depth
        except Exception as e:
            print(f"Depth-Anything-V2 inference error: {e}")

    # 2. MiDaS inference
    if MIDAS_MODEL and MIDAS_TRANSFORM:
        try:
            input_batch = MIDAS_TRANSFORM(img_rgb)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_batch = input_batch.to(device)
            with torch.no_grad():
                prediction = MIDAS_MODEL(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth = prediction.cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            return depth
        except Exception as e:
            print(f"MiDaS inference error: {e}")

    # 3. High-detail gradient fallback
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) / 255.0
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(dx**2 + dy**2)
    depth = 1.0 - (blur * 0.7 + grad * 0.3)
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)


def compute_depth_normals(depth_map: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Computes exact per-pixel surface normals directly from the AI depth map gradients."""
    dz_dx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3) * scale
    dz_dy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3) * scale
    
    # Normal vector: (-dz/dx, -dz/dy, 1.0)
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(depth_map)
    
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    nx /= norm
    ny /= norm
    nz /= norm
    
    return np.dstack([nx, ny, nz])


def remove_flying_pixels(mesh: o3d.geometry.TriangleMesh, max_edge_length: float = 0.08) -> o3d.geometry.TriangleMesh:
    """
    Removes stretched triangles (flying pixels) caused by depth map boundaries.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    if len(triangles) == 0:
        return mesh

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    
    edge1 = np.linalg.norm(v0 - v1, axis=1)
    edge2 = np.linalg.norm(v1 - v2, axis=1)
    edge3 = np.linalg.norm(v2 - v0, axis=1)
    
    # Keep only triangles where all 3 edges are within the threshold
    valid_mask = (edge1 <= max_edge_length) & (edge2 <= max_edge_length) & (edge3 <= max_edge_length)
    mesh.triangles = o3d.utility.Vector3iVector(triangles[valid_mask])
    mesh.remove_unreferenced_vertices()
    return mesh


def generate_ai_depth_3d_mesh(input_image_path: str, output_stl_path: str, model_type: str = "foundation") -> str:
    """
    Generates high-detail 3D model from single 2D image using Custom PyTorch / Depth-Anything-V2 / MiDaS
    AI monocular depth estimation, per-pixel depth gradient surface normals, Poisson surface
    reconstruction, and GLB texture projection.
    """
    in_p = Path(input_image_path)
    out_stl = Path(output_stl_path)
    out_glb = out_stl.with_suffix(".glb")
    out_stl.parent.mkdir(parents=True, exist_ok=True)

    print(f"1. AI Depth 3D ({model_type}): Isolating background for {in_p.name}...")
    temp_rgba_path = out_stl.parent / f"temp_ai_{in_p.stem}_rgba.png"
    process_single_image(str(in_p), str(temp_rgba_path))

    rgba = cv2.imread(str(temp_rgba_path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[2] < 4:
        raise ValueError("Failed to load 4-channel RGBA image.")

    # Clamp resolution to optimal max 512 dimension for fast inference & reconstruction
    max_dim = 512
    h_orig, w_orig = rgba.shape[:2]
    if max(h_orig, w_orig) > max_dim:
        scale_img = max_dim / max(h_orig, w_orig)
        new_w, new_h = int(w_orig * scale_img), int(h_orig * scale_img)
        rgba = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    b, g, r, alpha = cv2.split(rgba)
    img_bgr = cv2.merge([b, g, r])
    mask = alpha > 15

    print(f"2. Predicting per-pixel AI depth map & surface normals ({model_type})...")
    ai_depth = estimate_ai_depth(img_bgr, model_type=model_type)
    
    # Smooth boundary edge feathering to preserve exact inner subject depth
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    edge_feather = np.clip(dist / 6.0, 0.0, 1.0)
    
    # 100% fine AI depth inside the subject, tapered smoothly to 0 at boundary
    final_depth = ai_depth * edge_feather
    final_depth[~mask] = 0.0

    normal_map = compute_depth_normals(final_depth, scale=3.0)

    print("3. Generating high-density normal-guided 3D volumetric point cloud...")
    h, w = alpha.shape
    cx, cy = w / 2.0, h / 2.0
    scale = 2.0 / max(h, w)
    rgb = cv2.merge([r, g, b]) / 255.0

    y_idx, x_idx = np.where(mask)
    if len(y_idx) == 0:
        raise ValueError("No valid foreground subject found in masked image.")

    # Bound maximum point cloud size to ~100k points for clean, fast reconstruction
    max_pts = 100000
    if len(y_idx) > max_pts:
        step = int(np.ceil(len(y_idx) / max_pts))
        y_idx = y_idx[::step]
        x_idx = x_idx[::step]

    z_front = (final_depth[y_idx, x_idx] * 0.9).astype(np.float64)
    x_3d = ((x_idx - cx) * scale).astype(np.float64)
    y_3d = (-(y_idx - cy) * scale).astype(np.float64)
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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"   Point cloud: {len(points)} points")

    print("4. Reconstructing high-detail Poisson 3D Surface (depth=8)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, scale=1.1, linear_fit=True
    )

    # Trim low-density boundary artifacts — keep 98% of detail
    densities_arr = np.asarray(densities)
    if len(densities_arr) > 0:
        density_threshold = np.quantile(densities_arr, 0.02)
        vertices_to_remove = densities_arr < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

    # 4b. Remove flying pixels (stretched edge discontinuity triangles)
    mesh = remove_flying_pixels(mesh, max_edge_length=0.10)

    mesh.compute_vertex_normals()
    print(f"   Poisson mesh: {len(mesh.vertices)} vertices, {len(np.asarray(mesh.triangles))} faces")

    temp_stl = out_stl.parent / f"temp_ai_{in_p.stem}_raw.stl"
    o3d.io.write_triangle_mesh(str(temp_stl), mesh)

    print("5. Gentle mesh refinement (preserving detail)...")
    refine_mesh(str(temp_stl), str(out_stl), smoothing_iterations=1, decimation_ratio=0.95)

    print("6. Projecting RGB texture onto 3D mesh vertices & exporting GLB...")
    project_texture_to_mesh(str(out_stl), str(input_image_path), str(out_glb))

    if temp_rgba_path.exists():
        try: temp_rgba_path.unlink()
        except Exception: pass
    if temp_stl.exists():
        try: temp_stl.unlink()
        except Exception: pass

    print(f"High-Detail Textured 3D Model created: STL={out_stl}, GLB={out_glb}")
    return str(out_stl)

if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/frame_000.jpg"
    outp = sys.argv[2] if len(sys.argv) > 2 else "../data/output/ai_depth_model.stl"
    generate_ai_depth_3d_mesh(inp, outp)
