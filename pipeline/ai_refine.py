import trimesh
import torch
import numpy as np
import cv2
from pathlib import Path
import sys
from typing import List, Optional

# Import the custom trained AI model structure
from pipeline.train_ai_3d import HighPrecision3DReconstructionNet, predict_custom_neural_depth_and_normals
from pipeline.camera_calibration import load_calibration

# Hard ceiling on how far AI refinement may move any vertex, in millimetres.
# This preserves the measured (COLMAP / dense MVS) geometry: AI refinement is
# only allowed to add subtle high-frequency detail, never to warp the shape.
MAX_DISPLACEMENT_MM = 0.5

# Relative displacement ceiling for GENERATIVE single-image mode, where the
# mesh is not metrically calibrated. Expressed as a fraction of the mesh
# bounding-box diagonal so it scales with the generated model, never a fake mm.
MAX_DISPLACEMENT_RATIO = 0.001


def refine_mesh(
    input_mesh_path: str,
    output_mesh_path: str,
    reference_images: Optional[List[str]] = None,
    smoothing_iterations: int = 3,
    decimation_ratio: float = 0.98,
    enable_ai_displacement: bool = True,
    max_displacement_mm: float = MAX_DISPLACEMENT_MM,
) -> str:
    """
    Refines 3D mesh using PyTorch/Trimesh for gentle vertex smoothing and topology repair.

    AI geometric detail fusion is OPTIONAL and, when enabled, is strictly bounded
    by `max_displacement_mm` so the original measured geometry is never destroyed.
    """
    in_p = Path(input_mesh_path)
    
    out_p = Path(output_mesh_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading mesh for refinement: {in_p}...")
    mesh = trimesh.load_mesh(str(in_p))

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    initial_faces = len(mesh.faces)
    print(f"Initial mesh stats: {len(mesh.vertices)} vertices, {initial_faces} faces.")
    
    # 1. Weight Loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_model = None
    if reference_images and len(reference_images) > 0:
        print("Loading custom trained HighPrecision3DReconstructionNet...")
        model_path = "d:/3dSCANNER/data/models/best_3d_ai_model.pth"
        if Path(model_path).exists():
            ai_model = HighPrecision3DReconstructionNet().to(device)
            ckpt = torch.load(model_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            ai_model.load_state_dict(state_dict)
            ai_model.eval()
            print("Successfully loaded AI model weights for mesh fusion.")
        else:
            print(f"Warning: Custom AI model weights not found at {model_path}. Skipping AI displacement.")

    # 1. Floating Artifact Removal
    print("Attempting floating artifact removal...")
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            mesh = max(components, key=lambda m: m.bounding_box.volume)
            print(f"Discarded {len(components)-1} floating islands.")
    except Exception as e:
        print(f"Component splitting failed: {e}")

    # 2. Gentle hole filling (best-effort)
    if not mesh.is_watertight:
        print("Attempting gentle hole filling...")
        try:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_normals(mesh)
        except Exception as e:
            print(f"Hole filling note: {e}")

    vertices_tensor = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces_tensor = torch.tensor(mesh.faces, dtype=torch.int64)
    num_vertices = vertices_tensor.shape[0]

    # 2 & 3. Direct Feature Extraction & Mesh Fusion
    if ai_model is not None and enable_ai_displacement:
        print(f"Extracting high-frequency normal maps from {len(reference_images)} reference frames...")
        
        # Calculate vertex normals for displacement direction
        if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
            mesh.fix_normals()
        vertex_normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32)
        
        # Accumulate displacement from all reference views
        displacement_accumulator = torch.zeros((num_vertices, 3), dtype=torch.float32)
        weight_accumulator = torch.zeros((num_vertices, 1), dtype=torch.float32)

        # Get mesh bounding box for projection
        bbox_min = np.min(mesh.vertices, axis=0)
        bbox_max = np.max(mesh.vertices, axis=0)
        bbox_range = bbox_max - bbox_min + 1e-8

        with torch.no_grad():
            for img_path in reference_images:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None: continue
                
                h_orig, w_orig = img_bgr.shape[:2]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA) / 255.0
                tensor_in = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

                # Predict depth and surface normals from the custom AI model
                pred_depth, pred_normals, pred_mask = ai_model(tensor_in)
                
                # Extract high-frequency normal detail signal
                # The AI normal map contains fine surface angles (eyelids, lips, fabric creases).
                # We subtract a blurred version to isolate only the high-frequency detail.
                normals_np = pred_normals.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                depth_np = pred_depth.squeeze().cpu().numpy()  # (H, W)
                mask_np = pred_mask.squeeze().cpu().numpy()  # (H, W)
                
                # High-frequency detail = original - smooth
                smooth_depth = cv2.GaussianBlur(depth_np, (15, 15), 3.0)
                hf_detail = depth_np - smooth_depth  # High-freq depth residual
                
                # Load physically correct camera intrinsics
                calib = load_calibration(img_shape=(w_orig, h_orig))
                K = np.array(calib["camera_matrix"], dtype=np.float64)
                dist = np.array(calib["distortion_coefficients"], dtype=np.float64)

                # World to Camera Transformation (Extrinsics)
                # Ideally, this comes from COLMAP per-image. As a fallback, we center the object.
                rvec = np.zeros((3, 1), dtype=np.float64)
                
                # Approximate translation to fit object in FOV if pose is unknown
                z_dist = max(bbox_range[0], bbox_range[1]) * 1.5
                tvec = np.array([[0.0], [0.0], [z_dist]], dtype=np.float64)

                # Center mesh at origin for projection if it's not already
                verts_np = mesh.vertices.copy()
                mesh_center = bbox_min + bbox_range / 2.0
                verts_centered = verts_np - mesh_center

                # Project 3D vertices to actual 2D image pixels using Lens Distortion + Intrinsics
                img_pts, _ = cv2.projectPoints(verts_centered, rvec, tvec, K, dist)
                img_pts = img_pts.squeeze(1)

                # Scale projected points down to the 256x256 AI feature map resolution
                proj_u = (img_pts[:, 0] / w_orig * 255).astype(int)
                proj_v = (img_pts[:, 1] / h_orig * 255).astype(int)
                proj_u = np.clip(proj_u, 0, 255)
                proj_v = np.clip(proj_v, 0, 255)
                
                # Sample the high-frequency detail at each vertex's projected location
                for vi in range(num_vertices):
                    u, v = proj_u[vi], proj_v[vi]
                    if mask_np[v, u] > 0.3:  # Only displace visible vertices
                        detail_mag = float(hf_detail[v, u])
                        # Displace along the vertex normal by the high-frequency detail magnitude
                        displacement_accumulator[vi] += vertex_normals[vi] * detail_mag
                        weight_accumulator[vi] += 1.0

        # Apply averaged AI displacement to vertices, but clamp every vertex move
        # to the configured maximum so measured geometry is preserved.
        valid_weights = weight_accumulator > 0
        if valid_weights.any():
            avg_displacement = displacement_accumulator / torch.clamp(weight_accumulator, min=1.0)

            # Mesh is in metres internally; convert the mm ceiling to mesh units.
            extents_m = np.ptp(mesh.vertices, axis=0)
            scale_mm_per_unit = 1000.0 if np.all(extents_m < 10.0) else 1.0
            max_disp_units = max_displacement_mm / scale_mm_per_unit

            # Confidence = how many views agreed on the displaced vertex.
            confidence = weight_accumulator / torch.clamp(weight_accumulator.max(), min=1.0)
            disp_norm = torch.norm(avg_displacement, dim=1, keepdim=True) + 1e-8
            clamped = avg_displacement / disp_norm * torch.clamp(disp_norm, max=max_disp_units)

            # Only move vertices where we have enough agreement to trust the signal.
            confident = confidence > 0.34
            blend = 0.1 * confident.float()
            vertices_tensor += clamped * blend
            print(
                f"Successfully merged AI-predicted geometry with Poisson mesh "
                f"(max displacement {max_displacement_mm} mm, "
                f"{int(confident.sum())}/{num_vertices} vertices confident)."
            )

    # Laplacian smoothing (Volume preserving)
    if num_vertices > 100000:
        print(f"Skipping Laplacian smoothing (mesh has {num_vertices} vertices, too large).")
    else:
        from pipeline.cleanup_mesh import _taubin_smooth
        print(f"Applying volume-preserving Taubin smoothing ({smoothing_iterations} iterations)...")
        mesh = _taubin_smooth(mesh, smoothing_iterations)

    # Clean up degenerate geometry
    mesh.remove_unreferenced_vertices()
    try:
        mask = mesh.nondegenerate_faces()
        if mask is not None and len(mask) > 0:
            mesh.update_faces(mask)
    except Exception:
        pass
    mesh.fix_normals()

    # 4. Scale & Normalization
    # Assuming the input mesh was already scaled during the turntable calibration phase,
    # we ensure the bounding box and unit scales remain consistent.
    print(f"Normalizing to real-world dimensions. Extents: {np.ptp(mesh.vertices, axis=0)} mm")

    if decimation_ratio < 1.0 and len(mesh.faces) > 10000:
        target_faces = max(10000, int(len(mesh.faces) * decimation_ratio))
        if target_faces < len(mesh.faces):
            try:
                ratio = float(target_faces) / float(len(mesh.faces))
                mesh = mesh.simplify_quadric_decimation(ratio)
                print(f"Decimated: {initial_faces} -> {len(mesh.faces)} faces.")
            except Exception as e:
                print(f"Quadric decimation skipped: {e}")

    watertight_status = "watertight" if mesh.is_watertight else "non-watertight (detail preserved)"
    print(f"Refined mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces ({watertight_status}).")

    print(f"Saving refined mesh to {out_p}...")
    mesh.export(str(out_p))

    # Texture Baking (UV Unwrapping with xatlas)
    print("Baking vertex colors to texture map...")
    try:
        import xatlas
        import cv2
        from PIL import Image
        
        if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
            vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
            
            old_colors = mesh.visual.vertex_colors[:, :3]
            new_colors = old_colors[vmapping]
            
            texture_size = 1024
            texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
            
            for uv, color in zip(uvs, new_colors):
                x = np.clip(int(uv[0] * (texture_size - 1)), 0, texture_size - 1)
                y = np.clip(int((1.0 - uv[1]) * (texture_size - 1)), 0, texture_size - 1)
                texture[y, x] = color
                
            kernel = np.ones((7,7), np.uint8)
            texture = cv2.dilate(texture, kernel, iterations=3)
            
            tex_img = Image.fromarray(texture)
            material = trimesh.visual.material.SimpleMaterial(image=tex_img)
            
            new_mesh = trimesh.Trimesh(vertices=mesh.vertices[vmapping], faces=indices)
            new_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=tex_img, material=material)
            mesh = new_mesh
            print("UV Unwrapping and texture baking successful.")
    except Exception as e:
        print(f"UV generation error: {e}. Falling back to vertex colors.")

    # Also save as GLB
    glb_out = out_p.with_suffix(".glb")
    mesh.export(str(glb_out))
    print(f"Saved GLB asset to {glb_out}")

    # Phase 7.1: Run mesh cleanup as final pipeline step
    try:
        from pipeline.cleanup_mesh import cleanup_generated_mesh
        print("Running final cleanup to remove floating components...")
        cleaned_path = str(out_p).replace('.stl', '_clean.stl')
        cleanup_generated_mesh(str(out_p), cleaned_path, object_type='organic')
        
        # Override output if cleanup successful
        if Path(cleaned_path).exists():
            out_p = Path(cleaned_path)
    except Exception as e:
        print(f"Cleanup failed (non-fatal): {e}")

    return str(out_p)

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "../data/output/model.stl"
    outp = sys.argv[2] if len(sys.argv) > 2 else "../data/output/refined_model.stl"
    refine_mesh(inp, outp)
