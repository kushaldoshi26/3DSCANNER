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
            raise RuntimeError(f'Custom depth checkpoint is not ready: {e}') from e

    load_ai_depth_model()
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Depth-Anything-V2 inference
    if DEPTH_ANYTHING_PIPE is not None:
        try:
            from PIL import Image
            pil_img = Image.fromarray(img_rgb)
            result = DEPTH_ANYTHING_PIPE(pil_img)
            predicted=result.get('predicted_depth')
            depth=predicted.detach().float().cpu().squeeze().numpy() if torch.is_tensor(predicted) else np.asarray(result['depth'],dtype=np.float32)
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

    raise RuntimeError('No trained depth model is available. Choose TripoSR; brightness gradients are not 3D depth.')


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
    from PIL import Image
    import json
    from pipeline.preprocess_input import preprocess_for_generation
    from pipeline.point_cloud import depth_surface
    from pipeline.mesh_quality import validate_mesh
    out=Path(output_stl_path);out.parent.mkdir(parents=True,exist_ok=True)
    prep=preprocess_for_generation(input_image_path,str(out.parent/'prep'))
    rgba=np.asarray(Image.open(prep['foreground_rgba']).convert('RGBA'))
    ratio=min(1.,512/max(rgba.shape[:2]))
    if ratio<1: rgba=cv2.resize(rgba,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
    alpha=rgba[:,:,3:4].astype(np.float32)/255
    rgb=(rgba[:,:,:3]*alpha+127*(1-alpha)).astype(np.uint8)
    depth=estimate_ai_depth(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),model_type=model_type)
    mesh=depth_surface(rgba,depth)
    temporary=out.parent/'preview_cloud.tmp.npz'
    np.savez_compressed(temporary,points=np.asarray(mesh.vertices),colors=np.asarray(mesh.visual.vertex_colors[:,:3]))
    temporary.replace(out.parent/'preview_cloud.npz')
    mesh.export(str(out));scene=trimesh.Scene(mesh);scene.metadata['source_view']=[0,0,1];scene.metadata['geometry_kind']='estimated_depth_surface';scene.export(str(out.with_suffix('.glb')))
    qc=validate_mesh(mesh,coordinate_space='normalized')
    qc['warnings'].append('Single-view depth is an estimated open surface, not a closed 360-degree reconstruction or calibrated measurement.')
    (out.parent/'quality_report.json').write_text(json.dumps({'quality':qc,'engine_used':'ai_depth','warnings':qc['warnings'],'geometry_kind':'estimated_depth_surface'},indent=2))
    return str(out)


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/frame_000.jpg"
    outp = sys.argv[2] if len(sys.argv) > 2 else "../data/output/ai_depth_model.stl"
    generate_ai_depth_3d_mesh(inp, outp)
