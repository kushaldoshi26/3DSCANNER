"""
TripoSR (Stability AI / Tripo AI) Single-Image 3D Large Reconstruction Model (LRM)

Generates true 360-degree, solid, watertight, textured 3D meshes from a single photograph
in < 1 second using Tri-plane Transformer NeRF / Signed Distance Field representations.
Eliminates 2.5D depth extrusion streaks, hollow backsides, and flying pixel artifacts.
"""

import torch
import numpy as np
import trimesh
from PIL import Image
from pathlib import Path
import sys

# Add project root to sys.path to find the local 'tsr' module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

TRIPOSR_MODEL = None

def get_triposr_model():
    """Initializes and caches TripoSR model on CUDA/CPU."""
    global TRIPOSR_MODEL
    if TRIPOSR_MODEL is None:
        print("Loading pre-trained TripoSR 3D Foundation Model (stabilityai/TripoSR)...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        TRIPOSR_MODEL = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt"
        )
        TRIPOSR_MODEL.renderer.set_chunk_size(8192)
        TRIPOSR_MODEL.to(device)
        print(f"TripoSR loaded successfully on device: {device}")
    return TRIPOSR_MODEL


def generate_triposr_3d_mesh(
    image_path: str,
    output_stl_path: str,
    output_glb_path: str = None,
    resolution: int = 512,
    mc_threshold: float = 25.0
) -> str:
    """
    Generates a full 360-degree solid textured 3D model from a single photo using TripoSR.
    
    Args:
        image_path: Path to input 2D image
        output_stl_path: Output STL file path
        output_glb_path: Output GLB file path (default: same name as STL)
        resolution: Marching cubes voxel resolution (default: 256 for high detail)
        mc_threshold: Marching cubes surface extraction threshold
    
    Returns:
        Path to output STL file
    """
    in_p = Path(image_path)
    out_stl = Path(output_stl_path)
    out_glb = Path(output_glb_path) if output_glb_path else out_stl.with_suffix(".glb")
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    out_glb.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = get_triposr_model()

    print(f"1. Preprocessing input image for 360-degree 3D LRM: {in_p.name}...")
    pil_image = Image.open(str(in_p))

    # Preprocess: remove background & center subject
    if pil_image.mode != "RGBA":
        print("   Isolating foreground subject with alpha channel...")
        pil_image = remove_background(pil_image, rembg_session=None)
    
    pil_image = resize_foreground(pil_image, 0.85)

    # Composite onto neutral background for 3-channel RGB tokenizer
    rgba_np = np.array(pil_image).astype(np.float32) / 255.0
    if rgba_np.shape[2] == 4:
        rgb_composite = rgba_np[:, :, :3] * rgba_np[:, :, 3:4] + (1.0 - rgba_np[:, :, 3:4]) * 0.5
        pil_image = Image.fromarray((rgb_composite * 255.0).astype(np.uint8))
    else:
        pil_image = pil_image.convert("RGB")

    print("2. Running TripoSR Transformer Triplane 3D inference...")
    with torch.no_grad():
        scene_codes = model([pil_image], device=device)
        print(f"3. Extracting 360-degree isosurface mesh at {resolution}x{resolution}x{resolution} resolution...")
        meshes = model.extract_mesh(scene_codes, True, resolution=resolution, threshold=mc_threshold)

    mesh = meshes[0]

    # Convert to Trimesh representation for normalization and export
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    # Rotate upright if needed (TripoSR coordinate system standard)
    transform = np.array([
        [1,  0,  0, 0],
        [0,  0, -1, 0],
        [0,  1,  0, 0],
        [0,  0,  0, 1]
    ])
    mesh.apply_transform(transform)

    # Scale to unit bounding box centered at origin
    mesh.vertices -= mesh.center_mass
    max_ext = np.max(mesh.extents)
    if max_ext > 0:
        mesh.vertices /= max_ext

    # Clean degenerate geometry
    mesh.remove_unreferenced_vertices()
    try:
        mask = mesh.nondegenerate_faces()
        if mask is not None and len(mask) > 0:
            mesh.update_faces(mask)
    except Exception:
        pass
    
    # 4. Mesh Refinement - Floating artifact removal
    print("4. Refining mesh: Removing floating artifacts...")
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            mesh = max(components, key=lambda m: m.bounding_box.volume)
            print(f"   Kept largest component, discarded {len(components)-1} floating islands.")
    except Exception as e:
        print(f"   Component splitting failed: {e}")

    # 5. Mesh Refinement - Laplacian Smoothing
    print("5. Refining mesh: Applying Laplacian smoothing...")
    try:
        trimesh.smoothing.filter_laplacian(mesh, iterations=3)
    except Exception as e:
        print(f"   Smoothing failed: {e}")

    # 6. Mesh Refinement - Decimation
    target_faces = 50000
    if len(mesh.faces) > target_faces:
        print(f"6. Refining mesh: Decimating from {len(mesh.faces)} to ~{target_faces} faces...")
        try:
            ratio = float(target_faces) / float(len(mesh.faces))
            mesh = mesh.simplify_quadric_decimation(ratio)
        except Exception as e:
            print(f"   Decimation failed: {e}")

    mesh.fix_normals()

    # 7. Texture Baking (UV Unwrapping with xatlas)
    print("7. Baking vertex colors to texture map...")
    try:
        import xatlas
        import cv2
        
        if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
            vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
            
            # Map old colors to new unwrapped vertices
            old_colors = mesh.visual.vertex_colors[:, :3]
            new_colors = old_colors[vmapping]
            
            # Create a texture map
            texture_size = 1024
            texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
            
            for uv, color in zip(uvs, new_colors):
                x = np.clip(int(uv[0] * (texture_size - 1)), 0, texture_size - 1)
                y = np.clip(int((1.0 - uv[1]) * (texture_size - 1)), 0, texture_size - 1)
                texture[y, x] = color
                
            # Dilate to fill gaps
            kernel = np.ones((7,7), np.uint8)
            texture = cv2.dilate(texture, kernel, iterations=3)
            
            tex_img = Image.fromarray(texture)
            material = trimesh.visual.material.SimpleMaterial(image=tex_img)
            
            # Create a new mesh structure with the new vertices/faces/uvs
            new_mesh = trimesh.Trimesh(vertices=mesh.vertices[vmapping], faces=indices)
            new_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=tex_img, material=material)
            mesh = new_mesh
            print("   UV Unwrapping and texture baking successful.")
    except Exception as e:
        print(f"   UV generation error: {e}. Falling back to vertex colors.")

    print(f"8. Exporting proper 360-degree 3D model ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)...")
    mesh.export(str(out_stl))
    mesh.export(str(out_glb))
    print(f"   [SUCCESS] Proper 3D model saved: STL={out_stl}, GLB={out_glb}")

    return str(out_stl)


if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else "data/raw/frame_000.jpg"
    test_out = sys.argv[2] if len(sys.argv) > 2 else "data/output/triposr_model.stl"
    generate_triposr_3d_mesh(test_img, test_out)
