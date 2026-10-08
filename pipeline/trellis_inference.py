"""
TRELLIS (Microsoft) 3D Asset Generation Pipeline
Unified Structured LATent (SLAT) model for ultra-high geometric fidelity.

Produces intricate watertight 3D meshes and textured GLB models directly from a single photo
with native support for NVIDIA GPUs (e.g. RTX 5060 Ti 16GB).
"""

import os
import torch
import numpy as np
import trimesh
from PIL import Image
from pathlib import Path

TRELLIS_PIPELINE = None

def get_trellis_pipeline():
    """Initializes and caches TRELLIS 3D Pipeline on CUDA/CPU."""
    global TRELLIS_PIPELINE
    if TRELLIS_PIPELINE is None:
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"Loading Microsoft TRELLIS model (microsoft/TRELLIS-image-large) on {device}...")
            TRELLIS_PIPELINE = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
            if hasattr(TRELLIS_PIPELINE, "to"):
                TRELLIS_PIPELINE.to(device)
            print("TRELLIS pipeline loaded successfully.")
        except Exception as e:
            print(f"TRELLIS native package notice: {e}. Fallback pipeline will be used.")
            TRELLIS_PIPELINE = "FALLBACK"
    return TRELLIS_PIPELINE


def generate_trellis_3d_mesh(
    image_path: str,
    output_stl_path: str,
    output_glb_path: str = None
) -> str:
    """
    Generates a high-precision 3D mesh model using Microsoft TRELLIS.
    
    Args:
        image_path: Path to input 2D image
        output_stl_path: Output STL path
        output_glb_path: Output GLB path (optional)
        
    Returns:
        Path to output STL
    """
    in_p = Path(image_path)
    out_stl = Path(output_stl_path)
    out_glb = Path(output_glb_path) if output_glb_path else out_stl.with_suffix(".glb")
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    out_glb.parent.mkdir(parents=True, exist_ok=True)

    print(f"1. Microsoft TRELLIS: Ingesting image {in_p.name}...")
    pipeline = get_trellis_pipeline()

    if pipeline is not None and pipeline != "FALLBACK":
        try:
            print("2. Running TRELLIS SLAT Latent 3D inference...")
            pil_img = Image.open(str(in_p)).convert("RGB")
            outputs = pipeline.run(image=pil_img)
            mesh = outputs['mesh'][0]
            
            # Export via trimesh or direct format
            if hasattr(mesh, 'export'):
                mesh.export(str(out_glb))
                mesh.export(str(out_stl))
            else:
                # Convert if custom mesh object, ensuring tensors are moved to CPU and numpy
                verts = mesh.vertices.detach().cpu().numpy() if hasattr(mesh.vertices, 'detach') else mesh.vertices
                faces = mesh.faces.detach().cpu().numpy() if hasattr(mesh.faces, 'detach') else mesh.faces
                t_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                t_mesh.export(str(out_stl))
                t_mesh.export(str(out_glb))

            print(f"   [SUCCESS] TRELLIS mesh exported: STL={out_stl}, GLB={out_glb}")
            return str(out_stl)
        except Exception as err:
            print(f"TRELLIS inference runtime issue: {err}.")
            raise RuntimeError(f"TRELLIS inference failed: {err}") from err
    else:
        raise RuntimeError("TRELLIS pipeline not installed or unavailable.")


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/raw/frame_000.jpg"
    outp = sys.argv[2] if len(sys.argv) > 2 else "data/output/trellis_model.stl"
    generate_trellis_3d_mesh(inp, outp)
