"""
Hunyuan3D 2.0 (Tencent) 3D Synthesis Pipeline
Two-Stage Generation:
1. Shape Generation (Hunyuan3D-DiT Flow Matching)
2. Texture Synthesis (Hunyuan3D-Paint)

Delivers ultra-vibrant, high-resolution textures and crisp geometric detail.
"""

import os
import torch
import numpy as np
import trimesh
from PIL import Image
from pathlib import Path

SHAPE_PIPELINE = None
PAINT_PIPELINE = None

def get_hunyuan3d_pipelines():
    """Initializes and caches Hunyuan3D-2 Shape and Paint pipelines."""
    global SHAPE_PIPELINE, PAINT_PIPELINE
    if SHAPE_PIPELINE is None:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"Loading Tencent Hunyuan3D 2.0 pipelines on {device}...")
            
            SHAPE_PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
            if hasattr(SHAPE_PIPELINE, "to"):
                SHAPE_PIPELINE.to(device)

            PAINT_PIPELINE = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
            if hasattr(PAINT_PIPELINE, "to"):
                PAINT_PIPELINE.to(device)

            print("Hunyuan3D 2.0 pipelines loaded successfully.")
        except Exception as e:
            print(f"Hunyuan3D 2.0 native package notice: {e}. Fallback pipeline will be used.")
            SHAPE_PIPELINE = "FALLBACK"
            PAINT_PIPELINE = "FALLBACK"
    return SHAPE_PIPELINE, PAINT_PIPELINE


def generate_hunyuan3d_mesh(
    image_path: str,
    output_stl_path: str,
    output_glb_path: str = None,
    paint_texture: bool = True
) -> str:
    """
    Generates a high-fidelity 3D mesh with optional high-res painted texture using Hunyuan3D 2.0.
    
    Args:
        image_path: Path to input 2D image
        output_stl_path: Output STL path
        output_glb_path: Output GLB path (optional)
        paint_texture: Whether to run stage 2 texture synthesis
        
    Returns:
        Path to output STL
    """
    in_p = Path(image_path)
    out_stl = Path(output_stl_path)
    out_glb = Path(output_glb_path) if output_glb_path else out_stl.with_suffix(".glb")
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    out_glb.parent.mkdir(parents=True, exist_ok=True)

    print(f"1. Tencent Hunyuan3D 2.0: Processing image {in_p.name}...")
    shape_pipe, paint_pipe = get_hunyuan3d_pipelines()

    if shape_pipe is not None and shape_pipe != "FALLBACK":
        try:
            print("2. Stage 1: Generating bare geometry with Hunyuan3D-DiT Flow Matching...")
            bare_mesh = shape_pipe(image=str(in_p))[0]

            print("2.5 Refining bare geometry (Artifact removal, Smoothing, Decimation)...")
            try:
                if not isinstance(bare_mesh, trimesh.Trimesh):
                    bare_mesh = trimesh.Trimesh(vertices=bare_mesh.vertices, faces=bare_mesh.faces)
                
                components = bare_mesh.split(only_watertight=False)
                if len(components) > 1:
                    bare_mesh = max(components, key=lambda m: m.bounding_box.volume)
                    print(f"    Discarded {len(components)-1} floating islands.")

                trimesh.smoothing.filter_laplacian(bare_mesh, iterations=3)
                print("    Applied 3-iteration Laplacian smoothing.")

                target_faces = 50000
                if len(bare_mesh.faces) > target_faces:
                    print(f"    Decimating from {len(bare_mesh.faces)} to ~{target_faces} faces...")
                    ratio = float(target_faces) / float(len(bare_mesh.faces))
                    bare_mesh = bare_mesh.simplify_quadric_decimation(ratio)
                    
                bare_mesh.fix_normals()
            except Exception as ref_err:
                print(f"    Refinement warning: {ref_err}")

            final_mesh = bare_mesh
            if paint_texture and paint_pipe is not None and paint_pipe != "FALLBACK":
                print("3. Stage 2: Painting high-resolution texture with Hunyuan3D-Paint...")
                textured_mesh = paint_pipe(bare_mesh, image=str(in_p))[0]
                final_mesh = textured_mesh

            if hasattr(final_mesh, 'export'):
                final_mesh.export(str(out_glb))
                final_mesh.export(str(out_stl))
            else:
                t_mesh = trimesh.Trimesh(vertices=final_mesh.vertices, faces=final_mesh.faces)
                t_mesh.export(str(out_stl))
                t_mesh.export(str(out_glb))

            print(f"   [SUCCESS] Hunyuan3D 2.0 mesh exported: STL={out_stl}, GLB={out_glb}")
            return str(out_stl)
        except Exception as err:
            print(f"Hunyuan3D inference notice: {err}.")
            raise RuntimeError(f"Hunyuan3D inference failed: {err}") from err
    else:
        raise RuntimeError("Hunyuan3D pipeline not installed or unavailable.")


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/raw/frame_000.jpg"
    outp = sys.argv[2] if len(sys.argv) > 2 else "data/output/hunyuan3d_model.stl"
    generate_hunyuan3d_mesh(inp, outp)
