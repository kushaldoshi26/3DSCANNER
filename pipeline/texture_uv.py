"""
UV Texture pipeline — separates geometry from appearance (thin wrapper).

The actual UV-unwrap + texture-atlas + PBR-material logic lives in
`pipeline.texture_projector.project_texture_to_mesh`, which is the single
authoritative texture implementation (avoiding duplicate logic). This module
keeps the spec's `apply_uv_texture(mesh_path, image_path, output_glb)` API so
callers (and any older tests) continue to work.
"""

from typing import Optional
from pathlib import Path


def apply_uv_texture(
    mesh_path: str,
    image_path: str,
    output_glb: str,
    uv_fallback: bool = True,
) -> str:
    """
    Textures the mesh from `image_path`, preferring a proper UV atlas and
    falling back to vertex-color projection when UV unwrapping is infeasible.
    Returns the output GLB path.
    """
    out_path = Path(output_glb)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from pipeline.texture_projector import project_texture_to_mesh
    return project_texture_to_mesh(
        mesh_path,
        image_path,
        str(out_path),
        use_uv_atlas=uv_fallback,
    )
