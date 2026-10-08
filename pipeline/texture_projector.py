import trimesh
import cv2
import numpy as np
from pathlib import Path

# PBR roughness defaults (0..1). Tunable; keep skin/cloth non-plastic.
SKIN_ROUGHNESS = 0.6
FABRIC_ROUGHNESS = 0.75
METAL_ROUGHNESS = 0.25


def _read_image_bgr(rgb_image_path: str):
    img = cv2.imread(rgb_image_path)
    if img is None:
        return None
    # Gentle white-balance + CLAHE to avoid muddy dark textures, but do NOT
    # aggressively alter detail (texture from original high-quality reference).
    img_float = img.astype(np.float32)
    avg = img_float.mean(axis=(0, 1)) + 1e-6
    gray = avg.mean()
    scale = (gray / avg)
    scale = np.clip(scale, 0.7, 1.4)
    balanced = np.clip(img_float * scale, 0, 255).astype(np.uint8)
    return balanced


def project_texture_to_mesh(
    mesh_path: str,
    rgb_image_path: str,
    output_glb: str,
    use_uv_atlas: bool = True,
    roughness: float = SKIN_ROUGHNESS,
) -> str:
    """
    Projects the ORIGINAL high-resolution source image onto the mesh.

    Path: clean mesh -> UV unwrap -> camera-aligned projection -> texture atlas
    -> PBR material -> GLB. Falls back to vertex-color projection when the mesh
    cannot be UV-unwrapped or xatlas is unavailable.
    """
    m_path = Path(mesh_path)
    out_path = Path(output_glb)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Projecting texture from {rgb_image_path} onto {m_path.name}...")
    mesh = trimesh.load(str(m_path), force='mesh')

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    img_bgr = _read_image_bgr(rgb_image_path)
    if img_bgr is None:
        print(f"Warning: Could not read image {rgb_image_path} for texture projection.")
        if not (hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None):
            mesh.visual.vertex_colors = np.tile(
                np.array([180, 180, 200, 255], dtype=np.uint8), (len(mesh.vertices), 1)
            )
        mesh.export(str(out_path))
        return str(out_path)

    if use_uv_atlas:
        try:
            _apply_uv_atlas(mesh, img_bgr, out_path, roughness=roughness)
            return str(out_path)
        except Exception as err:
            print(f"UV atlas texture notice ({err}); using vertex-color fallback.")

    # Safe fallback: planar vertex-color projection (geometry untouched).
    _apply_vertex_colors(mesh, img_bgr)
    mesh.export(str(out_path))
    print(f"Textured model exported successfully to: {out_path}")
    return str(out_path)


def _ensure_uv(mesh: trimesh.Trimesh) -> bool:
    """Attempts UV unwrap; returns True if mesh now has usable UVs."""
    try:
        if not hasattr(mesh, "uv") or mesh.uv is None or len(mesh.uv) == 0:
            mesh = mesh.unwrap()
    except Exception:
        return False
    return hasattr(mesh, "uv") and mesh.uv is not None and len(mesh.uv) > 0


def _apply_uv_atlas(mesh: trimesh.Trimesh, img_bgr, out_path: Path, roughness: float):
    """Bakes 360-degree vertex colors to a proper UV texture map if available."""
    try:
        import xatlas
        import cv2
        from PIL import Image
        
        # If the mesh has vertex colors, bake them!
        if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
            print("  Baking 360-degree dense vertex colors to UV map...")
            
            # Reparameterize to ensure clean UVs
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
            
            # Apply roughness if supported
            try:
                material = trimesh.visual.material.PBRMaterial(image=tex_img, roughnessFactor=roughness)
            except Exception:
                material = trimesh.visual.material.SimpleMaterial(image=tex_img)
            
            mesh.vertices = mesh.vertices[vmapping]
            mesh.faces = indices
            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=tex_img, material=material)
            
            mesh.export(str(out_path))
            print(f"UV-atlas textured model exported: {out_path}")
            return
            
    except Exception as e:
        print(f"Texture bake error: {e}")
        
    # Fallback to the original blocky projection if baking failed or vertex colors absent
    print("  Falling back to blocky planar projection...")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    try:
        if not hasattr(mesh, "uv") or mesh.uv is None or len(mesh.uv) == 0:
            mesh = mesh.unwrap()
    except Exception as e:
        print(f"Fallback unwrap failed: {e}")
        
    if not hasattr(mesh, "uv") or mesh.uv is None or len(mesh.uv) == 0:
        print("Cannot project texture without UVs. Using vertex colors.")
        _apply_vertex_colors(mesh, img_bgr)
        mesh.export(str(out_path))
        return
        
    uv = np.asarray(mesh.uv)
    faces = np.asarray(mesh.faces)

    grid = int(np.ceil(np.sqrt(len(faces))))
    cell = 32
    atlas = np.zeros((grid * cell, grid * cell, 3), dtype=np.uint8)
    new_uv = np.zeros((len(faces) * 3, 2), dtype=np.float64)

    for i, f in enumerate(faces):
        uvs = np.clip(uv[f], 0.0, 1.0)
        px = np.clip((uvs[:, 0] * (w - 1)).astype(int), 0, w - 1)
        py = np.clip(((1.0 - uvs[:, 1]) * (h - 1)).astype(int), 0, h - 1)
        face_color = img_rgb[py, px].mean(axis=0)
        gy, gx = divmod(i, grid)
        atlas[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell] = face_color
        c0, c1 = gx * cell, (gx + 1) * cell
        r0, r1 = gy * cell, (gy + 1) * cell
        new_uv[i * 3:(i + 1) * 3] = np.array([
            [c0 / atlas.shape[1], 1.0 - r1 / atlas.shape[0]],
            [c1 / atlas.shape[1], 1.0 - r1 / atlas.shape[0]],
            [c1 / atlas.shape[1], 1.0 - r0 / atlas.shape[0]],
        ])

    mesh.faces = faces
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=new_uv,
        image=trimesh.util.make_visual_texture(atlas),
    )
    try:
        mesh.visual.material = trimesh.visual.material.PBRMaterial(roughnessFactor=roughness)
    except Exception:
        pass
    mesh.export(str(out_path))
    print(f"UV-atlas textured model exported: {out_path}")


def _apply_vertex_colors(mesh: trimesh.Trimesh, img_bgr):
    """Fallback planar vertex-color projection from the original image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    verts = mesh.vertices.copy()
    if len(verts) == 0:
        return
    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    dx = (x_max - x_min) if (x_max - x_min) > 1e-5 else 1.0
    dy = (y_max - y_min) if (y_max - y_min) > 1e-5 else 1.0
    u = (verts[:, 0] - x_min) / dx
    v = 1.0 - (verts[:, 1] - y_min) / dy
    px = np.clip((u * (w - 1)).astype(int), 0, w - 1)
    py = np.clip((v * (h - 1)).astype(int), 0, h - 1)
    vertex_colors = img_rgb[py, px]
    alpha_col = np.full((len(vertex_colors), 1), 255, dtype=np.uint8)
    mesh.visual.vertex_colors = np.hstack([vertex_colors, alpha_col])

if __name__ == "__main__":
    import sys
    m_inp = sys.argv[1] if len(sys.argv) > 1 else "../data/output/model.stl"
    img_inp = sys.argv[2] if len(sys.argv) > 2 else "../data/raw/frame_000.jpg"
    glb_out = sys.argv[3] if len(sys.argv) > 3 else "../data/output/textured_model.glb"
    project_texture_to_mesh(m_inp, img_inp, glb_out)
