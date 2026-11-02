"""Preserve generated 360-degree appearance, with a source projection fallback."""
from pathlib import Path
import cv2
import numpy as np
import trimesh

SKIN_ROUGHNESS=0.6
FABRIC_ROUGHNESS=0.75
METAL_ROUGHNESS=0.25

def has_native_appearance(mesh):
    return bool(getattr(mesh.visual, 'defined', False))

def _read_image_bgr(path):
    return cv2.imread(str(path))

def _apply_vertex_colors(mesh, img_bgr):
    image=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    extents=np.maximum(mesh.extents,1e-8)
    uv=(mesh.vertices-mesh.bounds[0])/extents
    px=np.clip(uv[:,0]*(image.shape[1]-1),0,image.shape[1]-1).astype(np.float32)
    py=np.clip((1-uv[:,1])*(image.shape[0]-1),0,image.shape[0]-1).astype(np.float32)
    x0=px.astype(np.int64);y0=py.astype(np.int64)
    x1=np.minimum(x0+1,image.shape[1]-1);y1=np.minimum(y0+1,image.shape[0]-1)
    wx=(px-x0)[:,None];wy=(py-y0)[:,None]
    rgb=((1-wx)*(1-wy)*image[y0,x0]+wx*(1-wy)*image[y0,x1]+(1-wx)*wy*image[y1,x0]+wx*wy*image[y1,x1]).round().astype(np.uint8)
    mesh.visual.vertex_colors=np.column_stack((rgb,np.full(len(rgb),255,dtype=np.uint8)))

def project_texture_to_mesh(mesh_path,rgb_image_path,output_glb,use_uv_atlas=True,roughness=SKIN_ROUGHNESS):
    mesh=trimesh.load(str(mesh_path),force='mesh')
    if not has_native_appearance(mesh):
        image=_read_image_bgr(rgb_image_path)
        if image is None: raise ValueError(f'Cannot read texture source: {rgb_image_path}')
        _apply_vertex_colors(mesh,image)
    # GLB supports smooth vertex colors directly; a sparse atlas loses detail.
    out=Path(output_glb);out.parent.mkdir(parents=True,exist_ok=True)
    mesh.export(str(out),include_normals=True)
    return str(out)
