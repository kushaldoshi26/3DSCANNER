"""Point clouds sampled from reconstructed surfaces, never flat image pixels."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import trimesh

@lru_cache(maxsize=8)
def surface_cloud(path,mtime,size,count=16000):
    mesh=trimesh.load(path,force='mesh')
    if mesh.is_empty:raise ValueError('No reconstructed surface available')
    points,face_ids=trimesh.sample.sample_surface(mesh,count,seed=42)
    if mesh.visual.kind=='texture':
        vertex_colors=mesh.visual.to_color().vertex_colors[:,:3]
    else:vertex_colors=mesh.visual.vertex_colors[:,:3]
    tri=mesh.triangles[face_ids]
    weights=trimesh.triangles.points_to_barycentric(tri,points)
    colors=(vertex_colors[mesh.faces[face_ids]]*weights[:,:,None]).sum(axis=1).clip(0,255).astype(np.uint8)
    return {'positions':np.round(points,5).reshape(-1).tolist(),'colors':colors.reshape(-1).tolist(),
            'kind':'reconstructed_surface','estimated':True,'count':len(points)}

def get_cloud(output):
    output=Path(output)
    for name in ('final_model.glb','raw_ai_mesh.glb','native_model.glb'):
        path=output/name
        if path.exists():
            stat=path.stat()
            return surface_cloud(str(path),stat.st_mtime_ns,stat.st_size)
    path=output/'preview_cloud.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            p=data['points'];c=data['colors'];step=max(1,len(p)//16000)
            return {'positions':p[::step].round(5).reshape(-1).tolist(),'colors':c[::step].astype(np.uint8).reshape(-1).tolist(),
                    'kind':'estimated_depth_surface','estimated':True,'count':len(p[::step])}
    raise FileNotFoundError('3D points are still being reconstructed')

def depth_surface(rgba,depth):
    """A coloured foreground depth surface, without Poisson balloons or background texture."""
    import cv2
    mask=rgba[:,:,3]>127
    ys,xs=np.where(mask)
    if len(xs)<16:raise ValueError('No valid isolated foreground')
    d=np.asarray(depth,dtype=np.float32)
    if d.shape!=mask.shape or not np.isfinite(d[mask]).all():raise ValueError('Invalid depth estimate')
    low,high=np.percentile(d[mask],[2,98])
    if high-low<1e-6:raise ValueError('Depth model returned a flat or invalid prediction')
    d=np.clip((d-low)/(high-low),0,1).astype(np.float32)
    d=cv2.bilateralFilter(d,5,.08,3)
    centre=np.array([(xs.min()+xs.max())/2,(ys.min()+ys.max())/2])
    scale=2/max(np.ptp(xs)+1,np.ptp(ys)+1)
    points=np.column_stack(((xs-centre[0])*scale,(centre[1]-ys)*scale,d[ys,xs]*.55)).astype(np.float32)
    colors=rgba[ys,xs,:3]
    index=np.full(mask.shape,-1,np.int64);index[ys,xs]=np.arange(len(points))
    a=index[:-1,:-1];b=index[:-1,1:];c=index[1:,:-1];e=index[1:,1:]
    f1=np.column_stack((a.ravel(),c.ravel(),b.ravel()));f2=np.column_stack((b.ravel(),c.ravel(),e.ravel()))
    faces=np.concatenate((f1,f2));faces=faces[(faces>=0).all(axis=1)]
    # Reject discontinuity bridges instead of stretching triangles across gaps.
    z=points[faces,2];faces=faces[np.ptp(z,axis=1)<.09]
    mesh=trimesh.Trimesh(points,faces,vertex_colors=colors,process=False)
    mesh.remove_unreferenced_vertices();mesh.fix_normals()
    return mesh
