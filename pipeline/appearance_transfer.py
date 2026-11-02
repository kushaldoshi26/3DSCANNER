"""Transfer preview colours to an unchanged full-resolution surface in batches."""
import numpy as np
import trimesh
from trimesh.visual.color import uv_to_color

def transfer_preview_appearance(full,preview,progress=None,batch_size=1000):
    if full.is_empty or preview.is_empty:raise ValueError('Empty appearance transfer mesh')
    extent=float(full.extents.max())
    if extent<=0 or np.max(np.abs(full.bounds-preview.bounds))>.01*extent:
        raise ValueError('Full shape and texture preview are not aligned')
    image=getattr(preview.visual.material,'baseColorTexture',None)
    if image is None:image=getattr(preview.visual.material,'image',None)
    if image is None or preview.visual.uv is None:raise ValueError('Painted preview has no texture')
    colours=np.empty((len(full.vertices),4),np.uint8);distances=np.empty(len(full.vertices))
    for start in range(0,len(full.vertices),batch_size):
        stop=min(start+batch_size,len(full.vertices))
        closest,distance,tri=trimesh.proximity.closest_point(preview,full.vertices[start:stop])
        bary=trimesh.triangles.points_to_barycentric(preview.triangles[tri],closest)
        uv=(preview.visual.uv[preview.faces[tri]]*bary[:,:,None]).sum(axis=1)
        colours[start:stop]=uv_to_color(uv,image);distances[start:stop]=distance
        if progress and (start%(batch_size*20)==0 or stop==len(full.vertices)):
            progress(stop/len(full.vertices))
    p99=float(np.quantile(distances,.99))
    if not np.isfinite(distances).all() or p99>.01*extent:
        raise ValueError('Texture correspondence is too distant; original shape retained')
    result=full.copy();result.visual=trimesh.visual.ColorVisuals(mesh=result,vertex_colors=colours)
    return result,{'full_shape_faces':len(full.faces),'paint_preview_faces':len(preview.faces),
                   'geometry_preserved':True,'colour_distance_p99_relative':p99/extent}
