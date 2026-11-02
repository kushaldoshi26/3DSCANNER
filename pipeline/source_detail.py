"""Project source detail onto visible faces; preserve inferred rear appearance.

This changes appearance only. It does not claim measured facial geometry or
recover unseen surfaces. A silhouette/color search estimates the source view.
"""
from pathlib import Path
import numpy as np
import cv2
import trimesh
from PIL import Image


def _warp_uv(coords,flow,batch_size=16000):
    if flow is None:return coords
    output=np.empty_like(coords)
    for start in range(0,len(coords),batch_size):
        part=coords[start:start+batch_size]
        xy=np.column_stack((part[:,0]*255,(1-part[:,1])*255)).astype(np.float32)
        delta=cv2.remap(flow,xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR).reshape(-1,2)
        output[start:start+len(part)]=part+delta*np.array([1/255,-1/255])
    return output

def _source_image(path, size=256, max_texture_size=4096):
    image=Image.open(path).convert('RGBA')
    alpha=np.asarray(image)[:,:,3]
    y,x=np.where(alpha>127)
    if len(x)<16: raise ValueError('No isolated source foreground')
    crop=image.crop((int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1))
    side=int(np.ceil(max(crop.size)/.85))
    result=Image.new('RGBA',(side,side),(127,127,127,0))
    result.paste(crop,((side-crop.width)//2,(side-crop.height)//2))
    result.thumbnail((max_texture_size,max_texture_size),Image.Resampling.LANCZOS)
    return result, np.asarray(result.resize((size,size),Image.Resampling.LANCZOS))

def _basis(yaw,elevation):
    a,e=np.radians([yaw,elevation])
    d=np.array([np.sin(a)*np.cos(e),np.sin(e),-np.cos(a)*np.cos(e)])
    right=np.cross([0,1,0],d);right/=np.linalg.norm(right)
    up=np.cross(d,right)
    return np.array([right,up,d])

def _fit(vertices,basis):
    projected=vertices@basis.T
    centre=(projected.max(axis=0)+projected.min(axis=0))/2
    scale=1.7/max(np.ptp(projected[:,1]),1e-8)
    return projected-centre,centre,scale

def enhance_source_detail(mesh_path,foreground_path,output_path,debug_dir=None,*,infer_rear_fabric=False,allow_textured=False,align_photo=False,visibility_size=1024):
    import moderngl
    mesh=trimesh.load(str(mesh_path),force='mesh')
    if mesh.is_empty or mesh.visual.kind not in ('vertex','texture'): return {'applied':False,'reason':'Native appearance required'}
    if mesh.visual.kind=='texture' and not allow_textured: return {'applied':False,'reason':'Existing texture preserved; source projection requires explicit opt-in'}
    image,target=_source_image(foreground_path)
    target_mask=target[:,:,3]>127
    target_rgb=target[:,:,:3].astype(np.float32)/255
    vertices=np.asarray(mesh.vertices,dtype=np.float32)
    native_colors=mesh.visual.to_color().vertex_colors if mesh.visual.kind=='texture' else mesh.visual.vertex_colors
    colors=np.asarray(native_colors[:,:3],dtype=np.float32)/255
    ctx=moderngl.create_standalone_context()
    resources=[]
    try:
        program=ctx.program(vertex_shader='''#version 330
        in vec3 in_pos;in vec3 in_color;uniform mat3 basis;uniform vec3 centre;uniform float scale;out vec3 color;
        void main(){vec3 p=basis*in_pos-centre;gl_Position=vec4(p.xy*scale,-p.z*.9,1.0);color=in_color;}''',fragment_shader='''#version 330
        in vec3 color;out vec4 frag;void main(){frag=vec4(color,1.0);}''')
        vb=ctx.buffer(np.column_stack((vertices,colors)).astype('f4').tobytes());ib=ctx.buffer(np.asarray(mesh.faces,dtype='i4').tobytes())
        vao=ctx.vertex_array(program,[(vb,'3f 3f','in_pos','in_color')],ib)
        texture=ctx.texture((256,256),4);depth=ctx.depth_texture((256,256));fbo=ctx.framebuffer([texture],depth)
        resources.extend([vao,vb,ib,fbo,texture,depth,program]);fbo.use();ctx.enable(moderngl.DEPTH_TEST)
        def render(basis):
            _,centre,scale=_fit(vertices,basis)
            program['basis'].write(np.asarray(basis.T,dtype='f4').tobytes());program['centre'].value=tuple(centre);program['scale'].value=float(scale)
            fbo.clear(0,0,0,0,depth=1);vao.render()
            pixels=np.frombuffer(texture.read(),np.uint8).reshape(256,256,4)[::-1].copy()
            return pixels,centre,scale
        best=None
        for yaw in range(-180,180,15):
            for elevation in (-10,0,10):
                basis=_basis(yaw,elevation);pixels,centre,scale=render(basis)
                mask=pixels[:,:,3]>0;overlap=mask&target_mask
                iou=overlap.sum()/max((mask|target_mask).sum(),1)
                error=np.abs(pixels[:,:,:3].astype(np.float32)/255-target_rgb)[overlap].mean() if overlap.any() else 1
                score=.7*iou+.3*(1-error)
                if best is None or score>best[0]:best=(float(score),basis,yaw,elevation,float(iou))
        score,basis,yaw,elevation,iou=best
        pixels,centre,scale=render(basis)
        # Exact raster visibility avoids native-colour speckles on small/steep
        # facial triangles where a low-resolution centroid depth test fails.
        id_program=ctx.program(vertex_shader='''#version 330
        in vec3 in_pos;uniform mat3 basis;uniform vec3 centre;uniform float scale;
        void main(){vec3 p=basis*in_pos-centre;gl_Position=vec4(p.xy*scale,-p.z*.9,1.0);}''',fragment_shader='''#version 330
        layout(location=0) out uint face_id;void main(){face_id=uint(gl_PrimitiveID)+1u;}''')
        id_vao=ctx.vertex_array(id_program,[(vb,'3f 12x','in_pos')],ib)
        ids=ctx.texture((visibility_size,visibility_size),1,dtype='u4');id_depth=ctx.depth_texture((visibility_size,visibility_size));id_fbo=ctx.framebuffer([ids],id_depth)
        resources.extend([id_vao,id_program,id_fbo,ids,id_depth]);id_fbo.use();id_fbo.clear(depth=1)
        id_program['basis'].write(np.asarray(basis.T,dtype='f4').tobytes());id_program['centre'].value=tuple(centre);id_program['scale'].value=float(scale)
        id_vao.render()
        visible_ids=np.unique(np.frombuffer(ids.read(),np.uint32));visible_ids=visible_ids[visible_ids>0]-1
    finally:
        for resource in resources:resource.release()
        ctx.release()
    if iou<.65: return {'applied':False,'reason':'Source silhouette does not align safely','silhouette_iou':iou}
    flow=None;flow_valid=None
    if align_photo:
        first=cv2.cvtColor(pixels[:,:,:3],cv2.COLOR_RGB2GRAY)
        second=cv2.cvtColor(target[:,:,:3],cv2.COLOR_RGB2GRAY)
        first[pixels[:,:,3]==0]=127;second[target[:,:,3]<128]=127
        solver=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow=solver.calc(first,second,None)
        reverse=solver.calc(second,first,None)
        gy,gx=np.mgrid[:256,:256].astype(np.float32)
        mx=gx+flow[:,:,0];my=gy+flow[:,:,1]
        backward=cv2.remap(reverse,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=1000)
        alpha=cv2.remap(target[:,:,3],mx,my,cv2.INTER_NEAREST)
        flow_valid=(np.linalg.norm(flow+backward,axis=2)<1.5)&(np.linalg.norm(flow,axis=2)<32)&(alpha>220)
    projected=vertices@basis.T-centre
    uv=np.column_stack((projected[:,0]*scale*.5+.5,projected[:,1]*scale*.5+.5))
    visible=np.zeros(len(mesh.faces),bool);visible[visible_ids]=True
    # Never paint background pixels or hidden faces onto a reconstructed rear.
    tx=np.clip((uv[:,0]*255).astype(int),0,255);ty=np.clip(((1-uv[:,1])*255).astype(int),0,255)
    inside=((uv>=0)&(uv<=1)).all(axis=1)&(target[ty,tx,3]>220)
    facing=(mesh.face_normals@basis[2])>.3
    selected=visible&inside[mesh.faces].all(axis=1)&facing
    if flow_valid is not None:selected&=flow_valid[ty,tx][mesh.faces].all(axis=1)
    if selected.sum()<100: return {'applied':False,'reason':'Too few safely visible faces'}
    front=mesh.submesh([np.flatnonzero(selected)],append=True,repair=False)
    fp=np.asarray(front.vertices)@basis.T-centre
    front_uv=np.column_stack((fp[:,0]*scale*.5+.5,fp[:,1]*scale*.5+.5))
    front_uv=_warp_uv(front_uv,flow)
    front.visual=trimesh.visual.texture.TextureVisuals(uv=front_uv,material=trimesh.visual.material.PBRMaterial(baseColorTexture=image.convert('RGB'),metallicFactor=0,roughnessFactor=1,doubleSided=False))
    scene=trimesh.Scene();scene.add_geometry(front,node_name='Source photo detail')
    from pipeline.fabric_detail import infer_fabric
    # Face detection also matches statues; never invent repeated rear clothing by default.
    fabric=infer_fabric(image) if infer_rear_fabric else None
    fabric_faces=np.zeros(len(mesh.faces),bool)
    if fabric:
        left,bottom,right,top=fabric['bounds']
        face_uv=uv[mesh.faces].mean(axis=1)
        centres=projected[mesh.faces].mean(axis=1)
        fabric_faces=(~selected)&(face_uv[:,0]>left)&(face_uv[:,0]<right)&(face_uv[:,1]>bottom)&(face_uv[:,1]<top)
        fabric_faces&=(mesh.face_normals@basis[2])<.2
        fabric_faces&=centres[:,2]<np.quantile(projected[:,2],.5)
        if fabric_faces.any():
            cloth=mesh.submesh([np.flatnonzero(fabric_faces)],append=True,repair=False)
            cp=np.asarray(cloth.vertices)@basis.T-centre
            cloth_uv=np.column_stack((cp[:,0]*scale*.5+.5,cp[:,1]*scale*.5+.5))
            cloth.visual=trimesh.visual.texture.TextureVisuals(uv=cloth_uv,material=trimesh.visual.material.PBRMaterial(baseColorTexture=fabric['texture'],metallicFactor=0,roughnessFactor=1))
            scene.add_geometry(cloth,node_name='Inferred rear fabric pattern')
    remaining=(~selected)&(~fabric_faces)
    if remaining.any():scene.add_geometry(mesh.submesh([np.flatnonzero(remaining)],append=True,repair=False),node_name='Inferred unseen surfaces')
    scene.metadata['source_view']=basis[2].tolist()
    scene.metadata['source_detail']={'appearance_only':True,'geometry_unchanged':True,'silhouette_iou':round(iou,4),'unseen_surfaces':'inferred','back_pattern_inferred':bool(fabric_faces.any())}
    scene.export(str(output_path))
    report={'applied':True,'appearance_only':True,'yaw':yaw,'elevation':elevation,'silhouette_iou':round(iou,4),'source_textured_faces':int(selected.sum()),'faces':len(mesh.faces),'texture_resolution':list(image.size),'photo_alignment':'forward_backward_checked_optical_flow' if align_photo else 'global_projection','back_texture':fabric['report'] if fabric and fabric_faces.any() else None}
    if debug_dir:
        out=Path(debug_dir);out.mkdir(parents=True,exist_ok=True)
        Image.fromarray(pixels).save(out/'aligned_native.png');image.save(out/'source_texture.png')
    return report
