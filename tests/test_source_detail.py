from pathlib import Path
import importlib.util
import numpy as np
import trimesh
from PIL import Image,ImageDraw

spec=importlib.util.spec_from_file_location('source_detail',Path(__file__).resolve().parent.parent/'pipeline'/'source_detail.py')
detail=importlib.util.module_from_spec(spec);spec.loader.exec_module(detail)

def test_foreground_border_and_white_detail_survive(tmp_path):
    p=tmp_path/'source.png';im=Image.new('RGBA',(80,120));ImageDraw.Draw(im).rectangle((10,0,69,119),fill=(255,255,255,255));im.save(p)
    image,small=detail._source_image(p)
    assert image.getbbox() is not None
    assert (np.asarray(image)[:,:,3]>0).sum()==60*120
    assert small[:,:,3].max()==255

def test_projection_changes_appearance_without_geometry(tmp_path,monkeypatch):
    import pipeline.fabric_detail as fabric
    def forbidden(*args,**kwargs):
        raise AssertionError('Automatic rear fabric inference must stay disabled')
    monkeypatch.setattr(fabric,'infer_fabric',forbidden)
    mesh=trimesh.creation.icosphere(subdivisions=3)
    mesh.visual.vertex_colors=np.tile([180,130,80,255],(len(mesh.vertices),1))
    native=tmp_path/'native.glb';mesh.export(native)
    image=Image.new('RGBA',(256,256));draw=ImageDraw.Draw(image);draw.ellipse((8,8,248,248),fill=(180,130,80,255));draw.rectangle((110,40,140,215),fill=(240,50,40,255));image.save(tmp_path/'source.png')
    output=tmp_path/'detail.glb'
    report=detail.enhance_source_detail(native,tmp_path/'source.png',output)
    assert report['applied'] and report['appearance_only']
    assert report['back_texture'] is None
    scene=trimesh.load(output,force='scene')
    assert scene.metadata['source_detail']['appearance_only']
    assert len(scene.geometry)==2
    assert sum(len(g.faces) for g in scene.geometry.values())==len(mesh.faces)
    np.testing.assert_allclose(scene.bounds,mesh.bounds,atol=1e-6)
    np.testing.assert_allclose(sum(g.area for g in scene.geometry.values()),mesh.area,rtol=1e-6)
    front=next(g for g in scene.geometry.values() if g.visual.kind=='texture')
    assert front.visual.material.baseColorTexture is not None
    assert any(g.visual.kind=='vertex' for g in scene.geometry.values())

def test_existing_texture_is_not_overwritten(tmp_path):
    mesh=trimesh.creation.box();mesh.visual=trimesh.visual.texture.TextureVisuals(uv=np.zeros((len(mesh.vertices),2)),image=Image.new('RGB',(8,8),'red'))
    path=tmp_path/'textured.glb';mesh.export(path)
    report=detail.enhance_source_detail(path,tmp_path/'absent.png',tmp_path/'result.glb')
    assert not report['applied']


def test_high_resolution_photo_is_not_reduced_to_2048(tmp_path):
    path=tmp_path/'large.png'
    Image.new('RGBA',(2200,2400),(180,130,80,255)).save(path)
    texture,_=detail._source_image(path)
    assert max(texture.size)>2048
    assert max(texture.size)<=4096

def test_opt_in_photo_alignment_preserves_textured_rear_and_geometry(tmp_path):
    mesh=trimesh.creation.icosphere(subdivisions=3)
    vertices=np.asarray(mesh.vertices)
    uv=np.column_stack((np.arctan2(vertices[:,0],vertices[:,2])/(2*np.pi)+.5,np.arcsin(vertices[:,1])/np.pi+.5))
    mesh.visual=trimesh.visual.texture.TextureVisuals(uv=uv,image=Image.new('RGB',(32,32),(180,130,80)))
    native=tmp_path/'native.glb';mesh.export(native)
    image=Image.new('RGBA',(256,256));ImageDraw.Draw(image).ellipse((8,8,248,248),fill=(180,130,80,255));image.save(tmp_path/'source.png')
    output=tmp_path/'detail.glb'
    report=detail.enhance_source_detail(native,tmp_path/'source.png',output,allow_textured=True,align_photo=True)
    assert report['applied'] and report['appearance_only']
    assert report['photo_alignment']=='forward_backward_checked_optical_flow'
    assert report['back_texture'] is None
    scene=trimesh.load(output,force='scene')
    assert len(scene.geometry)==2
    assert sum(len(g.faces) for g in scene.geometry.values())==len(mesh.faces)
    np.testing.assert_allclose(scene.bounds,mesh.bounds,atol=1e-6)
    np.testing.assert_allclose(sum(g.area for g in scene.geometry.values()),mesh.area,rtol=1e-6)
    assert all(g.visual.kind=='texture' for g in scene.geometry.values())
    assert not scene.metadata['source_detail']['back_pattern_inferred']


def test_photo_warp_handles_more_than_opencv_short_limit():
    coords=np.full((40000,2),.5,dtype=np.float64)
    flow=np.empty((256,256,2),np.float32);flow[:]=[1,-1]
    warped=detail._warp_uv(coords,flow)
    np.testing.assert_allclose(warped,coords+1/255)
    np.testing.assert_array_equal(coords,np.full((40000,2),.5))
