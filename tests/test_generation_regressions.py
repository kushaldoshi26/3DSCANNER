from pathlib import Path
import numpy as np
import pytest
import trimesh
from PIL import Image
from pipeline.cleanup_mesh import cleanup_generated_mesh, _face_aware_smooth
from pipeline.mesh_quality import validate_mesh, surface_roughness
from pipeline.preprocess_input import preprocess_for_generation
from pipeline.texture_projector import project_texture_to_mesh
from pipeline.ai import engine_manager as em

def test_alpha_red_subject_and_edge_pixels_survive(tmp_path):
    rgba=np.zeros((40,60,4),np.uint8);rgba[1:40,20:60]=[255,0,0,255]
    image=tmp_path/'red.png';Image.fromarray(rgba).save(image)
    paths=preprocess_for_generation(str(image),str(tmp_path/'prep'))
    fg=np.array(Image.open(paths['foreground_rgba']))
    assert np.array_equal(fg,rgba)
    padded=np.array(Image.open(paths['padded_rgba']))
    assert padded.shape[0]==padded.shape[1]
    assert (padded[:,:,3]>0).sum()==39*40
    assert padded[0,:,3].max()==0

def test_empty_foreground_fails_before_generation(tmp_path):
    image=tmp_path/'empty.png';Image.new('RGBA',(32,32)).save(image)
    with pytest.raises(ValueError,match='No foreground'): preprocess_for_generation(str(image),str(tmp_path/'p'))

def test_face_smoothing_does_not_ignore_strengths():
    mesh=trimesh.creation.icosphere(subdivisions=2)
    mesh.vertices += np.random.default_rng(4).normal(0,.025,mesh.vertices.shape)
    before=mesh.vertices.copy()
    result=_face_aware_smooth(mesh,None,0.,0.,iterations=3)
    assert np.array_equal(result.vertices,before)

def test_smoothing_reduces_noise_without_melting(tmp_path):
    mesh=trimesh.creation.icosphere(subdivisions=3)
    mesh.vertices+=np.random.default_rng(6).normal(0,.018,mesh.vertices.shape)
    source=tmp_path/'raw.glb';mesh.export(source)
    out=tmp_path/'smooth.glb';cleanup_generated_mesh(str(source),str(out))
    result=trimesh.load(out,force='mesh')
    assert surface_roughness(result)<surface_roughness(mesh)
    assert abs(abs(result.volume)/abs(mesh.volume)-1)<.05

def test_hard_surface_retains_corners_and_faces(tmp_path):
    mesh=trimesh.creation.box().subdivide().subdivide()
    source=tmp_path/'box.glb';mesh.export(source)
    out=tmp_path/'out.glb';cleanup_generated_mesh(str(source),str(out),object_type='hard_surface')
    result=trimesh.load(out,force='mesh')
    assert len(result.faces)==len(mesh.faces)
    assert np.allclose(result.bounds,mesh.bounds)

def test_qc_counts_nonmanifold_edges():
    m=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],faces=[[0,1,2],[1,0,3],[0,1,4]],process=False)
    result=validate_mesh(m,min_vertices=1,min_faces=1)
    assert result['metrics']['non_manifold_edges']==1
    assert not result['passed']

def test_empty_mesh_is_failed_not_exception():
    assert not validate_mesh(trimesh.Trimesh())['passed']

def test_native_colors_are_not_replaced_by_grey_atlas(tmp_path):
    mesh=trimesh.creation.icosphere(subdivisions=2)
    color=np.tile([220,40,15,255],(len(mesh.vertices),1));mesh.visual.vertex_colors=color
    source=tmp_path/'colors.glb';mesh.export(source)
    out=tmp_path/'out.glb';project_texture_to_mesh(str(source),'unused.png',str(out))
    result=trimesh.load(out,force='mesh')
    assert np.all(result.visual.vertex_colors[:,:3]==[220,40,15])

def test_manager_preserves_colors_and_matching_exports(tmp_path,monkeypatch):
    image=tmp_path/'image.png';Image.new('RGBA',(32,32),(255,0,0,128)).save(image)
    for e in em._ENGINES.values():monkeypatch.setattr(e,'is_available',lambda:False)
    tripo=em._ENGINES['tripo'];monkeypatch.setattr(tripo,'is_available',lambda:True)
    import pipeline.source_detail as detail
    monkeypatch.setattr(detail,'enhance_source_detail',lambda *a,**k: (_ for _ in ()).throw(AssertionError('Photo overlay must be opt-in')))
    called=[]
    def generate(image,stl,glb,**kw):
        called.append(kw)
        m=trimesh.creation.icosphere(subdivisions=2)
        m.visual.vertex_colors=np.tile([210,30,50,255],(len(m.vertices),1))
        m.export(stl);m.export(glb)
        return {'generation':{'source_view':[0,0,1]}}
    monkeypatch.setattr(tripo,'generate',generate)
    result=em.run_single_image('tripo',str(image),str(tmp_path/'final.stl'),str(tmp_path/'final.glb'),quality='ultra',refine=True)
    stl=trimesh.load(result['files']['stl'],force='mesh');glb=trimesh.load(result['files']['glb'],force='mesh')
    assert np.allclose(stl.bounds,glb.bounds)
    assert len(stl.faces)==len(glb.faces)
    assert np.all(glb.visual.vertex_colors[:,:3]==[210,30,50])
    assert called[0]['texture'] is True and called[0]['quality']=='ultra'
    assert result['quality']['passed']
    assert result['source_detail'] is None
    assert trimesh.load(result['files']['glb'],force='scene').metadata['source_view']==[0,0,1]
    assert not any('Source detail unavailable' in warning for warning in result['warnings'])

def test_quality_presets_control_actual_tripo_resolution():
    from pipeline.ai.tripo_engine import _QUALITY_RESOLUTION
    assert [_QUALITY_RESOLUTION[k] for k in ['fast','high','ultra']]==[256,384,512]

def test_streamed_density_matches_full_grid():
    import torch
    from types import SimpleNamespace
    from tsr.system import TSR
    observed=[]
    class Helper:
        points_range=(0,1)
        def __call__(self,level):
            observed.append(level)
            return torch.tensor([[0.,0,0],[1.,0,0],[0,1.,0]]),torch.tensor([[0,1,2]])
    class Renderer:
        chunk_size=17
        cfg=SimpleNamespace(radius=1.)
        def query_triplane(self,decoder,positions,code):
            assert len(positions)<=17
            return {'density_act':(positions[:,0]+2*positions[:,1]+3*positions[:,2])[:,None],'color':torch.ones(len(positions),3)*.5}
    obj=SimpleNamespace(renderer=Renderer(),decoder=None,isosurface_helper=Helper(),set_marching_cubes_resolution=lambda n:None)
    TSR.extract_mesh(obj,torch.zeros((1,1)),True,resolution=4,threshold=0)
    coords=torch.stack(torch.meshgrid(*[torch.linspace(-1,1,4)]*3,indexing='ij'),dim=-1).reshape(-1,3)
    assert torch.allclose(observed[0],-(coords@torch.tensor([1.,2.,3.])),atol=1e-6)

def test_cpu_marching_cubes_preserves_axis_order():
    import torch
    from tsr.models.isosurface import MarchingCubeHelper
    helper=MarchingCubeHelper(32)
    coords=helper.grid_vertices
    field=((coords-.5)/torch.tensor([.12,.2,.35])).square().sum(dim=1)-1
    vertices,faces=helper(field)
    extents=(vertices.max(dim=0).values-vertices.min(dim=0).values).numpy()
    assert extents[2]>extents[1]>extents[0]
    assert np.allclose(extents,[.24,.4,.7],atol=.02)

@pytest.mark.parametrize('free_gb,expected',[(3,'cpu'),(8,'cuda:0')])
def test_busy_gpu_routes_to_cpu_without_stopping_training(monkeypatch,free_gb,expected):
    from pipeline import triposr_inference as tripo
    monkeypatch.delenv('SCANNER_INFERENCE_DEVICE',raising=False)
    monkeypatch.setattr(tripo.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(tripo,'_other_python_gpu_job',lambda:False)
    monkeypatch.setattr(tripo.torch.cuda,'mem_get_info',lambda:(free_gb*1024**3,16*1024**3))
    assert tripo._generation_device()==expected

def test_active_training_prefers_cpu_even_between_memory_peaks(monkeypatch):
    from pipeline import triposr_inference as tripo
    monkeypatch.delenv('SCANNER_INFERENCE_DEVICE',raising=False)
    monkeypatch.setattr(tripo.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(tripo,'_other_python_gpu_job',lambda:True)
    monkeypatch.setattr(tripo.torch.cuda,'mem_get_info',lambda:(12*1024**3,16*1024**3))
    assert tripo._generation_device()=='cpu'

def test_z_up_geometry_is_exported_upright_for_web_viewer():
    from pipeline.triposr_inference import Y_UP_TRANSFORM
    assert np.allclose(Y_UP_TRANSFORM@np.array([0.,0.,1.,1.]),[0.,1.,0.,1.])
    assert np.isclose(np.linalg.det(Y_UP_TRANSFORM[:3,:3]),1.)
