import io,json
from pathlib import Path
import numpy as np
import pytest,trimesh
from PIL import Image
from pipeline.appearance_transfer import transfer_preview_appearance
from pipeline import hunyuan_runtime as runtime
from pipeline.cleanup_mesh import _remove_invalid_geometry

def test_colour_transfer_preserves_full_geometry_and_small_parts():
    full=trimesh.creation.icosphere(subdivisions=2)
    preview=full.copy()
    preview.visual=trimesh.visual.texture.TextureVisuals(uv=np.full((len(preview.vertices),2),.5),image=Image.new('RGB',(16,16),'red'))
    restored,report=transfer_preview_appearance(full,preview,batch_size=17)
    np.testing.assert_array_equal(restored.vertices,full.vertices)
    np.testing.assert_array_equal(restored.faces,full.faces)
    assert restored.is_watertight and report['geometry_preserved']
    assert np.all(restored.visual.vertex_colors[:,:3]==[255,0,0])
    accessory=trimesh.creation.box(extents=[.005]*3);accessory.apply_translation([2,0,0])
    combined=trimesh.util.concatenate([full,accessory])
    cleaned=_remove_invalid_geometry(combined.copy())
    assert len(cleaned.faces)==len(combined.faces)
    assert len(cleaned.split())==2

def test_colour_transfer_rejects_wrong_coordinate_frame():
    full=trimesh.creation.box();preview=full.copy();preview.apply_translation([2,0,0])
    with pytest.raises(ValueError,match='not aligned'):transfer_preview_appearance(full,preview)

class FakeProcess:
    def __init__(self,code=0):
        self.returncode=code;self.killed=False
        self.stdout=io.StringIO(json.dumps({'event':'progress','stage':'Shape','percent':40,'message':'sampling'})+'\n')
    def poll(self):return self.returncode
    def wait(self,timeout=None):return self.returncode
    def kill(self):self.killed=True;self.returncode=-9

def test_worker_native_failure_is_a_normal_exception(tmp_path,monkeypatch):
    process=FakeProcess(-1073741819)
    monkeypatch.setattr(runtime.subprocess,'Popen',lambda *args,**kw:process)
    with pytest.raises(RuntimeError,match='exited with code'):
        runtime.run_hunyuan_worker('input.png',tmp_path/'m.stl',tmp_path/'m.glb')
    assert (tmp_path/'hunyuan-worker.log').exists()

def test_worker_timeout_kills_only_its_own_child(tmp_path,monkeypatch):
    process=FakeProcess(None)
    monkeypatch.setattr(runtime.subprocess,'Popen',lambda *args,**kw:process)
    with pytest.raises(TimeoutError):runtime.run_hunyuan_worker('input.png',tmp_path/'m.stl',tmp_path/'m.glb',timeout=-1)
    assert process.killed

def test_worker_checks_exports_and_forwards_progress(tmp_path,monkeypatch):
    process=FakeProcess()
    monkeypatch.setattr(runtime.subprocess,'Popen',lambda *args,**kw:process)
    with pytest.raises(RuntimeError,match='both model exports'):
        runtime.run_hunyuan_worker('input.png',tmp_path/'m.stl',tmp_path/'m.glb')
    for suffix in ('stl','glb'):(tmp_path/f'm.{suffix}').write_text('test file')
    (tmp_path/'m.generation.json').write_text(json.dumps({'geometry_preserved':True}))
    process=FakeProcess();events=[]
    result=runtime.run_hunyuan_worker('input.png',tmp_path/'m.stl',tmp_path/'m.glb',progress_cb=lambda *args:events.append(args))
    assert result['geometry_preserved'] and events[0]==('Shape',40,'sampling')


def test_worker_progress_survives_tqdm_prefix():
    line='Diffusion Sampling:: 2%|x| 1/50'+json.dumps({'event':'progress','stage':'Shape','percent':12,'message':'2/50'})
    assert runtime.parse_worker_progress(line)==('Shape',12,'2/50')
    assert runtime.parse_worker_progress('Volume Decoding:  50%|xxxx|')==('Decoding surface',50,'Extracting detailed surface: 50%')
    assert runtime.parse_worker_progress('unrelated log') is None
    assert runtime.parse_worker_progress('{"event": "progress" bad data') is None
