import io
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from api import routes

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(routes,'BASE_DIR',tmp_path)
    app=FastAPI();app.include_router(routes.router)
    return TestClient(app)

def test_uploaded_form_settings_reach_worker(client,monkeypatch):
    received=[]
    monkeypatch.setattr(routes,'execute_reconstruct_pipeline',lambda *args:received.append(args))
    image=io.BytesIO();Image.new('RGBA',(32,32),(180,0,0,120)).save(image,format='PNG')
    response=client.post('/api/reconstruct',files={'file':('../chair.png',image.getvalue(),'image/png')},
        data={'engine':'triposr','quality':'ultra','object_type':'hard_surface','texture':'false','refine':'false','printable':'true'})
    assert response.status_code==200
    assert received[0][1:] == ('chair.png','triposr','hard_surface',False,'ultra',False,True,False)
    job=routes.JOBS[response.json()['job_id']]
    assert job['quality']=='ultra' and job['object_type']=='hard_surface'

def test_corrupt_image_rejected_before_queue(client):
    response=client.post('/api/reconstruct',files={'file':('bad.png',b'not an image','image/png')})
    assert response.status_code==400

def test_invalid_object_type_rejected(client):
    response=client.post('/api/reconstruct',files={'file':('bad.png',b'x','image/png')},data={'object_type':'invalid'})
    assert response.status_code==400

def test_saved_generation_reports_relative_scale_after_restart(client):
    import json
    out=routes.get_job_dirs('saved_quality_demo')[4]
    (out/'final_model.stl').write_text('exists')
    quality={'status':'PASS','passed':True,'metrics':{'vertices':150,'faces':300,
        'coordinate_space':'normalized','bounding_box_units':[.6,1.,.4],
        'watertight':True,'winding_consistent':True,'volume_units3':.15}}
    (out/'quality_report.json').write_text(json.dumps({'quality':quality}))
    response=client.get('/api/scan/topology/saved_quality_demo')
    assert response.status_code==200
    assert response.json()['scale_calibrated'] is False
    assert response.json()['dimensions_units']['y']==1.
    assert client.get('/api/scan/quality/saved_quality_demo').json()==quality


def test_interrupted_job_is_not_reported_completed_after_restart(client):
    import json
    directory=routes.BASE_DIR/'jobs'/'interrupted';directory.mkdir(parents=True)
    (directory/'job_status.json').write_text(json.dumps({'job_id':'interrupted','status':'processing','progress':45}))
    result=client.get('/api/scan/status/interrupted').json()
    assert result['status']=='failed' and 'restarted' in result['error']

def test_completed_job_retains_engine_and_elapsed_time_after_restart(client):
    import json
    directory=routes.BASE_DIR/'jobs'/'finished';directory.mkdir(parents=True)
    status={'job_id':'finished','status':'completed','engine_used':'hunyuan3d','elapsed_seconds':123,'progress':100}
    (directory/'job_status.json').write_text(json.dumps(status))
    assert client.get('/api/scan/status/finished').json()==status

def test_photo_detail_setting_reaches_worker(client,monkeypatch):
    received=[]
    monkeypatch.setattr(routes,'execute_reconstruct_pipeline',lambda *args:received.append(args))
    image=io.BytesIO();Image.new('RGBA',(32,32),(180,0,0,120)).save(image,format='PNG')
    result=client.post('/api/reconstruct',files={'file':('photo.png',image.getvalue(),'image/png')},data={'photo_detail':'true'})
    assert result.status_code==200 and received[0][-1] is True

def test_ply_download_uses_new_generation_export(client):
    out=routes.get_job_dirs('points')[4];(out/'final_model.ply').write_bytes(b'ply\nformat ascii 1.0\nend_header\n')
    result=client.get('/api/scan/download/points/ply')
    assert result.status_code==200 and result.content.startswith(b'ply')
