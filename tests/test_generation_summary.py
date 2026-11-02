import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from api import routes

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(routes,'BASE_DIR',tmp_path)
    app=FastAPI();app.include_router(routes.router)
    return TestClient(app)

def save(report):
    out=routes.BASE_DIR/'jobs'/'test_model'/'output';out.mkdir(parents=True)
    (out/'quality_report.json').write_text(json.dumps(report))

def test_summary_reports_fallback_without_internal_paths(client):
    save({'engine_requested':'hunyuan3d','engine_used':'tripo','engine_label':'TripoSR',
          'degraded':True,'glb_path':'D:/private/model.glb','logs':['private log'],
          'quality':{'passed':True,'status':'PASS WITH WARNINGS','metrics':{'coordinate_space':'normalized'},'warnings':['Two parts']},
          'warnings':['Two parts','Back is inferred']})
    response=client.get('/api/scan/report/test_model');assert response.status_code==200
    data=response.json();assert data['degraded'] and data['engine_used']=='tripo'
    assert data['warnings']==['Two parts','Back is inferred']
    assert data['relative_scale'] and not data['fine_geometry_verified']
    assert 'private' not in response.text and 'logs' not in data and 'glb_path' not in data

def test_photo_trial_is_distinguished_from_geometry_improvement(client):
    save({'experimental':True,'engine_used':'hunyuan3d','detail_transfer':{'appearance_only':True},
          'quality':{'passed':False,'status':'FAILED','errors':['Non-manifold edges']}})
    data=client.get('/api/scan/report/test_model').json()
    assert data['experimental'] and data['appearance_only'] and not data['mesh_passed']
    assert data['warnings']==['Non-manifold edges']

def test_missing_summary_does_not_create_job_directories(client):
    assert client.get('/api/scan/report/unknown').status_code==404
    assert not (routes.BASE_DIR/'jobs').exists()

def test_corrupt_saved_summary_has_readable_failure(client):
    save({});p=routes.BASE_DIR/'jobs/test_model/output/quality_report.json';p.write_text('{unfinished')
    assert client.get('/api/scan/report/test_model').status_code==503

def test_invalid_identifier_rejected(client):
    assert client.get('/api/scan/report/bad%3Aname').status_code==400
