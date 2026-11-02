import io
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from api import routes

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(routes,'BASE_DIR',tmp_path)
    app=FastAPI();app.include_router(routes.router)
    return TestClient(app)

def test_missing_and_invalid_preview(client):
    assert client.get('/api/scan/source-preview/12345678').status_code==404
    assert client.get('/api/scan/source-preview/not-a-scan').status_code==400

def test_preview_waits_for_complete_png(client):
    folder=routes.BASE_DIR/'jobs'/'12345678'/'output'/'prep';folder.mkdir(parents=True)
    image=io.BytesIO();Image.new('RGBA',(8,8),(255,255,255,255)).save(image,format='PNG')
    payload=image.getvalue();path=folder/'source_foreground.png'
    path.write_bytes(payload[:20]);assert client.get('/api/scan/source-preview/12345678').status_code==404
    path.write_bytes(payload);response=client.get('/api/scan/source-preview/12345678')
    assert response.status_code==200 and response.content==payload
    assert int(response.headers['content-length'])==len(payload)
    assert response.headers['cache-control']=='no-store'
