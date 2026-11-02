import importlib.util
import sys
import types
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
spec=importlib.util.spec_from_file_location('tested_oauth',Path(__file__).resolve().parent.parent/'api'/'oauth.py')
oauth=importlib.util.module_from_spec(spec);spec.loader.exec_module(oauth)

@pytest.fixture
def client(monkeypatch,tmp_path):
    import api.db as db
    monkeypatch.setattr(db,'DB_PATH',tmp_path/'models.db');db.init_db()
    for key in ['GOOGLE_CLIENT_ID','GOOGLE_CLIENT_SECRET','FACEBOOK_APP_ID','FACEBOOK_APP_SECRET','FACEBOOK_GRAPH_VERSION','PUBLIC_BASE_URL']: monkeypatch.delenv(key,raising=False)
    oauth.pending.clear();oauth.handoffs.clear()
    app=FastAPI();app.include_router(oauth.router)
    return TestClient(app)

def test_unconfigured(client):
    assert client.get('/api/auth/oauth/providers').json()=={'google':False,'facebook':False}
    assert client.get('/api/auth/oauth/google/start').status_code==503

def configure(monkeypatch):
    monkeypatch.setenv('GOOGLE_CLIENT_ID','test-id');monkeypatch.setenv('GOOGLE_CLIENT_SECRET','test-secret');monkeypatch.setenv('PUBLIC_BASE_URL','http://testserver')

def test_state_binding_and_replay(client,monkeypatch):
    configure(monkeypatch)
    response=client.get('/api/auth/oauth/google/start',follow_redirects=False)
    assert 'code_challenge=' in response.headers['location']
    state=next(iter(oauth.pending))
    client.cookies.clear()
    assert client.get(f'/api/auth/oauth/google/callback?state={state}&code=bad').status_code==400
    assert state not in oauth.pending

def test_success_and_single_use_handoff(client,monkeypatch):
    configure(monkeypatch)
    sessions={};monkeypatch.setitem(sys.modules,'api.routes',types.SimpleNamespace(ACTIVE_SESSIONS=sessions))
    class Reply:
        def __init__(self,data):self.data=data
        def raise_for_status(self):pass
        def json(self):return self.data
    monkeypatch.setattr(oauth.requests,'post',lambda *a,**k:Reply({'access_token':'provider-secret'}))
    monkeypatch.setattr(oauth.requests,'get',lambda *a,**k:Reply({'sub':'123','name':'admin','email':'existing@example.com'}))
    client.get('/api/auth/oauth/google/start',follow_redirects=False)
    state=next(iter(oauth.pending))
    response=client.get(f'/api/auth/oauth/google/callback?state={state}&code=valid',follow_redirects=False)
    assert response.headers['location']=='/?oauth_success=1'
    assert 'provider-secret' not in str(response.headers)
    response=client.post('/api/auth/oauth/session',headers={'Origin':'http://testserver'})
    assert response.status_code==200
    assert response.json()['user']['is_admin']==0
    assert response.json()['token'] in sessions
    assert client.post('/api/auth/oauth/session',headers={'Origin':'http://testserver'}).status_code==401

def test_session_rejects_cross_origin(client):
    assert client.post('/api/auth/oauth/session',headers={'Origin':'https://evil.example'}).status_code==403

def test_identity_is_stable_and_not_linked_by_email(client):
    first=oauth.identity_user('google','123','admin');second=oauth.identity_user('google','123','another')
    assert first['id']==second['id']
    assert first['is_admin']==0

def test_facebook_code_exchange_and_profile(client,monkeypatch):
    monkeypatch.setenv('PUBLIC_BASE_URL','http://testserver')
    monkeypatch.setenv('FACEBOOK_APP_ID','fb-id');monkeypatch.setenv('FACEBOOK_APP_SECRET','fb-secret');monkeypatch.setenv('FACEBOOK_GRAPH_VERSION','vTEST')
    class Reply:
        def __init__(self,data):self.data=data
        def raise_for_status(self):pass
        def json(self):return self.data
    calls=[]
    monkeypatch.setattr(oauth.requests,'post',lambda *a,**k:Reply({'access_token':'facebook-token'}))
    def profile(*a,**kwargs):
        calls.append(kwargs);return Reply({'id':'fb123','name':'Test'})
    monkeypatch.setattr(oauth.requests,'get',profile)
    response=client.get('/api/auth/oauth/facebook/start',follow_redirects=False)
    assert response.headers['location'].startswith('https://www.facebook.com/vTEST/dialog/oauth?')
    state=next(iter(oauth.pending))
    response=client.get(f'/api/auth/oauth/facebook/callback?state={state}&code=ok',follow_redirects=False)
    assert response.headers['location']=='/?oauth_success=1'
    assert len(calls[0]['params']['appsecret_proof'])==64
