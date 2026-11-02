"""Server-side social login. Provider credentials stay on the server."""
import hashlib
import hmac
import os
import secrets
import time
from urllib.parse import urlencode

import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from api.db import _get_conn, create_user

router = APIRouter(prefix='/api/auth/oauth')
pending = {}
handoffs = {}

def config(provider):
    if provider == 'google':
        return os.getenv('GOOGLE_CLIENT_ID'), os.getenv('GOOGLE_CLIENT_SECRET')
    if provider == 'facebook':
        return os.getenv('FACEBOOK_APP_ID'), os.getenv('FACEBOOK_APP_SECRET')
    raise HTTPException(404, 'Unknown login provider')

def enabled(provider):
    return all(config(provider)) and bool(os.getenv('PUBLIC_BASE_URL')) and (provider != 'facebook' or bool(os.getenv('FACEBOOK_GRAPH_VERSION')))

def callback_url(provider):
    return os.environ['PUBLIC_BASE_URL'].rstrip('/') + f'/api/auth/oauth/{provider}/callback'

def secure():
    return os.getenv('PUBLIC_BASE_URL', '').startswith('https://')

def prune(store):
    for key in list(store):
        if store[key]['expires'] < time.time(): store.pop(key, None)

@router.get('/providers')
def providers():
    return {p: enabled(p) for p in ('google','facebook')}

@router.get('/{provider}/start')
def start(provider: str):
    if not enabled(provider): raise HTTPException(503, 'Configure provider credentials and PUBLIC_BASE_URL on the server first.')
    prune(pending)
    state, binding = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(48)
    pending[state] = {'provider':provider,'binding':binding,'verifier':verifier,'expires':time.time()+600}
    client_id, _ = config(provider)
    params = dict(client_id=client_id, redirect_uri=callback_url(provider), response_type='code', state=state)
    if provider == 'google':
        import base64
        params.update(scope='openid email profile', code_challenge_method='S256', code_challenge=base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode())
        endpoint='https://accounts.google.com/o/oauth2/v2/auth'
    else:
        params['scope']='public_profile,email'
        endpoint=f"https://www.facebook.com/{os.environ['FACEBOOK_GRAPH_VERSION']}/dialog/oauth"
    response=RedirectResponse(endpoint+'?'+urlencode(params), status_code=302)
    response.set_cookie('scanner_oauth_state',binding,max_age=600,httponly=True,secure=secure(),samesite='lax',path='/api/auth/oauth')
    response.headers['Cache-Control']='no-store'
    return response

def identity_user(provider, subject, name):
    # Link by immutable provider subject only; never silently merge accounts by email.
    with _get_conn() as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS oauth_identities (provider TEXT NOT NULL, subject TEXT NOT NULL, user_id INTEGER NOT NULL REFERENCES users(id), PRIMARY KEY(provider,subject))')
        row=conn.execute('SELECT u.id,u.username,u.email,u.is_admin FROM users u JOIN oauth_identities o ON u.id=o.user_id WHERE o.provider=? AND o.subject=?',(provider,subject)).fetchone()
        if row: return dict(row)
    suffix=secrets.token_hex(8)
    user=create_user(f'{provider}_{suffix}', f'{provider}_{suffix}@social.invalid', secrets.token_urlsafe(48), is_admin=0)
    if not user: raise RuntimeError('Unable to create social account')
    with _get_conn() as conn:
        conn.execute('INSERT INTO oauth_identities(provider,subject,user_id) VALUES(?,?,?)',(provider,subject,user['id']))
    return user

@router.get('/{provider}/callback')
def callback(provider: str, request: Request, state: str='', code: str=''):
    config(provider)
    record=pending.pop(state,None)
    if not record or record['expires']<time.time() or record['provider']!=provider or not secrets.compare_digest(record['binding'],request.cookies.get('scanner_oauth_state','')):
        raise HTTPException(400,'Invalid or expired login state')
    response=RedirectResponse('/?oauth_error=1',status_code=302)
    try:
        if not code or not enabled(provider): raise ValueError('Login cancelled or unconfigured')
        client_id,client_secret=config(provider)
        payload=dict(client_id=client_id,client_secret=client_secret,redirect_uri=callback_url(provider),code=code)
        if provider=='google':
            payload.update(grant_type='authorization_code',code_verifier=record['verifier'])
            token_response=requests.post('https://oauth2.googleapis.com/token',data=payload,timeout=20)
            token_response.raise_for_status(); access=token_response.json()['access_token']
            profile_response=requests.get('https://openidconnect.googleapis.com/v1/userinfo',headers={'Authorization':'Bearer '+access},timeout=20)
            profile_response.raise_for_status(); profile=profile_response.json(); subject=profile['sub']
        else:
            graph=f"https://graph.facebook.com/{os.environ['FACEBOOK_GRAPH_VERSION']}"
            token_response=requests.post(graph+'/oauth/access_token',data=payload,timeout=20)
            token_response.raise_for_status(); access=token_response.json()['access_token']
            proof=hmac.new(client_secret.encode(),access.encode(),hashlib.sha256).hexdigest()
            profile_response=requests.get(graph+'/me',params={'fields':'id,name','appsecret_proof':proof},headers={'Authorization':'Bearer '+access},timeout=20)
            profile_response.raise_for_status(); profile=profile_response.json(); subject=profile['id']
        if not subject: raise ValueError('Provider returned no identity')
        user=identity_user(provider,str(subject),profile.get('name',''))
        prune(handoffs); ticket=secrets.token_urlsafe(32)
        handoffs[ticket]={'user':user,'expires':time.time()+60}
        response=RedirectResponse('/?oauth_success=1',status_code=302)
        response.set_cookie('scanner_oauth_ticket',ticket,max_age=60,httponly=True,secure=secure(),samesite='strict',path='/api/auth/oauth/session')
    except (requests.RequestException,KeyError,ValueError,RuntimeError):
        pass  # Do not expose tokens, secrets or provider responses in the browser.
    response.delete_cookie('scanner_oauth_state',path='/api/auth/oauth')
    response.headers['Cache-Control']='no-store';response.headers['Referrer-Policy']='no-referrer'
    return response

@router.post('/session')
def session(request: Request):
    from api.routes import ACTIVE_SESSIONS
    base=os.getenv('PUBLIC_BASE_URL','').rstrip('/')
    if request.headers.get('origin') != base: raise HTTPException(403,'Invalid origin')
    record=handoffs.pop(request.cookies.get('scanner_oauth_ticket',''),None)
    if not record or record['expires']<time.time(): raise HTTPException(401,'Login expired')
    token=secrets.token_hex(32);ACTIVE_SESSIONS[token]=record['user']
    response=JSONResponse({'token':token,'user':record['user']})
    response.delete_cookie('scanner_oauth_ticket',path='/api/auth/oauth/session');response.headers['Cache-Control']='no-store'
    return response
