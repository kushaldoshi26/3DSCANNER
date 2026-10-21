"""3D SCANNER local companion: optional TripoSR engine and Uno connection.
ESP32-CAM capture is planned; no depth-camera dependencies or fake scan results.
"""
import base64, io, json, os, secrets, shutil, subprocess, sys, threading, time, uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parent
TOKEN=secrets.token_urlsafe(24)
ORIGINS={'https://form-scanner-electronics.zippyshell32.chatgpt.site','http://127.0.0.1:8787','http://localhost:8787','http://127.0.0.1:5173'}
controller=None
job={'state':'idle','message':'AI engine is idle.'}
model=None
lock=threading.Lock()

def engine():
    root=os.environ.get('TRIPOSR_ROOT','')
    return Path(root).resolve() if root and (Path(root)/'run.py').is_file() else None

def status():
    try:
        from serial.tools import list_ports
        ports=[{'device':p.device,'description':p.description} for p in list_ports.comports()]
    except ImportError:ports=[]
    return {'camera':False,'camera_type':'esp32-cam','camera_state':'planned','controller':bool(controller and controller.is_open),'ai_available':engine() is not None,'ports':ports}

def disconnect():
    global controller
    if controller:
        try:controller.write(b'STOP\n');controller.close()
        finally:controller=None

def connect(port):
    global controller
    import serial
    from serial.tools import list_ports
    if port not in [p.device for p in list_ports.comports()]:raise ValueError('Select a connected Arduino port.')
    disconnect()
    link=serial.Serial(port,115200,timeout=.2,write_timeout=2)
    try:
        time.sleep(2);link.reset_input_buffer();link.write(b'HELLO\n')
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if link.readline().strip()==b'FORM_READY':controller=link;return status()
        raise ValueError('Upload the companion firmware: Arduino did not answer the handshake.')
    except Exception:link.close();raise

def run_ai(data,quality,root):
    global model
    try:
        from PIL import Image
        image=Image.open(io.BytesIO(data))
        if image.width*image.height>45000000:raise ValueError('Image exceeds 45 megapixels.')
        folder=ROOT/'jobs'/uuid.uuid4().hex;folder.mkdir(parents=True)
        source=folder/'input.png';image.convert('RGBA').save(source)
        python=os.environ.get('TRIPOSR_PYTHON',sys.executable)
        args=[python,str(root/'run.py'),str(source),'--output-dir',str(folder/'output'),'--model-save-format','glb','--mc-resolution',str(quality),'--chunk-size','4096']
        job.update(state='running',message='TripoSR reconstruction is running. Initial model downloads may take longer.')
        process=subprocess.run(args,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1800,check=False,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (folder/'engine.log').write_bytes(process.stdout)
        if process.returncode:
            raise ValueError('AI engine failed. Check jobs/'+folder.name+'/engine.log for missing dependencies or GPU memory errors.')
        output=folder/'output/0/mesh.glb'
        if not output.is_file() or output.stat().st_size>150*1024*1024:raise ValueError('Engine did not produce a supported GLB output.')
        raw=output.read_bytes()
        if raw[:4]!=b'glTF':raise ValueError('Engine output is not a valid GLB.')
        model=raw;job.update(state='complete',message='Full 3D mesh is ready.')
    except Exception as error:job.update(state='error',message=str(error))

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,data,code=200,kind='application/json'):
        raw=json.dumps(data).encode() if kind=='application/json' else data
        self.send_response(code)
        origin=self.headers.get('Origin')
        if origin in ORIGINS:self.send_header('Access-Control-Allow-Origin',origin);self.send_header('Vary','Origin')
        self.send_header('Access-Control-Allow-Methods','GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers','Authorization, Content-Type')
        self.send_header('Access-Control-Allow-Private-Network','true')
        self.send_header('Cache-Control','no-store');self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(raw)));self.end_headers()
        if code!=204:self.wfile.write(raw)
    def allowed(self):
        if self.headers.get('Origin') and self.headers['Origin'] not in ORIGINS:self.reply({'error':'Origin not allowed.'},403);return False
        if not secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+TOKEN):self.reply({'error':'Incorrect or missing pairing token.'},401);return False
        return True
    def do_OPTIONS(self):
        self.reply({},204 if self.headers.get('Origin') in ORIGINS else 403)
    def do_GET(self):
        path=urlparse(self.path).path
        if path.startswith('/api/'):
            if not self.allowed():return
            if path=='/api/status':return self.reply(status())
            if path=='/api/generate':return self.reply(dict(job))
            if path=='/api/model' and model and job['state']=='complete':return self.reply(model,kind='model/gltf-binary')
            return self.reply({'error':'No result available.'},404)
        import mimetypes
        web=(ROOT/'web').resolve();file=(web/path.lstrip('/')).resolve()
        if path=='/':file=web/'workspace.html'
        if not file.is_relative_to(web) or not file.is_file():return self.reply({'error':'Place the website build in the companion web folder for local access.'},404)
        self.reply(file.read_bytes(),kind=mimetypes.guess_type(file)[0] or 'application/octet-stream')
    def do_POST(self):
        global model
        if not self.allowed():return
        try:
            path=urlparse(self.path).path
            length=int(self.headers.get('Content-Length','0'))
            limit=22*1024*1024 if path=='/api/generate' else 8192
            if not 0<=length<=limit:raise ValueError('Request is too large.')
            data=json.loads(self.rfile.read(length) or b'{}')
            with lock:
                if path=='/api/connect':return self.reply(connect(data.get('port','')))
                if path=='/api/disconnect':disconnect();return self.reply({'disconnected':True})
                if path=='/api/stop':
                    if controller:controller.write(b'STOP\n')
                    return self.reply({'stopped':True})
                if path in ['/api/scan','/api/calibration']:raise ValueError('ESP32-CAM capture is planned and is not active yet.')
                if path=='/api/generate':
                    if job['state'] in ['starting','running']:raise ValueError('An AI generation is already running.')
                    root=engine()
                    if not root:raise ValueError('Set TRIPOSR_ROOT and install the local TripoSR engine first.')
                    text=data.get('image','');prefix,encoded=text.split(',',1)
                    if prefix not in ['data:image/png;base64','data:image/jpeg;base64','data:image/webp;base64']:raise ValueError('Use a PNG, JPG or WEBP image.')
                    raw=base64.b64decode(encoded,validate=True)
                    if not raw or len(raw)>15*1024*1024:raise ValueError('Image must be below 15 MB.')
                    quality=int(data.get('quality',256))
                    if quality not in [256,384]:raise ValueError('Choose a supported quality level.')
                    model=None;job.update(state='starting',message='Starting the configured AI engine…')
                    threading.Thread(target=run_ai,args=(raw,quality,root),daemon=True).start()
                    return self.reply(dict(job))
            self.reply({'error':'Unknown operation.'},404)
        except Exception as error:self.reply({'error':str(error)},400)

def main():
    server=ThreadingHTTPServer(('127.0.0.1',8787),Handler)
    print('3D SCANNER companion: http://127.0.0.1:8787',flush=True)
    print('Pairing token: '+TOKEN,flush=True)
    print('AI engine: '+('configured' if engine() else 'not configured'),flush=True)
    print('ESP32-CAM: planned. Phone webcam: coming later. No motor motion on startup.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:disconnect();server.server_close()
