"""Keep native model crashes and timeouts outside the API server process."""
from pathlib import Path
import os,sys,time,json,queue,threading,subprocess,re


def parse_worker_progress(line):
    # tqdm and JSON share the redirected stream; JSON can follow a progress bar.
    marker=line.find('{"event": "progress"')
    if marker>=0:
        try:
            event,_=json.JSONDecoder().raw_decode(line[marker:])
            return event['stage'],event['percent'],event['message']
        except (ValueError,KeyError,TypeError):
            return None
    match=re.search(r'Volume Decoding:\s*(\d+)%',line)
    if match:
        percent=min(100,int(match.group(1)))
        return 'Decoding surface',45+int(percent*.1),f'Extracting detailed surface: {percent}%'
    return None

def run_hunyuan_worker(image,stl,glb,*,texture=True,quality='high',progress_cb=None,timeout=1800):
    root=Path(__file__).resolve().parent.parent
    log_path=Path(stl).parent/'hunyuan-worker.log';log_path.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','-X','faulthandler',str(root/'pipeline/hunyuan_worker.py'),
             '--image',str(image),'--stl',str(stl),'--glb',str(glb),'--quality',quality]
    if texture:command.append('--texture')
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['HF_HUB_OFFLINE']='1'
    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    process=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,encoding='utf-8',errors='replace',creationflags=creationflags)
    lines=queue.Queue();started=time.monotonic()
    def read():
        try:
            for line in process.stdout:lines.put(line)
        finally:lines.put(None)
    reader=threading.Thread(target=read,daemon=True);reader.start()
    try:
        last_event=None
        with log_path.open('w',encoding='utf-8') as log:
            while True:
                if time.monotonic()-started>timeout:
                    raise TimeoutError('Hunyuan worker exceeded its generation time limit')
                try:line=lines.get(timeout=.5)
                except queue.Empty:continue
                if line is None:break
                log.write(line);log.flush()
                event=parse_worker_progress(line)
                if event and event!=last_event:
                    last_event=event
                    if progress_cb:progress_cb(*event)
        code=process.wait(timeout=10)
        if code:raise RuntimeError(f'Hunyuan worker exited with code {code}; inspect hunyuan-worker.log')
        if not all(Path(p).is_file() and Path(p).stat().st_size>0 for p in (stl,glb)):
            raise RuntimeError('Hunyuan worker did not produce both model exports')
    finally:
        if process.poll() is None:process.kill();process.wait()
        process.stdout.close();reader.join(timeout=2)
    metadata=Path(glb).with_suffix('.generation.json')
    return json.loads(metadata.read_text(encoding='utf-8')) if metadata.exists() else {}
