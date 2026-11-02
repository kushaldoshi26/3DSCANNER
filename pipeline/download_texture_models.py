"""Required official Hunyuan texture weights; range-resumable, no duplicate .bin weights."""
from pathlib import Path
import os,json,time,hashlib,concurrent.futures,requests
from huggingface_hub import HfApi
ROOT=Path('D:/3dSCANNER');target=ROOT/'engines/models/tencent/Hunyuan3D-2';status=ROOT/'data/software_checks/hunyuan_texture_setup';status.mkdir(parents=True,exist_ok=True)
files=[f for f in HfApi().list_repo_tree('tencent/Hunyuan3D-2',recursive=True) if hasattr(f,'size') and f.path.startswith(('hunyuan3d-delight-v2-0/','hunyuan3d-paint-v2-0-turbo/'))]
names={f.path for f in files}
files=[f for f in files if f.path == 'hunyuan3d-paint-v2-0-turbo/unet/diffusion_pytorch_model.bin' or not(f.path.endswith('.bin') and (f.path[:-4]+'.safetensors' in names or f.path.replace('pytorch_model.bin','model.safetensors') in names))]
total=sum(f.size for f in files);finished=0;started=time.time();block=16*1024**2
baseline_bytes=None
def record(stage,**extra):
    global baseline_bytes
    if baseline_bytes is None:baseline_bytes=finished
    elapsed=time.time()-started
    current=finished
    if stage=='downloading' and extra.get('file'):
        chunks=target/(extra['file']+'.parts')
        if chunks.exists():current+=sum(p.stat().st_size for p in chunks.iterdir() if p.is_file() and p.suffix!='.tmp')
    transferred=max(0,current-baseline_bytes)
    report={'stage':stage,'completed_bytes':current,'total_bytes':total,'elapsed_seconds':elapsed,'eta_seconds':elapsed/transferred*(total-current) if transferred else None,'updated':time.time(),**extra}
    temp=status/'download.tmp';temp.write_text(json.dumps(report,indent=2));os.replace(temp,status/'download.json')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as s:
        while data:=s.read(8*1024**2):h.update(data)
    return h.hexdigest()
try:
    for entry in sorted(files,key=lambda f:f.size):
        path=target/entry.path;path.parent.mkdir(parents=True,exist_ok=True)
        sha=getattr(getattr(entry,'lfs',None),'sha256',None)
        if path.exists() and path.stat().st_size==entry.size and (not sha or digest(path)==sha):finished+=entry.size;continue
        record('downloading',file=entry.path)
        url='https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/'+entry.path
        parts=path.parent/(path.name+'.parts');parts.mkdir(exist_ok=True)
        def fetch(index):
            start=index*block;end=min(entry.size-1,start+block-1);part=parts/f'{index:04d}'
            if part.exists() and part.stat().st_size==end-start+1:return part
            for attempt in range(4):
                try:
                    with requests.get(url+f'?download=true&part={index}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,90)) as r:
                        r.raise_for_status()
                        if r.status_code==206 and r.headers.get('Content-Range')!=f'bytes {start}-{end}/{entry.size}':raise RuntimeError('Range mismatch')
                        if r.status_code!=206 and not(start==0 and entry.size<=block):raise RuntimeError('Server ignored range')
                        tmp=part.with_suffix('.tmp')
                        with tmp.open('wb') as s:
                            for data in r.iter_content(1024*1024):s.write(data)
                        if tmp.stat().st_size!=end-start+1:raise RuntimeError('Incomplete data')
                        os.replace(tmp,part);return part
                except Exception:
                    if attempt==3:raise
                    time.sleep(2)
        count=(entry.size+block-1)//block
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for future in concurrent.futures.as_completed([pool.submit(fetch,i) for i in range(count)]):
                future.result();record('downloading',file=entry.path,parts_ready=sum(1 for p in parts.iterdir() if p.suffix!='.tmp'),parts_total=count)
        temp=path.with_suffix(path.suffix+'.assembling')
        with temp.open('wb') as s:
            for i in range(count):
                with (parts/f'{i:04d}').open('rb') as inp:
                    while data:=inp.read(8*1024**2):s.write(data)
        if temp.stat().st_size!=entry.size or (sha and digest(temp)!=sha):raise RuntimeError('Downloaded model verification failed')
        os.replace(temp,path);finished+=entry.size;print('Verified',entry.path,flush=True)
    record('completed')
except Exception as error:record('failed',error=str(error));raise
