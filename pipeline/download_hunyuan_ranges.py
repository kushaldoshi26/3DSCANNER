"""Resumable official fp16 download with verified byte ranges."""
from pathlib import Path
import requests,concurrent.futures,os,json,time
root=Path('D:/3dSCANNER/engines/models/tenc ent'.replace('tenc ent','tencent'))/'Hunyuan3D-2/hunyuan3d-dit-v2-0'
root.mkdir(parents=True,exist_ok=True)
url='https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-dit-v2-0/model.fp16.safetensors'
size=4928151562;block=16*1024**2;parts=root/'download-parts';parts.mkdir(exist_ok=True)
def part(index):
    start=index*block;end=min(size-1,start+block-1);target=parts/f'{index:04d}.part'
    if target.exists() and target.stat().st_size==end-start+1:return
    for attempt in range(4):
        try:
            with requests.get(url+f'?download=true&part={index}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,90)) as response:
                response.raise_for_status()
                if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise RuntimeError('Unexpected range response')
                temporary=target.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024):stream.write(chunk)
                if temporary.stat().st_size!=end-start+1:raise RuntimeError('Incomplete range')
                os.replace(temporary,target)
            return
        except Exception:
            if attempt==3:raise
            time.sleep(2)
count=(size+block-1)//block
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(part,index) for index in range(count)]
    for completed,future in enumerate(concurrent.futures.as_completed(futures),1):
        future.result();(root/'download-progress.json').write_text(json.dumps({'completed_parts':completed,'total_parts':count,'total_bytes':size,'updated':time.time()}));print(completed,'/',count,flush=True)
target=root/'model.fp16.safetensors';temp=target.with_suffix('.assembling')
with temp.open('wb') as output:
    for index in range(count):
        with (parts/f'{index:04d}.part').open('rb') as stream:
            while data:=stream.read(8*1024**2):output.write(data)
assert temp.stat().st_size==size
from safetensors import safe_open
with safe_open(temp,framework='pt',device='cpu') as weights:print('Verified tensors',len(weights.keys()),flush=True)
os.replace(temp,target);print('COMPLETED',target,flush=True)
