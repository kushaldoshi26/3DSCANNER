"""TripoSR reconstruction with bounded memory and native vertex-color export."""
from pathlib import Path
from threading import RLock
import os
import numpy as np
import torch
import trimesh
from PIL import Image, ImageOps
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

TRIPOSR_MODEL=None
_MODEL_LOCK=RLock()
# TripoSR uses Z-up; the web viewer uses Y-up.
Y_UP_TRANSFORM=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])

def _other_python_gpu_job():
    """Avoid competing with independent training jobs during their VRAM dips."""
    import csv, shutil, subprocess
    executable=shutil.which('nvidia-smi')
    if not executable: return False
    try:
        result=subprocess.run([executable,'--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=2,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for row in csv.reader(result.stdout.splitlines()):
            if len(row)>=2 and row[0].strip()!=str(os.getpid()) and 'python' in row[1].lower():
                return True
    except (OSError,subprocess.TimeoutExpired):
        pass
    return False

def _generation_device():
    requested=os.environ.get('SCANNER_INFERENCE_DEVICE','auto')
    if requested=='cpu' or not torch.cuda.is_available(): return 'cpu'
    if requested=='auto' and _other_python_gpu_job():
        print('Another Python GPU job is active; using CPU without interrupting training.',flush=True)
        return 'cpu'
    free,_=torch.cuda.mem_get_info()
    minimum_free=float(os.environ.get('SCANNER_MIN_FREE_VRAM_GB','6'))
    if requested=='auto' and free < minimum_free*1024**3:
        print(f'GPU is busy (less than {minimum_free:g} GiB free); using CPU for reconstruction.',flush=True)
        return 'cpu'
    return 'cuda:0'

def get_triposr_model(device=None):
    global TRIPOSR_MODEL
    with _MODEL_LOCK:
        if TRIPOSR_MODEL is None:
            print('Loading TripoSR weights...',flush=True)
            model=TSR.from_pretrained('stabilityai/TripoSR',config_name='config.yaml',weight_name='model.ckpt')
            model.renderer.set_chunk_size(8192)
            model.eval()
            TRIPOSR_MODEL=model
        TRIPOSR_MODEL.to(device or _generation_device())
        return TRIPOSR_MODEL

def generate_triposr_3d_mesh(image_path, output_stl_path, output_glb_path=None,
                           resolution=384, mc_threshold=25.0):
    if not 64 <= resolution <= 512: raise ValueError('Extraction resolution must be between 64 and 512.')
    out=Path(output_stl_path);out.parent.mkdir(parents=True,exist_ok=True)
    glb=Path(output_glb_path) if output_glb_path else out.with_suffix('.glb')
    glb.parent.mkdir(parents=True,exist_ok=True)
    with Image.open(image_path) as source:
        image=ImageOps.exif_transpose(source).convert('RGBA')
    alpha=np.asarray(image)[:,:,3]
    if not np.any(alpha < 255): image=remove_background(image,rembg_session=None)
    image=resize_foreground(image,0.85)
    rgba=np.asarray(image,dtype=np.float32)/255.
    rgb=rgba[:,:,:3]*rgba[:,:,3:4]+(1-rgba[:,:,3:4])*0.5
    image=Image.fromarray(np.clip(rgb*255,0,255).astype(np.uint8))
    with _MODEL_LOCK:
        device=_generation_device()
        if device=='cpu':
            torch.set_num_threads(max(1,min(torch.get_num_threads(),int(os.environ.get('SCANNER_CPU_THREADS','4')))))
        model=get_triposr_model(device)
        print(f'TripoSR inference on {device}; extraction resolution {resolution}.',flush=True)
        try:
            with torch.inference_mode():
                codes=model([image],device=device)
                mesh=model.extract_mesh(codes,True,resolution=resolution,threshold=mc_threshold)[0]
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError('Not enough GPU memory. Finish training or set SCANNER_INFERENCE_DEVICE=cpu; no incomplete mesh was exported.') from error
        finally:
            model.to('cpu')
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    if mesh.is_empty or not np.isfinite(mesh.vertices).all():
        raise RuntimeError('TripoSR did not produce valid surface geometry.')
    mesh.apply_transform(Y_UP_TRANSFORM)
    mesh.vertices-=mesh.bounds.mean(axis=0)
    extent=float(np.max(mesh.extents))
    if extent <= 0: raise RuntimeError('TripoSR mesh has zero extent.')
    mesh.vertices/=extent
    mesh.fix_normals(multibody=True)
    mesh.export(str(out))
    mesh.export(str(glb),include_normals=True)
    print(f'TripoSR exported {len(mesh.vertices)} vertices / {len(mesh.faces)} faces.',flush=True)
    return str(out)
