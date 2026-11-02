from pathlib import Path
import os,sys,time,json,traceback
ROOT=Path('D:/3dSCANNER');sys.path[:0]=[str(ROOT),str(ROOT/'engines/Hunyuan3D-2-main'),str(ROOT/'engines/python-packages')]
os.environ['HY3DGEN_MODELS']=str(ROOT/'engines/models');os.environ['HF_HOME']=str(ROOT/'data/cache/huggingface');os.environ['OMP_NUM_THREADS']='4'
out=ROOT/'data/software_checks/hunyuan_texture_setup';out.mkdir(parents=True,exist_ok=True)
def record(stage,**extra):(out/'texture_test.json').write_text(json.dumps({'stage':stage,'updated':time.time(),**extra},indent=2))
try:
    # Fail before loading multi-GB models when Windows commit memory is exhausted.
    if sys.platform == 'win32':
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),
                      ('total_phys',ctypes.c_ulonglong),('avail_phys',ctypes.c_ulonglong),
                      ('total_page',ctypes.c_ulonglong),('avail_page',ctypes.c_ulonglong),
                      ('total_virtual',ctypes.c_ulonglong),('avail_virtual',ctypes.c_ulonglong),
                      ('avail_extended',ctypes.c_ulonglong)]
        memory=MemoryStatus();memory.length=ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise ctypes.WinError()
        free_gb=memory.avail_page/1024**3
        if free_gb<12:
            raise RuntimeError(f'Insufficient Windows commit headroom: {free_gb:.1f} GiB available; free at least 12 GiB by saving and closing memory-heavy apps before retrying. Downloaded weights are intact.')
    import torch,trimesh
    torch.set_num_threads(4)
    from PIL import Image
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from hy3dgen.texgen.pipelines import Hunyuan3DTexGenConfig
    record('loading_texture_models')
    model_root=ROOT/'engines/models/tencent/Hunyuan3D-2'
    config=Hunyuan3DTexGenConfig(str(model_root/'hunyuan3d-delight-v2-0'),str(model_root/'hunyuan3d-paint-v2-0-turbo'),'hunyuan3d-paint-v2-0-turbo')
    config.device='cpu'
    pipe=Hunyuan3DPaintPipeline(config)
    if hasattr(pipe,'enable_model_cpu_offload'):pipe.enable_model_cpu_offload()
    mesh=trimesh.load(ROOT/'data/software_checks/day1_baseline/hunyuan_clean_shape.glb',force='mesh')
    # Reduce geometry for UV baking only; preserve the original high-res baseline.
    if len(mesh.faces)>100000:mesh=mesh.simplify_quadric_decimation(face_count=100000)
    image=Image.open(ROOT/'data/jobs/35c5a5d8/output/prep/IMG_0030_foreground.png').convert('RGBA')
    record('painting_portrait')
    result=pipe(mesh,image=image);result=result[0] if isinstance(result,(list,tuple)) else result
    result.export(out/'portrait_textured.glb')
    record('completed',output=str(out/'portrait_textured.glb'),estimated_unseen_texture=True)
except Exception as error:record('failed',error=str(error) or repr(error),error_type=type(error).__name__);traceback.print_exc();raise
