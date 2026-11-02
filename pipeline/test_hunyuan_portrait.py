from pathlib import Path
import sys,os,json,time,traceback
ROOT=Path('D:/3dSCANNER');sys.path.insert(0,str(ROOT));sys.path.insert(0,'D:/3dSCANNER/engines/Hunyuan3D-2-main')
os.environ['HY3DGEN_MODELS']='D:/3dSCANNER/engines/models'
os.environ['HF_HOME']='D:/3dSCANNER/engines/hf-cache'
os.environ['OMP_NUM_THREADS']='4'
out=ROOT/'data/software_checks/hunyuan_portrait';out.mkdir(parents=True,exist_ok=True)
status=out/'status.json'
def record(stage,**extra):status.write_text(json.dumps({'stage':stage,'updated':time.time(),**extra},indent=2))
try:
    import torch
    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    record('loading_shape')
    torch.set_num_threads(4)
    pipeline=Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2',device='cuda',use_safetensors=True)
    # Shape stage fits the 16GB GPU; avoid incompatible upstream offload helper.
    record('generating_shape')
    image=Image.open(ROOT/'data/jobs/35c5a5d8/output/prep/IMG_0030_foreground.png').convert('RGBA')
    meshes=pipeline(image=image,num_inference_steps=30,octree_resolution=384,num_chunks=8000,generator=torch.Generator(device='cuda').manual_seed(42))
    mesh=meshes[0];mesh.export(out/'shape.glb');mesh.export(out/'shape.stl')
    from pipeline.mesh_quality import validate_mesh
    report=validate_mesh(str(out/'shape.glb'),coordinate_space='normalized',check_self_intersection=False)
    (out/'quality_report.json').write_text(json.dumps(report,indent=2))
    record('shape_completed',vertices=len(mesh.vertices),faces=len(mesh.faces),quality=report)
    print(json.dumps(report,indent=2),flush=True)
except Exception as error:record('failed',error=str(error));traceback.print_exc();raise
