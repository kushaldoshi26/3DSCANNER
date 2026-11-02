from pathlib import Path
import sys,time,json,traceback,os
sys.path.insert(0,'D:/3dSCANNER')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']='4'
from pipeline.hunyuan3d_inference import generate_hunyuan3d_mesh
out=Path('D:/3dSCANNER/data/software_checks/hunyuan_statue');out.mkdir(parents=True,exist_ok=True)
started=time.time()
def record(stage,**kw):(out/'status.json').write_text(json.dumps({'stage':stage,'updated':time.time(),'elapsed_seconds':time.time()-started,**kw},indent=2))
try:
    record('generating_shape_and_texture')
    generate_hunyuan3d_mesh('D:/3dSCANNER/data/jobs/9db914a1/output/prep/Screenshot 2026-08-09 182222_foreground.png',out/'statue.stl',out/'statue_textured.glb',quality='high',paint_face_limit=100000)
    record('completed',output=str(out/'statue_textured.glb'))
except BaseException as e:
    record('failed',error=repr(e));traceback.print_exc();raise