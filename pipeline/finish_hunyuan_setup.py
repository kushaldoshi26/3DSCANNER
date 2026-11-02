"""Continue a real local setup after the resumable weight download finishes."""
from pathlib import Path
import sys,time,subprocess,json,os
root=Path('D:/3dSCANNER/data/software_checks/hunyuan_portrait');root.mkdir(parents=True,exist_ok=True)
weights=Path('D:/3dSCANNER/engines/models/tencent/Hunyuan3D-2/hunyuan3d-dit-v2-0/model.fp16.safetensors')
status=root/'setup_progress.json';started=time.time()
while not weights.exists():
    status.write_text(json.dumps({'stage':'waiting_for_weights','updated':time.time(),'weights':str(weights)},indent=2))
    if time.time()-started>12*3600:raise RuntimeError('Download not completed within 12 hours; rerun after connection recovers')
    time.sleep(15)
status.write_text(json.dumps({'stage':'testing_portrait','updated':time.time()}))
with (root/'test.log').open('w',encoding='utf-8') as log:
    code=subprocess.call([sys.executable,str(Path(__file__).with_name('test_hunyuan_portrait.py'))],stdout=log,stderr=log)
status.write_text(json.dumps({'stage':'shape_test_completed' if code==0 else 'shape_test_failed','returncode':code,'updated':time.time()},indent=2))
