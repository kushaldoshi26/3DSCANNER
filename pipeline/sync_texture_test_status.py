from pathlib import Path
import json,time,os
out=Path('D:/3dSCANNER/data/software_checks/hunyuan_texture_setup')
test=json.loads((out/'texture_test.json').read_text());started=test['updated']
for _ in range(3600):
    test=json.loads((out/'texture_test.json').read_text())
    stage=test.get('stage','testing_portrait_texture')
    payload={'stage':stage,'elapsed_seconds':max(0,time.time()-started),'updated':time.time(),'timer_scope':'texture phase','error':test.get('error')}
    p=out/'setup.sync.tmp';p.write_text(json.dumps(payload,indent=2));os.replace(p,out/'setup.json')
    if stage in ('completed','failed'):break
    time.sleep(2)