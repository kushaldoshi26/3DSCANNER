"""Run official archives sequentially with durable status and no extraction."""
from pathlib import Path
import subprocess,sys,json,time,os
ROOT=Path('D:/3dSCANNER/data/datasets/thuman4')
ROOT.mkdir(parents=True,exist_ok=True)
state={'status':'downloading','completed':[],'failed':[],'root':str(ROOT),'updated':time.time()}
def save():
    state['updated']=time.time();p=ROOT/'download_all.tmp';p.write_text(json.dumps(state,indent=2));os.replace(p,ROOT/'download_all_status.json')
for subject in ['00','01','02']:
    state['current_subject']=subject;save()
    with (ROOT/f'subject{subject}.log').open('a',encoding='utf-8') as log:
        code=subprocess.call([sys.executable,'D:/3dSCANNER/pipeline/download_thuman4.py','--subject',subject,'--root',str(ROOT),'--research-terms-accepted'],stdout=log,stderr=log)
    state['completed' if code==0 else 'failed'].append(subject);save()
state['status']='completed' if not state['failed'] else 'incomplete';save()
