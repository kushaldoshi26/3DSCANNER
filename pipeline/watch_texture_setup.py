from pathlib import Path
import json,time
root=Path(__file__).resolve().parent.parent/'data/software_checks/hunyuan_texture_setup'
print('Texture setup monitor | Ctrl+C closes monitoring only.',flush=True)
def read(name):
    try:return json.loads((root/name).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
def clock(seconds):
    if seconds is None:return 'calculating'
    s=max(0,int(seconds));return f'{s//3600:02}:{s%3600//60:02}:{s%60:02}'
try:
    while True:
        setup=read('setup.json');download=read('download.json');test=read('texture_test.json')
        run_started=setup.get('updated',0)-setup.get('elapsed_seconds',0)
        if test.get('updated',0)<run_started:
            test={'stage':'queued_after_download'}
        elapsed=setup.get('elapsed_seconds',0)
        if setup.get('stage') not in ['completed','failed']:
            elapsed+=max(0,time.time()-setup.get('updated',time.time()))
        print('\r'+f"Stage: {setup.get('stage','starting')} | elapsed {clock(elapsed)} | weights {download.get('completed_bytes',0)/1e9:.2f}/{download.get('total_bytes',0)/1e9:.2f} GB | download ETA ~{clock(download.get('eta_seconds')) if download.get('completed_bytes',0)>50e6 else 'calculating'} | test: {test.get('stage','pending')}".ljust(200),end='',flush=True)
        if setup.get('stage') in ['completed','failed'] or download.get('stage')=='failed':print('\nInspect setup.json / download.json for details.');break
        time.sleep(2)
except KeyboardInterrupt:print('\nMonitor closed; setup continues.')
