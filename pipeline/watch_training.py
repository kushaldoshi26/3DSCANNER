"""Read-only terminal timer; Ctrl+C exits monitoring, not training."""
from pathlib import Path
import argparse,json,time,sys
def duration(seconds):
    if seconds is None:return 'calculating'
    seconds=max(0,int(seconds));return f'{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}'
parser=argparse.ArgumentParser();parser.add_argument('run',nargs='?',default='domain_v5_candidate');args=parser.parse_args()
name=Path(args.run)
if name.name!=args.run or args.run in ['.','..']:raise SystemExit('Use a candidate folder name only.')
root=Path(__file__).resolve().parent.parent;folder=root/'data/models'/name;status=folder/'status.json'
print('Training timer | '+args.run+' | Ctrl+C stops this monitor only.',flush=True)
try:
    while True:
        try:
            d=json.loads(status.read_text());state=d.get('status','unknown');terminal=state in ['completed','failed','cancelled']
            epoch=d.get('current_epoch',d.get('epochs',0));total=d.get('total_epochs','?');elapsed=d.get('elapsed_seconds')
            age=max(0,time.time()-d['updated_at']) if 'updated_at' in d else None
            if not terminal and elapsed is not None and age is not None:elapsed+=age
            eta=d.get('eta_seconds');phase=d.get('phase','')
            if eta is not None and age is not None:eta=max(0,eta-age)
            text=f'{state.upper()} | {phase} | epoch {epoch}/{total} | elapsed {duration(elapsed)}'
            if not terminal:text+=' | remaining ~'+duration(eta)+' (max epochs; test extra)'
            if age is not None and age>60 and not terminal:text+=' | NO RECENT UPDATE: inspect training process/log'
            print('\r'+text.ljust(160),end='',flush=True)
            if terminal:
                print('\nTraining process reports '+state+'.');break
        except FileNotFoundError:print('\rWaiting for '+str(status),end='',flush=True)
        except json.JSONDecodeError:pass
        time.sleep(1)
except KeyboardInterrupt:print('\nMonitor closed; training was not stopped.')
