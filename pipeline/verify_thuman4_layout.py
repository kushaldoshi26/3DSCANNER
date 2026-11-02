"""Index official THuman4 camera views without inventing depth targets."""
from pathlib import Path
import json,argparse
import numpy as np
def inspect(subject):
    subject=Path(subject);calibration=json.loads((subject/'calibration.json').read_text())
    cameras={};warnings=[]
    for name,values in calibration.items():
        k=np.asarray(values['K'],dtype=float).reshape(3,3);r=np.asarray(values['R'],dtype=float).reshape(3,3);t=np.asarray(values['T'],dtype=float).reshape(3)
        if not all(np.isfinite(v).all() for v in [k,r,t]) or min(k[0,0],k[1,1])<=0:raise ValueError('Invalid calibration: '+name)
        if not np.allclose(r@r.T,np.eye(3),atol=1e-3):raise ValueError('Invalid camera rotation: '+name)
        images={p.stem:p for p in (subject/'images'/name).glob('*.jpg')};masks={p.stem:p for p in (subject/'masks'/name).glob('*.jpg')}
        frames=sorted(images.keys()&masks.keys())
        cameras[name]={'frames':frames,'K':k.tolist(),'R':r.tolist(),'T':t.tolist()}
        if not frames:warnings.append('No aligned RGB/mask frames for '+name)
    if len(cameras)<2:raise ValueError('At least two calibrated views are required')
    common=sorted(set.intersection(*(set(c['frames']) for c in cameras.values())))
    return {'subject':subject.name,'camera_count':len(cameras),'common_frames':common,'cameras':cameras,'warnings':warnings,
            'depth_training_ready':False,'reason':'RGB/masks/calibration are indexed. Reconstruct and verify clothed-surface depth before merging with supervised depth training; body fittings alone are not garment ground truth.'}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('subject');parser.add_argument('--output',required=True);args=parser.parse_args()
    result=inspect(args.subject);Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['cameras','common_frames']},indent=2))
