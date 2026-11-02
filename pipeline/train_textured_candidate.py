"""Train an isolated candidate; never auto-promote synthetic-only scores."""
from pathlib import Path
import os,sys,json,time,hashlib,random,traceback
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
ROOT=Path('D:/3dSCANNER');sys.path.insert(0,str(ROOT))
from pipeline.training_data import PhotoDepthDataset,CONTRACT,read_manifest
from pipeline.train_ai_3d import HighPrecision3DReconstructionNet
RUN=ROOT/'data/datasets/photo_v2_run';OUT=ROOT/'data/models/photo_v2_textured_candidate';OUT.mkdir(parents=True,exist_ok=True)
def save(path,value):
    if path.name=='status.json':
        value=dict(value,updated_at=time.time())
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temp,path)
def objective(pred,batch):
    depth,normal,mask=pred;target=batch['depth'];fg=batch['mask'];n=batch['normals']
    depth_loss=((depth-target).abs()*fg).sum()/fg.sum().clamp_min(1)
    interior=1-F.max_pool2d(1-fg,3,1,1)
    normal_loss=((1-(normal*n).sum(1,keepdim=True))*interior).sum()/interior.sum().clamp_min(1)
    edge=depth.new_tensor(0.)
    for axis in [2,3]:
        pd=torch.diff(depth,dim=axis);td=torch.diff(target,dim=axis)
        a=[slice(None)]*4;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
        valid=fg[tuple(a)]*fg[tuple(b)]
        edge+=((pd-td).abs()*valid).sum()/valid.sum().clamp_min(1)
    with torch.amp.autocast('cuda',enabled=False):
        mask_loss=F.binary_cross_entropy(mask.float().clamp(1e-6,1-1e-6),fg.float())
    return depth_loss+.2*normal_loss+.5*edge+.25*mask_loss

def evaluate(model,loader,device):
    model.eval();totals={'depth_mae':0.,'mask_iou':0.,'constant_depth_mae':0.};count=0
    with torch.no_grad():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()};d,n,m=model(batch['image']);fg=batch['mask'];target=batch['depth']
            for i in range(len(d)):
                valid=fg[i]>.5;mae=(d[i][valid]-target[i][valid]).abs().mean()
                base=(target[i][valid]-.5).abs().mean();pm=m[i]>.5
                totals['depth_mae']+=float(mae);totals['constant_depth_mae']+=float(base)
                totals['mask_iou']+=float((pm&valid).sum()/(pm|valid).sum().clamp_min(1));count+=1
    return {k:v/max(count,1) for k,v in totals.items()}

def main():
    torch.set_num_threads(4);random.seed(20261022);np.random.seed(20261022);torch.manual_seed(20261022)
    status=RUN/'preparation_status.json';started=time.time()
    while True:
        state=json.loads(status.read_text()) if status.exists() else {}
        if state.get('stage')=='prepared':break
        if state.get('stage')=='failed':raise RuntimeError(state.get('error'))
        if time.time()-started>3600:raise RuntimeError('Preparation timeout; training not started')
        save(OUT/'status.json',{'status':'waiting_for_preparation','preparation':state});time.sleep(10)
    manifest=ROOT/'data/datasets/training_manifest.json';doc=read_manifest(manifest)
    train=PhotoDepthDataset(manifest,'train',384,True);val=PhotoDepthDataset(manifest,'val',384,False)
    # Verify every aligned sample before allocating the candidate model.
    for dataset in [train,val]:
        for i in range(len(dataset)):dataset[i]
    if not torch.cuda.is_available():raise RuntimeError('GPU not available; no silent CPU training')
    device=torch.device('cuda');model=HighPrecision3DReconstructionNet().to(device)
    training=DataLoader(train,batch_size=4,shuffle=True,num_workers=0);validation=DataLoader(val,batch_size=4,num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)
    scaler=torch.amp.GradScaler('cuda');best=float('inf');stale=0;history=[]
    manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
    run_training_start=time.time();last_heartbeat=0
    for epoch in range(1,31):
        model.train();loss_sum=0.;batches=0;epoch_start=time.time()
        for batch in training:
            batch={k:v.to(device) for k,v in batch.items()}
            # Mild colour variation; retain spatial alignment and supervision.
            gain=.85+.3*torch.rand(len(batch['image']),1,1,1,device=device)
            batch['image']=(batch['image']*gain).clamp(0,1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):loss=objective(model(batch['image']),batch)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update();loss_sum+=float(loss.detach());batches+=1
            now=time.time()
            if now-last_heartbeat>=2:
                completed=(epoch-1)+batches/max(len(training),1)
                remaining=(now-run_training_start)/completed*(30-completed) if completed>0 else None
                save(OUT/'status.json',{'status':'training','phase':'training_batches','current_epoch':epoch,'total_epochs':30,
                    'batch':batches,'total_batches':len(training),'elapsed_seconds':now-started,'eta_seconds':remaining,
                    'eta_kind':'estimate_to_max_epochs_excludes_final_test','early_stopping_possible':True,'candidate_only':True})
                last_heartbeat=now
        save(OUT/'status.json',{'status':'training','phase':'validation','current_epoch':epoch,'total_epochs':30,'elapsed_seconds':time.time()-started,'candidate_only':True})
        metrics=evaluate(model,validation,device);scheduler.step()
        row={'epoch':epoch,'train_loss':loss_sum/max(batches,1),**metrics,'seconds':time.time()-epoch_start};history.append(row)
        checkpoint={'model_state_dict':model.state_dict(),'img_size':192,'epoch':epoch,'data_contract':CONTRACT,'manifest_sha256':manifest_hash,'synthetic_only':True,'metrics':metrics}
        if metrics['depth_mae']<best:
            best=metrics['depth_mae'];stale=0;temporary=OUT/'best.tmp';torch.save(checkpoint,temporary);os.replace(temporary,OUT/'best.pth')
        else:stale+=1
        save(OUT/'history.json',history);save(OUT/'status.json',{'status':'training','current_epoch':epoch,'total_epochs':30,'train_pairs':len(train),'validation_pairs':len(val),'best_depth_mae':best,'metrics':metrics,'elapsed_seconds':time.time()-started,'candidate_only':True})
        elapsed=time.time()-started;remaining=sum(r['seconds'] for r in history[-5:])/len(history[-5:])*(30-epoch)
        print(f'[TIMER] Epoch {epoch}/30 | elapsed {elapsed/60:.1f} min | remaining about {remaining/60:.1f} min (max epochs; final test extra; early stop possible)',flush=True)
        print(json.dumps(row),flush=True)
        if stale>=8:break
    checkpoint=torch.load(OUT/'best.pth',map_location=device,weights_only=True);model.load_state_dict(checkpoint['model_state_dict'])
    # The final test objects are never used to choose the best epoch.
    test=PhotoDepthDataset(manifest,'val',192,False);test.records=doc['held_out_test']
    save(OUT/'status.json',{'status':'evaluating','phase':'final_test','current_epoch':history[-1]['epoch'],'total_epochs':30,'elapsed_seconds':time.time()-started,'candidate_only':True})
    test_metrics=evaluate(model,DataLoader(test,batch_size=6),device)
    report={'status':'needs_real_photo_evaluation','contract':CONTRACT,'held_out_objects':True,'synthetic_only':True,'checkpoint_sha256':hashlib.sha256((OUT/'best.pth').read_bytes()).hexdigest(),'best_epoch':checkpoint['epoch'],'test_pairs':len(test),'test_metrics':test_metrics,'activated':False,'reason':'Synthetic test performance does not qualify real-photo inference or full textured 360 reconstruction.'}
    save(OUT/'best.evaluation.json',report)
    save(OUT/'status.json',{'status':'completed','epochs':history[-1]['epoch'],'best_epoch':checkpoint['epoch'],'train_pairs':len(train),'validation_pairs':len(val),'test_pairs':len(test),'test_metrics':test_metrics,'candidate_only':True,'activated':False,'elapsed_seconds':time.time()-started})

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(OUT/'status.json',{'status':'failed','error':str(error)});traceback.print_exc();sys.exit(1)
