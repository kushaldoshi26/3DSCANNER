"""Explicit photo/depth contracts and object-disjoint validation."""
from pathlib import Path
import json,hashlib,random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

ROOT=Path(__file__).resolve().parent.parent
CONTRACT='rgb_depth_nearness_v2'

def read_manifest(path):
    path=Path(path)
    if not path.is_file():raise ValueError('Photo training is not ready: add training_manifest.json with aligned RGB, float depth, normals and masks. Geometry-only NPZ files are not photo training pairs.')
    doc=json.loads(path.read_text(encoding='utf-8'))
    if doc.get('contract')!=CONTRACT:raise ValueError('Manifest must use '+CONTRACT)
    samples=doc.get('samples',[])
    if not samples:raise ValueError('Training manifest has no samples')
    groups={}
    for sample in samples:
        if not sample.get('object_id'):raise ValueError('Each sample needs an object_id for leakage-free splitting')
        split=sample.get('split')
        if split not in ['train','val']:raise ValueError('Each sample needs train or val split')
        previous=groups.setdefault(sample['object_id'],split)
        if previous!=split:raise ValueError('Object leakage between training and validation: '+sample['object_id'])
        if sample.get('source_type') not in ['photo','textured_render']:raise ValueError('Training needs real photos or textured renders, not point splats')
        for key in ['rgb','depth','normals','mask']:
            if not (path.parent/sample.get(key,'')).is_file():raise ValueError('Missing aligned '+key+' file')
    if set(groups.values())!={'train','val'}:raise ValueError('At least one held-out validation object and one training object are required')
    return doc

class PhotoDepthDataset(Dataset):
    def __init__(self,path,split,img_size=256,augment=False):
        self.path=Path(path);self.doc=read_manifest(path);self.records=[s for s in self.doc['samples'] if s['split']==split]
        self.img_size=img_size;self.augment=augment
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        sample=self.records[index];base=self.path.parent
        rgb=cv2.imread(str(base/sample['rgb']));mask=cv2.imread(str(base/sample['mask']),cv2.IMREAD_GRAYSCALE)
        depth=np.load(base/sample['depth'],allow_pickle=False).astype(np.float32)
        normals=np.load(base/sample['normals'],allow_pickle=False).astype(np.float32)
        if rgb is None or mask is None:raise ValueError('Corrupt RGB or mask')
        if depth.shape!=rgb.shape[:2] or mask.shape!=depth.shape or normals.shape!=(*depth.shape,3):raise ValueError('RGB/depth/normal/mask alignment mismatch')
        selected=mask>127
        if not selected.any() or not np.isfinite(depth[selected]).all() or not np.isfinite(normals[selected]).all():raise ValueError('Empty or nonfinite supervision')
        if depth[selected].min()<0 or depth[selected].max()>1:raise ValueError('Depth must be relative nearness in [0,1]')
        shape=(self.img_size,self.img_size)
        rgb=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB),shape,interpolation=cv2.INTER_AREA).astype(np.float32)/255
        mask=cv2.resize(selected.astype(np.float32),shape,interpolation=cv2.INTER_NEAREST)
        depth=cv2.resize(np.where(selected,depth,0),shape,interpolation=cv2.INTER_NEAREST)*mask
        normals=cv2.resize(normals,shape,interpolation=cv2.INTER_NEAREST)
        normals/=np.maximum(np.linalg.norm(normals,axis=2,keepdims=True),1e-8);normals*=mask[:,:,None]
        if self.augment and random.random()<.5:
            rgb=rgb[:,::-1].copy();depth=depth[:,::-1].copy();mask=mask[:,::-1].copy();normals=normals[:,::-1].copy();normals[:,:,0]*=-1
        return {'image':torch.from_numpy(rgb.copy()).permute(2,0,1),'depth':torch.from_numpy(depth.copy())[None],
                'normals':torch.from_numpy(normals.copy()).permute(2,0,1),'mask':torch.from_numpy(mask.copy())[None]}

def training_readiness(root=None):
    root=Path(root or ROOT/'data'/'datasets');manifest=root/'training_manifest.json'
    result={'raw_meshes':sum(1 for p in root.rglob('*') if p.suffix.lower() in ['.stl','.obj','.ply','.glb','.gltf','.off']),
            'geometry_npz':len(list((root/'processed').glob('*.npz'))),'photo_training_ready':False,'photo_pairs':0,
            'rgb_depth_png_pairs':len(list((root/'renders').rglob('*_rgb.png')))+len(list((root/'sculpture_renders').rglob('*_rgb.png'))),
            'reason':'Legacy point-splat renders lack dense float depth, masks, normals and object-disjoint splits.',
            'contract':CONTRACT}
    try:
        doc=read_manifest(manifest)
        real=sum(s.get('source_type')=='photo' for s in doc['samples'])
        result.update(photo_training_ready=True,photo_pairs=len(doc['samples']),real_photo_pairs=real,
                      textured_render_pairs=len(doc['samples'])-real,synthetic_only=real==0,
                      reason='Aligned training contract available; synthetic renders are not real-photo qualification.')
    except ValueError as error:result['reason']=str(error)
    audit=root/'geometry_audit.json'
    if audit.is_file():
        checked=json.loads(audit.read_text(encoding='utf-8'))
        if checked.get('geometry_npz')==result['geometry_npz']:
            result['geometry_valid_npz']=checked['valid_geometry_npz']
            result['geometry_excluded_npz']=checked['invalid_geometry_npz']
    result['custom_checkpoint']=checkpoint_status()
    return result

def checkpoint_status(path=None):
    path=Path(path or ROOT/'data'/'models'/'photo_depth_v2_best.pth')
    # A low legacy loss is not evidence of generalisation to real photos.
    evaluation=path.with_suffix('.evaluation.json')
    if not path.exists():return {'ready':False,'reason':'Legacy checkpoints were trained on point-splat geometry and are not qualified for photos. Retrain using the v2 photo contract.'}
    if not evaluation.exists():return {'ready':False,'reason':'Checkpoint needs a held-out photo evaluation before use.'}
    report=json.loads(evaluation.read_text(encoding='utf-8'))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    ready=report.get('status')=='passed' and report.get('contract')==CONTRACT and report.get('checkpoint_sha256')==digest and report.get('held_out_objects',False)
    return {'ready':bool(ready),'reason':'Held-out photo evaluation passed' if ready else 'Checkpoint evaluation missing, failed or stale','path':str(path)}
