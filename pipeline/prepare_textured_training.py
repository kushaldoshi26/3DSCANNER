"""Bounded, resumable Objaverse acquisition and dense aligned supervision."""
from pathlib import Path
import os,json,gzip,hashlib,re,random,shutil,time
import requests
import numpy as np
from PIL import Image
import trimesh,pyrender

ROOT=Path('D:/3dSCANNER'); DATA=ROOT/'data/datasets'; RUN=DATA/'photo_v2_run'
RUN.mkdir(parents=True,exist_ok=True)
CACHE=DATA/'objaverse_cache'
STATUS=RUN/'preparation_status.json'
def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temporary,path)
def get(url,target,limit=100*1024**2):
    if target.exists():return
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_suffix('.part');size=0
    try:
        with requests.get(url,stream=True,timeout=(15,90)) as response:
            response.raise_for_status()
            if int(response.headers.get('Content-Length',0))>limit:raise ValueError('Asset exceeds per-file budget')
            with partial.open('wb') as stream:
                for chunk in response.iter_content(1024*1024):
                    size+=len(chunk)
                    if size>limit:raise ValueError('Asset exceeds per-file budget')
                    stream.write(chunk)
        os.replace(partial,target)
    finally:
        if partial.exists():partial.unlink()

def render_asset(path,uid,renderer):
    source=trimesh.load(path,force='scene');scene=source.copy()
    geometries=list(scene.geometry.values())
    if not geometries or sum(len(g.faces) for g in geometries)>600000:raise ValueError('Empty or oversized geometry')
    if not any(g.visual.kind=='texture' and getattr(g.visual,'uv',None) is not None for g in geometries):raise ValueError('No textured surface')
    if not np.isfinite(scene.bounds).all():raise ValueError('Nonfinite geometry')
    centre=scene.bounds.mean(axis=0);extent=np.ptp(scene.bounds,axis=0).max()
    if extent<1e-8:raise ValueError('Degenerate geometry')
    transform=np.eye(4);transform[:3,:3]/=extent;transform[:3,3]=-centre/extent
    scene.apply_transform(transform)
    render_scene=pyrender.Scene.from_trimesh_scene(scene,bg_color=[.5,.5,.5,0],ambient_light=[.65]*3)
    camera=pyrender.OrthographicCamera(xmag=.7,ymag=.7,znear=.01,zfar=10)
    node=render_scene.add(camera)
    light=render_scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=1.3))
    folder=RUN/'pairs'/uid;folder.mkdir(parents=True,exist_ok=True)
    pairs=[]
    for index in range(12):
        yaw=index*2*np.pi/12;elevation=np.radians(10 if index%2 else 20)
        position=2*np.array([np.sin(yaw)*np.cos(elevation),np.sin(elevation),np.cos(yaw)*np.cos(elevation)])
        z=position/np.linalg.norm(position);x=np.cross([0,1,0],z);x/=np.linalg.norm(x);y=np.cross(z,x)
        pose=np.eye(4);pose[:3,:3]=np.column_stack([x,y,z]);pose[:3,3]=position
        render_scene.set_pose(node,pose);render_scene.set_pose(light,pose)
        rgb,depth=renderer.render(render_scene,flags=pyrender.RenderFlags.RGBA)
        mask=depth>0
        if mask.sum()<1000:continue
        low,high=np.percentile(depth[mask],[0,100]);span=high-low
        if span<1e-4:continue
        near=np.where(mask,.05+.9*(high-depth)/span,0).astype('f4')
        # Camera-space visible-surface normals from metric orthographic depth.
        filled=depth.copy()
        import cv2
        filled[~mask]=float(np.median(depth[mask]))
        dzdy,dzdx=np.gradient(filled,1.4/192)
        normals=np.stack([dzdx,-dzdy,np.ones_like(depth)],axis=-1)
        normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-8);normals[~mask]=0
        rgb[~mask,:3]=127;rgb[:,:,3]=255
        prefix=folder/f'{index:02d}'
        Image.fromarray(rgb[:,:,:3]).save(str(prefix)+'_rgb.png')
        Image.fromarray(mask.astype('uint8')*255).save(str(prefix)+'_mask.png')
        np.save(str(prefix)+'_depth.npy',near);np.save(str(prefix)+'_normals.npy',normals.astype('f4'))
        pairs.append({'object_id':'objaverse:'+uid,'source_type':'textured_render',
                      'rgb':str(Path(str(prefix)+'_rgb.png').relative_to(DATA)).replace('\\','/'),
                      'mask':str(Path(str(prefix)+'_mask.png').relative_to(DATA)).replace('\\','/'),
                      'depth':str(Path(str(prefix)+'_depth.npy').relative_to(DATA)).replace('\\','/'),
                      'normals':str(Path(str(prefix)+'_normals.npy').relative_to(DATA)).replace('\\','/'),
                      'camera_pose':pose.tolist(),'projection':'orthographic','world_width':1.4})
    if len(pairs)<6:raise ValueError('Insufficient usable views')
    return pairs

def main():
    annotations=json.load(gzip.open(CACHE/'metadata/000-023.json.gz'))
    paths=json.load(gzip.open(CACHE/'object-paths.json.gz'))
    cached={p.stem:p for p in (CACHE/'glbs').rglob('*.glb')}
    eligible=[uid for uid,a in annotations.items() if a.get('license') in ['cc0','by'] and a.get('vertexCount',0)<300000]
    rng=random.Random(20261022);rng.shuffle(eligible)
    human=re.compile(r'person|human|woman|dress|shirt|statue|bust|chair',re.I)
    fresh=sorted([u for u in eligible if u not in cached],key=lambda u:not bool(human.search(annotations[u].get('name',''))))[:96]
    reused=[u for u in eligible if u in cached][:160]
    selected=fresh+reused
    records=[];sources=[];errors=[];seen=set();renderer=pyrender.OffscreenRenderer(192,192)
    downloads=0;download_bytes=0
    try:
        for uid in selected:
            if len(sources)>=128:break
            state={'stage':'preparing','objects':len(sources),'pairs':len(records),'downloaded':downloads,'download_bytes':download_bytes,'errors':len(errors),'current':uid,'updated':time.time()};save(STATUS,state);print(json.dumps(state),flush=True)
            try:
                if uid in cached:path=cached[uid]
                else:
                    if downloads>=64 or download_bytes>6*1024**3:continue
                    path=RUN/'originals'/f'{uid}.glb';existed=path.exists()
                    get('https://huggingface.co/datasets/allenai/objaverse/resolve/main/'+paths[uid],path)
                    downloads+=1;download_bytes+=path.stat().st_size
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen:continue
                seen.add(digest)
                pair_file=RUN/'pairs'/uid/'records.json'
                if pair_file.exists():pairs=json.loads(pair_file.read_text())
                else:pairs=render_asset(path,uid,renderer);save(pair_file,pairs)
                a=annotations[uid]
                sources.append({'uid':uid,'name':a.get('name'),'license':a.get('license'),'author':a.get('user',{}).get('displayName'),'url':a.get('viewerUrl'),'sha256':digest,'path':str(path)})
                records.extend(pairs)
            except Exception as error:
                errors.append({'uid':uid,'error':str(error)});print('SKIP',uid,str(error),flush=True)
        if len(sources)<24:raise RuntimeError('Too few verified textured objects; do not train')
        ids=sorted({r['object_id'] for r in records});rng.shuffle(ids)
        val=set(ids[:max(4,len(ids)//8)]);test=set(ids[max(4,len(ids)//8):max(4,len(ids)//8)*2])
        samples=[];test_samples=[]
        for record in records:
            if record['object_id'] in test:test_samples.append(record);continue
            record['split']='val' if record['object_id'] in val else 'train';samples.append(record)
        manifest={'contract':'rgb_depth_nearness_v2','seed':20261022,'sources':'photo_v2_run/sources.json','synthetic_only':True,'samples':samples,'held_out_test':test_samples}
        save(RUN/'sources.json',sources);save(RUN/'errors.json',errors)
        target=DATA/'training_manifest.json'
        if target.exists():shutil.copy2(target,RUN/f'previous_manifest_{int(time.time())}.json')
        save(target,manifest)
        save(STATUS,{'stage':'prepared','objects':len(sources),'train_pairs':sum(s['split']=='train' for s in samples),'val_pairs':sum(s['split']=='val' for s in samples),'test_pairs':len(test_samples),'downloaded':downloads,'download_bytes':download_bytes,'errors':len(errors),'synthetic_only':True,'updated':time.time()})
    finally:renderer.delete()

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(STATUS,{'stage':'failed','error':str(error),'updated':time.time()});raise
