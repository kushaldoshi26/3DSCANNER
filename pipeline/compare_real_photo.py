from pathlib import Path
import sys,json,gc
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
ROOT=Path('D:/3dSCANNER');sys.path.insert(0,str(ROOT))
from pipeline.train_ai_3d import HighPrecision3DReconstructionNet
from pipeline.point_cloud import depth_surface
out=ROOT/'data/software_checks/real_photo_comparison';out.mkdir(parents=True,exist_ok=True)
rgba=Image.open(ROOT/'data/jobs/35c5a5d8/output/prep/IMG_0030_foreground.png').convert('RGBA');rgba.thumbnail((512,512));array=np.asarray(rgba);foreground=array[:,:,3]>127
rgb=array[:,:,:3].copy();rgb[~foreground]=127
tensor=torch.from_numpy(cv2.resize(rgb,(192,192)).astype('f4')/255).permute(2,0,1)[None]
device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4);tensor=tensor.to(device)
sheet=Image.new('RGB',(960,470),(16,38,48));draw=ImageDraw.Draw(sheet);results=[]
source=Image.fromarray(rgb);source.thumbnail((300,410));sheet.paste(source,(10,40));draw.text((10,12),'Same isolated source photo',fill='white')
for index,name in enumerate(['merged_v3_candidate','merged_v4_candidate'],1):
    model=HighPrecision3DReconstructionNet().to(device);checkpoint=torch.load(ROOT/'data/models'/name/'best.pth',map_location=device,weights_only=True);model.load_state_dict(checkpoint['model_state_dict']);model.eval()
    with torch.no_grad():depth,normal,mask=model(tensor)
    depth=cv2.resize(depth[0,0].cpu().numpy(),rgba.size);predmask=cv2.resize(mask[0,0].cpu().numpy(),rgba.size)>.5
    overlap=float((predmask&foreground).sum()/max((predmask|foreground).sum(),1))
    mesh=depth_surface(array,depth);mesh.export(out/(name+'.glb'))
    low,high=np.percentile(depth[foreground],[2,98]);preview=np.zeros(depth.shape,dtype='uint8');preview[foreground]=(np.clip((depth[foreground]-low)/(high-low+1e-8),0,1)*255).astype('uint8')
    picture=Image.fromarray(preview).convert('RGB');picture.thumbnail((300,410));sheet.paste(picture,(index*320+10,40));draw.text((index*320+10,12),name+' depth',fill='white')
    results.append({'model':name,'mask_overlap_with_preprocessor':overlap,'reference_is_ground_truth':False,'depth_accuracy_available':False,'vertices':len(mesh.vertices),'faces':len(mesh.faces)})
    del model;gc.collect();torch.cuda.empty_cache()
sheet.save(out/'v3-v4-comparison.jpg');(out/'results.json').write_text(json.dumps({'source':'IMG_0030.jpg','real_photo_3d_ground_truth':False,'activated':False,'results':results},indent=2));print(json.dumps(results,indent=2))
