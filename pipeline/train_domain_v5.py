"""Test background/lighting robustness; separate output, no auto promotion."""
from pathlib import Path
import sys,random
import torch
ROOT=Path('D:/3dSCANNER');sys.path.insert(0,str(ROOT))
import pipeline.training_data as data
original=data.PhotoDepthDataset.__getitem__
def augmented(self,index):
    sample=original(self,index)
    if self.augment:
        image=sample['image'];mask=sample['mask'];h,w=image.shape[1:]
        # Procedural backgrounds only: no use of evaluation portraits for training.
        colours=torch.rand(3,2,2)
        background=torch.nn.functional.interpolate(colours[None],size=(h,w),mode='bilinear',align_corners=False)[0]
        background=(background+torch.randn_like(background)*.025).clamp(0,1)
        gain=torch.empty(3,1,1).uniform_(.85,1.15);offset=random.uniform(-.04,.04)
        sample['image']=((image*gain+offset).clamp(0,1)*mask+background*(1-mask)).contiguous()
    return sample
data.PhotoDepthDataset.__getitem__=augmented
source=(ROOT/'pipeline/train_textured_candidate.py').read_text()
source=source.replace("RUN=ROOT/'data/datasets/photo_v2_run';OUT=ROOT/'data/models/photo_v2_textured_candidate'","RUN=ROOT/'data/datasets/merged_v3_run';OUT=ROOT/'data/models/domain_v5_candidate'")
source=source.replace("manifest=ROOT/'data/datasets/training_manifest.json'","manifest=ROOT/'data/datasets/merged_training_manifest.json'")
source=source.replace('lr=2e-4','lr=5e-5')
source=source.replace('model=HighPrecision3DReconstructionNet().to(device)',"model=HighPrecision3DReconstructionNet().to(device);model.load_state_dict(torch.load(ROOT/'data/models/merged_v4_candidate/best.pth',map_location=device,weights_only=True)['model_state_dict'])")
exec(compile(source,str(ROOT/'pipeline/train_textured_candidate.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'pipeline/train_textured_candidate.py')})
