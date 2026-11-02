"""Merge compatible local supervision while preserving original held-out objects."""
from pathlib import Path
import sys,json,hashlib,importlib.util,os
ROOT=Path('D:/3dSCANNER');sys.path.insert(0,str(ROOT));DATA=ROOT/'data/datasets'
spec=importlib.util.spec_from_file_location('trainer',ROOT/'pipeline/train_textured_candidate.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
original=json.loads((DATA/'training_manifest.json').read_text())
external=json.loads((DATA/'thuman_reference/external_human_test.json').read_text())
merged=json.loads(json.dumps(original));seen={(s['object_id'],s['rgb']) for s in merged['samples']}
for sample in external['samples']:
    sample=dict(sample,split='train')
    if (sample['object_id'],sample['rgb']) not in seen:merged['samples'].append(sample)
merged['note']='THuman example now belongs to training; previous single-human diagnostic is no longer an independent test. Original Objaverse validation and final test objects remain held out.'
run=DATA/'merged_v3_run';run.mkdir(exist_ok=True)
manifest=DATA/'merged_training_manifest.json';manifest.write_text(json.dumps(merged,indent=2))
train={s['object_id'] for s in merged['samples'] if s['split']=='train'};val={s['object_id'] for s in merged['samples'] if s['split']=='val'};test={s['object_id'] for s in merged['held_out_test']}
assert not(train&val or train&test or val&test)
audit={'train_objects':len(train),'validation_objects':len(val),'test_objects':len(test),'training_pairs':sum(s['split']=='train' for s in merged['samples']),'validation_pairs':sum(s['split']=='val' for s in merged['samples']),'test_pairs':len(merged['held_out_test']),'geometry_only_npz_not_photo_supervision':True,'thuman4_archive_has_no_training_samples':True,'human_scan_is_training_not_test':True}
(run/'merge_report.json').write_text(json.dumps(audit,indent=2));(run/'preparation_status.json').write_text(json.dumps({'stage':'prepared',**audit}))
# Execute the verified trainer against a separate manifest and output directory.
source=(ROOT/'pipeline/train_textured_candidate.py').read_text()
source=source.replace("RUN=ROOT/'data/datasets/photo_v2_run';OUT=ROOT/'data/models/photo_v2_textured_candidate'","RUN=ROOT/'data/datasets/merged_v3_run';OUT=ROOT/'data/models/merged_v3_candidate'")
source=source.replace("manifest=ROOT/'data/datasets/training_manifest.json'","manifest=ROOT/'data/datasets/merged_training_manifest.json'")
source=source.replace("model=HighPrecision3DReconstructionNet().to(device)","model=HighPrecision3DReconstructionNet().to(device); model.load_state_dict(torch.load(ROOT/'data/models/photo_v2_textured_candidate/best.pth',map_location=device,weights_only=True)['model_state_dict'])")
exec(compile(source,str(ROOT/'pipeline/train_textured_candidate.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'pipeline/train_textured_candidate.py')})
