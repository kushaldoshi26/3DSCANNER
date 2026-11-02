import json
import numpy as np
import pytest
import cv2
import trimesh
from pipeline.point_cloud import depth_surface,surface_cloud
from pipeline.training_data import PhotoDepthDataset,read_manifest,checkpoint_status,CONTRACT
from pipeline.fabric_detail import infer_fabric
from PIL import Image

def test_surface_points_have_real_depth_and_only_foreground_colors(tmp_path):
    rgba=np.zeros((32,32,4),np.uint8);rgba[:,:,:3]=[0,255,0]
    rgba[4:28,8:24]=[220,180,60,255]
    depth=np.indices((32,32))[0].astype(np.float32);depth[rgba[:,:,3]==0]=1e6
    mesh=depth_surface(rgba,depth)
    assert np.ptp(mesh.vertices[:,2])>.2
    assert (mesh.visual.vertex_colors[:,:3]==[220,180,60]).all()
    assert len(mesh.faces)>100 and not mesh.is_watertight
    path=tmp_path/'surface.glb';mesh.export(path)
    cloud=surface_cloud(str(path),1,1,count=500)
    points=np.asarray(cloud['positions']).reshape(-1,3)
    assert np.ptp(points[:,2])>.2 and cloud['kind']=='reconstructed_surface'

def test_flat_depth_is_rejected():
    rgba=np.full((16,16,4),255,np.uint8)
    with pytest.raises(ValueError,match='flat'):depth_surface(rgba,np.ones((16,16),np.float32))

def manifest(tmp_path):
    cv2.imwrite(str(tmp_path/'rgb.png'),np.full((16,16,3),100,np.uint8));cv2.imwrite(str(tmp_path/'mask.png'),np.full((16,16),255,np.uint8))
    np.save(tmp_path/'depth.npy',np.full((16,16),.5,np.float32));normal=np.zeros((16,16,3),np.float32);normal[:,:,2]=1;np.save(tmp_path/'normals.npy',normal)
    samples=[dict(object_id=obj,split=split,source_type='textured_render',rgb='rgb.png',depth='depth.npy',normals='normals.npy',mask='mask.png') for obj,split in [('object-a','train'),('object-b','val')]]
    path=tmp_path/'training_manifest.json';path.write_text(json.dumps({'contract':CONTRACT,'samples':samples}));return path

def test_photo_contract_has_disjoint_deterministic_validation(tmp_path):
    path=manifest(tmp_path);validation=PhotoDepthDataset(path,'val',img_size=32,augment=False)
    first=validation[0];second=validation[0]
    assert np.array_equal(first['image'].numpy(),second['image'].numpy())
    assert first['depth'].shape==(1,32,32)
    assert first['normals'][2].min()==1

def test_object_leakage_is_rejected(tmp_path):
    path=manifest(tmp_path);doc=json.loads(path.read_text());doc['samples'][1]['object_id']='object-a';path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='leakage'):read_manifest(path)

def test_training_requires_photo_pairs_and_preserves_legacy_weights(tmp_path,monkeypatch):
    import pipeline.train_ai_3d as train
    monkeypatch.setattr(train,'DATASETS_DIR',tmp_path);monkeypatch.setattr(train,'MODELS_DIR',tmp_path/'models')
    with pytest.raises(ValueError,match='Photo training is not ready'):train.train_custom_3d_ai_model(epochs=1)
    assert train.TRAINING_STATE['status']=='failed'
    assert not list((tmp_path/'models').glob('*.pth'))

def test_unqualified_checkpoint_is_not_ready(tmp_path):
    assert not checkpoint_status(tmp_path/'absent.pth')['ready']
    file=tmp_path/'weights.pth';file.write_bytes(b'not qualified')
    assert not checkpoint_status(file)['ready']

def test_fabric_continuation_does_not_apply_without_a_face():
    assert infer_fabric(Image.new('RGBA',(512,512),(200,170,60,255))) is None

def test_cloth_patch_excludes_face_and_is_labelled_inferred(monkeypatch):
    import pipeline.fabric_detail as fabric
    class Detector:
        def detectMultiScale(self,*args,**kwargs):return np.array([[210,30,80,80]])
    monkeypatch.setattr(fabric.cv2,'CascadeClassifier',lambda *a:Detector())
    image=np.zeros((512,512,4),np.uint8);image[20:490,130:380]=[210,160,40,255]
    image[30:110,210:290]=[170,110,95,255]
    result=infer_fabric(Image.fromarray(image))
    assert result is not None and result['report']['observed_back'] is False
    assert result['report']['patch_pixels'][1]>110
