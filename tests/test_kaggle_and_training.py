import os
import sys
import tempfile
import pytest
import torch
import numpy as np
import trimesh
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.download_kaggle_datasets import (
    check_kaggle_credentials,
    render_multiview_training_pairs,
    process_all_datasets_into_npz,
    normalize_and_save_mesh,
    DATASET_SPECS
)
from pipeline.train_ai_3d import (
    HighPrecision3DReconstructionNet,
    ResidualBlock2D,
    Kaggle3DFoundationDataset,
    get_dataset_inventory,
    TRAINING_STATE
)
from pipeline.losses import ChamferLoss, LaplacianEdgeLoss, NormalConsistencyLoss
from api.main import app
from fastapi.testclient import TestClient


def test_kaggle_credentials_and_specs():
    # Should safely return boolean without throwing
    res = check_kaggle_credentials()
    assert isinstance(res, bool)
    assert "shapenet_part" in DATASET_SPECS
    assert "objaverse_xl" in DATASET_SPECS
    assert "ob3d_multiview" in DATASET_SPECS
    assert "retexture_3d" in DATASET_SPECS


def test_resnet_3d_neural_network_forward():
    model = HighPrecision3DReconstructionNet()
    model.eval()

    # Batch of 2 synthetic RGB images (2, 3, 64, 64)
    dummy_input = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        depth, normals, mask = model(dummy_input)

    assert depth.shape == (2, 1, 64, 64)
    assert normals.shape == (2, 3, 64, 64)
    assert mask.shape == (2, 1, 64, 64)

    # Check normal vectors are unit normalized
    norm_mag = torch.norm(normals, dim=1)
    assert torch.allclose(norm_mag, torch.ones_like(norm_mag), atol=1e-3)

    # Check depth and mask values are bounded in [0, 1]
    assert depth.min() >= 0.0 and depth.max() <= 1.0
    assert mask.min() >= 0.0 and mask.max() <= 1.0


def test_multi_term_losses():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pred_depth = torch.rand(2, 1, 32, 32, device=device)
    target_depth = torch.rand(2, 1, 32, 32, device=device)

    # Chamfer loss
    chamfer = ChamferLoss()
    l_chamfer = chamfer(pred_depth.squeeze(1), target_depth.squeeze(1))
    assert isinstance(l_chamfer, torch.Tensor)
    assert l_chamfer.item() >= 0.0

    # Laplacian edge loss
    laplacian = LaplacianEdgeLoss().to(device)
    l_lap = laplacian(pred_depth)
    assert isinstance(l_lap, torch.Tensor)
    assert l_lap.item() >= 0.0

    # Normal consistency loss
    normal_loss = NormalConsistencyLoss().to(device)
    l_norm = normal_loss(pred_depth)
    assert isinstance(l_norm, torch.Tensor)
    assert l_norm.item() >= 0.0


def test_kaggle_dataset_loader():
    dataset = Kaggle3DFoundationDataset(img_size=32, augment=False)
    assert len(dataset) > 0
    sample = dataset[0]

    assert "image" in sample
    assert "depth" in sample
    assert "normals" in sample
    assert "mask" in sample

    assert sample["image"].shape == (3, 32, 32)
    assert sample["depth"].shape == (1, 32, 32)
    assert sample["normals"].shape == (3, 32, 32)
    assert sample["mask"].shape == (1, 32, 32)


def test_api_dataset_and_training_endpoints():
    client = TestClient(app)

    # 1. Dataset Inventory endpoint
    res = client.get("/api/dataset/inventory")
    assert res.status_code == 200
    data = res.json()
    assert "total_3d_models" in data
    assert "processed_npz_count" in data
    assert data["total_3d_models"] > 0

    # 2. Training Status endpoint
    res = client.get("/api/train/status")
    assert res.status_code == 200
    state = res.json()
    assert "status" in state
    assert "metrics" in state

    # 3. Trigger Dataset Download endpoint
    res = client.post("/api/dataset/download-kaggle")
    assert res.status_code == 200
    assert res.json()["status"] == "started"
