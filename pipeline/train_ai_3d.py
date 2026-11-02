"""
High-Precision PyTorch 3D Reconstruction Neural Network & GPU Training Engine

Architecture:
- ResNet Residual Feature Pyramid Encoder (Captures sub-pixel edges, high-frequency contours, curvature)
- Convolutional 3D Spatial Bottleneck (Resolution-agnostic spatial volumetric representation)
- Multi-Head High-Resolution Decoder:
    1. Depth Map Head (Per-pixel continuous depth prediction in [0, 1])
    2. Surface Normal Head (3-channel unit normal vectors Nx, Ny, Nz)
    3. Foreground Salience Mask Head (Sub-pixel object boundary segmentation)

Training Features:
- Multi-view 360-degree orbital & elevation viewpoints (Yaw: 0-360°, Pitch: 0-30°)
- Automatic Mixed Precision (AMP FP16) training on CUDA for maximum precision and speed
- Multi-term Loss Objective: MSE + Bidirectional Chamfer + Laplacian + Normal Consistency
- Cosine Annealing Learning Rate Scheduler with smooth decay
- Automatic best-model checkpointing to data/models/best_3d_ai_model.pth
"""

import os
import sys
import time
import math
import random
import cv2
import trimesh
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pipeline.losses import ChamferLoss, LaplacianEdgeLoss, NormalConsistencyLoss, MaskedMSELoss, SSIMLoss

def mask_bce(predicted, target):
    # Sigmoid mask probabilities must use full precision outside autocast.
    with torch.amp.autocast('cuda', enabled=False):
        return F.binary_cross_entropy(predicted.float().clamp(1e-6, 1-1e-6), target.float())

DATASETS_DIR = Path("d:/3dSCANNER/data/datasets")
PROCESSED_DIR = DATASETS_DIR / "processed"
MODELS_DIR = Path("d:/3dSCANNER/data/models")

# Global Training State Tracker for API / Dashboard Polling
TRAINING_STATE: Dict[str, Any] = {
    "status": "idle",           # idle, training, completed, failed
    "current_epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "best_val_loss": None,
    "best_epoch": 0,
    "current_lr": 0.0,
    "metrics": {
        "mse": 0.0,
        "chamfer": 0.0,
        "laplacian": 0.0,
        "normal": 0.0
    },
    "history": [],
    "logs": [],
    "model_path": None,
    "error": None
}


def get_dataset_inventory() -> List[Dict[str, Any]]:
    """
    Lists all 3D datasets with topology, vertex/face metrics, watertightness, and size.
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    inventory = []

    supported_exts = ('.stl', '.obj', '.ply', '.glb', '.gltf', '.off')
    files = sorted([f for f in DATASETS_DIR.glob("*") if f.suffix.lower() in supported_exts])

    for f in files:
        try:
            mesh = trimesh.load(str(f), force='mesh')
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
            
            extents_mm = [round(float(x) * 100, 2) for x in mesh.extents] if len(mesh.vertices) > 0 else [0, 0, 0]
            inventory.append({
                "filename": f.name,
                "filepath": str(f),
                "size_kb": round(f.stat().st_size / 1024.0, 2),
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "is_watertight": bool(mesh.is_watertight),
                "extents_mm": extents_mm,
                "format": f.suffix.upper().replace('.', '')
            })
        except Exception as e:
            inventory.append({
                "filename": f.name,
                "filepath": str(f),
                "size_kb": round(f.stat().st_size / 1024.0, 2),
                "error": str(e)
            })

    return inventory


# =========================================================================
#  ResNet Residual 3D Neural Network Architecture
# =========================================================================

class ResidualBlock2D(nn.Module):
    """Residual Convolutional Block with Skip Connection & Batch Normalization."""
    def __init__(self, channels: int):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.act2(out)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention for adaptive feature weighting."""
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class HighPrecision3DReconstructionNet(nn.Module):
    """
    High-Precision ResNet Residual Feature Pyramid 3D Reconstruction Network.
    Fully resolution-agnostic (supports 64x64, 128x128, 256x256+ input sizes).
    Outputs:
      1. Continuous Depth Map (1, H, W) in [0, 1]
      2. Unit Surface Normal Map (3, H, W) with ||N|| = 1
      3. Foreground Confidence Mask (1, H, W) in [0, 1]
    """
    def __init__(self):
        super(HighPrecision3DReconstructionNet, self).__init__()

        # Stage 1: Initial feature extraction
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock2D(32)
        )

        # Stage 2: Downsample /2
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock2D(64)
        )

        # Stage 3: Downsample /4
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock2D(128)
        )

        # Stage 4: Downsample /8
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock2D(256)
        )

        # 3D Spatial Volumetric Latent Bottleneck (Convolutional Residual Stack + Attention)
        self.bottleneck = nn.Sequential(
            ResidualBlock2D(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock2D(256),
            SEBlock(256)
        )

        # Decoder Stage 1: Upsample *2 (with skip from down2)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec_block1 = ResidualBlock2D(128 + 128)

        # Decoder Stage 2: Upsample *4 (with skip from down1)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec_block2 = ResidualBlock2D(64 + 64)

        # Decoder Stage 3: Upsample *8 (with skip from in_conv)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec_block3 = ResidualBlock2D(32 + 32)

        # Multi-Head Prediction Layers
        # Head 1: Continuous 3D Depth Map
        self.depth_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Head 2: Surface Normal Map (Nx, Ny, Nz)
        self.normal_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # Head 3: Foreground Confidence Mask
        self.mask_head = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        # Encoder with skip connections
        f1 = self.in_conv(x)       # (B, 32, H, W)
        f2 = self.down1(f1)        # (B, 64, H/2, W/2)
        f3 = self.down2(f2)        # (B, 128, H/4, W/4)
        f4 = self.down3(f3)        # (B, 256, H/8, W/8)

        # Spatial Latent Volume Bottleneck
        latent = self.bottleneck(f4)

        # Decoder with U-Net Skip Concatenation
        u1 = self.up1(latent)                  # (B, 128, H/4, W/4)
        cat1 = torch.cat([u1, f3], dim=1)      # (B, 256, H/4, W/4)
        d1 = self.dec_block1(cat1)

        u2 = self.up2(d1)                      # (B, 64, H/2, W/2)
        cat2 = torch.cat([u2, f2], dim=1)      # (B, 128, H/2, W/2)
        d2 = self.dec_block2(cat2)

        u3 = self.up3(d2)                      # (B, 32, H, W)
        cat3 = torch.cat([u3, f1], dim=1)      # (B, 64, H, W)
        d3 = self.dec_block3(cat3)             # (B, 64, H, W)

        # Multi-Head Outputs
        pred_depth = self.depth_head(d3)
        raw_normals = self.normal_head(d3)
        pred_mask = self.mask_head(d3)

        # Normalize normal vectors to unit magnitude
        norm_mag = torch.norm(raw_normals, dim=1, keepdim=True) + 1e-8
        pred_normals = raw_normals / norm_mag

        return pred_depth, pred_normals, pred_mask


# Backward-compatible alias
Neural3DReconstructionNet = HighPrecision3DReconstructionNet


# =========================================================================
#  Dataset Loader with Multi-View & Point Cloud Ingestion
# =========================================================================

class Kaggle3DFoundationDataset(Dataset):
    """
    High-Precision Dataset Loader for Kaggle, ShapeNetPart, Objaverse, and CAD models.
    Loads processed .npz point clouds dynamically and renders multi-view training pairs on the fly.
    """
    def __init__(self, img_size: int = 128, augment: bool = True):
        self.img_size = img_size
        self.augment = augment
        self.file_paths = []
        self._load_all_datasets()

    def _load_all_datasets(self):
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        npz_files = sorted(list(PROCESSED_DIR.glob("*.npz")))

        if len(npz_files) == 0:
            try:
                from pipeline.gcs_utils import list_gcs_files, stream_npz_from_gcs, GCS_CACHE_DIR, enforce_cache_limit
                gcs_npz_blobs = [b for b in list_gcs_files(prefix="processed/") if b.endswith(".npz")]
                if gcs_npz_blobs:
                    print(f"[GCS] Found {len(gcs_npz_blobs)} .npz files on Google Cloud Storage. Streaming to local cache...")
                    enforce_cache_limit(max_gb=50.0)
                    for blob_name in gcs_npz_blobs:
                        stream_npz_from_gcs(blob_name, cache_locally=True)
                    npz_files = sorted(list(PROCESSED_DIR.glob("*.npz")))
            except ImportError:
                pass
            except Exception as e:
                print(f"[GCS Notice] Could not load from cloud: {e}. Falling back to local files.")

        self.file_paths = []
        exclusion_file=DATASETS_DIR/'geometry_exclusions.json'
        if exclusion_file.exists():
            import json
            excluded=set(json.loads(exclusion_file.read_text(encoding='utf-8')).get('npz_files',[]))
            npz_files=[f for f in npz_files if f.name not in excluded]
        if npz_files:
            print(f"Discovered {len(npz_files)} pre-processed .NPZ dataset targets from {PROCESSED_DIR} at {self.img_size}x{self.img_size} resolution.")
            self.file_paths.extend(npz_files)
        
        print(f"Searching {DATASETS_DIR} for raw 3D models...")
        raw_files = []
        for ext in ["*.stl", "*.obj", "*.glb", "*.ply", "*.gltf"]:
            raw_files.extend(list(DATASETS_DIR.rglob(ext)))
            
        if raw_files:
            print(f"Discovered {len(raw_files)} raw 3D dataset files.")
            if not self.file_paths:
                print("No pre-processed NPZ files found. Falling back to rendering multi-view on the fly (WARNING: This will be very slow and CPU-bound).")
                self.file_paths.extend(raw_files)
            else:
                print("Skipping on-the-fly rendering of raw models because pre-processed NPZ files exist. This ensures maximum GPU training speed.")
            
        if not self.file_paths:
            print("No datasets found.")

    def __len__(self):
        return max(len(self.file_paths) * 16, 1)  # 16 virtual views per file

    def __getitem__(self, idx):
        if not self.file_paths:
            raise ValueError('No geometry dataset available')
            
        file_idx = (idx // 16) % len(self.file_paths)
        file_path = self.file_paths[file_idx]
        
        points, normals = None, None
        
        try:
            if file_path.suffix == '.npz':
                with np.load(str(file_path)) as data:
                    points = data['points'] if 'points' in data else None
                    normals = data['normals'] if 'normals' in data else None
            elif file_path.suffix.lower() in ['.stl', '.obj', '.glb', '.ply', '.gltf']:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mesh = trimesh.load(str(file_path), force='mesh')
                    if isinstance(mesh, trimesh.Scene):
                        mesh = trimesh.util.concatenate(mesh.dump())
                    
                    if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                        pts, face_idx = trimesh.sample.sample_surface(mesh, 20000)
                        points = pts
                        normals = mesh.face_normals[face_idx]
                        
                        # Verify no NaN or Inf values in the corrupt dataset files
                        if not np.isfinite(points).all() or not np.isfinite(normals).all():
                            points = None
        except Exception:
            pass

        if points is None or len(points) == 0:
            # Return dummy on failure
            raise ValueError(f'Invalid geometry dataset: {file_path}')

        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]
        if normals is not None:
            normals = normals[valid_mask]

        if len(points) == 0:
            raise ValueError(f'Invalid geometry dataset: {file_path}')

        norm_pts = points - points.mean(axis=0)
        max_ext = np.max(np.abs(norm_pts))
        if max_ext > 0:
            norm_pts /= (max_ext * 2.0)

        # Randomize view
        pitch_deg = random.choice([0, 10, 20, 30, 45, -15])
        angle = random.randint(0, 359)
        
        pitch_rad = np.radians(pitch_deg)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        rot_pitch = np.array([
            [1, 0,      0],
            [0, cos_p, -sin_p],
            [0, sin_p,  cos_p]
        ], dtype=np.float32)

        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot_yaw = np.array([
            [cos_a,  0, sin_a],
            [0,      1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=np.float32)

        rot_matrix = rot_pitch @ rot_yaw
        rot_pts = norm_pts @ rot_matrix.T
        rot_norms = normals @ rot_matrix.T if normals is not None else None

        img_2d = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        depth_3d = np.zeros((self.img_size, self.img_size, 1), dtype=np.float32)
        normal_map = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        mask_map = np.zeros((self.img_size, self.img_size, 1), dtype=np.float32)

        light_dir = np.array([0.35, 0.55, 0.75], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)

        pts_sub = rot_pts[:40000]
        px = np.clip(((pts_sub[:, 0] + 0.5) * (self.img_size - 1)).astype(np.int32), 0, self.img_size - 1)
        py = np.clip(((0.5 - pts_sub[:, 1]) * (self.img_size - 1)).astype(np.int32), 0, self.img_size - 1)
        z_val = (pts_sub[:, 2] + 0.5).astype(np.float32)

        if rot_norms is not None:
            norms_sub = rot_norms[:40000]
            dots = np.clip(norms_sub @ light_dir, 0.18, 1.0)
        else:
            norms_sub = None
            dots = np.full(len(pts_sub), 0.8, dtype=np.float32)

        order = np.argsort(pts_sub[:, 2])
        s_px, s_py, s_z, s_dots = px[order], py[order], z_val[order], dots[order]

        img_2d[s_py, s_px, 0] = s_dots * 0.85 + 0.10
        img_2d[s_py, s_px, 1] = s_dots * 0.78 + 0.12
        img_2d[s_py, s_px, 2] = s_dots * 0.92 + 0.05
        depth_3d[s_py, s_px, 0] = s_z
        
        if norms_sub is not None:
            normal_map[s_py, s_px] = norms_sub[order]
        else:
            normal_map[s_py, s_px, 2] = 1.0
            
        mask_map[s_py, s_px, 0] = 1.0

        sample = {
            "image": torch.tensor(img_2d).permute(2, 0, 1),
            "depth": torch.tensor(depth_3d).permute(2, 0, 1),
            "normals": torch.tensor(normal_map).permute(2, 0, 1),
            "mask": torch.tensor(mask_map).permute(2, 0, 1)
        }

        if self.augment:
            # Color jitter (brightness/contrast)
            if random.random() > 0.5:
                scale = 0.85 + random.random() * 0.3
                sample["image"] = torch.clamp(sample["image"] * scale, 0.0, 1.0)
            # Gaussian noise for robustness
            if random.random() > 0.7:
                noise = torch.randn_like(sample["image"]) * 0.02
                sample["image"] = torch.clamp(sample["image"] + noise, 0.0, 1.0)
            # Random horizontal flip (must also flip depth, normals, mask)
            if random.random() > 0.5:
                sample["image"] = torch.flip(sample["image"], [-1])
                sample["depth"] = torch.flip(sample["depth"], [-1])
                sample["normals"] = torch.flip(sample["normals"], [-1])
                sample["normals"][0] *= -1  # flip Nx component
                sample["mask"] = torch.flip(sample["mask"], [-1])

        return sample

# Backward-compatible wrapper
class Mesh3DPairDataset(Dataset):
    def __init__(self, dataset_files: List[Path], img_size: int = 256):
        self.inner = Kaggle3DFoundationDataset(img_size=img_size, augment=False)
    def __len__(self):
        return len(self.inner)
    def __getitem__(self, idx):
        item = self.inner[idx]
        return item["image"], item["depth"]


# =========================================================================
#  Neural Network Training Engine
# =========================================================================

def train_custom_3d_ai_model(
    epochs: int = 300,
    batch_size: int = 8,
    lr: float = 3e-4,
    img_size: int = 256,
    device_name: Optional[str] = None,
    resume: bool = True,
    patience: int = 50,
    warmup_epochs: int = 5
) -> str:
    """
    Trains the High-Precision ResNet Residual 3D Reconstruction Network on all datasets.
    
    Features:
    - Automatic Mixed Precision (AMP FP16) on CUDA
    - Multi-term Loss: MSE + Chamfer + Laplacian + Normal Consistency
    - Cosine Annealing LR Scheduler with linear warmup for first N epochs
    - ReduceLROnPlateau backup scheduler (halves LR if val_loss stagnates)
    - Gradient norm clipping to 1.0 for training stability
    - Early stopping with configurable patience (default 30 epochs)
    - Automatic checkpoint resume from best model if available
    - Train / Validation Split (85% / 15%) with best-model checkpointing
    - Global state tracking for real-time Web Dashboard & API progress reporting
    
    Args:
        epochs: Total training epochs (default: 150)
        batch_size: Training batch size (default: 16)
        lr: Peak learning rate after warmup (default: 5e-4)
        img_size: Training image resolution (default: 128)
        device_name: Force specific device (e.g. 'cuda:0', 'cpu')
        resume: Resume from existing best checkpoint if available (default: True)
        patience: Early stopping patience in epochs (default: 30)
        warmup_epochs: Number of epochs for linear LR warmup (default: 5)
    """
    global TRAINING_STATE

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_model_path = MODELS_DIR / "photo_depth_v2_latest.pth"
    best_model_path = MODELS_DIR / "photo_depth_v2_best.pth"

    # Reset state
    TRAINING_STATE["status"] = "training"
    TRAINING_STATE["current_epoch"] = 0
    TRAINING_STATE["total_epochs"] = epochs
    TRAINING_STATE["best_val_loss"] = None
    TRAINING_STATE["history"] = []
    TRAINING_STATE["logs"] = []
    TRAINING_STATE["error"] = None

    def log_msg(msg: str):
        print(msg, flush=True)
        TRAINING_STATE["logs"].append(msg)

    log_msg("=" * 68)
    log_msg("  PyTorch High-Precision 3D Reconstruction Neural Network Training")
    log_msg("=" * 68)

    from pipeline.training_data import PhotoDepthDataset,CONTRACT
    manifest=DATASETS_DIR/'training_manifest.json'
    try:
        train_dataset=PhotoDepthDataset(manifest,'train',img_size=img_size,augment=True)
        val_dataset=PhotoDepthDataset(manifest,'val',img_size=img_size,augment=False)
    except ValueError as error:
        TRAINING_STATE.update(status='failed',error=str(error))
        raise
    total_samples=len(train_dataset)+len(val_dataset)
    train_size,val_size=len(train_dataset),len(val_dataset)
    data_metadata={'contract':CONTRACT,'object_disjoint':True,'train_objects':sorted({s['object_id'] for s in train_dataset.records}),
                   'validation_objects':sorted({s['object_id'] for s in val_dataset.records})}
    log_msg(f'Using {train_size} aligned training pairs and {val_size} held-out pairs; no shared objects.')

    import multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    num_workers = 0  # Use single-process DataLoader on Windows to avoid shared memory issues
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    # Duplicate DataLoader definitions removed (using the Windows‑safe configuration above)
    log_msg(f"Dataset split: {train_size} training samples / {val_size} validation samples (Workers: {num_workers})")

    # Select Device
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg(f"Training hardware device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    log_msg(f"Early stopping patience: {patience} epochs | LR warmup: {warmup_epochs} epochs")

    # Model & Criteria
    model = HighPrecision3DReconstructionNet().to(device)
    mse_criterion = MaskedMSELoss().to(device)  # Foreground-masked MSE
    ssim_criterion = SSIMLoss().to(device)      # Structural similarity
    chamfer_criterion = ChamferLoss()
    laplacian_criterion = LaplacianEdgeLoss().to(device)
    normal_criterion = NormalConsistencyLoss().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Primary: OneCycleLR converges faster and prevents getting stuck in early local minima
    total_steps = epochs * len(train_loader)
    onecycle_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos'
    )
    # Backup: Reduce LR on plateau if validation loss stagnates for 10 epochs
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=lr * 0.001
    )

    # Resume from existing checkpoint if available
    start_epoch = 1
    if resume and best_model_path.exists():
        try:
            ckpt = torch.load(str(best_model_path), map_location=device, weights_only=True)
            assert ckpt.get('data_contract',{}).get('contract') == CONTRACT, 'Cannot resume an incompatible legacy checkpoint'
            state_dict = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state_dict)
            prev_epoch = ckpt.get('epoch', 0)
            prev_val_loss = ckpt.get('val_loss', float('inf'))
            start_epoch = prev_epoch + 1
            best_val_loss = prev_val_loss
            best_epoch = prev_epoch
            log_msg(f"Resumed from checkpoint: epoch {prev_epoch}, val_loss={prev_val_loss:.5f}")
            log_msg(f"Continuing training from epoch {start_epoch} to {epochs}...")
        except Exception as e:
            log_msg(f"Could not resume from checkpoint: {e}. Starting fresh.")
            start_epoch = 1

    # Mixed Precision Scaler
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Loss Weights — Balanced to prevent any single term from dominating training
    W_MSE = 1.0
    W_SSIM = 0.5
    W_CHAMFER = 1.0
    W_LAPLACIAN = 0.5
    W_NORMAL = 1.0

    start_time = time.time()
    avg_train = 0.0
    avg_val = 0.0
    if start_epoch == 1:
        best_val_loss = float('inf')
        best_epoch = 0
    epochs_without_improvement = 0

    log_msg("-" * 68)
    remaining = epochs - start_epoch + 1
    log_msg(f"Training epochs {start_epoch}-{epochs} ({remaining} remaining) | Batch Size: {batch_size} | Peak LR: {lr} | AMP: {use_amp}")
    log_msg(f"Loss Terms: {W_MSE}*MSE + {W_CHAMFER}*Chamfer + {W_LAPLACIAN}*Laplacian + {W_NORMAL}*Normal")
    log_msg("-" * 68)

    try:
        for epoch in range(start_epoch, epochs + 1):
            TRAINING_STATE["current_epoch"] = epoch
            
            # --- TRAINING PHASE ---
            model.train()
            train_loss_total = 0.0
            train_loss_mse = 0.0
            train_loss_chamfer = 0.0
            train_loss_lap = 0.0
            train_loss_norm = 0.0
            num_train_batches = 0

            chamfer_w = W_CHAMFER * min(1.0, epoch / 5.0)

            # OneCycleLR handles warmup natively, so we skip the manual loop
            # if epoch <= warmup_epochs: ...

            from tqdm import tqdm
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
            for batch in pbar:
                imgs = batch["image"].to(device, non_blocking=True)
                target_depth = batch["depth"].to(device, non_blocking=True)
                target_normals = batch["normals"].to(device, non_blocking=True)
                target_mask = batch["mask"].to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred_depth, pred_normals, pred_mask = model(imgs)

                    l_mse = mse_criterion(pred_depth, target_depth)
                    l_ssim = ssim_criterion(pred_depth, target_depth)
                    l_chamfer = chamfer_criterion(pred_depth.squeeze(1), target_depth.squeeze(1))
                    l_lap = laplacian_criterion(pred_depth)
                    l_norm = normal_criterion(pred_depth) + 0.5 * mse_criterion(pred_normals, target_normals)

                    loss = (W_MSE * l_mse 
                            + W_SSIM * l_ssim
                            + chamfer_w * l_chamfer 
                            + W_LAPLACIAN * l_lap 
                            + W_NORMAL * l_norm + 0.25 * mask_bce(pred_mask, target_mask))

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss_total += loss.item()
                train_loss_mse += l_mse.item()
                train_loss_chamfer += l_chamfer.item()
                train_loss_lap += l_lap.item()
                train_loss_norm += l_norm.item()
                num_train_batches += 1
                
                # Update progress bar
                pbar.set_postfix({"Loss": f"{(train_loss_total / num_train_batches):.4f}"})
                onecycle_scheduler.step()

            # --- VALIDATION PHASE ---
            model.eval()
            val_loss_total = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(device)
                    target_depth = batch["depth"].to(device)
                    target_normals = batch["normals"].to(device)
                    target_mask = batch["mask"].to(device)

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        pred_depth, pred_normals, pred_mask = model(imgs)

                        l_mse = mse_criterion(pred_depth, target_depth)
                        l_ssim = ssim_criterion(pred_depth, target_depth)
                        l_chamfer = chamfer_criterion(pred_depth.squeeze(1), target_depth.squeeze(1))
                        l_lap = laplacian_criterion(pred_depth)
                        l_norm = normal_criterion(pred_depth) + 0.5 * mse_criterion(pred_normals, target_normals)

                        loss = (W_MSE * l_mse 
                                + W_SSIM * l_ssim
                                + chamfer_w * l_chamfer 
                                + W_LAPLACIAN * l_lap 
                                + W_NORMAL * l_norm + 0.25 * mask_bce(pred_mask, target_mask))

                    val_loss_total += loss.item()
                    num_val_batches += 1

            # Epoch Averages
            avg_train = train_loss_total / max(num_train_batches, 1)
            avg_val = val_loss_total / max(num_val_batches, 1)
            avg_mse = train_loss_mse / max(num_train_batches, 1)
            avg_chamfer = train_loss_chamfer / max(num_train_batches, 1)
            avg_lap = train_loss_lap / max(num_train_batches, 1)
            avg_norm = train_loss_norm / max(num_train_batches, 1)
            current_lr = optimizer.param_groups[0]['lr']

            # Step schedulers (warmup takes priority during warmup phase)
            if epoch > warmup_epochs:
                plateau_scheduler.step(avg_val)

            # Checkpoint Best Model
            is_best = avg_val < best_val_loss
            if is_best:
                best_val_loss = avg_val
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'data_contract': data_metadata,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val,
                    'train_loss': avg_train,
                    'samples_count': total_samples,
                    'img_size': img_size,
                    'history': list(TRAINING_STATE["history"]),
                    'metrics': dict(TRAINING_STATE["metrics"])
                }, str(best_model_path))
            else:
                epochs_without_improvement += 1

            # Update State Tracker
            TRAINING_STATE["train_loss"] = round(avg_train, 6)
            TRAINING_STATE["val_loss"] = round(avg_val, 6)
            TRAINING_STATE["best_val_loss"] = round(best_val_loss, 6)
            TRAINING_STATE["best_epoch"] = best_epoch
            TRAINING_STATE["current_lr"] = round(current_lr, 6)
            TRAINING_STATE["metrics"] = {
                "mse": round(avg_mse, 6),
                "chamfer": round(avg_chamfer, 6),
                "laplacian": round(avg_lap, 6),
                "normal": round(avg_norm, 6)
            }
            TRAINING_STATE["history"].append({
                "epoch": epoch,
                "train_loss": round(avg_train, 6),
                "val_loss": round(avg_val, 6),
                "lr": round(current_lr, 6),
                "mse": round(avg_mse, 6),
                "chamfer": round(avg_chamfer, 6)
            })

            best_tag = " [BEST]" if is_best else ""
            stall_msg = f" (no improvement for {epochs_without_improvement} epochs)" if epochs_without_improvement > 0 else ""
            log_msg(f"Epoch [{epoch:03d}/{epochs:03d}] "
                    f"Train: {avg_train:.5f} | Val: {avg_val:.5f} | "
                    f"LR: {current_lr:.6f}{best_tag}{stall_msg}")
            log_msg(f"  |-- MSE: {avg_mse:.5f}  Chamfer: {avg_chamfer:.5f}  "
                    f"Laplacian: {avg_lap:.5f}  Normal: {avg_norm:.5f}")

            # Early stopping check
            if epochs_without_improvement >= patience:
                log_msg(f"\nEarly stopping triggered after {patience} epochs without improvement.")
                log_msg(f"Best model was at epoch {best_epoch} with val_loss={best_val_loss:.5f}")
                break

        elapsed = round(time.time() - start_time, 2)
        log_msg("-" * 68)
        log_msg(f"Training successfully completed in {elapsed}s | Best Epoch: {best_epoch} (val_loss: {best_val_loss:.5f})")
        log_msg("-" * 68)

        # Save Final Checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
                    'data_contract': data_metadata,
            'epochs': epochs,
            'final_train_loss': avg_train,
            'final_val_loss': avg_val,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'samples_count': total_samples,
            'img_size': img_size
        }, str(out_model_path))

        log_msg(f"Trained AI model weights saved to: {out_model_path}")
        log_msg(f"Best AI checkpoint saved to:       {best_model_path}")

        TRAINING_STATE["status"] = "completed"
        TRAINING_STATE["model_path"] = str(best_model_path)
        return str(best_model_path)

    except Exception as e:
        TRAINING_STATE["status"] = "failed"
        TRAINING_STATE["error"] = str(e)
        log_msg(f"Training failed: {e}")
        raise e


def predict_custom_neural_depth_and_normals(
    img_bgr: np.ndarray,
    model_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs high-precision inference using the trained PyTorch 3D AI model.
    Returns:
      (depth_map [H, W], normal_map [H, W, 3], mask [H, W])
    """
    from pipeline.training_data import checkpoint_status
    status=checkpoint_status(model_path)
    if not status['ready']:
        raise RuntimeError(status['reason'])
    model_path=status['path']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HighPrecision3DReconstructionNet().to(device)

    # Read resolution from checkpoint (supports both old 128 and new 256 models)
    img_size = 128  # default fallback for older checkpoints
    if Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        img_size = ckpt.get('img_size', 128)

    model.eval()

    h_orig, w_orig = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to model's training resolution for neural inference
    img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA) / 255.0
    tensor_in = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        depth, normals, mask = model(tensor_in)

    depth_np = depth.squeeze().cpu().numpy()
    normals_np = normals.squeeze().permute(1, 2, 0).cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()

    # Resize back to original image dimensions
    depth_out = cv2.resize(depth_np, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    normals_out = cv2.resize(normals_np, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    mask_out = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    # Normalize depth to [0, 1]
    depth_out = (depth_out - depth_out.min()) / (depth_out.max() - depth_out.min() + 1e-8)

    # Renormalize normals
    norm_mag = np.linalg.norm(normals_out, axis=2, keepdims=True) + 1e-8
    normals_out /= norm_mag

    return depth_out, normals_out, mask_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train High-Precision PyTorch 3D AI Neural Network on Datasets")
    parser.add_argument("--epochs", type=int, default=500, help="Total training epochs (default: 500)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate after warmup (default: 0.0003)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--img_size", type=int, default=256, help="Training image resolution (default: 256)")
    parser.add_argument("--no_resume", action="store_true", help="Start training from scratch (don't resume checkpoint)")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience in epochs (default: 30)")
    parser.add_argument("--warmup", type=int, default=5, help="LR warmup epochs (default: 5)")
    parser.add_argument("--inventory_only", action="store_true", help="Print dataset inventory and exit")
    args = parser.parse_args()

    if args.inventory_only:
        print("\n--- DATASET INVENTORY ---")
        inv = get_dataset_inventory()
        print(f"Total dataset 3D models: {len(inv)}")
        for item in inv[:12]:
            print(f"  * {item.get('filename')} ({item.get('faces', 0)} faces, {item.get('size_kb', 0)} KB, watertight={item.get('is_watertight')})")
        if len(inv) > 12:
            print(f"  ... and {len(inv) - 12} more dataset models.")
        print("--------------------------\n")
    else:
        train_custom_3d_ai_model(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            img_size=args.img_size,
            resume=not args.no_resume,
            patience=args.patience,
            warmup_epochs=args.warmup
        )
