# 3DINVENZA — Phone 3D Scanner AI

A production-grade, end-to-end 3D reconstruction platform that converts smartphone photos into watertight, textured, dimensionally-accurate 3D models. The system supports **single-image instant 3D**, **multi-view turntable scanning**, **Intel RealSense D455 depth capture**, and **custom neural network training** — all served through a FastAPI backend and an interactive Three.js WebGL dashboard.

---

## ✨ Key Capabilities

| Capability | Description |
| :--- | :--- |
| **Single-Image → 3D** | Generate full 360° meshes from one photo via TripoSR, TRELLIS, Hunyuan3D 2.0, or Depth-Anything-V2 |
| **Multi-View Turntable** | Fuse 24–72 calibrated turntable frames into metric-accurate models |
| **TSDF Volumetric Scan** | Real-time depth-camera integration with stepper-motor turntable via TSDF voxel grids |
| **D455 RealSense Capture** | Live 3D point-cloud streaming and capture with Intel RealSense D455 |
| **AI Neural Training** | Train custom ResNet-based depth/normal/mask prediction networks on your own datasets |
| **Multi-View Diffusion** | Hallucinate unseen geometry (back, sides, top) from a single photo before reconstruction |
| **Physical Metrology** | OBB auto-measurement, ArUco scale references, and scan quality scoring |
| **Interactive Dashboard** | Drag-and-drop upload, real-time pipeline tracking, WebGL 3D viewer, instant STL/GLB/PLY export |

---

## 🏗️ Architecture & Project Structure

```text
3dSCANNER/
├── api/
│   ├── main.py                          # FastAPI server & static file host
│   └── routes.py                        # All REST API endpoints (scan, train, measure, calibrate…)
│
├── pipeline/
│   ├── preprocess.py                    # Background removal (rembg + OpenCV fallback)
│   ├── colmap_poses.py                  # SfM pose estimation (pycolmap + OpenCV SIFT fallback)
│   ├── open3d_mesh.py                   # Poisson surface meshing & density pruning
│   ├── ai_refine.py                     # PyTorch Laplacian smoothing & watertight repair
│   ├── single_image_3d.py              # Single-photo → 3D via Depth-Anything-V2 + Poisson
│   ├── ai_depth_3d.py                  # Monocular AI depth estimation & normal computation
│   ├── triposr_inference.py            # TripoSR (Stability AI) LRM inference
│   ├── trellis_inference.py            # TRELLIS (Microsoft) SLAT inference
│   ├── hunyuan3d_inference.py          # Hunyuan3D 2.0 (Tencent) shape + paint synthesis
│   ├── multiview_diffusion.py          # 360° novel-view hallucination from single photo
│   ├── multiview_fusion.py             # Multi-view point-cloud fusion & meshing
│   ├── turntable_calibrated_pipeline.py # Calibrated turntable scan (known-angle reconstruction)
│   ├── tsdf_turntable_scanner.py       # TSDF volumetric integration with stepper-motor poses
│   ├── camera_calibration.py           # Checkerboard camera calibration & undistortion
│   ├── scale_reference.py             # ArUco marker & turntable baseline metrology
│   ├── canonical_alignment.py         # Ground-plane alignment & upright orientation
│   ├── dimensional_eval.py            # OBB physical dimension measurement (L×W×H mm)
│   ├── scan_confidence.py             # Scan quality scoring & diagnostics (0–100%)
│   ├── texture_projector.py           # RGB → vertex color projection & GLB export
│   ├── mesh_topology.py               # Topology analysis & non-manifold repair
│   ├── depth_normal_refine.py         # Depth-map & surface-normal refinement
│   ├── losses.py                       # Chamfer, Laplacian edge, & normal consistency losses
│   ├── train_ai_3d.py                  # ResNet Feature Pyramid neural training engine
│   ├── download_ai_models.py          # MiDaS / Depth model weight downloader
│   ├── download_dataset.py            # Sample dataset downloader
│   ├── download_kaggle_datasets.py    # Kaggle 3D dataset ingestion pipeline
│   ├── download_sculpture_datasets.py # Sculpture/statue dataset scraper
│   ├── generate_complex_datasets.py   # Procedural parametric mesh generator (20+ shapes)
│   ├── 01_preprocess.py … 05_*.py     # CLI wrapper scripts for each pipeline stage
│   └── __pycache__/
│
├── static/
│   ├── index.html                      # 3D Dashboard web interface
│   ├── style.css                       # Dark-mode glassmorphism styling
│   └── app.js                          # Three.js WebGL viewer & upload controller
│
├── tsr/                                # TripoSR model runtime (system, utils, bake_texture)
├── triposr_repo/                       # TripoSR upstream repository (Stability AI)
│
├── tests/
│   ├── test_pipeline.py                # Core pipeline unit tests
│   ├── test_ai_pipeline.py             # AI depth & single-image tests
│   ├── test_high_detail.py             # High-detail reconstruction tests
│   ├── test_kaggle_and_training.py     # Dataset ingestion & training tests
│   ├── test_obb_and_diffusion.py       # OBB measurement & diffusion tests
│   └── test_turntable_and_calibration.py # Turntable & camera calibration tests
│
├── data/
│   ├── raw/                            # Uploaded raw turntable images
│   ├── datasets/                       # Training datasets (procedural + downloaded)
│   ├── models/                         # Trained model checkpoints (.pth)
│   ├── output/                         # Final 3D models (.stl, .glb, .ply)
│   └── jobs/                           # Per-job processing directories
│
├── orbiter/                            # Orbiter 2-axis turntable subsystem (see below)
├── D455_SCANNER.PY                     # Intel RealSense D455 live 3D capture script
├── CHECK_USB.PY                        # RealSense USB 3.0 connection validator
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. System Requirements

| Requirement | Details |
| :--- | :--- |
| **Python** | 3.9, 3.10, or 3.11 |
| **GPU (recommended)** | NVIDIA GPU with CUDA (RTX 3060+ / 8 GB+ VRAM for neural engines) |
| **COLMAP (optional)** | For traditional SfM. Auto-falls back to OpenCV SIFT if unavailable |
| **Intel RealSense SDK** | Required only for D455 depth-camera capture (`pyrealsense2`) |

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Optional — for advanced AI engines:**

```bash
# TripoSR (bundled in tsr/ — works out of the box)

# TRELLIS (Microsoft)
pip install trellis

# Hunyuan3D 2.0 (Tencent)
pip install hy3dgen

# Intel RealSense D455
pip install pyrealsense2
```

### 3. Download AI Model Weights

```bash
python pipeline/download_ai_models.py
```

This downloads MiDaS depth-estimation weights and generates sample 3D-printing calibration datasets.

### 4. Start the Server

```bash
python api/main.py
```

Or with hot-reload:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

---

## 🕹️ Web Dashboard

The interactive dashboard provides a complete scan-to-model workflow:

- **Drag & Drop Upload** — Single photo or multi-frame turntable sequence (JPG, PNG, ZIP)
- **AI Engine Selector** — Choose between TripoSR, TRELLIS, Hunyuan3D, or AI Depth
- **Real-Time Pipeline Tracker** — Live progress indicators for each reconstruction stage
- **WebGL 3D Viewport** — Inspect models with OrbitControls, solid/wireframe modes, dynamic lighting
- **Physical Measurements** — Auto-displayed OBB dimensions (L × W × H mm)
- **Instant Export** — Download as `.stl`, `.glb`, `.ply`, or `.obj` with one click

---

## 🤖 Single-Image 3D Engines

The platform includes four swappable AI engines for single-photo 3D generation, with automatic fallback chains:

| Engine | Provider | Technique | Speed | Quality |
| :--- | :--- | :--- | :--- | :--- |
| **TripoSR** | Stability AI / Tripo | Tri-plane LRM + SDF | ~1 s | ★★★★☆ |
| **TRELLIS** | Microsoft | Structured Latent (SLAT) | ~3 s | ★★★★★ |
| **Hunyuan3D 2.0** | Tencent | DiT Flow Matching + Paint | ~5 s | ★★★★★ |
| **AI Depth** | Depth-Anything-V2 | Monocular 2.5D depth | <1 s | ★★★☆☆ |

Engine selection is available via the dashboard dropdown or the `engine` parameter in the API:

```bash
curl -X POST http://localhost:8000/api/scan/single-image \
  -F "file=@photo.jpg" \
  -F "engine=trellis"
```

If the selected engine is unavailable (missing weights or no CUDA), the system automatically walks the fallback chain until a working engine is found.

---

## ⚡ CLI Pipeline Usage

Run any pipeline stage directly from the terminal:

```bash
# Background removal
python pipeline/01_preprocess.py data/raw data/processed

# COLMAP sparse reconstruction & pose estimation
python pipeline/02_colmap_poses.py data/processed data/workspace

# Poisson surface meshing
python pipeline/03_open3d_mesh.py data/workspace/sparse_cloud.ply data/output/model.stl

# PyTorch Laplacian smoothing & watertight repair
python pipeline/04_ai_refine.py data/output/model.stl data/output/refined_model.stl

# OBB physical dimensional evaluation
python pipeline/05_dimensional_eval.py data/output/refined_model.stl
```

---

## 📷 Intel RealSense D455 Capture

For real-time depth-camera 3D scanning with an Intel RealSense D455:

```bash
# 1. Verify USB 3.0 connection
python CHECK_USB.PY

# 2. Launch live 3D point-cloud viewer
python D455_SCANNER.PY
# Press Ctrl+C to stop — saves capture.ply automatically
```

> **Note:** The D455 scanner requires a USB 3.0 port and the `pyrealsense2` package. The script streams aligned RGB-D frames, builds a live Open3D point cloud, and exports to PLY on exit.

---

## 🧠 Trained Neural Network

The project includes a fully trained custom 3D reconstruction neural network. Two checkpoint files are ready for inference in `data/models/`:

| Checkpoint | Size | Description |
| :--- | :--- | :--- |
| `best_3d_ai_model.pth` | 87.7 MB | Best validation-loss checkpoint (auto-saved during training) |
| `custom_3d_ai_model.pth` | 29.3 MB | Final epoch checkpoint |

### Network Architecture

**HighPrecision3DReconstructionNet** — a ResNet Residual Feature Pyramid encoder-decoder (U-Net style) with skip connections:

```
Input RGB (3, H, W)
  │
  ├─ Encoder ─────────────────────────────────────────────┐
  │   Stage 1:  Conv 3→32  + ResBlock(32)                 │ skip f1
  │   Stage 2:  Conv 32→64  ↓/2 + ResBlock(64)            │ skip f2
  │   Stage 3:  Conv 64→128 ↓/4 + ResBlock(128)           │ skip f3
  │   Stage 4:  Conv 128→256 ↓/8 + ResBlock(256)          │
  │                                                       │
  │   Bottleneck: ResBlock(256) → Conv(256) → ResBlock(256)│
  │                                                       │
  ├─ Decoder (U-Net skip concatenation) ──────────────────┘
  │   Up1: ConvTranspose 256→128 ↑×2 + cat(f3) → ResBlock(256)
  │   Up2: ConvTranspose 256→64  ↑×4 + cat(f2) → ResBlock(128)
  │   Up3: ConvTranspose 128→32  ↑×8 + cat(f1) → ResBlock(64)
  │
  └─ Multi-Head Output (3 simultaneous predictions)
       ├─ Depth Head:   Conv 64→32→1 + Sigmoid     → Depth Map (1, H, W)
       ├─ Normal Head:  Conv 64→32→3 + Tanh + Norm → Surface Normals (3, H, W)
       └─ Mask Head:    Conv 64→16→1 + Sigmoid     → Foreground Mask (1, H, W)
```

- Resolution-agnostic: supports 64×64, 128×128, 256×256+ inputs
- LeakyReLU(0.2) activations with BatchNorm throughout
- Unit-normalized surface normal output (‖N‖ = 1)

### Training Data

The model was trained on **42 3D meshes** (42 processed `.npz` datasets) spanning 6 categories:

| Category | Examples | Count |
| :--- | :--- | :--- |
| **Stanford / Classic** | Bunny, Dragon, Armadillo, Spot Cow | 4 |
| **Museum Sculptures** | Nefertiti Bust, Max Planck | 2 |
| **ShapeNet CAD** | Airplane, Car, Chair, Lamp, Mug, Table | 8 |
| **Procedural Organic** | Asteroid, Capsule, Dumbbell, Ellipsoid, Mushroom, Pebble | 8 |
| **Procedural Mechanical** | Gear, Stepped Column, Helix Bar, Washer, L-Bracket, T-Connector | 10 |
| **Primitives / Calibration** | Cube, Icosphere, Cylinder, Torus, Cone, Pyramid, Prism, Disc | 10 |

Each mesh is rendered from **32 multi-view viewpoints** (16 yaw angles × 2 elevation pitches at 0° and 20°), producing **~1,344 training samples** at 128×128 resolution with paired depth maps, surface normals, and foreground masks.

### Training Configuration

| Parameter | Value |
| :--- | :--- |
| Epochs | 150 |
| Batch Size | 16 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Peak Learning Rate | 5e-4 |
| LR Schedule | Linear warmup (5 epochs) → Cosine Annealing (η_min=LR×0.01) |
| Backup Schedule | ReduceLROnPlateau (factor=0.5, patience=10) |
| Early Stopping | 30 epochs patience |
| Precision | AMP FP16 on CUDA, FP32 on CPU |
| Gradient Clipping | max_norm=1.0 |
| Train / Val Split | 85% / 15% (seed=42) |

**Multi-term loss function:**

```
L = 1.0 × MSE + 0.5 × Chamfer + 0.1 × Laplacian + 0.1 × Normal
```

- **MSE Loss** — Per-pixel depth regression
- **Bidirectional Chamfer Distance** — Point cloud geometric accuracy
- **Laplacian Edge Loss** — Surface smoothness regularization
- **Normal Consistency Loss** — Surface orientation coherence + 0.5 × Normal MSE

### Re-Training or Fine-Tuning

```bash
# Generate / download training datasets
python pipeline/generate_complex_datasets.py
python pipeline/download_kaggle_datasets.py
python pipeline/download_sculpture_datasets.py

# Train from scratch (or auto-resume from best checkpoint)
python pipeline/train_ai_3d.py
```

Training automatically resumes from `best_3d_ai_model.pth` if it exists. Training progress is exposed in real-time via the API (`GET /api/training/status`) and the web dashboard.

---

## 📡 API Reference

### Scan & Reconstruction

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/scan/upload` | Upload turntable frame batch or ZIP to start multi-view scan |
| `POST` | `/api/scan/single-image` | Upload single photo + select engine for instant 3D |
| `GET` | `/api/scan/status/{job_id}` | Poll job progress, current step, and logs |
| `GET` | `/api/scan/download/{job_id}/{fmt}` | Download model (`stl`, `glb`, `ply`, `obj`) |
| `POST` | `/scan` | Direct multi-part upload with synchronous download |

### AI Engines & Configuration

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/engines` | List available 3D engines, CUDA status, and fallback chains |

### Neural Training & Datasets

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/training/start` | Launch neural network training run |
| `GET` | `/api/training/status` | Real-time epoch, loss, and progress metrics |
| `GET` | `/api/datasets/inventory` | List available training datasets and mesh counts |
| `POST` | `/api/datasets/ingest` | Download and ingest datasets from Kaggle |

### Metrology & Quality

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/measure` | OBB dimensional measurement of a mesh (L × W × H mm) |
| `POST` | `/api/topology/analyze` | Mesh topology analysis (manifold, watertight, genus) |
| `POST` | `/api/topology/repair` | Auto-repair non-manifold edges and holes |

### Camera Calibration

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/calibration/run` | Run checkerboard camera calibration |
| `GET` | `/api/calibration` | Retrieve stored calibration parameters |

### Dashboard

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive 3D Dashboard UI |

---

## 🧪 Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run individual test modules
pytest tests/test_pipeline.py -v                    # Core pipeline
pytest tests/test_ai_pipeline.py -v                 # AI depth & single-image
pytest tests/test_high_detail.py -v                 # High-detail reconstruction
pytest tests/test_turntable_and_calibration.py -v   # Turntable & calibration
pytest tests/test_obb_and_diffusion.py -v           # OBB measurement & diffusion
pytest tests/test_kaggle_and_training.py -v         # Dataset ingestion & training
```

---

## 🔧 Orbiter Turntable Integration

This project integrates the [Orbiter](https://github.com/RawMechatronicsOrg/Orbiter) two-axis motorized camera turntable under `orbiter/`. Orbiter provides hardware-controlled, calibrated photo capture with known camera poses — feeding directly into the reconstruction pipeline.

### Orbiter Components

| Directory | Purpose |
| :--- | :--- |
| `orbiter/server/` | FastAPI server with WebSocket scene-graph, ESP32 proxy, scan session management |
| `orbiter/ui/` | Vite + React + react-three-fiber browser UI — live 3D view + photo library |
| `orbiter/colmap/` | COLMAP integration with SfM priors from Orbiter poses |
| `orbiter/docker/` | Docker Compose stack (server + UI + COLMAP) |
| `orbiter/firmware/` | ESP32 firmware for stepper/encoder control |
| `orbiter/cad/` | 3D-printable mechanical parts (STL, GLB, FBX) |
| `orbiter/docs/` | Subsystem documentation |

### Running Orbiter (Docker)

```bash
cd orbiter/docker
cp .env.example .env    # Set ESP32_IP and CAMERA_URL
docker compose up
```

Open **http://localhost:5173** for the Orbiter UI.

### Running Orbiter (Standalone)

```bash
# Server
cd orbiter/server
pip install -e .
uvicorn orbiter_server.app:create_app --factory --host 0.0.0.0 --port 8001

# UI (separate terminal)
cd orbiter/ui
npm install
npm run dev
```

### How They Connect

1. **Orbiter** captures calibrated photos with known (azimuth, elevation) poses
2. Export the session via Orbiter's SfM export → produces images + `manifest.json`
3. Feed those images into the reconstruction pipeline (`01_preprocess.py` → `04_ai_refine.py`)
4. Or use Orbiter's COLMAP integration for dense reconstruction with prior poses

---

## 📐 Reconstruction Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT METHODS                                     │
├──────────────┬──────────────┬──────────────┬────────────────────────────┤
│ Single Photo │ Turntable    │ D455 Depth   │ Orbiter Hardware           │
│              │ Sequence     │ Camera       │ Turntable                  │
└──────┬───────┴──────┬───────┴──────┬───────┴────────────┬───────────────┘
       │              │              │                    │
       ▼              ▼              ▼                    ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ AI Engine    │ │ Background   │ │ TSDF Volume  │ │ Calibrated   │
│ Selection    │ │ Removal      │ │ Integration  │ │ Pose Export  │
│ (TripoSR /  │ │ (rembg)      │ │ (Open3D)     │ │ (manifest)   │
│ TRELLIS /    │ │              │ │              │ │              │
│ Hunyuan3D)   │ │      ↓       │ │      ↓       │ │      ↓       │
│              │ │ COLMAP SfM / │ │ Marching     │ │ Multi-View   │
│      ↓       │ │ Multi-View   │ │ Cubes Mesh   │ │ Fusion       │
│ 360° Mesh    │ │ Fusion       │ │ Extraction   │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │              │              │                    │
       └──────────────┴──────────────┴────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │ Poisson Surface Meshing   │
                    │ + Laplacian Smoothing     │
                    │ + Watertight Repair       │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │ Texture Projection (GLB) │
                    │ Ground-Plane Alignment   │
                    │ OBB Measurement (mm)     │
                    │ Scan Quality Score       │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │ Export: STL / GLB / PLY   │
                    │ Dashboard 3D Viewer      │
                    │ API Download             │
                    └──────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | FastAPI, Uvicorn, Python 3.9+ |
| **AI / ML** | PyTorch, Depth-Anything-V2, MiDaS, TripoSR, TRELLIS, Hunyuan3D 2.0 |
| **3D Processing** | Open3D, Trimesh, pycolmap, OpenCV |
| **Frontend** | Three.js, WebGL, vanilla HTML/CSS/JS |
| **Hardware** | Intel RealSense D455, ESP32 stepper motor (Orbiter) |
| **Containerization** | Docker Compose (Orbiter subsystem) |

---

## 📄 License

See [orbiter/LICENSE](orbiter/LICENSE) for the Orbiter subsystem license.
