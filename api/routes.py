import os
import shutil
import zipfile
import uuid
import asyncio
import math
import time
import secrets
from pathlib import Path
from typing import List, Dict, Optional
import trimesh
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Query, Depends, Header, Form
from fastapi.responses import FileResponse

from api.db import (save_model, get_all_models, get_model, get_model_by_job, 
                    delete_model, get_model_count, create_user, authenticate_user,
                    get_user_settings, update_user_settings, get_all_users_db)

from pipeline.preprocess import process_images
from pipeline.colmap_poses import run_colmap_reconstruction
from pipeline.open3d_mesh import pointcloud_to_mesh
from pipeline.ai_refine import refine_mesh
from pipeline.single_image_3d import single_image_to_3d_mesh
from pipeline.ai_depth_3d import generate_ai_depth_3d_mesh
from pipeline.triposr_inference import generate_triposr_3d_mesh
from pipeline.trellis_inference import generate_trellis_3d_mesh
from pipeline.hunyuan3d_inference import generate_hunyuan3d_mesh
from pipeline.multiview_fusion import reconstruct_multiview_turntable_mesh
from pipeline.mesh_topology import analyze_mesh_topology, repair_mesh_topology
from pipeline.train_ai_3d import get_dataset_inventory, train_custom_3d_ai_model, TRAINING_STATE
from pipeline.download_kaggle_datasets import run_full_dataset_ingestion, DATASET_SPECS
from pipeline.multiview_diffusion import generate_diffusion_3d_mesh
from pipeline.dimensional_eval import auto_measure_mesh
from pipeline.ai.engine_manager import run_single_image as engine_manager_run
from pipeline.mesh_quality import validate_mesh
from pipeline.texture_projector import project_texture_to_mesh
from pipeline.turntable_calibrated_pipeline import run_calibrated_turntable_scan
from pipeline.camera_calibration import load_calibration, calibrate_camera_from_checkerboard, save_calibration
from pipeline.scan_confidence import evaluate_scan_quality

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent / "data"
JOBS: Dict[str, Dict] = {}


def _persist_generation_job(job_id):
    import time,json
    job=JOBS[job_id];job['updated_at']=time.time()
    if job.get('started_at'):
        end=job.get('finished_at') or job['updated_at'];job['elapsed_seconds']=max(0,end-job['started_at'])
    keys=('job_id','status','progress','step','mode','engine_requested','engine_used','engine_label','degraded','warnings','error','created_at','started_at','updated_at','finished_at','elapsed_seconds')
    directory=BASE_DIR/'jobs'/job_id;directory.mkdir(parents=True,exist_ok=True)
    temp=directory/'job_status.tmp';temp.write_text(json.dumps({key:job[key] for key in keys if key in job},indent=2),encoding='utf-8');temp.replace(directory/'job_status.json')

def get_job_dirs(job_id: str):
    job_dir = BASE_DIR / "jobs" / job_id
    raw_dir = job_dir / "raw"
    proc_dir = job_dir / "processed"
    workspace_dir = job_dir / "workspace"
    out_dir = job_dir / "output"
    for d in [raw_dir, proc_dir, workspace_dir, out_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return job_dir, raw_dir, proc_dir, workspace_dir, out_dir


# =========================================================================
#  Single-Image 3D Engine Registry (Model Selector Backend)
# =========================================================================
#  Every engine exposes the same call signature:
#      fn(image_path: str, output_stl_path: str, output_glb_path: str) -> str
#  so the dashboard model selector can swap engines with no other changes.
#  'fallbacks' is an ordered degradation chain used when weights or the
#  native package are unavailable (e.g. no CUDA, no hy3dgen install).

def _run_triposr(img: str, stl: str, glb: str) -> str:
    return generate_triposr_3d_mesh(img, stl, glb)


def _run_trellis(img: str, stl: str, glb: str) -> str:
    return generate_trellis_3d_mesh(img, stl, glb)


def _run_hunyuan3d(img: str, stl: str, glb: str) -> str:
    return generate_hunyuan3d_mesh(img, stl, glb, paint_texture=True)


def _run_ai_depth(img: str, stl: str, glb: str) -> str:
    return generate_ai_depth_3d_mesh(img, stl, model_type="foundation")


def _run_custom_3d(img: str, stl: str, glb: str) -> str:
    return generate_ai_depth_3d_mesh(img, stl, model_type="custom")


SINGLE_IMAGE_ENGINES: Dict[str, Dict] = {
    "triposr": {
        "label": "TripoSR (Stability AI)",
        "detail": "Fast tri-plane LRM. ~1s inference, solid 360° geometry.",
        "runner": _run_triposr,
        "probe": ("tsr.system", "TSR"),
        "fallbacks": ["custom_3d", "ai_depth"],
    },
    "custom_3d": {
        "label": "Custom Trained 3D AI (PyTorch)",
        "detail": "Trained ResNet Feature Pyramid 3D neural network (best_3d_ai_model.pth).",
        "runner": _run_custom_3d,
        "probe": ("torch", "__version__"),
        "fallbacks": ["ai_depth"],
    },
    "trellis": {
        "label": "TRELLIS (Microsoft)",
        "detail": "Structured LATent (SLAT). Highest geometric fidelity.",
        "runner": _run_trellis,
        "probe": ("trellis.pipelines", "TrellisImageTo3DPipeline"),
        "fallbacks": ["triposr", "custom_3d", "ai_depth"],
    },
    "hunyuan3d": {
        "label": "Hunyuan3D 2.0 (Tencent)",
        "detail": "DiT flow-matching shape + Paint texture synthesis.",
        "runner": _run_hunyuan3d,
        "probe": ("hy3dgen.shapegen", "Hunyuan3DDiTFlowMatchingPipeline"),
        "fallbacks": ["triposr", "custom_3d", "ai_depth"],
    },
    "ai_depth": {
        "label": "AI Depth (Depth-Anything-V2)",
        "detail": "Monocular 2.5D depth preview. Fastest, not a true 360° scan.",
        "runner": _run_ai_depth,
        "probe": ("transformers", "pipeline"),
        "fallbacks": [],
    },
}

DEFAULT_SINGLE_IMAGE_ENGINE = "trellis"


def _probe_engine(engine_key: str) -> bool:
    """Best-effort check that an engine's native package is importable."""
    if engine_key == 'hunyuan3d':
        from pipeline.ai.hunyuan3d_engine import Hunyuan3DEngine
        return Hunyuan3DEngine().is_available()
    if engine_key == "custom_3d":
        from pipeline.training_data import checkpoint_status
        return checkpoint_status()["ready"]
    spec = SINGLE_IMAGE_ENGINES.get(engine_key, {}).get("probe")
    if not spec:
        return False
    module_name, attr = spec
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return hasattr(mod, attr)
    except Exception:
        return False


def resolve_engine(engine: str) -> str:
    """
    Normalizes a user-supplied engine key, falling back to the default.
    Accepts loose spellings from API/IoT clients: 'Hunyuan-3D', 'hunyuan_3d',
    'TRELLIS', 'depth-anything', etc. Unknown values resolve to the default.
    """
    import re
    key = re.sub(r"[^a-z0-9]", "", (engine or "").lower())
    aliases = {
        "custom": "custom_3d",
        "custom3d": "custom_3d",
        "customai": "custom_3d",
        "custom3dai": "custom_3d",
        "bestmodel": "custom_3d",
        "pytorch3d": "custom_3d",
        "triposr": "triposr",
        "tripo": "triposr",
        "tripo3d": "triposr",
        "lrm": "triposr",
        "trellis": "trellis",
        "slat": "trellis",
        "hunyuan": "hunyuan3d",
        "hunyuan3d": "hunyuan3d",
        "hunyuan3d2": "hunyuan3d",
        "hunyuan3d20": "hunyuan3d",
        "hy3d": "hunyuan3d",
        "aidepth": "ai_depth",
        "depth": "ai_depth",
        "depthanything": "ai_depth",
        "depthanythingv2": "ai_depth",
        "midas": "ai_depth",
    }
    return aliases.get(key, DEFAULT_SINGLE_IMAGE_ENGINE)


def run_single_image_engine(engine: str, img: str, stl: str, glb: str, logs: List[str]) -> Dict:
    """
    Executes the selected engine, walking its fallback chain on failure.
    Returns {'engine_requested', 'engine_used', 'degraded'}.
    """
    requested = resolve_engine(engine)
    chain = [requested] + list(SINGLE_IMAGE_ENGINES[requested]["fallbacks"])
    last_err = None

    for candidate in chain:
        meta = SINGLE_IMAGE_ENGINES[candidate]
        try:
            if candidate != requested:
                logs.append(f"Falling back to {meta['label']}...")
            else:
                logs.append(f"Running {meta['label']} — {meta['detail']}")
            meta["runner"](img, stl, glb)
            return {
                "engine_requested": requested,
                "engine_used": candidate,
                "engine_label": meta["label"],
                "degraded": candidate != requested,
            }
        except Exception as err:
            last_err = err
            logs.append(f"Notice: {meta['label']} unavailable ({err}).")

    raise RuntimeError(f"All engines failed for '{requested}'. Last error: {last_err}")


def execute_3d_scan_pipeline(job_id: str):
    JOBS[job_id]["status"] = "processing"
    JOBS[job_id]["progress"] = 5
    JOBS[job_id]["logs"].append("Starting 3D reconstruction pipeline...")

    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)
    final_stl = out_dir / "final_model.stl"
    final_glb = out_dir / "final_model.glb"
    ply_path = workspace_dir / "sparse_cloud.ply"

    try:
        raw_images = sorted(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpeg")))
        if not raw_images:
            raise RuntimeError("No uploaded images found in job directory.")

        JOBS[job_id]["step"] = "1. Multi-View Background Isolation"
        JOBS[job_id]["progress"] = 20
        JOBS[job_id]["logs"].append(f"Isolating foreground subject across {len(raw_images)} turntable photos...")

        JOBS[job_id]["step"] = "2. 360° AI Depth Point Cloud Fusion"
        JOBS[job_id]["progress"] = 55
        JOBS[job_id]["logs"].append("Extracting Depth-Anything-V2 dense point clouds and rotating into 360° world frame...")

        try:
            reconstruct_multiview_turntable_mesh(
                image_paths=raw_images,
                output_stl_path=str(final_stl),
                output_glb_path=str(final_glb),
                poisson_depth=10,
                point_stride=2
            )
        except Exception as fusion_err:
            JOBS[job_id]["logs"].append(f"Notice: Turntable fusion notice ({fusion_err}). Running COLMAP / Poisson pipeline...")
            
            # Step 1: Preprocessing & Background removal
            processed_files = process_images(str(raw_dir), str(proc_dir))
            if not processed_files:
                raise RuntimeError("Background removal produced no output images.")

            # Step 2: COLMAP Pose Estimation & Point Cloud
            ply_path = run_colmap_reconstruction(str(proc_dir), str(workspace_dir))

            # Step 3: Open3D Poisson Reconstruction
            initial_stl = out_dir / "initial_mesh.stl"
            pointcloud_to_mesh(str(ply_path), str(initial_stl))

            # Step 4: Refinement
            refine_mesh(str(initial_stl), str(final_stl))

            # Step 5: Texture projection
            texture_src = str(raw_images[0])
            project_texture_to_mesh(str(final_stl), texture_src, str(final_glb))

        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["step"] = "Completed"
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["stl_path"] = str(final_stl)
        JOBS[job_id]["glb_path"] = str(final_glb)
        JOBS[job_id]["ply_path"] = str(ply_path)
        JOBS[job_id]["logs"].append("3D Scan completed successfully!")

        # Persist to model database
        _save_job_to_db(job_id, engine="colmap", engine_label="Multi-View COLMAP", mode="turntable")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["logs"].append(f"Error in pipeline: {e}")


def execute_single_image_3d_pipeline(job_id: str, image_filename: str, engine: str = DEFAULT_SINGLE_IMAGE_ENGINE):
    resolved = resolve_engine(engine)
    engine_label = SINGLE_IMAGE_ENGINES[resolved]["label"]

    JOBS[job_id]["status"] = "processing"
    JOBS[job_id]["progress"] = 10
    JOBS[job_id]["engine"] = resolved
    JOBS[job_id]["engine_label"] = engine_label
    JOBS[job_id]["logs"].append(f"Starting Single-Image 3D Generation for '{image_filename}' using {engine_label}...")

    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)
    input_img_path = raw_dir / image_filename

    try:
        JOBS[job_id]["step"] = "1. AI Subject Isolation"
        JOBS[job_id]["progress"] = 25
        JOBS[job_id]["logs"].append("Isolating foreground subject and preparing alpha tensor...")

        JOBS[job_id]["step"] = f"2. 360° Reconstruction — {engine_label}"
        JOBS[job_id]["progress"] = 65

        final_stl = out_dir / "final_model.stl"
        final_glb = out_dir / "final_model.glb"

        result = run_single_image_engine(
            resolved,
            str(input_img_path),
            str(final_stl),
            str(final_glb),
            JOBS[job_id]["logs"]
        )

        JOBS[job_id]["engine_used"] = result["engine_used"]
        JOBS[job_id]["engine_label"] = result["engine_label"]
        JOBS[job_id]["degraded"] = result["degraded"]
        if result["degraded"]:
            JOBS[job_id]["logs"].append(
                f"Note: '{resolved}' was unavailable — output produced by '{result['engine_used']}' instead."
            )

        JOBS[job_id]["step"] = "3. OBB Physical Auto-Measurement"
        JOBS[job_id]["progress"] = 90
        try:
            dim_report = auto_measure_mesh(str(final_stl))
            dim = dim_report.get("dimensions", dim_report)
            JOBS[job_id]["dimensions"] = dim
            JOBS[job_id]["logs"].append(
                f"OBB Dimensions: {dim.get('length_mm', 0)} x {dim.get('width_mm', 0)} x {dim.get('height_mm', 0)} mm"
            )
        except Exception as measure_err:
            JOBS[job_id]["logs"].append(f"Notice: OBB measurement skipped ({measure_err}).")

        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["step"] = "Completed"
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["stl_path"] = str(final_stl)
        JOBS[job_id]["glb_path"] = str(final_glb)
        JOBS[job_id]["ply_path"] = str(workspace_dir / "sparse_cloud.ply")
        JOBS[job_id]["logs"].append(
            f"Single Image 360° Solid 3D Model created successfully via {result['engine_label']}!"
        )

        # Persist to model database
        _save_job_to_db(job_id, engine=result.get("engine_used", resolved), engine_label=result.get("engine_label", ""), mode="single_image")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["logs"].append(f"Error in single-image generation: {e}")


@router.get("/api/engines")
def list_single_image_engines():
    """
    Returns the available single-image 3D engines for the dashboard model selector,
    including whether each engine's native package is currently importable.
    """
    engines = []
    for key, meta in SINGLE_IMAGE_ENGINES.items():
        engines.append({
            "key": key,
            "label": meta["label"],
            "detail": meta["detail"],
            "available": _probe_engine(key),
            "fallbacks": meta["fallbacks"],
        })
    return {
        "default": DEFAULT_SINGLE_IMAGE_ENGINE,
        "cuda_available": _torch_cuda_available(),
        "engines": engines
    }


@router.post("/api/reconstruct")
async def reconstruct_single_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    engine: str = Form(DEFAULT_SINGLE_IMAGE_ENGINE),
    object_type: str = Form("organic"),
    texture: bool = Form(True),
    quality: str = Form("high"),
    refine: bool = Form(False),
    printable: bool = Form(False),
    photo_detail: bool = Form(False),
):
    """
    Unified single-image 3D reconstruction endpoint (Mode A: AI generation).

    Engine priority (auto-resolved when `engine` is unavailable):
    TRELLIS -> Hunyuan3D -> TripoSR.
    Single-image mode never routes through the custom depth->Poisson or the
    turntable COLMAP path.

    Returns job_id for status polling on /api/scan/status/{job_id}.
    """
    if resolve_engine(engine) == "custom_3d":
        from pipeline.training_data import checkpoint_status
        status=checkpoint_status()
        if not status["ready"]: raise HTTPException(409,status["reason"])
    image_name = Path(file.filename or "image.png").name
    if Path(image_name).suffix.lower() not in {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}:
        raise HTTPException(status_code=400, detail="Upload a PNG, JPG, WEBP, BMP or TIFF image.")
    if object_type not in {'organic','hard_surface'}:
        raise HTTPException(status_code=400, detail="Unknown object type.")
    # Bound the upload and reject corrupt images before an expensive engine starts.
    image_bytes = await file.read(25 * 1024 * 1024 + 1)
    if len(image_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the 25 MB upload limit.")
    try:
        from PIL import Image
        import io
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError("Image exceeds 40 megapixels.")
            image.verify()
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid image: {error}") from error
    valid_quality = quality if quality in ("fast", "high", "ultra") else "high"
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "step": "Image Uploaded",
        "mode": "single_image",
        "engine_requested": engine,
        "object_type": object_type,
        "texture": texture,
        "quality": valid_quality,
        "refine": refine,
        "printable": printable,
        "logs": [f"Image '{file.filename}' uploaded. Engine: {engine}, object_type={object_type}, texture={texture}, quality={valid_quality}, refine={refine}, printable={printable}. Job queued."],
        "error": None,
    }

    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)
    file_loc = raw_dir / image_name
    file_loc.write_bytes(image_bytes)
    import time
    JOBS[job_id]["created_at"]=time.time()
    _persist_generation_job(job_id)

    background_tasks.add_task(
        execute_reconstruct_pipeline,
        job_id,
        image_name,
        engine,
        object_type,
        texture,
        valid_quality,
        refine,
        printable,
        photo_detail,
    )
    return {
        "job_id": job_id,
        "engine": engine,
        "message": "Single-image 3D reconstruction started.",
    }


def execute_reconstruct_pipeline(
    job_id: str,
    image_filename: str,
    engine: str,
    object_type: str,
    texture: bool,
    quality: str,
    refine: bool = True,
    printable: bool = False,
    photo_detail: bool = False,
):
    """Background worker for the unified /api/reconstruct endpoint."""
    import time
    JOBS[job_id]["started_at"]=time.time()
    _persist_generation_job(job_id)
    def _progress_cb(stage, percent, message):
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["step"] = stage
        JOBS[job_id]["progress"] = max(JOBS[job_id]["progress"], percent)
        JOBS[job_id]["logs"].append(message)
        _persist_generation_job(job_id)

    try:
        # The engine manager owns the full pipeline:
        # generate -> cleanup -> bounded refine -> texture -> QC -> multi-export.
        result = engine_manager_run(
            engine,
            str(get_job_dirs(job_id)[1] / image_filename),
            str(get_job_dirs(job_id)[4] / "final_model.stl"),
            str(get_job_dirs(job_id)[4] / "final_model.glb"),
            object_type=object_type,
            texture=texture,
            quality=quality,
            progress_cb=_progress_cb,
            refine=refine,
            printable=printable,
            photo_detail=photo_detail,
        )
        JOBS[job_id]["engine_used"] = result["engine_used"]
        JOBS[job_id]["engine_label"] = result["engine_label"]
        JOBS[job_id]["degraded"] = result["degraded"]
        JOBS[job_id]["logs"].extend(result["logs"])
        if result["degraded"]:
            JOBS[job_id]["logs"].append(
                f"Note: '{engine}' was unavailable — produced by '{result['engine_used']}'."
            )

        JOBS[job_id]["warnings"] = result.get("warnings", [])
        JOBS[job_id]["logs"].extend(f"Quality note: {warning}" for warning in result.get("warnings", []))
        JOBS[job_id]["quality"] = result.get("quality")
        JOBS[job_id]["files"] = result.get("files", {})
        JOBS[job_id]["mesh_path"] = result.get("mesh_path")
        JOBS[job_id]["glb_path"] = result.get("glb_path")
        JOBS[job_id]["stl_path"] = result.get("files", {}).get("stl")

        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["step"] = "Completed"
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["finished_at"]=time.time()
        _persist_generation_job(job_id)
        JOBS[job_id]["logs"].append(
            f"Single-image 3D model created successfully via {result['engine_label']}!"
        )

        # Persist to model database
        _save_job_to_db(
            job_id,
            engine=result.get("engine_used", engine),
            engine_label=result.get("engine_label", ""),
            mode="single_image",
            object_type=object_type,
            quality=quality,
        )
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["logs"].append(f"Error in reconstruction: {e}")
        JOBS[job_id]["finished_at"]=time.time()
        _persist_generation_job(job_id)


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@router.post("/api/scan/single-image")
async def create_single_image_3d(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    engine: str = Form(DEFAULT_SINGLE_IMAGE_ENGINE)
):
    """
    Single-Image 3D Endpoint with selectable engine.
    engine = 'triposr' | 'trellis' | 'hunyuan3d' | 'ai_depth'
    """
    resolved = resolve_engine(engine)
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "step": "Single Image Uploaded",
        "engine": resolved,
        "engine_label": SINGLE_IMAGE_ENGINES[resolved]["label"],
        "logs": [f"Image '{file.filename}' uploaded. Engine: {SINGLE_IMAGE_ENGINES[resolved]['label']}. Job queued."],
        "error": None
    }

    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)
    file_loc = raw_dir / file.filename
    with open(file_loc, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(execute_single_image_3d_pipeline, job_id, file.filename, resolved)
    return {
        "job_id": job_id,
        "engine": resolved,
        "engine_label": SINGLE_IMAGE_ENGINES[resolved]["label"],
        "message": "Single image uploaded. 3D Model generation started."
    }


def execute_ai_depth_3d_pipeline(job_id: str, image_filename: str):
    JOBS[job_id]["status"] = "processing"
    JOBS[job_id]["progress"] = 10
    JOBS[job_id]["logs"].append(f"Starting PyTorch Neural AI Depth 3D Generation for '{image_filename}'...")

    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)
    input_img_path = raw_dir / image_filename

    try:
        JOBS[job_id]["step"] = "1. AI Subject Isolation"
        JOBS[job_id]["progress"] = 25
        JOBS[job_id]["logs"].append("Isolating subject background...")

        JOBS[job_id]["step"] = "2. PyTorch Neural Depth Inference"
        JOBS[job_id]["progress"] = 60
        JOBS[job_id]["logs"].append("Predicting per-pixel AI monocular depth map...")

        final_stl = out_dir / "final_model.stl"
        generate_ai_depth_3d_mesh(str(input_img_path), str(final_stl))

        final_glb = out_dir / "final_model.glb"

        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["step"] = "Completed"
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["stl_path"] = str(final_stl)
        JOBS[job_id]["glb_path"] = str(final_glb)
        JOBS[job_id]["ply_path"] = str(workspace_dir / "sparse_cloud.ply")
        JOBS[job_id]["logs"].append("PyTorch AI Depth Textured 3D Model created successfully!")

        # Persist to model database
        _save_job_to_db(job_id, engine="ai_depth", engine_label="AI Depth (Depth-Anything-V2)", mode="ai_depth")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["logs"].append(f"Error in AI depth generation: {e}")


@router.post("/api/scan/ai-depth")
async def create_ai_depth_3d(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "step": "Image Uploaded",
        "logs": [f"Image '{file.filename}' uploaded for AI Neural Depth. Job queued."],
        "error": None
    }

    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)
    file_loc = raw_dir / file.filename
    with open(file_loc, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(execute_ai_depth_3d_pipeline, job_id, file.filename)
    return {"job_id": job_id, "message": "AI Neural Depth 3D generation started."}


def execute_diffusion_3d_pipeline(job_id: str, image_filename: str):
    JOBS[job_id]["status"] = "processing"
    JOBS[job_id]["progress"] = 10
    JOBS[job_id]["logs"].append(f"Starting Multi-View Diffusion 3D Pipeline for '{image_filename}'...")

    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)
    input_img_path = raw_dir / image_filename

    try:
        JOBS[job_id]["step"] = "1. Generative Multi-View Diffusion"
        JOBS[job_id]["progress"] = 30
        JOBS[job_id]["logs"].append("Synthesizing 360-degree novel views (front, sides, back) to resolve unseen geometry...")

        final_stl = out_dir / "final_model.stl"
        final_glb = out_dir / "final_model.glb"

        JOBS[job_id]["step"] = "2. 360° Solid Mesh Fusion"
        JOBS[job_id]["progress"] = 65
        JOBS[job_id]["logs"].append("Fusing synthesized multi-view cloud into watertight Poisson surface...")

        report = generate_diffusion_3d_mesh(str(input_img_path), str(final_stl), str(final_glb))

        JOBS[job_id]["step"] = "3. OBB Physical Auto-Measurement"
        JOBS[job_id]["progress"] = 90
        dim = report.get("dimensions", {})
        JOBS[job_id]["logs"].append(f"OBB Dimensions: {dim.get('length_mm', 0)} x {dim.get('width_mm', 0)} x {dim.get('height_mm', 0)} mm")

        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["step"] = "Completed"
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["stl_path"] = str(final_stl)
        JOBS[job_id]["glb_path"] = str(final_glb)
        JOBS[job_id]["ply_path"] = str(workspace_dir / "sparse_cloud.ply")
        JOBS[job_id]["dimensions"] = dim
        JOBS[job_id]["logs"].append("Multi-View Diffusion 360° Solid 3D Model created successfully!")

        # Persist to model database
        _save_job_to_db(job_id, engine="diffusion", engine_label="Multi-View Diffusion", mode="diffusion")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["logs"].append(f"Error in Multi-View Diffusion generation: {e}")


@router.post("/api/scan/diffusion-3d")
async def create_diffusion_3d(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Multi-View Diffusion Endpoint: Synthesizes 360-degree unseen backside geometry from a single photo
    and generates a solid watertight 3D model with OBB measurements.
    """
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "step": "Single Photo Uploaded for 360° Diffusion",
        "logs": [f"Image '{file.filename}' uploaded for Multi-View Diffusion. Job queued."],
        "error": None
    }

    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)
    file_loc = raw_dir / file.filename
    with open(file_loc, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(execute_diffusion_3d_pipeline, job_id, file.filename)
    return {"job_id": job_id, "message": "Multi-View Diffusion 3D generation started."}


def execute_calibrated_turntable_pipeline(job_id: str, rotation_step_deg: float = 15.0, expected_frames: int = 24):
    JOBS[job_id]["status"] = "processing"
    JOBS[job_id]["progress"] = 10
    JOBS[job_id]["logs"].append(f"Starting 3DINVENZA Calibrated Turntable Pipeline (Step: {rotation_step_deg}°)...")

    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)
    images = sorted(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpeg")))

    if not images:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = "No valid turntable sequence images found."
        return

    try:
        JOBS[job_id]["step"] = "1. Camera Calibration & Undistortion"
        JOBS[job_id]["progress"] = 25
        JOBS[job_id]["logs"].append("Applying camera intrinsics and silhouette masking...")

        final_stl = out_dir / "final_model.stl"
        final_glb = out_dir / "final_model.glb"

        JOBS[job_id]["step"] = "2. Orbital Geometric Point Cloud Fusion"
        JOBS[job_id]["progress"] = 55
        JOBS[job_id]["logs"].append("Fusing 360° orbital point cloud using known turntable rotation matrices...")

        report = run_calibrated_turntable_scan(
            image_paths=images,
            output_stl_path=str(final_stl),
            output_glb_path=str(final_glb),
            rotation_step_deg=rotation_step_deg,
            expected_frames=expected_frames,
            poisson_depth=10
        )

        JOBS[job_id]["step"] = "3. Canonical Alignment & OBB Measurement"
        JOBS[job_id]["progress"] = 90
        dim = report.get("dimensions", {})
        qual = report.get("quality", {})
        JOBS[job_id]["logs"].append(f"Canonical Ground Aligned | OBB: {dim.get('length_mm', 0)} x {dim.get('width_mm', 0)} x {dim.get('height_mm', 0)} mm")
        JOBS[job_id]["logs"].append(f"Scan Quality Score: {qual.get('overall_score', 0)}%")

        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["step"] = "Completed"
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["stl_path"] = str(final_stl)
        JOBS[job_id]["glb_path"] = str(final_glb)
        JOBS[job_id]["ply_path"] = str(workspace_dir / "sparse_cloud.ply")
        JOBS[job_id]["dimensions"] = dim
        JOBS[job_id]["quality"] = qual
        JOBS[job_id]["logs"].append("3DINVENZA Calibrated Turntable 3D Reconstruction completed successfully!")

        # Persist to model database
        _save_job_to_db(job_id, engine="turntable", engine_label="Calibrated Turntable", mode="turntable")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["logs"].append(f"Error in calibrated turntable reconstruction: {e}")


@router.post("/api/scan/turntable-calibrated")
async def create_calibrated_turntable_scan(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    rotation_step: float = Form(15.0),
    expected_frames: int = Form(24)
):
    """
    3DINVENZA Calibrated Turntable Scan Endpoint:
    Ingests 24 to 72 turntable frames with known angular steps, camera intrinsics,
    scale reference, canonical ground alignment, and OBB metrology.
    """
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "step": f"Uploaded {len(files)} Turntable Frames",
        "logs": [f"Received {len(files)} frames for Calibrated Turntable Scan ({rotation_step}° step). Job queued."],
        "error": None
    }

    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)

    # Save uploaded files (support .zip or multiple images)
    for file in files:
        if file.filename.lower().endswith(".zip"):
            zip_path = job_dir / file.filename
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            zip_path.unlink()
        else:
            file_loc = raw_dir / file.filename
            with open(file_loc, "wb") as f:
                shutil.copyfileobj(file.file, f)

    background_tasks.add_task(execute_calibrated_turntable_pipeline, job_id, rotation_step, expected_frames)
    return {"job_id": job_id, "frames_count": len(files), "message": "Calibrated Turntable 3D Scan initiated."}




@router.post("/api/scan/upload")
async def create_multiview_scan(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Unconstrained Multi-View Sequence Endpoint.
    Accepts loose sequence images or a .zip archive, then runs AI-depth turntable
    fusion with automatic COLMAP + Poisson fallback.
    """
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "step": f"Uploaded {len(files)} Sequence Frames",
        "logs": [f"Received {len(files)} file(s) for Multi-View reconstruction. Job queued."],
        "error": None
    }

    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)

    for file in files:
        if file.filename.lower().endswith(".zip"):
            zip_path = job_dir / file.filename
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            zip_path.unlink()
        else:
            file_loc = raw_dir / file.filename
            with open(file_loc, "wb") as f:
                shutil.copyfileobj(file.file, f)

    frame_count = len(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpeg")))
    if frame_count == 0:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = "No valid images found in upload."
        raise HTTPException(status_code=400, detail="No valid images (.jpg/.jpeg/.png) found in upload.")

    JOBS[job_id]["logs"].append(f"{frame_count} image frames staged for reconstruction.")
    background_tasks.add_task(execute_3d_scan_pipeline, job_id)
    return {"job_id": job_id, "frames_count": frame_count, "message": "Multi-View 3D reconstruction initiated."}


@router.post("/api/iot/scan")
async def iot_camera_scan_trigger(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    mode: str = "ai_depth"  # 'ai_depth', 'single', or 'turntable'
):
    """
    IoT API Endpoint: Receives frame streams from IoT cameras (ESP32-CAM, RealSense D455, Raspberry Pi)
    and executes AI 3D model generation.
    """
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "step": f"IoT Camera Ingestion ({len(files)} frames)",
        "logs": [f"Received {len(files)} IoT frame(s). Mode: {mode}"],
        "error": None
    }

    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)
    saved_files = []
    for file in files:
        file_loc = raw_dir / file.filename
        with open(file_loc, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_files.append(file.filename)

    if mode == "ai_depth" and saved_files:
        background_tasks.add_task(execute_ai_depth_3d_pipeline, job_id, saved_files[0])
    elif mode == "single" and saved_files:
        background_tasks.add_task(execute_single_image_3d_pipeline, job_id, saved_files[0])
    else:
        background_tasks.add_task(execute_3d_scan_pipeline, job_id)

    return {"job_id": job_id, "mode": mode, "status": "queued", "message": "IoT trigger successful."}


def _saved_generation_quality(out_dir):
    import json
    path = Path(out_dir) / 'quality_report.json'
    if path.is_file():
        try:
            report = json.loads(path.read_text(encoding='utf-8'))
            quality = report.get('quality', {})
            if quality.get('metrics', {}).get('coordinate_space') == 'normalized':
                return quality
        except (OSError, ValueError, AttributeError):
            pass
    return None


@router.get("/api/scan/topology/{job_id}")
def get_mesh_topology_report(job_id: str):
    """
    Returns 3D Printing Topology Inspector metrics (watertightness, faces, volume, printability score).
    """
    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)
    final_stl = out_dir / "final_model.stl"

    if not final_stl.exists():
        raise HTTPException(status_code=404, detail=f"3D model for job '{job_id}' not found on disk.")

    try:
        saved = _saved_generation_quality(out_dir)
        if saved:
            metrics = saved['metrics']
            dims = metrics['bounding_box_units']
            return {'vertices': metrics['vertices'], 'faces': metrics['faces'],
                'is_watertight': metrics['watertight'], 'is_winding_consistent': metrics['winding_consistent'],
                'coordinate_space': 'normalized', 'scale_calibrated': False,
                'dimensions_units': {'x': dims[0], 'y': dims[1], 'z': dims[2]},
                'volume_units3': metrics.get('volume_units3')}
        report = analyze_mesh_topology(str(final_stl))
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze mesh topology: {e}")


@router.get("/api/scan/dimensions/{job_id}")
def get_mesh_dimensions_report(job_id: str):
    """
    Returns exact physical Length, Width, Height, and Volume calculated via Oriented Bounding Box (OBB).
    """
    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)
    final_stl = out_dir / "final_model.stl"

    if not final_stl.exists():
        raise HTTPException(status_code=404, detail=f"3D model for job '{job_id}' not found on disk.")

    try:
        report = auto_measure_mesh(str(final_stl))
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate OBB dimensions: {e}")



@router.get('/api/scan/report/{job_id}')
def get_generation_summary(job_id: str):
    """Public generation provenance and caveats, without local paths or logs."""
    import re,json
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',job_id):
        raise HTTPException(400,'Invalid model identifier.')
    path=BASE_DIR/'jobs'/job_id/'output'/'quality_report.json'
    if not path.is_file():raise HTTPException(404,'Saved generation details are unavailable.')
    try:
        report=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(report,dict):raise ValueError('Expected a report object')
    except (ValueError,OSError) as error:
        raise HTTPException(503,'Saved generation details could not be read.') from error
    quality=report.get('quality') or {}
    metrics=quality.get('metrics') or {}
    source=report.get('source_detail') or report.get('appearance_trial') or report.get('detail_transfer') or {}
    warnings=[]
    for group in (report.get('warnings',[]),quality.get('warnings',[]),quality.get('errors',[])):
        if isinstance(group,list):
            warnings.extend(text for text in group if isinstance(text,str))
    engine=report.get('engine_used')
    label=report.get('engine_label') or SINGLE_IMAGE_ENGINES.get(engine,{}).get('label')
    return {
        'job_id':job_id,'engine_requested':report.get('engine_requested'),'engine_used':engine,
        'engine_label':label,'degraded':report.get('degraded') is True,
        'experimental':any(report.get(key) is True for key in ('experimental','benchmark_only')),
        'appearance_only':report.get('appearance_only') is True or source.get('appearance_only') is True,
        'mesh_passed':quality.get('passed') is True,'mesh_status':quality.get('status','NOT CHECKED'),
        'relative_scale':metrics.get('coordinate_space')=='normalized',
        'fine_geometry_verified':False,'warnings':list(dict.fromkeys(warnings)),
    }

@router.get("/api/scan/quality/{job_id}")
def get_scan_quality_report(job_id: str):
    """
    Returns 3DINVENZA Scan Quality Score (Coverage, Calibration, Density, Scale Confidence, Diagnostics).
    """
    if job_id in JOBS and isinstance(JOBS[job_id].get("quality"), dict):
        return JOBS[job_id]["quality"]

    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)
    final_stl = out_dir / "final_model.stl"

    if not final_stl.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    saved = _saved_generation_quality(out_dir)
    if saved:
        return saved

    try:
        mesh = trimesh.load(str(final_stl), force='mesh')
        calib = load_calibration()
        quality = evaluate_scan_quality(
            images_count=24,
            expected_images=24,
            is_camera_calibrated=calib.get("is_calibrated", False),
            rms_reprojection=calib.get("rms_reprojection_error", 0.45),
            point_cloud_size=len(mesh.vertices),
            mesh_faces=len(mesh.faces),
            is_watertight=bool(mesh.is_watertight)
        )
        return quality
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate scan quality report: {e}")


@router.get("/api/calibration/camera")
async def get_camera_calibration():
    """Returns the current camera calibration profile."""
    return load_calibration()


@router.post("/api/calibration/camera")
async def calibrate_camera_endpoint(
    files: List[UploadFile] = File(...),
    columns: int = 9,
    rows: int = 6,
    square_size_mm: float = 25.0
):
    """Calibrates camera intrinsics from uploaded checkerboard calibration photos."""
    temp_dir = BASE_DIR / "calibration" / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for f in files:
        fpath = temp_dir / f.filename
        with open(fpath, "wb") as buff:
            shutil.copyfileobj(f.file, buff)
        saved_paths.append(fpath)

    try:
        report = calibrate_camera_from_checkerboard(
            image_paths=saved_paths,
            pattern_size=(columns, rows),
            square_size_mm=square_size_mm
        )
        for p in saved_paths:
            try: p.unlink()
            except Exception: pass
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calibration error: {e}")


@router.get("/api/scan/latest")
async def get_latest_scan():
    """Returns the most recent completed 3D scan job ID."""
    jobs_dir = BASE_DIR / "jobs"
    if not jobs_dir.exists():
        raise HTTPException(status_code=404, detail="No scans found.")

    completed = []
    for j in jobs_dir.glob("*"):
        if j.is_dir() and (j / "output" / "final_model.stl").exists():
            saved=j/'job_status.json'
            if saved.exists():
                try:
                    import json
                    if json.loads(saved.read_text(encoding='utf-8')).get('status')!='completed':continue
                except (ValueError,OSError):continue
            completed.append((j.stat().st_mtime, j.name))

    if not completed:
        raise HTTPException(status_code=404, detail="No completed 3D scans available.")

    completed.sort(reverse=True)
    latest_id = completed[0][1]
    return {"job_id": latest_id, "status": "completed"}


@router.get("/api/scan/status/{job_id}")
async def get_scan_status(job_id: str):
    if job_id in JOBS:
        import time
        job=dict(JOBS[job_id])
        if job.get('started_at'):job['elapsed_seconds']=max(0,(job.get('finished_at') or time.time())-job['started_at'])
        return job
    import re,json
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',job_id):raise HTTPException(400,'Invalid job identifier.')
    saved=BASE_DIR/'jobs'/job_id/'job_status.json'
    if saved.is_file():
        try:job=json.loads(saved.read_text(encoding='utf-8'))
        except (ValueError,OSError):raise HTTPException(503,'Saved job status could not be read.')
        if job.get('status') in ('queued','processing'):
            job.update(status='failed',error='The server restarted before generation completed. Please retry.',step='Interrupted')
        return job
    
    # Fallback to checking disk state
    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)
    if (out_dir / "final_model.stl").exists():
        return {
            "job_id": job_id,
            "status": "completed",
            "progress": 100,
            "step": "Completed",
            "stl_path": str(out_dir / "final_model.stl"),
            "glb_path": str(out_dir / "final_model.glb"),
            "logs": ["Loaded from disk."]
        }
    raise HTTPException(status_code=404, detail="Job not found")


@router.get("/api/scan/download/{job_id}/{fmt}")
async def download_scan_result(job_id: str, fmt: str):
    fmt = fmt.lower()
    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)

    if fmt == "stl":
        file_path = out_dir / "final_model.stl"
        media_type = "model/stl"
    elif fmt == "glb":
        file_path = out_dir / "final_model.glb"
        media_type = "model/gltf-binary"
    elif fmt == "ply":
        file_path = out_dir/"final_model.ply" if (out_dir/"final_model.ply").is_file() else workspace_dir/"sparse_cloud.ply"
        media_type = "application/x-ply"
    elif fmt == "obj":
        file_path = out_dir / "final_model.obj"
        media_type = "model/obj"
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Choose stl, glb, ply, or obj.")

    if not file_path.exists():
        # If GLB was requested but only STL exists, or vice versa
        if fmt == "glb" and (out_dir / "final_model.stl").exists():
            import trimesh as _tri
            _m = _tri.load_mesh(str(out_dir / "final_model.stl"))
            _m.export(str(out_dir / "final_model.glb"))
            file_path = out_dir / "final_model.glb"
        elif fmt == "stl" and (out_dir / "final_model.glb").exists():
            import trimesh as _tri
            _m = _tri.load_mesh(str(out_dir / "final_model.glb"))
            _m.export(str(out_dir / "final_model.stl"))
            file_path = out_dir / "final_model.stl"
        else:
            raise HTTPException(status_code=404, detail=f"File in format '{fmt}' not found on disk.")

    return FileResponse(path=str(file_path), filename=f"3d_scan_{job_id}.{fmt}", media_type=media_type)


# Direct sync endpoint for prompt backward compatibility
@router.post("/scan")
async def create_scan_legacy(files: List[UploadFile] = File(...)):
    raw_dir = BASE_DIR / "raw"
    proc_dir = BASE_DIR / "processed"
    workspace_dir = BASE_DIR / "workspace"
    out_dir = BASE_DIR / "output"
    for d in [raw_dir, proc_dir, workspace_dir, out_dir]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*"):
            if f.is_file(): f.unlink()

    for file in files:
        file_location = raw_dir / file.filename
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)

    try:
        process_images(str(raw_dir), str(proc_dir))
        ply_path = run_colmap_reconstruction(str(proc_dir), str(workspace_dir))
        output_stl = out_dir / "final_model.stl"
        pointcloud_to_mesh(str(ply_path), str(output_stl))
        refine_mesh(str(output_stl), str(output_stl))

        return FileResponse(
            path=str(output_stl),
            filename="scanned_object.stl",
            media_type="model/stl"
        )
    except Exception as e:
        return {"error": str(e)}


# =========================================================================
#  Kaggle Dataset & AI Neural Network Training Endpoints
# =========================================================================

@router.get("/api/dataset/inventory")
def get_datasets_inventory_endpoint(detailed: bool=False, limit: int=Query(50,ge=1,le=200)):
    from pipeline.training_data import training_readiness
    root=BASE_DIR/'datasets'
    paths=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.stl','.obj','.glb','.gltf','.ply','.off'})
    inventory=[]
    for path in paths[:limit]:
        entry={'filename':path.name,'filepath':str(path),'size_kb':round(path.stat().st_size/1024,2),'format':path.suffix[1:].upper(),'topology_checked':False}
        if detailed:
            try:
                mesh=trimesh.load(str(path),force='mesh')
                entry.update(vertices=len(mesh.vertices),faces=len(mesh.faces),is_watertight=bool(mesh.is_watertight),topology_checked=True)
            except Exception as error:entry['error']=str(error)
        inventory.append(entry)
    return {'total_3d_models':len(paths),'processed_npz_count':len(list((root/'processed').glob('*.npz'))),
            'categories':DATASET_SPECS,'models':inventory,'returned_models':len(inventory),'readiness':training_readiness(root)}


def _async_dataset_download_worker():
    try:
        run_full_dataset_ingestion()
    except Exception as e:
        print(f"Background dataset ingestion error: {e}")


@router.post("/api/dataset/download-kaggle")
async def trigger_kaggle_dataset_download(background_tasks: BackgroundTasks):
    """
    Triggers ingestion of Kaggle 3D benchmark datasets (ShapeNetPart, Objaverse-XL, OB3D, ReTexture 3D).
    """
    background_tasks.add_task(_async_dataset_download_worker)
    return {
        "status": "started",
        "message": "Kaggle & Open 3D Foundation Dataset Ingestion started in background."
    }


def _async_objaverse_worker(count: int, format_str: str):
    try:
        from pipeline.download_objaverse import download_objaverse_dataset
        from pipeline.download_kaggle_datasets import process_all_datasets_into_npz
        download_objaverse_dataset(count=count, target_format=format_str)
        process_all_datasets_into_npz()
    except Exception as e:
        print(f"Background Objaverse download error: {e}")


@router.post("/api/dataset/download-objaverse")
async def trigger_objaverse_download(
    background_tasks: BackgroundTasks,
    count: int = 10,
    format: str = "stl"
):
    """
    Triggers download and ingestion of official Objaverse 3D models into training pipeline.
    """
    background_tasks.add_task(_async_objaverse_worker, count, format)
    return {
        "status": "started",
        "count": count,
        "format": format,
        "message": f"Objaverse download of {count} models started in background."
    }


def _async_train_worker(epochs: int, batch_size: int, lr: float):
    try:
        train_custom_3d_ai_model(epochs=epochs, batch_size=batch_size, lr=lr)
    except Exception as e:
        print(f"Background training error: {e}")


@router.post("/api/train/start")
async def trigger_ai_training(
    background_tasks: BackgroundTasks,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 0.001
):
    """
    Triggers High-Precision PyTorch 3D AI Neural Network training on all datasets.
    """
    if TRAINING_STATE["status"] == "training":
        return {
            "status": "already_running",
            "message": "AI Training is already actively running.",
            "current_epoch": TRAINING_STATE["current_epoch"],
            "total_epochs": TRAINING_STATE["total_epochs"]
        }

    background_tasks.add_task(_async_train_worker, epochs, batch_size, lr)
    return {
        "status": "started",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "message": f"PyTorch 3D AI training started for {epochs} epochs."
    }


@router.get("/api/train/status")
async def get_ai_training_status():
    """
    Returns real-time PyTorch AI training state, loss breakdown, and convergence metrics.
    """
    state_copy = dict(TRAINING_STATE)
    if state_copy.get("best_val_loss") is not None and (math.isinf(state_copy["best_val_loss"]) or math.isnan(state_copy["best_val_loss"])):
        state_copy["best_val_loss"] = None
    return state_copy

# =========================================================================
#  ESP32 Hardware Controller Endpoints
# =========================================================================

try:
    from hardware.turntable_controller import turntable
except ImportError:
    turntable = None

@router.post("/api/hardware/rotate")
async def hardware_rotate(angle: float = 360.0):
    if not turntable:
        raise HTTPException(status_code=500, detail="Hardware controller not available.")
    try:
        # Run in a background thread to not block the event loop
        await asyncio.to_thread(turntable.rotate, angle, True)
        return {"status": "success", "message": f"Rotated {angle} degrees."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/hardware/stop")
async def hardware_stop():
    if not turntable:
        raise HTTPException(status_code=500, detail="Hardware controller not available.")
    try:
        turntable.stop()
        return {"status": "success", "message": "Turntable stopped."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/hardware/status")
async def hardware_status():
    if not turntable:
        raise HTTPException(status_code=500, detail="Hardware controller not available.")
    try:
        status = turntable.get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
#  Model Database Helper & Gallery Endpoints
# =========================================================================

def _save_job_to_db(
    job_id: str,
    engine: str = "",
    engine_label: str = "",
    mode: str = "single_image",
    object_type: str = "organic",
    quality: str = "high",
):
    """Utility: save a completed job into the persistent model database."""
    job = JOBS.get(job_id, {})
    job_dir, raw_dir, proc_dir, workspace_dir, out_dir = get_job_dirs(job_id)

    stl_path = job.get("stl_path", str(out_dir / "final_model.stl"))
    glb_path = job.get("glb_path", str(out_dir / "final_model.glb"))
    ply_path = job.get("ply_path", str(workspace_dir / "sparse_cloud.ply"))

    # Gather mesh stats if available
    vertices = 0
    faces = 0
    length_mm = 0.0
    width_mm = 0.0
    height_mm = 0.0
    volume_mm3 = 0.0
    is_watertight = False

    try:
        mesh_path = stl_path if os.path.isfile(stl_path) else glb_path
        if os.path.isfile(mesh_path):
            mesh = trimesh.load(mesh_path, force='mesh')
            vertices = len(mesh.vertices)
            faces = len(mesh.faces)
            is_watertight = bool(mesh.is_watertight)
            if is_watertight:
                volume_mm3 = round(abs(mesh.volume), 2)
            bbox = mesh.bounding_box_oriented
            extents = sorted(bbox.extents, reverse=True)
            length_mm = round(extents[0], 2) if len(extents) > 0 else 0
            width_mm = round(extents[1], 2) if len(extents) > 1 else 0
            height_mm = round(extents[2], 2) if len(extents) > 2 else 0
    except Exception as e:
        print(f"[DB] Mesh stats extraction notice for {job_id}: {e}")

    # Overwrite with job-level dims if available
    dims = job.get("dimensions", {})
    if dims.get("length_mm"):
        length_mm = dims["length_mm"]
        width_mm = dims.get("width_mm", width_mm)
        height_mm = dims.get("height_mm", height_mm)

    try:
        save_model(
            job_id=job_id,
            name=f"Scan {job_id}",
            engine=engine,
            engine_label=engine_label,
            mode=mode,
            object_type=job.get("object_type", object_type),
            quality=job.get("quality", quality) if isinstance(job.get("quality"), str) else quality,
            stl_path=stl_path,
            glb_path=glb_path,
            ply_path=ply_path,
            vertices=vertices,
            faces=faces,
            length_mm=length_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            volume_mm3=volume_mm3,
            is_watertight=is_watertight,
        )
        print(f"[DB] Model saved: job_id={job_id}, engine={engine}, verts={vertices}, faces={faces}")
    except Exception as e:
        print(f"[DB] Failed to save model {job_id}: {e}")


@router.get("/api/models")
async def list_models(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    """Returns all saved 3D models for the gallery, newest first."""
    models = get_all_models(limit=limit, offset=offset)
    total = get_model_count()
    return {"total": total, "models": models}


@router.get("/api/models/{model_id}")
async def get_model_detail(model_id: int):
    """Returns details for a single model."""
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.delete("/api/models/{model_id}")
async def delete_model_endpoint(model_id: int):
    """Deletes a model from the database."""
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    delete_model(model_id)
    return {"status": "deleted", "model_id": model_id}


@router.post("/api/models/scan-existing")
async def scan_existing_jobs():
    """
    Scans the data/jobs directory for existing completed jobs
    and adds them to the database if not already present.
    """
    jobs_dir = BASE_DIR / "jobs"
    if not jobs_dir.exists():
        return {"scanned": 0, "added": 0}

    scanned = 0
    added = 0
    for job_folder in sorted(jobs_dir.iterdir()):
        if not job_folder.is_dir():
            continue
        job_id = job_folder.name
        scanned += 1

        # Skip if already in DB
        if get_model_by_job(job_id):
            continue

        out_dir = job_folder / "output"
        stl = out_dir / "final_model.stl"
        glb = out_dir / "final_model.glb"

        if stl.exists() or glb.exists():
            _save_job_to_db(job_id, engine="unknown", engine_label="Imported", mode="unknown")
            added += 1

    return {"scanned": scanned, "added": added}


# =========================================================================
#  Authentication & Profile Routes
# =========================================================================

# Simple in-memory session store (token -> user_dict)
ACTIVE_SESSIONS = {}

def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1]
    user = ACTIVE_SESSIONS.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class SettingsUpdateRequest(BaseModel):
    theme: str
    default_engine: str
    notifications: int

@router.post("/api/auth/register")
async def register(req: RegisterRequest):
    user = create_user(req.username, req.email, req.password)
    if not user:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    return {"message": "Registration successful"}

@router.post("/api/auth/login")
async def login(req: LoginRequest):
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    token = secrets.token_hex(32)
    ACTIVE_SESSIONS[token] = {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "is_admin": user.get("is_admin", 0)
    }
    return {"token": token, "user": ACTIVE_SESSIONS[token]}

@router.get("/api/user/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    settings = get_user_settings(current_user["id"])
    return {
        "user": current_user,
        "settings": settings
    }

@router.post("/api/user/settings")
async def update_settings(req: SettingsUpdateRequest, current_user: dict = Depends(get_current_user)):
    success = update_user_settings(current_user["id"], req.theme, req.default_engine, req.notifications)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update settings")
    return {"message": "Settings updated"}

@router.post("/api/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        if token in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[token]
    return {"message": "Logged out"}

# ==============================================================================
# ADMIN DASHBOARD
# ==============================================================================

@router.get("/api/admin/users")
async def admin_get_users(current_user: dict = Depends(get_current_user)):
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Forbidden: Admins only")
    users = get_all_users_db()
    return {"users": users}

@router.get("/api/admin/stats")
async def admin_get_stats(current_user: dict = Depends(get_current_user)):
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Forbidden: Admins only")
    total_users = len(get_all_users_db())
    total_models = get_model_count()
    return {
        "total_users": total_users,
        "total_models": total_models
    }


@router.get('/api/scan/source-preview/{job_id}')
def source_preview(job_id: str):
    """Return only the isolated foreground produced by this scan's preprocessing."""
    import re
    if not re.fullmatch(r'[a-f0-9]{8,32}',job_id):
        raise HTTPException(400,'Invalid scan identifier')
    output=BASE_DIR/'jobs'/job_id/'output'
    candidates=list((output/'prep').glob('*_foreground.png'))
    candidates.extend(output.glob('temp_ai_*_rgba.png'))
    if not candidates:
        raise HTTPException(404,'Foreground preview is not ready')
    from fastapi.responses import Response
    from PIL import Image
    from io import BytesIO
    try:
        payload=candidates[0].read_bytes()
        Image.open(BytesIO(payload)).verify()
    except (OSError,ValueError,SyntaxError):
        raise HTTPException(404,'Foreground preview is still being written')
    return Response(payload,media_type='image/png',headers={'Cache-Control':'no-store'})


@router.get('/api/dataset/readiness')
def dataset_readiness():
    from pipeline.training_data import training_readiness
    return training_readiness(BASE_DIR/'datasets')

@router.get('/api/scan/point-cloud/{job_id}')
def reconstructed_point_cloud(job_id: str):
    import re
    if not re.fullmatch(r'[a-f0-9]{8,32}',job_id):raise HTTPException(400,'Invalid scan identifier')
    from pipeline.point_cloud import get_cloud
    try:return get_cloud(BASE_DIR/'jobs'/job_id/'output')
    except Exception:raise HTTPException(404,'3D points are still being reconstructed')
