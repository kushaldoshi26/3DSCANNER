"""
Engine Manager — unified single-image 3D engine selection & orchestration.

Design goals (per 3DINVENZA upgrade spec):
  - Priority chain: TRELLIS -> Hunyuan3D -> TripoSR (fast fallback).
  - Detect available GPU / native packages and select the best available engine.
  - Automatically fall back down the chain when a requested engine is missing
    or fails at runtime, returning a clear `degraded` flag.
  - Expose generation progress to callers (dashboard / API).
  - Never synthesize a fake mesh; report a clear error instead.

The manager wraps the existing, tested inference modules via thin
`EngineBase` subclasses so no working functionality is removed.
"""

from typing import Callable, Dict, Any, List, Optional

from pipeline.ai.engine_base import EngineBase, ProgressCallback
from pipeline.ai.trellis_engine import TrellisEngine
from pipeline.ai.hunyuan3d_engine import Hunyuan3DEngine
from pipeline.ai.tripo_engine import TripoEngine

# Engine registry. `priority` defines the default selection order.
_ENGINE_CLASSES = [TrellisEngine, Hunyuan3DEngine, TripoEngine]
_PRIORITY_ORDER = ["trellis", "hunyuan3d", "tripo"]

_ENGINES: Dict[str, EngineBase] = {}
for _cls in _ENGINE_CLASSES:
    _inst = _cls()
    _ENGINES[_inst.key] = _inst


def list_engines() -> List[Dict[str, Any]]:
    """Returns metadata for every engine, including live availability."""
    out = []
    for key in _PRIORITY_ORDER:
        eng = _ENGINES.get(key)
        if eng is None:
            continue
        out.append({
            "key": eng.key,
            "label": eng.label,
            "detail": eng.detail,
            "available": eng.is_available(),
            "fallbacks": eng.fallbacks,
        })
    return out


def get_engine(key: str) -> Optional[EngineBase]:
    return _ENGINES.get(key)


def resolve_engine(requested: Optional[str]) -> str:
    """
    Normalizes a user-supplied engine key (loose spellings supported).
    If the requested engine is unknown, returns the best currently-available
    engine in priority order; if none available, returns the highest-priority
    key so the caller gets a clear availability error rather than a silent pass.
    """
    import re
    aliases = {
        "trellis": "trellis",
        "slat": "trellis",
        "hunyuan": "hunyuan3d",
        "hunyuan3d": "hunyuan3d",
        "hunyuan3d2": "hunyuan3d",
        "hy3d": "hunyuan3d",
        "triposr": "tripo",
        "tripo": "tripo",
        "tripo3d": "tripo",
        "lrm": "tripo",
    }
    norm = re.sub(r"[^a-z0-9]", "", (requested or "").lower())
    if norm in aliases:
        return aliases[norm]

    # Unknown key: pick best available in priority order.
    for key in _PRIORITY_ORDER:
        eng = _ENGINES.get(key)
        if eng is not None and eng.is_available():
            return key
    return _PRIORITY_ORDER[0]


def select_best_available(prefer: Optional[str] = None) -> Optional[EngineBase]:
    """
    Builds the engine selection chain (requested first, then its fallbacks,
    then remaining priority engines) and returns the first *available* one.
    """
    chain: List[str] = []
    if prefer:
        chain.append(prefer)
    eng = _ENGINES.get(prefer) if prefer else None
    if eng:
        chain.extend(eng.fallbacks)
    for key in _PRIORITY_ORDER:
        if key not in chain:
            chain.append(key)

    for key in chain:
        candidate = _ENGINES.get(key)
        if candidate is not None and candidate.is_available():
            return candidate
    return None


# Quality presets map a named preset to cleanup/refinement settings.
# FAST: light; HIGH: balanced face-preserving; ULTRA: max quality.
QUALITY_PRESETS = {
    "fast": {
        "taubin_iterations": 2,
        "face_strength": 0.15,
        "body_strength": 0.4,
        "refine": False,
        "export_obj": False,
        "printable_default": False,
    },
    "high": {
        "taubin_iterations": 3,
        "face_strength": 0.25,
        "body_strength": 0.6,
        "refine": True,
        "export_obj": True,
        "printable_default": False,
    },
    "ultra": {
        "taubin_iterations": 5,
        "face_strength": 0.3,
        "body_strength": 0.7,
        "refine": True,
        "export_obj": True,
        "printable_default": False,
    },
}


def run_single_image(
    engine: Optional[str],
    image_path: str,
    output_stl: str,
    output_glb: str,
    *,
    object_type: str = "organic",
    texture: bool = True,
    quality: str = "high",
    progress_cb: Optional[ProgressCallback] = None,
    refine: bool = True,
    printable: bool = False,
    export_obj: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Runs single-image 3D generation through the engine manager, then applies
    the structured cleanup, optional bounded refinement, texture, QC, and
    multi-format export pipeline.

    Args:
        engine: requested engine key (loose spelling accepted); None = auto.
        image_path: input 2D image (original reference; used for texture).
        output_stl / output_glb: primary output paths (STL + GLB).
        texture: request painted/vertex texture.
        quality: "fast" | "high" | "ultra".
        progress_cb: optional (stage, percent, message) callback.
        refine: enable bounded AI detail refinement.
        printable: if True, optionally close an open bottom (bust base).
        export_obj: also export OBJ + MTL.
        debug: save intermediate artifacts (01_input ... quality_report.json).

    Returns dict including engine selection, mesh/texture/QC status, and files.
    """
    from pipeline.cleanup_mesh import cleanup_generated_mesh
    from pipeline.texture_projector import project_texture_to_mesh
    from pipeline.mesh_quality import validate_mesh
    from pipeline.ai_refine import refine_mesh, MAX_DISPLACEMENT_RATIO

    logs: List[str] = []
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["high"])
    
    # If hard_surface is requested, strongly prefer Trellis as it produces 
    # much cleaner flat surfaces than TripoSR's NeRF-based approach.
    if object_type == "hard_surface":
        engine = "trellis"
        
    requested_key = resolve_engine(engine)

    def _progress(stage, percent, message):
        if progress_cb:
            progress_cb(stage, percent, message)
        logs.append(f"[{stage}] {message}")

    _progress("Preparing image", 10, "Validating input and foreground extraction...")

    # --- High-quality input preprocessing ---
    from pathlib import Path as _Path
    from pipeline.preprocess_input import preprocess_for_generation
    out_dir = _Path(output_stl).parent
    prep = preprocess_for_generation(image_path, str(out_dir / "prep"), padding_ratio=0.12)
    generator_input = prep["padded_rgba"]
    original_rgb = prep["original_rgb"]

    selected = select_best_available(prefer=requested_key)
    if selected is None:
        raise RuntimeError(
            "No single-image 3D engine is available. "
            "Install one of: microsoft/TRELLIS, tencent/Hunyuan3D-2, or stabilityai/TripoSR. "
            "AI Depth preview (pipeline.ai_depth_3d) remains available as a non-360 fallback."
        )

    # Build the full chain starting from whatever was selected, preserving
    # the global priority order so we always try the best engines first.
    chain: List[str] = []
    if selected.key not in chain:
        chain.append(selected.key)
    chain.extend(selected.fallbacks)
    for key in _PRIORITY_ORDER:
        if key not in chain:
            chain.append(key)

    # Raw AI mesh is saved so we can compare raw vs cleaned (debug / tuning).
    raw_stl = str(out_dir / "raw_ai_mesh.stl")
    raw_glb = str(out_dir / "raw_ai_mesh.glb")

    last_err: Optional[Exception] = None
    for candidate_key in chain:
        candidate = _ENGINES.get(candidate_key)
        if candidate is None or not candidate.is_available():
            logs.append(f"Skipping {candidate_key}: package not available.")
            continue
        try:
            if candidate_key != requested_key:
                logs.append(f"TRELLIS unavailable -> using {candidate.label}" if requested_key == "trellis" else f"Falling back to {candidate.label}...")
            _progress("Generating base geometry", 35, f"Running {candidate.label}...")
            candidate.generate(
                generator_input,
                raw_stl,
                raw_glb,
                texture=False,  # we texture from the original image ourselves
                quality=quality,
                progress_cb=lambda s, p, m: _progress("Generating base geometry", 35 + int(p * 0.2), m),
            )

            # --- Structured cleanup ---
            _progress("Cleaning mesh", 60, "Removing artifacts & face-aware smoothing...")
            clean_stl = output_stl
            cleanup_generated_mesh(
                raw_stl,
                clean_stl,
                image_path=original_rgb,
                object_type=object_type,
                smoothing_enabled=True,
                taubin_iterations=preset["taubin_iterations"],
                face_strength=preset["face_strength"],
                body_strength=preset["body_strength"],
                debug=debug,
            )

            # --- Optional bounded AI refinement (relative ratio for generative) ---
            _progress("Preserving facial detail", 72, "Optional bounded detail refinement...")
            if refine and preset["refine"]:
                refined_stl = str(out_dir / "refined_mesh.stl")
                
                # IMPORTANT: Generative models (Trellis, Tripo, Hunyuan) already output perfect 
                # high-frequency 3D details. Applying the 2.5D neural depth network over them 
                # destroys their topology, causing rippled/bumpy artifacts. We only enable 
                # displacement for raw 2.5D depth extrusions.
                should_displace = (candidate_key in ["ai_depth", "custom_3d"])
                
                refine_mesh(
                    str(clean_stl),
                    refined_stl,
                    reference_images=[original_rgb],
                    enable_ai_displacement=should_displace,
                    max_displacement_mm=MAX_DISPLACEMENT_RATIO * 1000.0,  # relative ratio scaled
                    smoothing_iterations=0,
                )
                clean_stl = refined_stl

            # --- Texture from ORIGINAL high-res image ---
            _progress("Generating texture", 82, "UV unwrap + high-resolution texture projection...")
            textured_glb = output_glb
            if texture:
                project_texture_to_mesh(str(clean_stl), original_rgb, textured_glb, use_uv_atlas=True)
            else:
                _copy_mesh(str(clean_stl), textured_glb)

            # --- Multi-format export ---
            _progress("Preparing exports", 90, "Exporting STL / GLB / OBJ...")
            files = {
                "stl": str(output_stl),
                "glb": str(output_glb),
            }
            if export_obj or preset["export_obj"]:
                obj_path = str(_Path(output_stl).with_suffix(".obj"))
                _copy_mesh(str(clean_stl), obj_path)
                files["obj"] = obj_path
            if printable:
                files["stl_printable"] = _make_printable_bust(str(clean_stl), str(out_dir / "printable.stl"))

            # --- QC ---
            _progress("Mesh validation", 95, "Running mesh quality control...")
            qc = validate_mesh(str(output_stl), require_watertight=False)
            if debug:
                import json as _json
                with open(str(out_dir / "quality_report.json"), "w") as _f:
                    _json.dump(qc, _f, indent=2)

            return {
                "engine_requested": requested_key,
                "engine_used": candidate.key,
                "engine_label": candidate.label,
                "degraded": candidate.key != requested_key,
                "mesh_path": str(clean_stl),
                "glb_path": str(output_glb),
                "texture_path": str(output_glb) if texture else None,
                "quality": qc,
                "files": files,
                "warnings": qc.get("warnings", []),
                "logs": logs,
            }
        except Exception as err:
            last_err = err
            logs.append(f"Notice: {candidate.label} failed ({err}).")

    raise RuntimeError(
        f"All single-image engines failed for '{requested_key}'. "
        f"Last error: {last_err}"
    )


def _copy_mesh(src: str, dst: str) -> None:
    import shutil
    m = trimesh_load_safe(src)
    m.export(dst)


def trimesh_load_safe(p: str):
    import trimesh
    m = trimesh.load(str(p), force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(m.dump())
    return m


def _make_printable_bust(stl: str, out_path: str) -> str:
    """Closes an open bottom with a flat bust base (explicit printable option)."""
    m = trimesh_load_safe(stl)
    try:
        if not m.is_watertight:
            # Cut at lower 15% and cap with a flat bottom.
            extents = m.extents
            z_min = m.bounds[0][2]
            cut_z = z_min + extents[2] * 0.12
            section = m.section(plane_origin=[0, 0, cut_z], plane_normal=[0, 0, 1])
            if section is not None:
                poly = section.to_planar()[0]
                cap = trimesh.path.polygons.paths_to_mesh(poly)
                capped = trimesh.util.concatenate([m, cap])
                capped.export(out_path)
                return out_path
    except Exception:
        pass
    _copy_mesh(stl, out_path)
    return out_path
