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


# Neural models are shared, mutable and compete for the same GPU.
from threading import Lock
_GENERATION_LOCK = Lock()

def run_single_image(engine, image_path, output_stl, output_glb, *,
                     object_type="organic", texture=True, quality="high",
                     progress_cb=None, refine=False, printable=False,
                     export_obj=True, debug=False, photo_detail=False):
    if progress_cb:
        progress_cb("Waiting for engine", 1, "Waiting for the generation slot...")
    with _GENERATION_LOCK:
        return _run_single_image(engine, image_path, output_stl, output_glb,
            object_type=object_type, texture=texture, quality=quality,
            progress_cb=progress_cb, refine=refine, printable=printable,
            export_obj=export_obj, debug=debug, photo_detail=photo_detail)

def _run_single_image(engine, image_path, output_stl, output_glb, *,
                      object_type, texture, quality, progress_cb, refine,
                      printable, export_obj, debug, photo_detail=False):
    import json
    import numpy as np
    from pathlib import Path
    from pipeline.cleanup_mesh import cleanup_generated_mesh
    from pipeline.preprocess_input import preprocess_for_generation
    from pipeline.texture_projector import project_texture_to_mesh, has_native_appearance
    from pipeline.mesh_quality import validate_mesh
    logs=[]
    preset=QUALITY_PRESETS.get(quality, QUALITY_PRESETS['high'])
    quality=quality if quality in QUALITY_PRESETS else 'high'
    requested_key=resolve_engine(engine)
    def progress(stage, percent, message):
        logs.append(f"[{stage}] {message}")
        if progress_cb: progress_cb(stage,percent,message)
    selected=select_best_available(prefer=requested_key)
    if selected is None:
        raise RuntimeError("No single-image 3D engine is available. Install TRELLIS, Hunyuan3D or TripoSR.")
    out=Path(output_stl).parent;out.mkdir(parents=True,exist_ok=True)
    progress('Preparing image',10,'Preserving input alpha, orientation and foreground detail...')
    prep=preprocess_for_generation(image_path,str(out/'prep'))
    raw_stl=str(out/'raw_ai_mesh.stl');raw_glb=str(out/'raw_ai_mesh.glb')
    chain=list(dict.fromkeys([selected.key]+selected.fallbacks+_PRIORITY_ORDER))
    failures=[]
    used=None
    engine_metadata={}
    for key in chain:
        candidate=_ENGINES.get(key)
        if candidate is None or not candidate.is_available(): continue
        try:
            # Failed candidates must not leave stale outputs for their successor.
            for path in (raw_stl,raw_glb): Path(path).unlink(missing_ok=True)
            progress('Generating base geometry',35,f'Running {candidate.label} ({quality})...')
            generation_result=candidate.generate(prep['padded_rgba'],raw_stl,raw_glb,
                texture=texture,quality=quality,
                progress_cb=lambda stage,percent,message: progress('Generating base geometry',35+int(percent*.2),message))
            source=raw_glb if Path(raw_glb).is_file() else raw_stl
            raw=trimesh_load_safe(source)
            if raw.is_empty or not np.isfinite(raw.vertices).all():
                raise RuntimeError('Engine returned empty or non-finite geometry.')
            engine_metadata=(generation_result or {}).get('generation',{})
            used=candidate;break
        except Exception as err:
            failures.append(f'{candidate.label}: {err}')
            logs.append(f'Engine failed: {failures[-1]}')
    if used is None: raise RuntimeError('All single-image engines failed. '+'; '.join(failures))
    # Post-processing failures must not rerun an expensive neural engine.
    progress('Cleaning mesh',60,'Removing invalid faces and gently smoothing while retaining native colors...')
    clean=str(out/'cleaned_model.glb')
    cleanup_generated_mesh(source,clean,image_path=prep['original_rgb'],
        object_type=object_type,smoothing_enabled=refine,
        taubin_iterations=preset['taubin_iterations'],
        face_strength=preset['face_strength'],body_strength=preset['body_strength'],debug=debug)
    final=trimesh_load_safe(clean)
    if refine:
        # Generative geometry already has inferred 360-degree structure.
        # A separate monocular depth network must not displace this mesh.
        import trimesh
        trimesh.repair.fill_holes(final)
        final.fix_normals(multibody=True)
    warnings=['Unseen geometry and texture are inferred from a single image.', 'Facial expression, facial depth and accessory accuracy have not been validated against a 3D reference.']
    if texture:
        if not has_native_appearance(final):
            final.export(clean)
            project_texture_to_mesh(clean,prep['padded_rgba'],output_glb,use_uv_atlas=False)
            final=trimesh_load_safe(output_glb)
            warnings.append('Engine had no native colors; source-image projection is approximate, especially on unseen surfaces.')
    else:
        import trimesh
        final.visual=trimesh.visual.ColorVisuals(mesh=final)
    progress('Mesh validation',85,'Checking final geometry before writing all exports...')
    qc=validate_mesh(final,require_watertight=printable,coordinate_space='normalized')
    warnings.extend(qc['warnings'])
    if not qc['passed']:
        raise RuntimeError('Final mesh failed quality checks: '+'; '.join(qc['errors']))
    progress('Preparing exports',92,'Writing matching STL, GLB and OBJ geometry...')
    final.export(str(output_stl))
    import trimesh
    export_scene=trimesh.Scene(final)
    if engine_metadata.get('source_view'):
        export_scene.metadata['source_view']=engine_metadata['source_view']
    export_scene.export(str(output_glb),include_normals=True)
    source_detail=None
    if texture and photo_detail:
        # Keep geometry and native appearance available; enrich visible GLB faces only.
        import shutil
        native_glb=out/'native_model.glb'
        shutil.copy2(output_glb,native_glb)
        try:
            from pipeline.source_detail import enhance_source_detail
            progress('Preserving photo detail',95,'Matching visible source detail; keeping unseen surfaces inferred...')
            enriched=out/'source_detail_model.glb'
            source_detail=enhance_source_detail(native_glb,prep['foreground_rgba'],enriched,allow_textured=photo_detail,align_photo=photo_detail,visibility_size=2048 if photo_detail else 1024)
            if source_detail.get('applied'):
                shutil.copy2(enriched,output_glb)
                warnings.append('Visible GLB surfaces use source-photo projection. This improves appearance, not measured geometry; unseen surfaces remain inferred.')
                if (source_detail.get('back_texture') or {}).get('observed_back') is False:
                    warnings.append('Rear fabric pattern is estimated from a visible clothing patch, not an observed back photograph.')
            else:
                warnings.append('Source detail skipped: '+source_detail.get('reason','alignment unavailable'))
        except Exception as error:
            warnings.append('Source detail unavailable; retained native appearance.')
            logs.append('Source detail: '+str(error))
    files={'stl':str(output_stl),'glb':str(output_glb)}
    import trimesh
    point_path=out/'final_model.ply'
    colours=final.visual.to_color().vertex_colors if final.visual.kind=='texture' else final.visual.vertex_colors
    trimesh.points.PointCloud(final.vertices,colors=colours).export(point_path)
    files['ply']=str(point_path)
    if export_obj or preset['export_obj']:
        obj=str(Path(output_stl).with_suffix('.obj'));final.export(obj);files['obj']=obj
    if printable:
        printable_path=str(out/'printable.stl');final.export(printable_path);files['stl_printable']=printable_path
    if used.key != requested_key:
        warnings.append(f'Requested {requested_key}; generated with {used.label}.')
    report={'engine_requested':requested_key,'engine_used':used.key,
        'engine_label':used.label,'degraded':used.key!=requested_key,
        'mesh_path':str(output_stl),'glb_path':str(output_glb),
        'texture_path':str(output_glb) if texture else None,
        'quality':qc,'files':files,'warnings':warnings,'logs':logs,'source_detail':source_detail,
        'generation':engine_metadata,'experimental':bool(photo_detail and source_detail and source_detail.get('applied'))}
    (out/'quality_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


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
    """Repair only small holes; never cut away a model to claim printability."""
    import trimesh
    mesh=trimesh_load_safe(stl)
    trimesh.repair.fill_holes(mesh)
    mesh.fix_normals(multibody=True)
    if not mesh.is_watertight:
        raise RuntimeError('Mesh is not watertight; additional manual repair is required for printing.')
    mesh.export(out_path)
    return out_path
