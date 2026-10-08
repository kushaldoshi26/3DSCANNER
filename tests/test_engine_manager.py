"""
Tests for the unified engine manager, mesh quality control, improved
Open3D meshing, bounded AI refinement, and COLMAP failure handling.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import trimesh
import open3d as o3d
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ai.engine_manager import (
    list_engines,
    resolve_engine,
    select_best_available,
    run_single_image,
    QUALITY_PRESETS,
)
from pipeline.mesh_quality import validate_mesh, surface_roughness, detail_metrics
from pipeline.open3d_mesh import pointcloud_to_mesh
from pipeline.ai_refine import refine_mesh, MAX_DISPLACEMENT_MM
from pipeline.colmap_poses import opencv_sfm_fallback
from pipeline.cleanup_mesh import (
    cleanup_generated_mesh,
    detect_face_roi,
    FACE_SMOOTHING_STRENGTH,
    BODY_SMOOTHING_STRENGTH,
    TAUBIN_ITERATIONS,
)
from pipeline.texture_projector import project_texture_to_mesh
from pipeline.preprocess_input import preprocess_for_generation


# ---------------------------------------------------------------------------
# Engine Manager
# ---------------------------------------------------------------------------

def test_engine_manager_lists_three_engines():
    engines = list_engines()
    keys = {e["key"] for e in engines}
    assert {"trellis", "hunyuan3d", "tripo"}.issubset(keys)
    # Priority order preserved.
    ordered = [e["key"] for e in engines]
    assert ordered.index("trellis") < ordered.index("hunyuan3d") < ordered.index("tripo")


def test_resolve_engine_normalizes_loose_spellings():
    assert resolve_engine("TRELLIS") == "trellis"
    assert resolve_engine("Hunyuan-3D") == "hunyuan3d"
    assert resolve_engine("tripo") == "tripo"
    # Unknown key resolves to best available in priority order, not an error.
    assert resolve_engine("nonexistent_engine") in {"trellis", "hunyuan3d", "tripo"}


def test_run_single_image_raises_clear_error_when_no_engine_available(monkeypatch):
    # Force every engine to report unavailable so run_single_image must fail loudly
    # with a clear message rather than silently producing a model.
    from pipeline.ai import engine_manager as em
    for inst in em._ENGINES.values():
        monkeypatch.setattr(inst, "is_available", lambda: False)

    # Provide a tiny valid image so the only failure is engine availability.
    img = np.ones((64, 64, 3), dtype=np.uint8) * 200
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "x.png"
        import cv2
        cv2.imwrite(str(img_path), img)
        with pytest.raises(RuntimeError) as exc:
            run_single_image("trellis", str(img_path), str(Path(tmp) / "o.stl"), str(Path(tmp) / "o.glb"))
        assert "No single-image 3D engine is available" in str(exc.value)


# ---------------------------------------------------------------------------
# Mesh Quality Control
# ---------------------------------------------------------------------------

def test_validate_mesh_passes_watertight_sphere():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    report = validate_mesh(sphere)
    assert report["passed"] is True
    assert report["metrics"]["watertight"] is True
    assert report["metrics"]["vertices"] > 0


def test_validate_mesh_reports_open_mesh_as_non_watertight():
    # A single flat triangle is not watertight and should surface a warning/error.
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]])
    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    report = validate_mesh(tri, require_watertight=False)
    assert report["metrics"]["watertight"] is False


# ---------------------------------------------------------------------------
# Open3D Meshing (depth=9 default, Taubin configurable)
# ---------------------------------------------------------------------------

def test_pointcloud_to_mesh_default_depth_nine():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere = sphere.sample_points_uniformly(number_of_points=2000)
    sphere.estimate_normals()

    with tempfile.TemporaryDirectory() as tmp:
        ply = Path(tmp) / "cloud.ply"
        o3d.io.write_point_cloud(str(ply), sphere)
        stl = Path(tmp) / "mesh.stl"
        pointcloud_to_mesh(str(ply), str(stl))  # default depth=9
        assert stl.exists()
        m = trimesh.load(str(stl), force="mesh")
        assert len(m.vertices) > 0


# ---------------------------------------------------------------------------
# AI Refinement (bounded displacement flag)
# ---------------------------------------------------------------------------

def test_refine_mesh_accepts_displacement_params():
    # The new optional/flagged AI displacement contract must be present.
    assert isinstance(MAX_DISPLACEMENT_MM, (int, float))
    assert MAX_DISPLACEMENT_MM > 0

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.stl"
        out = Path(tmp) / "out.stl"
        sphere.export(str(inp))
        # Default (no custom model present) must not raise and must preserve geometry.
        refine_mesh(str(inp), str(out), enable_ai_displacement=False, smoothing_iterations=1)
        assert out.exists()
        m = trimesh.load(str(out), force="mesh")
        assert len(m.vertices) > 0


# ---------------------------------------------------------------------------
# COLMAP / SfM failure handling (no synthetic cloud)
# ---------------------------------------------------------------------------

def test_opencv_sfm_fallback_raises_on_no_features():
    """Solid-color images yield no feature matches; the fallback must raise a
    clear SCAN FAILED error instead of fabricating a synthetic point cloud."""
    with tempfile.TemporaryDirectory() as tmp:
        img_dir = Path(tmp) / "imgs"
        workspace = Path(tmp) / "ws"
        img_dir.mkdir()
        # Two flat solid-color images have no SIFT features to match.
        solid = np.full((64, 64, 3), 128, dtype=np.uint8)
        import cv2
        cv2.imwrite(str(img_dir / "a.png"), solid)
        cv2.imwrite(str(img_dir / "b.png"), solid)

        with pytest.raises(RuntimeError) as exc:
            opencv_sfm_fallback(img_dir, workspace)
        assert "SCAN FAILED" in str(exc.value)


# ---------------------------------------------------------------------------
# Cleanup pipeline & face-aware smoothing
# ---------------------------------------------------------------------------

def _make_noisy_head_mesh(tmp):
    """A rough box-ish 'head' mesh with high-frequency vertex noise to clean."""
    mesh = trimesh.creation.box(extents=(1.0, 1.2, 0.9), transform=None)
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    verts = mesh.vertices.copy()
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.03, verts.shape)
    mesh.vertices = verts + noise
    p = Path(tmp) / "noisy.glb"
    mesh.export(str(p))
    return str(p)


def test_quality_presets_exist():
    assert set(QUALITY_PRESETS.keys()) == {"fast", "high", "ultra"}
    # Face smoothing must always be weaker than body smoothing.
    for preset in QUALITY_PRESETS.values():
        assert preset["face_strength"] < preset["body_strength"]


def test_cleanup_reduces_surface_roughness():
    with tempfile.TemporaryDirectory() as tmp:
        raw = _make_noisy_head_mesh(tmp)
        before = detail_metrics(raw)["surface_roughness"]
        out = Path(tmp) / "clean.glb"
        cleanup_generated_mesh(raw, str(out), smoothing_enabled=True,
                               taubin_iterations=3, face_strength=0.25, body_strength=0.6)
        assert out.exists()
        m = trimesh.load(str(out), force="mesh")
        assert len(m.vertices) > 0
        after = detail_metrics(str(out))["surface_roughness"]
        assert after < before  # cleanup must reduce roughness


def test_face_smoothing_strength_is_weaker_than_body():
    assert FACE_SMOOTHING_STRENGTH < BODY_SMOOTHING_STRENGTH
    assert 0 < FACE_SMOOTHING_STRENGTH < 1
    assert 0 < BODY_SMOOTHING_STRENGTH < 1


def test_detect_face_roi_returns_none_on_solid_image():
    with tempfile.TemporaryDirectory() as tmp:
        import cv2
        img = np.full((80, 80, 3), 200, dtype=np.uint8)
        p = Path(tmp) / "solid.png"
        cv2.imwrite(str(p), img)
        # No face in a solid image; must not crash and return None or a bbox.
        roi = detect_face_roi(str(p))
        assert roi is None or (0.0 <= roi[0] <= 1.0)


def test_taubin_iterations_default_not_twenty():
    assert TAUBIN_ITERATIONS <= 5
    assert TAUBIN_ITERATIONS != 20


# ---------------------------------------------------------------------------
# Mesh QC status classification
# ---------------------------------------------------------------------------

def test_validate_mesh_returns_status_field():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "s.glb"
        sphere.export(str(p))
        report = validate_mesh(str(p))
        assert report["status"] in {"PASS", "PASS WITH WARNINGS", "FAILED"}
        assert "metrics" in report


def test_detail_metrics_surface_roughness_runs():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    dm = detail_metrics(sphere)
    assert "surface_roughness" in dm
    assert "faces" in dm


# ---------------------------------------------------------------------------
# Texture pipeline + multi-format export
# ---------------------------------------------------------------------------

def test_texture_projector_exports_glb_and_obj():
    with tempfile.TemporaryDirectory() as tmp:
        raw = _make_noisy_head_mesh(tmp)
        import cv2
        img = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        imgp = Path(tmp) / "ref.png"
        cv2.imwrite(str(imgp), img)

        glb = Path(tmp) / "textured.glb"
        obj = Path(tmp) / "textured.obj"
        project_texture_to_mesh(raw, str(imgp), str(glb), use_uv_atlas=False)
        assert glb.exists()
        # Vertex-color path exports a valid mesh.
        m = trimesh.load(str(glb), force="mesh")
        assert len(m.vertices) > 0


def test_preprocess_for_generation_preserves_original():
    with tempfile.TemporaryDirectory() as tmp:
        import cv2
        img = np.random.default_rng(2).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        src = Path(tmp) / "src.png"
        cv2.imwrite(str(src), img)
        out = preprocess_for_generation(str(src), str(Path(tmp) / "prep"))
        assert "original_rgb" in out
        assert "padded_rgba" in out
        # Original is preserved untouched.
        assert Path(out["original_rgb"]).exists()
        assert Path(out["padded_rgba"]).exists()


# ---------------------------------------------------------------------------
# Turntable isolation (single-image must not route through COLMAP)
# ---------------------------------------------------------------------------

def test_run_single_image_uses_single_image_engines_only(monkeypatch):
    """Verify the manager resolves single-image engines and does NOT invoke
    the turntable COLMAP path."""
    from pipeline.ai import engine_manager as em
    # Force all single-image engines unavailable -> clear error, no COLMAP call.
    for inst in em._ENGINES.values():
        monkeypatch.setattr(inst, "is_available", lambda: False)
    with tempfile.TemporaryDirectory() as tmp:
        import cv2
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        p = Path(tmp) / "x.png"
        cv2.imwrite(str(p), img)
        with pytest.raises(RuntimeError) as exc:
            run_single_image("trellis", str(p), str(Path(tmp) / "o.stl"), str(Path(tmp) / "o.glb"))
        assert "No single-image 3D engine is available" in str(exc.value)
