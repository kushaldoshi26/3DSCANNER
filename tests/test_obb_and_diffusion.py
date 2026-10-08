import pytest
import sys
import numpy as np
import trimesh
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from api.main import app
from pipeline.dimensional_eval import auto_measure_mesh, format_dimensional_report
from pipeline.mesh_topology import analyze_mesh_topology
from pipeline.multiview_diffusion import synthesize_multiview_diffusion

client = TestClient(app)


def test_obb_rotation_invariance(tmp_path):
    """
    Tests that Oriented Bounding Box (OBB) dimensions remain constant
    regardless of how the 3D mesh is rotated in space.
    """
    # Create a box with known dimensions: 10 x 5 x 2 units
    box = trimesh.creation.box(extents=[10.0, 5.0, 2.0])
    
    # 1. Measure unrotated box
    m1 = auto_measure_mesh(box)
    assert pytest.approx(m1["length_mm"], rel=1e-2) == 10.0
    assert pytest.approx(m1["width_mm"], rel=1e-2) == 5.0
    assert pytest.approx(m1["height_mm"], rel=1e-2) == 2.0

    # 2. Rotate box by 45 degrees along X, Y, and Z axes
    rot_matrix = trimesh.transformations.euler_matrix(0.785, 0.523, 1.047)
    box_rotated = box.copy()
    box_rotated.apply_transform(rot_matrix)

    m2 = auto_measure_mesh(box_rotated)
    
    # OBB dimensions must be invariant to rotation (within numerical tolerance)
    assert pytest.approx(m2["length_mm"], rel=1e-2) == 10.0
    assert pytest.approx(m2["width_mm"], rel=1e-2) == 5.0
    assert pytest.approx(m2["height_mm"], rel=1e-2) == 2.0


def test_obb_report_formatting():
    """Tests the CLI dimensional report string formatter."""
    box = trimesh.creation.box(extents=[2.0, 1.0, 0.5])
    report = auto_measure_mesh(box)
    formatted = format_dimensional_report(report)
    
    assert "AUTOMATED DIMENSIONAL & OBB MEASUREMENT REPORT" in formatted
    assert "Length (Longest side):" in formatted
    assert "Width  (Middle side):" in formatted
    assert "Height (Shortest side):" in formatted
    assert "OBB Bounding Volume:" in formatted


def test_mesh_topology_obb_integration(tmp_path):
    """Tests that analyze_mesh_topology returns complete OBB physical metrics."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    stl_path = tmp_path / "test_sphere.stl"
    mesh.export(str(stl_path))

    res = analyze_mesh_topology(str(stl_path))
    assert "length_mm" in res
    assert "width_mm" in res
    assert "height_mm" in res
    assert "obb_volume_mm3" in res
    assert "surface_area_mm2" in res
    assert "dimensions_obb" in res
    assert res["is_watertight"] is True
    assert res["printability_score"] >= 80


def test_multiview_diffusion_synthesis(tmp_path):
    """Tests multi-view novel view synthesis from a single input image."""
    img_path = Path("data/raw/frame_000.jpg")
    if not img_path.exists():
        pytest.skip("data/raw/frame_000.jpg not found")

    out_views_dir = tmp_path / "synth_views"
    views = synthesize_multiview_diffusion(str(img_path), str(out_views_dir), num_views=4)
    
    assert len(views) == 4
    for v in views:
        assert v.exists()
        assert v.stat().st_size > 0


def test_diffusion_and_dimension_api_endpoints(tmp_path):
    """Tests POST /api/scan/diffusion-3d and GET /api/scan/dimensions/{job_id}."""
    img_path = Path("data/raw/frame_000.jpg")
    if not img_path.exists():
        pytest.skip("data/raw/frame_000.jpg not found")

    with open(img_path, "rb") as f:
        resp = client.post("/api/scan/diffusion-3d", files={"file": ("frame_000.jpg", f, "image/jpeg")})
    
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    job_id = data["job_id"]

    # Verify status endpoint returns job tracking
    status_resp = client.get(f"/api/scan/status/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["job_id"] == job_id
