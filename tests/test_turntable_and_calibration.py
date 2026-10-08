"""
Automated Test Suite for 3DINVENZA Calibrated Turntable Scanner & Metrology
"""

import sys
import pytest
import numpy as np
import trimesh
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app
from pipeline.camera_calibration import (
    get_default_camera_matrix,
    load_calibration,
    save_calibration,
    undistort_image
)
from pipeline.scale_reference import (
    detect_aruco_scale_reference,
    detect_turntable_platter_scale,
    compute_dataset_scale_reference
)
from pipeline.canonical_alignment import align_mesh_to_ground_plane
from pipeline.scan_confidence import evaluate_scan_quality, format_scan_quality_report
from pipeline.dimensional_eval import auto_measure_mesh

client = TestClient(app)


def test_camera_calibration_default():
    """Verifies default smartphone camera matrix construction."""
    K, dist = get_default_camera_matrix(1920, 1080, 65.0)
    assert K.shape == (3, 3)
    assert K[0, 0] > 1000  # fx
    assert K[1, 1] > 1000  # fy
    assert K[0, 2] == 960  # cx
    assert K[1, 2] == 540  # cy
    assert len(dist) == 5


def test_camera_undistortion():
    """Verifies image undistortion pipeline."""
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    undist = undistort_image(dummy_img)
    assert undist.shape == (480, 640, 3)


def test_scale_reference_detection():
    """Verifies physical scale detection and millimeter conversion."""
    dummy_img = np.zeros((600, 800, 3), dtype=np.uint8)
    scale_res = detect_turntable_platter_scale(dummy_img, default_platter_dia_mm=200.0)
    assert "mm_per_pixel" in scale_res
    assert scale_res["mm_per_pixel"] > 0
    assert scale_res["scale_confidence"] >= 0.80


def test_canonical_ground_plane_alignment():
    """Verifies mesh canonical ground plane alignment (Z=0, centered)."""
    # Create a box floating at Z=50 to 100
    box = trimesh.creation.box(extents=(60, 40, 20))
    box.apply_translation([10, 20, 50])

    aligned, transform = align_mesh_to_ground_plane(box)
    verts = aligned.vertices
    min_z = float(np.min(verts[:, 2]))
    center_xy = np.mean(verts[:, :2], axis=0)

    # Base should sit flat on Z=0
    assert abs(min_z) < 1e-4
    # Center XY should be at origin (0, 0)
    assert abs(center_xy[0]) < 1e-3
    assert abs(center_xy[1]) < 1e-3


def test_scan_quality_evaluation():
    """Verifies Scan Quality scoring and diagnostics."""
    report = evaluate_scan_quality(
        images_count=24,
        expected_images=24,
        mask_coverage_avg=0.95,
        is_camera_calibrated=True,
        rms_reprojection=0.35,
        point_cloud_size=50000,
        mesh_faces=80000,
        is_watertight=True,
        scale_confidence=0.95
    )

    assert 80 <= report["overall_score"] <= 100
    assert report["metrics"]["coverage_percent"] == 100.0
    assert report["metrics"]["scale_percent"] == 95.0
    assert report["is_watertight"] is True

    ascii_box = format_scan_quality_report(report)
    assert "3DINVENZA SCAN QUALITY SCORE" in ascii_box


def test_api_camera_calibration_endpoint():
    """Verifies GET /api/calibration/camera."""
    response = client.get("/api/calibration/camera")
    assert response.status_code == 200
    data = response.json()
    assert "camera_matrix" in data
    assert "fx" in data


def test_api_turntable_calibrated_endpoint():
    """Verifies POST /api/scan/turntable-calibrated."""
    raw_images = sorted(list(Path("data/raw").glob("frame_*.jpg")))[:3]
    if raw_images:
        files = [("files", (p.name, open(p, "rb"), "image/jpeg")) for p in raw_images]
        response = client.post("/api/scan/turntable-calibrated?rotation_step=30.0&expected_frames=12", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["frames_count"] == len(raw_images)
