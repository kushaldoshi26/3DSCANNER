import os
import sys
import tempfile
import pytest
import numpy as np
import cv2
import trimesh
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ai_depth_3d import generate_ai_depth_3d_mesh, estimate_ai_depth
from pipeline.mesh_topology import analyze_mesh_topology, repair_mesh_topology
from pipeline.download_ai_models import download_sample_3d_printing_datasets
from api.main import app
from fastapi.testclient import TestClient


def test_estimate_ai_depth():
    # Synthetic RGB image (100x100 circle)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    cv2.circle(img, (50, 50), 30, (50, 50, 220), -1)

    depth = estimate_ai_depth(img)
    assert depth.shape == (100, 100)
    assert depth.min() >= 0.0 and depth.max() <= 1.0


def test_mesh_topology_analysis_and_repair():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mesh_path = tmp_path / "sphere.stl"
        repaired_path = tmp_path / "sphere_repaired.stl"

        # Create a sphere mesh
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        sphere.export(str(mesh_path))

        analysis = analyze_mesh_topology(str(mesh_path))
        assert "printability_score" in analysis
        assert analysis["vertices"] > 0
        assert analysis["faces"] > 0

        # Run topology repair
        repair_mesh_topology(str(mesh_path), str(repaired_path))
        assert repaired_path.exists()


def test_ai_depth_and_iot_api_endpoints():
    client = TestClient(app)

    # Create dummy JPEG image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.circle(img, (50, 50), 25, (0, 0, 255), -1)
    _, img_encoded = cv2.imencode('.jpg', img)

    # Test IoT scan trigger endpoint
    response = client.post(
        "/api/iot/scan",
        files=[("files", ("iot_frame.jpg", img_encoded.tobytes(), "image/jpeg"))],
        data={"mode": "ai_depth"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"

def test_api_reconstruct_endpoint():
    client = TestClient(app)
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode('.jpg', img)

    response = client.post(
        "/api/reconstruct",
        data={"engine": "trellis"},
        files=[("file", ("test.jpg", img_encoded.tobytes(), "image/jpeg"))]
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "message" in data

def test_engine_resolution():
    from api.routes import resolve_engine
    assert resolve_engine("trellis") == "trellis"
    assert resolve_engine("hunyuan3d2") == "hunyuan3d"
    assert resolve_engine("depth") == "ai_depth"
    assert resolve_engine("invalid_engine") == "trellis"

