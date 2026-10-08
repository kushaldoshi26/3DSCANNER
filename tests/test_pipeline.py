import os
import sys
import tempfile
import pytest
import numpy as np
import cv2
import open3d as o3d
import trimesh
from pathlib import Path

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preprocess import process_images, remove_background_opencv
from pipeline.colmap_poses import opencv_sfm_fallback
from pipeline.open3d_mesh import pointcloud_to_mesh
from pipeline.ai_refine import refine_mesh
from api.main import app
from fastapi.testclient import TestClient


def test_remove_background_opencv():
    # Create synthetic test image (200x200 red circle on white background)
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.circle(img, (100, 100), 50, (0, 0, 255), -1)

    result = remove_background_opencv(img)
    assert result.shape == (200, 200, 4)
    # Check transparent alpha channel exists
    assert np.any(result[:, :, 3] == 0)


def test_open3d_meshing_and_ai_refinement():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ply_file = tmp_path / "test_cloud.ply"
        stl_file = tmp_path / "test_mesh.stl"
        refined_stl = tmp_path / "refined_mesh.stl"

        # Generate synthetic sphere point cloud
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).sample_points_uniformly(number_of_points=500)
        o3d.io.write_point_cloud(str(ply_file), sphere)

        # Run meshing
        pointcloud_to_mesh(str(ply_file), str(stl_file))
        assert stl_file.exists()

        # Run AI Refinement
        refine_mesh(str(stl_file), str(refined_stl))
        assert refined_stl.exists()

        # Verify output mesh valid
        loaded_mesh = trimesh.load(str(refined_stl))
        assert len(loaded_mesh.vertices) > 0


def test_fastapi_endpoints():
    client = TestClient(app)

    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200

    # Test invalid job status
    response = client.get("/api/scan/status/nonexistent_job")
    assert response.status_code == 404
