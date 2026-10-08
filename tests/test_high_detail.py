import os
import sys
import tempfile
import pytest
import numpy as np
import cv2
import torch
import trimesh
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.download_dataset import prepare_mesh_dataset
from pipeline.depth_normal_refine import estimate_surface_normals, image_to_detailed_pointcloud
from pipeline.texture_projector import project_texture_to_mesh
from pipeline.losses import chamfer_distance_loss


def test_prepare_mesh_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_dir = tmp_path / "raw"
        proc_dir = tmp_path / "processed"
        raw_dir.mkdir()

        # Create sample mesh
        cube = trimesh.creation.box(extents=[10, 10, 10])
        cube.export(str(raw_dir / "test_box.stl"))

        prepare_mesh_dataset(str(raw_dir), str(proc_dir))
        out_files = list(proc_dir.glob("*.npz"))
        assert len(out_files) == 1
        # Use allow_pickle=False and explicitly close file handle (Windows file locking)
        with np.load(str(out_files[0])) as data:
            assert "points" in data and len(data["points"]) == 10000


def test_surface_normals_and_detailed_pointcloud():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_path = tmp_path / "test_img.jpg"

        # Synthetic image (100x100)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 200
        cv2.circle(img, (50, 50), 30, (0, 0, 255), -1)
        cv2.imwrite(str(img_path), img)

        normals = estimate_surface_normals(img)
        assert normals.shape == (100, 100, 3)

        depth = np.ones((100, 100), dtype=np.float64) * 0.5
        pcd = image_to_detailed_pointcloud(str(img_path), depth, normals)
        assert len(pcd.points) > 0


def test_chamfer_distance_loss():
    p1 = torch.rand(2, 50, 3)
    p2 = torch.rand(2, 50, 3)

    loss = chamfer_distance_loss(p1, p2)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0


def test_texture_projection_to_glb():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        stl_path = tmp_path / "mesh.stl"
        img_path = tmp_path / "texture.png"
        glb_path = tmp_path / "model.glb"

        # Create mesh and texture
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        sphere.export(str(stl_path))

        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(img_path), img)

        out_glb = project_texture_to_mesh(str(stl_path), str(img_path), str(glb_path))
        assert Path(out_glb).exists()
