"""
Metrology-Grade Turntable 3D Scanner via Scalable TSDF Volumetric Integration

Fuses multi-view RGB + Depth frames using mathematically exact camera extrinsic poses
derived from known Arduino stepper motor turntable rotation angles.

Eliminates feature-matching failure on textureless/glossy surfaces and integrates
depth measurements into a metric Open3D ScalableTSDFVolume voxel grid with dimensional accuracy evaluation.
"""

import math
import numpy as np
import open3d as o3d
import trimesh
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any


def generate_turntable_camera_poses(
    num_views: int = 24,
    radius_mm: float = 300.0,
    elevation_mm: float = 80.0,
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> List[np.ndarray]:
    """
    Calculates exact 4x4 camera extrinsic matrices (world-to-camera) from turntable stepper motor angles.
    
    Args:
        num_views: Number of captures per full 360-degree rotation (e.g. 24 = 15° steps)
        radius_mm: Physical distance from camera lens to turntable center in millimeters
        elevation_mm: Camera height above turntable platform in millimeters
        look_at: Target 3D coordinate at the center of the turntable
    
    Returns:
        List of 4x4 camera extrinsic matrices [R | t]
    """
    poses = []
    angle_step_deg = 360.0 / num_views
    target = np.array(look_at, dtype=np.float64)

    for i in range(num_views):
        angle_rad = math.radians(i * angle_step_deg)

        # Camera eye position orbiting around Y-axis (turntable rotation)
        cam_x = radius_mm * math.sin(angle_rad)
        cam_y = elevation_mm
        cam_z = radius_mm * math.cos(angle_rad)
        eye = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

        # Look-At camera coordinate frame (Open3D standard: +Z forward, +Y down, +X right)
        forward = target - eye
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            forward = np.array([0.0, 0.0, -1.0])
        else:
            forward = forward / forward_norm

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_norm

        up = np.cross(right, forward)

        # Open3D extrinsic matrix maps world coords to camera coords
        # Camera coordinates: X = right, Y = -up (down), Z = forward
        R = np.vstack([right, -up, forward])  # 3x3
        t = -R @ eye                         # 3x1

        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        poses.append(extrinsic)

    return poses


def create_default_camera_intrinsics(width: int = 1280, height: int = 720, fov_deg: float = 60.0) -> o3d.camera.PinholeCameraIntrinsic:
    """Constructs pinhole camera intrinsic matrix based on sensor resolution and field of view."""
    fx = (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def run_tsdf_turntable_integration(
    color_images: List[np.ndarray],
    depth_images: List[np.ndarray],
    camera_poses: List[np.ndarray],
    intrinsics: Optional[o3d.camera.PinholeCameraIntrinsic] = None,
    voxel_length_mm: float = 1.5,
    sdf_trunc_mm: float = 5.0,
    depth_scale: float = 1000.0,
    depth_trunc_mm: float = 600.0
) -> o3d.geometry.TriangleMesh:
    """
    Integrates multi-view RGB-D frames into an Open3D ScalableTSDFVolume voxel grid.
    
    Args:
        color_images: List of RGB uint8 numpy images (H, W, 3)
        depth_images: List of depth float32/uint16 numpy images (H, W) in millimeters
        camera_poses: List of 4x4 extrinsic matrices
        intrinsics: Pinhole camera intrinsics
        voxel_length_mm: Metric size of each TSDF voxel in mm (e.g. 1.0 = 1mm resolution)
        sdf_trunc_mm: Truncation distance for Signed Distance Function in mm
        depth_scale: Depth unit scale (1000.0 = mm to meters for Open3D)
        depth_trunc_mm: Maximum depth clipping threshold in mm
    """
    n_frames = min(len(color_images), len(depth_images), len(camera_poses))
    if n_frames == 0:
        raise ValueError("No frames provided for TSDF integration.")

    h, w = color_images[0].shape[:2]
    if intrinsics is None:
        intrinsics = create_default_camera_intrinsics(width=w, height=h)

    print(f"Initializing Open3D ScalableTSDFVolume (Voxel Resolution: {voxel_length_mm} mm, SDF Trunc: {sdf_trunc_mm} mm)...")
    
    # Scale from mm to meters for Open3D TSDF Volume
    voxel_length_m = voxel_length_mm / 1000.0
    sdf_trunc_m = sdf_trunc_mm / 1000.0
    depth_trunc_m = depth_trunc_mm / 1000.0

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length_m,
        sdf_trunc=sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i in range(n_frames):
        color_rgb = color_images[i]
        if color_rgb.shape[2] == 4:
            color_rgb = cv2.cvtColor(color_rgb, cv2.COLOR_RGBA2RGB)
        elif color_rgb.dtype != np.uint8:
            color_rgb = (color_rgb * 255).astype(np.uint8)

        depth_mm = depth_images[i].astype(np.float32)
        # Remove depth outliers/NaNs
        depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)

        o3d_color = o3d.geometry.Image(np.ascontiguousarray(color_rgb))
        o3d_depth = o3d.geometry.Image(np.ascontiguousarray(depth_mm))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc_m,
            convert_rgb_to_intensity=False
        )

        # Scale pose translation from mm to meters
        extrinsic_m = camera_poses[i].copy()
        extrinsic_m[:3, 3] = extrinsic_m[:3, 3] / 1000.0

        volume.integrate(rgbd, intrinsics, extrinsic_m)

    print("Extracting 360-degree watertight triangle mesh from TSDF volume...")
    mesh = volume.extract_triangle_mesh()
    
    # Clean up low-density outliers extracted from the volume
    if len(mesh.vertices) > 0:
        cl, ind = mesh.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        mesh = mesh.select_by_index(ind)
    
    # Scale extracted mesh vertices back from meters to millimeters
    mesh.scale(1000.0, center=np.array([0.0, 0.0, 0.0]))
    mesh.compute_vertex_normals()
    
    print(f"TSDF Mesh extracted: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles.")
    return mesh


def evaluate_scanner_accuracy(mesh_path: str, real_dimensions_mm: Dict[str, float]) -> Dict[str, Any]:
    """
    Evaluates physical metrology scanner accuracy by comparing reconstructed bounding box
    and volume against physical digital caliper measurements.
    
    Args:
        mesh_path: Path to reconstructed STL/GLB file
        real_dimensions_mm: Dict with 'length', 'width', 'height' (or 'x', 'y', 'z') in millimeters
    
    Returns:
        Dict containing error analysis and accuracy score percentage
    """
    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    bounds = mesh.extents  # [width, height, depth] in mm
    generated_dims = np.sort(bounds)
    real_vals = [
        real_dimensions_mm.get('length', real_dimensions_mm.get('x', 0.0)),
        real_dimensions_mm.get('width', real_dimensions_mm.get('y', 0.0)),
        real_dimensions_mm.get('height', real_dimensions_mm.get('z', 0.0))
    ]
    real_dims = np.sort([float(x) for x in real_vals if x > 0])

    if len(real_dims) < 3:
        raise ValueError("Must provide 3 non-zero physical caliper dimensions (length, width, height in mm).")

    axes = ['Shortest (Z)', 'Middle (Y)', 'Longest (X)']
    axis_errors = []
    total_pct_error = 0.0

    print("\n" + "=" * 60)
    print("      METROLOGY 3D SCANNER DIMENSIONAL ACCURACY REPORT")
    print("=" * 60)

    for i in range(3):
        err_mm = abs(real_dims[i] - generated_dims[i])
        err_pct = (err_mm / real_dims[i]) * 100.0 if real_dims[i] > 0 else 0.0
        total_pct_error += err_pct
        
        axis_errors.append({
            "axis": axes[i],
            "caliper_mm": round(float(real_dims[i]), 2),
            "scan_mm": round(float(generated_dims[i]), 2),
            "error_mm": round(float(err_mm), 2),
            "error_pct": round(float(err_pct), 2)
        })
        print(f"  {axes[i]:14s} -> Caliper: {real_dims[i]:6.2f} mm | Scan: {generated_dims[i]:6.2f} mm | Error: {err_mm:5.2f} mm ({err_pct:4.1f}%)")

    mean_err_pct = total_pct_error / 3.0
    accuracy_score = max(0.0, min(100.0, 100.0 - mean_err_pct))

    print("-" * 60)
    print(f"  Mean Dimensional Error: {mean_err_pct:.2f}%")
    print(f"  Physical Metrology Accuracy Score: {accuracy_score:.1f}%")
    print("=" * 60 + "\n")

    return {
        "mesh_file": Path(mesh_path).name,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "axis_errors": axis_errors,
        "mean_error_pct": round(mean_err_pct, 2),
        "accuracy_score_pct": round(accuracy_score, 1)
    }
