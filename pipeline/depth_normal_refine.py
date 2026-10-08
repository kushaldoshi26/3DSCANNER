import cv2
import numpy as np
import open3d as o3d
import torch
from pathlib import Path

def estimate_surface_normals(img_bgr: np.ndarray) -> np.ndarray:
    """
    Estimates 3D surface normal map (Nx, Ny, Nz) from image gradients for high-frequency detail.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) / 255.0
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Compute Sobel gradients
    dx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

    # Construct normal vectors (-dx, -dy, dz) normalized
    # Scale dz based on gradient magnitude to preserve structure
    grad_mag = np.sqrt(dx**2 + dy**2)
    dz = np.ones_like(dx) * np.clip(1.0 - grad_mag, 0.1, 1.0)
    normals = np.stack([-dx, -dy, dz], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    normals /= norm

    return normals


def image_to_detailed_pointcloud(img_path: str, depth_map: np.ndarray, normal_map: np.ndarray = None) -> o3d.geometry.PointCloud:
    """
    Converts 2D image pixels into a high-density, normal-guided 3D point cloud.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    h, w, _ = img.shape
    if normal_map is None:
        normal_map = estimate_surface_normals(img)

    fx = fy = max(h, w)
    cx, cy = w / 2.0, h / 2.0

    points = []
    normals = []
    colors = []

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    for v in range(0, h, 2):
        for u in range(0, w, 2):
            z = depth_map[v, u]
            if z <= 0.01 or z > 5.0:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, y, z])
            normals.append(normal_map[v, u])
            colors.append(img_rgb[v, u])

    pcd = o3d.geometry.PointCloud()
    if points:
        pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
        pcd.normals = o3d.utility.Vector3dVector(np.array(normals, dtype=np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))

    return pcd
