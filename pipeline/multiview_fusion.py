"""
Multi-View Turntable 3D Reconstruction via AI Depth Point Cloud Fusion

Fuses multi-view turntable photos using Depth-Anything-V2 foundation depth maps,
known stepper turntable orbital rotation matrices, and Open3D Poisson surface reconstruction.
Solves feature-matching failure on textureless glossy objects (like white chargers, plastics, metals).
"""

import math
import numpy as np
import open3d as o3d
import trimesh
import cv2
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preprocess import process_single_image
from pipeline.ai_depth_3d import estimate_ai_depth, compute_depth_normals, remove_flying_pixels


def reconstruct_multiview_turntable_mesh(
    image_paths: List[Path],
    output_stl_path: str,
    output_glb_path: Optional[str] = None,
    poisson_depth: int = 10,
    point_stride: int = 2
) -> str:
    """
    Reconstructs a full 360-degree watertight 3D mesh from a set of turntable photos.
    
    Args:
        image_paths: List of Paths to turntable images in sequential rotational order
        output_stl_path: Path where output STL file is saved
        output_glb_path: Optional path for textured GLB export (defaults to same stem as STL)
        poisson_depth: Open3D Poisson octree depth (default: 10 for high resolution)
        point_stride: Pixel sampling stride (2 = every second pixel)
    
    Returns:
        Path to output STL file
    """
    out_stl = Path(output_stl_path)
    out_glb = Path(output_glb_path) if output_glb_path else out_stl.with_suffix(".glb")
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    out_glb.parent.mkdir(parents=True, exist_ok=True)

    num_imgs = len(image_paths)
    if num_imgs < 2:
        raise ValueError("Multi-view reconstruction requires at least 2 images.")

    print(f"Starting Multi-View Turntable 3D Fusion across {num_imgs} photos...")
    angle_step_deg = 360.0 / num_imgs

    all_points = []
    all_normals = []
    all_colors = []

    for idx, img_path in enumerate(image_paths):
        print(f"  [{idx+1}/{num_imgs}] Processing {img_path.name}...")
        temp_rgba = out_stl.parent / f"temp_mv_{img_path.stem}_rgba.png"
        
        try:
            process_single_image(str(img_path), str(temp_rgba))
            rgba = cv2.imread(str(temp_rgba), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[2] < 4:
                continue

            b, g, r, alpha = cv2.split(rgba)
            img_bgr = cv2.merge([b, g, r])
            rgb = cv2.merge([r, g, b]) / 255.0
            mask = alpha > 20

            # 1. High-precision depth and surface normals
            depth = estimate_ai_depth(img_bgr)
            dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            edge_feather = np.clip(dist / 8.0, 0.0, 1.0)
            final_depth = depth * edge_feather
            final_depth[~mask] = 0.0

            normals = compute_depth_normals(final_depth, scale=2.5)

            h, w = alpha.shape
            cx, cy = w / 2.0, h / 2.0
            scale = 2.0 / max(h, w)

            # 2. Turntable rotation matrix around Y-axis
            angle_rad = math.radians(idx * angle_step_deg)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            rot_matrix = np.array([
                [cos_a,  0, sin_a],
                [0,      1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=np.float64)

            # 3. Extract unprojected 3D points
            for y in range(0, h, point_stride):
                for x in range(0, w, point_stride):
                    if mask[y, x] and edge_feather[y, x] > 0.08:
                        z = float(final_depth[y, x]) * 0.85
                        x_local = (x - cx) * scale
                        y_local = -(y - cy) * scale
                        col = rgb[y, x]
                        norm_local = normals[y, x]

                        # Transform into unified 360-degree world frame
                        pt_local = np.array([x_local, y_local, z], dtype=np.float64)
                        pt_world = rot_matrix @ pt_local
                        norm_world = rot_matrix @ norm_local

                        all_points.append(pt_world)
                        all_normals.append(norm_world)
                        all_colors.append(col)

        finally:
            if temp_rgba.exists():
                try:
                    temp_rgba.unlink()
                except Exception:
                    pass

    if len(all_points) < 500:
        raise RuntimeError("Failed to extract sufficient 3D points from multi-view images.")

    print(f"Total fused 360-degree point cloud: {len(all_points)} points.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(all_points, dtype=np.float64))
    pcd.normals = o3d.utility.Vector3dVector(np.array(all_normals, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors, dtype=np.float64))

    # 4. Statistical outlier filtering
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    print(f"Cleaned 360-degree point cloud: {len(pcd.points)} points.")

    # 5. Poisson Surface Reconstruction
    print(f"Running high-resolution Poisson Surface Reconstruction (depth={poisson_depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, scale=1.1, linear_fit=True
    )

    densities_arr = np.asarray(densities)
    if len(densities_arr) > 0:
        density_threshold = np.quantile(densities_arr, 0.03)
        mesh.remove_vertices_by_mask(densities_arr < density_threshold)

    # 6. Remove flying pixels across depth discontinuities
    mesh = remove_flying_pixels(mesh, max_edge_length=0.12)
    mesh.compute_vertex_normals()

    # 7. Auto-trim bottom turntable platform artifacts
    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    y_height = max_bound[1] - min_bound[1]
    if y_height > 0.1:
        # Slice off bottom 5% where turntable contacts the object
        min_bound[1] += y_height * 0.05
        crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        try:
            cropped = mesh.crop(crop_box)
            if len(cropped.triangles) > 1000:
                mesh = cropped
        except Exception as crop_err:
            print(f"Base crop notice: {crop_err}")

    # 8. Open3D Quadric Decimation (optimizes polygon density for 60 FPS WebGL)
    if len(mesh.triangles) > 250000:
        target_triangles = 180000
        print(f"Decimating mesh from {len(mesh.triangles)} to {target_triangles} triangles via Open3D for instant WebGL loading...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        mesh.compute_vertex_normals()

    # 8. Save STL & GLB
    o3d.io.write_triangle_mesh(str(out_stl), mesh)

    tri_mesh = trimesh.load(str(out_stl), force='mesh')
    if isinstance(tri_mesh, trimesh.Scene):
        tri_mesh = trimesh.util.concatenate(tri_mesh.dump())
    
    tri_mesh.remove_unreferenced_vertices()
    try:
        non_deg = tri_mesh.nondegenerate_faces()
        if non_deg is not None and len(non_deg) > 0:
            tri_mesh.update_faces(non_deg)
    except Exception:
        pass

    tri_mesh.export(str(out_stl))
    tri_mesh.export(str(out_glb))

    print(f"Multi-View 3D Model created: STL={out_stl}, GLB={out_glb} ({len(tri_mesh.vertices)} verts, {len(tri_mesh.faces)} faces)")
    return str(out_stl)
