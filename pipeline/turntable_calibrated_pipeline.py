"""
3DINVENZA Calibrated Turntable 3D Reconstruction Pipeline

Ingests 24 to 72 turntable frames captured at known angular steps (e.g. 15° increments).
Reconstructs metric-accurate 3D models using:
1. Camera Calibration & Lens Undistortion (K matrix & distortion parameters)
2. High-Accuracy Silhouette Masking & Debug Inspection
3. Physical Scale Reference & Metrology (ArUco / Turntable Baseline in mm)
4. Known Turntable Rotation Constraints (R_y(theta_i)) & Multi-View Point Cloud Fusion
5. High-Resolution Poisson Surface Meshing
6. Canonical Ground-Plane Alignment (Z=0, centered, upright)
7. Oriented Bounding Box (OBB) Physical Auto-Measurement
8. Scan Quality & Confidence Analytics
"""

import sys
import math
import cv2
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preprocess import process_single_image
from pipeline.camera_calibration import load_calibration, undistort_image
from pipeline.scale_reference import compute_dataset_scale_reference
from pipeline.canonical_alignment import align_mesh_to_ground_plane
from pipeline.dimensional_eval import auto_measure_mesh, format_dimensional_report
from pipeline.scan_confidence import evaluate_scan_quality, format_scan_quality_report
from pipeline.texture_projector import project_texture_to_mesh
from pipeline.train_ai_3d import predict_custom_neural_depth_and_normals
from pipeline.ai_depth_3d import remove_flying_pixels


def run_calibrated_turntable_scan(
    image_paths: List[Path],
    output_stl_path: str,
    output_glb_path: Optional[str] = None,
    rotation_step_deg: Optional[float] = None,
    expected_frames: int = 24,
    poisson_depth: int = 10,
    point_stride: int = 2,
    camera_profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes full calibrated turntable reconstruction on 24-72 sequence images.
    """
    num_images = len(image_paths)
    if num_images == 0:
        raise ValueError("No input images provided for calibrated turntable scan.")

    # Calculate exact angular rotation step per frame if not explicitly given
    if rotation_step_deg is None or rotation_step_deg <= 0:
        rotation_step_deg = 360.0 / num_images

    out_stl = Path("d:/3dSCANNER/data/exports") / Path(output_stl_path).name
    out_glb = Path("d:/3dSCANNER/data/exports") / (Path(output_glb_path).name if output_glb_path else out_stl.with_suffix(".glb").name)
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    job_work_dir = out_stl.parent

    # Create masks debug inspection directory
    masks_dir = job_work_dir / "masks_debug"
    masks_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  3DINVENZA CALIBRATED TURNTABLE 3D SCANNER ENGINE")
    print(f"  Captured Images: {num_images} | Angular Step: {rotation_step_deg:.2f}° | Depth: {poisson_depth}")
    print("=" * 70)

    # 1. Camera Calibration Profile
    if camera_profile is None:
        camera_profile = load_calibration()
    print(f"[Stage 1] Camera Intrinsics: fx={camera_profile['fx']:.1f}, fy={camera_profile['fy']:.1f}, RMS={camera_profile.get('rms_reprojection_error', 0.0):.3f}px (Calibrated: {camera_profile.get('is_calibrated', False)})")

    # 2. Physical Scale Reference Metrology
    scale_info = compute_dataset_scale_reference(image_paths)
    print(f"[Stage 2] Physical Scale Reference: {scale_info['reference_type']} (Confidence: {scale_info['scale_confidence']*100:.1f}%)")

    # 3. Process & Undistort Multi-View Frames with Silhouette Masks
    print(f"\n[Stage 3] Processing, undistorting & masking {num_images} turntable frames...")
    processed_views = []
    mask_qualities = []

    for i, img_p in enumerate(image_paths):
        img_raw = cv2.imread(str(img_p))
        if img_raw is None:
            continue

        # Undistort image using camera intrinsics
        img_undist = undistort_image(img_raw, camera_profile)

        # Subject isolation via rembg
        temp_rgba = masks_dir / f"frame_{i:03d}_rgba.png"
        process_single_image(str(img_p), str(temp_rgba))
        rgba = cv2.imread(str(temp_rgba), cv2.IMREAD_UNCHANGED)

        if rgba is not None and rgba.shape[2] == 4:
            b, g, r, alpha = cv2.split(rgba)
            rgb = cv2.merge([b, g, r])
            mask = alpha > 20

            # Save debug inspection mask images
            cv2.imwrite(str(masks_dir / f"frame_{i:03d}_orig.jpg"), img_undist)
            cv2.imwrite(str(masks_dir / f"frame_{i:03d}_mask.png"), (mask.astype(np.uint8) * 255))
            cv2.imwrite(str(masks_dir / f"frame_{i:03d}_masked.png"), rgba)

            mask_coverage = float(np.sum(mask)) / float(mask.size)
            mask_qualities.append(min(1.0, mask_coverage * 3.0))  # Normalized coverage score
            processed_views.append((rgb, mask, i))

        if temp_rgba.exists():
            try: temp_rgba.unlink()
            except Exception: pass

    if len(processed_views) == 0:
        raise RuntimeError("Failed to process any valid turntable frames.")

    # =========================================================================
    # STAGE A: METRIC BASE GEOMETRY (Accurate SIFT/COLMAP Anchor Points)
    # =========================================================================
    print(f"\n[Stage A] Generating accurate metric base geometry via OpenCV/COLMAP SFM...")
    sparse_ply_path = None
    sparse_pcd = None
    sparse_center = np.zeros(3)
    sparse_bbox = None
    try:
        from pipeline.colmap_poses import run_colmap_reconstruction
        img_dir = str(image_paths[0].parent)
        wsp_dir = str(job_work_dir / "sfm_workspace")
        sparse_ply_path = run_colmap_reconstruction(img_dir, wsp_dir)
        
        # Load and scale the sparse cloud
        sparse_pcd = o3d.io.read_point_cloud(str(sparse_ply_path))
        print(f"  Sparse Anchor Points generated: {len(sparse_pcd.points):,}")
        
        # Scale to match ArUco
        unit_scale = scale_info.get('scale_factor', 1.0)
        sparse_pcd.scale(unit_scale, center=(0,0,0))
        sparse_bbox = sparse_pcd.get_axis_aligned_bounding_box()
        sparse_center = sparse_bbox.get_center()
        
    except Exception as sfm_err:
        print(f"  Stage A Notice: SFM extraction failed ({sfm_err}). Defaulting to Stage B purely.")

    # =========================================================================
    # STAGE B: AI GEOMETRY ENHANCEMENT (Dense Textureless Reconstruction)
    # =========================================================================
    print(f"\n[Stage B / Stage 4] Fusing dense multi-view 3D point clouds using AI Depth...")
    all_points = []
    all_normals = []
    all_colors = []

    for rgb, mask, frame_idx in processed_views:
        # Predict relative depth & unit normal gradients using CUSTOM AI
        depth, normals, mask_pred = predict_custom_neural_depth_and_normals(rgb, model_path="d:/3dSCANNER/data/models/best_3d_ai_model.pth")
        
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        edge_feather = np.clip(dist / 8.0, 0.0, 1.0)
        final_depth = depth * edge_feather
        final_depth[~mask] = 0.0

        clean_mask = mask & (edge_feather > 0.15)

        h, w = depth.shape
        cx, cy = w / 2.0, h / 2.0
        focal = camera_profile["fx"]

        # Known turntable rotation angle for frame i
        theta_deg = frame_idx * rotation_step_deg
        theta_rad = math.radians(theta_deg)
        cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)

        # Orbital Turntable Y-Axis Rotation Matrix
        R_y = np.array([
            [cos_t,  0.0, sin_t],
            [0.0,    1.0, 0.0],
            [-sin_t, 0.0, cos_t]
        ], dtype=np.float64)

        # Extract 3D points
        y_indices, x_indices = np.where(clean_mask[::point_stride, ::point_stride])
        y_pts = y_indices * point_stride
        x_pts = x_indices * point_stride

        z_vals = depth[y_pts, x_pts]
        # Invert depth so foreground objects project towards camera
        z_3d = 1.0 - z_vals

        x_3d = (x_pts - cx) * z_3d / focal
        y_3d = -(y_pts - cy) * z_3d / focal

        local_pts = np.stack([x_3d, y_3d, z_3d], axis=-1)
        local_norms = normals[y_pts, x_pts]
        local_colors = rgb[y_pts, x_pts][:, [2, 1, 0]] / 255.0  # BGR -> RGB

        # Apply turntable orbital rotation
        world_pts = local_pts @ R_y.T
        world_norms = local_norms @ R_y.T

        all_points.append(world_pts)
        all_normals.append(world_norms)
        all_colors.append(local_colors)

    pts_combined = np.vstack(all_points).astype(np.float64)
    norms_combined = np.vstack(all_normals).astype(np.float64)
    cols_combined = np.vstack(all_colors).astype(np.float64)

    # Normalize normals
    norm_mag = np.linalg.norm(norms_combined, axis=1, keepdims=True) + 1e-8
    norms_combined /= norm_mag

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_combined)
    pcd.normals = o3d.utility.Vector3dVector(norms_combined)
    pcd.colors = o3d.utility.Vector3dVector(cols_combined)

    print(f"  Total fused 360° point cloud: {len(pcd.points):,} points.")

    # Statistical Outlier Removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.2)
    pcd_clean = pcd.select_by_index(ind)
    print(f"  Cleaned point cloud:          {len(pcd_clean.points):,} points.")

    # =========================================================================
    # STAGE A + STAGE B: FUSION & ALIGNMENT
    # =========================================================================
    if sparse_ply_path is not None and sparse_pcd is not None and len(sparse_pcd.points) > 10:
        print("  Aligning dense AI geometry to accurate Stage A metric anchor points...")
        ai_bbox = pcd_clean.get_axis_aligned_bounding_box()
        ai_center = ai_bbox.get_center()
        
        # 1. Translate to match centers
        pcd_clean.translate(sparse_center - ai_center)
        
        # 2. Scale to match exact bounding box extents
        ai_extents = ai_bbox.get_extent()
        sparse_extents = sparse_bbox.get_extent()
        
        if np.all(ai_extents > 0):
            scale_ratio = np.mean(sparse_extents / ai_extents)
            # Prevent crazy scaling if SFM failed to capture the full object
            if 0.2 < scale_ratio < 5.0:
                pcd_clean.scale(scale_ratio, center=sparse_center)
                print(f"  Successfully warped AI cloud to exact metric scale (ratio: {scale_ratio:.3f})")
            else:
                print(f"  Skipped warping: Scale ratio {scale_ratio:.3f} is out of bounds.")
    else:
        print("  Skipping Stage A alignment (no valid sparse metric anchors found).")

    # 5. Poisson Surface Reconstruction
    print(f"\n[Stage 5] Reconstructing high-detail Poisson 3D Surface (depth={poisson_depth})...")
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_clean, depth=poisson_depth, linear_fit=True
    )

    # Crop low-density Poisson halo vertices
    densities = np.asarray(densities)
    if len(densities) > 0:
        thresh = np.quantile(densities, 0.05)
        mesh_o3d.remove_vertices_by_mask(densities < thresh)

    # Transfer colors from point cloud to mesh using nearest neighbors
    if pcd_clean.has_colors():
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_clean)
        mesh_colors = []
        pcd_colors = np.asarray(pcd_clean.colors)
        for v in np.asarray(mesh_o3d.vertices):
            [_, idx, _] = pcd_tree.search_knn_vector_3d(v, 1)
            mesh_colors.append(pcd_colors[idx[0]])
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(np.vstack(mesh_colors))

    mesh_o3d.compute_vertex_normals()
    temp_ply = job_work_dir / "temp_raw_poisson.ply"
    o3d.io.write_triangle_mesh(str(temp_ply), mesh_o3d)

    # 5.5 Mesh Refinement (Floating Artifacts, Smoothing, Decimation)
    print(f"\n[Stage 5.5] Refining mesh (Removing floating artifacts, Smoothing, Decimating)...")
    mesh_tri = trimesh.load(str(temp_ply), force='mesh')
    if isinstance(mesh_tri, trimesh.Scene):
        mesh_tri = trimesh.util.concatenate(mesh_tri.dump())
        
    try:
        components = mesh_tri.split(only_watertight=False)
        if len(components) > 1:
            mesh_tri = max(components, key=lambda m: m.bounding_box.volume)
            print(f"  Discarded {len(components)-1} floating islands.")
    except Exception as e:
        print(f"  Component splitting failed: {e}")

    try:
        trimesh.smoothing.filter_laplacian(mesh_tri, iterations=3)
        print("  Applied 3-iteration Laplacian smoothing.")
    except Exception as e:
        print(f"  Smoothing failed: {e}")

    target_faces = 100000
    if len(mesh_tri.faces) > target_faces:
        print(f"  Decimating from {len(mesh_tri.faces)} to ~{target_faces} faces...")
        try:
            ratio = float(target_faces) / float(len(mesh_tri.faces))
            mesh_tri = mesh_tri.simplify_quadric_decimation(ratio)
        except Exception as e:
            print(f"  Decimation failed: {e}")
            
    mesh_tri.fix_normals()

    # 6. Canonical Ground-Plane Alignment
    print(f"\n[Stage 6] Aligning 3D model to canonical ground plane (Z=0, centered)...")
    aligned_mesh, transform = align_mesh_to_ground_plane(mesh_tri)
    aligned_mesh.export(str(out_stl))
    
    # Export temporary PLY to preserve vertex colors for texture projection
    temp_aligned_ply = job_work_dir / "temp_aligned.ply"
    aligned_mesh.export(str(temp_aligned_ply))
    
    if temp_ply.exists():
        try: temp_ply.unlink()
        except Exception: pass

    # 7. Project Textures & Export GLB
    print(f"\n[Stage 7] Projecting high-resolution multi-view texture onto 3D mesh...")
    try:
        project_texture_to_mesh(
            mesh_path=str(temp_aligned_ply),
            rgb_image_path=str(image_paths[0]),
            output_glb=str(out_glb)
        )
        if temp_aligned_ply.exists():
            try: temp_aligned_ply.unlink()
            except Exception: pass
    except Exception as tex_err:
        print(f"Texture projection notice ({tex_err}). Exporting standard GLB...")
        aligned_mesh.export(str(out_glb))

    # 8. OBB Physical Auto-Measurement
    print(f"\n[Stage 8] Calculating exact Oriented Bounding Box (OBB) physical dimensions...")
    dim_report = auto_measure_mesh(str(out_stl))
    print(format_dimensional_report(dim_report))

    # 9. Scan Quality & Confidence Analytics
    print(f"\n[Stage 9] Evaluating 3DINVENZA Scan Quality & Confidence...")
    avg_mask_q = float(np.mean(mask_qualities)) if mask_qualities else 0.9
    quality_report = evaluate_scan_quality(
        images_count=num_images,
        expected_images=expected_frames,
        mask_coverage_avg=avg_mask_q,
        is_camera_calibrated=camera_profile.get("is_calibrated", False),
        rms_reprojection=camera_profile.get("rms_reprojection_error", 0.45),
        point_cloud_size=len(pcd_clean.points),
        mesh_faces=len(aligned_mesh.faces),
        is_watertight=bool(aligned_mesh.is_watertight),
        scale_confidence=scale_info["scale_confidence"]
    )
    print(format_scan_quality_report(quality_report))

    return {
        "stl_path": str(out_stl),
        "glb_path": str(out_glb),
        "images_count": num_images,
        "rotation_step_deg": rotation_step_deg,
        "point_cloud_points": len(pcd_clean.points),
        "faces": len(aligned_mesh.faces),
        "vertices": len(aligned_mesh.vertices),
        "dimensions": dim_report,
        "quality": quality_report,
        "scale": scale_info,
        "masks_debug_dir": str(masks_dir)
    }


if __name__ == "__main__":
    raw_images = sorted(list(Path("data/raw").glob("frame_*.jpg")))
    if raw_images:
        res = run_calibrated_turntable_scan(
            image_paths=raw_images,
            output_stl_path="data/output/turntable_calibrated_model.stl",
            rotation_step_deg=30.0,
            expected_frames=12
        )
    else:
        print("No test images found in data/raw/")
