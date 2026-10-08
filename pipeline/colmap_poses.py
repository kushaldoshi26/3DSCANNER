import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path
import open3d as o3d

try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False
    print("Notice: 'pycolmap' module not found. Will use OpenCV SFM fallback if COLMAP binary isn't on PATH.")


def opencv_sfm_fallback(image_dir: Path, workspace: Path) -> Path:
    """
    Fallback Structure-from-Motion implementation using OpenCV SIFT feature matching 
    and triangulation to produce a sparse point cloud PLY when COLMAP is absent.
    """
    print("Running OpenCV Structure-from-Motion fallback pipeline...")
    images = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if len(images) < 2:
        raise ValueError("Need at least 2 images for reconstruction.")

    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []
    img_objs = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        keypoints_list.append(kp)
        descriptors_list.append(des)
        img_objs.append(img)

    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
    points_3d = []
    colors_3d = []

    # Camera intrinsic approximation (assuming standard 60-degree FOV phone camera)
    h, w = img_objs[0].shape[:2]
    focal_length = max(w, h)
    K = np.array([[focal_length, 0, w / 2.0],
                  [0, focal_length, h / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)

    # Match consecutive pairs and triangulate 3D points
    for i in range(len(images) - 1):
        j = i + 1
        if descriptors_list[i] is None or descriptors_list[j] is None:
            continue
            
        matches = flann.knnMatch(descriptors_list[i], descriptors_list[j], k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 15:
            continue

        pts1 = np.float32([keypoints_list[i][m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([keypoints_list[j][m.trainIdx].pt for m in good_matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            continue

        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

        # Projection matrices
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))

        pts1_inliers = pts1[mask.ravel() == 255]
        pts2_inliers = pts2[mask.ravel() == 255]

        if len(pts1_inliers) == 0:
            continue

        pts4d = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        # Extract RGB colors for 3D points
        for idx_pt, pt in enumerate(pts3d):
            if np.all(np.isfinite(pt)) and pt[2] > 0 and pt[2] < focal_length * 10:
                px, py = int(pts1_inliers[idx_pt][0]), int(pts1_inliers[idx_pt][1])
                if 0 <= py < h and 0 <= px < w:
                    bgr = img_objs[i][py, px]
                    rgb = [bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0]
                    points_3d.append(pt)
                    colors_3d.append(rgb)

    if not points_3d:
        # Never fabricate a synthetic point cloud. Report the failure explicitly
        # so the caller can surface a real "SCAN FAILED" result with guidance.
        raise RuntimeError(
            "SCAN FAILED: OpenCV SfM fallback extracted no triangulated points. "
            "Reason: insufficient feature matches across the turntable sequence "
            "(likely low texture, motion blur, or poor lighting). "
            "Suggested fix: capture more frames with diffuse lighting, a textured "
            "background reference, or use the calibrated turntable pipeline with a "
            "scale reference."
        )
    else:
        points_3d = np.array(points_3d)
        colors_3d = np.array(colors_3d)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_3d)

    ply_path = workspace / "sparse_cloud.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"Fallback sparse point cloud created with {len(points_3d)} points: {ply_path}")
    return ply_path


def run_colmap_reconstruction(image_dir: str, workspace_dir: str) -> Path:
    """
    Runs COLMAP sparse reconstruction on turntable images.
    Falls back gracefully if pycolmap or COLMAP system binaries are unavailable.
    """
    image_path = Path(image_dir).resolve()
    workspace = Path(workspace_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    database_path = workspace / "database.db"

    if database_path.exists():
        database_path.unlink()

    if PYCOLMAP_AVAILABLE:
        try:
            print("1. Extracting features with PyCOLMAP...")
            pycolmap.extract_features(database_path, image_path)

            print("2. Exhaustive feature matching...")
            pycolmap.match_exhaustive(database_path)

            print("3. Running incremental mapping...")
            maps = pycolmap.incremental_mapping(database_path, image_path, workspace)

            if maps:
                best_model = maps[0]
                best_model.write(workspace)
                ply_path = workspace / "sparse_cloud.ply"
                best_model.export_PLY(ply_path)
                print(f"COLMAP sparse point cloud saved to {ply_path}")
                return ply_path
            else:
                print("COLMAP incremental mapping produced empty maps. Falling back to SFM module.")
        except Exception as e:
            print(f"PyCOLMAP reconstruction attempt failed: {e}. Falling back to OpenCV SFM.")

    return opencv_sfm_fallback(image_path, workspace)


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "../data/processed"
    wsp = sys.argv[2] if len(sys.argv) > 2 else "../data/workspace"
    run_colmap_reconstruction(inp, wsp)
