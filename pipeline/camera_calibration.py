"""
Camera Calibration & Lens Undistortion Module

Calculates camera intrinsic matrix K (fx, fy, cx, cy) and distortion coefficients (k1, k2, p1, p2, k3)
using Checkerboard or ChArUco calibration patterns.
Enables metric-accurate ray projection and undistorted image processing for 3DINVENZA.
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CALIB_DIR = Path("d:/3dSCANNER/data/calibration")
CALIB_FILE = CALIB_DIR / "camera_intrinsics.json"


def get_default_camera_matrix(image_width: int = 1920, image_height: int = 1080, fov_deg: float = 65.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a standard smartphone camera matrix based on field of view (FOV).
    fx = fy = (width / 2) / tan(fov / 2)
    cx = width / 2, cy = height / 2
    """
    fov_rad = np.radians(fov_deg)
    focal_length = (image_width / 2.0) / np.tan(fov_rad / 2.0)
    
    K = np.array([
        [focal_length, 0.0, image_width / 2.0],
        [0.0, focal_length, image_height / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros(5, dtype=np.float64)  # Zero distortion for ideal camera
    return K, dist_coeffs


def calibrate_camera_from_checkerboard(
    image_paths: List[Path],
    pattern_size: Tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0
) -> Dict[str, Any]:
    """
    Performs standard OpenCV checkerboard camera calibration across multiple views.
    
    Args:
        image_paths: List of file paths to checkerboard calibration images.
        pattern_size: Number of inner corners (columns, rows). Default: (9, 6).
        square_size_mm: Physical side length of each checkerboard square in mm. Default: 25.0 mm.

    Returns:
        Dictionary containing camera matrix K, distortion coefficients, and RMS reprojection error.
    """
    # Prepare 3D object points in real world space (Z = 0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size_mm

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    img_shape = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            objpoints.append(objp)
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_sub)

    if len(objpoints) < 3:
        # Fallback to default smartphone intrinsics if insufficient calibration frames
        print(f"[Camera Calibration] Warning: Only {len(objpoints)} valid calibration views found. Generating standard smartphone profile.")
        w, h = img_shape if img_shape else (1920, 1080)
        K, dist = get_default_camera_matrix(w, h)
        rms = 0.0
        is_calibrated = False
    else:
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
        rms = float(ret)
        is_calibrated = True
        print(f"[Camera Calibration] Success! RMS Reprojection Error: {rms:.4f} px across {len(objpoints)} frames.")

    calib_data = {
        "is_calibrated": is_calibrated,
        "image_width": img_shape[0] if img_shape else 1920,
        "image_height": img_shape[1] if img_shape else 1080,
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "distortion_coefficients": dist.flatten().tolist(),
        "camera_matrix": K.tolist(),
        "rms_reprojection_error": rms,
        "valid_calibration_frames": len(objpoints)
    }

    save_calibration(calib_data)
    return calib_data


def save_calibration(calib_data: Dict[str, Any], filepath: Optional[Path] = None):
    """Saves camera calibration profile to disk."""
    out_file = filepath or CALIB_FILE
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(calib_data, f, indent=4)
    print(f"[Camera Calibration] Saved calibration profile to: {out_file}")


def load_calibration(filepath: Optional[Path] = None, img_shape: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """Loads camera calibration profile from disk, or constructs default smartphone intrinsics."""
    in_file = filepath or CALIB_FILE
    if in_file.exists():
        try:
            with open(in_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Camera Calibration] Notice reading {in_file}: {e}")

    # Construct default calibrated profile
    w, h = img_shape if img_shape else (1920, 1080)
    K, dist = get_default_camera_matrix(w, h)
    default_data = {
        "is_calibrated": False,
        "image_width": w,
        "image_height": h,
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "distortion_coefficients": dist.flatten().tolist(),
        "camera_matrix": K.tolist(),
        "rms_reprojection_error": 0.0,
        "profile_name": "Standard Smartphone Camera"
    }
    return default_data


def undistort_image(img_bgr: np.ndarray, calib_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Applies radial and tangential lens undistortion to an input image using camera intrinsics.
    """
    h, w = img_bgr.shape[:2]
    if calib_data is None:
        calib_data = load_calibration(img_shape=(w, h))

    K = np.array(calib_data["camera_matrix"], dtype=np.float64)
    dist = np.array(calib_data["distortion_coefficients"], dtype=np.float64)

    # Only apply cv2.undistort if non-zero distortion is present
    if np.any(np.abs(dist) > 1e-6):
        undistorted = cv2.undistort(img_bgr, K, dist, None, K)
        return undistorted
    return img_bgr


if __name__ == "__main__":
    calib = load_calibration()
    print("--- 3DINVENZA Camera Calibration Profile ---")
    print(json.dumps(calib, indent=4))
