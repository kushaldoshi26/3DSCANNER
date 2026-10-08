"""
Scan Quality & Confidence Analytics Module

Evaluates 3D capture quality, coverage completeness, calibration integrity,
and scale confidence for 3DINVENZA.
Generates an overall Scan Quality Score (0–100%) and actionable user diagnostics.
"""

import sys
from typing import Dict, Any, List, Optional
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) if 'Path' in globals() else None
from pathlib import Path


def evaluate_scan_quality(
    images_count: int,
    expected_images: int = 24,
    mask_coverage_avg: float = 0.95,
    is_camera_calibrated: bool = True,
    rms_reprojection: float = 0.45,
    point_cloud_size: int = 45000,
    mesh_faces: int = 80000,
    is_watertight: bool = True,
    scale_confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Computes rigorous multi-factor scan quality diagnostics and confidence scoring.
    """
    # 1. Multi-View Frame Coverage (0 - 100%)
    frame_ratio = min(1.0, images_count / max(expected_images, 1))
    coverage_score = round(frame_ratio * 100.0, 1)

    # 2. Mask Quality (0 - 100%)
    mask_score = round(min(1.0, mask_coverage_avg) * 100.0, 1)

    # 3. Camera Geometry & Feature Consistency (0 - 100%)
    if is_camera_calibrated:
        calib_score = round(max(0.0, 1.0 - (rms_reprojection / 2.0)) * 100.0, 1)
    else:
        calib_score = 75.0  # Standard fallback default

    # 4. Point Cloud Density & Mesh Completeness (0 - 100%)
    if point_cloud_size > 20000:
        geometry_score = 95.0 if is_watertight else 85.0
    elif point_cloud_size > 5000:
        geometry_score = 75.0
    else:
        geometry_score = 45.0

    # 5. Scale Confidence (0 - 100%)
    scale_score = round(scale_confidence * 100.0, 1)

    # 6. Weighted Overall Scan Quality Score
    overall_score = round(
        0.25 * coverage_score +
        0.20 * mask_score +
        0.20 * calib_score +
        0.20 * geometry_score +
        0.15 * scale_score
    )
    overall_score = max(0, min(100, overall_score))

    # Diagnostics & Warnings
    warnings = []
    suggestions = []

    if images_count < 24:
        warnings.append(f"Low image count ({images_count}/{expected_images} frames).")
        suggestions.append("Capture at least 24 frames at 15-deg turntable increments for complete 360-deg coverage.")

    if not is_camera_calibrated:
        suggestions.append("Calibrate smartphone camera using a checkerboard pattern for sub-millimeter metrology.")

    if not is_watertight:
        warnings.append("Partial occlusions detected on object base or thin geometry.")
        suggestions.append("Capture additional photos at 45-deg camera elevation to resolve top/bottom cavities.")

    if point_cloud_size < 10000:
        warnings.append("Low point cloud density.")
        suggestions.append("Ensure balanced, diffuse studio lighting without intense specular glare.")

    return {
        "overall_score": overall_score,
        "metrics": {
            "coverage_percent": coverage_score,
            "mask_quality_percent": mask_score,
            "calibration_percent": calib_score,
            "geometry_percent": geometry_score,
            "scale_percent": scale_score
        },
        "images_captured": f"{images_count}/{expected_images}",
        "is_calibrated": is_camera_calibrated,
        "point_cloud_points": point_cloud_size,
        "mesh_faces": mesh_faces,
        "is_watertight": is_watertight,
        "scale_confidence": scale_confidence,
        "warnings": warnings,
        "suggestions": suggestions
    }


def format_scan_quality_report(report: Dict[str, Any]) -> str:
    """Formats scan quality metrics into a formatted ASCII diagnostic box."""
    m = report["metrics"]
    lines = [
        "+" + "=" * 62 + "+",
        f"|             3DINVENZA SCAN QUALITY SCORE: {report['overall_score']:>3d}%                 |",
        "+" + "=" * 62 + "+",
        f"|  * Frame Coverage (Images {report['images_captured']}):           {m['coverage_percent']:>5.1f}%       |",
        f"|  * Mask & Silhouette Quality:                  {m['mask_quality_percent']:>5.1f}%       |",
        f"|  * Camera Geometry & Calibration:              {m['calibration_percent']:>5.1f}%       |",
        f"|  * Point Cloud Density & Geometry:             {m['geometry_percent']:>5.1f}%       |",
        f"|  * Physical Scale Metrology:                   {m['scale_percent']:>5.1f}%       |",
        "+" + "=" * 62 + "+"
    ]
    if report["warnings"]:
        lines.append("Warnings:")
        for w in report["warnings"]:
            lines.append(f"  [!] {w}")
    if report["suggestions"]:
        lines.append("Suggestions:")
        for s in report["suggestions"]:
            lines.append(f"  [+] {s}")
    return "\n".join(lines)


if __name__ == "__main__":
    rep = evaluate_scan_quality(images_count=24, expected_images=24, point_cloud_size=35000, is_watertight=True)
    print(format_scan_quality_report(rep))
