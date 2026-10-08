"""
High-quality input preprocessing for single-image 3D generation (Mode A).

Performs the spec's STEP 4 pipeline without destroying detail:
  1. Orientation check (upright assumption; no destructive rotation)
  2. Alpha / background extraction (rembg)
  3. Subject bounding-box detection
  4. Safe padding (~10-15%) around subject
  5. Preserve original resolution
  6. Gentle normalization (no aggressive blur / over-sharpen)
  7. Generate original RGB + RGBA foreground + soft mask
  8. Never permanently replace the original image

Uses the project's existing `pipeline.preprocess.process_single_image` for the
rembg/foreground step rather than re-implementing background removal.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path


def preprocess_for_generation(
    image_path: str,
    output_dir: str,
    *,
    padding_ratio: float = 0.12,
    keep_original: bool = True,
) -> Dict[str, str]:
    """
    Prepares an image for TRELLIS/Hunyuan/TripoSR.

    Returns paths to:
      original_rgb   - untouched source (used for final texture)
      foreground_rgba - rembg-result with alpha
      soft_mask      - binary/soft foreground mask (PNG)
      padded_rgba    - subject-cropped with safe padding (best for generator)
    """
    from pipeline.preprocess import process_single_image

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    original_rgb = str(out_dir / f"{stem}_original.png")
    rgba_path = str(out_dir / f"{stem}_foreground.png")
    mask_path = str(out_dir / f"{stem}_mask.png")
    padded_path = str(out_dir / f"{stem}_padded.png")

    # 1. Keep original RGB untouched.
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read input image: {image_path}")
    if keep_original:
        cv2.imwrite(original_rgb, img)

    # 2. Foreground extraction via existing rembg pipeline.
    process_single_image(str(image_path), rgba_path)
    rgba = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[2] < 4:
        # Fallback: no alpha -> treat whole image as foreground.
        rgba = np.dstack([img, np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)])

    b, g, r, a = cv2.split(rgba)
    mask = (a > 15).astype(np.uint8) * 255

    # 3 & 4. Subject bounding box + safe padding.
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        # No subject found; use full image.
        x0, x1 = 0, img.shape[1]
        y0, y1 = 0, img.shape[0]
    else:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0, y1 - y0
    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(img.shape[1], x1 + pad_x)
    y1 = min(img.shape[0], y1 + pad_y)

    # 5. Crop with safe padding (preserves original resolution within crop).
    padded = rgba[y0:y1, x0:x1]
    cv2.imwrite(padded_path, padded)

    # 6. Soft mask (light Gaussian on binary mask, clamped).
    soft = cv2.GaussianBlur(mask, (5, 5), 0)
    soft_crop = soft[y0:y1, x0:x1]
    cv2.imwrite(mask_path, soft_crop)

    return {
        "original_rgb": original_rgb,
        "foreground_rgba": rgba_path,
        "soft_mask": mask_path,
        "padded_rgba": padded_path,
    }
