import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: 'rembg' library not installed. Falling back to OpenCV segmentation.")


def remove_background_opencv(img: np.ndarray) -> np.ndarray:
    """
    Fallback background removal using Otsu thresholding & GrabCut.
    Produces a 4-channel RGBA image with transparent background.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu threshold to estimate foreground mask
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Create mask for GrabCut
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    except Exception:
        fg_mask = thresh

    # Smooth mask edges
    fg_mask = cv2.GaussianBlur(fg_mask, (3, 3), 0)

    # Convert BGR to RGBA
    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, fg_mask])
    return rgba


def process_single_image(input_path: str, output_path: str) -> str:
    """
    Process a single image to remove background, isolate foreground object,
    and eliminate turntable base platform artifacts.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")

    if REMBG_AVAILABLE:
        try:
            output_img = remove(img)
        except Exception as e:
            print(f"Rembg processing failed for {input_path}: {e}. Using OpenCV fallback.")
            output_img = remove_background_opencv(img)
    else:
        output_img = remove_background_opencv(img)

    output_img = np.array(output_img, copy=True)
    if output_img.shape[2] == 4:
        # Suppress bright saturated turntable colors (e.g. red/green turntable base in lower region)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_red1 = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([14, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([165, 60, 40]), np.array([180, 255, 255]))
        turntable_mask = mask_red1 | mask_red2

        # Apply turntable color suppression specifically to bottom 45% of frame
        h_img = img.shape[0]
        turntable_zone = int(h_img * 0.55)
        turntable_mask[:turntable_zone, :] = 0

        # Erode and clean turntable mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        turntable_mask = cv2.dilate(turntable_mask, kernel, iterations=2)
        output_img[:, :, 3] = np.where(turntable_mask > 0, 0, output_img[:, :, 3])

    cv2.imwrite(output_path, output_img)
    return output_path


def process_images(input_dir: str, output_dir: str) -> List[str]:
    """
    Process all turntable images in input_dir and save isolated subject PNGs into output_dir.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    
    files = sorted([f for f in in_path.iterdir() if f.suffix.lower() in supported_exts])
    if not files:
        print(f"No valid images found in {input_dir}")
        return processed_files

    print(f"Starting background removal for {len(files)} images...")
    for idx, file_path in enumerate(files, 1):
        out_file = out_path / f"{file_path.stem}.png"
        try:
            process_single_image(str(file_path), str(out_file))
            processed_files.append(str(out_file))
            print(f"[{idx}/{len(files)}] Processed: {file_path.name}")
        except Exception as e:
            print(f"[{idx}/{len(files)}] Failed to process {file_path.name}: {e}")

    print(f"Successfully processed {len(processed_files)} images to {output_dir}")
    return processed_files


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "../data/raw"
    outp = sys.argv[2] if len(sys.argv) > 2 else "../data/processed"
    process_images(inp, outp)
