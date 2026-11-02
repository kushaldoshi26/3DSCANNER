"""Non-destructive, alpha-aware preparation for single-image reconstruction."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

def preprocess_for_generation(image_path, output_dir, *, padding_ratio=0.12, keep_original=True):
    from pipeline.preprocess import process_single_image
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    paths = {k: str(out / f"{stem}_{name}.png") for k,name in
             [('original_rgb','original'),('foreground_rgba','foreground'),('soft_mask','mask'),('padded_rgba','padded')]}
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert('RGBA')
    image.convert('RGB').save(paths['original_rgb'])
    process_single_image(str(image_path), paths['foreground_rgba'])
    foreground = Image.open(paths['foreground_rgba']).convert('RGBA')
    alpha = np.asarray(foreground)[:,:,3]
    ys,xs = np.where(alpha > 15)
    if not len(xs):
        raise ValueError('No foreground object detected. Use a clear image or a transparent PNG.')
    bbox = (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)
    crop = foreground.crop(bbox)
    side = max(2, int(np.ceil(max(crop.size) * (1 + 2 * max(0., padding_ratio)))))
    padded = Image.new('RGBA',(side,side),(0,0,0,0))
    padded.paste(crop,((side-crop.width)//2,(side-crop.height)//2))
    padded.save(paths['padded_rgba'])
    padded.getchannel('A').save(paths['soft_mask'])
    return paths
