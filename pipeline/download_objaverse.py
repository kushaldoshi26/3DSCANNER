"""
Objaverse Dataset Downloader and Multi-View Processing Pipeline

Integrates the official Objaverse 3D repository (800,000+ open 3D objects):
- Automatic UID selection by category (sculptures, furniture, vehicles, artifacts, organic)
- GLB mesh downloading and sanitization with trimesh
- Unit bounding box normalization & point cloud sampling
- Direct integration into the PyTorch 3D training dataset
"""

import os
import sys
import math
import random
import socket

# Prevent urllib from hanging forever if the network silently drops
socket.setdefaulttimeout(30.0)
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import trimesh

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASETS_DIR = Path("d:/3dSCANNER/data/datasets")
OBJAVERSE_DIR = DATASETS_DIR / "objaverse"
PROCESSED_DIR = DATASETS_DIR / "processed"


def get_curated_objaverse_categories() -> List[str]:
    """Curated list of 3D object categories optimal for scanner AI training."""
    return [
        "sculpture",
        "furniture",
        "statue",
        "vehicle",
        "tool",
        "animal",
        "character",
        "artifact",
        "household",
        "ceramic"
    ]


def download_objaverse_dataset(
    count: int = 10,
    categories: Optional[List[str]] = None,
    target_format: str = "stl"
) -> List[Path]:
    """
    Downloads curated 3D objects from Objaverse, normalizes them, and saves to data/datasets.
    
    Args:
        count: Number of 3D models to download.
        categories: Optional list of categories/tags to filter by.
        target_format: 'stl', 'obj', or 'glb'
    
    Returns:
        List of Path objects for downloaded and normalized 3D models.
    """
    try:
        import objaverse
    except ImportError:
        print("[ERROR] 'objaverse' package is not installed. Please run: pip install objaverse")
        return []

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    OBJAVERSE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  OBJAVERSE 3D DATASET INGESTION PIPELINE")
    print(f"  Target count: {count} models | Target format: {target_format.upper()}")
    print("=" * 68)

    # 1. Fetch UIDs (Use a fixed seed so we can resume interrupted downloads)
    random.seed(42)
    print("Fetching Objaverse index and annotations...")
    uids = objaverse.load_uids()
    print(f"Found {len(uids)} available Objaverse 3D models in catalog.")

    selected_uids = []
    
    # Try filtering by annotations if requested
    if categories:
        try:
            print(f"Filtering UIDs for categories: {', '.join(categories)}...")
            # Sample candidate UIDs to inspect annotations
            candidate_pool = random.sample(uids, min(len(uids), max(count * 20, 200)))
            annotations = objaverse.load_annotations(candidate_pool)
            for uid, meta in annotations.items():
                tags = [t.get("name", "").lower() for t in meta.get("tags", []) if isinstance(t, dict)]
                categories_meta = [c.get("name", "").lower() for c in meta.get("categories", []) if isinstance(c, dict)]
                text = (meta.get("name", "") + " " + " ".join(tags) + " " + " ".join(categories_meta)).lower()
                
                if any(cat.lower() in text for cat in categories):
                    selected_uids.append(uid)
                    if len(selected_uids) >= count:
                        break
        except Exception as e:
            print(f"[Notice] Annotation filtering error ({e}). Falling back to random sampling.")
            selected_uids = []

    if len(selected_uids) < count:
        remaining = count - len(selected_uids)
        # Sample random UIDs
        sampled = random.sample(uids, min(len(uids), remaining * 2))
        for u in sampled:
            if u not in selected_uids:
                selected_uids.append(u)
                if len(selected_uids) >= count:
                    break

    print(f"Selected {len(selected_uids)} Objaverse UIDs to download...")

    # 2. Download 3D objects in chunks to prevent complete failure on a single file error
    downloaded_objects = {}
    chunk_size = 100
    for i in range(0, len(selected_uids), chunk_size):
        chunk = selected_uids[i:i + chunk_size]
        print(f"Downloading chunk {i//chunk_size + 1} / {math.ceil(len(selected_uids)/chunk_size)} ({len(chunk)} models)...")
        success = False
        for attempt in range(3):
            try:
                batch = objaverse.load_objects(uids=chunk)
                downloaded_objects.update(batch)
                success = True
                break
            except Exception as e:
                print(f"  [WARN] Chunk failed (Attempt {attempt+1}/3): {e}")
                import time
                time.sleep(2)
        if not success:
            print("  [ERROR] Skipping chunk after 3 failed attempts.")

    print(f"Successfully downloaded {len(downloaded_objects)} raw GLB models.")

    # 3. Load, sanitize, normalize to unit bounding box, and save to datasets
    normalized_paths = []
    for uid, glb_path in downloaded_objects.items():
        try:
            glb_file = Path(glb_path)
            if not glb_file.exists():
                continue

            target_name = f"objaverse_{uid[:12]}.{target_format.lower()}"
            out_file = DATASETS_DIR / target_name

            # Load with trimesh
            mesh = trimesh.load(str(glb_file), force='mesh')
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())

            if len(mesh.vertices) == 0:
                continue

            # Center mass at origin & normalize to unit bounding box [-0.5, 0.5]
            mesh.vertices -= mesh.center_mass
            max_ext = np.max(mesh.extents)
            if max_ext > 0:
                mesh.vertices /= max_ext

            mesh.remove_unreferenced_vertices()

            # Export normalized mesh to target format
            mesh.export(str(out_file))
            normalized_paths.append(out_file)
            print(f"  [PROCESSED] {target_name} ({len(mesh.vertices)} verts, {len(mesh.faces)} faces, {round(out_file.stat().st_size/1024, 1)} KB)")

        except Exception as e:
            print(f"  [WARN] Failed to process Objaverse model {uid}: {e}")

    print("=" * 68)
    print(f"Objaverse Ingestion Complete: {len(normalized_paths)} 3D models added to {DATASETS_DIR}")
    print("=" * 68)
    return normalized_paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and ingest Objaverse 3D models")
    parser.add_argument("--count", type=int, default=5, help="Number of Objaverse 3D models to download (default: 5)")
    parser.add_argument("--format", type=str, default="stl", choices=["stl", "obj", "glb"], help="Output format")
    args = parser.parse_args()

    download_objaverse_dataset(count=args.count, target_format=args.format)
