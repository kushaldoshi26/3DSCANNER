"""
3DINVENZA Master Dataset Downloader & Cloud Ingestion Pipeline

Downloads the top 10 critical datasets for multi-view, turntable, and organic reconstruction,
processes them into point clouds, and streams them directly to Google Cloud Storage.

Core 10-Dataset Setup:
1. OmniObject3D (100GB subset)
2. MVImgNet 2.0 (100GB subset)
3. Amazon Berkeley Objects (ABO) (75GB subset)
4. Google Scanned Objects (7.5GB)
5. CO3D (8.9GB subset)
6. ShapeNetCore (30GB subset)
7. Pix3D (20GB)
8. Completion3D (5GB)
9. PCN (5GB)
10. THuman2.0 (30-50GB subset)
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.download_kaggle_datasets import download_from_kaggle
from pipeline.gcs_utils import upload_to_gcs

DATASETS_DIR = Path("d:/3dSCANNER/data/datasets")
KAGGLE_CACHE_DIR = DATASETS_DIR / "kaggle_raw"

# We map datasets to their Kaggle/HF slugs where available.
# In a real enterprise setup, some of these would be pulled via AWS S3 / HuggingFace datasets.
DATASET_SPECS = {
    "omniobject3d": {
        "slug": "omniobject3d/subset", # Placeholder slug
        "desc": "Real scanned objects",
        "size_gb": 100
    },
    "mvimgnet_2": {
        "slug": "mvimgnet/mvimgnet-2", # Placeholder slug
        "desc": "Multi-view RGB + masks + poses + point clouds",
        "size_gb": 100
    },
    "amazon_berkeley_objects": {
        "slug": "amazon/berkeley-objects", # Placeholder slug
        "desc": "360° product/turntable data",
        "size_gb": 75
    },
    "google_scanned_objects": {
        "slug": "google/scanned-objects", # Placeholder slug
        "desc": "Real scanned objects",
        "size_gb": 7.5
    },
    "co3d_v2": {
        "slug": "facebook/co3d-subset", # Placeholder slug
        "desc": "Real multi-view reconstruction",
        "size_gb": 8.9
    },
    "shapenet_core": {
        "slug": "shapenet/shapenet-core", # Placeholder slug
        "desc": "Clean 3D geometry",
        "size_gb": 30
    },
    "pix3d": {
        "slug": "pix3d/pix3d", # Placeholder slug
        "desc": "Real image <-> 3D model",
        "size_gb": 20
    },
    "completion3d": {
        "slug": "completion3d/benchmark", # Placeholder slug
        "desc": "Partial -> complete point cloud",
        "size_gb": 5
    },
    "pcn": {
        "slug": "pcn/point-cloud-completion", # Placeholder slug
        "desc": "Point-cloud completion",
        "size_gb": 5
    },
    "thuman2": {
        "slug": "thuman/thuman-2", # Placeholder slug
        "desc": "Human/body/head reconstruction",
        "size_gb": 40
    }
}

def ingest_master_datasets():
    print("===============================================================")
    print("  3DINVENZA CLOUD-NATIVE MASTER DATASET INGESTION PIPELINE")
    print("===============================================================")
    print("Initializing pipeline for Core 10 Datasets...")
    print("WARNING: This will stream approximately 400GB of data. Ensure your GCS Bucket is configured.\n")
    
    KAGGLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    total_gb_processed = 0
    
    for key, spec in DATASET_SPECS.items():
        print(f"--- Ingesting {key.upper()} ---")
        print(f"Description: {spec['desc']}")
        print(f"Target Size: {spec['size_gb']} GB")
        
        # 1. Download to local cache
        # Note: In production, you would stream this directly to GCS via memory buffers 
        # to avoid hitting local disk limits entirely.
        success = download_from_kaggle(spec["slug"], KAGGLE_CACHE_DIR / key)
        
        if success:
            # 2. Extract, Process into .NPZ, and Upload to GCS
            # (In this script we just mock the upload of the raw zip for demonstration)
            zip_file = KAGGLE_CACHE_DIR / key / f"{spec['slug'].split('/')[-1]}.zip"
            if zip_file.exists():
                gcs_blob_name = f"raw_archives/{key}.zip"
                upload_to_gcs(zip_file, gcs_blob_name, delete_local=True)
                total_gb_processed += spec["size_gb"]
                
    print("\n===============================================================")
    print(f"MASTER INGESTION COMPLETE. Processed ~{total_gb_processed} GB to Cloud Storage.")
    print("===============================================================")

if __name__ == "__main__":
    ingest_master_datasets()
