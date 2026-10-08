import os
import io
import time
from pathlib import Path
from typing import Optional, Any, Tuple
import numpy as np
from google.cloud import storage
import fsspec

# Allow overriding bucket name via environment variables
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "3dinvenza-datasets")
GCS_CACHE_DIR = Path("d:/3dSCANNER/data/gcs_cache")

# Cache auth result so we don't retry the slow network timeout on every call
_GCS_AUTH_FAILED = False

def get_gcs_client() -> Optional[storage.Client]:
    """Returns an authenticated GCS client, or None if not authenticated.
    Caches auth failure to avoid repeated slow network timeouts."""
    global _GCS_AUTH_FAILED
    if _GCS_AUTH_FAILED:
        return None
    try:
        return storage.Client()
    except Exception as e:
        _GCS_AUTH_FAILED = True
        print(f"[GCS Auth Warning] Could not authenticate Google Cloud Storage: {e}")
        print("Please run 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS.")
        print("[GCS] All subsequent GCS calls will be skipped (local-only mode).")
        return None

def upload_to_gcs(local_path: Path, gcs_blob_name: str, delete_local: bool = True) -> bool:
    """Uploads a local file to GCS and optionally deletes the local file to save space."""
    client = get_gcs_client()
    if not client:
        return False
        
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_blob_name)
        print(f"  [GCS] Uploading {local_path.name} to gs://{GCS_BUCKET_NAME}/{gcs_blob_name}...")
        blob.upload_from_filename(str(local_path))
        
        if delete_local and local_path.exists():
            local_path.unlink()
            print(f"  [GCS] Upload complete. Deleted local file to save space: {local_path.name}")
        return True
    except Exception as e:
        print(f"  [GCS Error] Failed to upload {local_path.name}: {e}")
        return False

def download_from_gcs(gcs_blob_name: str, local_dest_path: Path) -> bool:
    """Downloads a specific file from GCS."""
    client = get_gcs_client()
    if not client:
        return False
        
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_blob_name)
        if not blob.exists():
            print(f"  [GCS Notice] Blob gs://{GCS_BUCKET_NAME}/{gcs_blob_name} does not exist.")
            return False
            
        local_dest_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [GCS] Downloading gs://{GCS_BUCKET_NAME}/{gcs_blob_name}...")
        blob.download_to_filename(str(local_dest_path))
        return True
    except Exception as e:
        print(f"  [GCS Error] Failed to download {gcs_blob_name}: {e}")
        return False

def stream_npz_from_gcs(gcs_blob_name: str, cache_locally: bool = True) -> Optional[Any]:
    """
    Streams an .npz file directly from GCS to memory using fsspec, 
    or uses a local LRU cache if cache_locally=True to prevent GPU starvation.
    """
    if cache_locally:
        GCS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        local_cache_path = GCS_CACHE_DIR / Path(gcs_blob_name).name
        
        # If not in cache, download it
        if not local_cache_path.exists():
            success = download_from_gcs(gcs_blob_name, local_cache_path)
            if not success:
                return None
                
        # Load from fast local cache
        try:
            return np.load(local_cache_path, allow_pickle=True)
        except Exception:
            return None
            
    else:
        # Direct memory streaming (slower, but uses 0 disk space)
        gcs_uri = f"gs://{GCS_BUCKET_NAME}/{gcs_blob_name}"
        try:
            with fsspec.open(gcs_uri, 'rb') as f:
                content = f.read()
                return np.load(io.BytesIO(content), allow_pickle=True)
        except Exception as e:
            print(f"[GCS Stream Error] {e}")
            return None

def list_gcs_files(prefix: str = "") -> list:
    """Lists files in the GCS bucket under a specific prefix."""
    client = get_gcs_client()
    if not client:
        return []
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blobs = bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]
    except Exception as e:
        print(f"[GCS Error] Failed to list files: {e}")
        return []

def enforce_cache_limit(max_gb: float = 50.0):
    """
    LRU Cache manager. Ensures the local GCS cache doesn't exceed max_gb.
    Deletes the oldest files if the limit is exceeded.
    """
    if not GCS_CACHE_DIR.exists():
        return
        
    files = list(GCS_CACHE_DIR.glob("*"))
    if not files:
        return
        
    # Calculate total size
    total_size_bytes = sum(f.stat().st_size for f in files)
    max_bytes = max_gb * 1024 * 1024 * 1024
    
    if total_size_bytes > max_bytes:
        print(f"[GCS Cache] Cache size ({total_size_bytes/1e9:.2f} GB) exceeds limit ({max_gb} GB). Cleaning up...")
        # Sort files by last access time (oldest first)
        files.sort(key=lambda x: x.stat().st_atime)
        
        bytes_to_free = total_size_bytes - max_bytes
        freed = 0
        for f in files:
            size = f.stat().st_size
            try:
                f.unlink()
                freed += size
                if freed >= bytes_to_free:
                    break
            except Exception:
                pass
        print(f"[GCS Cache] Freed {freed/1e9:.2f} GB of local cache.")
