import os
import sys
from pathlib import Path

# Add project root directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preprocess import process_images

if __name__ == "__main__":
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/raw"
    proc_dir = sys.argv[2] if len(sys.argv) > 2 else "../data/processed"
    process_images(raw_dir, proc_dir)
