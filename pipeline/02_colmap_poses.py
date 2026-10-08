import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.colmap_poses import run_colmap_reconstruction

if __name__ == "__main__":
    proc_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/processed"
    workspace_dir = sys.argv[2] if len(sys.argv) > 2 else "../data/workspace"
    run_colmap_reconstruction(proc_dir, workspace_dir)
