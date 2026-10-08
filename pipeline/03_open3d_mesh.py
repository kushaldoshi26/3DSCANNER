import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.open3d_mesh import pointcloud_to_mesh

if __name__ == "__main__":
    ply_path = sys.argv[1] if len(sys.argv) > 1 else "../data/workspace/sparse_cloud.ply"
    output_stl = sys.argv[2] if len(sys.argv) > 2 else "../data/output/model.stl"
    pointcloud_to_mesh(ply_path, output_stl)
