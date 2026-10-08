import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ai_refine import refine_mesh

if __name__ == "__main__":
    input_mesh = sys.argv[1] if len(sys.argv) > 1 else "../data/output/model.stl"
    output_mesh = sys.argv[2] if len(sys.argv) > 2 else "../data/output/refined_model.stl"
    refine_mesh(input_mesh, output_mesh)
