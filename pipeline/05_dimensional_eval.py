import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.dimensional_eval import auto_measure_mesh, format_dimensional_report


if __name__ == "__main__":
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "data/output/ai_trained_model.stl"
    
    try:
        report = auto_measure_mesh(mesh_path)
        print(format_dimensional_report(report))
    except Exception as e:
        print(f"Dimensional evaluation failed for '{mesh_path}': {e}")
        sys.exit(1)
