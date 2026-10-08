import open3d as o3d
import numpy as np
from pathlib import Path
import sys

def pointcloud_to_mesh(
    ply_path: str,
    output_stl: str,
    depth: int = 9,
    taubin_iterations: int = 1,
) -> str:
    """
    Converts PLY point cloud to mesh using Poisson Surface Reconstruction.

    Defaults: Poisson depth=9 (preserves real geometry/sharp edges without
    over-smoothing), Taubin 1 iteration (volume-preserving; removes staircase
    artifacts while keeping measured features). Both are configurable so the
    caller can trade detail vs. smoothness explicitly.
    """
    ply_p = Path(ply_path)
    out_p = Path(output_stl)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading point cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(str(ply_p))

    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud at {ply_path} is empty.")

    print(f"Loaded {len(pcd.points)} points. Cleaning statistical outliers...")
    # Remove statistical outliers from point cloud
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(ind) > 0:
        pcd = pcd.select_by_index(ind)

    print("Estimating point cloud normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(100)

    print(f"Running Poisson Surface Reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1, linear_fit=True
    )

    if len(mesh.triangles) == 0:
        print("Poisson reconstruction produced 0 triangles. Falling back to Ball Pivoting algorithm...")
        radii = [0.005, 0.01, 0.02, 0.05]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

    # Filter out low-density boundary noise — crop bottom 5% for cleaner facial geometry
    if len(densities) > 0:
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.05)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Apply Taubin smoothing to eliminate staircase stepping without shrinking features.
    # Default 1 iteration (volume-preserving); increase only when extra smoothing is desired.
    print(f"Applying Taubin smoothing ({taubin_iterations} iteration(s)) for clean surface texture...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=taubin_iterations)

    print("Computing mesh vertex normals...")
    mesh.compute_vertex_normals()

    print(f"Saving generated mesh to {out_p}...")
    o3d.io.write_triangle_mesh(str(out_p), mesh)
    
    # Also write OBJ/PLY format if needed
    obj_p = out_p.with_suffix(".obj")
    o3d.io.write_triangle_mesh(str(obj_p), mesh)

    print(f"Successfully generated mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
    return str(out_p)

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "../data/workspace/sparse_cloud.ply"
    outp = sys.argv[2] if len(sys.argv) > 2 else "../data/output/model.stl"
    pointcloud_to_mesh(inp, outp)
