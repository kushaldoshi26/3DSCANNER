"""
Procedural Complex 3D Mesh Dataset Generator

Generates 20+ complex parametric 3D meshes for training the Neural 3D Reconstruction network.
These meshes go beyond simple primitives (cube, sphere, cylinder, torus) to include:
- Multi-body compositions (sphere + cylinder, stacked shapes)
- Boolean-like unions (multi-component assemblies)
- Organic shapes (perturbed spheres, twisted forms)
- Mechanical part approximations (stepped cylinders, flanged tubes)
- Varied topologies (high genus surfaces, hollowed shapes)

All meshes are exported as STL files to data/datasets/ for automatic pickup by the training pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trimesh
import numpy as np
from trimesh.creation import (
    box, cylinder, icosphere, capsule, annulus
)

DATASETS_DIR = Path("d:/3dSCANNER/data/datasets")


def _save_mesh(mesh: trimesh.Trimesh, name: str):
    """Normalize mesh to unit bounding box and save as STL."""
    if len(mesh.vertices) == 0:
        print(f"  Skipping {name}: empty mesh")
        return
    # Center and scale to unit bounding box
    mesh.vertices -= mesh.center_mass
    max_ext = np.max(mesh.extents)
    if max_ext > 0:
        mesh.vertices /= max_ext
    out_path = DATASETS_DIR / f"{name}.stl"
    mesh.export(str(out_path))
    print(f"  [OK] {name}.stl ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")


def generate_complex_datasets():
    """Generate 20+ complex parametric meshes into data/datasets/."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Generating Complex 3D Training Dataset Meshes")
    print("=" * 60)

    # 1. Capsule (rounded cylinder — pharmaceutical/organic shape)
    m = capsule(height=2.0, radius=0.5, count=[16, 16])
    _save_mesh(m, "capsule_organic")

    # 2. Tall thin cylinder (pen/rod shape)
    m = cylinder(radius=0.15, height=3.0, sections=32)
    _save_mesh(m, "thin_rod_cylinder")

    # 3. Flat disc (coin shape)
    m = cylinder(radius=1.5, height=0.1, sections=48)
    _save_mesh(m, "flat_disc_coin")

    # 4. Multi-stacked cylinders (stepped column)
    parts = []
    for i, (r, h) in enumerate([(0.8, 0.5), (0.6, 0.8), (0.4, 1.0), (0.25, 0.6)]):
        c = cylinder(radius=r, height=h, sections=32)
        offset = sum(hh for _, hh in [(0.8, 0.5), (0.6, 0.8), (0.4, 1.0), (0.25, 0.6)][:i]) + h / 2
        c.apply_translation([0, 0, offset])
        parts.append(c)
    m = trimesh.util.concatenate(parts)
    _save_mesh(m, "stepped_column")

    # 5. Sphere with extruded cylinder (lollipop/mushroom)
    sphere = icosphere(subdivisions=3, radius=0.6)
    sphere.apply_translation([0, 0, 1.5])
    stick = cylinder(radius=0.1, height=1.5, sections=16)
    stick.apply_translation([0, 0, 0.75])
    m = trimesh.util.concatenate([sphere, stick])
    _save_mesh(m, "mushroom_lollipop")

    # 6. L-shaped bracket
    h_bar = box(extents=[2.0, 0.3, 0.3])
    h_bar.apply_translation([1.0, 0, 0])
    v_bar = box(extents=[0.3, 0.3, 2.0])
    v_bar.apply_translation([0, 0, 1.0])
    m = trimesh.util.concatenate([h_bar, v_bar])
    _save_mesh(m, "l_bracket")

    # 7. T-shaped connector
    top = box(extents=[2.0, 0.3, 0.3])
    stem = box(extents=[0.3, 0.3, 1.5])
    stem.apply_translation([0, 0, -0.9])
    m = trimesh.util.concatenate([top, stem])
    _save_mesh(m, "t_connector")

    # 8. Cross / Plus shape
    h = box(extents=[3.0, 0.4, 0.4])
    v = box(extents=[0.4, 0.4, 3.0])
    m = trimesh.util.concatenate([h, v])
    _save_mesh(m, "cross_plus")

    # 9. Perturbed sphere (organic/asteroid shape)
    m = icosphere(subdivisions=3, radius=1.0)
    noise = np.random.RandomState(42).randn(len(m.vertices), 3) * 0.08
    m.vertices += noise
    _save_mesh(m, "asteroid_organic")

    # 10. High-subdivision icosphere (smooth organic)
    m = icosphere(subdivisions=4, radius=1.0)
    _save_mesh(m, "smooth_hires_sphere")

    # 11. Elongated ellipsoid
    m = icosphere(subdivisions=3, radius=1.0)
    m.vertices[:, 2] *= 2.5  # Stretch Z axis
    _save_mesh(m, "elongated_ellipsoid")

    # 12. Flattened ellipsoid (pebble/stone)
    m = icosphere(subdivisions=3, radius=1.0)
    m.vertices[:, 2] *= 0.3  # Flatten Z
    m.vertices[:, 0] *= 1.5  # Stretch X
    _save_mesh(m, "flat_pebble_stone")

    # 13. Double sphere (dumbbell/figure-8)
    s1 = icosphere(subdivisions=3, radius=0.5)
    s1.apply_translation([-0.7, 0, 0])
    s2 = icosphere(subdivisions=3, radius=0.5)
    s2.apply_translation([0.7, 0, 0])
    bridge = cylinder(radius=0.15, height=1.0, sections=16)
    bridge.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    m = trimesh.util.concatenate([s1, s2, bridge])
    _save_mesh(m, "dumbbell_figure8")

    # 14. Pyramid (4-sided)
    verts = np.array([
        [1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0],  # base
        [0, 0, 1.5]  # apex
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # base
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]  # sides
    ])
    m = trimesh.Trimesh(vertices=verts, faces=faces)
    _save_mesh(m, "pyramid_4sided")

    # 15. Hexagonal prism
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    r = 1.0
    h_half = 0.8
    top_verts = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.full(6, h_half)])
    bot_verts = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.full(6, -h_half)])
    verts = np.vstack([top_verts, bot_verts])
    faces = []
    for i in range(6):
        j = (i + 1) % 6
        faces.append([i, j, j + 6])
        faces.append([i, j + 6, i + 6])
    # Top and bottom caps (fan triangulation)
    for i in range(1, 5):
        faces.append([0, i, i + 1])
        faces.append([6, 6 + i + 1, 6 + i])
    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    _save_mesh(m, "hexagonal_prism")

    # 16. Wedge / ramp shape
    verts = np.array([
        [0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0],  # base
        [0, 0, 1], [0, 1, 1]  # top edge (wedge)
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [0, 4, 5], [0, 5, 3],  # left side
        [0, 1, 4],             # front triangle
        [3, 5, 2],             # back triangle
        [1, 2, 5], [1, 5, 4]   # slope
    ])
    m = trimesh.Trimesh(vertices=verts, faces=faces)
    _save_mesh(m, "wedge_ramp")

    # 17. Multi-cube assembly (building blocks)
    cubes = []
    positions = [
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1]
    ]
    for pos in positions:
        c = box(extents=[0.9, 0.9, 0.9])
        c.apply_translation(pos)
        cubes.append(c)
    m = trimesh.util.concatenate(cubes)
    _save_mesh(m, "building_blocks")

    # 18. Ring with square cross-section (washer)
    outer = cylinder(radius=1.2, height=0.4, sections=48)
    inner = cylinder(radius=0.6, height=0.5, sections=48)
    # Approximate ring by concatenating (visual approximation)
    parts = []
    for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        segment = box(extents=[0.3, 0.12, 0.4])
        cx = 0.9 * np.cos(angle)
        cy = 0.9 * np.sin(angle)
        segment.apply_translation([cx, cy, 0])
        rot = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        segment.apply_transform(rot)
        parts.append(segment)
    m = trimesh.util.concatenate(parts)
    _save_mesh(m, "square_washer_ring")

    # 19. Twisted bar (helical extrusion approximation)
    segments = []
    n_seg = 20
    for i in range(n_seg):
        t = i / n_seg
        seg = box(extents=[0.3, 0.08, 0.15])
        seg.apply_translation([0, 0, t * 3.0])
        angle = t * np.pi * 2
        rot = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        seg.apply_transform(rot)
        segments.append(seg)
    m = trimesh.util.concatenate(segments)
    _save_mesh(m, "twisted_helix_bar")

    # 20. Arch / bridge shape
    parts = []
    # Two pillars
    p1 = box(extents=[0.3, 0.3, 1.5])
    p1.apply_translation([-1.0, 0, 0.75])
    parts.append(p1)
    p2 = box(extents=[0.3, 0.3, 1.5])
    p2.apply_translation([1.0, 0, 0.75])
    parts.append(p2)
    # Arch top (semicircle of small boxes)
    for angle in np.linspace(0, np.pi, 12):
        seg = box(extents=[0.25, 0.3, 0.12])
        x = 1.0 * np.cos(angle)
        z = 1.5 + 0.5 * np.sin(angle)
        seg.apply_translation([x, 0, z])
        rot = trimesh.transformations.rotation_matrix(angle - np.pi / 2, [0, 1, 0])
        seg.apply_transform(rot)
        parts.append(seg)
    m = trimesh.util.concatenate(parts)
    _save_mesh(m, "arch_bridge")

    # 21. Cone
    verts = []
    faces_list = []
    n_sides = 32
    apex = [0, 0, 2.0]
    for i in range(n_sides):
        angle = 2 * np.pi * i / n_sides
        verts.append([np.cos(angle), np.sin(angle), 0])
    verts.append(apex)
    verts.append([0, 0, 0])  # base center
    apex_idx = n_sides
    center_idx = n_sides + 1
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces_list.append([i, j, apex_idx])
        faces_list.append([j, i, center_idx])
    m = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces_list))
    _save_mesh(m, "cone_sharp")

    # 22. Star extruded shape
    verts_2d = []
    n_points = 5
    for i in range(n_points * 2):
        angle = np.pi * i / n_points - np.pi / 2
        r = 1.0 if i % 2 == 0 else 0.4
        verts_2d.append([r * np.cos(angle), r * np.sin(angle)])
    # Create star as a prism by extruding
    top_v = [[x, y, 0.2] for x, y in verts_2d]
    bot_v = [[x, y, -0.2] for x, y in verts_2d]
    all_v = top_v + bot_v
    star_faces = []
    n = len(verts_2d)
    # Side faces
    for i in range(n):
        j = (i + 1) % n
        star_faces.append([i, j, j + n])
        star_faces.append([i, j + n, i + n])
    # Top/bottom fan
    for i in range(1, n - 1):
        star_faces.append([0, i, i + 1])
        star_faces.append([n, n + i + 1, n + i])
    m = trimesh.Trimesh(vertices=np.array(all_v), faces=np.array(star_faces))
    _save_mesh(m, "star_extruded")

    # 23. Gear-like shape (toothed cylinder)
    teeth = []
    base = cylinder(radius=0.8, height=0.3, sections=48)
    teeth.append(base)
    n_teeth = 12
    for i in range(n_teeth):
        angle = 2 * np.pi * i / n_teeth
        tooth = box(extents=[0.15, 0.3, 0.3])
        tooth.apply_translation([1.0 * np.cos(angle), 1.0 * np.sin(angle), 0])
        rot = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        tooth.apply_transform(rot)
        teeth.append(tooth)
    m = trimesh.util.concatenate(teeth)
    _save_mesh(m, "gear_toothed")

    # 24. Staircase
    stairs = []
    for i in range(6):
        step = box(extents=[1.0, 0.4, 0.2])
        step.apply_translation([0, i * 0.4, i * 0.2])
        stairs.append(step)
    m = trimesh.util.concatenate(stairs)
    _save_mesh(m, "staircase_steps")

    print(f"\n{'=' * 60}")
    total = len(list(DATASETS_DIR.glob("*.stl")))
    print(f"Dataset generation complete: {total} STL files in {DATASETS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    generate_complex_datasets()
