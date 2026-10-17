"""Prepare source STLs with Blender, without modifying source files.

Run using: blender-launcher.exe --background --factory-startup --python tools/prepare_models.py
"""
import bpy
import math
import json
import time
from pathlib import Path
from mathutils import Vector
import bmesh

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / '3D scanner 3D printed parts'
OUT = ROOT / 'public' / 'models'
REPORTS = ROOT / 'model-reports'
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
PARTS = {
    'frame': ('frame.stl', 70000),
    'top_plate': ('top plate.stl', 30000),
    'small_gear': ('small gear.stl', 16000),
    'nut_holder': ('nutholder.stl', 10000),
    'bearing_holder': ('bearing holder.stl', 10000),
    'camera_stand': ('camera stand/camera stand in 1 part (requires 220mm print bed and supports).stl', 24000),
}
status_path = REPORTS / 'pipeline-status.txt'
def log(message):
    print(message, flush=True)
    with status_path.open('a', encoding='utf-8') as handle:
        handle.write(message + '\n')

status_path.write_text('Starting Blender STL preparation\n', encoding='utf-8')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
objects = {}
report = {
    'source': 'User-supplied printable STL files',
    'blender_version': bpy.app.version_string,
    'units': 'Original STL numerical units, assumed millimetres. STL does not encode units.',
    'coordinate_mapping': 'Blender original XYZ (Z-up) exported as glTF XYZ (Y-up): (x, z, -y). Each mesh is centered independently.',
    'assembly_status': 'Visual assembly is inferred from part geometry and requires confirmation against an assembled reference.',
    'parts': {},
}

def bounds(obj):
    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    mins = [min(v[i] for v in verts) for i in range(3)]
    maxs = [max(v[i] for v in verts) for i in range(3)]
    dims = [maxs[i] - mins[i] for i in range(3)]
    return mins, maxs, dims

def tris(obj):
    obj.data.calc_loop_triangles()
    return len(obj.data.loop_triangles)

def save_report():
    text = json.dumps(report, indent=2)
    (REPORTS / 'geometry-report.json').write_text(text, encoding='utf-8')
    (OUT / 'manifest.json').write_text(text, encoding='utf-8')

for name, (filename, target) in PARTS.items():
    bpy.ops.wm.stl_import(filepath=str(SOURCE / filename), use_mesh_validate=True)
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name + '_mesh'
    original_min, original_max, dimensions = bounds(obj)
    center = Vector([(original_min[i] + original_max[i]) / 2 for i in range(3)])
    report['parts'][name] = {
        'source_file': filename,
        'source_size_bytes': (SOURCE / filename).stat().st_size,
        'source_bounds_min': original_min,
        'source_bounds_max': original_max,
        'source_dimensions': dimensions,
        'source_center': list(center),
        'source_triangles': tris(obj),
        'gltf_dimensions': [dimensions[0], dimensions[2], dimensions[1]],
        'recenter_translation_blender': [-value for value in center],
        'path': '/models/' + name + '.glb',
    }
    for vertex in obj.data.vertices:
        vertex.co -= center
    objects[name] = obj
    log(f'Imported {name}: {report["parts"][name]["source_triangles"]} triangles, dimensions {dimensions}')
save_report()

mat = bpy.data.materials.new('Warm matte mineral')
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Base Color'].default_value = (0.55, 0.48, 0.36, 1)
bsdf.inputs['Roughness'].default_value = 0.34
bsdf.inputs['Metallic'].default_value = 0.25

for name, obj in objects.items():
    started = time.time()
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=0.0001)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data)
    bm.free()
    # Remove redundant coplanar tessellation before silhouette-sensitive reduction.
    planar = obj.modifiers.new('Simplify coplanar surfaces', 'DECIMATE')
    planar.decimate_type = 'DISSOLVE'
    planar.angle_limit = math.radians(0.3)
    planar.use_dissolve_boundaries = False
    bpy.ops.object.modifier_apply(modifier=planar.name)
    after_planar = tris(obj)
    target = PARTS[name][1]
    if after_planar > target:
        collapse = obj.modifiers.new('Web triangle budget', 'DECIMATE')
        collapse.decimate_type = 'COLLAPSE'
        collapse.ratio = target / after_planar
        collapse.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=collapse.name)
    for face in obj.data.polygons:
        face.use_smooth = True
    obj.data.set_sharp_from_angle(angle=math.radians(38))
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    weighted = obj.modifiers.new('Preserve machined edges', 'WEIGHTED_NORMAL')
    weighted.keep_sharp = True
    weighted.weight = 40
    bpy.ops.object.modifier_apply(modifier=weighted.name)
    dest = OUT / (name + '.glb')
    bpy.ops.export_scene.gltf(filepath=str(dest), export_format='GLB', use_selection=True, export_yup=True, export_apply=True, export_materials='EXPORT', export_normals=True, export_texcoords=False, export_animations=False)
    part = report['parts'][name]
    part['after_planar_triangles'] = after_planar
    part['exported_triangles'] = tris(obj)
    part['triangle_reduction_percent'] = round(100 * (1 - part['exported_triangles'] / part['source_triangles']), 2)
    part['glb_size_bytes'] = dest.stat().st_size
    log(f'Exported {name}: {part["exported_triangles"]} triangles, {part["glb_size_bytes"]} bytes, {time.time()-started:.1f}s')
    save_report()

report['totals'] = {
    'source_triangles': sum(p['source_triangles'] for p in report['parts'].values()),
    'exported_triangles': sum(p['exported_triangles'] for p in report['parts'].values()),
    'glb_size_bytes': sum(p['glb_size_bytes'] for p in report['parts'].values()),
}
save_report()

# Preserve actual dimensions in the reusable asset library blend.
bpy.ops.wm.save_as_mainfile(filepath=str(REPORTS / 'scanner-assets.blend'))
log('Saved reusable scanner-assets.blend')

# Display the six independent geometries at comparable scales for visual inspection.
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 24
scene.cycles.use_denoising = True
scene.render.resolution_x = 1500
scene.render.resolution_y = 1000
scene.render.resolution_percentage = 100
scene.world.color = (0.15, 0.15, 0.15)
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.13, 0.15, 0.19, 1)
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.45
scene.view_settings.view_transform = 'AgX'
for idx, (name, obj) in enumerate(objects.items()):
    scale = 105 / max(obj.dimensions)
    obj.scale = (scale, scale, scale)
    column = idx % 3
    row = idx // 3
    obj.location = ((column - 1) * 160, (0.5-row)*150, 42)

def aim(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat('-Z', 'Y').to_euler()

bpy.ops.object.camera_add(location=(280, -470, 690))
camera = bpy.context.object
camera.name = 'Contact sheet camera'
camera.data.type = 'ORTHO'
camera.data.clip_end = 10000
camera.data.ortho_scale = 610
aim(camera, (0, 0, 30))
scene.camera = camera
for pos, energy, size in [((50,-130,450),1800000,380),((-300,0,250),900000,300),((180,260,350),1800000,250)]:
    bpy.ops.object.light_add(type='AREA', location=pos)
    light = bpy.context.object
    light.data.energy = energy
    light.data.shape = 'DISK'
    light.data.size = size
    aim(light, (0,0,0))

# Render labels as geometry facing the camera; labels remain readable in the PNG.
label_mat = bpy.data.materials.new('Label ink')
label_mat.use_nodes = True
label_bsdf = label_mat.node_tree.nodes.get('Principled BSDF')
label_bsdf.inputs['Base Color'].default_value = (0.95,0.95,0.95,1)
label_bsdf.inputs['Emission Color'].default_value = (0.95,0.95,0.95,1)
label_bsdf.inputs['Emission Strength'].default_value = 0.5
for idx, (name, obj) in enumerate(objects.items()):
    bpy.ops.object.text_add(location=(obj.location.x, obj.location.y - 60, 4))
    text_obj = bpy.context.object
    text_obj.data.body = str(idx+1) + '. ' + name.replace('_', ' ')
    text_obj.data.align_x = 'CENTER'
    text_obj.data.size = 8
    text_obj.rotation_euler = camera.rotation_euler
    text_obj.data.materials.append(label_mat)

scene.render.filepath = str(REPORTS / 'parts-contact-sheet.png')
bpy.ops.render.render(write_still=True)
log('Rendered parts-contact-sheet.png')

# Frame, top plate, and stand individually from three useful geometric angles.
for obj in scene.objects:
    if obj.type in {'MESH', 'FONT'}:
        obj.hide_render = True
scene.render.resolution_x = 900
scene.render.resolution_y = 900
for name in ['frame', 'top_plate', 'camera_stand']:
    obj = objects[name]
    obj.hide_render = False
    obj.location = (0,0,0)
    obj.scale = (1,1,1)
    bpy.context.view_layer.update()
    maximum = max(report['parts'][name]['source_dimensions'])
    camera.data.ortho_scale = maximum*1.35
    for view, direction in [('perspective',(1,-1.5,1.3)),('reverse',(1,1.5,-1.3)),('top',(0,-0.001,1)),('front',(0,-1,0.001)),('back',(0,1,0.001))]:
        camera.location = Vector(direction)*maximum*3
        aim(camera, (0,0,0))
        scene.render.filepath = str(REPORTS / f'{name}-{view}.png')
        bpy.ops.render.render(write_still=True)
        log(f'Rendered {name}-{view}.png')
    obj.hide_render = True
log('COMPLETE')
