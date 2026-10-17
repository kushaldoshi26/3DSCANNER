"""Render a transparent fallback of the website's inferred scanner assembly."""
import bpy
import json
import math
from pathlib import Path
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'model-reports'
bpy.ops.wm.open_mainfile(filepath=str(REPORTS / 'scanner-assets.blend'))
scene = bpy.context.scene
assembly = {
    'frame': {'position': [0,-1.4,0], 'rotation': [math.pi/2,0,0], 'color': [0.045,0.07,0.09,1]},
    'top_plate': {'position': [0,-0.93,0], 'rotation': [0,0,0], 'color': [0.72,0.68,0.58,1]},
    'small_gear': {'position': [0.7,-1.3,-0.5], 'rotation': [0,0,0], 'color': [0.49,0.31,0.16,1]},
    'bearing_holder': {'position': [0,-1.16,0], 'rotation': [math.pi/2,0,0], 'color': [0.42,0.34,0.24,1]},
    'nut_holder': {'position': [-0.4,-1.25,0.9], 'rotation': [math.pi/2,0,0], 'color': [0.49,0.31,0.16,1]},
    'camera_stand': {'position': [2.25,0.873,0], 'rotation': [0,0,0], 'color': [0.045,0.07,0.09,1]},
}
for name, values in assembly.items():
    obj = bpy.data.objects[name]
    obj.hide_render = False
    x, y, z = values['position']
    obj.location = (x, -z, y)
    obj.rotation_euler = values['rotation']
    obj.scale = (0.023,0.023,0.023)
    material = bpy.data.materials.new(name + '_presentation')
    material.use_nodes = True
    shader = material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = values['color']
    shader.inputs['Roughness'].default_value = 0.33
    shader.inputs['Metallic'].default_value = 0.28
    obj.data.materials.clear()
    obj.data.materials.append(material)

def aim(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat('-Z', 'Y').to_euler()

bpy.ops.object.camera_add(location=(7,-10,5.5))
camera = bpy.context.object
camera.data.type = 'ORTHO'
camera.data.ortho_scale = 7.3
camera.data.clip_end = 10000
aim(camera, (0.7,0,0.8))
scene.camera = camera
for location, power, size in [((1,-4,8),1300,6),((-4,-1,3),950,5),((3,5,6),1800,4),((6,-6,1),700,3)]:
    bpy.ops.object.light_add(type='AREA', location=location)
    light = bpy.context.object
    light.data.energy = power
    light.data.shape = 'DISK'
    light.data.size = size
    aim(light, (0.7,0,0.3))

scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.3,0.4,0.5,1)
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.35
scene.render.engine = 'CYCLES'
scene.cycles.samples = 72
scene.cycles.use_denoising = True
scene.render.resolution_x = 1300
scene.render.resolution_y = 1300
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.view_settings.view_transform = 'AgX'
scene.render.filepath = str(ROOT / 'public' / 'models' / 'assembly-poster.png')
bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(REPORTS / 'scanner-assembly-preview.blend'))

for file in [REPORTS / 'geometry-report.json', ROOT / 'public' / 'models' / 'manifest.json']:
    report = json.loads(file.read_text(encoding='utf-8'))
    report['assembly_preview'] = {
        'coordinate_system': 'Three.js Y-up, for already-centered GLB assets',
        'uniform_scale': 0.023,
        'status': 'Inferred visual arrangement; requires assembled project photo to validate mechanical alignment.',
        'parts': {name: {k:v for k,v in values.items() if k != 'color'} for name, values in assembly.items()},
        'poster': '/models/assembly-poster.png',
    }
    file.write_text(json.dumps(report, indent=2), encoding='utf-8')
(REPORTS / 'assembly-render-complete.txt').write_text('Completed transparent assembly poster and scene.\n', encoding='utf-8')
