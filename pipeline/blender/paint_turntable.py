"""A character turning 360 degrees, as depth and as colour: Wan's guide.

Run as ``blender -b -P paint_turntable.py -- <rigged.blend> <out_dir>
[frames] [width] [height]``. The figure stays still and faces +Y; an
orthographic camera circles it, level with its middle, and frame ``i`` looks
from azimuth ``360 * i / (frames - 1)`` degrees (frame 0 and the last are the
front). For every frame it writes ``depth_####.exr``, the distance from the camera
over the far plane (the background is 1), which ``character_paint`` turns
into the annotator's picture; and for the
first frame ``colour_0000.png``: the character as its current material paints
it, flat-lit, on grey. ``camera.json`` records everything needed to project
the generated frames back onto the mesh exactly.
"""

import json
import math
import os
import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
blend, out_dir = args[0], args[1]
frames = int(args[2]) if len(args) > 2 else 81
width = int(args[3]) if len(args) > 3 else 480
height = int(args[4]) if len(args) > 4 else 832
os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=blend)
scene = bpy.context.scene
for ob in list(scene.objects):
    if ob.type in ("LIGHT", "CAMERA"):
        bpy.data.objects.remove(ob)
meshes = [o for o in scene.objects if o.type == "MESH" and o.visible_get()]
depsgraph = bpy.context.evaluated_depsgraph_get()
points = []
for ob in meshes:
    ev = ob.evaluated_get(depsgraph)
    me = ev.to_mesh()
    points.extend(ev.matrix_world @ v.co for v in me.vertices)
    ev.to_mesh_clear()
xs = [p.x for p in points]
ys = [p.y for p in points]
zs = [p.z for p in points]
cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
zlo, zhi = min(zs), max(zs)
cz = (zlo + zhi) / 2.0
radius = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys))
# The frame holds the figure's height or its widest turn, with a margin.
ortho = max((zhi - zlo), 2.0 * radius * height / float(width)) * 1.08

scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = width, height
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
cam = bpy.data.objects.new("paint_cam", bpy.data.cameras.new("paint_cam"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.type = "ORTHO"
# ortho_scale spans the frame's height, landscape or portrait, as the framing
# above and character_paint.project both assume (AUTO spans the longer side).
cam.data.sensor_fit = "VERTICAL"
cam.data.ortho_scale = ortho
distance = radius * 4.0 + 5.0
cam.data.clip_start = 0.05
cam.data.clip_end = distance * 2.0


def place(i):
    a = math.radians(360.0 * i / float(frames - 1))
    # Azimuth 0 looks at the front (the figure faces +Y); the camera then
    # goes round to -X, the figure's anatomical left -- the side the
    # turnaround sheet shows in profile.
    cam.location = (cx - distance * math.sin(a), cy + distance * math.cos(a),
                    cz)
    cam.rotation_euler = (math.radians(90.0), 0.0, math.radians(180.0) + a)
    return a


world = bpy.data.worlds.new("paint_world")
scene.world = world
world.use_nodes = True
background = world.node_tree.nodes["Background"]

# -- colour of the front, flat-lit on grey ----------------------------------
background.inputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
background.inputs[1].default_value = 1.0
def photo_only():
    """Every character material shows only its front photo, unlit.

    The colour frame Wan keeps must be the photo itself: rendered through
    the blended multi-view material it carried that material's faults (a
    dark mark on the arm) into every painted frame, and lit by the world it
    came out darker than the photo. Returns, per material, the emission
    node and the photo's solidity socket, for ``solid_0000.png``."""
    found = []
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        tree = mat.node_tree
        photo = None
        for node in tree.nodes:
            if node.bl_idname != "ShaderNodeTexImage":
                continue
            link = node.inputs["Vector"].links
            if link and getattr(link[0].from_node, "uv_map", "") == "photo":
                photo = node
        out = next((n for n in tree.nodes
                    if n.bl_idname == "ShaderNodeOutputMaterial"), None)
        if photo is None or out is None:
            continue
        emit = tree.nodes.new("ShaderNodeEmission")
        # The photo only where it is solidly inside the figure; elsewhere
        # the model's own colour. At a grazing surface the photo's projection
        # lands just outside the figure, on empty pixels, and drew a black rim
        # down every inner arm -- which Wan then carried round the turn.
        albedo = tree.nodes.new("ShaderNodeVertexColor")
        albedo.layer_name = "albedo"
        solid = tree.nodes.new("ShaderNodeMath")
        solid.operation = "POWER"
        solid.inputs[1].default_value = 6.0
        tree.links.new(photo.outputs["Alpha"], solid.inputs[0])
        mix = tree.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        tree.links.new(solid.outputs[0], mix.inputs["Factor"])
        tree.links.new(albedo.outputs["Color"], mix.inputs["A"])
        tree.links.new(photo.outputs["Color"], mix.inputs["B"])
        tree.links.new(mix.outputs["Result"], emit.inputs["Color"])
        tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        found.append((tree, emit, solid.outputs[0]))
    return found


photo_nodes = []
if os.environ.get("PAINT_PHOTO_ONLY", "1") == "1":
    photo_nodes = photo_only()
colour_all = os.environ.get("PAINT_COLOUR_ALL") == "1"
front_scale = int(os.environ.get("PAINT_FRONT_SCALE", "1"))
for i in (range(frames) if colour_all else [0]):
    place(i)
    scene.render.resolution_percentage = 100 * (front_scale if i == 0 else 1)
    scene.render.filepath = os.path.join(out_dir, "colour_%04d.png" % i)
    bpy.ops.render.render(write_still=True)

# -- where the front frame is the photo -------------------------------------
# Elsewhere (a grazing arm's edge, beside the armholes) the front frame shows
# the model's own dark albedo, which is no colour anyone should keep: Wan
# paints those pixels instead, and the projection's front layer skips them.
if photo_nodes:
    place(0)
    scene.render.resolution_percentage = 100 * front_scale
    background.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    for tree, emit, solid in photo_nodes:
        tree.links.new(solid, emit.inputs["Color"])
    scene.render.filepath = os.path.join(out_dir, "solid_0000.png")
    bpy.ops.render.render(write_still=True)
scene.render.resolution_percentage = 100

# -- depth for every frame --------------------------------------------------
near = distance - radius
far = distance + radius
mat = bpy.data.materials.new("paint_depth")
mat.use_nodes = True
tree = mat.node_tree
for node in list(tree.nodes):
    if node.bl_idname != "ShaderNodeOutputMaterial":
        tree.nodes.remove(node)
output = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
camdata = tree.nodes.new("ShaderNodeCameraData")
# Raw distance, scaled into 0..1 by the far plane; the Python side normalises
# each frame to the figure's own depth range, as a depth annotator would.
scale = tree.nodes.new("ShaderNodeMath")
scale.operation = "DIVIDE"
scale.inputs[1].default_value = far
tree.links.new(camdata.outputs["View Z Depth"], scale.inputs[0])
emit = tree.nodes.new("ShaderNodeEmission")
tree.links.new(scale.outputs[0], emit.inputs["Color"])
tree.links.new(emit.outputs["Emission"], output.inputs["Surface"])
scene.view_layers[0].material_override = mat
background.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)   # = far
background.inputs[1].default_value = 1.0
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.color_mode = "RGB"
scene.view_settings.view_transform = "Standard"
for i in range(frames):
    place(i)
    scene.render.filepath = os.path.join(out_dir, "depth_%04d.exr" % i)
    bpy.ops.render.render(write_still=True)

json.dump({
    "frames": frames, "width": width, "height": height,
    "ortho_scale": ortho, "centre": [cx, cy, cz], "distance": distance,
    "near": near, "far": far, "depth_exr_scale": far,
    "azimuth_deg": [360.0 * i / float(frames - 1) for i in range(frames)],
}, open(os.path.join(out_dir, "camera.json"), "w"), indent=1)
print("PAINT_TURNTABLE done", frames)
