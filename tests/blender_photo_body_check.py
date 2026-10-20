"""Render a photo-built character at rest and posed, for a person to look at.

    blender -b --factory-startup --python tests/blender_photo_body_check.py -- \
        <project> <character> <out_dir> [take]

Writes ``<character>_rest.png``, ``_posed.png``, ``_posed34.png`` and
``_back.png``: the rest pose from the front, then the posed figure from the
front, three-quarters and behind. Not picked up by unittest: it needs Blender and a
built project, and what it checks is whether the arms came down and the photo
is on the right side of the head, which is a thing to look at.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bpy  # noqa: E402

from pipeline import jsonio  # noqa: E402
from pipeline.blender import _common  # noqa: E402

B = _common.install()

args = sys.argv[sys.argv.index("--") + 1:]
project_root, cid, out_dir = args[0], args[1], args[2]
take = args[3] if len(args) > 3 else None
if not os.path.isabs(project_root):
    project_root = os.path.join(os.path.dirname(HERE), "projects", project_root)
os.makedirs(out_dir, exist_ok=True)

base = os.path.join(project_root, "assets", "characters", cid, "rigged")
manifest = jsonio.read(os.path.join(base, "rigged.json"))
_common.open_blend(os.path.join(base, "rigged.blend"))
rig = bpy.data.objects[manifest["rig"]]
mesh = bpy.data.objects[manifest["mesh"]]
kind = "quadruped" if rig.get("rig_kind") == "quadruped" else "human"
take = take or f"idle_{kind}"
body = jsonio.read(os.path.join(project_root, "motion", take, "body.json"))
height = float(manifest.get("top_m") or manifest["height_m"])

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = 520, 760
scene.world = bpy.data.worlds.new("w")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (
    0.42, 0.42, 0.45, 1)
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.2
sun.rotation_euler = (math.radians(55), 0, math.radians(-150))
scene.collection.objects.link(sun)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.lens = 50


def shoot(name, yaw_deg):
    a = math.radians(yaw_deg)
    dist = height * 2.4
    # The character faces +Y, so the camera that sees its front is on +Y.
    cam.location = (dist * math.sin(a), dist * math.cos(a), height * 0.55)
    look = (0.0 - cam.location.x, 0.0 - cam.location.y,
            height * 0.5 - cam.location.z)
    cam.rotation_euler = (math.atan2(math.hypot(look[0], look[1]), -look[2]),
                          0.0, math.atan2(look[1], look[0]) - math.pi / 2)
    path = os.path.join(out_dir, f"{cid}_{name}.png")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


shots = [shoot("rest", 0)]
report = B.apply_motion(rig, body["poses"], root=body.get("root_m"), start=1,
                        frames=len(body["poses"]), retime="loop",
                        axes=body.get("axes", "blender"))
scene.frame_set(max(1, len(body["poses"]) // 3))
shots += [shoot("posed", 0), shoot("posed34", 40), shoot("back", 180)]
print("apply_motion:", report)
print("SHOTS", " ".join(shots))
