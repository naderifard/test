"""j_mouth, in Blender: where each speaker's face is, frame by frame.

MuseTalk repaints the lower half of a face box. Its own tools find that box
with two detectors (DWPose through mmpose, and an SFD face detector), neither
of which is needed here: the scene knows where every face is. The body manifest
carries the face landmarks the character was built from -- eyes, nose, mouth,
in rest-pose mesh coordinates -- so each one is pinned to its nearest mesh
vertex, the deformed mesh is evaluated per frame, and the vertices are
projected through the shot camera. That is exact for a stylised face as much
as a realistic one, and it follows the jaw, the head take and the camera move.

Also per frame, how far the face is turned from the camera. The face normal is
the cross product of the eye line and the eye-to-mouth line, oriented at rest
to point the way the figure faces (+Y); MuseTalk is trained on faces within
about a profile-and-a-half of frontal, so the host step skips frames turned
further than that rather than paint a frontal mouth onto a profile.

Output ``faces.json``: ``{character: [{frame, points: {name: [u, v]}, depth,
turn_deg, visible}]}`` with u, v normalised to the frame (0,0 top left), so it
holds for any render scale.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from pipeline import jsonio                                      # noqa: E402
from pipeline.blender import _common                             # noqa: E402
from pipeline.stages import _base                                # noqa: E402

import bpy                                                       # noqa: E402
from bpy_extras.object_utils import world_to_camera_view          # noqa: E402
from mathutils import Vector                                      # noqa: E402

POINTS = ("left_eye", "right_eye", "nose", "mouth", "mouth_left",
          "mouth_right")


def _abs(job, relative):
    return os.path.join(job.project_root, str(relative).replace("/", os.sep))


def _nearest(mesh, point):
    target = Vector(point)
    best, best_d = 0, float("inf")
    for vertex in mesh.data.vertices:
        d = (vertex.co - target).length_squared
        if d < best_d:
            best, best_d = vertex.index, d
    return best, math.sqrt(best_d)


def _normal(p):
    across = p["right_eye"] - p["left_eye"]
    eyes = (p["right_eye"] + p["left_eye"]) * 0.5
    down = p["mouth"] - eyes
    n = across.cross(down)
    return n.normalized() if n.length > 1e-9 else Vector((0, 1, 0))


def track(job, unit):
    params = unit["params"]
    shot = jsonio.read(_abs(job, params["shot_json"]))
    _common.open_blend(_abs(job, params["scene_blend"]))
    scene = bpy.context.scene
    camera = scene.camera
    out = {}
    notes = []
    for cid in params["speakers"]:
        body = jsonio.read(_abs(job, f"assets/characters/{cid}/body/body.json"),
                           default={}) or {}
        face = {n: v for n, v in (body.get("face") or {}).items()
                if n in POINTS and isinstance(v, list)}
        name = shot["provenance"]["cast"].get(cid, {}).get("mesh") or cid
        mesh = bpy.data.objects.get(name)
        if mesh is None or len(face) < 4:
            notes.append(f"{cid}: no mesh or no face landmarks; not tracked")
            continue
        pins, worst = {}, 0.0
        for point_name, point in face.items():
            pins[point_name], d = _nearest(mesh, point)
            worst = max(worst, d)
        rest = {n: Vector(face[n]) for n in face}
        sign = 1.0 if _normal(rest).y >= 0 else -1.0
        rows = []
        for frame in range(shot["frame_start"], shot["frame_end"] + 1):
            scene.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            evaluated = mesh.evaluated_get(depsgraph)
            data = evaluated.to_mesh()
            world = {n: evaluated.matrix_world @ data.vertices[i].co
                     for n, i in pins.items()}
            evaluated.to_mesh_clear()
            normal = _normal(world) * sign
            centre = (world["left_eye"] + world["right_eye"]
                      + world["mouth"]) / 3.0
            to_camera = (camera.matrix_world.translation - centre).normalized()
            turn = math.degrees(math.acos(max(-1.0, min(1.0,
                                                        normal.dot(to_camera)))))
            points, depth, inside = {}, 0.0, True
            for n, p in world.items():
                view = world_to_camera_view(scene, camera, p)
                points[n] = [round(view.x, 5), round(1.0 - view.y, 5)]
                depth = max(depth, view.z)
                inside = inside and 0.0 <= view.x <= 1.0 and 0.0 <= view.y <= 1.0
            rows.append({"frame": frame, "points": points,
                         "depth": round(depth, 4), "turn_deg": round(turn, 1),
                         "visible": bool(inside and depth > 0.0)})
        out[cid] = rows
        notes.append(f"{cid}: {len(rows)} frames tracked, landmarks pinned "
                     f"within {worst * 1000:.1f} mm")
    scene.frame_set(shot["frame_start"])
    jsonio.write(_base.out_path(unit, "faces.json"), out)
    return {"metrics": {"tracked": sorted(out)}, "notes": notes}


if __name__ == "__main__":
    sys.exit(_base.stage_main(track, from_blender=True))
