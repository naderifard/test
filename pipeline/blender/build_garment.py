"""c_garment, in Blender: shells over the body's joint frame, bound by proximity.

The shell table is deliberately coarse. What matters for M1 is that a garment
exists as its own object in the body's rest space with the right material role,
because that is the shape of the thing M5 replaces. Radii are fractions of the
body's own ``girth``, and offsets fractions of its ``factor``, so a shell fits
Rostam at 2.10 m and a 1.6 m character without a second table.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from pipeline import jsonio                                        # noqa: E402
from pipeline.blender import _common                               # noqa: E402
from pipeline.stages import _base                                  # noqa: E402

import bpy                                                         # noqa: E402

B = _common.install()

# (from joint, dz at from, to joint, dz at to, r0, r1, sides). The dz values are
# in units of ``factor`` and the radii in units of ``girth``.
SHELLS = {
    "torso": [("spine1", -0.05, "spine3", 0.06, 0.175, 0.205, 14)],
    "hips": [("pelvis", -0.06, "spine1", 0.02, 0.190, 0.180, 12)],
    "arms": [("left_shoulder", 0.0, "left_elbow", 0.0, 0.082, 0.070, 8),
             ("right_shoulder", 0.0, "right_elbow", 0.0, 0.082, 0.070, 8)],
    "legs": [("left_hip", 0.0, "left_knee", 0.0, 0.115, 0.092, 8),
             ("right_hip", 0.0, "right_knee", 0.0, 0.115, 0.092, 8)],
    "head": [("neck", 0.02, "head", 0.10, 0.120, 0.100, 10)],
    "feet": [("left_ankle", 0.0, "left_foot", 0.02, 0.075, 0.065, 8),
             ("right_ankle", 0.0, "right_foot", 0.02, 0.075, 0.065, 8)],
}


def _shell(name, joints, dims, specs, role):
    """One mesh from every span the garment covers, in one object."""
    verts, faces = [], []
    for start, dz0, end, dz1, r0, r1, sides in specs:
        if start not in joints or end not in joints:
            return None, f"{name}: this rig has no {start}/{end}"
        a = list(joints[start])
        b = list(joints[end])
        a[2] += dz0 * dims["factor"]
        b[2] += dz1 * dims["factor"]
        piece_v, piece_f = B._prism(a, b, r0 * dims["girth"],
                                   r1 * dims["girth"], sides=sides)
        offset = len(verts)
        verts.extend(piece_v)
        faces.extend(tuple(index + offset for index in face) for face in piece_f)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    mesh.materials.append(B.role_material(role))
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(ob)
    B.smooth(ob, 50.0)
    return ob, None


# Enough displacement to prove the garment is driven by the armature rather
# than sitting in the same place looking correct.
FOLLOW_ENOUGH_M = 0.002


def _travel(rig, garment, joint):
    """Key a throwaway pose on one bone and measure how far the garment goes."""
    rig.animation_data_clear()
    poses = [{joint: (0.0, 0.0, 0.55 * (index / 7.0))} for index in range(8)]
    B.apply_motion(rig, poses, start=1, frames=8, retime="none")
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    rest = _common.world_points(garment)
    bpy.context.scene.frame_set(8)
    bpy.context.view_layer.update()
    posed = _common.world_points(garment)
    rig.animation_data_clear()
    if not rest:
        return 0.0
    return max((a - b).length for a, b in zip(rest, posed))


def _follows(rig, garment, bones):
    """How far the garment travels when the bones it is bound to are posed.

    It has to be *those* bones. This used to rotate the left shoulder for every
    garment on every character, which works for a cuirass and is meaningless for
    a pair of boots: rotating a shoulder moves nothing at the ankle, so a
    correctly bound sandal reported zero travel and failed the gate. The first
    production plan with footwear in it found that immediately.

    Each bound bone is tried in turn on its own action, and the largest
    displacement wins; the search stops as soon as one is decisive, so the
    common case still costs one pose.
    """
    best, tested = 0.0, []
    for joint in bones:
        if rig.pose.bones.get(joint) is None:
            continue
        moved = _travel(rig, garment, joint)
        tested.append(joint)
        best = max(best, moved)
        if best >= FOLLOW_ENOUGH_M:
            break
    return best, tested


def run(job, unit):
    params = unit["params"]
    cid = params["subject"]
    _common.open_blend(unit["inputs"]["body"])
    body = jsonio.read(unit["inputs"]["body_json"], default={}) or {}
    joints = body.get("joints") or {}
    dims = body.get("dims") or {"factor": 1.0, "girth": 1.0}

    collection = bpy.data.collections.get(f"char_{cid}")
    mesh_ob = bpy.data.objects.get(body.get("mesh") or cid)
    rig = bpy.data.objects.get(body.get("rig") or (cid + "_rig"))
    if mesh_ob is None or rig is None:
        raise RuntimeError(f"{cid}: the body file has no mesh/rig named in "
                           f"body.json ({body.get('mesh')}, {body.get('rig')})")

    made, records, skipped = [], [], []
    for garment in params.get("garments") or []:
        gid = garment.get("id") or "garment"
        specs = []
        for cover in garment.get("covers") or ["torso"]:
            specs.extend(SHELLS.get(cover) or [])
        if not specs:
            skipped.append(f"{gid}: covers {garment.get('covers')} — no shell "
                           "for that")
            continue
        ob, problem = _shell(f"{cid}_{gid}", joints, dims, specs,
                             garment.get("role") or "cloth_primary")
        if ob is None:
            skipped.append(problem)
            continue
        stats = B.bind_garment(mesh_ob, ob, offset=float(params.get("offset_m", 0.0)))
        measured = _common.measure([ob])
        records.append({
            "id": gid,
            "object": ob.name,
            "role": garment.get("role"),
            "covers": garment.get("covers"),
            "bound": stats.get("bound"),
            "of": stats.get("of"),
            "bones": len(stats.get("bones") or []),
            "bound_bones": list(stats.get("bones") or []),
            "tris": measured.get("tris"),
            "size_m": measured.get("size_m"),
        })
        made.append(ob)

    _common.collect(f"char_{cid}", [mesh_ob, rig] + made)
    stats = _common.measure(_common.meshes(collection))
    blend = _common.save_blend(_base.out_path(unit, "garmented.blend"))

    # After the save, so a throwaway action cannot end up in the artifact - and
    # after the measurement, which wants the rest pose's bounding box.
    for record, ob in zip(records, made):
        moved, tested = _follows(rig, ob, record.get("bound_bones") or [])
        record["moved_m"] = round(moved, 5)
        record["tested_bones"] = tested

    jsonio.write(_base.out_path(unit, "garment.json"), {
        "schema_version": 1,
        "subject": cid,
        "collection": f"char_{cid}",
        "blend": os.path.basename(blend),
        "mesh": mesh_ob.name,
        "rig": rig.name,
        "garments": records,
        "skipped": skipped,
        "bind": {"method": "proximity_weights",
                 "offset_m": params.get("offset_m")},
        "measured": stats,
    })
    return {
        "metrics": {
            "subject": cid,
            "garments": records,
            "skipped": skipped,
            **stats,
        },
        "notes": ([f"{r['id']}: {r['bound']}/{r['of']} verts on {r['bones']} "
                   f"bones, moved {r['moved_m'] * 100:.2f} cm"
                   for r in records]
                  or ["no garments"]),
    }


if __name__ == "__main__":
    sys.exit(_base.stage_main(run, from_blender=True))
