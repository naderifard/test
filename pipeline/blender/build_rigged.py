"""c_rigged, in Blender: measure the finished character and publish it.

Nothing is built here. The file is opened, measured against §11-C, annotated so
that a .blend opened by hand says what it is, and written out under the name the
shot builder consumes. The copy is deliberate: ``h_assemble`` appends
``rigged/rigged.blend``, and when M5 inserts a real garment solver between the
two the name it has to satisfy stays put.

Two measurements are taken by moving the rig rather than by reading it. The jaw
key is opened and the mesh re-evaluated, because a shape key that exists and
deforms nothing passes every structural check and then produces a film in which
nobody's lips part; and each attachment point named by the production plan is
resolved to a real bone, whose rest head is recorded so the shot builder places
a prop without reopening this file.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from pipeline import jsonio                                        # noqa: E402
from pipeline.blender import _common                               # noqa: E402
from pipeline.stages import _base                                  # noqa: E402
from pipeline.validate import VISEME_CHANNELS                      # noqa: E402

import bpy                                                         # noqa: E402

B = _common.install()

# What the production plan calls an attachment point, in the order a rig might
# name it. The plan says ``hand_r`` because that is what a director says; SMPL-X
# says ``right_wrist`` and a quadruped says ``right_hoof_f``. The unqualified
# name is tried last, so a plan that already uses bone names just works.
ATTACH_ALIASES = {
    "hand_r": ("right_wrist", "right_hoof_f"),
    "hand_l": ("left_wrist", "left_hoof_f"),
    "head": ("head",),
    "jaw": ("jaw",),
    "back": ("spine3", "chest"),
    "hip": ("pelvis", "root"),
    "foot_r": ("right_foot", "right_hoof_b"),
    "foot_l": ("left_foot", "left_hoof_b"),
    "saddle": ("spine2", "spine3"),
    "muzzle": ("muzzle", "head"),
}


def _resolve_attach(rig, names):
    """Attachment point -> bone, with the bone's rest head in world space."""
    bones = rig.data.bones
    found, missing = {}, []
    for name in names:
        for candidate in tuple(ATTACH_ALIASES.get(name, ())) + (name,):
            bone = bones.get(candidate)
            if bone is not None:
                head = rig.matrix_world @ bone.head_local
                found[name] = {
                    "bone": candidate,
                    "head_m": [round(v, 5) for v in head],
                    "length_m": round(bone.length, 5),
                }
                break
        else:
            missing.append(name)
    return found, missing


def _jaw_move(mesh):
    """How far the mesh travels when jaw_open goes from 0 to 1, in metres."""
    keys = mesh.data.shape_keys
    if keys is None or "jaw_open" not in keys.key_blocks:
        return None
    block = keys.key_blocks["jaw_open"]
    was = block.value
    try:
        block.value = 0.0
        bpy.context.view_layer.update()
        shut = _common.world_points(mesh)
        block.value = 1.0
        bpy.context.view_layer.update()
        wide = _common.world_points(mesh)
    finally:
        block.value = was
        bpy.context.view_layer.update()
    if not shut:
        return 0.0
    return max((a - b).length for a, b in zip(shut, wide))


def _height(rig, low, high):
    """The figure's height as the plan means it.

    A primitive body is exactly as tall as its bounding box. A body built from
    a photo is not: Keyumars' crown and Soroush's wings stand above their heads,
    so photo_body scaled the figure by its eye line and recorded the stature it
    fitted as ``stature_m``. That, times any scale the rig has picked up since,
    is the number the plan's ``height_m`` is compared with.
    """
    stature = rig.get("stature_m")
    if stature:
        return round(float(stature) * rig.matrix_world.to_scale().z, 4)
    return round(high - low, 4)


def run(job, unit):
    params = unit["params"]
    cid = params["subject"]
    _common.open_blend(unit["inputs"]["garmented"])
    body = jsonio.read(unit["inputs"]["body_json"], default={}) or {}
    garment = jsonio.read(unit["inputs"]["garment_json"], default={}) or {}
    face = jsonio.read(unit["inputs"]["face"], default={}) or {}

    mesh = bpy.data.objects.get(body.get("mesh") or cid)
    rig = bpy.data.objects.get(body.get("rig") or (cid + "_rig"))
    if mesh is None or rig is None:
        raise RuntimeError(f"{cid}: the garmented file has no {body.get('mesh')}"
                           f"/{body.get('rig')} to publish")
    garment_obs = [ob for ob in (bpy.data.objects.get(g.get("object"))
                                 for g in garment.get("garments") or [])
                   if ob is not None]

    # Before anything is measured: the shape model's hands are mittens, and
    # the published character carries SMPL-X's in their place.
    hands = None
    if params.get("hands") == "smplx":
        from pipeline.blender import hands as hands_mod
        if hands_mod.available():
            hands = hands_mod.transplant(mesh, rig)

    collection = _common.collect(f"char_{cid}", [mesh, rig] + garment_obs)
    skinned = [mesh] + garment_obs
    stats = _common.measure(skinned)
    points = _common.world_points(mesh)
    low = min(point.z for point in points)
    high = max(point.z for point in points)

    blocks = mesh.data.shape_keys.key_blocks if mesh.data.shape_keys else {}
    missing = [name for name in VISEME_CHANNELS if name not in blocks]
    attach, unresolved = _resolve_attach(rig, params.get("attach") or [])

    manifest = {
        "schema_version": 1,
        "subject": cid,
        "kind": params.get("kind"),
        "rig_mode": params.get("rig_mode"),
        "collection": collection.name,
        "mesh": mesh.name,
        "rig": rig.name,
        "blend": "rigged.blend",
        "identity": face.get("identity"),
        "asked_height_m": float(params.get("height_m") or 0.0),
        "height_m": _height(rig, low, high),
        # The top of the figure, crown and wings included. For a primitive body
        # it is the height; for a photo body it is not, and says by how much.
        "top_m": round(high - low, 4),
        "feet_z_m": round(low, 4),
        "bones": list(body.get("bones") or [bone.name for bone in rig.data.bones]),
        "joints": body.get("joints") or {},
        "dims": body.get("dims") or {},
        "channels": list(VISEME_CHANNELS),
        "roles": body.get("roles") or [],
        "garments": [{"id": g.get("id"), "object": g.get("object"),
                      "role": g.get("role"), "covers": g.get("covers")}
                     for g in garment.get("garments") or []],
        "attach": attach,
        "props": params.get("props") or [],
        "anchors": face.get("anchors") or {},
        "measured": stats,
    }

    # A .blend that says what it is. h_assemble reads the JSON, but a human
    # opening the asset should not have to go looking for the sidecar.
    collection["bstudio_subject"] = cid
    collection["bstudio_kind"] = str(params.get("kind"))
    collection["bstudio_height_m"] = manifest["height_m"]
    collection["bstudio_mesh"] = mesh.name
    collection["bstudio_rig"] = rig.name
    collection["bstudio_attach"] = ",".join(f"{k}={v['bone']}"
                                            for k, v in sorted(attach.items()))

    blend = _common.save_blend(_base.out_path(unit, "rigged.blend"))
    manifest["blend"] = os.path.basename(blend)
    jsonio.write(_base.out_path(unit, "rigged.json"), manifest)

    # After the save: opening the jaw moves the mesh, and the artifact should
    # hold the rest pose no matter how this measurement ends.
    jaw = _jaw_move(mesh)

    return {
        "metrics": {
            "subject": cid,
            "kind": params.get("kind"),
            "asked_height_m": manifest["asked_height_m"],
            "height_m": manifest["height_m"],
            "feet_z_m": manifest["feet_z_m"],
            "bones": len(manifest["bones"]),
            "shape_keys": len(blocks),
            "missing_channels": missing,
            "jaw_move_m": None if jaw is None else round(jaw, 5),
            "unweighted": _common.unweighted(skinned, rig),
            "untagged_slots": _common.untagged_slots(skinned),
            "garments": len(manifest["garments"]),
            "attach": {k: v["bone"] for k, v in attach.items()},
            "unresolved_attach": unresolved,
            "collection": collection.name,
            "hands": hands,
            **stats,
        },
        "notes": [f"{manifest['height_m']:.3f} m, {stats['tris']} tris, "
                  f"{len(manifest['bones'])} bones, "
                  f"{len(manifest['garments'])} garment(s), jaw opens "
                  f"{(jaw or 0) * 1000:.1f} mm"]
        + [f"{name} -> {rec['bone']}" for name, rec in sorted(attach.items())],
    }


if __name__ == "__main__":
    sys.exit(_base.stage_main(run, from_blender=True))
