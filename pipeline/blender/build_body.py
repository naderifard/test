"""c_body, in Blender: build one character body per pending unit.

One process handles every character (§10 rule 1), so each unit starts with a
``reset()`` - otherwise Rostam's mesh would end up in Sohrab's file and both
measurements would be wrong in the same direction.

The measurements written to ``metrics`` are the ones ``c_body._gate`` cannot
take on trust: the *evaluated* height and floor gap, the count of vertices no
bone reaches, the missing mouth channels, and any material slot without a
``style_role``. The gate runs in the host before the artifact is committed, so
anything it needs has to travel in ``metrics`` rather than be read back off
disk.
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

B = _common.install()


def _photo(job, unit):
    """The character built from its reference photo (c_asset -> photo_body)."""
    from pipeline.blender import photo_body

    params = unit["params"]
    cid = params["subject"]
    B.reset(engine="EEVEE", res=job.resolution, fps=job.fps)
    built = photo_body.build(B, cid, os.path.dirname(unit["inputs"]["asset"]),
                             float(params["height_m"]), params.get("kind"),
                             # The stage's first step leaves this for a
                             # character it could fit SMPL-X to.
                             smplx_path=os.path.join(unit["out_dir"],
                                                     "smplx_fit.npz"))
    mesh, rig = built["mesh"], built["rig"]
    points = _common.world_points(mesh)
    low = min(point.z for point in points)
    high = max(point.z for point in points)
    stats = _common.measure([mesh])
    blocks = mesh.data.shape_keys.key_blocks if mesh.data.shape_keys else {}
    missing = [name for name in VISEME_CHANNELS if name not in blocks]
    collection = _common.collect(f"char_{cid}", [mesh, rig])
    blend = _common.save_blend(_base.out_path(unit, "body.blend"))
    height = float(rig.get("stature_m") or (high - low))

    body = {
        "schema_version": 2,
        "source": "sf3d_photo",
        "subject": cid,
        "kind": params.get("kind"),
        "rig_mode": params.get("rig_mode"),
        "collection": collection.name,
        "mesh": mesh.name,
        "rig": rig.name,
        "blend": os.path.basename(blend),
        "asked_height_m": float(params["height_m"]),
        "height_m": round(height, 4),
        "top_m": round(high - low, 4),
        "feet_z_m": round(low, 4),
        "joints": {name: [round(v, 5) for v in xyz]
                   for name, xyz in built["joints"].items()},
        "dims": built["dims"],
        "face": {name: ([round(v, 5) for v in value]
                        if isinstance(value, (list, tuple)) else value)
                 for name, value in built["face"].items()},
        "bones": list(built["bones"]),
        "roles": built["roles"],
        "channels": list(VISEME_CHANNELS),
        "skinning": built["skinning"],
        "rest_correction_bones": built["rest_correction_bones"],
        "measured": stats,
    }
    jsonio.write(_base.out_path(unit, "body.json"), body)
    jaw = built["viseme_move_m"].get("jaw_open", 0.0)
    return {
        "metrics": {
            "subject": cid,
            "kind": body["kind"],
            "source": "sf3d_photo",
            "asked_height_m": body["asked_height_m"],
            "height_m": body["height_m"],
            "top_m": body["top_m"],
            "feet_z_m": body["feet_z_m"],
            "bones": len(built["bones"]),
            "shape_keys": len(blocks),
            "missing_channels": missing,
            "unweighted": _common.unweighted([mesh], rig),
            "untagged_slots": _common.untagged_slots([mesh]),
            "roles": 1,
            "faces_before_decimate": built["faces_before"],
            "skinning": built["skinning"],
            "smplx_skin": built.get("smplx_skin"),
            "distance_weighted": built["distance_weighted"],
            "rest_corrected_bones": len(built["rest_correction_bones"]),
            "jaw_open_max_m": jaw,
            "collection": collection.name,
            **stats,
        },
        "notes": [f"photo body {body['height_m']:.3f} m (top "
                  f"{body['top_m']:.3f} m), {stats['tris']} tris from "
                  f"{built['faces_before']}, skinned by {built['skinning']}, "
                  f"{len(built['rest_correction_bones'])} bones rest-corrected, "
                  f"jaw opens {jaw * 1000:.1f} mm"],
    }


def run(job, unit):
    params = unit["params"]
    cid = params["subject"]
    if unit["inputs"].get("asset") and os.path.exists(unit["inputs"]["asset"]):
        return _photo(job, unit)
    face = jsonio.read(unit["inputs"]["face"], default={}) or {}

    B.reset(engine="EEVEE", res=job.resolution, fps=job.fps)
    built = B.character(
        cid,
        height_m=float(params["height_m"]),
        preset=params.get("preset") or "neutral",
        kind=params.get("kind") or "human",
        clothed=bool(params.get("clothed", True)),
        beard=bool(face.get("beard")),
        material_roles=params.get("material_roles") or {},
    )
    mesh, rig = built["mesh"], built["rig"]

    points = _common.world_points(mesh)
    low = min(point.z for point in points)
    high = max(point.z for point in points)
    stats = _common.measure([mesh])

    blocks = (mesh.data.shape_keys.key_blocks
              if mesh.data.shape_keys else {})
    missing = [name for name in VISEME_CHANNELS if name not in blocks]

    collection = _common.collect(f"char_{cid}", [mesh, rig])
    blend = _common.save_blend(_base.out_path(unit, "body.blend"))

    body = {
        "schema_version": 1,
        "subject": cid,
        "kind": params.get("kind"),
        "preset": params.get("preset"),
        "rig_mode": params.get("rig_mode"),
        "collection": collection.name,
        "mesh": mesh.name,
        "rig": rig.name,
        "blend": os.path.basename(blend),
        "identity": face.get("identity"),
        "beard": bool(face.get("beard")),
        "asked_height_m": float(params["height_m"]),
        "height_m": round(high - low, 4),
        "feet_z_m": round(low, 4),
        "joints": {name: [round(v, 5) for v in xyz]
                   for name, xyz in built["joints"].items()},
        "dims": {k: (round(v, 5) if isinstance(v, float) else v)
                 for k, v in built["dims"].items()},
        "bones": list(built["bones"]),
        "roles": built["roles"],
        "channels": list(VISEME_CHANNELS),
        "measured": stats,
    }
    jsonio.write(_base.out_path(unit, "body.json"), body)

    return {
        "metrics": {
            "subject": cid,
            # gates.check_body ranges by kind: a 0.9 m gazelle is correct and a
            # 0.9 m man is not.
            "kind": body["kind"],
            "asked_height_m": body["asked_height_m"],
            "height_m": body["height_m"],
            "feet_z_m": body["feet_z_m"],
            "bones": len(built["bones"]),
            "shape_keys": len(blocks),
            "missing_channels": missing,
            "unweighted": _common.unweighted([mesh], rig),
            "untagged_slots": _common.untagged_slots([mesh]),
            "roles": len(built["roles"]),
            "collection": collection.name,
            **stats,
        },
        "notes": [f"{body['height_m']:.3f} m, {stats['tris']} tris, "
                  f"{len(built['bones'])} bones, "
                  f"{len(built['roles'])} material roles"],
    }


if __name__ == "__main__":
    sys.exit(_base.stage_main(run, from_blender=True))
