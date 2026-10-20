"""e_motion step 2: does the take actually move a skeleton the way it claims?

The take is numbers in a file, and every interesting thing about it is a
question about metres after forward kinematics: do the feet stay where they were
put, does any joint snap, does the last frame of the cycle step into the first as
smoothly as any other frame. §11 says measure the artifact, so the artifact is
applied to a real armature here and read back off the evaluated depsgraph.

The rig is a reference body at the prelude's rest height rather than any
character's. A take is scale-free - ``apply_motion(root_scale=...)`` fits it to a
2.10 m Rostam at assembly - so measuring it against one particular character
would report that character's height as if it were the take's property. Skate is
therefore quoted at 1.70 m and scales with whoever performs it.

Velocity and the loop seam are measured analytically on the authored document,
on each bone's own basis rotation, the quantity the take writes, rather than on
world orientation: a degree at the pelvis would otherwise be counted again at
the head, and a gate that fires on the sum cannot say which joint is jittering.
Blender stores pose matrices in single precision, where a sub-degree idle reads
out of quantisation noise - mirrored joints with identical keyframes measured
different seams, which integer harmonics make impossible - so those numbers are
computed here in double precision (``e_motion.velocity``) and the armature is
kept only for what only a real forward pass can answer: skate, hand placement,
root lift, keyed-frame counts and missing bones.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from pipeline import jsonio                                        # noqa: E402
from pipeline.blender import _common                               # noqa: E402
from pipeline.stages import _base, e_motion                       # noqa: E402

import bpy                                                         # noqa: E402

B = _common.install()

# The joints that carry the animal's weight, and are therefore the ones whose
# horizontal drift is skating rather than motion.
CONTACTS = {
    "human": ("left_ankle", "right_ankle"),
    "quadruped": ("left_hoof_f", "right_hoof_f",
                  "left_hoof_b", "right_hoof_b"),
}
# Where the reference rig is measured from, per topology.
REFERENCE_HEIGHT_M = {"human": 1.70, "quadruped": 1.60}
# The pair whose separation says whether the arms came down out of the T-pose.
ARM_PROBE = {"human": ("left_shoulder", "left_wrist")}


def _nan_count(document):
    """Non-finite numbers anywhere in the take, counted rather than assumed."""
    bad = 0
    stack = [document.get("poses") or [], document.get("root_m") or []]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, bool) or item is None:
            continue
        elif isinstance(item, (int, float)) and not math.isfinite(item):
            bad += 1
    return bad


def _sample(rig, points, first, last):
    """World position of every probe point.

    One pass over the frame range, off the evaluated depsgraph, because reading
    the original armature gives the rest pose for anything the depsgraph has not
    flushed back. Rotation is not sampled here: the seam and per-frame velocity
    are measured in double precision off the authored document (see the module
    docstring), because Blender's single-precision pose storage quantises a
    sub-degree idle past the point it can be read back.
    """
    positions = {name: [] for name in points}
    scene = bpy.context.scene
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        graph = bpy.context.evaluated_depsgraph_get()
        evaluated = rig.evaluated_get(graph)
        matrix = evaluated.matrix_world
        for name in points:
            bone = evaluated.pose.bones.get(name)
            if bone is not None:
                positions[name].append(matrix @ bone.head.copy())
    return positions


# A foot is on the ground when it is within this of the lowest it gets.
CONTACT_BAND_M = 0.03


def _skate(positions):
    """Horizontal drift per frame of each contact joint *while it is planted*.

    Skating is a foot sliding over the ground. A foot that lifts and moves is
    stepping, and counting it as skating made every lunge, walk and fall warn
    -- a strike's forward step read as 6 cm a frame of skate. So a step between
    two frames counts only when the joint is within ``CONTACT_BAND_M`` of its
    lowest height in both; for a take whose feet never lift (the idle) that is
    every frame, and the number is what it always was.
    """
    out = {}
    for name, track in positions.items():
        if len(track) < 2:
            continue
        low = min(point.z for point in track)
        planted = [point.z <= low + CONTACT_BAND_M for point in track]
        steps = [math.hypot(track[i + 1].x - track[i].x,
                            track[i + 1].y - track[i].y)
                 for i in range(len(track) - 1)
                 if planted[i] and planted[i + 1]] or [0.0]
        heights = [point.z for point in track]
        out[name] = {
            "skate_max_m": round(max(steps), 6),
            "skate_mean_m": round(sum(steps) / len(steps), 6),
            "travel_m": round(math.hypot(track[-1].x - track[0].x,
                                         track[-1].y - track[0].y), 6),
            "z_min_m": round(min(heights), 5),
            "z_range_m": round(max(heights) - min(heights), 6),
        }
    return out


def _peak(records, field):
    values = [record[field] for record in records.values() if field in record]
    return max(values) if values else None


def run(job, unit):
    params = unit["params"]
    take = params["take"]
    kind = params.get("kind") or "human"
    body = jsonio.read(_base.out_path(unit, "body.json"))
    head = jsonio.read(_base.out_path(unit, "head_pose.json"))
    fps = int(body.get("fps") or job.fps)
    height = REFERENCE_HEIGHT_M.get(kind, 1.70)

    B.reset(engine="EEVEE", res=job.resolution, fps=fps)
    reference = B.character(f"ref_{kind}", height_m=height, preset="neutral",
                            kind=kind, clothed=False)
    rig = reference["rig"]

    applied = B.apply_motion(rig, body["poses"], root=body.get("root_m"),
                             start=1, retime="none", axes=body.get("axes"))
    head_applied = B.apply_motion(rig, head["poses"], start=1, retime="none",
                                  axes=head.get("axes"))

    have = {bone.name for bone in rig.data.bones}
    wanted = set(body.get("bones") or []) | set(head.get("bones") or [])
    missing = sorted(wanted - have)

    contacts = [name for name in CONTACTS.get(kind, ()) if name in have]
    probe = [name for name in ARM_PROBE.get(kind, ()) if name in have]
    root_bone = next((b.name for b in rig.pose.bones if b.parent is None), None)

    frames = int(body.get("cycle_frames") or len(body["poses"]))
    head_frames = int(head.get("cycle_frames") or len(head["poses"]))
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, frames

    points = list(dict.fromkeys(contacts + probe + ([root_bone] if root_bone
                                                    else [])))
    positions = _sample(rig, points, 1, frames)

    # Double precision, off the authored document, per the module docstring:
    # Blender's single-precision pose storage cannot resolve a sub-degree idle.
    velocity = e_motion.velocity(body["poses"])
    head_velocity = e_motion.velocity(head["poses"])
    skate = _skate({name: positions[name] for name in contacts
                    if name in positions})

    # How far out the hand ends up, averaged over the cycle: the number that
    # tells a T-pose apart from an idle without anyone having to look.
    hand_side = hand_drop = None
    if len(probe) == 2 and all(positions.get(name) for name in probe):
        shoulder, wrist = (positions[probe[0]], positions[probe[1]])
        hand_side = sum(abs(point.x) for point in wrist) / len(wrist)
        hand_drop = sum(s.z - w.z for s, w in zip(shoulder, wrist)) / len(wrist)

    root_track = positions.get(root_bone) or []
    root_z = (round(max(p.z for p in root_track)
                    - min(p.z for p in root_track), 6) if root_track else None)
    nan_count = _nan_count(body) + _nan_count(head)

    quality = {
        "schema_version": 1,
        "take": take,
        "kind": kind,
        "measured_on": {"rig": rig.name, "height_m": height,
                        "bones": len(have), "preset": "neutral"},
        "fps": fps,
        "frames": frames,
        "head_frames": head_frames,
        "keyed_frames": applied.get("frames"),
        "head_keyed_frames": head_applied.get("frames"),
        "bones_keyed": applied.get("bones"),
        "bones_missing": missing,
        "nan_count": nan_count,
        "velocity": velocity,
        "head_velocity": head_velocity,
        "contacts": skate,
        "root_lift_m": root_z,
        "hand_side_m": None if hand_side is None else round(hand_side, 5),
        "hand_drop_m": None if hand_drop is None else round(hand_drop, 5),
    }
    jsonio.write(_base.out_path(unit, "quality.json"), quality)

    # A .blend of the take on the reference rig. Nothing downstream reads it -
    # h_assemble keys the JSON onto the real character - but a take that looks
    # wrong is diagnosed by scrubbing it, not by reading a table of degrees.
    blend = _common.save_blend(_base.out_path(unit, "take.blend"))

    peak = _peak(velocity, "max_deg_per_frame")
    seam = _peak(velocity, "seam_deg")
    head_peak = _peak(head_velocity, "max_deg_per_frame")
    head_seam = _peak(head_velocity, "seam_deg")
    worst = max(skate.values(), key=lambda r: r["skate_max_m"]) if skate else None

    return {
        "metrics": {
            "keyed_frames": applied.get("frames"),
            "head_keyed_frames": head_applied.get("frames"),
            "bones_keyed": applied.get("bones"),
            "bones_missing": missing,
            "nan_count": nan_count,
            "max_joint_deg_per_frame": peak,
            "head_max_deg_per_frame": head_peak,
            "loop_seam_deg": seam,
            "head_loop_seam_deg": head_seam,
            "foot_skate_m_per_frame": (worst["skate_max_m"] if worst else None),
            "foot_travel_m": (worst["travel_m"] if worst else None),
            "contacts": len(skate),
            "root_lift_m": root_z,
            "hand_side_m": quality["hand_side_m"],
            "hand_drop_m": quality["hand_drop_m"],
            "reference_height_m": height,
            "blend": os.path.basename(blend),
        },
        "notes": [
            f"measured on a {height:.2f} m reference {kind}: "
            f"{applied.get('bones')} bones over {frames} frames, "
            f"peak {(peak or 0.0):.2f} deg/frame, seam {(seam or 0.0):.2f} deg",
            (f"feet slide {(worst['skate_max_m'] if worst else 0.0) * 1000:.2f} "
             f"mm/frame at worst, {len(skate)} contact(s)"
             + (f"; hands hang {quality['hand_drop_m'] * 100:.0f} cm and "
                f"{quality['hand_side_m'] * 100:.0f} cm out"
                if quality["hand_side_m"] is not None else "")),
        ],
    }


if __name__ == "__main__":
    sys.exit(_base.stage_main(run, from_blender=True))
