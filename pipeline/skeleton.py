"""Where the bones go inside a mesh built from a photo.

``c_asset`` pushes each BlazePose landmark through the SF3D mesh (see
``sf3d_dense.joint_from_pixel``) and gets two points per landmark: the *skin*,
where the ray enters, and the *middle*, halfway to where it leaves. This module
turns those into the joint set ``prelude``'s rigs are built from -- the first 22
SMPL-X joints plus ``jaw`` for a person, the ``QUAD_JOINTS`` for an animal --
so everything downstream (motion, anchors, prop attachment) keeps its names.

Pure Python over tuples, no numpy and no bpy, so it runs in every interpreter
and the host tests can exercise it.

**Sides are decided by position, not by anatomy.** ``prelude``'s rest pose puts
every ``left_*`` joint on +X while the body faces +Y, which is the mirror image
of a real person (whose left, facing +Y with +Z up, is -X). Every take in the
project -- the authored idle, its ``_mirror`` helper, the motion library -- is
written against that convention, so here the landmark pair that lands on +X is
named ``left_``, whichever side BlazePose thought it was. A photo that confuses
BlazePose about which way the figure faces therefore cannot mirror the rig.

**Human proportions come from SMPL-X's own rest pose.** Where BlazePose has no
landmark (pelvis, spine, collars, the head *joint*, the jaw), the joint is
placed at the fraction of the way between landmarks that the 1.70 m rest body
in ``prelude.SMPLX_REST`` puts it, so a rig built here and a rig built there
agree about what "spine2" means.
"""

from __future__ import annotations

import math

# The subset of prelude.SMPLX_REST this module needs, copied rather than
# imported so the module stays free of bpy. tests/test_skeleton.py asserts the
# two tables agree.
SMPLX_Z = {"pelvis": 0.94, "hips": 0.86, "spine1": 1.05, "spine2": 1.16,
           "spine3": 1.25, "neck": 1.42, "collar": 1.36, "shoulder": 1.40,
           "head": 1.52, "jaw": 1.56}
SMPLX_HEIGHT = 1.70
# Eye height over stature for an adult. Used to express offsets in the figure's
# own scale when only landmarks are known.
EYE_OVER_STATURE = 0.936

HUMAN_PAIRS = (("shoulder", "shoulder"), ("elbow", "elbow"),
               ("wrist", "wrist"), ("hip", "hip"), ("knee", "knee"),
               ("ankle", "ankle"))

# prelude.QUAD_REST, same reason as above.
QUAD_REST = {
    "root": (0.00, 0.00, 0.62), "spine1": (0.00, 0.22, 0.66),
    "spine2": (0.00, 0.46, 0.68), "chest": (0.00, 0.66, 0.66),
    "neck": (0.00, 0.78, 0.80), "head": (0.00, 0.92, 0.92),
    "muzzle": (0.00, 1.06, 0.86), "jaw": (0.00, 1.00, 0.84),
    "tail": (0.00, -0.14, 0.62), "tail_end": (0.00, -0.40, 0.34),
    "left_shoulder": (0.13, 0.62, 0.60), "left_knee": (0.14, 0.62, 0.34),
    "left_hoof_f": (0.14, 0.64, 0.03),
    "right_shoulder": (-0.13, 0.62, 0.60), "right_knee": (-0.14, 0.62, 0.34),
    "right_hoof_f": (-0.14, 0.64, 0.03),
    "left_hip": (0.15, -0.02, 0.58), "left_stifle": (0.16, -0.06, 0.36),
    "left_hoof_b": (0.16, 0.02, 0.03),
    "right_hip": (-0.15, -0.02, 0.58), "right_stifle": (-0.16, -0.06, 0.36),
    "right_hoof_b": (-0.16, 0.02, 0.03),
}
QUAD_SPAN_Y = (-0.40, 1.06)
QUAD_CROWN = 1.06
QUAD_HALF_WIDTH = 0.19


class SkeletonError(ValueError):
    pass


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _mul(a, k):
    return tuple(x * k for x in a)


def _lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def _mid(a, b):
    return _lerp(a, b, 0.5)


def _len(a):
    return math.sqrt(sum(x * x for x in a))


def human_joints(points):
    """SMPL-X joints from pushed-through landmarks.

    ``points`` maps a BlazePose landmark name to ``{"mid": (x, y, z),
    "skin": (x, y, z)}`` in Blender axes (figure facing +Y, up +Z), in whatever
    units the mesh is in. Returns ``(joints, info)``; raises
    :class:`SkeletonError` naming what is missing when the figure cannot be
    rigged from what was found.
    """
    need = ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
    missing = [name for name in need if name not in points]
    if missing:
        raise SkeletonError("no landmark for " + ", ".join(missing))

    def mid(name):
        return tuple(points[name]["mid"])

    # Which BlazePose side is +X.
    flip = bool(mid("left_shoulder")[0] < mid("right_shoulder")[0])

    def side(rig_side, part):
        blaze = rig_side if not flip else ("right" if rig_side == "left"
                                           else "left")
        return f"{blaze}_{part}"

    eyes = [tuple(points[n]["mid"]) for n in ("left_eye", "right_eye")
            if n in points]
    ankles = _mid(mid("left_ankle"), mid("right_ankle"))
    if eyes:
        eye = _mid(*eyes) if len(eyes) == 2 else eyes[0]
        stature = (eye[2] - ankles[2]) / (EYE_OVER_STATURE - 0.047)
    else:
        shoulders = _mid(mid("left_shoulder"), mid("right_shoulder"))
        stature = (shoulders[2] - ankles[2]) / ((SMPLX_Z["shoulder"] - 0.08)
                                                / SMPLX_HEIGHT)
        eye = None
    if stature <= 0:
        raise SkeletonError("the eyes or shoulders are below the ankles; the "
                            "landmarks do not describe a standing figure")
    unit = stature / SMPLX_HEIGHT          # one SMPL-X metre, in mesh units
    up = (0.0, 0.0, 1.0)
    forward = (0.0, 1.0, 0.0)

    joints = {}
    for rig_side in ("left", "right"):
        for part in ("shoulder", "elbow", "wrist", "knee", "ankle"):
            joints[f"{rig_side}_{part}"] = mid(side(rig_side, part))
        # BlazePose's hip landmarks sit on the outside of the pelvis; SMPL-X's
        # hip joints are the heads of the femurs, a little further in.
        hip = mid(side(rig_side, "hip"))
        joints[f"{rig_side}_hip"] = hip
        heel = points.get(side(rig_side, "heel"))
        toe = points.get(side(rig_side, "foot_index"))
        ankle = joints[f"{rig_side}_ankle"]
        if heel and toe:
            foot = _lerp(tuple(heel["mid"]), tuple(toe["mid"]), 0.6)
        else:
            foot = _add(ankle, (0.0, 0.13 * unit, -0.05 * unit))
        joints[f"{rig_side}_foot"] = (foot[0], foot[1],
                                      max(foot[2], ankle[2] - 0.05 * unit))

    hips = _mid(joints["left_hip"], joints["right_hip"])
    inward = 0.12
    for rig_side, other in (("left", "right"), ("right", "left")):
        joints[f"{rig_side}_hip"] = _lerp(joints[f"{rig_side}_hip"], hips,
                                          inward)
    shoulders = _mid(joints["left_shoulder"], joints["right_shoulder"])
    pelvis = _add(hips, _mul(up, (SMPLX_Z["pelvis"] - SMPLX_Z["hips"]) * unit))
    neck = _add(shoulders, _mul(up, (SMPLX_Z["neck"] - SMPLX_Z["shoulder"])
                                * unit))
    joints["pelvis"] = pelvis
    joints["neck"] = neck
    span = SMPLX_Z["neck"] - SMPLX_Z["pelvis"]
    for name in ("spine1", "spine2", "spine3"):
        joints[name] = _lerp(pelvis, neck,
                             (SMPLX_Z[name] - SMPLX_Z["pelvis"]) / span)
    for rig_side in ("left", "right"):
        shoulder = joints[f"{rig_side}_shoulder"]
        collar = _lerp(neck, shoulder, 0.39)
        joints[f"{rig_side}_collar"] = _add(
            collar, _mul(up, (SMPLX_Z["collar"] - SMPLX_Z["neck"]) * unit))

    if eye is not None:
        head = (eye[0], eye[1],
                eye[2] - (EYE_OVER_STATURE * SMPLX_HEIGHT - SMPLX_Z["head"])
                * unit)
    else:
        head = _add(neck, _mul(up, (SMPLX_Z["head"] - SMPLX_Z["neck"]) * unit))
    joints["head"] = head
    joints["jaw"] = _add(head, _add(
        _mul(up, (SMPLX_Z["jaw"] - SMPLX_Z["head"]) * unit),
        _mul(forward, 0.06 * unit)))

    face = {}
    for name in ("nose", "mouth_left", "mouth_right", "left_eye", "right_eye"):
        if name in points:
            face[name] = tuple(points[name]["skin"])
    if "mouth_left" in face and "mouth_right" in face:
        face["mouth"] = _mid(face["mouth_left"], face["mouth_right"])
        face["mouth_width"] = _len(_sub(face["mouth_left"],
                                        face["mouth_right"]))
    info = {"stature": stature, "unit": unit, "flipped_sides": flip,
            "eye_height": (eye[2] if eye else None), "face": face}
    return joints, info


# On a human arm the wrist is 57% of the way from the elbow to the fingertips
# (forearm 0.25 m, hand 0.19 m). A tip much farther than that, or off in
# another direction, means the detector put the wrist on the forearm.
WRIST_SHARE = 0.57
TIP_TOO_FAR = 1.3
TIP_OFF_AXIS_DEG = 35.0
TIP_POINTS = 20
# How far past the elbow a hand can reach, in upper-arm lengths (a human's
# forearm and hand are about 1.5 of them): measured from the upper arm,
# because the forearm is the thing that may have been detected wrong.
# The first attempt worked on the mesh and kept finding the thigh.
TIP_MAX_UPPER_ARMS = 2.2


def refine_wrist_pixels(points, alpha):
    """Wrist landmarks moved to where the matte says the hands are.

    ``points`` is BlazePose's ``{name: [x, y, visibility]}`` in image pixels;
    ``alpha`` the figure's matte (0..1 or 0..255). Working in the image, not
    on the mesh, because an A-pose reference hangs its hands clear of the
    body with background between -- the image check sees to that -- while
    the reconstructed mesh fuses arm and body and has no clean far end of an
    arm (a mesh search kept finding the thigh). The fingertip is the
    silhouette pixel farthest from the elbow, below it and at least as far
    out; when the detected wrist is far short of it or off its line, the
    wrist moves to ``WRIST_SHARE`` of the way from elbow to tip. Returns
    ``(points, moved)``.
    """
    import math

    import numpy as np
    a = np.asarray(alpha, dtype=np.float32)
    if a.max() > 1.0:
        a = a / 255.0
    ys, xs = np.nonzero(a > 0.5)
    if len(xs) == 0:
        return points, {}
    out = {k: list(v) for k, v in points.items()}
    moved = {}
    centre = (points["left_shoulder"][0] + points["right_shoulder"][0]) / 2.0
    for side in ("left", "right"):
        need = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]
        if any(n not in points for n in need):
            continue
        sx, sy = points[need[0]][:2]
        ex, ey = points[need[1]][:2]
        wx, wy = points[need[2]][:2]
        out_sign = 1.0 if ex >= centre else -1.0
        upper = math.hypot(ex - sx, ey - sy)
        pick = ((ys > ey) & (out_sign * (xs - ex) > -0.15 * upper)
                & (np.hypot(xs - ex, ys - ey) < TIP_MAX_UPPER_ARMS * upper))
        if pick.sum() < TIP_POINTS:
            continue
        px, py = xs[pick], ys[pick]
        dist = np.hypot(px - ex, py - ey)
        far = np.argsort(-dist)[:TIP_POINTS]
        tx, ty = float(px[far].mean()), float(py[far].mean())
        forearm = math.hypot(wx - ex, wy - ey)
        reach = math.hypot(tx - ex, ty - ey)
        expected = forearm / WRIST_SHARE
        cosine = (((wx - ex) * (tx - ex) + (wy - ey) * (ty - ey))
                  / max(1e-9, forearm * reach))
        off_axis = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if reach > TIP_TOO_FAR * expected or off_axis > TIP_OFF_AXIS_DEG:
            nx, ny = ex + WRIST_SHARE * (tx - ex), ey + WRIST_SHARE * (ty - ey)
            out[need[2]] = [nx, ny] + list(points[need[2]][2:])
            moved[side] = {"from_px": [round(wx, 1), round(wy, 1)],
                           "to_px": [round(nx, 1), round(ny, 1)],
                           "tip_px": [round(tx, 1), round(ty, 1)],
                           "off_axis_deg": round(off_axis, 1),
                           "reach_ratio": round(reach / expected, 2)}
    return out, moved


def quadruped_joints(low, high):
    """``prelude.QUAD_REST`` stretched into a mesh's bounding box.

    An animal photographed head-on (which is what the seed prompt asks for, so
    that every reference faces the camera the same way) has no BlazePose
    landmarks to speak of, and the template is a better guess than any of them:
    head at the front of the box, tail at the back, hooves on the floor.
    """
    (x0, y0, z0), (x1, y1, z1) = low, high
    length = y1 - y0
    height = z1 - z0
    half = (x1 - x0) / 2.0
    cx = (x0 + x1) / 2.0
    if min(length, height, half) <= 0:
        raise SkeletonError("the mesh's bounding box is degenerate")
    a, b = QUAD_SPAN_Y
    joints = {}
    for name, (x, y, z) in QUAD_REST.items():
        joints[name] = (cx + x / QUAD_HALF_WIDTH * half * 0.8,
                        y0 + (y - a) / (b - a) * length,
                        z0 + z / QUAD_CROWN * height)
    return joints, {"stature": height, "unit": height / QUAD_CROWN,
                    "flipped_sides": False, "face": {}}
