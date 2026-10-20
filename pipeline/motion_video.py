"""A recorded performance becomes a body take: BlazePose per frame, retargeted.

When the user attaches a video of themselves doing a move, that video is the
take. Every frame goes through BlazePose (``pose_mp``), whose *world* landmarks
are metric 3D points around the hips, and each limb's direction is re-expressed
in the frame of the body part it hangs from -- the arms in the chest's frame,
the legs in the pelvis's -- which is exactly the pose description
``motion_library.to_bones`` already turns into rotations. So a recorded take and
an authored one are the same data, and the rig's rest correction applies to
both.

**Axes and sides.** BlazePose's world frame has x toward image right, y down
and z away from the camera; a performer faces the camera, so their anatomical
left is on image right. The rig faces +Y and its ``left_*`` bones sit on +X,
which is its anatomical *right*. So points are turned by a proper rotation,
``X = -x``, ``Y = -z`` (toward the camera is forward), ``Z = -y``, and each
landmark pair is read by the rig side it drives (``_rig_sides``): the
performer's right hand drives the character's right hand, not its mirror.

**What is kept of the performer's travel.** World landmarks have no global
position. Sideways travel is read from the hips' image position, scaled by the
performer's own height in pixels; crouching from how far the hips sit above
the lower foot. Travel toward the camera is not recovered from one view and is
left at zero. The performer's overall yaw at the start is removed, so the
staging decides which way the character faces and the video decides how they
move.
"""

from __future__ import annotations

import math

from . import motion_library as ml

SMOOTH_FRAMES = 5
EYE_TO_ANKLE_REF_M = (0.936 - 0.047) * 1.70


def _vec(p):
    """BlazePose world point -> rig axes (a rotation, not a reflection)."""
    return (-p[0], -p[2], -p[1])


def _rig_sides(rows):
    """Landmark rows re-indexed so ``left_*`` is the rig's left, i.e. the
    performer's right (see Axes and sides)."""
    from .pose_mp import INDEX
    out = list(rows)
    for name, index in INDEX.items():
        for side, other in (("left_", "right_"), ("right_", "left_")):
            if name.startswith(side):
                out[INDEX[other + name[len(side):]]] = rows[index]
    return out


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _mid(a, b):
    return tuple((x + y) / 2.0 for x, y in zip(a, b))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _frame(across, up):
    """A rotation (quaternion) whose X is ``across`` and Z is ``up``."""
    x = ml._norm(across)
    upward = tuple(u - _dot(up, x) * c for u, c in zip(up, x))
    z = ml._norm(upward)
    y = ml._cross(z, x)
    return _matrix_to_quat(((x[0], y[0], z[0]), (x[1], y[1], z[1]),
                            (x[2], y[2], z[2])))


def _matrix_to_quat(m):
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return ml._norm(((0.25 * s), (m[2][1] - m[1][2]) / s,
                         (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s))
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return ml._norm(((m[2][1] - m[1][2]) / s, 0.25 * s,
                         (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s))
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ml._norm(((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s,
                         0.25 * s, (m[1][2] + m[2][1]) / s))
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ml._norm(((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s,
                     (m[1][2] + m[2][1]) / s, 0.25 * s))


def _yaw(q):
    forward = ml.q_rotate(q, (0.0, 1.0, 0.0))
    return math.degrees(math.atan2(-forward[0], forward[1]))


def read(path, fps, estimator=None):
    """Landmarks for every output frame of a video, resampled to ``fps``."""
    import cv2
    import numpy as np

    from . import pose_mp
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open the move video {path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    estimator = estimator or pose_mp.Estimator()
    frames, box = [], None
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        found = estimator.detect(bgr, figure_box=box)
        if found:
            xs = [p[0] for p in found["image"]]
            ys = [p[1] for p in found["image"]]
            box = (min(xs), min(ys), max(xs), max(ys))
        frames.append(found)
    capture.release()
    if not frames or not any(frames):
        raise RuntimeError(f"no performer was found in {path}")
    # Hold the last good reading over any frame the detector missed.
    last = next(f for f in frames if f)
    filled = []
    for found in frames:
        last = found or last
        filled.append(last)
    count = max(2, int(round(len(filled) / source_fps * fps)))
    picks = [min(len(filled) - 1, int(round(i * source_fps / fps)))
             for i in range(count)]
    world = np.array([_rig_sides([_vec(p) for p in filled[i]["world"]])
                      for i in picks])
    image = np.array([_rig_sides([p[:2] for p in filled[i]["image"]])
                      for i in picks])
    # A centred moving average: BlazePose jitters a few millimetres a frame.
    if len(world) >= SMOOTH_FRAMES:
        kernel = np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES
        pad = SMOOTH_FRAMES // 2
        for arr in (world, image):
            padded = np.concatenate([arr[:1].repeat(pad, 0), arr,
                                     arr[-1:].repeat(pad, 0)])
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    arr[:, j, k] = np.convolve(padded[:, j, k], kernel,
                                               mode="valid")
    missed = sum(1 for f in frames if not f)
    return world, image, {"source_fps": round(source_fps, 3),
                          "source_frames": len(frames),
                          "missed_frames": missed, "frames": count}


def retarget(world, image):
    """Per-frame body and head poses (quaternions) and root offsets."""
    from .pose_mp import INDEX as I

    def P(frame, name):
        return tuple(float(v) for v in world[frame][I[name]])

    def px(frame, name):
        return tuple(float(v) for v in image[frame][I[name]])

    # Metres per pixel from the performer's own eye-to-ankle height, frame 0.
    eye0 = _mid(px(0, "left_eye"), px(0, "right_eye"))
    ankle0 = _mid(px(0, "left_ankle"), px(0, "right_ankle"))
    height_px = max(1.0, ankle0[1] - eye0[1])
    per_px = EYE_TO_ANKLE_REF_M / height_px
    hips0 = _mid(px(0, "left_hip"), px(0, "right_hip"))

    def hip_height(frame):
        hips = _mid(P(frame, "left_hip"), P(frame, "right_hip"))
        feet = min(P(frame, "left_ankle")[2], P(frame, "right_ankle")[2])
        return hips[2] - feet
    standing = hip_height(0)
    yaw0 = None
    body, head, root = [], [], []
    for f in range(len(world)):
        ls, rs = P(f, "left_shoulder"), P(f, "right_shoulder")
        lh, rh = P(f, "left_hip"), P(f, "right_hip")
        mid_s, mid_h = _mid(ls, rs), _mid(lh, rh)
        up = _sub(mid_s, mid_h)
        torso = _frame(_sub(ls, rs), up)
        pelvis = _frame(_sub(lh, rh), up)
        if yaw0 is None:
            yaw0 = _yaw(pelvis)
        unyaw = ml.q_axis((0, 0, 1), -yaw0)
        pelvis_q = ml.q_mul(unyaw, pelvis)
        torso_q = ml.q_mul(unyaw, torso)
        bones = {"pelvis": pelvis_q}
        spine = ml.q_mul(ml.q_inv(pelvis_q), torso_q)
        third = ml.q_slerp((1.0, 0.0, 0.0, 0.0), spine, 1.0 / 3.0)
        for name in ("spine1", "spine2", "spine3"):
            bones[name] = third
        chest_inv = ml.q_inv(torso)
        pelvis_inv = ml.q_inv(pelvis)
        for side in ("left", "right"):
            upper = ml.q_rotate(chest_inv, _sub(P(f, f"{side}_elbow"),
                                                P(f, f"{side}_shoulder")))
            fore = ml.q_rotate(chest_inv, _sub(P(f, f"{side}_wrist"),
                                               P(f, f"{side}_elbow")))
            t = ml.ARM_T[side]
            m_upper = ml.q_from_to(t, upper)
            bones[f"{side}_collar"] = (1.0, 0.0, 0.0, 0.0)
            bones[f"{side}_shoulder"] = m_upper
            bones[f"{side}_elbow"] = ml.q_from_to(
                t, ml.q_rotate(ml.q_inv(m_upper), fore))
            thigh = ml.q_rotate(pelvis_inv, _sub(P(f, f"{side}_knee"),
                                                 P(f, f"{side}_hip")))
            shin = ml.q_rotate(pelvis_inv, _sub(P(f, f"{side}_ankle"),
                                                P(f, f"{side}_knee")))
            m_thigh = ml.q_from_to(ml.LEG_T, thigh)
            m_shin = ml.q_from_to(ml.LEG_T, ml.q_rotate(ml.q_inv(m_thigh),
                                                        shin))
            bones[f"{side}_hip"] = m_thigh
            bones[f"{side}_knee"] = m_shin
            bones[f"{side}_ankle"] = ml.q_slerp(
                (1.0, 0.0, 0.0, 0.0), ml.q_inv(ml.q_mul(m_thigh, m_shin)), 0.8)
        ears = _mid(P(f, "left_ear"), P(f, "right_ear"))
        face = ml.q_rotate(chest_inv, _sub(P(f, "nose"), ears))
        across = ml.q_rotate(chest_inv, _sub(P(f, "left_ear"),
                                             P(f, "right_ear")))
        # The nose points forward and a little down in a level head, so the
        # head's frame comes from the ear-to-ear axis and the face direction.
        head_q = _head_frame(across, face)
        neck = ml.q_slerp((1.0, 0.0, 0.0, 0.0), head_q, 0.4)
        head.append({"neck": neck,
                     "head": ml.q_mul(ml.q_inv(neck), head_q)})
        body.append(bones)
        hips = _mid(px(f, "left_hip"), px(f, "right_hip"))
        # Image right is the performer's left, which is the rig's -X.
        root.append((round(-(hips[0] - hips0[0]) * per_px, 5), 0.0,
                     round(min(0.0, hip_height(f) - standing), 5)))
    return body, head, root


# A level head's nose, relative to the ear midpoint, in chest axes.
LEVEL_FACE = ml._norm((0.0, 1.0, -0.25))


def _head_frame(across, face):
    """The head's rotation from rest, from its ear-to-ear axis and its face."""
    turn = ml.q_from_to((1.0, 0.0, 0.0), across)
    level = ml.q_rotate(turn, LEVEL_FACE)
    nod = ml.q_from_to(level, face)
    return ml.q_mul(nod, turn)


def build(path, fps, estimator=None):
    """Video file -> ``(body_poses, head_poses, root, report)`` like
    :func:`motion_library.build`, rotations as rotation vectors."""
    world, image, report = read(path, fps, estimator)
    body, head, root = retarget(world, image)
    body_rv = [{n: ml.q_rotvec(q) for n, q in b.items()} for b in body]
    head_rv = [{n: ml.q_rotvec(q) for n, q in h.items()} for h in head]
    report["seconds"] = round(len(body) / float(fps), 3)
    return body_rv, head_rv, [list(r) for r in root], report
