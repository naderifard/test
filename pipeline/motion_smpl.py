"""SMPL body joints become a body take: the shared retarget for GVHMR and MoMask.

Both learned motion sources end in the same place -- the 22 SMPL body joints
per frame, metric, gravity-aligned. GVHMR recovers them from a video of the
user performing the move; MoMask generates them from the move's words. This
module turns either into the take ``apply_motion`` plays, the same way
``motion_video`` turns BlazePose landmarks into one: every limb's *direction*
is re-expressed in the frame of the body part it hangs from, and
``motion_library``'s conventions turn those directions into rotations. So the
rig's rest correction, the head take and the shot builder see nothing new.

**Axes and sides.** SMPL is y-up, faces +z and has its anatomical left on +x.
The rig faces +Y and is +Z up, and its ``left_*`` bones sit on +X -- which,
for a body facing +Y, is its anatomical *right* (the photo character's right
shoulder is on the rig's +X; checked against Siamak's sash). So the joints are
turned by a proper rotation, ``X = -x, Y = z, Z = y``, which puts the
performer's left on the rig's -X, and the SMPL ``left_*`` joints are read as
the rig's ``right_*`` bones (``RIG_SIDE``). A performer who raises their right
hand makes the character raise its right hand, not its mirror image.

**Travel is real.** Unlike a BlazePose world frame these joints keep their
global position, so a walk walks. The start is moved to the origin and the
performer's starting yaw removed, so the staging still decides where the
character stands and which way they face; the motion decides how they move
from there. Heights are rescaled to the 1.70 m reference by leg length, so a
tall performer's stride lands a 1.70 m rig's stride, and ``apply_motion``'s
``root_scale`` takes it on to the character's own height.
"""

from __future__ import annotations

import math

from . import motion_library as ml
from .motion_video import _frame, _mid, _sub, _yaw

SMPL22 = ("pelvis", "left_hip", "right_hip", "spine1", "left_knee",
          "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
          "left_foot", "right_foot", "neck", "left_collar", "right_collar",
          "head", "left_shoulder", "right_shoulder", "left_elbow",
          "right_elbow", "left_wrist", "right_wrist")
J = {name: index for index, name in enumerate(SMPL22)}

# The reference rig's leg, hip to ankle, and how high a standing ankle sits.
REF_LEG_M = ml.THIGH_M + ml.SHIN_M
REF_HIP_Z = ml.HIP_REST["left"][2]
# A planted foot moves less than this per second and sits this near the floor.
CONTACT_SPEED_MS = 0.25
CONTACT_HEIGHT_M = 0.06
# The foot bone at rest, ankle to toe, in the shin's frame: forward and down.
FOOT_REST = ml._norm((0.0, 0.119, -0.056))
# SMPL's head joint sits forward of its neck joint at rest: a level head's
# neck-to-head line leans 38 degrees forward, measured on the neutral body.
HEAD_REST = ml._norm((0.0, 0.051, 0.065))


# Rig bone side -> the SMPL joint side that drives it (see Axes and sides).
RIG_SIDE = {"left": "right", "right": "left"}


def to_rig(points):
    """SMPL (y up, facing +z) -> rig axes (Z up, facing +Y), rotated not
    reflected, with each SMPL joint moved to the index of the rig bone it
    drives -- so ``frames[f][J["left_hip"]]`` is the rig's left hip."""
    turned = [(-p[0], p[2], p[1]) for p in points]
    out = list(turned)
    for index, name in enumerate(SMPL22):
        for side, other in RIG_SIDE.items():
            if name.startswith(side + "_"):
                out[J[other + name[len(side):]]] = turned[index]
    return out


def resample(frames, source_fps, fps):
    """Linear resampling of a joint sequence to the project's frame rate."""
    if not frames:
        return []
    count = max(2, int(round((len(frames) - 1) / float(source_fps) * fps)) + 1)
    out = []
    for i in range(count):
        t = i * source_fps / float(fps)
        a = min(len(frames) - 1, int(math.floor(t)))
        b = min(len(frames) - 1, a + 1)
        w = t - a
        out.append([tuple(pa[k] + (pb[k] - pa[k]) * w for k in range(3))
                    for pa, pb in zip(frames[a], frames[b])])
    return out


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _leg_length(frames):
    """The performer's hip-to-ankle length, from the straightest frames."""
    lengths = []
    for f in frames:
        for side in ("left", "right"):
            lengths.append(_dist(f[J[f"{side}_hip"]], f[J[f"{side}_knee"]])
                           + _dist(f[J[f"{side}_knee"]],
                                   f[J[f"{side}_ankle"]]))
    lengths.sort()
    return lengths[int(len(lengths) * 0.9)] if lengths else REF_LEG_M


def contacts(frames, fps):
    """Per frame, whether each ankle is planted: slow and near the floor."""
    floor = sorted(min(f[J["left_ankle"]][2], f[J["right_ankle"]][2])
                   for f in frames)[max(0, len(frames) // 20)]
    out = []
    for i, f in enumerate(frames):
        g = frames[min(len(frames) - 1, i + 1)]
        e = frames[max(0, i - 1)]
        row = {}
        for side in ("left", "right"):
            a = J[f"{side}_ankle"]
            span = max(1, min(len(frames) - 1, i + 1) - max(0, i - 1))
            speed = _dist(g[a], e[a]) * fps / span
            row[side] = (speed < CONTACT_SPEED_MS
                         and f[a][2] - floor < CONTACT_HEIGHT_M)
        out.append(row)
    return out, floor


def skate(frames, fps):
    """Mean horizontal slide of planted ankles, metres per frame."""
    planted, _ = contacts(frames, fps)
    total, count = 0.0, 0
    for i in range(1, len(frames)):
        for side in ("left", "right"):
            if planted[i][side] and planted[i - 1][side]:
                a = J[f"{side}_ankle"]
                total += math.hypot(frames[i][a][0] - frames[i - 1][a][0],
                                    frames[i][a][1] - frames[i - 1][a][1])
                count += 1
    return total / count if count else 0.0


def retarget(frames, fps):
    """Rig-axis joint frames -> ``(body, head, root, report)``, quaternions."""
    scale = REF_LEG_M / max(1e-6, _leg_length(frames))
    _, floor = contacts(frames, fps)

    def P(f, name):
        return frames[f][J[name]]

    start = _mid(P(0, "left_hip"), P(0, "right_hip"))
    yaw0 = None
    body, head, root = [], [], []
    for f in range(len(frames)):
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
            # The foot from the recovered toe, in the shin's own frame.
            shin_world = ml.q_mul(pelvis, ml.q_mul(m_thigh, m_shin))
            foot = ml.q_rotate(ml.q_inv(shin_world),
                               _sub(P(f, f"{side}_foot"),
                                    P(f, f"{side}_ankle")))
            bones[f"{side}_ankle"] = ml.q_from_to(FOOT_REST, ml._norm(foot))
        # The head from the neck-to-head line in the chest's frame: nods and
        # tilts. Joint positions carry no head yaw; it follows the chest.
        rise = ml.q_rotate(chest_inv, _sub(P(f, "head"), P(f, "neck")))
        head_q = ml.q_from_to(HEAD_REST, ml._norm(rise))
        neck = ml.q_slerp((1.0, 0.0, 0.0, 0.0), head_q, 0.4)
        head.append({"neck": neck, "head": ml.q_mul(ml.q_inv(neck), head_q)})
        body.append(bones)
        offset = ml.q_rotate(unyaw, _sub(mid_h, start))
        # Hip height over the planted ankles, rescaled, against the rest rig's:
        # zero when the performer stands, negative when they crouch or kneel.
        height = (mid_h[2] - floor) * scale - (REF_HIP_Z - ml.ANKLE_REST_Z)
        root.append((round(offset[0] * scale, 5), round(offset[1] * scale, 5),
                     round(min(height, 0.25), 5)))
    report = {"frames": len(body), "scale_to_reference": round(scale, 4),
              "skate_m_per_frame": round(skate(frames, fps) * scale, 5)}
    return body, head, root, report


def _pose_gap(a, b):
    """Mean joint distance of two frames with the pelvis's travel removed."""
    pa, pb = a[J["pelvis"]], b[J["pelvis"]]
    total = 0.0
    for ja, jb in zip(a, b):
        total += _dist((ja[0] - pa[0], ja[1] - pa[1], ja[2]),
                       (jb[0] - pb[0], jb[1] - pb[1], jb[2]))
    return total / len(a)


def loop(frames, fps, blend_s=0.5, min_s=3.0):
    """A seamless cycle cut from a longer generated take.

    The cut ``e`` is the frame after ``min_s`` whose pose, and the pose of the
    ``blend`` frames after it, best match the start. The travel between start
    and cut is taken out linearly so the cycle stays on its spot, and the
    frames after the cut are cross-faded into the opening ones: the cycle's
    first frames are ``S[e + i]`` fading into ``S[i]``, so the last frame
    before the wrap flows into the first as any two neighbours do.
    """
    blend = max(2, int(round(blend_s * fps)))
    first = int(round(min_s * fps))
    last = len(frames) - blend - 1
    if last <= first:
        raise ValueError(f"{len(frames)} frames are too few for a {min_s} s "
                         f"loop with a {blend_s} s blend")
    cut = min(range(first, last + 1),
              key=lambda e: sum(_pose_gap(frames[e + i], frames[i])
                                for i in range(blend)))
    p0, pe = frames[0][J["pelvis"]], frames[cut][J["pelvis"]]
    drift = ((pe[0] - p0[0]) / cut, (pe[1] - p0[1]) / cut)
    steady = [[(p[0] - drift[0] * i, p[1] - drift[1] * i, p[2]) for p in f]
              for i, f in enumerate(frames)]
    out = [list(f) for f in steady[:cut]]
    for i in range(blend):
        w = ml.smooth((i + 0.5) / blend)
        out[i] = [tuple(a[k] + (b[k] - a[k]) * w for k in range(3))
                  for a, b in zip(steady[cut + i], steady[i])]
    gap = _pose_gap(out[-1], out[0])
    step = sum(_pose_gap(out[i], out[i + 1]) for i in range(len(out) - 1)) \
        / max(1, len(out) - 1)
    return out, {"cut_frame": cut, "loop_s": round(cut / float(fps), 3),
                 "wrap_gap_m": round(gap, 4), "mean_step_m": round(step, 4)}


def take(frames, fps):
    """Rig-axis joint frames already at ``fps`` -> a take, rotation vectors.

    Returns ``(body_poses, head_poses, root, report)`` like
    :func:`motion_video.build`.
    """
    body, head, root, report = retarget(frames, fps)
    body_rv = [{n: ml.q_rotvec(q) for n, q in b.items()} for b in body]
    head_rv = [{n: ml.q_rotvec(q) for n, q in h.items()} for h in head]
    report["seconds"] = round(len(body) / float(fps), 3)
    return body_rv, head_rv, [list(r) for r in root], report


def build(frames, source_fps, fps):
    """Rig-axis joint frames at ``source_fps`` -> a take at ``fps``."""
    return take(resample(frames, source_fps, fps), fps)
