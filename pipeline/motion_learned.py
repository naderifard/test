"""Learned motion for ``e_motion``: MoMask for words, GVHMR for videos.

``e_motion`` asks this module for a take and falls back to the authored clip
library (``motion_library``) or to BlazePose (``motion_video``) when it says
no -- a machine without the models still builds a film, and the take's
``backend`` says which one made it, so a fallback is visible rather than
silent.

What is learned and what stays authored:

* **human action takes** -- the move's own words through MoMask. A move whose
  clips aim the head at someone ("looks at Soroush") keeps the library's head
  take, because MoMask knows no targets; the body is still MoMask's.
* **human talk and fight cycles** -- MoMask with a fixed prompt, cut into a
  seamless loop (``motion_smpl.loop``), shared by every human as before.
* **idle and all quadruped motion** -- authored. The idle is a breathing
  cycle that is already right, and no text-to-motion model covers animals.
* **recorded moves** -- GVHMR, world-grounded; BlazePose only as fallback.
"""

from __future__ import annotations

import os
import re

from . import motion_smpl

CYCLE_PROMPTS = {
    "talk": ("a person stands and speaks, his arms mostly at his sides with "
             "small hand gestures"),
    "fight": "a person fights in place, throwing punches and blocking",
}
CYCLE_GENERATE_S = 8.0
CYCLE_MIN_S = 3.5
LOOK_CLIPS = ("look",)
ACTION_PAD = 1.25              # MoMask moves read better unhurried
# After a one-shot move the character holds its last pose; a held pose that
# does not breathe reads as a freeze-frame, so the hold breathes, as the
# library's own tail does (motion_library.TAIL_S).
TAIL_S = 12.0
BREATH_PERIOD_S = 4.2
BREATH = {"spine1": 0.6, "spine2": 0.9, "spine3": 1.1, "neck": -0.6,
          "left_collar": 0.7, "right_collar": 0.7}      # degrees of pitch

_GENERATOR = None


def available():
    """``(True, "")`` when MoMask can run here, else ``(False, reason)``."""
    try:
        import clip  # noqa: F401
        import smplx  # noqa: F401
        import torch
    except ImportError as exc:
        return False, f"the motion env is missing {exc.name}"
    from . import envs
    for path in (os.path.join(envs.VENDOR_ROOT, "momask-codes"),
                 os.path.join(envs.MODEL_ROOT, "momask", "t2m")):
        if not os.path.isdir(path):
            return False, f"not on this machine: {path}"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    return True, ""


def video_available():
    try:
        import hydra  # noqa: F401
        import smplx  # noqa: F401
    except ImportError as exc:
        return False, f"the motion env is missing {exc.name}"
    from . import envs
    for path in (os.path.join(envs.VENDOR_ROOT, "GVHMR"),
                 os.path.join(envs.MODEL_ROOT, "gvhmr", "gvhmr")):
        if not os.path.isdir(path):
            return False, f"not on this machine: {path}"
    return True, ""


def _generator():
    global _GENERATOR
    if _GENERATOR is None:
        from . import motion_text
        _GENERATOR = motion_text.Generator()
    return _GENERATOR


def _quat(rv):
    import math
    angle = math.sqrt(sum(v * v for v in rv))
    if angle < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(angle / 2.0) / angle
    return (math.cos(angle / 2.0), rv[0] * s, rv[1] * s, rv[2] * s)


def breathing_tail(body, head, root, fps, seconds=TAIL_S):
    """The last pose held for ``seconds``, breathing: a slow pitch of the
    spine, collars and neck around it, starting from zero so the join is
    seamless."""
    import math

    from . import motion_library as ml
    frames = int(round(seconds * fps))
    last_b, last_h, last_r = body[-1], head[-1], root[-1]
    for i in range(1, frames + 1):
        wave = math.sin(2.0 * math.pi * i / (BREATH_PERIOD_S * fps))
        pose = {}
        for name, rv in last_b.items():
            amp = BREATH.get(name)
            q = _quat(rv)
            if amp:
                q = ml.q_mul(q, ml.q_axis((1, 0, 0), amp * wave))
            pose[name] = ml.q_rotvec(q)
        hpose = {}
        for name, rv in last_h.items():
            amp = BREATH.get(name)
            q = _quat(rv)
            if amp:
                q = ml.q_mul(q, ml.q_axis((1, 0, 0), amp * wave))
            hpose[name] = ml.q_rotvec(q)
        body.append(pose)
        head.append(hpose)
        root.append(list(last_r))
    return body, head, root


TRAVEL = re.compile(
    r"\b(walk|walks|walking|run|runs|running|step|steps|stepping|approach"
    r"|approaches|goes|go|moves|move|comes|come|charge|charges|flee|flees"
    r"|retreat|retreats|advance|advances|leap|leaps|jump|jumps|towards?"
    r"|across|away|paces|circles|crosses|enters|exits|leaves|rushes)\b",
    re.I)


def travels(text):
    """Whether a move line asks the character to go somewhere."""
    return bool(TRAVEL.search(text or ""))


def action(text, seconds, names, fps):
    """A one-shot take from the move's words, at ``fps``, then a breathing
    hold.

    A move that names no travel ("bows his head, unable to answer") is asked
    for in place and the sample that wanders least is kept: MoMask's bows and
    gestures often come with a step, and a step the staging did not plan walks
    the character off their mark and off the ground under it.
    """
    from . import motion_text
    stay = not travels(text)
    frames, report = _generator().generate(
        text, seconds * ACTION_PAD, names, in_place=stay)
    frames = motion_smpl.resample(frames, motion_text.FPS, fps)
    if FALLS.search(text or ""):
        frames, report["fall_cut_frame"] = _end_on_the_ground(frames)
    body, head, root, took = motion_smpl.take(frames, fps)
    report.update(took)
    report["action_s"] = round(len(body) / float(fps), 3)
    body, head, root = breathing_tail(body, head, root, fps)
    return body, head, root, report


HEAD_UP_MAX_DEG = 6.0
HEAD_PITCH_BONES = ("neck", "head")


def cap_head_up(poses, limit_deg=HEAD_UP_MAX_DEG):
    """Neck and head together tipped back no further than ``limit_deg``
    (positive pitch is up; a bow's is negative). For a speaker: the face is
    what the shot is for, and tipped back it showed the underside of the
    chin -- the part the painting never saw -- and MuseTalk smeared it."""
    import math
    limit = math.radians(limit_deg)
    out = []
    for pose in poses:
        ups = {n: pose[n][0] for n in HEAD_PITCH_BONES
               if n in pose and pose[n][0] > 0.0}
        total = sum(ups.values())
        if total <= limit:
            out.append(pose)
            continue
        k = limit / total
        capped = dict(pose)
        for name, up in ups.items():
            rv = list(pose[name])
            rv[0] = up * k
            capped[name] = rv
        out.append(capped)
    return out


# MoMask's "falls to the ground and lies still" falls, and then as often as
# not gets up again, and the held last pose was Siamak standing over his own
# death. A fall ends where the body is lowest.
FALLS = re.compile(r"\b(falls?|falling|collapses?|dies|lies|lying|slumps?)\b",
                   re.I)


def _end_on_the_ground(frames):
    """``frames`` cut at the lowest the pelvis goes, and the cut index."""
    pelvis = motion_smpl.J["pelvis"]
    heights = [f[pelvis][2] for f in frames]
    low = min(range(len(heights)), key=heights.__getitem__)
    return frames[:low + 1], low


# Talking is shared by every human, the angel with his wings among them, and a
# talking sample that throws both arms wide drags a fused sleeve or wing with
# them: the talk cycle keeps the sample whose hands stay nearest the body.
TALK_SAMPLES = 6
REACH_WEIGHT = 0.02          # per metre of mean hand reach


def _reach(frames):
    """Mean distance of the wrists from a hand's natural place, metres: how
    far out from the pelvis horizontally, plus how far above the chest."""
    import math
    J = motion_smpl.J
    total = 0.0
    for f in frames:
        p = f[J["pelvis"]]
        chest = f[J["spine3"]][2]
        for side in ("left_wrist", "right_wrist"):
            w = f[J[side]]
            total += math.hypot(w[0] - p[0], w[1] - p[1])
            total += 2.0 * max(0.0, w[2] - chest)
    return total / (2.0 * max(1, len(frames)))


ARM_BONES = ("left_collar", "right_collar", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow")
TALK_ARM_DAMP = 0.72


def damp_arms(body, amount):
    """Arm rotations pulled ``amount`` of the way to the neutral hanging arm.

    MoMask's talkers open their arms wide every few seconds; people mostly
    gesture low and small, and a fused sleeve or wing (Soroush) stretches
    with every wide gesture. Damping keeps the rhythm and shrinks the reach.
    """
    from . import motion_library as ml
    neutral, _ = ml.to_bones(dict(ml.NEUTRAL))
    out = []
    for pose in body:
        damped = dict(pose)
        for name in ARM_BONES:
            if name in pose and name in neutral:
                q = ml.q_slerp(_quat(pose[name]), neutral[name], amount)
                damped[name] = ml.q_rotvec(q)
        out.append(damped)
    return out


# A fight is an exchange: one strikes while the other blocks, then the other
# way round. From one sentence ("fights in place, throwing punches and
# blocking") MoMask put its punches and blocks wherever it liked, and two
# fighters playing it swung at the air at the same moments. So the loop is
# built in two halves of one length -- a strike, then a block -- and
# ``h_assemble`` plays it half a cycle apart for two fighters facing each
# other: each blocks as the other strikes.
EXCHANGE_PROMPTS = (
    "a person in a fighting stance throws a hard punch forward with the "
    "right fist and pulls back into the stance",
    "a person in a fighting stance raises both forearms in front of the face "
    "to block a blow and leans back",
)
EXCHANGE_HALF_S = 2.0


def exchange(fps):
    """The fight loop: strike, then block, each ``EXCHANGE_HALF_S``, on the
    spot. Returns ``(body, head, root, report)`` like :func:`cycle`."""
    from . import motion_library as ml
    from . import motion_text
    halves, reports = [], []
    for text in EXCHANGE_PROMPTS:
        frames, report = _generator().generate(text, EXCHANGE_HALF_S,
                                               in_place=True)
        frames = motion_smpl.resample(frames, motion_text.FPS, fps)
        # Each half on the same spot: the pelvis's start at the origin.
        p0 = frames[0][motion_smpl.J["pelvis"]]
        frames = [[(p[0] - p0[0], p[1] - p0[1], p[2]) for p in f]
                  for f in frames]
        halves.append(frames)
        reports.append(report)
    strike, block = halves
    blend = max(2, int(round(0.4 * fps)))

    def join(a, b):
        """``a`` cross-faded into ``b`` over ``blend`` frames."""
        mixed = [[tuple(p[k] + (q[k] - p[k]) * ml.smooth((i + 0.5) / blend)
                        for k in range(3))
                  for p, q in zip(a[len(a) - blend + i], b[i])]
                 for i in range(blend)]
        return list(a[:len(a) - blend]) + mixed + list(b[blend:])

    # strike -> block -> strike again: the loop's cut closes onto the second
    # strike, so the cycle is one strike and one block long.
    sequence = join(join(strike, block), strike)
    cycle_frames = len(strike) + len(block) - 2 * blend
    half = cycle_frames // 2
    frames, looped = motion_smpl.loop(
        sequence, fps, blend_s=0.4,
        min_s=max(0.5, (cycle_frames - 2) / float(fps)))
    body, head, root, took = motion_smpl.take(frames, fps)
    report = {"exchange": [r.get("caption") for r in reports],
              "half_frames": half}
    report.update(took)
    report.update(looped)
    return body, head, root, report


def cycle(clip, fps):
    """A shared looping take for ``talk`` or ``fight``, at ``fps``."""
    from . import motion_text
    if clip == "fight":
        return exchange(fps)
    talk = clip == "talk"
    frames, report = _generator().generate(
        CYCLE_PROMPTS[clip], CYCLE_GENERATE_S, in_place=True,
        samples=TALK_SAMPLES if talk else motion_text.SAMPLES,
        penalty=(lambda f: REACH_WEIGHT * _reach(f)) if talk else None)
    report["hand_reach_m"] = round(_reach(frames), 3)
    frames = motion_smpl.resample(frames, motion_text.FPS, fps)
    frames, looped = motion_smpl.loop(frames, fps, min_s=CYCLE_MIN_S)
    body, head, root, took = motion_smpl.take(frames, fps)
    if talk:
        body = damp_arms(body, TALK_ARM_DAMP)
        report["arm_damp"] = TALK_ARM_DAMP
    report.update(took)
    report.update(looped)
    return body, head, root, report


# -- heading: where the learned motion goes, and which way it faces ----------
# MoMask knows the words, not the stage: "walks to the king" walks wherever
# the sample happened to walk, and more than one sample turned right round at
# the end -- Sorush delivered his command with his back to the camera.
TRAVEL_MIN_M = 0.5
FEATURED_TURN_LIMIT_DEG = 60.0     # MuseTalk's own limit is about this
TURN_SMOOTH_FRAMES = 12


def _yaw_of(rv):
    import math

    from . import motion_library as ml
    forward = ml.q_rotate(_quat(rv), (0.0, 1.0, 0.0))
    return math.degrees(math.atan2(-forward[0], forward[1]))


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _turn(body, root, deltas):
    """Each frame's pelvis turned by ``deltas[i]`` degrees about +Z."""
    from . import motion_library as ml
    out = []
    for pose, delta in zip(body, deltas):
        turned = dict(pose)
        if "pelvis" in pose and delta:
            q = ml.q_mul(ml.q_axis((0, 0, 1), delta), _quat(pose["pelvis"]))
            turned["pelvis"] = ml.q_rotvec(q)
        out.append(turned)
    return out


def aim_travel(body, root, want_deg):
    """The whole take rotated so its travel heads ``want_deg`` (rig frame,
    CCW from +Y). Returns ``(body, root, turned_deg)``; unchanged when the
    take hardly travels."""
    import math
    end = root[-1] if root else [0.0, 0.0, 0.0]
    if math.hypot(end[0], end[1]) < TRAVEL_MIN_M:
        return body, root, 0.0
    have = math.degrees(math.atan2(-end[0], end[1]))
    delta = _wrap(want_deg - have)
    c, s_ = math.cos(math.radians(delta)), math.sin(math.radians(delta))
    root = [[r[0] * c - r[1] * s_, r[0] * s_ + r[1] * c, r[2]] for r in root]
    body = _turn(body, root, [delta] * len(body))
    return body, root, round(delta, 1)


def limit_turn(body, root, facing_deg, limit=FEATURED_TURN_LIMIT_DEG):
    """Keep the body within ``limit`` degrees of facing the camera.

    ``facing_deg`` is the character's staged facing (180 faces the camera).
    The per-frame correction is smoothed, so the character still turns --
    just not away. Returns ``(body, largest correction in degrees)``.
    """
    raw = [_wrap(facing_deg - 180.0 + _yaw_of(p["pelvis"])) if "pelvis" in p
           else 0.0 for p in body]
    # Unwrapped, so a turn through the back reads 170, 190, 210 and not 170,
    # -170, -150: a wrapped curve flips the correction's sign and it cancels.
    rel = raw[:1]
    for value in raw[1:]:
        rel.append(rel[-1] + _wrap(value - rel[-1]))
    want = [max(-limit, min(limit, r)) - r for r in rel]
    n = TURN_SMOOTH_FRAMES
    smooth = []
    for i in range(len(want)):
        window = want[max(0, i - n):i + n + 1]
        # The largest correction in reach, not the mean: a smoothed-away
        # correction is a frame spent facing away.
        smooth.append(max(window, key=abs) if window else 0.0)
    eased = []
    for i in range(len(smooth)):
        window = smooth[max(0, i - n // 2):i + n // 2 + 1]
        eased.append(sum(window) / len(window))
    return _turn(body, root, eased), round(max(map(abs, eased), default=0.0),
                                           1)


def video(path, work_dir, fps):
    """A take from a recorded performance, via GVHMR, at ``fps``."""
    from . import mocap_gvhmr
    frames, report = mocap_gvhmr.track(path, work_dir)
    body, head, root, took = motion_smpl.build(frames, mocap_gvhmr.FPS, fps)
    report.update(took)
    return body, head, root, report
