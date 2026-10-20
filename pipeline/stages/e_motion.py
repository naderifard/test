"""Stage E — the body take: what a character does while the camera is on them.

§7-E. Each cast member's move in the main prompt arrives as a clip sequence
(``cast[].action``, see ``moves.py``) and becomes one of four kinds of take:

* **idle** -- nothing to do: the seamless breathing idle below, shared by every
  character of a rig topology and phased apart by ``h_assemble``.
* **cycle** -- only talking, or only fighting: one looping take per topology,
  authored by ``motion_library`` and shared the same way.
* **action** -- anything else ("walks to the throne and kneels"): the
  character's own take for that shot, authored clip by clip, played once and
  held.
* **video** -- the user recorded the move: ``performance/<take>/video.mp4`` is
  retargeted frame by frame (``motion_video``) and wins over the clips.

Which one a cast member gets is decided by ``moves.resolve_take``, which
``h_assemble`` also calls, so the two stages cannot disagree about where a
character's motion lives. All four are written in one document shape, so the
shot builder never learns which it is keying.

The rest of this docstring is about the idle, which is where the document
shape was worked out.

Four decisions are worth stating.

**The unit is a take, not a character.** ``cast[].take`` names a performance and
two characters in one shot can share it — a take is rotations, tied to a rig
topology rather than to a body. So the take id defaults to ``idle_<kind>`` and
Rostam and Sohrab draw one cache entry between them. They must not therefore
breathe in unison: ``cast_member.offset_frames`` exists for exactly that, and
``h_assemble`` gives each character its own phase into the shared cycle.

**Rotations are in armature space** (``axes="blender"``), which is what
``apply_motion`` conjugates by each bone's rest basis. With +X to the character's
left, +Y forward and +Z up: pitch about X, roll about Y, yaw about Z. A pose and
its mirror image differ by negating the Y and Z components of the rotation
vector — that is :func:`_mirror`, and the reason only the left side is authored.

**The arms are part of the take.** The rest pose is a T-pose, so a take that
leaves the shoulders alone yields a film of people standing with their arms
straight out. Bringing them down is a constant offset inside the take rather
than a change to the rest pose, because a recovered take carries arm position
too and the rest pose is what the skinning was built against.

**Every period is a whole number of frames and every wave an integer harmonic of
the cycle**, so the last frame steps into the first by exactly as much as any
other frame steps into its neighbour. That is what makes ``retime="loop"``
invisible, and the second step measures it rather than trusting it.

The measuring step runs in Blender because foot skate is a question about metres
after forward kinematics, and the host interpreter has no numpy to answer it
with. It measures on a reference rig at the prelude's own rest height: the take
is scale-free, ``apply_motion(root_scale=...)`` is what fits it to a 2.10 m
Rostam, so the skate figure is quoted at 1.70 m and scales with the character.
"""

from __future__ import annotations

import math
import os
import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio

TAU = math.tau
DEG = math.pi / 180.0

# One breath, one weight shift. 4 s for a person at rest is about fifteen
# breaths a minute, 5 s for a horse about twelve. Whole seconds, so the cycle is
# a whole number of frames at any sane fps.
CYCLE_S = {"human": 4.0, "quadruped": 5.0}

# The head drifts on its own period. 3 s against the body's 4 s means the two
# re-synchronise every 12 s, which is longer than any shot in the slice.
HEAD_CYCLE_S = {"human": 3.0, "quadruped": 3.5}

# Peak of ``sin(x) + 0.22 sin(2x)``, so :func:`_wave` stays inside +/-1 and the
# amplitudes below are degrees rather than degrees times something.
_WAVE_PEAK = 1.0772


def _wave(phase):
    """A breath: periodic, smooth, and not a pure sine.

    The second harmonic makes the inhale steeper than the exhale, which is what
    stops a stub idle from reading as a machine. It is an integer harmonic, so
    the seam at the end of the cycle stays exact.
    """
    return (math.sin(phase) + 0.22 * math.sin(2.0 * phase)) / _WAVE_PEAK


def _mirror(rotation):
    """The same rotation on the other side of the body.

    Mirroring through the YZ plane conjugates a rotation by ``diag(-1, 1, 1)``,
    which for a rotation *vector* negates the Y and Z components and leaves X:
    both arms pitch forward together, but they roll and yaw in opposition.
    """
    return (rotation[0], -rotation[1], -rotation[2])


def _quat(rotation):
    """A rotation vector as a unit quaternion, in double precision."""
    x, y, z = rotation
    angle = math.sqrt(x * x + y * y + z * z)
    if angle < 1e-15:
        return (1.0, 0.0, 0.0, 0.0)
    half = math.sin(angle / 2.0) / angle
    return (math.cos(angle / 2.0), x * half, y * half, z * half)


def _angle_deg(first, second):
    """Degrees between two rotations, in the numerically stable form.

    This is measured here rather than off the posed armature on purpose.
    ``apply_motion`` writes ``basis^-1 M basis``, and a conjugation by a rotation
    preserves the angle, so the number below *is* the bone's per-frame rotation
    on the rig - while Blender stores pose matrices in single precision, where a
    hundredth of a degree is indistinguishable from zero and a sub-degree idle
    would be measured entirely out of quantisation noise.
    """
    a, b = _quat(first), _quat(second)
    dot = sum(x * y for x, y in zip(a, b))
    vx = a[0] * b[1] - b[0] * a[1] - (a[2] * b[3] - a[3] * b[2])
    vy = a[0] * b[2] - b[0] * a[2] - (a[3] * b[1] - a[1] * b[3])
    vz = a[0] * b[3] - b[0] * a[3] - (a[1] * b[2] - a[2] * b[1])
    return 2.0 * math.atan2(math.sqrt(vx * vx + vy * vy + vz * vz),
                            abs(dot)) / DEG


def velocity(poses):
    """Per-bone rotation speed, and the cost of the wrap back to frame one.

    ``seam_deg`` is the step a looped take takes from its last frame into its
    first. For a cycle built from integer harmonics it is an ordinary step; when
    it is not, the take pops once per loop, which is the defect this number
    exists to catch. ``measure_motion`` imports this for its quality report.
    """
    out = {}
    for name in sorted({name for pose in poses for name in pose}):
        track = [pose.get(name, (0.0, 0.0, 0.0)) for pose in poses]
        if len(track) < 2:
            continue
        steps = [_angle_deg(track[i], track[i + 1])
                 for i in range(len(track) - 1)]
        out[name] = {
            "max_deg_per_frame": round(max(steps), 6),
            "mean_deg_per_frame": round(sum(steps) / len(steps), 6),
            "peak_frame": steps.index(max(steps)) + 1,
            "seam_deg": round(_angle_deg(track[-1], track[0]), 6),
        }
    return out


def _peak(records, field):
    values = [record[field] for record in records.values()]
    return round(max(values), 4) if values else None


# Degrees on the left side, about (X pitch, Y roll, Z yaw) in armature space.
# The arm comes down eighty degrees in three pieces because that is how a
# shoulder does it: the collar drops, the upper arm swings, and the forearm
# keeps a little flexion so the elbow does not read as a hinge locked straight.
ARM_DOWN = {
    "left_collar": (0.0, 6.0, 0.0),
    "left_shoulder": (0.0, 60.0, 4.0),
    "left_elbow": (0.0, 14.0, 12.0),
}


def _human(frames):
    """The idle body cycle for an SMPL-X skeleton.

    Amplitudes are degrees. They are small on purpose: at 1.70 m a degree at the
    pelvis is 8 mm at the head, and the thing being avoided is a stub that
    wobbles. What has to be visible is the chest, and three degrees spread over
    the three spine joints lifts it about a centimetre.
    """
    poses, root = [], []
    for index in range(frames):
        phase = TAU * index / frames
        breath = _wave(phase)
        # The weight shift is its own phase so the body is never doing one
        # single thing; the arms then break the mirror by drifting apart.
        shift = math.sin(phase - 0.40 * TAU)
        swing_l = math.sin(phase)
        swing_r = math.sin(phase + 0.37 * TAU)

        pose = {
            "pelvis": (0.0, 1.6 * shift, 0.8 * shift),
            "spine1": (0.9 * breath, -0.5 * shift, 0.0),
            "spine2": (1.2 * breath, -0.6 * shift, 0.4 * shift),
            "spine3": (0.9 * breath, -0.4 * shift, 0.0),
        }
        collar = (0.0, ARM_DOWN["left_collar"][1] - 1.2 * breath, 0.0)
        pose["left_collar"] = collar
        pose["right_collar"] = _mirror(collar)
        for side, swing in (("left", swing_l), ("right", swing_r)):
            shoulder = (1.5 * swing,
                        ARM_DOWN["left_shoulder"][1] + 0.8 * breath,
                        ARM_DOWN["left_shoulder"][2])
            elbow = (0.0, ARM_DOWN["left_elbow"][1],
                     ARM_DOWN["left_elbow"][2] + 1.0 * swing)
            if side == "right":
                shoulder, elbow = _mirror(shoulder), _mirror(elbow)
            pose[side + "_shoulder"] = shoulder
            pose[side + "_elbow"] = elbow

        poses.append({name: [v * DEG for v in value]
                      for name, value in pose.items()})
        root.append([round(0.012 * shift, 5), round(0.004 * breath, 5),
                     round(0.003 * breath, 5)])
    return poses, root


def _human_head(frames):
    """Neck and head, kept out of the body take.

    Separate because the two are driven by different things: the body loops on
    its own cycle while a talking head follows the line. ``h_assemble`` keys them
    from separate files at separate offsets, and in M3 the head comes from a
    tracker rather than from GVHMR, so the split is where the seam already is.

    The head counter-rotates against the neck so the gaze stays roughly level -
    a stub whose head swings with its neck looks like it is being shaken.
    """
    poses = []
    for index in range(frames):
        phase = TAU * index / frames
        nod = math.sin(phase)
        turn = math.sin(2.0 * phase + 0.20 * TAU)
        drift = math.sin(phase + 0.30 * TAU)
        pose = {
            "neck": (1.8 * nod, 0.9 * turn, 1.4 * turn),
            "head": (0.6 * drift - 1.0 * nod, -0.5 * turn, 1.0 * drift),
        }
        poses.append({name: [v * DEG for v in value]
                      for name, value in pose.items()})
    return poses


def _quadruped(frames):
    """The idle body cycle for the quadruped rig.

    A horse at rest breathes with its barrel rather than its shoulders, so the
    amplitude climbs from the loins to the chest, and the tail carries the
    liveliness: two swings and a third harmonic per breath, which is slow enough
    to read as a swat and fast enough not to look like the animal is asleep.
    """
    poses, root = [], []
    for index in range(frames):
        phase = TAU * index / frames
        breath = _wave(phase)
        shift = math.sin(phase - 0.40 * TAU)
        swing = math.sin(2.0 * phase)
        flick = math.sin(3.0 * phase + 0.25 * TAU)

        pose = {
            "root": (0.0, 1.2 * shift, 0.5 * shift),
            "spine1": (0.5 * breath, -0.4 * shift, 0.0),
            "spine2": (0.8 * breath, -0.5 * shift, 0.0),
            "chest": (1.0 * breath, -0.3 * shift, 0.0),
            "tail": (0.8 * breath, 3.5 * swing, 1.2 * flick),
            "tail_end": (0.0, 4.5 * flick, 2.0 * swing),
        }
        for joint in ("shoulder", "hip"):
            lean = (0.0, 0.6 * shift, 0.0)
            pose["left_" + joint] = lean
            pose["right_" + joint] = _mirror(lean)

        poses.append({name: [v * DEG for v in value]
                      for name, value in pose.items()})
        root.append([round(0.008 * shift, 5), 0.0,
                     round(0.002 * breath, 5)])
    return poses, root


def _quadruped_head(frames):
    """Neck and head for the quadruped: a slow graze-and-lift."""
    poses = []
    for index in range(frames):
        phase = TAU * index / frames
        nod = math.sin(phase)
        turn = math.sin(2.0 * phase + 0.20 * TAU)
        drift = math.sin(phase + 0.30 * TAU)
        pose = {
            "neck": (2.2 * nod, 0.8 * turn, 1.6 * turn),
            "head": (-1.2 * nod, -0.5 * turn, 0.9 * drift),
        }
        poses.append({name: [v * DEG for v in value]
                      for name, value in pose.items()})
    return poses


AUTHORS = {
    "human": (_human, _human_head),
    "quadruped": (_quadruped, _quadruped_head),
}

# What kinds share a rig topology, and therefore a take.
RIG_KIND = {"human": "human", "humanoid": "human",
            "quadruped": "quadruped", "creature": "quadruped",
            "animal": "quadruped"}


def plan(project, graph):
    from pipeline import moves, staging

    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    characters = {c["id"]: c for c in resolved.get("characters") or []}
    slots = {f"{x} {d}": (xv, dv) for x, xv in staging.X_SLOTS.items()
             for d, dv in staging.DEPTH_SLOTS.items()}
    slots.update({x: (xv, 0.8) for x, xv in staging.X_SLOTS.items()})

    takes = {}
    for shot in resolved.get("shots") or []:
        positions = {m["character"]: list(m.get("position_m") or [0, 0, 0])
                     for m in shot.get("cast") or []}
        for member in shot.get("cast") or []:
            cid = member.get("character")
            character = characters.get(cid) or {}
            raw_kind = character.get("kind", "human")
            kind = RIG_KIND.get(raw_kind, "human")
            own = member.get("take")
            video = (project.path("performance", own, "video.mp4")
                     if own else None)
            has_video = bool(video and os.path.exists(video))
            take, shared = moves.resolve_take(member, raw_kind, has_video)
            clips = (member.get("action") or {}).get("clips") or []
            if shared:
                mode = "idle" if take.startswith("idle_") else "cycle"
            elif has_video:
                mode = "video"
            elif clips:
                mode = "action"
            else:
                # A legacy plan naming a take with no clips and no video.
                mode = "idle"
            speakers = {line.get("character") or line.get("speaker")
                        for line in shot.get("dialogue") or []}
            aim = (shot.get("camera") or {}).get("aim")
            subject = aim.split(".")[0] if isinstance(aim, str) else None
            entry = takes.setdefault(take, {"kind": kind, "mode": mode,
                                            "featured": (speakers
                                                         | {subject}),
                                            "users": [], "shot": shot["id"],
                                            "member": member,
                                            "positions": positions,
                                            "height": float(character.get(
                                                "height_m") or 1.7),
                                            "video": video if has_video
                                            else None,
                                            "clips": clips})
            if cid not in entry["users"]:
                entry["users"].append(cid)

    units = []
    for take, entry in sorted(takes.items()):
        kind, mode = entry["kind"], entry["mode"]
        params = {"take": take, "kind": kind, "fps": project.fps(),
                  "mode": mode, "learned": LEARNED_VERSION}
        inputs = {"video": entry["video"]}
        if mode == "cycle":
            params["clip"] = take.rsplit("_", 1)[0]
            if params["clip"] == "fight":
                # The fight loop is a strike and a block (an exchange).
                params["exchange"] = EXCHANGE_VERSION
        if mode == "action":
            member = entry["member"]
            reference = REFERENCE_HEIGHT_M.get(kind, 1.70)
            params.update({
                "featured": member.get("character") in entry.get(
                    "featured", ()),
                "text": ((member.get("action") or {}).get("text") or ""),
                "names": sorted(characters),
                "clips": entry["clips"],
                "position_m": list(member.get("position_m") or [0, 0, 0]),
                "facing_deg": float(member.get("facing_deg") or 0.0),
                "targets": {k: v for k, v in entry["positions"].items()
                            if k != member.get("character")},
                "scale": round(entry["height"] / reference, 5),
                "slots": {k: list(v) for k, v in slots.items()},
            })
            from pipeline import motion_learned
            if motion_learned.FALLS.search(params["text"]):
                # A fall now ends on the ground (motion_learned.action).
                params["ends_down"] = 1
            from pipeline import staging
            if any(c.get("clip") in staging.CONTACT_GAP and c.get("target")
                   for c in params["clips"]):
                # Contact moves stop short of their partner.
                params["contact_stop"] = 1
        units.append(Unit(
            f"take_{take}",
            inputs=inputs,
            # fps is a parameter and not just a job field: the cycle is a whole
            # number of frames, so the artifact genuinely differs at 30 fps and
            # would otherwise be served from a 24 fps cache entry.
            params=params,
            out_rel=f"motion/{take}",
            label=f"motion/{take} ({mode}: {', '.join(entry['users'])})",
        ))
    return units


def nothing_to_do(project):
    return "no resolved production plan — a_production has not run"


def _finite(values, path, bad):
    """Count the non-finite numbers, rather than asserting there are none."""
    for index, value in enumerate(values):
        if isinstance(value, (list, tuple)):
            _finite(value, f"{path}[{index}]", bad)
        elif isinstance(value, dict):
            for name, item in value.items():
                _finite(item if isinstance(item, (list, tuple)) else [item],
                        f"{path}.{name}", bad)
        elif not math.isfinite(value):
            bad.append(f"{path}[{index}]")
    return bad


def _document(take, kind, fps, cycle_s, poses, root=None):
    """The take, in the shape ``apply_motion`` reads and M3 will write.

    JSON rather than the design's ``.npz`` for two reasons: the host interpreter
    has no numpy, and a stub whose numbers cannot be read in a text editor is a
    stub that cannot be checked. GVHMR writes the same document out of the array
    it recovers, so ``h_assemble`` never learns which one it is looking at.
    """
    return {
        "schema_version": 1,
        "take": take,
        "kind": kind,
        "source": "authored_idle",
        "backend": "placeholder",
        "fps": fps,
        "cycle_s": cycle_s,
        "cycle_frames": len(poses),
        "loopable": True,
        "retime": "loop",
        "axes": "blender",
        "rotation": "axis_angle_rad",
        "reference_height_m": 1.70 if kind == "human" else 1.60,
        "bones": sorted({name for pose in poses for name in pose}),
        "poses": poses,
        "root_m": root,
    }


REFERENCE_HEIGHT_M = {"human": 1.70, "quadruped": 1.60}


LEARNED_VERSION = 10
EXCHANGE_VERSION = 2


CONTACT_MIN_M = 0.45    # pelvis to pelvis, two bodies chest to chest


def _stop_at_contact(root, report, params):
    """A contact move's travel cut short of walking into its partner.

    MoMask's embrace walks each man half a metre forward, and from arm's
    length Keyumars and Siamak walked into one another and stood merged. The
    root's horizontal travel is scaled so each covers at most half of what
    separates them beyond ``CONTACT_MIN_M``."""
    import math

    from pipeline import staging
    targets = params.get("targets") or {}
    target = next((c.get("target") for c in params.get("clips") or []
                   if c.get("clip") in staging.CONTACT_GAP
                   and c.get("target") in targets), None)
    if not target or not root:
        return root
    here = params.get("position_m") or [0.0, 0.0, 0.0]
    there = targets[target]
    gap = math.hypot(there[0] - here[0], there[1] - here[1])
    allowed = max(0.0, (gap - CONTACT_MIN_M) / 2.0) \
        / float(params.get("scale") or 1.0)
    x0, y0 = root[0][0], root[0][1]
    travel = max(math.hypot(r[0] - x0, r[1] - y0) for r in root)
    if travel <= allowed or travel < 1e-6:
        return root
    k = allowed / travel
    report["contact_travel_cut_m"] = round(travel - allowed, 3)
    return [[x0 + (r[0] - x0) * k, y0 + (r[1] - y0) * k] + list(r[2:])
            for r in root]


def _stage_heading(poses, root, report, params, text):
    """Point a learned take where the staging wants it (motion_learned)."""
    import math

    from pipeline import motion_learned
    targets = params.get("targets") or {}
    target = next((c.get("target") for c in params.get("clips") or []
                   if c.get("target") in targets), None)
    if target and motion_learned.travels(text):
        here = params.get("position_m") or [0.0, 0.0, 0.0]
        there = targets[target]
        world = math.degrees(math.atan2(-(there[0] - here[0]),
                                        there[1] - here[1]))
        local = motion_learned._wrap(world - float(params.get("facing_deg")
                                                    or 0.0))
        poses, root, turned = motion_learned.aim_travel(poses, root, local)
        report["aimed_at"] = target
        report["aim_turn_deg"] = turned
    if params.get("featured"):
        poses, corrected = motion_learned.limit_turn(
            poses, root, float(params.get("facing_deg") or 180.0))
        report["turn_limited_deg"] = corrected
    return poses, root, report


def _authored_cycle(kind, clip, fps):
    """The library's own loop for a clip: the fallback, and every quadruped's."""
    from pipeline import motion_library
    ctx = motion_library.Context(kind, fps)
    table = (motion_library.QUADRUPED if ctx.kind == "quadruped"
             else motion_library.HUMAN)
    start = dict(motion_library.NEUTRAL)
    duration, sampler = table[clip](start, {}, ctx)
    frames = max(2, int(round(duration * fps)))
    poses, head, root = [], [], []
    for index in range(frames):
        bones, offset = motion_library.to_bones(
            sampler(index / float(fps)), ctx.kind)
        head.append({n: motion_library.q_rotvec(bones.pop(n))
                     for n in motion_library.HEAD_BONES if n in bones})
        poses.append({n: motion_library.q_rotvec(q)
                      for n, q in bones.items()})
        root.append([round(v, 5) for v in offset])
    report = {"clips": [{"clip": clip}], "frames": frames,
              "seconds": round(duration, 3)}
    return poses, head, root, report


def _authored(job, unit):
    """A cycle, an action or a video take, in the idle's document shape."""
    from pipeline import motion_learned, motion_library

    params = unit["params"]
    take, kind, mode = params["take"], params.get("kind") or "human", \
        params["mode"]
    fps = int(params.get("fps") or job.fps)
    backend, fallback = None, None
    if mode == "video":
        ok, why = motion_learned.video_available()
        if ok:
            try:
                poses, head, root, report = motion_learned.video(
                    unit["inputs"]["video"],
                    _base.out_path(unit, "gvhmr"), fps)
                backend = "gvhmr"
            except Exception as exc:          # noqa: BLE001 - fall back, say so
                why = f"{type(exc).__name__}: {exc}"
        if backend is None:
            fallback = f"GVHMR not used ({why}); BlazePose instead"
            from pipeline import motion_video
            poses, head, root, report = motion_video.build(
                unit["inputs"]["video"], fps)
            backend = "blazepose_retarget"
        source, loopable = "video", False
    elif mode == "cycle":
        clip = params["clip"]
        if kind == "human" and clip in motion_learned.CYCLE_PROMPTS:
            ok, why = motion_learned.available()
            if ok:
                poses, head, root, report = motion_learned.cycle(clip, fps)
                report["clips"] = [{"clip": clip}]
                backend = "momask"
            else:
                fallback = f"MoMask not used ({why}); authored cycle instead"
        if backend is None:
            poses, head, root, report = _authored_cycle(kind, clip, fps)
            backend = "motion_library"
        source, loopable = f"cycle_{clip}", True
    else:
        ctx = motion_library.Context(
            kind, fps, params["position_m"], params["facing_deg"],
            params.get("targets"), params.get("scale") or 1.0,
            params.get("slots"))
        poses, head, root, report = motion_library.build(params["clips"], ctx)
        backend = "motion_library"
        text = (params.get("text") or "").strip()
        if kind == "human" and text:
            ok, why = motion_learned.available()
            if ok:
                # The move itself, not the breathing tail the library adds.
                seconds = float(report.get("action_s")
                                or len(poses) / float(fps))
                authored_head = head
                poses, head, root, learned = motion_learned.action(
                    text, seconds, params.get("names") or [], fps)
                if any(c.get("clip") in motion_learned.LOOK_CLIPS
                       for c in params["clips"]):
                    # MoMask knows no targets; the library aims the head.
                    head = (authored_head + authored_head[-1:]
                            * max(0, len(poses) - len(authored_head)))
                    head = head[:len(poses)]
                learned["clips"] = report.get("clips")
                poses, root, learned = _stage_heading(
                    poses, root, learned, params, text)
                root = _stop_at_contact(root, learned, params)
                report = learned
                backend = "momask"
            else:
                fallback = f"MoMask not used ({why}); authored clips instead"
        source, loopable = "clips", False

    cycle_s = round(len(poses) / float(fps), 5)
    body_doc = _document(take, kind, fps, cycle_s, poses, root)
    head_doc = _document(take, kind, fps, cycle_s, head)
    for doc in (body_doc, head_doc):
        doc["source"] = source
        doc["backend"] = backend
        doc["loopable"] = loopable
        doc["retime"] = "loop" if loopable else "hold_last_pose"
        doc["clips"] = report.get("clips")
    jsonio.write(_base.out_path(unit, "body.json"), body_doc)
    jsonio.write(_base.out_path(unit, "head_pose.json"), head_doc)
    bad = _finite(poses, "poses", []) + _finite(head, "head", []) \
        + _finite(root, "root", [])
    travel = max([abs(entry[axis]) for entry in root for axis in range(2)]
                 or [0.0])
    return {
        "metrics": {
            "take": take, "kind": kind, "fps": fps, "mode": mode,
            "frames": len(poses), "cycle_frames": len(poses),
            "cycle_s": cycle_s, "head_cycle_frames": len(head),
            "authored_bones": len(body_doc["bones"]),
            "head_bones": len(head_doc["bones"]),
            "root_travel_m": round(travel, 4), "nan_count": len(bad),
            "source": source, "loopable": loopable,
            "clips": [c.get("clip") for c in report.get("clips") or []],
            "backend": backend,
            **({k: report[k] for k in ("caption", "loop_s", "wrap_gap_m",
                                       "skate_m_per_frame",
                                       "keypoint_confidence")
                if k in report}),
            **({k: report[k] for k in ("missed_frames", "source_frames")
                if k in report}),
        },
        "notes": ([fallback] if fallback else []) + [
                  f"{take}: {mode} by {backend}, {len(poses)} frames"
                  + (f", clips {', '.join(c['clip'] for c in report['clips'])}"
                     if report.get("clips") else "")
                  + (f", {report['missed_frames']} of "
                     f"{report['source_frames']} video frames had no performer"
                     if "missed_frames" in report else "")
                  + (f", '{report['caption']}'" if report.get("caption")
                     else "")],
    }


def run(job, unit):
    params = unit["params"]
    if params.get("mode") in ("cycle", "action", "video"):
        return _authored(job, unit)
    take = params["take"]
    kind = params.get("kind") or "human"
    fps = int(params.get("fps") or job.fps)
    body_author, head_author = AUTHORS.get(kind, AUTHORS["human"])

    cycle_s = CYCLE_S.get(kind, 4.0)
    head_cycle_s = HEAD_CYCLE_S.get(kind, 3.0)
    frames = max(2, int(round(cycle_s * fps)))
    head_frames = max(2, int(round(head_cycle_s * fps)))

    poses, root = body_author(frames)
    head = head_author(head_frames)

    body_doc = _document(take, kind, fps, cycle_s, poses, root)
    head_doc = _document(take, kind, fps, head_cycle_s, head)
    head_doc["source"] = "authored_drift"
    jsonio.write(_base.out_path(unit, "body.json"), body_doc)
    jsonio.write(_base.out_path(unit, "head_pose.json"), head_doc)

    bad = _finite(poses, "poses", []) + _finite(head, "head", []) \
        + _finite(root, "root", [])
    travel = max(abs(entry[axis]) for entry in root for axis in range(3))

    return {
        "metrics": {
            "take": take,
            "kind": kind,
            "fps": fps,
            "frames": frames,
            "cycle_frames": frames,
            "cycle_s": cycle_s,
            "head_cycle_frames": head_frames,
            "authored_bones": len(body_doc["bones"]),
            "head_bones": len(head_doc["bones"]),
            "root_travel_m": round(travel, 5),
            "nan_count": len(bad),
            "source": "authored_idle",
        },
        "notes": [f"{take}: {frames}f body cycle ({cycle_s:.1f} s), "
                  f"{head_frames}f head cycle, "
                  f"{len(body_doc['bones'])} bones, "
                  f"root travel {travel * 100:.1f} cm"],
    }


# A T-posed wrist sits 0.69 m out at the reference height. A hand hanging by the
# thigh sits about 0.27 m out, so anything past 0.45 m means the arms never came
# down - the one failure that a structural check would call a success.
HAND_OUT_FAIL_M = 0.45
HAND_OUT_WARN_M = 0.36
SEAM_SLACK_DEG = 0.25


def _seam(out, name, seam, step):
    """The loop is only seamless if the wrap costs no more than any other frame."""
    if seam is None or step is None:
        return
    if seam > step + SEAM_SLACK_DEG:
        out.append(gates.Verdict(
            gates.FAIL, name,
            f"the cycle jumps {seam:.2f} deg from its last frame back to its "
            f"first against {step:.2f} deg between neighbours, so a looped take "
            "would pop once per cycle", round(seam, 3), round(step, 3)))


def _gate(project, unit, metrics):
    out = [v for v in gates.check_motion(metrics) if v.level != gates.PASS]

    for bone in metrics.get("bones_missing") or []:
        out.append(gates.Verdict(
            gates.FAIL, "motion.bones",
            f"the take rotates {bone!r}, which this rig has not got; "
            "apply_motion skips it silently and the joint stays at rest"))

    keyed = metrics.get("keyed_frames")
    cycle = metrics.get("cycle_frames")
    if keyed is not None and cycle and keyed != cycle:
        out.append(gates.Verdict(
            gates.FAIL, "motion.keys",
            f"{keyed} frames landed on the rig out of {cycle} authored",
            keyed, cycle))

    if metrics.get("loopable") is False:
        # A take that plays once and holds has no seam to pop across, and its
        # hands are wherever the move put them: a raised arm or a celebration
        # is not a T-pose. The skate, velocity and NaN checks above still apply.
        return out or [gates.Verdict(
            gates.PASS, "motion",
            f"{metrics.get('take')}: {metrics.get('mode')} take, "
            f"{metrics.get('frames')} frames"
            + (f" ({', '.join(metrics.get('clips') or [])})"
               if metrics.get("clips") else "")
            + ", feet slide %.1f mm/frame at worst"
            % ((metrics.get("foot_skate_m_per_frame") or 0.0) * 1000))]

    _seam(out, "motion.seam", metrics.get("loop_seam_deg"),
          metrics.get("max_joint_deg_per_frame"))
    _seam(out, "motion.head_seam", metrics.get("head_loop_seam_deg"),
          metrics.get("head_max_deg_per_frame"))

    side = metrics.get("hand_side_m")
    if side is not None and side > HAND_OUT_FAIL_M:
        out.append(gates.Verdict(
            gates.FAIL, "motion.arms",
            f"the hands sit {side * 100:.0f} cm out from the centreline at "
            "1.70 m, which is the T-pose the rig rests in, not an idle",
            round(side, 3), HAND_OUT_FAIL_M))
    elif side is not None and side > HAND_OUT_WARN_M:
        out.append(gates.Verdict(
            gates.WARN, "motion.arms",
            f"the hands hang {side * 100:.0f} cm out from the centreline; the "
            "arms read as held away from the body", round(side, 3),
            HAND_OUT_WARN_M))

    skate = metrics.get("foot_skate_m_per_frame")
    return out or [gates.Verdict(
        gates.PASS, "motion",
        f"{metrics.get('take')}: {metrics.get('frames')} frames, seam "
        f"{metrics.get('loop_seam_deg', 0.0):.2f} deg against "
        f"{metrics.get('max_joint_deg_per_frame', 0.0):.2f} deg/frame, feet "
        f"slide {(skate or 0.0) * 1000:.2f} mm/frame")]


STAGE = StageSpec(
    name="e_motion",
    version="3",
    summary="author or retarget a body take per move and measure it on a rig",
    needs=["a_production"],
    # The motion environment: MoMask and GVHMR (and OpenCV for the BlazePose
    # fallback). It resolves to comfyui when the motion venv is absent, where
    # the learned backends report themselves unavailable and the authored
    # takes, which are pure Python, run instead.
    steps=[Step("motion"),
           Step("blender", "pipeline/blender/measure_motion.py")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(run))
