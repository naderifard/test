"""Stage H — resolve one shot, then compile it into a .blend.

Section 7-H. This is deterministic host code, not LLM code: the model
contributes the *content* of the shot, never the assembly. Two steps, because
the two halves need different interpreters and section 5.1 keeps them one stage:

1. **host** — turn the shot's intent into ``shot.json``, section 6.3's fully
   resolved shot. Real frame counts from the aligned audio, real asset paths
   from the store, the chosen style pack, the resolved camera. Schema-validated
   before anything opens Blender, because a shot that will not validate is
   cheaper to reject in fifty milliseconds than after a scene build.
2. **blender** — ``pipeline/blender/build_shot.py`` compiles that into
   ``scene.blend`` and measures it.

The split is the point. ``shot.json`` is the thing to hand-edit when a shot is
nearly right, and it cannot be that unless every number in it was decided here,
in one readable pass, rather than inside a Blender script.

**The audio is the master clock (section 5.2).** ``frames = ceil((speech_end -
speech_start + head_handle + tail_handle) * fps)``, and every other time in the
shot is expressed against that. A silent shot has to say ``silent_duration_s``
instead -- with no audio there is nothing to derive a length from, which is why
``validate.py`` refuses a shot that has neither.

**Where each of the three clocks starts** is the only genuinely fiddly part of
this file, so it is written down once here and obeyed below:

* *shot time* runs from 0 at ``frame_start``.
* *speech* begins at shot time ``head_handle_s``. That is where the first line's
  ``align.speech_start_s`` lands, and therefore where frame 0 of its
  ``visemes.json`` lands, because g_align writes its curves from
  ``speech_start_s`` onward.
* *the audio strip* is placed so its speech lands at that same moment. Under the
  default ``silence_lead_policy: trim`` the recording's own lead-in is cut off
  and the strip simply starts there; under ``keep`` the strip starts earlier by
  the length of the lead-in, and is clipped if that would fall before frame one.

Under trim the strip start and the viseme start are the same frame computed by
the same expression, which is what makes the section 11 lip-sync check a check
on the pipeline rather than on arithmetic done twice.

**The take is fitted to the window, not the other way round.** ``retime`` and
``offset_frames`` come from the cast member. ``offset_frames`` means two
different things and both are wanted: against a looping idle it is a *phase*
into the cycle, and against a real take it is a *shift* of the performance
relative to the line. Section 5.2 asks for the second; e_motion asks for the
first, because Rostam and Sohrab share one idle take and must not breathe in
unison. So a looping take rotates and any other take shifts, and a cast member
who names no offset gets a stable per-character phase derived from their id --
stable so that a character breathes the same way in every shot they appear in.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio, moves, validate

# Where a camera anchor sits on a rig. The plan says "sohrab.chest" because that
# is what a director says; the rigs say ``spine3`` for a person and ``chest``
# for a horse. First match wins, except ``hands``, which averages whatever it
# finds so a two-handed grip aims between the wrists.
ANCHOR_JOINTS = {
    "head": ("head",),
    "eyes": ("head",),
    "chest": ("spine3", "chest", "spine2"),
    "pelvis": ("pelvis", "root"),
    "root": ("root", "pelvis"),
    "hands": ("left_wrist", "right_wrist", "left_hoof_f", "right_hoof_f"),
}
AVERAGED_ANCHORS = ("hands",)

# The eyes are not a joint on either rig. They are a little above and in front
# of the head joint, in multiples of the body's own scale factor so the offset
# is right for a 2.1 m Rostam and a 1.95 m horse alike.
ANCHOR_OFFSET = {"eyes": (0.0, 0.045, 0.075)}

# Two lines in one shot do not run into each other.
INTER_LINE_GAP_S = 0.18

# How far the perspective compression of a near-orthographic pack may go. A 4x
# dolly-and-zoom is already flatter than any real lens most people have seen;
# past that the near plane starts clipping the set. The pack's own
# ``focal_max_mm`` bounds it further, and usually first: §8 gives the miniature
# and stylized packs a 50-85 mm policy, so a shot authored at 85 mm has already
# asked for the flat perspective and is left alone.
MAX_COMPRESSION = 4.0
DEFAULT_FOCAL_MAX_MM = 200.0

GROUND_TOLERANCE_M = 0.02


# -- planning ----------------------------------------------------------------
def plan(project, graph):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    characters = {c["id"]: c for c in resolved.get("characters") or []}
    envs = {e["id"]: e for e in resolved.get("envs") or []}
    props = {p["id"]: p for p in resolved.get("props") or []}
    config = project.config

    units = []
    for shot in resolved.get("shots") or []:
        units.append(_unit(project, graph, config, shot, characters, envs,
                           props))
    return units


def nothing_to_do(project):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    if not resolved:
        return "no resolved production plan — a_production has not run"
    return "the plan declares no shots"


def _unit(project, graph, config, shot, characters, envs, props):
    """One shot: everything it reads as inputs, everything it is as params.

    The resolved production plan is deliberately *not* an input. Hashing the
    whole file would re-key every shot whenever any other shot was edited, and
    "units, not stages, are the cache grain" is the rule that makes this graph
    worth having. So this shot's own block travels in ``params`` and only the
    artifacts it actually consumes are hashed as inputs.
    """
    shot_id = shot["id"]
    inputs, cast, used_props = {}, [], []
    struck = {c.get("target") for m in shot.get("cast") or []
              for c in (m.get("action") or {}).get("clips") or []
              if c.get("clip") == "strike" and c.get("target")}

    for member in shot.get("cast") or []:
        cid = member["character"]
        character = characters.get(cid) or {}
        kind = character.get("kind", "human")
        # The same decision e_motion planned from: a shared looping take for
        # idling, talking or fighting, the member's own take for anything
        # else or anything the user recorded.
        own = member.get("take")
        has_video = bool(own) and os.path.exists(
            project.path("performance", own, "video.mp4"))
        take, shared = moves.resolve_take(member, kind, has_video)
        retime = (member.get("retime") or "loop") if shared \
            else "hold_last_pose"
        base = ("assets", "characters", cid)
        # c_paint's copy: the rigged character with its painted colour, or
        # the rigged file itself where nothing was painted.
        inputs[f"char_{cid}_blend"] = project.path(*base, "painted",
                                                  "painted.blend")
        inputs[f"char_{cid}_manifest"] = project.path(*base, "rigged",
                                                     "rigged.json")
        inputs[f"take_{cid}_body"] = project.path("motion", take, "body.json")
        inputs[f"take_{cid}_head"] = project.path("motion", take,
                                                  "head_pose.json")
        cast.append({
            "character": cid,
            "name_fa": character.get("name_fa"),
            "kind": kind,
            "rig_mode": character.get("rig_mode", "smplx"),
            "height_m": float(character.get("height_m") or 1.75),
            "material_roles": character.get("material_roles") or {},
            "position_m": [float(v) for v in
                           (member.get("position_m") or [0.0, 0.0, 0.0])],
            "facing_deg": float(member.get("facing_deg") or 0.0),
            "retime": retime,
            "offset_frames": member.get("offset_frames"),
            "struck": cid in struck,
            "take": take,
            "opponent": next(
                (c.get("target") for c in
                 (member.get("action") or {}).get("clips") or []
                 if c.get("clip") in ("fight", "strike") and c.get("target")),
                None),
            "props": list(member.get("props") or []),
        })
        used_props.extend(member.get("props") or [])

    env_id = shot.get("env")
    env = envs.get(env_id) or {}
    inputs["env_blend"] = project.path("assets", "envs", env_id, "build",
                                       "env.blend")
    inputs["env_manifest"] = project.path("assets", "envs", env_id, "build",
                                          "env.json")

    prop_records = {}
    for prop_id in sorted(set(used_props)):
        inputs[f"prop_{prop_id}_blend"] = project.path("assets", "props",
                                                       prop_id, "build",
                                                       "prop.blend")
        inputs[f"prop_{prop_id}_manifest"] = project.path("assets", "props",
                                                          prop_id, "build",
                                                          "prop.json")
        prop_records[prop_id] = props.get(prop_id) or {}

    dialogue = []
    for line in shot.get("dialogue") or []:
        line_id = line["line_id"]
        inputs[f"align_{line_id}"] = project.path("audio", line_id,
                                                  "align.json")
        inputs[f"visemes_{line_id}"] = project.path("audio", line_id,
                                                    "visemes.json")
        dialogue.append({
            "line_id": line_id,
            "character": line.get("character"),
            "text_fa": line.get("text_fa", ""),
            "emotion": line.get("emotion", "neutral"),
            "starts_at_s": line.get("starts_at_s"),
        })

    return Unit(
        f"shot_{shot_id}",
        inputs=inputs,
        params={
            "shot_id": shot_id,
            "style_pack": graph.style_pack,
            "fps": project.fps(),
            "resolution": list(config["resolution"]),
            "head_handle_s": float(config["head_handle_s"]),
            "tail_handle_s": float(config["tail_handle_s"]),
            "min_shot_s": float(config.get("min_shot_s") or 0.0),
            "silence_lead_policy": config.get("silence_lead_policy") or "trim",
            "silent_duration_s": shot.get("silent_duration_s"),
            "env": {"id": env_id,
                    "interior": bool(env.get("interior")),
                    "ground_role": env.get("ground") or "ground",
                    "light_intent": env.get("light_intent") or "",
                    "scatter": env.get("scatter") or []},
            "cast": cast,
            "dialogue": dialogue,
            "camera": shot.get("camera") or {},
            "beats": shot.get("beats") or [],
            "props": prop_records,
            "title": shot.get("title") or shot_id,
        },
        out_rel=f"shots/{shot_id}",
        label=f"shot/{shot_id} ({', '.join(m['character'] for m in cast)})",
    )


RIG_KIND = {"human": "human", "humanoid": "human",
            "quadruped": "quadruped", "creature": "quadruped",
            "animal": "quadruped"}


def _rig_kind(kind):
    return RIG_KIND.get(kind, "human")


# -- geometry the host can do without Blender --------------------------------
def _place(point, position, facing_deg):
    """A point in a character's rest space, put where the character stands."""
    angle = math.radians(facing_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    x, y, z = (float(point[0]), float(point[1]), float(point[2]))
    return [round(position[0] + x * cos - y * sin, 5),
            round(position[1] + x * sin + y * cos, 5),
            round(position[2] + z, 5)]


def _anchor_point(manifest, member, anchor):
    """Where ``sohrab.chest`` is, in world metres, with the shot's staging on."""
    joints = manifest.get("joints") or {}
    factor = float((manifest.get("dims") or {}).get("factor") or 1.0)
    names = ANCHOR_JOINTS.get(anchor, (anchor,))
    found = [joints[name] for name in names if name in joints]
    if not found:
        return None, f"the rig has none of {list(names)}"
    if anchor in AVERAGED_ANCHORS:
        local = [sum(p[axis] for p in found) / len(found) for axis in range(3)]
    else:
        local = list(found[0])
    offset = ANCHOR_OFFSET.get(anchor)
    if offset:
        local = [local[axis] + offset[axis] * factor for axis in range(3)]
    return _place(local, member["position_m"], member["facing_deg"]), None


def _distance(a, b):
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


LEAN_MAX_DEG = 25.0
REACT_DELAY_S = 0.8     # a strike's wind-up, before the blow lands


def _lean_to(member, cast):
    """Degrees the taller of two fighters bends toward the shorter, so his
    blows land at the other's height and not above it: the shoulders come
    down by about 60% of the difference in height. 0 for anyone else."""
    import math
    other = next((m for m in cast
                  if m["character"] == member.get("opponent")), None)
    if other is None:
        return 0.0
    mine, theirs = float(member["height_m"]), float(other["height_m"])
    if mine <= theirs:
        return 0.0
    lean = math.degrees(math.atan2(0.6 * (mine - theirs), 0.4 * mine))
    return round(min(LEAN_MAX_DEG, lean), 1)


def _exchange_phase(member, cast, cycle_frames):
    """For two characters fighting each other on the shared fight loop --
    a strike, then a block (``motion_learned.exchange``) -- 0 for one and
    half a cycle for the other, so each blocks as the other strikes. None
    for anyone else. Who goes first is the cast order."""
    other = member.get("opponent")
    if not other or not cycle_frames or member.get("retime") != "loop":
        return None
    them = next((m for m in cast if m["character"] == other), None)
    if them is None or them.get("take") != member.get("take") \
            or them.get("opponent") != member["character"]:
        return None
    order = [m["character"] for m in cast]
    first = order.index(member["character"]) < order.index(other)
    return 0 if first else int(cycle_frames) // 2


def _phase(character_id, cycle_frames):
    """A stable per-character phase into a shared take.

    Rostam and Sohrab draw one ``idle_human`` cache entry between them, and two
    characters breathing in unison reads as one puppet with two bodies. Derived
    from the character id rather than from the shot so that a character's idle
    is continuous across cuts, and from a digest rather than ``hash()`` because
    Python's string hash is salted per process and would give a different film
    on every run.
    """
    if not cycle_frames:
        return 0
    digest = hashlib.sha256(character_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(cycle_frames)


# -- timing (section 5.2) ----------------------------------------------------
def _timing(params, aligns):
    """The master clock. Returns the timing block and the per-line placements."""
    fps = params["fps"]
    head = params["head_handle_s"]
    tail = params["tail_handle_s"]
    minimum = params["min_shot_s"]

    if not params["dialogue"]:
        silent = float(params.get("silent_duration_s") or 0.0)
        if silent <= 0.0:
            raise ValueError(
                f"{params['shot_id']}: a shot with no dialogue needs "
                "silent_duration_s — with no audio there is nothing to derive "
                "its length from (§5.2)")
        duration = max(silent, minimum)
        return {
            "source": "silent_duration",
            "speech_start_s": 0.0,
            "speech_end_s": 0.0,
            "head_handle_s": head,
            "tail_handle_s": tail,
            "duration_s": round(duration, 5),
            "min_duration_s": minimum,
            "padded_s": round(max(0.0, duration - silent), 5),
        }, []

    placements, cursor = [], head
    for line in params["dialogue"]:
        align = aligns[line["line_id"]]
        speech = max(1.0 / fps, align["speech_end_s"] - align["speech_start_s"])
        start = cursor if line.get("starts_at_s") is None \
            else float(line["starts_at_s"])
        placements.append({"line": line, "align": align, "start_s": start,
                           "speech_s": speech})
        cursor = start + speech + INTER_LINE_GAP_S

    speech_end = max(p["start_s"] + p["speech_s"] for p in placements)
    duration = speech_end + tail
    padded = max(0.0, minimum - duration)
    return {
        "source": "audio",
        "speech_start_s": round(min(p["start_s"] for p in placements), 5),
        "speech_end_s": round(speech_end, 5),
        "head_handle_s": head,
        "tail_handle_s": tail,
        "duration_s": round(duration + padded, 5),
        "min_duration_s": minimum,
        "padded_s": round(padded, 5),
    }, placements


def _audio_tracks(params, placements, frame_start, project):
    """Where each recording sits in the shot, in the muxer's own units."""
    fps = params["fps"]
    trim = (params["silence_lead_policy"] or "trim") == "trim"
    tracks, notes = [], []
    for placement in placements:
        align = placement["align"]
        lead = align["speech_start_s"]
        if trim:
            strip_start_s = placement["start_s"]
            trim_start_s = lead
            duration = placement["speech_s"]
        else:
            strip_start_s = placement["start_s"] - lead
            trim_start_s = 0.0
            duration = align["duration_s"]
            if strip_start_s < 0.0:
                # The lead-in is longer than the handle, so keeping it whole
                # would start the strip before frame one. Clip exactly the
                # overflow and say so rather than sliding the speech.
                trim_start_s = -strip_start_s
                duration -= trim_start_s
                strip_start_s = 0.0
                notes.append(
                    f"{placement['line']['line_id']}: silence_lead_policy is "
                    f"'keep' but the take's {lead:.2f} s lead-in does not fit "
                    f"in a {params['head_handle_s']:.2f} s handle; "
                    f"{trim_start_s:.2f} s was clipped")
        tracks.append({
            "line_id": placement["line"]["line_id"],
            "character": placement["line"].get("character"),
            "path": align["audio"],
            "start_frame": frame_start + int(round(strip_start_s * fps)),
            "trim_start_s": round(max(0.0, trim_start_s), 5),
            "duration_s": round(max(1.0 / fps, duration), 5),
            "text_fa": placement["line"].get("text_fa", ""),
        })
    return tracks, notes


# -- camera ------------------------------------------------------------------
# How close to the image's own viewpoint a shot camera on an image set may
# come from behind. The set is a photograph pushed into depth; behind the place
# it was taken from, the camera sees the grid's near edge.
IMAGE_SET_CLEARANCE_M = 0.3
# The widest lens a clamped camera widens to; wider distorts faces.
MIN_FOCAL_MM = 18.0
IMAGE_SET_MAX_LOOKDOWN_DEG = 10.0


# An image set is one photograph pushed into depth. From the photograph's own
# viewpoint it is exact; every metre the camera moves away from that point
# shows the faces the depth stretched across each rock edge, as radial smears
# (the mourning shot's camera stood 8.9 m in front of its photograph). So a
# shot camera may come at most this share of the way from the viewpoint to the
# stage, and stay this near the viewpoint's height; it frames by lengthening
# the lens instead -- a telephoto look, with the depth of field softening the
# set, which is how a painted backdrop has always been shot.
PARALLAX_FORWARD_SHARE = 0.35
PARALLAX_HEIGHT_M = 0.7
PARALLAX_LATERAL_SHARE = 0.25
MAX_FOCAL_MM = 150.0


def _budget_floor_y(reference, stage_distance):
    ref_y = float(reference["position_m"][1])
    return ref_y + max(IMAGE_SET_CLEARANCE_M,
                       PARALLAX_FORWARD_SHARE * max(0.0, stage_distance))


def _parallax_budget(position, aim_point, focal, reference, stage_distance):
    """The camera kept inside the image set's parallax budget, the lens
    changed so the subject keeps the size in frame the plan asked for.
    Returns ``(position, focal, note)``."""
    ref = [float(v) for v in reference["position_m"]]
    before = _distance(position, aim_point)
    moved = list(position)
    floor_y = _budget_floor_y(reference, stage_distance)
    if moved[1] > floor_y and aim_point[1] > floor_y:
        # Back along the line of sight to the budget's edge.
        t = (aim_point[1] - floor_y) / max(1e-6, aim_point[1] - moved[1])
        moved = [aim_point[i] + (moved[i] - aim_point[i]) * t
                 for i in range(3)]
    elif moved[1] < ref[1] + IMAGE_SET_CLEARANCE_M:
        t = ((aim_point[1] - ref[1] - IMAGE_SET_CLEARANCE_M)
             / max(1e-6, aim_point[1] - moved[1]))
        moved = [aim_point[i] + (moved[i] - aim_point[i]) * t
                 for i in range(3)]
    moved[2] = min(ref[2] + PARALLAX_HEIGHT_M,
                   max(ref[2] - PARALLAX_HEIGHT_M, moved[2]))
    lateral = PARALLAX_LATERAL_SHARE * max(1.0, aim_point[1] - moved[1])
    moved[0] = min(ref[0] + lateral, max(ref[0] - lateral, moved[0]))
    after = _distance(moved, aim_point)
    if max(abs(moved[i] - position[i]) for i in range(3)) < 1e-4:
        return position, focal, None
    new_focal = min(MAX_FOCAL_MM,
                    max(MIN_FOCAL_MM, focal * after / max(1e-6, before)))
    note = (f"camera kept near the set's viewpoint: {before:.2f} m from the "
            f"subject became {after:.2f} m, lens {focal:.0f} -> "
            f"{new_focal:.0f} mm (an image set is exact only near the place "
            f"its picture was taken)")
    return [round(v, 5) for v in moved], round(new_focal, 3), note


def _camera(params, pack, manifests, aim_point, frame_start, frame_count,
            env_manifest=None):
    """The resolved camera: where it is, what it looks at, and how it moves."""
    spec = params["camera"]
    position = [float(v) for v in (spec.get("position_m") or [0.0, -6.0, 1.6])]
    planned = list(position)
    focal = float(spec.get("focal_mm") or pack.camera.get("focal_mm") or 50.0)
    distance = _distance(position, aim_point)
    image_set = (env_manifest or {}).get("strategy") == "image_set"
    reference = (env_manifest or {}).get("reference_camera") or {}

    # Section 8 gives the miniature pack a near-orthographic camera. A real
    # orthographic projection would flatten the backdrop planes into one another,
    # so the compression is done the way it is done on a set: pull back and
    # lengthen the lens by the same factor, which preserves the subject's size
    # in frame and takes the perspective out of it.
    # An image set is already exactly as flat as the photograph it was built
    # from, and pulling the camera back 1.75x would put it behind the place
    # that photograph was taken.
    bias = 0.0 if image_set else float(pack.camera.get("orthographic_bias")
                                       or 0.0)
    ceiling = float(pack.camera.get("focal_max_mm") or DEFAULT_FOCAL_MAX_MM)
    compression = min(MAX_COMPRESSION, 1.0 + 3.0 * max(0.0, bias))
    if compression > 1.0 and distance > 1e-6:
        capped = max(1.0, min(compression, ceiling / max(1e-6, focal)))
        position = [round(aim_point[i] + (position[i] - aim_point[i]) * capped, 5)
                    for i in range(3)]
        focal = round(focal * capped, 3)
        compression = capped
        distance = _distance(position, aim_point)

    clamped_note = None
    if image_set and reference.get("position_m"):
        position, focal, clamped_note = _parallax_budget(
            position, aim_point, focal, reference,
            float(env_manifest.get("stage_distance_m") or 0.0))
        distance = _distance(position, aim_point)
    floor_y = None
    if image_set and reference.get("position_m"):
        floor_y = (float(reference["position_m"][1]) + IMAGE_SET_CLEARANCE_M,
                   _budget_floor_y(reference,
                                   float(env_manifest.get("stage_distance_m")
                                         or 0.0)))
        # Looking steeply down on an image set shows its near ground, which the
        # photograph saw at a grazing angle from far away: the picture there is
        # a few rows of pixels stretched over metres (the mourning shot's
        # smear). So the look-down is capped; the height comes down, not the
        # distance.
        horizontal = math.hypot(position[0] - aim_point[0],
                                position[1] - aim_point[1])
        rise = position[2] - aim_point[2]
        limit = math.tan(math.radians(IMAGE_SET_MAX_LOOKDOWN_DEG))
        if horizontal > 1e-6 and rise / horizontal > limit:
            position[2] = round(aim_point[2] + horizontal * limit, 5)
            clamped_note = ((clamped_note + "; ") if clamped_note else "") + (
                f"look-down capped at {IMAGE_SET_MAX_LOOKDOWN_DEG:.0f} deg on "
                "an image set")
            distance = _distance(position, aim_point)
    camera = {
        "clamped_note": clamped_note,
        "position_m": position,
        "aim_point_m": [round(v, 5) for v in aim_point],
        "aim_target": spec.get("aim") if isinstance(spec.get("aim"), str)
                      else None,
        "focal_mm": focal,
        "dof": bool(spec.get("dof", pack.camera.get("dof"))),
        "focus_distance_m": round(distance, 4),
        "orthographic": False,
        "ortho_scale": None,
        "move": _camera_move(spec.get("move"), position, aim_point,
                             compression, frame_start, frame_count,
                             planned=planned, floor_y=floor_y),
    }
    return camera, round(compression, 4)


def _camera_move(move, position, aim_point, compression, frame_start,
                 frame_count, planned=None, floor_y=None):
    """A camera move in frames, resolved from the plan's normalised times.

    The destination is compressed by the same factor as the start, about the
    same point. Compressing only the start would turn a 60 cm push into a
    several-metre lunge at the miniature pack's 1.75x, which is the kind of
    thing that looks like a camera bug and is a units bug.
    """
    if not move or move.get("kind") in (None, "static"):
        return None
    span = max(1, frame_count - 1)
    from_frame = frame_start + int(round(float(move.get("from_pct") or 0.0)
                                         * span))
    to_frame = frame_start + int(round(float(move.get("to_pct") or 1.0) * span))
    out = {
        "kind": move["kind"],
        "from_frame": from_frame,
        "to_frame": max(to_frame, from_frame + 1),
        "ease": move.get("ease") or "in_out",
        "from_position_m": [round(v, 5) for v in position],
    }
    if move.get("to_position_m"):
        target = [aim_point[i]
                  + (float(move["to_position_m"][i]) - aim_point[i])
                  * compression for i in range(3)]
        if planned is not None:
            # The plan's destination is relative to the plan's start. When the
            # start was moved (clamped, capped), the move goes with it: a push
            # keeps its ratio to the aim, a pan or crane its offset.
            base = [aim_point[i] + (planned[i] - aim_point[i]) * compression
                    for i in range(3)]
            if move.get("kind") in ("push_in", "pull_out"):
                ratio = (_distance(target, aim_point)
                         / max(1e-6, _distance(base, aim_point)))
                target = [aim_point[i] + (position[i] - aim_point[i]) * ratio
                          for i in range(3)]
            else:
                target = [position[i] + target[i] - base[i] for i in range(3)]
        if floor_y is not None:
            # The move stays inside the image set's viewpoint budget too.
            target[1] = min(floor_y[1], max(floor_y[0], target[1]))
        out["to_position_m"] = [round(v, 5) for v in target]
    if move.get("orbit_deg") is not None:
        out["orbit_deg"] = float(move["orbit_deg"])
    return out


def ease(fraction, kind):
    """The shape of a camera move, as a value rather than as an fcurve.

    Smoothstep for ``in_out``, quadratic for ``in`` and ``out``, linear
    otherwise. build_shot bakes this into the keyed positions instead of setting
    keyframe interpolation, so a move eases the same way whoever opens the file
    and whatever they do to the curve handles - and so the arithmetic can be
    tested here, where there is no bpy.
    """
    if kind == "in_out":
        return fraction * fraction * (3.0 - 2.0 * fraction)
    if kind == "in":
        return fraction * fraction
    if kind == "out":
        return 1.0 - (1.0 - fraction) ** 2
    return fraction


# -- the host step -----------------------------------------------------------
def resolve(job, unit):
    """Turn the shot's intent into section 6.3's fully resolved ``shot.json``."""
    params = unit["params"]
    project = job.project
    shot_id = params["shot_id"]
    fps = params["fps"]
    pack = job.style()

    manifests = {}
    for member in params["cast"]:
        manifests[member["character"]] = jsonio.read(
            unit["inputs"][f"char_{member['character']}_manifest"], default={})
    env_manifest = jsonio.read(unit["inputs"]["env_manifest"], default={})

    aligns = {}
    for line in params["dialogue"]:
        aligns[line["line_id"]] = jsonio.read(
            unit["inputs"][f"align_{line['line_id']}"], default={})

    timing, placements = _timing(params, aligns)
    frame_start = 1
    frame_count = max(1, int(math.ceil(timing["duration_s"] * fps - 1e-9)))
    frame_end = frame_start + frame_count - 1

    tracks, notes = _audio_tracks(params, placements, frame_start, project)

    # Which frame each line's mouth curves start on. Frame 0 of visemes.json is
    # align.speech_start_s, so this is the same expression that placed the
    # trimmed audio strip - deliberately, so the two cannot drift.
    viseme_start = {p["line"]["line_id"]: frame_start
                    + int(round(p["start_s"] * fps)) for p in placements}
    speakers = {}
    for placement in placements:
        speakers.setdefault(placement["line"]["character"], []).append(
            placement["line"]["line_id"])

    cast, provenance_cast = [], {}
    for member in params["cast"]:
        cid = member["character"]
        manifest = manifests.get(cid) or {}
        body = jsonio.read(unit["inputs"][f"take_{cid}_body"], default={})
        cycle = int(body.get("cycle_frames") or 0)
        offset = member["offset_frames"]
        derived = offset is None
        if derived:
            offset = _phase(cid, cycle) if member["retime"] == "loop" else 0
            exchange = _exchange_phase(member, params["cast"], cycle)
            if exchange is not None:
                offset = exchange
            elif member.get("struck") and member["retime"] != "loop":
                # The blow lands, then he falls: the two takes start
                # together, and Siamak was already falling as the div
                # wound up.
                offset = int(round(REACT_DELAY_S * fps))
        cast.append({
            "character": cid,
            "name_fa": member.get("name_fa") or cid,
            "rig_mode": member["rig_mode"],
            "position_m": member["position_m"],
            "facing_deg": member["facing_deg"],
            "height_m": member["height_m"],
            "retime": member["retime"],
            "offset_frames": int(offset),
            "lean_deg": _lean_to(member, params["cast"]),
            "material_roles": member["material_roles"],
            "assets": {
                "body": project.rel(unit["inputs"][f"char_{cid}_blend"]),
                "rig": project.rel(unit["inputs"][f"char_{cid}_manifest"]),
                "shape": None,
                "face": None,
                "albedo": None,
                "garments": [],
                "motion": project.rel(unit["inputs"][f"take_{cid}_body"]),
                "head_pose": project.rel(unit["inputs"][f"take_{cid}_head"]),
                "visemes": [project.rel(unit["inputs"][f"visemes_{line}"])
                            for line in speakers.get(cid, [])],
            },
        })
        provenance_cast[cid] = {
            "take": member["take"],
            "take_cycle_frames": cycle,
            "take_reference_height_m": body.get("reference_height_m"),
            "root_scale": round(member["height_m"]
                                / float(body.get("reference_height_m") or 1.7),
                                5),
            "offset_derived": derived,
            "collection": manifest.get("collection"),
            "mesh": manifest.get("mesh"),
            "rig": manifest.get("rig"),
            "garments": [g.get("id") for g in manifest.get("garments") or []],
            "attach": manifest.get("attach") or {},
            "props": member["props"],
            "lines": speakers.get(cid, []),
            "viseme_start_frame": {line: viseme_start[line]
                                   for line in speakers.get(cid, [])},
        }

    aim_point, aim_note = _aim(params, manifests, provenance_cast)
    if aim_note:
        notes.append(aim_note)
    camera, compression = _camera(params, pack, manifests, aim_point,
                                  frame_start, frame_count, env_manifest)
    clamped = camera.pop("clamped_note", None)
    if clamped:
        notes.append(clamped)

    beats = []
    for beat in params["beats"]:
        beats.append({
            "from_frame": frame_start + int(round(float(beat.get("from_pct")
                                                        or 0.0)
                                                  * (frame_count - 1))),
            "to_frame": frame_start + int(round(float(beat.get("to_pct") or 1.0)
                                                * (frame_count - 1))),
            "what": beat.get("what", ""),
        })

    shot = {
        "schema_version": 1,
        "shot_id": shot_id,
        "title": params.get("title") or shot_id,
        "fps": fps,
        "resolution": params["resolution"],
        "style_pack": pack.name,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": frame_count,
        "timing": timing,
        "env": {
            "id": params["env"]["id"],
            "strategy": env_manifest.get("strategy")
                        or pack.env.get("strategy") or "procedural",
            "interior": params["env"]["interior"],
            "ground_role": params["env"]["ground_role"],
            "light_intent": params["env"]["light_intent"],
            "blend": project.rel(unit["inputs"]["env_blend"]),
            "backdrop": env_manifest.get("backdrop_plate"),
            "hdri": None,
            "props_json": None,
            "scatter": params["env"]["scatter"],
        },
        "cast": cast,
        "camera": camera,
        "audio": {"tracks": tracks},
        "beats": beats,
        "provenance": {
            "stage": "h_assemble",
            "cast": provenance_cast,
            "env_collection": env_manifest.get("collection"),
            "props": {pid: {
                "blend": project.rel(unit["inputs"][f"prop_{pid}_blend"]),
                "manifest": project.rel(unit["inputs"][f"prop_{pid}_manifest"]),
            } for pid in sorted(params["props"])},
            "viseme_profile": pack.visemes.get("profile"),
            "viseme_start_frame": viseme_start,
            "silence_lead_policy": params["silence_lead_policy"],
            "perspective_compression": compression,
            "align": {line_id: {
                "backend": align.get("backend"),
                "approximate": align.get("approximate"),
                "speech_start_s": align.get("speech_start_s"),
                "speech_end_s": align.get("speech_end_s"),
            } for line_id, align in sorted(aligns.items())},
        },
    }

    report = validate.check_schema(shot, "shot")
    jsonio.write(_base.out_path(unit, "shot.json"), shot)

    return {
        "metrics": {
            "shot_id": shot_id,
            "fps": fps,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_count": frame_count,
            "duration_s": timing["duration_s"],
            "timing_source": timing["source"],
            "speech_start_s": timing["speech_start_s"],
            "speech_end_s": timing["speech_end_s"],
            "cast": [m["character"] for m in cast],
            "retimes": {m["character"]: m["retime"] for m in cast},
            "offsets": {m["character"]: m["offset_frames"] for m in cast},
            "lines": [t["line_id"] for t in tracks],
            "audio_start_frames": {t["line_id"]: t["start_frame"]
                                   for t in tracks},
            "viseme_start_frames": viseme_start,
            "aim_target": camera["aim_target"],
            "aim_point_m": camera["aim_point_m"],
            "focal_mm": camera["focal_mm"],
            "focus_distance_m": camera["focus_distance_m"],
            "perspective_compression": compression,
            "camera_move": (camera["move"] or {}).get("kind") or "static",
            "beats": len(beats),
            "shot_schema_errors": [str(p) for p in report.errors],
            "shot_schema_warnings": [str(p) for p in report.warnings],
        },
        "notes": [f"{shot_id}: {frame_count} frames "
                  f"({timing['duration_s']:.2f} s at {fps} fps) from "
                  f"{timing['source']}, cast "
                  f"{', '.join(m['character'] for m in cast)}, "
                  f"{len(tracks)} audio track(s)"] + notes,
    }


def _aim(params, manifests, provenance_cast):
    """Resolve ``camera.aim`` to a world point, however it was written."""
    aim = (params["camera"] or {}).get("aim")
    if isinstance(aim, (list, tuple)) and len(aim) == 3:
        return [float(v) for v in aim], None
    if not isinstance(aim, str):
        return [0.0, 0.0, 1.5], "no camera aim; aiming at the origin"

    target, _, anchor = aim.partition(".")
    if target == "env":
        for member in params["cast"]:
            if anchor in (member.get("props") or []):
                record = (manifests.get(member["character"]) or {})
                attach = (record.get("attach") or {})
                held = params["props"].get(anchor) or {}
                slot = attach.get(held.get("attach") or "hand_r")
                if slot:
                    return _place(slot["head_m"], member["position_m"],
                                  member["facing_deg"]), None
        return [0.0, 0.0, 1.0], None

    member = next((m for m in params["cast"] if m["character"] == target), None)
    if member is None:
        return [0.0, 0.0, 1.5], (
            f"the camera aims at {target!r}, who is not in this shot; aiming at "
            "the origin instead")
    point, problem = _anchor_point(manifests.get(target) or {}, member,
                                   anchor or "chest")
    if point is None:
        fallback = [member["position_m"][0], member["position_m"][1],
                    member["position_m"][2] + 0.6 * member.get("height_m", 1.75)]
        return fallback, (f"{target}.{anchor}: {problem}; aiming at "
                          "mid-height instead")
    return point, None


# -- the gate ----------------------------------------------------------------
def _gate(project, unit, metrics):
    out = [v for v in gates.check_assembly(metrics) if v.level != gates.PASS]

    for problem in metrics.get("shot_schema_errors") or []:
        out.append(gates.Verdict(
            gates.FAIL, "assembly.schema",
            f"shot.json does not validate: {problem}"))

    # Section 5.2, checked rather than assumed: the frame range has to be the
    # one the audio implies, because everything downstream trusts it.
    fps = metrics.get("fps") or 24
    expected = int(math.ceil((metrics.get("duration_s") or 0.0) * fps - 1e-9))
    count = metrics.get("frame_count") or 0
    if count != expected:
        out.append(gates.Verdict(
            gates.FAIL, "assembly.frames",
            f"{count} frames for a {metrics.get('duration_s')} s shot at "
            f"{fps} fps", count, expected))
    if metrics.get("built_frame_range") \
            and metrics["built_frame_range"] != [metrics.get("frame_start"),
                                                 metrics.get("frame_end")]:
        out.append(gates.Verdict(
            gates.FAIL, "assembly.range",
            f"the .blend renders {metrics['built_frame_range']} against the "
            f"resolved {[metrics.get('frame_start'), metrics.get('frame_end')]}"))

    for cid, keyed in (metrics.get("motion_bones") or {}).items():
        if not keyed:
            out.append(gates.Verdict(
                gates.FAIL, "assembly.motion",
                f"{cid} stands in the shot with no pose keyed on any bone"))
    for cid, missing in (metrics.get("motion_missing_bones") or {}).items():
        if missing:
            out.append(gates.Verdict(
                gates.FAIL, "assembly.motion_bones",
                f"{cid}: the take rotates {', '.join(missing[:4])}, which this "
                "rig has not got"))

    for line_id, keyed in (metrics.get("viseme_keys") or {}).items():
        if not keyed:
            out.append(gates.Verdict(
                gates.FAIL, "assembly.visemes",
                f"{line_id} is spoken in this shot and keyed no mouth channel"))

    if metrics.get("env_strategy") == "painted_backdrop" \
            and metrics.get("env_plate"):
        if not metrics.get("materials_kept"):
            out.append(gates.Verdict(
                gates.FAIL, "assembly.backdrop",
                f"{metrics.get('env_plate')} was painted onto the backdrop and "
                "the style pack swapped it back to a flat colour, so the "
                "reference image does not appear in the shot"))
        for plate, size in sorted((metrics.get("env_plate_px") or {}).items()):
            if not size or not all(size):
                out.append(gates.Verdict(
                    gates.FAIL, "assembly.backdrop_pixels",
                    f"{plate} is linked into the backdrop and carries no "
                    f"pixels ({size}); it renders as flat black, which looks "
                    "like a style choice rather than a missing image",
                    size, "non-zero"))
        if not metrics.get("backdrop_plate_visible"):
            out.append(gates.Verdict(
                gates.FAIL, "assembly.backdrop_hidden",
                f"{metrics.get('env_plate')} is behind the land bands in this "
                f"camera: they reach {metrics.get('backdrop_land_top')} of "
                f"frame height and the plate reaches "
                f"{metrics.get('backdrop_plate_top')}, so none of the painted "
                "image is in shot",
                metrics.get("backdrop_land_top"), 1.0))
        elif (metrics.get("backdrop_plate_frame_fraction") or 0.0) < 0.05:
            out.append(gates.Verdict(
                gates.WARN, "assembly.backdrop_sliver",
                f"{metrics.get('env_plate')} shows over only "
                f"{metrics.get('backdrop_plate_frame_fraction'):.1%} of the "
                "frame height; the painted backdrop is a sliver at the top",
                metrics.get("backdrop_plate_frame_fraction"), 0.05))

    for name in metrics.get("nan_transforms") or []:
        out.append(gates.Verdict(gates.FAIL, "assembly.nan",
                                 f"{name} has a non-finite transform"))

    for prop_id, attached in (metrics.get("props_attached") or {}).items():
        if not attached:
            out.append(gates.Verdict(
                gates.WARN, "assembly.props",
                f"{prop_id} is in the shot but is parented to nothing; it will "
                "not move with the hand holding it"))

    for problem in metrics.get("shot_schema_warnings") or []:
        out.append(gates.Verdict(gates.WARN, "assembly.schema", problem))

    return out or [gates.Verdict(
        gates.PASS, "assembly",
        f"{metrics.get('shot_id')}: {metrics.get('frame_count')} frames, "
        f"{len(metrics.get('cast') or [])} in shot, cast fills "
        f"{(metrics.get('subject_frame_fraction') or 0.0):.0%} of frame, "
        f"{metrics.get('objects')} objects all with materials")]


STAGE = StageSpec(
    name="h_assemble",
    # 2: a speaker's head is not tipped back; the taller fighter leans.
    version="2",
    summary="resolve each shot against the audio clock and compile it to a .blend",
    needs=["a_production", "c_paint", "d_env", "e_motion", "g_align"],
    steps=[Step("host"), Step("blender", "pipeline/blender/build_shot.py")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(resolve))
