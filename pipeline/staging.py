"""Words to metres: where the cast stands and where the camera goes.

The main prompt stages a shot in a small vocabulary -- "left, back, facing
siamak"; "medium on keyumars, low angle, slow push in" -- and this module is the
only place those words become numbers. Pure Python, no Blender, so the host can
compute every coordinate before anything expensive runs and the tests can check
the arithmetic directly.

**The stage.** The origin is a point on the ground in the lower middle of the
scene image, where ``scene_depth`` puts it, and the image's own camera stands
on -Y looking +Y. So the cast is laid out across X and a few metres along Y,
and every shot camera stays on the -Y side within :data:`MAX_YAW_DEG` of the
image's line of sight: a set built from one photograph looks right from near
where that photograph was taken and nowhere else.

**The camera** is solved from the shot size the way a camera operator thinks:
a size says how much of the subject fills the frame height, the lens is chosen
for the size, and the distance follows from the lens's field of view. The aim
point sits at the height the size is about (the eyes for a close-up, the chest
for a medium) and the camera swings a little to the side the subject faces, so
a character in profile is not shot in profile.
"""

from __future__ import annotations

import math
import re

X_SLOTS = {"far_left": -3.2, "left": -1.7, "center": 0.0, "centre": 0.0,
           "right": 1.7, "far_right": 3.2}
DEPTH_SLOTS = {"front": -1.0, "middle": 0.8, "back": 2.6}
MIN_SPACING_M = 0.95

# (frame height as a multiple of the subject's height, lens mm, aim height as a
# fraction of the subject's height)
SIZES = {
    "extreme_wide": (3.6, 24.0, 0.50),
    "wide": (2.3, 28.0, 0.52),
    "full": (1.25, 35.0, 0.55),
    # A medium keeps the whole head with room above it: at 0.62 the frame's
    # top edge fell through Soroush's hair once his push-in tightened it.
    "medium": (0.74, 45.0, 0.74),
    "medium_close": (0.50, 55.0, 0.81),
    "close_up": (0.25, 70.0, 0.89),
    "extreme_close_up": (0.15, 90.0, 0.93),
}
SIZE_WORDS = {"ews": "extreme_wide", "ws": "wide", "ms": "medium",
              "mcu": "medium_close", "cu": "close_up",
              "ecu": "extreme_close_up", "long": "wide",
              "establishing": "extreme_wide"}
# Blender's default sensor: 36 mm wide, so 20.25 mm tall at 16:9.
SENSOR_H_MM = 36.0 * 9.0 / 16.0
MAX_YAW_DEG = 32.0
YAW_PER_FACING = 0.3
HIGH_ANGLE_DEG = 25.0
LOW_ANGLE_DEG = 15.0
CLOSE_LOW_ANGLE_DEG = 5.0
CLOSE_SIZES = ("medium_close", "close_up", "extreme_close_up")
EYE_LEVEL_M = 1.6
MIN_CAMERA_Z_M = 0.3
MOVES = ("static", "push_in", "pull_out", "pan_left", "pan_right",
         "orbit_left", "orbit_right", "crane_up", "crane_down")
ANGLES = ("eye_level", "low_angle", "high_angle")
# For a quadruped the frame is sized by its length as much as its height.
QUADRUPED_KINDS = ("quadruped", "creature", "animal")


def _norm(text):
    return re.sub(r"[\s\-,;]+", "_", (text or "").strip().lower()).strip("_")


def facing_to(source, target):
    """Degrees for a character at ``source`` to face ``target``.

    ``facing_deg`` is the rig's rotation about +Z, counter-clockwise, applied
    to a body that faces +Y at rest (``build_shot`` passes it straight to
    ``prelude.place``). So a facing of ``f`` looks along ``(-sin f, cos f)``:
    0 is +Y, 90 is **-X**, 180 is toward the camera on -Y. The schema's
    description once said 90 was +X; the code never did.
    """
    dx, dy = target[0] - source[0], target[1] - source[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 180.0
    return round(math.degrees(math.atan2(-dx, dy)), 2)


def _wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


# How far from facing the camera a character turned toward someone may end up.
# Actors on a stage "cheat out": a character downstage addressing someone
# upstage turns only part of the way, so the audience sees a three-quarter
# face rather than a back. Unclamped, Soroush delivered his warning with his
# back square to the lens.
MAX_TURN_FROM_CAMERA_DEG = 75.0
# The shot's subject and anyone speaking are cheated harder: their face is
# what the shot is for, and MuseTalk repaints a mouth only within about 60
# degrees of frontal. An image set cannot be shot from the reverse angle (its
# picture exists only from its own viewpoint), so the face has to come to the
# camera rather than the camera going round to the face.
FEATURED_TURN_DEG = 35.0


def _cheat(degrees, limit=MAX_TURN_FROM_CAMERA_DEG):
    """A facing turned toward the camera until no more than ``limit``."""
    away = _wrap(degrees - 180.0)
    if abs(away) <= limit:
        return degrees
    return round(180.0 + (limit if away > 0 else -limit), 2)


def place_cast(members, featured=()):
    """Positions and facings for one shot's cast.

    ``members`` is ``[{"character", "placement", "height_m", "kind"}]`` with the
    placement text as written; ``featured`` names the shot's subject and its
    speakers, who are cheated to ``FEATURED_TURN_DEG``. Returns ``{id:
    {"position_m", "facing_deg", "facing_target"}}`` and a list of notes about
    anything assumed.
    """
    notes, spots = [], {}
    unplaced = []
    for member in members:
        words = [_norm(w) for w in re.split(r",", member.get("placement") or "")]
        x = y = None
        facing = None
        for word in words:
            if word in X_SLOTS:
                x = X_SLOTS[word]
            elif word in DEPTH_SLOTS:
                y = DEPTH_SLOTS[word]
            elif word.startswith("facing_"):
                facing = word[len("facing_"):]
            else:
                parts = word.split("_")
                for part in parts:
                    if part in DEPTH_SLOTS and y is None:
                        y = DEPTH_SLOTS[part]
                    if part in X_SLOTS and x is None:
                        x = X_SLOTS[part]
        if x is None:
            unplaced.append(member["character"])
        spots[member["character"]] = {"x": x, "y": 0.8 if y is None else y,
                                      "facing": facing or "camera"}
    # Anyone without an x is spread across the gaps, centre first.
    if unplaced:
        free = [v for v in (0.0, 1.7, -1.7, 3.2, -3.2)
                if v not in {s["x"] for s in spots.values()}]
        for index, cid in enumerate(unplaced):
            spots[cid]["x"] = free[index % len(free)] if free else index * 1.0
            notes.append(f"{cid}: no stage position given, placed at x="
                         f"{spots[cid]['x']}")
    # Two characters on one mark are pushed apart along x.
    ordered = sorted(spots.items(), key=lambda kv: (kv[1]["y"], kv[1]["x"]))
    taken = []
    for cid, spot in ordered:
        for other in taken:
            if (abs(other["x"] - spot["x"]) < MIN_SPACING_M
                    and abs(other["y"] - spot["y"]) < MIN_SPACING_M):
                spot["x"] = round(other["x"] + MIN_SPACING_M
                                  * (1 if spot["x"] >= 0 else -1), 3)
                notes.append(f"{cid}: shared a mark, moved to x={spot['x']}")
        taken.append(spot)

    out = {}
    for cid, spot in spots.items():
        position = [spot["x"], spot["y"], 0.0]
        facing = spot["facing"]
        target = None
        if facing in ("camera", "viewer", "front"):
            degrees = 180.0
        elif facing in ("away", "back"):
            degrees = _cheat(0.0, FEATURED_TURN_DEG) if cid in featured                 else 0.0
        elif facing == "left":            # screen left, -X
            degrees = 90.0
        elif facing == "right":           # screen right, +X
            degrees = -90.0
        elif facing in spots:
            other = spots[facing]
            limit = (FEATURED_TURN_DEG if cid in featured
                     else MAX_TURN_FROM_CAMERA_DEG)
            degrees = _cheat(facing_to(position, (other["x"], other["y"])),
                             limit)
            target = facing
        else:
            degrees = 180.0
            notes.append(f"{cid}: 'facing {facing}' not understood; facing "
                         f"the camera")
        out[cid] = {"position_m": position, "facing_deg": degrees,
                    "facing_target": target}
    return out, notes


# Pelvis-to-pelvis distance for two characters in contact, as a fraction of
# their mean height. The placement words know only "left" and "right", 3.4 m
# apart: Siamak and the div "grappled" at that distance, swinging at air, and
# the div's claws felled Siamak from 2.5 m away.
CONTACT_GAP = {"fight": 0.50, "strike": 0.60, "block": 0.60,
               # Arm's length, not chest to chest: each plays an embrace of
               # his own, and at 0.26 Keyumars stood inside his son.
               "embrace": 0.55}


def engage(placed, clips, bodies):
    """Bring characters whose moves reach each other into range, facing.

    ``placed`` is :func:`place_cast`'s output, changed in place; ``clips``
    maps a character to its clip list; ``bodies`` to ``{"height_m"}``. A pair
    is engaged when either one's contact clip (:data:`CONTACT_GAP`) targets
    the other; the two close in on their midpoint to the clip's distance,
    side by side across the screen, and face each other square -- no cheat
    to the camera, which sees a fight best from the side. Returns notes.
    """
    notes = []
    done = set()
    for cid, clip_list in clips.items():
        for clip in clip_list or []:
            other = clip.get("target")
            gap = CONTACT_GAP.get(clip.get("clip"))
            if (gap is None or other not in placed or cid not in placed
                    or other == cid or frozenset((cid, other)) in done):
                continue
            done.add(frozenset((cid, other)))
            heights = [float((bodies.get(c) or {}).get("height_m") or 1.75)
                       for c in (cid, other)]
            distance = gap * sum(heights) / 2.0
            a = placed[cid]["position_m"]
            b = placed[other]["position_m"]
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
            half = distance / 2.0
            # Across the screen, left and right kept: along the line they
            # stood on, the div struck Siamak with Siamak's back to the lens.
            ux, uy = (1.0 if b[0] >= a[0] else -1.0), 0.0
            placed[cid]["position_m"] = [round(mid[0] - ux * half, 3),
                                         round(mid[1] - uy * half, 3), 0.0]
            placed[other]["position_m"] = [round(mid[0] + ux * half, 3),
                                           round(mid[1] + uy * half, 3), 0.0]
            for me, you in ((cid, other), (other, cid)):
                placed[me]["facing_deg"] = facing_to(
                    placed[me]["position_m"], placed[you]["position_m"])
                placed[me]["facing_target"] = you
                placed[me]["engaged"] = you
            notes.append(f"{cid} and {other}: in contact for "
                         f"{clip['clip']}, {distance:.2f} m apart (were "
                         f"{length:.2f} m)")
    return notes


def parse_camera(text, cast_ids):
    """``(size, subject, angle, move, slow)`` from a camera line, and notes."""
    notes = []
    words = _norm(text)
    size = None
    for name in sorted(SIZES, key=len, reverse=True):
        if name in words:
            size = name
            break
    if size is None:
        for short, name in SIZE_WORDS.items():
            if re.search(rf"(^|_){short}(_|$)", words):
                size = name
                break
    if size is None:
        size = "medium"
        notes.append("no shot size; using medium")
    subject = None
    match = re.search(r"(?:^|_)on_([a-z][a-z0-9_]*?)(?:_(?:eye|low|high|"
                      + "|".join(m.split("_")[0] for m in MOVES)
                      + r"|slow)|$)", words)
    if match:
        candidate = match.group(1).strip("_")
        if candidate in cast_ids or candidate in ("group", "all", "everyone"):
            subject = candidate
        else:
            for cid in sorted(cast_ids, key=len, reverse=True):
                if candidate.startswith(cid):
                    subject = cid
                    break
    if subject is None:
        for cid in cast_ids:
            if cid in words:
                subject = cid
                break
    if subject in ("all", "everyone"):
        subject = "group"
    if subject is None:
        subject = "group" if len(cast_ids) > 1 else (cast_ids[0] if cast_ids
                                                     else "group")
        notes.append(f"no subject; framing {subject}")
    angle = "eye_level"
    for name in ANGLES:
        if name.split("_")[0] + "_angle" in words or name in words:
            angle = name
    if "low" in words.split("_") and angle == "eye_level":
        angle = "low_angle"
    if "high" in words.split("_") and angle == "eye_level":
        angle = "high_angle"
    move = "static"
    for name in MOVES:
        if name in words:
            move = name
    slow = "slow" in words.split("_")
    return (size, subject, angle, move, slow), notes


def _subject_box(subject, placed, bodies):
    """Aim point basis: position, height and horizontal width of the subject."""
    if subject != "group" and subject in placed:
        pos = placed[subject]["position_m"]
        body = bodies.get(subject) or {}
        height = float(body.get("height_m") or 1.75)
        if body.get("kind") in QUADRUPED_KINDS:
            height = max(height, 0.8 * height + 0.5)
        # Two in contact stand side-on by design (``engage``): the camera
        # does not swing round to find a face, which in s08 swung it past
        # the edge of a set made from one photograph.
        facing = 180.0 if placed[subject].get("engaged") \
            else placed[subject]["facing_deg"]
        return pos, height, 0.8, facing
    xs = [p["position_m"][0] for p in placed.values()] or [0.0]
    ys = [p["position_m"][1] for p in placed.values()] or [0.0]
    heights = [float((bodies.get(c) or {}).get("height_m") or 1.75)
               for c in placed] or [1.75]
    centre = [(min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, 0.0]
    return centre, max(heights), (max(xs) - min(xs)) + 1.4, 180.0


def solve_camera(text, placed, bodies, aspect=16.0 / 9.0):
    """The shot's camera in production.json's terms, and notes.

    ``placed`` is :func:`place_cast`'s output; ``bodies`` maps a character id to
    ``{"height_m", "kind"}``.
    """
    (size, subject, angle, move, slow), notes = parse_camera(text,
                                                             list(placed))
    frame_mult, lens, aim_frac = SIZES[size]
    base, height, width, facing = _subject_box(subject, placed, bodies)
    frame_h = frame_mult * height
    if subject == "group":
        frame_h = max(frame_h, width / aspect * 1.15)
    distance = frame_h * lens / SENSOR_H_MM
    aim = [base[0], base[1], round(aim_frac * height, 3)]

    # Swing toward the side the subject faces: a facing of -90 looks along +X,
    # so the camera moves to +X to see the face.
    yaw = max(-MAX_YAW_DEG, min(MAX_YAW_DEG,
                                YAW_PER_FACING * _wrap(facing - 180.0)))
    # The distance above is along the line of sight, so an angle tilts that
    # line rather than lifting the camera straight up: 3 m up for a medium
    # shot put the camera nearly overhead the first time this was written.
    elevation = {"high_angle": HIGH_ANGLE_DEG,
                 "low_angle": -LOW_ANGLE_DEG}.get(angle, 0.0)
    if angle == "low_angle" and size in CLOSE_SIZES:
        # Close on a face, a low camera looks up the nose and under the
        # chin -- the one part of a character no view of its painting saw.
        elevation = -CLOSE_LOW_ANGLE_DEG
    if angle == "eye_level" and size in ("extreme_wide", "wide", "full"):
        # A wide shot at eye level is a person's eye level, not the subject's
        # waist, which is where its aim point is.
        cam_z = EYE_LEVEL_M
        horizontal = math.sqrt(max(0.25, distance ** 2
                                   - (cam_z - aim[2]) ** 2))
    else:
        cam_z = aim[2] + distance * math.sin(math.radians(elevation))
        if cam_z < MIN_CAMERA_Z_M:
            cam_z = MIN_CAMERA_Z_M
        horizontal = math.sqrt(max(0.25, distance ** 2
                                   - (cam_z - aim[2]) ** 2))
    rad = math.radians(yaw)
    position = [round(aim[0] + horizontal * math.sin(rad), 3),
                round(aim[1] - horizontal * math.cos(rad), 3),
                round(cam_z, 3)]

    camera = {"position_m": position,
              "aim": (f"{subject}.{_anchor(size)}" if subject != "group"
                      else [round(v, 3) for v in aim]),
              "focal_mm": lens}
    if move != "static":
        span = (0.0, 1.0) if slow else (0.1, 0.9)
        spec = {"from_pct": span[0], "to_pct": span[1], "ease": "in_out"}
        if move in ("push_in", "pull_out"):
            factor = (0.72 if move == "push_in" else 1.35)
            if slow:
                factor = 1.0 + (factor - 1.0) * 0.6
            spec["kind"] = move
            spec["to_position_m"] = [round(aim[i] + (position[i] - aim[i])
                                           * factor, 3) for i in range(3)]
        elif move in ("pan_left", "pan_right"):
            shift = (-1 if move == "pan_left" else 1) * 0.18 * distance
            spec["kind"] = "pan"
            spec["to_position_m"] = [round(position[0] + shift, 3),
                                     position[1], position[2]]
        elif move in ("orbit_left", "orbit_right"):
            spec["kind"] = "orbit"
            # A set made from one photograph shows its edge past about ten
            # degrees round: at 18 the struggle's orbit ended on grey.
            spec["orbit_deg"] = (-1 if move == "orbit_left" else 1) * (
                6.0 if slow else 9.0)
        elif move in ("crane_up", "crane_down"):
            lift = 0.9 if move == "crane_up" else -0.7
            spec["kind"] = "crane"
            spec["to_position_m"] = [position[0], position[1],
                                     round(max(0.4, position[2] + lift), 3)]
        camera["move"] = spec
    notes = [f"camera: {n}" for n in notes]
    return camera, {"size": size, "subject": subject, "angle": angle,
                    "move": move, "slow": slow, "distance_m": round(distance, 3),
                    "yaw_deg": round(yaw, 2)}, notes


def _anchor(size):
    if size in ("close_up", "extreme_close_up"):
        return "head"
    if size in ("medium", "medium_close"):
        return "chest"
    return "pelvis"
