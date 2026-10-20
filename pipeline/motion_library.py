"""The move library: clips from ``moves.py``, authored as body takes.

``e_motion`` used to write one take, a breathing idle. A shot in the main prompt
now says what each character *does* -- "walks to the throne and kneels" -- and
the analysis maps that onto a sequence of clips. This module turns a clip
sequence into a take in exactly the document shape the idle has always had
(``poses[frame][bone]`` as armature-space rotation vectors, ``root_m[frame]``),
so ``apply_motion``, the rest correction and the shot builder read it without
knowing it is anything new.

**Limbs are authored as directions.** A pose says "the upper arm points
forward and down, the forearm up toward the face", in the chest's own frame
(the legs in the pelvis's), and :func:`to_bones` turns that into rotations down
the chain. It reads like a director's note, it survives any bone roll, and it
is exactly what the video retargeter produces from a performance -- so authored
and recorded takes are one kind of data.

**The frame is prelude's.** Armature space, the character facing +Y, up +Z,
every ``left_*`` bone on +X (see ``skeleton.py`` for why that is the mirror of a
real body). Rotations compose down the chain as ``G_child = G_parent * M``,
which is how ``apply_motion`` keys them. Distances are metres on the 1.70 m
reference body; ``apply_motion``'s ``root_scale`` fits them to the character.

**A sequence is clips end to end.** Each clip starts where the last left the
body -- its position, its heading, its pose -- with a short blend across the
join, and the take ends with the last pose held and breathing, long enough to
outlast any shot.
"""

from __future__ import annotations

import math

REFERENCE_HEIGHT_M = {"human": 1.70, "quadruped": 1.60}
BLEND_S = 0.3
LEAD_S = 0.25
TAIL_S = 12.0
DEG = math.pi / 180.0

HEAD_BONES = ("neck", "head")


# -- quaternions (w, x, y, z), pure Python ----------------------------------

def q_axis(axis, degrees):
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    half = degrees * DEG / 2.0
    s = math.sin(half) / n
    return (math.cos(half), x * s, y * s, z * s)


def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def q_inv(q):
    return (q[0], -q[1], -q[2], -q[3])


def q_rotate(q, v):
    p = q_mul(q_mul(q, (0.0,) + tuple(v)), q_inv(q))
    return (p[1], p[2], p[3])


def _norm(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / n for c in v)


def q_from_to(u, v):
    """The shortest rotation carrying direction ``u`` onto ``v``."""
    u, v = _norm(u), _norm(v)
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(u, v))))
    if dot > 1.0 - 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    if dot < -1.0 + 1e-9:
        axis = (1.0, 0.0, 0.0) if abs(u[0]) < 0.9 else (0.0, 1.0, 0.0)
        cross = _cross(u, axis)
        return q_axis(cross, 180.0)
    cross = _cross(u, v)
    q = (1.0 + dot,) + cross
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def q_euler(pitch=0.0, roll=0.0, yaw=0.0):
    """Degrees about X (pitch), Y (roll), Z (yaw), applied pitch-roll-yaw."""
    return q_mul(q_axis((0, 0, 1), yaw),
                 q_mul(q_axis((0, 1, 0), roll), q_axis((1, 0, 0), pitch)))


def q_slerp(a, b, t):
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b, dot = tuple(-c for c in b), -dot
    if dot > 0.9995:
        q = tuple(x + (y - x) * t for x, y in zip(a, b))
    else:
        theta = math.acos(dot)
        s = math.sin(theta)
        wa, wb = math.sin((1 - t) * theta) / s, math.sin(t * theta) / s
        q = tuple(wa * x + wb * y for x, y in zip(a, b))
    n = math.sqrt(sum(c * c for c in q)) or 1.0
    return tuple(c / n for c in q)


def q_rotvec(q):
    """A unit quaternion as an axis-angle vector in radians."""
    w, x, y, z = q
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    s = math.sqrt(x * x + y * y + z * z)
    if s < 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(s, w)
    return [round(x / s * angle, 6), round(y / s * angle, 6),
            round(z / s * angle, 6)]


def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def dlerp(a, b, t):
    """Blend two directions (normalised lerp, fine for poses)."""
    return _norm(lerp(a, b, t))


def mirror(direction):
    """A left-side direction as the matching right-side one."""
    return (-direction[0], direction[1], direction[2])


# -- human poses --------------------------------------------------------------

ARM_T = {"left": (1.0, 0.0, 0.0), "right": (-1.0, 0.0, 0.0)}
LEG_T = (0.0, 0.0, -1.0)
SPINE_SHARE = (("spine1", 0.3), ("spine2", 0.35), ("spine3", 0.35))

# The rest of every pose: arms hanging a little away from the body with the
# elbows soft, legs straight. Directions in the chest (arms) or pelvis (legs)
# frame, left side; the right side mirrors it unless a pose says otherwise.
NEUTRAL = {
    "root": (0.0, 0.0, 0.0), "heading": 0.0,
    "pelvis": (0.0, 0.0, 0.0), "spine": (0.0, 0.0, 0.0),
    "neck": (0.0, 0.0, 0.0), "head": (0.0, 0.0, 0.0),
    "arm_left": ((0.22, 0.03, -0.97), (0.12, 0.16, -0.98)),
    "arm_right": ((-0.22, 0.03, -0.97), (-0.12, 0.16, -0.98)),
    "leg_left": ((0.03, 0.0, -1.0), (0.0, 0.0, -1.0)),
    "leg_right": ((-0.03, 0.0, -1.0), (0.0, 0.0, -1.0)),
}


def pose(**changes):
    """NEUTRAL with some fields replaced. ``arm``/``leg`` set both sides."""
    out = dict(NEUTRAL)
    arm = changes.pop("arm", None)
    if arm is not None:
        out["arm_left"] = arm
        out["arm_right"] = (mirror(arm[0]), mirror(arm[1]))
    leg = changes.pop("leg", None)
    if leg is not None:
        out["leg_left"] = leg
        out["leg_right"] = (mirror(leg[0]), mirror(leg[1]))
    out.update(changes)
    return out


def blend_pose(a, b, t):
    """Two poses mixed: directions nlerped, angles and positions lerped."""
    out = {}
    for key in a:
        va, vb = a[key], b.get(key, a[key])
        if key.startswith(("arm_", "leg_")):
            out[key] = tuple(dlerp(x, y, t) for x, y in zip(va, vb))
        elif key == "heading":
            out[key] = va + (vb - va) * t
        else:
            out[key] = lerp(va, vb, t)
    return out


def to_bones(p, kind="human"):
    """A pose as ``{bone: quaternion}`` plus the root offset."""
    if kind == "quadruped":
        return _quad_bones(p)
    bones = {}
    pelvis = q_mul(q_axis((0, 0, 1), p["heading"]), q_euler(*p["pelvis"]))
    bones["pelvis"] = pelvis
    for name, share in SPINE_SHARE:
        bones[name] = q_euler(*(share * v for v in p["spine"]))
    bones["neck"] = q_euler(*p["neck"])
    bones["head"] = q_euler(*p["head"])
    for side in ("left", "right"):
        upper, fore = p[f"arm_{side}"]
        t = ARM_T[side]
        m_upper = q_from_to(t, upper)
        m_fore = q_from_to(t, q_rotate(q_inv(m_upper), fore))
        bones[f"{side}_collar"] = (1.0, 0.0, 0.0, 0.0)
        bones[f"{side}_shoulder"] = m_upper
        bones[f"{side}_elbow"] = m_fore
        thigh, shin = p[f"leg_{side}"]
        m_thigh = q_from_to(LEG_T, thigh)
        m_shin = q_from_to(LEG_T, q_rotate(q_inv(m_thigh), shin))
        bones[f"{side}_hip"] = m_thigh
        bones[f"{side}_knee"] = m_shin
        # The foot stays level with the pelvis rather than riding the shin,
        # so a bent knee does not point the toes at the sky.
        bones[f"{side}_ankle"] = q_slerp((1.0, 0.0, 0.0, 0.0),
                                         q_inv(q_mul(m_thigh, m_shin)), 0.8)
    return bones, tuple(p["root"])


# -- quadruped poses ----------------------------------------------------------

QUAD_NEUTRAL = {
    "root": (0.0, 0.0, 0.0), "heading": 0.0, "body": (0.0, 0.0, 0.0),
    "spine": (0.0, 0.0, 0.0), "neck": (0.0, 0.0, 0.0),
    "head": (0.0, 0.0, 0.0), "jaw": 0.0, "tail": (0.0, 0.0, 0.0),
    # (upper, lower) pitch in degrees about X for each leg: + swings forward.
    "front_left": (0.0, 0.0), "front_right": (0.0, 0.0),
    "hind_left": (0.0, 0.0), "hind_right": (0.0, 0.0),
}
QUAD_LEGS = {"front_left": ("left_shoulder", "left_knee"),
             "front_right": ("right_shoulder", "right_knee"),
             "hind_left": ("left_hip", "left_stifle"),
             "hind_right": ("right_hip", "right_stifle")}


def quad_pose(**changes):
    out = dict(QUAD_NEUTRAL)
    out.update(changes)
    return out


def _quad_bones(p):
    bones = {"root": q_mul(q_axis((0, 0, 1), p["heading"]),
                           q_euler(*p["body"]))}
    for name, share in (("spine1", 0.3), ("spine2", 0.35), ("chest", 0.35)):
        bones[name] = q_euler(*(share * v for v in p["spine"]))
    bones["neck"] = q_euler(*p["neck"])
    bones["head"] = q_euler(*p["head"])
    bones["jaw"] = q_euler(-p["jaw"], 0.0, 0.0)
    bones["tail"] = q_euler(*p["tail"])
    for leg, (upper, lower) in QUAD_LEGS.items():
        a, b = p[leg]
        bones[upper] = q_euler(a, 0.0, 0.0)
        bones[lower] = q_euler(b, 0.0, 0.0)
    return bones, tuple(p["root"])


def blend_any(a, b, t, kind):
    if kind != "quadruped":
        return blend_pose(a, b, t)
    out = {}
    for key in a:
        va, vb = a[key], b.get(key, a[key])
        if isinstance(va, tuple):
            out[key] = lerp(va, vb, t)
        else:
            out[key] = va + (vb - va) * t
    return out


# -- context: where the character is, where things are ------------------------

class Context:
    """What a clip needs to know about the shot, in the character's frame.

    ``position`` and ``facing_deg`` are the character's world staging;
    ``targets`` maps an id to a world position; ``scale`` is the character's
    height over the reference height, so world metres become take metres.
    """

    def __init__(self, kind="human", fps=24, position=(0.0, 0.0, 0.0),
                 facing_deg=180.0, targets=None, scale=1.0, slots=None):
        self.kind = "quadruped" if kind in ("quadruped", "creature",
                                            "animal") else "human"
        self.fps = int(fps)
        self.position = tuple(position)
        self.facing = float(facing_deg)
        self.targets = dict(targets or {})
        self.scale = float(scale) or 1.0
        self.slots = dict(slots or {})

    def local(self, world):
        """A world point in the character's rest frame, in take metres."""
        dx = world[0] - self.position[0]
        dy = world[1] - self.position[1]
        a = -self.facing * DEG
        x = dx * math.cos(a) - dy * math.sin(a)
        y = dx * math.sin(a) + dy * math.cos(a)
        return (x / self.scale, y / self.scale)

    def goal(self, clip, default_distance=2.5):
        """Where a clip is headed, in take metres, or None."""
        target = clip.get("target")
        if target == "camera":
            return self.local((self.position[0], self.position[1] - 3.0))
        if target in self.targets:
            return self.local(self.targets[target])
        if clip.get("to") and clip["to"] in self.slots:
            return self.local(self.slots[clip["to"]])
        direction = clip.get("direction")
        d = default_distance
        # Directions are the audience's: "left" is screen left (-X in world).
        world = {"left": (-d, 0.0), "right": (d, 0.0),
                 "toward_camera": (0.0, -d), "away_from_camera": (0.0, d)}
        if direction in world:
            wx, wy = world[direction]
            return self.local((self.position[0] + wx, self.position[1] + wy))
        if direction == "forward":
            return (0.0, d / self.scale)
        if direction == "back":
            return (0.0, -d / self.scale)
        return None


def heading_to(point, origin=(0.0, 0.0)):
    """Yaw (deg) that turns +Y toward ``point``, counter-clockwise about +Z."""
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(-dx, dy))


def _wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


# -- clips ----------------------------------------------------------------------
#
# A clip is a function (state, clip, ctx) -> (duration_s, sampler). ``state`` is
# the pose the previous clip ended on; ``sampler(t)`` returns the pose at t.
# Root and heading are absolute in the take, so each clip carries the body on
# from where the last one left it.

def _keyed(start, keys):
    """A sampler that eases through ``[(time_s, pose), ...]`` after ``start``."""
    frames = [(0.0, start)] + list(keys)

    def sample(t):
        for (t0, p0), (t1, p1) in zip(frames, frames[1:]):
            if t <= t1:
                return blend_pose(p0, p1, smooth((t - t0) / max(1e-6, t1 - t0)))
        return frames[-1][1]
    return frames[-1][0], sample


def _at(state, **changes):
    """``changes`` applied to a neutral body standing where ``state`` stands.

    A leg the change does not set explicitly is solved: its foot stays planted
    under its hip and the knee bends to reach it (:func:`plant`). Without that,
    any pose that lowers the pelvis -- a lunge, a buckle, a cower -- drives the
    feet through the floor: the div's strike put them 0.47 m under.
    """
    p = pose(**changes)
    p["root"] = changes.get("root", (state["root"][0], state["root"][1], 0.0))
    p["heading"] = changes.get("heading", state["heading"])
    explicit = {"leg_left", "leg_right"} if "leg" in changes else \
        {k for k in ("leg_left", "leg_right") if k in changes}
    # Feet stay where they stood when the clip began: a lunge moves the
    # pelvis over a planted back foot rather than dragging the foot along.
    # Planting under the new pelvis instead skated the fighters 6 cm a frame.
    anchor = tuple(state["root"][:2])
    for side in ("left", "right"):
        if f"leg_{side}" not in explicit:
            p[f"leg_{side}"] = plant(p, side, anchor)
    return p


# The reference body's leg: hip joints at (+-0.09, 0, 0.86), knees at 0.47,
# ankles at 0.08 (prelude.SMPLX_REST), so both bones are 0.39 m.
PELVIS_REST = (0.0, 0.0, 0.94)
HIP_REST = {"left": (0.09, 0.0, 0.86), "right": (-0.09, 0.0, 0.86)}
ANKLE_REST_Z = 0.08
THIGH_M = SHIN_M = 0.39


def plant(p, side, anchor=None):
    """Two-bone IK: thigh and shin directions that put the ankle on the floor.

    The ankle target is below the hip's rest position at ``anchor`` (where the
    feet were planted; the root's own position when not given) and the body's
    heading; the knee bends toward the body's forward direction.
    Directions come back in the pelvis frame, which is what :func:`to_bones`
    expects of a leg.
    """
    q_pelvis = q_mul(q_axis((0, 0, 1), p["heading"]), q_euler(*p["pelvis"]))
    root = p["root"]
    offset = tuple(h - c for h, c in zip(HIP_REST[side], PELVIS_REST))
    rotated = q_rotate(q_pelvis, offset)
    hip = tuple(root[i] + PELVIS_REST[i] + rotated[i] for i in range(3))
    turn = q_axis((0, 0, 1), p["heading"])
    foot_offset = q_rotate(turn, (HIP_REST[side][0], 0.0, 0.0))
    base = anchor if anchor is not None else root[:2]
    ankle = (base[0] + foot_offset[0], base[1] + foot_offset[1], ANKLE_REST_Z)
    reach = tuple(a - h for a, h in zip(ankle, hip))
    d = math.sqrt(sum(c * c for c in reach))
    d = max(1e-4, min(d, THIGH_M + SHIN_M - 1e-4))
    u = _norm(reach)
    forward = q_rotate(turn, (0.0, 1.0, 0.0))
    along = sum(a * b for a, b in zip(forward, u))
    perp = _norm(tuple(f - along * c for f, c in zip(forward, u)))
    a = (THIGH_M ** 2 - SHIN_M ** 2 + d * d) / (2.0 * d)
    h = math.sqrt(max(0.0, THIGH_M ** 2 - a * a))
    knee = tuple(hip[i] + a * u[i] + h * perp[i] for i in range(3))
    thigh = _norm(tuple(k - hh for k, hh in zip(knee, hip)))
    shin = _norm(tuple(an - k for an, k in zip(ankle, knee)))
    back = q_inv(q_pelvis)
    return (q_rotate(back, thigh), q_rotate(back, shin))


def _clip_hold(state, clip, ctx, seconds=0.6, **changes):
    target = _at(state, **changes)
    return _keyed(state, [(seconds, target)])


def _clip_turn(state, clip, ctx):
    goal = ctx.goal(clip)
    if goal is None:
        return _clip_hold(state, clip, ctx, 0.3)
    wanted = heading_to(goal, state["root"][:2])
    delta = _wrap(wanted - state["heading"])
    seconds = 0.35 + abs(delta) / 180.0
    end = _at(state, heading=state["heading"] + delta)

    def sample(t):
        u = smooth(t / seconds)
        p = blend_pose(state, end, u)
        # A shuffle of the feet while the body comes round.
        step = math.sin(u * math.pi * 2.0) * 8.0 * (1 - abs(2 * u - 1))
        p["leg_left"] = ((0.03, math.sin(step * DEG), -math.cos(step * DEG)),
                         p["leg_left"][1])
        p["leg_right"] = ((-0.03, -math.sin(step * DEG), -math.cos(step * DEG)),
                          p["leg_right"][1])
        return p
    return seconds, sample


def _gait(amplitude, knee, arm_swing, elbow, lean, bob, period):
    """A walk or run cycle as a pose function of phase in radians."""
    def at(phase, heading, root_xy, extra_root_z=0.0):
        p = pose()
        for side, offset in (("left", 0.0), ("right", math.pi)):
            s = 1.0 if side == "left" else -1.0
            h = amplitude * math.sin(phase + offset)
            k = knee[0] + knee[1] * max(0.0, math.cos(phase + offset)) ** 2
            thigh = (0.03 * s, math.sin(h * DEG), -math.cos(h * DEG))
            shin = (0.0, math.sin((h - k) * DEG), -math.cos((h - k) * DEG))
            p[f"leg_{side}"] = (thigh, shin)
            a = -arm_swing * math.sin(phase + offset)
            upper = (0.2 * s, math.sin(a * DEG), -math.cos(a * DEG))
            fore = (0.12 * s, math.sin((a + elbow) * DEG),
                    -math.cos((a + elbow) * DEG))
            p[f"arm_{side}"] = (_norm(upper), _norm(fore))
        p["heading"] = heading
        p["pelvis"] = (0.0, 3.0 * math.sin(2 * phase), 4.0 * math.sin(phase))
        p["spine"] = (-lean, 0.0, -3.0 * math.sin(phase))
        p["root"] = (root_xy[0], root_xy[1],
                     -bob + bob * math.cos(2 * phase) + extra_root_z)
        return p
    return at


# Thigh length on the 1.70 m reference is 0.39 m and the leg 0.86 m. The ground
# speed is what the stance foot sweeps: four times the leg times the sine of the
# swing amplitude per cycle, so a planted foot does not skate.
LEG_M = 0.86
WALK = {"amplitude": 22.0, "knee": (6.0, 52.0), "arm_swing": 14.0,
        "elbow": 14.0, "lean": 2.0, "bob": 0.018, "period": 1.1}
RUN = {"amplitude": 34.0, "knee": (20.0, 85.0), "arm_swing": 32.0,
       "elbow": 75.0, "lean": 10.0, "bob": 0.04, "period": 0.72}


def _clip_travel(state, clip, ctx, style):
    goal = ctx.goal(clip, default_distance=3.0)
    if goal is None:
        goal = (state["root"][0] - 2.0 * math.sin(state["heading"] * DEG),
                state["root"][1] + 2.0 * math.cos(state["heading"] * DEG))
    here = state["root"][:2]
    stop_short = 0.9 / ctx.scale if clip.get("target") in ctx.targets else 0.0
    dx, dy = goal[0] - here[0], goal[1] - here[1]
    distance = max(0.0, math.hypot(dx, dy) - stop_short)
    heading = heading_to(goal, here) if distance > 1e-3 else state["heading"]
    turn = _wrap(heading - state["heading"])
    turn_s = 0.3 + abs(turn) / 240.0 if abs(turn) > 15.0 else 0.0
    speed = 4.0 * LEG_M * math.sin(style["amplitude"] * DEG) / style["period"]
    cycles = max(1, int(round(distance / (speed * style["period"]))))
    walk_s = cycles * style["period"]
    speed = distance / walk_s if walk_s else 0.0
    settle_s = 0.35
    total = turn_s + walk_s + settle_s
    gait = _gait(**style)
    direction = (-math.sin(heading * DEG), math.cos(heading * DEG))
    start_heading = state["heading"]

    def sample(t):
        if t < turn_s:
            u = smooth(t / turn_s)
            p = blend_pose(state, _at(state, heading=start_heading + turn * u),
                           u)
            return p
        if t < turn_s + walk_s:
            w = t - turn_s
            travelled = speed * w
            phase = 2.0 * math.pi * w / style["period"]
            p = gait(phase, start_heading + turn,
                     (here[0] + direction[0] * travelled,
                      here[1] + direction[1] * travelled))
            # Ease into the gait over the first half cycle.
            ramp = smooth(w / (0.5 * style["period"]))
            return blend_pose(_at(state, heading=start_heading + turn,
                                  root=p["root"]), p, ramp)
        end = gait(0.0, start_heading + turn,
                   (here[0] + direction[0] * distance,
                    here[1] + direction[1] * distance))
        rest = _at({"root": end["root"], "heading": end["heading"]},
                   heading=end["heading"])
        return blend_pose(end, rest, smooth((t - turn_s - walk_s) / settle_s))
    return total, sample


def _toward(ctx, state, clip, elevation=0.0):
    """A direction in the chest frame pointing at the clip's target."""
    goal = ctx.goal(clip)
    if goal is None:
        return _norm((0.0, 1.0, math.tan(elevation * DEG)))
    world_yaw = heading_to(goal, state["root"][:2]) - state["heading"]
    a = world_yaw * DEG
    return _norm((-math.sin(a), math.cos(a), math.tan(elevation * DEG)))


def _clip_strike(state, clip, ctx):
    wind = _at(state, spine=(4.0, 0.0, 14.0), pelvis=(0.0, 0.0, 5.0),
               arm_right=((-0.30, -0.35, 0.89), (-0.20, -0.60, 0.77)),
               arm_left=((0.35, 0.35, -0.87), (0.10, 0.80, -0.60)))
    # The weight goes into the blow through the hips, over feet that stay
    # planted (leg IK); an earlier version stepped the front foot forward at
    # floor height, which slid it 6 cm a frame.
    lean = 0.10
    h = state["heading"] * DEG
    hit = _at(state, spine=(-12.0, 0.0, -12.0), pelvis=(0.0, 0.0, -4.0),
              root=(state["root"][0] - lean * math.sin(h),
                    state["root"][1] + lean * math.cos(h), -0.06),
              arm_right=((-0.15, 0.85, -0.50), (-0.05, 0.90, -0.43)),
              arm_left=((0.30, -0.10, -0.95), (0.20, -0.05, -0.98)))
    guard = _at(state, spine=(-4.0, 0.0, -4.0),
                arm_right=((-0.30, 0.35, -0.88), (-0.20, 0.80, -0.56)),
                arm_left=((0.30, 0.40, -0.87), (0.10, 0.85, -0.52)))
    return _keyed(state, [(0.35, wind), (0.65, hit), (1.1, guard)])


def _clip_block(state, clip, ctx):
    up = _at(state, spine=(6.0, 0.0, 0.0),
             root=(state["root"][0], state["root"][1] - 0.10, -0.04),
             arm_left=((0.35, 0.55, 0.10), (-0.60, 0.40, 0.70)),
             arm_right=((-0.35, 0.55, 0.10), (0.60, 0.40, 0.70)))
    return _keyed(state, [(0.3, up), (0.8, up)])


def _clip_fight(state, clip, ctx):
    duration_a, strike = _clip_strike(state, clip, ctx)
    after = strike(duration_a)
    duration_b, block = _clip_block(after, clip, ctx)
    total = duration_a + duration_b + 0.4

    def sample(t):
        if t < duration_a:
            return strike(t)
        if t < duration_a + duration_b:
            return block(t - duration_a)
        return blend_pose(block(duration_b), state,
                          smooth((t - duration_a - duration_b) / 0.4))
    return total, sample


def _clip_shoot_bow(state, clip, ctx):
    aim = _toward(ctx, state, clip, elevation=4.0)
    turn = _at(state, spine=(0.0, 0.0, -25.0), head=(0.0, 0.0, 25.0),
               arm_left=(aim, aim),
               arm_right=((-0.55, 0.60, 0.10), (0.30, 0.90, 0.20)))
    draw = _at(state, spine=(0.0, 0.0, -25.0), head=(0.0, 0.0, 25.0),
               arm_left=(aim, aim),
               arm_right=((-0.85, -0.10, 0.12), (0.95, 0.25, 0.05)))
    loose = _at(state, spine=(0.0, 0.0, -25.0), head=(0.0, 0.0, 25.0),
                arm_left=(aim, aim),
                arm_right=((-0.90, -0.30, 0.05), (-0.60, -0.70, 0.10)))
    return _keyed(state, [(0.6, turn), (1.4, draw), (1.8, draw), (1.95, loose),
                          (2.6, loose)])


def _clip_throw(state, clip, ctx):
    aim = _toward(ctx, state, clip, elevation=20.0)
    wind = _at(state, spine=(6.0, 0.0, 18.0),
               arm_right=((-0.35, -0.55, 0.76), (-0.10, -0.40, 0.91)))
    release = _at(state, spine=(-14.0, 0.0, -14.0),
                  root=(state["root"][0], state["root"][1] + 0.2, -0.04),
                  arm_right=(aim, aim),
                  leg_left=((0.03, 0.40, -0.92), (0.0, 0.05, -1.0)))
    return _keyed(state, [(0.4, wind), (0.6, release), (1.1, release)])


def _clip_kneel(state, clip, ctx):
    x, y = state["root"][:2]
    down = _at(state, root=(x, y + 0.05, -0.42), spine=(-8.0, 0.0, 0.0),
               leg_left=((0.03, 0.98, -0.20), (0.0, 0.05, -1.0)),
               leg_right=((-0.03, 0.12, -0.99), (0.0, -0.98, -0.20)),
               arm_left=((0.20, 0.45, -0.87), (0.05, 0.90, -0.40)))
    return _keyed(state, [(1.1, down)])


def _clip_sit(state, clip, ctx):
    x, y = state["root"][:2]
    seated = _at(state, root=(x, y - 0.1, -0.80), spine=(-12.0, 0.0, 0.0),
                 leg=((0.05, 0.85, 0.52), (0.0, 0.45, -0.89)),
                 arm=((0.20, 0.55, -0.80), (0.0, 0.95, -0.30)))
    return _keyed(state, [(1.4, seated)])


def _clip_rise(state, clip, ctx):
    return _keyed(state, [(1.2, _at(state))])


def _clip_bow(state, clip, ctx):
    low = _at(state, spine=(-35.0, 0.0, 0.0), head=(-15.0, 0.0, 0.0),
              arm_right=((-0.20, 0.30, -0.93), (0.75, 0.60, 0.25)))
    up = _at(state, spine=(-5.0, 0.0, 0.0),
             arm_right=((-0.20, 0.30, -0.93), (0.75, 0.60, 0.25)))
    return _keyed(state, [(0.6, low), (1.1, low), (1.7, up)])


def _clip_point(state, clip, ctx):
    aim = _toward(ctx, state, clip, elevation=8.0)
    goal = ctx.goal(clip)
    look = 0.0
    if goal is not None:
        look = max(-60.0, min(60.0, _wrap(heading_to(goal, state["root"][:2])
                                          - state["heading"])))
    pointing = _at(state, arm_right=(aim, aim), head=(0.0, 0.0, look))
    return _keyed(state, [(0.7, pointing), (1.4, pointing)])


def _clip_raise_arm(state, clip, ctx):
    raised = _at(state, spine=(5.0, 0.0, 0.0), head=(10.0, 0.0, 0.0),
                 arm_right=((-0.25, 0.25, 0.93), (-0.10, 0.15, 0.98)))
    return _keyed(state, [(0.9, raised), (1.6, raised)])


def _clip_beckon(state, clip, ctx):
    keys = []
    for index in range(3):
        base = 0.5 + index * 0.45
        keys.append((base, _at(state, arm_right=((-0.25, 0.75, -0.6),
                                                  (-0.10, 0.60, 0.79)))))
        keys.append((base + 0.22, _at(state, arm_right=((-0.25, 0.75, -0.6),
                                                         (-0.10, 0.98, 0.10)))))
    return _keyed(state, keys)


# The pelvis joint sits inside the body, so a figure lying on its back has it
# about a body's half-thickness off the ground: 0.22 m on the reference body.
# -0.82 put it at 0.12 and Siamak's back 28 cm into the earth.
LYING_ROOT_Z = -0.72
LYING = {"pelvis": (90.0, 0.0, 0.0), "spine": (0.0, 0.0, 0.0),
         "arm": ((0.30, 0.0, -0.95), (0.25, 0.10, -0.96)),
         "leg": ((0.05, 0.0, -1.0), (0.0, 0.0, -1.0))}


def _clip_fall(state, clip, ctx):
    x, y = state["root"][:2]
    h = state["heading"] * DEG
    back = (x + 0.85 * math.sin(h), y - 0.85 * math.cos(h))
    buckle = _at(state, root=(x, y, -0.22), spine=(-20.0, 0.0, 0.0),
                 arm=((0.60, 0.30, -0.74), (0.50, 0.50, -0.70)))
    # Between the buckle and the ground the body sits back onto its seat, legs
    # forward along the ground; going straight from planted feet to lying
    # swung the feet through the earth on the way down.
    half = ((x + back[0]) / 2.0, (y + back[1]) / 2.0)
    seat = _at(state, root=(half[0], half[1], -0.55), pelvis=(45.0, 0.0, 0.0),
               spine=(-10.0, 0.0, 0.0),
               arm=((0.55, -0.20, -0.81), (0.45, -0.10, -0.89)),
               leg=((0.03, 0.70, -0.70), (0.0, 0.75, -0.66)))
    down = _at(state, root=(back[0], back[1], LYING_ROOT_Z), **LYING)
    return _keyed(state, [(0.4, buckle), (0.85, seat), (1.4, down)])


def _clip_lie(state, clip, ctx):
    x, y = state["root"][:2]
    h = state["heading"] * DEG
    back = (x + 0.85 * math.sin(h), y - 0.85 * math.cos(h))
    return _keyed(state, [(0.9, _at(state, root=(back[0], back[1],
                                                 LYING_ROOT_Z), **LYING))])


def _clip_mourn(state, clip, ctx):
    grief = _at(state, spine=(-18.0, 0.0, 0.0), head=(-25.0, 0.0, 0.0),
                neck=(-8.0, 0.0, 0.0),
                arm_left=((0.25, 0.60, -0.76), (-0.45, 0.45, 0.77)),
                arm_right=((-0.25, 0.60, -0.76), (0.45, 0.45, 0.77)))
    return _keyed(state, [(1.2, grief)])


def _clip_cower(state, clip, ctx):
    x, y = state["root"][:2]
    shrink = _at(state, root=(x, y - 0.15, -0.12), spine=(12.0, 0.0, 0.0),
                 head=(-10.0, 0.0, 0.0),
                 arm_left=((0.30, 0.50, 0.10), (-0.50, 0.20, 0.85)),
                 arm_right=((-0.30, 0.50, 0.10), (0.50, 0.20, 0.85)))
    return _keyed(state, [(0.5, shrink)])


def _clip_celebrate(state, clip, ctx):
    joy = _at(state, spine=(8.0, 0.0, 0.0), head=(15.0, 0.0, 0.0),
              arm=((0.45, 0.10, 0.89), (0.35, 0.05, 0.94)))
    return _keyed(state, [(0.7, joy), (1.4, joy)])


def _clip_embrace(state, clip, ctx):
    wrap = _at(state, spine=(-6.0, 0.0, 0.0),
               arm_left=((0.35, 0.93, 0.0), (-0.75, 0.60, 0.10)),
               arm_right=((-0.35, 0.93, 0.0), (0.75, 0.60, 0.10)))
    return _keyed(state, [(0.9, wrap), (2.0, wrap)])


def _clip_look(state, clip, ctx):
    goal = ctx.goal(clip)
    yaw = 0.0
    if goal is not None:
        yaw = max(-70.0, min(70.0, _wrap(heading_to(goal, state["root"][:2])
                                         - state["heading"])))
    looking = _at(state, neck=(0.0, 0.0, yaw * 0.35), head=(0.0, 0.0, yaw * 0.65))
    return _keyed(state, [(0.6, looking)])


def _clip_nod(state, clip, ctx):
    down = _at(state, head=(-14.0, 0.0, 0.0))
    return _keyed(state, [(0.25, down), (0.5, _at(state)), (0.75, down),
                          (1.05, _at(state))])


def _clip_shake_head(state, clip, ctx):
    keys, t = [], 0.0
    for yaw in (18.0, -18.0, 16.0, -14.0, 0.0):
        t += 0.25
        keys.append((t, _at(state, head=(0.0, 0.0, yaw))))
    return _keyed(state, keys)


def _clip_talk(state, clip, ctx):
    """A speaker's gestures: one hand, then the other, the head nodding."""
    gesture_r = _at(state, spine=(0.0, 0.0, 3.0), head=(-4.0, 0.0, 4.0),
                    arm_right=((-0.25, 0.25, -0.93), (-0.25, 0.75, -0.30)))
    gesture_l = _at(state, spine=(0.0, 0.0, -3.0), head=(3.0, 0.0, -4.0),
                    arm_left=((0.25, 0.25, -0.93), (0.25, 0.75, -0.30)))
    rest = _at(state)
    return _keyed(state, [(0.5, rest), (1.1, gesture_r), (1.8, gesture_r),
                          (2.4, rest), (3.0, gesture_l), (3.6, gesture_l),
                          (4.2, rest)])


def _clip_idle(state, clip, ctx):
    return _keyed(state, [(1.0, _at(state))])


HUMAN = {
    "idle": _clip_idle, "talk": _clip_talk,
    "walk": lambda s, c, x: _clip_travel(s, c, x, WALK),
    "run": lambda s, c, x: _clip_travel(s, c, x, RUN),
    "turn": _clip_turn, "look": _clip_look, "strike": _clip_strike,
    "block": _clip_block, "fight": _clip_fight, "shoot_bow": _clip_shoot_bow,
    "throw": _clip_throw, "kneel": _clip_kneel, "sit": _clip_sit,
    "rise": _clip_rise, "bow": _clip_bow, "point": _clip_point,
    "raise_arm": _clip_raise_arm, "beckon": _clip_beckon, "fall": _clip_fall,
    "lie": _clip_lie, "mourn": _clip_mourn, "cower": _clip_cower,
    "celebrate": _clip_celebrate, "embrace": _clip_embrace, "nod": _clip_nod,
    "shake_head": _clip_shake_head,
}


# -- quadruped clips ------------------------------------------------------------

def _qkeyed(start, keys):
    frames = [(0.0, start)] + list(keys)

    def sample(t):
        for (t0, p0), (t1, p1) in zip(frames, frames[1:]):
            if t <= t1:
                return blend_any(p0, p1, smooth((t - t0) / max(1e-6, t1 - t0)),
                                 "quadruped")
        return frames[-1][1]
    return frames[-1][0], sample


def _qat(state, **changes):
    p = quad_pose(**changes)
    p["root"] = changes.get("root", (state["root"][0], state["root"][1], 0.0))
    p["heading"] = changes.get("heading", state["heading"])
    return p


QUAD_LEG_M = 0.57
QUAD_WALK = {"amplitude": 24.0, "lower": 32.0, "period": 1.0,
             "phases": {"hind_left": 0.0, "front_left": 0.25,
                        "hind_right": 0.5, "front_right": 0.75},
             "bob": 0.012}
QUAD_RUN = {"amplitude": 32.0, "lower": 55.0, "period": 0.6,
            "phases": {"front_left": 0.0, "hind_right": 0.0,
                       "front_right": 0.5, "hind_left": 0.5},
            "bob": 0.035}


def _qgait(style, phase, heading, root_xy):
    p = quad_pose(heading=heading)
    for leg, offset in style["phases"].items():
        ph = phase + 2.0 * math.pi * offset
        a = style["amplitude"] * math.sin(ph)
        flex = style["lower"] * max(0.0, math.cos(ph)) ** 2
        p[leg] = (a, -flex if leg.startswith("front") else flex * 0.7)
    p["neck"] = (3.0 * math.sin(2 * phase), 0.0, 0.0)
    p["tail"] = (0.0, 0.0, 6.0 * math.sin(phase))
    p["root"] = (root_xy[0], root_xy[1],
                 -style["bob"] + style["bob"] * math.cos(2 * phase))
    return p


def _qclip_travel(state, clip, ctx, style):
    goal = ctx.goal(clip, default_distance=3.0)
    if goal is None:
        goal = (state["root"][0] - 2.0 * math.sin(state["heading"] * DEG),
                state["root"][1] + 2.0 * math.cos(state["heading"] * DEG))
    here = state["root"][:2]
    stop_short = 1.1 / ctx.scale if clip.get("target") in ctx.targets else 0.0
    distance = max(0.0, math.hypot(goal[0] - here[0], goal[1] - here[1])
                   - stop_short)
    heading = heading_to(goal, here) if distance > 1e-3 else state["heading"]
    turn = _wrap(heading - state["heading"])
    turn_s = 0.4 + abs(turn) / 200.0 if abs(turn) > 15.0 else 0.0
    speed = 4.0 * QUAD_LEG_M * math.sin(style["amplitude"] * DEG) / style["period"]
    cycles = max(1, int(round(distance / (speed * style["period"]))))
    walk_s = cycles * style["period"]
    speed = distance / walk_s
    direction = (-math.sin(heading * DEG), math.cos(heading * DEG))
    start = state["heading"]

    def sample(t):
        if t < turn_s:
            return blend_any(state, _qat(state, heading=start + turn
                                         * smooth(t / turn_s)),
                             smooth(t / turn_s), "quadruped")
        if t < turn_s + walk_s:
            w = t - turn_s
            return _qgait(style, 2 * math.pi * w / style["period"],
                          start + turn, (here[0] + direction[0] * speed * w,
                                         here[1] + direction[1] * speed * w))
        end = (here[0] + direction[0] * distance,
               here[1] + direction[1] * distance)
        return _qat(state, heading=start + turn, root=(end[0], end[1], 0.0))
    return turn_s + walk_s + 0.3, sample


def _qclip_turn(state, clip, ctx):
    goal = ctx.goal(clip)
    if goal is None:
        return _qkeyed(state, [(0.3, _qat(state))])
    delta = _wrap(heading_to(goal, state["root"][:2]) - state["heading"])
    return _qkeyed(state, [(0.5 + abs(delta) / 150.0,
                            _qat(state, heading=state["heading"] + delta))])


def _qclip_look(state, clip, ctx):
    goal = ctx.goal(clip)
    yaw = 0.0
    if goal is not None:
        yaw = max(-60.0, min(60.0, _wrap(heading_to(goal, state["root"][:2])
                                         - state["heading"])))
    return _qkeyed(state, [(0.6, _qat(state, neck=(0.0, 0.0, 0.5 * yaw),
                                      head=(0.0, 0.0, 0.5 * yaw)))])


def _qclip_lie(state, clip, ctx):
    x, y = state["root"][:2]
    down = _qat(state, root=(x, y, -0.27), front_left=(70.0, -150.0),
                front_right=(70.0, -150.0), hind_left=(60.0, -120.0),
                hind_right=(60.0, -120.0), neck=(-10.0, 0.0, 0.0))
    return _qkeyed(state, [(1.3, down)])


def _qclip_fall(state, clip, ctx):
    x, y = state["root"][:2]
    down = _qat(state, root=(x, y, -0.34), body=(0.0, 85.0, 0.0),
                front_left=(20.0, -20.0), hind_left=(-15.0, 10.0),
                neck=(-20.0, 0.0, 0.0))
    return _qkeyed(state, [(1.2, down)])


def _qclip_graze(state, clip, ctx):
    low = _qat(state, neck=(-55.0, 0.0, 0.0), head=(-20.0, 0.0, 0.0),
               front_left=(-6.0, 0.0), front_right=(-6.0, 0.0))
    return _qkeyed(state, [(1.0, low)])


def _qclip_rear(state, clip, ctx):
    up = _qat(state, body=(40.0, 0.0, 0.0), hind_left=(-30.0, 10.0),
              hind_right=(-30.0, 10.0), front_left=(50.0, -80.0),
              front_right=(45.0, -70.0), neck=(10.0, 0.0, 0.0))
    return _qkeyed(state, [(0.9, up), (1.5, up), (2.3, _qat(state))])


def _qclip_roar(state, clip, ctx):
    roar = _qat(state, neck=(15.0, 0.0, 0.0), head=(20.0, 0.0, 0.0), jaw=28.0)
    return _qkeyed(state, [(0.5, roar), (1.4, roar), (1.9, _qat(state))])


def _qclip_idle(state, clip, ctx):
    return _qkeyed(state, [(1.0, _qat(state))])


QUADRUPED = {
    "idle": _qclip_idle,
    "walk": lambda s, c, x: _qclip_travel(s, c, x, QUAD_WALK),
    "run": lambda s, c, x: _qclip_travel(s, c, x, QUAD_RUN),
    "turn": _qclip_turn, "look": _qclip_look, "lie": _qclip_lie,
    "fall": _qclip_fall, "graze": _qclip_graze, "rear": _qclip_rear,
    "roar": _qclip_roar,
}


# -- the take -----------------------------------------------------------------

def _breathe(p, t, kind, strength=1.0):
    """The settled hold: the last pose, alive. Chest and head only."""
    q = dict(p)
    breath = math.sin(2 * math.pi * t / 4.0) + 0.22 * math.sin(4 * math.pi * t / 4.0)
    drift = math.sin(2 * math.pi * t / 3.0 + 1.3)
    if kind == "quadruped":
        q["spine"] = tuple(a + b for a, b in zip(p["spine"],
                                                 (0.8 * breath * strength, 0, 0)))
        q["tail"] = tuple(a + b for a, b in zip(p["tail"],
                                                (0, 0, 4.0 * drift * strength)))
        return q
    q["spine"] = tuple(a + b for a, b in zip(p["spine"],
                                             (1.2 * breath * strength,
                                              0.0, 0.8 * drift * strength)))
    q["head"] = tuple(a + b for a, b in zip(p["head"],
                                            (0.8 * drift * strength, 0.0,
                                             1.0 * breath * strength)))
    return q


def build(clips, ctx):
    """A take from a clip sequence: ``(body_doc, head_doc, report)``.

    The take opens on a short still lead, plays the clips end to end with a
    blend at each join, and closes with the final pose held and breathing for
    :data:`TAIL_S`, so ``hold_last_pose`` never has to freeze a live figure.
    """
    table = QUADRUPED if ctx.kind == "quadruped" else HUMAN
    neutral = (dict(QUAD_NEUTRAL) if ctx.kind == "quadruped"
               else dict(NEUTRAL))
    neutral["root"] = (0.0, 0.0, 0.0)
    neutral["heading"] = 0.0
    blend = blend_any

    segments, state, t0, played = [], neutral, LEAD_S, []
    for clip in clips:
        author = table.get(clip.get("clip"))
        if author is None:
            continue
        duration, sampler = author(state, clip, ctx)
        segments.append((t0, duration, sampler, state))
        played.append({"clip": clip.get("clip"), "start_s": round(t0, 3),
                       "duration_s": round(duration, 3)})
        state = sampler(duration)
        t0 += duration
    final = state
    total = t0 + TAIL_S
    frames = max(2, int(round(total * ctx.fps)))

    def at(t):
        if t < LEAD_S or not segments:
            base = neutral if not segments else segments[0][3]
            return _breathe(base, t, ctx.kind, 0.5)
        for start, duration, sampler, before in segments:
            if t < start + duration:
                local = t - start
                p = sampler(local)
                if local < BLEND_S and before is not neutral:
                    # Joins are already continuous (each clip starts from the
                    # last pose); this only softens velocity at the join.
                    p = blend(before, p, smooth(local / BLEND_S), ctx.kind)
                return p
        return _breathe(final, t - t0, ctx.kind,
                        min(1.0, (t - t0) / 1.0))

    body_poses, head_poses, root = [], [], []
    for index in range(frames):
        p = at(index / float(ctx.fps))
        bones, offset = to_bones(p, ctx.kind)
        head = {n: q_rotvec(bones.pop(n)) for n in HEAD_BONES if n in bones}
        body_poses.append({n: q_rotvec(q) for n, q in bones.items()})
        head_poses.append(head)
        root.append([round(v, 5) for v in offset])

    report = {"clips": played, "frames": frames, "seconds": round(total, 3),
              "action_s": round(t0, 3),
              "final_root_m": [round(v, 4) for v in final["root"]],
              "final_heading_deg": round(final["heading"], 2)}
    return body_poses, head_poses, root, report
