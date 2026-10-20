"""The vocabulary of moves: what a character can be asked to do in a shot.

The main prompt describes moves in plain English ("raises his staff slowly in
blessing", "walks to the throne and sits"). The pipeline cannot animate
English, so the in-loop model maps each description onto a short sequence of
*clips* from this table, and ``e_motion`` animates the clips. When the user
records a video of the move instead, the video wins and the clips are only a
label in the panel.

The table is the single source of truth for three readers: the prompt that
asks the model to classify a move, the validator that checks what came back,
and ``e_motion``'s authoring code, which must implement every name here.

Parameters a clip may take:

* ``target`` -- a character id in the shot, or ``camera``: whom to walk to,
  turn to, point at, strike, look at.
* ``direction`` -- one of :data:`DIRECTIONS`, for a move with no one to aim at.
* ``to`` -- a stage slot (``left``, ``center back``...) to walk or run to.
"""

from __future__ import annotations

HUMAN_CLIPS = {
    "idle": "stand at rest, breathing",
    "talk": "speak with natural hand gestures and small head movements",
    "walk": "walk to a place, toward someone, or in a direction",
    "run": "run to a place, toward someone, or in a direction",
    "turn": "turn the whole body to face someone or a direction",
    "look": "turn only the head to look at someone",
    "strike": "swing a weapon or a fist at someone",
    "block": "raise arms or a shield to defend",
    "fight": "fight: alternating strikes and blocks",
    "shoot_bow": "draw a bow and loose an arrow",
    "throw": "throw something overarm",
    "kneel": "go down on one knee",
    "sit": "sit down on the ground",
    "rise": "stand up from kneeling, sitting or lying",
    "bow": "bow the head and upper body in respect",
    "point": "point an arm at someone or somewhere",
    "raise_arm": "raise one arm high: bless, salute, command, call out",
    "beckon": "wave someone closer",
    "fall": "collapse to the ground: wounded, dying, fainting",
    "lie": "lie still on the ground",
    "mourn": "grieve: head bowed, hands to the face",
    "cower": "recoil in fear, arms raised, leaning away",
    "celebrate": "raise both arms in joy or triumph",
    "embrace": "open the arms and embrace someone",
    "nod": "nod the head yes",
    "shake_head": "shake the head no",
}

QUADRUPED_CLIPS = {
    "idle": "stand at rest, breathing",
    "walk": "walk to a place, toward someone, or in a direction",
    "run": "run to a place, toward someone, or in a direction",
    "turn": "turn the whole body to face someone or a direction",
    "look": "turn only the head to look at someone",
    "lie": "lie down and rest",
    "fall": "collapse to the ground",
    "graze": "lower the head to the ground to graze or sniff",
    "rear": "rear up on the hind legs",
    "roar": "raise the head and roar or call out",
}

DIRECTIONS = ("left", "right", "forward", "back", "toward_camera",
              "away_from_camera")

# Clips that travel: e_motion needs a destination for these.
TRAVELLING = ("walk", "run")
# Clips that repeat for as long as the shot lasts, rather than play once.
CYCLIC = ("idle", "talk", "fight")

QUADRUPED_KINDS = ("quadruped", "creature", "animal")


def clips_for(kind):
    """The clip table that applies to a character of this kind."""
    return QUADRUPED_CLIPS if kind in QUADRUPED_KINDS else HUMAN_CLIPS


def menu(kind):
    """The clip table as prompt text, one ``name: meaning`` per line."""
    return "\n".join(f"- {name}: {meaning}"
                     for name, meaning in clips_for(kind).items())


def check(clips, kind, targets):
    """Problems with a proposed clip sequence, as plain-English strings.

    ``targets`` is the set of ids a clip may aim at (the shot's cast plus
    ``camera``). An empty list means the sequence is usable.
    """
    problems = []
    table = clips_for(kind)
    if not clips:
        problems.append("no clips")
    for index, clip in enumerate(clips or []):
        name = clip.get("clip")
        if name not in table:
            problems.append(f"clip {index + 1}: {name!r} is not one of "
                            f"{', '.join(table)}")
        target = clip.get("target")
        if target and target not in targets:
            problems.append(f"clip {index + 1}: target {target!r} is not in "
                            f"this shot ({', '.join(sorted(targets))})")
        direction = clip.get("direction")
        if direction and direction not in DIRECTIONS:
            problems.append(f"clip {index + 1}: direction {direction!r} is not "
                            f"one of {', '.join(DIRECTIONS)}")
    return problems


# Words that settle a move without asking anyone, tried in order when the
# model is unavailable. Deliberately small: this is the floor, not the method.
KEYWORDS = (
    (("run", "runs", "running", "rushes", "charges", "gallop"), "run"),
    (("walk", "walks", "steps", "strides", "approaches", "comes to",
      "goes to", "paces"), "walk"),
    (("kneel", "kneels"), "kneel"),
    (("sits", "sit down", "seated"), "sit"),
    (("rises", "stands up", "gets up"), "rise"),
    (("falls", "collapses", "dies", "faints", "slumps"), "fall"),
    (("lies", "lying"), "lie"),
    (("bows", "bow "), "bow"),
    (("points", "point "), "point"),
    (("raises his", "raises her", "lifts his", "lifts her", "raises an arm",
      "blesses", "salutes"), "raise_arm"),
    (("weeps", "mourns", "grieves", "laments", "sobs"), "mourn"),
    (("recoils", "cowers", "flinches", "shrinks"), "cower"),
    (("fights", "battles", "duels", "wrestles"), "fight"),
    (("strikes", "swings", "slashes", "hits", "punches", "attacks"), "strike"),
    (("blocks", "parries", "shields"), "block"),
    (("shoots", "looses an arrow", "draws his bow", "draws her bow"),
     "shoot_bow"),
    (("throws", "hurls"), "throw"),
    (("embraces", "hugs"), "embrace"),
    (("celebrates", "rejoices", "cheers"), "celebrate"),
    (("nods",), "nod"),
    (("shakes his head", "shakes her head"), "shake_head"),
    (("beckons", "waves"), "beckon"),
    (("turns",), "turn"),
    (("looks", "glances", "gazes", "watches"), "look"),
    (("speaks", "says", "tells", "talks", "calls"), "talk"),
    (("grazes", "sniffs"), "graze"),
    (("rears",), "rear"),
    (("roars", "growls", "bellows"), "roar"),
)


def guess(text, kind):
    """A keyword reading of a move, for when the model cannot be asked."""
    lowered = f" {text.lower()} "
    table = clips_for(kind)
    found = []
    for words, clip in KEYWORDS:
        if clip in table and any(word in lowered for word in words):
            position = min(lowered.find(word) for word in words
                           if word in lowered)
            found.append((position, clip))
    found.sort()
    seen, clips = set(), []
    for _position, clip in found:
        if clip not in seen:
            seen.add(clip)
            clips.append({"clip": clip})
    return clips or [{"clip": "idle"}]


RIG = {"human": "human", "humanoid": "human", "quadruped": "quadruped",
       "creature": "quadruped", "animal": "quadruped"}
# Moves that every character of a topology can share one looping take for.
SHARED = {("idle",): "idle", ("talk",): "talk", ("fight",): "fight"}


def resolve_take(member, kind, has_video=False):
    """Which take a cast member plays, and whether it loops.

    Returns ``(take_id, shared)``. A move that is only idling, talking or
    fighting shares one looping take per rig topology (``idle_human``,
    ``talk_human``) -- the same cache entry for every character, phased apart by
    ``h_assemble``. Anything else, and anything the user recorded a video for,
    is the cast member's own take, played once and then held.

    ``e_motion`` plans from this and ``h_assemble`` reads the take it names, so
    the two cannot disagree about where a character's motion lives.
    """
    rig = RIG.get(kind, "human")
    clips = [c.get("clip") for c in (member.get("action") or {}).get("clips")
             or []]
    own = member.get("take")
    if has_video and own:
        return own, False
    if not clips:
        return (own, False) if own else (f"idle_{rig}", True)
    shared = SHARED.get(tuple(clips))
    if shared and (shared == "idle" or rig == "human"):
        return f"{shared}_{rig}", True
    return (own or f"idle_{rig}"), own is None
