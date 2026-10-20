"""Validation: JSON Schema first, then the cross-references a schema cannot see.

Two design rules here.

**Report everything at once.** A strong chat model that gets back "line 41: bad
enum" fixes one field and returns a file with four remaining errors. Handing it
the complete list turns three round trips into one.

**Point at the data, not at the validator.** Every problem is reported as a JSON
pointer into the document (``/shots/0/cast/1/character``) plus what was wrong,
so a human editing ``production.json`` by hand knows exactly where to look.
"""

from __future__ import annotations

import os

from . import jsonio

SCHEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema")

# kind -> rig mode, the last resort in the §6.2 resolution order.
KIND_RIG = {
    "human": "smplx",
    "humanoid": "smplx",
    "creature": "rigged",
    "quadruped": "rigged",
}


class Problem:
    """One validation failure, locatable in the document."""

    def __init__(self, pointer, message, severity="error"):
        self.pointer = pointer or "/"
        self.message = message
        self.severity = severity

    def __str__(self):
        mark = "ERROR" if self.severity == "error" else "warn "
        return f"{mark} {self.pointer}: {self.message}"

    def as_dict(self):
        return {"pointer": self.pointer, "message": self.message,
                "severity": self.severity}


class Report:
    """The outcome of validating one document."""

    def __init__(self, problems=None):
        self.problems = list(problems or [])

    def __bool__(self):
        return not self.errors

    @property
    def errors(self):
        return [p for p in self.problems if p.severity == "error"]

    @property
    def warnings(self):
        return [p for p in self.problems if p.severity == "warn"]

    def add(self, pointer, message, severity="error"):
        self.problems.append(Problem(pointer, message, severity))
        return self

    def extend(self, other):
        self.problems.extend(other.problems)
        return self

    def text(self):
        if not self.problems:
            return "valid"
        return "\n".join(str(p) for p in self.problems)

    def as_dict(self):
        return {
            "ok": bool(self),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "problems": [p.as_dict() for p in self.problems],
        }


def load_schema(name):
    return jsonio.read(os.path.join(SCHEMA_DIR, name + ".schema.json"))


def check_schema(document, schema_name):
    """Validate against a schema, collecting every violation."""
    report = Report()
    try:
        import jsonschema
    except ImportError:
        return report.add("/", "jsonschema is not installed in this interpreter; "
                               "schema validation was skipped", "warn")
    validator = jsonschema.Draft202012Validator(load_schema(schema_name))
    for error in sorted(validator.iter_errors(document), key=lambda e: list(e.path)):
        pointer = "/" + "/".join(str(p) for p in error.path)
        report.add(pointer, _phrase(error))
    return report


def _phrase(error):
    """Turn a jsonschema error into something an author can act on.

    The stock messages quote the whole failing subtree, which for a shot means
    two hundred characters of JSON before the actual complaint.
    """
    kind = error.validator
    if kind == "required":
        return error.message
    if kind == "additionalProperties":
        return error.message + " (fields are closed; check the spelling)"
    if kind == "enum":
        allowed = ", ".join(repr(v) for v in error.validator_value)
        return f"{error.instance!r} is not one of: {allowed}"
    if kind in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        return f"{error.instance!r} is out of range ({kind} {error.validator_value})"
    if kind == "pattern":
        return (f"{error.instance!r} does not match {error.validator_value} "
                "(ids are lower-case letters, digits and underscores)")
    if kind == "type":
        return f"expected {error.validator_value}, got {type(error.instance).__name__}"
    if kind in ("oneOf", "anyOf"):
        return error.message.split("\n")[0]
    return error.message


ANCHORS = ("head", "eyes", "chest", "pelvis", "root", "hands")

# The mouth channels the rig implements as shape keys, in the prelude's order.
# Host-side stages need the list too - g_align writes them, h_assemble keys them
# - and the prelude is not importable outside Blender, so the canonical copy
# lives here and ``tests/blender_character_check.py`` asserts the two agree.
VISEME_CHANNELS = ("jaw_open", "lip_wide", "lip_round", "lip_press",
                   "brow_raise", "brow_furrow", "squint", "smile")


# The one speaker who is not a character. See pipeline/main_prompt.py.
NARRATOR = "narrator"


def check_production(document):
    """Schema, then every cross-reference a schema cannot express (§7-A)."""
    report = check_schema(document, "production")
    if report.errors:
        # Structure is broken; pointers from here would be noise.
        return report

    characters = _index(document.get("characters", []), "id")
    envs = _index(document.get("envs", []), "id")
    props = _index(document.get("props", []), "id")

    _check_unique(report, document.get("characters", []), "id", "/characters")
    _check_unique(report, document.get("envs", []), "id", "/envs")
    _check_unique(report, document.get("props", []), "id", "/props")
    _check_unique(report, document.get("shots", []), "id", "/shots")

    for index, prop in enumerate(document.get("props", [])):
        holder = prop.get("held_by")
        if holder and holder not in characters:
            report.add(f"/props/{index}/held_by",
                       f"no character {holder!r}; cast is {sorted(characters)}")

    seen_lines = {}
    for si, shot in enumerate(document.get("shots", [])):
        base = f"/shots/{si}"
        _check_shot(report, base, shot, characters, envs, props, seen_lines)

    for char_id, character in characters.items():
        height = character.get("height_m", 0)
        if character.get("kind") in ("human", "humanoid") and not 1.2 <= height <= 3.0:
            report.add(f"/characters/{_pos(document['characters'], char_id)}/height_m",
                       f"{height} m is outside the plausible range for a "
                       f"{character['kind']}; SMPL-X scaling will look wrong",
                       "warn")
    return report


def _check_shot(report, base, shot, characters, envs, props, seen_lines):
    if shot.get("env") not in envs:
        report.add(base + "/env",
                   f"no env {shot.get('env')!r}; defined envs are {sorted(envs)}")

    cast_ids = []
    for ci, member in enumerate(shot.get("cast", [])):
        who = member.get("character")
        cast_ids.append(who)
        if who not in characters:
            report.add(f"{base}/cast/{ci}/character",
                       f"no character {who!r}; defined characters are "
                       f"{sorted(characters)}")
        for pi, prop_id in enumerate(member.get("props", [])):
            if prop_id not in props:
                report.add(f"{base}/cast/{ci}/props/{pi}",
                           f"no prop {prop_id!r}")
    duplicates = {c for c in cast_ids if cast_ids.count(c) > 1}
    if duplicates:
        report.add(base + "/cast",
                   f"{sorted(duplicates)} appear twice in one shot")

    dialogue = shot.get("dialogue", [])
    for di, line in enumerate(dialogue):
        pointer = f"{base}/dialogue/{di}"
        line_id = line.get("line_id")
        if line_id in seen_lines:
            report.add(pointer + "/line_id",
                       f"{line_id!r} is already used by shot {seen_lines[line_id]!r}; "
                       "line ids are the audio directory names and must be unique "
                       "across the film")
        else:
            seen_lines[line_id] = shot.get("id")
        if line.get("character") == NARRATOR:
            # Narration is spoken over the shot by no one in it: it is timed
            # like dialogue and never drives a mouth.
            continue
        if line.get("character") not in cast_ids:
            report.add(pointer + "/character",
                       f"{line.get('character')!r} speaks but is not in this "
                       f"shot's cast {cast_ids}")

    if not dialogue and not shot.get("silent_duration_s"):
        report.add(base,
                   "a shot with no dialogue needs silent_duration_s: with no "
                   "audio there is nothing to derive its length from (§5.2)")

    _check_aim(report, base, shot, cast_ids, envs, props)
    _check_ranges(report, base, shot)


def _check_aim(report, base, shot, cast_ids, envs, props):
    aim = (shot.get("camera") or {}).get("aim")
    if not isinstance(aim, str):
        return
    target, _, anchor = aim.partition(".")
    if target == "env":
        if anchor not in props and anchor not in envs:
            report.add(base + "/camera/aim",
                       f"env.{anchor} names neither a prop nor an env")
        return
    if target not in cast_ids:
        report.add(base + "/camera/aim",
                   f"the camera aims at {target!r}, who is not in this shot's "
                   f"cast {cast_ids}")
    elif anchor not in ANCHORS:
        report.add(base + "/camera/aim",
                   f"{anchor!r} is not an anchor; use one of {list(ANCHORS)}")


def _check_ranges(report, base, shot):
    """Beat and camera-move windows must be ordered and must not overlap.

    They are in normalised shot time (0..1) rather than frames, because the
    frame count is derived from the audio and does not exist when this file is
    authored. The §7-A "frame ranges non-overlapping" gate is enforced here in
    that normalised space instead.
    """
    spans = []
    for bi, beat in enumerate(shot.get("beats", [])):
        pointer = f"{base}/beats/{bi}"
        start, end = beat.get("from_pct", 0.0), beat.get("to_pct", 0.0)
        if end <= start:
            report.add(pointer, f"to_pct ({end}) must be greater than "
                                f"from_pct ({start})")
        else:
            spans.append((start, end, bi))
    spans.sort()
    for (a_start, a_end, a_i), (b_start, b_end, b_i) in zip(spans, spans[1:]):
        if b_start < a_end - 1e-9:
            report.add(f"{base}/beats/{b_i}",
                       f"overlaps beat {a_i} ({a_start}-{a_end})", "warn")

    move = (shot.get("camera") or {}).get("move") or {}
    if move:
        start = move.get("from_pct", 0.0)
        end = move.get("to_pct", 1.0)
        if end <= start:
            report.add(base + "/camera/move",
                       f"to_pct ({end}) must be greater than from_pct ({start})")
        if move.get("kind") in ("push_in", "pull_out", "crane") \
                and not move.get("to_position_m"):
            report.add(base + "/camera/move",
                       f"a {move['kind']} move needs to_position_m")
        if move.get("kind") == "orbit" and move.get("orbit_deg") in (None, 0):
            report.add(base + "/camera/move", "an orbit move needs orbit_deg")


def check_project_inputs(project):
    """The on-disk half of the §7-A gate: does every line have its audio?

    Separate from :func:`check_production` because ``production.json`` is
    portable and these checks are about one machine's copy of the project.
    """
    report = Report()
    document = project.production
    for si, shot in enumerate(document.get("shots", [])):
        for di, line in enumerate(shot.get("dialogue", [])):
            line_id = line.get("line_id") or "?"
            source = os.path.join(project.audio_dir(line_id), "source.wav")
            if not os.path.exists(source):
                report.add(f"/shots/{si}/dialogue/{di}/line_id",
                           f"no recording at {project.rel(source)} — the audio is "
                           "the master clock, so this line cannot be timed")
        for ci, member in enumerate(shot.get("cast", [])):
            take = member.get("take")
            if not take:
                continue
            video = os.path.join(project.performance_dir(take), "video.mp4")
            if not os.path.exists(video):
                report.add(f"/shots/{si}/cast/{ci}/take",
                           f"no performance video at {project.rel(video)}; "
                           "the motion stage will fall back to a placeholder",
                           "warn")
    for character in document.get("characters", []):
        folder = project.refs_dir(character["id"])
        found = _images(folder)
        if not found:
            report.add(f"/characters/{_pos(document['characters'], character['id'])}",
                       f"no reference images in {project.rel(folder)}",
                       "warn")
    return report


def resolve_rig_mode(character, overrides):
    """§6.2: overrides -> character.rig_default -> derived from kind."""
    chosen = overrides.get(character["id"])
    if chosen:
        return chosen, "override"
    if character.get("rig_default"):
        return character["rig_default"], "rig_default"
    return KIND_RIG.get(character.get("kind"), "rigged"), "kind"


IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")


def _images(folder):
    try:
        return sorted(name for name in os.listdir(folder)
                      if name.lower().endswith(IMAGE_EXT))
    except OSError:
        return []


def _index(items, key):
    return {item[key]: item for item in items if isinstance(item, dict) and key in item}


def _pos(items, value, key="id"):
    for index, item in enumerate(items):
        if item.get(key) == value:
            return index
    return 0


def _check_unique(report, items, key, pointer):
    seen = {}
    for index, item in enumerate(items):
        value = item.get(key)
        if value in seen:
            report.add(f"{pointer}/{index}/{key}",
                       f"{value!r} is already used at {pointer}/{seen[value]}")
        else:
            seen[value] = index
