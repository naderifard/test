"""The main prompt: the production instructions a strong model writes and the
app reads.

The flow is: the user gives a strong chat model (Opus) the seed prompt and the
Persian story; it returns the *main prompt* and one image prompt per character
and per scene. The main prompt is what the GUI takes first. It is English
throughout -- the in-loop model (Nemotron) works best in English -- except the
spoken lines, which are Persian and are carried through byte for byte.

**Why a line grammar and not JSON.** The main prompt is also something a person
reads and edits before pressing Analyse. A fixed grammar of headings, ``key:
value`` lines and ``-`` items is as easy for a strong model to follow as JSON,
far easier for a person to edit, and parses deterministically -- so the model in
the loop is asked to *repair* a section that does not parse, and to classify
free-text moves, rather than to transcribe the whole document and risk changing
a Persian line on the way.

**Staging is words, not numbers.** Where a character stands and how the camera
frames a shot are said in a small vocabulary ("left, back, facing siamak";
"medium on keyumars, low angle, slow push in"). ``staging.py`` turns the words
into metres. A strong model can pick the right words reliably; nobody, human or
model, reliably picks the right camera coordinates in a text box.

**Ids are stable across edits.** A line's id is its shot id plus its ordinal in
that shot (``s02_warning_1``) and a move's take is its shot id plus the
character. The recordings and videos the user attaches in the panel are filed
under those ids, so re-analysing an edited prompt keeps them attached as long as
the shot ids stay put.
"""

from __future__ import annotations

import re

NARRATOR = "narrator"

KINDS = ("human", "humanoid", "quadruped", "creature")
EMOTIONS = ("neutral", "grave", "angry", "grieving", "joyful", "afraid",
            "scornful", "tender", "resolute", "weary")
SETTINGS = ("outdoor", "indoor")

# Default standing height by kind, when the prompt gives none.
DEFAULT_HEIGHT_M = {"human": 1.78, "humanoid": 1.9, "quadruped": 1.2,
                    "creature": 2.2}

ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
PERSIAN_RE = re.compile(r"«([^»]+)»")
LINE_RE = re.compile(
    r"^(?P<who>[a-z][a-z0-9_]*)\s*(?:\[(?P<emotion>[a-z_ ]+)\])?\s*:\s*"
    r"«(?P<fa>[^»]+)»\s*(?:\(\s*en\s*:\s*(?P<en>.*?)\s*\))?\s*$")


FORMAT = """\
The main prompt is plain text in exactly this shape. Headings start with `#`
(sections) and `##` (one item). Every other line is either `key: value` or a
list item starting with `- `. English everywhere, except the spoken lines,
which are Persian inside « ».

```
# FILM
title: <English title>
title_fa: <Persian title>
narrator: <yes | no>

# CHARACTERS
## <id>
name: <English name>
name_fa: <Persian name>
kind: <human | humanoid | quadruped | creature>
height_m: <standing height in metres>
look: <one dense English sentence: appearance, dress, colours, age, build>
voice: <how the voice should sound, for the person recording it>

# SCENES
## <id>
name: <English name>
setting: <outdoor | indoor>
look: <one dense English sentence describing the place>
light: <plain-language light: "low golden sun from the left">

# SHOTS
## <id>
scene: <scene id>
summary: <one English sentence: what happens in this shot>
cast:
- <character id>: <x>, <depth>, facing <camera | away | left | right | character id>
camera: <size> on <character id | group>, <angle>, <move>
moves:
- <character id>: <what they do, in plain English>
lines:
- <speaker id> [<emotion>]: «<Persian line>» (en: <English gloss>)
duration_s: <seconds, ONLY for a shot with no lines>
```

Vocabulary:
* `<id>`: lower-case ASCII letters, digits and underscores, starting with a
  letter (`keyumars`, `black_div`, `s03_the_battle`).
* `<x>`: far_left, left, center, right, far_right.
* `<depth>`: front, middle, back.
* `<size>`: extreme_wide, wide, full, medium, medium_close, close_up,
  extreme_close_up.
* `<angle>`: eye_level, low_angle, high_angle.
* `<move>`: static, push_in, pull_out, pan_left, pan_right, orbit_left,
  orbit_right, crane_up, crane_down. Prefix `slow ` for a slow move.
* `<emotion>`: neutral, grave, angry, grieving, joyful, afraid, scornful,
  tender, resolute, weary.
* `<speaker id>` is a character in the shot's cast, or `narrator`.
"""


class MainPromptError(ValueError):
    pass


class Problem:
    """One thing wrong with the prompt, with where it is."""

    def __init__(self, where, message, severity="error"):
        self.where = where
        self.message = message
        self.severity = severity

    def __str__(self):
        return f"{self.where}: {self.message}"

    def as_dict(self):
        return {"where": self.where, "message": self.message,
                "severity": self.severity}


def split_sections(text):
    """``{"FILM": [lines], "CHARACTERS": [lines], ...}`` from the raw text.

    Tolerant of what a model wraps round the prompt: a Markdown fence, a line
    of preamble before the first heading, trailing whitespace.
    """
    sections, current = {}, None
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if line.strip().startswith("```"):
            continue
        heading = re.match(r"^#\s+([A-Za-z ]+?)\s*$", line.strip())
        if heading and not line.strip().startswith("##"):
            current = heading.group(1).strip().upper()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def split_items(lines):
    """``[(id, [lines])]`` for the ``## id`` items inside one section."""
    items, current = [], None
    preamble = []
    for line in lines:
        heading = re.match(r"^##\s+(.+?)\s*$", line.strip())
        if heading:
            current = (heading.group(1).strip(), [])
            items.append(current)
        elif current is not None:
            current[1].append(line)
        elif line.strip():
            preamble.append(line)
    return items, preamble


def fields(lines):
    """Key-value lines and the list blocks under them.

    Returns ``{"key": "value", "cast": ["item", ...], ...}``. A key with an
    empty value followed by ``- `` items becomes a list.
    """
    out, current_list = {}, None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") or stripped == "-":
            if current_list is None:
                out.setdefault("_loose", []).append(stripped[2:].strip())
            else:
                out[current_list].append(stripped[2:].strip())
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*?)\s*:\s*(.*)$", stripped)
        if not match:
            out.setdefault("_unparsed", []).append(stripped)
            continue
        key = match.group(1).strip().lower().replace(" ", "_")
        value = match.group(2).strip()
        if value:
            out[key] = value
            current_list = None
        else:
            out[key] = []
            current_list = key
    return out


def _slug(text):
    slug = re.sub(r"[^a-z0-9_]+", "_", text.strip().lower()).strip("_")
    if slug and not slug[0].isalpha():
        slug = "x_" + slug
    return slug[:48]


def _number(value, where, problems, default=None):
    if value in (None, ""):
        return default
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        problems.append(Problem(where, f"{value!r} is not a number"))
        return default
    return float(match.group(0))


def parse(text):
    """The prompt as a plain dict, plus every problem found on the way.

    Never raises on bad content: a problem list is what the panel shows and
    what the repair loop sends back to the model. Raises
    :class:`MainPromptError` only when there is nothing recognisable at all.
    """
    sections = split_sections(text)
    if not any(k in sections for k in ("CHARACTERS", "SCENES", "SHOTS")):
        raise MainPromptError(
            "no '# CHARACTERS', '# SCENES' or '# SHOTS' heading found; this "
            "does not look like a main prompt")
    problems = []
    doc = {"film": {}, "characters": [], "scenes": [], "shots": []}

    film = fields(sections.get("FILM", []))
    doc["film"] = {
        "title": film.get("title") or "Untitled",
        "title_fa": film.get("title_fa") or "",
        "narrator": str(film.get("narrator", "no")).lower().startswith("y"),
    }

    items, _ = split_items(sections.get("CHARACTERS", []))
    for raw_id, lines in items:
        where = f"CHARACTERS/{raw_id}"
        cid = _slug(raw_id)
        if not ID_RE.match(cid or ""):
            problems.append(Problem(where, f"{raw_id!r} cannot be made an id"))
            continue
        if cid == NARRATOR:
            problems.append(Problem(where, "'narrator' is reserved for "
                                           "narration; it is not a character"))
            continue
        f = fields(lines)
        kind = (f.get("kind") or "human").lower().strip()
        if kind not in KINDS:
            problems.append(Problem(where, f"kind {kind!r} is not one of "
                                           f"{', '.join(KINDS)}"))
            kind = "human"
        height = _number(f.get("height_m") or f.get("height"), where, problems,
                         DEFAULT_HEIGHT_M[kind])
        if not f.get("look"):
            problems.append(Problem(where, "no 'look:' line; the image prompt "
                                           "and the analysis both need it",
                                    "warn"))
        doc["characters"].append({
            "id": cid,
            "name": f.get("name") or raw_id,
            "name_fa": f.get("name_fa") or f.get("name") or raw_id,
            "kind": kind,
            "height_m": height,
            "look": f.get("look") or "",
            "voice": f.get("voice") or "",
        })

    items, _ = split_items(sections.get("SCENES", []))
    for raw_id, lines in items:
        where = f"SCENES/{raw_id}"
        sid = _slug(raw_id)
        f = fields(lines)
        setting = (f.get("setting") or "outdoor").lower()
        if setting not in SETTINGS:
            problems.append(Problem(where, f"setting {setting!r} is not "
                                           f"outdoor or indoor", "warn"))
            setting = "outdoor"
        doc["scenes"].append({
            "id": sid,
            "name": f.get("name") or raw_id,
            "setting": setting,
            "look": f.get("look") or "",
            "light": f.get("light") or "soft daylight",
        })

    character_ids = {c["id"] for c in doc["characters"]}
    scene_ids = {s["id"] for s in doc["scenes"]}
    items, _ = split_items(sections.get("SHOTS", []))
    seen_shots = set()
    for raw_id, lines in items:
        where = f"SHOTS/{raw_id}"
        shot_id = _slug(raw_id.split(":")[0])
        if shot_id in seen_shots:
            problems.append(Problem(where, f"shot id {shot_id!r} is used "
                                           f"twice"))
            continue
        seen_shots.add(shot_id)
        f = fields(lines)
        shot = {"id": shot_id, "scene": _slug(f.get("scene") or ""),
                "summary": f.get("summary") or "",
                "cast": [], "camera": f.get("camera") or "",
                "moves": [], "lines": [],
                "duration_s": _number(f.get("duration_s") or f.get("duration"),
                                      where, problems)}
        if shot["scene"] not in scene_ids:
            problems.append(Problem(where, f"scene {f.get('scene')!r} is not "
                                           f"one of the SCENES"))
        for item in f.get("cast") or []:
            who, _, placement = item.partition(":")
            cid = _slug(who)
            if cid not in character_ids:
                problems.append(Problem(where + "/cast",
                                        f"{who.strip()!r} is not a character"))
                continue
            shot["cast"].append({"character": cid,
                                 "placement": placement.strip()})
        cast_ids = {m["character"] for m in shot["cast"]}
        for item in f.get("moves") or []:
            who, _, what = item.partition(":")
            cid = _slug(who)
            if cid not in cast_ids:
                problems.append(Problem(where + "/moves",
                                        f"{who.strip()!r} moves but is not in "
                                        f"the cast"))
                continue
            if what.strip():
                shot["moves"].append({"character": cid, "text": what.strip()})
        for index, item in enumerate(f.get("lines") or []):
            match = LINE_RE.match(item.strip())
            if not match:
                problems.append(Problem(
                    where + f"/lines/{index + 1}",
                    "not in the form 'speaker [emotion]: «Persian» (en: ...)'"))
                continue
            who = match.group("who")
            if who != NARRATOR and who not in cast_ids:
                problems.append(Problem(where + f"/lines/{index + 1}",
                                        f"{who!r} speaks but is not in the "
                                        f"cast"))
                continue
            emotion = (match.group("emotion") or "neutral").strip()
            if emotion not in EMOTIONS:
                problems.append(Problem(where + f"/lines/{index + 1}",
                                        f"emotion {emotion!r} is not one of "
                                        f"the list; using neutral", "warn"))
                emotion = "neutral"
            shot["lines"].append({
                "line_id": line_id(shot_id, len(shot["lines"]) + 1),
                "speaker": who,
                "emotion": emotion,
                "text_fa": match.group("fa").strip(),
                "text_en": (match.group("en") or "").strip(),
            })
        if not shot["cast"]:
            problems.append(Problem(where, "the shot has no cast"))
        if not shot["lines"] and not shot["duration_s"]:
            problems.append(Problem(where, "a shot with no lines needs "
                                           "'duration_s:'", "warn"))
            shot["duration_s"] = 4.0
        if not shot["camera"]:
            problems.append(Problem(where, "no 'camera:' line; a medium shot "
                                           "on the first speaker is used",
                                    "warn"))
        doc["shots"].append(shot)

    if not doc["characters"]:
        problems.append(Problem("CHARACTERS", "no characters"))
    if not doc["scenes"]:
        problems.append(Problem("SCENES", "no scenes"))
    if not doc["shots"]:
        problems.append(Problem("SHOTS", "no shots"))
    if any(l["speaker"] == NARRATOR for s in doc["shots"] for l in s["lines"]):
        doc["film"]["narrator"] = True
    return doc, problems


def line_id(shot_id, ordinal):
    """``s02_warning_1``: stable while the shot id and line order are."""
    return f"{shot_id[:44]}_{ordinal}"


def take_id(shot_id, character_id):
    """The take a move is filed under: shot id plus character, 48 chars max."""
    base = f"{shot_id}__{character_id}"
    if len(base) <= 48:
        return base
    return f"{shot_id[:47 - len(character_id) - 2]}__{character_id}"[:48]


def persian_lines(text):
    """Every «...» string in the raw prompt, in order. What must survive."""
    return [m.group(1).strip() for m in PERSIAN_RE.finditer(text)]
