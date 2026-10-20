"""Stage K — the film: shots in order, and Persian subtitles that are free.

Section 7-K. Two steps:

1. **host** — decide the edit. Where each shot starts in film time, where each
   spoken line falls once its shot has been offset, and the subtitle file. All
   of it arithmetic over ``shot.json`` and ``align.json``, written to
   ``edit.json`` so the cut is a document that can be read and hand-edited
   rather than a decision buried in an encoder.
2. **blender** — ``pipeline/blender/build_film.py`` lays the shot MP4s end to
   end in a sequencer and encodes ``final.mp4``.

**The subtitles cost nothing, which is the point.** Section 7-K says so plainly:
the alignment already contains word timings, so a subtitle is a line of text
with two numbers that were measured three stages ago. Nothing here re-derives a
timing; ``align.json`` says where the speech is inside the recording, the shot
says where the recording sits in the shot, and the edit says where the shot sits
in the film. Three additions.

**RTL is not a font problem.** Persian subtitle text is stored logically -- the
first character of the string is the first character *read*, which is the one
that renders rightmost -- and every cue is prefixed with U+200F RIGHT-TO-LEFT
MARK so a player that guesses paragraph direction from the first strong
character cannot get it wrong when a line opens with a digit or a quotation
mark. Reversing the string, which is the usual mistake, would produce something
that looks right in a naive viewer and is wrong everywhere else.

Like ``j_render``, the film is published under ``film/<style_pack>/`` rather
than section 5.3's bare ``film/``: the whole point of section 13's "render it in
all three styles" is to have the three side by side.
"""

from __future__ import annotations

import os
import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio

from . import j_mouth

# A beat of black between shots would be a cut to nothing; shots butt together.
# Kept as a named constant because "why is there no gap" is a real question.
SHOT_GAP_FRAMES = 0

# Under this a subtitle flashes rather than reads. The cue is extended rather
# than dropped, because a line that was spoken should be readable.
MIN_CUE_S = 1.2
MIN_VIDEO_BYTES = 8192

# RIGHT-TO-LEFT MARK. Fixes the paragraph direction of a cue whose first strong
# character is not Persian.
RLM = "‏"


def plan(project, graph):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    shots = [shot["id"] for shot in resolved.get("shots") or []]
    if not shots:
        return []

    inputs, order = {}, []
    for shot_id in shots:
        document = jsonio.read(project.path("shots", shot_id, "shot.json"),
                               default={})
        inputs[f"shot_{shot_id}"] = project.path("shots", shot_id, "shot.json")
        # The repainted-mouth version when j_mouth made one (j_mouth.py).
        inputs[f"video_{shot_id}"] = j_mouth.output_video(
            project, shot_id, graph.style_pack, document)
        for track in (document.get("audio") or {}).get("tracks") or []:
            inputs[f"align_{track['line_id']}"] = project.path(
                "audio", track["line_id"], "align.json")
        order.append(shot_id)

    return [Unit(
        "film",
        inputs=inputs,
        params={
            "order": order,
            "style_pack": graph.style_pack,
            "fps": project.fps(),
            # The resolution the shots were rendered at. A draft (--scale 0.5)
            # cut at the project's full resolution put each 960x540 frame in
            # the middle of a black 1920x1080 canvas, because the sequencer
            # centres a smaller image rather than fitting it. The scale is a
            # parameter, so a draft film can never be served as a final one.
            "resolution": [int(round(v * float(graph.resolution_scale or 1.0)))
                           for v in project.config["resolution"]],
            "title": resolved.get("title") or project.name,
            "language": project.config.get("language") or "fa",
        },
        out_rel=f"film/{graph.style_pack}",
        label=f"film ({graph.style_pack}, {len(order)} shot(s))",
    )]


def nothing_to_do(project):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    if not resolved:
        return "no resolved production plan — a_production has not run"
    return "the plan declares no shots to cut together"


# -- subtitles ---------------------------------------------------------------
def _timecode(seconds):
    """SRT's ``HH:MM:SS,mmm``.

    Rounded to milliseconds *first* and then decomposed with integer division.
    Splitting the seconds first and rounding the remainder afterwards lets the
    remainder round up to a full 1000, which writes ``00:00:01,1000`` -- four
    digits in a three-digit field, which a player will either reject or read as
    a completely different time.
    """
    millis = int(round(max(0.0, float(seconds)) * 1000.0))
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    secs, millis = divmod(millis, 1000)
    return "%02d:%02d:%02d,%03d" % (hours, minutes, secs, millis)


def _srt(cues):
    """A SubRip file. UTF-8, LF, logical order, one blank line between cues."""
    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append("%d\n%s --> %s\n%s\n"
                      % (index, _timecode(cue["start_s"]),
                         _timecode(cue["end_s"]), RLM + cue["text"]))
    return "\n".join(blocks)


# -- the host step -----------------------------------------------------------
def cut(job, unit):
    """Decide the edit, write ``edit.json`` and the subtitle file."""
    params = unit["params"]
    project = job.project
    fps = params["fps"]

    entries, cues, cursor, warnings = [], [], 0, []
    for shot_id in params["order"]:
        shot = jsonio.read(unit["inputs"][f"shot_{shot_id}"], default={})
        video = unit["inputs"][f"video_{shot_id}"]
        if not shot:
            raise RuntimeError(f"{shot_id}: no shot.json to cut with")
        if not os.path.exists(video):
            raise RuntimeError(
                f"{shot_id}: no rendered video at {project.rel(video)} — "
                f"j_render has not run for the {params['style_pack']} pack")

        count = shot["frame_count"]
        entries.append({
            "shot_id": shot_id,
            "video": project.rel(video),
            "frame_count": count,
            "film_start_frame": cursor + 1,
            "film_end_frame": cursor + count,
            "start_s": round(cursor / float(fps), 5),
            "duration_s": round(count / float(fps), 5),
            "cast": [m["character"] for m in shot.get("cast") or []],
            "lines": [t["line_id"] for t in shot["audio"]["tracks"]],
        })

        for track in shot["audio"]["tracks"]:
            align = jsonio.read(unit["inputs"][f"align_{track['line_id']}"],
                                default={})
            text = (track.get("text_fa") or align.get("text") or "").strip()
            if not text:
                warnings.append(f"{track['line_id']}: no Persian text to "
                                "subtitle with")
                continue
            # Film time = where the shot starts + where the strip starts in it.
            start = (cursor + track["start_frame"] - shot["frame_start"]) \
                / float(fps)
            end = start + track["duration_s"]
            if end - start < MIN_CUE_S:
                end = start + MIN_CUE_S
            cues.append({
                "line_id": track["line_id"],
                "character": track.get("character"),
                "shot": shot_id,
                "start_s": round(start, 5),
                "end_s": round(end, 5),
                "text": text,
                "words": len(align.get("words") or []),
            })
        cursor += count + SHOT_GAP_FRAMES

    # Two cues that overlap are two subtitles on screen at once. Shots butt
    # together, so this can only happen if a tail handle is longer than the next
    # shot's head - worth saying, not worth stopping for.
    for first, second in zip(cues, cues[1:]):
        if second["start_s"] < first["end_s"] - 1e-6:
            warnings.append(
                f"{second['line_id']} starts at {second['start_s']:.2f} s while "
                f"{first['line_id']} is still on screen until "
                f"{first['end_s']:.2f} s")

    edit = {
        "schema_version": 1,
        "title": params["title"],
        "style_pack": params["style_pack"],
        "language": params["language"],
        "fps": fps,
        "resolution": params["resolution"],
        "frame_count": cursor,
        "duration_s": round(cursor / float(fps), 5),
        "shots": entries,
        "subtitles": {"file": "subs_fa.srt", "cues": cues},
    }
    jsonio.write(_base.out_path(unit, "edit.json"), edit)

    target = _base.out_path(unit, "subs_fa.srt")
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_srt(cues))

    return {
        "metrics": {
            "style_pack": params["style_pack"],
            "shots": len(entries),
            "shot_order": [e["shot_id"] for e in entries],
            "frame_count": cursor,
            "duration_s": edit["duration_s"],
            "cues": len(cues),
            "subtitled_lines": [c["line_id"] for c in cues],
            "cue_seconds": round(sum(c["end_s"] - c["start_s"] for c in cues),
                                 3),
            "edit_warnings": warnings,
        },
        "notes": [f"{len(entries)} shot(s), {cursor} frames "
                  f"({edit['duration_s']:.2f} s), {len(cues)} subtitle cue(s)"]
        + [f"{e['shot_id']}: frames {e['film_start_frame']}-"
           f"{e['film_end_frame']}" for e in entries]
        + warnings,
    }


# -- the gate ----------------------------------------------------------------
def _gate(project, unit, metrics):
    out = []

    wanted = metrics.get("frame_count") or 0
    built = metrics.get("built_frames")
    if built is not None and built != wanted:
        out.append(gates.Verdict(
            gates.FAIL, "edit.frames",
            f"the film is {built} frames against the {wanted} the cut asks for",
            built, wanted))

    for shot_id in metrics.get("missing_video") or []:
        out.append(gates.Verdict(gates.FAIL, "edit.source",
                                 f"{shot_id} has no rendered video"))

    if (metrics.get("video_bytes") or 0) < MIN_VIDEO_BYTES:
        out.append(gates.Verdict(
            gates.FAIL, "edit.video",
            f"the film encoded to {metrics.get('video_bytes')} bytes",
            metrics.get("video_bytes"), MIN_VIDEO_BYTES))

    silent = [s for s in (metrics.get("shot_audio") or {})
              if not metrics["shot_audio"][s]]
    spoken = set(metrics.get("subtitled_lines") or [])
    if spoken and metrics.get("audio_strips") == 0:
        out.append(gates.Verdict(
            gates.FAIL, "edit.audio",
            "the film has dialogue in its cut and no sound in its timeline"))

    # A subtitle per spoken line, or the reason there is not one.
    for note in metrics.get("edit_warnings") or []:
        out.append(gates.Verdict(gates.WARN, "edit.subtitles", note))

    if silent:
        out.append(gates.Verdict(
            gates.WARN, "edit.silent_shots",
            f"{', '.join(sorted(silent))} carry no dialogue audio"))

    return out or [gates.Verdict(
        gates.PASS, "edit",
        f"{metrics.get('shots')} shots, {metrics.get('frame_count')} frames "
        f"({(metrics.get('duration_s') or 0.0):.2f} s), "
        f"{metrics.get('cues')} subtitle cue(s), "
        f"{(metrics.get('video_bytes') or 0) / 1048576.0:.1f} MB")]


STAGE = StageSpec(
    name="k_edit",
    version="1",
    summary="cut the shots together, burn in the timings, write the RTL subtitles",
    needs=["j_mouth"],
    steps=[Step("host"), Step("blender", "pipeline/blender/build_film.py")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(cut))
