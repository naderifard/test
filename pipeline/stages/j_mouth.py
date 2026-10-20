"""Stage J2 — the mouths: MuseTalk repaints each speaker's lower face.

Between the render and the edit. A shot where somebody speaks gets two steps:

1. **blender** — ``pipeline/blender/face_track.py`` follows each speaker's
   face landmarks through the animated scene and projects them through the
   shot camera: where the face is and how far it is turned, every frame.
2. **motion** — ``pipeline/lipsync_musetalk.py`` builds each speaker's own
   audio on the shot's timeline, repaints their lower face over the frames of
   their lines with MuseTalk, and encodes the shot again with its original
   sound.

Narration moves no mouth, so the narrator is never a speaker here; neither is
an animal. A shot nobody speaks in has no unit and ``k_edit`` cuts its render
directly. A frame MuseTalk cannot do well -- a small face, a face turned away,
a face off screen -- keeps the rendered mouth, and each painted run fades in
and out so the switch does not pop. How much of each line was painted is a
metric, and a line that was hardly painted is a warning, because then the
mouth on screen is the old one and nobody asked for that.

The film cuts ``shots/<id>/<style>_mouth/shot.mp4`` when it exists
(``output_video``), so a machine without MuseTalk still cuts a film.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio

NARRATOR = "narrator"
SPEAKING_KINDS = ("human", "humanoid")
PAINT_WARN_SHARE = 0.5
CHUNK = 24


def speakers(document, kinds):
    """The characters who speak in a shot and have a face MuseTalk can do."""
    out = []
    for track in (document.get("audio") or {}).get("tracks") or []:
        cid = track.get("character")
        if not cid or cid == NARRATOR or cid in out:
            continue
        if kinds.get(cid, "human") in SPEAKING_KINDS:
            out.append(cid)
    return out


def _kinds(project):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={}) or {}
    return {c["id"]: c.get("kind", "human")
            for c in resolved.get("characters") or []}


def output_video(project, shot_id, style_pack, document=None):
    """The shot video the edit should cut: the repainted one for a shot with
    speakers, once this stage has made it; the render otherwise."""
    rendered = project.path("shots", shot_id, style_pack, "shot.mp4")
    if document is None:
        document = jsonio.read(project.path("shots", shot_id, "shot.json"),
                               default={}) or {}
    if not speakers(document, _kinds(project)):
        return rendered
    mouthed = project.path("shots", shot_id, f"{style_pack}_mouth", "shot.mp4")
    return mouthed if os.path.exists(mouthed) else rendered


def plan(project, graph):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={}) or {}
    kinds = _kinds(project)
    units = []
    for shot in resolved.get("shots") or []:
        shot_id = shot["id"]
        document = jsonio.read(project.path("shots", shot_id, "shot.json"),
                               default={}) or {}
        cast = speakers(document, kinds)
        if not cast:
            continue
        inputs = {
            "scene": project.path("shots", shot_id, "scene.blend"),
            "shot": project.path("shots", shot_id, "shot.json"),
            "render": project.path("shots", shot_id, graph.style_pack,
                                   "shot.mp4"),
        }
        for track in (document.get("audio") or {}).get("tracks") or []:
            if track.get("character") in cast:
                inputs[f"audio_{track['line_id']}"] = project.path(
                    *track["path"].split("/"))
        units.append(Unit(
            f"mouth_{shot_id}",
            inputs=inputs,
            params={
                "shot_id": shot_id,
                "style_pack": graph.style_pack,
                "speakers": cast,
                "scene_blend": f"shots/{shot_id}/scene.blend",
                "shot_json": f"shots/{shot_id}/shot.json",
                "frames_rel": f"shots/{shot_id}/{graph.style_pack}/frames",
                "video_rel": f"shots/{shot_id}/{graph.style_pack}/shot.mp4",
                "resolution_scale": float(
                    getattr(graph, "resolution_scale", None) or 1.0),
                "backend": "musetalk_v15",
            },
            out_rel=f"shots/{shot_id}/{graph.style_pack}_mouth",
            label=f"mouth/{shot_id} ({', '.join(cast)})",
        ))
    return units


def nothing_to_do(project):
    return "no shot has a speaking character"


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _encode(frames_dir, fps, source_video, output):
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run([ffmpeg, "-loglevel", "error", "-y",
                    "-framerate", str(fps), "-i",
                    os.path.join(frames_dir, "%04d.png"),
                    "-i", source_video, "-map", "0:v", "-map", "1:a?",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                    "-c:a", "copy", "-shortest", output], check=True)


def paint(job, unit):
    import cv2
    import numpy as np

    from pipeline import lipsync_musetalk as lm

    params = unit["params"]
    root = job.project_root

    def absolute(rel):
        return os.path.join(root, rel.replace("/", os.sep))

    shot = jsonio.read(absolute(params["shot_json"]))
    fps = int(shot["fps"])
    faces = jsonio.read(_base.out_path(unit, "faces.json"), default={}) or {}
    source_dir = absolute(params["frames_rel"])
    names = sorted(n for n in os.listdir(source_dir) if n.endswith(".png"))
    count = shot["frame_end"] - shot["frame_start"] + 1
    if len(names) != count:
        raise RuntimeError(f"{len(names)} rendered frames against {count} in "
                           "the shot's range")
    frames_dir = os.path.dirname(_base.out_path(unit, "frames", "x"))
    for stale in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, stale))
    # Output frames are numbered from 1, like the render's.
    out_names = [f"{i + 1:04d}.png" for i in range(count)]

    painter = lm.Painter()
    painted = set()
    report = {}

    def current(i):
        """A frame as it stands: repainted already (an earlier speaker in
        the shot) or as rendered."""
        path = (os.path.join(frames_dir, out_names[i]) if i in painted
                else os.path.join(source_dir, names[i]))
        return cv2.imread(path)
    for cid in params["speakers"]:
        rows = faces.get(cid)
        if not rows:
            report[cid] = {"note": "no face track"}
            continue
        tracks = []
        for track in shot["audio"]["tracks"]:
            if track.get("character") == cid:
                t = dict(track)
                t["abs_path"] = absolute(track["path"])
                tracks.append(t)
        wav = _base.out_path(unit, f"voice_{cid}.wav")
        windows = lm.speaker_audio(tracks, shot, fps, wav)
        chunks = painter.chunks(wav, fps)
        speaking = [False] * count
        for first, last in windows:
            for i in range(max(0, first - lm.PAD_FRAMES),
                           min(count, last + lm.PAD_FRAMES)):
                speaking[i] = True
        sample = cv2.imread(os.path.join(source_dir, names[0]))
        height, width = sample.shape[:2]
        boxes, ok, why = [], [], {}
        for i in range(count):
            box = lm.face_box(rows[i]["points"], width, height)
            good, reason = lm.usable(rows[i], box, height)
            good = good and speaking[i] and i < len(chunks)
            if speaking[i] and not good and reason:
                why[reason] = why.get(reason, 0) + 1
            boxes.append(box)
            ok.append(good)
        spoken = sum(speaking)
        done = 0
        smeared = 0
        for run in lm.runs(ok):
            first, last = run
            before, after, written = [], [], []
            # A run is painted a slice at a time: a long line at 1080p would
            # otherwise hold gigabytes of frames.
            for lo in range(first, last + 1, CHUNK):
                idx = list(range(lo, min(last, lo + CHUNK - 1) + 1))
                images = [current(i) for i in idx]
                padded = [lm.padded(image, boxes[i])
                          for i, image in zip(idx, images)]
                faces_out = painter.paint([p[0] for p in padded],
                                          [p[1] for p in padded],
                                          [chunks[i] for i in idx])
                for i, image, face, (big, box, pad) in zip(
                        idx, images, faces_out, padded):
                    blended = lm.unpadded(painter.blend(big, face, box), pad)
                    alpha = lm.fade(i, run)
                    if alpha < 1.0:
                        blended = (image.astype(np.float32) * (1.0 - alpha)
                                   + blended.astype(np.float32) * alpha
                                   ).round().astype(np.uint8)
                    before.append(lm.detail(image, boxes[i]))
                    after.append(lm.detail(blended, boxes[i]))
                    cv2.imwrite(os.path.join(frames_dir, out_names[i]),
                                blended)
                    written.append(i)
            if lm.kept_detail(before, after) < lm.DETAIL_KEEP:
                # The repaint smeared this run: it keeps the rendered frames
                # (restored below, with every frame never painted).
                for i in written:
                    os.remove(os.path.join(frames_dir, out_names[i]))
                smeared += len(written)
                continue
            painted.update(written)
            done += len(written)
        heights = [boxes[i][3] - boxes[i][1] for i in range(count) if ok[i]]
        report[cid] = {"spoken_frames": spoken, "painted_frames": done,
                       "painted_share": round(done / float(spoken), 3)
                       if spoken else 0.0,
                       "skipped": dict(why, **({"smeared by the repaint":
                                                smeared} if smeared else {})),
                       "face_px_median": (sorted(heights)[len(heights) // 2]
                                          if heights else 0)}

    for i in range(count):
        if i in painted:
            continue
        target = os.path.join(frames_dir, out_names[i])
        if os.path.exists(target):
            os.remove(target)
        _link_or_copy(os.path.join(source_dir, names[i]), target)
    output = _base.out_path(unit, "shot.mp4")
    _encode(frames_dir, fps, absolute(params["video_rel"]), output)
    jsonio.write(_base.out_path(unit, "mouth.json"), report)
    notes = []
    for cid, r in report.items():
        if "note" in r:
            notes.append(f"{cid}: {r['note']}")
            continue
        skipped = ", ".join(f"{n} {k}" for k, n in r["skipped"].items())
        notes.append(f"{cid}: painted {r['painted_frames']} of "
                     f"{r['spoken_frames']} speaking frames, face "
                     f"{r['face_px_median']} px"
                     + (f" (kept the rendered mouth: {skipped})"
                        if skipped else ""))
    return {"metrics": {"speakers": report,
                        "painted_frames": len(painted),
                        "frames": count},
            "notes": notes}


def _gate(project, unit, metrics):
    out = []
    for cid, r in (metrics.get("speakers") or {}).items():
        if "note" in r:
            out.append(gates.Verdict(gates.WARN, "mouth.track",
                                     f"{cid}: {r['note']}; the rendered mouth "
                                     "stays"))
            continue
        share = r.get("painted_share", 0.0)
        if r.get("spoken_frames") and share < PAINT_WARN_SHARE:
            out.append(gates.Verdict(
                gates.WARN, "mouth.coverage",
                f"{cid}: only {share:.0%} of the speaking frames were "
                f"repainted ({r.get('skipped')}); the rest show the rendered "
                "mouth", share, PAINT_WARN_SHARE))
    return out or [gates.Verdict(
        gates.PASS, "mouth",
        f"{metrics.get('painted_frames')} of {metrics.get('frames')} frames "
        "repainted")]


STAGE = StageSpec(
    name="j_mouth",
    # 2: a repaint that smears the mouth's detail is refused.
    version="2",
    summary="repaint each speaker's mouth over the render with MuseTalk",
    needs=["j_render"],
    steps=[Step("blender", "pipeline/blender/face_track.py"), Step("motion")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(paint))
