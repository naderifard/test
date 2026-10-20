"""Stage J — render the shot, mux the dialogue, and measure the sequence.

Section 7-J. Two steps, and the split is forced by where the libraries are:

1. **blender** — ``pipeline/blender/render_shot.py`` opens ``scene.blend``,
   enables OptiX, turns on ``use_persistent_data``, renders the whole frame
   range in one call, and encodes the PNGs and the dialogue into one MP4.
2. **host** — this module measures what came out. Pillow lives in the host
   interpreter and not in Blender's, and both new section 11 metrics need it or
   need the audio files, so the measuring is done here.

The two new metrics are the reason this stage is more than a wrapper around a
render call, because both catch failures that are *invisible in a single still*:

**Temporal flicker** is the mean absolute luminance delta between consecutive
frames. EEVEE sample noise and unstable shadows make a sequence boil while every
individual frame passes every existing check, and that is the difference between
an animation that looks finished and one that looks cheap. The limit is per
style pack (``gates.FLICKER_LIMIT``) because the miniature pack is the fast path
that is meant to carry thousands of EEVEE frames, and one global number would
either be too loose for Cycles or too tight to let the fast path run.

**Lip-sync score** correlates the jaw against the audio envelope, and is worth
more here than anywhere else upstream because *both ends of the comparison are
artifacts*. The jaw comes out of the built scene, sampled per frame off the
evaluated shape key by the Blender step; the envelope comes out of the WAV file
that was muxed. Neither is the number that placed them. So a curve keyed at the
wrong frame, or the wrong line attached to the wrong shot, shows up as an argmax
that is not at zero lag -- which is exactly what section 11 says this metric is
for, and what it could not do if it merely re-read ``visemes.json``.

The frames are published under ``shots/<id>/<style_pack>/`` rather than section
5.3's ``shots/<id>/frames/``. Section 5.3 was written before any look had a
rendered artifact; the store keys per pack already, so the readable tree has to
as well or rendering the slice in all three looks -- which section 13 asks for
by name -- leaves only the last one on disk.
"""

from __future__ import annotations

import os
import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio

# The luminance grid the still analysis is measured on. Wider than the flicker
# grid because this one is asked about a *fraction* of the frame, not about a
# global drift, and 48 cells across is too coarse to say "how much of this is
# black" with a straight face.
ANALYSIS_GRID = 128
NEAR_BLACK = 0.06

# A frame smaller than this is a write that went wrong, whatever the file says.
MIN_FRAME_BYTES = 1024
MIN_VIDEO_BYTES = 4096


# -- planning ----------------------------------------------------------------
def plan(project, graph):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    from blender_ai_studio import styles
    pack = styles.load(graph.style_pack)

    units = []
    for shot in resolved.get("shots") or []:
        shot_id = shot["id"]
        document = jsonio.read(project.path("shots", shot_id, "shot.json"),
                               default={})
        inputs = {
            "scene": project.path("shots", shot_id, "scene.blend"),
            "shot": project.path("shots", shot_id, "shot.json"),
        }
        for track in (document.get("audio") or {}).get("tracks") or []:
            # The recording is hashed as an input because it is *in* the
            # deliverable: re-record a line and the video has to be remade even
            # though not one pixel of it changed.
            inputs[f"audio_{track['line_id']}"] = project.path(
                *track["path"].split("/"))
        units.append(Unit(
            f"shot_{shot_id}",
            inputs=inputs,
            params={
                "shot_id": shot_id,
                "style_pack": graph.style_pack,
                "scene_blend": f"shots/{shot_id}/scene.blend",
                "shot_json": f"shots/{shot_id}/shot.json",
                # `--scale` overrides the pack's own draft setting. Both are
                # in the key: 960x540 and 1920x1080 are different artifacts.
                "resolution_scale": float(
                    getattr(graph, "resolution_scale", None)
                    or pack.render.get("resolution_scale") or 1.0),
                "samples": pack.render.get("samples"),
            },
            out_rel=f"shots/{shot_id}/{graph.style_pack}",
            label=f"render/{shot_id} ({graph.style_pack})",
        ))
    return units


def nothing_to_do(project):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    if not resolved:
        return "no resolved production plan — a_production has not run"
    return "the plan declares no shots"


# -- measurement (the host step) ---------------------------------------------
def _analyse(path):
    """The existing still checks of section 11-J, on one representative frame."""
    stats = gates.measure_image(path, thumb=ANALYSIS_GRID)
    grid = gates.luma_grid(path, width=ANALYSIS_GRID)
    if not grid:
        return {"note": stats.get("note") or "Pillow unavailable"}
    return {
        "frame": os.path.basename(path),
        "mean_luminance": round(sum(grid) / len(grid), 5),
        "contrast": stats.get("stddev_luma"),
        "fraction_near_black": round(
            sum(1 for v in grid if v < NEAR_BLACK) / float(len(grid)), 5),
        "unique_levels": stats.get("unique_levels"),
        "width": stats.get("width"),
        "height": stats.get("height"),
    }


def _envelope(project, track, fps):
    """The RMS envelope of exactly the audio that was muxed, frame by frame."""
    path = project.path(*track["path"].split("/"))
    samples, rate = gates.read_wav_mono(path)
    frames = max(1, int(round(track["duration_s"] * fps)))
    return gates.rms_envelope(samples, rate, fps, frame_count=frames,
                              offset_s=track["trim_start_s"])


def _lipsync(project, shot, record):
    """Section 11's lip-sync score, per line, off the built scene and the WAV."""
    fps = shot["fps"]
    start = shot["frame_start"]
    jaw = record.get("jaw_open") or {}
    speaker_of = {t["line_id"]: t.get("character")
                  for t in shot["audio"]["tracks"]}
    out = {}
    for track in shot["audio"]["tracks"]:
        line_id = track["line_id"]
        if speaker_of.get(line_id) == "narrator":
            # Narration is heard over the picture and spoken by no one in it.
            out[line_id] = {"usable": False, "narration": True,
                            "note": "narration: no mouth to sync"}
            continue
        curve = jaw.get(speaker_of.get(line_id) or "")
        if not curve:
            out[line_id] = {"usable": False,
                            "note": "the speaker's rig drives no jaw_open"}
            continue
        envelope = _envelope(project, track, fps)
        first = max(0, track["start_frame"] - start)
        window = curve[first:first + len(envelope)]
        if len(window) < 3:
            out[line_id] = {"usable": False,
                            "note": "the line does not fit inside the shot"}
            continue
        score = gates.lip_sync_score(window, envelope)
        score["frames"] = len(window)
        score["from_frame"] = start + first
        out[line_id] = score
    return out


def _worst(scores):
    """The line the gate should judge the shot by."""
    usable = [s for s in scores.values() if s.get("usable")]
    if not usable:
        return None
    return min(usable, key=lambda s: (s["r_at_zero"] - abs(s["best_lag"])))


def measure(job, unit):
    project = job.project
    out_dir = unit["out_dir"]
    record = jsonio.read(os.path.join(out_dir, "render.json"), default=None)
    if record is None:
        raise RuntimeError("the render step wrote no render.json to measure")
    shot = jsonio.read(project.path(*unit["params"]["shot_json"].split("/")),
                       default={})

    frames_dir = os.path.join(out_dir, "frames")
    frames = [os.path.join(frames_dir, name)
              for name in sorted(os.listdir(frames_dir))
              if name.lower().endswith(".png")] if os.path.isdir(frames_dir) \
        else []
    small = [os.path.basename(p) for p in frames
             if os.path.getsize(p) < MIN_FRAME_BYTES]

    analysis = _analyse(frames[len(frames) // 2]) if frames else {
        "render_error": "the render produced no frames"}
    flicker = gates.temporal_flicker(frames) if len(frames) > 1 else {
        "usable": False}
    lipsync = _lipsync(project, shot, record) if shot.get("audio") else {}
    worst = _worst(lipsync)

    video = os.path.join(out_dir, record.get("video") or "shot.mp4")
    video_bytes = os.path.getsize(video) if os.path.exists(video) else 0

    record["measured"] = {"analysis": analysis, "flicker": flicker,
                          "lipsync": lipsync}
    jsonio.write(os.path.join(out_dir, "render.json"), record)

    seconds = record["frame_count"] / float(record["fps"] or 24)
    return {
        "metrics": {
            "frames_on_disk": len(frames),
            "undersized_frames": small,
            "video": os.path.basename(video),
            "video_bytes": video_bytes,
            "video_seconds": round(seconds, 3),
            "analysis": analysis,
            "flicker": flicker,
            "lipsync_per_line": lipsync,
            "lipsync": worst,
            "megapixels_rendered": round(
                len(frames) * (analysis.get("width") or 0)
                * (analysis.get("height") or 0) / 1e6, 2),
        },
        "notes": [
            "%s: %d frames, %.2f s, %.1f MB, mean luma %.3f, flicker %.4f"
            % (record["shot_id"], len(frames), seconds,
               video_bytes / 1048576.0,
               analysis.get("mean_luminance") or 0.0,
               (flicker or {}).get("mean_delta") or 0.0)]
        + ["%s: jaw-to-audio r=%.2f at lag %+d over %d frames"
           % (line, score.get("r_at_zero") or 0.0, score.get("best_lag") or 0,
              score.get("frames") or 0)
           for line, score in sorted(lipsync.items()) if score.get("usable")]
        + ["%s: lip sync not measurable — %s" % (line, score.get("note"))
           for line, score in sorted(lipsync.items())
           if not score.get("usable")],
    }


# -- the gate ----------------------------------------------------------------
def _gate(project, unit, metrics):
    out = [v for v in gates.check_render(metrics.get("analysis") or {},
                                         flicker=metrics.get("flicker"),
                                         lipsync=metrics.get("lipsync"),
                                         style_pack=metrics.get("style_pack"))
           if v.level != gates.PASS]

    wanted = metrics.get("frames_expected") or 0
    got = metrics.get("frames_on_disk") or 0
    if wanted and got != wanted:
        out.append(gates.Verdict(
            gates.FAIL, "render.frames",
            f"{got} frames on disk against {wanted} in the shot's range",
            got, wanted))
    for name in metrics.get("missing_frames") or []:
        out.append(gates.Verdict(gates.FAIL, "render.missing",
                                 f"{name} was never written"))
    for name in (metrics.get("empty_frames") or []) \
            + (metrics.get("undersized_frames") or []):
        out.append(gates.Verdict(gates.FAIL, "render.truncated",
                                 f"{name} is too small to be an image"))

    if (metrics.get("video_bytes") or 0) < MIN_VIDEO_BYTES:
        out.append(gates.Verdict(
            gates.FAIL, "render.video",
            f"the encode produced {metrics.get('video_bytes')} bytes",
            metrics.get("video_bytes"), MIN_VIDEO_BYTES))

    # Section 15-4: encode_video used to build a video-only file. The fix was
    # written and never exercised, so this is the check that keeps it honest.
    expected_audio = set(metrics.get("audio_expected") or [])
    muxed = set(metrics.get("audio_muxed") or [])
    for line_id in sorted(expected_audio - muxed):
        out.append(gates.Verdict(
            gates.FAIL, "render.audio",
            f"{line_id} is spoken in this shot and is not in the video"))
    for line_id, (placed, wanted_frame) in sorted(
            (metrics.get("audio_placement") or {}).items()):
        if placed != wanted_frame:
            out.append(gates.Verdict(
                gates.FAIL, "render.audio_offset",
                f"{line_id} was muxed on frame {placed} and belongs on "
                f"{wanted_frame}", placed, wanted_frame))

    for line_id, score in sorted(
            (metrics.get("lipsync_per_line") or {}).items()):
        if not score.get("usable") and not score.get("narration"):
            out.append(gates.Verdict(
                gates.WARN, "render.lipsync_missing",
                f"{line_id}: lip sync could not be measured — "
                f"{score.get('note')}"))

    # A Cycles render that quietly fell back to the CPU is not wrong, but it is
    # the difference between minutes and hours and must not be silent (§15-1).
    if str(metrics.get("engine", "")).upper().startswith("CYCLES") \
            and not metrics.get("devices"):
        out.append(gates.Verdict(
            gates.WARN, "render.device",
            "Cycles rendered on the CPU: no GPU compute backend was enabled in "
            "this process"))

    flicker = metrics.get("flicker") or {}
    lipsync = metrics.get("lipsync") or {}
    return out or [gates.Verdict(
        gates.PASS, "render",
        f"{metrics.get('shot_id')}/{metrics.get('style_pack')}: "
        f"{metrics.get('frames_on_disk')} frames, flicker "
        f"{flicker.get('mean_delta', 0.0):.4f}, "
        + (f"lip sync r={lipsync.get('r_at_zero'):.2f} at lag "
           f"{lipsync.get('best_lag'):+d}" if lipsync.get("usable")
           else "no dialogue"))]


STAGE = StageSpec(
    name="j_render",
    version="1",
    summary="render each shot headless with OptiX, mux the dialogue, measure it",
    needs=["h_assemble"],
    steps=[Step("blender", "pipeline/blender/render_shot.py"), Step("host")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(measure))
