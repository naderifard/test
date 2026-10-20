"""k_edit, in Blender: the shots laid end to end, with their dialogue.

Nothing is decided here either. ``edit.json``, written by the host step, says
which shot goes where and on which frame; this lays them out in a sequencer and
encodes ``final.mp4``.

Two notes on why it is built the way it is.

**The picture and the sound are two strips over the same file.** A movie strip
carries video only; Blender reads the audio out of the same container as a
separate sound strip, and a film assembled from movie strips alone comes out
silent -- which is section 15-4's defect all over again, one level up. So every
shot that has dialogue in the cut gets both, and the gate counts the sound
strips rather than trusting that they were added.

**A silent shot is asked for no sound strip at all**, rather than being handed
one and having it fail. ``edit.json`` already knows which shots have lines,
because the host step read their ``shot.json`` to place the subtitles.

The strip's own measured length is reported next to the length the edit asked
for. A movie file that decodes to a different number of frames than it was
written with is rare, real, and would otherwise show up as a film that drifts
further out of sync with every cut.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from pipeline import jsonio                                      # noqa: E402
from pipeline.blender import _common                             # noqa: E402
from pipeline.stages import _base                                # noqa: E402

import bpy                                                       # noqa: E402

B = _common.install()

VIDEO_CHANNEL = 1
AUDIO_CHANNEL = 2


def _abs(job, relative):
    return os.path.join(job.project_root, str(relative).replace("/", os.sep))


def _strips(editor):
    """5.x renamed ``sequences`` to ``strips``; an empty collection is falsy."""
    collection = getattr(editor, "strips", None)
    if collection is None:
        collection = editor.sequences
    return collection


def _lay_out(job, scene, edit):
    """One movie strip per shot, plus a sound strip for every shot that speaks."""
    strips = _strips(scene.sequence_editor_create())
    placed, missing, audio = [], [], 0
    for index, entry in enumerate(edit["shots"]):
        path = _abs(job, entry["video"])
        if not os.path.exists(path):
            missing.append(entry["shot_id"])
            continue
        movie = strips.new_movie(name=entry["shot_id"], filepath=path,
                                 channel=VIDEO_CHANNEL,
                                 frame_start=entry["film_start_frame"])
        record = {
            "shot_id": entry["shot_id"],
            "wanted_frames": entry["frame_count"],
            "decoded_frames": int(movie.frame_final_duration),
            "from_frame": int(movie.frame_final_start),
            "sound": False,
        }
        if entry.get("lines"):
            try:
                sound = strips.new_sound(
                    name=entry["shot_id"] + "_audio", filepath=path,
                    channel=AUDIO_CHANNEL + (index % 2),
                    frame_start=entry["film_start_frame"])
            except Exception as exc:                            # noqa: BLE001
                record["sound_error"] = f"{type(exc).__name__}: {exc}"
            else:
                sound.frame_final_duration = max(
                    1, min(int(sound.frame_final_duration),
                           entry["frame_count"]))
                record["sound"] = True
                record["sound_frames"] = int(sound.frame_final_duration)
                audio += 1
        placed.append(record)
    return placed, missing, audio


def _encode(scene, edit, output):
    scene.render.resolution_x, scene.render.resolution_y = edit["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.fps = int(edit["fps"])
    scene.frame_start = 1
    scene.frame_end = int(edit["frame_count"])
    scene.render.image_settings.media_type = "VIDEO"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.gopsize = 12
    scene.render.ffmpeg.audio_codec = "AAC"
    scene.render.ffmpeg.audio_bitrate = 192
    scene.render.filepath = output
    os.makedirs(os.path.dirname(output), exist_ok=True)
    bpy.ops.render.render(animation=True, scene=scene.name)
    return _finished(output)


def _finished(output):
    """Blender appends its own frame range to a video filename."""
    if os.path.exists(output):
        return output
    stem, extension = os.path.splitext(output)
    folder = os.path.dirname(output) or "."
    base = os.path.basename(stem)
    for candidate in sorted(os.listdir(folder)):
        if candidate.startswith(base) \
                and candidate.lower().endswith(extension.lower()):
            found = os.path.join(folder, candidate)
            if found != output:
                try:
                    os.replace(found, output)
                    return output
                except OSError:
                    return found
    raise RuntimeError("the encode finished but produced no video file")


def build(job, unit):
    edit = jsonio.read(_base.out_path(unit, "edit.json"), default=None)
    if edit is None:
        raise RuntimeError("the host step wrote no edit.json to build from")

    _common.new_file()
    scene = bpy.context.scene
    placed, missing, audio = _lay_out(job, scene, edit)
    video = _encode(scene, edit, os.path.join(unit["out_dir"], "final.mp4"))

    drift = [p for p in placed
             if abs(p["decoded_frames"] - p["wanted_frames"]) > 1]
    return {
        "metrics": {
            "built_frames": int(scene.frame_end),
            "video": os.path.basename(video),
            "video_bytes": os.path.getsize(video),
            "missing_video": missing,
            "audio_strips": audio,
            "shot_audio": {p["shot_id"]: p["sound"] for p in placed},
            "shot_frames": {p["shot_id"]: p["decoded_frames"] for p in placed},
            "decode_drift": [p["shot_id"] for p in drift],
            "strips": placed,
        },
        "notes": [
            "%s: %d shot(s) over %d frames, %d with dialogue, %.1f MB"
            % (edit["style_pack"], len(placed), int(scene.frame_end), audio,
               os.path.getsize(video) / 1048576.0)]
        + ["%s: %d frames at %d%s" % (p["shot_id"], p["decoded_frames"],
                                      p["from_frame"],
                                      "" if p["sound"] else " (no dialogue)")
           for p in placed]
        + ["%s decodes to %d frames, the cut expects %d"
           % (p["shot_id"], p["decoded_frames"], p["wanted_frames"])
           for p in drift],
    }


if __name__ == "__main__":
    sys.exit(_base.stage_main(build, from_blender=True))
