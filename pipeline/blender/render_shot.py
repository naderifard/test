"""j_render, in Blender: the headless batch worker, and the audio mux.

Section 7-J. The socket render in ``session.py`` stays as the *preview* path --
it is what makes the live viewport and honest progress reporting possible, and
that was a good decision. This is the other one: a Blender process that opens
the shot, renders the whole range in a single ``render(animation=True)`` call
with ``use_persistent_data`` on and OptiX enabled, and exits. For a 360-frame
Cycles shot the difference between that and a frame at a time over a socket is
the difference between minutes and hours.

Three things this file is careful about.

**OptiX has to be re-enabled here.** The graph starts Blender with
``--factory-startup``, and the compute device lives in *addon preferences*, not
in the .blend -- so a scene saved with ``cycles.device = "GPU"`` opens in a
process that has no GPU configured and renders on the CPU without complaining.
The backend and the device names it found are reported as metrics, so "why did
this take an hour" is answerable from the run summary.

**The jaw is measured off the built scene, not off the file it came from.**
Section 11's lip-sync score is only worth having if the curve it reads is the
one that will render. So the shape key value is sampled per frame out of the
evaluated scene here, written into ``render.json``, and correlated against the
*audio file* by the host step. Both ends of that comparison are then artifacts
rather than intentions, which is what makes it catch a curve keyed at the wrong
frame rather than merely agreeing with the number that placed it.

**The mux is part of the encode, not a pass after it.** Blender bundles its own
ffmpeg, so the dialogue goes into the same throwaway VSE scene as the image
sequence and comes out in one file -- but the audio codec has to be set
explicitly or the container comes out silent, which is exactly the defect
section 15-4 recorded and ``session.py:243`` fixed for the preview path. The
recipe here is that one, with the per-line trim and placement ``shot.json``
carries.
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

FRAME_PATTERN = "frame_#####"
FRAME_GLOB = "frame_%05d.png"

# The channel the picture sits on in the encode scene; dialogue stacks above it.
VIDEO_CHANNEL = 1
AUDIO_CHANNEL_BASE = 2


def _abs(job, relative):
    return os.path.join(job.project_root, str(relative).replace("/", os.sep))


# -- the render --------------------------------------------------------------
def _configure(scene, shot, pack, params):
    """Batch-render settings, on top of whatever the style pack already set."""
    render = scene.render
    render.image_settings.media_type = "IMAGE"
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGB"
    render.image_settings.compression = 15
    render.resolution_x, render.resolution_y = shot["resolution"]
    render.resolution_percentage = int(round(
        100.0 * float(params.get("resolution_scale") or 1.0)))
    render.fps = shot["fps"]
    scene.frame_start = shot["frame_start"]
    scene.frame_end = shot["frame_end"]

    # Section 7-J asks for this by name, and it is worth having on both engines:
    # it keeps the BVH and the shader compilation between frames of one shot.
    render.use_persistent_data = True

    backend, devices, cycles = None, [],         scene.render.engine.upper().startswith("CYCLES")
    if cycles:
        backend, devices = B.enable_gpu_compute()
        try:
            scene.cycles.device = "GPU" if devices else "CPU"
        except Exception:                                       # noqa: BLE001
            pass
        samples = pack.render.get("samples")
        if samples:
            scene.cycles.samples = int(samples)
    return {
        "engine": scene.render.engine,
        # Only meaningful for Cycles: EEVEE has no compute-device setting, and
        # reporting the untouched scene.cycles.device would read as "this
        # rendered on the CPU".
        "device": (getattr(scene.cycles, "device", None) if cycles else None),
        "compute_backend": backend,
        "devices": devices,
        "samples": (getattr(scene.cycles, "samples", None) if cycles
                    else getattr(scene.eevee, "taa_render_samples", None)),
        "persistent_data": bool(render.use_persistent_data),
        "resolution": [render.resolution_x, render.resolution_y],
        "resolution_percentage": render.resolution_percentage,
        "view_transform": scene.view_settings.view_transform,
        "compositing": bool(render.use_compositing)
                       and scene.compositing_node_group is not None,
    }


def _sample_jaw(scene, shot):
    """The jaw-open value per frame, straight out of the evaluated scene.

    This is the curve that will render. Reading it from ``visemes.json``
    instead would make section 11's lip-sync check a comparison between two
    numbers the host already agreed on; reading it here makes it a check on
    whether ``h_assemble`` actually keyed what it said it keyed.
    """
    curves = {}
    for member in shot["cast"]:
        cid = member["character"]
        mesh = bpy.data.objects.get(
            shot["provenance"]["cast"][cid].get("mesh") or cid)
        keys = getattr(getattr(mesh, "data", None), "shape_keys", None)
        if keys is None or "jaw_open" not in keys.key_blocks:
            continue
        block = keys.key_blocks["jaw_open"]
        track = []
        for frame in range(shot["frame_start"], shot["frame_end"] + 1):
            scene.frame_set(frame)
            track.append(round(block.value, 5))
        curves[cid] = track
    scene.frame_set(shot["frame_start"])
    return curves


def _render(scene, out_dir):
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    scene.render.filepath = os.path.join(frames_dir, FRAME_PATTERN)
    scene.render.use_file_extension = True
    bpy.ops.render.render(animation=True, write_still=False)
    return frames_dir


def _frame_paths(frames_dir, shot):
    return [os.path.join(frames_dir, FRAME_GLOB % frame)
            for frame in range(shot["frame_start"], shot["frame_end"] + 1)]


# -- the encode --------------------------------------------------------------
def _strips(editor):
    """5.x renamed ``sequences`` to ``strips``; an empty collection is falsy,
    so ``a or b`` would discard the real one. Test for None."""
    collection = getattr(editor, "strips", None)
    if collection is None:
        collection = editor.sequences
    return collection


def _encode(job, shot, frames_dir, output, fps):
    """PNG sequence plus dialogue into one H.264 file, in a throwaway scene."""
    files = sorted(name for name in os.listdir(frames_dir)
                   if name.lower().endswith(".png"))
    if not files:
        raise RuntimeError("there are no rendered frames to encode")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    scene = bpy.data.scenes.new("bstudio_encode")
    scene.render.resolution_x, scene.render.resolution_y = shot["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.fps = int(fps)
    scene.frame_start = 1
    scene.frame_end = len(files)
    # FFMPEG only appears as a format once the media type is VIDEO.
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

    editor = scene.sequence_editor_create()
    strips = _strips(editor)
    picture = strips.new_image(name="frames",
                               filepath=os.path.join(frames_dir, files[0]),
                               channel=VIDEO_CHANNEL, frame_start=1)
    for name in files[1:]:
        picture.elements.append(name)

    muxed = []
    for index, track in enumerate(shot["audio"]["tracks"]):
        path = _abs(job, track["path"])
        if not os.path.exists(path):
            muxed.append({"line_id": track["line_id"], "muxed": False,
                          "note": f"no audio at {track['path']}"})
            continue
        # Encode-scene frame 1 is the shot's first frame.
        placed = track["start_frame"] - shot["frame_start"] + 1
        offset = int(round(track["trim_start_s"] * fps))
        sound = strips.new_sound(name=track["line_id"], filepath=path,
                                 channel=AUDIO_CHANNEL_BASE + index,
                                 frame_start=placed - offset)
        # frame_start is where the *file* begins; the offset skips the take's
        # own lead-in, so the speech lands on `placed` exactly.
        sound.frame_offset_start = offset
        sound.frame_final_duration = max(
            1, int(round(track["duration_s"] * fps)))
        muxed.append({
            "line_id": track["line_id"],
            "muxed": True,
            "channel": AUDIO_CHANNEL_BASE + index,
            "starts_on_frame": int(sound.frame_final_start),
            "wanted_frame": placed,
            "frames": int(sound.frame_final_duration),
        })

    bpy.ops.render.render(animation=True, scene=scene.name)
    bpy.data.scenes.remove(scene)
    return _finished(output), muxed


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


# -- the step ----------------------------------------------------------------
def render(job, unit):
    params = unit["params"]
    project_shot = _abs(job, params["shot_json"])
    shot = jsonio.read(project_shot, default=None)
    if shot is None:
        raise RuntimeError(f"no shot.json at {params['shot_json']}")
    pack = job.style()

    _common.open_blend(_abs(job, params["scene_blend"]))
    scene = bpy.context.scene
    settings = _configure(scene, shot, pack, params)
    jaw = _sample_jaw(scene, shot)

    frames_dir = _render(scene, unit["out_dir"])
    expected = _frame_paths(frames_dir, shot)
    missing = [os.path.basename(p) for p in expected if not os.path.exists(p)]
    empty = [os.path.basename(p) for p in expected
             if os.path.exists(p) and os.path.getsize(p) == 0]

    video, muxed = _encode(job, shot, frames_dir,
                           os.path.join(unit["out_dir"], "shot.mp4"),
                           shot["fps"])

    record = {
        "schema_version": 1,
        "shot_id": shot["shot_id"],
        "style_pack": pack.name,
        "fps": shot["fps"],
        "frame_start": shot["frame_start"],
        "frame_end": shot["frame_end"],
        "frame_count": shot["frame_count"],
        "render": settings,
        "frames": len(expected) - len(missing),
        "missing_frames": missing,
        "empty_frames": empty,
        "video": os.path.basename(video),
        "video_bytes": os.path.getsize(video),
        "audio": muxed,
        # What the host step correlates against the recordings. Sampled off the
        # evaluated scene, so it is the curve that rendered.
        "jaw_open": jaw,
    }
    jsonio.write(os.path.join(unit["out_dir"], "render.json"), record)

    return {
        "metrics": {
            "shot_id": shot["shot_id"],
            "style_pack": pack.name,
            "frames_rendered": record["frames"],
            "frames_expected": len(expected),
            "missing_frames": missing,
            "empty_frames": empty,
            "video_bytes": record["video_bytes"],
            "audio_muxed": [m["line_id"] for m in muxed if m["muxed"]],
            "audio_expected": [t["line_id"] for t in shot["audio"]["tracks"]],
            "audio_placement": {m["line_id"]: [m.get("starts_on_frame"),
                                               m.get("wanted_frame")]
                                for m in muxed if m["muxed"]},
            "jaw_characters": sorted(jaw),
            "jaw_peak": {cid: round(max(track), 4) if track else 0.0
                         for cid, track in jaw.items()},
            **settings,
        },
        "notes": [
            "%s/%s: %d frames at %sx%s on %s%s, %.1f MB of video"
            % (shot["shot_id"], pack.name, record["frames"],
               settings["resolution"][0], settings["resolution"][1],
               settings["engine"],
               (" (%s: %s)" % (settings["compute_backend"],
                               ", ".join(settings["devices"])))
               if settings["devices"] else "",
               record["video_bytes"] / 1048576.0)]
        + ["%s muxed on frame %d (wanted %d)"
           % (m["line_id"], m["starts_on_frame"], m["wanted_frame"])
           for m in muxed if m["muxed"]]
        + ["%s was NOT muxed: %s" % (m["line_id"], m.get("note"))
           for m in muxed if not m["muxed"]],
    }


if __name__ == "__main__":
    sys.exit(_base.stage_main(render, from_blender=True))
