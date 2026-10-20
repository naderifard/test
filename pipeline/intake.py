"""From an analysed main prompt to a project, and from the panel to its files.

The GUI's panel asks for five kinds of input. Each has exactly one place in the
project layout, which is where the stages already look -- so attaching a file
here is the whole integration, and a file dropped into the same place by hand
works identically:

=================  ===========================================  ===============
panel              file                                          read by
=================  ===========================================  ===============
character image    ``refs/characters/<id>/<id>_front.png``        b_refs, c_asset
voice sample       ``audio/voices/<id>.wav``                      f_voice
scene image        ``refs/envs/<id>/<id>_wide.png``               b_refs, d_env
sentence voice     ``audio/<line_id>/source.wav``                 f_voice
move video         ``performance/<take>/video.mp4``               e_motion
=================  ===========================================  ===============

Every attachment is normalised on the way in (PNG; mono 16-bit WAV at the
recording's own rate; H.264 MP4), because the stages downstream were written
against one format each and a user's phone recording is an M4A.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from . import analysis as analysis_mod
from . import jsonio, main_prompt
from .project import Project

DEFAULT_STYLE = "stylized"
KINDS = ("character_image", "voice_sample", "scene_image", "line_audio",
         "move_video")
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus")
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


class IntakeError(RuntimeError):
    pass


def target(project, kind, key):
    """Where an attachment of ``kind`` for ``key`` lives in the project."""
    if kind == "character_image":
        return project.path("refs", "characters", key, f"{key}_front.png")
    if kind == "voice_sample":
        return project.voice_ref(key)
    if kind == "scene_image":
        return project.path("refs", "envs", key, f"{key}_wide.png")
    if kind == "line_audio":
        return project.path("audio", key, "source.wav")
    if kind == "move_video":
        return project.path("performance", key, "video.mp4")
    raise IntakeError(f"unknown attachment kind {kind!r}")


def ffmpeg():
    found = shutil.which("ffmpeg")
    if not found:
        raise IntakeError("ffmpeg is not on PATH; it converts recordings and "
                          "videos to the formats the pipeline reads")
    return found


def _run(command):
    done = subprocess.run(command, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if done.returncode != 0:
        raise IntakeError("ffmpeg failed: " + (done.stderr or "")[-600:])


def attach(project, kind, key, source):
    """Copy ``source`` into its place, converting it. Returns the path."""
    if not os.path.isfile(source):
        raise IntakeError(f"no such file: {source}")
    dest = target(project, kind, key)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    ext = os.path.splitext(source)[1].lower()
    if kind in ("character_image", "scene_image"):
        from PIL import Image
        with Image.open(source) as image:
            image.load()
            mode = "RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB"
            image.convert(mode).save(dest, "PNG")
    elif kind in ("voice_sample", "line_audio"):
        if ext not in AUDIO_EXT:
            raise IntakeError(f"{os.path.basename(source)} is not an audio "
                              f"file this app reads ({', '.join(AUDIO_EXT)})")
        _run([ffmpeg(), "-y", "-loglevel", "error", "-i", source, "-ac", "1",
              "-c:a", "pcm_s16le", dest])
    elif kind == "move_video":
        if ext not in VIDEO_EXT:
            raise IntakeError(f"{os.path.basename(source)} is not a video "
                              f"file this app reads ({', '.join(VIDEO_EXT)})")
        _run([ffmpeg(), "-y", "-loglevel", "error", "-i", source, "-an",
              "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", dest])
    return dest


def detach(project, kind, key):
    """Remove an attachment. Returns whether there was one."""
    path = target(project, kind, key)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def create(name, text, analysis, fps=24):
    """A project for this main prompt: layout, story files, production.json.

    Re-running on an existing project rewrites the story files and keeps every
    attachment, because attachments are filed by id and the ids are stable.
    """
    title = analysis["film"].get("title_fa") or analysis["film"]["title"]
    # Stylized rather than the project default, miniature: characters and
    # sets built from photographs are photographic, and miniature's ink
    # outline pass (edge detection on normals and depth) tore them into black
    # blotches. Changing project.json's style_pack still picks another.
    project = Project.create(name, title=title, style_pack=DEFAULT_STYLE)
    story = project.path("story")
    os.makedirs(story, exist_ok=True)
    with open(os.path.join(story, "main_prompt.md"), "w",
              encoding="utf-8") as handle:
        handle.write(text if text.endswith("\n") else text + "\n")
    jsonio.write(os.path.join(story, "analysis.json"), analysis)
    production = analysis_mod.to_production(analysis, fps=fps)
    jsonio.write(project.production_path, production)
    return project


def status(project, analysis):
    """For every panel row, whether its file is there. The GUI's checkmarks."""
    panel = analysis_mod.inputs_panel(analysis)
    out = {"characters": [], "scenes": [], "sentences": [], "moves": []}
    for row in panel["characters"]:
        out["characters"].append(dict(
            row,
            image=_exists(target(project, "character_image", row["id"])),
            voice=_exists(target(project, "voice_sample", row["id"]))))
    for row in panel["scenes"]:
        out["scenes"].append(dict(
            row, image=_exists(target(project, "scene_image", row["id"]))))
    for row in panel["sentences"]:
        out["sentences"].append(dict(
            row, audio=_exists(target(project, "line_audio", row["line_id"]))))
    for row in panel["moves"]:
        out["moves"].append(dict(
            row, video=_exists(target(project, "move_video", row["take"]))))
    if panel["narrator"]:
        out["narrator_voice"] = _exists(target(project, "voice_sample",
                                               main_prompt.NARRATOR))
    out["missing"] = missing(out)
    return out


def missing(state):
    """What must still be attached before a build can succeed, in words.

    Voice samples, the narrator's sample and move videos are optional: without
    a sample a line keeps the recorded voice, without a video a move is
    authored from its clips.
    """
    needed = []
    for row in state["characters"]:
        if not row["image"]:
            needed.append(f"image for {row['name']}")
    for row in state["scenes"]:
        if not row["image"]:
            needed.append(f"image for scene {row['name']}")
    for row in state["sentences"]:
        if not row["audio"]:
            needed.append(f"recording of {row['line_id']} ({row['speaker']})")
    return needed


def _exists(path):
    return path if os.path.exists(path) else None
