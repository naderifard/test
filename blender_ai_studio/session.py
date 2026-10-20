"""Everything the host does *to* Blender: install the helpers, inspect the
scene, and render the final image or video.

Rendering happens inside the already-running GUI instance rather than a
headless subprocess. A 960x540 EEVEE frame takes well under a second there,
while a fresh headless Blender costs ~25s of startup before it draws anything.
Rendering frame by frame also means the host can report real progress and keep
the viewport preview alive between frames.
"""

from __future__ import annotations

import base64
import json
import os
import posixpath

from .blender_link import BlenderError, BlenderLink

PRELUDE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prelude.py")

OK_MARKER = "__BSTUDIO_OK__"
ERROR_MARKER = "__BSTUDIO_ERROR__"


class CodeResult:
    """Outcome of one snippet: what it printed and whether it raised."""

    def __init__(self, output):
        self.raw = output or ""
        self.failed = ERROR_MARKER in self.raw
        self.output = (
            self.raw.replace(OK_MARKER, "").replace(ERROR_MARKER, "").strip()
        )

    def __bool__(self):
        return not self.failed

    def summary(self, limit=3000):
        body = self.output or ("(no output)" if not self.failed else "")
        if len(body) > limit:
            body = body[:limit] + f"\n... [{len(body) - limit} more characters]"
        return body


def _bpath(path):
    """Blender's Python is happiest with forward slashes on Windows too."""
    return str(path).replace("\\", "/")


class BlenderSession:
    """A live Blender the agent can build in."""

    def __init__(self, link=None, port=9876, on_log=None):
        self.link = link or BlenderLink(port=port)
        self.on_log = on_log
        self._prelude_installed = False

    def _log(self, message):
        if self.on_log:
            self.on_log(message)

    # -- setup --------------------------------------------------------------
    def install_prelude(self, force=False):
        """Inject ``prelude.py`` into Blender as the module ``bstudio``."""
        if self._prelude_installed and not force:
            return True
        with open(PRELUDE_PATH, "r", encoding="utf-8") as fh:
            source = fh.read()
        encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
        installer = (
            "import base64, sys, types\n"
            "_src = base64.b64decode('" + encoded + "').decode('utf-8')\n"
            "_mod = types.ModuleType('bstudio')\n"
            "_mod.__file__ = '<bstudio>'\n"
            "exec(compile(_src, '<bstudio>', 'exec'), _mod.__dict__)\n"
            "sys.modules['bstudio'] = _mod\n"
            "print('installed', len(_mod.__all__), 'helpers')\n"
        )
        result = self.run(installer, timeout=120)
        if result.failed:
            raise BlenderError("Could not install the bstudio helpers:\n" + result.output)
        self._prelude_installed = True
        self._log("Helper library installed in Blender.")
        return True

    def prepare(self, engine="EEVEE", res=(1920, 1080), fps=24, samples=64):
        self.install_prelude()
        code = (
            "import bstudio as B\n"
            f"B.reset(engine={engine!r}, res=({int(res[0])}, {int(res[1])}), fps={int(fps)})\n"
            f"B.set_engine({engine!r}, samples={int(samples)})\n"
            "print(B.report())\n"
        )
        return self.run(code, timeout=180)

    # -- execution ----------------------------------------------------------
    def run(self, code, timeout=240.0):
        return CodeResult(self.link.run_python(code, timeout=timeout))

    def report(self, verbose=False):
        result = self.run(
            "import bstudio as B\nprint(B.report(verbose=%s))" % bool(verbose),
            timeout=120,
        )
        return result.summary()

    def analyse(self, width=240):
        result = self.run(
            "import bstudio as B, json\nprint(json.dumps(B.analyse(width=%d)))" % int(width),
            timeout=300,
        )
        if result.failed:
            return {"render_error": result.summary(1200), "hints": []}
        text = result.output.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return {"render_error": "analyse produced no JSON: " + text[:400], "hints": []}
        try:
            return json.loads(text[start:end + 1])
        except ValueError as exc:
            return {"render_error": f"analyse JSON was malformed: {exc}", "hints": []}

    def screenshot(self, filepath, max_size=900):
        return self.link.viewport_screenshot(_bpath(filepath), max_size=max_size)

    def save_blend(self, filepath):
        code = (
            "import bpy\n"
            f"bpy.ops.wm.save_as_mainfile(filepath={_bpath(filepath)!r})\n"
            "print('saved')\n"
        )
        return self.run(code, timeout=180)

    # -- rendering ----------------------------------------------------------
    def render_still(self, filepath, res=(1920, 1080), samples=None, timeout=1800.0):
        """Render the current frame to a PNG."""
        target = _bpath(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        sample_line = ""
        if samples:
            sample_line = (
                "if sc.render.engine == 'CYCLES':\n"
                f"    sc.cycles.samples = {int(samples)}\n"
                "else:\n"
                f"    sc.eevee.taa_render_samples = {int(samples)}\n"
            )
        code = (
            "import bpy, os\n"
            "sc = bpy.context.scene\n"
            f"sc.render.resolution_x = {int(res[0])}\n"
            f"sc.render.resolution_y = {int(res[1])}\n"
            "sc.render.resolution_percentage = 100\n"
            "sc.render.image_settings.media_type = 'IMAGE'\n"
            "sc.render.image_settings.file_format = 'PNG'\n"
            "sc.render.image_settings.color_mode = 'RGBA'\n"
            + sample_line +
            f"sc.render.filepath = {target!r}\n"
            "bpy.ops.render.render(write_still=True)\n"
            f"print('rendered', os.path.exists({target!r}))\n"
        )
        result = self.run(code, timeout=timeout)
        if result.failed:
            raise BlenderError("Render failed:\n" + result.summary(2000))
        if not os.path.exists(filepath):
            raise BlenderError(f"Render reported success but {filepath} is missing.")
        return filepath

    def render_animation(self, frame_dir, res=(1920, 1080), samples=None,
                         start=None, end=None, on_frame=None, should_stop=None,
                         timeout=1800.0):
        """Render each frame to its own PNG, reporting progress as it goes.

        Returns the list of frame paths, in order.
        """
        os.makedirs(frame_dir, exist_ok=True)
        info = self.run(
            "import bpy\nsc = bpy.context.scene\n"
            "print(sc.frame_start, sc.frame_end, sc.render.fps)",
            timeout=60,
        )
        try:
            scene_start, scene_end, _fps = (int(v) for v in info.output.split()[:3])
        except (ValueError, IndexError):
            scene_start, scene_end = 1, 96
        start = int(scene_start if start is None else start)
        end = int(scene_end if end is None else end)
        if end < start:
            start, end = end, start

        sample_line = ""
        if samples:
            sample_line = (
                "if sc.render.engine == 'CYCLES':\n"
                f"    sc.cycles.samples = {int(samples)}\n"
                "else:\n"
                f"    sc.eevee.taa_render_samples = {int(samples)}\n"
            )
        setup = (
            "import bpy\n"
            "sc = bpy.context.scene\n"
            f"sc.render.resolution_x = {int(res[0])}\n"
            f"sc.render.resolution_y = {int(res[1])}\n"
            "sc.render.resolution_percentage = 100\n"
            "sc.render.image_settings.media_type = 'IMAGE'\n"
            "sc.render.image_settings.file_format = 'PNG'\n"
            "sc.render.image_settings.color_mode = 'RGB'\n"
            + sample_line +
            "print('ready')\n"
        )
        result = self.run(setup, timeout=120)
        if result.failed:
            raise BlenderError("Could not configure the animation render:\n"
                               + result.summary(1500))

        paths = []
        total = end - start + 1
        for index, frame in enumerate(range(start, end + 1)):
            if should_stop and should_stop():
                self._log("Render cancelled.")
                break
            name = f"frame_{frame:05d}"
            target = _bpath(os.path.join(frame_dir, name))
            code = (
                "import bpy\n"
                "sc = bpy.context.scene\n"
                f"sc.frame_set({frame})\n"
                f"sc.render.filepath = {target!r}\n"
                "bpy.ops.render.render(write_still=True)\n"
                "print('ok')\n"
            )
            frame_result = self.run(code, timeout=timeout)
            if frame_result.failed:
                raise BlenderError(f"Frame {frame} failed to render:\n"
                                   + frame_result.summary(1500))
            path = os.path.join(frame_dir, name + ".png")
            if os.path.exists(path):
                paths.append(path)
            if on_frame:
                on_frame(index + 1, total, path if os.path.exists(path) else None)
        return paths

    def encode_video(self, frame_dir, output_path, fps=24, res=(1920, 1080),
                     audio_path=None, audio_offset_frames=0, timeout=1200.0):
        """Turn a PNG sequence into an H.264 .mp4 using Blender's own encoder.

        Done in a throwaway scene so the built scene is left exactly as the
        agent made it.

        ``audio_path`` muxes a soundtrack in during the same encode. Blender
        bundles its own ffmpeg, so this needs no external binary - but the
        audio codec has to be set explicitly or the container comes out silent.
        """
        frames = sorted(f for f in os.listdir(frame_dir) if f.lower().endswith(".png"))
        if not frames:
            raise BlenderError("There are no rendered frames to encode.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        audio_line = ""
        if audio_path and os.path.exists(audio_path):
            audio_line = (
                "sc.render.ffmpeg.audio_codec = 'AAC'\n"
                "sc.render.ffmpeg.audio_bitrate = 192\n"
                f"coll.new_sound(name='dialogue', filepath={_bpath(audio_path)!r},"
                f" channel=2, frame_start=1 + {int(audio_offset_frames)})\n"
            )
        elif audio_path:
            self._log(f"No audio at {audio_path}; encoding the video silent.")

        code = (
            "import bpy, os\n"
            f"src = {_bpath(frame_dir)!r}\n"
            f"out = {_bpath(output_path)!r}\n"
            "files = sorted(f for f in os.listdir(src) if f.lower().endswith('.png'))\n"
            "sc = bpy.data.scenes.new('bstudio_encode')\n"
            f"sc.render.resolution_x = {int(res[0])}\n"
            f"sc.render.resolution_y = {int(res[1])}\n"
            "sc.render.resolution_percentage = 100\n"
            f"sc.render.fps = {int(fps)}\n"
            "sc.frame_start = 1\n"
            "sc.frame_end = len(files)\n"
            # In Blender 5.x FFMPEG only appears once media_type is VIDEO.
            "sc.render.image_settings.media_type = 'VIDEO'\n"
            "sc.render.image_settings.file_format = 'FFMPEG'\n"
            "sc.render.ffmpeg.format = 'MPEG4'\n"
            "sc.render.ffmpeg.codec = 'H264'\n"
            "sc.render.ffmpeg.constant_rate_factor = 'HIGH'\n"
            "sc.render.ffmpeg.ffmpeg_preset = 'GOOD'\n"
            "sc.render.ffmpeg.gopsize = 12\n"
            "sc.render.filepath = out\n"
            "se = sc.sequence_editor_create()\n"
            # An empty bpy_prop_collection is falsy, so `a or b` would discard
            # the real (but empty) collection. Test for None explicitly.
            "coll = getattr(se, 'strips', None)\n"
            "if coll is None:\n"
            "    coll = se.sequences\n"
            "strip = coll.new_image(name='seq', filepath=os.path.join(src, files[0]),"
            " channel=1, frame_start=1)\n"
            "for f in files[1:]:\n"
            "    strip.elements.append(f)\n"
            + audio_line +
            "bpy.ops.render.render(animation=True, scene=sc.name)\n"
            "bpy.data.scenes.remove(sc)\n"
            "print('encoded', os.path.exists(out))\n"
        )
        result = self.run(code, timeout=timeout)
        if result.failed:
            raise BlenderError("Video encoding failed:\n" + result.summary(2000))

        # Blender appends its own frame range to the name for video output.
        if os.path.exists(output_path):
            return output_path
        stem, ext = os.path.splitext(output_path)
        folder = os.path.dirname(output_path) or "."
        base = os.path.basename(stem)
        for candidate in sorted(os.listdir(folder)):
            if candidate.startswith(base) and candidate.lower().endswith(ext.lower()):
                found = os.path.join(folder, candidate)
                if found != output_path:
                    try:
                        os.replace(found, output_path)
                        return output_path
                    except OSError:
                        return found
        raise BlenderError("Encoding finished but no video file was produced.")
