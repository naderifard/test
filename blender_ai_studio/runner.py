"""End-to-end orchestration: prompt in, rendered file out.

Used by both the GUI and the command line, so it takes plain callbacks rather
than knowing anything about Qt.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import traceback

from . import agent, config
from .blender_link import BlenderLink, BlenderUnavailable, launch_blender
from .llm import Chat
from .session import BlenderSession


def slugify(text, limit=40):
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return (slug[:limit].rstrip("-") or "scene")


class RunResult:
    def __init__(self):
        self.output_path = None
        self.preview_path = None
        self.blend_path = None
        self.brief = ""
        self.summary = ""
        self.run_dir = ""
        self.analysis = None
        self.cancelled = False
        self.error = None


class Runner:
    """One scene-generation run."""

    def __init__(self, settings, on_log=None, on_event=None, on_progress=None,
                 should_stop=None):
        self.settings = settings
        self.on_log = on_log
        self.on_event = on_event
        self.on_progress = on_progress
        self.should_stop = should_stop or (lambda: False)
        self.session = None
        self.blender = None

    # -- helpers ------------------------------------------------------------
    def log(self, message):
        if self.on_log:
            self.on_log(str(message))

    def event(self, kind, payload):
        if self.on_event:
            self.on_event(kind, payload)

    def progress(self, fraction, label=""):
        if self.on_progress:
            self.on_progress(fraction, label)

    def _check_stop(self):
        if self.should_stop():
            raise agent.Cancelled("Stopped by the user.")

    # -- blender ------------------------------------------------------------
    def ensure_blender(self):
        """Attach to a running Blender, or start one."""
        port = int(self.settings.get("port", 9876))
        exe = self.settings.get("blender_exe") or None
        self.blender = launch_blender(port=port, executable=exe, on_log=self.log)
        self.session = BlenderSession(link=BlenderLink(port=port), on_log=self.log)
        self.session.install_prelude()
        return self.session

    # -- the run ------------------------------------------------------------
    def run(self, user_prompt, mode="image"):
        result = RunResult()
        mode = "animation" if str(mode).lower().startswith("anim") else "image"
        settings = self.settings

        res = config.parse_resolution(settings.get("resolution", "1920x1080"))
        engine = settings.get("engine", "EEVEE")
        samples = int(settings.get("samples", 64))
        frames = int(settings.get("frames", 96))
        fps = int(settings.get("fps", 24))

        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(config.ensure_output_dir(),
                               f"{stamp}_{slugify(user_prompt)}")
        os.makedirs(run_dir, exist_ok=True)
        result.run_dir = run_dir
        self.log(f"Output folder: {run_dir}")

        try:
            self.progress(0.02, "Starting Blender")
            self.ensure_blender()
            self._check_stop()

            self.progress(0.06, "Preparing the scene")
            self.session.prepare(engine=engine, res=res, fps=fps, samples=samples)

            chat = Chat(
                api_key=config.resolve_api_key(settings.get("api_key")),
                on_log=self.log,
            )

            # -- stage 1: art direction
            self.progress(0.10, "Writing the shot brief")
            self._check_stop()
            brief = agent.write_brief(
                chat, user_prompt, mode, frames=frames, fps=fps,
                thinking=bool(settings.get("director_thinking", True)),
                on_log=self.log,
            )
            result.brief = brief
            self.event("brief", {"text": brief})
            with open(os.path.join(run_dir, "brief.md"), "w", encoding="utf-8") as fh:
                fh.write(f"# Brief\n\n**Prompt:** {user_prompt}\n\n**Mode:** {mode}\n\n{brief}\n")
            self.log("Brief ready.")

            # -- stage 2: build
            self.progress(0.16, "Building the scene")
            self._check_stop()
            toolbox, messages = agent.build_scene(
                chat, self.session, user_prompt, brief, mode, res, engine,
                frames=frames, fps=fps,
                thinking=bool(settings.get("builder_thinking", False)),
                max_steps=int(settings.get("max_steps", 26)),
                on_log=self.log, on_event=self.event,
                should_stop=self.should_stop,
            )
            result.summary = toolbox.finish_summary
            result.analysis = toolbox.last_analysis
            self._save_transcript(run_dir, messages)

            # -- save the project before rendering, so nothing is ever lost
            blend_path = os.path.join(run_dir, "scene.blend")
            try:
                self.session.save_blend(blend_path)
                result.blend_path = blend_path
                self.log(f"Saved {blend_path}")
            except Exception as exc:  # noqa: BLE001
                self.log(f"Could not save the .blend file: {exc}")

            # -- exposure safety net
            # The model cannot see the render, so however carefully it set the
            # lights, the frame can still come out crushed or blown. Correcting
            # in exposure stops is non-destructive: it leaves the lighting the
            # model designed intact and only touches colour management.
            self._check_stop()
            self.progress(0.72, "Balancing the exposure")
            self._auto_exposure()

            # -- stage 3: render
            self._check_stop()
            if mode == "animation":
                result.output_path = self._render_animation(
                    run_dir, res, samples, frames, fps
                )
            else:
                result.output_path = self._render_image(run_dir, res, samples)

            self.progress(1.0, "Done")
            return result

        except agent.Cancelled:
            result.cancelled = True
            self.log("Run cancelled.")
            return result
        except Exception as exc:  # noqa: BLE001 - report, never crash the GUI
            result.error = str(exc)
            self.log(f"ERROR: {exc}")
            self.log(traceback.format_exc(limit=6))
            return result

    def _auto_exposure(self):
        result = self.session.run(
            "import bstudio as B, json\nprint(json.dumps(B.auto_exposure()))",
            timeout=300,
        )
        if result.failed:
            self.log("Exposure balancing skipped (it errored); rendering as built.")
            return None
        try:
            text = result.output
            data = json.loads(text[text.find("{"):text.rfind("}") + 1])
        except (ValueError, IndexError):
            return None
        moved = data.get("exposure", 0.0) - data.get("start_exposure", 0.0)
        if abs(moved) < 0.05:
            self.log("Exposure was already balanced.")
        else:
            self.log(
                f"Exposure adjusted by {moved:+.2f} stops "
                f"(median luminance now {data.get('median_luminance', 0):.2f})."
            )
        return data

    # -- rendering ----------------------------------------------------------
    def _render_image(self, run_dir, res, samples):
        self.progress(0.80, "Rendering the final image")
        self.log(f"Rendering {res[0]}x{res[1]}...")
        target = os.path.join(run_dir, "render.png")
        path = self.session.render_still(target, res=res, samples=samples)
        self.log(f"Rendered {path}")
        self.event("result_image", {"path": path})
        return path

    def _render_animation(self, run_dir, res, samples, frames, fps):
        frame_dir = os.path.join(run_dir, "frames")
        self.log(f"Rendering {frames} frames at {res[0]}x{res[1]}...")

        def on_frame(index, total, path):
            fraction = 0.35 + 0.55 * (index / max(total, 1))
            self.progress(fraction, f"Rendering frame {index}/{total}")
            if path and index % 2 == 1:
                self.event("result_image", {"path": path})

        paths = self.session.render_animation(
            frame_dir, res=res, samples=samples, on_frame=on_frame,
            should_stop=self.should_stop,
        )
        if not paths:
            raise RuntimeError("No frames were rendered.")
        self._check_stop()

        self.progress(0.92, "Encoding the video")
        self.log(f"Encoding {len(paths)} frames to MP4...")
        video = os.path.join(run_dir, "animation.mp4")
        try:
            path = self.session.encode_video(frame_dir, video, fps=fps, res=res)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Video encoding failed ({exc}). The PNG frames are in {frame_dir}.")
            return frame_dir
        self.log(f"Wrote {path}")
        # A still from the middle of the clip makes a better poster frame.
        middle = paths[len(paths) // 2]
        poster = os.path.join(run_dir, "poster.png")
        try:
            shutil.copyfile(middle, poster)
            self.event("result_image", {"path": poster})
        except OSError:
            pass
        return path

    # -- bookkeeping --------------------------------------------------------
    def _save_transcript(self, run_dir, messages):
        try:
            with open(os.path.join(run_dir, "transcript.json"), "w", encoding="utf-8") as fh:
                json.dump(messages, fh, indent=1, default=str)
        except OSError:
            pass
        # The generated Blender code on its own is the useful artefact.
        try:
            chunks = []
            for message in messages:
                for call in message.get("tool_calls") or []:
                    if call["function"]["name"] != "execute_blender_code":
                        continue
                    try:
                        args = json.loads(call["function"]["arguments"])
                    except ValueError:
                        continue
                    chunks.append(
                        f"# --- {args.get('step', 'step')} ---\n{args.get('code', '')}\n"
                    )
            if chunks:
                path = os.path.join(run_dir, "scene_build.py")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(chunks))
        except OSError:
            pass
