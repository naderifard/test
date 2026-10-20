"""The tool surface exposed to the scene-building model.

Deliberately small. A 120B open model degrades quickly when given twenty
overlapping tools, so there is exactly one way to change the scene
(``execute_blender_code``), two ways to observe it, two asset helpers, and a
way to stop.
"""

from __future__ import annotations

import json

from .blender_link import BlenderError

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "execute_blender_code",
            "description": (
                "Run a Python snippet inside Blender to build part of the scene. "
                "Each call starts with a fresh namespace, so begin with "
                "'import bstudio as B'. Returns whatever the snippet printed, "
                "or a full traceback if it raised."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "step": {
                        "type": "string",
                        "description": "Short label for this build step, e.g. 'ground and rocks'.",
                    },
                    "code": {
                        "type": "string",
                        "description": "The Python source to execute in Blender.",
                    },
                },
                "required": ["step", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_scene",
            "description": (
                "Read back what currently exists: engine, resolution, frame range, "
                "camera, lights, and every object with its position and materials."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "verbose": {
                        "type": "boolean",
                        "description": "List every object instead of the first 25.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyse_frame",
            "description": (
                "Render a small thumbnail through the scene camera and measure it. "
                "Returns brightness, contrast, which objects are inside the frame "
                "and how much of it they fill, plus hints about what to fix. This "
                "is the only way to find out what the camera actually sees."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_textures",
            "description": (
                "Search Poly Haven for free PBR textures or HDRI environments to "
                "use instead of procedural materials. Returns matching asset ids."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_type": {
                        "type": "string",
                        "enum": ["textures", "hdris"],
                        "description": "What kind of asset to look for.",
                    },
                    "categories": {
                        "type": "string",
                        "description": (
                            "Comma-separated category filter, e.g. 'wood', 'metal', "
                            "'brick', 'outdoor', 'sunset'."
                        ),
                    },
                },
                "required": ["asset_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_texture",
            "description": (
                "Download a Poly Haven asset and apply it. For a texture, pass the "
                "object to put it on; the object is UV unwrapped first. For an "
                "HDRI, omit object_name and it becomes the world environment light."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Asset id from search_textures.",
                    },
                    "asset_type": {
                        "type": "string",
                        "enum": ["textures", "hdris"],
                    },
                    "object_name": {
                        "type": "string",
                        "description": "Object to texture. Leave empty for HDRIs.",
                    },
                    "resolution": {
                        "type": "string",
                        "enum": ["1k", "2k", "4k"],
                        "description": "Texture resolution. 2k is a good default.",
                    },
                },
                "required": ["asset_id", "asset_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Declare the scene complete. Only call this once the scene matches "
                "the brief and analyse_frame shows the subject framed and lit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One paragraph describing what you built.",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]


class ToolBox:
    """Executes tool calls against a live Blender session."""

    def __init__(self, session, on_log=None, on_event=None):
        self.session = session
        self.on_log = on_log
        self.on_event = on_event
        self.finished = False
        self.finish_summary = ""
        self.code_calls = 0
        self.failed_calls = 0
        self.last_analysis = None

    def _log(self, message):
        if self.on_log:
            self.on_log(message)

    def _event(self, kind, payload):
        if self.on_event:
            self.on_event(kind, payload)

    # -- dispatch -----------------------------------------------------------
    def call(self, name, arguments):
        handler = getattr(self, "_tool_" + name, None)
        if handler is None:
            return f"There is no tool called {name!r}. Available: " + ", ".join(
                schema["function"]["name"] for schema in TOOL_SCHEMAS
            )
        try:
            return handler(**arguments)
        except TypeError as exc:
            return f"Wrong arguments for {name}: {exc}"
        except BlenderError as exc:
            return f"Blender rejected the command: {exc}"
        except Exception as exc:  # noqa: BLE001 - surface anything to the model
            return f"The tool failed: {type(exc).__name__}: {exc}"

    # -- tools --------------------------------------------------------------
    def _tool_execute_blender_code(self, code, step=""):
        self.code_calls += 1
        label = step or f"step {self.code_calls}"
        self._log(f"Building: {label}")
        self._event("step", {"label": label, "code": code})
        result = self.session.run(code, timeout=420)
        if result.failed:
            self.failed_calls += 1
            self._log(f"  ! {label} raised an error; the model will see the traceback.")
            return (
                "The snippet raised an exception. Fix the specific line named in "
                "the traceback and try again.\n\n" + result.summary(2500)
            )
        body = result.summary(2500)
        return f"Executed. Output:\n{body}" if body else "Executed with no output."

    def _tool_inspect_scene(self, verbose=False):
        return self.session.report(verbose=verbose)

    def _tool_analyse_frame(self):
        self._log("Analysing what the camera sees...")
        analysis = self.session.analyse()
        self.last_analysis = analysis
        self._event("analysis", analysis)
        return json.dumps(analysis, indent=1)

    def _tool_search_textures(self, asset_type, categories=""):
        params = {"asset_type": asset_type}
        if categories:
            params["categories"] = categories
        try:
            result = self.session.link.send("search_polyhaven_assets", params, timeout=90)
        except BlenderError as exc:
            return (
                f"Poly Haven search is unavailable ({exc}). Use B.procedural(...) "
                "materials instead - they need no download."
            )
        assets = (result or {}).get("assets", {})
        if not assets:
            return (
                "No assets matched. Try a broader category, or use "
                "B.procedural(...) instead."
            )
        lines = []
        for asset_id, meta in list(assets.items())[:14]:
            name = meta.get("name", asset_id)
            cats = ", ".join(meta.get("categories", [])[:3])
            lines.append(f"{asset_id}  -  {name} [{cats}]")
        return (
            f"{len(assets)} matches (showing {len(lines)}). Use the id on the left "
            "with apply_texture:\n" + "\n".join(lines)
        )

    def _tool_apply_texture(self, asset_id, asset_type, object_name="", resolution="2k"):
        self._log(f"Downloading Poly Haven asset '{asset_id}' ({resolution})...")
        try:
            self.session.link.send(
                "download_polyhaven_asset",
                {"asset_id": asset_id, "asset_type": asset_type, "resolution": resolution},
                timeout=300,
            )
        except BlenderError as exc:
            return (
                f"Download failed: {exc}. Fall back to B.procedural(...) for this "
                "surface."
            )
        if asset_type == "hdris":
            return (
                f"HDRI '{asset_id}' is now lighting the scene. Call analyse_frame "
                "to check the exposure."
            )
        if not object_name:
            return (
                "The texture downloaded but no object_name was given, so it was "
                "not applied. Call apply_texture again with object_name set."
            )
        unwrap = self.session.run(
            f"import bstudio as B\nB.unwrap({object_name!r})\nprint('unwrapped')",
            timeout=180,
        )
        if unwrap.failed:
            return (
                f"Could not UV unwrap {object_name!r}, so the texture would look "
                f"wrong:\n{unwrap.summary(900)}"
            )
        try:
            self.session.link.send(
                "set_texture",
                {"object_name": object_name, "texture_id": asset_id},
                timeout=180,
            )
        except BlenderError as exc:
            return f"Could not apply the texture to {object_name!r}: {exc}"
        return f"Texture '{asset_id}' applied to {object_name!r} at {resolution}."

    def _tool_finish(self, summary=""):
        self.finished = True
        self.finish_summary = summary
        return "Acknowledged."
