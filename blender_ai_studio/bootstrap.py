"""Runs INSIDE Blender at startup (blender.exe --python bootstrap.py).

Does not modify the BlenderMCP addon; it only enables it, turns on the
Poly Haven integration (which the addon leaves off by default) and makes sure
the socket server is listening on the port we asked for.
"""

import os
import sys

import bpy

PORT = int(os.environ.get("BSTUDIO_PORT", "9876"))
READY_FILE = os.environ.get("BSTUDIO_READY_FILE", "")
ADDON = "blender_mcp"


def _log(msg):
    print(f"[bstudio-bootstrap] {msg}", flush=True)


def _enable_addon():
    try:
        if ADDON not in bpy.context.preferences.addons:
            bpy.ops.preferences.addon_enable(module=ADDON)
            _log(f"enabled addon {ADDON}")
        else:
            _log(f"addon {ADDON} already enabled")
        return True
    except Exception as exc:  # pragma: no cover - depends on user install
        _log(f"could not enable {ADDON}: {exc}")
        return False


def _configure_scene():
    scene = bpy.context.scene
    # Poly Haven ships disabled; the agent needs it for real PBR textures/HDRIs.
    for prop, value in (
        ("blendermcp_use_polyhaven", True),
        ("blendermcp_port", PORT),
        ("blendermcp_auto_start_server", True),
    ):
        if hasattr(scene, prop):
            try:
                setattr(scene, prop, value)
            except Exception as exc:
                _log(f"could not set {prop}: {exc}")


def _start_server():
    server = getattr(bpy.types, "blendermcp_server", None)
    if server is not None and getattr(server, "running", False):
        if getattr(server, "port", PORT) == PORT:
            _log(f"server already running on {PORT}")
            return True
        try:
            bpy.ops.blendermcp.stop_server()
        except Exception:
            pass
    try:
        bpy.ops.blendermcp.start_server()
        _log(f"server started on port {PORT}")
        return True
    except Exception as exc:
        _log(f"could not start server: {exc}")
        return False


def _prepare_viewport():
    """Material-preview shading so the human watching sees textures live."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            space.shading.type = "MATERIAL"
            space.overlay.show_overlays = True
            space.show_gizmo = True
            try:
                space.clip_end = 10000.0
            except Exception:
                pass
    _log("viewport set to material preview")


def _deferred():
    """Run once Blender's UI is fully up (registered as a one-shot timer)."""
    _enable_addon()
    _configure_scene()
    _start_server()
    try:
        _prepare_viewport()
    except Exception as exc:
        _log(f"viewport setup skipped: {exc}")
    if READY_FILE:
        try:
            with open(READY_FILE, "w", encoding="utf-8") as fh:
                fh.write(str(PORT))
        except Exception as exc:
            _log(f"could not write ready file: {exc}")
    _log("ready")
    return None


if bpy.app.background:
    _log("background mode - BlenderMCP cannot serve; bootstrap skipped")
else:
    bpy.app.timers.register(_deferred, first_interval=0.5)
