"""c_paint, in Blender: the renders Wan paints over, and the painted result.

Step 0 (``prepare``) renders, for each character, the depth turntable Wan
follows (``turn/``), a front view at twice the size for the projection's front
layer (``front/``), and the texture space the painting is baked into
(``bake/``). Step 2 (``apply``) makes the painted texture the character's
colour and saves ``painted.blend``. Step 1 is the motion env's (Wan).

Each piece runs as its own Blender process with the scripts that also serve
as tools on their own (``paint_turntable.py``, ``paint_bake.py``): every one of
them opens the character file fresh, and a process each keeps one's changes to
the materials from leaking into the next.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from pipeline import jsonio                                        # noqa: E402
from pipeline.stages import _base                                  # noqa: E402

import bpy                                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _blender(script, *args, marker, env=None):
    """Run ``script`` in its own Blender; it must print ``marker`` (each
    script's last line) or the step fails with the end of its log."""
    cmd = [bpy.app.binary_path, "-b", "--factory-startup", "-P",
           os.path.join(HERE, script), "--"] + [str(a) for a in args]
    full = dict(os.environ)
    full.update(env or {})
    done = subprocess.run(cmd, env=full, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if done.returncode != 0 or marker not in done.stdout:
        tail = "\n".join((done.stdout + done.stderr).splitlines()[-25:])
        raise RuntimeError(f"{script} failed:\n{tail}")
    return done.stdout


def prepare(job, unit):
    params = unit["params"]
    out = unit["out_dir"]
    rigged = unit["inputs"]["rigged"]
    if not params.get("paint"):
        return {"metrics": {"painted": False}, "notes": [params.get("why")]}
    _blender("paint_turntable.py", rigged, os.path.join(out, "turn"),
             params["frames"], params["width"], params["height"],
             marker="PAINT_TURNTABLE done")
    _blender("paint_turntable.py", rigged, os.path.join(out, "front"), 2,
             params["width"], params["height"], marker="PAINT_TURNTABLE done",
             env={"PAINT_FRONT_SCALE": str(params["front_scale"])})
    log = _blender("paint_bake.py", "bake", rigged, os.path.join(out, "bake"),
                   params["texture_size"], marker="PAINT_BAKE done")
    overlap = [line for line in log.splitlines()
               if line.startswith("PAINT_UV overlap")]
    left = int(overlap[-1].split()[-1]) if overlap else None
    return {"metrics": {"uv_overlap_triangles": left}}


def apply(job, unit):
    params = unit["params"]
    out = unit["out_dir"]
    target = _base.out_path(unit, "painted.blend")
    report = jsonio.read(os.path.join(out, "paint.json"), default={}) or {}
    if not params.get("paint") or not report.get("texture"):
        # Nothing painted (no photo, or no video model on this machine): the
        # rigged character goes on as it is, so the shot builder has one
        # path to read either way.
        shutil.copyfile(unit["inputs"]["rigged"], target)
        return {"metrics": {"painted": False},
                "notes": [params.get("why") or report.get("why")]}
    _blender("paint_bake.py", "apply",
             os.path.join(out, "bake", "paint_source.blend"),
             os.path.join(out, report["texture"]), target,
             marker="PAINT_APPLY done")
    return {"metrics": dict(report.get("metrics") or {}, painted=True)}


if __name__ == "__main__":
    sys.exit(_base.stage_main(prepare, prepare, apply, from_blender=True))
