"""What every stage's subprocess side looks like.

The graph hands a stage a job file and expects a result file back. This module
is that contract, so a stage module only has to write the interesting part::

    STAGE = StageSpec(name="f_voice", version="1", summary="...", needs=[...])

    def plan(project, graph):        # in the host, decides the units
        ...

    def run(job, unit):             # in the stage's own interpreter
        ...
        return {"metrics": {...}, "notes": [...]}

    if __name__ == "__main__":
        base.stage_main(run)

A unit that raises is recorded as failed and the rest of the units still run:
one bad line of dialogue should not cost the other nine their alignment.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from pipeline import jsonio                                    # noqa: E402
from pipeline.graph import StageSpec, Step, Unit                # noqa: E402,F401
from pipeline.project import Project                            # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--result", required=True)
    return parser.parse_args(argv)


def blender_args():
    """Blender puts the script's own arguments after a bare ``--``."""
    argv = sys.argv
    return parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])


class Job:
    """The job file, with the few things every stage asks it for."""

    def __init__(self, data):
        self.data = data
        self.stage = data["stage"]
        self.version = data.get("version", "1")
        self.project_root = data["project_root"]
        self.repo_root = data.get("repo_root")
        self.style_pack = data.get("style_pack", "miniature")
        self.fps = int(data.get("fps", 24))
        self.resolution = list(data.get("resolution", [1920, 1080]))
        self.units = [dict(u) for u in data.get("units", [])]

    @property
    def project(self):
        return Project(self.project_root)

    def style(self):
        from blender_ai_studio import styles
        return styles.load(self.style_pack)


def load_job(path):
    return Job(jsonio.read(path))


def out_path(unit, *parts):
    """A path inside the unit's output directory, parents created."""
    target = os.path.join(unit["out_dir"], *parts)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return target


def stage_main(*handlers, argv=None, from_blender=False):
    """Run one step of a stage over every unit in its job file.

    ``handlers`` is one callable per step, in order. Each is called as
    ``handler(job, unit)`` and returns a dict with ``metrics`` and ``notes``.
    """
    args = blender_args() if from_blender else parse_args(argv)
    job = load_job(args.job)
    handler = handlers[args.step] if args.step < len(handlers) else handlers[-1]

    records = {}
    for unit in job.units:
        label = unit.get("label") or unit["id"]
        print(f"--- {job.stage} :: {label}", flush=True)
        try:
            payload = handler(job, unit) or {}
            records[unit["id"]] = {
                "metrics": payload.get("metrics") or {},
                "notes": payload.get("notes") or [],
                "error": None,
            }
        except Exception as exc:
            traceback.print_exc()
            records[unit["id"]] = {
                "metrics": {}, "notes": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
    jsonio.write(args.result, {"stage": job.stage, "step": args.step,
                               "units": records})
    # Exit 0 even when a unit failed: the graph reads per-unit errors from the
    # result file and a non-zero exit would hide the ones that succeeded.
    return 0
