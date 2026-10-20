"""The stage graph: dependency order, caching, subprocess isolation, gates.

A stage is a standalone CLI with declared inputs and outputs. It reads files,
writes files, and exits (§5.1). This module is what walks them.

Three ideas carry the design.

**Units, not stages, are the cache grain.** Rostam's body is one unit of
``c_body``; his line is one unit of ``g_align``. Caching per stage would rebuild
every character because one changed; caching per unit is what makes "Rostam's
body is built once and reused by every shot" true.

**One subprocess per stage, not per unit.** §10 rule 1 wants the process to exit
so VRAM is freed, but loading a 1.2 GB aligner once per *line* would be absurd.
So all of a stage's pending units go into one job file, the process does them
all, and then it exits.

**Steps, so one stage can span two interpreters.** ``h_assemble`` resolves the
shot in the host (where the schema validator lives) and then builds the .blend
in Blender. Both steps write into the same unit directory and the store commits
the result once, so the fourteen stages stay fourteen.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time

from . import envs, gates, jsonio
from .project import Project

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The fourteen stages of §12, in the only order their dependencies allow.
STAGE_ORDER = (
    "a_production", "b_refs",
    # c_asset turns the reference image into geometry; the fitters that
    # follow measure and dress it. It sits before c_face because a face
    # reference is the input to both and the mesh is the expensive part.
    "c_asset",
    "c_face", "c_body", "c_texture", "c_garment", "c_rigged", "c_paint",
    "d_env", "e_motion", "f_voice", "g_align",
    "h_assemble", "j_render", "j_mouth", "k_edit",
)


class Step:
    """One interpreter's worth of work inside a stage."""

    def __init__(self, env, script=None):
        self.env = env
        self.script = script  # only for blender steps

    @property
    def is_blender(self):
        return self.env == "blender"


class StageSpec:
    """What a stage declares about itself."""

    def __init__(self, name, version, summary, needs=(), steps=None,
                 gate=None, optional=False, implemented=True):
        self.name = name
        self.version = str(version)
        self.summary = summary
        self.needs = tuple(needs)
        # ``None`` means "the default, one host step"; an empty list means
        # "nothing to run", which is what an unbuilt stage declares. `or`
        # would collapse the two, since an empty list is falsy.
        self.steps = [Step("host")] if steps is None else list(steps)
        self.gate = gate
        self.optional = optional
        self.implemented = implemented

    def __repr__(self):
        return f"<StageSpec {self.name} v{self.version}>"


NOT_BUILT_YET = "not built yet"


def unbuilt_spec(name):
    """The stand-in for a stage that ``STAGE_ORDER`` names and nobody has written.

    ``STAGE_ORDER`` is the declared shape of the pipeline, and it is worth more
    as a fourteen-row plan with five rows marked than as a list edited down to
    whatever happens to exist this week. So a missing module becomes one of
    these rather than an ImportError, ``status`` renders it as a progress board,
    and a bounded run stays exactly as strict as it was.

    A stage that is missing is *skipped*, never faked. Nothing downstream can
    mistake it for done: its outputs are simply absent, and the first stage that
    needs them fails on its own missing inputs with a path in the message.
    """
    return StageSpec(name, version="0", summary=NOT_BUILT_YET,
                     steps=[], implemented=False)


class Unit:
    """One cacheable piece of work: one character, one line, one shot."""

    def __init__(self, uid, inputs=None, params=None, out_rel=None, label=None):
        self.uid = uid
        self.inputs = dict(inputs or {})
        self.params = dict(params or {})
        # Where the outputs are published in the readable project tree.
        self.out_rel = out_rel
        self.label = label or uid

    def as_dict(self):
        return {"id": self.uid, "inputs": self.inputs, "params": self.params,
                "out_rel": self.out_rel, "label": self.label}


class StageError(RuntimeError):
    pass


def load_stage(name, missing_ok=False):
    """Import a stage module. ``None`` when it is not written yet.

    The two ``ModuleNotFoundError`` cases are not the same failure and must not
    be reported as one: ``pipeline.stages.g_align`` missing means the stage is
    unwritten, while ``whisperx`` missing means a written stage cannot run in
    this interpreter. ``exc.name`` is what separates them, and conflating the
    two would let a broken environment read as an unbuilt pipeline.
    """
    try:
        module = importlib.import_module(f"pipeline.stages.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"pipeline.stages.{name}":
            raise StageError(
                f"{name}: needs {exc.name!r}, which this interpreter has not "
                f"got ({exc})") from exc
        if missing_ok:
            return None
        raise StageError(
            f"stage {name!r} is declared in STAGE_ORDER but not implemented "
            f"yet — expected pipeline/stages/{name}.py") from exc
    if not hasattr(module, "STAGE"):
        raise StageError(f"pipeline.stages.{name} declares no STAGE")
    return module


def implemented(name):
    """Whether a declared stage has a module behind it."""
    return importlib.util.find_spec(f"pipeline.stages.{name}") is not None


def order(only=None, start=None, stop=None):
    """The stages to walk, in dependency order."""
    names = list(STAGE_ORDER)
    if only:
        wanted = set(only)
        unknown = wanted - set(names)
        if unknown:
            raise StageError(f"unknown stages: {sorted(unknown)}")
        return [n for n in names if n in wanted]
    if start:
        if start not in names:
            raise StageError(f"unknown stage {start!r}")
        names = names[names.index(start):]
    if stop:
        if stop not in names:
            raise StageError(f"unknown stage {stop!r}")
        names = names[:names.index(stop) + 1]
    return names


class UnitResult:
    def __init__(self, unit, status, key=None, metrics=None, verdicts=(),
                 notes=(), error=None, seconds=0.0):
        self.unit = unit
        self.status = status         # cached | built | skipped | failed
        self.key = key
        self.metrics = dict(metrics or {})
        self.verdicts = list(verdicts)
        self.notes = list(notes)
        self.error = error
        self.seconds = seconds

    @property
    def level(self):
        return gates.worst(self.verdicts) if self.verdicts else gates.PASS

    def as_dict(self):
        return {
            "unit": self.unit.uid, "status": self.status, "key": self.key,
            "metrics": self.metrics, "notes": self.notes, "error": self.error,
            "seconds": round(self.seconds, 2),
            "verdicts": [v.as_dict() for v in self.verdicts],
        }


class StageResult:
    def __init__(self, name, units=None, skipped=None, seconds=0.0):
        self.name = name
        self.units = list(units or [])
        self.skipped = skipped        # a reason string when the stage had no work
        self.seconds = seconds

    @property
    def level(self):
        return gates.worst([v for u in self.units for v in u.verdicts]) \
            if any(u.verdicts for u in self.units) else gates.PASS

    @property
    def failed(self):
        return any(u.status == "failed" for u in self.units) \
            or self.level == gates.FAIL

    def as_dict(self):
        return {"stage": self.name, "skipped": self.skipped,
                "seconds": round(self.seconds, 2),
                "units": [u.as_dict() for u in self.units]}


class Graph:
    """Walks the stages for one project."""

    def __init__(self, project, on_log=None, style_pack=None, force=(),
                 dry_run=False, resolution_scale=None):
        self.project = project
        self.on_log = on_log or (lambda message: None)
        self.style_pack = style_pack or project.config["style_pack"]
        self.force = set(force or ())
        self.dry_run = dry_run
        # A draft render, as a fraction of the project's resolution. None means
        # "whatever the style pack asks for". It travels into the unit's params
        # rather than being applied behind the store's back, so a draft can
        # never be served from the cache as a final.
        self.resolution_scale = resolution_scale
        self.results = []

    def log(self, message):
        self.on_log(message)

    # -- planning -----------------------------------------------------------
    def plan(self, stage_names):
        """What each stage would do, without doing any of it."""
        plans = []
        for name in stage_names:
            module = load_stage(name, missing_ok=True)
            if module is None:
                plans.append((unbuilt_spec(name), []))
                continue
            spec = module.STAGE
            try:
                units = list(module.plan(self.project, self) or [])
            except Exception as exc:                    # a planning bug is a bug
                raise StageError(f"{name}: planning failed: {exc}") from exc
            plans.append((spec, units))
        return plans

    # -- running ------------------------------------------------------------
    def run(self, stage_names=None, stop_on_fail=True):
        stage_names = stage_names or list(STAGE_ORDER)
        self.results = []
        for name in stage_names:
            result = self.run_stage(name)
            self.results.append(result)
            if result.failed and stop_on_fail:
                self.log(f"  stopping: {name} failed its gate")
                break
        self.project.store.flush()
        self._write_summary()
        return self.results

    def run_stage(self, name):
        started = time.time()
        module = load_stage(name, missing_ok=True)
        if module is None:
            self.log(f"{name:<12} skipped — {NOT_BUILT_YET}")
            return StageResult(name, skipped=NOT_BUILT_YET,
                               seconds=time.time() - started)
        spec = module.STAGE
        units = list(module.plan(self.project, self) or [])
        if not units:
            reason = getattr(module, "nothing_to_do", lambda p: "no units")(self.project)
            self.log(f"{name:<12} skipped — {reason}")
            return StageResult(name, skipped=reason,
                              seconds=time.time() - started)

        store = self.project.store
        keyed = []
        for unit in units:
            key, resolved = store.key(name, spec.version, unit.inputs, unit.params)
            keyed.append((unit, key, resolved))

        forced = name in self.force or "all" in self.force
        pending = [(u, k, r) for u, k, r in keyed
                   if forced or not store.has(name, k)]
        cached = [(u, k, r) for u, k, r in keyed if (u, k, r) not in pending]

        results = []
        for unit, key, _ in cached:
            self._materialise(name, key, unit)
            meta = store.meta(name, key)
            results.append(UnitResult(unit, "cached", key,
                                      metrics=meta.get("metrics"),
                                      verdicts=_rehydrate(meta.get("verdicts")),
                                      notes=meta.get("notes")))

        if pending and self.dry_run:
            for unit, key, _ in pending:
                results.append(UnitResult(unit, "skipped", key,
                                          notes=["dry run"]))
        elif pending:
            results.extend(self._build(spec, module, pending))

        elapsed = time.time() - started
        self._report(name, results, elapsed)
        return StageResult(name, results, seconds=elapsed)

    # -- the subprocess dance -----------------------------------------------
    def _build(self, spec, module, pending):
        name = spec.name
        store = self.project.store
        workspace = tempfile.mkdtemp(prefix=f"{name}-",
                                     dir=_ensure(self.project.cache("work")))
        results = []
        try:
            job_units = []
            for unit, key, _resolved in pending:
                out_dir = os.path.join(workspace, unit.uid)
                os.makedirs(out_dir, exist_ok=True)
                entry = unit.as_dict()
                entry["out_dir"] = out_dir.replace("\\", "/")
                entry["key"] = key
                job_units.append(entry)
            job = {
                "stage": name,
                "version": spec.version,
                "project_root": self.project.root.replace("\\", "/"),
                "repo_root": REPO_ROOT.replace("\\", "/"),
                "style_pack": self.style_pack,
                "fps": self.project.fps(),
                "resolution": self.project.config["resolution"],
                "units": job_units,
            }
            job_path = os.path.join(workspace, "job.json")
            jsonio.write(job_path, job)

            merged = {u["id"]: {"metrics": {}, "notes": [], "ok": True,
                                "error": None} for u in job_units}
            for index, step in enumerate(spec.steps):
                self.log(f"{name:<12} step {index + 1}/{len(spec.steps)} "
                         f"in {step.env} — {len(job_units)} unit(s)")
                payload = self._run_step(spec, step, index, job_path, workspace)
                for uid, record in payload.get("units", {}).items():
                    slot = merged.setdefault(uid, {"metrics": {}, "notes": [],
                                                   "ok": True, "error": None})
                    slot["metrics"].update(record.get("metrics") or {})
                    slot["notes"].extend(record.get("notes") or [])
                    if record.get("error"):
                        slot["ok"] = False
                        slot["error"] = record["error"]

            for unit, key, resolved in pending:
                results.append(self._finish(spec, module, unit, key, resolved,
                                            workspace, merged.get(unit.uid, {})))
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
        store.flush()
        return results

    def _run_step(self, spec, step, index, job_path, workspace):
        result_path = os.path.join(workspace, f"result.step{index}.json")
        if step.is_blender:
            exe = envs.resolve("blender")
            script = os.path.join(REPO_ROOT, step.script.replace("/", os.sep))
            if not os.path.exists(script):
                raise StageError(f"{spec.name}: no such Blender script {script}")
            command = [exe, "--background", "--factory-startup",
                       "--python", script, "--",
                       "--job", job_path, "--step", str(index),
                       "--result", result_path]
        else:
            exe = envs.resolve(step.env)
            command = [exe, "-m", f"pipeline.stages.{spec.name}",
                       "--job", job_path, "--step", str(index),
                       "--result", result_path]

        environment = dict(os.environ)
        # The repo first, then whatever vendored source this env needs
        # (SF3D is a checkout, not an installed package), then anything
        # the caller already had set.
        extra, vendor = envs.step_env(step.env)
        previous = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            [REPO_ROOT] + vendor + ([previous] if previous else []))
        environment.update(extra)
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"

        log_path = os.path.join(_ensure(self.project.store.logs),
                               f"{spec.name}.step{index}.log")
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            log.write(" ".join(command) + "\n\n")
            log.flush()
            done = subprocess.run(command, cwd=REPO_ROOT, env=environment,
                                  stdout=log, stderr=subprocess.STDOUT)
        if done.returncode != 0:
            raise StageError(
                f"{spec.name} step {index + 1} ({step.env}) exited "
                f"{done.returncode}. Log: {log_path}\n" + _tail(log_path))
        payload = jsonio.read(result_path, default=None)
        if payload is None:
            raise StageError(
                f"{spec.name} step {index + 1} wrote no result file. "
                f"Log: {log_path}\n" + _tail(log_path))
        return payload

    def _finish(self, spec, module, unit, key, resolved, workspace, record):
        """Gate a finished unit, then commit and publish it."""
        out_dir = os.path.join(workspace, unit.uid)
        metrics = record.get("metrics") or {}
        notes = record.get("notes") or []
        if record.get("error"):
            return UnitResult(unit, "failed", key, metrics=metrics, notes=notes,
                              error=record["error"])

        produced = _list_files(out_dir)
        if not produced:
            return UnitResult(unit, "failed", key, metrics=metrics, notes=notes,
                              error="the stage wrote no files")

        gate = spec.gate or getattr(module, "gate", None)
        verdicts = list(gate(self.project, unit, metrics) or []) if gate else []
        level = gates.worst(verdicts) if verdicts else gates.PASS
        if level == gates.FAIL:
            # A failed unit is not committed: caching a bad artifact would make
            # the failure sticky across runs.
            return UnitResult(unit, "failed", key, metrics=metrics,
                              verdicts=verdicts, notes=notes,
                              error="; ".join(v.message for v in verdicts
                                              if v.level == gates.FAIL))

        self.project.store.commit(spec.name, key, out_dir, {
            "version": spec.version,
            "unit": unit.uid,
            "label": unit.label,
            "inputs": resolved,
            "input_paths": {k: _relative(self.project, v)
                            for k, v in unit.inputs.items()},
            "params": unit.params,
            "out_rel": unit.out_rel,
            "metrics": metrics,
            "notes": notes,
            "verdicts": [v.as_dict() for v in verdicts],
            "style_pack": self.style_pack,
        })
        self._materialise(spec.name, key, unit)
        return UnitResult(unit, "built", key, metrics=metrics,
                          verdicts=verdicts, notes=notes)

    def _materialise(self, stage, key, unit):
        if not unit.out_rel:
            return []
        dest = self.project.path(unit.out_rel)
        os.makedirs(dest, exist_ok=True)
        return self.project.store.materialise(stage, key, dest)

    # -- reporting ----------------------------------------------------------
    def _report(self, name, results, elapsed):
        built = sum(1 for r in results if r.status == "built")
        cached = sum(1 for r in results if r.status == "cached")
        failed = [r for r in results if r.status == "failed"]
        parts = []
        if built:
            parts.append(f"{built} built")
        if cached:
            parts.append(f"{cached} cached")
        if failed:
            parts.append(f"{len(failed)} FAILED")
        self.log(f"{name:<12} {', '.join(parts) or 'nothing'}  ({elapsed:.1f}s)")
        for result in results:
            for verdict in result.verdicts:
                if verdict.level != gates.PASS:
                    self.log(f"             {result.unit.uid}: {verdict}")
                elif result.status == "built":
                    self.log(f"             {result.unit.uid}: {verdict.message}")
            if result.error:
                self.log(f"             {result.unit.uid}: {result.error}")

    def _write_summary(self):
        path = self.project.cache("last_run.json")
        jsonio.write(path, {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "style_pack": self.style_pack,
            "stages": [r.as_dict() for r in self.results],
        })
        return path


def _rehydrate(records):
    return [gates.Verdict(r.get("level", gates.PASS), r.get("name", "?"),
                          r.get("message", ""), r.get("value"), r.get("threshold"))
            for r in (records or [])]


def _relative(project, value):
    if isinstance(value, str) and os.path.isabs(value):
        try:
            return project.rel(value)
        except ValueError:
            return value
    return value


def _list_files(folder):
    found = []
    for base, dirs, files in os.walk(folder):
        dirs.sort()
        found.extend(os.path.join(base, name) for name in sorted(files))
    return found


def _ensure(folder):
    os.makedirs(folder, exist_ok=True)
    return folder


def _tail(path, lines=30):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read().splitlines()
    except OSError:
        return ""
    return "\n".join("  | " + line for line in content[-lines:])
