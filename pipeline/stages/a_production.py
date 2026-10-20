"""Stage A — validate the hand-written production plan and resolve it once.

Stage A itself is a person and a chat model (§2, §7-A). What runs here is the
half that has to be mechanical: check the JSON against the schema, check that
every cross-reference resolves, check that every line of dialogue actually has a
recording, and then write ``production.resolved.json``.

The resolved document is the point. Twelve later stages need to know Rostam's rig
mode, and §6.2 says that comes from the override table, then the character's
``rig_default``, then its ``kind``. Resolving that in each stage means twelve
chances to disagree. Resolving it once, here, and hashing the result means a
change to the override table re-keys exactly the units that depended on it.

Nothing downstream reads ``production.json``.
"""

from __future__ import annotations

import os
import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio, validate

# The project settings that change what gets built. Deliberately not the whole
# config: `title` is in there, and a retitled film should not rebuild the plan.
SETTINGS = ("fps", "resolution", "language", "head_handle_s", "tail_handle_s",
            "min_shot_s", "aligner_backend", "silence_lead_policy")


def plan(project, graph):
    config = project.config
    if not os.path.exists(project.production_path):
        return []
    return [Unit(
        "production",
        inputs={"production": project.production_path,
                "rig_overrides": project.rig_overrides_path,
                "settings": {k: config.get(k) for k in SETTINGS}},
        params={},
        out_rel="story",
        label="production plan",
    )]


def nothing_to_do(project):
    return (f"no {project.rel(project.production_path)} yet — run "
            f"`python -m pipeline seed {project.name}`, paste the reply")


def run(job, unit):
    project = job.project
    production = project.production
    overrides = project.rig_overrides
    config = project.config

    report = validate.check_production(production)
    inputs = validate.check_project_inputs(project)
    problems = list(report.problems) + list(inputs.problems)

    resolved = dict(production)
    resolved["schema_version"] = 1
    resolved["fps"] = int(production.get("fps") or config["fps"])
    resolved["settings"] = {k: config.get(k) for k in SETTINGS}

    rigs = {}
    characters = []
    for character in production.get("characters") or []:
        entry = dict(character)
        mode, source = validate.resolve_rig_mode(character, overrides)
        entry["rig_mode"] = mode
        entry["rig_source"] = source
        rigs[character["id"]] = mode
        characters.append(entry)
    resolved["characters"] = characters

    # A flat index of every line, in film order, so g_align and h_assemble do
    # not each re-walk the shot list to find where a recording lives.
    index = []
    for order, shot in enumerate(production.get("shots") or []):
        for position, line in enumerate(shot.get("dialogue") or []):
            source = project.path("audio", line["line_id"], "source.wav")
            index.append({
                "line_id": line["line_id"],
                "character": line["character"],
                "shot": shot["id"],
                "shot_order": order,
                "position": position,
                "emotion": line.get("emotion", "neutral"),
                "text_fa": line.get("text_fa", ""),
                "starts_at_s": line.get("starts_at_s"),
                "source_wav": project.rel(source),
                "recorded": os.path.exists(source),
            })
    resolved["dialogue_index"] = index
    resolved["rig_modes"] = rigs

    jsonio.write(_base.out_path(unit, "production.resolved.json"), resolved)
    jsonio.write(_base.out_path(unit, "validate.json"), {
        "errors": [p.as_dict() for p in problems if p.severity == "error"],
        "warnings": [p.as_dict() for p in problems if p.severity != "error"],
    })

    kinds = {}
    for character in characters:
        kinds[character["kind"]] = kinds.get(character["kind"], 0) + 1
    return {
        "metrics": {
            "characters": len(characters),
            "envs": len(production.get("envs") or []),
            "props": len(production.get("props") or []),
            "shots": len(production.get("shots") or []),
            "lines": len(index),
            "unrecorded": sum(1 for e in index if not e["recorded"]),
            "errors": sum(1 for p in problems if p.severity == "error"),
            "warnings": sum(1 for p in problems if p.severity != "error"),
            "kinds": kinds,
            "rig_modes": rigs,
        },
        "notes": [f"{p.severity}: {p}" for p in problems[:20]],
    }


def _gate(project, unit, metrics):
    out = []
    errors = metrics.get("errors", 0)
    if errors:
        out.append(gates.Verdict(gates.FAIL, "production.schema",
                                 f"{errors} validation error(s) — see "
                                 "story/validate.json", errors, 0))
    unrecorded = metrics.get("unrecorded", 0)
    if unrecorded:
        out.append(gates.Verdict(gates.FAIL, "production.audio",
                                 f"{unrecorded} line(s) have no recording; the "
                                 "audio is the master clock", unrecorded, 0))
    warnings = metrics.get("warnings", 0)
    if warnings:
        out.append(gates.Verdict(gates.WARN, "production.inputs",
                                 f"{warnings} input warning(s)", warnings, 0))
    return out or [gates.Verdict(
        gates.PASS, "production",
        f"{metrics.get('characters', 0)} character(s), "
        f"{metrics.get('shots', 0)} shot(s), {metrics.get('lines', 0)} line(s)")]


STAGE = StageSpec(
    name="a_production",
    version="1",
    summary="validate production.json and resolve rig modes and the line index",
    steps=[Step("host")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(run))
