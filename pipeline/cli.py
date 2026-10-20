"""``python -m pipeline`` — the command line over the stage graph."""

from __future__ import annotations

import argparse
import os
import sys

from blender_ai_studio import console

from . import envs, gates, graph, jsonio, validate
from .project import Project

STYLE_PACKS = ("miniature", "stylized", "photoreal")


def _log(message):
    print(console.safe(message), flush=True)


def cmd_init(args):
    resolution = None
    if args.resolution:
        width, _, height = args.resolution.lower().partition("x")
        resolution = [int(width), int(height)]
    project = Project.create(args.name, title=args.title, fps=args.fps,
                             resolution=resolution, style_pack=args.style)
    _log(f"Created {project.root}")
    for folder in sorted(os.listdir(project.root)):
        _log("  " + folder)
    _log("\nNext: write story/story.md, then paste the strong model's reply into")
    _log("story/production.json and run:  python -m pipeline validate " + args.name)
    return 0


def cmd_list(args):
    names = Project.list_all()
    if not names:
        _log("No projects yet. Create one with:  python -m pipeline init <name>")
        return 0
    for name in names:
        project = Project.open(name)
        config = project.config
        shots = len(project.production.get("shots", []))
        _log(f"{name:<24} {config['style_pack']:<10} "
             f"{config['resolution'][0]}x{config['resolution'][1]} @{config['fps']}  "
             f"{shots} shot(s)")
    return 0


def cmd_validate(args):
    project = Project.open(args.project)
    if not os.path.exists(project.production_path):
        _log(f"No {project.rel(project.production_path)} yet.")
        _log("Stage A is manual: paste the strong model's reply into that file.")
        return 2

    report = validate.check_production(project.production)
    config_report = validate.check_schema(project.config, "project")
    overrides_report = validate.check_schema(project.rig_overrides, "rig_overrides")
    inputs_report = validate.Report() if report.errors \
        else validate.check_project_inputs(project)

    total = validate.Report()
    for part, label in ((config_report, "project.json"),
                        (overrides_report, "rig_overrides.json"),
                        (report, "production.json"),
                        (inputs_report, "inputs")):
        if part.problems:
            _log(f"\n{label}:")
            for problem in part.problems:
                _log("  " + str(problem))
        total.extend(part)

    if not total.problems:
        _log("valid — schema, cross-references and inputs all check out")
    else:
        _log(f"\n{len(total.errors)} error(s), {len(total.warnings)} warning(s)")

    if not report.errors:
        _log("")
        _summarise_production(project)
    jsonio.write(project.cache("validate.json"), total.as_dict())
    return 1 if total.errors else 0


def _summarise_production(project):
    document = project.production
    overrides = project.rig_overrides
    _log(f"{document.get('title', '(untitled)')} — {document.get('fps')} fps")
    for character in document.get("characters", []):
        mode, source = validate.resolve_rig_mode(character, overrides)
        _log(f"  {character['id']:<16} {character['kind']:<10} "
             f"{character['height_m']:>5.2f} m   rig={mode} (from {source})")
    for env in document.get("envs", []):
        scatter = sum(s["count"] for s in env.get("scatter", []))
        _log(f"  env {env['id']:<12} ground={env['ground']:<14} "
             f"{scatter} scattered")
    for shot in document.get("shots", []):
        cast = ", ".join(m["character"] for m in shot.get("cast", []))
        lines = len(shot.get("dialogue", []))
        _log(f"  shot {shot['id']:<11} env={shot['env']:<14} "
             f"cast=[{cast}] {lines} line(s)")


def cmd_run(args):
    project = Project.open(args.project)
    report = validate.check_production(project.production)
    if report.errors:
        _log("production.json does not validate; refusing to run:")
        for problem in report.errors:
            _log("  " + str(problem))
        return 2

    walker = graph.Graph(project, on_log=_log, style_pack=args.style,
                         force=args.force or (), dry_run=args.dry_run,
                         resolution_scale=args.scale)
    stages = graph.order(only=args.only, start=getattr(args, "from"), stop=args.to)
    _log(f"{project.name}: {len(stages)} stage(s), style pack "
         f"{walker.style_pack}\n")
    results = walker.run(stages, stop_on_fail=not args.keep_going)

    _log("")
    failed = [r for r in results if r.failed]
    built = sum(1 for r in results for u in r.units if u.status == "built")
    cached = sum(1 for r in results for u in r.units if u.status == "cached")
    ran = [r for r in results if not r.skipped]
    _log(f"{len(ran)}/{len(stages)} stage(s) did work — "
         f"{built} unit(s) built, {cached} cached")
    if failed:
        for result in failed:
            for unit in result.units:
                if unit.status == "failed":
                    _log(f"FAILED {result.name}/{unit.unit.uid}: {unit.error}")
        return 1
    warnings = [v for r in results for u in r.units for v in u.verdicts
                if v.level == gates.WARN]
    if warnings:
        _log(f"{len(warnings)} warning(s) — see .cache/last_run.json")
    return 0


def cmd_status(args):
    project = Project.open(args.project)
    walker = graph.Graph(project, on_log=_log, style_pack=args.style)
    _log(f"{project.name}  style={walker.style_pack}\n")
    _log(f"{'stage':<14}{'units':>6}{'cached':>8}{'pending':>9}  detail")
    unbuilt = 0
    for spec, units in walker.plan(graph.order()):
        if not spec.implemented:
            unbuilt += 1
            _log(f"{spec.name:<14}{'-':>6}{'-':>8}{'-':>9}  "
                 f"{graph.NOT_BUILT_YET}")
            continue
        if not units:
            _log(f"{spec.name:<14}{0:>6}{'':>8}{'':>9}  (nothing to do)")
            continue
        cached, pending = [], []
        for unit in units:
            key, _ = project.store.key(spec.name, spec.version, unit.inputs,
                                       unit.params)
            (cached if project.store.has(spec.name, key) else pending).append(unit)
        detail = ", ".join(u.uid for u in (pending or cached))
        if len(detail) > 52:
            detail = detail[:49] + "..."
        _log(f"{spec.name:<14}{len(units):>6}{len(cached):>8}{len(pending):>9}  "
             f"{detail}")
    if unbuilt:
        _log("")
        _log(f"{unbuilt} of {len(graph.STAGE_ORDER)} declared stage(s) are "
             "not written yet; a run skips them and anything needing their "
             "output fails on the missing input.")
    return 0


def cmd_doctor(args):
    _log("Interpreters:")
    for logical, exe in envs.describe():
        _log(f"  {logical:<9} {exe}")
    _log(f"\nconda envs: {', '.join(envs.list_envs()) or '(none)'}")
    try:
        import jsonschema  # noqa: F401
        _log("jsonschema  available in the host interpreter")
    except ImportError:
        _log("jsonschema  MISSING in the host interpreter — schema validation "
             "will be skipped")
    try:
        from PIL import Image  # noqa: F401
        _log("Pillow      available (image and flicker gates enabled)")
    except ImportError:
        _log("Pillow      MISSING — image and flicker gates will be skipped")
    _log(f"\nprojects: {', '.join(Project.list_all()) or '(none)'}")
    return 0


def cmd_seed(args):
    """Write the Stage A seed prompt into a project, ready to paste."""
    project = Project.open(args.project)
    from .seed import build_seed_prompt

    target = project.path("story", "seed_prompt.md")
    text = build_seed_prompt(project)
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    _log(f"Wrote {project.rel(target)} ({len(text)} characters)")
    _log("Paste it, plus story/story.md, into the strong chat model, then save")
    _log(f"the JSON reply as {project.rel(project.production_path)}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="The Shahnameh production pipeline: a staged graph over a "
                    "content-addressed artifact store.")
    subs = parser.add_subparsers(dest="command", required=True)

    init = subs.add_parser("init", help="create a project directory")
    init.add_argument("name")
    init.add_argument("--title")
    init.add_argument("--fps", type=int)
    init.add_argument("--resolution", help='e.g. "1920x1080"')
    init.add_argument("--style", choices=STYLE_PACKS)
    init.set_defaults(func=cmd_init)

    listing = subs.add_parser("list", help="list projects")
    listing.set_defaults(func=cmd_list)

    check = subs.add_parser("validate", help="validate production.json and inputs")
    check.add_argument("project")
    check.set_defaults(func=cmd_validate)

    run = subs.add_parser("run", help="walk the stage graph")
    run.add_argument("project")
    run.add_argument("--only", nargs="+", metavar="STAGE")
    run.add_argument("--from", metavar="STAGE")
    run.add_argument("--to", metavar="STAGE")
    run.add_argument("--force", nargs="+", metavar="STAGE",
                     help="rebuild these stages even if cached ('all' for every one)")
    run.add_argument("--style", choices=STYLE_PACKS)
    run.add_argument("--scale", type=float, metavar="FRACTION",
                     help="render at this fraction of the project resolution "
                          "(0.5 is a quarter of the pixels and about a quarter "
                          "of the time); it enters the artifact key, so a draft "
                          "is never served as a final")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--keep-going", action="store_true",
                     help="do not stop at the first failing stage")
    run.set_defaults(func=cmd_run)

    status = subs.add_parser("status", help="what is cached and what is pending")
    status.add_argument("project")
    status.add_argument("--style", choices=STYLE_PACKS)
    status.set_defaults(func=cmd_status)

    seed = subs.add_parser("seed", help="write the Stage A seed prompt")
    seed.add_argument("project")
    seed.set_defaults(func=cmd_seed)

    doctor = subs.add_parser("doctor", help="what this machine can run")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv=None):
    console.configure()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (graph.StageError, envs.EnvError, ModuleNotFoundError,
            FileNotFoundError, ValueError) as exc:
        _log(f"error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
