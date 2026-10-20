"""Stage D — the set, and the things the cast carry.

§7-D. Three strategies, chosen by the style pack, because "the environment"
means three different things in the three looks: a painted backdrop on stacked
planes for *miniature*, procedural ground for *stylized*, displaced terrain under
an HDRI for *photoreal*. The pack name therefore travels in the unit's params:
the artifact genuinely differs per pack, so it must key differently per pack, and
the character stages - whose output does not differ - keep their one cache entry.

Props live here too. The design gives them no stage of their own, they are
published as ``prop_<id>`` collections exactly like an env is published as
``env_<id>``, and in M1 a prop is a primitive scaled to the size the plan asked
for. Adding a fifteenth stage to hold two boxes would cost more than it explains.

``scatter[].avoid_radius_m`` is read the way ``pipeline/seed.py`` documents it -
"to keep them off the actors' marks". The marks belong to a shot and this asset
is shared by every shot that uses the set, so the radius is applied around the
origin, which is where the marks are. Read the other way round - as separation
between instances - 240 items at 2.4 m apart do not fit in 26x26 m at all, and
the meadow the plan describes would come out as a lawn with a hundred tufts.
"""

from __future__ import annotations

import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio

SCATTER_MIN_FRACTION = 0.5
# The depth fit puts the ground under the stage at z=0 by construction; this is
# how far off it may land before the fit is worth a look.
STAGE_Z_TOLERANCE_M = 0.35
PROP_TOLERANCE_M = 0.005


def _env_inputs(project, eid):
    """refs.json, plus the image file itself so replacing it rebuilds the set."""
    refs = project.path("assets", "envs", eid, "refs", "refs.json")
    inputs = {"refs": refs}
    image = scene_image(project.root, refs)
    if image:
        inputs["image"] = image
    return inputs


def plan(project, graph):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    units = []
    for env in resolved.get("envs") or []:
        eid = env["id"]
        units.append(Unit(
            f"env_{eid}",
            inputs=_env_inputs(project, eid),
            params={"kind": "env",
                    "subject": eid,
                    "style_pack": graph.style_pack,
                    "interior": bool(env.get("interior")),
                    "ground_role": env.get("ground") or "ground",
                    "light_intent": env.get("light_intent") or "",
                    "scatter": env.get("scatter") or []},
            out_rel=f"assets/envs/{eid}/build",
            label=f"env/{eid}",
        ))
    for prop in resolved.get("props") or []:
        pid = prop["id"]
        units.append(Unit(
            f"prop_{pid}",
            params={"kind": "prop",
                    "subject": pid,
                    "style_pack": graph.style_pack,
                    "role": prop.get("role") or "wood",
                    "size_m": [float(v) for v in
                               (prop.get("size_m") or [0.1, 0.1, 1.0])],
                    "held_by": prop.get("held_by"),
                    "attach": prop.get("attach")},
            out_rel=f"assets/props/{pid}/build",
            label=f"prop/{pid}",
        ))
    return units


def nothing_to_do(project):
    return "no resolved production plan — a_production has not run"


def _gate_env(metrics):
    out = [v for v in gates.check_asset(metrics) if v.level != gates.PASS]

    if not metrics.get("ground"):
        out.append(gates.Verdict(gates.FAIL, "env.ground",
                                 "the set has no ground object to stand on"))

    if metrics.get("strategy") == "image_set":
        # The set reaches a far dome ~250 m out on purpose: that is where the
        # image's sky and distance live. The generic 200 m scale check was
        # written for props and characters.
        out = [v for v in out if v.name != "asset.bbox"]
        stage = metrics.get("stage_z_m")
        if stage is None:
            out.append(gates.Verdict(
                gates.FAIL, "env.stage",
                "nothing under the stage: a ray straight down at the origin "
                "misses the set, so the cast would stand on air"))
        elif abs(stage) > STAGE_Z_TOLERANCE_M:
            out.append(gates.Verdict(
                gates.WARN, "env.stage",
                f"the ground under the stage is at z={stage:+.2f} m, not 0; "
                f"the cast is dropped onto it, but the depth fit is off",
                stage, STAGE_Z_TOLERANCE_M))
        for name in metrics.get("untagged_slots") or []:
            out.append(gates.Verdict(gates.FAIL, "env.roles",
                                     f"{name} carries no style_role"))
        return out or [gates.Verdict(
            gates.PASS, "env",
            f"image set, horizon at {metrics.get('horizon_fraction', 0):.0%}, "
            f"stage ground z={stage:+.3f} m, {metrics.get('tris')} tris")]

    reach = metrics.get("ground_span_m") or 0.0
    needed = metrics.get("scatter_span_m") or 0.0
    if needed and reach < needed:
        out.append(gates.Verdict(
            gates.FAIL, "env.extent",
            f"ground spans {reach:.1f} m but the scatter covers {needed:.1f} m, "
            "so part of the set hangs off the edge of the world",
            round(reach, 2), round(needed, 2)))

    grown = metrics.get("ground_enlarged") or 0.0
    if grown:
        out.append(gates.Verdict(
            gates.WARN, "env.extent",
            f"the plan scatters over {metrics.get('scatter_span_m')} m and the "
            f"{metrics.get('style_pack')} pack's floor is "
            f"{metrics.get('ground_span_pack_m')} m, so the floor was grown to "
            f"{metrics.get('ground_span_m')} m to hold it",
            metrics.get("ground_span_m"), metrics.get("ground_span_pack_m")))

    for record in metrics.get("scatter") or []:
        asked = record.get("asked") or 0
        placed = record.get("placed") or 0
        if asked and placed < asked * SCATTER_MIN_FRACTION:
            out.append(gates.Verdict(
                gates.WARN, "env.scatter",
                f"{record.get('role')}: {placed} of {asked} placed; the area is "
                "too small for that many", placed,
                int(asked * SCATTER_MIN_FRACTION)))

    if metrics.get("backdrop_wanted") and not metrics.get("backdrop_plate"):
        out.append(gates.Verdict(
            gates.WARN, "env.backdrop",
            f"{metrics.get('strategy')} paints the reference onto the backdrop "
            "and this set has none yet; the planes are flat role colour"))

    for name in metrics.get("untagged_slots") or []:
        out.append(gates.Verdict(gates.FAIL, "env.roles",
                                 f"{name} carries no style_role"))

    placed = sum(r.get("placed") or 0 for r in metrics.get("scatter") or [])
    return out or [gates.Verdict(
        gates.PASS, "env",
        f"{metrics.get('strategy')}, {metrics.get('ground_span_m')} m ground, "
        f"{placed} scattered, {metrics.get('tris')} tris")]


def _gate_prop(metrics):
    out = [v for v in gates.check_asset(metrics) if v.level != gates.PASS]

    asked = metrics.get("asked_size_m") or []
    built = metrics.get("size_m") or []
    if asked and built and len(asked) == len(built):
        off = max(abs(a - b) for a, b in zip(asked, built))
        if off > PROP_TOLERANCE_M:
            out.append(gates.Verdict(
                gates.FAIL, "prop.size",
                f"built {built} against the plan's {asked}",
                round(off, 4), PROP_TOLERANCE_M))

    for name in metrics.get("untagged_slots") or []:
        out.append(gates.Verdict(gates.FAIL, "prop.roles",
                                 f"{name} carries no style_role"))

    return out or [gates.Verdict(
        gates.PASS, "prop",
        f"{built[2] if len(built) > 2 else 0:.2f} m of {metrics.get('role')}, "
        f"{metrics.get('tris')} tris, grip at "
        f"z={metrics.get('grip_z_m', 0.0):.2f}")]


def _gate(project, unit, metrics):
    if metrics.get("kind") == "prop":
        return _gate_prop(metrics)
    return _gate_env(metrics)


def scene_image(project_root, refs_json):
    """The scene's reference image, or None. Same slot order as build_env._plate."""
    import os
    refs = jsonio.read(refs_json, default={}) or {}
    slots = refs.get("slots") or {}
    images = refs.get("images") or []
    name = (slots.get("wide") or slots.get("front") or slots.get("back")
            or (images[0].get("path") if images else None))
    folder = refs.get("folder")
    if not name or not folder:
        return None
    path = os.path.join(project_root, folder.replace("/", os.sep), name)
    return path if os.path.exists(path) else None


def depth(job, unit):
    """The ml step: the scene image's depth, as a grid in world metres.

    Writes ``set.npz`` and ``set.json`` into the unit directory, where the
    Blender step builds the set from them. Props have no image and no step.
    """
    import os

    import numpy as np
    from PIL import Image

    from pipeline import scene_depth

    params = unit["params"]
    if params.get("kind") == "prop":
        return {"metrics": {}, "notes": []}
    image_path = scene_image(job.project_root, unit["inputs"]["refs"])
    if image_path is None:
        raise RuntimeError(
            f"scene {params['subject']!r} has no image. The set is built from "
            f"it; add one in the Scenes panel, or drop it into "
            f"refs/envs/{params['subject']}/{params['subject']}_wide.png.")
    image = Image.open(image_path)
    disparity = scene_depth.estimate(image)
    built = scene_depth.build(disparity)
    out_dir = unit["out_dir"]
    np.savez_compressed(os.path.join(out_dir, "set.npz"),
                        verts=built.pop("verts"), uv=built.pop("uv"),
                        keep=built.pop("keep"),
                        backdrop=built.pop("backdrop"))
    built["image"] = os.path.basename(image_path)
    jsonio.write(os.path.join(out_dir, "set.json"), built)
    return {
        "metrics": {
            "horizon_fraction": round(built["camera"]["horizon_fraction"], 4),
            "stage_distance_m": round(built["stage_distance_m"], 3),
            "sky_fraction": round(built["sky_fraction"], 4),
            "torn_fraction": round(built["torn_fraction"], 4),
        },
        "notes": [f"depth: horizon at {built['camera']['horizon_fraction']:.0%}"
                  f" of the frame, stage {built['stage_distance_m']:.1f} m from "
                  f"the reference camera, {built['torn_fraction']:.1%} of the "
                  f"grid torn at occlusion edges"],
    }


STAGE = StageSpec(
    name="d_env",
    version="2",
    summary="build each scene from its image's depth, and each prop",
    needs=["b_refs"],
    # Depth in the ml environment, then the set in Blender. Both write into the
    # same unit directory, so the Blender step reads set.npz from there.
    steps=[Step("ml"),
           Step("blender", "pipeline/blender/build_env.py")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(depth, lambda job, unit: {}))
