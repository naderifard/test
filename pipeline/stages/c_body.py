"""Stage C-body — the body mesh and its armature, one .blend per character.

§7-C2 is SMPL-X. M1 substitutes ``prelude.character()``, which builds a welded
primitive body against the *same interface*: SMPL-X joint names in SMPL-X order,
the eight mouth channels as shape keys, material roles instead of materials, and
the requested height measured from the floor. M3 swaps the geometry; the rest of
the pipeline never learns that it changed.

The stage is one Blender step. There is no host half to speak of - the unit's
parameters and ``face.json`` are all the builder needs - and a second
interpreter would only add a process.

What it measures is the thing that has bitten every one of these builds so far:
the height it was asked for, the gap between the lowest vertex and the floor,
and whether a single vertex escaped the armature. A body that is 4 cm short or
sunk into the ground fails here rather than in the render, where the cause is
three stages away.
"""

from __future__ import annotations

import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio
from pipeline.validate import VISEME_CHANNELS

HEIGHT_TOLERANCE_M = 0.02


def plan(project, graph):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    units = []
    for character in resolved.get("characters") or []:
        cid = character["id"]
        units.append(Unit(
            f"char_{cid}",
            inputs={"face": project.path("assets", "characters", cid, "face",
                                         "face.json"),
                    # c_asset's photo mesh. When it exists the body is built
                    # from it; the welded-primitive stand-in is only for a
                    # character that has no reference image at all.
                    "asset": project.path("assets", "characters", cid, "asset",
                                          "asset.json"),
                    "asset_mesh": project.path("assets", "characters", cid,
                                               "asset", "mesh.npz"),
                    "asset_photo": project.path("assets", "characters", cid,
                                                "asset", "photo.png")},
            params={
                "subject": cid,
                "kind": character.get("kind", "human"),
                "preset": character.get("shape_preset", "neutral"),
                "height_m": float(character.get("height_m") or 1.75),
                "material_roles": character.get("material_roles") or {},
                "rig_mode": character.get("rig_mode", "smplx"),
                "clothed": True,
            },
            out_rel=f"assets/characters/{cid}/body",
            label=f"body/{cid}",
        ))
    return units


def nothing_to_do(project):
    return "no resolved production plan — a_production has not run"


def _gate(project, unit, metrics):
    out = list(gates.check_body(metrics)) + list(gates.check_asset(metrics))
    out = [v for v in out if v.level != gates.PASS]

    asked = metrics.get("asked_height_m") or 0.0
    built = metrics.get("height_m") or 0.0
    delta = abs(built - asked)
    if delta > HEIGHT_TOLERANCE_M:
        out.append(gates.Verdict(
            gates.FAIL, "body.fit",
            f"built {built:.3f} m against a requested {asked:.3f} m; the fit "
            "step did not take", round(delta, 4), HEIGHT_TOLERANCE_M))

    feet = metrics.get("feet_z_m")
    if feet is not None and abs(feet) > HEIGHT_TOLERANCE_M:
        out.append(gates.Verdict(
            gates.FAIL, "body.floor",
            f"the lowest vertex sits at z={feet:+.3f}; a body that is not on "
            "the floor puts every shot's contact off",
            round(feet, 4), HEIGHT_TOLERANCE_M))

    if metrics.get("unweighted"):
        out.append(gates.Verdict(
            gates.FAIL, "body.weights",
            f"{metrics['unweighted']} vertices are weighted to no bone and "
            "would stay behind when the body moves",
            metrics["unweighted"], 0))

    missing = metrics.get("missing_channels") or []
    if missing:
        out.append(gates.Verdict(
            gates.FAIL, "body.visemes",
            f"no shape key for {', '.join(missing)}", len(missing), 0))

    if not metrics.get("bones"):
        out.append(gates.Verdict(gates.FAIL, "body.rig",
                                 "the body has no armature"))

    untagged = metrics.get("untagged_slots") or []
    if untagged:
        out.append(gates.Verdict(
            gates.FAIL, "body.roles",
            f"{len(untagged)} material slot(s) carry no style_role, so the "
            f"style packs cannot swap them: {', '.join(untagged[:4])}",
            len(untagged), 0))

    return out or [gates.Verdict(
        gates.PASS, "body",
        f"{built:.3f} m, {metrics.get('tris')} tris, {metrics.get('bones')} "
        f"bones, {len(VISEME_CHANNELS)} mouth channels")]


SMPLX_KINDS = ("human", "humanoid")
SURFACE_EVERY = 3


def fit_smplx(job, unit):
    """Step 1, in the motion env: SMPL-X fitted to the character's joints and
    surface, left as ``smplx_fit.npz`` for the Blender step to skin from.

    Skipped, with a note, for a quadruped, for a character with no photo
    mesh, and on a machine without SMPL-X: the Blender step then weights the
    mesh as it did before. A fit that does not hold (a beast's proportions)
    is written anyway and refused there, by its error."""
    import os

    import numpy as np

    from pipeline import smplx_fit
    params = unit["params"]
    asset_path = unit["inputs"]["asset"]
    mesh_path = unit["inputs"]["asset_mesh"]
    if params.get("kind") not in SMPLX_KINDS:
        return {"notes": ["no SMPL-X fit: not a human figure"]}
    if not (os.path.exists(asset_path) and os.path.exists(mesh_path)):
        return {"notes": ["no SMPL-X fit: no photo mesh"]}
    if not smplx_fit.available():
        return {"notes": ["no SMPL-X fit: the model is not on this machine"]}
    asset = jsonio.read(asset_path)
    if (asset.get("rig") or {}).get("method") == "quadruped_template":
        return {"notes": ["no SMPL-X fit: a quadruped's rig"]}
    surface = np.load(mesh_path)["verts"][::SURFACE_EVERY]
    result = smplx_fit.fit(asset["joints"], list(smplx_fit.SMPLX_BODY),
                           surface=surface)
    smplx_fit.save(_base.out_path(unit, "smplx_fit.npz"), result)
    return {"metrics": {"smplx_fit_error": result["error"]}}


STAGE = StageSpec(
    name="c_body",
    # 14: skin weights from a fitted SMPL-X body; a hem hangs from the hips.
    version="14",
    summary="rig each character's photo mesh (or a stand-in, without one)",
    needs=["c_face", "c_asset"],
    steps=[Step("motion"),
           Step("blender", "pipeline/blender/build_body.py")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(fit_smplx))
