"""Stage C-garment — dress the body, and prove the clothes move with it.

§7-C4. M1 builds each garment as a primitive shell over the body's own joint
frame and binds it by proximity, which is what ``prelude.bind_garment`` does.
M5 replaces both halves - a real garment mesh, Surface Deform plus correctives -
and the contract that survives is the one this stage checks: a garment is a
*separate object*, modelled in the body's rest space, that ends up driven by the
body's armature.

That last clause is the whole measurement. A garment that is merely parented, or
weighted to nothing, looks perfect in the asset file and then stands still while
the character walks out of it. So after the artifact is written this stage keys a
throwaway pose onto the rig and measures how far the garment's vertices actually
travelled. The .blend is already on disk by then, so the test cannot contaminate
what was saved.

A character with no garments still produces an artifact: the dressed body is the
input every later stage names, and making that path conditional would mean two
kinds of character asset for the rest of the pipeline to keep straight.
"""

from __future__ import annotations

import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio

BIND_MIN_FRACTION = 0.95
MOVE_MIN_M = 0.002


def plan(project, graph):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    units = []
    for character in resolved.get("characters") or []:
        cid = character["id"]
        base = ("assets", "characters", cid)
        # A body built from a photo is already dressed: the pelt, the crown and
        # the robe are in the reference image, so in the mesh. Primitive shells
        # over it would be exactly the boxes that body replaced.
        body = jsonio.read(project.path(*base, "body", "body.json"),
                           default={}) or {}
        photo = body.get("source") == "sf3d_photo"
        units.append(Unit(
            f"char_{cid}",
            inputs={"body": project.path(*base, "texture", "textured.blend"),
                    "body_json": project.path(*base, "body", "body.json")},
            params={"subject": cid,
                    "kind": character.get("kind", "human"),
                    "garments": ([] if photo
                                 else character.get("garments") or []),
                    "material_roles": character.get("material_roles") or {},
                    "offset_m": 0.004},
            out_rel=f"assets/characters/{cid}/garment",
            label=f"garment/{cid}",
        ))
    return units


def nothing_to_do(project):
    return "no resolved production plan — a_production has not run"


def _gate(project, unit, metrics):
    out = [v for v in gates.check_asset(metrics) if v.level != gates.PASS]
    garments = metrics.get("garments") or []

    for garment in garments:
        gid = garment.get("id")
        of = garment.get("of") or 0
        bound = garment.get("bound") or 0
        if of and bound < of * BIND_MIN_FRACTION:
            out.append(gates.Verdict(
                gates.FAIL, "garment.bind",
                f"{gid}: only {bound} of {of} vertices found a bone to follow",
                bound, int(of * BIND_MIN_FRACTION)))
        if (garment.get("bones") or 0) < 2:
            out.append(gates.Verdict(
                gates.WARN, "garment.bones",
                f"{gid}: reaches {garment.get('bones')} bone(s); a shell bound "
                "to one bone moves rigidly", garment.get("bones"), 2))
        moved = garment.get("moved_m")
        if moved is not None and moved < MOVE_MIN_M:
            out.append(gates.Verdict(
                gates.FAIL, "garment.follow",
                f"{gid}: the test pose moved the body but the garment only "
                f"{moved * 1000:.2f} mm — it is not following the armature",
                round(moved, 5), MOVE_MIN_M))

    for note in metrics.get("skipped") or []:
        out.append(gates.Verdict(gates.WARN, "garment.skipped", note))

    if not garments:
        return out or [gates.Verdict(gates.PASS, "garment",
                                     "no garments; the body passes through")]
    return out or [gates.Verdict(
        gates.PASS, "garment",
        f"{len(garments)} garment(s), "
        f"{sum(g.get('tris') or 0 for g in garments)} tris, all following "
        "the rig")]


STAGE = StageSpec(
    name="c_garment",
    version="1",
    summary="build and bind each character's garments over the rigged body",
    needs=["c_texture"],
    steps=[Step("blender", "pipeline/blender/build_garment.py")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(lambda job, unit: {}))
