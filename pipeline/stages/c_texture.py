"""Stage C-texture — give the body a UV layout, and measure it.

§7-C3 generates albedo from the reference images. M1 stubs the *generation*:
there is no texture synthesis here, and there is no image file in the artifact.
What is not stubbed is the part every later texture depends on and that nothing
else in the chain produces - a UV unwrap, plus the measurement that says whether
it is usable.

Skipping the stage entirely would have been the other option. It is the wrong
one: ``c_garment`` and ``c_rigged`` would then have to be re-keyed when M4's
albedo lands, and the render would have no UVs for any material that samples
them. Unwrapping now costs a second per character and makes the M4 substitution
a change to one stage.
"""

from __future__ import annotations

import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio


def plan(project, graph):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    units = []
    for character in resolved.get("characters") or []:
        cid = character["id"]
        units.append(Unit(
            f"char_{cid}",
            inputs={"body": project.path("assets", "characters", cid, "body",
                                         "body.blend"),
                    "refs": project.path("assets", "characters", cid, "refs",
                                         "refs.json")},
            params={"subject": cid, "angle_limit_deg": 66.0,
                    "island_margin": 0.02},
            out_rel=f"assets/characters/{cid}/texture",
            label=f"texture/{cid}",
        ))
    return units


def nothing_to_do(project):
    return "no resolved production plan — a_production has not run"


def _gate(project, unit, metrics):
    out = [v for v in gates.check_asset(metrics) if v.level != gates.PASS]
    coverage = metrics.get("uv_coverage")
    if coverage is None:
        out.append(gates.Verdict(gates.FAIL, "texture.uv",
                                 "the unwrap produced no UV layer at all"))
    if metrics.get("unwrapped_meshes", 0) < 1:
        out.append(gates.Verdict(gates.FAIL, "texture.meshes",
                                 "no mesh was unwrapped"))
    return out or [gates.Verdict(
        gates.PASS, "texture",
        f"UVs cover {coverage:.0%} of the 0-1 square over "
        f"{metrics.get('roles')} role(s)")]


STAGE = StageSpec(
    name="c_texture",
    version="1",
    summary="unwrap the body and measure the UV layout the albedo will need",
    needs=["c_body"],
    steps=[Step("blender", "pipeline/blender/build_texture.py")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(lambda job, unit: {}))
