"""c_texture, in Blender: unwrap the body that c_body built.

Opens the previous artifact rather than rebuilding the character. That is the
whole reason the chain is a chain: a rebuild would let the two stages disagree
about the geometry, and then the UVs would belong to a body nobody rendered.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from pipeline import jsonio                                        # noqa: E402
from pipeline.blender import _common                               # noqa: E402
from pipeline.stages import _base                                  # noqa: E402

import bpy                                                         # noqa: E402

B = _common.install()


def run(job, unit):
    params = unit["params"]
    cid = params["subject"]
    _common.open_blend(unit["inputs"]["body"])

    collection = bpy.data.collections.get(f"char_{cid}")
    bodies = _common.meshes(collection)
    if not bodies:
        raise RuntimeError(f"char_{cid} holds no mesh to unwrap")

    per_mesh = []
    for ob in bodies:
        # A photo body already carries its UVs -- the exact projection of the
        # reference -- and a smart-project unwrap would only add a second map
        # nothing samples, at a minute a character.
        if not ob.data.get("photo_body"):
            B.unwrap(ob, angle=float(params.get("angle_limit_deg", 66.0)))
        layer = ob.data.uv_layers.active
        per_mesh.append({
            "object": ob.name,
            "uv_layer": layer.name if layer else None,
            "loops": len(layer.data) if layer else 0,
            "coverage": _common.uv_coverage(ob.data),
            "roles": [m.get("style_role") for m in ob.data.materials],
        })

    stats = _common.measure(bodies)
    blend = _common.save_blend(_base.out_path(unit, "textured.blend"))
    jsonio.write(_base.out_path(unit, "texture.json"), {
        "schema_version": 1,
        "subject": cid,
        "collection": f"char_{cid}",
        "blend": os.path.basename(blend),
        "albedo": None,          # M4 writes the generated maps here
        "unwrap": {"method": "smart_project",
                   "angle_limit_deg": params.get("angle_limit_deg"),
                   "island_margin": params.get("island_margin")},
        "meshes": per_mesh,
        "measured": stats,
    })
    return {
        "metrics": {
            "subject": cid,
            "unwrapped_meshes": len(bodies),
            "roles": len({r for m in per_mesh for r in m["roles"] if r}),
            **stats,
        },
        "notes": [f"unwrapped {len(bodies)} mesh(es), UV coverage "
                  f"{stats.get('uv_coverage')}"],
    }


if __name__ == "__main__":
    sys.exit(_base.stage_main(run, from_blender=True))
