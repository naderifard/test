"""Stage C-rigged — publish the character, and refuse to publish a broken one.

§7-C5. By the time a unit reaches this stage the geometry, the UVs and the
garments already exist, so nothing here builds anything. Two things make it a
stage rather than a rename:

* It writes the manifest ``h_assemble`` reads. That includes the one mapping the
  production plan needs and the rig does not have - ``attach: hand_r`` in a prop
  record against ``right_wrist`` on an SMPL-X armature. Resolving it here means
  the shot builder never guesses a bone name.
* It is the acceptance gate for a character (§11-C). Height, floor contact,
  weights, mouth channels, role tags and triangle count are all checked against
  the *evaluated* mesh, and the mouth is checked by opening it: a jaw key that
  exists but moves nothing would pass every structural test and then produce a
  film where nobody's lips part.
"""

from __future__ import annotations

import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio
from pipeline.validate import VISEME_CHANNELS

HEIGHT_TOLERANCE_M = 0.02
JAW_MIN_MOVE_M = 0.002


CLAW_WORDS = ("claw", "talon", "paw", "hoof", "hooves")


def hands_for(character):
    """"smplx" to give the character real hands (``blender/hands.py``), None
    to keep what the shape model made: a person gets them, a humanoid too
    unless its description gives it claws -- the black div's are claws, and
    SMPL-X's hand on him would be a man's."""
    kind = character.get("kind", "human")
    look = (character.get("image_prompt") or "").lower()
    if kind == "human":
        return "smplx"
    if kind == "humanoid" and not any(w in look for w in CLAW_WORDS):
        return "smplx"
    return None


def plan(project, graph):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    props = {}
    for prop in resolved.get("props") or []:
        if prop.get("held_by"):
            props.setdefault(prop["held_by"], []).append(
                {"id": prop["id"], "attach": prop.get("attach") or "hand_r"})

    units = []
    for character in resolved.get("characters") or []:
        cid = character["id"]
        base = ("assets", "characters", cid)
        units.append(Unit(
            f"char_{cid}",
            inputs={"garmented": project.path(*base, "garment",
                                              "garmented.blend"),
                    "body_json": project.path(*base, "body", "body.json"),
                    "garment_json": project.path(*base, "garment",
                                                 "garment.json"),
                    "face": project.path(*base, "face", "face.json")},
            params={"subject": cid,
                    "kind": character.get("kind", "human"),
                    "rig_mode": character.get("rig_mode", "smplx"),
                    "height_m": float(character.get("height_m") or 1.75),
                    "hands": hands_for(character),
                    "attach": sorted({p["attach"] for p in props.get(cid, [])}),
                    "props": props.get(cid, [])},
            out_rel=f"assets/characters/{cid}/rigged",
            label=f"rigged/{cid}",
        ))
    return units


def nothing_to_do(project):
    return "no resolved production plan — a_production has not run"


def _gate(project, unit, metrics):
    out = [v for v in gates.check_asset(metrics) if v.level != gates.PASS]
    out += [v for v in gates.check_body(metrics) if v.level != gates.PASS]

    asked = metrics.get("asked_height_m") or 0.0
    built = metrics.get("height_m") or 0.0
    if abs(built - asked) > HEIGHT_TOLERANCE_M:
        out.append(gates.Verdict(
            gates.FAIL, "rigged.height",
            f"{built:.3f} m against the plan's {asked:.3f} m",
            round(abs(built - asked), 4), HEIGHT_TOLERANCE_M))

    feet = metrics.get("feet_z_m")
    if feet is not None and abs(feet) > HEIGHT_TOLERANCE_M:
        out.append(gates.Verdict(
            gates.FAIL, "rigged.floor",
            f"the asset's lowest vertex is at z={feet:+.3f}, not on the floor",
            round(feet, 4), HEIGHT_TOLERANCE_M))

    if metrics.get("unweighted"):
        out.append(gates.Verdict(
            gates.FAIL, "rigged.weights",
            f"{metrics['unweighted']} vertices follow no bone",
            metrics["unweighted"], 0))

    missing = metrics.get("missing_channels") or []
    if missing:
        out.append(gates.Verdict(gates.FAIL, "rigged.visemes",
                                 f"no shape key for {', '.join(missing)}",
                                 len(missing), 0))

    jaw = metrics.get("jaw_move_m")
    if jaw is not None and jaw < JAW_MIN_MOVE_M:
        out.append(gates.Verdict(
            gates.FAIL, "rigged.jaw",
            f"opening jaw_open moved the mesh {jaw * 1000:.2f} mm; the mouth "
            "channel is connected to nothing", round(jaw, 5), JAW_MIN_MOVE_M))

    untagged = metrics.get("untagged_slots") or []
    if untagged:
        out.append(gates.Verdict(
            gates.FAIL, "rigged.roles",
            f"{len(untagged)} material slot(s) have no style_role: "
            f"{', '.join(untagged[:4])}", len(untagged), 0))

    unresolved = metrics.get("unresolved_attach") or []
    if unresolved:
        out.append(gates.Verdict(
            gates.FAIL, "rigged.attach",
            f"no bone for attachment point(s) {', '.join(unresolved)}",
            len(unresolved), 0))

    return out or [gates.Verdict(
        gates.PASS, "rigged",
        f"{built:.3f} m, {metrics.get('tris')} tris, {metrics.get('bones')} "
        f"bones, {len(VISEME_CHANNELS)} channels, jaw opens "
        f"{(jaw or 0) * 1000:.1f} mm")]


STAGE = StageSpec(
    name="c_rigged",
    # 2: SMPL-X hands in place of the shape model's mittens.
    version="2",
    summary="publish the finished character asset and check it can be rendered",
    needs=["c_garment"],
    steps=[Step("blender", "pipeline/blender/build_rigged.py")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(lambda job, unit: {}))
