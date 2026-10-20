"""Stage C-face — pick the head identity, once, for the whole film.

§7-C1 wants a FLAME identity per character. M1 stubs the *fitting*: nothing here
looks at a photograph and solves for coefficients. What it does instead is pick
a hand-chosen identity from :data:`IDENTITIES` by the character's shape preset
and record it, with the coefficients written out in full.

That sounds like a file that does nothing, and it is the cheapest way to make
the substitution in M4 a substitution. Twelve stages downstream ask "which head
is this?"; if the answer lived in each of their parameters, swapping in a real
MICA fit would mean touching twelve stages. Because it lives in one artifact
that ``c_body`` takes as an input, replacing the *contents* of this file
re-keys every character asset that depended on it and nothing else changes.

The one measurement it makes is whether a face reference actually exists. That
is a warning, not an error (§7-B): with an empty ``refs/`` the whole slice still
builds, on presets alone.
"""

from __future__ import annotations

import hashlib
import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio
from pipeline.validate import ANCHORS, VISEME_CHANNELS

# Hand-picked FLAME identities. The vectors are the first eight shape
# coefficients of the FLAME 2020 generic model, chosen by eye against the
# epithets in the production plan rather than fitted to anything: sigma is
# roughly 1.0 per coefficient, so these stay inside a plausible face.
IDENTITIES = {
    "heroic": {
        "index": 11,
        "note": "wide heavy-browed skull, deep-set eyes, broad jaw",
        "shape": [1.42, -0.86, 0.61, 0.44, -0.30, 0.18, -0.12, 0.07],
        "head_scale": 1.06,
    },
    "youth": {
        "index": 4,
        "note": "short round skull, high cheek, small chin, open eyes",
        "shape": [-1.15, 0.72, -0.48, 0.21, 0.33, -0.14, 0.09, -0.05],
        "head_scale": 0.97,
    },
    "neutral": {
        "index": 0,
        "note": "the model mean",
        "shape": [0.0] * 8,
        "head_scale": 1.0,
    },
    "giant": {
        "index": 19,
        "note": "long heavy skull — the quadruped reads it as head length only",
        "shape": [1.90, 0.35, -0.72, 0.55, 0.12, -0.28, 0.16, 0.04],
        "head_scale": 1.10,
    },
}

# Which words in the image prompt decide whether the body gets a beard. The
# prompt is the one place the plan describes a face in prose, and "beardless"
# has to win over "beard" because it contains it.
BEARD_NO = ("beardless", "clean-shaven", "clean shaven", "no beard")
BEARD_YES = ("beard", "bearded", "moustache", "whiskers")


def _beard(character):
    prompt = (character.get("image_prompt") or "").lower()
    if any(word in prompt for word in BEARD_NO):
        return False, "prompt says beardless"
    if any(word in prompt for word in BEARD_YES):
        return True, "prompt describes a beard"
    return False, "prompt says nothing about a beard"


def plan(project, graph):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    units = []
    for character in resolved.get("characters") or []:
        cid = character["id"]
        # A quadruped has no FLAME head, and gets a unit anyway: c_body takes
        # face.json as an input, and a missing input would make the horse the
        # one asset whose chain has a hole in it.
        units.append(Unit(
            f"char_{cid}",
            inputs={"refs": project.path("assets", "characters", cid, "refs",
                                         "refs.json")},
            params={
                "subject": cid,
                "kind": character.get("kind", "human"),
                "preset": character.get("shape_preset", "neutral"),
                "image_prompt": character.get("image_prompt", ""),
            },
            out_rel=f"assets/characters/{cid}/face",
            label=f"face/{cid}",
        ))
    return units


def nothing_to_do(project):
    return "no resolved production plan — a_production has not run"


def run(job, unit):
    params = unit["params"]
    cid = params["subject"]
    preset = params.get("preset") or "neutral"
    kind = params.get("kind", "human")

    refs = jsonio.read(unit["inputs"]["refs"], default={}) or {}
    slots = refs.get("slots") or {}
    used = slots.get("face") or slots.get("front")

    identity = dict(IDENTITIES.get(preset) or IDENTITIES["neutral"])
    identity["preset"] = preset
    # A stable per-character seed. Every stub that needs to look deliberate
    # rather than random — the idle take, the scatter — draws from the same
    # number, so a character looks like itself in every shot.
    identity["seed"] = int(hashlib.sha256(cid.encode("utf-8")).hexdigest()[:8], 16)
    identity["source"] = "flame_preset"
    identity["fitted_from"] = used

    beard, why = _beard(params)
    if kind == "quadruped":
        beard, why = False, "quadruped"

    jsonio.write(_base.out_path(unit, "face.json"), {
        "schema_version": 1,
        "subject": cid,
        "kind": kind,
        "identity": identity,
        "beard": beard,
        "beard_reason": why,
        "visemes": {"channels": list(VISEME_CHANNELS), "rest": "closed"},
        "anchors": list(ANCHORS),
        "refs": {"face": slots.get("face"), "front": slots.get("front")},
    })
    return {
        "metrics": {
            "subject": cid,
            "identity_index": identity["index"],
            "preset": preset,
            "beard": beard,
            "has_face_ref": bool(used),
            "channels": len(VISEME_CHANNELS),
        },
        "notes": [f"identity {identity['index']} ({identity['note']}), {why}"],
    }


def _gate(project, unit, metrics):
    out = []
    if not metrics.get("has_face_ref"):
        out.append(gates.Verdict(
            gates.WARN, "face.refs",
            f"no face or front reference for {metrics.get('subject')}; the "
            f"identity is the hand-picked {metrics.get('preset')} preset"))
    if metrics.get("channels") != len(VISEME_CHANNELS):
        out.append(gates.Verdict(
            gates.FAIL, "face.channels",
            "the viseme channel set is not the eight the rig expects",
            metrics.get("channels"), len(VISEME_CHANNELS)))
    return out or [gates.Verdict(
        gates.PASS, "face",
        f"identity {metrics.get('identity_index')} "
        f"({metrics.get('preset')}), beard={metrics.get('beard')}")]


STAGE = StageSpec(
    name="c_face",
    version="1",
    summary="choose and record the head identity for each character",
    needs=["b_refs"],
    steps=[Step("host")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(run))
