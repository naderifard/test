"""Stage C-paint — paint each character's colour with a video model.

A photo-built character's colour came from projecting its front photo and the
turnaround views onto the shape, and those are separate drawings that disagree
about where things are: some features came out twice (two eyes, two ears)
however the views were blended. Here the character's own shape turns a full
circle as a depth video and Wan2.1-VACE paints that turn, holding the front
photo as the first and last frame and as its reference. A video model keeps a
subject consistent from frame to frame and the depth keeps every frame on the
geometry, so each feature exists once and turns with the shape. The frames
are projected back into a texture (``pipeline.character_paint``).

Steps: Blender renders the depth turn, a front view and the texture space;
the motion env runs Wan and the projection; Blender applies the texture and
saves ``painted.blend`` -- the file the shot builder reads. A character with
no photo, or a machine without the video model, passes through unpainted.
"""

from __future__ import annotations

import os
import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio

FRAMES = 33
SIZE = (480, 832)
STEPS = 25
SEED = 42
FRONT_SCALE = 2
TEXTURE_SIZE = 2048
UNSEEN_WARN = 0.30

FRAME = ("Full body, plain grey studio background, soft even light, sharp "
         "natural detail, photorealistic.")
LOOK = ("The same face, hair, clothing, colours and patterns from every "
        "side. " + FRAME)
# No clothing for an animal: asked for "the same clothing", Wan put a
# blanket round the lion's middle.
ANIMAL_LOOK = ("The same face, mane, fur, colours and markings from every "
               "side, bare fur with nothing worn. " + FRAME)
HUMAN_TURN = ("A 360-degree turntable of the character from the reference "
              "image, standing perfectly still in an A-pose while the camera "
              "circles around: ")
ANIMAL_TURN = ("A 360-degree turntable of the animal from the reference image, "
               "standing perfectly still on all four legs while the camera "
               "circles around it: ")


def turn_prompt(character):
    """What Wan is asked to paint: the turn, the character's own description
    from the production plan, and the look."""
    animal = character.get("kind") == "quadruped"
    start, end = (ANIMAL_TURN, ANIMAL_LOOK) if animal else (HUMAN_TURN, LOOK)
    look = (character.get("image_prompt") or "").strip()
    if look and not look.endswith("."):
        look += "."
    return start + (look + " " if look else "") + end


def frame_size(character):
    """(width, height): portrait for a figure standing up, landscape for an
    animal on four legs -- in a portrait frame the lion's turn filled a
    fifth of it."""
    return SIZE[::-1] if character.get("kind") == "quadruped" else SIZE


def plan(project, graph):
    from pipeline import wan_vace
    ready = wan_vace.available()
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    units = []
    for character in resolved.get("characters") or []:
        cid = character["id"]
        base = ("assets", "characters", cid)
        width, height = frame_size(character)
        units.append(Unit(
            f"char_{cid}",
            inputs={"rigged": project.path(*base, "rigged", "rigged.blend"),
                    "photo": project.path(*base, "asset", "photo.png")},
            params={"subject": cid, "kind": character.get("kind", "human"),
                    "paint": ready,
                    "why": None if ready else
                    "Wan2.1-VACE is not installed; the character keeps its "
                    "projected colour",
                    "prompt": turn_prompt(character),
                    "frames": FRAMES, "width": width, "height": height,
                    "steps": STEPS, "seed": SEED, "front_scale": FRONT_SCALE,
                    "texture_size": TEXTURE_SIZE},
            out_rel=f"assets/characters/{cid}/painted",
            label=f"painted/{cid}",
        ))
    return units


def nothing_to_do(project):
    return "no resolved production plan — a_production has not run"


_LOADED = {}


def _wan(job):
    """Encode every unit's prompt once, then load the model once, for the
    whole job: the text encoder and the video model never share memory."""
    if "vace" not in _LOADED:
        from pipeline import wan_vace
        prompts = sorted({u["params"]["prompt"] for u in job.units
                          if u["params"].get("paint")})
        table = wan_vace.encode_prompts(prompts + [wan_vace.negative_prompt()])
        _LOADED["vace"] = wan_vace.load(table)
    return _LOADED["vace"]


def paint(job, unit):
    """Wan over the depth turn, then the projection into the texture."""
    import numpy as np
    from PIL import Image

    from pipeline import character_paint as cp
    params = unit["params"]
    out = unit["out_dir"]
    report_path = os.path.join(out, "paint.json")
    photo = unit["inputs"]["photo"]
    if not params.get("paint"):
        jsonio.write(report_path, {"why": params.get("why")})
        return {"metrics": {"painted": False}}
    if not os.path.exists(photo):
        why = "no photo: this character was not built from one"
        jsonio.write(report_path, {"why": why})
        return {"metrics": {"painted": False}, "notes": [why]}
    turn = os.path.join(out, "turn")
    bake = os.path.join(out, "bake")
    _wan(job)
    frames = cp.generate_turn(turn, params["prompt"], photo,
                              steps=params["steps"], seed=params["seed"])
    position = np.load(os.path.join(bake, "position.npy"))
    normal = np.load(os.path.join(bake, "normal.npy"))
    valid = np.load(os.path.join(bake, "valid.npy"))[..., 0] > 0.5
    hand_path = os.path.join(bake, "hand.npy")
    hand = np.load(hand_path)[..., 0] if os.path.exists(hand_path) else None
    head_path = os.path.join(bake, "head.npy")
    head = np.load(head_path)[..., 0] if os.path.exists(head_path) else None
    front = os.path.join(out, "front", "colour_0000.png")
    scale = Image.open(front).width / params["width"]
    texture, _weight, report = cp.bake_texture(
        turn, list(frames), position, normal, valid, front_path=front,
        front_scale=scale, hand=hand, head=head)
    Image.fromarray(cp.dilate(texture, valid, 8)).save(
        _base.out_path(unit, "texture.png"))
    metrics = {"texels": report["texels"],
               "unseen_share": report["unseen_share"],
               "front_share": report.get("front_share")}
    jsonio.write(report_path, {"texture": "texture.png", "metrics": metrics})
    return {"metrics": metrics}


def _gate(project, unit, metrics):
    if not metrics.get("painted"):
        return [gates.Verdict(gates.WARN, "paint.skipped",
                              "the character keeps its projected colour",
                              False, True)]
    out = []
    left = metrics.get("uv_overlap_triangles")
    if left:
        out.append(gates.Verdict(
            gates.WARN, "paint.uv_overlap",
            f"{left} triangles still share texels with another part of the "
            "body; their colour may come from there", left, 0))
    unseen = metrics.get("unseen_share") or 0.0
    if unseen > UNSEEN_WARN:
        out.append(gates.Verdict(
            gates.WARN, "paint.unseen",
            f"{unseen:.0%} of the surface was seen by no frame and took its "
            "neighbours' colour", unseen, UNSEEN_WARN))
    return out or [gates.Verdict(
        gates.PASS, "paint",
        f"{metrics.get('texels')} texels painted, {unseen:.0%} filled from "
        f"neighbours, {metrics.get('front_share') or 0:.0%} from the photo")]


STAGE = StageSpec(
    name="c_paint",
    # 3: the head painted from one source: the photo, else the best frame.
    version="3",
    summary="paint each character's colour with a video model turning round "
            "its shape",
    needs=["c_rigged"],
    steps=[Step("blender", "pipeline/blender/paint_stage.py"),
           Step("motion"),
           Step("blender", "pipeline/blender/paint_stage.py")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(paint, paint, paint))
