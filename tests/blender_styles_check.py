"""Measure the styles layer inside Blender. Run headless:

    blender -b --factory-startup --python tests/blender_styles_check.py

Builds every material role and the compositor group for all three packs and
asserts what the render actually depends on -- an Output node with a linked
Surface, a compositor group whose output traces back to Render Layers -- rather
than trusting that the builders ran without raising. Exits non-zero on failure.
"""

import os
import sys

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from blender_ai_studio import styles                            # noqa: E402
from blender_ai_studio.styles import postfx_common, shading_common  # noqa: E402

FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)
    return bool(condition)


def upstream(node, seen=None):
    """Every node feeding ``node``, however indirectly."""
    seen = seen if seen is not None else set()
    for socket in node.inputs:
        for link in socket.links:
            source = link.from_node
            if source.name in seen:
                continue
            seen.add(source.name)
            upstream(source, seen)
    return seen


def check_materials(pack, engine):
    built = 0
    for role in styles.ROLES:
        try:
            material = shading_common.material_for(pack, role, engine)
        except Exception as exc:                                # noqa: BLE001
            FAILURES.append(f"{pack.name}/{role}: raised {exc!r}")
            continue
        tree = material.node_tree
        outputs = [n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"]
        if not check(len(outputs) == 1,
                     f"{pack.name}/{role}: {len(outputs)} output nodes"):
            continue
        surface = outputs[0].inputs["Surface"]
        if not check(bool(surface.links),
                     f"{pack.name}/{role}: Surface not connected"):
            continue
        feeders = upstream(outputs[0])
        check(len(feeders) >= 1, f"{pack.name}/{role}: nothing upstream")
        shader = material.get("style_shader")
        check(shader is not None, f"{pack.name}/{role}: no style_shader tag")
        if shader in ("flat", "toon"):
            ramps = [n for n in tree.nodes if n.type == "VALTORGB"]
            check(bool(ramps), f"{pack.name}/{role}: flat without a ramp")
            for ramp in ramps:
                check(ramp.color_ramp.interpolation == "CONSTANT",
                      f"{pack.name}/{role}: ramp is not CONSTANT")
        built += 1
    print(f"   materials: {built}/{len(styles.ROLES)} built")
    return built


def check_postfx(pack, scene, expect):
    try:
        applied = postfx_common.build(pack, scene)
    except Exception as exc:                                    # noqa: BLE001
        FAILURES.append(f"{pack.name}/postfx: raised {exc!r}")
        return []
    tree = scene.compositing_node_group
    if not check(tree is not None, f"{pack.name}/postfx: no group assigned"):
        return applied
    outs = [n for n in tree.nodes if n.type == "GROUP_OUTPUT"]
    if not check(len(outs) == 1,
                 f"{pack.name}/postfx: {len(outs)} group outputs"):
        return applied
    linked = bool(outs[0].inputs[0].links)
    check(linked, f"{pack.name}/postfx: group output not connected")
    if linked:
        feeders = upstream(outs[0])
        kinds = {tree.nodes[name].type for name in feeders}
        check("R_LAYERS" in kinds,
              f"{pack.name}/postfx: output does not trace back to Render Layers")
    orphans = [n.name for n in tree.nodes
               if n.type not in ("GROUP_OUTPUT", "R_LAYERS", "IMAGE",
                                 "ELLIPSEMASK")
               and not any(s.links for s in n.outputs)]
    check(not orphans, f"{pack.name}/postfx: dead nodes {orphans}")
    for token in expect:
        check(any(entry.startswith(token) for entry in applied),
              f"{pack.name}/postfx: expected {token!r} in {applied}")
    check_modes(pack, tree)
    print(f"   postfx: {len(tree.nodes)} nodes, applied={applied}")
    return applied


# A menu socket that silently keeps its default is the failure mode that hurts
# most: a Filter left on "Soften" still builds a clean-looking chain and simply
# produces no outline. So the modes get measured, not assumed.
MODES = {"FILTER": ("Type", "Sobel"),
         "BLUR": ("Type", "Fast Gaussian"),
         "ELLIPSEMASK": ("Operation", "Add"),
         "COLORBALANCE": ("Type", "Lift/Gamma/Gain"),
         "KUWAHARA": ("Type", "Anisotropic")}


def check_modes(pack, tree):
    seen = []
    for node in tree.nodes:
        wanted = MODES.get(node.type)
        if wanted is None:
            continue
        socket = postfx_common._socket(node, wanted[0])
        if socket is None:
            continue
        got = socket.default_value
        check(got == wanted[1],
              f"{pack.name}/postfx: {node.type}.{wanted[0]} is {got!r}, "
              f"wanted {wanted[1]!r}")
        seen.append(f"{node.type}={got}")
    if seen:
        print("   modes:", ", ".join(seen))


EXPECTED = {
    "miniature": ("outline@", "contrast=", "saturation=", "paper="),
    "stylized": ("contrast=", "saturation=", "grain=", "vignette="),
    "photoreal": ("grain=", "vignette="),
}


def main():
    scene = bpy.context.scene
    scene.render.resolution_x = 640
    scene.render.resolution_y = 360
    for name in styles.PACKS:
        print(f"== {name}")
        pack = styles.load(name)
        engine = pack.render.get("engine", "BLENDER_EEVEE")
        try:
            scene.render.engine = engine
        except (TypeError, ValueError) as exc:
            FAILURES.append(f"{name}: engine {engine} not settable ({exc})")
        check_materials(pack, engine)
        check_postfx(pack, scene, EXPECTED[name])

    print(f"\nmaterials in file: {len(bpy.data.materials)}")
    dupes = [m.name for m in bpy.data.materials if m.name[-4:-3] == "."
             and m.name[-3:].isdigit()]
    check(not dupes, f"materials rebuilt instead of reused: {dupes}")
    if FAILURES:
        print(f"\nFAILED ({len(FAILURES)}):")
        for line in FAILURES:
            print("  -", line)
        sys.exit(1)
    print("\nOK: every role and every compositor chain measured clean")


main()
