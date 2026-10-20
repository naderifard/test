"""Gold leaf, which the shared vocabulary cannot express.

A Persian miniature's gold is not a metal in the PBR sense: it is beaten leaf
laid on paper, so it reads as a *flat* gold field that glints only where the
surface turns away from the eye. Simulating it with a metallic Principled goes
wrong twice over here -- this pack renders in EEVEE with raytracing off, so a
metal has nothing to reflect and comes out muddy, and even with reflections a
mirror finish looks like jewellery rather than paint.

So the leaf is painted: an emission of the pack's gold, broken up by a high
frequency noise so it looks hand-laid rather than printed, lifted toward a
burnish colour at grazing angles by a Layer Weight. That keeps it immune to the
lighting rig, which is the same bargain the pack's ``flat`` shader makes.
"""

from __future__ import annotations

import bpy

from .. import shading_common as common

GOLD_SHADERS = ("gold_leaf", "leaf", "burnished")


def build_material(pack, role, recipe, engine, name):
    """Style hook. Returns a material for gold-leaf recipes, else ``None``."""
    if (recipe.get("shader") or "").lower() not in GOLD_SHADERS:
        return None

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()

    base = common._rgba(recipe.get("base_color", [0.83, 0.69, 0.22]))
    laid = tuple(value * 0.78 for value in base[:3]) + (1.0,)
    burnish = tuple(min(1.0, value * 1.35 + 0.08)
                    for value in base[:3]) + (1.0,)

    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.location = (-900, -200)
    common._set(noise, "Scale", 120.0)
    common._set(noise, "Detail", 2.0)
    common._set(noise, "Roughness", 0.7)

    unevenness = tree.nodes.new("ShaderNodeMath")
    unevenness.operation = "MULTIPLY"
    unevenness.location = (-700, -200)
    tree.links.new(noise.outputs["Factor"], unevenness.inputs[0])
    unevenness.inputs[1].default_value = 0.45       # keep the break-up subtle

    field = tree.nodes.new("ShaderNodeMixRGB")
    field.blend_type = "MIX"
    field.location = (-500, 0)
    field.inputs["Color1"].default_value = base
    field.inputs["Color2"].default_value = laid
    tree.links.new(unevenness.outputs["Value"], field.inputs["Factor"])

    weight = tree.nodes.new("ShaderNodeLayerWeight")
    weight.location = (-700, 220)
    common._set(weight, "Blend", 0.32)

    glint = tree.nodes.new("ShaderNodeMixRGB")
    glint.blend_type = "MIX"
    glint.location = (-250, 0)
    glint.inputs["Color2"].default_value = burnish
    tree.links.new(field.outputs["Color"], glint.inputs["Color1"])
    tree.links.new(weight.outputs["Facing"], glint.inputs["Factor"])

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.location = (50, 0)
    common._set(emission, "Strength",
                1.0 + float(recipe.get("emission", 0.0) or 0.0))
    tree.links.new(glint.outputs["Color"], emission.inputs["Color"])

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (300, 0)
    tree.links.new(emission.outputs[0], output.inputs["Surface"])

    if hasattr(material, "use_backface_culling"):
        material.use_backface_culling = False
    material["style_shader"] = "gold_leaf"
    return material
