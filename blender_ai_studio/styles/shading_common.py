"""Turn a style pack's material recipe into a Blender node tree.

Imported inside Blender only. Every pack shares this builder; a pack's own
``shading.py`` handles what the shared vocabulary cannot express and delegates
the rest here, so the three packs do not carry three copies of "how to wire a
Principled BSDF".

Socket names are the Blender 5.x Principled ones ("Subsurface Weight",
"Specular IOR Level", ...) and are looked up defensively, because a rename
between versions should degrade to "that parameter was ignored" rather than
crash a render at stage j.
"""

from __future__ import annotations

import bpy

# ``flat``/toon shading routes a BSDF through Shader-to-RGB, which only exists
# in EEVEE's pipeline. In Cycles that node renders black, so the builder
# substitutes a plain Principled instead of quietly producing a black character.
TOON_ENGINES = ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT")


def _rgba(value, alpha=1.0):
    seq = list(value or (0.8, 0.8, 0.8))
    while len(seq) < 3:
        seq.append(seq[-1] if seq else 0.8)
    return (seq[0], seq[1], seq[2], seq[3] if len(seq) > 3 else alpha)


def _set(node, socket, value):
    """Set a socket if this Blender has it. Returns whether it landed."""
    entry = node.inputs.get(socket)
    if entry is None:
        return False
    try:
        if hasattr(entry.default_value, "__len__") and not isinstance(
                entry.default_value, (str, bytes)):
            entry.default_value = _rgba(value)
        else:
            entry.default_value = float(value)
    except (TypeError, ValueError):
        return False
    return True


def material_for(pack, role, engine="BLENDER_EEVEE", suffix=""):
    """Get or build the material for a role. Reused across every object.

    Materials are named ``<pack>_<role>`` so one Rostam mesh and one Sohrab mesh
    that both ask for ``skin`` share a single material, which keeps the .blend
    small and makes a style swap a matter of relinking rather than rebuilding.
    """
    name = f"{pack.name}_{role}{suffix}"
    existing = bpy.data.materials.get(name)
    if existing is not None:
        return existing

    recipe = pack.material(role)
    hook = pack.hook("shading")
    if hook is not None and hasattr(hook, "build_material"):
        built = hook.build_material(pack, role, recipe, engine, name)
        if built is not None:
            return built
    return build(name, recipe, engine)


def build(name, recipe, engine="BLENDER_EEVEE"):
    """Build one material from a recipe dict."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)

    shader = (recipe.get("shader") or "principled").lower()
    if shader in ("flat", "toon") and engine not in TOON_ENGINES:
        shader = "principled"

    if shader == "emission":
        node = _emission(tree, recipe)
    elif shader == "glass":
        node = _glass(tree, recipe)
    elif shader in ("flat", "toon"):
        node = _toon(tree, recipe)
    else:
        node = _principled(tree, recipe)

    tree.links.new(node.outputs[0], output.inputs["Surface"])
    _apply_settings(material, recipe, shader)
    return material


def _principled(tree, recipe):
    node = tree.nodes.new("ShaderNodeBsdfPrincipled")
    node.location = (100, 0)
    _set(node, "Base Color", recipe.get("base_color", [0.8, 0.8, 0.8]))
    _set(node, "Roughness", recipe.get("roughness", 0.5))
    _set(node, "Metallic", recipe.get("metallic", 0.0))
    _set(node, "IOR", recipe.get("ior", 1.45))
    if "specular" in recipe:
        _set(node, "Specular IOR Level", recipe["specular"])
    if "sheen" in recipe:
        _set(node, "Sheen Weight", recipe["sheen"])
        _set(node, "Sheen Roughness", 0.35)
    if "anisotropy" in recipe:
        _set(node, "Anisotropic", recipe["anisotropy"])
    if "transmission" in recipe:
        _set(node, "Transmission Weight", recipe["transmission"])
    if recipe.get("subsurface"):
        _set(node, "Subsurface Weight", recipe["subsurface"])
        _set(node, "Subsurface Scale", recipe.get("subsurface_scale", 0.12))
        radius = recipe.get("subsurface_radius")
        if radius:
            entry = node.inputs.get("Subsurface Radius")
            if entry is not None:
                entry.default_value = tuple(radius[:3])
    if recipe.get("translucency"):
        # Cheap two-sided leaf look: lift diffuse roughness and thin the wall.
        _set(node, "Diffuse Roughness", min(1.0, recipe["translucency"] + 0.3))
        _set(node, "Thin Wall", 1.0)
    if recipe.get("emission"):
        _set(node, "Emission Color", recipe.get("emission_color",
                                                recipe.get("base_color")))
        _set(node, "Emission Strength", recipe["emission"])
    if recipe.get("noise_bump"):
        _bump(tree, node, recipe["noise_bump"])
    return node


def _bump(tree, target, strength, scale=None):
    """Procedural surface break-up, so a flat primitive is not flatly lit."""
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.location = (-500, -300)
    _set(noise, "Scale", scale or 14.0)
    _set(noise, "Detail", 6.0)
    _set(noise, "Roughness", 0.55)
    bump = tree.nodes.new("ShaderNodeBump")
    bump.location = (-200, -300)
    _set(bump, "Strength", min(1.0, float(strength)))
    _set(bump, "Distance", 0.04)
    tree.links.new(noise.outputs["Factor"], bump.inputs["Height"])
    normal = target.inputs.get("Normal")
    if normal is not None:
        tree.links.new(bump.outputs["Normal"], normal)
    return bump


def _emission(tree, recipe):
    node = tree.nodes.new("ShaderNodeEmission")
    node.location = (100, 0)
    _set(node, "Color", recipe.get("base_color", [1, 1, 1]))
    _set(node, "Strength", recipe.get("strength", 1.0))
    return node


def _glass(tree, recipe):
    node = tree.nodes.new("ShaderNodeBsdfGlass")
    node.location = (100, 0)
    _set(node, "Color", recipe.get("base_color", [0.8, 0.9, 0.9]))
    _set(node, "Roughness", recipe.get("roughness", 0.05))
    _set(node, "IOR", recipe.get("ior", 1.33))
    return node


def _toon(tree, recipe):
    """Banded shading: diffuse -> Shader to RGB -> stepped ramp -> emission.

    The ramp is CONSTANT-interpolated so the terminator is a hard edge, which is
    what makes the miniature pack read as paint rather than as a 3D render with
    the lights turned down.
    """
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.location = (-700, 0)
    _set(diffuse, "Color", [1.0, 1.0, 1.0])
    _set(diffuse, "Roughness", 0.0)

    to_rgb = tree.nodes.new("ShaderNodeShaderToRGB")
    to_rgb.location = (-500, 0)
    tree.links.new(diffuse.outputs["BSDF"], to_rgb.inputs["Shader"])

    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-300, 0)
    ramp.color_ramp.interpolation = "CONSTANT"
    steps = max(1, int(recipe.get("steps", 2)))
    shade_at = float(recipe.get("shade_at", 0.5))
    base = _rgba(recipe.get("base_color", [0.8, 0.8, 0.8]))
    shade = _rgba(recipe.get("shade_color", recipe.get("base_color")))
    elements = ramp.color_ramp.elements
    elements[0].position = 0.0
    elements[0].color = shade
    elements[1].position = min(0.999, max(0.001, shade_at))
    elements[1].color = base
    for index in range(1, steps - 1):
        mid = elements.new(shade_at * (index / max(1, steps - 1)))
        blend = index / max(1, steps - 1)
        mid.color = tuple(shade[i] + (base[i] - shade[i]) * blend * 0.6
                          for i in range(4))
    tree.links.new(to_rgb.outputs["Color"], ramp.inputs["Factor"])

    node = tree.nodes.new("ShaderNodeEmission")
    node.location = (100, 0)
    _set(node, "Strength", 1.0 + float(recipe.get("emission", 0.0)))
    tree.links.new(ramp.outputs["Color"], node.inputs["Color"])
    return node


def _apply_settings(material, recipe, shader):
    """Material-level flags the node tree cannot carry."""
    if shader == "glass" or recipe.get("transmission"):
        material.use_backface_culling = False
        for attr, value in (("blend_method", "BLEND"),
                            ("shadow_method", "HASHED")):
            if hasattr(material, attr):
                try:
                    setattr(material, attr, value)
                except (TypeError, ValueError):
                    pass
    if shader in ("flat", "toon", "emission"):
        # Flat and emissive surfaces do not want to catch a specular highlight
        # from the rig; they already encode their own lighting.
        if hasattr(material, "use_backface_culling"):
            material.use_backface_culling = False
    material["style_shader"] = shader
    return material





