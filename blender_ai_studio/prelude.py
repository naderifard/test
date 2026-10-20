"""Helper library installed INSIDE Blender as the module ``bstudio``.

The scene-building model is small and text-only. Every low-level Blender API
detail it has to remember is a chance to fail, and it cannot look at the
viewport to notice that it failed. So this module does two jobs:

1. Collapse the fiddly, version-sensitive parts of ``bpy`` (Blender 5.x node
   sockets, colour management, EEVEE settings, the 5.x video-output split)
   into a handful of forgiving calls that take degrees, tuples and names.
2. Give the model a substitute for eyesight - :func:`report` describes the
   scene as text, and :func:`analyse` renders a thumbnail and measures it, so
   "the frame is 97% black" comes back as a fact it can act on.

Injected verbatim by ``session.install_prelude``; it is never imported by the
host application.
"""

import json
import math
import os

import bpy
import mathutils
from bpy_extras.object_utils import world_to_camera_view

__all__ = [
    "reset", "set_engine", "set_output", "set_view_transform", "set_frames",
    "enable_gpu_compute",
    "pbr", "glass", "emission_material", "procedural", "assign", "unwrap",
    "world_color", "world_gradient", "sky", "hdri_file", "sun", "area_light",
    "point_light", "spot_light", "three_point",
    "add", "floor", "text", "smooth", "bevel", "subdiv", "solidify", "place",
    "look_at", "scatter", "join", "boolean",
    "camera", "frame_all", "orbit_camera",
    "key", "anim_orbit", "anim_spin", "ease",
    "report", "analyse", "auto_exposure", "bounds", "obj",
    # the character layer
    "character", "apply_motion", "apply_visemes", "bind_garment", "style",
    "load_style", "role_material",
    "SMPLX_JOINTS", "SMPLX_PARENTS", "SMPLX_REST", "QUAD_JOINTS",
    "QUAD_PARENTS", "QUAD_REST", "VISEME_CHANNELS", "SHAPE_PRESETS",
    "DEFAULT_PART_ROLES",
]

_D = math.radians


# --------------------------------------------------------------------------
# small coercions - the model is sloppy about types, so meet it halfway
# --------------------------------------------------------------------------
def obj(target):
    """Accept an object or its name; raise a message that names the mistake."""
    if isinstance(target, str):
        found = bpy.data.objects.get(target)
        if found is None:
            raise KeyError(
                "No object named %r. Existing objects: %s"
                % (target, [o.name for o in bpy.data.objects])
            )
        return found
    return target


def _vec(value, default=(0.0, 0.0, 0.0)):
    if value is None:
        return mathutils.Vector(default)
    # A name is a legal stand-in for a position: the helper reference promises
    # the model that anything called `target` takes an object or its name, and
    # `B.camera(target="Ball")` reaches this function through look_at().
    if isinstance(value, str):
        value = obj(value)
    if hasattr(value, "matrix_world"):
        return value.matrix_world.translation.copy()
    if isinstance(value, (int, float)):
        return mathutils.Vector((value, value, value))
    return mathutils.Vector(tuple(float(v) for v in value))


def _rgba(color, alpha=1.0):
    if color is None:
        return (0.8, 0.8, 0.8, alpha)
    if isinstance(color, str):
        h = color.lstrip("#")
        rgb = tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        # hex is sRGB; Blender sockets want linear
        rgb = tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)
        return rgb + (alpha,)
    color = tuple(float(c) for c in color)
    if len(color) == 3:
        return color + (alpha,)
    return color


def _bsdf(mat):
    for node in mat.node_tree.nodes:
        if node.bl_idname == "ShaderNodeBsdfPrincipled":
            return node
    return mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")


# --------------------------------------------------------------------------
# scene setup
# --------------------------------------------------------------------------
def reset(engine="EEVEE", res=(1920, 1080), fps=24):
    """Delete everything and start from a clean, sane scene."""
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.curves,
                  bpy.data.lights, bpy.data.cameras, bpy.data.images):
        for item in list(block):
            if getattr(item, "users", 0) == 0:
                block.remove(item)
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    world_color((0.05, 0.05, 0.06), 1.0)
    set_engine(engine)
    set_output(res[0], res[1], fps)
    set_view_transform("AgX")
    return scene


def enable_gpu_compute(prefer=("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL")):
    """Turn on the best available Cycles GPU backend.

    Returns ``(backend, [device names])``, or ``(None, [])`` when the machine
    genuinely has no GPU backend and Cycles has to fall back to the CPU.

    Blender only exposes a backend once its addon preferences have been asked
    for devices of that type, so the order here matters: set the type, pull the
    device list, then enable the non-CPU entries.
    """
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        return (None, [])

    try:
        available = {row[0] for row in prefs.get_device_types(bpy.context)}
    except Exception:
        available = set(prefer)

    for backend in prefer:
        if backend not in available:
            continue
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue  # not compiled into this build
        try:
            devices = list(prefs.get_devices_for_type(backend))
        except Exception:
            try:
                prefs.refresh_devices()
            except Exception:
                pass
            devices = [d for d in getattr(prefs, "devices", []) if d.type == backend]
        # Enable the accelerators, leave the CPU off: mixing the two costs more
        # memory than it buys speed, and this is an 8 GB card.
        names = []
        for device in devices:
            use = device.type != "CPU"
            try:
                device.use = use
            except Exception:
                continue
            if use:
                names.append(device.name)
        if names:
            return (backend, names)
    return (None, [])


def set_engine(engine="EEVEE", samples=None):
    """``EEVEE`` (fast, the default) or ``CYCLES`` (physically accurate)."""
    scene = bpy.context.scene
    name = str(engine).upper()
    # Substring, not prefix: the style packs declare Blender's own identifier
    # ("BLENDER_EEVEE"), which does not start with "EEVEE" and would otherwise
    # fall through to Cycles - silently rendering the fast preview style on the
    # slow engine.
    if "EEVEE" in name:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = int(samples or 64)
        scene.eevee.taa_samples = 16
        for flag, value in (("use_raytracing", True), ("use_shadows", True)):
            if hasattr(scene.eevee, flag):
                try:
                    setattr(scene.eevee, flag, value)
                except Exception:
                    pass
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = int(samples or 96)
        scene.cycles.use_denoising = True
        backend, devices = enable_gpu_compute()
        try:
            scene.cycles.device = "GPU" if devices else "CPU"
        except Exception:
            pass
        if devices:
            print("Cycles: %s on %s" % (backend, ", ".join(devices)))
        else:
            print("Cycles: no GPU backend found, rendering on the CPU")
    return scene.render.engine


def set_output(res_x=1920, res_y=1080, fps=24):
    scene = bpy.context.scene
    scene.render.resolution_x = int(res_x)
    scene.render.resolution_y = int(res_y)
    scene.render.resolution_percentage = 100
    scene.render.fps = int(fps)
    scene.render.film_transparent = False
    return (scene.render.resolution_x, scene.render.resolution_y)


def set_view_transform(name="AgX", look="None", exposure=0.0):
    """AgX (filmic, forgiving), Standard, Filmic, Khronos PBR Neutral, Raw."""
    scene = bpy.context.scene
    try:
        scene.view_settings.view_transform = name
    except Exception:
        scene.view_settings.view_transform = "Standard"
    try:
        scene.view_settings.look = look
    except Exception:
        pass
    scene.view_settings.exposure = float(exposure)


def set_frames(start=1, end=96, fps=24):
    scene = bpy.context.scene
    scene.frame_start = int(start)
    scene.frame_end = int(end)
    scene.render.fps = int(fps)
    scene.frame_set(int(start))
    return (scene.frame_start, scene.frame_end)


# --------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------
def pbr(name, color=(0.8, 0.8, 0.8), roughness=0.5, metallic=0.0, ior=1.45,
        alpha=1.0, transmission=0.0, emission=None, emission_strength=0.0,
        coat=0.0, sheen=0.0, subsurface=0.0):
    """A Principled material. Socket names here are the Blender 4.x/5.x ones."""
    mat = bpy.data.materials.new(name)
    node = _bsdf(mat)
    node.inputs["Base Color"].default_value = _rgba(color)
    node.inputs["Roughness"].default_value = float(roughness)
    node.inputs["Metallic"].default_value = float(metallic)
    node.inputs["IOR"].default_value = float(ior)
    node.inputs["Alpha"].default_value = float(alpha)
    node.inputs["Transmission Weight"].default_value = float(transmission)
    node.inputs["Coat Weight"].default_value = float(coat)
    node.inputs["Sheen Weight"].default_value = float(sheen)
    node.inputs["Subsurface Weight"].default_value = float(subsurface)
    if emission is not None or emission_strength:
        node.inputs["Emission Color"].default_value = _rgba(emission or color)
        node.inputs["Emission Strength"].default_value = float(emission_strength or 1.0)
    if alpha < 1.0 or transmission > 0.0:
        mat.blend_method = "BLEND" if hasattr(mat, "blend_method") else mat.blend_method
    return mat


def glass(name, color=(1.0, 1.0, 1.0), roughness=0.05, ior=1.45):
    return pbr(name, color=color, roughness=roughness, transmission=1.0, ior=ior)


def emission_material(name, color=(1.0, 0.9, 0.7), strength=8.0):
    return pbr(name, color=color, roughness=0.5, emission=color, emission_strength=strength)


_PROCEDURAL_PRESETS = {
    # kind: (base, accent, roughness, metallic, scale, distortion, detail, bump)
    "wood":     ((0.28, 0.15, 0.07), (0.52, 0.33, 0.16), 0.55, 0.0, 3.0, 2.0, 6.0, 0.20),
    "marble":   ((0.86, 0.85, 0.83), (0.30, 0.30, 0.32), 0.18, 0.0, 2.2, 4.5, 8.0, 0.05),
    "rust":     ((0.24, 0.10, 0.05), (0.55, 0.28, 0.12), 0.85, 0.55, 9.0, 1.5, 10.0, 0.45),
    "concrete": ((0.42, 0.41, 0.39), (0.28, 0.27, 0.26), 0.88, 0.0, 14.0, 0.6, 12.0, 0.30),
    "fabric":   ((0.32, 0.16, 0.18), (0.45, 0.24, 0.26), 0.95, 0.0, 45.0, 0.4, 4.0, 0.35),
    "metal":    ((0.55, 0.56, 0.58), (0.38, 0.39, 0.41), 0.28, 1.0, 20.0, 0.5, 6.0, 0.08),
    "stone":    ((0.34, 0.33, 0.31), (0.20, 0.19, 0.18), 0.80, 0.0, 6.0, 1.2, 10.0, 0.40),
    "sand":     ((0.72, 0.60, 0.40), (0.55, 0.44, 0.28), 0.92, 0.0, 60.0, 0.3, 8.0, 0.25),
    "dirt":     ((0.20, 0.14, 0.09), (0.32, 0.23, 0.14), 0.93, 0.0, 12.0, 1.0, 10.0, 0.40),
    "plastic":  ((0.10, 0.35, 0.75), (0.06, 0.22, 0.50), 0.32, 0.0, 30.0, 0.2, 3.0, 0.04),
    "leather":  ((0.18, 0.10, 0.07), (0.28, 0.17, 0.11), 0.72, 0.0, 55.0, 0.8, 8.0, 0.30),
    "ice":      ((0.80, 0.90, 0.98), (0.60, 0.75, 0.90), 0.10, 0.0, 8.0, 2.0, 6.0, 0.15),
}


def procedural(name, kind="stone", color=None, accent=None, scale=None,
               roughness=None, metallic=None, bump=None):
    """A textured material built from noise/voronoi nodes - no downloads needed.

    ``kind`` is one of: wood, marble, rust, concrete, fabric, metal, stone,
    sand, dirt, plastic, leather, ice.
    """
    preset = _PROCEDURAL_PRESETS.get(str(kind).lower(), _PROCEDURAL_PRESETS["stone"])
    p_base, p_accent, p_rough, p_metal, p_scale, p_dist, p_detail, p_bump = preset
    base = _rgba(color if color is not None else p_base)
    acc = _rgba(accent if accent is not None else p_accent)
    scale = float(p_scale if scale is None else scale)
    rough = float(p_rough if roughness is None else roughness)
    metal = float(p_metal if metallic is None else metallic)
    bump_amt = float(p_bump if bump is None else bump)

    mat = bpy.data.materials.new(name)
    tree = mat.node_tree
    nodes, links = tree.nodes, tree.links
    bsdf = _bsdf(mat)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal

    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    links.new(coord.outputs["Object"], mapping.inputs["Vector"])

    kind_l = str(kind).lower()
    if kind_l in ("stone", "concrete", "rust", "dirt", "sand"):
        tex = nodes.new("ShaderNodeTexVoronoi")
        tex.inputs["Scale"].default_value = scale
        value_out = tex.outputs["Distance"]
    elif kind_l == "wood":
        tex = nodes.new("ShaderNodeTexWave")
        tex.wave_type = "RINGS"
        tex.inputs["Scale"].default_value = scale
        tex.inputs["Distortion"].default_value = p_dist
        tex.inputs["Detail"].default_value = p_detail
        value_out = tex.outputs["Fac"]
    elif kind_l == "marble":
        tex = nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = scale
        tex.inputs["Detail"].default_value = p_detail
        tex.inputs["Distortion"].default_value = p_dist
        value_out = tex.outputs["Fac"]
    else:
        tex = nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = scale
        tex.inputs["Detail"].default_value = p_detail
        value_out = tex.outputs["Fac"]
    links.new(mapping.outputs["Vector"], tex.inputs["Vector"])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = base
    ramp.color_ramp.elements[1].color = acc
    links.new(value_out, ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # roughness breakup keeps highlights from looking CG-uniform
    rough_ramp = nodes.new("ShaderNodeMapRange")
    rough_ramp.inputs["To Min"].default_value = max(0.0, rough - 0.15)
    rough_ramp.inputs["To Max"].default_value = min(1.0, rough + 0.15)
    links.new(value_out, rough_ramp.inputs["Value"])
    links.new(rough_ramp.outputs["Result"], bsdf.inputs["Roughness"])

    if bump_amt > 0:
        bump_node = nodes.new("ShaderNodeBump")
        bump_node.inputs["Strength"].default_value = bump_amt
        links.new(value_out, bump_node.inputs["Height"])
        links.new(bump_node.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def assign(target, material, slot_clear=True):
    """Give ``target`` (object or name) this material."""
    ob = obj(target)
    if not hasattr(ob.data, "materials"):
        raise TypeError("%s cannot take a material" % ob.name)
    if slot_clear:
        ob.data.materials.clear()
    ob.data.materials.append(material)
    return ob


def unwrap(target, angle=66.0):
    """Smart UV project - required before any image texture will look right."""
    ob = obj(target)
    if ob.type != "MESH":
        return ob
    bpy.context.view_layer.objects.active = ob
    for other in bpy.context.selected_objects:
        other.select_set(False)
    ob.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=_D(angle), island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
    return ob


# --------------------------------------------------------------------------
# world and lighting
# --------------------------------------------------------------------------
def _world_tree():
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    tree = world.node_tree
    for node in list(tree.nodes):
        if node.bl_idname != "ShaderNodeOutputWorld":
            tree.nodes.remove(node)
    out = next((n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputWorld"), None)
    if out is None:
        out = tree.nodes.new("ShaderNodeOutputWorld")
    return tree, out


def world_color(color=(0.05, 0.05, 0.06), strength=1.0):
    tree, out = _world_tree()
    bg = tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = _rgba(color)
    bg.inputs["Strength"].default_value = float(strength)
    tree.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return bpy.context.scene.world


def world_gradient(top=(0.10, 0.16, 0.30), bottom=(0.02, 0.02, 0.03), strength=1.0):
    tree, out = _world_tree()
    bg = tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = float(strength)
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = _rgba(bottom)
    ramp.color_ramp.elements[1].color = _rgba(top)
    grad = tree.nodes.new("ShaderNodeTexGradient")
    grad.gradient_type = "EASING"
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (_D(90), 0, 0)
    coord = tree.nodes.new("ShaderNodeTexCoord")
    tree.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], grad.inputs["Vector"])
    tree.links.new(grad.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    tree.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return bpy.context.scene.world


def sky(sun_elevation=25.0, sun_rotation=135.0, strength=1.0, turbidity=2.5):
    """Physical sky - the most reliable way to get believable outdoor light."""
    tree, out = _world_tree()
    bg = tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = float(strength)
    tex = tree.nodes.new("ShaderNodeTexSky")
    try:
        tex.sky_type = "NISHITA"
    except Exception:
        pass
    for attr, value in (("sun_elevation", _D(sun_elevation)),
                        ("sun_rotation", _D(sun_rotation)),
                        ("turbidity", turbidity)):
        if hasattr(tex, attr):
            try:
                setattr(tex, attr, value)
            except Exception:
                pass
    tree.links.new(tex.outputs["Color"], bg.inputs["Color"])
    tree.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return bpy.context.scene.world


def hdri_file(path, strength=1.0, rotation=0.0):
    """Light the scene from an .hdr/.exr already on disk."""
    if not os.path.exists(path):
        raise FileNotFoundError("HDRI not found: %s" % path)
    tree, out = _world_tree()
    bg = tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = float(strength)
    env = tree.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(path, check_existing=True)
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0, 0, _D(rotation))
    coord = tree.nodes.new("ShaderNodeTexCoord")
    tree.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    tree.links.new(env.outputs["Color"], bg.inputs["Color"])
    tree.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return bpy.context.scene.world


def _new_light(kind, name, location, energy, color):
    data = bpy.data.lights.new(name, type=kind)
    data.energy = float(energy)
    data.color = _rgba(color)[:3]
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = _vec(location)
    return light


def sun(direction=(45.0, 0.0, 135.0), energy=3.0, color=(1.0, 0.96, 0.9), angle=2.0):
    """A sun lamp aimed by euler angles in degrees."""
    light = _new_light("SUN", "Sun", (0, 0, 12), energy, color)
    light.rotation_euler = tuple(_D(a) for a in direction)
    try:
        light.data.angle = _D(angle)
    except Exception:
        pass
    return light


def area_light(location=(4, -4, 5), target=(0, 0, 0), energy=500.0,
               size=3.0, color=(1.0, 1.0, 1.0), name="AreaLight"):
    light = _new_light("AREA", name, location, energy, color)
    light.data.size = float(size)
    look_at(light, target)
    return light


def point_light(location=(3, -3, 4), energy=300.0, color=(1.0, 1.0, 1.0),
                radius=0.25, name="PointLight"):
    light = _new_light("POINT", name, location, energy, color)
    light.data.shadow_soft_size = float(radius)
    return light


def spot_light(location=(4, -4, 6), target=(0, 0, 0), energy=800.0,
               spot_size=45.0, blend=0.3, color=(1.0, 1.0, 1.0), name="Spot"):
    light = _new_light("SPOT", name, location, energy, color)
    light.data.spot_size = _D(spot_size)
    light.data.spot_blend = float(blend)
    look_at(light, target)
    return light


def three_point(target=(0, 0, 0), distance=6.0, key=600.0, fill=180.0, rim=400.0,
                key_color=(1.0, 0.95, 0.88), fill_color=(0.85, 0.9, 1.0),
                rim_color=(1.0, 1.0, 1.0)):
    """Key / fill / rim around ``target`` - a safe default for any subject."""
    t = _vec(target)
    d = float(distance)
    lights = [
        area_light((t.x + d * 0.8, t.y - d * 0.8, t.z + d * 0.7), t, key, d * 0.5, key_color, "Key"),
        area_light((t.x - d * 0.9, t.y - d * 0.5, t.z + d * 0.3), t, fill, d * 0.7, fill_color, "Fill"),
        area_light((t.x - d * 0.3, t.y + d * 1.0, t.z + d * 0.8), t, rim, d * 0.35, rim_color, "Rim"),
    ]
    return lights


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
_PRIMITIVES = {
    "cube": "primitive_cube_add",
    "sphere": "primitive_uv_sphere_add",
    "uv_sphere": "primitive_uv_sphere_add",
    "icosphere": "primitive_ico_sphere_add",
    "cylinder": "primitive_cylinder_add",
    "cone": "primitive_cone_add",
    "torus": "primitive_torus_add",
    "plane": "primitive_plane_add",
    "circle": "primitive_circle_add",
    "monkey": "primitive_monkey_add",
    "grid": "primitive_grid_add",
}


# Each primitive operator spells its main dimension differently, and passing
# the wrong one is a hard TypeError from Blender rather than a warning.
_SIZE_ARG = {
    "primitive_cube_add": "size",
    "primitive_plane_add": "size",
    "primitive_grid_add": "size",
    "primitive_monkey_add": "size",
    "primitive_circle_add": "radius",
    "primitive_uv_sphere_add": "radius",
    "primitive_ico_sphere_add": "radius",
    "primitive_cylinder_add": "radius",
    "primitive_cone_add": "radius1",
    "primitive_torus_add": "major_radius",
}


def add(kind="cube", name=None, location=(0, 0, 0), rotation=(0, 0, 0),
        scale=(1, 1, 1), size=None, material=None, **kwargs):
    """Add a primitive. ``rotation`` is in DEGREES."""
    op_name = _PRIMITIVES.get(str(kind).lower())
    if op_name is None:
        raise ValueError("Unknown primitive %r. Choose from: %s"
                         % (kind, sorted(_PRIMITIVES)))
    op = getattr(bpy.ops.mesh, op_name)
    params = dict(kwargs)
    params["location"] = tuple(_vec(location))
    if size is not None:
        params.setdefault(_SIZE_ARG.get(op_name, "size"), float(size))

    # Say which keywords this primitive actually takes, instead of letting
    # Blender raise a bare "keyword unrecognized".
    allowed = {p.identifier for p in op.get_rna_type().properties}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise TypeError(
            "%s() does not accept %s. For kind=%r the accepted keywords are: %s"
            % (op_name, ", ".join(repr(u) for u in unknown), kind,
               ", ".join(sorted(allowed - {"rna_type"})))
        )
    op(**params)
    ob = bpy.context.object
    if name:
        ob.name = name
    ob.rotation_euler = tuple(_D(a) for a in _vec(rotation))
    ob.scale = _vec(scale, (1, 1, 1))
    if material is not None:
        assign(ob, material)
    return ob


def floor(size=40.0, material=None, z=0.0, name="Floor"):
    ground = add("plane", name=name, location=(0, 0, z), size=size)
    if material is not None:
        assign(ground, material)
    return ground


def text(body="Hello", name="Text", location=(0, 0, 0), rotation=(90, 0, 0),
         size=1.0, extrude=0.06, bevel=0.01, align="CENTER", material=None):
    curve = bpy.data.curves.new(name, type="FONT")
    curve.body = str(body)
    curve.size = float(size)
    curve.extrude = float(extrude)
    curve.bevel_depth = float(bevel)
    curve.align_x = align
    curve.align_y = "CENTER"
    ob = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(ob)
    ob.location = _vec(location)
    ob.rotation_euler = tuple(_D(a) for a in _vec(rotation))
    if material is not None:
        assign(ob, material)
    return ob


def smooth(target, angle=35.0):
    ob = obj(target)
    if ob.type != "MESH":
        return ob
    bpy.context.view_layer.objects.active = ob
    for other in bpy.context.selected_objects:
        other.select_set(False)
    ob.select_set(True)
    try:
        bpy.ops.object.shade_auto_smooth(angle=_D(angle))
    except Exception:
        bpy.ops.object.shade_smooth()
    return ob


def bevel(target, width=0.02, segments=3):
    ob = obj(target)
    mod = ob.modifiers.new("Bevel", "BEVEL")
    mod.width = float(width)
    mod.segments = int(segments)
    mod.limit_method = "ANGLE"
    return ob


def subdiv(target, levels=2, render_levels=None):
    ob = obj(target)
    mod = ob.modifiers.new("Subdivision", "SUBSURF")
    mod.levels = int(levels)
    mod.render_levels = int(levels if render_levels is None else render_levels)
    return ob


def solidify(target, thickness=0.05):
    ob = obj(target)
    mod = ob.modifiers.new("Solidify", "SOLIDIFY")
    mod.thickness = float(thickness)
    return ob


def boolean(target, cutter, operation="DIFFERENCE"):
    ob, cut = obj(target), obj(cutter)
    mod = ob.modifiers.new("Boolean", "BOOLEAN")
    mod.operation = operation
    mod.object = cut
    cut.hide_render = True
    cut.hide_viewport = True
    return ob


def place(target, location=None, rotation=None, scale=None):
    """Move/rotate/scale. ``rotation`` is in DEGREES."""
    ob = obj(target)
    if location is not None:
        ob.location = _vec(location)
    if rotation is not None:
        ob.rotation_euler = tuple(_D(a) for a in _vec(rotation))
    if scale is not None:
        ob.scale = _vec(scale, (1, 1, 1))
    return ob


def look_at(target, point=(0, 0, 0)):
    """Aim an object's -Z axis at ``point`` (how cameras and lights point)."""
    ob = obj(target)
    # matrix_world is only refreshed on a depsgraph update, so without this a
    # freshly assigned .location would still read as the previous position and
    # the object would aim from where it used to be.
    bpy.context.view_layer.update()
    direction = _vec(point) - ob.matrix_world.translation
    if direction.length < 1e-6:
        return ob
    ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return ob


def join(targets, name=None):
    objects = [obj(t) for t in targets]
    if not objects:
        raise ValueError("join() needs at least one object")
    for other in bpy.context.selected_objects:
        other.select_set(False)
    for ob in objects:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    result = bpy.context.object
    if name:
        result.name = name
    return result


def scatter(target, count=20, area=(10.0, 10.0), z=0.0, seed=0,
            scale_range=(0.7, 1.3), rotate=True, avoid_radius=0.0,
            center=(0.0, 0.0)):
    """Duplicate an object across an area with jitter - cheap scene complexity.

    Returns the list of new objects (the original is left where it is).
    """
    import random

    rng = random.Random(seed)
    source = obj(target)
    half_x, half_y = float(area[0]) / 2.0, float(area[1]) / 2.0
    placed = []
    made = []
    attempts = 0
    while len(made) < int(count) and attempts < int(count) * 40:
        attempts += 1
        x = center[0] + rng.uniform(-half_x, half_x)
        y = center[1] + rng.uniform(-half_y, half_y)
        if avoid_radius > 0:
            if any((x - px) ** 2 + (y - py) ** 2 < avoid_radius ** 2 for px, py in placed):
                continue
        placed.append((x, y))
        dup = source.copy()
        dup.data = source.data  # linked duplicate: cheap in memory
        bpy.context.collection.objects.link(dup)
        dup.location = (x, y, float(z))
        s = rng.uniform(*scale_range)
        dup.scale = (source.scale.x * s, source.scale.y * s, source.scale.z * s)
        if rotate:
            dup.rotation_euler = (source.rotation_euler.x,
                                  source.rotation_euler.y,
                                  rng.uniform(0, 2 * math.pi))
        made.append(dup)
    return made


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------
def camera(location=(7, -7, 5), target=(0, 0, 0), focal=50.0, name="Camera",
           dof_target=None, f_stop=2.8, shift=(0.0, 0.0)):
    """Create (or move) the scene camera and point it at ``target``."""
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None or cam.type != "CAMERA":
        data = bpy.data.cameras.new(name)
        cam = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(cam)
        scene.camera = cam
    cam.location = _vec(location)
    cam.data.lens = float(focal)
    cam.data.shift_x, cam.data.shift_y = float(shift[0]), float(shift[1])
    look_at(cam, target)
    if dof_target is not None:
        cam.data.dof.use_dof = True
        cam.data.dof.focus_object = obj(dof_target)
        cam.data.dof.aperture_fstop = float(f_stop)
    return cam


def _renderable():
    return [o for o in bpy.context.scene.objects
            if o.type in {"MESH", "FONT", "CURVE"} and not o.hide_render]


def _subjects():
    """Renderable objects minus ground planes.

    Framing on a 60m floor pushes the camera so far back the actual subject
    becomes a speck, so a large flat slab is treated as staging, not subject.
    """
    objects = _renderable()
    if len(objects) < 2:
        return objects
    footprints = []
    for ob in objects:
        size = _local_size(ob)
        footprints.append(max(size.x * size.y, 1e-6))
    median = sorted(footprints)[len(footprints) // 2]
    keep = []
    for ob, area in zip(objects, footprints):
        size = _local_size(ob)
        flat = size.z < 0.05 * max(size.x, size.y, 1e-6)
        if flat and area > 8.0 * median:
            continue
        keep.append(ob)
    return keep or objects


def _local_size(ob):
    corners = [ob.matrix_world @ mathutils.Vector(c) for c in ob.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return mathutils.Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))


def bounds(objects=None):
    """World-space bounding box of the renderable meshes."""
    bpy.context.view_layer.update()
    if objects is None:
        objects = _renderable()
    else:
        objects = [obj(o) for o in objects]
    if not objects:
        return None
    lo = mathutils.Vector((1e18, 1e18, 1e18))
    hi = mathutils.Vector((-1e18, -1e18, -1e18))
    for ob in objects:
        for corner in ob.bound_box:
            world = ob.matrix_world @ mathutils.Vector(corner)
            lo = mathutils.Vector((min(lo.x, world.x), min(lo.y, world.y), min(lo.z, world.z)))
            hi = mathutils.Vector((max(hi.x, world.x), max(hi.y, world.y), max(hi.z, world.z)))
    return lo, hi, (lo + hi) / 2.0, (hi - lo)


def frame_all(margin=1.25, elevation=28.0, azimuth=-55.0, focal=50.0, target=None,
              objects=None, include_ground=False):
    """Put the camera where the subject fits in frame.

    This is the single most useful call in the module: it removes the most
    common failure mode, a beautifully built scene that the camera misses.
    Large flat ground planes are excluded from the fit unless
    ``include_ground=True``, so a big floor does not shrink the subject.
    """
    box = bounds(objects if objects is not None
                 else (_renderable() if include_ground else _subjects()))
    if box is None:
        return camera()
    _lo, _hi, center, size = box
    if target is not None:
        center = _vec(target)
    radius = max(size.length / 2.0, 0.5)
    scene = bpy.context.scene
    aspect = scene.render.resolution_x / max(scene.render.resolution_y, 1)
    sensor = 36.0
    fov = 2.0 * math.atan(sensor / (2.0 * float(focal)))
    if aspect < 1.0:
        fov = 2.0 * math.atan(math.tan(fov / 2.0) * aspect)
    distance = (radius * float(margin)) / max(math.tan(fov / 2.0), 1e-3)
    el, az = _D(elevation), _D(azimuth)
    offset = mathutils.Vector((
        distance * math.cos(el) * math.cos(az),
        distance * math.cos(el) * math.sin(az),
        distance * math.sin(el),
    ))
    return camera(center + offset, center, focal=focal)


def orbit_camera(target=(0, 0, 0), radius=8.0, height=4.0, angle=0.0, focal=50.0):
    a = _D(angle)
    t = _vec(target)
    return camera((t.x + radius * math.cos(a), t.y + radius * math.sin(a), t.z + height),
                  t, focal=focal)


# --------------------------------------------------------------------------
# animation
# --------------------------------------------------------------------------
def key(target, frame, location=None, rotation=None, scale=None,
        energy=None, hide=None):
    """Insert keyframes. ``rotation`` is in DEGREES."""
    ob = obj(target)
    frame = int(frame)
    if location is not None:
        ob.location = _vec(location)
        ob.keyframe_insert("location", frame=frame)
    if rotation is not None:
        ob.rotation_euler = tuple(_D(a) for a in _vec(rotation))
        ob.keyframe_insert("rotation_euler", frame=frame)
    if scale is not None:
        ob.scale = _vec(scale, (1, 1, 1))
        ob.keyframe_insert("scale", frame=frame)
    if energy is not None and getattr(ob, "data", None) and hasattr(ob.data, "energy"):
        ob.data.energy = float(energy)
        ob.data.keyframe_insert("energy", frame=frame)
    if hide is not None:
        ob.hide_render = bool(hide)
        ob.keyframe_insert("hide_render", frame=frame)
    return ob


def anim_orbit(target=(0, 0, 0), start=1, end=96, radius=None, height=None,
               turns=1.0, focal=50.0, elevation=25.0):
    """Fly the camera in a circle around ``target`` - the safest camera move."""
    box = bounds(_subjects())
    if radius is None or height is None:
        if box is not None:
            _lo, _hi, center, size = box
            auto_r = max(size.length * 0.9, 2.0)
            radius = auto_r if radius is None else radius
            height = center.z + auto_r * math.sin(_D(elevation)) if height is None else height
        else:
            radius = radius or 8.0
            height = height or 4.0
    t = _vec(target)
    cam = camera((t.x + radius, t.y, height), t, focal=focal)
    frames = max(int(end) - int(start), 1)
    steps = max(int(frames / 6), 8)
    for i in range(steps + 1):
        f = int(start) + int(round(frames * i / steps))
        a = 2.0 * math.pi * turns * i / steps
        cam.location = (t.x + radius * math.cos(a), t.y + radius * math.sin(a), height)
        look_at(cam, t)
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
    ease(cam, "LINEAR")
    set_frames(start, end, bpy.context.scene.render.fps)
    return cam


def anim_spin(target, start=1, end=96, axis="Z", turns=1.0):
    ob = obj(target)
    idx = {"X": 0, "Y": 1, "Z": 2}[str(axis).upper()]
    base = list(ob.rotation_euler)
    ob.rotation_euler = base
    ob.keyframe_insert("rotation_euler", frame=int(start))
    end_rot = list(base)
    end_rot[idx] = base[idx] + 2.0 * math.pi * float(turns)
    ob.rotation_euler = end_rot
    ob.keyframe_insert("rotation_euler", frame=int(end))
    ease(ob, "LINEAR")
    return ob


def ease(target, interpolation="BEZIER"):
    """``LINEAR`` for constant motion, ``BEZIER`` for ease in/out."""
    ob = obj(target)
    action = getattr(getattr(ob, "animation_data", None), "action", None)
    if action is None:
        return ob
    curves = getattr(action, "fcurves", None)
    if curves is None:  # Blender 5.x slotted actions
        curves = []
        for layer in getattr(action, "layers", []):
            for strip in getattr(layer, "strips", []):
                for bag in getattr(strip, "channelbags", []):
                    curves.extend(bag.fcurves)
    for fcurve in curves:
        for kp in fcurve.keyframe_points:
            kp.interpolation = interpolation
    return ob


# --------------------------------------------------------------------------
# feedback - the model cannot see, so describe and measure for it
# --------------------------------------------------------------------------
def report(verbose=False):
    """A compact text description of the scene. Print the return value."""
    scene = bpy.context.scene
    lines = []
    lines.append("engine=%s  res=%dx%d  fps=%d  frames=%d-%d  view_transform=%s"
                 % (scene.render.engine, scene.render.resolution_x,
                    scene.render.resolution_y, scene.render.fps,
                    scene.frame_start, scene.frame_end,
                    scene.view_settings.view_transform))
    cam = scene.camera
    if cam is None:
        lines.append("CAMERA: NONE - nothing will render!")
    else:
        loc = cam.location
        lines.append("camera=%s at (%.2f, %.2f, %.2f) focal=%.0fmm dof=%s"
                     % (cam.name, loc.x, loc.y, loc.z, cam.data.lens,
                        cam.data.dof.use_dof))

    lights = [o for o in scene.objects if o.type == "LIGHT"]
    if not lights:
        lines.append("LIGHTS: NONE - the render will be black unless the world emits!")
    else:
        lines.append("lights: " + ", ".join(
            "%s(%s, %.0fW)" % (l.name, l.data.type, l.data.energy) for l in lights))

    world_strength = "?"
    if scene.world and scene.world.node_tree:
        for node in scene.world.node_tree.nodes:
            if node.bl_idname == "ShaderNodeBackground":
                world_strength = "%.2f" % node.inputs["Strength"].default_value
    lines.append("world background strength=%s" % world_strength)

    meshes = [o for o in scene.objects if o.type in {"MESH", "FONT", "CURVE"}]
    lines.append("objects: %d renderable" % len(meshes))
    box = bounds()
    if box:
        _lo, _hi, center, size = box
        lines.append("scene bounds: center=(%.2f, %.2f, %.2f) size=(%.2f, %.2f, %.2f)"
                     % (center.x, center.y, center.z, size.x, size.y, size.z))

    shown = meshes if verbose else meshes[:25]
    for ob in shown:
        mats = [s.material.name for s in ob.material_slots if s.material]
        lines.append("  - %s (%s) loc=(%.2f, %.2f, %.2f) scale=(%.2f, %.2f, %.2f) mats=%s%s"
                     % (ob.name, ob.type, ob.location.x, ob.location.y, ob.location.z,
                        ob.scale.x, ob.scale.y, ob.scale.z,
                        mats or "NONE (grey default!)",
                        " ANIMATED" if ob.animation_data and ob.animation_data.action else ""))
    if len(meshes) > len(shown):
        lines.append("  ... and %d more" % (len(meshes) - len(shown)))
    return "\n".join(lines)


def _visibility():
    """Per-object framing: is it inside the camera frustum, and how big?"""
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        return []
    bpy.context.view_layer.update()
    rows = []
    for ob in scene.objects:
        if ob.type not in {"MESH", "FONT", "CURVE"} or ob.hide_render:
            continue
        xs, ys, in_front = [], [], False
        for corner in ob.bound_box:
            world = ob.matrix_world @ mathutils.Vector(corner)
            ndc = world_to_camera_view(scene, cam, world)
            xs.append(ndc.x)
            ys.append(ndc.y)
            if ndc.z > 0:
                in_front = True
        x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
        y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
        coverage = max(0.0, x1 - x0) * max(0.0, y1 - y0) if in_front else 0.0
        rows.append({
            "name": ob.name,
            "in_frame": bool(in_front and x1 > x0 and y1 > y0),
            "screen_coverage": round(coverage, 4),
            "fully_inside": bool(in_front and min(xs) >= 0 and max(xs) <= 1
                                 and min(ys) >= 0 and max(ys) <= 1),
        })
    rows.sort(key=lambda r: -r["screen_coverage"])
    return rows


def _measure(width=200, samples=8):
    """Render a thumbnail and return (mean, median) luminance."""
    import tempfile

    import numpy as np

    scene = bpy.context.scene
    saved = (scene.render.resolution_x, scene.render.resolution_y,
             scene.render.filepath, scene.render.resolution_percentage,
             scene.render.image_settings.media_type,
             scene.render.image_settings.file_format)
    eevee_saved = getattr(scene.eevee, "taa_render_samples", None)
    height = max(1, int(width * scene.render.resolution_y / max(scene.render.resolution_x, 1)))
    path = os.path.join(tempfile.gettempdir(), "bstudio_measure.png")
    try:
        scene.render.resolution_x, scene.render.resolution_y = int(width), height
        scene.render.resolution_percentage = 100
        scene.render.image_settings.media_type = "IMAGE"
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = path
        if eevee_saved is not None:
            scene.eevee.taa_render_samples = int(samples)
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(path)
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        bpy.data.images.remove(image)
        rgb = pixels.reshape(-1, 4)[:, :3]
        luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        return float(luma.mean()), float(np.median(luma))
    finally:
        (scene.render.resolution_x, scene.render.resolution_y,
         scene.render.filepath, scene.render.resolution_percentage,
         scene.render.image_settings.media_type,
         scene.render.image_settings.file_format) = saved
        if eevee_saved is not None:
            scene.eevee.taa_render_samples = eevee_saved


def auto_exposure(target=0.38, tolerance=0.05, max_iter=5, limit=5.0):
    """Nudge the scene exposure until the frame is neither crushed nor blown.

    A model that cannot see the render has no way to notice that a physical sky
    plus a sun plus a 900 W key light adds up to a white image. This measures
    the result and corrects it in exposure stops, which is non-destructive - it
    touches colour management, not the lights the model set up.

    Returns a dict describing what it did.
    """
    scene = bpy.context.scene
    start = float(scene.view_settings.exposure)
    mean, median = _measure()
    history = [{"exposure": round(start, 3), "median": round(median, 4)}]

    for _ in range(int(max_iter)):
        # Median tracks the subject better than the mean when a bright sky
        # fills part of the frame.
        if abs(median - target) <= tolerance:
            break
        if median <= 1e-4:
            step = 2.0  # pitch black: open up hard and re-measure
        else:
            step = math.log2(target / max(median, 1e-4))
        step = max(-3.0, min(3.0, step))
        new_exposure = max(-limit, min(limit, scene.view_settings.exposure + step))
        if abs(new_exposure - scene.view_settings.exposure) < 0.01:
            break
        scene.view_settings.exposure = new_exposure
        mean, median = _measure()
        history.append({"exposure": round(new_exposure, 3), "median": round(median, 4)})

    return {
        "exposure": round(float(scene.view_settings.exposure), 3),
        "start_exposure": round(start, 3),
        "median_luminance": round(median, 4),
        "mean_luminance": round(mean, 4),
        "target": target,
        "steps": history,
    }


def analyse(width=240, samples=16):
    """Render a thumbnail and measure it. Returns a dict; print it as JSON.

    This is the model's substitute for looking at the viewport: it turns
    "too dark", "nothing in frame" and "blown out" into numbers.
    """
    import tempfile

    import numpy as np

    scene = bpy.context.scene
    saved = (scene.render.resolution_x, scene.render.resolution_y,
             scene.render.filepath, scene.render.resolution_percentage,
             scene.render.image_settings.media_type,
             scene.render.image_settings.file_format)
    eevee_saved = getattr(scene.eevee, "taa_render_samples", None)

    height = max(1, int(width * scene.render.resolution_y / max(scene.render.resolution_x, 1)))
    path = os.path.join(tempfile.gettempdir(), "bstudio_analyse.png")
    result = {}
    try:
        scene.render.resolution_x, scene.render.resolution_y = int(width), height
        scene.render.resolution_percentage = 100
        scene.render.image_settings.media_type = "IMAGE"
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = path
        if eevee_saved is not None:
            scene.eevee.taa_render_samples = int(samples)
        bpy.ops.render.render(write_still=True)

        image = bpy.data.images.load(path)
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        bpy.data.images.remove(image)
        rgb = pixels.reshape(-1, 4)[:, :3]
        luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        result = {
            "mean_luminance": round(float(luma.mean()), 4),
            "median_luminance": round(float(np.median(luma)), 4),
            "fraction_near_black": round(float((luma < 0.02).mean()), 4),
            "fraction_blown_out": round(float((luma > 0.98).mean()), 4),
            "contrast": round(float(luma.std()), 4),
            "mean_rgb": [round(float(c), 4) for c in rgb.mean(axis=0)],
        }
    except Exception as exc:
        result = {"render_error": str(exc)}
    finally:
        (scene.render.resolution_x, scene.render.resolution_y,
         scene.render.filepath, scene.render.resolution_percentage,
         scene.render.image_settings.media_type,
         scene.render.image_settings.file_format) = saved
        if eevee_saved is not None:
            scene.eevee.taa_render_samples = eevee_saved

    result["objects_in_frame"] = _visibility()[:15]
    hints = []
    if result.get("fraction_near_black", 0) > 0.85:
        hints.append("Almost the whole frame is black: check that lights exist and "
                     "have enough energy, that the world strength is above 0, and "
                     "that the camera is actually pointing at the subject.")
    if result.get("fraction_blown_out", 0) > 0.25:
        hints.append("Large blown-out areas: reduce light energy or world strength.")
    if result.get("median_luminance", 0) > 0.72:
        hints.append("The frame is washed out - almost everything is near white, so "
                     "materials and shapes have stopped reading. Cut the light "
                     "energy and world/sky strength, then call "
                     "bstudio.auto_exposure().")
    if result.get("contrast", 1) < 0.03 and result.get("mean_luminance", 0) > 0.02:
        hints.append("The frame is nearly flat - the camera may be inside an object "
                     "or facing an empty backdrop.")
    visible = [r for r in result["objects_in_frame"] if r["in_frame"]]
    subject_names = {o.name for o in _subjects()}
    subjects_visible = [r for r in visible if r["name"] in subject_names]
    if not visible:
        hints.append("No object is inside the camera frustum. Call bstudio.frame_all().")
    elif not subjects_visible:
        hints.append("Only the ground/backdrop is in frame - the subject is off camera. "
                     "Call bstudio.frame_all().")
    elif subjects_visible[0]["screen_coverage"] < 0.04:
        hints.append("The main subject fills under 4%% of the frame (largest is %r). "
                     "Tighten the shot with bstudio.frame_all(objects=['%s'], margin=1.2)."
                     % (subjects_visible[0]["name"], subjects_visible[0]["name"]))
    result["hints"] = hints
    return result


def analyse_json(width=240):
    return json.dumps(analyse(width=width), indent=1)


# --------------------------------------------------------------------------
# the character layer - bodies, motion, mouths, garments, style packs
#
# This is what the pipeline's Blender steps are written against, and it lives
# here rather than beside them so the interactive path gets the same calls.
#
# The skeleton is the load-bearing decision. Its bones are the first 22 SMPL-X
# joints, in SMPL-X order, with SMPL-X's parenting, plus ``jaw``. Nothing in M1
# fits a real body - the mesh is primitives - but a motion capture stage that
# emits ``poses[frame][joint]`` in that order can drive it with no translation
# layer, which is the whole point of stubbing rather than improvising.
# --------------------------------------------------------------------------
SMPLX_JOINTS = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "jaw",
)

SMPLX_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    15,
)

# Rest positions for a 1.70 m body, in Blender axes: +X to the character's
# left, +Y forward, +Z up. Scaled uniformly by height_m / 1.70.
SMPLX_REST = {
    "pelvis": (0.00, 0.00, 0.94), "left_hip": (0.09, 0.00, 0.86),
    "right_hip": (-0.09, 0.00, 0.86), "spine1": (0.00, 0.00, 1.05),
    "left_knee": (0.10, 0.00, 0.47), "right_knee": (-0.10, 0.00, 0.47),
    "spine2": (0.00, 0.00, 1.16), "left_ankle": (0.10, 0.00, 0.08),
    "right_ankle": (-0.10, 0.00, 0.08), "spine3": (0.00, 0.00, 1.25),
    "left_foot": (0.10, 0.13, 0.03), "right_foot": (-0.10, 0.13, 0.03),
    "neck": (0.00, 0.00, 1.42), "left_collar": (0.07, 0.00, 1.36),
    "right_collar": (-0.07, 0.00, 1.36), "head": (0.00, 0.00, 1.52),
    "left_shoulder": (0.18, 0.00, 1.40), "right_shoulder": (-0.18, 0.00, 1.40),
    "left_elbow": (0.44, 0.00, 1.40), "right_elbow": (-0.44, 0.00, 1.40),
    "left_wrist": (0.69, 0.00, 1.40), "right_wrist": (-0.69, 0.00, 1.40),
    "jaw": (0.00, 0.06, 1.56),
}

REST_HEIGHT_M = 1.70

# The eight channels ``visemes.schema.json`` allows, which are therefore the
# eight shape keys a body must have for a mouth curve to land on it.
VISEME_CHANNELS = ("jaw_open", "lip_wide", "lip_round", "lip_press",
                   "brow_raise", "brow_furrow", "squint", "smile")

# §7-C: illustration references defeat automatic fitting, so proportion comes
# from a named preset. Multipliers on (shoulder width, limb girth, head size,
# torso length).
SHAPE_PRESETS = {
    "neutral": (1.00, 1.00, 1.00, 1.00),
    "heroic": (1.22, 1.18, 0.94, 1.04),
    "giant": (1.30, 1.30, 0.90, 1.08),
    "slight": (0.88, 0.84, 1.04, 0.98),
    "aged": (1.02, 0.96, 1.00, 0.96),
    "youth": (0.92, 0.90, 1.08, 0.94),
}


# -- geometry built by hand, not by operators -------------------------------
# from_pydata needs no operator context, no active collection and no selection
# state, so it behaves the same in --background as in the viewport. Every
# primitive helper returns (verts, faces) in world space.
def _prism(a, b, r0, r1, sides=8, up=(0.0, 0.0, 1.0)):
    """A tapered prism from point ``a`` to point ``b``: limbs, necks, torsos."""
    a, b = mathutils.Vector(a), mathutils.Vector(b)
    axis = b - a
    length = axis.length
    if length < 1e-6:
        axis = mathutils.Vector((0.0, 0.0, 1.0))
        length = 1e-6
    axis = axis / length
    # ``up`` only names which way "around" the ring starts. When it happens to
    # be parallel to the axis the cross product is zero, and normalising zero
    # leaves zero - every ring vertex would collapse onto the centre line and
    # the prism would come out as a fan of zero-area faces. Fall back through
    # two more directions; no axis is parallel to all three.
    side = axis.cross(mathutils.Vector(up))
    for fallback in ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)):
        if side.length >= 1e-4:
            break
        side = axis.cross(mathutils.Vector(fallback))
    side.normalize()
    other = axis.cross(side)
    verts = []
    for radius, centre in ((r0, a), (r1, b)):
        for i in range(sides):
            angle = 2.0 * math.pi * i / sides
            verts.append(tuple(centre
                               + side * (math.cos(angle) * radius)
                               + other * (math.sin(angle) * radius)))
    faces = [(i, (i + 1) % sides, sides + (i + 1) % sides, sides + i)
             for i in range(sides)]
    faces.append(tuple(range(sides - 1, -1, -1)))
    faces.append(tuple(range(sides, sides * 2)))
    return verts, faces


def _ball(centre, radii, segments=12, rings=7):
    """A UV sphere with per-axis radii, so a head can be egg-shaped."""
    cx, cy, cz = centre
    rx, ry, rz = radii if isinstance(radii, (list, tuple)) else (radii,) * 3
    verts = [(cx, cy, cz + rz)]
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        for seg in range(segments):
            theta = 2.0 * math.pi * seg / segments
            verts.append((cx + rx * math.sin(phi) * math.cos(theta),
                          cy + ry * math.sin(phi) * math.sin(theta),
                          cz + rz * math.cos(phi)))
    verts.append((cx, cy, cz - rz))
    faces = []
    for seg in range(segments):
        faces.append((0, 1 + seg, 1 + (seg + 1) % segments))
    for ring in range(rings - 2):
        base, nxt = 1 + ring * segments, 1 + (ring + 1) * segments
        for seg in range(segments):
            faces.append((base + seg, nxt + seg,
                          nxt + (seg + 1) % segments,
                          base + (seg + 1) % segments))
    last, tip = 1 + (rings - 2) * segments, len(verts) - 1
    for seg in range(segments):
        faces.append((last + seg, tip, last + (seg + 1) % segments))
    return verts, faces


def _brick(centre, size, shear=(0.0, 0.0)):
    """An axis-aligned box, optionally sheared along X at the top."""
    cx, cy, cz = centre
    hx, hy, hz = (s * 0.5 for s in size)
    sx, sy = shear
    return ([(cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
             (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
             (cx - hx + sx, cy - hy + sy, cz + hz),
             (cx + hx + sx, cy - hy + sy, cz + hz),
             (cx + hx + sx, cy + hy + sy, cz + hz),
             (cx - hx + sx, cy + hy + sy, cz + hz)],
            [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1),
             (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)])


# -- material roles ---------------------------------------------------------
# Assets never name a material. They name a *role*, tagged onto a grey
# placeholder, and a style pack swaps the placeholder out at assemble time.
# The tag rides on the material rather than on a side table so it survives
# bpy.data.libraries.load into a shot file.
DEFAULT_PART_ROLES = {
    "skin": "skin", "hair": "hair", "eye_white": "eye_white",
    "eye_iris": "eye_iris", "lip": "lip", "tunic": "cloth_primary",
    "sash": "cloth_secondary", "trim": "cloth_trim", "boots": "leather",
    "armour": "metal_armor", "blade": "metal_blade", "hide": "skin",
    "mane": "hair", "tack": "leather", "bosses": "gold_leaf",
    "hoof": "leather", "muzzle": "skin",
}

_PLACEHOLDER_GREY = {
    "skin": (0.78, 0.62, 0.50), "hair": (0.16, 0.13, 0.11),
    "eye_white": (0.92, 0.92, 0.90), "eye_iris": (0.20, 0.14, 0.09),
    "lip": (0.66, 0.34, 0.30), "cloth_primary": (0.24, 0.34, 0.62),
    "cloth_secondary": (0.20, 0.58, 0.58), "cloth_trim": (0.80, 0.70, 0.30),
    "leather": (0.30, 0.22, 0.16), "metal_armor": (0.55, 0.55, 0.58),
    "metal_blade": (0.72, 0.73, 0.76), "gold_leaf": (0.83, 0.69, 0.22),
    "wood": (0.42, 0.29, 0.17), "stone": (0.55, 0.53, 0.50),
    "ground": (0.42, 0.44, 0.28), "foliage": (0.24, 0.42, 0.20),
    "water": (0.20, 0.36, 0.48), "sky": (0.62, 0.74, 0.88),
    "ink_outline": (0.06, 0.05, 0.06), "backdrop": (0.86, 0.82, 0.72),
}


def role_material(role):
    """The grey stand-in for a role, created once per blend file and tagged."""
    name = "role_" + role
    existing = bpy.data.materials.get(name)
    if existing is not None:
        existing["style_role"] = role
        return existing
    material = pbr(name, color=_PLACEHOLDER_GREY.get(role, (0.6, 0.6, 0.6)),
                   roughness=0.55)
    material["style_role"] = role
    return material


def _resolve_role(part_key, material_roles):
    wanted = (material_roles or {}).get(part_key)
    return wanted or DEFAULT_PART_ROLES.get(part_key, "skin")


def _dome(centre, radii, segments=12, rings=5, sweep=0.62):
    """A partial sphere closed with a flat ring: hair caps, shoulder pads."""
    cx, cy, cz = centre
    rx, ry, rz = radii if isinstance(radii, (list, tuple)) else (radii,) * 3
    verts = [(cx, cy, cz + rz)]
    for ring in range(1, rings + 1):
        phi = math.pi * sweep * ring / rings
        for seg in range(segments):
            theta = 2.0 * math.pi * seg / segments
            verts.append((cx + rx * math.sin(phi) * math.cos(theta),
                          cy + ry * math.sin(phi) * math.sin(theta),
                          cz + rz * math.cos(phi)))
    faces = [(0, 1 + seg, 1 + (seg + 1) % segments) for seg in range(segments)]
    for ring in range(rings - 1):
        base, nxt = 1 + ring * segments, 1 + (ring + 1) * segments
        for seg in range(segments):
            faces.append((base + seg, nxt + seg,
                          nxt + (seg + 1) % segments,
                          base + (seg + 1) % segments))
    rim = 1 + (rings - 1) * segments
    faces.append(tuple(range(rim + segments - 1, rim - 1, -1)))
    return verts, faces


# -- the rest skeleton, scaled to a character's height and preset ------------
_WIDE = ("left_collar", "right_collar", "left_shoulder", "right_shoulder",
         "left_elbow", "right_elbow", "left_wrist", "right_wrist")
_ABOVE_PELVIS = ("spine1", "spine2", "spine3", "neck", "head", "jaw") + _WIDE


def _rest_joints(height_m=REST_HEIGHT_M, preset="neutral"):
    """Joint positions and the radii the mesh is built from.

    The preset stretches the rest pose, so the crown moves; every coordinate is
    then scaled uniformly to put the top of the head back at ``height_m``. That
    makes the measured height match the requested one, which is what
    ``gates.check_body`` looks at.
    """
    shoulder, girth, head_mul, torso = SHAPE_PRESETS.get(
        preset, SHAPE_PRESETS["neutral"])
    pelvis_z = SMPLX_REST["pelvis"][2]
    joints = {}
    for name, (x, y, z) in SMPLX_REST.items():
        if name in _WIDE:
            x *= shoulder
        if name in _ABOVE_PELVIS:
            z = pelvis_z + (z - pelvis_z) * torso
        joints[name] = [x, y, z]
    crown = joints["head"][2] + (0.078 + 0.132 * head_mul)
    factor = float(height_m) / crown
    for name, xyz in joints.items():
        joints[name] = tuple(component * factor for component in xyz)
    return joints, {"factor": factor, "girth": girth * factor,
                    "head": head_mul * factor, "shoulder": shoulder,
                    "height": float(height_m), "preset": preset}


def _lerp(a, b, t):
    return tuple(p + (q - p) * t for p, q in zip(a, b))


def _shift(point, dx=0.0, dy=0.0, dz=0.0):
    return (point[0] + dx, point[1] + dy, point[2] + dz)


def _human_parts(joints, dims, clothed=True, beard=False):
    """The part list for a biped: ``[(part_key, verts, faces), ...]``.

    Clothing is not a second surface laid over the skin — where the character is
    dressed, the surface *is* the garment. Two coincident shells would z-fight,
    and a primitive body has nothing to gain from the skin underneath.
    """
    g = dims["girth"]
    f = dims["factor"]
    parts = []
    j = joints

    torso = "tunic" if clothed else "skin"
    limb = "tunic" if clothed else "skin"
    lower = "boots" if clothed else "skin"

    hip_c = _lerp(j["left_hip"], j["right_hip"], 0.5)
    parts.append((torso, *_prism(_shift(hip_c, dz=-0.03 * f), j["spine1"],
                                 0.140 * g, 0.128 * g, sides=10)))
    parts.append((torso, *_prism(j["spine1"], j["spine3"],
                                 0.128 * g, 0.158 * g, sides=10)))
    if clothed:
        parts.append(("sash", *_prism(_shift(hip_c, dz=0.06 * f),
                                      _shift(hip_c, dz=0.14 * f),
                                      0.148 * g, 0.144 * g, sides=10)))
        # A knee-length tunic. The thighs live inside it, exactly as they do
        # under a real one, and the skirt is what the camera sees.
        parts.append((torso, *_prism(_shift(hip_c, dz=0.02 * f),
                                     _shift(hip_c, dz=-0.36 * f),
                                     0.150 * g, 0.196 * g, sides=12)))
        parts.append(("trim", *_prism(_shift(hip_c, dz=-0.355 * f),
                                      _shift(hip_c, dz=-0.395 * f),
                                      0.198 * g, 0.196 * g, sides=12)))

    for side in ("left", "right"):
        collar, shoulder = j[side + "_collar"], j[side + "_shoulder"]
        elbow, wrist = j[side + "_elbow"], j[side + "_wrist"]
        sign = 1.0 if side == "left" else -1.0
        parts.append((torso, *_prism(collar, shoulder,
                                     0.082 * g, 0.070 * g, sides=8)))
        parts.append((limb, *_prism(shoulder, elbow,
                                    0.062 * g, 0.050 * g, sides=8)))
        parts.append(("skin", *_prism(elbow, wrist,
                                      0.050 * g, 0.038 * g, sides=8)))
        if clothed:
            parts.append(("trim", *_prism(
                _lerp(shoulder, elbow, 0.92), _lerp(shoulder, elbow, 1.02),
                0.054 * g, 0.052 * g, sides=8)))
        parts.append(("skin", *_brick(
            _shift(wrist, dx=sign * 0.085 * f, dz=-0.012 * f),
            (0.170 * f, 0.052 * g, 0.092 * g))))

        hip, knee, ankle = j[side + "_hip"], j[side + "_knee"], j[side + "_ankle"]
        parts.append(("skin", *_prism(_shift(hip, dz=-0.02 * f), knee,
                                      0.094 * g, 0.072 * g, sides=8)))
        parts.append((lower, *_prism(knee, ankle,
                                     0.074 * g, 0.048 * g, sides=8)))
        parts.append((lower, *_brick(
            _shift(ankle, dy=0.072 * f, dz=-0.042 * f),
            (0.098 * g, 0.245 * f, 0.082 * f))))
    head, anchors = _head_parts(joints, dims, beard=beard)
    return parts + head, anchors


def _head_parts(joints, dims, beard=False):
    """Head, face and hair, plus the anchor points the mouth shapes need.

    The anchors come back rather than being recomputed later because the shape
    keys are built from a radial falloff around them, and a face whose mouth
    key opens the wrong vertices is the most visible failure in the pipeline.
    """
    f, g, h = dims["factor"], dims["girth"], dims["head"]
    z = joints["head"][2]
    face_y = 0.052 * f + 0.050 * h
    parts = [
        ("skin", *_prism(joints["spine3"], _shift(joints["head"], dz=-0.055 * f),
                         0.062 * g, 0.056 * g, sides=8)),
        ("skin", *_ball((0.0, 0.006 * f, z + 0.075 * f),
                        (0.092 * h, 0.105 * h, 0.125 * h),
                        segments=14, rings=9)),
        ("skin", *_brick((0.0, 0.058 * f, z + 0.004 * f),
                         (0.132 * h, 0.118 * h, 0.078 * h))),
        ("lip", *_brick((0.0, face_y, z + 0.002 * f),
                        (0.064 * h, 0.026 * h, 0.022 * h))),
    ]
    for sign in (1.0, -1.0):
        eye = (sign * 0.038 * h, face_y - 0.004 * h, z + 0.082 * f)
        parts.append(("eye_white", *_ball(eye, 0.017 * h, segments=8, rings=5)))
        parts.append(("eye_iris", *_ball(_shift(eye, dy=0.009 * h),
                                         0.0095 * h, segments=8, rings=5)))
        parts.append(("skin", *_brick(
            (sign * 0.040 * h, face_y + 0.002 * h, z + 0.112 * f),
            (0.066 * h, 0.020 * h, 0.016 * h))))
    parts.append(("hair", *_dome((0.0, -0.008 * f, z + 0.078 * f),
                                 (0.100 * h, 0.112 * h, 0.132 * h),
                                 segments=14, rings=5, sweep=0.60)))
    if beard:
        parts.append(("hair", *_ball((0.0, 0.062 * f, z - 0.048 * f),
                                     (0.088 * h, 0.086 * h, 0.090 * h),
                                     segments=12, rings=7)))
    anchors = {
        "head": tuple(joints["head"]),
        "mouth": (0.0, face_y, z + 0.002 * f),
        "brow": (0.0, face_y, z + 0.112 * f),
        "eye_l": (0.038 * h, face_y, z + 0.082 * f),
        "eye_r": (-0.038 * h, face_y, z + 0.082 * f),
        "radius": 0.075 * h,
    }
    return parts, anchors


# -- the quadruped ----------------------------------------------------------
# Rakhsh is not a person, so SMPL-X does not apply and §6.2 routes his kind to
# ``rig_mode: rigged``. He still needs a skeleton in the same shape of data -
# names, parents, rest positions - so that ``apply_motion`` is one function and
# not two. Coordinates are normalised to a 1.0 m crown and scaled on use; the
# animal faces +Y.
QUAD_JOINTS = (
    "root", "spine1", "spine2", "chest", "neck", "head", "muzzle", "jaw",
    "tail", "tail_end",
    "left_shoulder", "left_knee", "left_hoof_f",
    "right_shoulder", "right_knee", "right_hoof_f",
    "left_hip", "left_stifle", "left_hoof_b",
    "right_hip", "right_stifle", "right_hoof_b",
)

QUAD_PARENTS = (-1, 0, 1, 2, 3, 4, 5, 5, 0, 8,
                3, 10, 11, 3, 13, 14,
                0, 16, 17, 0, 19, 20)

QUAD_REST = {
    "root": (0.00, 0.00, 0.62), "spine1": (0.00, 0.22, 0.66),
    "spine2": (0.00, 0.46, 0.68), "chest": (0.00, 0.66, 0.66),
    "neck": (0.00, 0.78, 0.80), "head": (0.00, 0.92, 0.92),
    "muzzle": (0.00, 1.06, 0.86), "jaw": (0.00, 1.00, 0.84),
    "tail": (0.00, -0.14, 0.62), "tail_end": (0.00, -0.40, 0.34),
    "left_shoulder": (0.13, 0.62, 0.60), "left_knee": (0.14, 0.62, 0.34),
    "left_hoof_f": (0.14, 0.64, 0.03),
    "right_shoulder": (-0.13, 0.62, 0.60), "right_knee": (-0.14, 0.62, 0.34),
    "right_hoof_f": (-0.14, 0.64, 0.03),
    "left_hip": (0.15, -0.02, 0.58), "left_stifle": (0.16, -0.06, 0.36),
    "left_hoof_b": (0.16, 0.02, 0.03),
    "right_hip": (-0.15, -0.02, 0.58), "right_stifle": (-0.16, -0.06, 0.36),
    "right_hoof_b": (-0.16, 0.02, 0.03),
}

QUAD_CROWN = 1.06          # head top in normalised units


def _quad_joints(height_m=1.60, preset="neutral"):
    _, girth, head_mul, _ = SHAPE_PRESETS.get(preset, SHAPE_PRESETS["neutral"])
    factor = float(height_m) / QUAD_CROWN
    joints = {name: tuple(c * factor for c in xyz)
              for name, xyz in QUAD_REST.items()}
    return joints, {"factor": factor, "girth": girth * factor,
                    "head": head_mul * factor, "shoulder": 1.0,
                    "height": float(height_m), "preset": preset}


def _quad_parts(joints, dims, tack=True):
    f, g, h = dims["factor"], dims["girth"], dims["head"]
    j = joints
    parts = [
        ("hide", *_prism(j["root"], j["chest"], 0.205 * g, 0.225 * g,
                         sides=12, up=(0.0, 1.0, 0.0))),
        ("hide", *_prism(j["tail"], j["root"], 0.170 * g, 0.210 * g,
                         sides=10, up=(0.0, 1.0, 0.0))),
        ("hide", *_prism(j["chest"], j["neck"], 0.195 * g, 0.120 * g, sides=10)),
        ("hide", *_prism(j["neck"], j["head"], 0.115 * g, 0.085 * g, sides=10)),
        ("hide", *_ball(_shift(j["head"], dy=0.03 * f, dz=-0.02 * f),
                        (0.075 * h, 0.120 * h, 0.098 * h),
                        segments=12, rings=7)),
        ("muzzle", *_prism(_shift(j["head"], dy=0.05 * f, dz=-0.04 * f),
                           j["muzzle"], 0.068 * h, 0.052 * h, sides=10)),
        ("mane", *_prism(_shift(j["chest"], dz=0.19 * g),
                         _shift(j["head"], dz=0.075 * h),
                         0.058 * g, 0.038 * g, sides=6)),
        ("mane", *_prism(j["tail"], j["tail_end"], 0.058 * g, 0.022 * g,
                         sides=8, up=(0.0, 1.0, 0.0))),
    ]
    for sign in (1.0, -1.0):
        eye = (sign * 0.070 * h, j["head"][1] + 0.055 * f,
               j["head"][2] + 0.025 * f)
        parts.append(("eye_white", *_ball(eye, 0.021 * h, segments=8, rings=5)))
        parts.append(("eye_iris", *_ball(_shift(eye, dx=sign * 0.009 * h),
                                         0.012 * h, segments=8, rings=5)))
        parts.append(("hide", *_prism(
            (sign * 0.048 * h, j["head"][1] - 0.02 * f, j["head"][2] + 0.06 * f),
            (sign * 0.060 * h, j["head"][1] - 0.05 * f, j["head"][2] + 0.15 * f),
            0.024 * h, 0.006 * h, sides=6)))
        for mid, hoof in (("knee", "hoof_f"), ("stifle", "hoof_b")):
            side = "left" if sign > 0 else "right"
            top = j[side + ("_shoulder" if mid == "knee" else "_hip")]
            joint, foot = j[side + "_" + mid], j[side + "_" + hoof]
            parts.append(("hide", *_prism(top, joint, 0.076 * g, 0.050 * g,
                                          sides=8)))
            parts.append(("hide", *_prism(joint, foot, 0.050 * g, 0.036 * g,
                                          sides=8)))
            parts.append(("hoof", *_brick(_shift(foot, dz=-0.015 * f),
                                          (0.082 * g, 0.105 * f, 0.060 * f))))
    if tack:
        seat = _lerp(j["spine1"], j["spine2"], 0.5)
        parts.append(("tack", *_brick(_shift(seat, dz=0.20 * g),
                                      (0.36 * g, 0.44 * f, 0.10 * f))))
        parts.append(("tack", *_prism(_shift(seat, dz=0.18 * g),
                                      _shift(seat, dz=-0.20 * g),
                                      0.215 * g, 0.215 * g, sides=12)))
        band = _lerp(j["head"], j["muzzle"], 0.45)
        parts.append(("tack", *_prism(_shift(band, dy=-0.02 * f),
                                      _shift(band, dy=0.02 * f),
                                      0.070 * h, 0.070 * h, sides=10)))
        for sign in (1.0, -1.0):
            parts.append(("bosses", *_ball(
                _shift(band, dx=sign * 0.062 * h), 0.020 * h,
                segments=8, rings=5)))
    anchors = {
        "head": tuple(j["head"]),
        "mouth": _lerp(j["muzzle"], j["head"], 0.15),
        "brow": _shift(j["head"], dy=0.02 * f, dz=0.07 * f),
        "eye_l": (0.070 * h, j["head"][1] + 0.055 * f, j["head"][2] + 0.025 * f),
        "eye_r": (-0.070 * h, j["head"][1] + 0.055 * f, j["head"][2] + 0.025 * f),
        "radius": 0.090 * h,
    }
    return parts, anchors


# -- welding the primitives into one deformable mesh ------------------------
# One mesh, not a parented pile: a garment binder needs a single surface to
# sample, shape keys cannot span objects, and an armature modifier per limb
# would be twelve modifiers to keep in step.
HUMAN_TAILS = {"pelvis": "spine1", "spine3": "neck", "head": None}
QUAD_TAILS = {"root": "spine1", "chest": "neck", "head": "muzzle"}


def _weld(parts):
    """``(verts, faces, spans)`` where a span is (part_key, face_from, face_to)."""
    verts, faces, spans = [], [], []
    for key, part_verts, part_faces in parts:
        base, first = len(verts), len(faces)
        verts.extend(tuple(v) for v in part_verts)
        faces.extend(tuple(i + base for i in face) for face in part_faces)
        spans.append((key, first, len(faces), base, len(verts)))
    return verts, faces, spans


def _fit(verts, joints, anchors, dims, height_m):
    """Rescale the welded body so it measures ``height_m`` with its feet at z=0.

    The part builders work in a normalised frame that the presets then distort,
    and the tallest thing on a body is not always near the joint an analytic
    crown formula was written around - a horse's ears out-reach its skull, and a
    hoof block hangs below the hoof joint. Rather than keep such a formula in
    step with every primitive, measure what was actually welded and scale
    everything that shares its units: the vertices, the joint frame the armature
    is built from, the anchors the shape keys deform around, and the length
    scalars in ``dims``. ``gates.check_body`` measures the bounding box, so this
    is the number it reads.
    """
    if not verts:
        return verts, joints, anchors, dims
    low = min(v[2] for v in verts)
    span = max(v[2] for v in verts) - low
    scale = (float(height_m) / span) if span > 1e-9 else 1.0
    lift = -low * scale

    def move(point):
        return (point[0] * scale, point[1] * scale, point[2] * scale + lift)

    fitted = {}
    for key, value in anchors.items():
        fitted[key] = (move(value) if hasattr(value, "__len__")
                       else value * scale)
    sized = dict(dims)
    for key in ("factor", "girth", "head"):
        sized[key] = sized[key] * scale
    sized["fit_scale"] = scale
    return ([move(v) for v in verts],
            {name: move(xyz) for name, xyz in joints.items()},
            fitted, sized)


def _bone_frame(joints, order, parents, tails=None, stub=0.06):
    """Head and tail for every bone, resolved once and shared with the weights."""
    tails = tails or {}
    children = {}
    for index, parent in enumerate(parents):
        if parent >= 0:
            children.setdefault(order[parent], []).append(order[index])
    frame = {}
    for index, name in enumerate(order):
        head = mathutils.Vector(joints[name])
        kids = children.get(name) or []
        wanted = tails.get(name, "" if kids else None)
        if wanted:
            tail = mathutils.Vector(joints[wanted])
        elif wanted == "" and kids:
            tail = mathutils.Vector(joints[kids[0]])
        else:
            parent = parents[index]
            if parent >= 0:
                direction = head - mathutils.Vector(joints[order[parent]])
            else:
                direction = mathutils.Vector((0.0, 0.0, 1.0))
            if direction.length < 1e-6:
                direction = mathutils.Vector((0.0, 0.0, 1.0))
            tail = head + direction.normalized() * stub
        if (tail - head).length < 1e-5:
            tail = head + mathutils.Vector((0.0, 0.0, stub))
        frame[name] = (head, tail)
    return frame


def _closest_on_segment(point, head, tail):
    axis = tail - head
    length_sq = axis.dot(axis)
    if length_sq < 1e-12:
        return (point - head).length
    t = max(0.0, min(1.0, (point - head).dot(axis) / length_sq))
    return (point - (head + axis * t)).length


def _bone_weights(verts, frame, blend=2):
    """Weight every vertex to its nearest bone segments, softly.

    Blender's automatic weights run heat diffusion and give up on non-manifold
    input, which welded primitives always are. Distance to the bone segment is
    cruder but never fails, and for a stand-in body it deforms cleanly.
    """
    names = list(frame)
    groups = {name: [] for name in names}
    for index, vert in enumerate(verts):
        point = mathutils.Vector(vert)
        ranked = sorted(
            ((_closest_on_segment(point, *frame[name]), name) for name in names),
            key=lambda pair: pair[0])[:max(1, blend)]
        weights = [(1.0 / (distance + 1e-4) ** 3, name)
                   for distance, name in ranked]
        total = sum(weight for weight, _ in weights) or 1.0
        for weight, name in weights:
            share = weight / total
            if share > 0.004:
                groups[name].append((index, share))
    return groups


def _falloff(verts, centre, radius, power=2.0, cap_z=None):
    centre = mathutils.Vector(centre)
    out = {}
    for index, vert in enumerate(verts):
        if cap_z is not None and vert[2] > cap_z:
            continue
        distance = (mathutils.Vector(vert) - centre).length
        if distance < radius:
            out[index] = (1.0 - distance / radius) ** power
    return out


def _build_mesh(name, verts, faces, spans, material_roles=None):
    """One mesh from the welded parts, with a material slot per role."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(ob)

    slots, index_of = [], {}
    for key, _, _, _, _ in spans:
        role = _resolve_role(key, material_roles)
        if role not in index_of:
            index_of[role] = len(slots)
            slots.append(role)
            mesh.materials.append(role_material(role))
    for key, face_from, face_to, _, _ in spans:
        slot = index_of[_resolve_role(key, material_roles)]
        for polygon in mesh.polygons[face_from:face_to]:
            polygon.material_index = slot
    for key, _, _, vert_from, vert_to in spans:
        group = ob.vertex_groups.get(key) or ob.vertex_groups.new(name=key)
        group.add(list(range(vert_from, vert_to)), 1.0, "REPLACE")
    mesh.validate(clean_customdata=False)
    mesh.update()
    return ob, slots


def _face_groups(verts, anchors):
    """Mouth, brow and eye weights - the fields every viseme key deforms."""
    radius = anchors["radius"]
    mouth = _falloff(verts, anchors["mouth"], radius * 1.40, power=1.6,
                     cap_z=anchors["mouth"][2] + radius * 0.95)
    brow = _falloff(verts, anchors["brow"], radius * 0.95, power=1.6)
    eyes = {}
    for key in ("eye_l", "eye_r"):
        for index, weight in _falloff(verts, anchors[key], radius * 0.60,
                                      power=1.4).items():
            eyes[index] = max(eyes.get(index, 0.0), weight)
    return mouth, brow, eyes


def _shape_deform(channel, data, groups, anchors, h):
    """Write one viseme channel into a shape key's point data."""
    mouth, brow, eyes = groups
    mz = anchors["mouth"][2]
    ez = anchors["eye_l"][2]
    if channel == "jaw_open":
        for i, w in mouth.items():
            x, y, z = data[i].co
            data[i].co = (x, y + 0.004 * h * w,
                          mz + (z - mz) * (1.0 + 0.95 * w) - 0.026 * h * w)
    elif channel == "lip_wide":
        for i, w in mouth.items():
            x, y, z = data[i].co
            data[i].co = (x * (1.0 + 0.50 * w), y,
                          mz + (z - mz) * (1.0 - 0.45 * w))
    elif channel == "lip_round":
        for i, w in mouth.items():
            x, y, z = data[i].co
            data[i].co = (x * (1.0 - 0.42 * w), y + 0.013 * h * w,
                          mz + (z - mz) * (1.0 + 0.28 * w))
    elif channel == "lip_press":
        for i, w in mouth.items():
            x, y, z = data[i].co
            data[i].co = (x, y - 0.005 * h * w,
                          mz + (z - mz) * (1.0 - 0.62 * w))
    elif channel == "brow_raise":
        for i, w in brow.items():
            x, y, z = data[i].co
            data[i].co = (x, y, z + 0.018 * h * w)
    elif channel == "brow_furrow":
        for i, w in brow.items():
            x, y, z = data[i].co
            data[i].co = (x * (1.0 - 0.18 * w), y, z - 0.011 * h * w)
    elif channel == "squint":
        for i, w in eyes.items():
            x, y, z = data[i].co
            data[i].co = (x, y, ez + (z - ez) * (1.0 - 0.62 * w))
    elif channel == "smile":
        span = max([abs(data[i].co[0]) for i in mouth] or [1.0]) or 1.0
        for i, w in mouth.items():
            x, y, z = data[i].co
            corner = min(1.0, abs(x) / span)
            data[i].co = (x * (1.0 + 0.12 * w), y,
                          z + 0.013 * h * w * corner)


def _shape_keys(ob, anchors, dims):
    """The eight ``VISEME_CHANNELS`` keys, plus the face groups they came from."""
    mesh = ob.data
    verts = [tuple(v.co) for v in mesh.vertices]
    groups = _face_groups(verts, anchors)
    for name, weights in zip(("face_mouth", "face_brow", "face_eye"), groups):
        group = ob.vertex_groups.get(name) or ob.vertex_groups.new(name=name)
        for index, weight in weights.items():
            group.add([index], weight, "REPLACE")
    ob.shape_key_add(name="Basis", from_mix=False)
    for channel in VISEME_CHANNELS:
        key = ob.shape_key_add(name=channel, from_mix=False)
        key.slider_min, key.slider_max = 0.0, 1.0
        key.value = 0.0
        _shape_deform(channel, key.data, groups, anchors, dims["head"])
    return {name: len(weights) for name, weights
            in zip(("mouth", "brow", "eye"), groups)}


def _build_armature(name, frame, order, parents):
    armature = bpy.data.armatures.new(name)
    ob = bpy.data.objects.new(name, armature)
    bpy.context.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bones = {}
        for joint in order:
            head, tail = frame[joint]
            bone = armature.edit_bones.new(joint)
            bone.head, bone.tail = head, tail
            bones[joint] = bone
        for index, joint in enumerate(order):
            parent = parents[index]
            if parent >= 0:
                bones[joint].parent = bones[order[parent]]
                bones[joint].use_connect = (
                    (bones[joint].head - bones[order[parent]].tail).length < 1e-5)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    ob.show_in_front = True
    return ob


def character(name, height_m=1.75, preset="neutral", kind="human",
              clothed=True, beard=False, material_roles=None,
              at=(0.0, 0.0, 0.0), facing_deg=0.0, smooth_angle=50.0):
    """A rigged stand-in body: one mesh, one armature, eight viseme keys.

    M1 substitutes this for the SMPL-X/FLAME path. What it gets right is the
    *interface* - SMPL-X joint names and order, the eight ``VISEME_CHANNELS``
    shape keys, material roles instead of materials - so a later stage replaces
    the geometry without touching anything that reads it.

    ``smooth_angle`` defaults above the 45 degrees between two faces of an
    eight-sided prism, so limbs shade round while the boxes stay crisp.
    """
    quadruped = str(kind).lower() in ("quadruped", "animal", "creature")
    if quadruped:
        joints, dims = _quad_joints(height_m, preset)
        parts, anchors = _quad_parts(joints, dims, tack=clothed)
        order, parents, tails = QUAD_JOINTS, QUAD_PARENTS, QUAD_TAILS
    else:
        joints, dims = _rest_joints(height_m, preset)
        parts, anchors = _human_parts(joints, dims, clothed=clothed, beard=beard)
        order, parents, tails = SMPLX_JOINTS, SMPLX_PARENTS, HUMAN_TAILS

    verts, faces, spans = _weld(parts)
    verts, joints, anchors, dims = _fit(verts, joints, anchors, dims, height_m)
    mesh_ob, roles = _build_mesh(name, verts, faces, spans, material_roles)
    face_counts = _shape_keys(mesh_ob, anchors, dims)

    frame = _bone_frame(joints, order, parents, tails,
                        stub=0.06 * dims["factor"])
    rig = _build_armature(name + "_rig", frame, order, parents)
    for bone, weights in _bone_weights(verts, frame).items():
        if not weights:
            continue
        group = (mesh_ob.vertex_groups.get(bone)
                 or mesh_ob.vertex_groups.new(name=bone))
        for index, weight in weights:
            group.add([index], weight, "REPLACE")
    modifier = mesh_ob.modifiers.new("Armature", "ARMATURE")
    modifier.object = rig
    modifier.use_vertex_groups = True
    mesh_ob.parent = rig
    smooth(mesh_ob, smooth_angle)
    place(rig, location=at, rotation=(0.0, 0.0, facing_deg))

    rig["character"], mesh_ob["character"] = name, name
    rig["rest_height_m"] = dims["height"]
    rig["rig_kind"] = "quadruped" if quadruped else "smplx"
    return {"name": name, "mesh": mesh_ob, "rig": rig, "joints": joints,
            "dims": dims, "anchors": anchors, "roles": roles,
            "bones": list(order), "shape_keys": list(VISEME_CHANNELS),
            "kind": "quadruped" if quadruped else "human",
            "verts": len(verts), "faces": len(faces),
            "face_groups": face_counts}


# -- motion -----------------------------------------------------------------
# SMPL's canonical frame is Y-up with the body facing -Z; the rest pose above is
# Blender's Z-up facing +Y. A rotation given in the former becomes
# ``B @ M @ Bt`` in the latter, and a rotation in armature space becomes a bone's
# local rotation as ``rest^-1 @ M @ rest``. Both conversions are here, chosen by
# ``axes=``, so M3's real mocap output is a keyword change rather than a patch.
SMPL_TO_BLENDER = mathutils.Matrix(((1.0, 0.0, 0.0),
                                    (0.0, 0.0, -1.0),
                                    (0.0, 1.0, 0.0)))


def _fcurves(action):
    """Every fcurve in an action, on 4.x flat actions or 5.x slotted ones."""
    curves = getattr(action, "fcurves", None)
    if curves is not None:
        try:
            return list(curves)
        except TypeError:
            pass
    found = []
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for bag in getattr(strip, "channelbags", []):
                found.extend(bag.fcurves)
    return found


def _rotation_matrix(value):
    """A 3x3 from an axis-angle vector (SMPL-X), a quaternion or a matrix."""
    rows = list(value)
    if rows and hasattr(rows[0], "__len__"):
        matrix = mathutils.Matrix([tuple(row) for row in rows])
        return matrix if len(rows) == 3 else matrix.to_3x3()
    if len(rows) == 3:
        vector = mathutils.Vector(rows)
        angle = vector.length
        if angle < 1e-9:
            return mathutils.Matrix.Identity(3)
        return mathutils.Matrix.Rotation(angle, 3, vector / angle)
    if len(rows) == 4:
        return mathutils.Quaternion(rows).to_matrix()
    if len(rows) == 9:
        return mathutils.Matrix((rows[0:3], rows[3:6], rows[6:9]))
    raise ValueError("cannot read a rotation from %d numbers" % len(rows))


# The four retime policies of §5.2, plus the prelude's own ``none``. The
# audio fixes the length of a shot and the body take is whatever length it was
# recorded at, so one of these decides what happens in the gap - and *which*
# one is a directorial choice, per cast member, which is why it travels in
# shot.json rather than being decided here.
#
#   trim            Cut the take to the window. Frames past the end of the take
#                   are not keyed at all: the performance ends, and nothing
#                   pretends otherwise. The count of discarded frames comes
#                   back, so "my take was twice as long as the line" is visible.
#   hold_last_pose  Freeze the final pose for the rest of the window, keyed
#                   explicitly on the last frame so the hold is in the file and
#                   not merely in Blender's extrapolation.
#   loop            Cycle the take. Only invisible when the take's last frame
#                   steps into its first by no more than any other frame steps
#                   into its neighbour, which is exactly what e_motion's seam
#                   metric measures and why that metric exists.
#   timewarp        Resample the take to fill the window exactly, at whatever
#                   speed that takes. Rotations are slerped rather than blended
#                   component-wise: a component-wise blend of two rotations
#                   passes through the inside of the sphere, which shortens
#                   every limb it touches on the way.
#   none            Key the take once at its own length and stop. Behaves as
#                   ``trim`` does; kept because it says something different -
#                   that no policy was applied, rather than that one was.
RETIME_POLICIES = ("trim", "hold_last_pose", "loop", "timewarp", "none")


def _retime_plan(retime, count, total):
    """One entry per output frame: a source index, a blend, or None.

    Separated from the keying so the policy is a table that can be read and
    tested rather than a branch inside a loop over bones. ``None`` means "do
    not key this frame", which is the thing ``trim`` needs and no arithmetic on
    a source index can express.
    """
    if retime == "loop":
        return [offset % count for offset in range(total)]
    if retime == "hold_last_pose":
        return [min(offset, count - 1) for offset in range(total)]
    if retime in ("trim", "none"):
        return [offset if offset < count else None for offset in range(total)]
    if retime == "timewarp":
        if total == 1 or count == 1:
            return [0] * total
        plan = []
        for offset in range(total):
            # The first output frame is the first source frame and the last is
            # the last, so the take is stretched onto the window rather than
            # sampled at a rate that drifts off the end of it.
            position = offset * (count - 1) / float(total - 1)
            low = min(int(math.floor(position)), count - 2)
            plan.append((low, low + 1, position - low))
        return plan
    raise ValueError(
        "unknown retime policy %r; expected one of %s"
        % (retime, ", ".join(RETIME_POLICIES)))


def _pose_locals(pose, convert, rest, bones, correction=None, parents=None):
    """One source pose as ``{bone: local rotation quaternion}``.

    Computed once per source frame rather than once per use: ``loop`` reads
    every frame several times over a long shot, and ``timewarp`` reads most of
    them twice. Bones the rig has not got are dropped here, so the keying loop
    below never has to look one up.

    ``correction`` is a rig's ``rest_correction`` (see :func:`_rest_correction`):
    per bone, the rotation ``C`` carrying that rig's rest direction onto the
    T-pose every take is authored against. A take's armature-space rotation
    ``M`` then becomes ``C_parent^-1 @ M @ C_bone``, and a bone the take does not
    mention still gets ``C_parent^-1 @ C_bone`` if it hangs below one the take
    does -- a wrist under a keyed elbow would otherwise sit at its photo angle
    composed with the elbow's correction.

    Only *below* keyed bones. A head take keys the neck and head and nothing
    else; filling every corrected bone in the rig for it put the arms back in
    the T-pose after the body take had lowered them.
    """
    out = {}
    correction = correction or {}
    parents = parents or {}
    names = set(pose or {})
    if correction:
        grew = True
        while grew:
            extra = {name for name in bones
                     if name not in names and parents.get(name) in names
                     and (name in correction
                          or parents.get(name) in correction)}
            names |= extra
            grew = bool(extra)
    identity = mathutils.Matrix.Identity(3)
    for name in names:
        if name not in bones:
            continue
        value = (pose or {}).get(name)
        matrix = identity if value is None else _rotation_matrix(value)
        if convert is not None and value is not None:
            matrix = convert @ matrix @ convert.transposed()
        if correction:
            mine = correction.get(name, identity)
            theirs = correction.get(parents.get(name), identity)
            matrix = theirs.transposed() @ matrix @ mine
        basis = rest[name]
        out[name] = (basis.inverted() @ matrix @ basis).to_quaternion()
    return out


def _rest_correction(ob):
    """A rig's ``rest_correction`` property as ``{bone: 3x3}``, or ``{}``.

    Written by ``pipeline/blender/photo_body.py`` for a rig whose rest pose is
    the pose of the photo it was built from rather than the T-pose. A rig
    without the property -- every rig ``character()`` builds -- gets ``{}`` and
    ``apply_motion`` behaves exactly as it always has.
    """
    import json
    raw = ob.get("rest_correction") if hasattr(ob, "get") else None
    if not raw:
        return {}
    try:
        table = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return {}
    return {name: mathutils.Quaternion(value).to_matrix()
            for name, value in table.items()}


def _blend_locals(low, high, factor):
    """Slerp between two poses. A bone in only one of them keeps its rotation."""
    out = {}
    for name, rotation in low.items():
        other = high.get(name)
        out[name] = rotation if other is None else rotation.slerp(other, factor)
    for name, rotation in high.items():
        out.setdefault(name, rotation)
    return out


def _root_at(root, entry, root_scale):
    """The world translation for one plan entry, blended if it is a blend."""
    if isinstance(entry, tuple):
        low, high, factor = entry
        a = mathutils.Vector(tuple(root[min(low, len(root) - 1)])[:3])
        b = mathutils.Vector(tuple(root[min(high, len(root) - 1)])[:3])
        vector = a.lerp(b, factor)
    else:
        vector = mathutils.Vector(tuple(root[min(entry, len(root) - 1)])[:3])
    return vector * float(root_scale)


def apply_motion(rig, poses, root=None, start=1, frames=None, retime="loop",
                 axes="blender", root_scale=1.0):
    """Key a pose sequence onto an armature, fitted to the shot's window.

    ``poses[i]`` maps bone name to a rotation - an axis-angle vector in radians
    (what SMPL-X emits), a quaternion, or a 3x3. ``root[i]`` is an optional
    world translation for the first bone. ``frames`` is the length of the window
    the take has to fill, which §5.2 derives from the dialogue audio and not
    from the take; ``retime`` is how it is fitted, and is one of
    :data:`RETIME_POLICIES`.

    An unrecognised policy raises. It used to fall through to ``hold_last_pose``,
    which meant a shot asking for ``timewarp`` got a freeze and nothing said so -
    the failure mode this whole module is written to avoid.
    """
    ob = obj(rig["rig"] if isinstance(rig, dict) else rig)
    if ob.type != "ARMATURE":
        raise TypeError("%s is not an armature" % ob.name)
    poses = list(poses or [])
    if not poses:
        return {"frames": 0, "bones": 0, "retime": retime, "source_frames": 0}
    count = len(poses)
    total = max(1, int(frames or count))
    convert = SMPL_TO_BLENDER if str(axes).lower().startswith("smpl") else None

    rest = {}
    for bone in ob.pose.bones:
        bone.rotation_mode = "QUATERNION"
        rest[bone.name] = bone.bone.matrix_local.to_3x3()
    root_bone = next((b for b in ob.pose.bones if b.parent is None), None)

    plan = _retime_plan(retime, count, total)
    correction = _rest_correction(ob)
    parents = {bone.name: (bone.parent.name if bone.parent else None)
               for bone in ob.pose.bones}
    local = [_pose_locals(pose, convert, rest, rest, correction, parents)
             for pose in poses]

    touched, written, previous = set(), 0, None
    for offset, entry in enumerate(plan):
        if entry is None:
            continue
        if entry == previous and offset not in (0, total - 1):
            # A held pose does not need a key per frame. The last frame is
            # always keyed anyway, so the hold is explicit at both ends.
            continue
        previous = entry
        frame = start + offset
        pose = (_blend_locals(local[entry[0]], local[entry[1]], entry[2])
                if isinstance(entry, tuple) else local[entry])
        for name, rotation in pose.items():
            bone = ob.pose.bones[name]
            bone.rotation_quaternion = rotation
            bone.keyframe_insert("rotation_quaternion", frame=frame)
            touched.add(name)
        if root and root_bone is not None:
            vector = _root_at(root, entry, root_scale)
            if convert is not None:
                vector = convert @ vector
            root_bone.location = rest[root_bone.name].inverted() @ vector
            root_bone.keyframe_insert("location", frame=frame)
        written += 1

    keyed = [offset for offset, entry in enumerate(plan) if entry is not None]
    return {
        "frames": written,
        "span": [start, start + total - 1],
        "keyed_span": ([start + keyed[0], start + keyed[-1]] if keyed
                       else [start, start]),
        "bones": len(touched),
        "retime": retime,
        "axes": axes,
        "source_frames": count,
        "target_frames": total,
        # What the policy actually did, so a shot's report says whether the
        # take was cut, held, cycled or stretched, and by how much.
        "trimmed_frames": max(0, count - total) if retime in ("trim", "none") else 0,
        "held_frames": max(0, total - count) if retime == "hold_last_pose" else 0,
        "cycles": round(total / float(count), 3) if retime == "loop" else None,
        "speed": (round((count - 1) / float(total - 1), 4)
                  if retime == "timewarp" and total > 1 else None),
        "missing_bones": sorted({name for pose in poses for name in (pose or {})}
                                - set(rest)),
    }


# -- visemes ----------------------------------------------------------------
_GAIN_GROUPS = {
    "jaw_gain": ("jaw_open",),
    "lip_gain": ("lip_wide", "lip_round", "lip_press", "smile"),
    "brow_gain": ("brow_raise", "brow_furrow", "squint"),
}


def _shape_target(target):
    if isinstance(target, dict):
        target = target.get("mesh") or target.get("rig")
    ob = obj(target)
    if getattr(getattr(ob, "data", None), "shape_keys", None):
        return ob
    for child in ob.children:
        if getattr(getattr(child, "data", None), "shape_keys", None):
            return child
    raise ValueError("%s has no shape keys to drive" % ob.name)


def _series(curves, channels):
    """``{channel: [value, ...]}`` from either that shape or a list of frames."""
    if isinstance(curves, dict):
        return {name: list(values) for name, values in curves.items()
                if name in channels}
    out = {}
    for index, frame in enumerate(curves or []):
        values = frame.get("channels", frame) if isinstance(frame, dict) else {}
        for name, value in values.items():
            if name not in channels:
                continue
            column = out.setdefault(name, [])
            while len(column) < index:
                column.append(column[-1] if column else 0.0)
            column.append(float(value))
    return out


def _smooth_series(values, passes):
    for _ in range(max(0, int(passes))):
        if len(values) < 3:
            break
        values = ([(values[0] * 3 + values[1]) / 4.0]
                  + [(values[i - 1] + values[i] * 2 + values[i + 1]) / 4.0
                     for i in range(1, len(values) - 1)]
                  + [(values[-1] * 3 + values[-2]) / 4.0])
    return values


def apply_visemes(target, curves, start=1, profile=None, interpolation=None):
    """Key the eight mouth channels onto a body's shape keys.

    ``profile`` is a style pack's ``visemes`` block, applied as given: the
    graphic look holds each sample for ``hold_frames`` and steps between them,
    the smooth looks run a box filter ``smoothing_frames`` times. Gains are
    per group, because a stylised mouth wants a wider one than a photoreal face.
    """
    ob = _shape_target(target)
    blocks = ob.data.shape_keys.key_blocks
    profile = profile or {}
    gains = {}
    for key, names in _GAIN_GROUPS.items():
        value = profile.get(key)
        if value is not None:
            for name in names:
                gains[name] = float(value)
    hold = int(profile.get("hold_frames") or 0)
    step = max(1, hold + 1)
    smoothing = int(profile.get("smoothing_frames") or 0)

    written, last, peaks = 0, start, {}
    for channel, values in _series(curves, set(VISEME_CHANNELS)).items():
        block = blocks.get(channel)
        if block is None or not values:
            continue
        values = _smooth_series([float(v) for v in values], smoothing)
        gain = gains.get(channel, 1.0)
        for index in range(len(values)):
            if step > 1 and index % step and index != len(values) - 1:
                continue
            sample = values[(index // step) * step] if step > 1 else values[index]
            block.value = max(0.0, min(1.0, sample * gain))
            block.keyframe_insert("value", frame=start + index)
            written += 1
        peaks[channel] = round(max(values) * gain, 4)
        last = max(last, start + len(values) - 1)

    action = getattr(getattr(ob.data.shape_keys, "animation_data", None),
                     "action", None)
    if action is not None:
        want = interpolation or ("CONSTANT" if hold else "BEZIER")
        for fcurve in _fcurves(action):
            for point in fcurve.keyframe_points:
                point.interpolation = want
    return {"channels": sorted(peaks), "keys": written,
            "span": [start, last], "peaks": peaks,
            "profile": profile.get("profile", "linear"),
            "hold_frames": hold, "smoothing_frames": smoothing}


# -- garments ---------------------------------------------------------------
def bind_garment(body, garment, offset=0.0, samples=3):
    """Make a garment deform with the body it was modelled over.

    Weights come from the nearest body vertices, inverse-distance blended. M5
    replaces this with Surface Deform plus correctives; what stays the same is
    the contract - the garment is a separate object, built in the body's *rest*
    space, that ends up driven by the body's armature.
    """
    body_ob = obj(body["mesh"] if isinstance(body, dict) else body)
    cloth = obj(garment)
    rig = body.get("rig") if isinstance(body, dict) else None
    if rig is None and body_ob.parent is not None:
        rig = body_ob.parent if body_ob.parent.type == "ARMATURE" else None
    rig = obj(rig) if rig is not None else None

    vertices = body_ob.data.vertices
    tree = mathutils.kdtree.KDTree(len(vertices))
    for vertex in vertices:
        tree.insert(body_ob.matrix_world @ vertex.co, vertex.index)
    tree.balance()

    bones = {bone.name for bone in rig.pose.bones} if rig else set()
    names = {group.index: group.name for group in body_ob.vertex_groups}
    weights = {}
    for vertex in vertices:
        row = [(names[entry.group], entry.weight) for entry in vertex.groups
               if names.get(entry.group) in bones]
        if row:
            weights[vertex.index] = row

    bound, used = 0, set()
    for vertex in cloth.data.vertices:
        near = tree.find_n(cloth.matrix_world @ vertex.co, max(1, int(samples)))
        blend, total = {}, 0.0
        for _, index, distance in near:
            row = weights.get(index)
            if not row:
                continue
            share = 1.0 / (distance + 1e-3)
            total += share
            for bone, weight in row:
                blend[bone] = blend.get(bone, 0.0) + weight * share
        if not blend or total <= 0.0:
            continue
        for bone, weight in blend.items():
            value = min(1.0, weight / total)
            if value <= 0.004:
                continue
            group = (cloth.vertex_groups.get(bone)
                     or cloth.vertex_groups.new(name=bone))
            group.add([vertex.index], value, "REPLACE")
            used.add(bone)
        bound += 1

    if offset:
        push = cloth.modifiers.new("Offset", "DISPLACE")
        push.mid_level = 0.0
        push.strength = float(offset)
        push.direction = "NORMAL"
    if rig is not None:
        modifier = cloth.modifiers.new("Armature", "ARMATURE")
        modifier.object = rig
        modifier.use_vertex_groups = True
        cloth.parent = rig
        cloth.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    return {"garment": cloth.name, "body": body_ob.name,
            "bound": bound, "of": len(cloth.data.vertices),
            # The names, not the count. Whether a garment follows the rig can
            # only be tested by posing a bone it is actually bound to, and the
            # count cannot say which those are.
            "bones": sorted(used), "offset": float(offset)}


# -- style packs ------------------------------------------------------------
# This module is exec'd into Blender as a synthetic module with no package, so
# it cannot import its siblings the ordinary way. Every caller inside the
# pipeline already knows the repo root; it passes it in and the import works.
def load_style(pack="miniature", repo_root=None):
    """A ``StylePack``, from a name or passed through if already loaded."""
    if hasattr(pack, "material") and hasattr(pack, "render"):
        return pack
    import sys
    if repo_root:
        root = os.path.abspath(repo_root)
        if root not in sys.path:
            sys.path.insert(0, root)
    from blender_ai_studio import styles
    return styles.load(pack)


def _apply_render(pack):
    scene = bpy.context.scene
    settings = pack.render
    engine = settings.get("engine", "BLENDER_EEVEE")
    set_engine(engine, samples=settings.get("samples"))
    set_view_transform(settings.get("view_transform", "AgX"),
                       look=settings.get("look", "None"),
                       exposure=settings.get("exposure", 0.0))
    scene.render.film_transparent = bool(settings.get("film_transparent"))
    scene.render.use_motion_blur = bool(settings.get("motion_blur"))
    if engine.upper().startswith("CYCLES"):
        cycles = scene.cycles
        for key, attribute in (("adaptive_threshold", "adaptive_threshold"),
                               ("max_bounces", "max_bounces"),
                               ("diffuse_bounces", "diffuse_bounces"),
                               ("glossy_bounces", "glossy_bounces"),
                               ("transmission_bounces", "transmission_bounces")):
            if settings.get(key) is not None:
                try:
                    setattr(cycles, attribute, settings[key])
                except Exception:                               # noqa: BLE001
                    pass
        cycles.use_adaptive_sampling = settings.get("adaptive_threshold") is not None
        cycles.use_denoising = bool(settings.get("denoise", True))
        if settings.get("denoiser"):
            try:
                cycles.denoiser = settings["denoiser"]
            except Exception:                                   # noqa: BLE001
                pass
        scene.render.use_persistent_data = bool(settings.get("persistent_data"))
    else:
        eevee = scene.eevee
        for key in ("use_raytracing", "use_shadows"):
            if settings.get(key) is not None and hasattr(eevee, key):
                try:
                    setattr(eevee, key, bool(settings[key]))
                except Exception:                               # noqa: BLE001
                    pass
        for key, attribute in (("shadow_ray_count", "shadow_ray_count"),
                               ("shadow_step_count", "shadow_step_count")):
            if settings.get(key) is not None and hasattr(eevee, attribute):
                try:
                    setattr(eevee, attribute, int(settings[key]))
                except Exception:                               # noqa: BLE001
                    pass
    return scene.render.engine


def _lamp_position(spec, target, scale=1.0):
    """Azimuth 0 is the camera side; positive azimuth swings to the left."""
    azimuth = math.radians(float(spec.get("azimuth_deg", 0.0)))
    elevation = math.radians(float(spec.get("elevation_deg", 35.0)))
    distance = float(spec.get("distance_m", 6.0)) * float(scale)
    return (target[0] + distance * math.cos(elevation) * math.sin(azimuth),
            target[1] - distance * math.cos(elevation) * math.cos(azimuth),
            target[2] + distance * math.sin(elevation))


def _apply_lighting(pack, target=(0.0, 0.0, 0.0), distance_scale=1.0,
                    clear=True):
    lighting = pack.lighting
    target = tuple(_vec(target))
    if clear:
        for existing in [o for o in bpy.data.objects if o.type == "LIGHT"]:
            bpy.data.objects.remove(existing, do_unlink=True)

    hdri = lighting.get("hdri") or {}
    path = hdri.get("path")
    colour = lighting.get("world_color") or (0.5, 0.5, 0.5)
    strength = float(lighting.get("world_strength", 1.0))
    if path and os.path.exists(str(path)):
        hdri_file(str(path), strength=float(hdri.get("strength", 1.0)),
                  rotation=float(hdri.get("rotation_deg", 0.0)))
        world = "hdri:" + os.path.basename(str(path))
    elif lighting.get("rig") == "hdri":
        # The pack asks for an environment map and has none. A flat colour would
        # kill every reflection, so stand in a sky gradient and say so.
        world_gradient(top=[min(1.0, c * 1.18) for c in colour[:3]],
                       bottom=[c * 0.42 for c in colour[:3]],
                       strength=strength)
        world = "gradient (no hdri path set)"
    else:
        world_color(colour, strength=strength)
        world = "color"

    made = []
    for slot in ("key", "fill", "rim"):
        spec = lighting.get(slot)
        if not spec:
            continue
        kind = str(spec.get("type", "SUN")).upper()
        energy = float(spec.get("energy", 3.0))
        colour = spec.get("color", (1.0, 1.0, 1.0))
        if kind == "SUN":
            light = sun(direction=(90.0 - float(spec.get("elevation_deg", 45.0)),
                                   0.0, float(spec.get("azimuth_deg", 0.0))),
                        energy=energy, color=colour,
                        angle=float(spec.get("angle_deg", 2.0)))
        elif kind == "AREA":
            light = area_light(
                location=_lamp_position(spec, target, distance_scale),
                target=target, energy=energy,
                size=float(spec.get("size_m", 3.0)), color=colour)
        else:
            light = point_light(
                location=_lamp_position(spec, target, distance_scale),
                energy=energy, color=colour)
        light.name = "%s_%s" % (kind.title(), slot)
        made.append(light.name)
    return {"rig": lighting.get("rig"), "world": world, "lights": made}


def _swap_materials(pack, engine=None):
    """Replace every role-tagged *placeholder* with this pack's version of it.

    A material carrying ``style_plate`` is skipped. It was painted from one of
    the reference images the user generated by hand (§7-B, decision 5), which
    makes it content rather than a role placeholder: swapping it throws the
    image away and leaves a flat colour. That is precisely what happened to the
    miniature pack's painted backdrop — §7-D's whole reason for that strategy
    existing — from the first env reference image the project ever had.

    The role tag stays on it, so the material is still identifiable and a pack
    that genuinely wants to override a plate can still find it. Both counts come
    back so a run says what it replaced *and* what it left alone.
    """
    from blender_ai_studio.styles import shading_common
    engine = engine or bpy.context.scene.render.engine
    swapped, kept = {}, {}
    prefix = pack.name + "_"
    for material in list(bpy.data.materials):
        role = material.get("style_role")
        if not role or material.name.startswith(prefix):
            continue
        # A character built from its photo carries ``style_photo`` for the
        # same reason a plate does: its colour is the user's image, and
        # swapping it for the role placeholder rendered every character in
        # the first photo-built film as flat cream.
        if material.get("style_plate") or material.get("style_photo"):
            kept[role] = kept.get(role, 0) + 1
            continue
        replacement = shading_common.material_for(pack, role, engine)
        replacement["style_role"] = role
        material.user_remap(replacement)
        swapped[role] = swapped.get(role, 0) + 1
    return swapped, kept


def style(pack="miniature", repo_root=None, target=(0.0, 0.0, 0.0),
          distance_scale=1.0, materials=True, lighting=True, render=True,
          postfx=False, clear_lights=True):
    """Apply a style pack to the current scene: render, lights, materials.

    Materials are swapped by their ``style_role`` tag, so this works on a scene
    that was linked in from asset .blend files built by an earlier stage, and it
    can be called twice to change pack without rebuilding anything.

    Resolution is deliberately not touched: ``render.resolution_scale`` belongs
    to the render stage, where it can enter the artifact key.
    """
    pack = load_style(pack, repo_root)
    out = {"pack": pack.name}
    if render:
        out["engine"] = _apply_render(pack)
    if lighting:
        out.update(_apply_lighting(pack, target, distance_scale, clear_lights))
    if materials:
        out["materials"], out["materials_kept"] = _swap_materials(pack)
    if postfx:
        from blender_ai_studio.styles import postfx_common
        out["postfx"] = postfx_common.build(pack)
    return out

















