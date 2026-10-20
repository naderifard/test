"""The compositor half of a style pack, built from ``style.json``'s ``postfx``.

Blender 5.x replaced the per-scene compositor with a node *group* assigned to
``scene.compositing_node_group``, and dropped several classic nodes
(``CompositorNodeComposite``, ``CompositorNodeMixRGB`` is now the generic shader
Mix, ``CompositorNodeMath`` is gone). Everything here was probed against
Blender 5.2 rather than carried over from 4.x habits.

The chain, in order, with every link optional and driven by data::

    Render Layers -> painterly -> ink outline -> contrast -> saturation
                  -> lift/gain -> glare -> vignette -> Group Output

:func:`build` returns the list of effects it actually applied, which the render
stage records as a metric — so "the outline is missing" is visible in the run
summary instead of being something you notice in the video.
"""

from __future__ import annotations

import bpy

GROUP_NAME = "style_postfx"


def _rgba(value, alpha=1.0):
    seq = [float(v) for v in (value or (0.0, 0.0, 0.0))]
    while len(seq) < 3:
        seq.append(seq[-1] if seq else 0.0)
    return (seq[0], seq[1], seq[2], seq[3] if len(seq) > 3 else alpha)


def _socket(node, *wanted):
    """Look a socket up by identifier first, then by label.

    5.x gives several nodes two sockets sharing a *label*: ColorBalance has
    ``Base Gain`` (a float) and ``Color Gain`` (a colour) both displayed as
    "Gain", so identifier has to win or a scalar lands on the wrong one.
    """
    for want in wanted:
        for sock in node.inputs:
            if sock.identifier == want:
                return sock
        for sock in node.inputs:
            if sock.name == want:
                return sock
    return None


def _set(node, wanted, value):
    """Set an input socket if this Blender has it. Returns whether it landed."""
    sock = _socket(node, wanted) if isinstance(wanted, str) else wanted
    if sock is None:
        return False
    try:
        current = sock.default_value
        if isinstance(current, str) or not hasattr(current, "__len__"):
            sock.default_value = value
        else:
            width = len(current)
            seq = list(value) if hasattr(value, "__len__") \
                else [float(value)] * width
            sock.default_value = _rgba(seq) if width == 4 \
                else tuple((seq + seq[-1:] * width)[:width])
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def _set_enum(node, prop, socket, *values):
    """Set a node mode, trying each candidate spelling in turn.

    5.x moved these from properties to menu sockets, and the socket form wants
    the label ("Sobel"), not the 4.x identifier ("SOBEL"), so callers pass both.
    """
    for value in values:
        if prop and hasattr(node, prop):
            try:
                setattr(node, prop, value)
                return True
            except (TypeError, ValueError):
                pass
        sock = _socket(node, socket) if socket else None
        if sock is not None:
            try:
                sock.default_value = value
                return True
            except (TypeError, ValueError):
                pass
    return False


def _mix(tree, blend_type, factor, a, b, location=(0, 0)):
    """Mix socket ``a`` with ``b``, which may be a socket or a literal colour."""
    node = tree.nodes.new("ShaderNodeMixRGB")
    node.blend_type = blend_type
    node.location = location
    _set(node, "Factor", float(factor))
    tree.links.new(a, node.inputs["Color1"])
    if hasattr(b, "node"):
        tree.links.new(b, node.inputs["Color2"])
    else:
        _set(node, "Color2", _rgba(b))
    return node


def build(pack, scene=None):
    """Wire the pack's post-processing. Returns the effects applied."""
    scene = scene or bpy.context.scene
    settings = dict(pack.postfx or {})
    hook = pack.hook("postfx")
    if hook is not None and hasattr(hook, "build"):
        applied = hook.build(pack, scene)
        if applied is not None:
            return applied

    outline = settings.get("outline") or {}
    needs_normal = bool(outline.get("enabled"))

    view_layer = scene.view_layers[0]
    if needs_normal:
        view_layer.use_pass_normal = True
        view_layer.use_pass_z = True
        view_layer.use_pass_mist = True
        # Mist is a 0..1 remap of depth, so Sobel on it is scale-independent in
        # a way Sobel on raw Z is not -- but only once its range covers the set.
        world = scene.world
        if world is not None and hasattr(world, "mist_settings"):
            reach = float((pack.env or {}).get("ground_scale_m", 30) or 30)
            world.mist_settings.start = 0.5
            world.mist_settings.depth = max(8.0, reach * 0.75)

    tree = bpy.data.node_groups.get(GROUP_NAME)
    if tree is not None:
        bpy.data.node_groups.remove(tree)
    tree = bpy.data.node_groups.new(GROUP_NAME, "CompositorNodeTree")
    tree.interface.new_socket("Image", in_out="OUTPUT",
                              socket_type="NodeSocketColor")
    scene.compositing_node_group = tree

    layers = tree.nodes.new("CompositorNodeRLayers")
    layers.scene = scene
    layers.layer = view_layer.name
    layers.location = (-900, 0)
    signal = layers.outputs["Image"]
    applied = []
    column = -600

    painterly = settings.get("painterly") or {}
    if painterly.get("kuwahara"):
        node = tree.nodes.new("CompositorNodeKuwahara")
        node.location = (column, 0)
        _set_enum(node, "variation", "Type", "Anisotropic", "ANISOTROPIC")
        _set(node, "Size", float(painterly["kuwahara"]))
        _set(node, "Uniformity", 6.0)
        tree.links.new(signal, node.inputs["Image"])
        signal = node.outputs["Image"]
        applied.append(f"kuwahara={painterly['kuwahara']}")
        column += 200

    if painterly.get("posterize"):
        node = tree.nodes.new("CompositorNodePosterize")
        node.location = (column, 0)
        _set(node, "Steps", int(painterly["posterize"]))
        tree.links.new(signal, node.inputs["Image"])
        signal = node.outputs["Image"]
        applied.append(f"posterize={painterly['posterize']}")
        column += 200

    if needs_normal:
        edges = _edges(tree, layers, outline, column)
        if edges is not None:
            ink = _mix(tree, "MIX", 1.0, signal, outline.get("color", [0, 0, 0]),
                       (column + 600, 0))
            tree.links.new(edges, ink.inputs["Factor"])
            signal = ink.outputs["Color"]
            applied.append(f"outline@{outline.get('threshold')}")
            column += 800

    grade = settings.get("grade") or {}
    if abs(float(grade.get("contrast", 1.0)) - 1.0) > 1e-6 \
            or abs(float(grade.get("lift", 0.0))) > 1e-6:
        node = tree.nodes.new("CompositorNodeBrightContrast")
        node.location = (column, 0)
        # Blender's Contrast is a -100..100 offset around 0, not a multiplier.
        _set(node, "Contrast", (float(grade.get("contrast", 1.0)) - 1.0) * 100.0)
        _set(node, "Brightness", float(grade.get("lift", 0.0)) * 100.0)
        tree.links.new(signal, node.inputs["Image"])
        signal = node.outputs["Image"]
        applied.append(f"contrast={grade.get('contrast')}")
        column += 200

    if abs(float(grade.get("saturation", 1.0)) - 1.0) > 1e-6:
        node = tree.nodes.new("CompositorNodeHueSat")
        node.location = (column, 0)
        _set(node, "Saturation", float(grade["saturation"]))
        tree.links.new(signal, node.inputs["Image"])
        signal = node.outputs["Image"]
        applied.append(f"saturation={grade['saturation']}")
        column += 200

    gain = float(grade.get("gain", 1.0))
    if abs(gain - 1.0) > 1e-6:
        node = tree.nodes.new("CompositorNodeColorBalance")
        node.location = (column, 0)
        _set_enum(node, "correction_method", "Type",
                  "Lift/Gamma/Gain", "LIFT_GAMMA_GAIN")
        if not _set(node, "Base Gain", gain):
            _set(node, "Color Gain", (gain, gain, gain, 1.0))
        tree.links.new(signal, node.inputs["Image"])
        signal = node.outputs["Image"]
        applied.append(f"gain={gain}")
        column += 200

    glare = settings.get("glare") or settings.get("bloom") or {}
    if glare:
        node = tree.nodes.new("CompositorNodeGlare")
        node.location = (column, 0)
        _set_enum(node, "glare_type", "Type", "Bloom", "BLOOM")
        _set_enum(node, "quality", "Quality", "Medium", "MEDIUM")
        _set(node, "Highlights Threshold", float(glare.get("threshold", 1.0)))
        _set(node, "Strength", float(glare.get("strength", 0.2)))
        _set(node, "Size", float(glare.get("size", 0.5)))
        tree.links.new(signal, node.inputs["Image"])
        signal = node.outputs["Image"]
        applied.append(f"glare={glare.get('strength', 0.2)}")
        column += 200

    paper = float(settings.get("paper_texture", 0.0) or 0.0)
    if paper > 1e-6:
        plate = _noise(tree, scene, "style_paper", 2.0, (column - 200, -700))
        if plate is not None:
            signal = _mix(tree, "OVERLAY", paper, signal, plate,
                          (column, 0)).outputs["Color"]
            applied.append(f"paper={paper}")
            column += 200

    grain = float(settings.get("grain", 0.0) or 0.0)
    if grain > 1e-6:
        plate = _noise(tree, scene, "style_grain", 0.0, (column - 200, -700),
                       seed=19)
        if plate is not None:
            signal = _mix(tree, "SOFT_LIGHT", min(1.0, grain * 8.0), signal,
                          plate, (column, 0)).outputs["Color"]
            applied.append(f"grain={grain}")
            column += 200

    vignette = float(settings.get("vignette", 0.0) or 0.0)
    if vignette > 1e-6:
        mask = tree.nodes.new("CompositorNodeEllipseMask")
        mask.location = (column, -420)
        _set_enum(mask, "mask_type", "Operation", "Add", "ADD")
        _set(mask, "Position", (0.5, 0.5))
        _set(mask, "Size", (0.58, 0.62))
        _set(mask, "Value", 1.0)
        blur = tree.nodes.new("CompositorNodeBlur")
        blur.location = (column + 200, -420)
        _set_enum(blur, "filter_type", "Type", "Fast Gaussian", "FAST_GAUSS")
        _set(blur, "Size", (150.0, 150.0))
        tree.links.new(mask.outputs["Mask"], blur.inputs["Image"])
        signal = _mix(tree, "MULTIPLY", vignette, signal,
                      blur.outputs["Image"], (column + 400, 0)).outputs["Color"]
        applied.append(f"vignette={vignette}")
        column += 600

    output = tree.nodes.new("NodeGroupOutput")
    output.location = (column + 200, 0)
    tree.links.new(signal, output.inputs[0])
    return applied


def _edges(tree, layers, outline, column):
    """A hard ink factor built from the Normal and Mist passes.

    Sobel on the normals catches creases and the silhouette; Sobel on mist
    catches one figure standing in front of another where their normals happen
    to agree. A CONSTANT ramp then turns the response into a line rather than a
    soft gradient, which is what separates an ink outline from a dark halo.
    """
    sources = []
    for name in ("Normal", "Mist"):
        source = layers.outputs.get(name)
        if source is None or not source.enabled:
            continue
        node = tree.nodes.new("CompositorNodeFilter")
        node.location = (column, -260 * (len(sources) + 1))
        if not _set_enum(node, "filter_type", "Type", "Sobel", "SOBEL"):
            return None
        _set(node, "Factor", 1.0)
        tree.links.new(source, node.inputs["Image"])
        sources.append(node.outputs["Image"])
    if not sources:
        return None

    combined = sources[0]
    for extra in sources[1:]:
        combined = _mix(tree, "LIGHTEN", 1.0, combined, extra,
                        (column + 200, -400)).outputs["Color"]

    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.location = (column + 400, -300)
    ramp.color_ramp.interpolation = "CONSTANT"
    threshold = float(outline.get("threshold", 0.3) or 0.3)
    elements = ramp.color_ramp.elements
    elements[0].position = 0.0
    elements[0].color = (0.0, 0.0, 0.0, 1.0)
    elements[1].position = min(0.99, max(0.01, threshold))
    elements[1].color = (1.0, 1.0, 1.0, 1.0)
    tree.links.new(combined, ramp.inputs["Factor"])
    edge = ramp.outputs["Color"]

    width = float(outline.get("width_px", 1.0) or 1.0)
    if width > 1.5:
        grow = tree.nodes.new("CompositorNodeDilateErode")
        grow.location = (column + 600, -300)
        _set(grow, "Size", int(round(width - 1.0)))
        tree.links.new(edge, grow.inputs["Mask"])
        edge = grow.outputs["Mask"]
    return edge


def _noise(tree, scene, name, blur_px=0.0, location=(0, 0), seed=7):
    """A static noise plate.

    5.x's compositor has no procedural texture node, so paper grain and film
    grain both come from a generated image datablock sized to the render. It is
    static rather than per-frame, which is what paper wants anyway; for film
    grain it trades a little life for not writing a texture per frame.

    Because a GENERATED image is re-filled on file load, this has to be built
    at render time, not baked into the .blend by ``h_assemble``.
    """
    if not hasattr(bpy.types, "CompositorNodeImage"):
        return None
    percent = getattr(scene.render, "resolution_percentage", 100) / 100.0
    width = max(16, int(scene.render.resolution_x * percent))
    height = max(16, int(scene.render.resolution_y * percent))
    image = bpy.data.images.get(name)
    if image is not None and tuple(image.size) != (width, height):
        bpy.data.images.remove(image)
        image = None
    if image is None:
        image = bpy.data.images.new(name, width, height, alpha=False,
                                    float_buffer=True)
        _fill_noise(image, width, height, seed)

    node = tree.nodes.new("CompositorNodeImage")
    node.image = image
    node.location = location
    out = node.outputs["Image"]
    if blur_px > 0.0:
        blur = tree.nodes.new("CompositorNodeBlur")
        blur.location = (location[0] + 200, location[1])
        _set_enum(blur, "filter_type", "Type", "Fast Gaussian", "FAST_GAUSS")
        _set(blur, "Size", (blur_px, blur_px))
        tree.links.new(out, blur.inputs["Image"])
        out = blur.outputs["Image"]
    return out


def _fill_noise(image, width, height, seed, spread=0.6):
    count = width * height
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        data = np.empty((count, 4), dtype="float32")
        grey = 0.5 + (rng.random((count, 1), dtype="float32") - 0.5) * spread
        data[:, 0:3] = grey
        data[:, 3] = 1.0
        image.pixels.foreach_set(data.reshape(-1))
    except ImportError:                      # Blender ships numpy; belt and braces
        import random
        rng = random.Random(seed)
        flat = [0.0] * (count * 4)
        for index in range(count):
            grey = 0.5 + (rng.random() - 0.5) * spread
            base = index * 4
            flat[base] = flat[base + 1] = flat[base + 2] = grey
            flat[base + 3] = 1.0
        image.pixels.foreach_set(flat)
    image.update()
    return image


