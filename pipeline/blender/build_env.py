"""d_env, in Blender: the set for one environment, or the primitive for one prop.

The strategy comes from the style pack (§7-D), and the three differ in what the
distance is made of: painted flats for *miniature*, nothing but a large ground
and the world gradient for *stylized*, displaced terrain for *photoreal*. Every
surface is a role material, so a pack still swaps the look at assemble time -
the strategy decides the geometry, never the shading.

Scatter instances are linked duplicates of one source mesh, so 240 shrubs cost
one mesh; over displaced terrain each one is dropped onto the surface with a
ray cast, because a photoreal set with plants hovering at z=0 is worse than a
flat one.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from pipeline import jsonio                                        # noqa: E402
from pipeline.blender import _common                               # noqa: E402
from pipeline.stages import _base                                  # noqa: E402

import bpy                                                         # noqa: E402

B = _common.install()

TERRAIN_STRATEGY = "hdri_terrain"
BACKDROP_STRATEGY = "painted_backdrop"

# Clearance between the outermost scattered thing and the edge of the world, so
# an instance whose own radius straddles the boundary still has floor under it.
GROUND_MARGIN_M = 2.0

# One source mesh per scatter role, in metres. The shrub is a flattened ball so
# it reads as a clump rather than a bead; the rock is left faceted on purpose.
SOURCES = {
    "foliage": {"kind": "icosphere", "radius": 0.35, "subdivisions": 1,
                "scale": (1.0, 1.0, 0.62), "z": 0.14, "smooth": True},
    "stone": {"kind": "icosphere", "radius": 0.40, "subdivisions": 1,
              "scale": (1.0, 0.78, 0.52), "z": 0.08, "smooth": False},
}
DEFAULT_SOURCE = {"kind": "cube", "size": 0.5, "scale": (1.0, 1.0, 1.0),
                  "z": 0.25, "smooth": False}


def _seed(*parts):
    return int(hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()[:8],
               16)


def _plate(project_root, refs_json):
    """The reference image a painted backdrop is painted with, if one exists."""
    refs = jsonio.read(refs_json, default={}) or {}
    images = refs.get("images") or []
    if not images:
        return None
    slots = refs.get("slots") or {}
    name = (slots.get("wide") or slots.get("front") or slots.get("back")
            or images[0].get("path"))
    folder = refs.get("folder")
    if not name or not folder:
        return None
    path = os.path.join(project_root, folder.replace("/", os.sep), name)
    return path if os.path.exists(path) else None


def _plate_material(name, path, role):
    """A role-tagged material that shows an image, not a flat colour.

    The pack still owns the look through ``style_role``: swapping the role
    replaces this material wholesale, which is the right behaviour for
    ``photoreal`` and exactly what ``miniature`` does not want, so only the
    painted-backdrop strategy asks for it.
    """
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree

    # Clear first, *then* fetch the output node. A Python reference to a node
    # does not survive `nodes.remove()` on the same tree: Blender reallocates
    # the collection and the old pointer dangles, so reading `.name` off it
    # comes back as freed bytes and `.inputs["Surface"]` raises KeyError. That
    # is what this did, and because a painted backdrop only builds when an env
    # reference image exists, the path had never once run. `prelude._world_tree`
    # has always done it in this order, which is why the world has never broken.
    for node in list(tree.nodes):
        if node.bl_idname != "ShaderNodeOutputMaterial":
            tree.nodes.remove(node)
    output = next((n for n in tree.nodes
                   if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    if output is None:
        output = tree.nodes.new("ShaderNodeOutputMaterial")

    shader = tree.nodes.new("ShaderNodeBsdfDiffuse")
    texture = tree.nodes.new("ShaderNodeTexImage")
    image = bpy.data.images.load(path, check_existing=True)

    # Pack the pixels into the .blend. Blender stores an image as a *path*, and
    # relative paths are resolved against the file holding them -- so an image
    # loaded here survives inside env.blend and then arrives empty when
    # h_assemble appends the collection into shots/<id>/scene.blend, two
    # directories away. The symptom is not an error: the image datablock is
    # present and linked, reports size (0, 0), and renders as flat black through
    # the diffuse shader, which looks exactly like a style choice.
    #
    # Packing makes the asset carry its own pixels, which is what an artifact in
    # a content-addressed store should do anyway: env.blend is then the whole
    # environment, not a file plus a hope about where it sits.
    try:
        image.pack()
    except Exception as exc:                                    # noqa: BLE001
        print("could not pack %s: %s" % (os.path.basename(path), exc))
    texture.image = image
    texture.extension = "EXTEND"
    tree.links.new(texture.outputs["Color"], shader.inputs["Color"])
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    material["style_role"] = role
    material["style_plate"] = os.path.basename(path)
    material["style_plate_px"] = "%dx%d" % tuple(image.size)
    return material


def _ground(strategy, span, role):
    """The floor. Flat, except for photoreal, which gets real relief."""
    if strategy != TERRAIN_STRATEGY:
        ground = B.floor(size=span, material=B.role_material(role),
                         name="ground")
        return ground, 0.0
    ground = B.add("grid", name="ground", size=span,
                   x_subdivisions=120, y_subdivisions=120)
    B.assign(ground, B.role_material(role))
    texture = bpy.data.textures.get("terrain") \
        or bpy.data.textures.new("terrain", type="CLOUDS")
    texture.noise_scale = span / 12.0
    texture.noise_depth = 3
    modifier = ground.modifiers.new("relief", type="DISPLACE")
    modifier.texture = texture
    modifier.strength = span / 90.0
    modifier.mid_level = 0.5
    B.smooth(ground, 60.0)
    return ground, round(modifier.strength, 4)


# Stacked flats, as (name, distance fraction, height fraction, role).
#
# The painted plate is the *sky* band, and the two land bands stand in front of
# it. Their heights are what decides whether the plate is ever seen: two bands
# at 0.34 and 0.56 of a 9 m backdrop are 3 m and 5 m walls at 8 and 11 m, and
# through the miniature pack's long compressed lens they filled the frame from
# the horizon to the top edge. The plate was behind them, fully occluded, in
# every shot -- measured in camera space, `backdrop_far` spanned 0.39 to 1.12 of
# frame height on a frame that ends at 1.0.
#
# So the land bands are now low scenery rather than walls: a hip-high near band
# and a chest-high far one, which is what painted flats in front of a backdrop
# actually are. `build_shot` measures the result and fails the shot if they
# reach the top of frame again, because the failure is invisible -- a flat
# coloured band reads perfectly well as a deliberate sky.
BACKDROP_BANDS = (
    ("backdrop_near", 0.58, 0.16, "near"),
    ("backdrop_far", 0.80, 0.26, "far"),
    ("backdrop_sky", 1.00, 1.00, "plate"),
)


def _backdrop(pack, plate, near_role="ground", far_role="stone"):
    """Stacked flats: two low land bands in front of one painted sky."""
    distance = float(pack.env.get("backdrop_distance_m") or 14.0)
    height = float(pack.env.get("backdrop_height_m") or 9.0)
    roles = {"near": near_role, "far": far_role, "plate": "backdrop"}
    made = []
    bands = [(name, d, h, roles[slot])
             for name, d, h, slot in BACKDROP_BANDS]
    for name, span_factor, height_factor, role in bands:
        y = distance * span_factor
        tall = height * height_factor
        plane = B.add("plane", name=name, size=1.0,
                      location=(0.0, y, tall / 2.0), rotation=(90.0, 0.0, 0.0))
        plane.scale = (max(6.0, y * 2.6), tall, 1.0)
        if role == "backdrop" and plate:
            B.assign(plane, _plate_material("backdrop_plate", plate, role))
        else:
            B.assign(plane, B.role_material(role))
        made.append(plane)
    return made


def _plate_size(backdrop):
    """The pixel size of the painted plate, read back off the built material.

    Measured rather than assumed: an image whose pixels did not arrive still
    has a datablock, a name and a live link, and reports 0x0.
    """
    for ob in backdrop or []:
        for slot in ob.material_slots:
            material = slot.material
            if material is not None and material.get("style_plate"):
                return material.get("style_plate_px")
    return None


def _source(role):
    """The one mesh a scatter role duplicates. Removed once the copies exist."""
    spec = dict(SOURCES.get(role) or DEFAULT_SOURCE)
    kwargs = {}
    if spec["kind"] == "icosphere":
        kwargs["subdivisions"] = spec.get("subdivisions", 1)
    ob = B.add(spec["kind"], name=f"src_{role}",
               size=spec.get("radius") or spec.get("size") or 0.4,
               location=(0.0, 0.0, 0.0), **kwargs)
    ob.scale = spec.get("scale") or (1.0, 1.0, 1.0)
    B.assign(ob, B.role_material(role))
    if spec.get("smooth"):
        B.smooth(ob, 60.0)
    return ob, spec


def _drop(objects, ground, base_z):
    """Sit each instance on the terrain instead of on the z=0 plane.

    The cast is against the ground object rather than the scene, so an instance
    already placed cannot shadow the one being placed.
    """
    graph = bpy.context.evaluated_depsgraph_get()
    surface = ground.evaluated_get(graph)
    inverse = ground.matrix_world.inverted()
    dropped = 0
    for ob in objects:
        origin = inverse @ ob.matrix_world.translation.copy()
        origin.z = 200.0
        hit, location, _normal, _index = surface.ray_cast(
            origin, (0.0, 0.0, -1.0))
        if hit:
            ob.location.z = (ground.matrix_world @ location).z + base_z
            dropped += 1
    return dropped


IMAGE_SET_STRATEGY = "image_set"
# How much of the set's colour is the image as painted (emission) against the
# image lit by the pack's lights (diffuse). All emission would look exactly like
# the image and cast no shadow of the cast onto it; all diffuse would light the
# painted light a second time. Chosen by eye on the mountain court.
SET_EMISSION = 0.55


def _grid_mesh(name, verts, uv, rows, columns, keep=None):
    """Quads over a rows x columns vertex grid, optionally skipping torn cells."""
    import numpy as np
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    index = np.arange(rows * columns).reshape(rows, columns)
    quads = np.stack([index[:-1, :-1].ravel(), index[1:, :-1].ravel(),
                      index[1:, 1:].ravel(), index[:-1, 1:].ravel()], 1)
    if keep is not None:
        quads = quads[np.asarray(keep).ravel()]
    mesh.loops.add(quads.size)
    mesh.loops.foreach_set("vertex_index", quads.astype(np.int32).ravel())
    mesh.polygons.add(len(quads))
    mesh.polygons.foreach_set("loop_start",
                              np.arange(0, quads.size, 4, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total",
                              np.full(len(quads), 4, dtype=np.int32))
    mesh.update()
    mesh.validate()
    layer = mesh.uv_layers.new(name="image")
    layer.data.foreach_set("uv", uv[quads.ravel()].astype(np.float32).ravel())
    mesh.polygons.foreach_set("use_smooth", np.ones(len(quads), dtype=bool))
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _set_material(name, path, lit=True):
    """The scene image on the set: part as painted, part lit, packed, kept.

    ``style_plate`` makes the pack's material swap leave it alone, exactly as
    for the painted backdrop; ``style_role`` keeps it identifiable.
    """
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    # Clear first, then fetch: see _plate_material for the dangling-pointer bug.
    for node in list(tree.nodes):
        if node.bl_idname != "ShaderNodeOutputMaterial":
            tree.nodes.remove(node)
    output = next(n for n in tree.nodes
                  if n.bl_idname == "ShaderNodeOutputMaterial")
    texture = tree.nodes.new("ShaderNodeTexImage")
    image = bpy.data.images.load(path, check_existing=True)
    try:
        image.pack()
    except Exception as exc:                                    # noqa: BLE001
        print("could not pack %s: %s" % (os.path.basename(path), exc))
    texture.image = image
    texture.extension = "EXTEND"
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
    if lit:
        emission.inputs["Strength"].default_value = SET_EMISSION
        diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        tree.links.new(texture.outputs["Color"], diffuse.inputs["Color"])
        add = tree.nodes.new("ShaderNodeAddShader")
        tree.links.new(emission.outputs["Emission"], add.inputs[0])
        tree.links.new(diffuse.outputs["BSDF"], add.inputs[1])
        tree.links.new(add.outputs["Shader"], output.inputs["Surface"])
    else:
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    material["style_role"] = "ground" if lit else "backdrop"
    material["style_plate"] = os.path.basename(path)
    material["style_plate_px"] = "%dx%d" % tuple(image.size)
    return material


def _image_set(job, unit, pack):
    """The set built from the scene image's depth (see pipeline/scene_depth)."""
    import numpy as np

    params = unit["params"]
    eid = params["subject"]
    out_dir = unit["out_dir"]
    data = np.load(os.path.join(out_dir, "set.npz"))
    info = jsonio.read(os.path.join(out_dir, "set.json"))
    rows, columns = info["grid"]
    image = _plate(job.project_root, unit["inputs"]["refs"])

    B.reset(engine="EEVEE", res=job.resolution, fps=job.fps)
    terrain = _grid_mesh("ground", data["verts"], data["uv"], rows, columns,
                         keep=data["keep"])
    backdrop = _grid_mesh("backdrop_image", data["backdrop"], data["uv"],
                          rows, columns)
    terrain.data.materials.append(_set_material("set_image", image, lit=True))
    backdrop.data.materials.append(
        _set_material("set_backdrop", image, lit=False))
    # The backdrop is scenery at 240 m: it neither casts nor needs shadows.
    backdrop.visible_shadow = False

    objects = [terrain, backdrop]
    collection = _common.collect(f"env_{eid}", objects)
    stats = _common.measure([terrain])
    collection["bstudio_env"] = eid
    collection["bstudio_strategy"] = IMAGE_SET_STRATEGY
    collection["bstudio_ground"] = terrain.name
    blend = _common.save_blend(_base.out_path(unit, "env.blend"))

    # Where the ground is under the stage, measured on the built mesh.
    graph = bpy.context.evaluated_depsgraph_get()
    hit, location, _n, _i = terrain.evaluated_get(graph).ray_cast(
        (0.0, 0.0, 100.0), (0.0, 0.0, -1.0))
    stage_z = round(location.z, 4) if hit else None
    ignored = [spec.get("role") for spec in params.get("scatter") or []]

    record = {
        "schema_version": 2,
        "subject": eid,
        "kind": "env",
        "style_pack": pack.name,
        "strategy": IMAGE_SET_STRATEGY,
        "collection": collection.name,
        "blend": os.path.basename(blend),
        "interior": bool(params.get("interior")),
        "light_intent": params.get("light_intent"),
        "ground": {"object": terrain.name, "span_m": None,
                   "role": "ground", "relief_m": None,
                   "stage_z_m": stage_z},
        "backdrop": [backdrop.name],
        "backdrop_plate": os.path.basename(image),
        "backdrop_plate_px": terrain.data.materials[0].get("style_plate_px"),
        "reference_camera": info["camera"],
        "stage_distance_m": info["stage_distance_m"],
        "scatter": [],
        "scatter_ignored": ignored,
        "measured": stats,
    }
    jsonio.write(_base.out_path(unit, "env.json"), record)
    return {
        "metrics": {
            "kind": "env",
            "subject": eid,
            "strategy": IMAGE_SET_STRATEGY,
            "style_pack": pack.name,
            "ground": terrain.name,
            "stage_z_m": stage_z,
            "horizon_fraction": round(info["camera"]["horizon_fraction"], 4),
            "stage_distance_m": round(info["stage_distance_m"], 3),
            "torn_fraction": round(info["torn_fraction"], 4),
            "backdrop": 1,
            "backdrop_plate": os.path.basename(image),
            "scatter": [],
            "untagged_slots": _common.untagged_slots(objects),
            "objects": len(objects),
            "collection": collection.name,
            **stats,
        },
        "notes": [f"image set from {os.path.basename(image)}: "
                  f"{len(terrain.data.polygons)} faces, stage ground at "
                  f"z={stage_z}, horizon at "
                  f"{info['camera']['horizon_fraction']:.0%}"]
        + ([f"scatter ({', '.join(ignored)}) not built: the image is the set"]
           if ignored else []),
    }


def _env(job, unit):
    params = unit["params"]
    eid = params["subject"]
    pack = job.style()
    if os.path.exists(os.path.join(unit["out_dir"], "set.npz")):
        return _image_set(job, unit, pack)
    strategy = pack.env.get("strategy") or "procedural"
    density = float(pack.env.get("scatter_density") or 1.0)

    # The pack's ground size is a *default*, not a limit. §8's central rule is
    # that production.json is style-agnostic, and a plan whose scatter area has
    # to be chosen to fit the miniature pack's 30 m floor is not: the same file
    # would need a different number for the photoreal pack's 120 m. So the floor
    # grows to hold whatever the plan puts on it, and the metrics record both
    # numbers so an enlargement is visible rather than silent.
    wanted = float(pack.env.get("ground_scale_m") or 40.0)
    widest = max([max(float(v) for v in (spec.get("area_m") or [0.0, 0.0])[:2])
                  for spec in params.get("scatter") or []] or [0.0])
    span = max(wanted, widest + GROUND_MARGIN_M if widest else 0.0)

    B.reset(engine="EEVEE", res=job.resolution, fps=job.fps)
    plate = (_plate(job.project_root, unit["inputs"]["refs"])
             if strategy == BACKDROP_STRATEGY else None)
    ground, relief = _ground(strategy, span,
                             params.get("ground_role") or "ground")
    backdrop = _backdrop(pack, plate) if strategy == BACKDROP_STRATEGY else []

    records, instances, widest = [], [], 0.0
    for spec in params.get("scatter") or []:
        role = spec.get("role") or "foliage"
        asked = max(0, int(round(float(spec.get("count") or 0) * density)))
        area = [float(v) for v in (spec.get("area_m") or [10.0, 10.0])][:2]
        clear = float(spec.get("avoid_radius_m") or 0.0)
        low, high = ([float(v) for v in
                      (spec.get("scale_range") or [0.8, 1.2])] + [1.2])[:2]
        widest = max(widest, max(area))

        source, source_spec = _source(role)
        placed = B.scatter(source, count=asked, area=area,
                           z=source_spec["z"], seed=_seed(eid, role),
                           scale_range=(low, high), rotate=True)
        bpy.data.objects.remove(source, do_unlink=True)

        kept, cleared = [], 0
        for ob in placed:
            if clear and math.hypot(ob.location.x, ob.location.y) < clear:
                bpy.data.objects.remove(ob, do_unlink=True)
                cleared += 1
            else:
                kept.append(ob)
        dropped = (_drop(kept, ground, source_spec["z"])
                   if strategy == TERRAIN_STRATEGY else 0)
        records.append({"role": role, "asked": asked, "placed": len(kept),
                        "cleared_from_marks": cleared, "area_m": area,
                        "clear_radius_m": clear, "scale_range": [low, high],
                        "dropped_to_terrain": dropped,
                        "seed": _seed(eid, role)})
        instances.extend(kept)

    objects = [ground] + backdrop + instances
    collection = _common.collect(f"env_{eid}", objects)
    stats = _common.measure(objects)
    collection["bstudio_env"] = eid
    collection["bstudio_strategy"] = strategy
    collection["bstudio_ground"] = ground.name
    blend = _common.save_blend(_base.out_path(unit, "env.blend"))

    jsonio.write(_base.out_path(unit, "env.json"), {
        "schema_version": 1,
        "subject": eid,
        "kind": "env",
        "style_pack": pack.name,
        "strategy": strategy,
        "collection": collection.name,
        "blend": os.path.basename(blend),
        "interior": bool(params.get("interior")),
        "light_intent": params.get("light_intent"),
        "ground": {"object": ground.name, "span_m": span,
                   "role": params.get("ground_role"), "relief_m": relief},
        "backdrop": [ob.name for ob in backdrop],
        "backdrop_plate": os.path.basename(plate) if plate else None,
        "backdrop_plate_px": _plate_size(backdrop),
        "scatter": records,
        "measured": stats,
    })

    return {
        "metrics": {
            "kind": "env",
            "subject": eid,
            "strategy": strategy,
            "style_pack": pack.name,
            "ground": ground.name,
            "ground_span_m": span,
            "ground_span_pack_m": wanted,
            "ground_enlarged": round(span - wanted, 3) if span > wanted else 0.0,
            "relief_m": relief,
            "scatter_span_m": round(widest, 3),
            "scatter": records,
            "backdrop": len(backdrop),
            "backdrop_wanted": strategy == BACKDROP_STRATEGY,
            "backdrop_plate": os.path.basename(plate) if plate else None,
            "backdrop_plate_px": _plate_size(backdrop),
            "untagged_slots": _common.untagged_slots(objects),
            "objects": len(objects),
            "collection": collection.name,
            **stats,
        },
        "notes": [f"{strategy}: {span:.0f} m ground"
                  + (f" with {relief:.2f} m relief" if relief else "")
                  + (f", {len(backdrop)} backdrop flat(s)" if backdrop else "")]
        + [f"{r['role']}: {r['placed']} placed"
           + (f", {r['cleared_from_marks']} cleared from the marks"
              if r["cleared_from_marks"] else "")
           + (f", {r['dropped_to_terrain']} dropped onto terrain"
              if r["dropped_to_terrain"] else "")
           for r in records],
    }


# A prop is modelled in a unit box - x and y in [-0.5, 0.5], z in [0, 1] - and
# then scaled to the ``size_m`` the plan asked for, which is why the shape table
# carries no metres. ``grip`` is where the hand goes, as a fraction of length.
CLUB = {
    "grip": 0.16,
    "prisms": [((0, 0, 0.0), (0, 0, 0.62), 0.13, 0.11, 10),
               ((0, 0, 0.60), (0, 0, 0.68), 0.18, 0.22, 10)],
    "balls": [((0, 0, 0.83), (0.5, 0.5, 0.17), 16, 9)],
}
BLADE = {
    "grip": 0.10,
    "prisms": [((0, 0, 0.02), (0, 0, 0.20), 0.30, 0.34, 8),
               ((0, 0, 0.20), (0, 0, 0.24), 0.50, 0.50, 4),
               ((0, 0, 0.24), (0, 0, 1.0), 0.42, 0.14, 4)],
    "balls": [((0, 0, 0.02), (0.34, 0.34, 0.03), 10, 5)],
}
BULKY = 0.12


def _prop_mesh(name, shape, size, role):
    """Weld the shape's primitives, normalise the box, scale it to size_m."""
    verts, faces = [], []
    for a, b, r0, r1, sides in shape["prisms"]:
        piece_v, piece_f = B._prism(a, b, r0, r1, sides=sides, up=(0.0, 1.0, 0.0))
        offset = len(verts)
        verts.extend([list(v) for v in piece_v])
        faces.extend(tuple(i + offset for i in face) for face in piece_f)
    for centre, radii, segments, rings in shape["balls"]:
        piece_v, piece_f = B._ball(centre, radii, segments=segments, rings=rings)
        offset = len(verts)
        verts.extend([list(v) for v in piece_v])
        faces.extend(tuple(i + offset for i in face) for face in piece_f)

    # Normalise before scaling so the built bounding box is the asked one to the
    # millimetre, whatever the primitives rounded to, and shift the origin to the
    # grip so h_assemble can parent the prop straight onto a wrist bone.
    low = [min(v[i] for v in verts) for i in range(3)]
    high = [max(v[i] for v in verts) for i in range(3)]
    mid = [(low[i] + high[i]) / 2.0 for i in range(3)]
    extent = [max(1e-6, high[i] - low[i]) for i in range(3)]
    grip = float(shape["grip"])
    for v in verts:
        v[0] = (v[0] - mid[0]) / extent[0] * size[0]
        v[1] = (v[1] - mid[1]) / extent[1] * size[1]
        v[2] = ((v[2] - low[2]) / extent[2] - grip) * size[2]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    mesh.materials.append(B.role_material(role))
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(ob)
    return ob, grip * size[2]


def _prop(job, unit):
    params = unit["params"]
    pid = params["subject"]
    size = [max(0.001, float(v)) for v in params["size_m"]]
    role = params.get("role") or "wood"
    bulky = max(size[0], size[1]) / size[2] > BULKY
    shape = CLUB if bulky else BLADE

    B.reset(engine="EEVEE", res=job.resolution, fps=job.fps)
    ob, grip_z = _prop_mesh(pid, shape, size, role)
    B.smooth(ob, 40.0)
    B.unwrap(ob, angle=66.0)
    collection = _common.collect(f"prop_{pid}", [ob])
    stats = _common.measure([ob])
    collection["bstudio_prop"] = pid
    collection["bstudio_grip_z_m"] = round(grip_z, 5)
    blend = _common.save_blend(_base.out_path(unit, "prop.blend"))

    jsonio.write(_base.out_path(unit, "prop.json"), {
        "schema_version": 1,
        "subject": pid,
        "kind": "prop",
        "collection": collection.name,
        "blend": os.path.basename(blend),
        "object": ob.name,
        "role": role,
        "shape": "club" if bulky else "blade",
        "asked_size_m": [round(v, 4) for v in size],
        "grip_z_m": round(grip_z, 5),
        "butt_z_m": round(-grip_z, 5),
        "tip_z_m": round(size[2] - grip_z, 5),
        "held_by": params.get("held_by"),
        "attach": params.get("attach"),
        "measured": stats,
    })

    return {
        "metrics": {
            "kind": "prop",
            "subject": pid,
            "role": role,
            "shape": "club" if bulky else "blade",
            "asked_size_m": [round(v, 4) for v in size],
            "grip_z_m": round(grip_z, 5),
            "untagged_slots": _common.untagged_slots([ob]),
            "collection": collection.name,
            **stats,
        },
        "notes": [f"{'club' if bulky else 'blade'} of {role}, "
                  f"{size[2]:.2f} m long, grip {grip_z * 100:.0f} cm from the "
                  f"butt, {stats['tris']} tris"],
    }


def run(job, unit):
    if unit["params"].get("kind") == "prop":
        return _prop(job, unit)
    return _env(job, unit)


if __name__ == "__main__":
    sys.exit(_base.stage_main(run, from_blender=True))
