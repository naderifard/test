"""What every Blender-side stage step shares.

The graph starts these as ``blender -b --factory-startup --python <script> --
--job ... --step N --result ...``, so a script in this package is the Blender
half of exactly one stage and reads the same job file its host half read. Three
constraints shape the module:

* Blender ignores ``PYTHONPATH`` unless started with ``--python-use-system-env``,
  so every script puts the repo root on ``sys.path`` itself, from ``__file__``.
* ``prelude.py`` is not imported as part of the package. It is exec'd into a
  synthetic module named ``bstudio``, exactly the way ``session.install_prelude``
  does it - which is why the character layer inside it imports no siblings.
* One process handles every pending unit of its stage (§10 rule 1), so a step
  that builds an asset per unit must open or reset the scene per unit rather
  than accumulating three characters in one file.

The measurement helpers live here rather than in ``gates`` because they need
``bpy``: §11 says measure the artifact, and for a .blend the only honest place
to measure is inside Blender, off the evaluated depsgraph.
"""

from __future__ import annotations

import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import bpy                                                         # noqa: E402
from bpy_extras.object_utils import world_to_camera_view            # noqa: E402

_PRELUDE = None


def install():
    """The prelude, as the module named ``bstudio``. Cached per process."""
    global _PRELUDE
    if _PRELUDE is not None:
        return _PRELUDE
    path = os.path.join(REPO_ROOT, "blender_ai_studio", "prelude.py")
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    module = types.ModuleType("bstudio")
    module.__file__ = path
    exec(compile(source, path, "exec"), module.__dict__)
    sys.modules["bstudio"] = module
    _PRELUDE = module
    return module


# -- files -------------------------------------------------------------------
def new_file():
    """A genuinely empty .blend, for a process that builds several units.

    ``prelude.reset()`` deletes the objects but not the collections, armatures,
    actions and shape keys behind them. That is enough for a stage that builds
    everything from primitives, and not enough for one that *appends*: the
    second unit's ``append_collection`` finds the first unit's collection still
    linked to the scene and refuses. Reading factory settings empty is the only
    reliable way to start over inside one process, which §10 rule 1 requires
    this to be.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def open_blend(path):
    """Continue from the previous stage's artifact.

    The character chain is additive - c_texture unwraps what c_body built,
    c_garment dresses it - and opening the file forward is both cheaper and
    more faithful than rebuilding from parameters, which would let the stages
    silently disagree about the geometry.
    """
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(path))
    return bpy.context.scene


def save_blend(path):
    """Write the current file to ``path`` without adopting it as the session's.

    ``copy=True`` matters: the next unit in this same process opens its own
    input file, and a session pointing at the file just written would make a
    later relative path resolve somewhere surprising.
    """
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=target, copy=True,
                               compress=True)
    return target


def append_collection(path, name):
    """Append one collection (and its dependencies) from another .blend.

    ``libraries.load(link=False)`` rather than ``wm.append``: it names the
    datablock explicitly instead of depending on operator context, and it
    rewires the internal pointers of everything it pulls, so a garment keeps
    the armature modifier that points at the body's rig.
    """
    with bpy.data.libraries.load(os.path.abspath(path), link=False) as (src, dst):
        if name not in src.collections:
            raise KeyError(f"{path} has no collection {name!r} "
                           f"(has {list(src.collections)})")
        dst.collections = [name]
    collection = bpy.data.collections.get(name)
    if collection is None:
        raise KeyError(f"{name} did not arrive from {path}")
    bpy.context.scene.collection.children.link(collection)
    return collection


def collect(name, objects):
    """Put ``objects`` in a collection of their own, unlinked from any other.

    Every asset .blend publishes exactly one collection, so the stage that
    consumes it names one thing rather than guessing which objects matter.
    """
    collection = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if name not in {c.name for c in bpy.context.scene.collection.children}:
        bpy.context.scene.collection.children.link(collection)
    for ob in objects:
        for parent in list(ob.users_collection):
            if parent is not collection:
                parent.objects.unlink(ob)
        if ob.name not in collection.objects:
            collection.objects.link(ob)
    return collection


def meshes(collection=None):
    source = collection.all_objects if collection else bpy.context.scene.objects
    return [ob for ob in source if ob.type == "MESH"]


# -- measurement -------------------------------------------------------------
def world_points(ob):
    """Evaluated, world-space vertex positions - modifiers and shape keys in."""
    graph = bpy.context.evaluated_depsgraph_get()
    copy = ob.evaluated_get(graph)
    mesh = copy.to_mesh()
    matrix = copy.matrix_world
    points = [matrix @ vertex.co.copy() for vertex in mesh.vertices]
    copy.to_mesh_clear()
    return points


def _mark(cells, a, b, c, grid):
    """Add every cell of the 0-1 grid that one UV triangle covers."""
    step = 1.0 / grid
    lo_x = max(0, min(grid - 1, int(min(a[0], b[0], c[0]) * grid)))
    hi_x = max(0, min(grid - 1, int(max(a[0], b[0], c[0]) * grid)))
    lo_y = max(0, min(grid - 1, int(min(a[1], b[1], c[1]) * grid)))
    hi_y = max(0, min(grid - 1, int(max(a[1], b[1], c[1]) * grid)))
    doubled = ((b[0] - a[0]) * (c[1] - a[1])
               - (c[0] - a[0]) * (b[1] - a[1]))
    hit = False
    if abs(doubled) > 1e-12:
        for cx in range(lo_x, hi_x + 1):
            px = (cx + 0.5) * step
            for cy in range(lo_y, hi_y + 1):
                py = (cy + 0.5) * step
                w0 = (b[0] - a[0]) * (py - a[1]) - (px - a[0]) * (b[1] - a[1])
                w1 = (c[0] - b[0]) * (py - b[1]) - (px - b[0]) * (c[1] - b[1])
                w2 = (a[0] - c[0]) * (py - c[1]) - (px - c[0]) * (a[1] - c[1])
                if (w0 >= 0 and w1 >= 0 and w2 >= 0) or \
                        (w0 <= 0 and w1 <= 0 and w2 <= 0):
                    cells.add((cx, cy))
                    hit = True
    if not hit:
        # A triangle smaller than a cell contains no cell centre. It still
        # occupies texture space, so credit the cell its centroid lands in;
        # without this the metric falls as a mesh gets denser.
        mx = (a[0] + b[0] + c[0]) / 3.0
        my = (a[1] + b[1] + c[1]) / 3.0
        if 0.0 <= mx <= 1.0 and 0.0 <= my <= 1.0:
            cells.add((min(grid - 1, int(mx * grid)),
                       min(grid - 1, int(my * grid))))


def uv_coverage(mesh, grid=64):
    """Fraction of the 0-1 square the UVs touch, on a coarse grid.

    The faces are rasterised, not their corners. Marking only the cells that
    contain a UV vertex sounds equivalent and is not: it caps the answer at
    ``loops / grid**2``, so a 276-loop prop could never score above 6.7% however
    well it was laid out, and the metric silently reported polygon density
    instead of coverage.

    An exact area sum would count overlapping islands twice and cost more; what
    ``gates.check_asset`` asks is whether the layout is present and spread at
    all, and a 64x64 occupancy grid answers that.
    """
    layer = mesh.uv_layers.active
    if layer is None or not len(layer.data):
        return None
    cells = set()
    for polygon in mesh.polygons:
        corners = [tuple(layer.data[index].uv)
                   for index in polygon.loop_indices]
        for index in range(1, len(corners) - 1):
            _mark(cells, corners[0], corners[index], corners[index + 1], grid)
    return round(len(cells) / float(grid * grid), 5)


def measure(objects, uv=True):
    """Triangles, bounding box, degenerate faces and UV coverage."""
    graph = bpy.context.evaluated_depsgraph_get()
    tris = verts = degenerate = 0
    low = [float("inf")] * 3
    high = [float("-inf")] * 3
    coverage = []
    for ob in objects:
        copy = ob.evaluated_get(graph)
        mesh = copy.to_mesh()
        matrix = copy.matrix_world
        for vertex in mesh.vertices:
            point = matrix @ vertex.co
            for axis in range(3):
                low[axis] = min(low[axis], point[axis])
                high[axis] = max(high[axis], point[axis])
        verts += len(mesh.vertices)
        for polygon in mesh.polygons:
            tris += max(0, len(polygon.vertices) - 2)
            if polygon.area < 1e-9:
                degenerate += 1
        if uv:
            found = uv_coverage(mesh)
            if found is not None:
                coverage.append(found)
        copy.to_mesh_clear()
    if not verts:
        return {"tris": 0, "verts": 0, "size_m": [0.0, 0.0, 0.0],
                "degenerate_faces": 0, "uv_coverage": None}
    return {
        "tris": tris,
        "verts": verts,
        "size_m": [round(high[i] - low[i], 4) for i in range(3)],
        "min_m": [round(v, 4) for v in low],
        "max_m": [round(v, 4) for v in high],
        "degenerate_faces": degenerate,
        "uv_coverage": (round(sum(coverage) / len(coverage), 5)
                        if coverage else None),
    }


def unweighted(objects, rig):
    """Vertices no bone can move - the silent cause of a torn deform.

    A vertex group whose name is not a bone counts for nothing, which is the
    case that matters: proximity binding writes groups from a joint table, and a
    joint the armature does not carry produces a group that looks weighted in
    the UI and moves nothing at render time.
    """
    bones = {bone.name for bone in rig.data.bones}
    total = 0
    for ob in objects:
        groups = {group.index: group.name for group in ob.vertex_groups}
        total += sum(1 for vertex in ob.data.vertices
                     if not any(groups.get(entry.group) in bones
                                and entry.weight > 0.0
                                for entry in vertex.groups))
    return total


def untagged_slots(objects):
    """Material slots a style pack could not swap: empty, or with no role."""
    out = []
    for ob in objects:
        for index, slot in enumerate(ob.material_slots):
            material = slot.material
            if material is None:
                out.append(f"{ob.name}.slot{index}:empty")
            elif not material.get("style_role"):
                out.append(material.name)
    return out


def materials_missing(objects):
    """Objects with no material, or with an empty slot - §11-H's first check."""
    out = []
    for ob in objects:
        slots = [slot.material for slot in ob.material_slots]
        if not slots or not any(slots):
            out.append(ob.name)
    return out


def framing(camera, objects):
    """How much of the frame the cast fills, and whether any of it is in shot.

    The projected bounding box over-reports a diagonal pose, but §11-H asks
    only whether the subject is present and not a speck, and the box answers
    that without a hull.
    """
    scene = bpy.context.scene
    xs, ys, inside = [], [], False
    for ob in objects:
        for point in world_points(ob):
            projected = world_to_camera_view(scene, camera, point)
            xs.append(projected.x)
            ys.append(projected.y)
            if projected.z > 0 and 0.0 <= projected.x <= 1.0 \
                    and 0.0 <= projected.y <= 1.0:
                inside = True
    if not xs:
        return {"subjects_in_frustum": False, "subject_frame_fraction": 0.0}
    width = max(0.0, min(1.0, max(xs)) - max(0.0, min(xs)))
    height = max(0.0, min(1.0, max(ys)) - max(0.0, min(ys)))
    return {
        "subjects_in_frustum": inside,
        "subject_frame_fraction": round(width * height, 5),
        "subject_box": [round(min(xs), 4), round(min(ys), 4),
                        round(max(xs), 4), round(max(ys), 4)],
    }


def feet_gap(objects):
    """Lowest evaluated vertex per object, which is how far off the floor it is."""
    out = {}
    for ob in objects:
        points = world_points(ob)
        if points:
            out[ob.name] = round(min(p.z for p in points), 4)
    return out
