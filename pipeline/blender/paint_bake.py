"""The character's own texture space, for ``character_paint``.

``blender -b -P paint_bake.py -- bake <rigged.blend> <out_dir> [size]``
    gives the character's mesh a non-overlapping UV layout (``paint``; the
    photo UVs are a projection, front and back on top of each other) and
    bakes, in that layout, the world position and normal of every texel and
    which texels the mesh covers, as ``position.npy``, ``normal.npy`` and
    ``valid.npy`` (row 0 is the top of the texture, as an image reads). The file with the new layout is saved as
    ``paint_source.blend``.

``blender -b -P paint_bake.py -- apply <paint_source.blend> <texture.png>
<out.blend>``
    makes the painted texture the character's colour and saves.
"""

import os
import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
mode = args[0]


def character_meshes():
    return [o for o in bpy.context.scene.objects
            if o.type == "MESH" and o.visible_get()
            and any(m.type == "ARMATURE" for m in o.modifiers)]


SEPARATE_M = 0.05
# Island margin in UV units: 2 px at 2048. A wider one left a furry
# character -- thousands of small islands -- covering a tenth of its texture.
MARGIN = float(os.environ.get("PAINT_UV_MARGIN", "0.001"))
UNTANGLE_ROUNDS = 4


def overlapping_faces(meshes, size):
    """Faces, per mesh, whose texels in the 'paint' layout are also covered
    by a part of the body more than ``SEPARATE_M`` away.

    Smart project flattens each island by one planar projection, and where
    the surface curls back inside an island -- an armpit, where the inner arm
    and the torso's side both face forward -- it lands on itself: 3% of a
    character's triangles shared texels with another part of the body 10-30
    cm away, and the arm showed the torso's colour."""
    import numpy as np
    owner = np.full((size, size), -1, np.int64)
    centres, where, bad = [], [], set()
    base = 0
    for mi, ob in enumerate(meshes):
        me = ob.data
        me.calc_loop_triangles()
        n = len(me.loop_triangles)
        loops = np.empty(n * 3, np.int32)
        me.loop_triangles.foreach_get("loops", loops)
        verts = np.empty(n * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", verts)
        polys = np.empty(n, np.int32)
        me.loop_triangles.foreach_get("polygon_index", polys)
        uv = np.empty(len(me.loops) * 2, np.float32)
        me.uv_layers["paint"].data.foreach_get("uv", uv)
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        mw = np.array(ob.matrix_world)
        co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        px = uv.reshape(-1, 2)[loops].reshape(n, 3, 2).astype(np.float64) * size
        px[..., 1] = size - px[..., 1]
        centres.append(co[verts].reshape(n, 3, 3).mean(1))
        where.extend((mi, int(p)) for p in polys)
        allc = np.concatenate(centres)
        for i in range(n):
            a, b, c = px[i]
            lo = np.floor(px[i].min(0)).astype(int).clip(0, size - 1)
            hi = np.ceil(px[i].max(0)).astype(int).clip(0, size - 1)
            xs, ys = np.meshgrid(np.arange(lo[0], hi[0] + 1) + 0.5,
                                 np.arange(lo[1], hi[1] + 1) + 0.5)
            e0 = (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])
            e1 = (c[0] - b[0]) * (ys - b[1]) - (c[1] - b[1]) * (xs - b[0])
            e2 = (a[0] - c[0]) * (ys - c[1]) - (a[1] - c[1]) * (xs - c[0])
            inside = (((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) |
                      ((e0 <= 0) & (e1 <= 0) & (e2 <= 0)))
            if not inside.any():
                continue
            iy = ys[inside].astype(int)
            ix = xs[inside].astype(int)
            prev = owner[iy, ix]
            prev = np.unique(prev[prev >= 0])
            if len(prev):
                far = prev[np.linalg.norm(allc[prev] - allc[base + i], axis=1)
                           > SEPARATE_M]
                if len(far):
                    bad.add(base + i)
                    bad.update(far.tolist())
            owner[iy, ix] = base + i
        base += n
    faces = {}
    for t in bad:
        mi, p = where[t]
        faces.setdefault(mi, set()).add(p)
    print("PAINT_UV texels covered", int((owner >= 0).sum()))
    return faces, len(bad)


def untangle(meshes, size):
    """Re-unwrap the overlapping faces into islands of their own and pack
    everything again, until nothing overlaps (or a few rounds pass)."""
    import numpy as np
    last = None
    for rnd in range(UNTANGLE_ROUNDS + 1):
        faces, tris = overlapping_faces(meshes, size)
        print("PAINT_UV overlap round", rnd, "triangles", tris)
        if not tris or rnd == UNTANGLE_ROUNDS:
            break
        bpy.ops.object.mode_set(mode="EDIT")
        # Selected through bmesh in edit mode: a face selection written in
        # object mode is overridden by the vertices still selected.
        import bmesh
        for mi, ob in enumerate(meshes):
            bm = bmesh.from_edit_mesh(ob.data)
            bm.faces.ensure_lookup_table()
            for f in bm.faces:
                f.select_set(False)
            for p in faces.get(mi, ()):
                bm.faces[p].select_set(True)
            bm.select_flush_mode()
            bmesh.update_edit_mesh(ob.data)
        if last is None or tris < last * 0.8:
            # Smaller islands: a narrower cone of normals per projection.
            bpy.ops.uv.smart_project(angle_limit=0.6, island_margin=MARGIN,
                                     area_weight=0.0, scale_to_bounds=False)
        else:
            # What a projection cannot separate gets a face per island,
            # each sized by its area.
            bpy.ops.uv.lightmap_pack(PREF_CONTEXT="SEL_FACES",
                                     PREF_PACK_IN_ONE=True,
                                     PREF_NEW_UVLAYER=False,
                                     PREF_MARGIN_DIV=0.2)
        last = tris
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.select_all(action="SELECT")
        # Islands sized by their area on the body, not by how the projection
        # that made them happened to scale them, before packing.
        bpy.ops.uv.average_islands_scale()
        bpy.ops.uv.pack_islands(margin=MARGIN, rotate=True)
        bpy.ops.object.mode_set(mode="OBJECT")


HAND_GROUPS = ("left_wrist", "right_wrist")
HAND_ATTRIBUTE = "paint_hand"
# The head: the face is painted from one source, never an average of views
# (``character_paint.bake_texture``).
HEAD_GROUPS = ("head", "jaw")
HEAD_ATTRIBUTE = "paint_head"


def group_weights(meshes, names, attribute):
    """How much each vertex belongs to the vertex groups ``names`` (summed),
    as a float attribute for a bake. False when no mesh has any of them (a
    hand on an animal)."""
    import numpy as np
    found = False
    for ob in meshes:
        groups = [ob.vertex_groups[g].index for g in names
                  if g in ob.vertex_groups]
        weights = np.zeros(len(ob.data.vertices), np.float32)
        for v in ob.data.vertices:
            for g in v.groups:
                if g.group in groups:
                    weights[v.index] += g.weight
        found = found or bool(groups)
        attr = ob.data.attributes.get(attribute) or \
            ob.data.attributes.new(attribute, "FLOAT", "POINT")
        attr.data.foreach_set("value", np.clip(weights, 0.0, 1.0))
    return found


def hand_weights(meshes):
    """How much each vertex belongs to a hand, for the ``hand`` bake.

    A hand is thirty pixels in Wan's frames, too small for the depth to say
    which way it faces, and Wan draws an open palm toward whichever side it
    is seen from -- the back of the hand came out as a second palm.
    ``character_paint`` lets only the front views colour the hands."""
    return group_weights(meshes, HAND_GROUPS, HAND_ATTRIBUTE)


def bake(source, out_dir, size):
    bpy.ops.wm.open_mainfile(filepath=source)
    os.makedirs(out_dir, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.device = "CPU"
    meshes = character_meshes()
    for ob in meshes:
        if "paint" not in ob.data.uv_layers:
            ob.data.uv_layers.new(name="paint")
        ob.data.uv_layers.active = ob.data.uv_layers["paint"]
    bpy.ops.object.select_all(action="DESELECT")
    for ob in meshes:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=MARGIN,
                             area_weight=0.0, scale_to_bounds=True)
    bpy.ops.object.mode_set(mode="OBJECT")
    untangle(meshes, size)

    originals = {ob.name: list(ob.data.materials) for ob in meshes}
    mat = bpy.data.materials.new("paint_bake")
    mat.use_nodes = True
    tree = mat.node_tree
    for node in list(tree.nodes):
        if node.bl_idname != "ShaderNodeOutputMaterial":
            tree.nodes.remove(node)
    out = next(n for n in tree.nodes
               if n.bl_idname == "ShaderNodeOutputMaterial")
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    emit = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    texture = tree.nodes.new("ShaderNodeTexImage")
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "paint"
    tree.links.new(uv.outputs["UV"], texture.inputs["Vector"])
    tree.nodes.active = texture
    for ob in meshes:
        ob.data.materials.clear()
        ob.data.materials.append(mat)

    def run(name, socket, float_buffer=True, fmt="OPEN_EXR"):
        image = bpy.data.images.new(name, size, size, alpha=False,
                                    float_buffer=float_buffer)
        image.generated_color = (0, 0, 0, 1)
        texture.image = image
        for link in list(tree.links):
            if link.to_socket == emit.inputs["Color"]:
                tree.links.remove(link)
        if socket is None:
            emit.inputs["Color"].default_value = (1, 1, 1, 1)
        else:
            tree.links.new(socket, emit.inputs["Color"])
        bpy.ops.object.bake(type="EMIT", margin=0, use_clear=True)
        # Raw pixels, not an image file: saving a float image applies the
        # display transform, which bent a 1.98 m figure into 1.34 m.
        import numpy as np
        pixels = np.empty(size * size * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        np.save(os.path.join(out_dir, name + ".npy"),
                pixels.reshape(size, size, 4)[::-1, :, :3])

    run("position", geometry.outputs["Position"])
    run("normal", geometry.outputs["Normal"])
    run("valid", None)
    if hand_weights(meshes):
        attribute = tree.nodes.new("ShaderNodeAttribute")
        attribute.attribute_name = HAND_ATTRIBUTE
        run("hand", attribute.outputs["Fac"])
    if group_weights(meshes, HEAD_GROUPS, HEAD_ATTRIBUTE):
        attribute = tree.nodes.new("ShaderNodeAttribute")
        attribute.attribute_name = HEAD_ATTRIBUTE
        run("head", attribute.outputs["Fac"])

    for ob in meshes:
        ob.data.materials.clear()
        for m in originals[ob.name]:
            ob.data.materials.append(m)
    bpy.data.materials.remove(mat)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir,
                                                      "paint_source.blend"))
    print("PAINT_BAKE done", [o.name for o in meshes])


def apply(source, texture_path, out_blend):
    bpy.ops.wm.open_mainfile(filepath=source)
    image = bpy.data.images.load(texture_path)
    image.pack()
    for ob in character_meshes():
        ob.data.uv_layers.active = ob.data.uv_layers["paint"]
        for mat in ob.data.materials:
            if mat is None or not mat.use_nodes:
                continue
            tree = mat.node_tree
            bsdf = next((n for n in tree.nodes
                         if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
            if bsdf is None:
                continue
            texture = tree.nodes.new("ShaderNodeTexImage")
            texture.image = image
            uv = tree.nodes.new("ShaderNodeUVMap")
            uv.uv_map = "paint"
            tree.links.new(uv.outputs["UV"], texture.inputs["Vector"])
            socket = bsdf.inputs["Base Color"]
            # Keep the lip-darkening mix if there is one: replace its input A
            # (the photo/albedo blend) with the painted texture.
            upstream = socket.links[0].from_node if socket.links else None
            if upstream is not None and upstream.bl_idname == "ShaderNodeMix":
                tree.links.new(texture.outputs["Color"], upstream.inputs["A"])
            else:
                tree.links.new(texture.outputs["Color"], socket)
            mat["paint_texture"] = os.path.basename(texture_path)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print("PAINT_APPLY done")


if mode == "bake":
    bake(args[1], args[2], int(args[3]) if len(args) > 3 else 2048)
elif mode == "apply":
    apply(args[1], args[2], args[3])
