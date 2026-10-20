"""Real hands for a photo-built character: the SMPL-X hand in place of a mitten.

The shape model resolves a body well and a hand badly: at the size a hand is
in the photographs, the fingers come out fused into a mitten with faint
grooves, and no texture can paint fingers onto that. SMPL-X carries a hand
with separate, well-formed fingers, so each of the character's hands is
replaced by it:

* The character's hand is **measured** from its own mesh: the wrist bone, the
  hand's principal axes (the long one points to the fingertips, the thinnest
  is the palm's normal), its length and the wrist's thickness. Which way the
  thumb points follows from which way the flat hand faces -- palms forward
  (the anatomical position) puts the thumbs out to the sides, palms to the
  thighs puts them forward.
* The SMPL-X hand of the same side is **placed** by the rotation that takes
  its own axes (wrist to middle knuckle; pinky knuckle to index knuckle) onto
  those, so a right hand stays a right hand, and scaled to the hand's length
  and the wrist's thickness.
* The mitten is **cut off** at the wrist and the template joined in its
  place. Its 3 cm cuff reaches back inside the forearm, tapered so it stays
  under the stump's surface rather than fighting it. It follows the wrist
  bone; its UVs and colours come from the nearest original vertex, so the
  hand looks right even before ``c_paint`` paints it.

Axes: SMPL-X stands on +Y facing +Z with its left on +X; joints 20 and 21 are
the wrists and 25-39 / 40-54 the left / right fingers (index, middle, pinky,
ring, thumb; three each). The rig here faces +Y and its ``left_*`` bones are
on +X, which is the figure's anatomical right: the SMPL-X *right* hand.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

CUFF_M = 0.075           # template forearm kept behind the wrist
# The stump is cut this far before the wrist bone: the mitten's thumb is a
# lump that starts behind the wrist, and a cut at -1 cm left it as a stub.
CUT_M = -0.045
LOOSE_VERTICES = 400     # smaller pieces left near a wrist are removed
BAND_M = 0.02            # the stump's radius is measured over this much
FIT = 0.97               # template radius at the cut, of the stump's
CUFF_TAPER = 0.10        # and this much thinner again at the cuff's end
HAND_WEIGHT = 0.5
REACH_M = 0.14           # how far from the arm's axis a hand vertex can be
# A hand is about a ninth of the stature, wrist to fingertip. A mitten's
# length is only roughly the hand's, and Soroush's gave him a giant's hands.
HAND_OF_STATURE = (0.095, 0.115)
SMPLX_WRIST = {"left": 20, "right": 21}
SMPLX_FINGERS = {"left": range(25, 40), "right": range(40, 55)}
SMPLX_INDEX1 = {"left": 25, "right": 40}
SMPLX_MIDDLE1 = {"left": 28, "right": 43}
SMPLX_PINKY1 = {"left": 31, "right": 46}
SMPLX_ELBOW = {"left": 18, "right": 19}


def _smplx_path():
    from pipeline import envs
    return os.path.join(envs.MODEL_ROOT, "body_models", "smplx",
                        "SMPLX_NEUTRAL.npz")


def available():
    try:
        return os.path.exists(_smplx_path())
    except Exception:                                       # noqa: BLE001
        return False


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def template_hand(side):
    """The SMPL-X ``side`` hand with a cuff of forearm: vertices, faces, and
    its frame (wrist, long axis, thumb-ward axis) and measurements."""
    data = np.load(_smplx_path(), allow_pickle=True)
    v = data["v_template"].astype(np.float64)
    f = data["f"].astype(np.int64)
    w = data["weights"]
    joints = data["J_regressor"].astype(np.float64) @ v
    wrist = joints[SMPLX_WRIST[side]]
    long_axis = _unit(joints[SMPLX_MIDDLE1[side]] - wrist)
    across = joints[SMPLX_INDEX1[side]] - joints[SMPLX_PINKY1[side]]
    across = _unit(across - long_axis * (across @ long_axis))
    arm = w[:, [SMPLX_ELBOW[side], SMPLX_WRIST[side]]
            + list(SMPLX_FINGERS[side])].sum(1) > 0.5
    along = (v - wrist) @ long_axis
    keep = arm & (along > -CUFF_M)
    index = -np.ones(len(v), np.int64)
    index[keep] = np.arange(keep.sum())
    faces = f[keep[f].all(1)]
    verts = v[keep]
    return {"verts": verts, "faces": index[faces], "wrist": wrist,
            "long": long_axis, "across": across,
            "length": float(((verts - wrist) @ long_axis).max()),
            "radius": _radius(verts, wrist, long_axis)}


def _radius(verts, wrist, long_axis):
    """Mean distance from the arm's axis over the forearm just before the
    cut -- the stump the new hand has to meet."""
    along = (verts - wrist) @ long_axis
    radial = np.linalg.norm(verts - wrist - np.outer(along, long_axis),
                            axis=1)
    band = (along < CUT_M) & (along > CUT_M - BAND_M) & (radial < REACH_M)
    return float(radial[band].mean()) if band.any() else 0.03


def character_hand(points, weights, wrist, body_centre, elbow=None):
    """The character's hand frame from its hand vertices (``weights`` above
    ``HAND_WEIGHT``): long axis to the fingertips, the thumb-ward axis, and
    length and wrist radius. ``body_centre`` tells outward from inward.

    With ``elbow``, the long axis is the forearm's, elbow to wrist: the new
    hand carries straight on from the arm, relaxed, and the cut is square to
    the arm. Along the mitten's own axis -- bent at the wrist -- the cut
    sliced the forearm on a slant and left a flared cuff."""
    hand = points[weights > HAND_WEIGHT]
    if len(hand) < 20:
        return None
    centred = hand - hand.mean(0)
    _s, _v, axes = np.linalg.svd(centred, full_matrices=False)
    long_axis = axes[0] if (hand.mean(0) - wrist) @ axes[0] > 0 else -axes[0]
    if elbow is not None:
        long_axis = _unit(wrist - elbow)
    width, thin = axes[1], axes[2]
    thin = _unit(thin - long_axis * (thin @ long_axis))
    width = _unit(np.cross(thin, long_axis))
    # The flat hand's thin direction is the palm's normal. Facing forward or
    # back (thin along Y): the thumb points out to the side, away from the
    # body. Facing the thighs (thin along X): the thumb points forward.
    if abs(thin[1]) >= abs(thin[0]):
        outward = np.array([np.sign(wrist[0] - body_centre[0]) or 1.0, 0, 0])
    else:
        outward = np.array([0.0, 1.0, 0.0])
    across = width if width @ outward >= 0 else -width
    across = _unit(across - long_axis * (across @ long_axis))
    long_axis = _unit(long_axis)
    # Length from the hand's own vertices: a few of the tunic's carry a
    # little wrist weight, and measured with them one hand came out 37 cm.
    length = float(((hand - wrist) @ long_axis).max())
    arm = points[weights > 0.05]
    return {"long": long_axis, "across": across, "wrist": wrist,
            "length": length, "radius": _radius(arm, wrist, long_axis)}


def placement(template, target):
    """(rotation, scale along, scale across the hand) taking template to
    target."""
    src = np.stack([template["long"], template["across"],
                    np.cross(template["long"], template["across"])], 1)
    dst = np.stack([target["long"], target["across"],
                    np.cross(target["long"], target["across"])], 1)
    rotation = dst @ src.T
    s_long = target["length"] / max(template["length"], 1e-6)
    s_hand = target["radius"] / max(template["radius"], 1e-6)
    # A mitten's wrist can be thick or thin for reasons that have nothing to
    # do with the hand: keep the hand's proportions within a quarter.
    s_hand = float(np.clip(s_hand, s_long * 0.8, s_long * 1.25))
    return rotation, s_long, s_hand


def radius_at(verts, wrist, long_axis, at, half=0.006):
    """Mean distance from the axis of the vertices within ``half`` of
    ``at`` along it."""
    along = (verts - wrist) @ long_axis
    radial = np.linalg.norm(verts - wrist - np.outer(along, long_axis),
                            axis=1)
    band = np.abs(along - at) < half
    return float(radial[band].mean()) if band.any() else None


def transform(template, target, rotation, s_long, s_hand, s_cut=None,
              shift=None):
    """Template vertices moved onto the character's wrist. Returns world
    positions and each one's distance along the hand from the wrist
    (negative on the cuff).

    Across the hand the template is scaled by ``s_hand``; from the wrist
    back to the cut the scale runs to ``s_cut``, which makes the template's
    forearm the stump's size there (FIT of it, so the stump's edge lies just
    outside), and past the cut the cuff narrows further inside the arm.
    ``shift`` moves the template sideways onto the stump's centre."""
    s_cut = s_hand if s_cut is None else s_cut
    rel = template["verts"] - template["wrist"]
    along = rel @ template["long"] * s_long
    radial = rel - np.outer(rel @ template["long"], template["long"])
    to_cut = np.clip(along / CUT_M, 0.0, 1.0)
    past_cut = np.clip((CUT_M - along) / (CUFF_M + CUT_M), 0.0, 1.0)
    scale = s_hand + (s_cut * FIT - s_hand) * to_cut
    scale = scale * (1.0 - CUFF_TAPER * past_cut)
    scaled = (np.outer(along, template["long"]) + radial * scale[:, None])
    out = target["wrist"] + scaled @ rotation.T
    if shift is not None:
        out = out + shift
    return out, along


# -- Blender -----------------------------------------------------------------

def _wrist_bones(rig):
    """{wrist bone: (wrist head, elbow head)} in world space."""
    out = {}
    for name in ("left_wrist", "right_wrist"):
        elbow = name.replace("wrist", "elbow")
        if name in rig.data.bones:
            out[name] = (rig.matrix_world @ rig.data.bones[name].head_local,
                         rig.matrix_world @ rig.data.bones[elbow].head_local
                         if elbow in rig.data.bones else None)
    return out


def _group_weights(ob, name):
    weights = np.zeros(len(ob.data.vertices))
    group = ob.vertex_groups.get(name)
    if group is None:
        return weights
    for v in ob.data.vertices:
        for g in v.groups:
            if g.group == group.index:
                weights[v.index] = g.weight
    return weights


def _islands(bm):
    """Connected pieces of ``bm`` as lists of vertices."""
    seen, parts = set(), []
    for start in bm.verts:
        if start in seen:
            continue
        part, stack = [], [start]
        seen.add(start)
        while stack:
            v = stack.pop()
            part.append(v)
            for e in v.link_edges:
                o = e.other_vert(v)
                if o not in seen:
                    seen.add(o)
                    stack.append(o)
        parts.append(part)
    return parts


def transplant(ob, rig, log=print):
    """Replace both hands of mesh ``ob`` (skinned to ``rig``). Returns a
    report; changes ``ob`` in place."""
    import bmesh
    import bpy
    from mathutils import kdtree

    mw = np.array(ob.matrix_world)
    co = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", co)
    points = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    centre = points.mean(0)
    # The rig's stature, not the mesh's height: Soroush's wings stand above
    # his head.
    stature = rig.get("stature_m")
    stature = (float(stature) * rig.matrix_world.to_scale().z if stature
               else float(points[:, 2].max() - points[:, 2].min()))
    report = {"hands": []}
    new_objects, cuts = [], []
    for bone, (wrist, elbow) in _wrist_bones(rig).items():
        weights = _group_weights(ob, bone)
        wrist = np.array(wrist)
        target = character_hand(points, weights, wrist, centre,
                                None if elbow is None else np.array(elbow))
        if target is None:
            log(f"hands: {bone}: too few hand vertices, left as it is")
            continue
        target["length"] = float(np.clip(
            target["length"], HAND_OF_STATURE[0] * stature,
            HAND_OF_STATURE[1] * stature))
        # left_* is on +X, the anatomical right: SMPL-X's right hand.
        side = "right" if wrist[0] > centre[0] else "left"
        template = template_hand(side)
        # The arm near this wrist, and only it, is what the cut may touch:
        # the tunic's hem hangs beside the hand.
        arm_along = (points - wrist) @ target["long"]
        radial = np.linalg.norm(points - wrist - np.outer(arm_along,
                                                          target["long"]),
                                axis=1)
        inside = (radial < REACH_M) & (arm_along > CUT_M - 3 * BAND_M)
        cuts.append((wrist + target["long"] * CUT_M, target["long"],
                     (weights > HAND_WEIGHT) & inside, inside))
        new_objects.append({"bone": bone, "side": side, "target": target,
                            "template": template})
    if not new_objects:
        return report

    # Nearest original vertex, for every attribute the new hand must carry.
    tree = kdtree.KDTree(len(points))
    for i, p in enumerate(points):
        tree.insert(p, i)
    tree.balance()
    carried = _per_vertex_attributes(ob)

    bpy.context.view_layer.objects.active = ob
    inv = np.linalg.inv(mw)
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    before = len(bm.verts)
    bm.verts.ensure_lookup_table()
    # What the cut may touch: the mesh reachable from the hand without
    # leaving a cylinder round the arm -- the whole of the forearm's ring at
    # the cut (by weight alone part of it was left, as ragged flaps), and not
    # the tunic's hem beside the hand, which is not joined to the arm.
    # Both regions are picked before either cut: a cut renumbers vertices.
    picked = []
    for point, normal, seeds, inside in cuts:
        chosen = {bm.verts[i] for i in np.nonzero(seeds)[0]}
        stack = list(chosen)
        while stack:
            v = stack.pop()
            for e in v.link_edges:
                o = e.other_vert(v)
                if o not in chosen and inside[o.index]:
                    chosen.add(o)
                    stack.append(o)
        picked.append((chosen, point, normal))
    rings = []
    for chosen, point, normal in picked:
        # A clean cut across the arm by a plane, the mitten side removed.
        chosen = {v for v in chosen if v.is_valid}
        faces = {f for v in chosen for f in v.link_faces}
        edges = {e for f in faces for e in f.edges}
        verts = {v for f in faces for v in f.verts}
        result = bmesh.ops.bisect_plane(
            bm, geom=list(verts) + list(edges) + list(faces),
            plane_co=tuple(inv[:3, :3] @ point + inv[:3, 3]),
            plane_no=tuple(_unit(inv[:3, :3] @ normal)), clear_outer=True)
        ring = [tuple(v.co) for v in result["geom_cut"]
                if isinstance(v, bmesh.types.BMVert) and v.is_valid]
        rings.append(np.array(ring) @ mw[:3, :3].T + mw[:3, 3]
                     if ring else None)
    # The cut was confined to the arm near the wrist; any piece of the mitten
    # outside it is now loose. Small pieces near a wrist go.
    wrists_local = [inv[:3, :3] @ p + inv[:3, 3] for p, _n, _s, _i in cuts]
    loose = []
    for part in _islands(bm):
        if len(part) >= LOOSE_VERTICES:
            continue
        centre_local = np.mean([tuple(v.co) for v in part], axis=0)
        if min(np.linalg.norm(centre_local - w) for w in wrists_local) \
                < 3 * REACH_M:
            loose.extend(part)
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    report["loose_removed"] = len(loose)
    report["removed_vertices"] = before - len(bm.verts)
    bm.to_mesh(ob.data)
    bm.free()

    for entry, ring in zip(new_objects, rings):
        bone, template, target = (entry["bone"], entry["template"],
                                  entry["target"])
        rotation, s_long, s_hand = placement(template, target)
        s_cut, shift = s_hand, None
        if ring is not None and len(ring) >= 6:
            # The stump's own edge: its centre and its size.
            centre_ring = ring.mean(0)
            ring_radius = float(np.linalg.norm(ring - centre_ring,
                                               axis=1).mean())
            on_axis = target["wrist"] + target["long"] * CUT_M
            offset = centre_ring - on_axis
            shift = offset - target["long"] * (offset @ target["long"])
            there = radius_at(template["verts"], template["wrist"],
                              template["long"], CUT_M / s_long)
            if there:
                s_cut = ring_radius / there
        verts, along = transform(template, target, rotation, s_long, s_hand,
                                 s_cut, shift)
        faces = template["faces"]
        report["hands"].append({
            "bone": bone, "side": entry["side"],
            "length_m": round(target["length"], 3),
            "scale": [round(s_long, 3), round(s_hand, 3), round(s_cut, 3)],
            "shift_m": None if shift is None
            else round(float(np.linalg.norm(shift)), 3)})
        local = verts @ inv[:3, :3].T + inv[:3, 3]
        me = bpy.data.meshes.new(f"hand_{bone}")
        me.from_pydata([tuple(p) for p in local], [], [tuple(f) for f in faces])
        me.update()
        hand = bpy.data.objects.new(f"hand_{bone}", me)
        hand.matrix_world = ob.matrix_world
        bpy.context.scene.collection.objects.link(hand)
        nearest = [tree.find(tuple(p))[1] for p in verts]
        _copy_attributes(carried, hand, nearest)
        elbow = bone.replace("wrist", "elbow")
        wrist_group = hand.vertex_groups.new(name=bone)
        elbow_group = hand.vertex_groups.new(name=elbow)
        for i, a in enumerate(along):
            # The elbow's at the cut, where the stump is; the wrist's from
            # the wrist on.
            share = float(np.clip(1.0 - a / CUT_M, 0.0, 1.0))
            wrist_group.add([i], share, "REPLACE")
            if share < 1.0:
                elbow_group.add([i], 1.0 - share, "REPLACE")
        for poly in me.polygons:
            poly.use_smooth = True
        bpy.ops.object.select_all(action="DESELECT")
        hand.select_set(True)
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.join()
    return report


def _per_vertex_attributes(ob):
    """Every UV map and colour attribute of ``ob`` as one value per vertex
    (a corner value taken from the vertex's first corner), read before the
    mesh is cut."""
    me = ob.data
    first = np.zeros(len(me.vertices), np.int64)
    corner_vertex = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", corner_vertex)
    first[corner_vertex[::-1]] = np.arange(len(me.loops))[::-1]
    material = np.zeros(len(me.vertices), np.int64)
    for poly in me.polygons:
        material[list(poly.vertices)] = poly.material_index
    out = {"uv": {}, "color": {}, "material": material,
           "materials": list(me.materials)}
    for layer in me.uv_layers:
        uv = np.empty(len(me.loops) * 2)
        layer.data.foreach_get("uv", uv)
        out["uv"][layer.name] = uv.reshape(-1, 2)[first]
    for attr in me.color_attributes:
        data = np.empty(len(attr.data) * 4)
        attr.data.foreach_get("color", data)
        data = data.reshape(-1, 4)
        out["color"][attr.name] = (attr.data_type, attr.domain,
                                   data if attr.domain == "POINT"
                                   else data[first])
    return out


def _copy_attributes(carried, target, nearest):
    """UV maps, colour attributes and material of the nearest original
    vertex. The materials first, in the same slots: joined without them the
    hand came in on an empty slot, which the render would draw as nothing."""
    me = target.data
    nearest = np.asarray(nearest)
    for material in carried["materials"]:
        me.materials.append(material)
    per_poly = [int(carried["material"][nearest[p.vertices[0]]])
                for p in me.polygons]
    me.polygons.foreach_set("material_index", per_poly)
    corner_vertex = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", corner_vertex)
    for name, per_vertex in carried["uv"].items():
        layer = me.uv_layers.new(name=name)
        layer.data.foreach_set("uv", per_vertex[nearest[corner_vertex]].ravel())
    for name, (data_type, domain, per_vertex) in carried["color"].items():
        attr = me.color_attributes.new(name, data_type, domain)
        values = per_vertex[nearest] if domain == "POINT" \
            else per_vertex[nearest[corner_vertex]]
        attr.data.foreach_set("color", values.ravel())


if __name__ == "__main__":
    import bpy
    args = sys.argv[sys.argv.index("--") + 1:]
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH"
                and any(m.type == "ARMATURE" for m in o.modifiers))
    print("HANDS", transplant(mesh, rig))
    bpy.ops.wm.save_as_mainfile(filepath=args[0], copy=True)
