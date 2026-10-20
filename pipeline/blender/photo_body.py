"""A character body built from ``c_asset``'s photo mesh: scaled, rigged, keyed.

This replaces ``prelude.character()`` -- the welded-primitive stand-in -- for
every character that has a reference image, and it keeps that function's whole
interface, because everything downstream was written against it: one mesh and
one armature in a ``char_<id>`` collection, SMPL-X joint names (or the quadruped
set), the eight ``VISEME_CHANNELS`` shape keys, a ``style_role`` on every slot,
and a ``joints``/``dims`` record ``h_assemble`` aims cameras with.

Five decisions carry it.

**Decimated before it is skinned.** ``c_asset`` delivers ~140k vertices so the
surface is smooth; a character on screen needs far fewer, and automatic weights
cost time in proportion. The decimator interpolates colour attributes, which is
why ``c_asset`` packed the photo-trust weight into the albedo's alpha.

**Scale from the eyes, not the crown.** Keyumars wears a tall crown and Soroush
has wings above his head; scaling the mesh's top to ``height_m`` would shrink
both men. ``skeleton.human_joints`` measured stature from the eye line, so the
figure is scaled by that and the crown is simply on top of it.

**Heat weights, with a floor.** Blender's automatic weights are the best
skinning available here and they need bones inside a closed mesh, which is
exactly what pushing landmarks through the surface produced. When they fail,
or leave vertices unweighted, prelude's distance weighting fills in; the count
is reported either way.

**The rest pose is the photo's, and a correction makes it the T-pose.** Every
take is authored as armature-space rotations away from prelude's T-pose rest.
The photo's arms hang at the sides, so rather than re-posing the mesh (and
smearing the armpits twice), each bone records the rotation ``C`` that carries
its rest direction onto the T-pose's; ``prelude.apply_motion`` keys
``C_parent^-1 * M * C_bone`` in place of ``M``. A take with the arms down puts
these arms down.

**The photo is content, not a role.** The material mixes the projected
reference over SF3D's albedo by the per-vertex trust weight, packs the image so
it survives being appended into a shot (bug 4 of the last review: a relative
path linked, loaded at 0x0 and rendered black), and carries ``style_photo`` so
a style pack's material swap leaves it alone, as it already does the plate.
"""

from __future__ import annotations

import json
import math
import os

import bpy
import mathutils
import numpy as np

# Faces kept for the render mesh. A close-up of a face at 1080p is well served
# by this and five characters in one shot stay light for EEVEE.
TARGET_FACES = 60000
MIN_WEIGHT_SHARE = 0.999

VISEME_CHANNELS = ("jaw_open", "lip_wide", "lip_round", "lip_press",
                   "brow_raise", "brow_furrow", "squint", "smile")


def _srgb_to_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def load_mesh(name, npz_path):
    """The c_asset surface as a Blender mesh, attributes and all."""
    data = np.load(npz_path)
    verts, faces = data["verts"], data["faces"]
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start",
                              np.arange(0, faces.size, 3, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total",
                              np.full(len(faces), 3, dtype=np.int32))
    mesh.update()
    mesh.validate()
    uv = mesh.uv_layers.new(name="photo")
    uv.data.foreach_set("uv", data["uv"][faces.ravel()].astype(np.float32)
                        .ravel())
    colour = data["color"].astype(np.float64)
    colour[:, :3] = _srgb_to_linear(colour[:, :3])
    attribute = mesh.color_attributes.new("albedo", "FLOAT_COLOR", "POINT")
    attribute.data.foreach_set("color", colour.astype(np.float32).ravel())
    mesh.color_attributes.active_color = attribute
    # A multi-view asset: one projection per view and the weights to blend
    # them by. The weights ride in a colour attribute because the decimator
    # interpolates those (see the module docstring on trust).
    for name in VIEW_NAMES:
        if f"uv_{name}" in data:
            layer = mesh.uv_layers.new(name=f"view_{name}")
            layer.data.foreach_set("uv", data[f"uv_{name}"][faces.ravel()]
                                   .astype(np.float32).ravel())
    if "view_weights" in data:
        weights = mesh.color_attributes.new("view_weights", "FLOAT_COLOR",
                                            "POINT")
        weights.data.foreach_set("color", data["view_weights"]
                                 .astype(np.float32).ravel())
    mesh.uv_layers.active = mesh.uv_layers["photo"]
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    return ob


SEEN_TRUST = 0.6
FILL_ITERATIONS = 400
FILL_KEEP_ALBEDO = 0.2


def fill_hidden(ob, photo_path):
    """Paint the surface the photo never saw with the photo's own colours.

    SF3D's albedo for the unseen side -- the back of a forearm, the palm, the
    rim of the silhouette -- is a pale grey guess, and the moment an arm turns
    toward the camera the character grows grey limbs. So every vertex the
    photo saw (trust times the photo's matte over ``SEEN_TRUST``) takes the
    photo's colour under it, and those colours are diffused across the mesh's
    edges into the rest: the back of an arm becomes the skin of its front, the
    back of a robe its front's cloth. A little of SF3D's albedo is kept
    (``FILL_KEEP_ALBEDO``) for its shading variation. Trust is unchanged, so
    the photo still wins wherever it was seen. Returns the unseen share.
    """
    mesh = ob.data
    n = len(mesh.vertices)
    attr = mesh.color_attributes.get("albedo")
    if attr is None or n == 0:
        return 0.0
    colour = np.empty(n * 4)
    attr.data.foreach_get("color", colour)
    colour = colour.reshape(-1, 4)
    loops = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loops)
    uv_loops = np.empty(len(mesh.loops) * 2)
    mesh.uv_layers["photo"].data.foreach_get("uv", uv_loops)
    uv = np.zeros((n, 2))
    count = np.zeros(n)
    np.add.at(uv, loops, uv_loops.reshape(-1, 2))
    np.add.at(count, loops, 1.0)
    uv /= np.maximum(count, 1.0)[:, None]

    image = bpy.data.images.load(photo_path, check_existing=True)
    w, h = image.size
    pixels = np.empty(w * h * image.channels, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(h, w, image.channels)
    px = np.clip((uv[:, 0] * (w - 1)).round().astype(int), 0, w - 1)
    py = np.clip((uv[:, 1] * (h - 1)).round().astype(int), 0, h - 1)
    sample = pixels[py, px]
    photo_rgb = _srgb_to_linear(sample[:, :3].astype(np.float64))
    matte = sample[:, 3] if image.channels == 4 else np.ones(n)
    seen = colour[:, 3] * matte > SEEN_TRUST
    if not seen.any():
        return 1.0

    edges = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", edges)
    a, b = edges.reshape(-1, 2).T
    value = np.where(seen[:, None], photo_rgb, 0.0)
    known = seen.astype(np.float64)
    for _ in range(FILL_ITERATIONS):
        total = np.zeros((n, 3))
        weight = np.zeros(n)
        np.add.at(total, a, value[b] * known[b, None])
        np.add.at(total, b, value[a] * known[a, None])
        np.add.at(weight, a, known[b])
        np.add.at(weight, b, known[a])
        grow = (~seen) & (weight > 0)
        value[grow] = total[grow] / weight[grow, None]
        known = np.where(seen, 1.0, np.where(weight > 0, 1.0, known))
    reached = (~seen) & (known > 0)
    colour[reached, :3] = ((1.0 - FILL_KEEP_ALBEDO) * value[reached]
                           + FILL_KEEP_ALBEDO * colour[reached, :3])
    attr.data.foreach_set("color", colour.astype(np.float32).ravel())
    mesh.update()
    return float((~seen).mean())


def decimate(ob, target=TARGET_FACES):
    before = len(ob.data.polygons)
    if before <= target:
        return before, before
    modifier = ob.modifiers.new("decimate", "DECIMATE")
    modifier.ratio = target / float(before)
    modifier.use_collapse_triangulate = True
    _apply(ob, modifier)
    return before, len(ob.data.polygons)


def _apply(ob, modifier):
    with bpy.context.temp_override(object=ob, active_object=ob,
                                   selected_objects=[ob]):
        bpy.ops.object.modifier_apply(modifier=modifier.name)


def fit(ob, joints, scale, face=None):
    """Scale about the origin, stand the figure on z=0, centre it on its hips.

    Returns the transformed joints and face points, and the offset applied.
    """
    coords = np.empty(len(ob.data.vertices) * 3, dtype=np.float64)
    ob.data.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3) * scale
    floor = coords[:, 2].min()
    centre = joints.get("pelvis") or joints.get("root")
    cx, cy = (float(centre[0]) * scale, float(centre[1]) * scale)
    offset = np.array([-cx, -cy, -floor])
    coords += offset
    ob.data.vertices.foreach_set("co", coords.astype(np.float32).ravel())
    ob.data.update()

    def move(point):
        return tuple(float(v) for v in (np.asarray(point) * scale + offset))

    moved = {name: move(point) for name, point in joints.items()}
    moved_face = {}
    for name, value in (face or {}).items():
        moved_face[name] = (move(value) if isinstance(value, (list, tuple))
                            else float(value) * scale)
    return moved, moved_face, [float(v) for v in offset], coords


def skin(mesh_ob, rig, B, frame):
    """Heat weights, then distance weights for anything heat left bare."""
    method = "heat"
    bpy.ops.object.select_all(action="DESELECT")
    mesh_ob.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError as exc:
        method = f"distance (heat failed: {exc})"
    bare = _unweighted_indices(mesh_ob)
    if method == "heat" and len(bare) > 0.02 * len(mesh_ob.data.vertices):
        method = f"distance (heat left {len(bare)} vertices bare)"
    if method != "heat":
        for group in list(mesh_ob.vertex_groups):
            mesh_ob.vertex_groups.remove(group)
        bare = list(range(len(mesh_ob.data.vertices)))
    if bare:
        verts = [tuple(mesh_ob.data.vertices[i].co) for i in bare]
        for bone, weights in B._bone_weights(verts, frame).items():
            if not weights:
                continue
            group = (mesh_ob.vertex_groups.get(bone)
                     or mesh_ob.vertex_groups.new(name=bone))
            for local, weight in weights:
                group.add([bare[local]], weight, "REPLACE")
    if not any(m.type == "ARMATURE" for m in mesh_ob.modifiers):
        modifier = mesh_ob.modifiers.new("Armature", "ARMATURE")
        modifier.object = rig
        mesh_ob.parent = rig
    return method, len(bare)


# Limb chains and the torso bone a vertex falls back to when it lies outside
# the limb. Human rigs only: a quadruped's template legs are the body.
LIMBS = {
    "left_arm": (("left_shoulder", "left_elbow", "left_wrist"), "spine3"),
    "right_arm": (("right_shoulder", "right_elbow", "right_wrist"), "spine3"),
    "left_leg": (("left_hip", "left_knee", "left_ankle", "left_foot"),
                 "pelvis"),
    "right_leg": (("right_hip", "right_knee", "right_ankle", "right_foot"),
                  "pelvis"),
}
CORE_WEIGHT = 0.9            # a vertex this owned by the limb is the limb
CAPSULE_PERCENTILE = 75
CAPSULE_SCALE = 1.5
CAPSULE_SOFT = 1.0           # of the limit, over which weight hands over
# The end bone of a chain (wrist, foot) carries the hand or foot past its tail:
# its capsule runs on this many bone lengths, or the fingers read as "outside
# the arm" and stayed at the hip while the arm rose.
END_REACH = 2.5
BEHIND_WEIGHT = 2.5
TORSO_BONES = ("pelvis", "spine1", "spine2", "spine3", "neck")
# The widest a limb (sleeve included) can be at 1.70 m. A core measured wider
# than this is a limb that already owns something that is not limb.
RADIUS_CAP_M = {"arm": 0.14, "leg": 0.16}


def _segment_distance(points, a, b, reach=1.0):
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(1e-12, float(ab @ ab)), 0.0, reach)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def confine_limbs(mesh_ob, rig, unit=1.0):
    """Keep limb weights on the limb: the capsule rule.

    A figure reconstructed from one image is one closed surface, so wherever
    an arm touched something in the picture -- the torso, a robe, the angel's
    wings behind his sleeves -- the surfaces are fused, and heat weighting
    lets the arm carry the fused part with it: Soroush's arms dragged slabs of
    wing, Siamak's forearms pulled torso into tubes. Here every vertex that a
    limb bone weights is measured against the limb's bone segments. The
    limb's own radius is taken from its core (vertices it owns outright); a
    vertex further out than ``CAPSULE_SCALE`` times that is not limb, and its
    limb weight is handed to the torso bone the limb hangs from, softly over
    ``CAPSULE_SOFT`` of the limit so there is no seam. Returns, per limb, the
    radius and how many vertices were handed over.
    """
    bones = rig.data.bones
    groups = {g.name: g.index for g in mesh_ob.vertex_groups}
    n = len(mesh_ob.data.vertices)
    names = list(groups)
    col = {name: i for i, name in enumerate(names)}
    W = np.zeros((n, len(names)))
    for vertex in mesh_ob.data.vertices:
        for g in vertex.groups:
            name = mesh_ob.vertex_groups[g.group].name
            W[vertex.index, col[name]] = g.weight
    co = np.empty(n * 3)
    mesh_ob.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    # Mesh coordinates into armature space (the mesh is parented to the rig).
    to_rig = np.array(rig.matrix_world.inverted() @ mesh_ob.matrix_world)
    co = co @ to_rig[:3, :3].T + to_rig[:3, 3]

    report = {}
    for limb, (chain, torso) in LIMBS.items():
        present = [b for b in chain if b in col and b in bones]
        if not present or torso not in bones:
            continue
        idx = [col[b] for b in present]
        owned = W[:, idx].sum(axis=1)
        touched = owned > 0.02
        if not touched.any():
            continue
        points = co
        if limb.endswith("_arm"):
            # Arms hang in the body's side plane; wings and cloaks sit behind
            # it. Distance behind the arm counts BEHIND_WEIGHT times, so a
            # wing fused to a sleeve stays with the back.
            ref = float(np.mean([bones[b].head_local[1] for b in present]))
            points = co.copy()
            behind = points[:, 1] < ref
            points[behind, 1] = ref + BEHIND_WEIGHT * (points[behind, 1] - ref)
        dist = np.full(n, np.inf)
        for name in present:
            bone = bones[name]
            dist = np.minimum(dist, _segment_distance(
                points, np.array(bone.head_local), np.array(bone.tail_local),
                END_REACH if name == present[-1] else 1.0))
        core = owned > CORE_WEIGHT
        if core.sum() < 20:
            continue
        radius = min(float(np.percentile(dist[core], CAPSULE_PERCENTILE)),
                     RADIUS_CAP_M[limb.split("_")[1]] * unit)
        limit = radius * CAPSULE_SCALE
        hand = np.clip((dist - limit) / (CAPSULE_SOFT * limit), 0.0, 1.0)
        hand[~touched] = 0.0
        if limb.endswith("_arm"):
            # Handed to the torso only if it is nearer the torso than the arm:
            # a div's claws hang half a metre past any "wrist", far outside
            # the capsule and nowhere near the chest, and handing them over
            # left them at the hip while the arm rose.
            to_torso = np.full(n, np.inf)
            for name in TORSO_BONES:
                if name in bones:
                    bone = bones[name]
                    to_torso = np.minimum(to_torso, _segment_distance(
                        co, np.array(bone.head_local),
                        np.array(bone.tail_local)))
            hand[dist <= to_torso] = 0.0
        moved = hand > 0
        if not moved.any():
            report[limb] = {"radius_m": round(radius, 4), "moved": 0}
            continue
        if torso not in col:
            mesh_ob.vertex_groups.new(name=torso)
            col[torso] = len(names)
            names.append(torso)
            W = np.hstack([W, np.zeros((n, 1))])
        give = W[:, idx] * hand[:, None]
        W[:, idx] -= give
        W[:, col[torso]] += give.sum(axis=1)
        report[limb] = {"radius_m": round(radius, 4),
                        "moved": int((hand > 0.5).sum())}
    for name, i in col.items():
        group = mesh_ob.vertex_groups.get(name) or \
            mesh_ob.vertex_groups.new(name=name)
        nonzero = np.nonzero(W[:, i] > 1e-4)[0]
        zero = np.nonzero(W[:, i] <= 1e-4)[0]
        if len(zero):
            group.remove(zero.tolist())
        for v in nonzero:
            group.add([int(v)], float(W[v, i]), "REPLACE")
    return report


def _unweighted_indices(ob):
    out = []
    for vertex in ob.data.vertices:
        if not any(g.weight > 1e-4 for g in vertex.groups):
            out.append(vertex.index)
    return out


# The bones whose rest needs carrying onto the canonical one: the arms, because
# every take is authored against a T-pose rest and the photo's arms hang at
# its sides. Nothing else. The photo's *standing* legs are the character's
# standing legs -- correcting them to the canonical straight leg lengthened the
# div's bent beast legs and put its foot 0.42 m under the floor in its idle --
# and correcting its feet drove the claws into the ground. A quadruped's rig
# is built on the template, so it has nothing to correct at all.
CORRECTED_BONES = (
    "left_collar", "right_collar", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
)


def rest_correction(frame, canonical_frame):
    """Per bone, the rotation carrying this rig's rest onto the canonical one.

    Both frames map bone -> (head, tail). Only the limb bones in
    :data:`CORRECTED_BONES` are considered, and of those only the ones that
    differ by more than a degree are recorded.
    """
    out = {}
    for name, (head, tail) in frame.items():
        if name not in canonical_frame or name not in CORRECTED_BONES:
            continue
        ours = (tail - head)
        c_head, c_tail = canonical_frame[name]
        theirs = (c_tail - c_head)
        if ours.length < 1e-6 or theirs.length < 1e-6:
            continue
        rotation = ours.normalized().rotation_difference(theirs.normalized())
        if abs(rotation.angle) > math.radians(1.0):
            out[name] = [round(v, 6) for v in rotation]
    return out


def _gauss(points, centre, radius):
    d2 = ((points - np.asarray(centre)) ** 2).sum(1)
    return np.exp(-d2 / (radius * radius))


def visemes(ob, face, joints, unit):
    """The eight mouth and brow channels as shape keys on the photo mesh.

    There is no FLAME head here, so each channel is a smooth displacement field
    around a landmark: the jaw rotates the chin about a hinge behind the mouth,
    the lip channels move the corners, the brows move above the eyes. Sizes are
    in units of the mouth's own width, so a div's muzzle and a boy's lips are
    both moved by amounts that suit them. A quadruped with no face landmarks
    gets the same fields around its muzzle joint.
    """
    coords = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", coords)
    base = coords.reshape(-1, 3)
    if ob.data.shape_keys is None:
        ob.shape_key_add(name="Basis", from_mix=False)

    mouth = face.get("mouth")
    width = face.get("mouth_width")
    if mouth is None:
        anchor = joints.get("muzzle") or joints.get("jaw") or joints["head"]
        mouth = tuple(anchor)
        width = 0.10 * unit
    width = max(float(width or 0.05 * unit), 0.03 * unit)
    mouth = np.asarray(mouth, dtype=np.float64)
    left_c = mouth + np.array([width / 2, 0, 0])
    right_c = mouth - np.array([width / 2, 0, 0])
    eyes = [np.asarray(face[n]) for n in ("left_eye", "right_eye") if n in face]
    eye_mid = (sum(eyes) / len(eyes)) if eyes else mouth + np.array(
        [0, 0, 1.3 * width])
    eye_gap = (float(np.linalg.norm(eyes[0] - eyes[1])) if len(eyes) == 2
               else 1.2 * width)
    front = base[:, 1] > (mouth[1] - 1.2 * width)

    fields = {}
    # jaw_open: rotate the chin region about a hinge behind and above the mouth.
    hinge = mouth + np.array([0.0, -1.1 * width, 0.55 * width])
    chin = mouth + np.array([0.0, 0.0, -0.9 * width])
    w = _gauss(base, chin, 1.25 * width) * (base[:, 2] < mouth[2] + 0.08 * width)
    w *= front
    angle = math.radians(16.0)
    rel = base - hinge
    rotated = np.stack([rel[:, 0],
                        rel[:, 1] * math.cos(angle) + rel[:, 2] * math.sin(angle),
                        -rel[:, 1] * math.sin(angle) + rel[:, 2] * math.cos(angle)],
                       1) + hinge
    fields["jaw_open"] = (rotated - base) * w[:, None]

    corners = (_gauss(base, left_c, 0.45 * width),
               _gauss(base, right_c, 0.45 * width))
    lips = _gauss(base, mouth, 0.6 * width) * front
    out_l = np.array([1.0, 0.0, 0.0])
    fields["lip_wide"] = (corners[0][:, None] * out_l * 0.22 * width
                          - corners[1][:, None] * out_l * 0.22 * width)
    fields["lip_round"] = (-corners[0][:, None] * out_l * 0.2 * width
                           + corners[1][:, None] * out_l * 0.2 * width
                           + lips[:, None] * np.array([0, 0.12 * width, 0]))
    above = (base[:, 2] > mouth[2]).astype(float) * 2 - 1
    fields["lip_press"] = (lips * -above)[:, None] * np.array(
        [0, 0, 0.06 * width])
    fields["smile"] = ((corners[0] + corners[1])[:, None]
                       * np.array([0, -0.03 * width, 0.14 * width])
                       + corners[0][:, None] * out_l * 0.08 * width
                       - corners[1][:, None] * out_l * 0.08 * width)
    brow = eye_mid + np.array([0.0, 0.0, 0.5 * eye_gap])
    brows = _gauss(base, brow, 0.75 * eye_gap) * front
    fields["brow_raise"] = brows[:, None] * np.array([0, 0, 0.10 * eye_gap])
    inward = np.where(base[:, 0] > eye_mid[0], -1.0, 1.0)
    fields["brow_furrow"] = brows[:, None] * np.stack(
        [inward * 0.05 * eye_gap, np.zeros(len(base)),
         np.full(len(base), -0.06 * eye_gap)], 1)
    lids = _gauss(base, eye_mid - np.array([0, 0, 0.12 * eye_gap]),
                  0.7 * eye_gap) * front
    fields["squint"] = lids[:, None] * np.array([0, 0, 0.035 * eye_gap])

    moved = {}
    for name in VISEME_CHANNELS:
        key = ob.data.shape_keys.key_blocks.get(name) or ob.shape_key_add(
            name=name, from_mix=False)
        key.slider_min, key.slider_max = 0.0, 1.0
        # A new shape key comes in at full strength in this Blender: left
        # alone, every photo character wore all eight at once -- jaw open,
        # smiling, squinting, brows up and down -- and lip sync only ever
        # overrode the channels it keyed.
        key.value = 0.0
        displaced = base + fields[name]
        key.data.foreach_set("co", displaced.astype(np.float32).ravel())
        moved[name] = round(float(np.linalg.norm(fields[name], axis=1).max()), 5)

    # The mouth line, for the material: a closed solid has no mouth cavity to
    # show when the jaw opens, so the lip line darkens with the jaw instead.
    mask = np.exp(-(((base[:, 0] - mouth[0]) / (0.55 * width)) ** 2
                    + ((base[:, 2] - mouth[2]) / (0.10 * width)) ** 2)) * front
    attribute = ob.data.attributes.new("mouth_mask", "FLOAT", "POINT")
    attribute.data.foreach_set("value", mask.astype(np.float32))
    return moved


VIEW_NAMES = ("left", "back", "right")
VIEW_BLEND_FULL = 0.08       # summed weight at which the views fully cover


def _view_blend(tree, photo_tex, views):
    """Every view's texture, blended by the per-vertex weights.

    ``views`` maps a view name to its image path. Returns the node socket of
    the blended colour and of the coverage (0..1) to mix it over the albedo.
    """
    nodes, links = tree.nodes, tree.links
    weights = nodes.new("ShaderNodeVertexColor")
    weights.layer_name = "view_weights"
    split = nodes.new("ShaderNodeSeparateColor")
    links.new(weights.outputs["Color"], split.inputs["Color"])
    sources = [(photo_tex, weights.outputs["Alpha"])]
    channel = {"left": "Red", "back": "Green", "right": "Blue"}
    for name, path in views.items():
        image = bpy.data.images.load(path)
        image.pack()
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.extension = "EXTEND"
        uvmap = nodes.new("ShaderNodeUVMap")
        uvmap.uv_map = f"view_{name}"
        links.new(uvmap.outputs["UV"], tex.inputs["Vector"])
        sources.append((tex, split.outputs[channel[name]]))
    total_colour, total_weight = None, None
    for tex, weight in sources:
        w = nodes.new("ShaderNodeMath")
        w.operation = "MULTIPLY"
        links.new(weight, w.inputs[0])
        links.new(tex.outputs["Alpha"], w.inputs[1])     # the view's matte
        scaled = nodes.new("ShaderNodeVectorMath")
        scaled.operation = "SCALE"
        links.new(tex.outputs["Color"], scaled.inputs[0])
        links.new(w.outputs[0], scaled.inputs["Scale"])
        if total_colour is None:
            total_colour, total_weight = scaled.outputs[0], w.outputs[0]
            continue
        add_c = nodes.new("ShaderNodeVectorMath")
        add_c.operation = "ADD"
        links.new(total_colour, add_c.inputs[0])
        links.new(scaled.outputs[0], add_c.inputs[1])
        add_w = nodes.new("ShaderNodeMath")
        add_w.operation = "ADD"
        links.new(total_weight, add_w.inputs[0])
        links.new(w.outputs[0], add_w.inputs[1])
        total_colour, total_weight = add_c.outputs[0], add_w.outputs[0]
    safe = nodes.new("ShaderNodeMath")
    safe.operation = "MAXIMUM"
    safe.inputs[1].default_value = 1e-4
    links.new(total_weight, safe.inputs[0])
    inverse = nodes.new("ShaderNodeMath")
    inverse.operation = "DIVIDE"
    inverse.inputs[0].default_value = 1.0
    links.new(safe.outputs[0], inverse.inputs[1])
    blended = nodes.new("ShaderNodeVectorMath")
    blended.operation = "SCALE"
    links.new(total_colour, blended.inputs[0])
    links.new(inverse.outputs[0], blended.inputs["Scale"])
    cover = nodes.new("ShaderNodeMath")
    cover.operation = "DIVIDE"
    cover.use_clamp = True
    cover.inputs[1].default_value = VIEW_BLEND_FULL
    links.new(total_weight, cover.inputs[0])
    return blended.outputs[0], cover.outputs[0]


def material(ob, photo_path, subject, views=None):
    """Photo over albedo by trust, a darkening lip line, packed and tagged.

    With ``views`` (a multi-view asset), every view's texture is blended by
    its weight instead, and the albedo only fills what no view sees.
    """
    mat = bpy.data.materials.new(f"{subject}_photo")
    mat.use_nodes = True
    tree = mat.node_tree
    nodes, links = tree.nodes, tree.links
    bsdf = nodes.get("Principled BSDF")
    image = bpy.data.images.load(photo_path)
    image.pack()
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.extension = "EXTEND"
    uvmap = nodes.new("ShaderNodeUVMap")
    uvmap.uv_map = "photo"
    links.new(uvmap.outputs["UV"], tex.inputs["Vector"])
    albedo = nodes.new("ShaderNodeVertexColor")
    albedo.layer_name = "albedo"
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    # Trust weight is the albedo attribute's alpha; the photo's own matte
    # multiplies it, so a background pixel never lands on the figure.
    if views:
        blended, cover = _view_blend(tree, tex, views)
        links.new(cover, mix.inputs["Factor"])
        links.new(albedo.outputs["Color"], mix.inputs["A"])
        links.new(blended, mix.inputs["B"])
    else:
        trust = nodes.new("ShaderNodeMath")
        trust.operation = "MULTIPLY"
        links.new(albedo.outputs["Alpha"], trust.inputs[0])
        links.new(tex.outputs["Alpha"], trust.inputs[1])
        links.new(trust.outputs[0], mix.inputs["Factor"])
        links.new(albedo.outputs["Color"], mix.inputs["A"])
        links.new(tex.outputs["Color"], mix.inputs["B"])

    mouth = nodes.new("ShaderNodeAttribute")
    mouth.attribute_name = "mouth_mask"
    jaw = nodes.new("ShaderNodeValue")
    jaw.name = jaw.label = "jaw_open"
    shade = nodes.new("ShaderNodeMath")
    shade.operation = "MULTIPLY"
    links.new(mouth.outputs["Fac"], shade.inputs[0])
    links.new(jaw.outputs[0], shade.inputs[1])
    dark = nodes.new("ShaderNodeMix")
    dark.data_type = "RGBA"
    dark.inputs["B"].default_value = (0.03, 0.01, 0.01, 1.0)
    links.new(shade.outputs[0], dark.inputs["Factor"])
    links.new(mix.outputs["Result"], dark.inputs["A"])
    links.new(dark.outputs["Result"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.72
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    mat["style_role"] = "skin"
    mat["style_photo"] = os.path.basename(photo_path)

    keys = ob.data.shape_keys
    if keys is not None and "jaw_open" in keys.key_blocks:
        curve = jaw.outputs[0].driver_add("default_value")
        driver = curve.driver
        driver.type = "AVERAGE"
        var = driver.variables.new()
        var.name = "jaw"
        var.type = "SINGLE_PROP"
        var.targets[0].id_type = "KEY"
        var.targets[0].id = keys
        var.targets[0].data_path = 'key_blocks["jaw_open"].value'
    return mat


def build(B, cid, asset_dir, height_m, kind, smplx_path=None):
    """Everything above, in order. Returns the same shape as prelude.character.

    ``smplx_path``: c_body's SMPL-X fit for this character, when it made one.
    A character it fits takes its skin weights from it (``smplx_skin``);
    any other is heat-weighted as before."""
    asset = json.load(open(os.path.join(asset_dir, "asset.json"),
                           encoding="utf-8"))
    ob = load_mesh(cid, os.path.join(asset_dir, asset.get("mesh", "mesh.npz")))
    before, after = decimate(ob)
    # A multi-view asset already coloured what the front cannot see from the
    # side and back images; filling it from the front would paint over them.
    unseen = (0.0 if asset.get("shape_backend") == "hunyuan3d_mv"
              else fill_hidden(ob, os.path.join(
                  asset_dir, asset.get("photo", "photo.png"))))

    quadruped = asset["rig"].get("method") == "quadruped_template"
    stature = float(asset["rig"].get("stature_units") or 1.0)
    scale = float(height_m) / stature
    joints_in = {n: tuple(p) for n, p in asset["joints"].items()}
    face_in = {n: (tuple(v) if isinstance(v, list) else v)
               for n, v in (asset.get("face") or {}).items()}
    joints, face, offset, coords = fit(ob, joints_in, scale, face_in)

    if quadruped:
        order, parents, tails = B.QUAD_JOINTS, B.QUAD_PARENTS, B.QUAD_TAILS
        canonical, _dims = B._quad_joints(height_m, "neutral")
    else:
        order, parents, tails = B.SMPLX_JOINTS, B.SMPLX_PARENTS, B.HUMAN_TAILS
        canonical, _dims = B._rest_joints(height_m, "neutral")
    unit = float(height_m) / (B.QUAD_CROWN if quadruped else B.REST_HEIGHT_M)
    frame = B._bone_frame(joints, order, parents, tails, stub=0.06 * unit)
    canonical_frame = B._bone_frame(canonical, order, parents, tails,
                                    stub=0.06 * unit)
    rig = B._build_armature(cid + "_rig", frame, order, parents)

    moved = visemes(ob, face, joints, unit)
    body_fit, smplx_report = None, None
    if smplx_path and os.path.exists(smplx_path) and not quadruped:
        from pipeline.blender import smplx_skin
        body_fit = smplx_skin.load(smplx_path)
    if body_fit is not None:
        smplx_report = smplx_skin.apply(ob, rig, body_fit, scale, offset,
                                        float(height_m))
        method, bare = "smplx", 0
    else:
        method, bare = skin(ob, rig, B, frame)
    confined = {} if quadruped else confine_limbs(ob, rig, unit)
    correction = rest_correction(frame, canonical_frame)
    rig["rest_correction"] = json.dumps(correction)
    rig["character"], ob["character"] = cid, cid
    rig["rest_height_m"] = float(height_m)
    if not quadruped:
        # Fitted by the eye line, so this is the height the plan asked for; see
        # build_rigged._height for why the bounding box is not.
        rig["stature_m"] = float(height_m)
    rig["rig_kind"] = "quadruped" if quadruped else "smplx"
    ob.data["photo_body"] = True
    for poly in ob.data.polygons:
        poly.use_smooth = True
    views = {name: os.path.join(asset_dir, file)
             for name, file in (asset.get("view_images") or {}).items()}
    material(ob, os.path.join(asset_dir, asset.get("photo", "photo.png")), cid,
             views=views or None)

    return {
        "name": cid, "mesh": ob, "rig": rig, "joints": joints,
        "dims": {"factor": unit, "height": float(height_m), "girth": 1.0},
        "bones": list(order), "roles": {"photo": "skin"},
        "kind": "quadruped" if quadruped else "human",
        "faces_before": before, "faces": after,
        "unseen_share": round(unseen, 3),
        "confined_limbs": confined,
        "skinning": method, "distance_weighted": bare,
        "smplx_skin": smplx_report,
        "rest_correction_bones": sorted(correction),
        "viseme_move_m": moved, "scale": scale, "offset": offset,
        "face": face,
    }
