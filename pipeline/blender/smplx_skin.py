"""Skin weights from a fitted SMPL-X body (``pipeline/smplx_fit.py``).

Every vertex of the character that lies on the body takes the weights of the
SMPL-X surface it lies on: weights learned from body scans, in place of
heat weighting's guess from bone positions.

Not every vertex lies on the body. A photo-built character is one fused
surface -- beard, hair, wings and loose cloth are part of it -- and the
nearest SMPL-X surface to a beard is the chest it hangs in front of, which
would tear it in two when the head turned. So the transfer has two parts:

* **On the body** (within ``NEAR`` of the SMPL-X surface, and facing the same
  way as it): the SMPL-X weights of the nearest few SMPL-X vertices, blended
  by distance. Facing matters: the front of a wing lies close behind an arm
  and faces the other way; it is not the arm.
* **Off the body**: the weights of the on-body vertex it is *attached to* --
  the nearest one along the mesh's own edges, not through the air. A beard is
  attached at the chin, a wing at the shoulder blades, a skirt at the waist.

A light smoothing over the mesh then takes out the seams between the two.
"""

from __future__ import annotations

import heapq
import json

import numpy as np

NEAR = 0.022            # of the stature: this close to SMPL-X is "on the body"
FACING = 0.2            # normals' dot above which the two surfaces agree
NEIGHBOURS = 6
SMOOTH_ROUNDS = 12
SMOOTH = 0.5
MAX_BONES = 4
MAX_FIT_ERROR = 0.035   # of the height: beyond this it is not a human's body


def load(path):
    """The fit, or None when its joints missed the rig's by too much (the
    black div: a man's body does not stand inside a beast's)."""
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    if meta["error"] > MAX_FIT_ERROR:
        return None
    return {"verts": data["verts"].astype(np.float64),
            "normals": data["normals"].astype(np.float64),
            "weights": data["weights"].astype(np.float64),
            "bones": [str(b) for b in data["bones"]], "error": meta["error"]}


def on_body(points, normals, body, stature):
    """For each point: SMPL-X weights by its nearest SMPL-X vertices, and
    whether it is on the body at all."""
    from mathutils import kdtree
    tree = kdtree.KDTree(len(body["verts"]))
    for i, v in enumerate(body["verts"]):
        tree.insert(v, i)
    tree.balance()
    weights = np.zeros((len(points), body["weights"].shape[1]))
    near = np.zeros(len(points), bool)
    limit = NEAR * stature
    for i, (p, n) in enumerate(zip(points, normals)):
        found = tree.find_n(p, NEIGHBOURS)
        index = np.array([f[1] for f in found])
        dist = np.array([f[2] for f in found])
        agree = body["normals"][index] @ n
        w = np.where(agree > FACING, 1.0, 0.0) / (dist ** 2 + 1e-8)
        if dist[0] < limit and w.sum() > 0:
            near[i] = True
        if w.sum() <= 0:
            w = 1.0 / (dist ** 2 + 1e-8)
        weights[i] = (body["weights"][index] * w[:, None]).sum(0) / w.sum()
    return weights, near


def attach(weights, near, points, edges):
    """Each off-body vertex takes the weights of the on-body vertex nearest
    to it along the mesh (multi-source Dijkstra over edge lengths)."""
    n = len(points)
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        length = float(np.linalg.norm(points[a] - points[b]))
        adjacency[a].append((b, length))
        adjacency[b].append((a, length))
    source = np.where(near, np.arange(n), -1)
    dist = np.where(near, 0.0, np.inf)
    heap = [(0.0, int(i)) for i in np.nonzero(near)[0]]
    heapq.heapify(heap)
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist[i]:
            continue
        for j, length in adjacency[i]:
            nd = d + length
            if nd < dist[j]:
                dist[j] = nd
                source[j] = source[i]
                heapq.heappush(heap, (nd, j))
    out = weights.copy()
    reached = (source >= 0) & ~near
    out[reached] = weights[source[reached]]
    return out, int((source < 0).sum())


def smooth(weights, edges, rounds=SMOOTH_ROUNDS, amount=SMOOTH):
    """A few rounds of each vertex moving toward its neighbours' mean."""
    a, b = edges[:, 0], edges[:, 1]
    degree = np.zeros(len(weights))
    np.add.at(degree, a, 1.0)
    np.add.at(degree, b, 1.0)
    degree = np.maximum(degree, 1.0)[:, None]
    for _ in range(rounds):
        total = np.zeros_like(weights)
        np.add.at(total, a, weights[b])
        np.add.at(total, b, weights[a])
        weights = (1.0 - amount) * weights + amount * total / degree
    return weights


def strongest(weights, keep=MAX_BONES):
    """Only each vertex's ``keep`` largest weights, summing to one."""
    if weights.shape[1] > keep:
        cut = np.partition(weights, -keep, axis=1)[:, -keep][:, None]
        weights = np.where(weights >= cut, weights, 0.0)
    return weights / np.maximum(weights.sum(1, keepdims=True), 1e-12)


# Hip, knee, ankle, foot. The foot is part of the leg here: a boot's sole lies
# past the ankle, "far from the leg" by hip-knee-ankle alone, and handed to
# the pelvis it stayed behind as the foot stepped -- a sheet stretched between
# Keyumars's feet. A boot hugs the foot; a gown's hem stands clear of it.
LEGS = {"left": ("left_hip", "left_knee", "left_ankle", "left_foot"),
        "right": ("right_hip", "right_knee", "right_ankle", "right_foot")}
LEG_SOFT = 1.0          # of a leg's radius, over which the hem hands over
SKIRT_PELVIS = 0.65     # a hem's weight left on the pelvis; the rest, thigh


def _segment_distance(points, a, b):
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(1e-12, float(ab @ ab)), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def hem(weights, near, points, rig, mesh_ob, body):
    """Cloth that hangs off the body below the hips -- a skirt, a gown --
    moves with the pelvis and, a little, the thigh on its side.

    Off the body, a vertex takes the weights of the body it is joined to,
    and a gown's hem is joined to the body at the feet: each leg then
    dragged its own side of Soroush's gown, and it creased and dented with
    every step. Below the hips, a vertex off the body and further from a
    leg than the leg is thick hands over (softly, ``LEG_SOFT``) to the
    pelvis and that side's thigh; trouser legs and boot cuffs, which hug
    the leg, keep the leg's weights. Returns the weights and how many
    vertices handed over."""
    bones = rig.data.bones
    column = {name: i for i, name in enumerate(body["bones"])}
    if "pelvis" not in column or not all(b in bones for leg in LEGS.values()
                                         for b in leg):
        return weights, 0
    to_rig = np.array(rig.matrix_world.inverted() @ mesh_ob.matrix_world)
    co = points @ to_rig[:3, :3].T + to_rig[:3, 3]
    smpl = body["verts"] @ to_rig[:3, :3].T + to_rig[:3, 3]
    pelvis_z = float(bones["pelvis"].head_local[2])
    dist, radius = {}, {}
    for side, chain in LEGS.items():
        joints = [np.array(bones[b].head_local) for b in chain]
        d = np.min([_segment_distance(co, a, b)
                    for a, b in zip(joints, joints[1:])], axis=0)
        ds = np.min([_segment_distance(smpl, a, b)
                     for a, b in zip(joints[:3], joints[1:3])], axis=0)
        own = body["weights"][:, [column[b] for b in chain[:2]]].sum(1) > 0.8
        radius[side] = float(np.percentile(ds[own], 90)) if own.any() \
            else 0.08
        dist[side] = d
    nearer_left = dist["left"] / radius["left"] <= \
        dist["right"] / radius["right"]
    d = np.where(nearer_left, dist["left"], dist["right"])
    r = np.where(nearer_left, radius["left"], radius["right"])
    share = np.clip((d - r) / (LEG_SOFT * r), 0.0, 1.0)
    share[near | (co[:, 2] > pelvis_z)] = 0.0
    if not share.any():
        return weights, 0
    skirt = np.zeros_like(weights)
    skirt[:, column["pelvis"]] = SKIRT_PELVIS
    thigh = np.where(nearer_left, column["left_hip"], column["right_hip"])
    skirt[np.arange(len(skirt)), thigh] += 1.0 - SKIRT_PELVIS
    out = weights * (1.0 - share[:, None]) + skirt * share[:, None]
    return out, int((share > 0.5).sum())


def apply(mesh_ob, rig, body, scale, offset, stature):
    """Give ``mesh_ob`` SMPL-X's weights and bind it to ``rig``.

    ``body`` is :func:`load`'s, in the asset's units; ``scale`` and ``offset``
    are what took the mesh from those units to the rig's (``photo_body.fit``).
    Returns a report.
    """
    body = dict(body, verts=body["verts"] * scale + np.asarray(offset))
    me = mesh_ob.data
    n = len(me.vertices)
    points = np.empty(n * 3)
    me.vertices.foreach_get("co", points)
    points = points.reshape(-1, 3)
    normals = np.array([tuple(v.normal) for v in me.vertices])
    edges = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)

    weights, near = on_body(points, normals, body, stature)
    weights, stranded = attach(weights, near, points, edges)
    weights, hemmed = hem(weights, near, points, rig, mesh_ob, body)
    weights = strongest(smooth(weights, edges))

    present = {bone.name for bone in rig.data.bones}
    for group in list(mesh_ob.vertex_groups):
        mesh_ob.vertex_groups.remove(group)
    for column, name in enumerate(body["bones"]):
        if name not in present:
            continue
        group = mesh_ob.vertex_groups.new(name=name)
        for i in np.nonzero(weights[:, column] > 1e-4)[0]:
            group.add([int(i)], float(weights[i, column]), "REPLACE")
    if not any(m.type == "ARMATURE" for m in mesh_ob.modifiers):
        modifier = mesh_ob.modifiers.new("Armature", "ARMATURE")
        modifier.object = rig
        mesh_ob.parent = rig
    return {"on_body_share": round(float(near.mean()), 3),
            "stranded": stranded, "hem_vertices": hemmed,
            "fit_error": body["error"]}
