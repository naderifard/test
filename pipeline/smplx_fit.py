"""SMPL-X fitted to a character's rig, for its skin weights.

A photo-built character is skinned by Blender's heat weighting, which knows
only where the bones are: shoulders collapse, an arm drags the torso beside
it, a thigh pulls the other thigh. SMPL-X's weights were learned from
thousands of body scans, and they are attached to a body, not to a mesh
topology -- so if SMPL-X stands inside the character, in its pose and at its
size, every vertex of the character can take the weights of the SMPL-X
surface it lies on (``blender/smplx_skin.py``).

This module does the standing-inside: it optimises SMPL-X's pose, shape,
scale and position so that its joints land on the rig's joints, which are
SMPL-X's own body joints by name. It runs in the motion env (torch, smplx).

Axes and sides. SMPL-X stands on +Y facing +Z with its left on +X. The rig
stands on +Z facing +Y, and its ``left_*`` bones are on +X: the map
``(x, y, z) -> (-x, z, y)`` is the rotation between the two, and it puts
SMPL-X's left on the rig's -X, which the rig calls right. So names swap sides.
"""

from __future__ import annotations

import os

SMPLX_BODY = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot",
    "right_foot", "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "jaw")
LEFT_HAND, RIGHT_HAND = range(25, 40), range(40, 55)
EYES = (23, 24)
# How much each joint's target is trusted: the limbs' come from landmarks on
# the photo, the spine's and collars' are interpolated between them.
TRUST = {"spine1": 0.5, "spine2": 0.5, "spine3": 0.5, "left_collar": 0.3,
         "right_collar": 0.3, "jaw": 0.2, "left_foot": 0.5, "right_foot": 0.5}
ITERATIONS = 500
POSE_PRIOR = 2e-3
SHAPE_PRIOR = 5e-3


def model_dir():
    from pipeline import envs
    return os.path.join(envs.MODEL_ROOT, "body_models")


def available():
    try:
        import smplx  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return os.path.exists(os.path.join(model_dir(), "smplx",
                                       "SMPLX_NEUTRAL.npz"))


def rig_name(smplx_name):
    """The rig's bone for an SMPL-X joint: the other side's (see module)."""
    if smplx_name.startswith("left_"):
        return "right_" + smplx_name[5:]
    if smplx_name.startswith("right_"):
        return "left_" + smplx_name[6:]
    return smplx_name


def bone_weights(lbs, bones):
    """SMPL-X's 55-joint weights collapsed onto the rig's ``bones``: fingers
    into the wrist, eyes into the head, and a joint the rig lacks into the
    nearest one it has (the jaw into the head)."""
    import numpy as np
    column = {name: i for i, name in enumerate(bones)}
    out = np.zeros((lbs.shape[0], len(bones)), np.float32)

    def add(index, name, fallback):
        out[:, column[name if name in column else fallback]] += lbs[:, index]

    for i, name in enumerate(SMPLX_BODY):
        add(i, rig_name(name), "head" if name == "jaw" else "pelvis")
    for i in EYES:
        add(i, "head", "head")
    for i in LEFT_HAND:
        add(i, rig_name("left_wrist"), "pelvis")
    for i in RIGHT_HAND:
        add(i, rig_name("right_wrist"), "pelvis")
    return out / np.maximum(out.sum(1, keepdims=True), 1e-9)


def to_rig_axes(points):
    """SMPL-X axes -> rig axes: (x, y, z) -> (-x, z, y), a rotation."""
    import torch
    return torch.stack([-points[..., 0], points[..., 2], points[..., 1]], -1)


SURFACE_WEIGHT = 0.6
SURFACE_EVERY = 25           # iterations between re-matching the surface
SURFACE_SAMPLE = 3           # every n-th SMPL-X vertex
SURFACE_REACH = 0.06         # of the height: further than this is not body


def fit(joints, bones, surface=None, iterations=ITERATIONS, log=print):
    """Fit SMPL-X to ``joints`` (rig bone name -> xyz, rig axes, any unit).

    ``surface``: the character's own vertices (M, 3), same space. The rig's
    joints are landmarks on a photograph given a guessed depth, and fitted
    to them alone the body stood a little forward of the character, its
    belly outside the tunic. With the surface, each SMPL-X vertex is also
    drawn to the nearest point of the character (re-matched as the fit
    moves), which settles depth and girth; a vertex with nothing of the
    character within ``SURFACE_REACH`` is left to the joints.

    Returns a dict: ``verts`` (N, 3) in the joints' space, ``normals``,
    ``faces``, ``weights`` (N, len(bones)) over ``bones``, ``joints`` (the
    fitted ones by rig name) and ``error`` (mean joint distance as a share of
    the figure's height).
    """
    import numpy as np
    import smplx
    import torch

    model = smplx.create(model_dir(), model_type="smplx", gender="neutral",
                         use_pca=False, flat_hand_mean=True, num_betas=10,
                         ext="npz")
    names = [rig_name(n) for n in SMPLX_BODY]
    have = [i for i, n in enumerate(names) if n in joints]
    target = torch.tensor([joints[names[i]] for i in have], dtype=torch.float32)
    trust = torch.tensor([TRUST.get(SMPLX_BODY[i], 1.0) for i in have])
    height = float(target[:, 2].max() - target[:, 2].min())

    betas = torch.zeros(1, 10, requires_grad=True)
    pose = torch.zeros(1, 63, requires_grad=True)
    orient = torch.zeros(1, 3, requires_grad=True)
    with torch.no_grad():
        rest = to_rig_axes(model().joints[0, :23])
        rest_height = float(rest[:, 2].max() - rest[:, 2].min())
    log_scale = torch.tensor([np.log(height / rest_height)],
                             dtype=torch.float32, requires_grad=True)
    shift = (target.mean(0) - rest[have].mean(0) * (height / rest_height)) \
        .clone().requires_grad_(True)

    def posed():
        out = model(betas=betas, body_pose=pose, global_orient=orient)
        scale = torch.exp(log_scale)
        return (to_rig_axes(out.vertices[0]) * scale + shift,
                to_rig_axes(out.joints[0, :23]) * scale + shift)

    optimiser = torch.optim.Adam([betas, pose, orient, log_scale, shift],
                                 lr=0.03)
    tree = None
    if surface is not None and len(surface):
        from scipy.spatial import cKDTree
        tree = cKDTree(np.asarray(surface, np.float64))
        sample = torch.arange(0, model.v_template.shape[0], SURFACE_SAMPLE)
    matched = None
    for step in range(iterations):
        optimiser.zero_grad()
        verts, fitted = posed()
        gap = ((fitted[have] - target) ** 2).sum(1)
        loss = (gap * trust).sum() / height ** 2 \
            + POSE_PRIOR * (pose ** 2).sum() + SHAPE_PRIOR * (betas ** 2).sum()
        # The surface only once the joints have put the body roughly in
        # place: matched from the T-pose, an arm finds the torso.
        if tree is not None and step >= iterations // 4:
            if matched is None or step % SURFACE_EVERY == 0:
                dist, index = tree.query(verts[sample].detach().numpy())
                near = torch.from_numpy(dist < SURFACE_REACH * height)
                matched = torch.from_numpy(
                    np.asarray(surface, np.float32)[index])
            pull = ((verts[sample] - matched) ** 2).sum(1)[near]
            if len(pull):
                loss = loss + SURFACE_WEIGHT * pull.mean() / height ** 2 \
                    * len(have)
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        verts, fitted = posed()
        error = float(((fitted[have] - target) ** 2).sum(1).sqrt().mean()
                      / height)
    verts = verts.numpy().astype(np.float32)
    faces = np.asarray(model.faces, np.int64)
    # (-x, z, y) is a rotation, so the faces keep their winding.
    tri = verts[faces]
    face_normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(verts)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normal)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True),
                          1e-12)
    log(f"smplx fit: joints within {error:.1%} of the height, betas "
        f"{[round(float(b), 2) for b in betas[0][:4]]}")
    return {"verts": verts, "normals": normals.astype(np.float32),
            "faces": faces,
            "weights": bone_weights(model.lbs_weights.numpy(), list(bones)),
            "bones": list(bones),
            "joints": {names[i]: [float(v) for v in fitted[i]]
                       for i in range(23)},
            "error": round(error, 4)}


def save(path, result):
    import json

    import numpy as np
    np.savez_compressed(
        path, verts=result["verts"], normals=result["normals"],
        faces=result["faces"], weights=result["weights"],
        bones=np.array(result["bones"]),
        meta=np.array(json.dumps({"joints": result["joints"],
                                  "error": result["error"]})))
