"""A character from three views: Hunyuan3D-2mv shape, colours from every view.

SF3D sees one front photo and guesses the rest, and it shows: the side of a
figure was a slab with the front photo smeared across it, the back a lumpy
guess. With the turnaround sheet the seed prompt now asks for (left profile
and back, made by editing the front image) the character's real depth and
back are known, and Hunyuan3D-2mv reconstructs the shape from front, left and
back together (the ``turbo`` variant: ten steps, about 85 s and 5.7 GB of VRAM
on the RTX 4070 Laptop).

Then every view paints the part of the surface it faces. Each view is an
orthographic camera fitted to its image -- the reference images are asked for
with a long lens, so orthographic is the honest model -- by matching the
mesh's projected silhouette to the image's matte: height to height, centre to
centre. A vertex takes each view's colour weighted by how squarely it faces
that view and whether the view can see it (a z-buffer); the right side, which
no image shows, borrows the left profile mirrored at a lower weight. The front
photo stays the authority: its projection is kept as the UV map and its trust
as the alpha, exactly as the SF3D path writes them, so ``photo_body`` renders
the front at the photo's full resolution and the views' colours everywhere
the front cannot see.

Axes. Hunyuan's mesh is y-up with the front toward +z and image-right toward
+x. The pipeline's internal frame (``sf3d_dense.to_blender``'s input) has the
figure facing +x, up +z, and image-right (the figure's anatomical left) on +y;
``from_hunyuan`` maps one to the other.
"""

from __future__ import annotations

import os
import sys

REPO = "tencent/Hunyuan3D-2mv"
SUBFOLDER = "hunyuan3d-dit-v2-mv-turbo"
STEPS = 10
OCTREE = 380
SEED = 12345
HEIGHT_UNITS = 0.92            # SF3D-sized, so sf3d_dense's tolerances hold
FAR = 100.0
VIEW_WEIGHT = {"front": 1.0, "left": 0.9, "back": 0.9, "right": 0.45}
FILL_ITERATIONS = 200

_PIPELINE = None


def available():
    from pipeline import envs
    code = os.path.join(envs.VENDOR_ROOT, "Hunyuan3D-2")
    weights = os.path.join(envs.MODEL_ROOT, "hunyuan3d", REPO, SUBFOLDER)
    for path in (code, weights):
        if not os.path.isdir(path):
            return False, f"not on this machine: {path}"
    try:
        import pymeshlab  # noqa: F401
    except ImportError:
        return False, "pymeshlab is not installed in this env"
    return True, ""


def _pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from pipeline import envs
        code = os.path.join(envs.VENDOR_ROOT, "Hunyuan3D-2")
        if code not in sys.path:
            sys.path.insert(0, code)
        os.environ.setdefault("HY3DGEN_MODELS",
                              os.path.join(envs.MODEL_ROOT, "hunyuan3d"))
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        _PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            REPO, subfolder=SUBFOLDER, variant="fp16", use_safetensors=True)
    return _PIPELINE


def release():
    global _PIPELINE
    if _PIPELINE is not None:
        import torch
        _PIPELINE = None
        torch.cuda.empty_cache()


def from_hunyuan(points):
    """Hunyuan axes (y up, front +z, image-right +x) -> internal axes."""
    import numpy as np
    p = np.asarray(points, dtype=np.float64)
    return np.stack([p[..., 2], p[..., 0], p[..., 1]], -1)


def shape(images):
    """A trimesh in internal axes, scaled to ``HEIGHT_UNITS`` tall and stood
    on z=0, from ``{"front", "left", "back"}`` RGBA PIL images."""
    import time

    import numpy as np
    import torch
    import trimesh
    from hy3dgen.shapegen import DegenerateFaceRemover, FloaterRemover
    started = time.time()
    mesh = _pipeline()(image=images, num_inference_steps=STEPS,
                       octree_resolution=OCTREE, num_chunks=20000,
                       generator=torch.manual_seed(SEED),
                       output_type="trimesh")[0]
    mesh = DegenerateFaceRemover()(FloaterRemover()(mesh))
    verts = from_hunyuan(mesh.vertices)
    low, high = verts.min(0), verts.max(0)
    scale = HEIGHT_UNITS / float(high[2] - low[2])
    centre = (low + high) / 2.0
    verts = (verts - [centre[0], centre[1], low[2]]) * scale
    out = trimesh.Trimesh(verts, np.asarray(mesh.faces), process=False)
    return out, {"seconds": round(time.time() - started, 1),
                 "vertices": int(len(verts)), "faces": int(len(mesh.faces)),
                 "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30,
                                  2)}


MAX_FACES = 200000


def reduce(mesh, max_faces=MAX_FACES):
    """Quadric decimation to ``max_faces``: Hunyuan returns up to 600k
    triangles, which made every ray and z-buffer after it slow, and
    photo_body decimates to 60k for the rig in any case."""
    if len(mesh.faces) <= max_faces:
        return mesh
    from hy3dgen.shapegen import FaceReducer
    return FaceReducer()(mesh, max_facenum=max_faces)


# -- views as cameras ------------------------------------------------------
# name -> (view direction, image-right, up), internal axes; the camera looks
# along the view direction. See the module docstring for the frame.
VIEW_AXES = {
    "front": ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "left": ((0, -1, 0), (-1, 0, 0), (0, 0, 1)),
    "back": ((1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "right": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
}


class OrthoView:
    """An orthographic camera fitted to one image of the figure.

    Pixels are the image's own. ``project`` and ``ray`` have the shape of
    ``sf3d_dense.Camera``'s, so ``photo_mapping`` and ``joint_from_pixel``
    work on it unchanged (through ``PhotoCond``).
    """

    def __init__(self, name, verts, alpha):
        import numpy as np
        view, right, up = (np.asarray(a, dtype=np.float64)
                           for a in VIEW_AXES[name])
        self.view, self.right, self.up = view, right, up
        a, b = verts @ right, verts @ up
        ys, xs = np.nonzero(np.asarray(alpha) > 0.5)
        self.scale = (ys.max() - ys.min()) / max(1e-9, b.max() - b.min())
        self.a0 = (a.max() + a.min()) / 2.0
        self.b0 = b.max()
        self.u0 = (xs.max() + xs.min()) / 2.0
        self.v0 = float(ys.min())
        self.T = -view * FAR          # for the facing term: toward the camera
        self.name = name
        self.px = int(max(np.asarray(alpha).shape))
        self.fit = self._refine(verts, alpha)

    def _silhouette(self, verts, shape, k):
        import numpy as np
        from scipy.ndimage import binary_closing, binary_fill_holes
        u, v, _ = self.project(verts)
        h, w = shape
        mask = np.zeros((h // k + 1, w // k + 1), bool)
        ui = np.clip((u / k).astype(int), 0, mask.shape[1] - 1)
        vi = np.clip((v / k).astype(int), 0, mask.shape[0] - 1)
        mask[vi, ui] = True
        return binary_fill_holes(binary_closing(mask, iterations=2))

    def _refine(self, verts, alpha, k=4):
        """Shift and scale searched to best overlap the model's silhouette
        with the image's matte (intersection over union). The height-and-
        centre fit is a start; wings, arms or a tail that the model and the
        picture hold slightly differently move a bounding-box centre, and a
        few pixels of misfit painted a second profile across a cheek."""
        import numpy as np
        a = np.asarray(alpha) > 0.5
        target = a[::k, ::k]
        target = np.pad(target, ((0, 1), (0, 1)))[:target.shape[0] + 1,
                                                  :target.shape[1] + 1]
        sample = verts[::max(1, len(verts) // 60000)]
        base = (self.scale, self.u0, self.v0)
        best, best_iou = base, -1.0
        rows = a.nonzero()[0]
        height = float(np.ptp(rows) or 1)
        for ds in (0.97, 0.985, 1.0, 1.015, 1.03):
            for du in np.linspace(-0.04, 0.04, 9) * height:
                for dv in (-0.01 * height, 0.0, 0.01 * height):
                    self.scale, self.u0, self.v0 = (base[0] * ds, base[1] + du,
                                                    base[2] + dv)
                    sil = self._silhouette(sample, a.shape, k)
                    hh = min(sil.shape[0], target.shape[0])
                    ww = min(sil.shape[1], target.shape[1])
                    s_, t_ = sil[:hh, :ww], target[:hh, :ww]
                    iou = (s_ & t_).sum() / max(1, (s_ | t_).sum())
                    if iou > best_iou:
                        best_iou, best = iou, (self.scale, self.u0, self.v0)
        self.scale, self.u0, self.v0 = best
        return {"iou": round(float(best_iou), 4),
                "scale": round(best[0] / base[0], 4),
                "shift_px": [round(best[1] - base[1], 1),
                             round(best[2] - base[2], 1)]}

    def project(self, points):
        import numpy as np
        p = np.asarray(points, dtype=np.float64)
        u = (p @ self.right - self.a0) * self.scale + self.u0
        v = -(p @ self.up - self.b0) * self.scale + self.v0
        depth = p @ self.view + FAR
        return u, v, depth

    def ray(self, u, v):
        import numpy as np
        a = (u - self.u0) / self.scale + self.a0
        b = -(v - self.v0) / self.scale + self.b0
        point = self.right * a + self.up * b
        return point - self.view * FAR, self.view.copy()

    def facing_normal_dot(self, normals):
        return -(normals @ self.view)


class PhotoCond:
    """What ``sf3d_dense.photo_mapping`` needs from a conditioning object,
    for an image seen through an ``OrthoView`` in its own pixels."""

    def __init__(self, rgba, alpha=None):
        import numpy as np
        array = np.asarray(rgba.convert("RGBA"))
        self.photo_size = rgba.size
        self.size = rgba.size[0]
        self.left = self.top = 0
        self.alpha = (np.asarray(alpha, dtype=np.float32) / 255.0
                      if alpha is not None else array[:, :, 3] / 255.0)
        from pipeline import sf3d_dense
        clean = np.asarray(sf3d_dense.bleed(rgba.convert("RGBA")))
        self.rgb = clean[:, :, :3].astype(np.float64) / 255.0

    def to_photo(self, u, v, cond_px):
        return u, v

    def to_cond(self, x, y, cond_px):
        return x, y


class FrontCamera:
    """An ``OrthoView`` of the front photo addressed in the SF3D crop's
    pixels, so ``joint_from_pixel`` and ``photo_mapping`` can be handed the
    front's ``Conditioned`` object exactly as the SF3D path does."""

    def __init__(self, ortho, cond):
        self.ortho, self.cond = ortho, cond
        self.px = cond.size
        self.T = ortho.T

    def project(self, points):
        u, v, depth = self.ortho.project(points)
        return u - self.cond.left, v - self.cond.top, depth

    def ray(self, u, v):
        return self.ortho.ray(u + self.cond.left, v + self.cond.top)


# Each view's weight falls off sharply with the angle from it, so that where
# two views overlap one of them clearly wins: a soft hand-over (the SF3D
# path's ramp) blended the front photo's face with the profile's across the
# side of the head, and Sorush had two faces at 45 degrees.
SHARP_LO, SHARP_HI, SHARP_POWER = 0.20, 1.00, 4.0
SIDE_VIEWS = ("left", "back", "right")


def view_mapping(mesh, camera, cond):
    """``photo_mapping``'s UVs and visibility, with the sharp weight."""
    import numpy as np

    from pipeline import sf3d_dense
    uv, weight, mapping = sf3d_dense.photo_mapping(mesh, camera, cond)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    toward = camera.T[None, :] - verts
    toward /= np.linalg.norm(toward, axis=1, keepdims=True)
    facing = (np.asarray(mesh.vertex_normals) * toward).sum(1)
    ramp = np.clip((facing - SHARP_LO) / (SHARP_HI - SHARP_LO), 0.0, 1.0)
    soft = np.clip((facing - sf3d_dense.FACING_LO)
                   / (sf3d_dense.FACING_HI - sf3d_dense.FACING_LO), 0.0, 1.0)
    # photo_mapping's weight is visibility x soft ramp x matte; swap the ramp.
    base = np.where(soft > 1e-6, weight / np.maximum(soft, 1e-6), 0.0)
    return uv, base * ramp ** SHARP_POWER, mapping


# Where views overlap, only the views within SEAM_KEEP of the best one paint,
# faded to nothing at that ratio: a seam a few pixels wide instead of a broad
# cross-fade. Two images of one eye blended over a band showed two eyes.
SEAM_KEEP = 0.75
# The front photo alone paints the face: the top of a standing figure (the
# head of an animal) wherever the front camera sees it at all.
HEAD_SHARE = {"human": 0.14, "humanoid": 0.16}
ANIMAL_HEAD_SHARE = 0.22
FACE_FACING = 0.5         # the front owns the face up to ~60 degrees round


def select_views(mesh, weights, front_camera, front_cond, kind):
    """Winner-takes-most view weights, and the face given to the front.

    ``weights`` maps view names to per-vertex weights; returns a new dict
    and the number of vertices the front was given as face.
    """
    import numpy as np

    from pipeline import sf3d_dense
    names = list(weights)
    stack = np.stack([weights[n] for n in names], 1)
    best = stack.max(1, keepdims=True)
    ratio = np.where(best > 0, stack / np.maximum(best, 1e-12), 0.0)
    fade = np.clip((ratio - SEAM_KEEP) / (1.0 - SEAM_KEEP), 0.0, 1.0)
    stack = stack * fade
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals)
    facing = -(normals @ front_camera.ortho.view)
    if kind in HEAD_SHARE:
        top = verts[:, 2].max()
        head = verts[:, 2] > top - HEAD_SHARE[kind] * (top - verts[:, 2].min())
    else:
        front_x = verts[:, 0].max()
        head = verts[:, 0] > front_x - ANIMAL_HEAD_SHARE * np.ptp(verts[:, 0])
    # Visibility and matte without the facing ramp, from the front camera.
    _uv, soft, _m = sf3d_dense.photo_mapping(mesh, front_camera, front_cond)
    face = head & (facing > FACE_FACING) & (soft > 0.0)
    fi = names.index("front")
    stack[face] = 0.0
    stack[face, fi] = np.maximum(soft[face], 1e-3)
    return {n: stack[:, i] for i, n in enumerate(names)}, int(face.sum())


def colours(mesh, views, kind="human"):
    """Per-vertex sRGB colour from every view, and a report.

    ``views`` maps a view name to ``(camera, cond)``. Returns ``(colour,
    weight_total, report)``; vertices no view sees get the colours of their
    neighbours by diffusion over the mesh's edges.
    """
    import numpy as np

    from pipeline import sf3d_dense
    n = len(mesh.vertices)
    total = np.zeros((n, 3))
    weight = np.zeros(n)
    report = {}
    per_view = {}
    for name, (camera, cond) in views.items():
        uv, w, mapping = view_mapping(mesh, camera, cond)
        h, wd = cond.rgb.shape[:2] if hasattr(cond, "rgb") else (None, None)
        if hasattr(cond, "rgb"):
            rgb = cond.rgb
        else:
            rgb = np.asarray(sf3d_dense.bleed(cond.cut).convert("RGB"),
                             dtype=np.float64) / 255.0
            h, wd = rgb.shape[:2]
        x = np.clip(uv[:, 0] * wd, 0, wd - 1).astype(int)
        y = np.clip((1.0 - uv[:, 1]) * h, 0, h - 1).astype(int)
        k = VIEW_WEIGHT.get(name, 0.5) * w
        per_view[name] = [uv, k, rgb[y, x]]
        report[name] = {"photo_share": mapping["photo_share"],
                        "visible": mapping["visible"],
                        "fit": getattr(getattr(camera, "ortho", camera),
                                       "fit", None)}
    front_camera, front_cond = views["front"]
    chosen, face = select_views(mesh, {n: v[1] for n, v in per_view.items()},
                                front_camera, front_cond, kind)
    report["face_vertices"] = face
    for name, (uv, _k, sampled) in list(per_view.items()):
        k = chosen[name]
        per_view[name] = (uv, k)
        total += sampled * k[:, None]
        weight += k
    colour = total / np.maximum(weight, 1e-9)[:, None]
    seen = weight > 0.05
    report["seen_share"] = round(float(seen.mean()), 4)
    edges = np.asarray(mesh.edges_unique)
    a, b = edges[:, 0], edges[:, 1]
    known = seen.astype(np.float64)
    for _ in range(FILL_ITERATIONS):
        if known.all():
            break
        acc = np.zeros((n, 3))
        cnt = np.zeros(n)
        np.add.at(acc, a, colour[b] * known[b, None])
        np.add.at(acc, b, colour[a] * known[a, None])
        np.add.at(cnt, a, known[b])
        np.add.at(cnt, b, known[a])
        grow = (known == 0) & (cnt > 0)
        colour[grow] = acc[grow] / cnt[grow, None]
        known[grow] = 1.0
    report["per_view"] = per_view
    return np.clip(colour, 0.0, 1.0), weight, report
