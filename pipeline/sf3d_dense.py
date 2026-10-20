"""A finer surface out of SF3D, coloured, with the reference photo pinned to it.

``sf3d_geometry`` gets SF3D loaded on this machine and its ``geometry()`` returns
the mesh SF3D's own path would: marching tetrahedra over the 160-grid that ships
with the checkpoint. That grid is the ceiling on detail -- a standing figure
comes out at 4,600 to 10,500 vertices, a face at a few dozen voxels -- and it is
a ceiling of the *extractor*, not of the model. The density field SF3D predicts
is a continuous function of position (a triplane sampled by an MLP), so this
module samples it on a grid of its own choosing, tight around the figure, and
runs marching cubes over that. At 6.3M grid points Keyumars
comes out at 137,000 vertices, watertight, one shell, with the density pass
taking 0.17 s.

Three more things come out of the same pass, because the same camera explains
all of them:

* **Albedo per vertex.** SF3D's decoder has a ``features`` head that is the
  base colour its own texture baker writes. Sampling it at the vertices gives
  colour everywhere, including the back and sides that no photo shows.
* **The photo, projected.** SF3D conditions on a fixed camera -- distance 1.6,
  vertical field of view 40 degrees, a 512 px square cropped around the figure
  by ``resize_foreground`` -- so every vertex has an exact pixel in the original
  reference image. The projection convention was measured rather than assumed:
  of the four sign choices, the right one puts 82% of Keyumars' vertices inside
  the photo's own mask and the wrong ones 57-67%.
* **How much to trust the photo at each vertex.** A vertex hidden behind
  another part of the figure, facing away, or landing on background pixels
  would drag the wrong colour onto the mesh. ``photo_w`` is visibility (a
  vertex z-buffer) times a facing ramp times the matte, and the Blender side
  mixes photo over albedo by it.

**Axes.** SF3D's internal frame has the camera on +X looking back at the origin,
up +Z, and image-right along +Y. This module hands out Blender's frame directly
-- up +Z, the figure facing +Y (the rest direction ``prelude`` rigs are built
in), so ``(xb, yb, zb) = (-yi, xi, zi)`` -- rather than applying glTF's export
rotations and undoing them again on import. Nothing here is scaled to metres;
that is the Blender step's job, once it knows what the figure is.
"""

from __future__ import annotations

import time

# Grid points in the density pass. Sized by a budget of points rather than by
# cells along the longest side, because the figure's box decides how much
# surface a given cell size produces: 384 cells along a standing man's height is
# 6.3M points and 137k vertices, while 384 along a lion's box was 647k
# triangles. At this budget the step adapts to the box and both come out alike.
GRID_POINTS = 6.3e6
# Padding around the coarse mesh's box, as a fraction of its longest side, so a
# surface that the finer grid finds slightly outside the coarse one is not cut.
GRID_PAD = 0.03
CHUNK = 1 << 18

# The ramp from edge-on to facing the camera over which the photo takes over.
FACING_LO, FACING_HI = 0.05, 0.40
# z-buffer resolution and depth slack, in SF3D units (the figure is ~0.9 tall).
ZBUF_PX = 384
ZBUF_TOL = 0.012

# A shell with fewer than this share of the vertices is debris from the
# isosurface, not part of the figure.
MIN_SHELL_SHARE = 0.02


SOLID_ALPHA = 245


def bleed(rgba):
    """The cut-out with every transparent pixel coloured like the figure's edge.

    The matte decides *whether* a pixel is figure; the renderer still filters
    colour across the boundary, and a transparent pixel keeps whatever RGB the
    matting left there -- light grey on this project's references, which
    showed as a pale halo around the dark div. Each transparent pixel takes the
    colour of its nearest opaque one, so filtering at the edge blends figure
    with figure. Alpha is untouched.
    """
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt

    array = np.array(rgba.convert("RGBA"))
    # Only fully solid pixels keep their colour: a semi-transparent fur edge
    # (alpha ~0.66 all round the black div) is part background, light grey,
    # and painted the div's inner legs grey wherever it was projected.
    solid = array[:, :, 3] >= SOLID_ALPHA
    if solid.all() or not solid.any():
        return rgba
    _dist, (iy, ix) = distance_transform_edt(~solid, return_indices=True)
    array[:, :, :3] = array[iy, ix, :3]
    return Image.fromarray(array, "RGBA")


def to_blender(points):
    """SF3D internal axes -> Blender axes, figure facing +Y. Works on (N, 3)."""
    import numpy as np
    points = np.asarray(points, dtype=np.float64)
    return np.stack([-points[..., 1], points[..., 0], points[..., 2]], -1)


class Conditioned:
    """One reference photo prepared exactly as SF3D sees it, plus the inverse.

    Keeps the crop box so a pixel in the 512 px conditioning image can be
    mapped back to the original photo, which is what makes the projection
    exact rather than fitted.
    """

    def __init__(self, photo, rembg_session, ratio=0.85):
        import numpy as np
        from sf3d.utils import get_bbox_from_mask

        from PIL import Image

        from pipeline import matte
        self.photo_size = photo.size
        # BiRefNet first, rembg and a plain-backdrop flood as fallbacks, and
        # never the whole frame (pipeline/matte.py: the Sorush slab).
        alpha, self.matte_method = matte.figure(photo, rembg_session)
        self.cut = photo.convert("RGB").convert("RGBA")
        self.cut.putalpha(Image.fromarray(alpha))
        alpha = np.array(self.cut)[:, :, 3]
        x1, y1, x2, y2 = get_bbox_from_mask(alpha, thr=0.5)
        yc, xc = (y1 + y2) / 2.0, (x1 + x2) / 2.0
        scale = max(y2 - y1, x2 - x1) / ratio
        # The same integer arithmetic resize_foreground uses, so the box here
        # and the crop SF3D conditions on are the same pixels.
        self.left = int(xc - scale / 2)
        self.top = int(yc - scale / 2)
        self.size = int(scale)
        self.image = self.cut.crop((self.left, self.top,
                                    self.left + self.size,
                                    self.top + self.size))
        self.figure_box = (int(x1), int(y1), int(x2), int(y2))
        self.alpha = alpha.astype(np.float32) / 255.0
        self.coverage = float((alpha > 127).mean())

    def to_photo(self, u, v, cond_px):
        """Conditioning-image pixels -> original photo pixels."""
        k = self.size / float(cond_px)
        return self.left + u * k, self.top + v * k

    def to_cond(self, x, y, cond_px):
        """Original photo pixels -> conditioning-image pixels."""
        k = cond_px / float(self.size)
        return (x - self.left) * k, (y - self.top) * k


class Camera:
    """SF3D's conditioning camera, as a projector and a ray caster."""

    def __init__(self, model):
        import numpy as np
        from sf3d.utils import create_intrinsic_from_fov_deg, default_cond_c2w

        self.px = int(model.cfg.cond_image_size)
        c2w = default_cond_c2w(model.cfg.default_distance).float().numpy()
        self.R, self.T = c2w[:3, :3].astype(np.float64), c2w[:3, 3].astype(np.float64)
        intrinsic, _ = create_intrinsic_from_fov_deg(
            model.cfg.default_fovy_deg, self.px, self.px)
        self.f = float(intrinsic[0, 0])
        self.c2w = c2w

    def project(self, points):
        """World points -> (u, v, depth) in conditioning pixels."""
        import numpy as np
        cam = (np.asarray(points, dtype=np.float64) - self.T) @ self.R
        depth = -cam[:, 2]
        u = self.px / 2.0 + self.f * cam[:, 0] / depth
        v = self.px / 2.0 - self.f * cam[:, 1] / depth
        return u, v, depth

    def ray(self, u, v):
        """Origin and unit direction of the ray through conditioning pixel u, v."""
        import numpy as np
        d_cam = np.array([(u - self.px / 2.0) / self.f,
                          -(v - self.px / 2.0) / self.f, -1.0])
        d = self.R @ d_cam
        return self.T.copy(), d / np.linalg.norm(d)


def scene_codes(model, cond):
    """SF3D's triplane for one conditioned image. The expensive pass."""
    import torch
    from sf3d.utils import create_intrinsic_from_fov_deg, default_cond_c2w

    device = next(model.parameters()).device
    n = model.cfg.cond_image_size
    with torch.no_grad():
        mask_cond, rgb_cond = model.prepare_image(cond.image)
        c2w = default_cond_c2w(model.cfg.default_distance).to(device)
        intrinsic, normed = create_intrinsic_from_fov_deg(
            model.cfg.default_fovy_deg, n, n)
        batch = {
            "rgb_cond": model.image_processor(rgb_cond, n),
            "mask_cond": model.image_processor(mask_cond, n),
            "c2w_cond": c2w.view(1, 1, 4, 4),
            "intrinsic_cond": intrinsic.to(device).view(1, 1, 3, 3),
            "intrinsic_normed_cond": normed.to(device).view(1, 1, 3, 3),
        }
        # See sf3d_geometry.geometry: the camera helpers are float32 and the
        # weights float16, and there is no autocast here to reconcile them.
        dtype = next(model.parameters()).dtype
        for key, value in batch.items():
            if torch.is_tensor(value) and value.is_floating_point():
                batch[key] = value.to(dtype)
        codes, _ = model.get_scene_codes(batch)
        coarse = model.triplane_to_meshes(codes)[0]
        low = coarse.v_pos.min(0).values.float().cpu().numpy()
        high = coarse.v_pos.max(0).values.float().cpu().numpy()
    return codes, (low, high)


def _decode(model, codes, points, head):
    import numpy as np
    import torch
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(points), CHUNK):
            chunk = torch.from_numpy(
                np.ascontiguousarray(points[start:start + CHUNK], dtype=np.float32)
            ).to(device)
            values = model.query_triplane(chunk, codes[0])
            decoded = model.decoder(values, include=[head])[head]
            out.append(decoded.float().reshape(len(chunk), -1).cpu().numpy())
    return np.concatenate(out)


def dense_mesh(model, codes, bounds, points=GRID_POINTS):
    """Marching cubes over SF3D's density, on a grid fitted to the figure."""
    import numpy as np
    import trimesh
    from skimage import measure

    started = time.time()
    low, high = (np.asarray(b, dtype=np.float64) for b in bounds)
    pad = GRID_PAD * float((high - low).max())
    low, high = low - pad, high + pad
    step = (float(np.prod(high - low)) / points) ** (1.0 / 3.0)
    dims = np.ceil((high - low) / step).astype(int) + 1
    axes = [low[i] + step * np.arange(dims[i]) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    density = _decode(model, codes, grid, "density")[:, 0] \
        - float(model.cfg.isosurface_threshold)
    volume = density.reshape(dims)
    if volume.max() <= 0.0:
        raise RuntimeError("SF3D's density never crosses the surface threshold")
    verts, faces, _normals, _values = measure.marching_cubes(volume, level=0.0)
    verts = low + verts * step
    mesh = trimesh.Trimesh(verts, faces, process=True)
    shells = mesh.split(only_watertight=False)
    total = sum(len(s.vertices) for s in shells) or 1
    kept = [s for s in shells if len(s.vertices) / total >= MIN_SHELL_SHARE]
    mesh = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
    # Marching cubes leaves a one-voxel staircase that shades as fine stripes;
    # Taubin smoothing removes it without shrinking the figure the way plain
    # Laplacian smoothing does.
    trimesh.smoothing.filter_taubin(mesh, iterations=6)
    return mesh, {
        "grid": [int(d) for d in dims],
        "cell": round(step, 6),
        "shells": len(shells),
        "shells_kept": len(kept),
        "seconds": round(time.time() - started, 2),
    }


def albedo(model, codes, verts):
    """SF3D's predicted base colour at each vertex, sRGB in 0..1."""
    import numpy as np
    return np.clip(_decode(model, codes, np.asarray(verts), "features"), 0, 1)


def photo_mapping(mesh, camera, cond):
    """UVs into the original photo and the per-vertex trust in them."""
    import numpy as np
    from scipy.ndimage import minimum_filter

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    u, v, depth = camera.project(verts)
    x, y = cond.to_photo(u, v, camera.px)
    width, height = cond.photo_size
    uv = np.stack([x / width, 1.0 - y / height], -1)

    # Visibility by splatting vertices into a coarse z-buffer. The mesh is dense
    # enough that vertex splats cover it, and the 3x3 minimum closes the gaps.
    zb = ZBUF_PX
    ui = np.clip((u * zb / camera.px).astype(int), 0, zb - 1)
    vi = np.clip((v * zb / camera.px).astype(int), 0, zb - 1)
    zbuf = np.full((zb, zb), np.inf)
    np.minimum.at(zbuf, (vi, ui), depth)
    zbuf = minimum_filter(zbuf, size=3)
    visible = depth <= zbuf[vi, ui] + ZBUF_TOL

    toward = camera.T[None, :] - verts
    toward /= np.linalg.norm(toward, axis=1, keepdims=True)
    facing = (np.asarray(mesh.vertex_normals) * toward).sum(1)
    ramp = np.clip((facing - FACING_LO) / (FACING_HI - FACING_LO), 0.0, 1.0)

    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    px = np.clip(x.astype(int), 0, width - 1)
    py = np.clip(y.astype(int), 0, height - 1)
    matte = cond.alpha[py, px] * inside
    weight = visible * ramp * matte
    return uv, weight, {
        "visible": round(float(visible.mean()), 4),
        "photo_share": round(float((weight > 0.5).mean()), 4),
        "inside_mask": round(float((matte > 0.5).mean()), 4),
    }


# Vertices that fit the colour transform: well inside the photo's trust.
MATCH_MIN_WEIGHT = 0.7
MATCH_MIN_SAMPLES = 400


def match_albedo(albedo, uv, weight, cond):
    """SF3D's albedo carried into the photo's colours, and how far it moved.

    Where the photo sees the figure, both colours are known for the same
    vertex; everywhere else only SF3D's guess is. The guess is often in the
    wrong register -- the black div came back pale grey, so its sides and back
    rendered white against a black front -- but its *shading* is right. So an
    affine colour transform (3x3 plus offset, least squares, sRGB) is fitted on
    the vertices the photo covers and applied to every vertex: the back of the
    div becomes black fur, and the transform is small wherever SF3D was
    already right. Returns ``(albedo, report)``.
    """
    import numpy as np
    rgb = np.asarray(cond.cut.convert("RGB"), dtype=np.float64) / 255.0
    height, width = rgb.shape[:2]
    x = np.clip(uv[:, 0] * width, 0, width - 1).astype(int)
    y = np.clip((1.0 - uv[:, 1]) * height, 0, height - 1).astype(int)
    photo = rgb[y, x]
    trusted = weight > MATCH_MIN_WEIGHT
    report = {"samples": int(trusted.sum())}
    if trusted.sum() < MATCH_MIN_SAMPLES:
        report["skipped"] = "too few vertices where the photo is trusted"
        return albedo, report
    source = np.concatenate([albedo[trusted], np.ones((trusted.sum(), 1))], 1)
    # A little ridge toward the identity, so a figure whose visible side is
    # all one colour cannot collapse the transform onto that colour.
    ridge = 0.02 * np.eye(4)
    ridge[3, 3] = 0.0
    identity = np.vstack([np.eye(3), np.zeros((1, 3))])
    lhs = source.T @ source + ridge * trusted.sum()
    rhs = source.T @ photo[trusted] + ridge @ identity * trusted.sum()
    transform = np.linalg.solve(lhs, rhs)
    full = np.concatenate([albedo, np.ones((len(albedo), 1))], 1) @ transform
    full = np.clip(full, 0.0, 1.0)
    before = float(np.abs(albedo[trusted] - photo[trusted]).mean())
    after = float(np.abs(full[trusted] - photo[trusted]).mean())
    report.update({"error_before": round(before, 4),
                   "error_after": round(after, 4),
                   "mean_shift": [round(float(v), 4) for v in
                                  (full - albedo).mean(0)]})
    return full, report


def joint_from_pixel(mesh, camera, cond, x, y, search_px=24):
    """A photo pixel pushed through the mesh: (entry, midpoint) or None.

    The midpoint of where the ray enters and leaves the figure is where a bone
    belongs -- the middle of the limb, whatever the limb is wearing. The entry
    is the skin, which is what a mouth or an eye wants. A landmark that sits a
    few pixels outside the silhouette (a wrist on the edge of a sleeve) is
    walked inward along a spiral until a ray hits.
    """
    import numpy as np

    triangles = np.asarray(mesh.vertices, dtype=np.float64)[
        np.asarray(mesh.faces)]
    u0, v0 = cond.to_cond(x, y, camera.px)
    k = camera.px / float(cond.size)
    for radius in range(0, int(search_px * k) + 2):
        offsets = [(0, 0)] if radius == 0 else [
            (radius * np.cos(a), radius * np.sin(a))
            for a in np.linspace(0, 2 * np.pi, 8 * radius, endpoint=False)]
        for du, dv in offsets:
            origin, direction = camera.ray(u0 + du, v0 + dv)
            distance = ray_hits(triangles, origin, direction)
            if len(distance) == 0:
                continue
            entry = origin + direction * distance[0]
            exit_ = origin + direction * distance[min(1, len(distance) - 1)]
            return entry, (entry + exit_) / 2.0
    return None


def ray_hits(triangles, origin, direction, eps=1e-9):
    """Sorted distances at which one ray crosses a triangle soup.

    Moller-Trumbore, vectorised over every triangle at once. trimesh's own ray
    queries want ``rtree`` for their broad phase, which is not in this
    environment; a few dozen rays against 270k triangles is a few milliseconds
    each without one.
    """
    import numpy as np
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(direction, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > eps
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origin - v0
    u = (s * p).sum(1) * inv
    q = np.cross(s, e1)
    v = (q * direction).sum(1) * inv
    t = (e2 * q).sum(1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps)
    return np.sort(t[hit])
