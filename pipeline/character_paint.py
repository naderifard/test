"""A character's texture painted by a video model turning round its own shape.

Projecting the front photo and the turnaround sheet onto the model painted
some features twice -- two eyes, then two ears -- because the three images
are separate drawings that disagree about where things are, and the model is
a compromise between them. No blending rule fixes that; consistent colour
has to come from something that sees the shape itself.

So the character's own shape is rendered turning a full circle as a depth
video (``blender/paint_turntable.py``), and Wan2.1-VACE paints that turn:
guided frame by frame by the depth, holding the front photo fixed as the
first and last frame, and given the photo as its reference. A video model
keeps a subject consistent from frame to frame, and the depth keeps every
frame on the geometry, so a feature exists once and turns with the shape.
Those frames are projected back onto the character's texture -- each texel
from the frames that see it squarely and unoccluded -- with the full-
resolution front photo over the front, where it is the best source there is.

Steps (``paint``): depth frames -> VACE -> projection into a texture baked
over position and normal maps from Blender.
"""

from __future__ import annotations

import json
import os

DEPTH_NEAR_VALUE, DEPTH_FAR_VALUE = 1.0, 0.2   # annotator-style, near bright
FACING_POWER = 4.0
DEPTH_TOLERANCE_M = 0.02
FRONT_POWER = 2.0
FRONT_WEIGHT = 4.0
UNSEEN_WEIGHT = 1e-3
FILL_NEIGHBOURS = 32
FILL_MIN_AGREE = 0.3


def _exr(path):
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def depth_video(turn_dir, camera):
    """Annotator-style depth frames (3, F, H, W) in [-1, 1] and the raw
    distances (F, H, W) in metres (inf on the background)."""
    import numpy as np
    import torch
    frames, raw = [], []
    for i in range(camera["frames"]):
        d = _exr(os.path.join(turn_dir, "depth_%04d.exr" % i))[..., 0]
        fg = d < 0.9995
        metres = np.where(fg, d * camera["depth_exr_scale"], np.inf)
        raw.append(metres)
        img = np.zeros_like(d)
        if fg.any():
            near, far = d[fg].min(), d[fg].max()
            t = (far - d) / max(1e-6, far - near)
            img = np.where(fg, DEPTH_FAR_VALUE + (DEPTH_NEAR_VALUE -
                                                  DEPTH_FAR_VALUE) * t, 0.0)
        frames.append(img)
    video = torch.from_numpy(np.stack(frames)).float()        # F, H, W
    video = video[None].repeat(3, 1, 1, 1) * 2.0 - 1.0          # 3, F, H, W
    return video, np.stack(raw)


def _image_tensor(path, size=None):
    import numpy as np
    import torch
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)             # 3, H, W


SOLID_KEEP = 0.5


def solid_mask(render_dir, size=None):
    """How much of each front-frame pixel is the photo itself (1) rather than
    the model's albedo showing through (0), from ``solid_0000.png``; None if
    the turntable did not render one. ``size`` resamples to (w, h)."""
    import numpy as np
    from PIL import Image
    path = os.path.join(render_dir, "solid_0000.png")
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("L")
    if size is not None and img.size != tuple(size):
        img = img.resize(tuple(size), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float64) / 255.0) ** 2.2   # sRGB out


def reference_image(photo_rgba_path, size, out_dir):
    """The photo's cut-out centred on white, the canvas VACE expects, saved
    in ``out_dir`` (not beside the photo, which is another stage's output)."""
    import torch
    from PIL import Image
    cut = Image.open(photo_rgba_path).convert("RGBA")
    box = cut.getchannel("A").point(lambda a: 255 if a > 127 else 0).getbbox()
    cut = cut.crop(box)
    w, h = size
    scale = min(w / cut.width, h / cut.height) * 0.92
    cut = cut.resize((max(1, int(cut.width * scale)),
                      max(1, int(cut.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGBA", size, (255, 255, 255, 255))
    canvas.alpha_composite(cut, ((w - cut.width) // 2, (h - cut.height) // 2))
    path = os.path.join(out_dir, "reference.png")
    canvas.convert("RGB").save(path)
    return _image_tensor(path)[:, None]                        # 3, 1, H, W


def generate_turn(turn_dir, prompt, photo_path, steps=25, seed=42):
    """VACE over the depth turntable, the front frame kept at both ends."""
    import numpy as np
    import torch

    from pipeline import wan_vace
    camera = json.load(open(os.path.join(turn_dir, "camera.json")))
    size = (camera["width"], camera["height"])
    video, _raw = depth_video(turn_dir, camera)
    masks = torch.ones((1,) + tuple(video.shape[1:]))
    front = _image_tensor(os.path.join(turn_dir, "colour_0000.png"), size)
    solid = solid_mask(turn_dir, size)
    keep = (torch.ones(front.shape[1:]) if solid is None else
            torch.from_numpy(solid > SOLID_KEEP).float())
    for i in (0, video.shape[1] - 1):
        # Kept where the photo is; painted over the depth where it is not.
        video[:, i] = front * keep + video[:, i] * (1 - keep)
        masks[:, i] = 1.0 - keep
    ref = reference_image(photo_path, size, turn_dir)
    out = wan_vace.generate(prompt, video, masks, [ref], size=size,
                            steps=steps, seed=seed)
    frames = ((out.float().clamp(-1, 1) + 1.0) * 127.5).round().byte()
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()        # F, H, W, 3
    from PIL import Image
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(turn_dir, "paint_%04d.png" % i))
    return frames


def _camera_basis(camera, i):
    import numpy as np
    a = np.radians(camera["azimuth_deg"][i])
    direction = np.array([np.sin(a), -np.cos(a), 0.0])       # camera looks
    right = np.array([-np.cos(a), -np.sin(a), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    centre = np.asarray(camera["centre"], dtype=np.float64)
    position = centre - direction * camera["distance"]
    return position, direction, right, up


def project(camera, i, points):
    """World points -> (x, y) pixels and depth along the view, frame ``i``."""
    import numpy as np
    position, direction, right, up = _camera_basis(camera, i)
    k = camera["height"] / camera["ortho_scale"]
    rel = points - np.asarray(camera["centre"])
    x = camera["width"] / 2.0 + (rel @ right) * k
    y = camera["height"] / 2.0 - (rel @ up) * k
    depth = (points - position) @ direction
    return x, y, depth


EDGE_PX = 3
EDGE_STEP_M = 0.05


def _reliable(depth):
    """Pixels a projection can trust: on the figure, and not within
    ``EDGE_PX`` of a depth step (a silhouette, an arm in front of the
    body). At those edges a pixel's colour belongs to whatever is in front
    or behind, and projecting it painted dark streaks down the inner arms."""
    import numpy as np
    from scipy.ndimage import binary_dilation, maximum_filter, minimum_filter
    fg = np.isfinite(depth)
    d = np.where(fg, depth, 1e3)
    span = maximum_filter(d, size=3) - minimum_filter(d, size=3)
    edge = binary_dilation(span > EDGE_STEP_M, iterations=EDGE_PX)
    return fg & ~edge


BACKGROUND_DISTANCE = 14.0
BACKGROUND_ERODE_PX = 1


def _painted_figure(frame):
    """Pixels where Wan painted the figure, not its plain backdrop.

    Wan follows the depth closely but not to the pixel, and a thin part --
    a finger -- can sit a pixel or two off: the geometry's fingertip then
    reads the grey behind Wan's finger, and every fingertip came out grey.
    The backdrop's colour is the frame border's median; anything that close
    to it, and connected to the border, is backdrop."""
    import numpy as np
    from scipy.ndimage import binary_dilation, label
    f = frame.astype(np.float32)
    border = np.concatenate([f[0], f[-1], f[:, 0], f[:, -1]])
    bg = np.median(border, axis=0)
    near = np.linalg.norm(f - bg, axis=2) < BACKGROUND_DISTANCE
    # Only backdrop reachable from the border: a grey tunic inside the
    # figure is not backdrop.
    lab, _n = label(near)
    edge_labels = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0],
                                            lab[:, -1]]))
    backdrop = np.isin(lab, edge_labels[edge_labels > 0])
    backdrop = binary_dilation(backdrop, iterations=BACKGROUND_ERODE_PX)
    return ~backdrop


CONSENSUS_SIGMA = 28.0
CONSENSUS_ROUNDS = 3


def consensus(samples, weights, chunk=200000):
    """Each texel's colour as the views that see it agree it is.

    ``samples``: (N, F, 3) uint8, what frame f shows at texel n; ``weights``:
    (N, F), how squarely it shows it. Wan's frames follow the depth closely
    but not to the pixel, and a frame whose arm sits a few pixels off the
    geometry puts the tunic's spots on the arm -- an average of the views
    kept those as a dark mark. So the colour starts from the weighted median
    (by brightness) and each view's say is cut the further it is from the
    others' colour, a few rounds; one misaligned view is outvoted."""
    import numpy as np
    out = np.zeros((len(samples), 3))
    luma = np.array([0.299, 0.587, 0.114], np.float32)
    for s in range(0, len(samples), chunk):
        rgb = samples[s:s + chunk].astype(np.float32)
        w = weights[s:s + chunk]
        order = np.argsort(rgb @ luma, axis=1)
        cum = np.cumsum(np.take_along_axis(w, order, 1), axis=1)
        half = cum[:, -1:] * 0.5
        pick = np.take_along_axis(order, (cum < half).sum(1, keepdims=True)
                                  .clip(0, rgb.shape[1] - 1), 1)[:, 0]
        centre = rgb[np.arange(len(rgb)), pick]
        for _ in range(CONSENSUS_ROUNDS):
            d = np.linalg.norm(rgb - centre[:, None], axis=2)
            ww = w / (1.0 + (d / CONSENSUS_SIGMA) ** 2)
            total = ww.sum(1, keepdims=True)
            centre = np.where(total > 1e-9,
                              (rgb * ww[..., None]).sum(1) /
                              np.maximum(total, 1e-9), centre)
        out[s:s + chunk] = centre
    return out


HAND_FROM, HAND_TO = 0.3, 0.7          # wrist weight: forearm .. hand


def _fill(p, n, colour, targets, sources):
    """``targets`` take the colour of the nearest ``sources`` in 3D (not in
    the texture, which would cross UV islands to an unrelated part), only
    from neighbours facing the same way: the inner arm, hidden from every
    horizontal angle, is nearest to the tunic beside it and took its dark
    armhole as a streak down the arm. In place."""
    import numpy as np
    if not targets.any() or not sources.any():
        return
    from scipy.spatial import cKDTree
    seen_p, seen_n, seen_c = p[sources], n[sources], colour[sources]
    k = min(FILL_NEIGHBOURS, len(seen_p))
    dist, near = cKDTree(seen_p).query(p[targets], k=k)
    if k == 1:
        dist, near = dist[:, None], near[:, None]
    agree = np.einsum("ij,ikj->ik", n[targets], seen_n[near])
    w = np.clip(agree - FILL_MIN_AGREE, 0.0, None) / (dist + 1e-3)
    fallback = w.sum(1) < 1e-9
    w[fallback] = 1.0 / (dist[fallback] + 1e-3)
    colour[targets] = (seen_c[near] * w[..., None]).sum(1) \
        / w.sum(1, keepdims=True)
FRONT_VIEW_DEG = 25.0


def hand_share(weight):
    """0 on the forearm, 1 on the hand, from the baked wrist weight."""
    import numpy as np
    return np.clip((np.asarray(weight, np.float64) - HAND_FROM)
                   / (HAND_TO - HAND_FROM), 0.0, 1.0)


def front_view(azimuth_deg):
    """1 for a frame within ``FRONT_VIEW_DEG`` of the front, easing to 0 by
    half as much again. A hand is too small in Wan's frames for the depth to
    say which way it faces, and Wan drew an open palm toward the camera from
    behind as well: the back of the hand came out as a second palm. Hands
    are coloured from the front views -- the photo's side -- only, and what
    those cannot see is filled from the forearm beside it."""
    a =abs((azimuth_deg + 180.0) % 360.0 - 180.0)
    return min(1.0, max(0.0, (1.5 * FRONT_VIEW_DEG - a)
                        / (0.5 * FRONT_VIEW_DEG)))


HEAD_FROM, HEAD_TO = 0.3, 0.7          # head weight: neck .. head
HEAD_PHOTO_FROM, HEAD_PHOTO_TO = 0.4, 0.65   # facing: photo hands over
HEAD_WINNER_POWER = 15.0


def head_share(weight):
    """0 below the neck, 1 on the head, from the baked head weight."""
    import numpy as np
    return np.clip((np.asarray(weight, np.float64) - HEAD_FROM)
                   / (HEAD_TO - HEAD_FROM), 0.0, 1.0)


def single_source(weights, share, power=HEAD_WINNER_POWER):
    """Frame weights sharpened toward the one frame that sees a texel most
    squarely, by ``share`` (0 leaves them, 1 is close to winner-takes-all).

    A face is forty pixels in Wan's frames and each frame puts the eyes a
    pixel or two elsewhere: averaged, the near-front frames stacked shifted
    copies of the same eye -- doubled eyes and brows that no outlier vote
    removes, since every frame disagrees differently. One frame's eye is
    one eye."""
    import numpy as np
    exponent = 1.0 + power * np.asarray(share, np.float64)[:, None]
    return np.power(np.clip(weights, 0.0, 1.0), exponent)


def bake_texture(turn_dir, frames, position, normal, valid, front_path=None,
                 front_scale=1, hand=None, head=None):
    """Every valid texel coloured from the frames that see it squarely.

    ``position``/``normal``: (H, W, 3) world-space maps from Blender's bake;
    ``valid``: (H, W) bool. ``hand``: (H, W) baked wrist weight, or None
    (an animal); see ``front_view``. ``head``: (H, W) baked head weight, or
    None: the head is painted from one source at a time, never an average --
    the photo wherever it sees the face, else the single best Wan frame
    (``single_source``). ``front_path``: a higher-resolution render of
    the front (frame 0's camera, ``front_scale`` times the size), laid over
    the front with a steeper weight. Returns (H, W, 3) uint8 and a report.
    """
    import numpy as np
    from PIL import Image
    camera = json.load(open(os.path.join(turn_dir, "camera.json")))
    _video, raw = depth_video(turn_dir, camera)
    idx = np.nonzero(valid)
    p = position[idx].astype(np.float64)
    n = normal[idx].astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    H, W = camera["height"], camera["width"]
    reliable = [_reliable(r) & _painted_figure(f)
                for r, f in zip(raw, frames)]

    def sample(i):
        x, y, depth = project(camera, i, p)
        xi = np.clip(np.round(x).astype(int), 0, W - 1)
        yi = np.clip(np.round(y).astype(int), 0, H - 1)
        seen = (np.abs(raw[i][yi, xi] - depth) < DEPTH_TOLERANCE_M)             & reliable[i][yi, xi]
        _pos, direction, _r, _u = _camera_basis(camera, i)
        facing = np.clip(-(n @ direction), 0.0, 1.0)
        rgb = frames[i][yi, xi].astype(np.float64)
        return rgb, seen * facing ** FACING_POWER

    count = len(frames) - 1                    # the last repeats frame 0
    samples = np.zeros((len(p), count, 3), np.uint8)
    weights = np.zeros((len(p), count), np.float32)
    in_hand = None if hand is None else hand_share(hand[idx])
    for i in range(count):
        rgb, w = sample(i)
        if in_hand is not None:
            w = w * (1.0 - in_hand * (1.0 - front_view(
                camera["azimuth_deg"][i])))
        samples[:, i] = np.clip(rgb, 0, 255).astype(np.uint8)
        weights[:, i] = w
    weight = weights.sum(1).astype(np.float64)
    in_head = None if head is None else head_share(head[idx])
    if in_head is not None:
        weights = single_source(weights, in_head).astype(np.float32)
    colour = consensus(samples, weights)
    report = {"texels": int(len(p)),
              "unseen_share": round(float((weight < 1e-3).mean()), 4)}
    if front_path:
        front = np.asarray(Image.open(front_path).convert("RGB"),
                           dtype=np.float64)
        x, y, depth = project(camera, 0, p)
        xs = np.clip(np.round(x * front_scale).astype(int), 0,
                     front.shape[1] - 1)
        ys = np.clip(np.round(y * front_scale).astype(int), 0,
                     front.shape[0] - 1)
        xi = np.clip(np.round(x).astype(int), 0, W - 1)
        yi = np.clip(np.round(y).astype(int), 0, H - 1)
        seen = np.abs(raw[0][yi, xi] - depth) < DEPTH_TOLERANCE_M
        _pos, direction, _r, _u = _camera_basis(camera, 0)
        facing = np.clip(-(n @ direction), 0.0, 1.0)
        # A wide, gentle hand-over: the photo's face and the painted turn's
        # differ a little, and a narrow one drew a line down the cheek.
        fw = seen * np.clip((facing - 0.35) / 0.6, 0.0, 1.0) ** FRONT_POWER
        if in_head is not None:
            # The face is the photo's wherever the photo sees it at all
            # squarely: the mesh was built from this photo, so its eyes and
            # mouth sit exactly on the geometry, once. The hand-over to Wan
            # falls at the side of the head, in the hair and the ear.
            on_face = seen * np.clip((facing - HEAD_PHOTO_FROM)
                                     / (HEAD_PHOTO_TO - HEAD_PHOTO_FROM),
                                     0.0, 1.0)
            fw = np.maximum(fw, in_head * on_face)
        solid = solid_mask(os.path.dirname(front_path))
        if solid is not None:
            fw = fw * np.clip((solid[ys, xs] - SOLID_KEEP) / (1 - SOLID_KEEP),
                              0.0, 1.0)
        colour = colour * (1 - fw[:, None]) + front[ys, xs] * fw[:, None]
        report["front_share"] = round(float((fw > 0.5).mean()), 4)
    # Texels no frame saw (under the arms, the crown's top, the back of a
    # hand) take the colour of the nearest seen point on the body.
    unseen = weight < UNSEEN_WEIGHT
    if in_hand is None:
        _fill(p, n, colour, unseen, ~unseen)
    else:
        # A hand hangs beside the hip, and the nearest seen surface to the
        # back of it is the tunic -- which also carries a little wrist
        # weight: the back of a hand fills only from the seen hand itself.
        _fill(p, n, colour, unseen & (in_hand <= 0.5), ~unseen)
        _fill(p, n, colour, unseen & (in_hand > 0.5),
              ~unseen & (in_hand > 0.5))
    out = np.zeros(valid.shape + (3,), np.uint8)
    out[idx] = np.clip(colour, 0, 255).astype(np.uint8)
    weight_map = np.zeros(valid.shape, np.float64)
    weight_map[idx] = weight
    return out, weight_map, report


def dilate(texture, valid, pixels=8):
    """Colour spread a few pixels past every UV island, so filtering and
    mip-mapping at an island's edge never fetch the black between them."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    dist, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    out = texture.copy()
    grow = (~valid) & (dist <= pixels)
    out[grow] = texture[iy[grow], ix[grow]]
    return out
