"""A scene image becomes a place a camera can stand in: depth, then metres.

The user generates one wide image per scene. Before this module the pipeline
hung that image on a flat card behind grey bands and scattered icospheres in
front of it. Here it becomes the set itself: Depth-Anything-V2 estimates depth
for every pixel, the depth is turned into metres, and the Blender step displaces
a grid to those depths with the image projected on it. Seen from where the image
was taken it *is* the image; seen from a camera a few metres to the side it has
real parallax, real occlusion and a ground the characters can stand on.

**From relative depth to metres.** Depth-Anything-V2 returns disparity -- large
near, small far -- up to an unknown scale. One assumption fixes it, and the seed
prompt asks the image generator for exactly that assumption: an eye-level camera
over roughly level ground. For such a camera, ground disparity falls linearly
with image row and reaches zero at the horizon, so a straight line fitted to the
disparity of the ground rows at the bottom of the image gives both the horizon
row and the scale: a ground point at row ``v`` is ``f * h / (v - v_h)`` metres
away, with ``h`` the camera height. Every other pixel follows from its disparity
with the same constant. Sky (disparity near zero) is sent to a far dome.

**The world frame the rest of the pipeline expects.** Shots put the cast near
the origin and cameras on -Y looking +Y. So the origin is placed on the ground
at the image's lower-middle -- the natural stage, in front of the reference
camera -- and the reference camera sits behind it on -Y at eye height. Shot
cameras that stay near that line of sight see the image as it was painted.

Runs in the ``ml`` environment (torch and transformers). No bpy.
"""

from __future__ import annotations

import math

MODEL_REPO = "depth-anything/Depth-Anything-V2-Small-hf"

# Assumptions about the camera that took the scene image, which the seed prompt
# asks the image generator to honour ("eye level, horizon level, wide").
HFOV_DEG = 60.0
CAMERA_HEIGHT_M = 1.6
# How far the lower-middle "stage" point may be from the reference camera. The
# ground row used is STAGE_ROW of the image height, clamped to these metres.
STAGE_ROW = 0.80
STAGE_MIN_M, STAGE_MAX_M = 4.0, 14.0
# Rows of the image used to fit the ground line, as fractions from the top.
GROUND_ROWS = (0.72, 0.97)
# Where the horizon may plausibly be, as fractions of the height.
HORIZON_LIMITS = (0.10, 0.70)
FAR_M = 260.0
NEAR_M = 0.6
# The grid is built past the image's edges so a camera that pans a little still
# has set to look at. The image is *mirrored* into that margin, depth and colour
# alike: clamping the border pixels instead smeared them into long stripes that
# no one would mistake for terrain.
MARGIN = 0.22
GRID_COLUMNS = 360
# A grid cell whose corners differ in depth by more than this ratio spans an
# occlusion edge -- a tree in front of the sky -- and is torn out. Kept, it is a
# rubber sheet from the tree to a far dome 260 m away, which any camera off the
# reference line sees as a pale translucent wedge.
#
# 2.5 left chains of thin slivers: the depth model blurs an occlusion edge, so
# a branch against the sky steps from 12 m to 260 m over several cells, none of
# which crosses 2.5. 1.6 breaks the chain. It also tears grazing ground near
# the horizon, which is why the backdrop layer below exists: every hole looks
# through to the same image, far away, instead of to an empty world.
TEAR_RATIO = 1.6
# The backdrop: the whole image (with its margins) on a surface this far away.
BACKDROP_M = 240.0

_MODEL = None


def estimate(image):
    """Depth-Anything-V2 disparity for a PIL image, at the image's own size."""
    import numpy as np
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    global _MODEL
    if _MODEL is None:
        processor = AutoImageProcessor.from_pretrained(MODEL_REPO)
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_REPO)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = (processor, model.to(device).eval(), device)
    processor, model, device = _MODEL
    rgb = image.convert("RGB")
    with torch.no_grad():
        inputs = processor(images=rgb, return_tensors="pt").to(device)
        predicted = model(**inputs).predicted_depth
        resized = torch.nn.functional.interpolate(
            predicted.unsqueeze(1), size=(rgb.height, rgb.width),
            mode="bicubic", align_corners=False)[0, 0]
    return resized.float().cpu().numpy().astype(np.float32)


def _mirror(coord, size):
    """Reflect coordinates outside ``[0, size)`` back into the image."""
    import numpy as np
    period = 2.0 * size
    folded = np.mod(coord, period)
    return np.where(folded < size, folded, period - folded - 1e-3)


def fit_ground(disparity):
    """Horizon row and disparity-per-row slope from the bottom of the image.

    Returns ``(v_h, slope, report)``. The median over the central columns of
    each row is used so a rock or a tree in the foreground cannot tilt the line.
    """
    import numpy as np
    height, width = disparity.shape
    rows = np.arange(int(GROUND_ROWS[0] * height), int(GROUND_ROWS[1] * height))
    centre = disparity[:, int(0.2 * width):int(0.8 * width)]
    medians = np.median(centre[rows], axis=1)
    slope, intercept = np.polyfit(rows.astype(np.float64), medians, 1)
    report = {"fit_rows": [int(rows[0]), int(rows[-1])],
              "slope": float(slope), "intercept": float(intercept)}
    if slope <= 0:
        # Disparity does not grow toward the bottom: not an eye-level ground.
        v_h = HORIZON_LIMITS[0] * height + 0.3 * height
        report["fallback"] = "ground disparity does not rise toward the viewer"
        slope = float(np.median(medians)) / max(1.0, height - v_h)
    else:
        v_h = -intercept / slope
    low, high = HORIZON_LIMITS[0] * height, HORIZON_LIMITS[1] * height
    if not low <= v_h <= high:
        report["horizon_clamped_from"] = float(v_h)
        v_h = min(max(v_h, low), high)
    return float(v_h), float(slope), report


def build(disparity, hfov_deg=HFOV_DEG, cam_h=CAMERA_HEIGHT_M,
          columns=GRID_COLUMNS, margin=MARGIN):
    """Grid vertices in world metres, their UVs, and the camera that saw them.

    Returns a dict of numpy arrays and numbers; the Blender step only has to
    make faces over the grid.
    """
    import numpy as np
    height, width = disparity.shape
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    cx = width / 2.0
    v_h, slope, report = fit_ground(disparity)
    # 1/Z = disparity / (slope * f * h): the constant that makes the fitted
    # ground rows land on z = 0.
    k = slope * f * cam_h

    rows = max(8, int(round(columns * height / float(width))))
    us = np.linspace(-margin * width, (1 + margin) * width, columns)
    vs = np.linspace(-margin * height, (1 + margin) * height, rows)
    uu, vv = np.meshgrid(us, vs)
    mu, mv = _mirror(uu, width), _mirror(vv, height)
    ui = np.clip(mu, 0, width - 1).astype(int)
    vi = np.clip(mv, 0, height - 1).astype(int)
    d = disparity[vi, ui].astype(np.float64)
    sky = d <= 1e-6
    depth = np.where(sky, FAR_M, k / np.maximum(d, 1e-6))
    depth = np.clip(depth, NEAR_M, FAR_M)
    # A pixel below the horizon cannot be further than the ground it looks at
    # would be, or the grid folds under the stage; clamp it to that.
    below = vv > v_h + 1.0
    ground_limit = np.where(below, f * cam_h / np.maximum(vv - v_h, 1e-3), FAR_M)
    depth = np.minimum(depth, np.maximum(ground_limit, NEAR_M))

    # Camera space: x right, y forward, z up; level camera, horizon at v_h.
    x = (uu - cx) / f * depth
    y = depth
    z = cam_h - (vv - v_h) / f * depth

    stage_v = STAGE_ROW * height
    stage_m = f * cam_h / max(stage_v - v_h, 1.0)
    stage_m = min(max(stage_m, STAGE_MIN_M), STAGE_MAX_M)
    verts = np.stack([x, y - stage_m, z], -1).reshape(-1, 3)
    uv = np.stack([mu / width, 1.0 - mv / height], -1).reshape(-1, 2)

    corners = np.stack([depth[:-1, :-1], depth[:-1, 1:], depth[1:, 1:],
                        depth[1:, :-1]], -1)
    keep = corners.max(-1) / corners.min(-1) <= TEAR_RATIO

    # Every grid ray pushed out to the backdrop distance: the same image, far
    # away, behind everything the tears expose.
    far = BACKDROP_M / np.maximum(y, 1e-6)
    backdrop = np.stack([x * far, y * far - stage_m,
                         cam_h + (z - cam_h) * far], -1).reshape(-1, 3)

    ground = (vv > v_h) & ~sky
    return {
        "backdrop": backdrop.astype(np.float32),
        "verts": verts.astype(np.float32),
        "uv": uv.astype(np.float32),
        "keep": keep,
        "torn_fraction": float(1.0 - keep.mean()),
        "grid": [int(rows), int(columns)],
        "camera": {"position_m": [0.0, -stage_m, cam_h],
                   "hfov_deg": hfov_deg,
                   "focal_px": f,
                   "horizon_row": v_h,
                   "horizon_fraction": v_h / height,
                   "image_px": [int(width), int(height)]},
        "stage_distance_m": stage_m,
        "sky_fraction": float(sky.mean()),
        "ground_fraction": float(ground.mean()),
        "depth_range_m": [float(depth.min()), float(depth.max())],
        "fit": report,
    }
