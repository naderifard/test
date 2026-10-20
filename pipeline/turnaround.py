"""A turnaround sheet becomes separate views, matched to the front image.

The seed prompt asks for each character's side and back on one canvas, made by
editing the front image (people: left profile | back, halves; animals: side
profile | rear, two thirds and a third). This module finds the figures on the
sheet -- by the matte, not by assuming the layout, because an image model
never quite keeps to it -- and returns each as its own image, cut out, on
the backdrop, and scaled so the figure is the same height in pixels as in the
front image and stands on the same ground line. A multi-view reconstruction
needs exactly that: every view the same size, the same framing.

``views(sheet, front, kind)`` returns ``({"left": img, "back": img}, report)``.
The report says how well each view agrees with the front (height ratio before
scaling, colour distance) so a sheet that drifted too far is flagged rather
than silently used.
"""

from __future__ import annotations

HUMAN_KINDS = ("human", "humanoid")
MIN_FIGURE_SHARE = 0.02
MAX_HEIGHT_DRIFT = 0.15         # a view's figure this much taller/shorter warns
MAX_COLOUR_DRIFT = 18.0         # Lab distance of mean colours that warns
BACKDROP = (128, 128, 128)      # neutral grey behind every extracted view


def _figures(alpha):
    """Bounding boxes of the large foreground blobs, left to right."""
    import cv2
    import numpy as np
    fg = (np.asarray(alpha) > 127).astype(np.uint8)
    # Close small gaps (a tail, a claw) so a figure is one blob.
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    h, w = fg.shape
    boxes = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area >= MIN_FIGURE_SHARE * h * w:
            boxes.append((int(x), int(y), int(x + bw), int(y + bh), int(area)))
    return sorted(boxes, key=lambda b: b[0])


def _crop(rgb, alpha, box):
    import numpy as np
    x1, y1, x2, y2 = box[:4]
    return rgb[y1:y2, x1:x2], np.asarray(alpha)[y1:y2, x1:x2]


def _place(rgb, alpha, height_px, ground_y, canvas_hw):
    """A cut-out figure scaled to ``height_px`` and stood on ``ground_y``,
    centred on a grey canvas the size of the front image."""
    import cv2
    import numpy as np
    H, W = canvas_hw
    h, w = alpha.shape
    scale = height_px / float(max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # A long view (a lion in profile) is wider than the front's canvas; the
    # canvas widens rather than crop the animal's head or tail.
    W = max(W, int(nw * 1.1))
    canvas = np.full((H, W, 3), BACKDROP, np.float32)
    mask = np.zeros((H, W), np.float32)
    top = int(round(ground_y - nh))
    left = int(round(W / 2.0 - nw / 2.0))
    ys, xs = max(0, top), max(0, left)
    ye, xe = min(H, top + nh), min(W, left + nw)
    if ye <= ys or xe <= xs:
        return canvas.astype(np.uint8), mask
    sub_rgb = rgb[ys - top:ye - top, xs - left:xe - left].astype(np.float32)
    sub_a = a[ys - top:ye - top, xs - left:xe - left].astype(np.float32) / 255.
    canvas[ys:ye, xs:xe] = (sub_rgb * sub_a[..., None]
                            + canvas[ys:ye, xs:xe] * (1 - sub_a[..., None]))
    mask[ys:ye, xs:xe] = sub_a
    return canvas.astype(np.uint8), mask


def _mean_lab(rgb, alpha):
    import cv2
    import numpy as np
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
    a = (np.asarray(alpha).reshape(-1) > 127)
    return lab[a].mean(0) if a.any() else lab.mean(0)


def views(sheet, front, kind="human"):
    """Split ``sheet`` (PIL) into views matched to ``front`` (PIL)."""
    import numpy as np
    from PIL import Image

    from pipeline import matte

    front_rgb = np.asarray(front.convert("RGB"))
    front_a, _ = matte.figure(front)
    fy, fx = np.nonzero(np.asarray(front_a) > 127)
    height = float(fy.max() - fy.min())
    ground = float(fy.max())

    sheet_rgb = np.asarray(sheet.convert("RGB"))
    sheet_a, method = matte.figure(sheet)
    boxes = _figures(sheet_a)
    report = {"matte": method, "figures_found": len(boxes)}
    if len(boxes) < 2:
        raise ValueError(f"expected two figures on the turnaround sheet, "
                         f"found {len(boxes)}")
    # The two largest, still left to right.
    boxes = sorted(sorted(boxes, key=lambda b: -b[4])[:2], key=lambda b: b[0])
    names = ("left", "back")
    out = {}
    front_lab = _mean_lab(front_rgb, front_a)
    for name, box in zip(names, boxes):
        rgb, a = _crop(sheet_rgb, sheet_a, box)
        ratio = (box[3] - box[1]) / max(1.0, boxes[0][3] - boxes[0][1])
        image, mask = _place(rgb, a, height, ground, front_rgb.shape[:2])
        drift = float(np.linalg.norm(_mean_lab(rgb, a) - front_lab))
        report[name] = {"box": list(box[:4]),
                        "height_vs_first": round(ratio, 3),
                        "colour_drift": round(drift, 1)}
        rgba = np.dstack([image, (mask * 255).astype(np.uint8)])
        out[name] = Image.fromarray(rgba, "RGBA")
    warnings = []
    for name in names:
        r = report[name]
        if abs(r["height_vs_first"] - 1.0) > MAX_HEIGHT_DRIFT and name != "left":
            warnings.append(f"the {name} view is {r['height_vs_first']:.0%} "
                            "the height of the side view")
        if r["colour_drift"] > MAX_COLOUR_DRIFT:
            warnings.append(f"the {name} view's colours differ from the front "
                            f"(distance {r['colour_drift']:.0f})")
    report["warnings"] = warnings
    return out, report


def sheet_path(folder):
    """The turnaround sheet in a character's refs folder, or None.

    Any image in the folder that is not the front and not a derived file is
    taken as the sheet; the newest if there are several.
    """
    import os
    names = []
    for name in os.listdir(folder):
        low = name.lower()
        if not low.endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        stem = os.path.splitext(low)[0]
        if stem.endswith("_front") or stem.endswith(("_left", "_back",
                                                      "_right")):
            continue
        names.append(name)
    if not names:
        return None
    names.sort(key=lambda n: os.path.getmtime(os.path.join(folder, n)))
    return os.path.join(folder, names[-1])
