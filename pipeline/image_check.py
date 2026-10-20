"""Is this character image good to build from? Checked before an hour's build.

A character image is reconstructed into a rigged, textured model, and each
defect of the image comes back later as a defect of the film: arms touching the
body become fused, sausage-like arms; a low camera becomes long legs and a
small head; hard sunlight becomes shadows painted into the skin; a busy or
same-coloured background becomes a ragged outline. All of these can be seen
in the image itself, so the Characters tab runs this the moment an image is
attached and says what to regenerate, in plain words.

It runs in the ``ml`` environment (rembg for the matte, BlazePose ONNX for the
joints) as ``python -m pipeline.image_check <image> [--kind human]`` and prints
one JSON object: ``{"verdict": "good"|"usable"|"poor", "findings": [{"level":
"ok"|"warn"|"fail", "check", "message"}], "metrics": {...}}``.

Every threshold was set on the Keyumars images, which failed in exactly these
ways, against a good A-pose reference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

HUMAN_KINDS = ("human", "humanoid")

MIN_SIDE_PX = 768
BORDER_SHARE = 0.05
BUSY_BACKGROUND_STD = 0.07        # luminance spread over the border, 0..1
MIN_OUTLINE_DE = 15.0             # colour difference across the outline (Lab)
DARK_SUBJECT = 0.18               # mean luminance of the figure, 0..1
MIN_FILL, MAX_FILL = 0.55, 0.97   # subject height / image height
EDGE_TOUCH_PX = 0                 # touching the edge is cut off
EDGE_MARGIN = 0.008               # closer than this is a tight margin
VERSION = 3                       # cached results older than this are redone
ARM_MIN_DEG = 14.0                # upper arm from vertical
LEG_TORSO_HIGH = 2.15             # hip-ankle over shoulder-hip: camera below
LEG_TORSO_LOW = 1.25              # camera above
MIN_EYE_GAP_SHARE = 0.022         # eye distance over image width
SIDE_LIGHT = 0.16                 # luminance difference, left half vs right
BLOWN_SHARE = 0.05


def _finding(level, check, message):
    return {"level": level, "check": check, "message": message}


def _matte(rgb):
    """Foreground probability 0..1 per pixel, the way c_asset will cut it."""
    import numpy as np
    from PIL import Image

    from pipeline import matte, sf3d_geometry
    alpha, _method = matte.figure(Image.fromarray(rgb),
                                  sf3d_geometry.new_rembg_session())
    return np.asarray(alpha, dtype=np.float32) / 255.0


def _luma(rgb):
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]) / 255.0


def _gap_between(matte, y, xa, xb):
    """Whether any background lies on row ``y`` strictly between two xs."""
    lo, hi = sorted((int(round(xa)), int(round(xb))))
    if hi - lo < 3 or not 0 <= y < matte.shape[0]:
        return None
    row = matte[int(y), lo + 1:hi]
    return bool((row < 0.5).any())


def check(path, kind="human", estimator=None):
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    rgb = np.asarray(image)
    h, w = rgb.shape[:2]
    out, metrics = [], {"size_px": [w, h]}

    if min(w, h) < MIN_SIDE_PX:
        out.append(_finding("warn", "size",
                            f"the image is small ({w}x{h}); generate at 1024 "
                            "px or more on the short side for a sharper model"))

    from pipeline import matte as matte_mod
    try:
        matte = _matte(rgb)
    except matte_mod.MatteError as exc:
        out.append(_finding("fail", "subject", str(exc)))
        return _result(out, metrics)
    fg = matte > 0.5
    if fg.sum() < 0.02 * w * h:
        out.append(_finding("fail", "subject",
                            "no clear figure was found against the background"))
        return _result(out, metrics)
    luma = _luma(rgb.astype(np.float32))

    # -- background ----------------------------------------------------------
    b = max(2, int(BORDER_SHARE * min(w, h)))
    border = np.zeros_like(fg)
    border[:b, :] = border[-b:, :] = True
    border[:, :b] = border[:, -b:] = True
    background = border & ~fg
    spread = float(luma[background].std()) if background.any() else 0.0
    metrics["background_spread"] = round(spread, 3)
    if spread > BUSY_BACKGROUND_STD:
        out.append(_finding(
            "warn", "background",
            "the background is busy (a landscape, a pattern or a painted "
            "backdrop); a plain seamless light grey background gives a clean "
            "outline and no background colour on the model's edges"))
    # Contrast is measured across the outline, in colour: mean brightness
    # alone called Siamak's warm skin on a brown backdrop inseparable, and
    # the matte found him perfectly.
    import cv2
    ring = np.ones((9, 9), np.uint8)
    solid = fg.astype(np.uint8)
    inner = fg & ~cv2.erode(solid, ring).astype(bool)
    outer = cv2.dilate(solid, ring).astype(bool) & ~fg
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    outline = float(np.linalg.norm(lab[inner].mean(0) - lab[outer].mean(0)))
    metrics["outline_contrast"] = round(outline, 1)
    if outline < MIN_OUTLINE_DE:
        out.append(_finding(
            "fail" if outline < MIN_OUTLINE_DE / 2 else "warn", "contrast",
            "the figure's edge is close in colour to the background, so its "
            "outline is hard to find; put a dark character on a white "
            "background and a pale one on grey"))
    darkness = float(luma[fg].mean())
    metrics["figure_brightness"] = round(darkness, 3)
    if darkness < DARK_SUBJECT:
        out.append(_finding(
            "warn", "dark",
            "the figure is very dark, so its texture will show little detail "
            "and its outline is easy to lose; use a white background and "
            "bright, soft light"))

    # -- framing -------------------------------------------------------------
    ys, xs = np.nonzero(fg)
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    fill = (bottom - top) / float(h)
    metrics["fill"] = round(fill, 3)
    cut, tight = [], []
    for name, gap, size in (("the top of the head", top, h),
                            ("the feet", h - 1 - bottom, h),
                            ("a side", min(left, w - 1 - right), w)):
        if gap <= EDGE_TOUCH_PX:
            cut.append(name)
        elif gap < EDGE_MARGIN * size:
            tight.append(name)
    if tight and not cut:
        out.append(_finding("warn", "framing",
                            f"very little margin at {', '.join(tight)}; "
                            "leave a little background all round"))
    if cut:
        out.append(_finding("fail", "framing",
                            f"the figure is cut off by the frame ({', '.join(cut)}); "
                            "the whole figure must be in the image with a "
                            "margin all round"))
    elif fill < MIN_FILL:
        out.append(_finding("warn", "framing",
                            f"the figure fills only {fill:.0%} of the height; "
                            "fill most of the frame for more detail"))

    # -- lighting ------------------------------------------------------------
    cx = (left + right) // 2
    left_half = fg.copy()
    left_half[:, cx:] = False
    right_half = fg.copy()
    right_half[:, :cx] = False
    if left_half.any() and right_half.any():
        sides = abs(float(luma[left_half].mean())
                    - float(luma[right_half].mean()))
        metrics["side_light"] = round(sides, 3)
        if sides > SIDE_LIGHT:
            out.append(_finding(
                "warn", "lighting",
                "one side of the figure is much darker than the other: "
                "strong side light will be painted into the skin and fight "
                "every scene's light; use soft flat front lighting"))
    blown = float((luma[fg] > 0.97).mean())
    metrics["blown_share"] = round(blown, 3)
    if blown > BLOWN_SHARE:
        out.append(_finding("warn", "lighting",
                            "parts of the figure are burnt out to white; "
                            "use softer light"))

    if kind in HUMAN_KINDS:
        out.extend(_body(rgb, fg, w, h, metrics, estimator, kind))
    return _result(out, metrics)


def _body(rgb, fg, w, h, metrics, estimator, kind="human"):
    import cv2

    from pipeline import pose_mp
    out = []
    estimator = estimator or pose_mp.Estimator()
    ys, xs = fg.nonzero()
    box = (xs.min(), ys.min(), xs.max(), ys.max())
    found = estimator.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                             figure_box=box)
    if not found:
        out.append(_finding("warn", "pose",
                            "no body pose was found; the skeleton will be "
                            "guessed from the outline"))
        return out
    P = pose_mp.named(found["image"])
    metrics["pose_seeded"] = bool(found.get("seeded"))

    def pt(name):
        return P[name][0], P[name][1]

    # Arms: angle from vertical, and a background gap between arm and body.
    touching, straight = [], []
    for side in ("left", "right"):
        sx, sy = pt(f"{side}_shoulder")
        ex, ey = pt(f"{side}_elbow")
        wx, wy = pt(f"{side}_wrist")
        hx, _hy = pt(f"{side}_hip")
        # The larger of the upper arm's and the whole arm's angle: an A-pose
        # can keep the elbows near the ribs and still hold the hands well out.
        angle = max(math.degrees(math.atan2(abs(ex - sx), max(1e-6, ey - sy))),
                    math.degrees(math.atan2(abs(wx - sx), max(1e-6, wy - sy))))
        metrics[f"{side}_arm_deg"] = round(angle, 1)
        if angle < ARM_MIN_DEG:
            straight.append(side)
        gaps = []
        for t in (0.5, 0.85):
            y = ey + (wy - ey) * t
            x = ex + (wx - ex) * t
            gaps.append(_gap_between(fg, y, hx, x))
        if all(g is False for g in gaps if g is not None) and \
                any(g is not None for g in gaps):
            touching.append(side)
    if touching:
        out.append(_finding(
            "warn", "arms",
            "an arm touches the body (" + " and ".join(touching) + " side): "
            "the model fuses what touches, and the arm drags the body with it; "
            "an A-pose with background showing between arm and body is best"))
    elif straight:
        out.append(_finding(
            "warn", "arms",
            "the arms hang straight down; angle them about 35 degrees out so "
            "the hands stay clear of the hips"))

    # Legs: a gap between the knees, ankles visible.
    lk, rk = pt("left_knee"), pt("right_knee")
    gap = _gap_between(fg, (lk[1] + rk[1]) / 2.0, lk[0], rk[0])
    ankles = min(P["left_ankle"][2], P["right_ankle"][2])
    metrics["ankle_visibility"] = round(float(ankles), 3)
    if gap is False:
        out.append(_finding(
            "warn", "legs",
            "the legs are together or hidden (a long robe?); the skeleton "
            "has to guess where the knees are, and the robe will bend with "
            "the legs; feet apart, legs showing, is best"))
    elif ankles < 0.5:
        out.append(_finding("warn", "legs",
                            "the ankles are not clearly visible"))

    # Camera height: legs against torso, in the picture.
    shoulders = ((pt("left_shoulder")[1] + pt("right_shoulder")[1]) / 2.0)
    hips = (pt("left_hip")[1] + pt("right_hip")[1]) / 2.0
    feet = (pt("left_ankle")[1] + pt("right_ankle")[1]) / 2.0
    ratio = (feet - hips) / max(1.0, hips - shoulders)
    metrics["leg_torso_ratio"] = round(ratio, 2)
    if kind != "human":
        pass     # a div's or an angel's legs are as long as its designer drew
    elif ratio > LEG_TORSO_HIGH:
        out.append(_finding(
            "warn", "camera",
            "the camera looks up from below (long legs, small head); a "
            "camera at chest height gives true proportions"))
    elif ratio < LEG_TORSO_LOW:
        out.append(_finding(
            "warn", "camera",
            "the camera looks down from above (short legs); a camera at "
            "chest height gives true proportions"))

    # Face: big enough for the mouth pass.
    le, re_ = pt("left_eye"), pt("right_eye")
    eye_gap = math.hypot(le[0] - re_[0], le[1] - re_[1]) / float(w)
    metrics["eye_gap_share"] = round(eye_gap, 4)
    if eye_gap < MIN_EYE_GAP_SHARE:
        out.append(_finding(
            "warn", "face",
            "the face is small in the image; generate larger, or the "
            "speaking mouth will be soft"))
    return out


def _result(findings, metrics):
    if any(f["level"] == "fail" for f in findings):
        verdict = "poor"
    elif len([f for f in findings if f["level"] == "warn"]) >= 3:
        verdict = "poor"
    elif findings:
        verdict = "usable"
    else:
        verdict = "good"
        findings = [_finding("ok", "all",
                             "a good reference: plain background, clear "
                             "outline, limbs apart, level camera")]
    return {"verdict": verdict, "findings": findings, "metrics": metrics,
            "version": VERSION}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image")
    parser.add_argument("--kind", default="human")
    args = parser.parse_args(argv)
    try:
        result = check(args.image, args.kind)
    except Exception as exc:                        # noqa: BLE001
        result = {"verdict": "error", "findings": [_finding(
            "warn", "check", f"the image could not be checked: {exc}")],
            "metrics": {}}
    sys.stdout.write("IMAGE_CHECK " + json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
