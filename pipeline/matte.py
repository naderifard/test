"""The figure's matte: BiRefNet, then rembg, then a plain-backdrop flood.

rembg (``u2netp``) separates most figures from most backgrounds. It failed on
the case the seed prompt now asks for most often: a pale figure on a plain
pale studio backdrop. Sorush -- white robe, gold skin, light grey behind, the
robe and the backdrop within one level of each other -- came back with the
whole frame as foreground, and SF3D dutifully turned the whole frame into a
flat white slab with the angel painted on its face.

``figure`` is the one entry point, and tries in order:

1. **BiRefNet** (``ZhengPeng7/BiRefNet_lite``, 170 MB): a segmentation network
   that uses shape as well as colour; it separated that robe cleanly.
2. **rembg**, the network the pipeline used before, if BiRefNet is absent.
3. **A plain-backdrop flood** from the image border, following colour steps
   small enough to be a backdrop's gradient and not a figure's edge.

Every candidate is checked with ``failed`` (the whole frame, or three edges
touched) and the next one tried; if all fail, ``MatteError`` says so rather
than let a slab be built.
"""

from __future__ import annotations

BIREFNET = "ZhengPeng7/BiRefNet_lite"
BIREFNET_SIZE = 1024
_BIREFNET = None


class MatteError(RuntimeError):
    pass

# A matte covering more than this of the frame, or touching three edges, is
# the background taken for the figure.
MAX_COVERAGE = 0.80
FLOOD_STEP = 4               # 0..255 per channel between neighbouring pixels
MIN_FIGURE_SHARE = 0.03


def failed(alpha):
    """Whether a 0..1 (or 0..255) matte is the whole frame, not a figure."""
    import numpy as np
    a = np.asarray(alpha, dtype=np.float32)
    if a.max() > 1.0:
        a = a / 255.0
    fg = a > 0.5
    if fg.mean() > MAX_COVERAGE:
        return True
    edges = [fg[0, :].mean(), fg[-1, :].mean(), fg[:, 0].mean(),
             fg[:, -1].mean()]
    return sum(e > 0.5 for e in edges) >= 3


def plain_backdrop(rgb, step=FLOOD_STEP):
    """A 0..255 matte for a figure on a plain backdrop, or None.

    The backdrop is flood-filled from every border pixel with a floating
    colour tolerance (a gradient is followed, an edge stops it); what is left,
    cleaned up and reduced to its largest piece with holes filled, is the
    figure.
    """
    import cv2
    import numpy as np
    image = np.ascontiguousarray(rgb[:, :, :3])
    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    diff = (step, step, step)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    for x in range(0, w, 7):
        for y in (0, h - 1):
            if mask[y + 1, x + 1] == 0:
                cv2.floodFill(blurred, mask, (x, y), 0, diff, diff, flags)
    for y in range(0, h, 7):
        for x in (0, w - 1):
            if mask[y + 1, x + 1] == 0:
                cv2.floodFill(blurred, mask, (x, y), 0, diff, diff, flags)
    figure = (mask[1:-1, 1:-1] == 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    figure = cv2.morphologyEx(figure, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(figure, 8)
    if count <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    figure = (labels == biggest).astype(np.uint8)
    # Holes inside the figure (a gap under an arm is open to the border and
    # stays background; a highlight on the robe is enclosed and is filled).
    inverse = (1 - figure).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 4)
    border = set(np.unique(np.concatenate([labels[0], labels[-1],
                                           labels[:, 0], labels[:, -1]])))
    for index in range(1, count):
        if index not in border:
            figure[labels == index] = 1
    if figure.mean() < MIN_FIGURE_SHARE or failed(figure):
        return None
    return cv2.GaussianBlur(figure * 255, (3, 3), 0)


def birefnet(image):
    """A 0..255 matte from BiRefNet, or None when it cannot run here."""
    global _BIREFNET
    import numpy as np
    try:
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
    except ImportError:
        return None
    if _BIREFNET is None:
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                BIREFNET, trust_remote_code=True)
        except Exception as exc:                    # noqa: BLE001
            print(f"BiRefNet unavailable: {exc}", flush=True)
            _BIREFNET = False
            return None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _BIREFNET = (model.eval().to(device), device, transforms.Compose([
            transforms.Resize((BIREFNET_SIZE, BIREFNET_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]))
    if _BIREFNET is False:
        return None
    import cv2
    model, device, tf = _BIREFNET
    rgb = image.convert("RGB")
    with torch.no_grad():
        pred = model(tf(rgb).unsqueeze(0).to(device))[-1].sigmoid()
    alpha = pred[0, 0].float().cpu().numpy()
    alpha = cv2.resize(alpha, rgb.size, interpolation=cv2.INTER_LINEAR)
    return (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)


def release():
    """Free BiRefNet's GPU memory before something bigger loads."""
    global _BIREFNET
    if _BIREFNET:
        import torch
        _BIREFNET = None
        torch.cuda.empty_cache()


def figure(image, rembg_session=None):
    """``(alpha 0..255, method)`` for a PIL image; raises ``MatteError``."""
    import numpy as np
    tried = []
    alpha = birefnet(image)
    if alpha is not None:
        if not failed(alpha):
            return alpha, "birefnet"
        tried.append("birefnet")
    if rembg_session is not None:
        import rembg
        mask = rembg.remove(image.convert("RGB"), session=rembg_session,
                            only_mask=True)
        alpha = np.asarray(mask, dtype=np.uint8)
        if not failed(alpha):
            return alpha, "rembg"
        tried.append("rembg")
    alpha = plain_backdrop(np.asarray(image.convert("RGB")))
    if alpha is not None:
        return alpha, "plain_backdrop"
    tried.append("plain_backdrop")
    raise MatteError(
        "the figure could not be separated from its background ("
        + ", ".join(tried) + " all took the whole frame); regenerate the "
        "image with a background that contrasts with the figure -- mid "
        "grey behind a pale or white-clad character, white behind a dark one")
