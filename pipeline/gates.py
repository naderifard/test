"""The quality gates of §11.

The best idea in the existing codebase is that the host *measures* the result
instead of believing a claim that it is finished. Every stage here is measured.

Two shapes of function live in this module and the split matters:

``measure_*``
    reads an artifact and returns numbers. Some of these can only run where the
    artifact's libraries are (a motion cache needs numpy), so a stage may
    compute its own metrics inside its own interpreter.

``check_*``
    takes numbers and returns verdicts. These never touch the disk, so the
    thresholds live in exactly one file no matter which interpreter measured.

Verdict levels follow §11: ``fail`` stops the graph, ``warn`` is recorded and
printed, ``pass`` is silent.
"""

from __future__ import annotations

import math
import os
import struct
import wave

FAIL, WARN, PASS = "fail", "warn", "pass"


class Verdict:
    def __init__(self, level, name, message, value=None, threshold=None):
        self.level = level
        self.name = name
        self.message = message
        self.value = value
        self.threshold = threshold

    def __bool__(self):
        return self.level != FAIL

    def __str__(self):
        return f"[{self.level}] {self.name}: {self.message}"

    def as_dict(self):
        return {"level": self.level, "name": self.name, "message": self.message,
                "value": self.value, "threshold": self.threshold}


def worst(verdicts):
    levels = [v.level for v in verdicts]
    if FAIL in levels:
        return FAIL
    if WARN in levels:
        return WARN
    return PASS


# -- audio -------------------------------------------------------------------
def wav_info(path):
    """Duration, rate and channels without numpy. Raises on a non-PCM wav."""
    with wave.open(path, "rb") as fh:
        frames = fh.getnframes()
        rate = fh.getframerate()
        return {
            "path": path,
            "frames": frames,
            "sample_rate": rate,
            "channels": fh.getnchannels(),
            "sample_width": fh.getsampwidth(),
            "duration_s": frames / float(rate) if rate else 0.0,
        }


def read_wav_mono(path, max_seconds=None):
    """Mono float samples in -1..1, plus the sample rate.

    Stdlib only: the host environment has no numpy, and the gates have to be
    runnable from the graph process that decides whether a stage passed.
    """
    with wave.open(path, "rb") as fh:
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        rate = fh.getframerate()
        count = fh.getnframes()
        if max_seconds:
            count = min(count, int(max_seconds * rate))
        raw = fh.readframes(count)
    if width == 2:
        values = struct.unpack("<%dh" % (len(raw) // 2), raw)
        scale = 32768.0
    elif width == 1:
        values = [b - 128 for b in raw]
        scale = 128.0
    elif width == 4:
        values = struct.unpack("<%di" % (len(raw) // 4), raw)
        scale = 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    if channels > 1:
        values = [sum(values[i:i + channels]) / channels
                  for i in range(0, len(values) - channels + 1, channels)]
    return [v / scale for v in values], rate


def rms_envelope(samples, rate, fps, frame_count=None, offset_s=0.0):
    """One RMS value per output frame. This is the reference the mouth is
    checked against, and it is also how the envelope aligner finds speech."""
    step = rate / float(fps)
    start = int(round(offset_s * rate))
    total = frame_count or max(1, int(math.floor((len(samples) - start) / step)))
    out = []
    for index in range(total):
        a = start + int(round(index * step))
        b = start + int(round((index + 1) * step))
        window = samples[max(0, a):max(0, b)]
        if not window:
            out.append(0.0)
            continue
        out.append(math.sqrt(sum(v * v for v in window) / len(window)))
    return out


def speech_bounds(samples, rate, floor_db=-38.0, min_run_s=0.04):
    """First and last moment of speech, in seconds.

    A fixed dBFS floor would misjudge a quiet recording, so the threshold is set
    relative to the loudest 100 ms in the take.
    """
    hop = max(1, int(rate * 0.01))
    frames = [math.sqrt(sum(v * v for v in samples[i:i + hop]) / max(1, len(samples[i:i + hop])))
              for i in range(0, len(samples), hop)]
    if not frames:
        return 0.0, 0.0
    peak = max(frames) or 1e-9
    threshold = peak * (10.0 ** (floor_db / 20.0))
    run_frames = max(1, int(min_run_s / 0.01))
    voiced = [i for i, v in enumerate(frames) if v >= threshold]
    if not voiced:
        return 0.0, len(samples) / float(rate)
    first, last = None, None
    for index in voiced:
        if all(frames[j] >= threshold
               for j in range(index, min(index + run_frames, len(frames)))):
            first = index
            break
    for index in reversed(voiced):
        if all(frames[j] >= threshold
               for j in range(max(0, index - run_frames + 1), index + 1)):
            last = index + 1
            break
    if first is None:
        first = voiced[0]
    if last is None:
        last = voiced[-1] + 1
    return first * 0.01, min(last * 0.01, len(samples) / float(rate))


def pearson(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    mean_a, mean_b = sum(a) / n, sum(b) / n
    da = [v - mean_a for v in a]
    db = [v - mean_b for v in b]
    num = sum(x * y for x, y in zip(da, db))
    den = math.sqrt(sum(x * x for x in da) * sum(y * y for y in db))
    return num / den if den > 1e-12 else 0.0


# How many scrambled versions of the same envelope the real one has to beat.
# Eight is a permutation test at p < 1/9 and costs eight correlations of a
# ninety-element list.
LIPSYNC_CONTROLS = 8


def lip_sync_score(jaw_curve, envelope, max_lag=8, controls=LIPSYNC_CONTROLS):
    """§11's lip-sync metric: does the mouth move when this sound happens?

    Three numbers come back and the *third* is the one that decides anything.

    ``r_at_zero`` is the correlation as placed. On its own it cannot be given a
    threshold, because how high it goes depends on things that have nothing to
    do with whether the sync is right: a jaw driven by phoneme shapes is only
    loosely tied to loudness, and §8's ``graphic`` viseme profile then holds
    each sample for two extra frames, which quantises the curve and lowers the
    correlation again. Measured on this project's own two lines, a correct,
    held, miniature-profile mouth scores 0.24-0.34 — under the 0.30 that was
    guessed for this metric before it had ever been run.

    ``best_lag`` is where the correlation peaks. It catches a curve keyed at
    the wrong frame, but only when the peak is sharp: a held curve trails its
    own samples by about half the hold, and on a shallow correlation surface
    the argmax wanders a frame or two for no reason worth acting on.

    ``control_r`` is the answer to both problems. The same jaw curve is
    correlated against the same envelope *rotated*, several times over, at
    offsets big enough to destroy the correspondence and small enough to keep
    every statistical property of the signal. The real alignment has to beat
    the best of them. That is self-calibrating — it assumes nothing about
    viseme profile, phoneme content or speech energy — and on this project it
    separates cleanly: correct mouths beat their strongest control by 0.08 to
    0.24, while the wrong line attached to the shot and a curve shifted four
    frames both come in 0.11 to 0.44 *below* it. Those two are precisely the
    failures §11 says this metric exists to catch.

    Positive lag means the jaw curve trails the audio.
    """
    if not jaw_curve or not envelope:
        return {"r_at_zero": 0.0, "best_r": 0.0, "best_lag": 0,
                "control_r": 0.0, "margin": 0.0, "usable": False}
    peak = max(envelope) or 1e-9
    reference = [v / peak for v in envelope]
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = jaw_curve[lag:], reference[:len(reference) - lag or None]
        else:
            a, b = jaw_curve[:len(jaw_curve) + lag or None], reference[-lag:]
        results[lag] = pearson(a, b)
    best_lag = max(results, key=lambda k: results[k])

    nulls = []
    for index in range(1, max(1, int(controls)) + 1):
        shift = int(round(index * len(reference) / float(controls + 1)))
        rotated = reference[shift:] + reference[:shift]
        nulls.append(pearson(jaw_curve, rotated))
    control = max(nulls) if nulls else 0.0

    return {
        "r_at_zero": round(results[0], 4),
        "best_r": round(results[best_lag], 4),
        "best_lag": best_lag,
        "control_r": round(control, 4),
        "control_mean": round(sum(nulls) / len(nulls), 4) if nulls else 0.0,
        "margin": round(results[0] - control, 4),
        "by_lag": {str(k): round(v, 4) for k, v in sorted(results.items())},
        "usable": True,
    }


# -- images ------------------------------------------------------------------
def _pillow():
    try:
        from PIL import Image
    except ImportError:
        return None
    return Image


def measure_image(path, thumb=128):
    """Resolution plus enough statistics to tell a real image from a blank one."""
    Image = _pillow()
    if Image is None:
        return {"path": path, "bytes": os.path.getsize(path),
                "note": "Pillow unavailable; only the file size was measured"}
    with Image.open(path) as image:
        width, height = image.size
        grey = image.convert("L")
        small = grey.resize((thumb, max(1, thumb * height // max(1, width))))
        pixels = list(small.getdata())
    count = len(pixels) or 1
    mean = sum(pixels) / count / 255.0
    variance = sum((p / 255.0 - mean) ** 2 for p in pixels) / count
    return {
        "path": path,
        "bytes": os.path.getsize(path),
        "width": width,
        "height": height,
        "mean_luma": round(mean, 4),
        "stddev_luma": round(math.sqrt(variance), 4),
        "unique_levels": len(set(pixels)),
    }


def luma_grid(path, width=48):
    """A tiny luminance grid, for frame-to-frame comparison."""
    Image = _pillow()
    if Image is None:
        return []
    with Image.open(path) as image:
        w, h = image.size
        small = image.convert("L").resize((width, max(1, width * h // max(1, w))))
        return [p / 255.0 for p in small.getdata()]


def temporal_flicker(frame_paths, stride=1):
    """§11's new metric: mean absolute luminance delta between frames.

    A still frame can pass every existing check while the sequence boils. This
    is measured on a 48 px grid because the failure is global sample noise, not
    a detail, and downsampling makes it cheap enough to run on every render.
    """
    picked = frame_paths[::stride]
    if len(picked) < 2:
        return {"frames": len(picked), "mean_delta": 0.0, "max_delta": 0.0,
                "usable": False}
    deltas = []
    previous = luma_grid(picked[0])
    for path in picked[1:]:
        current = luma_grid(path)
        if not current or len(current) != len(previous):
            previous = current
            continue
        deltas.append(sum(abs(a - b) for a, b in zip(current, previous))
                      / len(current))
        previous = current
    if not deltas:
        return {"frames": len(picked), "mean_delta": 0.0, "max_delta": 0.0,
                "usable": False}
    return {
        "frames": len(picked),
        "mean_delta": round(sum(deltas) / len(deltas), 5),
        "max_delta": round(max(deltas), 5),
        "usable": True,
    }


# -- thresholds --------------------------------------------------------------
# Flicker limits are per style pack on purpose. The miniature look is the fast
# path meant to carry thousands of frames on EEVEE, and EEVEE sample noise is
# exactly what this metric measures — a single global limit would either be too
# loose to catch a boiling photoreal render or too tight to let the fast path
# run at all.
FLICKER_LIMIT = {
    "miniature": {"warn": 0.020, "fail": 0.055},
    "stylized": {"warn": 0.014, "fail": 0.040},
    "photoreal": {"warn": 0.010, "fail": 0.030},
}

# A backstop, not the real check. The control comparison in `lip_sync_score` is
# what decides whether a mouth is synced; this only catches a mouth that barely
# moves at all, and is set below the 0.24 a correct held miniature-profile line
# measures on this project rather than above it.
LIPSYNC_MIN_R = 0.15
LIPSYNC_MAX_LAG = 1

MIN_REF_PIXELS = 1024
MIN_REF_STDDEV = 0.02

VOICE_DURATION_TOLERANCE_S = 0.010
MIN_PHONEME_S = 0.020
MIN_AUDIO_COVERAGE = 0.90


# A standing figure's head is in the top of the frame. Measured on this
# project's six references, the true faces sit at 8, 10, 16 and 26% of image
# height and every false positive at 39% or below: the claws of the div at 64
# and 65%, the gazelle's hooves at 84%, three more on Siamak's leopard skin at
# 57-73%. The gap between 26 and 39 is where this line goes.
FACE_MAX_HEIGHT_FRACTION = 0.35


def face_boxes(path):
    """Every frontal-face detection in an image, as (x, y, w, h).

    OpenCV's cascade, which is what is available without another model download.
    It is honest about faces and enthusiastic about anything with two dark
    patches above a light one, so the caller has to choose -- see ``head_box``.
    """
    try:
        import cv2
    except ImportError:
        return None                      # cannot tell, which is not "none found"
    cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades,
                     "haarcascade_frontalface_default.xml"))
    grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if grey is None or cascade.empty():
        return None
    return [tuple(int(v) for v in box)
            for box in cascade.detectMultiScale(grey, 1.1, 5)]


def choose_head_box(boxes, image_height):
    """Which of these detections is the figure's head, or None.

    Not the largest -- that rule picked a 147 px claw over a 54 px face on the
    black div, because the claw is bigger. The head of a standing figure is the
    *topmost* plausible detection, and anything below
    ``FACE_MAX_HEIGHT_FRACTION`` of the image is not a head at all.

    Separate from ``head_box`` so the choice can be tested without an image and
    a cascade: the detections above are the ones this project's own references
    actually produce.
    """
    if not boxes or not image_height:
        return None
    plausible = [b for b in boxes
                 if (b[1] + b[3] / 2.0) / float(image_height)
                 <= FACE_MAX_HEIGHT_FRACTION]
    if not plausible:
        return None
    # Topmost; ties broken by area, so a head found twice picks the fuller box.
    return min(plausible, key=lambda b: (b[1] + b[3] / 2.0, -b[2] * b[3]))


def head_box(path):
    """The detection that is actually this figure's head, or None."""
    boxes = face_boxes(path)
    if not boxes:
        return None
    image = _pillow()
    if image is None:
        return None
    with image.open(path) as opened:
        height = opened.size[1]
    return choose_head_box(boxes, height)


def check_refs(character_id, images):
    """§11-B. Reports what is missing per character rather than dying on the
    first gap, as §7-B asks."""
    out = []
    if not images:
        return [Verdict(WARN, "refs.present",
                        f"{character_id}: no reference images yet")]
    for stats in images:
        name = os.path.basename(stats.get("path", "?"))
        biggest = max(stats.get("width", 0), stats.get("height", 0))
        if biggest and biggest < MIN_REF_PIXELS:
            out.append(Verdict(FAIL, "refs.resolution",
                               f"{name} is {stats.get('width')}x{stats.get('height')}, "
                               f"under {MIN_REF_PIXELS} px on the long edge",
                               biggest, MIN_REF_PIXELS))
        if stats.get("stddev_luma", 1.0) < MIN_REF_STDDEV:
            out.append(Verdict(FAIL, "refs.blank",
                               f"{name} is effectively blank "
                               f"(luminance stddev {stats.get('stddev_luma')})",
                               stats.get("stddev_luma"), MIN_REF_STDDEV))
    return out or [Verdict(PASS, "refs", f"{character_id}: {len(images)} usable refs")]


# §11 C-body gives one plausible height range, and it is the range for a person:
# 1.4-2.6 m is Rostam at 2.1 and nobody at 4. Applied to the whole cast it fires
# on every animal in the film forever - a gazelle is 0.9 m and correct - and a
# warning that is always wrong is a warning people learn to scroll past. So the
# range is per kind, with §11's own numbers kept for the kind it was written for.
HEIGHT_RANGE_M = {
    "human": (1.4, 2.6),
    "humanoid": (1.4, 3.2),
    "creature": (0.3, 8.0),
    "quadruped": (0.3, 3.5),
    "animal": (0.3, 3.5),
}
DEFAULT_HEIGHT_RANGE_M = (0.3, 8.0)


def check_body(metrics):
    """§11 C-body: height after scale inside a plausible range, warn only."""
    height = metrics.get("height_m") or 0.0
    kind = (metrics.get("kind") or "human").lower()
    low, high = HEIGHT_RANGE_M.get(kind, DEFAULT_HEIGHT_RANGE_M)
    if not low <= height <= high:
        return [Verdict(WARN, "body.height",
                        f"built height is {height:.3f} m, outside {low}-{high} m "
                        f"for a {kind}", round(height, 3), [low, high])]
    return [Verdict(PASS, "body.height", f"height {height:.3f} m")]


def check_asset(metrics):
    """§11 C-asset: tri count, UV coverage, no zero-area faces, sane bbox."""
    out = []
    tris = metrics.get("tris") or 0
    if tris <= 0:
        out.append(Verdict(FAIL, "asset.tris", "the mesh has no triangles", tris, 1))
    elif tris > 400000:
        out.append(Verdict(WARN, "asset.tris",
                           f"{tris} triangles is heavy for 8 GB of VRAM", tris, 400000))
    if metrics.get("degenerate_faces"):
        out.append(Verdict(WARN, "asset.degenerate",
                           f"{metrics['degenerate_faces']} zero-area faces"))
    coverage = metrics.get("uv_coverage")
    if coverage is not None and coverage < 0.05:
        out.append(Verdict(WARN, "asset.uv",
                           f"UVs cover only {coverage:.1%} of the 0-1 square",
                           coverage, 0.05))
    size = metrics.get("size_m") or []
    if size and (max(size) > 200 or max(size) <= 0):
        out.append(Verdict(WARN, "asset.bbox",
                           f"bounding box {size} is not a plausible scale"))
    return out or [Verdict(PASS, "asset", f"{tris} tris")]


def check_motion(metrics):
    """§11-E: no NaN, joint velocity clamp, foot skate < 2 cm/frame."""
    out = []
    if metrics.get("nan_count"):
        out.append(Verdict(FAIL, "motion.nan",
                           f"{metrics['nan_count']} NaN values in the pose track",
                           metrics["nan_count"], 0))
    if (metrics.get("frames") or 0) < 2:
        out.append(Verdict(FAIL, "motion.frames",
                           "a motion cache needs at least two frames",
                           metrics.get("frames"), 2))
    skate = metrics.get("foot_skate_m_per_frame")
    if skate is not None and skate > 0.02:
        out.append(Verdict(WARN, "motion.skate",
                           f"feet slide {skate * 100:.1f} cm/frame; over 2 cm reads "
                           "as skating", round(skate, 4), 0.02))
    velocity = metrics.get("max_joint_deg_per_frame")
    if velocity is not None and velocity > 45.0:
        out.append(Verdict(WARN, "motion.velocity",
                           f"peak joint rotation {velocity:.1f} deg/frame is jittery",
                           round(velocity, 2), 45.0))
    return out or [Verdict(PASS, "motion", f"{metrics.get('frames')} frames clean")]


def check_voice(metrics):
    """§11-F. The duration check is the load-bearing one: a conversion that
    resynthesises with new timing silently invalidates every mouth curve."""
    out = []
    delta = abs((metrics.get("out_duration_s") or 0.0)
                - (metrics.get("in_duration_s") or 0.0))
    if delta > VOICE_DURATION_TOLERANCE_S:
        out.append(Verdict(FAIL, "voice.duration",
                           f"conversion changed the length by {delta * 1000:.1f} ms; "
                           "the mouth curves would no longer match",
                           round(delta, 5), VOICE_DURATION_TOLERANCE_S))
    similarity = metrics.get("speaker_similarity")
    if similarity is not None and similarity < 0.55:
        out.append(Verdict(WARN, "voice.similarity",
                           f"speaker similarity {similarity:.2f} to the reference",
                           similarity, 0.55))
    wer = metrics.get("asr_wer")
    if wer is not None and wer > 0.25:
        out.append(Verdict(FAIL, "voice.wer",
                           f"word error rate {wer:.0%} after conversion; the words "
                           "are no longer intelligible", wer, 0.25))
    return out or [Verdict(PASS, "voice",
                           f"duration preserved to {delta * 1000:.1f} ms")]


def check_align(align):
    """§11-G: every word timed, no phoneme under 20 ms, coverage over 90%."""
    out = []
    words = align.get("words") or []
    phones = align.get("phones") or []
    untimed = [w for w in words if w.get("end_s", 0) <= w.get("start_s", 0)]
    if untimed:
        out.append(Verdict(FAIL, "align.words",
                           f"{len(untimed)} of {len(words)} words have no duration",
                           len(untimed), 0))
    if not phones:
        out.append(Verdict(FAIL, "align.phones", "no phonemes were produced"))
        return out
    short = [p for p in phones
             if p.get("end_s", 0) - p.get("start_s", 0) < MIN_PHONEME_S]
    if short:
        out.append(Verdict(WARN, "align.short_phones",
                           f"{len(short)} of {len(phones)} phonemes are under "
                           f"{MIN_PHONEME_S * 1000:.0f} ms and will not read on screen",
                           len(short), 0))
    speech = max(1e-6, (align.get("speech_end_s") or 0.0)
                 - (align.get("speech_start_s") or 0.0))
    covered = sum(max(0.0, p.get("end_s", 0) - p.get("start_s", 0)) for p in phones)
    coverage = covered / speech
    if coverage < MIN_AUDIO_COVERAGE:
        out.append(Verdict(FAIL, "align.coverage",
                           f"phonemes cover {coverage:.0%} of the speech; under "
                           f"{MIN_AUDIO_COVERAGE:.0%} means the alignment drifted",
                           round(coverage, 3), MIN_AUDIO_COVERAGE))
    if align.get("approximate"):
        out.append(Verdict(WARN, "align.approximate",
                           f"timings came from the {align.get('backend')} fallback, "
                           "not a real forced alignment"))
    return out or [Verdict(PASS, "align",
                           f"{len(words)} words, {len(phones)} phonemes, "
                           f"{coverage:.0%} coverage")]


def check_assembly(metrics):
    """§11-H: every object has a material, feet on ground, subject in frustum."""
    out = []
    missing = metrics.get("objects_without_material") or []
    if missing:
        out.append(Verdict(FAIL, "assembly.materials",
                           f"no material on {', '.join(missing[:6])}"
                           + (" ..." if len(missing) > 6 else ""),
                           len(missing), 0))
    floating = metrics.get("characters_off_ground") or {}
    for name, gap in floating.items():
        out.append(Verdict(FAIL, "assembly.ground",
                           f"{name} is {gap:+.3f} m off the ground",
                           round(gap, 4), 0.02))
    in_frame = metrics.get("subjects_in_frustum")
    if in_frame is not None and not in_frame:
        out.append(Verdict(FAIL, "assembly.frustum",
                           "no cast member is inside the camera frustum"))
    coverage = metrics.get("subject_frame_fraction")
    if coverage is not None and coverage < 0.02:
        out.append(Verdict(WARN, "assembly.framing",
                           f"the cast fills only {coverage:.1%} of the frame",
                           round(coverage, 4), 0.02))
    return out or [Verdict(PASS, "assembly", "materials, ground and framing all sane")]


def check_render(analysis, flicker=None, lipsync=None, style_pack="miniature"):
    """§11-J: the existing still checks plus the two new sequence metrics."""
    out = []
    if analysis.get("render_error"):
        return [Verdict(FAIL, "render.error", str(analysis["render_error"])[:400])]

    mean = analysis.get("mean_luminance")
    if mean is not None:
        if mean < 0.06:
            out.append(Verdict(FAIL, "render.dark",
                               f"mean luminance {mean:.3f} — the frame is black",
                               mean, 0.06))
        elif mean > 0.88:
            out.append(Verdict(WARN, "render.bright",
                               f"mean luminance {mean:.3f} — the frame is blown out",
                               mean, 0.88))
    black = analysis.get("fraction_near_black")
    if black is not None and black > 0.90:
        out.append(Verdict(FAIL, "render.empty",
                           f"{black:.0%} of the frame is near-black", black, 0.90))
    contrast = analysis.get("contrast")
    if contrast is not None and contrast < 0.04:
        out.append(Verdict(WARN, "render.flat",
                           f"contrast {contrast:.3f} is flat", contrast, 0.04))

    if flicker and flicker.get("usable"):
        limits = FLICKER_LIMIT.get(style_pack, FLICKER_LIMIT["stylized"])
        value = flicker["mean_delta"]
        if value > limits["fail"]:
            out.append(Verdict(FAIL, "render.flicker",
                               f"mean frame-to-frame luminance delta {value:.4f} — "
                               "the sequence is boiling", value, limits["fail"]))
        elif value > limits["warn"]:
            out.append(Verdict(WARN, "render.flicker",
                               f"mean frame-to-frame luminance delta {value:.4f}; "
                               f"over {limits['warn']} for the {style_pack} pack",
                               value, limits["warn"]))

    if lipsync and lipsync.get("usable"):
        control = lipsync.get("control_r", 0.0)
        if lipsync["r_at_zero"] <= control:
            out.append(Verdict(FAIL, "render.lipsync",
                               f"the mouth matches a scrambled version of this "
                               f"line ({control:+.2f}) as well as the real one "
                               f"({lipsync['r_at_zero']:+.2f}) — it is not "
                               "tracking this voice",
                               lipsync["r_at_zero"], control))
        elif lipsync["r_at_zero"] < LIPSYNC_MIN_R:
            out.append(Verdict(FAIL, "render.lipsync",
                               f"jaw-to-audio correlation "
                               f"{lipsync['r_at_zero']:.2f} — the mouth barely "
                               "moves with the voice",
                               lipsync["r_at_zero"], LIPSYNC_MIN_R))
        elif abs(lipsync["best_lag"]) > LIPSYNC_MAX_LAG:
            # A warning and not a failure: §8's snap profile holds each viseme
            # for two extra frames, which delays the curve against its own
            # samples and moves the argmax without anything being wrong. The
            # control comparison above is what catches a genuinely shifted
            # curve, and it does so decisively.
            out.append(Verdict(WARN, "render.lipsync_offset",
                               f"the mouth correlates best "
                               f"{lipsync['best_lag']:+d} frames from the audio "
                               f"(r {lipsync.get('best_r', 0.0):.2f} against "
                               f"{lipsync['r_at_zero']:.2f} as placed)",
                               lipsync["best_lag"], LIPSYNC_MAX_LAG))
    return out or [Verdict(PASS, "render", "luminance, framing and stability all pass")]
