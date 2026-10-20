"""Stage G — the master clock, and the mouth that keeps time with it.

Section 7-G1. This stage does two jobs that look separate and are not.

**It fixes the length of every shot.** Section 5.2 makes the dialogue audio the
master clock: ``frames = ceil((speech_end - speech_start + head + tail) * fps)``.
Nothing else in the pipeline knows where the speech in a recording starts and
stops, so until ``align.json`` exists ``h_assemble`` cannot compute a frame
range at all. That is why this stage is written before the one that consumes it.

**It writes the mouth.** Phoneme timings become the eight channels of
``validate.VISEME_CHANNELS``, per frame, and because they are derived from the
same measurement of the same waveform they cannot drift against it. Section 5.2's
"the mouth curves are frame-locked to the audio by construction" is this file.

Three decisions are worth stating plainly.

**The backend is a ladder, and which rung ran is recorded.** Section 7-G wants
WhisperX with a Persian wav2vec2 model, with "Whisper word timestamps plus a
Persian G2P, distributing phonemes proportionally inside each word" as the named
fallback. Neither aligner is installed on this machine, so what runs today is
the fallback, built on :mod:`pipeline.persian` and on the real energy contour of
the real recording. :func:`_align` tries each rung in turn and reports why each
one it skipped was skipped, so a run says *which* aligner produced its timings
rather than leaving a reader to assume the best one did. Every artifact from the
fallback carries ``approximate: true``, which ``gates.check_align`` turns into a
warning on every single run. It is meant to be noisy: section 16 rates Persian
alignment quality a medium risk and asks for it to be evaluated early.

**The fallback measures the audio; it does not merely divide it up.** Word
durations start proportional to the phonemes in them and are then *snapped to
energy minima* in the recording, so a boundary lands in a real dip between two
words rather than at an arithmetic fraction of the line. The jaw is modulated by
the frame-rate RMS envelope on top of its phoneme shape, because a mouth is
open when sound is coming out of it. So the curves follow this take of this
line, and attaching the wrong recording to a shot produces visibly wrong
timings rather than plausible ones.

That last point has a consequence for section 11's lip-sync metric that has to
be said out loud, because a metric nobody distrusts is a metric that stops
working: the jaw curve is *partly derived from* the audio envelope it will later
be correlated against, so a high correlation is not evidence that the mouth
looks good. What the metric still genuinely catches, and what it is kept for, is
wiring — a curve keyed at the wrong frame, or the wrong line attached to the
wrong shot, both of which move the argmax off zero lag. :func:`_gate` runs that
same check here, against the offset this stage itself wrote, so an indexing
error is caught in the stage that caused it instead of two stages downstream.

**Frame zero of ``visemes.json`` is ``align.speech_start_s``, not zero.** The
schema has nowhere to record an offset, so the convention has to be stated
somewhere and obeyed everywhere. It is the right convention because
``silence_lead_policy`` defaults to trimming the lead-in: the audio strip in the
shot starts at ``speech_start_s`` too, so the curve and the sound are placed at
the same frame by the same number and no third party has to subtract anything.
"""

from __future__ import annotations

import math
import os
import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio, persian
from pipeline.validate import VISEME_CHANNELS

# The energy contour the word boundaries are snapped against. 10 ms is short
# enough to resolve the dip between two words and long enough that one glottal
# pulse is not mistaken for one.
HOP_S = 0.010
SMOOTH_PASSES = 2

# How far a boundary may be dragged from its proportional position, as a
# fraction of the shorter of the two words it separates. Unbounded snapping lets
# one loud syllable swallow a whole neighbouring word.
SNAP_FRACTION = 0.45
SNAP_MAX_S = 0.14

# A word has to be long enough to hold its own phonemes at the length section 11
# says a phoneme needs to read on screen.
MIN_PHONEME_S = gates.MIN_PHONEME_S

# The jaw is the product of a phoneme shape and the loudness at that moment.
# The floor keeps a quiet syllable from closing the mouth completely, which
# reads as a mumble rather than as quiet speech.
JAW_ENVELOPE_FLOOR = 0.35

# Brows do not snap on with the first phoneme. A quarter of a second in and out
# is fast enough to be there for the line and slow enough not to pop.
EMOTION_RAMP_S = 0.25

# How much better than lag zero another lag has to score before it is evidence
# of a shift rather than of a bumpy correlation surface. Measured on the
# Keyumars lines: four of five peak at zero by 0.05 to 0.17 over their
# neighbours, and the fifth peaks at zero and carries a second bump eight frames
# out that beats it by 0.009. Treating that as an off-by-eight would fail a
# correctly aligned line, and a gate that cries wolf is a gate people switch
# off. A lag that is genuinely wrong wins by much more than this.
LAG_DECISIVE_R = 0.05
# Evenly timed speech (a synthetic scratch line, a chant) bumps its correlation
# once per syllable, about seven frames out, so a shift must also beat zero by
# this share of the zero-lag score; and a one-frame shift (42 ms at 24 fps) is
# under what anyone can see, so it only warns.
LAG_DECISIVE_SHARE = 0.10
LAG_VISIBLE_FRAMES = 2

# Below this the mouth is not opening at all, whatever the phonemes claim.
JAW_PEAK_MIN = 0.20
JAW_ACTIVE_MIN_FRACTION = 0.25


class AlignerUnavailable(RuntimeError):
    """One rung of the ladder cannot run here, with the reason why."""


# -- planning ----------------------------------------------------------------
def plan(project, graph):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    config = project.config
    profile = _profile(graph.style_pack)
    units = []
    for entry in resolved.get("dialogue_index") or []:
        line_id = entry["line_id"]
        units.append(Unit(
            f"line_{line_id}",
            # f_voice publishes converted.wav beside the recording. Preferring
            # it when it exists means writing f_voice re-keys these units and
            # the alignment is redone against the audio that will actually be
            # heard - which is the whole point of section 7-F's duration gate.
            inputs={"audio": _audio_path(project, line_id)},
            params={
                "line_id": line_id,
                "character": entry.get("character"),
                "text_fa": entry.get("text_fa") or "",
                "emotion": entry.get("emotion") or "neutral",
                "language": config.get("language") or "fa",
                "fps": project.fps(),
                "backend": config.get("aligner_backend") or "auto",
                # The profile and not the pack name: section 8 gives miniature
                # snap visemes and the other two smooth ones, so photoreal and
                # stylized share this artifact and switching between them does
                # not re-run an aligner.
                "viseme_profile": profile,
            },
            out_rel=f"audio/{line_id}",
            label=f"align/{line_id} ({entry.get('character')})",
        ))
    return units


def nothing_to_do(project):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    if not resolved:
        return "no resolved production plan — a_production has not run"
    return "no dialogue lines in the plan; nothing to align"


def _audio_path(project, line_id):
    converted = project.path("audio", line_id, "converted.wav")
    if os.path.exists(converted):
        return converted
    return project.path("audio", line_id, "source.wav")


def _profile(style_pack):
    from blender_ai_studio import styles
    return styles.load(style_pack).visemes.get("profile") or "smooth"


# -- the backend ladder ------------------------------------------------------
def _try_whisperx(context):
    """Section 7-G's first choice: WhisperX with a Persian wav2vec2 model."""
    try:
        import whisperx                                        # noqa: F401
    except ImportError as exc:
        raise AlignerUnavailable(
            f"whisperx is not importable here ({exc.__class__.__name__}: "
            f"{exc})") from exc
    raise AlignerUnavailable(
        "whisperx imports but the M4 wiring for it is not written yet; this is "
        "the seam it goes in")


def _try_wav2vec2(context):
    """The same alignment without WhisperX: a Persian CTC model, forced-aligned.

    Deliberately never reached by ``auto``. A checkpoint has to be named — by
    ``aligner_model`` in project.json or ``PIPELINE_ALIGN_MODEL`` — because the
    alternative is a stage that quietly downloads a gigabyte from the network in
    the middle of a run, and a pipeline that does that once will do it on a
    machine that is offline.
    """
    model = context.get("model")
    if not model:
        raise AlignerUnavailable(
            "no Persian wav2vec2 checkpoint is configured; set aligner_model in "
            "project.json or PIPELINE_ALIGN_MODEL to a local model directory")
    try:
        import torch                                           # noqa: F401
        import transformers                                    # noqa: F401
    except ImportError as exc:
        raise AlignerUnavailable(
            f"torch/transformers are not importable here ({exc})") from exc
    raise AlignerUnavailable(
        f"a checkpoint is named ({model}) but the M4 CTC forced-alignment pass "
        "is not written yet")


def _try_envelope(context):
    """The fallback of section 7-G: a Persian G2P over the recording's energy."""
    return _align_envelope(context)


LADDER = (
    ("whisperx", _try_whisperx),
    ("wav2vec2", _try_wav2vec2),
    ("envelope", _try_envelope),
)


def _align(context):
    """Walk the ladder. Returns the alignment and one note per rung skipped."""
    wanted = (context.get("backend") or "auto").strip().lower()
    rungs = [(n, f) for n, f in LADDER if wanted in ("auto", n)]
    if not rungs:
        raise ValueError(
            f"aligner_backend {wanted!r} names no backend; known backends are "
            f"{', '.join(n for n, _ in LADDER)} (or 'auto')")
    skipped = []
    for name, function in rungs:
        try:
            return function(context), skipped
        except AlignerUnavailable as exc:
            skipped.append(f"{name}: {exc}")
    raise RuntimeError("no aligner could run:\n  " + "\n  ".join(skipped))


# -- the energy contour ------------------------------------------------------
def _contour(samples, rate, hop_s=HOP_S):
    """Per-hop RMS over the whole take. The thing boundaries are snapped to."""
    hop = max(1, int(round(rate * hop_s)))
    out = []
    for start in range(0, len(samples), hop):
        window = samples[start:start + hop]
        if not window:
            break
        out.append(math.sqrt(sum(v * v for v in window) / len(window)))
    return out


def _smooth(values, passes=SMOOTH_PASSES):
    """A three-tap box, twice. Enough to stop a pitch period reading as a gap."""
    for _ in range(max(0, passes)):
        if len(values) < 3:
            break
        values = ([(values[0] * 3 + values[1]) / 4.0]
                  + [(values[i - 1] + values[i] * 2 + values[i + 1]) / 4.0
                     for i in range(1, len(values) - 1)]
                  + [(values[-1] * 3 + values[-2]) / 4.0])
    return values


def _share(total, weights, minimum):
    """Split ``total`` across ``weights``, with nothing under ``minimum``.

    Water-filling rather than a single proportional pass: a proportional split
    of a fast word gives its stops two milliseconds, which is under the length
    section 11 says a phoneme needs, and simply clamping them would make the
    parts sum to more than the whole. Anything pinned at the minimum is taken
    out of the pool and the rest re-share what is left. When the word is too
    short to hold its phonemes at all, an even split is the honest answer and
    the gate is left to say so.
    """
    count = len(weights)
    if count == 0:
        return []
    if total <= count * minimum:
        return [total / count] * count
    pinned = [False] * count
    for _ in range(count):
        pool = total - minimum * sum(pinned)
        mass = sum(w for w, p in zip(weights, pinned) if not p) or 1.0
        shares = [minimum if pinned[i] else pool * weights[i] / mass
                  for i in range(count)]
        short = [i for i in range(count)
                 if not pinned[i] and shares[i] < minimum]
        if not short:
            return shares
        for index in short:
            pinned[index] = True
    return [total / count] * count


def _snap(boundary_s, contour, low_s, high_s):
    """Drag a word boundary to the quietest moment near it.

    This is the one place the fallback stops guessing and looks at the
    recording: a boundary that lands in a real dip in the energy is a boundary
    where a word really ended, whatever the phoneme arithmetic said.
    """
    first = int(math.floor(low_s / HOP_S))
    last = int(math.ceil(high_s / HOP_S))
    first = max(0, min(first, len(contour) - 1))
    last = max(first, min(last, len(contour) - 1))
    if last <= first:
        return boundary_s, False
    best = min(range(first, last + 1), key=lambda i: contour[i])
    moved = (best + 0.5) * HOP_S
    return moved, abs(moved - boundary_s) > HOP_S


def _align_envelope(context):
    """Persian G2P, proportional inside each word, boundaries on real energy."""
    text = context["text"]
    spelled, guessed = persian.g2p(text)
    if not spelled:
        raise ValueError(
            f"{context['line_id']}: no Persian text to align against. The "
            "envelope fallback derives phonemes from text_fa, so a line "
            "without it cannot be given a mouth.")

    start = context["speech_start_s"]
    end = context["speech_end_s"]
    span = max(1e-6, end - start)
    contour = _smooth(_contour(context["samples"], context["rate"]))

    weights = [sum(persian.weight_of(p) for p in phones)
               for _word, phones in spelled]
    floors = [max(MIN_PHONEME_S * len(phones), MIN_PHONEME_S)
              for _word, phones in spelled]
    lengths = _share(span, weights, min(floors) if floors else MIN_PHONEME_S)

    # Proportional boundaries first, then each interior one snapped to the
    # nearest energy minimum, then forced back into order in case two snapped
    # onto the same dip.
    edges, cursor = [start], start
    for length in lengths[:-1]:
        cursor += length
        edges.append(cursor)
    edges.append(end)

    snapped = 0
    for index in range(1, len(edges) - 1):
        left = edges[index] - edges[index - 1]
        right = edges[index + 1] - edges[index]
        reach = min(SNAP_MAX_S, SNAP_FRACTION * min(left, right))
        moved, did = _snap(edges[index], contour,
                           edges[index] - reach, edges[index] + reach)
        edges[index] = moved
        snapped += 1 if did else 0
    for index in range(1, len(edges)):
        edges[index] = max(edges[index], edges[index - 1] + MIN_PHONEME_S)
    if edges[-1] > end:                       # the clamp above may have run past
        edges = [start + (e - start) * (span / (edges[-1] - start))
                 for e in edges]

    words, phones = [], []
    for index, (word, sounds) in enumerate(spelled):
        word_start, word_end = edges[index], edges[index + 1]
        words.append({"text": word, "start_s": round(word_start, 4),
                      "end_s": round(word_end, 4), "score": None})
        shares = _share(word_end - word_start,
                        [persian.weight_of(p) for p in sounds], MIN_PHONEME_S)
        cursor = word_start
        for phone, length in zip(sounds, shares):
            phones.append({
                "phone": phone,
                "viseme": persian.viseme_of(phone),
                "start_s": round(cursor, 4),
                "end_s": round(cursor + length, 4),
                "word": index,
                "score": None,
            })
            cursor += length

    return {
        "backend": "envelope",
        "backend_model": None,
        "approximate": True,
        "confidence": None,
        "words": words,
        "phones": phones,
        "guessed_vowels": guessed,
        "boundaries_snapped": snapped,
    }


# -- the curves --------------------------------------------------------------
def _dominant(phones, low, high):
    """The phoneme that owns most of one frame.

    Not "the phoneme at the midpoint": at 24 fps a frame is 42 ms and a stop can
    be 30, so the midpoint rule drops short sounds entirely and a mouth loses
    its consonants. Overlap keeps whichever sound the frame is mostly made of.
    """
    best, best_overlap = None, 0.0
    for phone in phones:
        overlap = min(high, phone["end_s"]) - max(low, phone["start_s"])
        if overlap > best_overlap:
            best, best_overlap = phone, overlap
    return best


def _curves(align, samples, rate, fps, emotion, frames_hint=None):
    """The eight channels, one value per frame, from ``speech_start_s`` on."""
    start = align["speech_start_s"]
    span = max(1.0 / fps, align["speech_end_s"] - start)
    frames = frames_hint or max(1, int(math.ceil(span * fps - 1e-9)))

    envelope = gates.rms_envelope(samples, rate, fps, frame_count=frames,
                                  offset_s=start)
    peak = max(envelope) or 1e-9
    loudness = [min(1.0, v / peak) for v in envelope]

    face, known = persian.emotion_shape(emotion)
    ramp_frames = max(1, int(round(EMOTION_RAMP_S * fps)))

    channels = {name: [] for name in VISEME_CHANNELS}
    visemes = []
    for index in range(frames):
        low = start + index / float(fps)
        high = low + 1.0 / fps
        phone = _dominant(align["phones"], low, high)
        viseme = phone["viseme"] if phone else persian.REST_VISEME
        visemes.append(viseme)
        mouth = persian.shape_of(viseme)
        gain = JAW_ENVELOPE_FLOOR + (1.0 - JAW_ENVELOPE_FLOOR) * loudness[index]
        mouth["jaw_open"] *= gain

        ramp = min(1.0, (index + 1) / float(ramp_frames),
                   (frames - index) / float(ramp_frames))
        for name in VISEME_CHANNELS:
            value = mouth.get(name)
            if value is None:
                value = face.get(name, 0.0) * ramp
            channels[name].append(round(max(0.0, min(1.0, value)), 4))

    # A channel that is zero for the whole line is not written: the schema
    # allows it, and a rig with no such shape key would otherwise be keyed to
    # nothing frame after frame.
    return ({name: values for name, values in channels.items()
             if any(v > 0.0 for v in values)},
            visemes, loudness, known)


# -- running -----------------------------------------------------------------
def run(job, unit):
    params = unit["params"]
    project = job.project
    line_id = params["line_id"]
    audio = unit["inputs"]["audio"]
    fps = int(params.get("fps") or job.fps)

    samples, rate = gates.read_wav_mono(audio)
    if not samples:
        raise ValueError(f"{line_id}: {project.rel(audio)} holds no samples")
    duration = len(samples) / float(rate)
    speech_start, speech_end = gates.speech_bounds(samples, rate)
    if speech_end - speech_start < MIN_PHONEME_S:
        raise ValueError(
            f"{line_id}: no speech found in {project.rel(audio)} — the take is "
            f"{duration:.2f} s and the loudest run in it is under "
            f"{MIN_PHONEME_S * 1000:.0f} ms")

    context = {
        "line_id": line_id,
        "audio": audio,
        "samples": samples,
        "rate": rate,
        "text": params.get("text_fa") or "",
        "language": params.get("language") or "fa",
        "speech_start_s": speech_start,
        "speech_end_s": speech_end,
        "backend": params.get("backend") or "auto",
        "model": (params.get("aligner_model")
                  or os.environ.get("PIPELINE_ALIGN_MODEL")),
    }
    result, skipped = _align(context)

    align = {
        "schema_version": 1,
        "line_id": line_id,
        "backend": result["backend"],
        "backend_model": result.get("backend_model"),
        "language": context["language"],
        "audio": project.rel(audio),
        "sample_rate": rate,
        "duration_s": round(duration, 5),
        "speech_start_s": round(speech_start, 4),
        "speech_end_s": round(speech_end, 4),
        "text": context["text"],
        "confidence": result.get("confidence"),
        "approximate": bool(result.get("approximate")),
        "words": result["words"],
        "phones": result["phones"],
    }
    jsonio.write(_base.out_path(unit, "align.json"), align)

    channels, visemes, loudness, emotion_known = _curves(
        align, samples, rate, fps, params.get("emotion"))
    frame_count = len(next(iter(channels.values()))) if channels else 0
    curves = {
        "schema_version": 1,
        "line_id": line_id,
        "fps": fps,
        "frame_count": frame_count,
        "profile": params.get("viseme_profile") or "smooth",
        "emotion": params.get("emotion") or "neutral",
        "approximate": align["approximate"],
        "channels": channels,
    }
    jsonio.write(_base.out_path(unit, "visemes.json"), curves)

    jaw = channels.get("jaw_open") or []
    lipsync = gates.lip_sync_score(jaw, loudness)
    speech_s = speech_end - speech_start
    covered = sum(p["end_s"] - p["start_s"] for p in align["phones"])
    shortest = min((p["end_s"] - p["start_s"] for p in align["phones"]),
                   default=0.0)

    return {
        "metrics": {
            "line_id": line_id,
            "character": params.get("character"),
            "backend": align["backend"],
            "backend_model": align["backend_model"],
            "approximate": align["approximate"],
            "audio": align["audio"],
            "duration_s": align["duration_s"],
            "speech_start_s": align["speech_start_s"],
            "speech_end_s": align["speech_end_s"],
            "speech_s": round(speech_s, 4),
            "lead_silence_s": round(speech_start, 4),
            "tail_silence_s": round(duration - speech_end, 4),
            "word_count": len(align["words"]),
            "phone_count": len(align["phones"]),
            "guessed_vowels": result.get("guessed_vowels"),
            "boundaries_snapped": result.get("boundaries_snapped"),
            "coverage": round(covered / max(1e-6, speech_s), 4),
            "shortest_phone_s": round(shortest, 4),
            "fps": fps,
            "frame_count": frame_count,
            "viseme_profile": curves["profile"],
            "emotion": curves["emotion"],
            "emotion_known": emotion_known,
            "channels": sorted(channels),
            "peaks": {name: round(max(values), 4)
                      for name, values in sorted(channels.items())},
            "jaw_active_frames": sum(1 for v in jaw if v > 0.05),
            "visemes_used": sorted(set(visemes)),
            "lipsync": lipsync,
            "backends_skipped": skipped,
            # The gate needs the timings themselves, and the artifact directory
            # is gone by the time it runs. Scores are dropped: check_align never
            # reads them and they double the size of the run summary.
            "align": {
                "speech_start_s": align["speech_start_s"],
                "speech_end_s": align["speech_end_s"],
                "approximate": align["approximate"],
                "backend": align["backend"],
                "words": [{k: w[k] for k in ("text", "start_s", "end_s")}
                          for w in align["words"]],
                "phones": [{k: p[k] for k in
                            ("phone", "viseme", "start_s", "end_s")}
                           for p in align["phones"]],
            },
        },
        "notes": [f"{line_id}: {align['backend']} — {len(align['words'])} words, "
                  f"{len(align['phones'])} phones over "
                  f"{speech_s:.2f} s of speech in a {duration:.2f} s take, "
                  f"{frame_count} frames at {fps} fps"]
        + [f"skipped {note}" for note in skipped],
    }


# -- the gate ----------------------------------------------------------------
def _gate(project, unit, metrics):
    out = [v for v in gates.check_align(metrics.get("align") or {})
           if v.level != gates.PASS]

    if not metrics.get("emotion_known"):
        out.append(gates.Verdict(
            gates.WARN, "align.emotion",
            f"{metrics.get('emotion')!r} is not a known emotion; the face is "
            "neutral. Section 7-G makes this field the only source of facial "
            "expression, so an unrecognised one costs the line its acting"))

    peaks = metrics.get("peaks") or {}
    jaw_peak = peaks.get("jaw_open", 0.0)
    if jaw_peak < JAW_PEAK_MIN:
        out.append(gates.Verdict(
            gates.FAIL, "align.jaw",
            f"the jaw never opens past {jaw_peak:.2f}; whatever the phonemes "
            "say, this mouth does not move", round(jaw_peak, 3), JAW_PEAK_MIN))

    frames = metrics.get("frame_count") or 0
    active = metrics.get("jaw_active_frames") or 0
    if frames and active < frames * JAW_ACTIVE_MIN_FRACTION:
        out.append(gates.Verdict(
            gates.WARN, "align.jaw_activity",
            f"the mouth is open in {active} of {frames} frames; the line is "
            "mostly a closed mouth over speech", active,
            int(frames * JAW_ACTIVE_MIN_FRACTION)))

    # The curve was written from this envelope, so a good correlation proves
    # nothing about the acting. A non-zero best lag does prove something: the
    # curve and the sound disagree about which frame is frame one, which is an
    # indexing bug in this stage and is much cheaper to find here than in a
    # render.
    lipsync = metrics.get("lipsync") or {}
    if lipsync.get("usable") and lipsync.get("margin", 1.0) <= 0.0:
        out.append(gates.Verdict(
            gates.FAIL, "align.lipsync",
            f"the jaw curve matches a scrambled version of this line "
            f"({lipsync.get('control_r'):+.2f}) as well as the real one "
            f"({lipsync.get('r_at_zero'):+.2f})",
            lipsync.get("r_at_zero"), lipsync.get("control_r")))
    if lipsync.get("usable") and lipsync.get("best_lag"):
        gain = (lipsync.get("best_r") or 0.0) - (lipsync.get("r_at_zero") or 0.0)
        decisive = max(LAG_DECISIVE_R,
                       LAG_DECISIVE_SHARE * abs(lipsync.get("r_at_zero") or 0.0))
        level = (gates.FAIL if gain > decisive
                 and abs(lipsync["best_lag"]) >= LAG_VISIBLE_FRAMES
                 else gates.WARN)
        out.append(gates.Verdict(
            level, "align.offset",
            f"the jaw curve correlates best {lipsync['best_lag']:+d} frames "
            f"from the envelope it was built from "
            f"({lipsync.get('best_r'):+.2f} against "
            f"{lipsync.get('r_at_zero'):+.2f} as written)"
            + (", so frame zero of visemes.json is not speech_start_s"
               if level == gates.FAIL else
               f"; the peak beats zero by only {gain:.3f}, which is a bumpy "
               "correlation rather than a shift"),
            lipsync["best_lag"], 0))

    expected = int(math.ceil(
        (metrics.get("speech_s") or 0.0) * (metrics.get("fps") or 24) - 1e-9))
    if frames and abs(frames - expected) > 1:
        out.append(gates.Verdict(
            gates.FAIL, "align.frames",
            f"{frames} frames of curve for {expected} frames of speech",
            frames, expected))

    return out or [gates.Verdict(
        gates.PASS, "align",
        f"{metrics.get('line_id')}: {metrics.get('word_count')} words, "
        f"{metrics.get('phone_count')} phones, "
        f"{(metrics.get('coverage') or 0.0):.0%} coverage, "
        f"{frames} frames, jaw peaks {jaw_peak:.2f}")]


STAGE = StageSpec(
    name="g_align",
    version="1",
    summary="align each Persian line to its recording and write the mouth curves",
    # f_voice is named even though it is unwritten: the input path already
    # prefers converted.wav, so the dependency is real the moment it lands.
    needs=["a_production", "f_voice"],
    # The audio environment, not the host. What runs today is stdlib, but the
    # ladder's first two rungs need torch, and declaring the environment the
    # work will need is what makes their "not importable here" honest.
    steps=[Step("audio")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(run))
