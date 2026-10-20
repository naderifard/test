"""Stage F — the character's voice, from the user's performance of the line.

Section 7-F. This is voice *conversion*, not text-to-speech, and that choice
carries three things at once: Persian needs no TTS support, the user's own
prosody and acting survive, and the timing is preserved so the lip sync computed
from the audio stays valid.

The last of those is the whole gate. **A model that resynthesises with new
timing silently invalidates every mouth curve** -- not loudly, not with an
error, just with a film in which everyone is slightly out. So the duration check
is a hard failure at 10 ms and it is checked on the file that was written, not
promised by the model that wrote it.

In M1 no conversion happens. Section 13 says so explicitly: "voice: the user's
raw recording, no conversion". What runs is the ``passthrough`` rung of the same
backend ladder ``g_align`` uses -- Seed-VC first, REF-VC for noisy recordings,
and the pass-through last -- and it copies the recording to ``converted.wav``
verbatim. That is not a placeholder in the disparaging sense. It is the thing
that makes the seam real: every stage downstream already names
``converted.wav``, so the day Seed-VC lands, nothing but this file changes.

Two consequences of the pass-through being an *identity* worth stating:

* the duration gate is trivially satisfied today, and would still catch a
  conversion that broke it tomorrow, because the check reads both files;
* speaker similarity and ASR word error rate are **not reported at all** rather
  than reported as perfect. A stub that claims a similarity of 1.0 is a stub
  that will pass its gate forever. ``gates.check_voice`` skips a metric it is
  not given, and the gate below says out loud which checks did not run.
"""

from __future__ import annotations

import os
import shutil
import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio

# A voice reference under this is too short to characterise a speaker, whatever
# a model claims to do with it.
MIN_VOICE_REF_S = 3.0


class ConverterUnavailable(RuntimeError):
    """One rung of the ladder cannot run here, with the reason why."""


def plan(project, graph):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    config = project.config
    units = []
    for entry in resolved.get("dialogue_index") or []:
        line_id = entry["line_id"]
        character = entry.get("character") or ""
        units.append(Unit(
            f"line_{line_id}",
            inputs={
                "source": project.path("audio", line_id, "source.wav"),
                # The reference voice is hashed even when it does not exist:
                # dropping one in later has to re-key the conversion, and the
                # store records a missing input as missing rather than dropping
                # it.
                "voice_ref": project.voice_ref(character),
            },
            params={
                "line_id": line_id,
                "character": character,
                "backend": config.get("voice_backend") or "auto",
                "language": config.get("language") or "fa",
            },
            out_rel=f"audio/{line_id}",
            label=f"voice/{line_id} ({character})",
        ))
    return units


def nothing_to_do(project):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    if not resolved:
        return "no resolved production plan — a_production has not run"
    return "no dialogue lines in the plan; nothing to convert"


# -- the backend ladder ------------------------------------------------------
def _try_seedvc(context):
    """Section 7-F's first choice: Seed-VC, zero-shot diffusion transformer.

    Needs the character's voice sample: conversion is *to* a voice, and with no
    sample there is nothing to convert to, so the line falls through to the
    pass-through and the gate says so.
    """
    if not context.get("reference"):
        raise ConverterUnavailable(
            f"no voice sample for {context.get('character') or 'this speaker'}"
            f" (audio/voices/<id>.wav), so there is no voice to convert to")
    try:
        from pipeline import voice_seedvc
    except ImportError as exc:
        raise ConverterUnavailable(
            f"seed-vc cannot be imported here ({exc.__class__.__name__}: "
            f"{exc})") from exc
    return voice_seedvc.convert(context["source"], context["reference"],
                                context["target"])


def _try_refvc(context):
    """The fallback section 7-F names for recordings that turn out noisy."""
    raise ConverterUnavailable(
        "ref-vc is not installed; it is the option for noisy recordings, not "
        "the default")


def _try_passthrough(context):
    """M1 (section 13): the user's own recording, unconverted and unretimed."""
    shutil.copyfile(context["source"], context["target"])
    return {
        "backend": "passthrough",
        "backend_model": None,
        "converted": False,
        # Deliberately absent rather than 1.0: see the module docstring.
        "speaker_similarity": None,
        "asr_wer": None,
    }


LADDER = (
    ("seedvc", _try_seedvc),
    ("refvc", _try_refvc),
    ("passthrough", _try_passthrough),
)


def _convert(context):
    wanted = (context.get("backend") or "auto").strip().lower()
    rungs = [(n, f) for n, f in LADDER if wanted in ("auto", n)]
    if not rungs:
        raise ValueError(
            f"voice_backend {wanted!r} names no backend; known backends are "
            f"{', '.join(n for n, _ in LADDER)} (or 'auto')")
    skipped = []
    for name, function in rungs:
        try:
            return function(context), skipped
        except ConverterUnavailable as exc:
            skipped.append(f"{name}: {exc}")
    raise RuntimeError("no voice converter could run:\n  " + "\n  ".join(skipped))


# -- running -----------------------------------------------------------------
def run(job, unit):
    params = unit["params"]
    project = job.project
    line_id = params["line_id"]
    source = unit["inputs"]["source"]
    reference = unit["inputs"].get("voice_ref")
    target = _base.out_path(unit, "converted.wav")

    before = gates.wav_info(source)
    reference_info = None
    if reference and os.path.exists(reference):
        reference_info = gates.wav_info(reference)

    result, skipped = _convert({
        "line_id": line_id,
        "character": params.get("character"),
        "source": source,
        "target": target,
        "reference": reference if reference_info else None,
        "language": params.get("language"),
        "backend": params.get("backend") or "auto",
    })
    after = gates.wav_info(target)

    record = {
        "schema_version": 1,
        "line_id": line_id,
        "character": params.get("character"),
        "backend": result["backend"],
        "backend_model": result.get("backend_model"),
        "converted": bool(result.get("converted")),
        "source": project.rel(source),
        "output": "converted.wav",
        "voice_reference": project.rel(reference) if reference_info else None,
        "in_duration_s": round(before["duration_s"], 5),
        "out_duration_s": round(after["duration_s"], 5),
        "sample_rate": after["sample_rate"],
        "channels": after["channels"],
        "speaker_similarity": result.get("speaker_similarity"),
        "source_similarity": result.get("source_similarity"),
        "asr_wer": result.get("asr_wer"),
        "backends_skipped": skipped,
        "conversion": {k: result.get(k) for k in
                       ("diffusion_steps", "length_fix_samples", "gain",
                        "reference_s", "seconds", "load_s")
                       if result.get(k) is not None},
    }
    jsonio.write(_base.out_path(unit, "voice.json"), record)

    metrics = {
        "line_id": line_id,
        "character": params.get("character"),
        "backend": record["backend"],
        "converted": record["converted"],
        "in_duration_s": record["in_duration_s"],
        "out_duration_s": record["out_duration_s"],
        "duration_delta_ms": round(
            abs(record["out_duration_s"] - record["in_duration_s"]) * 1000, 4),
        "sample_rate": after["sample_rate"],
        "channels": after["channels"],
        "bytes": os.path.getsize(target),
        "has_voice_reference": reference_info is not None,
        "voice_reference_s": (round(reference_info["duration_s"], 3)
                              if reference_info else None),
        "backends_skipped": skipped,
    }
    # Only report a metric that was actually measured. check_voice skips what it
    # is not given, and a stub that claims a perfect score passes forever.
    for name in ("speaker_similarity", "asr_wer"):
        if record[name] is not None:
            metrics[name] = record[name]

    return {
        "metrics": metrics,
        "notes": [f"{line_id}: {record['backend']} — "
                  f"{record['in_duration_s']:.3f} s in, "
                  f"{record['out_duration_s']:.3f} s out, "
                  f"{after['sample_rate']} Hz"]
        + [f"skipped {note}" for note in skipped],
    }


# -- the gate ----------------------------------------------------------------
def _gate(project, unit, metrics):
    out = [v for v in gates.check_voice(metrics) if v.level != gates.PASS]

    if not metrics.get("converted"):
        # Said on every run, on purpose. M4 substitutes a model here, and until
        # it does the film is the user's own voice for every character.
        out.append(gates.Verdict(
            gates.WARN, "voice.passthrough",
            f"{metrics.get('line_id')}: no conversion ran ("
            f"{metrics.get('backend')}); this is the user's own recording, so "
            f"{metrics.get('character')} does not have a voice of his own yet"))
    else:
        for name, what in (("speaker_similarity", "speaker similarity"),
                           ("asr_wer", "intelligibility after conversion")):
            if metrics.get(name) is None:
                out.append(gates.Verdict(
                    gates.WARN, f"voice.{name}",
                    f"a conversion ran and {what} was not measured; §11-F wants "
                    "it checked"))

    if not metrics.get("has_voice_reference"):
        out.append(gates.Verdict(
            gates.WARN, "voice.reference",
            f"no reference voice at audio/voices/{metrics.get('character')}.wav; "
            "there is nothing for a converter to aim at"))
    elif (metrics.get("voice_reference_s") or 0.0) < MIN_VOICE_REF_S:
        out.append(gates.Verdict(
            gates.WARN, "voice.reference_length",
            f"the reference voice is {metrics.get('voice_reference_s'):.1f} s; "
            f"under {MIN_VOICE_REF_S:.0f} s is thin for a speaker embedding",
            metrics.get("voice_reference_s"), MIN_VOICE_REF_S))

    return out or [gates.Verdict(
        gates.PASS, "voice",
        f"{metrics.get('line_id')}: {metrics.get('backend')}, duration "
        f"preserved to {metrics.get('duration_delta_ms'):.1f} ms")]


STAGE = StageSpec(
    name="f_voice",
    version="2",
    summary="convert each recorded line to its character's voice, keeping the timing",
    needs=["a_production"],
    # The audio environment: torch, torchaudio, librosa and the vendored
    # Seed-VC checkout. The pass-through rung still works there when a line has
    # no voice sample to convert to.
    steps=[Step("audio")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(run))
