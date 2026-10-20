"""Stand in for the human recordings until the human records.

M1's contract is that every heavy model is stubbed *except* the alignment path,
and alignment needs real speech to align. A sine tone would let `g_align` run
and prove nothing: forced alignment on a tone produces a degenerate answer that
passes no gate and teaches nothing about the wiring.

So this makes speech. Windows ships SAPI voices; none of them speak Persian, but
a Persian line written out in Latin letters and read by an English voice produces
audio whose phoneme sequence is close enough that the Persian wav2vec2 aligner
has something real to lock onto, and whose envelope is genuinely speech-shaped —
syllable peaks, word gaps, breath at the ends. That is what the lip-sync gate
measures.

The romanisations live in ``audio/placeholder.json`` beside the recordings, keyed
by ``line_id``, because they are a property of this stand-in and not of the film:
``production.json`` has no field for them and should not grow one.

    python tools/placeholder_voice.py shahnameh
    python tools/placeholder_voice.py shahnameh --only sohrab_01 --force

Every file it writes goes to ``audio/<line_id>/source.wav`` — exactly where a
real recording goes — so replacing one is a copy, and the store re-runs only
what depended on it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline import jsonio                                      # noqa: E402
from pipeline.project import Project                             # noqa: E402

RATE = 48000
WIDTH = 2
LEAD_S = 0.35            # real silence for speech_bounds to find and trim
TAIL_S = 0.45

DEFAULT_VOICE = "Microsoft David Desktop"

# A grave male voice for Rostam, a lighter one for Sohrab. SAPI's en-US pair is
# the whole palette available, which is fine: this is a timing stand-in, not a
# performance.
VOICE_BY_EMOTION = {
    "grave": (DEFAULT_VOICE, -2),
    "weary": (DEFAULT_VOICE, -3),
    "resolute": ("Microsoft Zira Desktop", 0),
    "angry": ("Microsoft Zira Desktop", 2),
    "joyful": ("Microsoft Zira Desktop", 1),
}

PS_SCRIPT = """\
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$text = [IO.File]::ReadAllText($args[0], [Text.Encoding]::UTF8)
$out  = $args[1]
$want = $args[2]
$rate = [int]$args[3]
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($want -ne '') {
  $have = $s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }
  if ($have -contains $want) { $s.SelectVoice($want) }
}
$s.Rate = $rate
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    %(rate)d,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile($out, $fmt)
$s.Speak($text)
$s.SetOutputToNull()
Write-Output ('voice=' + $s.Voice.Name)
$s.Dispose()
""" % {"rate": RATE}


def synthesise(text, target, voice, rate):
    """SAPI to a 48 kHz mono 16-bit wav. Returns the voice actually used."""
    script = tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False,
                                         encoding="utf-8")
    words = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                        encoding="utf-8")
    try:
        script.write(PS_SCRIPT)
        script.close()
        words.write(text)
        words.close()
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", script.name, words.name, os.path.abspath(target),
             voice, str(rate)],
            capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or not os.path.exists(target):
            raise RuntimeError((proc.stderr or proc.stdout or "").strip()
                               or "powershell failed")
        for line in (proc.stdout or "").splitlines():
            if line.startswith("voice="):
                return line[6:].strip()
        return voice
    finally:
        for handle in (script, words):
            try:
                os.unlink(handle.name)
            except OSError:
                pass


def pad(path, lead_s=LEAD_S, tail_s=TAIL_S):
    """Bracket the speech with real silence, in place.

    SAPI starts on sample zero. That would make `speech_start_s` exactly 0 and
    leave the `silence_lead_policy` branch in `h_assemble` untested, which is
    the branch most likely to be wrong.
    """
    with wave.open(path, "rb") as fh:
        params = fh.getparams()
        body = fh.readframes(fh.getnframes())
    quiet = b"\x00" * int(params.framerate * params.sampwidth
                          * params.nchannels * lead_s)
    trailing = b"\x00" * int(params.framerate * params.sampwidth
                             * params.nchannels * tail_s)
    with wave.open(path, "wb") as fh:
        fh.setparams(params)
        fh.writeframes(quiet + body + trailing)


def lines_of(project):
    """Every dialogue line in shot order, with its emotion."""
    production = project.production
    out = []
    for shot in production.get("shots") or []:
        for line in shot.get("dialogue") or []:
            out.append({"line_id": line["line_id"],
                        "character": line.get("character", ""),
                        "text_fa": line.get("text_fa", ""),
                        "text_en": line.get("text_en", ""),
                        "emotion": line.get("emotion", "neutral"),
                        "shot": shot["id"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project")
    ap.add_argument("--only", action="append", default=None,
                    help="only these line ids")
    ap.add_argument("--force", action="store_true",
                    help="overwrite recordings that already exist")
    args = ap.parse_args(argv)

    project = Project.open(args.project)
    say_path = project.path("audio", "placeholder.json")
    romanised = jsonio.read(say_path, {})

    made, skipped, missing = [], [], []
    for line in lines_of(project):
        line_id = line["line_id"]
        if args.only and line_id not in args.only:
            continue
        target = project.path("audio", line_id, "source.wav")
        if os.path.exists(target) and not args.force:
            skipped.append(line_id)
            continue
        entry = romanised.get(line_id) or {}
        text = entry.get("say") or line["text_en"]
        if not text:
            missing.append(line_id)
            continue
        voice, rate = VOICE_BY_EMOTION.get(line["emotion"],
                                           (DEFAULT_VOICE, -1))
        voice = entry.get("voice") or voice
        rate = entry.get("rate", rate)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        used = synthesise(text, target, voice, rate)
        pad(target)
        info = wave.open(target, "rb")
        seconds = info.getnframes() / float(info.getframerate())
        info.close()
        made.append((line_id, seconds, used))
        print(f"  {line_id:<14} {seconds:5.2f}s  {used}")

    if skipped:
        print(f"kept {len(skipped)} existing: {', '.join(skipped)}")
    if missing:
        print(f"no romanisation and no text_en for: {', '.join(missing)}")
        print(f"add them to {project.rel(say_path)} as "
              '{"<line_id>": {"say": "..."}}')
    print(f"wrote {len(made)} placeholder recording(s) "
          f"totalling {sum(s for _, s, _ in made):.2f}s")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())


