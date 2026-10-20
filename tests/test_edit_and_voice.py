"""The two stages that bracket the render: voice conversion and the cut.

Both are mostly plumbing today -- section 13 stubs the voice model and the cut
is arithmetic over numbers three earlier stages measured -- which is exactly
why they are worth testing now rather than later. A backend ladder that silently
picks the wrong rung, or a subtitle timecode that is right for the first shot
and wrong for the third, are both invisible until someone watches the film.
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import shutil
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.stages import f_voice, k_edit                      # noqa: E402


class TestSubtitleTimecodes(unittest.TestCase):
    """SubRip is unforgiving about its format and silent when it is wrong."""

    def test_the_timecode_format_is_hours_minutes_seconds_milliseconds(self):
        self.assertEqual(k_edit._timecode(0.0), "00:00:00,000")
        self.assertEqual(k_edit._timecode(0.5415), "00:00:00,542")
        self.assertEqual(k_edit._timecode(61.25), "00:01:01,250")
        self.assertEqual(k_edit._timecode(3725.007), "01:02:05,007")

    def test_a_negative_time_clamps_rather_than_writing_a_minus_sign(self):
        self.assertEqual(k_edit._timecode(-2.0), "00:00:00,000")

    def test_milliseconds_round_rather_than_truncate(self):
        self.assertEqual(k_edit._timecode(1.9996), "00:00:02,000")

    def test_a_cue_carries_the_right_to_left_mark(self):
        """A player guessing direction from the first strong character must
        not be able to get a line opening with a digit wrong."""
        text = k_edit._srt([{"start_s": 0.5, "end_s": 3.0,
                             "text": "نام تو چیست؟"}])
        self.assertIn(k_edit.RLM, text)
        self.assertTrue(text.split("\n")[2].startswith(k_edit.RLM))

    def test_the_text_is_stored_logically_and_not_reversed(self):
        """Reversing looks right in a naive viewer and is wrong everywhere."""
        original = "نام تو چیست؟"
        text = k_edit._srt([{"start_s": 0.0, "end_s": 1.0, "text": original}])
        self.assertIn(original, text)

    def test_cues_are_numbered_from_one_and_separated_by_a_blank_line(self):
        text = k_edit._srt([
            {"start_s": 0.0, "end_s": 1.0, "text": "یک"},
            {"start_s": 2.0, "end_s": 3.0, "text": "دو"},
        ])
        # index, timecode, text, blank, index, ...
        lines = text.split("\n")
        self.assertEqual(lines[0], "1")
        self.assertEqual(lines[3], "")
        self.assertEqual(lines[4], "2")

    def test_an_empty_cut_writes_an_empty_file_rather_than_a_broken_one(self):
        self.assertEqual(k_edit._srt([]), "")


class TestVoiceLadder(unittest.TestCase):
    """Which rung ran has to be recorded, and the reasons have to be real."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="voice-")
        self.source = os.path.join(self.tmp, "source.wav")
        with wave.open(self.source, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(16000)
            fh.writeframes(struct.pack("<h", 1000) * 16000)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def context(self, backend="auto"):
        return {"line_id": "l1", "source": self.source,
                "target": os.path.join(self.tmp, "converted.wav"),
                "reference": None, "language": "fa", "backend": backend}

    def test_auto_falls_through_to_the_passthrough_and_says_what_it_skipped(self):
        result, skipped = f_voice._convert(self.context())
        self.assertEqual(result["backend"], "passthrough")
        self.assertFalse(result["converted"])
        self.assertEqual(len(skipped), 2, skipped)
        self.assertTrue(any("seed" in note for note in skipped))

    def test_the_passthrough_copies_the_recording_byte_for_byte(self):
        """The duration gate is only meaningful if this really is an identity."""
        result, _skipped = f_voice._convert(self.context())
        with open(self.source, "rb") as fh:
            original = fh.read()
        with open(self.context()["target"], "rb") as fh:
            copied = fh.read()
        self.assertEqual(original, copied)
        self.assertIsNone(result["speaker_similarity"],
                          "a stub must not claim a score it did not measure")
        self.assertIsNone(result["asr_wer"])

    def test_naming_a_backend_that_cannot_run_fails_rather_than_degrading(self):
        with self.assertRaises(RuntimeError) as caught:
            f_voice._convert(self.context(backend="seedvc"))
        self.assertIn("seed", str(caught.exception))

    def test_an_unknown_backend_name_is_refused_with_the_list(self):
        with self.assertRaises(ValueError) as caught:
            f_voice._convert(self.context(backend="rvc"))
        self.assertIn("passthrough", str(caught.exception))


class TestVoiceGate(unittest.TestCase):

    def levels(self, verdicts):
        return {v.name: v.level for v in verdicts}

    def base(self, **overrides):
        metrics = {
            "line_id": "rostam_01", "character": "rostam",
            "backend": "passthrough", "converted": False,
            "in_duration_s": 5.0, "out_duration_s": 5.0,
            "duration_delta_ms": 0.0,
            "has_voice_reference": True, "voice_reference_s": 12.0,
        }
        metrics.update(overrides)
        return metrics

    def test_the_passthrough_says_so_on_every_run(self):
        """M1's stub must never look like a finished stage."""
        levels = self.levels(f_voice._gate(None, None, self.base()))
        self.assertEqual(levels["voice.passthrough"], "warn")

    def test_a_conversion_that_changed_the_length_fails(self):
        verdicts = f_voice._gate(None, None, self.base(
            converted=True, backend="seedvc", out_duration_s=5.2,
            speaker_similarity=0.8, asr_wer=0.1))
        self.assertEqual(self.levels(verdicts)["voice.duration"], "fail")

    def test_a_conversion_that_measured_nothing_is_told_to_measure(self):
        verdicts = f_voice._gate(None, None, self.base(converted=True,
                                                       backend="seedvc"))
        levels = self.levels(verdicts)
        self.assertEqual(levels["voice.speaker_similarity"], "warn")
        self.assertEqual(levels["voice.asr_wer"], "warn")

    def test_a_missing_reference_voice_is_a_warning(self):
        verdicts = f_voice._gate(None, None,
                                 self.base(has_voice_reference=False,
                                           voice_reference_s=None))
        self.assertEqual(self.levels(verdicts)["voice.reference"], "warn")

    def test_a_reference_too_short_to_characterise_a_speaker_warns(self):
        verdicts = f_voice._gate(None, None, self.base(voice_reference_s=1.2))
        self.assertEqual(self.levels(verdicts)["voice.reference_length"],
                         "warn")


class TestEditGate(unittest.TestCase):

    def levels(self, verdicts):
        return {v.name: v.level for v in verdicts}

    def base(self, **overrides):
        metrics = {
            "shots": 3, "frame_count": 305, "built_frames": 305,
            "duration_s": 12.7, "cues": 2,
            "subtitled_lines": ["sohrab_01", "rostam_01"],
            "video_bytes": 4_000_000, "audio_strips": 2,
            "shot_audio": {"s01": True, "s02": True, "s03": False},
            "missing_video": [], "edit_warnings": [],
        }
        metrics.update(overrides)
        return metrics

    def test_a_clean_cut_passes_with_a_silent_shot_noted(self):
        verdicts = k_edit._gate(None, None, self.base())
        self.assertEqual(self.levels(verdicts)["edit.silent_shots"], "warn")

    def test_a_film_shorter_than_its_own_cut_fails(self):
        verdicts = k_edit._gate(None, None, self.base(built_frames=200))
        self.assertEqual(self.levels(verdicts)["edit.frames"], "fail")

    def test_dialogue_in_the_cut_and_no_sound_in_the_timeline_fails(self):
        """Section 15-4's defect, one level up: a silent film."""
        verdicts = k_edit._gate(None, None, self.base(audio_strips=0))
        self.assertEqual(self.levels(verdicts)["edit.audio"], "fail")

    def test_a_shot_with_no_rendered_video_fails_by_name(self):
        verdicts = k_edit._gate(None, None, self.base(missing_video=["s02"]))
        self.assertEqual(self.levels(verdicts)["edit.source"], "fail")
        self.assertIn("s02", verdicts[0].message)

    def test_an_empty_encode_fails(self):
        verdicts = k_edit._gate(None, None, self.base(video_bytes=0))
        self.assertEqual(self.levels(verdicts)["edit.video"], "fail")


if __name__ == "__main__":
    unittest.main()
