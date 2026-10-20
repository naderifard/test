"""The quality gates of section 11, measured against signals with known answers.

``gates.py`` is 523 lines of thresholds and DSP that nothing tested. That is the
band the e_motion regression lived in: too specific for the infrastructure
tests, too cheap to bother running end to end. It is also the worst possible
band for an untested module, because a gate that has quietly stopped failing
looks exactly like a pipeline that has started passing.

So the measurement halves are checked against synthesised audio whose answer is
known by construction -- a tone between two silences has its speech bounds on
the page -- and the threshold halves are checked at both sides of every boundary
that can stop a run.
"""

from __future__ import annotations

import math
import os
import shutil
import struct
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import gates                                      # noqa: E402

RATE = 16000


def tone(seconds, frequency=220.0, amplitude=0.5, rate=RATE):
    count = int(seconds * rate)
    return [amplitude * math.sin(2.0 * math.pi * frequency * i / rate)
            for i in range(count)]


def silence(seconds, rate=RATE):
    return [0.0] * int(seconds * rate)


def write_wav(path, samples, rate=RATE, channels=1):
    with wave.open(path, "wb") as fh:
        fh.setnchannels(channels)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, v)) * 32767))
            for v in samples))
    return path


class TestAudioMeasurement(unittest.TestCase):
    """The half that reads a waveform. Every answer is known by construction."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gates-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_wav_info_reads_a_take_off_disk(self):
        path = write_wav(os.path.join(self.tmp, "a.wav"), tone(1.5))
        info = gates.wav_info(path)
        self.assertEqual(info["sample_rate"], RATE)
        self.assertEqual(info["channels"], 1)
        self.assertAlmostEqual(info["duration_s"], 1.5, places=3)

    def test_a_read_round_trips_through_sixteen_bit_pcm(self):
        original = tone(0.2, amplitude=0.5)
        path = write_wav(os.path.join(self.tmp, "b.wav"), original)
        samples, rate = gates.read_wav_mono(path)
        self.assertEqual(rate, RATE)
        self.assertEqual(len(samples), len(original))
        worst = max(abs(a - b) for a, b in zip(samples, original))
        self.assertLess(worst, 1e-3, "quantisation and nothing else")

    def test_stereo_is_folded_to_one_value_per_frame(self):
        path = os.path.join(self.tmp, "c.wav")
        with wave.open(path, "wb") as fh:
            fh.setnchannels(2)
            fh.setsampwidth(2)
            fh.setframerate(RATE)
            fh.writeframes(b"".join(struct.pack("<hh", 16000, 0)
                                    for _ in range(RATE)))
        samples, _rate = gates.read_wav_mono(path)
        self.assertEqual(len(samples), RATE,
                         "one value per frame, not per sample")
        self.assertAlmostEqual(samples[0], 16000 / 32768.0 / 2.0, places=4)

    def test_an_unsupported_sample_width_is_refused_not_guessed(self):
        path = os.path.join(self.tmp, "d.wav")
        with wave.open(path, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(3)
            fh.setframerate(RATE)
            fh.writeframes(b"\x00\x00\x00" * 100)
        with self.assertRaises(ValueError):
            gates.read_wav_mono(path)

    def test_speech_bounds_find_a_tone_between_two_silences(self):
        samples = silence(0.5) + tone(1.0) + silence(0.7)
        start, end = gates.speech_bounds(samples, RATE)
        self.assertAlmostEqual(start, 0.5, delta=0.03)
        self.assertAlmostEqual(end, 1.5, delta=0.03)

    def test_speech_bounds_are_relative_so_a_quiet_take_still_works(self):
        """A fixed dBFS floor would call a quiet recording silent throughout."""
        loud = silence(0.3) + tone(0.8, amplitude=0.8) + silence(0.3)
        quiet = silence(0.3) + tone(0.8, amplitude=0.02) + silence(0.3)
        self.assertEqual(
            [round(v, 2) for v in gates.speech_bounds(loud, RATE)],
            [round(v, 2) for v in gates.speech_bounds(quiet, RATE)])

    def test_an_entirely_silent_take_reports_the_whole_file(self):
        start, end = gates.speech_bounds(silence(1.0), RATE)
        self.assertEqual(start, 0.0)
        self.assertAlmostEqual(end, 1.0, places=3)

    def test_the_envelope_has_exactly_one_value_per_output_frame(self):
        samples = tone(2.0)
        self.assertEqual(len(gates.rms_envelope(samples, RATE, 24, 48)), 48)
        self.assertEqual(len(gates.rms_envelope(samples, RATE, 24)), 48)

    def test_the_envelope_offset_skips_the_lead_in(self):
        samples = silence(1.0) + tone(1.0)
        without = gates.rms_envelope(samples, RATE, 24, frame_count=24)
        shifted = gates.rms_envelope(samples, RATE, 24, frame_count=24,
                                     offset_s=1.0)
        self.assertLess(max(without), 1e-6, "the first second is silent")
        self.assertGreater(min(shifted), 0.1, "the second is not")


class TestLipSyncScore(unittest.TestCase):
    """Section 11's new metric, and the thing it exists to catch."""

    def setUp(self):
        # A plausible envelope: syllables of speech across forty frames.
        self.envelope = [max(0.0, math.sin(i * math.pi / 7.0)) ** 2
                         for i in range(40)]

    def test_a_curve_that_follows_the_audio_scores_high_at_zero_lag(self):
        score = gates.lip_sync_score(list(self.envelope), self.envelope)
        self.assertTrue(score["usable"])
        self.assertGreater(score["r_at_zero"], 0.95)
        self.assertEqual(score["best_lag"], 0)

    def test_a_shifted_curve_is_caught_by_the_lag_search(self):
        """Correlation alone cannot tell 'right' from 'three frames early'."""
        shifted = [0.0, 0.0, 0.0] + list(self.envelope[:-3])
        score = gates.lip_sync_score(shifted, self.envelope)
        self.assertEqual(score["best_lag"], 3,
                         "a positive lag means the jaw trails the audio")
        self.assertGreater(score["best_r"], score["r_at_zero"])

    def test_a_mouth_moving_to_another_line_does_not_correlate(self):
        other = [max(0.0, math.cos(i * math.pi / 5.0)) ** 2 for i in range(40)]
        score = gates.lip_sync_score(other, self.envelope)
        self.assertLess(score["r_at_zero"], gates.LIPSYNC_MIN_R)

    def test_an_empty_curve_is_unusable_rather_than_scored_zero(self):
        self.assertFalse(gates.lip_sync_score([], self.envelope)["usable"])

    def test_the_right_line_beats_every_scrambled_control(self):
        """The self-calibrating half of the metric, on a curve with no hold."""
        score = gates.lip_sync_score(list(self.envelope), self.envelope)
        self.assertGreater(score["margin"], 0.0)
        self.assertGreater(score["r_at_zero"], score["control_r"])

    def test_a_shifted_curve_loses_to_its_own_controls(self):
        """The off-by-N bug §11 names, caught without an absolute threshold."""
        shifted = [0.0] * 6 + list(self.envelope[:-6])
        score = gates.lip_sync_score(shifted, self.envelope)
        self.assertLessEqual(score["margin"], 0.0)

    def test_the_wrong_line_loses_to_its_own_controls(self):
        other = [max(0.0, math.cos(i * math.pi / 5.0)) ** 2 for i in range(40)]
        score = gates.lip_sync_score(other, self.envelope)
        self.assertLessEqual(score["margin"], 0.0)

    def test_a_mouth_that_is_shaped_rather_than_loud_still_beats_control(self):
        """What a phoneme-driven jaw looks like: right timing, wrong amplitude.

        This is the case the old absolute threshold got wrong -- the curve is
        genuinely synced and correlates far below 0.3, because a mouth makes
        shapes rather than volume.
        """
        shapes = [0.9, 0.1, 0.5, 0.2, 0.85, 0.05]
        curve = [shapes[i % len(shapes)] if v > 0.05 else 0.0
                 for i, v in enumerate(self.envelope)]
        score = gates.lip_sync_score(curve, self.envelope)
        self.assertLess(score["r_at_zero"], 0.6)
        self.assertGreater(score["margin"], 0.0,
                           "shaped but correctly placed must still pass")

    def test_pearson_needs_three_points_and_survives_a_flat_series(self):
        self.assertEqual(gates.pearson([1.0, 2.0], [1.0, 2.0]), 0.0)
        self.assertEqual(gates.pearson([1.0] * 8, [1.0, 2.0, 3.0] * 3), 0.0)


class TestThresholds(unittest.TestCase):
    """The half that turns numbers into verdicts, checked either side of each."""

    def levels(self, verdicts):
        return {v.name: v.level for v in verdicts}

    def test_worst_ranks_fail_over_warn_over_pass(self):
        def make(level):
            return gates.Verdict(level, "n", "m")
        self.assertEqual(gates.worst([make(gates.PASS)]), gates.PASS)
        self.assertEqual(gates.worst([make(gates.PASS), make(gates.WARN)]),
                         gates.WARN)
        self.assertEqual(gates.worst([make(gates.WARN), make(gates.FAIL)]),
                         gates.FAIL)

    def test_a_verdict_is_falsy_only_when_it_fails(self):
        self.assertTrue(gates.Verdict(gates.WARN, "n", "m"))
        self.assertFalse(gates.Verdict(gates.FAIL, "n", "m"))

    def test_voice_duration_is_the_load_bearing_gate(self):
        """A conversion that retimes silently invalidates every mouth curve."""
        inside = gates.check_voice({"in_duration_s": 3.0,
                                    "out_duration_s": 3.008})
        self.assertEqual(gates.worst(inside), gates.PASS)
        outside = gates.check_voice({"in_duration_s": 3.0,
                                     "out_duration_s": 3.05})
        self.assertEqual(self.levels(outside)["voice.duration"], gates.FAIL)

    def test_voice_word_error_rate_fails_and_similarity_only_warns(self):
        levels = self.levels(gates.check_voice(
            {"in_duration_s": 1.0, "out_duration_s": 1.0,
             "asr_wer": 0.4, "speaker_similarity": 0.3}))
        self.assertEqual(levels["voice.wer"], gates.FAIL)
        self.assertEqual(levels["voice.similarity"], gates.WARN)

    def align(self, **overrides):
        base = {
            "speech_start_s": 0.0, "speech_end_s": 1.0, "approximate": False,
            "words": [{"text": "a", "start_s": 0.0, "end_s": 0.5},
                      {"text": "b", "start_s": 0.5, "end_s": 1.0}],
            "phones": [
                {"phone": "a", "viseme": "AA", "start_s": 0.0, "end_s": 0.5},
                {"phone": "b", "viseme": "MBP", "start_s": 0.5, "end_s": 1.0}],
        }
        base.update(overrides)
        return base

    def test_a_clean_alignment_passes(self):
        self.assertEqual(gates.worst(gates.check_align(self.align())),
                         gates.PASS)

    def test_an_untimed_word_fails(self):
        bad = self.align(words=[{"text": "a", "start_s": 0.4, "end_s": 0.4}])
        self.assertEqual(self.levels(gates.check_align(bad))["align.words"],
                         gates.FAIL)

    def test_no_phonemes_at_all_fails_and_stops_there(self):
        verdicts = gates.check_align(self.align(phones=[]))
        self.assertEqual(len(verdicts), 1)
        self.assertEqual(verdicts[0].name, "align.phones")

    def test_phonemes_under_twenty_milliseconds_warn(self):
        short = self.align(phones=[
            {"phone": "a", "viseme": "AA", "start_s": 0.0, "end_s": 0.99},
            {"phone": "b", "viseme": "MBP", "start_s": 0.99, "end_s": 1.0}])
        self.assertEqual(
            self.levels(gates.check_align(short))["align.short_phones"],
            gates.WARN)

    def test_an_alignment_covering_under_ninety_percent_fails(self):
        thin = self.align(phones=[
            {"phone": "a", "viseme": "AA", "start_s": 0.0, "end_s": 0.5}])
        self.assertEqual(self.levels(gates.check_align(thin))["align.coverage"],
                         gates.FAIL)

    def test_the_fallback_aligner_always_says_so(self):
        verdicts = gates.check_align(self.align(approximate=True,
                                                backend="envelope"))
        self.assertEqual(self.levels(verdicts)["align.approximate"], gates.WARN)

    def test_flicker_limits_are_per_style_pack(self):
        """One limit could not serve EEVEE's fast path and Cycles at once."""
        flicker = {"usable": True, "mean_delta": 0.017, "max_delta": 0.02}
        miniature = gates.check_render({}, flicker=flicker,
                                       style_pack="miniature")
        photoreal = gates.check_render({}, flicker=flicker,
                                       style_pack="photoreal")
        self.assertEqual(gates.worst(miniature), gates.PASS,
                         "0.017 is inside the fast path's 0.020 warn limit")
        self.assertEqual(self.levels(photoreal)["render.flicker"], gates.WARN,
                         "the same number is over photoreal's 0.010")

    def test_a_boiling_sequence_fails_though_every_still_is_fine(self):
        analysis = {"mean_luminance": 0.4, "contrast": 0.3,
                    "fraction_near_black": 0.1}
        self.assertEqual(
            gates.worst(gates.check_render(analysis, style_pack="stylized")),
            gates.PASS)
        boiling = gates.check_render(
            analysis,
            flicker={"usable": True, "mean_delta": 0.09, "max_delta": 0.2},
            style_pack="stylized")
        self.assertEqual(self.levels(boiling)["render.flicker"], gates.FAIL)

    def test_a_mouth_no_better_than_a_scrambled_control_fails(self):
        """The check that decides: is this curve tied to *this* line?"""
        verdicts = gates.check_render(
            {"mean_luminance": 0.4, "contrast": 0.3},
            lipsync={"usable": True, "r_at_zero": 0.21, "best_r": 0.3,
                     "best_lag": 0, "control_r": 0.26})
        self.assertEqual(self.levels(verdicts)["render.lipsync"], gates.FAIL)

    def test_a_mouth_that_barely_moves_fails_even_if_it_beats_the_control(self):
        verdicts = gates.check_render(
            {"mean_luminance": 0.4},
            lipsync={"usable": True, "r_at_zero": 0.05, "best_r": 0.05,
                     "best_lag": 0, "control_r": -0.4})
        self.assertEqual(self.levels(verdicts)["render.lipsync"], gates.FAIL)

    def test_a_correct_but_held_mouth_passes_on_its_margin_not_its_score(self):
        """0.24 against a 0.17 control is what §8's snap profile measures."""
        verdicts = gates.check_render(
            {"mean_luminance": 0.4, "contrast": 0.3},
            lipsync={"usable": True, "r_at_zero": 0.24, "best_r": 0.30,
                     "best_lag": -2, "control_r": 0.166})
        self.assertEqual(gates.worst(verdicts), gates.WARN,
                         "a wandering argmax on a held curve is a note")
        self.assertEqual(self.levels(verdicts)["render.lipsync_offset"],
                         gates.WARN)

    def test_a_render_error_short_circuits_every_other_check(self):
        verdicts = gates.check_render({"render_error": "OptiX out of memory",
                                       "mean_luminance": 0.0})
        self.assertEqual(len(verdicts), 1)
        self.assertEqual(verdicts[0].name, "render.error")

    def test_a_black_frame_fails_and_a_blown_out_one_warns(self):
        self.assertEqual(
            self.levels(gates.check_render(
                {"mean_luminance": 0.01}))["render.dark"], gates.FAIL)
        self.assertEqual(
            self.levels(gates.check_render(
                {"mean_luminance": 0.95}))["render.bright"], gates.WARN)

    def test_assembly_names_the_objects_without_a_material(self):
        verdicts = gates.check_assembly(
            {"objects_without_material": ["rock.003", "backdrop_far"]})
        self.assertEqual(self.levels(verdicts)["assembly.materials"],
                         gates.FAIL)
        self.assertIn("rock.003", verdicts[0].message)

    def test_assembly_fails_a_floating_character_and_an_empty_frame(self):
        levels = self.levels(gates.check_assembly({
            "characters_off_ground": {"rostam": 0.31},
            "subjects_in_frustum": False}))
        self.assertEqual(levels["assembly.ground"], gates.FAIL)
        self.assertEqual(levels["assembly.frustum"], gates.FAIL)

    def test_motion_nan_fails_and_foot_skate_warns(self):
        levels = self.levels(gates.check_motion(
            {"nan_count": 2, "frames": 96, "foot_skate_m_per_frame": 0.05}))
        self.assertEqual(levels["motion.nan"], gates.FAIL)
        self.assertEqual(levels["motion.skate"], gates.WARN)

    def test_a_reference_image_too_small_or_blank_fails(self):
        levels = self.levels(gates.check_refs("rostam", [
            {"path": "a.png", "width": 800, "height": 600, "stddev_luma": 0.3},
            {"path": "b.png", "width": 2048, "height": 2048,
             "stddev_luma": 0.001},
        ]))
        self.assertEqual(levels["refs.resolution"], gates.FAIL)
        self.assertEqual(levels["refs.blank"], gates.FAIL)

    def test_no_references_yet_is_a_warning_not_a_failure(self):
        """refs/ is empty by design in M1; this must never stop a run."""
        self.assertEqual(gates.worst(gates.check_refs("rostam", [])),
                         gates.WARN)


class TestFlickerMeasurement(unittest.TestCase):
    """Needs Pillow; skipped rather than silently passing without it."""

    def setUp(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed in this interpreter")
        self.Image = Image
        self.tmp = tempfile.mkdtemp(prefix="flicker-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def frames(self, values):
        paths = []
        for index, value in enumerate(values):
            path = os.path.join(self.tmp, "f%03d.png" % index)
            self.Image.new("RGB", (64, 36), (value, value, value)).save(path)
            paths.append(path)
        return paths

    def test_an_identical_sequence_does_not_flicker(self):
        result = gates.temporal_flicker(self.frames([128] * 6))
        self.assertTrue(result["usable"])
        self.assertEqual(result["mean_delta"], 0.0)

    def test_alternating_frames_measure_the_swing_between_them(self):
        result = gates.temporal_flicker(self.frames([100, 120] * 4))
        self.assertAlmostEqual(result["mean_delta"], 20 / 255.0, places=3)

    def test_one_frame_cannot_be_measured_and_says_so(self):
        self.assertFalse(gates.temporal_flicker(self.frames([128]))["usable"])


if __name__ == "__main__":
    unittest.main()


class TestBodyHeight(unittest.TestCase):
    """§11's one height range is a range for people.

    Applied to the whole cast it fires on every animal in the film forever, and
    a warning that is always wrong is one people learn to scroll past. The
    Keyumars plan — a 0.9 m gazelle, a 1.3 m lion, a 2.6 m div — is what found
    it.
    """

    def level(self, height, kind):
        return gates.check_body({"height_m": height, "kind": kind})[0].level

    def test_a_heroic_but_human_height_passes(self):
        self.assertEqual(self.level(2.1, "human"), gates.PASS)

    def test_an_impossible_human_still_warns(self):
        self.assertEqual(self.level(4.0, "human"), gates.WARN)
        self.assertEqual(self.level(0.4, "human"), gates.WARN)

    def test_a_gazelle_and_a_lion_are_not_warnings(self):
        self.assertEqual(self.level(0.9, "quadruped"), gates.PASS)
        self.assertEqual(self.level(1.3, "quadruped"), gates.PASS)

    def test_a_div_taller_than_any_man_is_not_a_warning(self):
        self.assertEqual(self.level(2.6, "creature"), gates.PASS)

    def test_an_unknown_kind_gets_the_widest_range_rather_than_the_human_one(self):
        self.assertEqual(self.level(0.9, "chimera"), gates.PASS)

    def test_every_declared_kind_has_a_range(self):
        from pipeline import validate
        for kind in validate.KIND_RIG:
            self.assertIn(kind, gates.HEIGHT_RANGE_M, kind)


class TestHeadBoxSelection(unittest.TestCase):
    """Which cascade detection is the head, given that most of them are not.

    Every box below is one this project's own reference images actually
    produced. The cascade is enthusiastic about anything with two dark patches
    above a light one, so choosing is the whole job: the first version of this
    took the largest box and gave the black div a claw for a head.
    """

    HEIGHT = 1536

    def test_the_largest_box_is_not_the_head(self):
        """black_div: a 54 px face at 26%, two ~145 px claws at 64 and 65%."""
        boxes = [(526, 370, 54, 54), (102, 915, 142, 142),
                 (791, 918, 147, 147)]
        self.assertEqual((526, 370, 54, 54),
                         gates.choose_head_box(boxes, self.HEIGHT))
        self.assertNotEqual(max(boxes, key=lambda b: b[2] * b[3]),
                            gates.choose_head_box(boxes, self.HEIGHT))

    def test_the_real_face_wins_over_three_false_positives_below_it(self):
        """siamak: the face at 10%, then leopard-skin patches at 57-73%."""
        boxes = [(439, 85, 131, 131), (451, 857, 50, 50),
                 (356, 990, 107, 107), (360, 1084, 89, 89)]
        self.assertEqual((439, 85, 131, 131),
                         gates.choose_head_box(boxes, self.HEIGHT))

    def test_detections_that_are_all_too_low_give_no_head_at_all(self):
        """gazelle: two boxes at 84% of height, which are its hooves."""
        boxes = [(116, 1222, 135, 135), (254, 1230, 128, 128)]
        self.assertIsNone(gates.choose_head_box(boxes, self.HEIGHT))

    def test_a_single_detection_high_in_the_frame_is_taken(self):
        boxes = [(448, 197, 104, 104)]
        self.assertEqual(boxes[0], gates.choose_head_box(boxes, self.HEIGHT))

    def test_nothing_detected_is_none_not_an_error(self):
        self.assertIsNone(gates.choose_head_box([], self.HEIGHT))
        self.assertIsNone(gates.choose_head_box(None, self.HEIGHT))

    def test_the_rule_is_the_topmost_centre_not_the_topmost_edge(self):
        """Two boxes starting at the same y are not tied: centres differ.

        This test first asserted that the fuller of two overlapping detections
        wins, which is not what the code does and not a case any of the six
        references produce -- after the height filter, exactly one box survives
        on every one of them. The rule is the topmost centre; the area
        comparison only breaks an exact tie.
        """
        boxes = [(450, 100, 90, 90), (445, 100, 130, 130)]
        self.assertEqual((450, 100, 90, 90),
                         gates.choose_head_box(boxes, self.HEIGHT))

    def test_an_exact_tie_in_centre_is_broken_by_area(self):
        boxes = [(450, 120, 100, 100), (300, 120, 140, 100)]
        self.assertEqual((300, 120, 140, 100),
                         gates.choose_head_box(boxes, self.HEIGHT))

    def test_the_threshold_separates_this_project_s_faces_from_its_lies(self):
        """26% is the lowest true face and 39% the highest false one."""
        self.assertGreater(gates.FACE_MAX_HEIGHT_FRACTION, 0.26)
        self.assertLess(gates.FACE_MAX_HEIGHT_FRACTION, 0.39)
