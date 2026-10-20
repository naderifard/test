"""The master clock, and the arithmetic that places everything against it.

Section 5.2 makes the dialogue audio decide how long a shot is, and everything
else in the shot -- the audio strip, the mouth curves, the beats, the camera
move -- is placed against that one number. All of it is pure host arithmetic
resolved before Blender opens, which makes it both easy to test and expensive to
get wrong: an off-by-one here is a shot whose mouth runs a frame ahead of its
voice, and the cheapest place to catch that is here rather than in a render.

The three clocks h_assemble has to keep straight are shot time, the speech
inside it, and the recording the speech came from. The tests below check the
places they meet.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.stages import h_assemble                            # noqa: E402

FPS = 24

# A recording with half a second of room tone before the line and a second
# after it -- which is what a real take looks like and why trimming exists.
ALIGN = {
    "line_id": "sohrab_01",
    "audio": "audio/sohrab_01/source.wav",
    "duration_s": 4.0,
    "speech_start_s": 0.5,
    "speech_end_s": 3.0,
}


def params(**overrides):
    base = {
        "shot_id": "s01",
        "fps": FPS,
        "head_handle_s": 0.5,
        "tail_handle_s": 0.7,
        "min_shot_s": 0.0,
        "silence_lead_policy": "trim",
        "silent_duration_s": None,
        "dialogue": [{"line_id": "sohrab_01", "character": "sohrab",
                      "text_fa": "...", "starts_at_s": None}],
        "camera": {},
        "cast": [],
        "props": {},
    }
    base.update(overrides)
    return base


class TestTiming(unittest.TestCase):
    """frames = ceil((speech_end - speech_start + head + tail) * fps)."""

    def test_the_shot_is_the_speech_plus_its_two_handles(self):
        timing, placements = h_assemble._timing(params(),
                                                {"sohrab_01": ALIGN})
        self.assertEqual(timing["source"], "audio")
        # 2.5 s of speech, half a second in front, seven tenths behind.
        self.assertAlmostEqual(timing["duration_s"], 3.7, places=5)
        self.assertAlmostEqual(timing["speech_start_s"], 0.5, places=5)
        self.assertAlmostEqual(timing["speech_end_s"], 3.0, places=5)
        self.assertEqual(len(placements), 1)

    def test_the_frame_count_is_section_five_two_to_the_letter(self):
        timing, _ = h_assemble._timing(params(), {"sohrab_01": ALIGN})
        expected = math.ceil((ALIGN["speech_end_s"] - ALIGN["speech_start_s"]
                              + 0.5 + 0.7) * FPS)
        self.assertEqual(math.ceil(timing["duration_s"] * FPS - 1e-9), expected)

    def test_the_lead_in_of_the_recording_does_not_lengthen_the_shot(self):
        """A take with a long run-up must not make the shot longer."""
        long_lead = dict(ALIGN, speech_start_s=2.0, speech_end_s=4.5,
                         duration_s=6.0)
        timing, _ = h_assemble._timing(params(), {"sohrab_01": long_lead})
        self.assertAlmostEqual(timing["duration_s"], 3.7, places=5,
                               msg="only the speech counts, not the silence")

    def test_a_silent_shot_uses_its_declared_duration(self):
        timing, placements = h_assemble._timing(
            params(dialogue=[], silent_duration_s=4.0), {})
        self.assertEqual(timing["source"], "silent_duration")
        self.assertEqual(timing["duration_s"], 4.0)
        self.assertEqual(placements, [])

    def test_a_silent_shot_without_a_duration_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            h_assemble._timing(params(dialogue=[]), {})
        self.assertIn("silent_duration_s", str(caught.exception))

    def test_min_shot_s_pads_the_end_and_records_that_it_did(self):
        timing, _ = h_assemble._timing(params(min_shot_s=5.0),
                                       {"sohrab_01": ALIGN})
        self.assertAlmostEqual(timing["duration_s"], 5.0, places=5)
        self.assertAlmostEqual(timing["padded_s"], 1.3, places=5)

    def test_two_lines_in_one_shot_do_not_run_into_each_other(self):
        second = dict(ALIGN, line_id="sohrab_02", speech_start_s=0.0,
                      speech_end_s=1.0, duration_s=1.0)
        timing, placements = h_assemble._timing(
            params(dialogue=[
                {"line_id": "sohrab_01", "character": "s", "starts_at_s": None},
                {"line_id": "sohrab_02", "character": "s", "starts_at_s": None}]),
            {"sohrab_01": ALIGN, "sohrab_02": second})
        gap = placements[1]["start_s"] - (placements[0]["start_s"]
                                          + placements[0]["speech_s"])
        self.assertAlmostEqual(gap, h_assemble.INTER_LINE_GAP_S, places=5)
        self.assertAlmostEqual(timing["speech_end_s"],
                               placements[1]["start_s"] + 1.0, places=5)

    def test_an_explicit_starts_at_s_wins_over_the_running_cursor(self):
        _timing, placements = h_assemble._timing(
            params(dialogue=[{"line_id": "sohrab_01", "character": "s",
                              "starts_at_s": 1.25}]),
            {"sohrab_01": ALIGN})
        self.assertAlmostEqual(placements[0]["start_s"], 1.25, places=5)


class TestAudioPlacement(unittest.TestCase):
    """Where the recording sits, in the muxer's own units."""

    def tracks(self, **overrides):
        setup = params(**overrides)
        _timing, placements = h_assemble._timing(setup, {"sohrab_01": ALIGN})
        return h_assemble._audio_tracks(setup, placements, 1, None)

    def test_trim_cuts_the_lead_in_and_starts_the_strip_at_the_handle(self):
        tracks, notes = self.tracks()
        track = tracks[0]
        self.assertEqual(track["start_frame"], 1 + round(0.5 * FPS))
        self.assertAlmostEqual(track["trim_start_s"], 0.5, places=5)
        self.assertAlmostEqual(track["duration_s"], 2.5, places=5)
        self.assertEqual(notes, [])

    def test_the_strip_and_the_mouth_start_on_the_same_frame(self):
        """The check section 11 makes is only meaningful if these agree."""
        setup = params()
        _timing, placements = h_assemble._timing(setup, {"sohrab_01": ALIGN})
        tracks, _notes = h_assemble._audio_tracks(setup, placements, 1, None)
        viseme_start = 1 + int(round(placements[0]["start_s"] * FPS))
        self.assertEqual(tracks[0]["start_frame"], viseme_start)

    def test_keep_places_the_strip_early_so_the_speech_still_lands_on_time(self):
        early = dict(ALIGN, speech_start_s=0.2)
        setup = params(silence_lead_policy="keep")
        _timing, placements = h_assemble._timing(setup, {"sohrab_01": early})
        tracks, notes = h_assemble._audio_tracks(setup, placements, 1, None)
        # Speech starts at 0.5 s of shot time and 0.2 s into the take, so the
        # strip has to begin at 0.3 s.
        self.assertEqual(tracks[0]["start_frame"], 1 + round(0.3 * FPS))
        self.assertEqual(tracks[0]["trim_start_s"], 0.0)
        self.assertEqual(notes, [])

    def test_keep_clips_rather_than_sliding_the_speech_when_it_will_not_fit(self):
        """A lead-in longer than the handle cannot start before frame one."""
        # 0.9 s of room tone in front of a 0.5 s handle: 0.4 s has nowhere to go.
        slow = dict(ALIGN, speech_start_s=0.9)
        setup = params(silence_lead_policy="keep")
        _timing, placements = h_assemble._timing(setup, {"sohrab_01": slow})
        tracks, notes = h_assemble._audio_tracks(setup, placements, 1, None)
        self.assertEqual(tracks[0]["start_frame"], 1,
                         "the strip starts at frame one, not before it")
        self.assertAlmostEqual(tracks[0]["trim_start_s"], 0.4, places=5)
        self.assertTrue(notes, "clipping a take is never done silently")
        self.assertIn("clipped", notes[0])


class TestStaging(unittest.TestCase):

    def test_a_point_rotates_about_the_character_before_it_is_moved(self):
        # A quarter turn takes the local +Y axis onto world -X.
        placed = h_assemble._place((0.0, 1.0, 0.0), (5.0, 0.0, 0.0), 90.0)
        self.assertAlmostEqual(placed[0], 4.0, places=4)
        self.assertAlmostEqual(placed[1], 0.0, places=4)

    def test_height_is_unaffected_by_facing(self):
        for facing in (0.0, 37.0, 105.0, -75.0, 180.0):
            self.assertAlmostEqual(
                h_assemble._place((0.3, 0.4, 1.7), (1.0, 2.0, 0.0), facing)[2],
                1.7, places=5)

    def manifest(self):
        return {
            "dims": {"factor": 1.2},
            "joints": {"head": [0.0, 0.0, 1.86], "spine3": [0.0, 0.0, 1.52],
                       "pelvis": [0.0, 0.0, 1.13],
                       "left_wrist": [1.0, 0.0, 1.71],
                       "right_wrist": [-1.0, 0.0, 1.71]},
        }

    def member(self, position=(0.0, 0.0, 0.0), facing=0.0):
        return {"position_m": list(position), "facing_deg": facing,
                "height_m": 2.1}

    def test_an_anchor_resolves_to_the_joint_that_carries_it(self):
        point, problem = h_assemble._anchor_point(self.manifest(),
                                                  self.member(), "head")
        self.assertIsNone(problem)
        self.assertAlmostEqual(point[2], 1.86, places=4)

    def test_chest_falls_through_the_names_the_two_rigs_use(self):
        """A person has spine3 and a horse has chest; both are "chest"."""
        person, _ = h_assemble._anchor_point(self.manifest(), self.member(),
                                             "chest")
        self.assertAlmostEqual(person[2], 1.52, places=4)
        horse = {"dims": {"factor": 1.0},
                 "joints": {"chest": [0.0, 1.18, 1.21], "root": [0, 0, 1.14]}}
        point, problem = h_assemble._anchor_point(horse, self.member(), "chest")
        self.assertIsNone(problem)
        self.assertAlmostEqual(point[2], 1.21, places=4)

    def test_hands_averages_so_a_two_handed_grip_aims_between_them(self):
        point, _ = h_assemble._anchor_point(self.manifest(), self.member(),
                                            "hands")
        self.assertAlmostEqual(point[0], 0.0, places=4)

    def test_eyes_sit_above_and_in_front_of_the_head_joint(self):
        eyes, _ = h_assemble._anchor_point(self.manifest(), self.member(),
                                           "eyes")
        head, _ = h_assemble._anchor_point(self.manifest(), self.member(),
                                           "head")
        self.assertGreater(eyes[2], head[2])
        self.assertGreater(eyes[1], head[1])

    def test_an_anchor_no_rig_carries_reports_what_it_looked_for(self):
        point, problem = h_assemble._anchor_point({"joints": {}}, self.member(),
                                                  "head")
        self.assertIsNone(point)
        self.assertIn("head", problem)

    def test_the_aim_falls_back_to_mid_height_rather_than_to_the_floor(self):
        setup = params(camera={"aim": "rostam.head"},
                       cast=[dict(self.member(), character="rostam",
                                  props=[])])
        point, note = h_assemble._aim(setup, {"rostam": {"joints": {}}}, {})
        self.assertIsNotNone(note)
        self.assertGreater(point[2], 1.0)

    def test_aiming_at_someone_offstage_says_so_instead_of_guessing(self):
        setup = params(camera={"aim": "sohrab.head"}, cast=[])
        _point, note = h_assemble._aim(setup, {}, {})
        self.assertIn("not in this shot", note)


class TestPhase(unittest.TestCase):
    """Two characters sharing one idle take must not breathe in unison."""

    def test_two_characters_get_different_phases_into_a_shared_take(self):
        self.assertNotEqual(h_assemble._phase("rostam", 96),
                            h_assemble._phase("sohrab", 96))

    def test_a_phase_is_inside_the_cycle(self):
        for name in ("rostam", "sohrab", "rakhsh", "div_e_sepid", "simorgh"):
            self.assertTrue(0 <= h_assemble._phase(name, 96) < 96, name)

    def test_a_phase_is_stable_across_processes(self):
        """Python's str hash is salted per run; a film cannot be.

        The value is pinned rather than merely compared with itself, because
        "stable" here means stable across the interpreter restarts that every
        stage subprocess is.
        """
        self.assertEqual(h_assemble._phase("rostam", 96), 89)
        self.assertEqual(h_assemble._phase("sohrab", 96), 70)

    def test_a_take_with_no_cycle_gets_no_phase(self):
        self.assertEqual(h_assemble._phase("rostam", 0), 0)


class Pack:
    """Just the camera policy; loading a real pack needs no Blender but this
    keeps the arithmetic under test rather than the pack file."""

    def __init__(self, **camera):
        self.camera = {"focal_mm": 45, "focal_max_mm": 85.0, "dof": False,
                       "sensor_mm": 36, "orthographic_bias": 0.0}
        self.camera.update(camera)


class TestCamera(unittest.TestCase):

    def resolve(self, pack, camera, aim=(0.0, 0.0, 1.5), frames=96):
        return h_assemble._camera(params(camera=camera), pack, {}, list(aim),
                                  1, frames)

    def test_a_pack_with_no_bias_leaves_the_shot_where_it_was_written(self):
        camera, compression = self.resolve(
            Pack(), {"position_m": [0.0, -6.0, 1.5], "focal_mm": 50})
        self.assertEqual(compression, 1.0)
        self.assertEqual(camera["position_m"], [0.0, -6.0, 1.5])
        self.assertEqual(camera["focal_mm"], 50)

    def test_a_near_ortho_pack_pulls_back_and_lengthens_by_the_same_factor(self):
        """Subject size is preserved; only the perspective flattens."""
        camera, compression = self.resolve(
            Pack(orthographic_bias=0.25),
            {"position_m": [0.0, -6.0, 1.5], "focal_mm": 50})
        self.assertAlmostEqual(compression, 1.7, places=4)
        self.assertAlmostEqual(camera["focal_mm"], 85.0, places=3)
        self.assertAlmostEqual(camera["focus_distance_m"], 6.0 * 1.7, places=3)

    def test_the_pack_focal_ceiling_is_section_eight_s_table_enforced(self):
        """A shot already at 85 mm has asked for the flat look; leave it."""
        camera, compression = self.resolve(
            Pack(orthographic_bias=0.25),
            {"position_m": [0.0, -6.0, 1.5], "focal_mm": 85})
        self.assertEqual(compression, 1.0)
        self.assertEqual(camera["focal_mm"], 85)

    def test_focus_distance_is_the_distance_to_what_the_camera_looks_at(self):
        camera, _ = self.resolve(
            Pack(dof=True), {"position_m": [0.0, -4.0, 1.5], "focal_mm": 50},
            aim=(0.0, 0.0, 1.5))
        self.assertTrue(camera["dof"])
        self.assertAlmostEqual(camera["focus_distance_m"], 4.0, places=4)

    def test_a_static_move_is_no_move_at_all(self):
        camera, _ = self.resolve(
            Pack(), {"position_m": [0.0, -6.0, 1.5], "focal_mm": 50,
                     "move": {"kind": "static"}})
        self.assertIsNone(camera["move"])

    def test_a_push_in_is_compressed_with_the_camera_that_makes_it(self):
        """Compressing only the start turns a 1 m push into a 1.7 m lunge."""
        camera, compression = self.resolve(
            Pack(orthographic_bias=0.25),
            {"position_m": [0.0, -6.0, 1.5], "focal_mm": 50,
             "move": {"kind": "push_in", "to_position_m": [0.0, -5.0, 1.5]}})
        move = camera["move"]
        travelled = abs(move["from_position_m"][1] - move["to_position_m"][1])
        self.assertAlmostEqual(travelled, 1.0 * compression, places=4)

    def test_move_percentages_become_frames_inside_the_shot(self):
        camera, _ = self.resolve(
            Pack(), {"position_m": [0.0, -6.0, 1.5], "focal_mm": 50,
                     "move": {"kind": "orbit", "orbit_deg": 22,
                              "from_pct": 0.25, "to_pct": 0.75}},
            frames=101)
        self.assertEqual(camera["move"]["from_frame"], 26)
        self.assertEqual(camera["move"]["to_frame"], 76)

    def test_a_move_never_ends_before_it_begins(self):
        camera, _ = self.resolve(
            Pack(), {"position_m": [0.0, -6.0, 1.5], "focal_mm": 50,
                     "move": {"kind": "orbit", "orbit_deg": 10,
                              "from_pct": 0.5, "to_pct": 0.5}},
            frames=96)
        self.assertGreater(camera["move"]["to_frame"],
                           camera["move"]["from_frame"])


class TestEasing(unittest.TestCase):
    """Ease is baked into the keyed positions, so the arithmetic is testable."""

    def test_every_ease_runs_from_zero_to_one(self):
        for kind in ("in_out", "in", "out", "linear", None):
            self.assertAlmostEqual(h_assemble.ease(0.0, kind), 0.0, places=9,
                                   msg=kind)
            self.assertAlmostEqual(h_assemble.ease(1.0, kind), 1.0, places=9,
                                   msg=kind)

    def test_every_ease_is_monotonic(self):
        for kind in ("in_out", "in", "out", "linear"):
            values = [h_assemble.ease(i / 20.0, kind) for i in range(21)]
            self.assertEqual(values, sorted(values), kind)

    def test_in_out_is_symmetric_and_starts_slower_than_linear(self):
        self.assertAlmostEqual(h_assemble.ease(0.5, "in_out"), 0.5, places=9)
        self.assertLess(h_assemble.ease(0.25, "in_out"), 0.25)
        self.assertGreater(h_assemble.ease(0.75, "in_out"), 0.75)


class TestContracts(unittest.TestCase):

    def test_the_retime_policies_are_the_ones_the_schema_declares(self):
        from pipeline import validate
        schema = validate.load_schema("shot")
        declared = set(schema["$defs"]["cast_member"]["properties"]
                       ["retime"]["enum"])
        self.assertEqual(declared,
                         {"trim", "hold_last_pose", "loop", "timewarp"})

    def test_every_camera_anchor_the_plan_allows_has_a_joint_table(self):
        from pipeline import validate
        for anchor in validate.ANCHORS:
            self.assertIn(anchor, h_assemble.ANCHOR_JOINTS,
                          f"{anchor} is an allowed aim and resolves to nothing")


if __name__ == "__main__":
    unittest.main()
