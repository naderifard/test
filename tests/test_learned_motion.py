"""The pure-Python parts of the learned motion and mouth passes.

The networks (MoMask, GVHMR, MuseTalk) need the motion env and a GPU and are
checked by running the stages; what is tested here is everything around them
that decides whether their output lands right: axes and sides, the loop cut,
prompts, the face box MuseTalk is given.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import lipsync_musetalk as lm           # noqa: E402
from pipeline import motion_learned, motion_smpl, motion_text  # noqa: E402

# SMPL neutral joints at zero pose (y up, facing +z), metres, rounded.
REST = [(0.0, 0.92, 0.0), (0.06, 0.83, -0.01), (-0.06, 0.83, -0.01),
        (0.0, 1.03, -0.02), (0.1, 0.45, 0.0), (-0.1, 0.45, 0.0),
        (0.0, 1.16, 0.0), (0.09, 0.05, -0.04), (-0.09, 0.05, -0.04),
        (0.0, 1.21, 0.01), (0.12, 0.0, 0.08), (-0.12, 0.0, 0.08),
        (0.0, 1.43, -0.01), (0.08, 1.35, 0.0), (-0.08, 1.35, 0.0),
        (0.0, 1.49, 0.04), (0.17, 1.39, -0.01), (-0.17, 1.39, -0.01),
        (0.26, 1.13, -0.03), (-0.26, 1.13, -0.03), (0.28, 0.88, 0.0),
        (-0.28, 0.88, 0.0)]


class TestAxesAndSides(unittest.TestCase):

    def test_rotation_not_reflection_and_sides_swapped(self):
        rig = motion_smpl.to_rig(REST)
        J = motion_smpl.J
        # The performer's anatomical left (SMPL +x) lands on the rig's -X ...
        self.assertLess(rig[J["right_shoulder"]][0], 0.0)
        # ... which is where the rig's right_* bones are.
        self.assertGreater(rig[J["left_shoulder"]][0], 0.0)
        # Forward is +Y, up is +Z.
        self.assertGreater(rig[J["head"]][2], rig[J["pelvis"]][2])
        self.assertGreater(rig[J["left_foot"]][1], rig[J["left_ankle"]][1])

    def test_standing_rest_is_near_neutral(self):
        frames = [motion_smpl.to_rig(REST)] * 3
        body, head, root, _ = motion_smpl.retarget(frames, 24)
        self.assertLess(abs(root[0][2]), 0.05, "standing is not a crouch")
        deg = 2 * math.degrees(math.acos(min(1.0, abs(head[0]["head"][0]))))
        self.assertLess(deg, 20.0)


class TestLoop(unittest.TestCase):

    def _walk_in_place(self, seconds, fps, drift=0.0):
        frames = []
        for i in range(int(seconds * fps)):
            t = i / float(fps)
            wave = math.sin(2 * math.pi * t / 1.2)
            f = [list(p) for p in motion_smpl.to_rig(REST)]
            for name in ("left_wrist", "left_elbow"):
                f[motion_smpl.J[name]][1] += 0.15 * wave
            for p in f:
                p[1] += drift * t
            frames.append([tuple(p) for p in f])
        return frames

    def test_cut_is_seamless_and_on_the_spot(self):
        frames = self._walk_in_place(8.0, 24, drift=0.05)
        out, report = motion_smpl.loop(frames, 24, min_s=3.0)
        self.assertGreaterEqual(len(out), 72)
        self.assertLessEqual(report["wrap_gap_m"], 2 * report["mean_step_m"]
                             + 1e-6)
        pelvis = motion_smpl.J["pelvis"]
        self.assertAlmostEqual(out[0][pelvis][1], out[-1][pelvis][1], delta=0.02)

    def test_the_fight_loop_is_a_strike_then_a_block(self):
        """``exchange`` with a stand-in for MoMask: the right wrist thrusts
        forward in the strike and rises in the block. The loop is one of
        each, the strike in its first half."""
        def half(kind):
            frames = []
            for i in range(40):                   # 2 s at MoMask's 20 fps
                s = math.sin(math.pi * i / 39.0)
                f = [list(p) for p in motion_smpl.to_rig(REST)]
                wrist = f[motion_smpl.J["right_wrist"]]
                if kind == "strike":
                    wrist[1] += 0.4 * s           # forward
                else:
                    wrist[2] += 0.4 * s           # up
                frames.append([tuple(p) for p in f])
            return frames

        class Stand:
            def generate(self, text, seconds, **kw):
                return half("strike" if "punch" in text else "block"), {
                    "caption": text}

        saved = motion_learned._GENERATOR
        motion_learned._GENERATOR = Stand()
        try:
            body, _head, _root, report = motion_learned.exchange(24)
        finally:
            motion_learned._GENERATOR = saved
        self.assertTrue(40 <= len(body) <= 110, len(body))
        self.assertEqual(2, len(report["exchange"]))
        self.assertLessEqual(report["wrap_gap_m"],
                             2 * report["mean_step_m"] + 0.02)

    def test_too_short_is_an_error(self):
        with self.assertRaises(ValueError):
            motion_smpl.loop(self._walk_in_place(2.0, 24), 24, min_s=3.0)


class TestExchangePhase(unittest.TestCase):

    def cast(self, b_opponent="a", b_take="fight_human"):
        return [{"character": "a", "take": "fight_human", "retime": "loop",
                 "opponent": "b"},
                {"character": "b", "take": b_take, "retime": "loop",
                 "opponent": b_opponent}]

    def test_two_fighters_are_half_a_cycle_apart(self):
        from pipeline.stages import h_assemble
        cast = self.cast()
        self.assertEqual(0, h_assemble._exchange_phase(cast[0], cast, 96))
        self.assertEqual(48, h_assemble._exchange_phase(cast[1], cast, 96))

    def test_the_giant_bends_to_the_man(self):
        from pipeline.stages import h_assemble
        cast = [{"character": "div", "height_m": 2.9, "opponent": "siamak"},
                {"character": "siamak", "height_m": 1.95, "opponent": "div"}]
        self.assertEqual(25.0, h_assemble._lean_to(cast[0], cast))
        self.assertEqual(0.0, h_assemble._lean_to(cast[1], cast))
        even = [dict(cast[0], height_m=1.95), cast[1]]
        self.assertEqual(0.0, h_assemble._lean_to(even[0], even))

    def test_a_fall_ends_on_the_ground(self):
        frames = []
        for i in range(60):
            f = [list(p) for p in motion_smpl.to_rig(REST)]
            # down to 0.2 m by frame 30, then back up again
            drop = 0.7 * math.sin(math.pi * min(i, 59) / 59.0)
            for p in f:
                p[2] -= drop
            frames.append([tuple(p) for p in f])
        cut, low = motion_learned._end_on_the_ground(frames)
        self.assertTrue(28 <= low <= 31, low)
        self.assertEqual(low + 1, len(cut))
        self.assertTrue(motion_learned.FALLS.search("falls and lies still"))
        self.assertFalse(motion_learned.FALLS.search("walks to the king"))

    def test_an_embrace_stops_at_the_other_body(self):
        from pipeline.stages import e_motion
        params = {"clips": [{"clip": "embrace", "target": "siamak"}],
                  "targets": {"siamak": [0.5, 0.0, 0.0]},
                  "position_m": [-0.5, 0.0, 0.0], "scale": 1.0}
        root = [[0.05 * i, 0.0, 0.0] for i in range(21)]   # walks 1 m
        report = {}
        out = e_motion._stop_at_contact(root, report, params)
        self.assertAlmostEqual((1.0 - e_motion.CONTACT_MIN_M) / 2.0,
                               out[-1][0], places=6)
        self.assertIn("contact_travel_cut_m", report)
        # No contact clip: the walk is left alone.
        params["clips"] = [{"clip": "walk", "target": "siamak"}]
        self.assertEqual(root, e_motion._stop_at_contact(root, {}, params))

    def test_a_speaker_does_not_look_at_the_sky(self):
        up = math.radians(12.0)
        poses = [{"neck": [up, 0.0, 0.0], "head": [up, 0.1, 0.0]},
                 {"neck": [-up, 0.0, 0.0], "head": [0.01, 0.0, 0.0]}]
        out = motion_learned.cap_head_up(poses)
        total = out[0]["neck"][0] + out[0]["head"][0]
        self.assertAlmostEqual(math.radians(motion_learned.HEAD_UP_MAX_DEG),
                               total, places=6)
        self.assertEqual(0.1, out[0]["head"][1])      # the turn is kept
        self.assertEqual(poses[1], out[1])             # a nod down is kept

    def test_a_smeared_repaint_is_refused(self):
        self.assertLess(lm.kept_detail([100, 100, 100], [30, 25, 40]),
                        lm.DETAIL_KEEP)
        self.assertGreater(lm.kept_detail([100, 100, 100], [90, 95, 99]),
                           lm.DETAIL_KEEP)

    def test_no_exchange_unless_they_fight_each_other(self):
        from pipeline.stages import h_assemble
        cast = self.cast(b_opponent=None)
        self.assertIsNone(h_assemble._exchange_phase(cast[0], cast, 96))
        cast = self.cast(b_take="talk_human")
        self.assertIsNone(h_assemble._exchange_phase(cast[0], cast, 96))


class TestPrompts(unittest.TestCase):

    def test_names_become_someone(self):
        caption = motion_text.prompt("fights siamak and strikes him down",
                                     ["siamak", "black_div"])
        self.assertEqual(caption, "a person fights someone and strikes him down")

    def test_underscore_names(self):
        caption = motion_text.prompt("turns toward the black div",
                                     ["black_div"])
        self.assertIn("someone", caption)
        self.assertNotIn("div", caption)

    def test_travel(self):
        self.assertFalse(motion_learned.travels("bows his head, unable to answer"))
        self.assertTrue(motion_learned.travels("walks to the throne and kneels"))
        self.assertTrue(motion_learned.travels("runs toward the div"))


class TestFaceBox(unittest.TestCase):

    POINTS = {"left_eye": [0.45, 0.30], "right_eye": [0.55, 0.30],
              "nose": [0.50, 0.36], "mouth": [0.50, 0.41],
              "mouth_left": [0.47, 0.41], "mouth_right": [0.53, 0.41]}

    def test_mask_line_sits_on_the_nose(self):
        x1, y1, x2, y2 = lm.face_box(self.POINTS, 1000, 1000)
        middle = (y1 + y2) / 2.0
        self.assertTrue(300 < middle < 400, middle)   # between eyes and mouth
        self.assertGreater(y2, 410)                   # below the mouth
        self.assertAlmostEqual((x1 + x2) / 2.0, 500, delta=5)

    def test_edge_face_is_padded_not_clamped(self):
        points = {k: [u, v - 0.28] for k, (u, v) in self.POINTS.items()}
        box = lm.face_box(points, 1000, 1000)
        self.assertLess(box[1], 0, "the box keeps its shape past the edge")
        import numpy as np
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        big, moved, pad = lm.padded(image, box)
        self.assertGreaterEqual(moved[1], 0)
        self.assertEqual(lm.unpadded(big, pad).shape, image.shape)

    def test_runs_and_fade(self):
        self.assertEqual(lm.runs([False, True, True, False, True]),
                         [(1, 2), (4, 4)])
        self.assertLess(lm.fade(10, (10, 40)), 1.0)
        self.assertEqual(lm.fade(25, (10, 40)), 1.0)


if __name__ == "__main__":
    unittest.main()


class TestHeading(unittest.TestCase):

    def _take(self, yaws, travel=(0.0, 0.0)):
        from pipeline import motion_library as ml
        body = [{"pelvis": ml.q_rotvec(ml.q_axis((0, 0, 1), y))} for y in yaws]
        n = len(yaws)
        root = [[travel[0] * i / (n - 1), travel[1] * i / (n - 1), 0.0]
                for i in range(n)]
        return body, root

    def test_travel_is_turned_toward_the_target(self):
        body, root = self._take([0.0] * 10, travel=(0.0, 2.0))   # +Y
        body, root, turned = motion_learned.aim_travel(body, root, 90.0)
        self.assertAlmostEqual(turned, 90.0, places=3)
        self.assertAlmostEqual(root[-1][0], -2.0, places=3)      # 90 is -X
        self.assertAlmostEqual(motion_learned._yaw_of(body[0]["pelvis"]),
                               90.0, places=3)

    def test_a_short_move_is_left_alone(self):
        body, root = self._take([0.0] * 10, travel=(0.0, 0.2))
        _, _, turned = motion_learned.aim_travel(body, root, 90.0)
        self.assertEqual(turned, 0.0)

    def test_a_speaker_never_turns_their_back(self):
        # Facing the camera (180), then spinning half round.
        body, root = self._take([i * 180.0 / 59 for i in range(60)])
        body, corrected = motion_learned.limit_turn(body, root, 180.0)
        worst = max(abs(motion_learned._wrap(
            motion_learned._yaw_of(p["pelvis"]))) for p in body)
        self.assertLessEqual(worst, motion_learned.FEATURED_TURN_LIMIT_DEG + 1)
        self.assertGreater(corrected, 100.0)
