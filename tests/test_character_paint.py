"""The parts of character painting that decide a texel's colour."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                             # noqa: E402

from pipeline import character_paint as cp                     # noqa: E402
from pipeline.stages import c_paint                            # noqa: E402


class TestConsensus(unittest.TestCase):

    def test_one_misaligned_view_is_outvoted(self):
        """Four views see skin; the most squarely facing one sees the tunic's
        dark spot beside the arm. The texel stays skin."""
        skin, spot = [190, 140, 110], [40, 30, 20]
        samples = np.array([[skin, skin, spot, skin, skin]], np.uint8)
        weights = np.array([[0.5, 0.8, 1.0, 0.8, 0.5]], np.float32)
        colour = cp.consensus(samples, weights)[0]
        self.assertLess(np.abs(colour - skin).max(), 12)

    def test_agreeing_views_average(self):
        samples = np.array([[[100, 100, 100], [110, 110, 110]]], np.uint8)
        weights = np.array([[1.0, 1.0]], np.float32)
        colour = cp.consensus(samples, weights)[0]
        self.assertTrue(np.allclose(colour, 105, atol=1))

    def test_unseen_texel_stays_finite(self):
        samples = np.zeros((1, 3, 3), np.uint8)
        weights = np.zeros((1, 3), np.float32)
        self.assertTrue(np.isfinite(cp.consensus(samples, weights)).all())


class TestPaintedFigure(unittest.TestCase):

    def test_backdrop_is_what_touches_the_border(self):
        """A grey patch inside the figure is not backdrop, however close its
        colour is to the backdrop's."""
        frame = np.full((60, 40, 3), 128, np.uint8)
        frame[10:50, 10:30] = [180, 120, 90]          # the figure
        frame[25:35, 15:25] = 128                     # a grey tunic on it
        figure = cp._painted_figure(frame)
        self.assertFalse(figure[2, 2])
        self.assertTrue(figure[30, 20])
        self.assertTrue(figure[15, 20])


class TestHands(unittest.TestCase):

    def test_only_front_views_colour_a_hand(self):
        self.assertEqual(cp.front_view(0.0), 1.0)
        self.assertEqual(cp.front_view(348.75), 1.0)
        self.assertEqual(cp.front_view(180.0), 0.0)
        self.assertEqual(cp.front_view(67.5), 0.0)
        self.assertTrue(0.0 < cp.front_view(33.75) < 1.0)

    def test_the_forearm_is_not_a_hand(self):
        share = cp.hand_share([0.0, 0.2, 0.5, 1.0])
        self.assertEqual(share[0], 0.0)
        self.assertEqual(share[1], 0.0)
        self.assertEqual(share[3], 1.0)


class TestHead(unittest.TestCase):

    def test_a_face_takes_one_frame_not_an_average(self):
        """Two near-front frames see a texel almost equally well: on the
        body they share it, on the head the better one has it nearly alone."""
        weights = np.array([[0.81, 0.66], [0.81, 0.66]])
        out = cp.single_source(weights, np.array([0.0, 1.0]))
        body, head = out[0] / out[0].sum(), out[1] / out[1].sum()
        self.assertLess(body[0], 0.6)
        self.assertGreater(head[0], 0.95)

    def test_the_neck_is_not_the_head(self):
        share = cp.head_share([0.0, 0.3, 0.5, 1.0])
        self.assertEqual(0.0, share[1])
        self.assertEqual(1.0, share[3])


class TestTurnPrompt(unittest.TestCase):

    def test_an_animal_stands_on_four_legs(self):
        prompt = c_paint.turn_prompt({"kind": "quadruped",
                                      "image_prompt": "A large lion"})
        self.assertIn("four legs", prompt)
        self.assertIn("A large lion.", prompt)

    def test_a_person_stands_in_an_a_pose(self):
        prompt = c_paint.turn_prompt({"kind": "human", "image_prompt": ""})
        self.assertIn("A-pose", prompt)


if __name__ == "__main__":
    unittest.main()
