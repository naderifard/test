"""The SMPL-X hand transplant's geometry (``pipeline/blender/hands.py``)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                             # noqa: E402

from pipeline.blender import hands                             # noqa: E402


def flat_hand(wrist, down, thin_axis, width_axis, n=200, seed=0):
    """Points of a flat hand hanging from ``wrist``: long along ``down``,
    wide along ``width_axis``, thin along ``thin_axis``."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 0.19, n)
    w = rng.uniform(-0.045, 0.045, n)
    h = rng.uniform(-0.012, 0.012, n)
    return wrist + np.outer(t, down) + np.outer(w, width_axis) \
        + np.outer(h, thin_axis)


class TestCharacterHand(unittest.TestCase):

    def frame(self, thin, width, wrist_x):
        wrist = np.array([wrist_x, 0.0, 0.9])
        pts = flat_hand(wrist, np.array([0, 0, -1.0]), thin, width)
        weights = np.ones(len(pts))
        return hands.character_hand(pts, weights, wrist,
                                    np.array([0.0, 0.0, 1.0]),
                                    elbow=wrist + np.array([0, 0, 0.26]))

    def test_palms_forward_put_the_thumb_outward(self):
        """The anatomical position: a hand on +X with its palm facing +Y has
        its thumb out on +X."""
        target = self.frame(np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), 0.3)
        self.assertGreater(target["across"][0], 0.9)
        target = self.frame(np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), -0.3)
        self.assertLess(target["across"][0], -0.9)

    def test_palms_to_the_thighs_put_the_thumb_forward(self):
        target = self.frame(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 0.3)
        self.assertGreater(target["across"][1], 0.9)

    def test_the_long_axis_is_the_forearm(self):
        target = self.frame(np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), 0.3)
        self.assertTrue(np.allclose(target["long"], [0, 0, -1], atol=1e-6))


class TestPlacement(unittest.TestCase):

    def test_a_right_hand_stays_a_right_hand(self):
        """The template's frame is taken onto the character's by a rotation,
        never a reflection."""
        template = {"long": np.array([-1.0, 0, 0]),
                    "across": np.array([0, 0, 1.0]),
                    "length": 0.19, "radius": 0.03}
        target = {"long": np.array([0, 0, -1.0]),
                  "across": np.array([1.0, 0, 0]),
                  "length": 0.2, "radius": 0.035}
        rotation, s_long, s_hand = hands.placement(template, target)
        self.assertAlmostEqual(1.0, np.linalg.det(rotation), places=6)
        self.assertTrue(np.allclose(rotation @ template["long"],
                                    target["long"]))
        self.assertTrue(np.allclose(rotation @ template["across"],
                                    target["across"]))
        self.assertLessEqual(s_hand, s_long * 1.25 + 1e-9)

    @unittest.skipUnless(hands.available(), "SMPL-X is not installed")
    def test_the_template_hand_is_a_hand(self):
        for side in ("left", "right"):
            t = hands.template_hand(side)
            self.assertTrue(0.15 < t["length"] < 0.25, t["length"])
            self.assertGreater(len(t["faces"]), 1000)


if __name__ == "__main__":
    unittest.main()
