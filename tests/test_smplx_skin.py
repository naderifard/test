"""SMPL-X skinning's pure parts: names and sides, weight collapsing, the
transfer along the mesh, smoothing (``smplx_fit``, ``blender/smplx_skin``)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                             # noqa: E402

from pipeline import smplx_fit                                 # noqa: E402
from pipeline.blender import smplx_skin                        # noqa: E402


class TestNames(unittest.TestCase):

    def test_sides_swap(self):
        """SMPL-X's left is on the rig's -X, which the rig calls right."""
        self.assertEqual("right_elbow", smplx_fit.rig_name("left_elbow"))
        self.assertEqual("left_hip", smplx_fit.rig_name("right_hip"))
        self.assertEqual("spine2", smplx_fit.rig_name("spine2"))

    def test_fingers_fold_into_the_wrist(self):
        bones = [smplx_fit.rig_name(n) for n in smplx_fit.SMPLX_BODY
                 if n != "jaw"]
        lbs = np.zeros((3, 55), np.float32)
        lbs[0, 30] = 1.0          # a left-hand finger joint
        lbs[1, 22] = 1.0          # the jaw, which the rig lacks
        lbs[2, 23] = 1.0          # an eye
        out = smplx_fit.bone_weights(lbs, bones)
        self.assertEqual(1.0, out[0, bones.index("right_wrist")])
        self.assertEqual(1.0, out[1, bones.index("head")])
        self.assertEqual(1.0, out[2, bones.index("head")])
        self.assertTrue(np.allclose(out.sum(1), 1.0))


class TestTransfer(unittest.TestCase):

    def test_off_body_takes_its_attachment_along_the_mesh(self):
        """A strip 0-1-2-3: 0 is on the body (bone A), 3 is on the body
        (bone B); 1 and 2 hang off it, each nearer along the mesh to one."""
        points = np.array([[0, 0, 0], [1, 0, 0], [2.5, 0, 0], [3, 0, 0.]])
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        weights = np.array([[1, 0], [0.5, 0.5], [0.5, 0.5], [0, 1.]])
        near = np.array([True, False, False, True])
        out, stranded = smplx_skin.attach(weights, near, points, edges)
        self.assertEqual([1, 0], list(out[1]))
        self.assertEqual([0, 1], list(out[2]))
        self.assertEqual(0, stranded)

    def test_unreachable_is_counted(self):
        points = np.zeros((3, 3))
        out, stranded = smplx_skin.attach(np.eye(3)[:, :2],
                                          np.array([True, False, False]),
                                          points, np.array([[0, 1]]))
        self.assertEqual(1, stranded)

    def test_strongest_keeps_four_and_normalises(self):
        w = np.array([[0.3, 0.25, 0.2, 0.15, 0.1]])
        out = smplx_skin.strongest(w, keep=4)
        self.assertEqual(0.0, out[0, 4])
        self.assertAlmostEqual(1.0, out.sum())

    def test_smoothing_keeps_weights_summing_to_one(self):
        w = np.array([[1, 0], [0, 1], [1, 0.]])
        out = smplx_skin.smooth(w, np.array([[0, 1], [1, 2]]), rounds=3)
        self.assertTrue(np.allclose(out.sum(1), 1.0))
        self.assertGreater(out[1, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
