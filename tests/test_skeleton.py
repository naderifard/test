"""Landmarks to joints: pipeline/skeleton.py, tested without a mesh or a GPU.

The fixture is a person built from prelude's own SMPL-X rest pose -- landmarks
placed where BlazePose would put them on a 1.70 m body -- so the joints that come
back can be compared with the joints ``prelude.character()`` would have used.
"""

from __future__ import annotations

import ast
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import skeleton                                    # noqa: E402


def prelude_table(name):
    """A module-level literal from prelude.py, read without importing bpy."""
    path = os.path.join(ROOT, "blender_ai_studio", "prelude.py")
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


REST = prelude_table("SMPLX_REST")


def person(mirror=False, scale=1.0, shift=(0.0, 0.0, 0.0)):
    """BlazePose-style landmarks on a rest-pose body, as pushed-through points.

    ``mirror`` swaps which side BlazePose calls left, the way a photo of a real
    person facing the camera does relative to prelude's convention.
    """
    def at(x, y, z):
        return {"mid": (x * scale + shift[0], y * scale + shift[1],
                        z * scale + shift[2]),
                "skin": (x * scale + shift[0], (y + 0.08) * scale + shift[1],
                         z * scale + shift[2])}

    points = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        blaze = side if not mirror else ("right" if side == "left" else "left")
        for part, joint in (("shoulder", "shoulder"), ("elbow", "elbow"),
                            ("wrist", "wrist"), ("knee", "knee"),
                            ("ankle", "ankle")):
            points[f"{blaze}_{part}"] = at(*REST[f"{side}_{joint}"])
        hx, hy, hz = REST[f"{side}_hip"]
        points[f"{blaze}_hip"] = at(hx / (1 - 0.12), hy, hz)
        points[f"{blaze}_heel"] = at(sign * 0.10, -0.03, 0.02)
        points[f"{blaze}_foot_index"] = at(sign * 0.10, 0.20, 0.02)
        points[f"{blaze}_eye"] = at(sign * 0.032, 0.0, 0.936 * 1.70)
    points["mouth_left"] = at(0.025, 0.0, 1.50)
    points["mouth_right"] = at(-0.025, 0.0, 1.50)
    points["nose"] = at(0.0, 0.0, 1.55)
    return points


def close(testcase, a, b, places=2):
    for x, y in zip(a, b):
        testcase.assertAlmostEqual(x, y, places=places)


class TestTablesAgreeWithPrelude(unittest.TestCase):
    """skeleton.py copies two prelude tables so it can run without bpy."""

    def test_the_smplx_heights_match(self):
        for name in ("pelvis", "spine1", "spine2", "spine3", "neck", "head",
                     "jaw"):
            self.assertAlmostEqual(REST[name][2], skeleton.SMPLX_Z[name],
                                   msg=name)
        self.assertAlmostEqual(REST["left_shoulder"][2],
                               skeleton.SMPLX_Z["shoulder"])
        self.assertAlmostEqual(REST["left_collar"][2],
                               skeleton.SMPLX_Z["collar"])

    def test_the_quadruped_rest_matches(self):
        self.assertEqual(prelude_table("QUAD_REST"), skeleton.QUAD_REST)
        self.assertEqual(prelude_table("QUAD_CROWN"), skeleton.QUAD_CROWN)


class TestHumanJoints(unittest.TestCase):

    def test_every_smplx_joint_comes_back(self):
        joints, _info = skeleton.human_joints(person())
        self.assertEqual(set(REST), set(joints))

    def test_a_rest_body_gives_back_its_own_spine(self):
        joints, info = skeleton.human_joints(person())
        self.assertAlmostEqual(1.70, info["stature"], places=2)
        for name in ("pelvis", "spine1", "spine2", "spine3", "neck", "head"):
            close(self, REST[name], joints[name])

    def test_hips_move_in_from_blazepose_to_the_femur_heads(self):
        joints, _info = skeleton.human_joints(person())
        close(self, REST["left_hip"], joints["left_hip"])

    def test_left_is_decided_by_position_not_by_blazepose(self):
        """A mirrored reading must not mirror the rig."""
        straight, _ = skeleton.human_joints(person())
        mirrored, info = skeleton.human_joints(person(mirror=True))
        self.assertTrue(info["flipped_sides"])
        for name in ("left_wrist", "right_knee", "left_foot"):
            close(self, straight[name], mirrored[name])
        self.assertGreater(mirrored["left_shoulder"][0], 0)

    def test_the_skeleton_scales_with_the_mesh(self):
        joints, info = skeleton.human_joints(person(scale=0.5,
                                                    shift=(0.1, 0.2, 0.0)))
        self.assertAlmostEqual(0.85, info["stature"], places=2)
        close(self, [v * 0.5 + s for v, s in zip(REST["neck"],
                                                  (0.1, 0.2, 0.0))],
              joints["neck"])

    def test_the_mouth_is_measured_on_the_skin(self):
        _joints, info = skeleton.human_joints(person())
        face = info["face"]
        self.assertAlmostEqual(0.05, face["mouth_width"], places=3)
        self.assertAlmostEqual(0.08, face["mouth"][1], places=3)

    def test_a_missing_limb_is_named_in_the_error(self):
        points = person()
        del points["left_wrist"]
        with self.assertRaises(skeleton.SkeletonError) as caught:
            skeleton.human_joints(points)
        self.assertIn("left_wrist", str(caught.exception))

    def test_an_upside_down_reading_is_refused(self):
        points = person()
        for name in ("left_ankle", "right_ankle"):
            x, y, z = points[name]["mid"]
            points[name]["mid"] = (x, y, 3.0)
        with self.assertRaises(skeleton.SkeletonError):
            skeleton.human_joints(points)


class TestQuadrupedJoints(unittest.TestCase):

    def test_the_template_fills_the_box(self):
        joints, info = skeleton.quadruped_joints((-0.3, -0.8, 0.0),
                                                 (0.3, 0.9, 1.2))
        self.assertAlmostEqual(0.9, joints["muzzle"][1], places=3)
        self.assertAlmostEqual(-0.8, joints["tail_end"][1], places=3)
        self.assertAlmostEqual(1.2, info["stature"])
        self.assertGreater(joints["left_hoof_f"][0], 0)
        self.assertLess(joints["right_hoof_f"][0], 0)
        self.assertLess(abs(joints["left_hoof_b"][2]), 0.05)

    def test_a_flat_box_is_refused(self):
        with self.assertRaises(skeleton.SkeletonError):
            skeleton.quadruped_joints((0, 0, 0), (0.3, 0.9, 0.0))


class TestRefineWrists(unittest.TestCase):
    """Wrists checked against the hands in the matte (the black div)."""

    @staticmethod
    def _matte(left_tip, right_tip):
        import cv2
        import numpy as np
        a = np.zeros((600, 400), np.uint8)
        cv2.rectangle(a, (170, 100), (230, 450), 255, -1)          # body
        cv2.line(a, (180, 120), (100, 200), 255, 18)               # upper arms
        cv2.line(a, (220, 120), (300, 200), 255, 18)
        cv2.line(a, (100, 200), left_tip, 255, 16)                 # forearms
        cv2.line(a, (300, 200), right_tip, 255, 16)
        return a

    @staticmethod
    def _points(left_wrist, right_wrist):
        return {"left_shoulder": [180, 120, 1], "right_shoulder": [220, 120, 1],
                "left_elbow": [100, 200, 1], "right_elbow": [300, 200, 1],
                "left_wrist": list(left_wrist) + [1],
                "right_wrist": list(right_wrist) + [1]}

    def test_a_human_wrist_stays(self):
        alpha = self._matte((60, 330), (340, 330))
        points = self._points((77, 274), (323, 274))
        _, moved = skeleton.refine_wrist_pixels(points, alpha)
        self.assertEqual(moved, {})

    def test_a_wrist_on_the_forearm_moves_to_the_hand(self):
        alpha = self._matte((60, 330), (380, 420))
        points = self._points((77, 274), (305, 215))   # right wrist too high
        points2, moved = skeleton.refine_wrist_pixels(points, alpha)
        self.assertIn("right_wrist".split("_")[0], moved)
        self.assertNotIn("left", moved)
        self.assertGreater(points2["right_wrist"][1], 280)


if __name__ == "__main__":
    unittest.main()
