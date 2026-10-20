"""The character image check's decisions that need no model.

``pipeline.image_check`` runs rembg and BlazePose in the ml env; what is
tested here is what it makes of their output: whether an arm is clear of the
body on a row, and how findings add up to a verdict.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                       # noqa: E402

from pipeline import image_check as ic                   # noqa: E402


class TestGap(unittest.TestCase):

    def test_background_between_arm_and_body_is_a_gap(self):
        matte = np.zeros((10, 40), dtype=np.float32)
        matte[:, 5:15] = 1.0        # body
        matte[:, 25:30] = 1.0       # arm, with background between
        self.assertTrue(ic._gap_between(matte, 5, 10, 27))

    def test_arm_against_the_body_is_not(self):
        matte = np.zeros((10, 40), dtype=np.float32)
        matte[:, 5:30] = 1.0
        self.assertFalse(ic._gap_between(matte, 5, 10, 27))

    def test_too_close_to_tell(self):
        matte = np.ones((10, 40), dtype=np.float32)
        self.assertIsNone(ic._gap_between(matte, 5, 10, 11))


class TestVerdict(unittest.TestCase):

    def test_no_findings_is_good(self):
        result = ic._result([], {})
        self.assertEqual(result["verdict"], "good")
        self.assertEqual(result["findings"][0]["level"], "ok")

    def test_a_failure_is_poor(self):
        result = ic._result([ic._finding("fail", "framing", "cut")], {})
        self.assertEqual(result["verdict"], "poor")

    def test_warnings_add_up(self):
        one = [ic._finding("warn", "arms", "x")]
        self.assertEqual(ic._result(one, {})["verdict"], "usable")
        three = one * 3
        self.assertEqual(ic._result(three, {})["verdict"], "poor")


if __name__ == "__main__":
    unittest.main()
