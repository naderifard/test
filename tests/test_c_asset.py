"""Stage C-asset: the parts that decide things, tested without a GPU.

Nothing here loads SF3D. What is worth testing is everything around it -- which
file a unit is built from, where the head crop lands, and what the gate does
with a set of measurements -- because those are the parts that fail quietly.
The mesh itself fails loudly and is measured by the gate at run time.

Two of these tests exist because the code was wrong in exactly that way first:
``refs.json`` stores bare filenames, so a planner that treats a slot value as a
path hashes "missing:" and never invalidates; and a checkpoint with shared
tensors needs ``remove_duplicate=False`` or 96 of them come back missing.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import gates, jsonio                               # noqa: E402
from pipeline.project import Project                             # noqa: E402
from pipeline.stages import c_asset                              # noqa: E402


# A set of measurements that passes everything, for tests to break one at a time.
# The numbers are Keyumars' from the first version-2 run.
GOOD = {
    "subject": "keyumars",
    "kind": "human",
    "tris": 278000,
    "size_m": [0.33, 0.22, 0.92],
    "degenerate_faces": 0,
    "vertices": 139070,
    "watertight": True,
    "shells_kept": 1,
    "photo_share": 0.35,
    "coverage": 0.12,
    "rig_method": "blazepose",
    "pose_confidence": 0.9999,
    "landmarks_inside": 1.0,
    "pose_seeded": False,
    "missed_landmarks": 0,
}


def verdicts(metrics):
    return {v.name: v for v in c_asset._gate(None, {}, metrics)}


class TestReferencePaths(unittest.TestCase):
    """refs.json records a folder and bare filenames, not paths."""

    REFS = {
        "folder": "refs/characters/keyumars",
        "slots": {"front": "keyumars_front.png"},
    }

    def test_a_slot_becomes_an_absolute_path_under_the_project(self):
        images = c_asset.ref_images(os.path.join("D:", os.sep, "proj"),
                                    self.REFS)
        self.assertTrue(os.path.isabs(images["front"]))
        self.assertTrue(images["front"].endswith(
            os.path.join("refs", "characters", "keyumars",
                         "keyumars_front.png")))

    def test_an_empty_slot_is_dropped_rather_than_becoming_the_folder(self):
        refs = dict(self.REFS, slots={"front": "a.png", "face": ""})
        images = c_asset.ref_images("/proj", refs)
        self.assertIn("front", images)
        self.assertNotIn("face", images)

    def test_no_slots_is_an_empty_mapping_not_an_error(self):
        self.assertEqual({}, c_asset.ref_images("/proj", {}))


class TestGate(unittest.TestCase):

    def test_a_good_figure_passes_with_the_numbers_in_the_message(self):
        found = verdicts(GOOD)
        self.assertEqual(["asset"], list(found))
        message = found["asset"].message
        self.assertIn("139070", message)
        self.assertIn("35%", message)
        self.assertIn("blazepose", message)

    def test_a_blob_fails_on_vertex_count(self):
        found = verdicts(dict(GOOD, vertices=900))
        self.assertEqual(gates.FAIL, found["asset.vertices"].level)

    def test_an_open_mesh_warns_rather_than_fails(self):
        """Skinning and lighting survive a hole; a close-up shows it."""
        found = verdicts(dict(GOOD, watertight=False))
        self.assertEqual(gates.WARN, found["asset.watertight"].level)

    def test_a_photo_that_does_not_land_on_the_mesh_fails(self):
        """The projection is exact or it is wrong; there is no in between."""
        found = verdicts(dict(GOOD, photo_share=0.04))
        self.assertEqual(gates.FAIL, found["asset.photo"].level)

    def test_an_unsure_skeleton_fails(self):
        found = verdicts(dict(GOOD, pose_confidence=0.3))
        self.assertEqual(gates.FAIL, found["asset.pose"].level)

    def test_landmarks_off_the_figure_warn(self):
        found = verdicts(dict(GOOD, landmarks_inside=0.6))
        self.assertEqual(gates.WARN, found["asset.landmarks"].level)

    def test_a_silhouette_seeded_skeleton_warns(self):
        """The div's case: no face for the detector, joints read anyway."""
        found = verdicts(dict(GOOD, pose_seeded=True))
        self.assertEqual(gates.WARN, found["asset.pose_seeded"].level)

    def test_an_animal_is_not_held_to_the_pose_checks(self):
        found = verdicts(dict(GOOD, kind="quadruped",
                              rig_method="quadruped_template",
                              pose_confidence=None, landmarks_inside=None))
        self.assertEqual(["asset"], list(found))

    def test_the_shared_asset_gate_still_applies(self):
        found = verdicts(dict(GOOD, tris=0))
        self.assertEqual(gates.FAIL, found["asset.tris"].level)


class TestPlan(unittest.TestCase):
    """What the planner decides, against a project on disk."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="c_asset_plan_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        jsonio.write(os.path.join(self.root, "story",
                                  "production.resolved.json"), {
            "characters": [
                {"id": "keyumars", "kind": "human", "height_m": 1.95},
                {"id": "lion", "kind": "quadruped", "height_m": 1.3},
            ]})
        for cid, slots in (("keyumars", {"front": "keyumars_front.png"}),
                           ("lion", {"front": "lion_front.png"})):
            jsonio.write(os.path.join(self.root, "assets", "characters", cid,
                                      "refs", "refs.json"),
                         {"folder": f"refs/characters/{cid}", "slots": slots})
        self.project = Project(self.root)

    def units(self):
        return {u.uid: u for u in c_asset.plan(self.project, None)}

    def test_one_unit_per_character(self):
        self.assertEqual({"char_keyumars", "char_lion"}, set(self.units()))

    def test_the_kind_travels_so_the_step_can_choose_a_rig(self):
        """BlazePose for a person, the quadruped template for an animal."""
        units = self.units()
        self.assertEqual("human", units["char_keyumars"].params["kind"])
        self.assertEqual("quadruped", units["char_lion"].params["kind"])

    def test_the_image_itself_is_an_input_so_replacing_it_rebuilds(self):
        """The bug this caught: a bare filename hashes as "missing:".

        refs.json records a filename, and the store hashes inputs by content. An
        input that is not a real path resolves to "missing:<name>" for every
        image with that name, so swapping in a better reference would leave the
        old mesh cached and nothing would say so.
        """
        unit = self.units()["char_keyumars"]
        self.assertIn("ref_front", unit.inputs)
        self.assertTrue(os.path.isabs(unit.inputs["ref_front"]),
                        f"{unit.inputs['ref_front']} is not a path the store "
                        f"can hash")

    def test_a_character_with_no_production_entry_plans_nothing(self):
        jsonio.write(os.path.join(self.root, "story",
                                  "production.resolved.json"), {})
        self.assertEqual([], c_asset.plan(Project(self.root), None))


if __name__ == "__main__":
    unittest.main()


class TestHostMemoryPreflight(unittest.TestCase):
    """The stage waits for host memory rather than failing on one sample.

    The competing consumer here is a training run whose footprint moves: free
    commit on this machine read 0.20 GB and 2.54 GB a minute apart. A preflight
    that sampled once would fail a six-unit stage on someone else's batch
    boundary, and one that waited forever would hang.
    """

    def setUp(self):
        from pipeline import sf3d_geometry
        self.module = sf3d_geometry
        self.original = sf3d_geometry.free_commit_gb
        self.addCleanup(setattr, sf3d_geometry, "free_commit_gb", self.original)

    def feed(self, *readings):
        values = list(readings)
        self.module.free_commit_gb = lambda: (values.pop(0) if len(values) > 1
                                              else values[0])

    def test_enough_memory_returns_at_once_without_waiting(self):
        self.feed(9.0)
        report = self.module.check_commit(wait_s=60, poll_s=0, log=None)
        self.assertTrue(report["available"])
        self.assertEqual(0, report["waited_s"])

    def test_a_transient_dip_is_waited_out_rather_than_failed(self):
        self.feed(0.2, 0.3, 9.0)
        report = self.module.check_commit(wait_s=60, poll_s=0, log=None)
        self.assertTrue(report["available"])

    def test_memory_that_never_comes_back_raises_and_says_the_numbers(self):
        self.feed(0.2)
        with self.assertRaises(self.module.Sf3dError) as caught:
            self.module.check_commit(wait_s=0, poll_s=0, log=None)
        message = str(caught.exception)
        self.assertIn("0.20 GB", message)
        self.assertIn("not VRAM", message)

    def test_a_platform_without_the_measurement_does_not_block_the_stage(self):
        self.module.free_commit_gb = lambda: None
        self.assertEqual({"available": False},
                         self.module.check_commit(wait_s=0, poll_s=0, log=None))
