"""The declared pipeline versus the built one.

``graph.STAGE_ORDER`` names the stages of §12, and for as long as some of them
are unwritten the runner has to tell the two kinds of absence apart:

* the module does not exist — the stage is not built yet, which is a fact about
  progress and should render as a row in ``status``, not a traceback;
* the module exists and one of *its* imports is missing — the stage is built and
  this interpreter cannot run it, which is an environment fault and must never
  be reported as "not built yet".

That distinction is one attribute lookup deep (``ModuleNotFoundError.name``) and
is exactly the kind of thing that regresses silently, so it is tested here
rather than left to the next full run.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import graph                                    # noqa: E402


class TestStageOrder(unittest.TestCase):

    def test_the_stages_of_the_design_are_declared(self):
        """Sixteen: fifteen, plus j_mouth (MuseTalk over the render).

        C-asset was in §7-C1 and in §11's gate table from the start -- gates.py
        has carried ``check_asset`` all along -- but STAGE_ORDER listed only the
        fitters that consume the mesh, never the stage that builds it. The count
        is pinned so that adding a stage stays a deliberate act.
        """
        self.assertEqual(len(graph.STAGE_ORDER), 17)
        self.assertEqual(len(set(graph.STAGE_ORDER)), 17, "no duplicates")

    def test_every_stage_with_a_declared_gate_is_in_the_order(self):
        """§11 names a gate per stage; a gate with no stage is a stage missing.

        This is how C-asset's absence would have been caught: ``check_asset``
        existed, was tested, and nothing ever called it.
        """
        from pipeline import gates
        for name in graph.STAGE_ORDER:
            checker = "check_" + name.split("_", 1)[1]
            if hasattr(gates, checker):
                self.assertIn(name, graph.STAGE_ORDER)
        # and the reverse: every check_* in gates belongs to a declared stage
        declared = {name.split("_", 1)[1] for name in graph.STAGE_ORDER}
        declared |= {"refs", "assembly", "render"}   # gates named for the artifact
        for attribute in dir(gates):
            if not attribute.startswith("check_"):
                continue
            subject = attribute[len("check_"):]
            self.assertIn(
                subject, declared,
                f"gates.{attribute} measures something no stage in "
                f"STAGE_ORDER produces")

    def test_every_declared_stage_has_a_module_or_is_reported_as_unbuilt(self):
        for name in graph.STAGE_ORDER:
            module = graph.load_stage(name, missing_ok=True)
            if module is None:
                self.assertFalse(graph.implemented(name))
                continue
            self.assertTrue(graph.implemented(name))
            self.assertEqual(module.STAGE.name, name,
                             "a stage must declare its own name")
            self.assertTrue(module.STAGE.implemented)

    def test_a_missing_stage_is_a_message_not_an_import_error(self):
        with self.assertRaises(graph.StageError) as caught:
            graph.load_stage("z_does_not_exist")
        self.assertIn("not implemented yet", str(caught.exception))

    def test_a_missing_stage_is_none_when_tolerated(self):
        self.assertIsNone(graph.load_stage("z_does_not_exist", missing_ok=True))

    def test_a_stage_whose_own_import_fails_is_never_called_unbuilt(self):
        """The case that must not be confused with an unwritten stage."""
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "pipeline", "stages",
            "z_broken_import.py")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("import a_library_that_is_not_installed_anywhere\n"
                     "STAGE = None\n")
        try:
            with self.assertRaises(graph.StageError) as caught:
                graph.load_stage("z_broken_import", missing_ok=True)
            message = str(caught.exception)
            self.assertIn("a_library_that_is_not_installed_anywhere", message)
            self.assertNotIn("not implemented yet", message)
        finally:
            os.remove(path)
            sys.modules.pop("pipeline.stages.z_broken_import", None)

    def test_the_unbuilt_placeholder_carries_no_steps_to_run(self):
        spec = graph.unbuilt_spec("g_align")
        self.assertFalse(spec.implemented)
        self.assertEqual(spec.steps, [], "an unbuilt stage must run nothing")
        self.assertEqual(spec.name, "g_align")


if __name__ == "__main__":
    unittest.main()
