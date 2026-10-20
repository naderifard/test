"""M0 acceptance test: the store, the keys and the graph, measured.

§11's rule is "measure the artifact, don't assume the stage worked", and it
applies to the machine itself before it applies to any model. Everything here
is checked by reading bytes off disk or the store's own metadata — never by
trusting a return value that says it went fine.

Run it with the host interpreter::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import graph, jsonio, project as project_mod   # noqa: E402
from pipeline.project import Project                          # noqa: E402
from pipeline.store import Store                              # noqa: E402

PERSIAN = "رستم دستان"


class TempProject(unittest.TestCase):
    """Each test gets its own projects root, so nothing leaks between them."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m0-")
        self._saved_root = project_mod.PROJECTS_ROOT
        project_mod.PROJECTS_ROOT = os.path.join(self.tmp, "projects")
        self.project = Project.create("scratch", title="Scratch", fps=24)

    def tearDown(self):
        project_mod.PROJECTS_ROOT = self._saved_root
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestJsonConventions(TempProject):
    """Content addressing is only as stable as the JSON writer under it."""

    def test_two_writes_of_the_same_data_are_byte_identical(self):
        target = self.project.path(".cache", "a.json")
        other = self.project.path(".cache", "b.json")
        jsonio.write(target, {"b": 1, "a": [2, 3], "t": PERSIAN})
        jsonio.write(other, {"t": PERSIAN, "a": [2, 3], "b": 1})
        with open(target, "rb") as fh:
            first = fh.read()
        with open(other, "rb") as fh:
            second = fh.read()
        self.assertEqual(first, second, "key order must not change the bytes")

    def test_persian_survives_as_persian(self):
        target = self.project.path(".cache", "fa.json")
        jsonio.write(target, {"line": PERSIAN})
        with open(target, "rb") as fh:
            raw = fh.read()
        self.assertIn(PERSIAN.encode("utf-8"), raw,
                      "Persian must be stored literally, not as \\uXXXX")
        self.assertTrue(raw.endswith(b"\n"), "file must end with a newline")
        self.assertEqual(jsonio.read(target)["line"], PERSIAN)

    def test_write_is_atomic_enough_to_not_truncate_on_failure(self):
        target = self.project.path(".cache", "keep.json")
        jsonio.write(target, {"value": 1})
        with self.assertRaises(TypeError):
            jsonio.write(target, {"value": object()})
        self.assertEqual(jsonio.read(target), {"value": 1},
                         "a failed write must leave the old file intact")


class TestStoreKeys(TempProject):
    """A key must move when anything that could change the artifact moves."""

    def setUp(self):
        super().setUp()
        self.store = self.project.store
        self.body = self.project.path(".cache", "body.bin")
        self.face = self.project.path(".cache", "face.bin")
        for path, payload in ((self.body, b"body-v1"), (self.face, b"face-v1")):
            with open(path, "wb") as fh:
                fh.write(payload)

    def key(self, **kwargs):
        inputs = kwargs.pop("inputs", {"body": self.body, "face": self.face})
        params = kwargs.pop("params", {"shape": "heroic"})
        version = kwargs.pop("version", "1")
        return self.store.key("c_rigged", version, inputs, params)[0]

    def test_same_work_gives_the_same_key(self):
        self.assertEqual(self.key(), self.key())

    def test_swapping_input_labels_changes_the_key(self):
        swapped = {"body": self.face, "face": self.body}
        self.assertNotEqual(self.key(), self.key(inputs=swapped),
                            "labels are part of the key, not just the file set")

    def test_changing_a_parameter_changes_the_key(self):
        self.assertNotEqual(self.key(), self.key(params={"shape": "aged"}))

    def test_bumping_the_stage_version_changes_the_key(self):
        self.assertNotEqual(self.key(), self.key(version="2"))

    def test_changing_a_file_changes_the_key(self):
        before = self.key()
        with open(self.body, "wb") as fh:
            fh.write(b"body-v2-different-length")
        self.assertNotEqual(before, self.key())

    def test_a_missing_input_is_marked_not_silently_dropped(self):
        _, resolved = self.store.key(
            "c_rigged", "1", {"body": self.project.path("nope.bin")}, {})
        self.assertTrue(resolved["body"].startswith("missing:"), resolved)


class TestStoreCommit(TempProject):

    def build_dir(self, **files):
        folder = tempfile.mkdtemp(dir=self.tmp)
        for name, text in files.items():
            with open(os.path.join(folder, name), "w", encoding="utf-8") as fh:
                fh.write(text)
        return folder

    def test_commit_records_a_measured_manifest(self):
        store = self.project.store
        source = self.build_dir(**{"note.txt": PERSIAN})
        store.commit("_selftest", "k1", source, {"version": "1"})
        meta = store.meta("_selftest", "k1")
        self.assertTrue(store.has("_selftest", "k1"))
        entry = meta["files"]["note.txt"]
        self.assertEqual(entry["bytes"],
                         len(PERSIAN.encode("utf-8")))
        self.assertEqual(len(entry["sha256"]), 64)

    def test_an_entry_without_meta_reads_as_absent(self):
        store = self.project.store
        os.makedirs(store.entry_dir("_selftest", "half"), exist_ok=True)
        with open(os.path.join(store.entry_dir("_selftest", "half"), "x.txt"),
                  "w") as fh:
            fh.write("partial")
        self.assertFalse(store.has("_selftest", "half"),
                         "a half-written commit must not count as a hit")

    def test_materialise_publishes_readable_files(self):
        store = self.project.store
        source = self.build_dir(**{"note.txt": "hello"})
        store.commit("_selftest", "k2", source, {"version": "1"})
        dest = self.project.path("assets", "props", "sword")
        os.makedirs(dest, exist_ok=True)
        written = store.materialise("_selftest", "k2", dest)
        self.assertEqual(len(written), 1)
        with open(os.path.join(dest, "note.txt"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "hello")

    def test_hash_file_memo_survives_a_reopen(self):
        path = self.project.path(".cache", "big.bin")
        with open(path, "wb") as fh:
            fh.write(b"x" * 4096)
        first = self.project.store.hash_file(path)
        self.project.store.flush()
        self.assertEqual(Store(self.project.path(".cache")).hash_file(path),
                         first)


class TestGraph(TempProject):
    """The whole machine, driven through the real subprocess path.

    ``_selftest`` is a genuine stage module invoked as ``python -m
    pipeline.stages._selftest`` in a child process, so this also proves the job
    file / result file contract and not merely the in-process bookkeeping.
    """

    def setUp(self):
        super().setUp()
        self._saved_order = graph.STAGE_ORDER
        graph.STAGE_ORDER = ("_selftest",)

    def tearDown(self):
        graph.STAGE_ORDER = self._saved_order
        super().tearDown()

    def set_plan(self, entries):
        jsonio.write(self.project.path(".cache", "_selftest", "plan.json"),
                     entries)

    def walk(self, **kwargs):
        walker = graph.Graph(self.project, **kwargs)
        return walker.run(graph.order())[0]

    def statuses(self, result):
        return {u.unit.uid: u.status for u in result.units}

    def test_first_run_builds_then_second_run_hits_the_cache(self):
        self.set_plan([{"uid": "alpha", "text": "alpha"},
                       {"uid": "beta", "text": PERSIAN}])
        first = self.walk()
        self.assertEqual(self.statuses(first),
                         {"alpha": "built", "beta": "built"})
        published = self.project.path("_selftest", "note.txt")
        self.assertTrue(os.path.exists(published), "output was not materialised")

        second = self.walk()
        self.assertEqual(self.statuses(second),
                         {"alpha": "cached", "beta": "cached"})
        # A cache hit must carry the metrics forward, not report an empty unit.
        for unit in second.units:
            self.assertGreater(unit.metrics.get("bytes_written", 0), 0,
                               unit.metrics)

    def test_the_subprocess_really_was_a_subprocess(self):
        self.set_plan([{"uid": "alpha", "text": "alpha"}])
        result = self.walk()
        metrics = result.units[0].metrics
        self.assertTrue(metrics["utf8_ok"])
        self.assertTrue(metrics["interpreter"].lower().startswith("python"),
                        metrics)

    def test_only_the_changed_unit_rebuilds(self):
        self.set_plan([{"uid": "alpha", "text": "alpha"},
                       {"uid": "beta", "text": "beta"}])
        self.walk()
        self.set_plan([{"uid": "alpha", "text": "alpha"},
                       {"uid": "beta", "text": "beta-rewritten"}])
        again = self.walk()
        self.assertEqual(self.statuses(again),
                         {"alpha": "cached", "beta": "built"},
                         "per-unit keying is what makes reuse across shots work")

    def test_force_rebuilds_a_cached_unit(self):
        self.set_plan([{"uid": "alpha", "text": "alpha"}])
        self.walk()
        self.assertEqual(self.statuses(self.walk(force=["_selftest"])),
                         {"alpha": "built"})
        self.assertEqual(self.statuses(self.walk(force=["all"])),
                         {"alpha": "built"})

    def test_deleting_the_published_copy_restores_it_from_the_store(self):
        self.set_plan([{"uid": "alpha", "text": "alpha"}])
        self.walk()
        published = self.project.path("_selftest", "note.txt")
        os.unlink(published)
        result = self.walk()
        self.assertEqual(self.statuses(result), {"alpha": "cached"})
        self.assertTrue(os.path.exists(published),
                        "a cache hit must still re-publish the readable copy")

    def test_dry_run_touches_nothing(self):
        self.set_plan([{"uid": "alpha", "text": "alpha"}])
        result = self.walk(dry_run=True)
        self.assertEqual(self.statuses(result), {"alpha": "skipped"})
        self.assertFalse(os.path.exists(self.project.path("_selftest",
                                                          "note.txt")))

    def test_a_failing_gate_is_not_cached(self):
        self.set_plan([{"uid": "alpha", "text": "alpha"},
                       {"uid": "bad", "text": "bad", "should_fail": True}])
        first = self.walk()
        self.assertEqual(self.statuses(first),
                         {"alpha": "built", "bad": "failed"})
        self.assertTrue(first.failed)

        store = self.project.store
        keys = {u.unit.uid: u.key for u in first.units}
        self.assertTrue(store.has("_selftest", keys["alpha"]))
        self.assertFalse(store.has("_selftest", keys["bad"]),
                         "a failed artifact must not become a sticky cache hit")

        # And it must fail again rather than quietly pass on the second walk.
        self.assertEqual(self.statuses(self.walk())["bad"], "failed")

    def test_a_unit_that_raises_does_not_cost_the_others_their_work(self):
        self.set_plan([{"uid": "alpha", "text": "alpha"},
                       {"uid": "boom", "text": None}])
        result = self.walk()
        statuses = self.statuses(result)
        self.assertEqual(statuses["alpha"], "built")
        self.assertEqual(statuses["boom"], "failed")
        self.assertIn("TypeError", result.units[
            [u.unit.uid for u in result.units].index("boom")].error)

    def test_an_empty_plan_reports_a_reason_not_a_silent_pass(self):
        self.set_plan([])
        result = self.walk()
        self.assertEqual(result.units, [])
        self.assertEqual(result.skipped, "the selftest plan is empty")

    def test_provenance_is_recorded_for_every_committed_unit(self):
        self.set_plan([{"uid": "alpha", "text": "alpha"}])
        result = self.walk()
        meta = self.project.store.meta("_selftest", result.units[0].key)
        for field in ("version", "unit", "label", "inputs", "params",
                      "metrics", "verdicts", "style_pack", "committed_at",
                      "files"):
            self.assertIn(field, meta)
        self.assertEqual(meta["params"]["text"], "alpha")
        self.assertEqual(meta["style_pack"], "miniature")

    def test_the_run_summary_lands_on_disk(self):
        self.set_plan([{"uid": "alpha", "text": "alpha"}])
        self.walk()
        summary = jsonio.read(self.project.path(".cache", "last_run.json"))
        self.assertEqual(summary["stages"][0]["stage"], "_selftest")
        self.assertEqual(summary["stages"][0]["units"][0]["status"], "built")


if __name__ == "__main__":
    unittest.main()






