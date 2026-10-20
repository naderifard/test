"""Validation: the cross-references a JSON Schema cannot see.

``production.json`` is section 6.1's contract with a chat model that has never
read this codebase, so it is the one file in the project most likely to arrive
subtly wrong -- an env id that does not exist, a character who speaks in a shot
he is not cast in, two lines sharing an id. None of those is a schema violation;
all of them break a later stage, and by then the message is about a missing file
rather than about the mistake.

Two properties are what make ``validate.py`` worth having, and they are what is
tested here:

* **every problem is reported at once**, because handing a model one error at a
  time turns one round trip into four;
* **every problem points at the data**, as a JSON pointer into the document, so
  a human editing the file by hand knows which line to look at.

A test that only asserted "this is invalid" would let both of those rot.
"""

from __future__ import annotations

import copy
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import jsonio, project as project_mod, validate    # noqa: E402
from pipeline.project import Project                             # noqa: E402

# A minimal production that validates cleanly. Every test below breaks exactly
# one thing in a copy of it, so a failure names the rule it broke.
GOOD = {
    "title": "رستم و سهراب",
    "fps": 24,
    "characters": [
        {"id": "rostam", "name_fa": "رستم", "kind": "human", "height_m": 2.1,
         "image_prompt": "an immense grey-bearded paladin in scale armour",
         "material_roles": {"skin": "skin"}},
        {"id": "rakhsh", "name_fa": "رخش", "kind": "quadruped",
         "height_m": 1.95,
         "image_prompt": "an enormous dappled chestnut stallion",
         "material_roles": {"hide": "skin"}},
    ],
    "envs": [
        {"id": "river_plain", "interior": False, "ground": "ground",
         "image_prompt": "a flowering meadow beside a slow river",
         "light_intent": "low afternoon sun from the left"},
    ],
    "props": [
        {"id": "gorz_mace", "role": "metal_blade", "size_m": [0.22, 0.22, 1.15],
         "image_prompt": "a heavy ox-headed Persian war mace",
         "held_by": "rostam", "attach": "hand_r"},
    ],
    "shots": [
        {
            "id": "s01",
            "env": "river_plain",
            "cast": [{"character": "rostam", "position_m": [0.0, 0.0, 0.0],
                      "facing_deg": 90, "retime": "loop",
                      "props": ["gorz_mace"]}],
            "dialogue": [{"line_id": "rostam_01", "character": "rostam",
                          "text_fa": "نام من تیغ من است.", "emotion": "grave"}],
            "camera": {"position_m": [0.0, -5.0, 1.7], "aim": "rostam.head",
                       "focal_mm": 50},
            "beats": [{"from_pct": 0.0, "to_pct": 0.5, "what": "he answers"},
                      {"from_pct": 0.5, "to_pct": 1.0, "what": "he looks away"}],
        },
    ],
}


def broken(**changes):
    """A copy of GOOD with one thing wrong, described by a mutator."""
    document = copy.deepcopy(GOOD)
    for _name, mutate in changes.items():
        mutate(document)
    return document


class ReportAssertions(unittest.TestCase):

    def pointers(self, report):
        return {p.pointer for p in report.problems}

    def messages(self, report):
        return "\n".join(str(p) for p in report.problems)

    def assertProblemAt(self, report, pointer, severity="error"):
        found = [p for p in report.problems if p.pointer == pointer]
        self.assertTrue(found,
                        f"expected a problem at {pointer}; got:\n"
                        + (self.messages(report) or "  (none)"))
        self.assertEqual(found[0].severity, severity, found[0].message)
        return found[0]


class TestBaseline(ReportAssertions):

    def test_the_good_document_has_no_errors(self):
        """Everything below is a delta from this, so this must be clean."""
        report = validate.check_production(GOOD)
        self.assertEqual(report.errors, [], self.messages(report))
        self.assertTrue(report)

    def test_the_real_project_still_validates(self):
        """The production.json the repo's example project ships."""
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "projects", "keyumars", "story",
            "production.json")
        if not os.path.exists(path):
            self.skipTest("the keyumars project is not present")
        report = validate.check_production(jsonio.read(path))
        self.assertEqual(report.errors, [], self.messages(report))


class TestCrossReferences(ReportAssertions):
    """The checks a schema cannot express, which is why they exist at all."""

    def test_a_shot_in_an_env_that_does_not_exist(self):
        def mutate(d):
            d["shots"][0]["env"] = "helmand_bank"
        report = validate.check_production(broken(env=mutate))
        problem = self.assertProblemAt(report, "/shots/0/env")
        self.assertIn("river_plain", problem.message,
                      "say what the defined envs are, not just that it is wrong")

    def test_a_cast_member_who_is_not_a_character(self):
        def mutate(d):
            d["shots"][0]["cast"][0]["character"] = "sohrab"
        self.assertProblemAt(validate.check_production(broken(cast=mutate)),
                             "/shots/0/cast/0/character")

    def test_a_prop_nobody_can_hold(self):
        def mutate(d):
            d["props"][0]["held_by"] = "esfandiyar"
        self.assertProblemAt(validate.check_production(broken(prop=mutate)),
                             "/props/0/held_by")

    def test_a_prop_a_cast_member_carries_that_does_not_exist(self):
        def mutate(d):
            d["shots"][0]["cast"][0]["props"] = ["kamand"]
        self.assertProblemAt(validate.check_production(broken(carry=mutate)),
                             "/shots/0/cast/0/props/0")

    def test_someone_speaking_who_is_not_in_the_shot(self):
        def mutate(d):
            d["shots"][0]["dialogue"][0]["character"] = "rakhsh"
        self.assertProblemAt(validate.check_production(broken(who=mutate)),
                             "/shots/0/dialogue/0/character")

    def test_the_same_character_cast_twice_in_one_shot(self):
        def mutate(d):
            d["shots"][0]["cast"].append(dict(d["shots"][0]["cast"][0]))
        self.assertProblemAt(validate.check_production(broken(twice=mutate)),
                             "/shots/0/cast")

    def test_duplicate_ids_are_caught_for_every_kind_of_thing(self):
        for key, pointer in (("characters", "/characters/2/id"),
                             ("envs", "/envs/1/id"),
                             ("shots", "/shots/1/id")):
            def mutate(d, key=key):
                d[key].append(copy.deepcopy(d[key][0]))
            report = validate.check_production(broken(dup=mutate))
            self.assertProblemAt(report, pointer)

    def test_a_line_id_reused_across_shots_is_an_error(self):
        """Line ids are the audio directory names; two shots cannot share one."""
        def mutate(d):
            second = copy.deepcopy(d["shots"][0])
            second["id"] = "s02"
            d["shots"].append(second)
        report = validate.check_production(broken(reuse=mutate))
        problem = self.assertProblemAt(report, "/shots/1/dialogue/0/line_id")
        self.assertIn("audio directory", problem.message)

    def test_a_silent_shot_must_say_how_long_it_is(self):
        """With no audio there is nothing to derive a length from (section 5.2)."""
        def mutate(d):
            d["shots"][0].pop("dialogue")
        report = validate.check_production(broken(silent=mutate))
        problem = self.assertProblemAt(report, "/shots/0")
        self.assertIn("silent_duration_s", problem.message)

    def test_a_silent_shot_with_a_duration_is_fine(self):
        def mutate(d):
            d["shots"][0].pop("dialogue")
            d["shots"][0]["silent_duration_s"] = 4.0
        report = validate.check_production(broken(silent=mutate))
        self.assertEqual(report.errors, [], self.messages(report))

    def test_every_problem_is_reported_in_one_pass(self):
        """One error at a time turns one round trip with a model into four."""
        def mutate(d):
            d["shots"][0]["env"] = "nowhere"
            d["shots"][0]["cast"][0]["character"] = "nobody"
            d["props"][0]["held_by"] = "no_one"
        report = validate.check_production(broken(several=mutate))
        self.assertGreaterEqual(len(report.errors), 3, self.messages(report))


class TestCamera(ReportAssertions):

    def test_the_camera_cannot_aim_at_someone_offstage(self):
        def mutate(d):
            d["shots"][0]["camera"]["aim"] = "rakhsh.head"
        self.assertProblemAt(validate.check_production(broken(aim=mutate)),
                             "/shots/0/camera/aim")

    def test_an_anchor_the_rig_has_not_got_is_rejected(self):
        def mutate(d):
            d["shots"][0]["camera"]["aim"] = "rostam.elbow"
        self.assertProblemAt(validate.check_production(broken(anchor=mutate)),
                             "/shots/0/camera/aim")

    def test_the_anchor_list_is_declared_in_exactly_one_shape(self):
        """The schema pattern and validate.ANCHORS are two copies of one list.

        The schema catches a bad anchor first, so ``_check_aim``'s own anchor
        branch only runs where jsonschema is absent -- which makes drift between
        the two silent until the day it matters. This pins them together.
        """
        pattern = validate.load_schema("production")["$defs"]["aim"]["oneOf"][0]
        alternatives = pattern["pattern"].split("(", 2)[2].split(")")[0]
        self.assertEqual(set(alternatives.split("|")), set(validate.ANCHORS))

    def test_aiming_at_a_prop_or_the_env_is_allowed(self):
        for target in ("env.gorz_mace", "env.river_plain"):
            def mutate(d, target=target):
                d["shots"][0]["camera"]["aim"] = target
            report = validate.check_production(broken(aim=mutate))
            self.assertEqual(report.errors, [], f"{target}: "
                                                + self.messages(report))

    def test_a_push_in_without_a_destination_is_an_error(self):
        def mutate(d):
            d["shots"][0]["camera"]["move"] = {"kind": "push_in"}
        self.assertProblemAt(validate.check_production(broken(move=mutate)),
                             "/shots/0/camera/move")

    def test_an_orbit_without_an_angle_is_an_error(self):
        def mutate(d):
            d["shots"][0]["camera"]["move"] = {"kind": "orbit"}
        self.assertProblemAt(validate.check_production(broken(move=mutate)),
                             "/shots/0/camera/move")


class TestBeatRanges(ReportAssertions):
    """Beats are in normalised shot time because the frame count does not
    exist yet -- section 7-A's non-overlap gate has to be enforced there."""

    def test_a_beat_that_ends_before_it_starts(self):
        def mutate(d):
            d["shots"][0]["beats"][0]["to_pct"] = 0.0
        self.assertProblemAt(validate.check_production(broken(beat=mutate)),
                             "/shots/0/beats/0")

    def test_overlapping_beats_warn_rather_than_fail(self):
        def mutate(d):
            d["shots"][0]["beats"][1]["from_pct"] = 0.2
        report = validate.check_production(broken(beat=mutate))
        self.assertProblemAt(report, "/shots/0/beats/1", severity="warn")
        self.assertEqual(report.errors, [], "overlap is a note, not a stop")


class TestPlausibility(ReportAssertions):

    def test_an_implausible_human_height_warns_but_does_not_stop_the_run(self):
        def mutate(d):
            d["characters"][0]["height_m"] = 0.4
        report = validate.check_production(broken(height=mutate))
        self.assertProblemAt(report, "/characters/0/height_m", severity="warn")
        self.assertEqual(report.errors, [])

    def test_a_broken_structure_stops_before_the_pointers_become_noise(self):
        report = validate.check_production({"title": "x"})
        self.assertTrue(report.errors)
        self.assertTrue(all(p.severity == "error" for p in report.problems))


class TestRigModeResolution(ReportAssertions):
    """Section 6.2: overrides, then rig_default, then derived from kind."""

    def test_the_override_table_wins(self):
        character = {"id": "simorgh", "kind": "creature", "rig_default": "smplx"}
        self.assertEqual(validate.resolve_rig_mode(character,
                                                   {"simorgh": "rigged"}),
                         ("rigged", "override"))

    def test_rig_default_beats_the_kind(self):
        character = {"id": "div", "kind": "creature", "rig_default": "smplx"}
        self.assertEqual(validate.resolve_rig_mode(character, {}),
                         ("smplx", "rig_default"))

    def test_the_kind_is_the_last_resort(self):
        self.assertEqual(
            validate.resolve_rig_mode({"id": "rostam", "kind": "human"}, {}),
            ("smplx", "kind"))
        self.assertEqual(
            validate.resolve_rig_mode({"id": "rakhsh", "kind": "quadruped"}, {}),
            ("rigged", "kind"))

    def test_an_unknown_kind_falls_back_to_the_rigged_path(self):
        """A creature nobody classified must not be handed to a mocap rig."""
        self.assertEqual(
            validate.resolve_rig_mode({"id": "x", "kind": "chimera"}, {}),
            ("rigged", "kind"))


class TestProjectInputs(ReportAssertions):
    """The on-disk half: does every line have the recording that times it?"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="validate-")
        self._saved = project_mod.PROJECTS_ROOT
        project_mod.PROJECTS_ROOT = os.path.join(self.tmp, "projects")
        self.project = Project.create("scratch", title="Scratch")
        jsonio.write(self.project.production_path, GOOD)

    def tearDown(self):
        project_mod.PROJECTS_ROOT = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def record(self, line_id):
        import struct
        import wave
        path = os.path.join(self.project.audio_dir(line_id), "source.wav")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with wave.open(path, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(16000)
            fh.writeframes(struct.pack("<h", 0) * 16000)
        return path

    def test_a_line_with_no_recording_is_an_error_not_a_warning(self):
        """The audio is the master clock; an untimed line cannot be built."""
        report = validate.check_project_inputs(self.project)
        problem = self.assertProblemAt(report, "/shots/0/dialogue/0/line_id")
        self.assertIn("master clock", problem.message)

    def test_a_recorded_line_leaves_only_the_missing_reference_warnings(self):
        self.record("rostam_01")
        report = validate.check_project_inputs(self.project)
        self.assertEqual(report.errors, [], self.messages(report))
        self.assertEqual(len(report.warnings), 2,
                         "one per character with an empty refs folder")

    def test_a_named_take_with_no_video_warns_because_a_stub_can_stand_in(self):
        self.record("rostam_01")
        document = copy.deepcopy(GOOD)
        document["shots"][0]["cast"][0]["take"] = "take_042"
        jsonio.write(self.project.production_path, document)
        report = validate.check_project_inputs(self.project)
        self.assertProblemAt(report, "/shots/0/cast/0/take", severity="warn")
        self.assertEqual(report.errors, [])


class TestReport(unittest.TestCase):

    def test_a_report_is_truthy_only_when_it_has_no_errors(self):
        report = validate.Report()
        self.assertTrue(report)
        report.add("/a", "just a note", "warn")
        self.assertTrue(report, "warnings do not make a document invalid")
        report.add("/b", "a real problem")
        self.assertFalse(report)

    def test_as_dict_counts_both_kinds_separately(self):
        report = validate.Report()
        report.add("/a", "one", "warn")
        report.add("/b", "two")
        payload = report.as_dict()
        self.assertEqual((payload["errors"], payload["warnings"]), (1, 1))
        self.assertFalse(payload["ok"])

    def test_extend_merges_without_losing_severity(self):
        first, second = validate.Report(), validate.Report()
        first.add("/a", "one", "warn")
        second.add("/b", "two")
        first.extend(second)
        self.assertEqual(len(first.errors), 1)
        self.assertEqual(len(first.warnings), 1)


class TestSchemaPhrasing(unittest.TestCase):
    """The stock jsonschema messages quote the whole failing subtree."""

    def test_a_bad_enum_lists_what_was_allowed(self):
        document = copy.deepcopy(GOOD)
        document["shots"][0]["cast"][0]["retime"] = "reverse"
        report = validate.check_schema(document, "production")
        if not report.problems:
            self.skipTest("jsonschema is not installed in this interpreter")
        message = "\n".join(p.message for p in report.problems)
        self.assertIn("'reverse'", message)
        self.assertIn("hold_last_pose", message)

    def test_a_misspelled_field_says_the_fields_are_closed(self):
        document = copy.deepcopy(GOOD)
        document["shots"][0]["camera"]["focal_length_mm"] = 50
        report = validate.check_schema(document, "production")
        if not report.problems:
            self.skipTest("jsonschema is not installed in this interpreter")
        self.assertIn("check the spelling",
                      "\n".join(p.message for p in report.problems))

    def test_every_schema_in_the_directory_loads(self):
        for name in ("production", "shot", "project", "rig_overrides",
                     "align", "visemes"):
            schema = validate.load_schema(name)
            self.assertIn("$schema", schema, name)


class TestProjectDefaultsMatchTheSchema(unittest.TestCase):
    """PROJECT_DEFAULTS and project.schema.json are two lists of one thing.

    project.json's schema is closed, so a setting added to PROJECT_DEFAULTS and
    not to the schema turns every ``pipeline validate`` into an error the moment
    a project actually writes it -- which is exactly how ``cache_dir`` shipped:
    the store moved, the run worked, the tests passed, and validate failed on a
    field the pipeline itself had just written.
    """

    def test_every_default_is_a_field_the_schema_allows(self):
        allowed = set(validate.load_schema("project")["properties"])
        unknown = sorted(set(project_mod.PROJECT_DEFAULTS) - allowed)
        self.assertEqual(
            [], unknown,
            "PROJECT_DEFAULTS writes %s, which project.schema.json does not "
            "declare and will reject" % unknown)

    def test_a_project_written_from_the_defaults_validates(self):
        document = dict(project_mod.PROJECT_DEFAULTS)
        document.update(schema_version=1, name="defaults_probe", fps=24,
                        resolution=[960, 540], style_pack="miniature")
        report = validate.check_schema(document, "project")
        self.assertEqual([], [p.message for p in report.problems])


if __name__ == "__main__":
    unittest.main()
