"""The main prompt, its analysis, and staging: everything before a GPU.

The worked example in ``examples/keyumars/main_prompt.txt`` is the fixture,
because it is what the seed prompt's format produces and what the GUI's "Load
example" loads -- if it stops parsing cleanly, the flow is broken for everyone.
The model is replaced by a scripted stand-in, so the checks the analysis makes
on a model's reply (Persian lines unchanged, clips from the vocabulary) are
tested against replies that fail them.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import (analysis, intake, main_prompt, moves, project,  # noqa
                      staging, validate)

EXAMPLE = os.path.join(ROOT, "examples", "keyumars", "main_prompt.txt")


def example():
    with open(EXAMPLE, encoding="utf-8") as handle:
        return handle.read()


class FakeChat:
    """Replies from a script, in order; records what it was asked."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.asked = []

    def text(self, system, user, **_kwargs):
        self.asked.append(user)
        return self.replies.pop(0) if self.replies else "{}"


class TestParse(unittest.TestCase):

    def test_the_example_parses_without_errors(self):
        doc, problems = main_prompt.parse(example())
        self.assertEqual([], [str(p) for p in problems
                              if p.severity == "error"])
        self.assertEqual(6, len(doc["characters"]))
        self.assertEqual(3, len(doc["scenes"]))
        self.assertEqual(6, len(doc["shots"]))
        self.assertTrue(doc["film"]["narrator"])

    def test_every_persian_line_arrives_byte_for_byte(self):
        text = example()
        doc, _ = main_prompt.parse(text)
        parsed = [l["text_fa"] for s in doc["shots"] for l in s["lines"]]
        self.assertEqual(main_prompt.persian_lines(text), parsed)

    def test_line_ids_are_the_shot_and_the_ordinal(self):
        doc, _ = main_prompt.parse(example())
        command = next(s for s in doc["shots"] if s["id"] == "s06_command")
        self.assertEqual(["s06_command_1", "s06_command_2"],
                         [l["line_id"] for l in command["lines"]])

    def test_narration_needs_no_cast_member(self):
        doc, problems = main_prompt.parse(example())
        first = doc["shots"][0]
        self.assertEqual("narrator", first["lines"][0]["speaker"])
        self.assertFalse(any("narrator" in p.message for p in problems))

    def test_a_speaker_outside_the_cast_is_a_problem(self):
        text = example().replace("- siamak [resolute]:", "- lion [resolute]:")
        _doc, problems = main_prompt.parse(text)
        self.assertTrue(any("lion" in p.message and "cast" in p.message
                            for p in problems))

    def test_narrator_is_not_a_character_id(self):
        text = example().replace("## gazelle", "## narrator")
        _doc, problems = main_prompt.parse(text)
        self.assertTrue(any("reserved" in p.message for p in problems))

    def test_a_fenced_prompt_with_preamble_still_parses(self):
        doc, problems = main_prompt.parse(
            "Here is the main prompt:\n```text\n" + example() + "\n```\n")
        self.assertEqual(6, len(doc["shots"]))
        self.assertFalse([p for p in problems if p.severity == "error"])

    def test_nothing_recognisable_raises(self):
        with self.assertRaises(main_prompt.MainPromptError):
            main_prompt.parse("Once upon a time there was a king.")

    def test_take_ids_fit_the_schema(self):
        take = main_prompt.take_id("s" + "x" * 46, "black_div")
        self.assertLessEqual(len(take), 48)
        self.assertTrue(take.endswith("__black_div"))


class TestStaging(unittest.TestCase):

    def members(self, *items):
        return [{"character": c, "placement": p, "height_m": 1.8,
                 "kind": "human"} for c, p in items]

    def test_facing_follows_the_rig_convention(self):
        """0 is +Y, 90 is -X (counter-clockwise), 180 faces the camera."""
        self.assertAlmostEqual(0.0, staging.facing_to((0, 0), (0, 5)))
        self.assertAlmostEqual(90.0, staging.facing_to((0, 0), (-5, 0)))
        self.assertAlmostEqual(-90.0, staging.facing_to((0, 0), (5, 0)))

    def test_left_is_screen_left(self):
        placed, _ = staging.place_cast(self.members(
            ("a", "left, middle, facing left"),
            ("b", "right, middle, facing camera")))
        self.assertLess(placed["a"]["position_m"][0], 0)
        self.assertEqual(90.0, placed["a"]["facing_deg"])
        self.assertEqual(180.0, placed["b"]["facing_deg"])

    def test_a_character_facing_upstage_cheats_toward_the_camera(self):
        placed, _ = staging.place_cast(self.members(
            ("king", "center, back, facing camera"),
            ("angel", "left, front, facing king")))
        away = abs(staging._wrap(placed["angel"]["facing_deg"] - 180.0))
        self.assertLessEqual(away, staging.MAX_TURN_FROM_CAMERA_DEG + 1e-6)

    def test_two_on_one_mark_are_pushed_apart(self):
        placed, notes = staging.place_cast(self.members(
            ("a", "center, middle"), ("b", "center, middle")))
        gap = abs(placed["a"]["position_m"][0] - placed["b"]["position_m"][0])
        self.assertGreaterEqual(gap, staging.MIN_SPACING_M - 1e-6)
        self.assertTrue(notes)

    def test_a_closer_size_puts_the_camera_closer(self):
        placed, _ = staging.place_cast(self.members(("k", "center, middle")))
        bodies = {"k": {"height_m": 1.8, "kind": "human"}}
        distances = []
        for size in ("wide", "full", "medium", "close_up"):
            _cam, framing, _ = staging.solve_camera(f"{size} on k", placed,
                                                    bodies)
            distances.append(framing["distance_m"])
        self.assertEqual(sorted(distances, reverse=True), distances)

    def test_the_camera_is_in_front_of_the_stage(self):
        placed, _ = staging.place_cast(self.members(("k", "center, back")))
        camera, _f, _n = staging.solve_camera(
            "medium on k, high_angle, slow push_in", placed,
            {"k": {"height_m": 1.9, "kind": "human"}})
        self.assertLess(camera["position_m"][1], placed["k"]["position_m"][1])
        self.assertEqual("push_in", camera["move"]["kind"])
        self.assertEqual("k.chest", camera["aim"])

    def test_fighters_close_in_and_face_each_other(self):
        """Placed left and right, 3.4 m apart; a fight brings them to about
        half their height apart, facing square."""
        placed, _ = staging.place_cast(self.members(
            ("siamak", "left, middle, facing div"),
            ("div", "right, middle, facing siamak")))
        bodies = {"siamak": {"height_m": 1.8}, "div": {"height_m": 1.8}}
        notes = staging.engage(
            placed, {"siamak": [{"clip": "fight", "target": "div"}],
                     "div": [{"clip": "fight", "target": "siamak"}]}, bodies)
        a, b = placed["siamak"]["position_m"], placed["div"]["position_m"]
        self.assertAlmostEqual(0.9, math.hypot(b[0] - a[0], b[1] - a[1]),
                               places=2)
        self.assertAlmostEqual(0.0, (a[0] + b[0]) / 2.0, places=3)
        self.assertEqual(-90.0, placed["siamak"]["facing_deg"])
        self.assertEqual(90.0, placed["div"]["facing_deg"])
        self.assertEqual(1, len(notes))

    def test_a_blow_is_struck_from_arm_reach(self):
        placed, _ = staging.place_cast(self.members(
            ("siamak", "center, front"), ("div", "right, middle")))
        staging.engage(placed, {"div": [{"clip": "strike",
                                         "target": "siamak"}],
                                "siamak": [{"clip": "fall",
                                            "target": "div"}]},
                       {"siamak": {"height_m": 1.8}, "div": {"height_m": 1.8}})
        a, b = placed["siamak"]["position_m"], placed["div"]["position_m"]
        self.assertAlmostEqual(1.08, math.hypot(b[0] - a[0], b[1] - a[1]),
                               places=2)
        self.assertEqual("div", placed["siamak"]["facing_target"])
        # Across the screen, not along the line they stood on: nobody's
        # back to the camera.
        self.assertEqual(a[1], b[1])
        self.assertEqual(-90.0, placed["siamak"]["facing_deg"])

    def test_no_contact_no_change(self):
        placed, _ = staging.place_cast(self.members(
            ("a", "left, middle"), ("b", "right, middle")))
        before = json.dumps(placed, sort_keys=True)
        notes = staging.engage(placed, {"a": [{"clip": "walk",
                                               "target": "b"}]}, {})
        self.assertEqual(before, json.dumps(placed, sort_keys=True))
        self.assertEqual([], notes)

    def test_commas_do_not_hide_the_subject(self):
        (_size, subject, angle, move, slow), notes = staging.parse_camera(
            "wide on group, eye_level, slow push_in", ["a", "b"])
        self.assertEqual(("group", "eye_level", "push_in", True),
                         (subject, angle, move, slow))
        self.assertEqual([], notes)


class TestMoves(unittest.TestCase):

    def test_an_unknown_clip_and_target_are_named(self):
        problems = moves.check([{"clip": "moonwalk", "target": "zal"}],
                               "human", {"rostam", "camera"})
        self.assertEqual(2, len(problems))

    def test_animals_have_their_own_clips(self):
        self.assertTrue(moves.check([{"clip": "kneel"}], "quadruped", set()))
        self.assertFalse(moves.check([{"clip": "graze"}], "quadruped", set()))

    def test_the_keyword_floor_reads_sequences_in_order(self):
        clips = moves.guess("walks to the throne and kneels", "human")
        self.assertEqual(["walk", "kneel"], [c["clip"] for c in clips])

    def test_talking_and_idling_share_a_take_and_actions_do_not(self):
        talk = {"take": "s1__k", "action": {"clips": [{"clip": "talk"}]}}
        act = {"take": "s1__k", "action": {"clips": [{"clip": "bow"}]}}
        self.assertEqual(("talk_human", True), moves.resolve_take(talk, "human"))
        self.assertEqual(("s1__k", False), moves.resolve_take(act, "human"))
        self.assertEqual(("s1__k", False),
                         moves.resolve_take(talk, "human", has_video=True))
        self.assertEqual(("idle_quadruped", True), moves.resolve_take(
            {"take": "s1__l", "action": {"clips": [{"clip": "idle"}]}},
            "quadruped"))


class TestAnalysis(unittest.TestCase):

    def test_without_a_model_the_example_becomes_a_valid_plan(self):
        result = analysis.analyse(example(), chat=None, log=lambda m: None)
        self.assertEqual(0, result["errors"])
        production = analysis.to_production(result)
        report = validate.check_production(production)
        self.assertFalse(report.errors, report.text())

    def test_the_model_s_clips_are_used_when_they_check_out(self):
        reply = json.dumps({"moves": [{"character": "keyumars",
                                       "clips": [{"clip": "raise_arm"}]}]})
        chat = FakeChat(*([reply] + ["{}"] * 20))
        result = analysis.analyse(example(), chat=chat, log=lambda m: None)
        king = result["shots"][0]["cast"][0]
        self.assertEqual("model", king["move"]["source"])
        self.assertEqual([{"clip": "raise_arm"}], king["move"]["clips"])

    def test_clips_outside_the_vocabulary_fall_back_to_keywords(self):
        bad = json.dumps({"moves": [{"character": "keyumars",
                                     "clips": [{"clip": "levitate"}]}]})
        chat = FakeChat(bad, bad, *(["{}"] * 20))
        result = analysis.analyse(example(), chat=chat, log=lambda m: None)
        king = result["shots"][0]["cast"][0]
        self.assertEqual("keywords", king["move"]["source"])
        self.assertIn("Your last reply had problems", chat.asked[1])

    def test_a_repair_that_changes_persian_is_rejected(self):
        broken = example().replace("scene: battlefield", "scene: nowhere")
        rewritten = broken.replace("scene: nowhere", "scene: battlefield") \
            .replace("بیدار باش", "بیدار شو")
        chat = FakeChat(rewritten, rewritten, *(["{}"] * 20))
        logs = []
        result = analysis.analyse(broken, chat=chat, log=logs.append)
        self.assertTrue(any("changed" in m and "Persian" in m for m in logs))
        self.assertGreater(result["errors"], 0)

    def test_a_faithful_repair_is_accepted(self):
        broken = example().replace("scene: battlefield", "scene: nowhere")
        chat = FakeChat(example(), *(["{}"] * 20))
        result = analysis.analyse(broken, chat=chat, log=lambda m: None)
        self.assertEqual(0, result["errors"])
        self.assertIn("repair", result["llm_used"])

    def test_restaging_needs_no_model_and_keeps_the_clips(self):
        result = analysis.analyse(example(), chat=None, log=lambda m: None)
        clips = [m["move"]["clips"] for s in result["shots"] for m in s["cast"]]
        again = analysis.restage(json.loads(json.dumps(result)))
        self.assertEqual(clips, [m["move"]["clips"] for s in again["shots"]
                                 for m in s["cast"]])


class TestIntake(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="intake_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.saved = project.PROJECTS_ROOT
        project.PROJECTS_ROOT = self.root
        intake.Project = project.Project
        self.addCleanup(setattr, project, "PROJECTS_ROOT", self.saved)
        self.result = analysis.analyse(example(), chat=None,
                                       log=lambda m: None)

    def test_a_project_is_created_with_its_story_files(self):
        made = intake.create("demo", example(), self.result)
        for name in ("main_prompt.md", "analysis.json", "production.json"):
            self.assertTrue(os.path.exists(made.path("story", name)), name)
        self.assertEqual("stylized", made.config["style_pack"])

    def test_every_panel_row_starts_missing_except_the_optional_ones(self):
        made = intake.create("demo", example(), self.result)
        state = intake.status(made, self.result)
        self.assertEqual(6 + 3 + 7, len(state["missing"]))
        self.assertFalse(any("voice sample" in m for m in state["missing"]))

    def test_an_attached_image_lands_where_b_refs_looks(self):
        from PIL import Image
        made = intake.create("demo", example(), self.result)
        source = os.path.join(self.root, "k.jpg")
        Image.new("RGB", (8, 12), (200, 100, 50)).save(source)
        path = intake.attach(made, "character_image", "keyumars", source)
        self.assertTrue(path.endswith(os.path.join(
            "refs", "characters", "keyumars", "keyumars_front.png")))
        state = intake.status(made, self.result)
        self.assertEqual(6 + 3 + 7 - 1, len(state["missing"]))


class TestMotionLibrary(unittest.TestCase):

    def setUp(self):
        from pipeline import motion_library
        self.ml = motion_library

    def test_every_clip_builds_finite_poses(self):
        for kind, table in (("human", self.ml.HUMAN),
                            ("quadruped", self.ml.QUADRUPED)):
            for name in table:
                ctx = self.ml.Context(kind, 24, (0, 0, 0), 180.0,
                                      {"other": (2.0, 1.0, 0.0)}, 1.1)
                body, head, root, report = self.ml.build(
                    [{"clip": name, "target": "other"}], ctx)
                values = [v for p in body for r in p.values() for v in r] \
                    + [v for r in root for v in r]
                self.assertTrue(all(math.isfinite(v) for v in values),
                                f"{kind}/{name}")
                self.assertEqual(len(body), len(head))

    def test_a_crouch_keeps_the_feet_on_the_floor(self):
        """Leg IK: pelvis 20 cm down, ankles still at their rest height."""
        p = self.ml._at({"root": (0.0, 0.0, 0.0), "heading": 30.0},
                        root=(0.0, 0.0, -0.2))
        q = self.ml.q_mul(self.ml.q_axis((0, 0, 1), p["heading"]),
                          self.ml.q_euler(*p["pelvis"]))
        for side in ("left", "right"):
            thigh, shin = (self.ml.q_rotate(q, d) for d in p[f"leg_{side}"])
            offset = tuple(h - c for h, c in zip(self.ml.HIP_REST[side],
                                                 self.ml.PELVIS_REST))
            hip_z = p["root"][2] + self.ml.PELVIS_REST[2] + \
                self.ml.q_rotate(q, offset)[2]
            ankle_z = hip_z + 0.39 * thigh[2] + 0.39 * shin[2]
            self.assertAlmostEqual(self.ml.ANKLE_REST_Z, ankle_z, places=3)

    def test_walking_to_a_target_stops_short_of_them(self):
        ctx = self.ml.Context("human", 24, (0, 0, 0), 180.0,
                              {"king": (0.0, -4.0, 0.0)}, 1.0)
        _b, _h, _r, report = self.ml.build([{"clip": "walk",
                                             "target": "king"}], ctx)
        travelled = math.hypot(*report["final_root_m"][:2])
        self.assertAlmostEqual(4.0 - 0.9, travelled, delta=0.05)


if __name__ == "__main__":
    unittest.main()
