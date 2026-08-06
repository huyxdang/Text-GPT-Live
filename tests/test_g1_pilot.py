from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.domain import UserState
from app.policy import SYSTEM_PROMPT_G1
from datagen.g1_pilot import (
    PILOT_FIRE_FLOOR,
    PILOT_WAIT_FLOOR,
    REQUIRED_COLLISION_PRIORITIES,
    REQUIRED_DEMOS,
    REQUIRED_EMPTY_KINDS,
    build_pilot_cards,
    has_vague_quantity_reference,
    has_vague_time_reference,
    idle,
    make_card,
    user,
    validate_pilot_coverage,
)
from scripts.g1_pilot import run_pilot
from train.g1_evaluation import evaluate_g1_predictions
from train.tinker_run import (
    completion_class,
    evaluate_pair_outputs,
    example_weight_for_pair,
    system_prompt_for_pair,
)


class G1PilotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cards = build_pilot_cards()
        self.rows = [card.row for card in self.cards]

    def test_coverage_guards_require_silence_fire_wait_and_boundaries(self) -> None:
        report = validate_pilot_coverage(self.cards)

        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(set(report["counts"]["empty_demos"]), REQUIRED_DEMOS)
        self.assertTrue(
            all(
                set(kinds) == REQUIRED_EMPTY_KINDS
                for kinds in report["counts"]["empty_matrix"].values()
            )
        )
        self.assertGreaterEqual(report["counts"]["should_fire"], PILOT_FIRE_FLOOR)
        self.assertGreaterEqual(report["counts"]["reminder_wait"], PILOT_WAIT_FLOOR)
        self.assertEqual(
            set(report["counts"]["timing_boundaries"]),
            {"after", "already-fired", "at", "before"},
        )
        self.assertEqual(
            set(report["counts"]["collision_priorities"]),
            REQUIRED_COLLISION_PRIORITIES,
        )

    def test_collision_priority_and_deferred_obligations(self) -> None:
        rows = {row["episode"]: row for row in self.rows}

        self.assertIn(
            '"message":"Still here."',
            rows["g1-pilot-collision-direct-over-reminder"]["completion"],
        )
        self.assertIn(
            '"message":"Drink water!"',
            rows["g1-pilot-collision-deferred-reminder"]["completion"],
        )
        self.assertIn(
            '"message":"Drink water!"',
            rows["g1-pilot-collision-reminder-over-standing-target"]["completion"],
        )
        self.assertEqual(
            rows["g1-pilot-collision-deferred-standing-target"]["expected_class"],
            "suggest_edit",
        )

    def test_runtime_and_spec_system_prompts_match_exactly(self) -> None:
        spec = (Path(__file__).resolve().parent.parent / "synthetic_data_spec.md").read_text(
            encoding="utf-8"
        )
        heading = spec.index("## The system prompt")
        start = spec.index("```\n", heading) + len("```\n")
        end = spec.index("\n```", start)

        self.assertEqual(spec[start:end], SYSTEM_PROMPT_G1)

    def test_ordinary_empty_text_is_idle_but_due_empty_text_fires(self) -> None:
        ordinary = [
            row
            for row in self.rows
            if row["current_content_empty"] and row["obligation"] == "none"
        ]
        due = [
            row
            for row in self.rows
            if row["current_content_empty"] and row["should_fire"]
        ]

        self.assertTrue(ordinary)
        self.assertTrue(due)
        self.assertTrue(all(row["expected_class"] == "idle" for row in ordinary))
        self.assertTrue(all(row["expected_class"] == "respond" for row in due))

    def test_always_idle_cannot_pass_or_hide_should_fire_failure(self) -> None:
        report = evaluate_g1_predictions(
            self.rows,
            ["<action>idle()</action>"] * len(self.rows),
            label="always-idle-test",
        )

        self.assertFalse(report["hard_gates"]["passed"])
        self.assertEqual(report["summary"]["should_fire_recall"], 0.0)
        self.assertGreater(report["summary"]["reminder_wait_accuracy"], 0.0)

    def test_training_path_selects_g1_prompt_and_parser(self) -> None:
        for row in self.rows:
            with self.subTest(episode=row["episode"]):
                self.assertEqual(system_prompt_for_pair(row), SYSTEM_PROMPT_G1)
                self.assertEqual(completion_class(row), row["expected_class"])
                self.assertEqual(example_weight_for_pair(row), 1.0)

    def test_release_gate_requires_approved_chinese_reference_review(self) -> None:
        release = validate_pilot_coverage(
            self.cards,
            require_reference_review=True,
        )
        self.assertTrue(release["passed"], release["errors"])
        self.assertFalse(release["warnings"])

        reviewed = next(row for row in self.rows if row.get("chinese_reference"))
        reviewed["reference_review"] = "pending-language-review"
        blocked = validate_pilot_coverage(
            self.cards,
            require_reference_review=True,
        )
        self.assertFalse(blocked["passed"])
        self.assertTrue(any("language review" in error for error in blocked["errors"]))

    def test_vague_time_references_are_detected(self) -> None:
        self.assertTrue(has_vague_time_reference("Meet me at eight."))
        self.assertTrue(has_vague_time_reference("Remind me later."))
        self.assertFalse(has_vague_time_reference("Meet me at eight tomorrow morning."))
        self.assertTrue(has_vague_quantity_reference("Highlight a few examples."))
        self.assertFalse(has_vague_quantity_reference("Highlight exactly three examples."))

    def test_ambiguity_guard_reads_actual_stream_events_not_optional_metadata(self) -> None:
        card = make_card(
            episode="ambiguous-without-source-text",
            demo="demo-1",
            situation="ordinary-text",
            history=[],
            current=user(1, "Remind me later.", UserState.IDLE, 650),
            expected=idle(),
        )
        report = validate_pilot_coverage(
            [*self.cards, card],
            require_reference_review=False,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("vague time or quantity" in error for error in report["errors"])
        )

    def test_translation_waits_before_boundary_then_emits_each_clause_once(self) -> None:
        translation = {
            row["episode"]: row
            for row in self.rows
            if row["demo"] == "demo-3"
        }
        self.assertEqual(
            translation["g1-pilot-d3-partial-clause"]["expected_class"],
            "idle",
        )
        self.assertIn(
            "今天早上市场很拥挤，",
            translation["g1-pilot-d3-first-clause-complete"]["completion"],
        )
        second = translation["g1-pilot-d3-second-clause-complete"]["completion"]
        self.assertIn("我在午饭前离开了。", second)
        self.assertNotIn("今天早上市场很拥挤，", second)

    def test_malformed_g1_gold_cannot_bypass_training_validation(self) -> None:
        malformed = dict(self.rows[0], completion=" <action>idle()</action>")
        with self.assertRaises(ValueError):
            example_weight_for_pair(malformed)

    def test_shared_evaluation_entry_point_routes_g1_to_fire_metrics(self) -> None:
        report = evaluate_pair_outputs(
            self.rows,
            [row["completion"] for row in self.rows],
            label="routing-test",
        )
        self.assertIn("should_fire_recall", report["summary"])
        self.assertEqual(report["summary"]["should_fire_recall"], 1.0)

    def test_pilot_run_writes_reloadable_artifacts_and_catches_idle_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = run_pilot(
                dataset_path=root / "pilot_g1.jsonl",
                out_dir=root / "report",
                inspection_ack="unit test",
            )

            rows = [
                json.loads(line)
                for line in (root / "pilot_g1.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), manifest["rows"])
            self.assertEqual(manifest["gold_summary"]["strict_row_accuracy"], 1.0)
            self.assertEqual(manifest["always_idle_summary"]["should_fire_recall"], 0.0)
            self.assertTrue((root / "report" / "inspection_samples.md").exists())
            self.assertEqual(manifest["manual_inspection"]["status"], "acknowledged")


if __name__ == "__main__":
    unittest.main()
