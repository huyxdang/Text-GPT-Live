from __future__ import annotations

import json
import unittest
from pathlib import Path

from train.g1_repair import (
    REPAIR_COUNTS,
    build_ladder_validation_cases,
    build_repair_rows,
    pack_repair_batches,
)
from scripts.g1_repair_train import build_focused_cases, finalize_repair_comparison


ROOT = Path(__file__).resolve().parent.parent
FULL_CORPUS_PATHS = [
    *(ROOT / "data" / f"train_g1-{index:05d}-of-00004.jsonl" for index in range(1, 5)),
    ROOT / "data" / "dev_g1.jsonl",
]
FULL_CORPUS_AVAILABLE = all(path.is_file() for path in FULL_CORPUS_PATHS)


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@unittest.skipUnless(
    FULL_CORPUS_AVAILABLE,
    "requires the generated full g1 corpus; run `python -m scripts.g1_full_build` first",
)
class G1RepairCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = [
            row
            for index in range(1, 5)
            for row in load_jsonl(ROOT / "data" / f"train_g1-{index:05d}-of-00004.jsonl")
        ]
        cls.dev = load_jsonl(ROOT / "data" / "dev_g1.jsonl")

    def test_repair_corpus_is_exact_deterministic_and_leak_free(self) -> None:
        first, manifest = build_repair_rows(self.train, self.dev)
        second, second_manifest = build_repair_rows(self.train, self.dev)
        self.assertEqual(first, second)
        self.assertEqual(manifest, second_manifest)
        self.assertEqual(len(first), 80)
        self.assertEqual(manifest["counts"], REPAIR_COUNTS)
        self.assertEqual(manifest["fire_modes"], {"fire-silent": 12, "fire-typing": 12})
        self.assertEqual(manifest["dev_episode_overlap"], 0)
        self.assertEqual(manifest["dev_prompt_overlap"], 0)
        self.assertEqual(manifest["duplicate_prompts"], 0)

    def test_batches_are_five_balanced_updates(self) -> None:
        rows, _ = build_repair_rows(self.train, self.dev)
        batches = pack_repair_batches(rows)
        self.assertEqual([len(batch) for batch in batches], [16] * 5)
        self.assertEqual(sum(len(batch) for batch in batches), 80)
        for batch in batches:
            self.assertIn(sum(row["repair_group"] == "fire" for row in batch), {4, 5})
            self.assertIn(sum(row["repair_group"] == "matched_wait" for row in batch), {4, 5})

    def test_focused_suite_is_exact_and_unique(self) -> None:
        cases = build_focused_cases(self.dev)
        self.assertEqual(len(cases), 65)
        self.assertEqual(len({case["prompt"] for case in cases}), 65)
        groups = {group: sum(case["focus_group"] == group for case in cases) for group in {
            "fire", "matched_wait", "silence", "reminder_restraint", "collision"
        }}
        self.assertEqual(
            groups,
            {"fire": 14, "matched_wait": 14, "silence": 25, "reminder_restraint": 8, "collision": 4},
        )

    def test_ladder_validation_is_train_only_and_disjoint(self) -> None:
        repair, _ = build_repair_rows(self.train, self.dev)
        first, audit = build_ladder_validation_cases(self.train, repair, self.dev)
        second, second_audit = build_ladder_validation_cases(self.train, repair, self.dev)
        self.assertEqual(first, second)
        self.assertEqual(audit, second_audit)
        self.assertEqual(len(first), 28)
        self.assertEqual(audit["pairs"], 14)
        self.assertEqual(audit["schedules"], 14)
        self.assertEqual(audit["fire_silent"], 7)
        self.assertEqual(audit["fire_typing"], 7)
        self.assertEqual(audit["repair_episode_overlap"], 0)
        self.assertEqual(audit["repair_prompt_overlap"], 0)
        self.assertEqual(audit["frozen_episode_overlap"], 0)
        self.assertEqual(audit["frozen_prompt_overlap"], 0)
        self.assertEqual({case["source"] for case in first}, {"train-only-ladder-validation"})


class G1RepairPromotionTests(unittest.TestCase):
    def test_promotion_fails_closed_until_semantic_review_passes(self) -> None:
        deterministic = {"deterministic_gates_passed": True}
        pending = finalize_repair_comparison(
            deterministic, semantic_regression_status="pending_separate_review"
        )
        self.assertFalse(pending["passed"])
        self.assertFalse(pending["promotion_eligible"])
        failed = finalize_repair_comparison(
            deterministic, semantic_regression_status="failed"
        )
        self.assertFalse(failed["passed"])
        passed = finalize_repair_comparison(
            deterministic, semantic_regression_status="passed"
        )
        self.assertTrue(passed["passed"])
        self.assertTrue(passed["promotion_eligible"])


if __name__ == "__main__":
    unittest.main()
