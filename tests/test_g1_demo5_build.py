from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.stream import parse_g1_action
from datagen.g1_authored_demo5 import validate_demo5_batches as _real_validate_demo5_batches
from datagen.g1_demo5 import Demo5Targets
from scripts.g1_demo5_build import (
    Demo5BuildError,
    build_demo5_artifacts,
    validate_demo5_rows,
)


SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "data" / "g1_authored_demo5_sample"

# The hand-written fixture: 24 generated schedules against the 14-entry bank
# cross-product to exactly 35 fires (see the sample README for how this
# number was derived and how to re-derive it if the fixture bank changes).
SAMPLE_TARGETS = Demo5Targets(
    schedules=24,
    fires=35,
    cards=190,
    empty_per_kind=10,
    min_post_cancel_idle=5,
    min_once_no_repeat=10,
    min_bait_idle=4,
    min_address_positive=5,
    min_silence_idle=12,
)


class RowValidationTests(unittest.TestCase):
    def _rows(self) -> list[dict]:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            build_demo5_artifacts(
                authored_root=SAMPLE_ROOT,
                train_base_path=out / "train.jsonl",
                dev_path=out / "dev.jsonl",
                artifact_dir=out / "artifacts",
                targets=SAMPLE_TARGETS,
                allow_small_corpus=True,
                minimum_train_shards=2,
            )
            rows: list[dict] = []
            for path in sorted(out.glob("train-*.jsonl")):
                rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
            rows.extend(json.loads(line) for line in (out / "dev.jsonl").read_text(encoding="utf-8").splitlines())
            return rows

    def setUp(self) -> None:
        self.rows = self._rows()

    def test_row_validation_accepts_a_real_build(self) -> None:
        summary = validate_demo5_rows(self.rows)
        self.assertEqual(summary["rows"], SAMPLE_TARGETS.cards)
        self.assertEqual(summary["unique_prompts"], SAMPLE_TARGETS.cards)
        self.assertGreater(summary["independently_timing_checked"], 0)

    def test_every_completion_parses_and_is_canonical(self) -> None:
        for row in self.rows:
            parsed = parse_g1_action(row["completion"])
            self.assertTrue(parsed.valid, row["candidate_id"])

    def test_a_mangled_completion_is_rejected(self) -> None:
        rows = [dict(row) for row in self.rows]
        rows[0]["completion"] = "<action>respond({'for': 1})</action>"
        with self.assertRaises(Demo5BuildError):
            validate_demo5_rows(rows)

    def test_fire_respond_must_carry_the_authored_fire_message(self) -> None:
        rows = [dict(row) for row in self.rows]
        target = next(index for index, row in enumerate(rows) if row["candidate_role"] in ("fire-typing", "fire-silent"))
        rows[target]["completion"] = rows[target]["completion"].replace(
            rows[target]["fire_message"], "Something else entirely!"
        )
        with self.assertRaises(Demo5BuildError):
            validate_demo5_rows(rows)

    def test_duplicate_prompts_with_differing_completions_are_rejected(self) -> None:
        rows = [dict(row) for row in self.rows]
        target = next(
            index
            for index, row in enumerate(rows)
            if row["completion"] != rows[0]["completion"]
        )
        rows[target] = dict(rows[target], prompt=rows[0]["prompt"])
        with self.assertRaises(Demo5BuildError) as caught:
            validate_demo5_rows(rows)
        message = str(caught.exception)
        self.assertIn("byte-identical", message)
        self.assertIn("completions differ", message)
        self.assertIn(rows[0]["candidate_id"], message)
        self.assertIn(rows[target]["candidate_id"], message)

    def test_duplicate_prompts_with_identical_completions_are_tolerated(self) -> None:
        rows = [dict(row) for row in self.rows]
        source, target = next(
            (i, j)
            for i, row_i in enumerate(rows)
            for j, row_j in enumerate(rows)
            if i < j and row_i["completion"] == row_j["completion"]
        )
        rows[target] = dict(rows[target], prompt=rows[source]["prompt"])
        summary = validate_demo5_rows(rows)
        self.assertEqual(summary["duplicate_prompt_completion_pairs"], 1)
        self.assertEqual(summary["unique_prompts"], SAMPLE_TARGETS.cards - 1)
        self.assertEqual(summary["rows"], SAMPLE_TARGETS.cards)

    def test_duplicate_prompt_completion_pairs_default_to_zero(self) -> None:
        summary = validate_demo5_rows(self.rows)
        self.assertEqual(summary["duplicate_prompt_completion_pairs"], 0)

    def test_missing_key_is_rejected(self) -> None:
        rows = [dict(row) for row in self.rows]
        rows[0].pop("fire_message")
        with self.assertRaises(Demo5BuildError) as caught:
            validate_demo5_rows(rows)
        self.assertIn("missing required keys", str(caught.exception))

    def test_disagreeing_timing_card_is_rejected(self) -> None:
        rows = [dict(row) for row in self.rows]
        target = next(
            index
            for index, row in enumerate(rows)
            if row["candidate_role"] == "fire-before" and row["schedule_kind"] == "every"
        )
        # A "before" idle card claiming it should have fired instead: the
        # independent prompt-timestamp check must catch this even though the
        # completion is a syntactically valid respond.
        rows[target] = dict(
            rows[target],
            completion=f'<action>respond({{"for":{rows[target]["current_event_index"]},'
            f'"message":"{rows[target]["fire_message"]}"}})</action>',
            expected_class="respond",
        )
        with self.assertRaises(Demo5BuildError) as caught:
            validate_demo5_rows(rows)
        self.assertIn("independent timing check", str(caught.exception))


class BuildArtifactTests(unittest.TestCase):
    def test_sample_build_publishes_manifest_coverage_and_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = build_demo5_artifacts(
                authored_root=SAMPLE_ROOT,
                train_base_path=out / "train.jsonl",
                dev_path=out / "dev.jsonl",
                artifact_dir=out / "artifacts",
                targets=SAMPLE_TARGETS,
                allow_small_corpus=True,
                minimum_train_shards=2,
            )
            self.assertEqual(manifest["demo"], "demo-5")
            self.assertEqual(manifest["row_counts"]["total"], SAMPLE_TARGETS.cards)
            self.assertFalse(manifest["distribution_gates_enforced"])
            self.assertTrue((out / "artifacts" / "manifest.json").exists())
            self.assertTrue((out / "artifacts" / "coverage.json").exists())
            self.assertTrue((out / "artifacts" / "inspection_samples.md").exists())
            shards = sorted(out.glob("train-*.jsonl"))
            self.assertEqual(len(shards), 2)
            self.assertFalse((out / "train.jsonl").exists())

    def test_build_is_byte_for_byte_reproducible(self) -> None:
        payloads: list[dict[str, str]] = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                build_demo5_artifacts(
                    authored_root=SAMPLE_ROOT,
                    train_base_path=out / "train.jsonl",
                    dev_path=out / "dev.jsonl",
                    artifact_dir=out / "artifacts",
                    targets=SAMPLE_TARGETS,
                    allow_small_corpus=True,
                    minimum_train_shards=2,
                )
                payloads.append(
                    {path.name: path.read_text(encoding="utf-8") for path in sorted(out.glob("train-*.jsonl"))}
                )
        self.assertEqual(payloads[0], payloads[1])

    def test_wrong_target_count_fails_with_a_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            bad_targets = Demo5Targets(
                schedules=24,
                fires=999,
                cards=999,
            )
            with self.assertRaises(ValueError) as caught:
                build_demo5_artifacts(
                    authored_root=SAMPLE_ROOT,
                    train_base_path=out / "train.jsonl",
                    dev_path=out / "dev.jsonl",
                    artifact_dir=out / "artifacts",
                    targets=bad_targets,
                    allow_small_corpus=True,
                    minimum_train_shards=2,
                )
            self.assertIn("999 fires", str(caught.exception))

    def test_missing_allow_small_corpus_enforces_distribution_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaises(Demo5BuildError) as caught:
                build_demo5_artifacts(
                    authored_root=SAMPLE_ROOT,
                    train_base_path=out / "train.jsonl",
                    dev_path=out / "dev.jsonl",
                    artifact_dir=out / "artifacts",
                    targets=SAMPLE_TARGETS,
                    allow_small_corpus=False,
                    minimum_train_shards=2,
                )
            self.assertIn("below the", str(caught.exception))


def _inject_warning(*args, **kwargs):
    result = dict(_real_validate_demo5_batches(*args, **kwargs))
    result["warnings"] = [*result["warnings"], "synthetic-test-warning: canary only"]
    return result


class WarningsAreNonFatalTests(unittest.TestCase):
    def test_warnings_are_non_fatal_by_default_and_land_in_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with mock.patch(
                "scripts.g1_demo5_build.validate_demo5_batches",
                side_effect=_inject_warning,
            ):
                manifest = build_demo5_artifacts(
                    authored_root=SAMPLE_ROOT,
                    train_base_path=out / "train.jsonl",
                    dev_path=out / "dev.jsonl",
                    artifact_dir=out / "artifacts",
                    targets=SAMPLE_TARGETS,
                    allow_small_corpus=True,
                    minimum_train_shards=2,
                )
            self.assertIn(
                "synthetic-test-warning: canary only", manifest["source_warnings"]
            )

    def test_fail_on_warnings_turns_the_same_warning_into_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with mock.patch(
                "scripts.g1_demo5_build.validate_demo5_batches",
                side_effect=_inject_warning,
            ):
                with self.assertRaises(Demo5BuildError) as caught:
                    build_demo5_artifacts(
                        authored_root=SAMPLE_ROOT,
                        train_base_path=out / "train.jsonl",
                        dev_path=out / "dev.jsonl",
                        artifact_dir=out / "artifacts",
                        targets=SAMPLE_TARGETS,
                        allow_small_corpus=True,
                        minimum_train_shards=2,
                        fail_on_warnings=True,
                    )
            self.assertIn("synthetic-test-warning", str(caught.exception))
            self.assertIn("--fail-on-warnings", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
