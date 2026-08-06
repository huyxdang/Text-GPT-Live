from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.stream import parse_g1_action
from datagen.g1_authored_demo3 import validate_demo3_batches as _real_validate_demo3_batches
from datagen.g1_demo3 import Demo3Targets
from scripts.g1_demo3_build import (
    CHINESE_REVIEW_DISCLAIMER,
    Demo3BuildError,
    _stage_and_publish,
    build_demo3_artifacts,
    validate_demo3_rows,
)
from tests.test_g1_authored_demo3 import _batch, _record


SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "data" / "g1_authored_demo3_sample"

# The hand-written fixture: 4 episodes, 12 clause commits, 7 trap idles, and one
# card of each empty kind. 12 + 4 + 7 + 3 = 26 mandatory, 29 neighbours = 55.
SAMPLE_TARGETS = Demo3Targets(
    episodes=4,
    clauses=12,
    cards=55,
    empty_per_kind=1,
    min_backtrack_idles=1,
    min_prequoted_idles=1,
    min_partial_idles=1,
)


def _write_wide_source(root: Path, count: int = 30) -> None:
    records = [_record(index) for index in range(count)]
    per_slot = count // 3
    batches = [
        _batch("slot-a", records[:per_slot]),
        _batch("slot-b", records[per_slot : 2 * per_slot]),
        _batch("slot-c", records[2 * per_slot :]),
    ]
    source = root / "demo3"
    source.mkdir(parents=True)
    for index, batch in enumerate(batches):
        (source / f"batch_{index}.json").write_text(
            json.dumps(batch, ensure_ascii=False), encoding="utf-8"
        )


class RowValidationTests(unittest.TestCase):
    def _rows(self) -> list[dict]:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            build_demo3_artifacts(
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
                rows.extend(
                    json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                )
            rows.extend(
                json.loads(line)
                for line in (out / "dev.jsonl").read_text(encoding="utf-8").splitlines()
            )
            return rows

    def setUp(self) -> None:
        self.rows = self._rows()

    def test_row_validation_accepts_a_real_build(self) -> None:
        summary = validate_demo3_rows(self.rows)
        self.assertEqual(summary["rows"], SAMPLE_TARGETS.cards)
        self.assertEqual(summary["unique_prompts"], SAMPLE_TARGETS.cards)
        self.assertGreater(summary["non_ascii_responds"], 0)

    def test_every_completion_parses_and_is_canonical(self) -> None:
        for row in self.rows:
            parsed = parse_g1_action(row["completion"])
            self.assertTrue(parsed.valid, row["candidate_id"])

    def test_a_mangled_completion_is_rejected(self) -> None:
        rows = [dict(row) for row in self.rows]
        rows[0]["completion"] = "<action>respond({'for': 1})</action>"
        with self.assertRaises(Demo3BuildError):
            validate_demo3_rows(rows)

    def test_escaped_chinese_completion_is_rejected_as_non_canonical(self) -> None:
        rows = [dict(row) for row in self.rows]
        target = next(
            index for index, row in enumerate(rows) if row["candidate_role"] == "clause-positive"
        )
        rows[target]["completion"] = (
            rows[target]["completion"].encode("unicode_escape").decode("ascii")
        )
        with self.assertRaises(Demo3BuildError):
            validate_demo3_rows(rows)

    def test_clause_respond_must_carry_the_authored_reference(self) -> None:
        rows = [dict(row) for row in self.rows]
        target = next(
            index for index, row in enumerate(rows) if row["candidate_role"] == "clause-positive"
        )
        rows[target]["clause_reference"] = "另一句完全不同的中文。"
        with self.assertRaises(Demo3BuildError) as caught:
            validate_demo3_rows(rows)
        self.assertIn("authored", str(caught.exception))

    def test_duplicate_prompts_with_differing_completions_are_rejected(self) -> None:
        rows = [dict(row) for row in self.rows]
        target = next(
            index
            for index, row in enumerate(rows)
            if row["completion"] != rows[0]["completion"]
        )
        rows[target] = dict(rows[target], prompt=rows[0]["prompt"])
        with self.assertRaises(Demo3BuildError) as caught:
            validate_demo3_rows(rows)
        message = str(caught.exception)
        self.assertIn("byte-identical", message)
        self.assertIn("completions differ", message)
        self.assertIn(rows[0]["candidate_id"], message)
        self.assertIn(rows[target]["candidate_id"], message)

    def test_duplicate_prompts_with_identical_completions_are_tolerated(self) -> None:
        rows = [dict(row) for row in self.rows]
        target = next(
            index
            for index, row in enumerate(rows)
            if index != 0 and row["completion"] == rows[0]["completion"]
        )
        rows[target] = dict(rows[target], prompt=rows[0]["prompt"])
        summary = validate_demo3_rows(rows)
        self.assertEqual(summary["duplicate_prompt_completion_pairs"], 1)
        self.assertEqual(summary["unique_prompts"], SAMPLE_TARGETS.cards - 1)
        self.assertEqual(summary["rows"], SAMPLE_TARGETS.cards)

    def test_duplicate_prompt_completion_pairs_default_to_zero(self) -> None:
        summary = validate_demo3_rows(self.rows)
        self.assertEqual(summary["duplicate_prompt_completion_pairs"], 0)

    def test_human_review_claim_on_a_row_is_rejected(self) -> None:
        rows = [dict(row) for row in self.rows]
        rows[0]["reference_review_method"] = "human"
        with self.assertRaises(Demo3BuildError):
            validate_demo3_rows(rows)

    def test_missing_key_is_rejected(self) -> None:
        rows = [dict(row) for row in self.rows]
        rows[0].pop("clause_reference")
        with self.assertRaises(Demo3BuildError) as caught:
            validate_demo3_rows(rows)
        self.assertIn("missing required keys", str(caught.exception))


class BuildArtifactTests(unittest.TestCase):
    def test_sample_build_publishes_manifest_coverage_and_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = build_demo3_artifacts(
                authored_root=SAMPLE_ROOT,
                train_base_path=out / "train.jsonl",
                dev_path=out / "dev.jsonl",
                artifact_dir=out / "artifacts",
                targets=SAMPLE_TARGETS,
                allow_small_corpus=True,
                minimum_train_shards=2,
            )
            self.assertEqual(manifest["demo"], "demo-3")
            self.assertEqual(manifest["row_counts"]["total"], SAMPLE_TARGETS.cards)
            self.assertFalse(manifest["distribution_gates_enforced"])
            self.assertTrue((out / "artifacts" / "manifest.json").exists())
            self.assertTrue((out / "artifacts" / "coverage.json").exists())
            self.assertTrue((out / "artifacts" / "inspection_samples.md").exists())
            shards = sorted(out.glob("train-*.jsonl"))
            self.assertEqual(len(shards), 2)
            self.assertFalse((out / "train.jsonl").exists())

    def test_manifest_records_machine_review_without_implying_human_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = build_demo3_artifacts(
                authored_root=SAMPLE_ROOT,
                train_base_path=out / "train.jsonl",
                dev_path=out / "dev.jsonl",
                artifact_dir=out / "artifacts",
                targets=SAMPLE_TARGETS,
                allow_small_corpus=True,
                minimum_train_shards=2,
            )
            review = manifest["chinese_reference_review"]
            self.assertIs(review["human_verified"], False)
            self.assertTrue(review["machine_reviewed"])
            self.assertEqual(list(review["methods"]), ["machine"])
            self.assertEqual(review["disclaimer"], CHINESE_REVIEW_DISCLAIMER)
            self.assertIn("not by a human reader", review["disclaimer"])
            self.assertEqual(
                review["targeted_semantic_review"]["references_corrected"], 11
            )
            self.assertFalse(
                review["targeted_semantic_review"]["human_certification"]
            )
            self.assertTrue(review["deterministic_checks"])
            samples = (out / "artifacts" / "inspection_samples.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("machine (LLM) review only", samples)

    def test_build_is_byte_for_byte_reproducible(self) -> None:
        payloads: list[dict[str, str]] = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                build_demo3_artifacts(
                    authored_root=SAMPLE_ROOT,
                    train_base_path=out / "train.jsonl",
                    dev_path=out / "dev.jsonl",
                    artifact_dir=out / "artifacts",
                    targets=SAMPLE_TARGETS,
                    allow_small_corpus=True,
                    minimum_train_shards=2,
                )
                payloads.append(
                    {
                        path.name: path.read_text(encoding="utf-8")
                        for path in sorted(out.glob("train-*.jsonl"))
                    }
                )
        self.assertEqual(payloads[0], payloads[1])

    def test_small_corpus_without_the_flag_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaises(Demo3BuildError) as caught:
                build_demo3_artifacts(
                    authored_root=SAMPLE_ROOT,
                    train_base_path=out / "train.jsonl",
                    dev_path=out / "dev.jsonl",
                    artifact_dir=out / "artifacts",
                    targets=SAMPLE_TARGETS,
                    minimum_train_shards=2,
                )
            self.assertIn("distribution", str(caught.exception))

    def test_missing_source_is_diagnosed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(Demo3BuildError) as caught:
                build_demo3_artifacts(
                    authored_root=Path(directory),
                    train_base_path=Path(directory) / "train.jsonl",
                    dev_path=Path(directory) / "dev.jsonl",
                    artifact_dir=Path(directory) / "artifacts",
                    targets=SAMPLE_TARGETS,
                    allow_small_corpus=True,
                )
            self.assertIn("No Demo 3 authored JSON files", str(caught.exception))

    def test_gated_build_runs_end_to_end_on_a_full_size_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            out = Path(directory) / "out"
            _write_wide_source(root)
            manifest = build_demo3_artifacts(
                authored_root=root,
                train_base_path=out / "train.jsonl",
                dev_path=out / "dev.jsonl",
                artifact_dir=out / "artifacts",
                targets=Demo3Targets(
                    episodes=30,
                    clauses=150,
                    cards=400,
                    empty_per_kind=5,
                    min_backtrack_idles=5,
                    min_prequoted_idles=5,
                    min_partial_idles=5,
                ),
                minimum_train_shards=2,
            )
            self.assertTrue(manifest["distribution_gates_enforced"])
            self.assertEqual(manifest["row_counts"]["total"], 400)
            self.assertEqual(manifest["row_validation"]["unique_prompts"], 400)
            self.assertEqual(manifest["row_validation"]["duplicate_prompt_completion_pairs"], 0)
            self.assertNotIn("unique_prompts", manifest["exact_target_evidence"])
            evidence = manifest["exact_target_evidence"]
            self.assertTrue(all(item["exact"] for item in evidence.values()))
            self.assertEqual(evidence["selected_clause_positive"]["actual"], 150)
            self.assertEqual(evidence["selected_instruction_ack"]["actual"], 30)
            self.assertGreater(manifest["row_counts"]["dev"], 0)

    def test_source_distribution_and_skeletons_reach_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            out = Path(directory) / "out"
            _write_wide_source(root)
            build_demo3_artifacts(
                authored_root=root,
                train_base_path=out / "train.jsonl",
                dev_path=out / "dev.jsonl",
                artifact_dir=out / "artifacts",
                targets=Demo3Targets(
                    episodes=30,
                    clauses=150,
                    cards=400,
                    empty_per_kind=5,
                    min_backtrack_idles=5,
                    min_prequoted_idles=5,
                    min_partial_idles=5,
                ),
                minimum_train_shards=2,
            )
            coverage = json.loads(
                (out / "artifacts" / "coverage.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(coverage["opening_shapes"]), 4)
            self.assertEqual(len(coverage["closing_shapes"]), 4)
            self.assertGreaterEqual(coverage["typing_delta_chars"]["min"], 4)
            self.assertLessEqual(coverage["typing_delta_chars"]["max"], 7)
            self.assertGreaterEqual(coverage["tick_gap_ms"]["max"], 10_000)
            self.assertEqual(
                set(coverage["selected_traps"]),
                {"backtrack", "half_clause", "prequoted_chinese"},
            )


def _inject_warning(*args, **kwargs):
    result = dict(_real_validate_demo3_batches(*args, **kwargs))
    result["warnings"] = [*result["warnings"], "synthetic-test-warning: canary only"]
    return result


class WarningsAreNonFatalTests(unittest.TestCase):
    def test_warnings_are_non_fatal_by_default_and_land_in_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with mock.patch(
                "scripts.g1_demo3_build.validate_demo3_batches",
                side_effect=_inject_warning,
            ):
                manifest = build_demo3_artifacts(
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
                "scripts.g1_demo3_build.validate_demo3_batches",
                side_effect=_inject_warning,
            ):
                with self.assertRaises(Demo3BuildError) as caught:
                    build_demo3_artifacts(
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

    def test_errors_still_fail_even_with_fail_on_warnings_unset(self) -> None:
        # test_small_corpus_without_the_flag_fails already covers the plain
        # error path; this asserts errors stay fatal when warnings are also
        # present and non-fatal-by-default is in effect.
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaises(Demo3BuildError):
                build_demo3_artifacts(
                    authored_root=SAMPLE_ROOT,
                    train_base_path=out / "train.jsonl",
                    dev_path=out / "dev.jsonl",
                    artifact_dir=out / "artifacts",
                    targets=SAMPLE_TARGETS,
                    minimum_train_shards=2,
                )


class PublicationTests(unittest.TestCase):
    def test_failed_publication_rolls_every_file_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kept = root / "kept.txt"
            kept.write_text("original", encoding="utf-8")
            blocked = root / "blocked"
            blocked.mkdir()
            with self.assertRaises(OSError):
                _stage_and_publish({kept: "replaced", blocked: "cannot write"})
            self.assertEqual(kept.read_text(encoding="utf-8"), "original")

    def test_obsolete_and_destination_overlap_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.jsonl"
            with self.assertRaises(Demo3BuildError):
                _stage_and_publish({path: "a"}, obsolete_paths=(path,))


if __name__ == "__main__":
    unittest.main()
