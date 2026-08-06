from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.stream import parse_g1_action
from datagen.g1_authored_demo4 import validate_demo4_batches as _real_validate_demo4_batches
from datagen.g1_demo4 import Demo4Targets
from scripts.g1_demo4_build import (
    Demo4BuildError,
    _stage_and_publish,
    build_demo4_artifacts,
    validate_demo4_rows,
)
from tests.test_g1_authored_demo4 import _batch, _check, _nudge, _request


SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "data" / "g1_demo4_sample"

# The hand-written fixture: 6 requests, 3 check + 3 nudge progress pairs,
# expanded to 15 episodes. See data/g1_demo4_sample/README.md for the by-hand
# arithmetic that reaches exactly 128 cards from this fixture.
SAMPLE_TARGETS = Demo4Targets(
    requests=6,
    progress_pairs=6,
    episodes=15,
    cards=128,
    empty_per_kind=10,
    min_check_positive=5,
    min_nudge_idle=5,
    min_narration_idle=5,
    min_failure_check=3,
)


def _write_wide_source(root: Path, per_author: int = 20) -> None:
    personas = [
        "journal-keeper",
        "meeting-note-taker",
        "group-chat-drafter",
        "seed-diary-writer",
        "student-planner",
        "family-organizer",
    ]
    domains = ["cooking", "work", "travel", "gardening", "school", "family"]
    registers = ["casual", "brisk", "warm", "plain"]
    batches = []
    counter = 0
    check_counter = 0
    nudge_counter = 0
    for author_index, slot in enumerate(("slot-a", "slot-b", "slot-c")):
        requests = []
        progress = []
        for _ in range(per_author):
            persona = personas[counter % len(personas)]
            domain = domains[counter % len(domains)]
            register = registers[counter % len(registers)]
            requests.append(_request(counter, persona=persona, domain=domain, register=register))
            if counter % 2 == 0:
                progress.append(_check(check_counter, persona=persona, domain=domain, register=register))
                check_counter += 1
            else:
                progress.append(_nudge(nudge_counter, persona=persona, domain=domain, register=register))
                nudge_counter += 1
            counter += 1
        batches.append(_batch(slot, requests, progress, tranche=author_index))
    source = root / "demo4"
    source.mkdir(parents=True)
    for index, batch in enumerate(batches):
        (source / f"batch_{index}.json").write_text(
            json.dumps(batch, ensure_ascii=False), encoding="utf-8"
        )


class RowValidationTests(unittest.TestCase):
    def _rows(self) -> list[dict]:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            build_demo4_artifacts(
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
        summary = validate_demo4_rows(self.rows)
        self.assertEqual(summary["rows"], SAMPLE_TARGETS.cards)

    def test_every_completion_parses_and_is_canonical(self) -> None:
        for row in self.rows:
            parsed = parse_g1_action(row["completion"])
            self.assertTrue(parsed.valid, row["candidate_id"])

    def test_a_mangled_completion_is_rejected(self) -> None:
        rows = [dict(row) for row in self.rows]
        rows[0]["completion"] = "<action>respond({'for': 1})</action>"
        with self.assertRaises(Demo4BuildError):
            validate_demo4_rows(rows)

    def test_delegate_task_drift_is_rejected(self) -> None:
        rows = [dict(row) for row in self.rows]
        target = next(
            index for index, row in enumerate(rows) if row["candidate_role"] == "request-positive"
        )
        rows[target] = dict(rows[target])
        rows[target]["request_task"] = "a completely different task"
        with self.assertRaises(Demo4BuildError) as caught:
            validate_demo4_rows(rows)
        self.assertIn("drifted", str(caught.exception))

    def test_duplicate_candidate_id_is_rejected(self) -> None:
        rows = [dict(row) for row in self.rows]
        rows[1] = dict(rows[1], candidate_id=rows[0]["candidate_id"])
        with self.assertRaises(Demo4BuildError) as caught:
            validate_demo4_rows(rows)
        self.assertIn("duplicate candidate_id", str(caught.exception))

    def test_missing_key_is_rejected(self) -> None:
        rows = [dict(row) for row in self.rows]
        del rows[0]["request_task"]
        with self.assertRaises(Demo4BuildError) as caught:
            validate_demo4_rows(rows)
        self.assertIn("missing required keys", str(caught.exception))

    def test_tool_ticks_carry_a_job_id_and_others_do_not(self) -> None:
        for row in self.rows:
            if row["candidate_role"] in {"accepted-idle", "completed-idle", "failed-idle"}:
                self.assertIsNotNone(row["job_id"])
            else:
                self.assertIsNone(row["job_id"])

    def test_all_five_traps_are_represented(self) -> None:
        roles = {row["candidate_role"] for row in self.rows}
        self.assertIn("nudge-idle", roles)  # trap 1
        self.assertIn("check-positive", roles)  # trap 2
        self.assertIn("completed-idle", roles)  # trap 3 (structural)
        self.assertIn("failed-idle", roles)  # trap 3, failure branch
        self.assertIn("narration-idle", roles)  # trap 4
        self.assertIn("failure-check-positive", roles)  # trap 5


class BuildArtifactTests(unittest.TestCase):
    def test_sample_build_publishes_manifest_coverage_and_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = build_demo4_artifacts(
                authored_root=SAMPLE_ROOT,
                train_base_path=out / "train.jsonl",
                dev_path=out / "dev.jsonl",
                artifact_dir=out / "artifacts",
                targets=SAMPLE_TARGETS,
                allow_small_corpus=True,
                minimum_train_shards=2,
            )
            self.assertEqual(manifest["demo"], "demo-4")
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
                build_demo4_artifacts(
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
            with self.assertRaises(Demo4BuildError) as caught:
                build_demo4_artifacts(
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
            with self.assertRaises(Demo4BuildError) as caught:
                build_demo4_artifacts(
                    authored_root=Path(directory),
                    train_base_path=Path(directory) / "train.jsonl",
                    dev_path=Path(directory) / "dev.jsonl",
                    artifact_dir=Path(directory) / "artifacts",
                    targets=SAMPLE_TARGETS,
                    allow_small_corpus=True,
                )
            self.assertIn("No Demo 4 authored JSON files", str(caught.exception))

    def test_gated_build_runs_end_to_end_on_a_full_size_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            out = Path(directory) / "out"
            _write_wide_source(root, per_author=20)
            targets = Demo4Targets(
                requests=60,
                progress_pairs=60,
                episodes=90,
                cards=560,
                empty_per_kind=25,
                min_check_positive=20,
                min_nudge_idle=20,
                min_narration_idle=20,
                min_failure_check=12,
            )
            manifest = build_demo4_artifacts(
                authored_root=root,
                train_base_path=out / "train.jsonl",
                dev_path=out / "dev.jsonl",
                artifact_dir=out / "artifacts",
                targets=targets,
                minimum_train_shards=2,
            )
            self.assertTrue(manifest["distribution_gates_enforced"])
            self.assertEqual(manifest["row_counts"]["total"], 560)
            evidence = manifest["exact_target_evidence"]
            self.assertTrue(all(item["exact"] for item in evidence.values()), evidence)
            self.assertGreater(manifest["row_counts"]["dev"], 0)

    def test_source_distribution_and_skeletons_reach_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = build_demo4_artifacts(
                authored_root=SAMPLE_ROOT,
                train_base_path=out / "train.jsonl",
                dev_path=out / "dev.jsonl",
                artifact_dir=out / "artifacts",
                targets=SAMPLE_TARGETS,
                allow_small_corpus=True,
                minimum_train_shards=2,
            )
            coverage = json.loads(
                (out / "artifacts" / "coverage.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(coverage["opening_shapes"]), 4)
            self.assertEqual(len(coverage["closing_shapes"]), 4)
            self.assertGreaterEqual(coverage["typing_delta_chars"]["min"], 4)
            self.assertLessEqual(coverage["typing_delta_chars"]["max"], 7)
            self.assertEqual(set(coverage["selected_outcomes"]), {"success", "failure"})
            self.assertEqual(
                set(coverage["selected_content_kinds"]), {"check", "nudge", "narration"}
            )
            self.assertIn(manifest["demo"], "demo-4")


def _inject_warning(*args, **kwargs):
    result = dict(_real_validate_demo4_batches(*args, **kwargs))
    result["warnings"] = [*result["warnings"], "synthetic-test-warning: canary only"]
    return result


class WarningsAreNonFatalTests(unittest.TestCase):
    def test_warnings_are_non_fatal_by_default_and_land_in_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with mock.patch(
                "scripts.g1_demo4_build.validate_demo4_batches",
                side_effect=_inject_warning,
            ):
                manifest = build_demo4_artifacts(
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
                "scripts.g1_demo4_build.validate_demo4_batches",
                side_effect=_inject_warning,
            ):
                with self.assertRaises(Demo4BuildError) as caught:
                    build_demo4_artifacts(
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
            with self.assertRaises(Demo4BuildError):
                _stage_and_publish({path: "a"}, obsolete_paths=(path,))


if __name__ == "__main__":
    unittest.main()
