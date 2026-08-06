from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datagen.g1_demo1 import Demo1Targets
from scripts.g1_demo1_build import (
    Demo1BuildError,
    _stage_and_publish,
    build_demo1_artifacts,
    validate_demo1_rows,
)


def _record(
    index: int,
    *,
    segment_count: int,
    address_index: int,
    text_characters: int,
) -> dict:
    segments: list[dict] = []
    first_narration = True
    for segment_index in range(segment_count):
        pause_after = ("none", "short", "medium", "long")[
            segment_index % 4
        ]
        if segment_index == address_index:
            segments.append(
                {
                    "kind": "address",
                    "text": f"What follows from fixture {index} at step {segment_index}?",
                    "gold_reply": f"Fixture {index} keeps its visible sequence intact.",
                    "pause_after": pause_after,
                }
            )
            continue
        if first_narration:
            text = (
                f"Why would fixture {index} lose its visible evidence? "
                f"Segment {segment_index} leaves the sequence visible."
            )
            traps = ["rhetorical_question"]
            first_narration = False
        else:
            text = (
                f"Fixture {index} segment {segment_index} adds a distinct practical "
                "detail to the continuing scene."
            )
            traps = []
        segments.append(
            {
                "kind": "narration",
                "text": text,
                "traps": traps,
                "pause_after": pause_after,
            }
        )
    current = sum(len(segment["text"]) for segment in segments)
    padding = text_characters - current
    if padding < 2:
        raise AssertionError("fixture target must leave room for deterministic padding")
    narration = next(segment for segment in segments if segment["kind"] == "narration")
    narration["text"] += " " + (chr(ord("a") + index) * (padding - 1))
    return {
        "id": f"fixture-{index:02d}",
        "persona": f"fixture-persona-{index}",
        "domain": f"fixture-domain-{index}",
        "register": f"fixture-register-{index}",
        "segments": segments,
    }


def _write_source(root: Path, *, warning: bool = False) -> None:
    records = [
        _record(0, segment_count=5, address_index=0, text_characters=420),
        _record(1, segment_count=6, address_index=0, text_characters=460),
        _record(2, segment_count=7, address_index=3, text_characters=650),
        _record(3, segment_count=7, address_index=3, text_characters=760),
        _record(4, segment_count=8, address_index=7, text_characters=1_000),
        _record(5, segment_count=5, address_index=4, text_characters=1_100),
    ]
    if warning:
        narration = next(
            segment
            for segment in records[0]["segments"]
            if segment["kind"] == "narration"
        )
        narration["text"] += " The guide explained the sequence."
    source_dir = root / "demo1"
    source_dir.mkdir(parents=True)
    for author_index in range(3):
        batch = {
            "schema_version": "g1-authored-1",
            "demo": "demo-1",
            "author": {
                "model": "fixture-model",
                "slot": f"fixture-author-{author_index}",
                "tranche": 1,
            },
            "records": records[author_index * 2 : author_index * 2 + 2],
        }
        (source_dir / f"fixture_{author_index}.json").write_text(
            json.dumps(batch), encoding="utf-8"
        )


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class Demo1FullBuildTests(unittest.TestCase):
    def test_small_strict_build_is_deterministic_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authored_root = root / "authored"
            train_base_path = root / "data" / "train.jsonl"
            dev_path = root / "data" / "dev.jsonl"
            artifact_dir = root / "artifacts"
            _write_source(authored_root)
            train_base_path.parent.mkdir(parents=True)
            train_base_path.write_text("obsolete single-file build\n", encoding="utf-8")
            stale_shards = (
                train_base_path.with_name("train-00001-of-00002.jsonl"),
                train_base_path.with_name("train-00002-of-00002.jsonl"),
            )
            for stale_shard in stale_shards:
                stale_shard.write_text("obsolete shard\n", encoding="utf-8")
            targets = Demo1Targets(
                addresses=6,
                narrations=32,
                cards=30,
                empty_per_kind=1,
            )

            first = build_demo1_artifacts(
                authored_root=authored_root,
                train_base_path=train_base_path,
                dev_path=dev_path,
                artifact_dir=artifact_dir,
                targets=targets,
                dev_fraction=1 / 3,
            )
            train_shard_paths = tuple(
                Path(entry["path"])
                for entry in first["files"]["train"]["shards"]
            )
            output_paths = (
                *train_shard_paths,
                dev_path,
                artifact_dir / "manifest.json",
                artifact_dir / "coverage.json",
                artifact_dir / "inspection_samples.md",
            )
            first_bytes = {path: path.read_bytes() for path in output_paths}
            second = build_demo1_artifacts(
                authored_root=authored_root,
                train_base_path=train_base_path,
                dev_path=dev_path,
                artifact_dir=artifact_dir,
                targets=targets,
                dev_fraction=1 / 3,
            )

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, {path: path.read_bytes() for path in output_paths})
            self.assertFalse(train_base_path.exists())
            self.assertTrue(all(not path.exists() for path in stale_shards))
            self.assertEqual(len(train_shard_paths), 3)
            train_rows = [
                row for path in train_shard_paths for row in _load_jsonl(path)
            ]
            dev_rows = _load_jsonl(dev_path)
            rows = train_rows + dev_rows
            self.assertEqual(len(rows), 30)
            self.assertTrue(all(row["schema_version"] == "g1" for row in rows))
            self.assertTrue(all(row["split"] == "train" for row in train_rows))
            self.assertTrue(all(row["split"] == "dev" for row in dev_rows))
            self.assertEqual(first["row_counts"]["total"], 30)
            self.assertEqual(first["source"]["counts"]["batches"], 3)
            self.assertEqual(first["source"]["counts"]["narration"], 32)
            aggregate = hashlib.sha256()
            for path, entry in zip(
                train_shard_paths,
                first["files"]["train"]["shards"],
                strict=True,
            ):
                payload = path.read_bytes()
                aggregate.update(payload)
                self.assertEqual(entry["bytes"], len(payload))
                self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertLessEqual(
                    entry["bytes"], first["files"]["train"]["max_shard_bytes"]
                )
            self.assertEqual(
                first["files"]["train"]["aggregate_sha256"],
                aggregate.hexdigest(),
            )
            self.assertEqual(
                first["files"]["dev"]["sha256"],
                hashlib.sha256(dev_path.read_bytes()).hexdigest(),
            )
            self.assertTrue(
                all(
                    result["exact"]
                    for result in first["exact_target_evidence"].values()
                )
            )
            coverage = json.loads(
                (artifact_dir / "coverage.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                coverage["selected_empty_kinds"],
                {"cleared": 1, "initial": 1, "unchanged": 1},
            )
            inspection = (artifact_dir / "inspection_samples.md").read_text(
                encoding="utf-8"
            )
            for role in coverage["selected_roles"]:
                self.assertIn(f"### role: {role}", inspection)
            self.assertIn("persona dominant", inspection)
            self.assertIn("persona sparse", inspection)
            self.assertIn("longest selected prompt", inspection)

    def test_scaled_build_subsamples_exactly_and_byte_identically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authored_root = root / "authored"
            train_base_path = root / "data" / "train.jsonl"
            dev_path = root / "data" / "dev.jsonl"
            artifact_dir = root / "artifacts"
            _write_source(authored_root)
            # Six source records, 32 narrations, 6 addresses.  The scaled slice
            # keeps 4 of the 6 address sites and drops most of the ballast.
            targets = Demo1Targets(
                addresses=6,
                narrations=32,
                cards=24,
                empty_per_kind=1,
                selected_addresses=4,
                hard_idles=3,
                strategy="stratified",
            )

            first = build_demo1_artifacts(
                authored_root=authored_root,
                train_base_path=train_base_path,
                dev_path=dev_path,
                artifact_dir=artifact_dir,
                targets=targets,
                dev_fraction=1 / 3,
            )
            train_shard_paths = tuple(
                Path(entry["path"]) for entry in first["files"]["train"]["shards"]
            )
            output_paths = (
                *train_shard_paths,
                dev_path,
                artifact_dir / "manifest.json",
                artifact_dir / "coverage.json",
            )
            first_bytes = {path: path.read_bytes() for path in output_paths}
            second = build_demo1_artifacts(
                authored_root=authored_root,
                train_base_path=train_base_path,
                dev_path=dev_path,
                artifact_dir=artifact_dir,
                targets=targets,
                dev_fraction=1 / 3,
            )

            self.assertEqual(first, second)
            self.assertEqual(
                first_bytes, {path: path.read_bytes() for path in output_paths}
            )
            self.assertEqual(first["row_counts"]["total"], 24)
            self.assertEqual(
                first["selected"]["roles"],
                {
                    "address-after": 4,
                    "address-before": 4,
                    "address-positive": 4,
                    "empty-cleared": 1,
                    "empty-initial": 1,
                    "empty-unchanged": 1,
                    "narration-ballast": 6,
                    "narration-hard-idle": 3,
                },
            )
            self.assertTrue(
                all(
                    result["exact"]
                    for result in first["exact_target_evidence"].values()
                )
            )
            self.assertEqual(first["exact_target_evidence"]["selected_hard_idles"],
                             {"target": 3, "actual": 3, "exact": True})
            self.assertEqual(first["exact_target_evidence"]["selected_ballast"],
                             {"target": 6, "actual": 6, "exact": True})
            # Source expectations stay at the full authored corpus.
            self.assertEqual(first["source"]["counts"]["records"], 6)
            self.assertEqual(first["source"]["counts"]["addresses"], 6)

            rows = [
                row for path in train_shard_paths for row in _load_jsonl(path)
            ] + _load_jsonl(dev_path)
            sites: dict[tuple[str, int], set[str]] = {}
            for row in rows:
                if str(row["candidate_role"]).startswith("address-"):
                    key = (row["episode"], row["segment_index"])
                    sites.setdefault(key, set()).add(row["candidate_role"])
            self.assertEqual(len(sites), 4)
            for key, roles in sites.items():
                self.assertEqual(
                    roles,
                    {"address-before", "address-positive", "address-after"},
                    f"incoherent address site {key}",
                )
            self.assertTrue(
                all(row["prompt"].endswith("<PREDICT_THIS_ACTION>") for row in rows)
            )

    def test_selection_distribution_gate_can_block_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authored_root = root / "authored"
            train_base_path = root / "data" / "train.jsonl"
            dev_path = root / "data" / "dev.jsonl"
            artifact_dir = root / "artifacts"
            _write_source(authored_root)
            # One address site cannot represent three authors, so the enforced
            # selected-card distribution gate must refuse to publish.
            targets = Demo1Targets(
                addresses=6,
                narrations=32,
                cards=6,
                empty_per_kind=1,
                selected_addresses=1,
                hard_idles=0,
                strategy="stratified",
            )

            with self.assertRaisesRegex(
                Demo1BuildError, "selected-card distribution failed"
            ):
                build_demo1_artifacts(
                    authored_root=authored_root,
                    train_base_path=train_base_path,
                    dev_path=dev_path,
                    artifact_dir=artifact_dir,
                    targets=targets,
                    dev_fraction=1 / 3,
                    enforce_selection_distribution=True,
                )
            self.assertFalse((artifact_dir / "manifest.json").exists())
            self.assertFalse(dev_path.exists())

    def test_warnings_are_non_fatal_by_default_and_land_in_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authored_root = root / "authored"
            train_base_path = root / "data" / "train.jsonl"
            dev_path = root / "data" / "dev.jsonl"
            artifact_dir = root / "artifacts"
            _write_source(authored_root, warning=True)

            manifest = build_demo1_artifacts(
                authored_root=authored_root,
                train_base_path=train_base_path,
                dev_path=dev_path,
                artifact_dir=artifact_dir,
                targets=Demo1Targets(
                    addresses=6,
                    narrations=32,
                    cards=30,
                    empty_per_kind=1,
                ),
            )

            self.assertTrue(manifest["source_warnings"])
            self.assertTrue((artifact_dir / "manifest.json").exists())
            self.assertTrue(dev_path.exists())

    def test_fail_on_warnings_makes_the_same_warning_release_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authored_root = root / "authored"
            train_base_path = root / "data" / "train.jsonl"
            dev_path = root / "data" / "dev.jsonl"
            artifact_dir = root / "artifacts"
            _write_source(authored_root, warning=True)

            with self.assertRaisesRegex(
                Demo1BuildError, "warnings are release-blocking"
            ):
                build_demo1_artifacts(
                    authored_root=authored_root,
                    train_base_path=train_base_path,
                    dev_path=dev_path,
                    artifact_dir=artifact_dir,
                    targets=Demo1Targets(
                        addresses=6,
                        narrations=32,
                        cards=30,
                        empty_per_kind=1,
                    ),
                    fail_on_warnings=True,
                )

            self.assertFalse(train_base_path.exists())
            self.assertFalse(list((root / "data").glob("train-*-of-*.jsonl")))
            self.assertFalse(dev_path.exists())
            self.assertFalse((artifact_dir / "manifest.json").exists())
            self.assertFalse((artifact_dir / "coverage.json").exists())
            self.assertFalse((artifact_dir / "inspection_samples.md").exists())

    def test_row_validator_rejects_non_g1_schema_and_invalid_completion(self) -> None:
        row = {
            "schema_version": "g1",
            "split": "train",
            "episode": "fixture",
            "demo": "demo-1",
            "situation": "empty-initial",
            "bucket": "empty-initial",
            "prompt": "<stream_event index=\"1\"></stream_event>\n<PREDICT_THIS_ACTION>",
            "completion": "<action>idle()</action>",
            "expected_class": "idle",
            "current_event_index": 1,
            "current_content_empty": True,
            "candidate_id": "fixture:empty-initial:initial",
            "candidate_role": "empty-initial",
            "source_record_id": "fixture",
            "source_persona": "fixture-persona",
            "source_domain": "fixture-domain",
            "source_register": "fixture-register",
            "source_author_slot": "fixture-author",
            "source_author_model": "fixture-model",
            "source_author_tranche": "1",
            "segment_index": None,
            "traps": [],
            "pause_after": None,
            "narration_idle_kind": None,
            "empty_kind": "initial",
            "obligation": "none",
        }
        self.assertEqual(validate_demo1_rows([row])["valid_g1_completions"], 1)
        bad_schema = dict(row, schema_version="g0")
        with self.assertRaisesRegex(Demo1BuildError, "schema_version"):
            validate_demo1_rows([bad_schema])
        bad_completion = dict(row, completion="idle()")
        with self.assertRaisesRegex(Demo1BuildError, "invalid g1 completion"):
            validate_demo1_rows([bad_completion])

    def test_publish_rolls_back_replacements_after_mid_commit_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.txt"
            second = root / "second.txt"
            obsolete = root / "legacy.txt"
            first.write_text("old first", encoding="utf-8")
            obsolete.write_text("old legacy", encoding="utf-8")
            real_replace = __import__("os").replace
            calls = 0

            def fail_second_replace(source: Path, destination: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected publication failure")
                real_replace(source, destination)

            with mock.patch(
                "scripts.g1_demo1_build.os.replace",
                side_effect=fail_second_replace,
            ):
                with self.assertRaisesRegex(OSError, "injected publication failure"):
                    _stage_and_publish(
                        {first: "new first", second: "new second"},
                        obsolete_paths=(obsolete,),
                    )

            self.assertEqual(first.read_text(encoding="utf-8"), "old first")
            self.assertFalse(second.exists())
            self.assertEqual(obsolete.read_text(encoding="utf-8"), "old legacy")


if __name__ == "__main__":
    unittest.main()
