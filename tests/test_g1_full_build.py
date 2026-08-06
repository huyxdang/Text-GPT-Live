from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datagen.g1_demo1 import DEFAULT_DEMO1_PROFILE, DEMO1_TARGET_PROFILES
from scripts import g1_full_build
from scripts.g1_full_build import (
    DEMOS,
    G1FullBuildError,
    merge_demo_rows,
    publish_full_artifacts,
)


IDLE = "<action>idle()</action>"


def _target_tokens(row: dict) -> int:
    return int(row.get("_target_tokens", 10))


def _row(
    demo: str,
    index: int,
    *,
    split: str,
    prompt: str | None = None,
    completion: str = IDLE,
    expected_class: str = "idle",
    episode: str | None = None,
) -> dict:
    return {
        "schema_version": "g1",
        "demo": demo,
        "split": split,
        "episode": episode or f"{demo}-episode-{index}",
        "candidate_id": f"{demo}-candidate-{index}",
        "prompt": prompt or f"<stream_event index=\"{index}\">{demo}</stream_event>",
        "completion": completion,
        "expected_class": expected_class,
        "current_content_empty": False,
        "obligation": "none",
        "should_fire": False,
        "reminder_eval_kind": None,
        "timing_boundary": None,
        "clause_state": None,
    }


def _fixture_rows() -> dict[str, list[dict]]:
    rows = {
        demo: [
            _row(demo, 1, split="train"),
            _row(demo, 2, split="dev"),
        ]
        for demo in DEMOS
    }
    rows["demo-1"][1]["current_content_empty"] = True
    rows["demo-3"][0]["clause_state"] = "partial"
    rows["demo-3"][1]["clause_state"] = "complete"
    rows["demo-5"].append(_row("demo-5", 3, split="dev"))
    rows["demo-5"][1].update(
        {
            "should_fire": True,
            "reminder_eval_kind": "fire",
            "timing_boundary": "at",
        }
    )
    rows["demo-5"][2].update(
        {
            "reminder_eval_kind": "wait",
            "timing_boundary": "before",
        }
    )
    return rows


def _fixture_manifests() -> dict[str, dict]:
    manifests = {
        demo: {
            "schema_version": f"{demo}-build",
            "dataset_schema": "g1",
            "source": {"counts": {"records": 2}},
            "targets": {"cards": 2},
            "source_warnings": [],
            "exact_target_evidence": {},
            "row_counts": {"total": 2, "train": 1, "dev": 1},
            "row_validation": {"rows": 2},
            **(
                {
                    "chinese_reference_review": {
                        "status": "machine_reviewed_with_targeted_codex_corrections"
                    }
                }
                if demo == "demo-3"
                else {}
            ),
        }
        for demo in DEMOS
    }
    manifests["demo-5"]["source"]["counts"]["records"] = 3
    manifests["demo-5"]["targets"]["cards"] = 3
    manifests["demo-5"]["row_counts"] = {"total": 3, "train": 1, "dev": 2}
    manifests["demo-5"]["row_validation"] = {"rows": 3}
    return manifests


class G1FullMergeTests(unittest.TestCase):
    def test_exact_duplicates_are_removed_and_cannot_cross_splits(self) -> None:
        rows = _fixture_rows()
        shared = "<stream_event index=\"1\"></stream_event>"
        rows["demo-1"][0]["prompt"] = shared
        rows["demo-5"][1]["prompt"] = shared

        merged, integrity = merge_demo_rows(rows)

        self.assertEqual(len(merged), 10)
        self.assertEqual(integrity["duplicate_prompt_groups"], 1)
        self.assertEqual(integrity["exact_pair_duplicates_removed"], 1)
        self.assertEqual(integrity["cross_split_duplicate_groups_resolved"], 1)
        train_prompts = {row["prompt"] for row in merged if row["split"] == "train"}
        dev_prompts = {row["prompt"] for row in merged if row["split"] == "dev"}
        self.assertFalse(train_prompts & dev_prompts)

    def test_conflicting_answers_for_one_prompt_fail(self) -> None:
        rows = _fixture_rows()
        shared = "<stream_event index=\"12\">hello</stream_event>"
        rows["demo-1"][0]["prompt"] = shared
        rows["demo-5"][1].update(
            {
                "prompt": shared,
                "completion": '<action>respond({"for":12,"message":"hello"})</action>',
                "expected_class": "respond",
            }
        )
        with self.assertRaisesRegex(G1FullBuildError, "Conflicting completions"):
            merge_demo_rows(rows)

    def test_episode_cannot_cross_train_and_dev(self) -> None:
        rows = _fixture_rows()
        rows["demo-2"][1]["episode"] = rows["demo-2"][0]["episode"]
        with self.assertRaisesRegex(G1FullBuildError, "Whole-episode split"):
            merge_demo_rows(rows)

    def test_completion_must_be_canonical(self) -> None:
        rows = _fixture_rows()
        rows["demo-3"][0]["completion"] = " idle() "
        with self.assertRaisesRegex(G1FullBuildError, "canonical g1 action"):
            merge_demo_rows(rows)


class G1FullPublishTests(unittest.TestCase):
    def test_orchestrator_uses_the_live_scaled_demo1_profile(self) -> None:
        builders = [
            "build_demo1_artifacts",
            "build_demo2_artifacts",
            "build_demo3_artifacts",
            "build_demo4_artifacts",
            "build_demo5_artifacts",
        ]
        patches = [mock.patch.object(g1_full_build, name, return_value={}) for name in builders]
        mocks = [patch.start() for patch in patches]
        self.addCleanup(lambda: [patch.stop() for patch in reversed(patches)])
        with mock.patch.object(g1_full_build, "_rows_from_manifest", return_value=[]):
            with tempfile.TemporaryDirectory() as temporary:
                g1_full_build._build_individual_demos(
                    authored_root=Path(temporary),
                    temporary_root=Path(temporary) / "staging",
                    fail_on_warnings=False,
                )
        self.assertEqual(
            mocks[0].call_args.kwargs["targets"],
            DEMO1_TARGET_PROFILES[DEFAULT_DEMO1_PROFILE],
        )

    def test_publish_writes_a_loader_compatible_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_base = root / "data" / "train_g1.jsonl"
            dev_path = root / "data" / "dev_g1.jsonl"
            artifact_dir = root / "artifacts" / "g1-full"
            manifest = publish_full_artifacts(
                rows_by_demo=_fixture_rows(),
                demo_manifests=_fixture_manifests(),
                train_base_path=train_base,
                dev_path=dev_path,
                artifact_dir=artifact_dir,
                minimum_train_shards=2,
                target_token_counter=_target_tokens,
            )

            self.assertEqual(manifest["schema_version"], "g1-full-build-1")
            self.assertEqual(manifest["dataset_schema"], "g1")
            self.assertEqual(manifest["row_counts"]["total"], 11)
            self.assertEqual(manifest["global_integrity"]["train_dev_prompt_overlap"], 0)
            self.assertEqual(
                sum(manifest["global_integrity"]["output_by_demo"].values()),
                manifest["global_integrity"]["output_rows"],
            )
            self.assertTrue(
                all(manifest["global_integrity"]["g1_eval_support"].values())
            )
            self.assertEqual(
                manifest["chinese_reference_review"]["status"],
                "machine_reviewed_with_targeted_codex_corrections",
            )
            train_entry = manifest["files"]["train"]
            aggregate = hashlib.sha256()
            for entry in train_entry["shards"]:
                path = Path(entry["path"])
                payload = path.read_bytes()
                aggregate.update(payload)
                self.assertEqual(entry["bytes"], len(payload))
                self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(train_entry["aggregate_sha256"], aggregate.hexdigest())
            dev_payload = dev_path.read_bytes()
            self.assertEqual(
                manifest["files"]["dev"]["sha256"],
                hashlib.sha256(dev_payload).hexdigest(),
            )
            on_disk = json.loads((artifact_dir / "manifest.json").read_text())
            self.assertEqual(on_disk, manifest)
            self.assertIn("Train/dev prompt overlap: 0", (artifact_dir / "inspection_samples.md").read_text())

    def test_publish_refuses_an_empty_required_evaluation_slice(self) -> None:
        rows = _fixture_rows()
        for row in rows["demo-5"]:
            row["should_fire"] = False
            row["reminder_eval_kind"] = "wait"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(G1FullBuildError, "should_fire"):
                publish_full_artifacts(
                    rows_by_demo=rows,
                    demo_manifests=_fixture_manifests(),
                    train_base_path=root / "data" / "train_g1.jsonl",
                    dev_path=root / "data" / "dev_g1.jsonl",
                    artifact_dir=root / "artifacts" / "g1-full",
                    minimum_train_shards=2,
                    target_token_counter=_target_tokens,
                )

    def test_publish_refuses_support_that_exists_only_in_train(self) -> None:
        rows = _fixture_rows()
        rows["demo-5"][0]["should_fire"] = True
        rows["demo-5"][0]["reminder_eval_kind"] = "fire"
        rows["demo-5"][1]["should_fire"] = False
        rows["demo-5"][1]["reminder_eval_kind"] = "wait"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(G1FullBuildError, "should_fire"):
                publish_full_artifacts(
                    rows_by_demo=rows,
                    demo_manifests=_fixture_manifests(),
                    train_base_path=root / "data" / "train_g1.jsonl",
                    dev_path=root / "data" / "dev_g1.jsonl",
                    artifact_dir=root / "artifacts" / "g1-full",
                    minimum_train_shards=2,
                    target_token_counter=_target_tokens,
                )

    def test_publish_removes_rows_above_the_tinker_limit(self) -> None:
        rows = _fixture_rows()
        rows["demo-1"][0]["_target_tokens"] = 11
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = publish_full_artifacts(
                rows_by_demo=rows,
                demo_manifests=_fixture_manifests(),
                train_base_path=root / "data" / "train_g1.jsonl",
                dev_path=root / "data" / "dev_g1.jsonl",
                artifact_dir=root / "artifacts" / "g1-full",
                minimum_train_shards=2,
                target_token_counter=_target_tokens,
                tinker_target_token_limit=10,
            )

            gate = manifest["global_integrity"]["tinker_target_token_gate"]
            self.assertEqual(gate["rows_removed"], 1)
            self.assertEqual(gate["max_before_filter"], 11)
            self.assertEqual(gate["max_after_filter"], 10)
            self.assertEqual(manifest["row_counts"]["total"], 10)
            self.assertEqual(gate["removed_by_demo"], {"demo-1": 1})
            self.assertEqual(
                manifest["global_integrity"]["output_by_demo"]["demo-1"], 1
            )
            self.assertEqual(
                manifest["global_integrity"]["removed_by_demo"]["demo-1"], 1
            )
            self.assertEqual(
                sum(manifest["global_integrity"]["output_by_demo"].values()), 10
            )
            payload = "".join(
                Path(entry["path"]).read_text()
                for entry in manifest["files"]["train"]["shards"]
            )
            self.assertNotIn("demo-1-candidate-1", payload)


if __name__ == "__main__":
    unittest.main()
