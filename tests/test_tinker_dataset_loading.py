from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from train import tinker_run


def _payload(*episodes: str, schema_version: str | int = "g1") -> bytes:
    return b"".join(
        (
            json.dumps(
                {"episode": episode, "schema_version": schema_version},
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for episode in episodes
    )


def _file_entry(path: str, payload: bytes) -> dict:
    return {
        "path": path,
        "bytes": len(payload),
        "rows": sum(bool(line.strip()) for line in payload.splitlines()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_fixture(root: Path) -> dict:
    data = root / "data"
    artifacts = root / "artifacts" / "g1-demo1"
    data.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    first = _payload("train-second-in-name-first-in-manifest")
    second = _payload("train-first-in-name-second-in-manifest", "train-tail")
    dev = _payload("dev-only")
    first_path = data / "z-shard.jsonl"
    second_path = data / "a-shard.jsonl"
    dev_path = data / "dev.jsonl"
    first_path.write_bytes(first)
    second_path.write_bytes(second)
    dev_path.write_bytes(dev)
    aggregate = hashlib.sha256()
    aggregate.update(first)
    aggregate.update(second)
    train_entries = [
        _file_entry("data/z-shard.jsonl", first),
        _file_entry("data/a-shard.jsonl", second),
    ]
    manifest = {
        "schema_version": "g1-demo1-build-1",
        "dataset_schema": "g1",
        "files": {
            "train": {
                "format": "ordered-jsonl-shards",
                "shards": train_entries,
                "bytes": len(first) + len(second),
                "rows": 3,
                "aggregate_sha256": aggregate.hexdigest(),
            },
            "dev": {"format": "jsonl", **_file_entry("data/dev.jsonl", dev)},
        },
    }
    _write_manifest(root, manifest)
    return manifest


def _write_manifest(root: Path, manifest: dict) -> None:
    path = root / "artifacts" / "g1-demo1" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


class TinkerManifestDatasetTests(unittest.TestCase):
    def test_paid_g1_dataset_resolution_fails_closed_without_full_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(tinker_run, "ROOT", root):
                with self.assertRaisesRegex(
                    SystemExit, "Paid training never falls back"
                ):
                    tinker_run.g1_train_dev_dataset_names()
            manifest = _write_fixture(root)
            manifest["schema_version"] = "g1-full-build-1"
            full_path = root / "artifacts" / "g1-full" / "manifest.json"
            full_path.parent.mkdir(parents=True)
            full_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(tinker_run, "ROOT", root):
                self.assertEqual(
                    tinker_run.g1_train_dev_dataset_names(),
                    ("train_g1", "dev_g1"),
                )

    def test_generated_train_and_dev_load_after_full_integrity_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            with mock.patch.object(tinker_run, "ROOT", root):
                self.assertEqual(
                    tinker_run.dataset_paths("train_g1_demo1"),
                    (root / "data/z-shard.jsonl", root / "data/a-shard.jsonl"),
                )
                self.assertEqual(
                    [
                        pair["episode"]
                        for pair in tinker_run.load_pairs("train_g1_demo1")
                    ],
                    [
                        "train-second-in-name-first-in-manifest",
                        "train-first-in-name-second-in-manifest",
                        "train-tail",
                    ],
                )
                self.assertEqual(
                    [pair["episode"] for pair in tinker_run.load_pairs("dev_g1_demo1")],
                    ["dev-only"],
                )

    def test_legacy_dataset_path_and_loading_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            path = data / "train_v4.jsonl"
            path.write_bytes(_payload("legacy", schema_version=4))
            with mock.patch.object(tinker_run, "ROOT", root):
                self.assertEqual(tinker_run.dataset_path("train"), path)
                self.assertEqual(tinker_run.dataset_paths("train"), (path,))
                self.assertEqual(
                    tinker_run.load_pairs("train")[0]["episode"], "legacy"
                )

    def test_missing_and_stale_manifests_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(tinker_run, "ROOT", root):
                with self.assertRaisesRegex(SystemExit, "Missing Demo 1 manifest"):
                    tinker_run.load_pairs("train_g1_demo1")

            manifest = _write_fixture(root)
            manifest["schema_version"] = "old-build"
            _write_manifest(root, manifest)
            with mock.patch.object(tinker_run, "ROOT", root):
                with self.assertRaisesRegex(SystemExit, "Stale or unsupported"):
                    tinker_run.load_pairs("train_g1_demo1")

    def test_each_train_file_and_aggregate_integrity_field_is_enforced(self) -> None:
        cases = (
            ("bytes", "byte count mismatch"),
            ("sha256", "SHA256 mismatch"),
            ("rows", "row count mismatch"),
            ("aggregate-bytes", "aggregate byte count mismatch"),
            ("aggregate-sha256", "aggregate SHA256 mismatch"),
            ("aggregate-rows", "aggregate row count mismatch"),
        )
        for case, message in cases:
            with (
                self.subTest(case=case),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                manifest = _write_fixture(root)
                train = manifest["files"]["train"]
                if case == "bytes":
                    train["shards"][0]["bytes"] += 1
                elif case == "sha256":
                    train["shards"][0]["sha256"] = "0" * 64
                elif case == "rows":
                    train["shards"][0]["rows"] += 1
                elif case == "aggregate-bytes":
                    train["bytes"] += 1
                elif case == "aggregate-sha256":
                    train["aggregate_sha256"] = "0" * 64
                else:
                    train["rows"] += 1
                _write_manifest(root, manifest)
                with mock.patch.object(tinker_run, "ROOT", root):
                    with self.assertRaisesRegex(SystemExit, message):
                        tinker_run.load_pairs("train_g1_demo1")

    def test_missing_shard_and_tampered_dev_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            (root / "data/z-shard.jsonl").unlink()
            with mock.patch.object(tinker_run, "ROOT", root):
                with self.assertRaisesRegex(
                    SystemExit, "Missing Demo 1 train dataset file"
                ):
                    tinker_run.load_pairs("train_g1_demo1")

            manifest = _write_fixture(root)
            manifest["files"]["dev"]["sha256"] = "0" * 64
            _write_manifest(root, manifest)
            with mock.patch.object(tinker_run, "ROOT", root):
                with self.assertRaisesRegex(SystemExit, "SHA256 mismatch"):
                    tinker_run.load_pairs("dev_g1_demo1")


if __name__ == "__main__":
    unittest.main()
