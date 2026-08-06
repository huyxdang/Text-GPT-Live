from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from app.policy import SYSTEM_PROMPT_G1
from scripts.g1_demo1_tokens import TokenAuditError, run_token_report


class FakeTokenizer:
    name_or_path = "fake-qwen"
    chat_template = "fake chat template"

    def __init__(self, model_max_length: int = 10_000) -> None:
        self.model_max_length = model_max_length
        self.chat_calls: list[tuple[bool, bool]] = []

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        if add_special_tokens:
            raise AssertionError("completion tokenization added special tokens")
        return {"input_ids": list(range(len(text.split()) or 1))}

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ):
        self.chat_calls.append((add_generation_prompt, enable_thinking))
        token_count = 5 + sum(len(message["content"].split()) for message in messages)
        # Match the BatchEncoding shape returned by the local Qwen tokenizer.
        return {"input_ids": list(range(token_count))}

    def convert_tokens_to_ids(self, token: str) -> int:
        if token != "<|im_end|>":
            raise AssertionError(f"unexpected token lookup: {token}")
        return 999


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def _row(
    split: str,
    role: str,
    prompt: str,
    completion: str,
    *,
    schema_version: str = "g1",
) -> dict:
    suffix = prompt.replace(" ", "-")
    return {
        "schema_version": schema_version,
        "split": split,
        "episode": f"episode-{suffix}",
        "candidate_id": f"candidate-{suffix}",
        "source_record_id": f"source-{suffix}",
        "candidate_role": role,
        "situation": role,
        "prompt": prompt,
        "completion": completion,
    }


def _write_fixture(
    root: Path, *, train_schema: str = "g1"
) -> tuple[Path, list[dict]]:
    train_rows = [
        _row(
            "train",
            "address-positive",
            "short prompt",
            "<action>idle()</action>",
            schema_version=train_schema,
        ),
        _row(
            "train",
            "narration-ballast",
            "a somewhat longer prompt here",
            "<action>idle()</action>",
        ),
    ]
    dev_rows = [
        _row(
            "dev",
            "address-positive",
            "the longest fixture prompt is in dev",
            '<action>respond({"for":1,"message":"ok"})</action>',
        )
    ]
    data_dir = root / "data"
    artifact_dir = root / "artifacts" / "g1-demo1"
    data_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    train_payloads = [_jsonl(train_rows[:1]), _jsonl(train_rows[1:])]
    train_entries = []
    aggregate = hashlib.sha256()
    for index, payload in enumerate(train_payloads, start=1):
        path = data_dir / f"train-{index}.jsonl"
        path.write_bytes(payload)
        aggregate.update(payload)
        train_entries.append(
            {
                "path": str(path.relative_to(root)),
                "rows": 1,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    dev_payload = _jsonl(dev_rows)
    dev_path = data_dir / "dev.jsonl"
    dev_path.write_bytes(dev_payload)
    manifest = {
        "schema_version": "g1-demo1-build-1",
        "dataset_schema": "g1",
        "files": {
            "train": {
                "format": "ordered-jsonl-shards",
                "rows": 2,
                "bytes": sum(len(payload) for payload in train_payloads),
                "aggregate_sha256": aggregate.hexdigest(),
                "shards": train_entries,
            },
            "dev": {
                "format": "jsonl",
                "path": str(dev_path.relative_to(root)),
                "rows": 1,
                "bytes": len(dev_payload),
                "sha256": hashlib.sha256(dev_payload).hexdigest(),
            },
        },
        "row_counts": {"train": 2, "dev": 1, "total": 3},
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, train_rows + dev_rows


class G1Demo1TokenAuditTests(unittest.TestCase):
    def test_audits_ordered_files_and_writes_split_role_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, rows = _write_fixture(root)
            output_path = manifest_path.parent / "token_report.json"
            tokenizer = FakeTokenizer()

            report = run_token_report(
                tokenizer=tokenizer,
                manifest_path=manifest_path,
                output_path=output_path,
                project_root=root,
                top_n=2,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(report, json.loads(output_path.read_text(encoding="utf-8")))
            self.assertEqual(report["overall"]["count"], 3)
            self.assertEqual(report["by_split"]["train"]["count"], 2)
            self.assertEqual(report["by_split"]["dev"]["count"], 1)
            self.assertEqual(report["by_role"]["address-positive"]["count"], 2)
            self.assertEqual(
                report["by_split_role"]["train"]["narration-ballast"]["count"],
                1,
            )
            self.assertEqual(len(report["top_longest"]), 2)
            self.assertEqual(report["top_longest"][0]["split"], "dev")
            longest_source = rows[-1]
            expected_prompt_tokens = (
                5
                + len(SYSTEM_PROMPT_G1.split())
                + len(longest_source["prompt"].split())
            )
            expected_completion_tokens = (
                len(longest_source["completion"].split()) + 1
            )
            self.assertEqual(
                report["top_longest"][0]["tokens"],
                expected_prompt_tokens + expected_completion_tokens,
            )
            self.assertEqual(
                report["top_longest"][0]["completion_tokens_including_im_end"],
                expected_completion_tokens,
            )
            self.assertTrue(report["hard_gate"]["passed"])
            self.assertEqual(
                report["rendering"]["chat_template_sha256"],
                report["tokenizer"]["chat_template_sha256"],
            )
            self.assertEqual(len(report["rendering"]["system_prompt_sha256"]), 64)
            self.assertEqual(len(report["tokenizer"]["fingerprint_sha256"]), 64)
            self.assertEqual(len(tokenizer.chat_calls), len(rows))
            self.assertTrue(all(call == (True, False) for call in tokenizer.chat_calls))
            self.assertEqual(
                list(output_path.parent.glob(".token_report.json.*.tmp")), []
            )

    def test_rejects_a_manifest_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = _write_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["dev"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(TokenAuditError, "sha256 mismatch"):
                run_token_report(
                    tokenizer=FakeTokenizer(),
                    manifest_path=manifest_path,
                    output_path=manifest_path.parent / "token_report.json",
                    project_root=root,
                )

    def test_rejects_stale_manifest_and_non_g1_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = _write_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = "old-build"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(TokenAuditError, "stale or unsupported"):
                run_token_report(
                    tokenizer=FakeTokenizer(),
                    manifest_path=manifest_path,
                    output_path=None,
                    project_root=root,
                )

    def test_requires_bytes_for_every_referenced_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = _write_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["files"]["train"]["shards"][0]["bytes"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(TokenAuditError, "bytes must be"):
                run_token_report(
                    tokenizer=FakeTokenizer(),
                    manifest_path=manifest_path,
                    output_path=None,
                    project_root=root,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = _write_fixture(root, train_schema="6")
            with self.assertRaisesRegex(
                TokenAuditError, "row schema_version must be 'g1'"
            ):
                run_token_report(
                    tokenizer=FakeTokenizer(),
                    manifest_path=manifest_path,
                    output_path=None,
                    project_root=root,
                )

    def test_writes_failed_report_then_raises_for_train_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = _write_fixture(root)
            output_path = manifest_path.parent / "token_report.json"

            with self.assertRaisesRegex(TokenAuditError, "full example"):
                run_token_report(
                    tokenizer=FakeTokenizer(model_max_length=10),
                    manifest_path=manifest_path,
                    output_path=output_path,
                    project_root=root,
                )

            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(report["hard_gate"]["passed"])
            self.assertGreater(report["hard_gate"]["examples_over"], 0)
            self.assertGreater(report["hard_gate"]["train_examples_over"], 0)
            self.assertEqual(report["by_split"]["dev"]["count"], 1)

    def test_dev_only_overflow_fails_the_context_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = _write_fixture(root)
            output_path = manifest_path.parent / "token_report.json"
            tokenizer = FakeTokenizer()
            baseline = run_token_report(
                tokenizer=tokenizer,
                manifest_path=manifest_path,
                output_path=None,
                project_root=root,
            )
            threshold = baseline["by_split"]["train"]["max"]
            self.assertGreater(baseline["by_split"]["dev"]["max"], threshold)
            tokenizer.model_max_length = threshold

            with self.assertRaisesRegex(TokenAuditError, "full example"):
                run_token_report(
                    tokenizer=tokenizer,
                    manifest_path=manifest_path,
                    output_path=output_path,
                    project_root=root,
                )

            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(report["hard_gate"]["passed"])
            self.assertEqual(report["hard_gate"]["train_examples_over"], 0)
            self.assertEqual(report["hard_gate"]["dev_examples_over"], 1)


if __name__ == "__main__":
    unittest.main()
