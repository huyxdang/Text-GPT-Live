"""Audit full training-example token lengths for the built Demo 1 g1 dataset.

The audit reads the build manifest, verifies its ordered JSONL files, and uses
the exact g1 training render: SYSTEM_PROMPT_G1 plus the row prompt through the
model-native chat template with thinking disabled, followed by the gold
completion and <|im_end|>.

Run:
    .venv/bin/python -m scripts.g1_demo1_tokens
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from app.policy import SYSTEM_PROMPT_G1


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "artifacts" / "g1-demo1" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "g1-demo1" / "token_report.json"
DEFAULT_MODEL = ROOT / "models" / "Qwen3.5-4B"
DEFAULT_TOP_N = 20
STANDARD_THRESHOLDS = (65_536, 131_072, 262_144)
REPORT_SCHEMA_VERSION = "g1-demo1-token-report-1"
BUILD_SCHEMA_VERSION = "g1-demo1-build-1"
TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


class TokenAuditError(RuntimeError):
    """The manifest, dataset, tokenizer, or context-length gate is invalid."""


class TokenizerLike(Protocol):
    model_max_length: int

    def __call__(
        self, text: str, *, add_special_tokens: bool = False
    ) -> Mapping[str, Any]: ...

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Any: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...


def _token_ids(value: Any, *, label: str) -> list[int]:
    """Normalize list and BatchEncoding-style tokenizer results."""

    if isinstance(value, list):
        ids = value
    else:
        try:
            ids = value["input_ids"]
        except (KeyError, TypeError) as exc:
            raise TokenAuditError(f"{label} did not expose input_ids") from exc
    if not isinstance(ids, list):
        try:
            ids = list(ids)
        except TypeError as exc:
            raise TokenAuditError(f"{label} input_ids are not a sequence") from exc
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise TokenAuditError(f"{label} unexpectedly returned a batch")
        ids = ids[0]
    if any(not isinstance(token_id, int) for token_id in ids):
        raise TokenAuditError(f"{label} input_ids contain non-integer values")
    return ids


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_dataset_path(project_root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TokenAuditError(f"{label} requires a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TokenAuditError(f"{label} must be a non-negative integer")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise TokenAuditError(f"{label} must be a lowercase SHA256 hex digest")
    return value


def _manifest_sources(
    manifest: Mapping[str, Any], project_root: Path
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    if (
        manifest.get("schema_version") != BUILD_SCHEMA_VERSION
        or manifest.get("dataset_schema") != "g1"
    ):
        raise TokenAuditError(
            "stale or unsupported manifest: expected "
            f"schema_version={BUILD_SCHEMA_VERSION!r} and dataset_schema='g1'"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TokenAuditError("manifest.files must be an object")
    train = files.get("train")
    dev = files.get("dev")
    if not isinstance(train, Mapping) or not isinstance(dev, Mapping):
        raise TokenAuditError("manifest.files requires train and dev entries")
    if train.get("format") != "ordered-jsonl-shards":
        raise TokenAuditError(
            "manifest files.train.format must be 'ordered-jsonl-shards'"
        )
    if dev.get("format") != "jsonl":
        raise TokenAuditError("manifest files.dev.format must be 'jsonl'")
    shards = train.get("shards")
    if not isinstance(shards, list) or not shards:
        raise TokenAuditError("manifest train entry requires ordered shards")

    sources: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(shards, start=1):
        if not isinstance(raw_entry, Mapping):
            raise TokenAuditError(f"train shard {index} must be an object")
        entry = dict(raw_entry)
        entry["split"] = "train"
        entry["order"] = index
        entry["resolved_path"] = _resolve_dataset_path(
            project_root, entry.get("path"), label=f"train shard {index}"
        )
        sources.append(entry)
    train_paths = [source["resolved_path"].resolve() for source in sources]
    if len(set(train_paths)) != len(train_paths):
        raise TokenAuditError("manifest lists a duplicate train shard path")

    dev_entry = dict(dev)
    dev_entry["split"] = "dev"
    dev_entry["order"] = 1
    dev_entry["resolved_path"] = _resolve_dataset_path(
        project_root, dev_entry.get("path"), label="dev file"
    )
    sources.append(dev_entry)
    _require_nonnegative_int(train.get("bytes"), label="manifest files.train.bytes")
    _require_nonnegative_int(train.get("rows"), label="manifest files.train.rows")
    _require_sha256(
        train.get("aggregate_sha256"),
        label="manifest files.train.aggregate_sha256",
    )
    return sources, train


def _nearest_rank(sorted_values: list[int], percentile: int) -> int:
    if not sorted_values:
        raise TokenAuditError("cannot summarize an empty token-length group")
    index = max(0, math.ceil(percentile / 100 * len(sorted_values)) - 1)
    return sorted_values[index]


def _length_stats(values: Iterable[int], thresholds: Iterable[int]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "over_thresholds": {str(value): 0 for value in thresholds},
        }
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _nearest_rank(ordered, 50),
        "p95": _nearest_rank(ordered, 95),
        "p99": _nearest_rank(ordered, 99),
        "max": ordered[-1],
        "over_thresholds": {
            str(threshold): sum(value > threshold for value in ordered)
            for threshold in thresholds
        },
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _tokenizer_name(tokenizer: TokenizerLike, fallback: str) -> str:
    value = getattr(tokenizer, "name_or_path", None)
    return str(value) if isinstance(value, (str, Path)) and str(value) else fallback


def _chat_template(tokenizer: TokenizerLike) -> str:
    getter = getattr(tokenizer, "get_chat_template", None)
    if callable(getter):
        try:
            value = getter()
        except (TypeError, ValueError):
            value = None
    else:
        value = None
    if not isinstance(value, str) or not value:
        value = getattr(tokenizer, "chat_template", None)
    if not isinstance(value, str) or not value:
        raise TokenAuditError("tokenizer does not expose its active chat template")
    return value


def _tokenizer_provenance(
    tokenizer: TokenizerLike, *, model_label: str | None
) -> dict[str, Any]:
    name = _tokenizer_name(tokenizer, model_label or "unknown")
    source = Path(name)
    if not source.is_dir() and model_label:
        source = Path(model_label)
    files: dict[str, dict[str, Any]] = {}
    if source.is_dir():
        for filename in TOKENIZER_FILES:
            path = source / filename
            if path.is_file():
                payload = path.read_bytes()
                files[filename] = {
                    "bytes": len(payload),
                    "sha256": _sha256_bytes(payload),
                }
    chat_template = _chat_template(tokenizer)
    template_sha = _sha256_bytes(chat_template.encode("utf-8"))
    fingerprint_input = {
        "name_or_path": name,
        "chat_template_sha256": template_sha,
        "files": files,
    }
    return {
        **fingerprint_input,
        "fingerprint_sha256": _sha256_bytes(
            json.dumps(
                fingerprint_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
    }


def run_token_report(
    *,
    tokenizer: TokenizerLike,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_path: Path | None = DEFAULT_OUTPUT,
    project_root: Path = ROOT,
    model_label: str | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    """Verify, tokenize, atomically report, and enforce the train context gate."""

    if top_n < 0:
        raise TokenAuditError("top_n must be non-negative")
    manifest_path = Path(manifest_path)
    project_root = Path(project_root)
    if not manifest_path.exists():
        raise TokenAuditError(f"manifest does not exist: {manifest_path}")
    manifest_payload = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_payload)
    except json.JSONDecodeError as exc:
        raise TokenAuditError(f"manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise TokenAuditError("manifest must contain one JSON object")
    sources, train_manifest = _manifest_sources(manifest, project_root)

    model_max_length = int(tokenizer.model_max_length)
    if model_max_length <= 0 or model_max_length > 10_000_000:
        raise TokenAuditError(
            f"tokenizer exposed an implausible model_max_length: {model_max_length}"
        )
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end_id, int) or isinstance(im_end_id, bool) or im_end_id < 0:
        raise TokenAuditError("tokenizer does not expose a valid <|im_end|> token")
    thresholds = tuple(sorted({*STANDARD_THRESHOLDS, model_max_length}))
    tokenizer_provenance = _tokenizer_provenance(
        tokenizer, model_label=model_label
    )

    all_lengths: list[int] = []
    by_split: dict[str, list[int]] = defaultdict(list)
    by_role: dict[str, list[int]] = defaultdict(list)
    by_split_role: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    longest: list[dict[str, Any]] = []
    verified_files: list[dict[str, Any]] = []
    split_rows: dict[str, int] = defaultdict(int)
    split_bytes: dict[str, int] = defaultdict(int)
    train_aggregate = hashlib.sha256()
    global_row_index = 0

    for source in sources:
        path = source["resolved_path"]
        split = str(source["split"])
        if not isinstance(path, Path) or not path.is_file():
            raise TokenAuditError(f"referenced {split} file does not exist: {path}")
        expected_rows = _require_nonnegative_int(
            source.get("rows"), label=f"{source.get('path')} rows"
        )
        if expected_rows == 0:
            raise TokenAuditError(
                f"{source.get('path')} must contain at least one row"
            )
        expected_sha = _require_sha256(
            source.get("sha256"), label=f"{source.get('path')} sha256"
        )
        expected_bytes = _require_nonnegative_int(
            source.get("bytes"), label=f"{source.get('path')} bytes"
        )

        digest = hashlib.sha256()
        file_rows = 0
        file_bytes = 0
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                file_bytes += len(raw_line)
                digest.update(raw_line)
                if split == "train":
                    train_aggregate.update(raw_line)
                if not raw_line.strip():
                    continue
                file_rows += 1
                split_rows[split] += 1
                global_row_index += 1
                try:
                    row = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise TokenAuditError(
                        f"{path}:{line_number}: invalid UTF-8 JSON row"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise TokenAuditError(f"{path}:{line_number}: row must be an object")
                if row.get("schema_version") != "g1":
                    raise TokenAuditError(
                        f"{path}:{line_number}: row schema_version must be 'g1'"
                    )
                if row.get("split") != split:
                    raise TokenAuditError(
                        f"{path}:{line_number}: row split {row.get('split')!r} "
                        f"does not match {split!r}"
                    )
                prompt = row.get("prompt")
                completion = row.get("completion")
                role = row.get(
                    "candidate_role", row.get("situation", row.get("bucket"))
                )
                if not isinstance(prompt, str) or not isinstance(completion, str):
                    raise TokenAuditError(
                        f"{path}:{line_number}: prompt and completion must be strings"
                    )
                if not isinstance(role, str) or not role:
                    raise TokenAuditError(f"{path}:{line_number}: row role is missing")

                rendered = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT_G1},
                        {"role": "user", "content": prompt},
                    ],
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt_ids = _token_ids(rendered, label="chat template")
                completion_result = tokenizer(
                    completion, add_special_tokens=False
                )
                completion_ids = _token_ids(
                    completion_result, label="completion tokenizer"
                )
                full_tokens = len(prompt_ids) + len(completion_ids) + 1
                all_lengths.append(full_tokens)
                by_split[split].append(full_tokens)
                by_role[role].append(full_tokens)
                by_split_role[split][role].append(full_tokens)
                longest.append(
                    {
                        "tokens": full_tokens,
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens_including_im_end": len(completion_ids) + 1,
                        "split": split,
                        "role": role,
                        "episode": row.get("episode"),
                        "candidate_id": row.get("candidate_id"),
                        "source_record_id": row.get("source_record_id"),
                        "global_row_index": global_row_index,
                        "split_row_index": split_rows[split],
                        "file": str(source.get("path")),
                        "line": line_number,
                    }
                )

        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            raise TokenAuditError(
                f"{source.get('path')}: sha256 mismatch; expected {expected_sha}, "
                f"found {actual_sha}"
            )
        if file_rows != expected_rows:
            raise TokenAuditError(
                f"{source.get('path')}: row mismatch; expected {expected_rows}, "
                f"found {file_rows}"
            )
        if file_bytes != expected_bytes:
            raise TokenAuditError(
                f"{source.get('path')}: byte mismatch; expected {expected_bytes}, "
                f"found {file_bytes}"
            )
        verified_files.append(
            {
                "split": split,
                "order": int(source["order"]),
                "path": str(source.get("path")),
                "rows": file_rows,
                "bytes": file_bytes,
                "sha256": actual_sha,
            }
        )
        split_bytes[split] += file_bytes

    actual_train_aggregate = train_aggregate.hexdigest()
    expected_train_aggregate = _require_sha256(
        train_manifest.get("aggregate_sha256"),
        label="manifest files.train.aggregate_sha256",
    )
    if actual_train_aggregate != expected_train_aggregate:
        raise TokenAuditError(
            "ordered train aggregate sha256 mismatch; "
            f"expected {expected_train_aggregate}, found {actual_train_aggregate}"
        )
    expected_train_bytes = _require_nonnegative_int(
        train_manifest.get("bytes"), label="manifest files.train.bytes"
    )
    if split_bytes["train"] != expected_train_bytes:
        raise TokenAuditError(
            "manifest train aggregate byte count is "
            f"{expected_train_bytes}, found {split_bytes['train']}"
        )
    expected_train_rows = _require_nonnegative_int(
        train_manifest.get("rows"), label="manifest files.train.rows"
    )
    if split_rows["train"] != expected_train_rows:
        raise TokenAuditError(
            "manifest train aggregate row count is "
            f"{expected_train_rows}, found {split_rows['train']}"
        )

    row_counts = manifest.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise TokenAuditError("manifest.row_counts must be an object")
    for split in ("train", "dev"):
        expected = _require_nonnegative_int(
            row_counts.get(split), label=f"manifest row_counts.{split}"
        )
        if split_rows[split] != expected:
            raise TokenAuditError(
                f"manifest {split} row count is {expected}, found {split_rows[split]}"
            )
    expected_total = _require_nonnegative_int(
        row_counts.get("total"), label="manifest row_counts.total"
    )
    if len(all_lengths) != expected_total:
        raise TokenAuditError(
            f"manifest total row count is {expected_total}, found {len(all_lengths)}"
        )

    longest.sort(
        key=lambda row: (
            -int(row["tokens"]),
            int(row["global_row_index"]),
        )
    )
    train_over_context = sum(
        value > model_max_length for value in by_split.get("train", [])
    )
    dev_over_context = sum(
        value > model_max_length for value in by_split.get("dev", [])
    )
    examples_over_context = train_over_context + dev_over_context
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "measured_at": datetime.now(UTC).isoformat(),
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_bytes(manifest_payload),
            "schema_version": manifest.get("schema_version"),
            "dataset_schema": manifest.get("dataset_schema"),
            "ordered_train_aggregate_sha256": actual_train_aggregate,
            "verified_files": verified_files,
        },
        "tokenizer": {
            **tokenizer_provenance,
            "model_max_length": model_max_length,
            "im_end_token": "<|im_end|>",
            "im_end_token_id": im_end_id,
            "weights_loaded": False,
        },
        "rendering": {
            "system_prompt": "SYSTEM_PROMPT_G1",
            "system_prompt_sha256": _sha256_bytes(
                SYSTEM_PROMPT_G1.encode("utf-8")
            ),
            "system_prompt_chars": len(SYSTEM_PROMPT_G1),
            "chat_template": "model-native",
            "chat_template_sha256": tokenizer_provenance[
                "chat_template_sha256"
            ],
            "add_generation_prompt": True,
            "enable_thinking": False,
            "completion_add_special_tokens": False,
            "completion_terminator": "<|im_end|>",
            "length_definition": "chat_template_prompt + completion + im_end",
        },
        "percentile_method": "nearest-rank",
        "thresholds": list(thresholds),
        "overall": _length_stats(all_lengths, thresholds),
        "by_split": {
            split: _length_stats(values, thresholds)
            for split, values in sorted(by_split.items())
        },
        "by_role": {
            role: _length_stats(values, thresholds)
            for role, values in sorted(by_role.items())
        },
        "by_split_role": {
            split: {
                role: _length_stats(values, thresholds)
                for role, values in sorted(roles.items())
            }
            for split, roles in sorted(by_split_role.items())
        },
        "top_longest": longest[:top_n],
        "hard_gate": {
            "name": "all full train and dev examples fit tokenizer model_max_length",
            "operator": "<=",
            "threshold": model_max_length,
            "examples_over": examples_over_context,
            "train_examples_over": train_over_context,
            "dev_examples_over": dev_over_context,
            "passed": examples_over_context == 0,
        },
    }
    destination = Path(output_path) if output_path is not None else None
    if destination is not None:
        _atomic_json(destination, report)
    if examples_over_context:
        report_location = f"; report written to {destination}" if destination else ""
        raise TokenAuditError(
            f"{examples_over_context} full example(s) exceed model_max_length "
            f"{model_max_length}{report_location}"
        )
    return report


def _print_stats(label: str, stats: Mapping[str, Any]) -> None:
    print(
        f"{label}: n={stats['count']:,} min={stats['min']:,} "
        f"p50={stats['p50']:,} p95={stats['p95']:,} "
        f"p99={stats['p99']:,} max={stats['max']:,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True
    )
    report = run_token_report(
        tokenizer=tokenizer,
        manifest_path=args.manifest,
        output_path=args.output,
        project_root=ROOT,
        model_label=str(args.model),
        top_n=args.top_n,
    )
    print(
        f"{report['tokenizer']['name_or_path']} · "
        f"context={report['tokenizer']['model_max_length']:,}"
    )
    _print_stats("all", report["overall"])
    for split, stats in report["by_split"].items():
        _print_stats(split, stats)
    print("over thresholds:", report["overall"]["over_thresholds"])
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
