"""Build the complete, validated Demo 1 g1 dataset and audit artifacts.

The authored source contract is fixed: all accepted Demo 1 batches must supply
exactly 700 addresses and 4,000 narrations across 700 records.  How many cards
are *selected* from that source is a profile, chosen with ``--profile``:

* ``scaled-1800`` (default) - the live 3x-scaled build.
* ``full-5400`` - the original build, kept reproducible for the archive.

Regenerating the archived 5,400-card build without touching the live one:

    .venv/bin/python -m scripts.g1_demo1_build \\
        --profile full-5400 \\
        --train-base data/archive/g1-demo1-5400/train_g1_demo1.jsonl \\
        --dev data/archive/g1-demo1-5400/dev_g1_demo1.jsonl \\
        --artifact-dir artifacts/g1-demo1-5400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain import ActionKind  # noqa: E402
from app.stream import parse_g1_action  # noqa: E402
from datagen.g1_authored import (  # noqa: E402
    authored_paths,
    load_authored_batches,
    validate_demo1_batches,
)
from datagen.g1_demo1 import (  # noqa: E402
    DEFAULT_DEMO1_PROFILE,
    DEMO1_TARGET_PROFILES,
    G1_SCHEMA_VERSION,
    Demo1Targets,
    compile_demo1_dataset,
    demo1_records_from_batches,
    demo1_selection_distribution_errors,
)


BUILD_SCHEMA_VERSION = "g1-demo1-build-1"
DEFAULT_AUTHORED_ROOT = ROOT / "data" / "g1_authored"
DEFAULT_TRAIN_BASE_PATH = ROOT / "data" / "train_g1_demo1.jsonl"
DEFAULT_DEV_PATH = ROOT / "data" / "dev_g1_demo1.jsonl"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "g1-demo1"
DEFAULT_TRAIN_SHARDS = 3
MAX_TRAIN_SHARD_BYTES = 90_000_000

REQUIRED_ROW_KEYS = {
    "schema_version",
    "split",
    "episode",
    "demo",
    "situation",
    "bucket",
    "prompt",
    "completion",
    "expected_class",
    "current_event_index",
    "current_content_empty",
    "candidate_id",
    "candidate_role",
    "source_record_id",
    "source_persona",
    "source_domain",
    "source_register",
    "source_author_slot",
    "source_author_model",
    "source_author_tranche",
    "segment_index",
    "traps",
    "pause_after",
    "narration_idle_kind",
    "empty_kind",
    "obligation",
}
SOURCE_DISTRIBUTION_FIELDS = ("persona", "domain", "register", "author_slot")


class Demo1BuildError(RuntimeError):
    """Raised before publication when a Demo 1 build contract is not met."""


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=indent,
        separators=(",", ":") if indent is None else None,
        sort_keys=True,
    )


def _sha256(payload: str | bytes) -> str:
    encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(encoded).hexdigest()


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _jsonl_payload(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(_json(dict(row)) + "\n" for row in rows)


def _train_shard_path(base_path: Path, index: int, count: int) -> Path:
    suffix = base_path.suffix or ".jsonl"
    stem = base_path.name[: -len(suffix)] if base_path.suffix else base_path.name
    return base_path.with_name(f"{stem}-{index:05d}-of-{count:05d}{suffix}")


def _existing_train_shard_paths(base_path: Path) -> list[Path]:
    suffix = base_path.suffix or ".jsonl"
    stem = base_path.name[: -len(suffix)] if base_path.suffix else base_path.name
    return sorted(base_path.parent.glob(f"{stem}-?????-of-?????{suffix}"))


def _train_shard_payloads(
    rows: Sequence[Mapping[str, Any]],
    *,
    base_path: Path,
    minimum_shards: int = DEFAULT_TRAIN_SHARDS,
    max_shard_bytes: int = MAX_TRAIN_SHARD_BYTES,
) -> list[tuple[Path, list[Mapping[str, Any]], str, int]]:
    """Split rows in order into stable, roughly equal, size-safe shards."""

    if minimum_shards <= 0:
        raise Demo1BuildError("minimum_shards must be positive")
    if max_shard_bytes <= 0:
        raise Demo1BuildError("max_shard_bytes must be positive")
    if len(rows) < minimum_shards:
        raise Demo1BuildError(
            f"cannot place {len(rows)} train rows into {minimum_shards} non-empty shards"
        )
    for shard_count in range(minimum_shards, len(rows) + 1):
        quotient, remainder = divmod(len(rows), shard_count)
        sizes = [
            quotient + (1 if index < remainder else 0)
            for index in range(shard_count)
        ]
        cursor = 0
        shards: list[tuple[Path, list[Mapping[str, Any]], str, int]] = []
        oversize = False
        for index, size in enumerate(sizes, start=1):
            shard_rows = list(rows[cursor : cursor + size])
            cursor += size
            payload = _jsonl_payload(shard_rows)
            payload_bytes = len(payload.encode("utf-8"))
            if payload_bytes > max_shard_bytes:
                oversize = True
                break
            shards.append(
                (
                    _train_shard_path(base_path, index, shard_count),
                    shard_rows,
                    payload,
                    payload_bytes,
                )
            )
        if not oversize:
            return shards
    raise Demo1BuildError(
        f"at least one train row exceeds the {max_shard_bytes}-byte shard limit"
    )


def validate_demo1_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Validate the emitted row envelope and every canonical g1 completion."""

    candidate_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    completion_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        location = f"row {index}"
        missing = sorted(REQUIRED_ROW_KEYS - set(row))
        if missing:
            raise Demo1BuildError(f"{location}: missing required keys {missing}")
        if row.get("schema_version") != G1_SCHEMA_VERSION:
            raise Demo1BuildError(
                f"{location}: schema_version must be {G1_SCHEMA_VERSION!r}"
            )
        if row.get("demo") != "demo-1":
            raise Demo1BuildError(f"{location}: demo must be 'demo-1'")
        if row.get("situation") != row.get("candidate_role") or row.get(
            "bucket"
        ) != row.get("candidate_role"):
            raise Demo1BuildError(
                f"{location}: situation, bucket, and candidate_role must match"
            )
        if row.get("episode") != row.get("source_record_id"):
            raise Demo1BuildError(
                f"{location}: episode and source_record_id must match"
            )
        split = row.get("split")
        if split not in {"train", "dev"}:
            raise Demo1BuildError(f"{location}: invalid split {split!r}")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise Demo1BuildError(f"{location}: candidate_id must be non-empty")
        if candidate_id in candidate_ids:
            raise Demo1BuildError(f"{location}: duplicate candidate_id {candidate_id!r}")
        candidate_ids.add(candidate_id)
        if not isinstance(row.get("prompt"), str) or not row["prompt"]:
            raise Demo1BuildError(f"{location}: prompt must be non-empty")
        if not row["prompt"].endswith("<PREDICT_THIS_ACTION>"):
            raise Demo1BuildError(
                f"{location}: prompt must end with <PREDICT_THIS_ACTION>"
            )
        if not isinstance(row.get("current_event_index"), int) or isinstance(
            row.get("current_event_index"), bool
        ) or int(row["current_event_index"]) <= 0:
            raise Demo1BuildError(
                f"{location}: current_event_index must be a positive integer"
            )
        if not isinstance(row.get("current_content_empty"), bool):
            raise Demo1BuildError(
                f"{location}: current_content_empty must be boolean"
            )
        if not isinstance(row.get("traps"), list) or any(
            not isinstance(trap, str) for trap in row["traps"]
        ):
            raise Demo1BuildError(f"{location}: traps must be an array of strings")
        if row.get("obligation") != "none":
            raise Demo1BuildError(f"{location}: Demo 1 obligation must be 'none'")
        completion = row.get("completion")
        if not isinstance(completion, str):
            raise Demo1BuildError(f"{location}: completion must be a string")
        action = parse_g1_action(completion)
        if not action.valid:
            raise Demo1BuildError(
                f"{location}: invalid g1 completion: {action.diagnostic}"
            )
        actual_class = (
            str(action.tool_name)
            if action.kind is ActionKind.TOOL
            else action.kind.value
        )
        if row.get("expected_class") != actual_class:
            raise Demo1BuildError(
                f"{location}: expected_class {row.get('expected_class')!r} "
                f"does not match completion class {actual_class!r}"
            )
        if action.kind is ActionKind.RESPOND and action.target != row.get(
            "current_event_index"
        ):
            raise Demo1BuildError(
                f"{location}: respond target does not match current_event_index"
            )
        split_counts[str(split)] += 1
        completion_counts[actual_class] += 1
    return {
        "rows": len(rows),
        "valid_g1_schemas": len(rows),
        "valid_g1_completions": len(rows),
        **{f"split_{name}": count for name, count in sorted(split_counts.items())},
        **{
            f"completion_{name}": count
            for name, count in sorted(completion_counts.items())
        },
    }


def _target_evidence(
    *,
    targets: Demo1Targets,
    source_counts: Mapping[str, Any],
    coverage: Mapping[str, Any],
    row_validation: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    empty_counts = coverage["selected_empty_kinds"]
    role_counts = coverage["selected_roles"]
    narration_cards = targets.narration_cards
    checks = {
        "source_records": (targets.addresses, source_counts["records"]),
        "source_addresses": (targets.addresses, source_counts["addresses"]),
        "source_narrations": (targets.narrations, source_counts["narration"]),
        "selected_cards": (targets.cards, coverage["selected_cards"]),
        "selected_address_before": (
            targets.address_sites,
            role_counts.get("address-before", 0),
        ),
        "selected_address_positive": (
            targets.address_sites,
            role_counts.get("address-positive", 0),
        ),
        "selected_address_after": (
            targets.address_sites,
            role_counts.get("address-after", 0),
        ),
        "selected_narrations": (
            narration_cards,
            role_counts.get("narration-hard-idle", 0)
            + role_counts.get("narration-ballast", 0),
        ),
        "empty_initial": (targets.empty_per_kind, empty_counts.get("initial", 0)),
        "empty_unchanged": (
            targets.empty_per_kind,
            empty_counts.get("unchanged", 0),
        ),
        "empty_cleared": (targets.empty_per_kind, empty_counts.get("cleared", 0)),
        "g1_schema_rows": (targets.cards, row_validation["valid_g1_schemas"]),
        "parseable_completions": (
            targets.cards,
            row_validation["valid_g1_completions"],
        ),
    }
    if targets.hard_idles is not None:
        checks["selected_hard_idles"] = (
            targets.hard_idles,
            role_counts.get("narration-hard-idle", 0),
        )
        checks["selected_ballast"] = (
            narration_cards - targets.hard_idles,
            role_counts.get("narration-ballast", 0),
        )
    for field, counts in coverage["selected_source_distribution"].items():
        checks[f"selected_{field}_cards"] = (targets.cards, sum(counts.values()))
    evidence = {
        name: {"target": target, "actual": actual, "exact": target == actual}
        for name, (target, actual) in checks.items()
    }
    failures = [name for name, result in evidence.items() if not result["exact"]]
    if failures:
        raise Demo1BuildError(
            "Demo 1 exact-target evidence failed: " + ", ".join(failures)
        )
    return evidence


def _stable_row(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(rows, key=lambda row: str(row["candidate_id"]))


def _render_sample(title: str, row: Mapping[str, Any]) -> list[str]:
    source = "/".join(
        str(row.get(field) or "-")
        for field in ("source_author_slot", "source_persona", "source_domain")
    )
    return [
        f"### {title}",
        "",
        f"- Candidate: `{row['candidate_id']}`",
        f"- Split / role: `{row['split']}` / `{row['candidate_role']}`",
        f"- Source: `{source}`",
        f"- Expected: `{row['expected_class']}`",
        f"- Prompt characters: `{len(str(row['prompt']))}`",
        "",
        "````text",
        str(row["prompt"]),
        "````",
        "",
        "````text",
        str(row["completion"]),
        "````",
        "",
    ]


def inspection_samples_markdown(
    rows: Sequence[Mapping[str, Any]], coverage: Mapping[str, Any]
) -> str:
    """Render deterministic role, distribution, and stream-extreme samples."""

    if not rows:
        raise Demo1BuildError("cannot create inspection samples from zero rows")
    lines = [
        "# Demo 1 inspection samples",
        "",
        "Deterministic representatives for every selected role, source-distribution "
        "extremes, and prompt-size extremes.",
        "",
        "## Selected-role representatives",
        "",
    ]
    for role in sorted(coverage["selected_roles"]):
        row = _stable_row(row for row in rows if row["candidate_role"] == role)
        lines.extend(_render_sample(f"role: {role}", row))

    lines.extend(["## Source-distribution representatives", ""])
    distributions = coverage["selected_source_distribution"]
    for field in SOURCE_DISTRIBUTION_FIELDS:
        counts = distributions.get(field, {})
        if not counts:
            continue
        dominant_value, dominant_count = min(
            counts.items(), key=lambda item: (-int(item[1]), str(item[0]))
        )
        sparse_value, sparse_count = min(
            counts.items(), key=lambda item: (int(item[1]), str(item[0]))
        )
        row_field = f"source_{field}"
        for label, value, count in (
            ("dominant", dominant_value, dominant_count),
            ("sparse", sparse_value, sparse_count),
        ):
            row = _stable_row(
                row for row in rows if str(row.get(row_field)) == str(value)
            )
            lines.extend(
                _render_sample(
                    f"{field} {label}: {value} ({count} selected cards)", row
                )
            )

    lines.extend(["## Stream extremes", ""])
    shortest = min(
        rows,
        key=lambda row: (len(str(row["prompt"])), str(row["candidate_id"])),
    )
    longest = min(
        rows,
        key=lambda row: (-len(str(row["prompt"])), str(row["candidate_id"])),
    )
    most_traps = min(
        rows,
        key=lambda row: (-len(row.get("traps") or []), str(row["candidate_id"])),
    )
    lines.extend(_render_sample("shortest selected prompt", shortest))
    lines.extend(_render_sample("longest selected prompt", longest))
    lines.extend(_render_sample("most trap labels on one selected card", most_traps))
    return "\n".join(lines).rstrip() + "\n"


def _stage_and_publish(
    payloads: Mapping[Path, str], *, obsolete_paths: Sequence[Path] = ()
) -> None:
    """Publish all files as one rollback-capable replacement transaction."""

    destinations = list(payloads)
    if len(set(destinations)) != len(destinations):
        raise Demo1BuildError("duplicate publication destination")
    overlap = set(destinations).intersection(obsolete_paths)
    if overlap:
        raise Demo1BuildError(
            f"publication destinations cannot also be obsolete: {sorted(overlap)}"
        )
    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path | None] = {}
    changed: list[Path] = []
    try:
        for destination, payload in payloads.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                staged.append((Path(handle.name), destination))
        for destination in [*destinations, *obsolete_paths]:
            if not destination.exists():
                backups[destination] = None
                continue
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".backup",
                delete=False,
            ) as handle:
                backup = Path(handle.name)
            backup.unlink()
            try:
                os.link(destination, backup)
            except OSError:
                shutil.copy2(destination, backup)
            backups[destination] = backup
        for temporary, destination in staged:
            os.replace(temporary, destination)
            changed.append(destination)
        for obsolete in obsolete_paths:
            if obsolete.exists():
                obsolete.unlink()
                changed.append(obsolete)
    except Exception:
        for destination in reversed(changed):
            backup = backups.get(destination)
            if backup is None:
                destination.unlink(missing_ok=True)
            elif backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            if backup is not None:
                backup.unlink(missing_ok=True)


def build_demo1_artifacts(
    *,
    authored_root: Path = DEFAULT_AUTHORED_ROOT,
    train_base_path: Path = DEFAULT_TRAIN_BASE_PATH,
    dev_path: Path = DEFAULT_DEV_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    targets: Demo1Targets = Demo1Targets(),
    dev_fraction: float = 0.1,
    minimum_train_shards: int = DEFAULT_TRAIN_SHARDS,
    max_train_shard_bytes: int = MAX_TRAIN_SHARD_BYTES,
    enforce_selection_distribution: bool = False,
    fail_on_warnings: bool = False,
) -> dict[str, Any]:
    """Validate all source batches, compile, audit, and publish one build."""

    paths = authored_paths(authored_root, "demo-1")
    if not paths:
        raise Demo1BuildError(
            f"No Demo 1 authored JSON files found under {authored_root}"
        )
    batches = load_authored_batches(paths)
    source_payloads = [path.read_bytes() for path in paths]
    for path, payload, batch in zip(paths, source_payloads, batches, strict=True):
        loaded_value = dict(batch)
        loaded_value.pop("_path", None)
        if json.loads(payload) != loaded_value:
            raise Demo1BuildError(f"Authored source changed while loading: {path}")
    source_validation = validate_demo1_batches(
        batches, enforce_distribution=True
    )
    if not source_validation["passed"]:
        raise Demo1BuildError(
            "Demo 1 authored validation failed: "
            + "; ".join(source_validation["errors"])
        )
    if source_validation["warnings"]:
        warning_text = "; ".join(source_validation["warnings"])
        print(
            "Demo 1 authored validation warnings (non-blocking canaries): "
            + warning_text,
            file=sys.stderr,
        )
        if fail_on_warnings:
            raise Demo1BuildError(
                "Demo 1 authored validation warnings are release-blocking "
                "(--fail-on-warnings): " + warning_text
            )

    source_records = demo1_records_from_batches(batches)
    build = compile_demo1_dataset(
        source_records,
        targets=targets,
        dev_fraction=dev_fraction,
    )
    rows = list(build.rows)
    row_validation = validate_demo1_rows(rows)
    rows_by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "dev")
    }
    if sum(len(items) for items in rows_by_split.values()) != len(rows):
        raise Demo1BuildError("not every compiled row belongs to train or dev")

    train_shards = _train_shard_payloads(
        rows_by_split["train"],
        base_path=train_base_path,
        minimum_shards=minimum_train_shards,
        max_shard_bytes=max_train_shard_bytes,
    )
    dev_payload = _jsonl_payload(rows_by_split["dev"])
    train_digest = hashlib.sha256()
    train_bytes = 0
    train_shard_entries: list[dict[str, Any]] = []
    for path, shard_rows, payload, payload_bytes in train_shards:
        train_digest.update(payload.encode("utf-8"))
        train_bytes += payload_bytes
        train_shard_entries.append(
            {
                "path": _path_label(path),
                "rows": len(shard_rows),
                "bytes": payload_bytes,
                "sha256": _sha256(payload),
            }
        )
    file_entries = {
        "train": {
            "format": "ordered-jsonl-shards",
            "rows": len(rows_by_split["train"]),
            "bytes": train_bytes,
            "aggregate_sha256": train_digest.hexdigest(),
            "max_shard_bytes": max_train_shard_bytes,
            "shards": train_shard_entries,
        },
        "dev": {
            "format": "jsonl",
            "path": _path_label(dev_path),
            "rows": len(rows_by_split["dev"]),
            "bytes": len(dev_payload.encode("utf-8")),
            "sha256": _sha256(dev_payload),
        },
    }
    selection_distribution_errors = demo1_selection_distribution_errors(build.coverage)
    if enforce_selection_distribution and selection_distribution_errors:
        raise Demo1BuildError(
            "Demo 1 selected-card distribution failed: "
            + "; ".join(selection_distribution_errors)
        )
    selection_distribution_gate = {
        "enforced": enforce_selection_distribution,
        "passed": not selection_distribution_errors,
        "errors": selection_distribution_errors,
    }
    source_counts = source_validation["counts"]
    evidence = _target_evidence(
        targets=targets,
        source_counts=source_counts,
        coverage=build.coverage,
        row_validation=row_validation,
    )
    source = {
        "root": _path_label(authored_root),
        "paths": [_path_label(path) for path in paths],
        "files": [
            {
                "path": _path_label(path),
                "sha256": _sha256(payload),
                "records": len(batch.get("records", [])),
            }
            for path, payload, batch in zip(
                paths, source_payloads, batches, strict=True
            )
        ],
        "counts": {
            key: source_counts[key]
            for key in ("batches", "records", "narration", "addresses")
        },
        "distributions": {
            key: source_counts[key]
            for key in (
                "agent",
                "persona",
                "domain",
                "register",
                "length_bucket",
                "event_count_bucket",
                "trigger_position",
                "traps",
            )
        },
    }
    row_counts = {
        "total": len(rows),
        "train": len(rows_by_split["train"]),
        "dev": len(rows_by_split["dev"]),
    }
    selected = {
        "roles": build.coverage["selected_roles"],
        "empty_kinds": build.coverage["selected_empty_kinds"],
        "splits": build.coverage["selected_splits"],
        "source_distribution": build.coverage["selected_source_distribution"],
        "distinct_records": build.coverage["selected_distinct_records"],
        "distribution_gate": selection_distribution_gate,
    }
    targets_dict = asdict(targets)
    manifest = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "dataset_schema": G1_SCHEMA_VERSION,
        "source": source,
        "targets": targets_dict,
        "source_warnings": source_validation["warnings"],
        "exact_target_evidence": evidence,
        "row_counts": row_counts,
        "row_validation": row_validation,
        "files": file_entries,
        "selected": selected,
        "artifacts": {
            "manifest": _path_label(artifact_dir / "manifest.json"),
            "coverage": _path_label(artifact_dir / "coverage.json"),
            "inspection_samples": _path_label(
                artifact_dir / "inspection_samples.md"
            ),
        },
    }
    coverage = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "dataset_schema": G1_SCHEMA_VERSION,
        "source": source,
        "targets": targets_dict,
        "source_warnings": source_validation["warnings"],
        "exact_target_evidence": evidence,
        "row_counts": row_counts,
        "row_validation": row_validation,
        "files": file_entries,
        "selected_roles": build.coverage["selected_roles"],
        "selected_empty_kinds": build.coverage["selected_empty_kinds"],
        "selected_splits": build.coverage["selected_splits"],
        "selected_source_distribution": build.coverage[
            "selected_source_distribution"
        ],
        "selected_distinct_records": build.coverage["selected_distinct_records"],
        "selected_distribution_gate": selection_distribution_gate,
        "source_splits": build.coverage["source_splits"],
        "source_turns": build.coverage["source_turns"],
        "typing_delta_chars": build.coverage["typing_delta_chars"],
    }
    inspection = inspection_samples_markdown(rows, build.coverage)
    payloads = {
        **{path: payload for path, _, payload, _ in train_shards},
        dev_path: dev_payload,
        artifact_dir / "manifest.json": _json(manifest, indent=2) + "\n",
        artifact_dir / "coverage.json": _json(coverage, indent=2) + "\n",
        artifact_dir / "inspection_samples.md": inspection,
    }
    current_shard_paths = {path for path, _, _, _ in train_shards}
    stale_shard_paths = [
        path
        for path in _existing_train_shard_paths(train_base_path)
        if path not in current_shard_paths
    ]
    _stage_and_publish(
        payloads,
        obsolete_paths=(train_base_path, *stale_shard_paths),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--authored-root", type=Path, default=DEFAULT_AUTHORED_ROOT)
    parser.add_argument(
        "--profile",
        choices=sorted(DEMO1_TARGET_PROFILES),
        default=DEFAULT_DEMO1_PROFILE,
        help="Selected-card mix. Source requirements are identical either way.",
    )
    parser.add_argument(
        "--train-base",
        type=Path,
        default=DEFAULT_TRAIN_BASE_PATH,
        help="Logical train JSONL name used to derive ordered shard names.",
    )
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Treat heuristic semantic warnings as acceptance failures.",
    )
    args = parser.parse_args()
    try:
        manifest = build_demo1_artifacts(
            authored_root=args.authored_root,
            train_base_path=args.train_base,
            dev_path=args.dev,
            artifact_dir=args.artifact_dir,
            targets=DEMO1_TARGET_PROFILES[args.profile],
            dev_fraction=args.dev_fraction,
            enforce_selection_distribution=True,
            fail_on_warnings=args.fail_on_warnings,
        )
    except (Demo1BuildError, ValueError) as exc:
        parser.exit(1, f"Demo 1 build failed: {exc}\n")
    counts = manifest["row_counts"]
    source = manifest["source"]["counts"]
    print(
        f"Demo 1 profile {args.profile}: {counts['total']} rows "
        f"(train={counts['train']} in {len(manifest['files']['train']['shards'])} "
        f"shards, dev={counts['dev']})"
    )
    print(
        f"Source: {source['batches']} batches, {source['records']} records, "
        f"{source['narration']} narrations, {source['addresses']} addresses"
    )
    roles = manifest["selected"]["roles"]
    print(
        "Selected roles: "
        + ", ".join(f"{role}={count}" for role, count in sorted(roles.items()))
    )
    print(
        f"Distinct source records represented: "
        f"{manifest['selected']['distinct_records']}"
    )
    print(f"Manifest: {manifest['artifacts']['manifest']}")


if __name__ == "__main__":
    main()
