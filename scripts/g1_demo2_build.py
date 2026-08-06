"""Build the complete, validated Demo 2 g1 dataset and audit artifacts.

Demo 2 is interjection: 2.1 ``suggest_edit`` under a standing correction
instruction, 2.2 ``highlight`` under a standing category instruction.  The
production contract is fixed by :class:`Demo2Targets` and proved, not assumed:
every emitted row is re-parsed with the serving parser and every ``suggest_edit``
quote is re-checked for uniqueness against the visible textbox recovered from
the prompt itself.
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
from html import unescape
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain import ActionKind  # noqa: E402
from app.stream import parse_g1_action  # noqa: E402
from datagen.g1_authored_demo2 import (  # noqa: E402
    demo2_authored_paths,
    load_authored_batches,
    occurrence_count,
    occurrence_start,
    validate_demo2_batches,
)
from datagen.g1_demo2 import (  # noqa: E402
    AFTER_ROLES,
    BEFORE_ROLES,
    EMPTY_KINDS,
    G1_SCHEMA_VERSION,
    TRAP_ROLES,
    Demo2Targets,
    compile_demo2_dataset,
    demo2_records_from_batches,
)


BUILD_SCHEMA_VERSION = "g1-demo2-build-1"
DEFAULT_AUTHORED_ROOT = ROOT / "data" / "g1_authored"
DEFAULT_TRAIN_BASE_PATH = ROOT / "data" / "train_g1_demo2.jsonl"
DEFAULT_DEV_PATH = ROOT / "data" / "dev_g1_demo2.jsonl"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "g1-demo2"
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
    "source_mode",
    "source_skeleton",
    "source_persona",
    "source_domain",
    "source_register",
    "source_author_slot",
    "source_author_model",
    "source_author_tranche",
    "segment_index",
    "trigger_key",
    "trigger_kind",
    "trap",
    "before_kind",
    "error_subtype",
    "category",
    "pause_after",
    "empty_kind",
    "obligation",
}
SOURCE_DISTRIBUTION_FIELDS = ("mode", "persona", "domain", "register", "author_slot")


class Demo2BuildError(RuntimeError):
    """Raised before publication when a Demo 2 build contract is not met."""


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
    if minimum_shards <= 0:
        raise Demo2BuildError("minimum_shards must be positive")
    if max_shard_bytes <= 0:
        raise Demo2BuildError("max_shard_bytes must be positive")
    if len(rows) < minimum_shards:
        raise Demo2BuildError(
            f"cannot place {len(rows)} train rows into {minimum_shards} non-empty shards"
        )
    for shard_count in range(minimum_shards, len(rows) + 1):
        quotient, remainder = divmod(len(rows), shard_count)
        sizes = [quotient + (1 if index < remainder else 0) for index in range(shard_count)]
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
                (_train_shard_path(base_path, index, shard_count), shard_rows, payload, payload_bytes)
            )
        if not oversize:
            return shards
    raise Demo2BuildError(
        f"at least one train row exceeds the {max_shard_bytes}-byte shard limit"
    )


def visible_text_from_prompt(prompt: str) -> str:
    """Recover the textbox the model actually sees on the graded tick.

    Reading it back out of the rendered prompt (rather than trusting the
    compiler's own variable) is what makes the quote-uniqueness check an
    independent audit instead of a restatement.
    """

    marker = prompt.rfind("<stream_event ")
    if marker < 0:
        raise Demo2BuildError("prompt contains no stream event")
    tail = prompt[marker:]
    try:
        open_end = tail.index(">") + 1
        close = tail.rindex("</stream_event>")
    except ValueError as exc:  # pragma: no cover - malformed prompt
        raise Demo2BuildError("prompt has an unterminated stream event") from exc
    return unescape(tail[open_end:close])


def validate_demo2_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Validate the row envelope, every completion, and every quote's meaning."""

    candidate_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    completion_counts: Counter[str] = Counter()
    responded_targets: set[tuple[str, int]] = set()
    edits_per_episode: dict[str, set[str]] = {}
    marks_per_episode: dict[str, set[tuple[str, int]]] = {}
    for index, row in enumerate(rows):
        location = f"row {index}"
        missing = sorted(REQUIRED_ROW_KEYS - set(row))
        if missing:
            raise Demo2BuildError(f"{location}: missing required keys {missing}")
        if row.get("schema_version") != G1_SCHEMA_VERSION:
            raise Demo2BuildError(f"{location}: schema_version must be {G1_SCHEMA_VERSION!r}")
        if row.get("demo") != "demo-2":
            raise Demo2BuildError(f"{location}: demo must be 'demo-2'")
        if row.get("situation") != row.get("candidate_role") or row.get("bucket") != row.get(
            "candidate_role"
        ):
            raise Demo2BuildError(
                f"{location}: situation, bucket, and candidate_role must match"
            )
        if row.get("episode") != row.get("source_record_id"):
            raise Demo2BuildError(f"{location}: episode and source_record_id must match")
        if row.get("source_mode") not in {"corrections", "highlights"}:
            raise Demo2BuildError(f"{location}: invalid source_mode {row.get('source_mode')!r}")
        split = row.get("split")
        if split not in {"train", "dev"}:
            raise Demo2BuildError(f"{location}: invalid split {split!r}")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise Demo2BuildError(f"{location}: candidate_id must be non-empty")
        if candidate_id in candidate_ids:
            raise Demo2BuildError(f"{location}: duplicate candidate_id {candidate_id!r}")
        candidate_ids.add(candidate_id)
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise Demo2BuildError(f"{location}: prompt must be non-empty")
        if not prompt.endswith("<PREDICT_THIS_ACTION>"):
            raise Demo2BuildError(f"{location}: prompt must end with <PREDICT_THIS_ACTION>")
        if "<interaction_context>" in prompt:
            raise Demo2BuildError(f"{location}: g1 prompts are header-free")
        if not isinstance(row.get("current_event_index"), int) or isinstance(
            row.get("current_event_index"), bool
        ) or int(row["current_event_index"]) <= 0:
            raise Demo2BuildError(f"{location}: current_event_index must be a positive integer")
        if not isinstance(row.get("current_content_empty"), bool):
            raise Demo2BuildError(f"{location}: current_content_empty must be boolean")
        if row.get("obligation") != "standing-instruction":
            raise Demo2BuildError(
                f"{location}: Demo 2 obligation must be 'standing-instruction'"
            )
        completion = row.get("completion")
        if not isinstance(completion, str):
            raise Demo2BuildError(f"{location}: completion must be a string")
        action = parse_g1_action(completion)
        if not action.valid:
            raise Demo2BuildError(f"{location}: invalid g1 completion: {action.diagnostic}")
        actual_class = (
            str(action.tool_name) if action.kind is ActionKind.TOOL else action.kind.value
        )
        if row.get("expected_class") != actual_class:
            raise Demo2BuildError(
                f"{location}: expected_class {row.get('expected_class')!r} does not match "
                f"completion class {actual_class!r}"
            )
        if actual_class not in {"idle", "respond", "suggest_edit", "highlight"}:
            raise Demo2BuildError(f"{location}: {actual_class!r} is not a Demo 2 action")

        episode = str(row["episode"])
        visible = visible_text_from_prompt(prompt)
        if action.kind is ActionKind.RESPOND:
            if action.target != row.get("current_event_index"):
                raise Demo2BuildError(
                    f"{location}: respond target does not match current_event_index"
                )
            key = (episode, int(action.target))
            if key in responded_targets:
                raise Demo2BuildError(f"{location}: two responds aimed at the same event")
            responded_targets.add(key)
        elif actual_class == "suggest_edit":
            quote = str(action.arguments["quote"])
            replacement = str(action.arguments["replacement"])
            if occurrence_count(visible, quote) != 1:
                raise Demo2BuildError(
                    f"{location}: suggest_edit quote {quote!r} occurs "
                    f"{occurrence_count(visible, quote)} times in the visible textbox"
                )
            if replacement == quote:
                raise Demo2BuildError(f"{location}: suggest_edit is a no-op")
            first_line = visible.split("\n", 1)[0]
            if quote and quote in first_line:
                raise Demo2BuildError(
                    f"{location}: suggest_edit quotes the standing instruction line"
                )
            seen = edits_per_episode.setdefault(episode, set())
            if quote in seen:
                raise Demo2BuildError(f"{location}: repeated suggest_edit quote in one episode")
            seen.add(quote)
        elif actual_class == "highlight":
            quote = str(action.arguments["quote"])
            occurrence = int(action.arguments["occurrence"])
            if " " in quote:
                raise Demo2BuildError(f"{location}: a highlight must quote one word")
            if occurrence_start(visible, quote, occurrence) < 0:
                raise Demo2BuildError(
                    f"{location}: highlight occurrence {occurrence} of {quote!r} does not exist"
                )
            seen_marks = marks_per_episode.setdefault(episode, set())
            if (quote, occurrence) in seen_marks:
                raise Demo2BuildError(f"{location}: repeated highlight in one episode")
            seen_marks.add((quote, occurrence))

        split_counts[str(split)] += 1
        completion_counts[actual_class] += 1

    before_counts = Counter(
        str(row["candidate_role"]) for row in rows if row["candidate_role"] in BEFORE_ROLES
    )
    after_counts = Counter(
        str(row["candidate_role"]) for row in rows if row["candidate_role"] in AFTER_ROLES
    )
    for before_role, after_role in zip(BEFORE_ROLES, AFTER_ROLES, strict=True):
        if before_counts.get(before_role, 0) != after_counts.get(after_role, 0):
            raise Demo2BuildError(
                f"{before_role} and {after_role} card counts must match; neighbours ship in pairs"
            )
    return {
        "rows": len(rows),
        "valid_g1_schemas": len(rows),
        "valid_g1_completions": len(rows),
        **{f"split_{name}": count for name, count in sorted(split_counts.items())},
        **{f"completion_{name}": count for name, count in sorted(completion_counts.items())},
    }


def _target_evidence(
    *,
    targets: Demo2Targets,
    source_counts: Mapping[str, Any],
    coverage: Mapping[str, Any],
    row_validation: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    roles = coverage["selected_roles"]
    empties = coverage["selected_empty_kinds"]
    trap_cards = sum(roles.get(role, 0) for role in TRAP_ROLES)
    checks = {
        "source_records": (targets.episodes, source_counts["records"]),
        "source_errors": (targets.errors, source_counts["errors_planted"]),
        "source_matches": (targets.matches, source_counts["literal_marks"]),
        "selected_cards": (targets.cards, coverage["selected_cards"]),
        "selected_instruction_acks": (targets.episodes, roles.get("instruction-ack", 0)),
        "selected_suggest_edits": (targets.errors, roles.get("error-positive", 0)),
        "selected_highlights": (targets.matches, roles.get("match-positive", 0)),
        "selected_hard_idles": (targets.hard_idle_cards, trap_cards),
        "g1_schema_rows": (targets.cards, row_validation["valid_g1_schemas"]),
        "parseable_completions": (targets.cards, row_validation["valid_g1_completions"]),
    }
    for empty_kind in EMPTY_KINDS:
        checks[f"empty_{empty_kind}"] = (
            targets.empty_per_kind,
            empties.get(empty_kind, 0),
        )
    for field, counts in coverage["selected_source_distribution"].items():
        checks[f"selected_{field}_cards"] = (targets.cards, sum(counts.values()))
    evidence = {
        name: {"target": target, "actual": actual, "exact": target == actual}
        for name, (target, actual) in checks.items()
    }
    failures = [name for name, result in evidence.items() if not result["exact"]]
    if failures:
        raise Demo2BuildError("Demo 2 exact-target evidence failed: " + ", ".join(failures))
    return evidence


def _stable_row(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(rows, key=lambda row: str(row["candidate_id"]))


def _render_sample(title: str, row: Mapping[str, Any]) -> list[str]:
    source = "/".join(
        str(row.get(field) or "-")
        for field in ("source_author_slot", "source_persona", "source_domain", "source_mode")
    )
    return [
        f"### {title}",
        "",
        f"- Candidate: `{row['candidate_id']}`",
        f"- Split / role: `{row['split']}` / `{row['candidate_role']}`",
        f"- Source: `{source}` (skeleton `{row['source_skeleton']}`)",
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
    if not rows:
        raise Demo2BuildError("cannot create inspection samples from zero rows")
    lines = [
        "# Demo 2 inspection samples",
        "",
        "Deterministic representatives for every selected role and trap, "
        "source-distribution extremes, and prompt-size extremes.",
        "",
        "## Selected-role representatives",
        "",
    ]
    for role in sorted(coverage["selected_roles"]):
        lines.extend(
            _render_sample(
                f"role: {role}", _stable_row(row for row in rows if row["candidate_role"] == role)
            )
        )

    lines.extend(["## Trap representatives", ""])
    for trap in sorted(coverage["selected_traps"]):
        lines.extend(
            _render_sample(
                f"trap: {trap}", _stable_row(row for row in rows if row.get("trap") == trap)
            )
        )

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
            lines.extend(
                _render_sample(
                    f"{field} {label}: {value} ({count} selected cards)",
                    _stable_row(row for row in rows if str(row.get(row_field)) == str(value)),
                )
            )

    lines.extend(["## Stream extremes", ""])
    shortest = min(rows, key=lambda row: (len(str(row["prompt"])), str(row["candidate_id"])))
    longest = min(rows, key=lambda row: (-len(str(row["prompt"])), str(row["candidate_id"])))
    lines.extend(_render_sample("shortest selected prompt", shortest))
    lines.extend(_render_sample("longest selected prompt", longest))
    return "\n".join(lines).rstrip() + "\n"


def _stage_and_publish(
    payloads: Mapping[Path, str], *, obsolete_paths: Sequence[Path] = ()
) -> None:
    """Publish all files as one rollback-capable replacement transaction."""

    destinations = list(payloads)
    if len(set(destinations)) != len(destinations):
        raise Demo2BuildError("duplicate publication destination")
    overlap = set(destinations).intersection(obsolete_paths)
    if overlap:
        raise Demo2BuildError(
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


def build_demo2_artifacts(
    *,
    authored_root: Path = DEFAULT_AUTHORED_ROOT,
    train_base_path: Path = DEFAULT_TRAIN_BASE_PATH,
    dev_path: Path = DEFAULT_DEV_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    targets: Demo2Targets = Demo2Targets(),
    dev_fraction: float = 0.1,
    minimum_train_shards: int = DEFAULT_TRAIN_SHARDS,
    max_train_shard_bytes: int = MAX_TRAIN_SHARD_BYTES,
    allow_small_corpus: bool = False,
    fail_on_warnings: bool = False,
) -> dict[str, Any]:
    """Validate all source batches, compile, audit, and publish one build."""

    paths = demo2_authored_paths(authored_root)
    if not paths:
        raise Demo2BuildError(f"No Demo 2 authored JSON files found under {authored_root}")
    batches = load_authored_batches(paths)
    source_payloads = [path.read_bytes() for path in paths]
    for path, payload, batch in zip(paths, source_payloads, batches, strict=True):
        loaded_value = dict(batch)
        loaded_value.pop("_path", None)
        if json.loads(payload) != loaded_value:
            raise Demo2BuildError(f"Authored source changed while loading: {path}")
    source_validation = validate_demo2_batches(
        batches, enforce_distribution=not allow_small_corpus
    )
    if not source_validation["passed"]:
        raise Demo2BuildError(
            "Demo 2 authored validation failed: " + "; ".join(source_validation["errors"])
        )
    if source_validation["warnings"]:
        warning_text = "; ".join(source_validation["warnings"])
        print(
            "Demo 2 authored validation warnings (non-blocking canaries): "
            + warning_text,
            file=sys.stderr,
        )
        if fail_on_warnings:
            raise Demo2BuildError(
                "Demo 2 authored validation warnings are release-blocking "
                "(--fail-on-warnings): " + warning_text
            )

    source_records = demo2_records_from_batches(batches)
    build = compile_demo2_dataset(source_records, targets=targets, dev_fraction=dev_fraction)
    rows = list(build.rows)
    row_validation = validate_demo2_rows(rows)
    rows_by_split = {
        split: [row for row in rows if row["split"] == split] for split in ("train", "dev")
    }
    if sum(len(items) for items in rows_by_split.values()) != len(rows):
        raise Demo2BuildError("not every compiled row belongs to train or dev")

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
            for path, payload, batch in zip(paths, source_payloads, batches, strict=True)
        ],
        "counts": {
            key: source_counts[key]
            for key in (
                "batches",
                "records",
                "errors_planted",
                "literal_marks",
                "bait_marks",
                "repairs",
                "asides",
                "clean_passages",
            )
        },
        "distributions": {
            key: source_counts[key]
            for key in (
                "agent",
                "mode",
                "persona",
                "domain",
                "register",
                "length_bucket",
                "event_count_bucket",
                "trigger_position",
                "error_subtypes",
                "categories",
            )
        },
    }
    row_counts = {
        "total": len(rows),
        "train": len(rows_by_split["train"]),
        "dev": len(rows_by_split["dev"]),
    }
    targets_dict = asdict(targets)
    audit = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "dataset_schema": G1_SCHEMA_VERSION,
        "source": source,
        "targets": targets_dict,
        "distribution_gates_enforced": source_validation["distribution_enforced"],
        "source_warnings": source_validation["warnings"],
        "exact_target_evidence": evidence,
        "row_counts": row_counts,
        "row_validation": row_validation,
        "files": file_entries,
    }
    manifest = {
        **audit,
        "selected": {
            "roles": build.coverage["selected_roles"],
            "traps": build.coverage["selected_traps"],
            "empty_kinds": build.coverage["selected_empty_kinds"],
            "before_kinds": build.coverage["selected_before_kinds"],
            "splits": build.coverage["selected_splits"],
            "source_distribution": build.coverage["selected_source_distribution"],
        },
        "artifacts": {
            "manifest": _path_label(artifact_dir / "manifest.json"),
            "coverage": _path_label(artifact_dir / "coverage.json"),
            "inspection_samples": _path_label(artifact_dir / "inspection_samples.md"),
        },
    }
    coverage = {
        **audit,
        **{
            key: build.coverage[key]
            for key in (
                "selected_roles",
                "selected_traps",
                "selected_empty_kinds",
                "selected_before_kinds",
                "selected_error_subtypes",
                "selected_categories",
                "selected_splits",
                "selected_source_distribution",
                "source_splits",
                "source_modes",
                "source_skeletons",
                "source_turns",
                "typing_delta_chars",
                "time_gap_ms",
            )
        },
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
    _stage_and_publish(payloads, obsolete_paths=(train_base_path, *stale_shard_paths))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authored-root", type=Path, default=DEFAULT_AUTHORED_ROOT)
    parser.add_argument(
        "--train-base",
        type=Path,
        default=DEFAULT_TRAIN_BASE_PATH,
        help="Logical train JSONL name used to derive ordered shard names.",
    )
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    defaults = Demo2Targets()
    parser.add_argument("--errors", type=int, default=defaults.errors)
    parser.add_argument("--matches", type=int, default=defaults.matches)
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--cards", type=int, default=defaults.cards)
    parser.add_argument("--empty-per-kind", type=int, default=defaults.empty_per_kind)
    parser.add_argument("--hard-idle-cards", type=int, default=defaults.hard_idle_cards)
    parser.add_argument(
        "--allow-small-corpus",
        action="store_true",
        help=(
            "Skip the scale-dependent distribution gates. For fixture and smoke "
            "builds only; the manifest records that the gates did not run."
        ),
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Treat heuristic semantic warnings as acceptance failures.",
    )
    args = parser.parse_args()
    try:
        manifest = build_demo2_artifacts(
            authored_root=args.authored_root,
            train_base_path=args.train_base,
            dev_path=args.dev,
            artifact_dir=args.artifact_dir,
            dev_fraction=args.dev_fraction,
            targets=Demo2Targets(
                errors=args.errors,
                matches=args.matches,
                episodes=args.episodes,
                cards=args.cards,
                empty_per_kind=args.empty_per_kind,
                hard_idle_cards=args.hard_idle_cards,
            ),
            allow_small_corpus=args.allow_small_corpus,
            fail_on_warnings=args.fail_on_warnings,
        )
    except (Demo2BuildError, ValueError) as exc:
        parser.exit(1, f"Demo 2 build failed: {exc}\n")
    counts = manifest["row_counts"]
    source = manifest["source"]["counts"]
    print(
        f"Demo 2: {counts['total']} rows "
        f"(train={counts['train']} in {len(manifest['files']['train']['shards'])} "
        f"shards, dev={counts['dev']})"
    )
    print(
        f"Source: {source['batches']} batches, {source['records']} records, "
        f"{source['errors_planted']} planted errors, {source['literal_marks']} highlight targets"
    )
    print(f"Manifest: {manifest['artifacts']['manifest']}")


if __name__ == "__main__":
    main()
