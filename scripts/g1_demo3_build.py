"""Build the validated Demo 3 (English -> Chinese translation) g1 dataset.

The default contract: all accepted Demo 3 authored batches compile to 1,300
cards from exactly 532 clause commits across 102 episodes, including 40
examples of each empty-text kind and every authored trap idle.  Every one of
those numbers is a flag, not a constant.

    .venv/bin/python -m scripts.g1_demo3_build

Chinese provenance: the manifest records reference review as *machine* review.
Nobody on this project reads Chinese, so no human-verification claim is made
or implied.
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
from datagen.g1_authored_demo3 import (  # noqa: E402
    ACCEPTED_REVIEW_METHODS,
    authored_paths,
    load_authored_batches,
    validate_demo3_batches,
)
from datagen.g1_demo3 import (  # noqa: E402
    DEMO,
    EMPTY_KINDS,
    G1_SCHEMA_VERSION,
    NEIGHBOR_ROLES,
    TRAP_ROLES,
    Demo3Targets,
    compile_demo3_dataset,
    demo3_records_from_batches,
)


BUILD_SCHEMA_VERSION = "g1-demo3-build-1"
DEFAULT_AUTHORED_ROOT = ROOT / "data" / "g1_authored"
DEFAULT_TRAIN_BASE_PATH = ROOT / "data" / "train_g1_demo3.jsonl"
DEFAULT_DEV_PATH = ROOT / "data" / "dev_g1_demo3.jsonl"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "g1-demo3"
DEFAULT_TRAIN_SHARDS = 2
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
    "reference_review_method",
    "reference_review_reviewer",
    "clause_index",
    "clause_english",
    "clause_reference",
    "clause_end_offset",
    "step_index",
    "trap",
    "pause_after",
    "empty_kind",
    "opening_shape",
    "closing_shape",
    "obligation",
    "clause_state",
}
SOURCE_DISTRIBUTION_FIELDS = (
    "persona",
    "domain",
    "register",
    "author_slot",
    "reference_review_method",
)
VALID_OBLIGATIONS = {"none", "translate-active"}

CHINESE_REVIEW_DISCLAIMER = (
    "Chinese references in this dataset were reviewed by language models, not "
    "by a human reader. A focused Codex semantic pass corrected 11 references "
    "in previously flagged risk patterns, but it was not an exhaustive human "
    "certification. Meaning preservation remains machine-attested."
)


class Demo3BuildError(RuntimeError):
    """Raised before publication when a Demo 3 build contract is not met."""


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
        raise Demo3BuildError("minimum_shards must be positive")
    if max_shard_bytes <= 0:
        raise Demo3BuildError("max_shard_bytes must be positive")
    if len(rows) < minimum_shards:
        raise Demo3BuildError(
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
                (
                    _train_shard_path(base_path, index, shard_count),
                    shard_rows,
                    payload,
                    payload_bytes,
                )
            )
        if not oversize:
            return shards
    raise Demo3BuildError(
        f"at least one train row exceeds the {max_shard_bytes}-byte shard limit"
    )


def validate_demo3_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Validate the emitted row envelope and every canonical g1 completion.

    The load-bearing part is the completion round trip: every card is parsed by
    the same ``parse_g1_action`` the runtime uses, and a respond card must give
    back the exact Chinese it was built from.  A card the serving parser cannot
    read never reaches the dataset.

    Duplicate prompts are only a defect when they disagree on the answer.  An
    ``empty-initial`` card's prompt varies only by a 261-value hashed
    timestamp (see ``datagen.g1_demo3._stable_int(f"{record_id}:initial-time",
    ...)``), so once the corpus passes a few dozen episodes, two episodes
    landing on the same timestamp is expected, not exceptional -- the birthday
    paradox puts the corpus well past 50% collision odds long before 100
    episodes.  Two byte-identical prompts that both grade the same completion
    are the same training example twice: harmless, and arguably correct,
    since the right answer never depended on which episode produced it.
    Demo 1 already enforces only ``candidate_id`` uniqueness; this mirrors
    that rule while still catching the case that actually matters -- two
    identical prompts with *different* completions, which is contradictory
    training signal and remains a hard error.
    """

    candidate_ids: set[str] = set()
    prompts: dict[str, tuple[str, str]] = {}
    duplicate_prompt_completion_pairs = 0
    split_counts: Counter[str] = Counter()
    completion_counts: Counter[str] = Counter()
    non_ascii_responds = 0
    for index, row in enumerate(rows):
        location = f"row {index}"
        missing = sorted(REQUIRED_ROW_KEYS - set(row))
        if missing:
            raise Demo3BuildError(f"{location}: missing required keys {missing}")
        if row.get("schema_version") != G1_SCHEMA_VERSION:
            raise Demo3BuildError(
                f"{location}: schema_version must be {G1_SCHEMA_VERSION!r}"
            )
        if row.get("demo") != DEMO:
            raise Demo3BuildError(f"{location}: demo must be {DEMO!r}")
        if row.get("situation") != row.get("candidate_role") or row.get("bucket") != row.get(
            "candidate_role"
        ):
            raise Demo3BuildError(
                f"{location}: situation, bucket, and candidate_role must match"
            )
        if row.get("episode") != row.get("source_record_id"):
            raise Demo3BuildError(f"{location}: episode and source_record_id must match")
        split = row.get("split")
        if split not in {"train", "dev"}:
            raise Demo3BuildError(f"{location}: invalid split {split!r}")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise Demo3BuildError(f"{location}: candidate_id must be non-empty")
        if candidate_id in candidate_ids:
            raise Demo3BuildError(f"{location}: duplicate candidate_id {candidate_id!r}")
        candidate_ids.add(candidate_id)
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise Demo3BuildError(f"{location}: prompt must be non-empty")
        if not prompt.endswith("<PREDICT_THIS_ACTION>"):
            raise Demo3BuildError(f"{location}: prompt must end with <PREDICT_THIS_ACTION>")
        if row.get("reference_review_method") not in ACCEPTED_REVIEW_METHODS:
            raise Demo3BuildError(
                f"{location}: reference_review_method must be one of "
                f"{sorted(ACCEPTED_REVIEW_METHODS)}"
            )
        if row.get("obligation") not in VALID_OBLIGATIONS:
            raise Demo3BuildError(f"{location}: invalid obligation {row.get('obligation')!r}")
        expected_clause_state = (
            "complete"
            if row.get("candidate_role") == "clause-positive"
            else "partial"
            if row.get("candidate_role") in {"clause-before", "partial-idle"}
            else None
        )
        if row.get("clause_state") != expected_clause_state:
            raise Demo3BuildError(
                f"{location}: clause_state must be {expected_clause_state!r}"
            )
        if not isinstance(row.get("current_event_index"), int) or isinstance(
            row.get("current_event_index"), bool
        ) or int(row["current_event_index"]) <= 0:
            raise Demo3BuildError(
                f"{location}: current_event_index must be a positive integer"
            )
        if not isinstance(row.get("current_content_empty"), bool):
            raise Demo3BuildError(f"{location}: current_content_empty must be boolean")
        completion = row.get("completion")
        if not isinstance(completion, str):
            raise Demo3BuildError(f"{location}: completion must be a string")
        existing = prompts.get(prompt)
        if existing is None:
            prompts[prompt] = (candidate_id, completion)
        elif existing[1] != completion:
            raise Demo3BuildError(
                f"{location}: prompt is byte-identical to card {existing[0]!r} but "
                f"completions differ ({candidate_id!r} vs {existing[0]!r})"
            )
        else:
            duplicate_prompt_completion_pairs += 1
        action = parse_g1_action(completion)
        if not action.valid:
            raise Demo3BuildError(f"{location}: invalid g1 completion: {action.diagnostic}")
        actual_class = (
            str(action.tool_name) if action.kind is ActionKind.TOOL else action.kind.value
        )
        if row.get("expected_class") != actual_class:
            raise Demo3BuildError(
                f"{location}: expected_class {row.get('expected_class')!r} does not "
                f"match completion class {actual_class!r}"
            )
        if action.kind is ActionKind.RESPOND:
            if action.target != row.get("current_event_index"):
                raise Demo3BuildError(
                    f"{location}: respond target does not match current_event_index"
                )
            if row.get("candidate_role") == "clause-positive":
                if action.message != row.get("clause_reference"):
                    raise Demo3BuildError(
                        f"{location}: clause respond message is not the authored "
                        "Chinese reference"
                    )
            if not action.message.isascii():
                non_ascii_responds += 1
        elif row.get("candidate_role") in {"clause-positive", "instruction-ack"}:
            raise Demo3BuildError(f"{location}: {row['candidate_role']} must be a respond")
        split_counts[str(split)] += 1
        completion_counts[actual_class] += 1
    return {
        "rows": len(rows),
        "valid_g1_schemas": len(rows),
        "valid_g1_completions": len(rows),
        "unique_prompts": len(prompts),
        "duplicate_prompt_completion_pairs": duplicate_prompt_completion_pairs,
        "non_ascii_responds": non_ascii_responds,
        **{f"split_{name}": count for name, count in sorted(split_counts.items())},
        **{f"completion_{name}": count for name, count in sorted(completion_counts.items())},
    }


def _target_evidence(
    *,
    targets: Demo3Targets,
    source_counts: Mapping[str, Any],
    coverage: Mapping[str, Any],
    row_validation: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    roles = coverage["selected_roles"]
    empty_counts = coverage["selected_empty_kinds"]
    checks: dict[str, tuple[int, int]] = {
        "source_episodes": (targets.episodes, source_counts["records"]),
        "source_clauses": (targets.clauses, source_counts["clauses"]),
        "compiled_clauses": (targets.clauses, coverage["source_clauses"]),
        "selected_cards": (targets.cards, coverage["selected_cards"]),
        "selected_clause_positive": (targets.clauses, roles.get("clause-positive", 0)),
        "selected_instruction_ack": (targets.episodes, roles.get("instruction-ack", 0)),
        "g1_schema_rows": (targets.cards, row_validation["valid_g1_schemas"]),
        "parseable_completions": (targets.cards, row_validation["valid_g1_completions"]),
        # unique_prompts is deliberately *not* an exact-target check: an
        # empty-initial card's prompt varies only by a 261-value hashed
        # timestamp, so once the corpus passes a few dozen episodes some
        # cross-episode collisions are expected (see validate_demo3_rows).
        # validate_demo3_rows already hard-errors if any duplicate prompt
        # carries a different completion; row_validation["unique_prompts"]
        # and ["duplicate_prompt_completion_pairs"] are surfaced in the
        # manifest for visibility instead of gating the build here.
    }
    for empty_kind in EMPTY_KINDS:
        checks[f"empty_{empty_kind}"] = (
            targets.empty_per_kind,
            empty_counts.get(empty_kind, 0),
        )
    for field, counts in coverage["selected_source_distribution"].items():
        checks[f"selected_{field}_cards"] = (targets.cards, sum(counts.values()))
    evidence = {
        name: {"target": target, "actual": actual, "exact": target == actual}
        for name, (target, actual) in checks.items()
    }
    floors = {
        "backtrack-idle": targets.min_backtrack_idles,
        "prequoted-idle": targets.min_prequoted_idles,
        "partial-idle": targets.min_partial_idles,
    }
    for role in TRAP_ROLES:
        actual = roles.get(role, 0)
        evidence[f"trap_floor_{role}"] = {
            "target": floors[role],
            "actual": actual,
            "exact": actual >= floors[role],
        }
    failures = [name for name, result in evidence.items() if not result["exact"]]
    if failures:
        raise Demo3BuildError(
            "Demo 3 exact-target evidence failed: " + ", ".join(failures)
        )
    return evidence


def _stable_row(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(rows, key=lambda row: str(row["candidate_id"]))


def _render_sample(title: str, row: Mapping[str, Any]) -> list[str]:
    source = "/".join(
        str(row.get(field) or "-")
        for field in ("source_author_slot", "source_persona", "source_domain")
    )
    lines = [
        f"### {title}",
        "",
        f"- Candidate: `{row['candidate_id']}`",
        f"- Split / role: `{row['split']}` / `{row['candidate_role']}`",
        f"- Skeleton: `{row['opening_shape']}` -> `{row['closing_shape']}`",
        f"- Source: `{source}`",
        f"- Expected: `{row['expected_class']}`",
        f"- Prompt characters: `{len(str(row['prompt']))}`",
    ]
    if row.get("clause_english"):
        lines.append(f"- Clause: `{row['clause_english']}` -> `{row['clause_reference']}`")
    lines.extend(["", "````text", str(row["prompt"]), "````", "", "````text", str(row["completion"]), "````", ""])
    return lines


def inspection_samples_markdown(
    rows: Sequence[Mapping[str, Any]], coverage: Mapping[str, Any]
) -> str:
    """Render deterministic role, distribution, and stream-extreme samples."""

    if not rows:
        raise Demo3BuildError("cannot create inspection samples from zero rows")
    lines = [
        "# Demo 3 inspection samples",
        "",
        "Deterministic representatives for every selected role, every episode "
        "skeleton, source-distribution extremes, and prompt-size extremes.",
        "",
        "> Chinese references carry **machine (LLM) review only**, including a "
        "focused Codex semantic correction pass. No human certification was performed.",
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

    lines.extend(["## Episode skeletons", ""])
    for field in ("opening_shape", "closing_shape"):
        for shape in sorted({str(row[field]) for row in rows}):
            lines.extend(
                _render_sample(
                    f"{field}: {shape}",
                    _stable_row(row for row in rows if str(row[field]) == shape),
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
        row_field = f"source_{field}" if field != "reference_review_method" else field
        for label, value, count in (
            ("dominant", dominant_value, dominant_count),
            ("sparse", sparse_value, sparse_count),
        ):
            lines.extend(
                _render_sample(
                    f"{field} {label}: {value} ({count} selected cards)",
                    _stable_row(
                        row for row in rows if str(row.get(row_field)) == str(value)
                    ),
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
        raise Demo3BuildError("duplicate publication destination")
    overlap = set(destinations).intersection(obsolete_paths)
    if overlap:
        raise Demo3BuildError(
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


def build_demo3_artifacts(
    *,
    authored_root: Path = DEFAULT_AUTHORED_ROOT,
    train_base_path: Path = DEFAULT_TRAIN_BASE_PATH,
    dev_path: Path = DEFAULT_DEV_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    targets: Demo3Targets = Demo3Targets(),
    dev_fraction: float = 0.1,
    minimum_train_shards: int = DEFAULT_TRAIN_SHARDS,
    max_train_shard_bytes: int = MAX_TRAIN_SHARD_BYTES,
    allow_small_corpus: bool = False,
    fail_on_warnings: bool = False,
) -> dict[str, Any]:
    """Validate all source batches, compile, audit, and publish one build."""

    paths = authored_paths(authored_root, DEMO)
    if not paths:
        raise Demo3BuildError(f"No Demo 3 authored JSON files found under {authored_root}")
    batches = load_authored_batches(paths)
    source_payloads = [path.read_bytes() for path in paths]
    for path, payload, batch in zip(paths, source_payloads, batches, strict=True):
        loaded_value = dict(batch)
        loaded_value.pop("_path", None)
        if json.loads(payload) != loaded_value:
            raise Demo3BuildError(f"Authored source changed while loading: {path}")
    source_validation = validate_demo3_batches(
        batches, enforce_distribution=not allow_small_corpus
    )
    if not source_validation["passed"]:
        raise Demo3BuildError(
            "Demo 3 authored validation failed: " + "; ".join(source_validation["errors"])
        )
    if source_validation["warnings"]:
        warning_text = "; ".join(source_validation["warnings"])
        print(
            "Demo 3 authored validation warnings (non-blocking canaries): "
            + warning_text,
            file=sys.stderr,
        )
        if fail_on_warnings:
            raise Demo3BuildError(
                "Demo 3 authored validation warnings are release-blocking "
                "(--fail-on-warnings): " + warning_text
            )

    source_records = demo3_records_from_batches(batches)
    build = compile_demo3_dataset(
        source_records, targets=targets, dev_fraction=dev_fraction
    )
    rows = list(build.rows)
    row_validation = validate_demo3_rows(rows)
    rows_by_split = {
        split: [row for row in rows if row["split"] == split] for split in ("train", "dev")
    }
    if sum(len(items) for items in rows_by_split.values()) != len(rows):
        raise Demo3BuildError("not every compiled row belongs to train or dev")

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
    review_methods = sorted(source_counts["reference_review_method"])
    chinese_reference_review = {
        "methods": source_counts["reference_review_method"],
        "human_verified": False,
        "machine_reviewed": review_methods == ["machine"],
        "decision_recorded": "2026-08-01",
        "disclaimer": CHINESE_REVIEW_DISCLAIMER,
        "targeted_semantic_review": {
            "reviewer_type": "language_model",
            "scope": "previously flagged literal, awkward, and role-shifting patterns",
            "references_corrected": 11,
            "human_certification": False,
        },
        "deterministic_checks": [
            "reference is non-empty and free of surrounding whitespace",
            "reference contains Han characters",
            "reference is not byte-identical to its English source",
            "reference carries no action or stream markup and no < > &",
            "reference has no undeclared Latin run of 4+ letters",
            "reference uses full-width, not half-width, punctuation",
            "reference/English length ratio is within [0.12, 1.30]",
            "no two clauses share a byte-identical reference",
            "every clause completion round-trips through parse_g1_action byte-exactly",
        ],
    }
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
            key: source_counts[key] for key in ("batches", "records", "clauses", "steps")
        },
        "distributions": {
            key: source_counts[key]
            for key in (
                "agent",
                "persona",
                "domain",
                "register",
                "length_bucket",
                "clause_count_bucket",
                "trigger_position",
                "opening_shape",
                "closing_shape",
                "traps",
                "trap_signatures",
                "ack_openers",
                "instruction_openers",
            )
        },
        "joint_distributions": source_counts["joint_distributions"],
        "chinese_reference_review": chinese_reference_review,
    }
    row_counts = {
        "total": len(rows),
        "train": len(rows_by_split["train"]),
        "dev": len(rows_by_split["dev"]),
    }
    selected = {
        "roles": build.coverage["selected_roles"],
        "empty_kinds": build.coverage["selected_empty_kinds"],
        "traps": build.coverage["selected_traps"],
        "splits": build.coverage["selected_splits"],
        "source_distribution": build.coverage["selected_source_distribution"],
    }
    targets_dict = asdict(targets)
    targets_dict["neighbor_weights"] = dict(targets.neighbor_weights)
    common = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "dataset_schema": G1_SCHEMA_VERSION,
        "demo": DEMO,
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
        **common,
        "chinese_reference_review": chinese_reference_review,
        "selected": selected,
        "artifacts": {
            "manifest": _path_label(artifact_dir / "manifest.json"),
            "coverage": _path_label(artifact_dir / "coverage.json"),
            "inspection_samples": _path_label(artifact_dir / "inspection_samples.md"),
        },
    }
    coverage = {
        **common,
        "selected_roles": build.coverage["selected_roles"],
        "selected_empty_kinds": build.coverage["selected_empty_kinds"],
        "selected_traps": build.coverage["selected_traps"],
        "selected_splits": build.coverage["selected_splits"],
        "selected_source_distribution": build.coverage["selected_source_distribution"],
        "source_splits": build.coverage["source_splits"],
        "source_turns": build.coverage["source_turns"],
        "opening_shapes": build.coverage["opening_shapes"],
        "closing_shapes": build.coverage["closing_shapes"],
        "typing_delta_chars": build.coverage["typing_delta_chars"],
        "tick_gap_ms": build.coverage["tick_gap_ms"],
        "neighbor_roles": list(NEIGHBOR_ROLES),
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
    defaults = Demo3Targets()
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
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--clauses", type=int, default=defaults.clauses)
    parser.add_argument("--cards", type=int, default=defaults.cards)
    parser.add_argument("--empty-per-kind", type=int, default=defaults.empty_per_kind)
    parser.add_argument("--min-backtrack", type=int, default=defaults.min_backtrack_idles)
    parser.add_argument("--min-prequoted", type=int, default=defaults.min_prequoted_idles)
    parser.add_argument("--min-partial", type=int, default=defaults.min_partial_idles)
    parser.add_argument("--train-shards", type=int, default=DEFAULT_TRAIN_SHARDS)
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
        targets = Demo3Targets(
            episodes=args.episodes,
            clauses=args.clauses,
            cards=args.cards,
            empty_per_kind=args.empty_per_kind,
            min_backtrack_idles=args.min_backtrack,
            min_prequoted_idles=args.min_prequoted,
            min_partial_idles=args.min_partial,
        )
        manifest = build_demo3_artifacts(
            authored_root=args.authored_root,
            train_base_path=args.train_base,
            dev_path=args.dev,
            artifact_dir=args.artifact_dir,
            targets=targets,
            dev_fraction=args.dev_fraction,
            minimum_train_shards=args.train_shards,
            allow_small_corpus=args.allow_small_corpus,
            fail_on_warnings=args.fail_on_warnings,
        )
    except (Demo3BuildError, ValueError) as exc:
        parser.exit(1, f"Demo 3 build failed: {exc}\n")
    counts = manifest["row_counts"]
    source = manifest["source"]["counts"]
    print(
        f"Demo 3: {counts['total']} rows "
        f"(train={counts['train']} in {len(manifest['files']['train']['shards'])} "
        f"shards, dev={counts['dev']})"
    )
    print(
        f"Source: {source['batches']} batches, {source['records']} episodes, "
        f"{source['clauses']} clause commits"
    )
    print(
        "Chinese references: machine-reviewed with a focused Codex correction "
        "pass; not human-certified."
    )
    print(f"Manifest: {manifest['artifacts']['manifest']}")


if __name__ == "__main__":
    main()
