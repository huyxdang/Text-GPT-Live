"""Build the validated Demo 4 (Talk-while-task) g1 dataset.

The default contract: all accepted Demo 4 authored banks (50 request
phrasings + 50 progress question/reply pairs) expand deterministically into
130 job-dialog episodes and compile to 870 cards, including the five required
traps (nudge-while-pending, honest-progress-check, unasked completion,
narration-during-job, failed-job-never-claims-success) and every empty-text
kind.  Every one of those numbers is a flag, not a constant.

    .venv/bin/python -m scripts.g1_demo4_build
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
from datagen.g1_authored_demo4 import (  # noqa: E402
    authored_paths,
    load_authored_batches,
    validate_demo4_batches,
)
from datagen.g1_demo4 import (  # noqa: E402
    DEMO,
    EMPTY_KINDS,
    G1_SCHEMA_VERSION,
    NEIGHBOR_ROLES,
    Demo4Targets,
    compile_demo4_dataset,
    demo4_banks_from_batches,
    expand_demo4_episodes,
)


BUILD_SCHEMA_VERSION = "g1-demo4-build-1"
DEFAULT_AUTHORED_ROOT = ROOT / "data" / "g1_authored"
DEFAULT_TRAIN_BASE_PATH = ROOT / "data" / "train_g1_demo4.jsonl"
DEFAULT_DEV_PATH = ROOT / "data" / "dev_g1_demo4.jsonl"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "g1-demo4"
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
    "job_id",
    "request_task",
    "content_kind",
    "outcome",
    "pause_after",
    "empty_kind",
    "opening_shape",
    "closing_shape",
    "obligation",
}
SOURCE_DISTRIBUTION_FIELDS = ("persona", "domain", "register", "author_slot")
VALID_OBLIGATIONS = {"none", "job-pending"}


class Demo4BuildError(RuntimeError):
    """Raised before publication when a Demo 4 build contract is not met."""


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
        raise Demo4BuildError("minimum_shards must be positive")
    if max_shard_bytes <= 0:
        raise Demo4BuildError("max_shard_bytes must be positive")
    if len(rows) < minimum_shards:
        raise Demo4BuildError(
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
    raise Demo4BuildError(
        f"at least one train row exceeds the {max_shard_bytes}-byte shard limit"
    )


def validate_demo4_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Validate the emitted row envelope and every canonical g1 completion.

    Every card is re-parsed by the same ``parse_g1_action`` the runtime uses.
    A card the serving parser cannot read never reaches the dataset.

    Unlike Demo 3's stricter check, byte-identical prompts across *different*
    candidate ids are not rejected here: an "empty-initial" tick (empty
    textbox, index 1, idle) is a context-free situation, and Demo 1's build
    accepts the same kind of incidental collision for the same reason. Only a
    duplicate ``candidate_id`` -- which would mean the same graded decision
    shipped twice -- is a hard failure.
    """

    candidate_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    completion_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        location = f"row {index}"
        missing = sorted(REQUIRED_ROW_KEYS - set(row))
        if missing:
            raise Demo4BuildError(f"{location}: missing required keys {missing}")
        if row.get("schema_version") != G1_SCHEMA_VERSION:
            raise Demo4BuildError(f"{location}: schema_version must be {G1_SCHEMA_VERSION!r}")
        if row.get("demo") != DEMO:
            raise Demo4BuildError(f"{location}: demo must be {DEMO!r}")
        if row.get("situation") != row.get("candidate_role") or row.get("bucket") != row.get(
            "candidate_role"
        ):
            raise Demo4BuildError(
                f"{location}: situation, bucket, and candidate_role must match"
            )
        if row.get("episode") != row.get("source_record_id"):
            raise Demo4BuildError(f"{location}: episode and source_record_id must match")
        split = row.get("split")
        if split not in {"train", "dev"}:
            raise Demo4BuildError(f"{location}: invalid split {split!r}")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise Demo4BuildError(f"{location}: candidate_id must be non-empty")
        if candidate_id in candidate_ids:
            raise Demo4BuildError(f"{location}: duplicate candidate_id {candidate_id!r}")
        candidate_ids.add(candidate_id)
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise Demo4BuildError(f"{location}: prompt must be non-empty")
        if not prompt.endswith("<PREDICT_THIS_ACTION>"):
            raise Demo4BuildError(f"{location}: prompt must end with <PREDICT_THIS_ACTION>")
        if "<interaction_context>" in prompt:
            raise Demo4BuildError(f"{location}: g1 prompts are header-free")
        if row.get("obligation") not in VALID_OBLIGATIONS:
            raise Demo4BuildError(f"{location}: invalid obligation {row.get('obligation')!r}")
        if not isinstance(row.get("current_event_index"), int) or isinstance(
            row.get("current_event_index"), bool
        ) or int(row["current_event_index"]) <= 0:
            raise Demo4BuildError(
                f"{location}: current_event_index must be a positive integer"
            )
        if not isinstance(row.get("current_content_empty"), bool):
            raise Demo4BuildError(f"{location}: current_content_empty must be boolean")
        completion = row.get("completion")
        if not isinstance(completion, str):
            raise Demo4BuildError(f"{location}: completion must be a string")
        action = parse_g1_action(completion)
        if not action.valid:
            raise Demo4BuildError(f"{location}: invalid g1 completion: {action.diagnostic}")
        actual_class = (
            str(action.tool_name) if action.kind is ActionKind.TOOL else action.kind.value
        )
        if row.get("expected_class") != actual_class:
            raise Demo4BuildError(
                f"{location}: expected_class {row.get('expected_class')!r} does not "
                f"match completion class {actual_class!r}"
            )
        role = row.get("candidate_role")
        if role == "request-positive":
            if actual_class != "delegate":
                raise Demo4BuildError(f"{location}: request-positive must be a delegate")
            if action.arguments.get("task") != row.get("request_task"):
                raise Demo4BuildError(f"{location}: delegate task drifted from the source")
        elif role in {
            "accepted-idle",
            "completed-idle",
            "failed-idle",
            "nudge-idle",
            "narration-idle",
        }:
            if actual_class != "idle":
                raise Demo4BuildError(f"{location}: {role} must be idle")
        elif role in {"ack-positive", "check-positive", "failure-check-positive"}:
            if actual_class != "respond":
                raise Demo4BuildError(f"{location}: {role} must be a respond")
            if action.target != row.get("current_event_index"):
                raise Demo4BuildError(
                    f"{location}: respond target does not match current_event_index"
                )
        # Tool-event ticks (accepted/completed/failed) are always graded idle
        # by construction, since respond can only target the current event and
        # a respond can never legally target a tool event. current_content_empty
        # is meaningless for tool events; job_id must be present exactly there.
        if role in {"accepted-idle", "completed-idle", "failed-idle"}:
            if not row.get("job_id"):
                raise Demo4BuildError(f"{location}: {role} row must carry a job_id")
        elif row.get("job_id") is not None:
            raise Demo4BuildError(f"{location}: {role} row must not carry a job_id")
        split_counts[str(split)] += 1
        completion_counts[actual_class] += 1
    return {
        "rows": len(rows),
        "valid_g1_schemas": len(rows),
        "valid_g1_completions": len(rows),
        **{f"split_{name}": count for name, count in sorted(split_counts.items())},
        **{f"completion_{name}": count for name, count in sorted(completion_counts.items())},
    }


def _target_evidence(
    *,
    targets: Demo4Targets,
    source_counts: Mapping[str, Any],
    coverage: Mapping[str, Any],
    row_validation: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    roles = coverage["selected_roles"]
    empties = coverage["selected_empty_kinds"]
    checks: dict[str, tuple[int, int]] = {
        "source_requests": (targets.requests, source_counts["requests"]),
        "source_progress_pairs": (targets.progress_pairs, source_counts["progress_pairs"]),
        "compiled_episodes": (targets.episodes, coverage["records"]),
        "selected_cards": (targets.cards, coverage["selected_cards"]),
        "selected_request_positive": (targets.episodes, roles.get("request-positive", 0)),
        "selected_accepted_idle": (targets.episodes, roles.get("accepted-idle", 0)),
        "selected_ack_positive": (targets.episodes, roles.get("ack-positive", 0)),
        "g1_schema_rows": (targets.cards, row_validation["valid_g1_schemas"]),
        "parseable_completions": (targets.cards, row_validation["valid_g1_completions"]),
    }
    for empty_kind in EMPTY_KINDS:
        checks[f"empty_{empty_kind}"] = (targets.empty_per_kind, empties.get(empty_kind, 0))
    for field, counts in coverage["selected_source_distribution"].items():
        checks[f"selected_{field}_cards"] = (targets.cards, sum(counts.values()))
    evidence = {
        name: {"target": target, "actual": actual, "exact": target == actual}
        for name, (target, actual) in checks.items()
    }
    floors = {
        "check-positive": targets.min_check_positive,
        "nudge-idle": targets.min_nudge_idle,
        "narration-idle": targets.min_narration_idle,
        "failure-check-positive": targets.min_failure_check,
    }
    for role, floor in floors.items():
        actual = roles.get(role, 0)
        evidence[f"trap_floor_{role}"] = {"target": floor, "actual": actual, "exact": actual >= floor}
    failures = [name for name, result in evidence.items() if not result["exact"]]
    if failures:
        raise Demo4BuildError("Demo 4 exact-target evidence failed: " + ", ".join(failures))
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
        f"- Content kind / outcome: `{row['content_kind']}` / `{row['outcome']}`",
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
    return lines


def inspection_samples_markdown(
    rows: Sequence[Mapping[str, Any]], coverage: Mapping[str, Any]
) -> str:
    """Render deterministic role, skeleton, and distribution representatives."""

    if not rows:
        raise Demo4BuildError("cannot create inspection samples from zero rows")
    lines = [
        "# Demo 4 inspection samples",
        "",
        "Deterministic representatives for every selected role, every episode "
        "skeleton, and source-distribution extremes.",
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

    lines.extend(["## Job outcomes", ""])
    for outcome in sorted({str(row["outcome"]) for row in rows}):
        lines.extend(
            _render_sample(
                f"outcome: {outcome}",
                _stable_row(row for row in rows if str(row["outcome"]) == outcome),
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
        raise Demo4BuildError("duplicate publication destination")
    overlap = set(destinations).intersection(obsolete_paths)
    if overlap:
        raise Demo4BuildError(
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


def build_demo4_artifacts(
    *,
    authored_root: Path = DEFAULT_AUTHORED_ROOT,
    train_base_path: Path = DEFAULT_TRAIN_BASE_PATH,
    dev_path: Path = DEFAULT_DEV_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    targets: Demo4Targets = Demo4Targets(),
    dev_fraction: float = 0.1,
    minimum_train_shards: int = DEFAULT_TRAIN_SHARDS,
    max_train_shard_bytes: int = MAX_TRAIN_SHARD_BYTES,
    allow_small_corpus: bool = False,
    fail_on_warnings: bool = False,
) -> dict[str, Any]:
    """Validate the two authored banks, expand, compile, audit, and publish."""

    paths = authored_paths(authored_root, DEMO)
    if not paths:
        raise Demo4BuildError(f"No Demo 4 authored JSON files found under {authored_root}")
    batches = load_authored_batches(paths)
    source_payloads = [path.read_bytes() for path in paths]
    for path, payload, batch in zip(paths, source_payloads, batches, strict=True):
        loaded_value = dict(batch)
        loaded_value.pop("_path", None)
        if json.loads(payload) != loaded_value:
            raise Demo4BuildError(f"Authored source changed while loading: {path}")
    source_validation = validate_demo4_batches(
        batches, enforce_distribution=not allow_small_corpus
    )
    if not source_validation["passed"]:
        raise Demo4BuildError(
            "Demo 4 authored validation failed: " + "; ".join(source_validation["errors"])
        )
    if source_validation["warnings"]:
        warning_text = "; ".join(source_validation["warnings"])
        print(
            "Demo 4 authored validation warnings (non-blocking canaries): "
            + warning_text,
            file=sys.stderr,
        )
        if fail_on_warnings:
            raise Demo4BuildError(
                "Demo 4 authored validation warnings are release-blocking "
                "(--fail-on-warnings): " + warning_text
            )

    requests, progress_pairs = demo4_banks_from_batches(batches)
    episodes = expand_demo4_episodes(requests, progress_pairs, count=targets.episodes)
    build = compile_demo4_dataset(episodes, targets=targets, dev_fraction=dev_fraction)
    rows = list(build.rows)
    row_validation = validate_demo4_rows(rows)
    rows_by_split = {
        split: [row for row in rows if row["split"] == split] for split in ("train", "dev")
    }
    if sum(len(items) for items in rows_by_split.values()) != len(rows):
        raise Demo4BuildError("not every compiled row belongs to train or dev")

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
                "requests": len(batch.get("requests", [])),
                "progress_pairs": len(batch.get("progress_pairs", [])),
            }
            for path, payload, batch in zip(paths, source_payloads, batches, strict=True)
        ],
        "counts": {
            key: source_counts[key] for key in ("batches", "requests", "progress_pairs", "checks", "nudges")
        },
        "distributions": {
            key: source_counts[key]
            for key in (
                "agent",
                "persona",
                "domain",
                "register",
                "length_bucket",
                "trigger_position",
                "record_kind",
                "request_openers",
                "reply_openers",
            )
        },
        "joint_distributions": source_counts["joint_distributions"],
    }
    row_counts = {"total": len(rows), "train": len(rows_by_split["train"]), "dev": len(rows_by_split["dev"])}
    selected = {
        "roles": build.coverage["selected_roles"],
        "empty_kinds": build.coverage["selected_empty_kinds"],
        "splits": build.coverage["selected_splits"],
        "outcomes": build.coverage["selected_outcomes"],
        "content_kinds": build.coverage["selected_content_kinds"],
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
        "selected_splits": build.coverage["selected_splits"],
        "selected_outcomes": build.coverage["selected_outcomes"],
        "selected_content_kinds": build.coverage["selected_content_kinds"],
        "selected_source_distribution": build.coverage["selected_source_distribution"],
        "source_splits": build.coverage["source_splits"],
        "source_turns": build.coverage["source_turns"],
        "opening_shapes": build.coverage["opening_shapes"],
        "closing_shapes": build.coverage["closing_shapes"],
        "outcomes": build.coverage["outcomes"],
        "content_kinds": build.coverage["content_kinds"],
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
        path for path in _existing_train_shard_paths(train_base_path) if path not in current_shard_paths
    ]
    _stage_and_publish(payloads, obsolete_paths=(train_base_path, *stale_shard_paths))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Demo4Targets()
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
    parser.add_argument("--requests", type=int, default=defaults.requests)
    parser.add_argument("--progress-pairs", type=int, default=defaults.progress_pairs)
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--cards", type=int, default=defaults.cards)
    parser.add_argument("--empty-per-kind", type=int, default=defaults.empty_per_kind)
    parser.add_argument("--min-check", type=int, default=defaults.min_check_positive)
    parser.add_argument("--min-nudge", type=int, default=defaults.min_nudge_idle)
    parser.add_argument("--min-narration", type=int, default=defaults.min_narration_idle)
    parser.add_argument("--min-failure-check", type=int, default=defaults.min_failure_check)
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
        targets = Demo4Targets(
            requests=args.requests,
            progress_pairs=args.progress_pairs,
            episodes=args.episodes,
            cards=args.cards,
            empty_per_kind=args.empty_per_kind,
            min_check_positive=args.min_check,
            min_nudge_idle=args.min_nudge,
            min_narration_idle=args.min_narration,
            min_failure_check=args.min_failure_check,
        )
        manifest = build_demo4_artifacts(
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
    except (Demo4BuildError, ValueError) as exc:
        parser.exit(1, f"Demo 4 build failed: {exc}\n")
    counts = manifest["row_counts"]
    source = manifest["source"]["counts"]
    print(
        f"Demo 4: {counts['total']} rows "
        f"(train={counts['train']} in {len(manifest['files']['train']['shards'])} "
        f"shards, dev={counts['dev']})"
    )
    print(
        f"Source: {source['batches']} batches, {source['requests']} requests, "
        f"{source['progress_pairs']} progress pairs ({source['checks']} check / "
        f"{source['nudges']} nudge)"
    )
    print(f"Manifest: {manifest['artifacts']['manifest']}")


if __name__ == "__main__":
    main()
