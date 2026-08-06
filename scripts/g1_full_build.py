"""Build one globally de-duplicated five-demo g1 train/dev dataset.

Each demo keeps its own source validator and compiler.  This command runs all
five of them in an isolated staging directory, then applies the checks that
only make sense after the demos are merged: prompt/answer consistency, exact
pair de-duplication, whole-episode splits, and zero prompt overlap between
train and dev.

    .venv/bin/python -m scripts.g1_full_build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from app.stream import g1_action_completion, parse_g1_action
from datagen.g1_demo1 import DEFAULT_DEMO1_PROFILE, DEMO1_TARGET_PROFILES
from datagen.g1_pilot import action_label
from scripts.g1_demo1_build import (
    _existing_train_shard_paths,
    _json,
    _jsonl_payload,
    _path_label,
    _sha256,
    _stage_and_publish,
    _train_shard_payloads,
    build_demo1_artifacts,
)
from scripts.g1_demo2_build import build_demo2_artifacts
from scripts.g1_demo3_build import build_demo3_artifacts
from scripts.g1_demo4_build import build_demo4_artifacts
from scripts.g1_demo5_build import build_demo5_artifacts
from train.tinker_run import (
    G1_BASE_MODEL,
    TINKER_MAX_TARGET_TOKENS,
    g1_target_token_count,
    get_tokenizer,
)


ROOT = Path(__file__).resolve().parent.parent
BUILD_SCHEMA_VERSION = "g1-full-build-1"
DATASET_SCHEMA_VERSION = "g1"
DEMOS = ("demo-1", "demo-2", "demo-3", "demo-4", "demo-5")
DEFAULT_AUTHORED_ROOT = ROOT / "data" / "g1_authored"
DEFAULT_TRAIN_BASE_PATH = ROOT / "data" / "train_g1.jsonl"
DEFAULT_DEV_PATH = ROOT / "data" / "dev_g1.jsonl"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "g1-full"
DEFAULT_TRAIN_SHARDS = 4
MAX_TRAIN_SHARD_BYTES = 90_000_000


class G1FullBuildError(RuntimeError):
    """Raised before publication when the merged build is not safe to train."""


def _stable_digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise G1FullBuildError(
                        f"Invalid JSON in {path} at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise G1FullBuildError(
                        f"{path} line {line_number} must contain a JSON object"
                    )
                rows.append(value)
    except OSError as exc:
        raise G1FullBuildError(f"Cannot read {path}: {exc}") from exc
    return rows


def _rows_from_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise G1FullBuildError("Per-demo manifest is missing files metadata")
    train = files.get("train")
    dev = files.get("dev")
    if not isinstance(train, Mapping) or not isinstance(dev, Mapping):
        raise G1FullBuildError("Per-demo manifest must describe train and dev")
    shards = train.get("shards")
    if not isinstance(shards, list) or not shards:
        raise G1FullBuildError("Per-demo train manifest must contain shards")
    paths: list[Path] = []
    for index, entry in enumerate(shards):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise G1FullBuildError(f"Invalid per-demo train shard entry {index}")
        paths.append(Path(str(entry["path"])))
    if not isinstance(dev.get("path"), str):
        raise G1FullBuildError("Invalid per-demo dev path")
    paths.append(Path(str(dev["path"])))
    return [row for path in paths for row in _load_jsonl(path)]


def _representative(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Choose one stable row without moving it away from its episode split."""

    prompt = str(rows[0]["prompt"])
    desired_split = "dev" if int(_stable_digest(prompt)[:8], 16) % 10 == 0 else "train"
    candidates = [row for row in rows if row.get("split") == desired_split] or list(rows)
    return min(
        candidates,
        key=lambda row: (
            str(row.get("demo", "")),
            str(row.get("episode", "")),
            str(row.get("candidate_id", "")),
        ),
    )


def merge_demo_rows(
    rows_by_demo: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and globally de-duplicate compiled rows from all five demos."""

    if set(rows_by_demo) != set(DEMOS):
        missing = sorted(set(DEMOS) - set(rows_by_demo))
        extra = sorted(set(rows_by_demo) - set(DEMOS))
        raise G1FullBuildError(f"Expected exactly demos 1-5; missing={missing}, extra={extra}")

    candidate_ids: set[str] = set()
    prompt_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    episode_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    input_counts: Counter[str] = Counter()
    input_action_counts: Counter[str] = Counter()

    for demo in DEMOS:
        for row_index, row in enumerate(rows_by_demo[demo]):
            location = f"{demo} row {row_index}"
            if row.get("schema_version") != DATASET_SCHEMA_VERSION:
                raise G1FullBuildError(f"{location}: schema_version must be 'g1'")
            if row.get("demo") != demo:
                raise G1FullBuildError(f"{location}: row demo does not match its source")
            split = row.get("split")
            if split not in {"train", "dev"}:
                raise G1FullBuildError(f"{location}: invalid split {split!r}")
            candidate_id = row.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise G1FullBuildError(f"{location}: candidate_id must be non-blank")
            if candidate_id in candidate_ids:
                raise G1FullBuildError(f"Duplicate candidate_id across demos: {candidate_id}")
            candidate_ids.add(candidate_id)
            prompt = row.get("prompt")
            completion = row.get("completion")
            if not isinstance(prompt, str) or not prompt:
                raise G1FullBuildError(f"{location}: prompt must be non-blank")
            if not isinstance(completion, str):
                raise G1FullBuildError(f"{location}: completion must be text")
            action = parse_g1_action(completion)
            if not action.valid or g1_action_completion(action) != completion:
                raise G1FullBuildError(
                    f"{location}: completion is not one canonical g1 action: {action.diagnostic}"
                )
            expected_class = action_label(action)
            if row.get("expected_class") != expected_class:
                raise G1FullBuildError(
                    f"{location}: expected_class disagrees with completion"
                )
            episode = row.get("episode")
            if not isinstance(episode, str) or not episode:
                raise G1FullBuildError(f"{location}: episode must be non-blank")
            episode_splits[(demo, episode)].add(str(split))
            prompt_groups[prompt].append(row)
            input_counts[demo] += 1
            input_action_counts[expected_class] += 1

    leaking_episodes = [key for key, splits in episode_splits.items() if len(splits) != 1]
    if leaking_episodes:
        raise G1FullBuildError(
            f"Whole-episode split invariant failed for {len(leaking_episodes)} episodes"
        )

    merged: list[dict[str, Any]] = []
    duplicate_groups = 0
    duplicates_removed = 0
    cross_split_groups = 0
    cross_demo_groups = 0
    for prompt, group in prompt_groups.items():
        completions = {str(row["completion"]) for row in group}
        if len(completions) != 1:
            diagnostics = sorted(
                (str(row.get("demo")), str(row.get("candidate_id")), str(row["completion"]))
                for row in group
            )
            raise G1FullBuildError(
                "Conflicting completions for one exact prompt: " + repr(diagnostics)
            )
        if len(group) > 1:
            duplicate_groups += 1
            duplicates_removed += len(group) - 1
            cross_split_groups += len({row["split"] for row in group}) > 1
            cross_demo_groups += len({row["demo"] for row in group}) > 1
        merged.append(dict(_representative(group)))

    merged.sort(
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            str(row["demo"]),
            str(row["episode"]),
            str(row["candidate_id"]),
        )
    )
    train_prompts = {row["prompt"] for row in merged if row["split"] == "train"}
    dev_prompts = {row["prompt"] for row in merged if row["split"] == "dev"}
    overlap = train_prompts & dev_prompts
    if overlap:
        raise G1FullBuildError(f"Global train/dev prompt overlap remains: {len(overlap)}")

    output_counts = Counter(str(row["demo"]) for row in merged)
    output_action_counts = Counter(str(row["expected_class"]) for row in merged)
    split_counts = Counter(str(row["split"]) for row in merged)
    integrity = {
        "input_rows": sum(input_counts.values()),
        "output_rows": len(merged),
        "unique_prompts": len(prompt_groups),
        "duplicate_prompt_groups": duplicate_groups,
        "exact_pair_duplicates_removed": duplicates_removed,
        "cross_split_duplicate_groups_resolved": cross_split_groups,
        "cross_demo_duplicate_groups": cross_demo_groups,
        "conflicting_prompt_groups": 0,
        "train_dev_prompt_overlap": 0,
        "episode_split_leaks": 0,
        "candidate_ids_unique": len(candidate_ids),
        "input_by_demo": dict(sorted(input_counts.items())),
        "output_by_demo": dict(sorted(output_counts.items())),
        "removed_by_demo": {
            demo: input_counts[demo] - output_counts[demo] for demo in DEMOS
        },
        "input_action_distribution": dict(sorted(input_action_counts.items())),
        "output_action_distribution": dict(sorted(output_action_counts.items())),
        "output_splits": dict(sorted(split_counts.items())),
    }
    return merged, integrity


def _manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        key: manifest[key]
        for key in (
            "schema_version",
            "dataset_schema",
            "source",
            "targets",
            "source_warnings",
            "exact_target_evidence",
            "row_counts",
            "row_validation",
        )
        if key in manifest
    }
    if "distribution_gates_enforced" in manifest:
        summary["distribution_gates_enforced"] = manifest["distribution_gates_enforced"]
    if "chinese_reference_review" in manifest:
        summary["chinese_reference_review"] = manifest["chinese_reference_review"]
    return summary


def _g1_eval_support(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    support = {
        "should_fire": sum(bool(row.get("should_fire")) for row in rows),
        "reminder_wait": sum(
            row.get("demo") == "demo-5" and row.get("reminder_eval_kind") == "wait"
            for row in rows
        ),
        "ordinary_silence": sum(
            bool(row.get("current_content_empty")) and row.get("obligation") == "none"
            for row in rows
        ),
        "clause_boundary": sum(
            row.get("clause_state") in {"partial", "complete"} for row in rows
        ),
    }
    missing = sorted(name for name, count in support.items() if count == 0)
    if missing:
        raise G1FullBuildError(
            "Published dataset would leave g1 evaluation slices empty: "
            + ", ".join(missing)
        )
    return support


def _apply_tinker_context_filter(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_token_counter: Callable[[Mapping[str, Any]], int],
    limit: int,
    base_model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove examples Tinker cannot accept and leave an auditable record."""

    if limit <= 0:
        raise G1FullBuildError("Tinker target-token limit must be positive")
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    lengths: list[int] = []
    kept_lengths: list[int] = []
    for row in rows:
        count = target_token_counter(row)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise G1FullBuildError(
                f"Invalid target-token count for {row.get('candidate_id')!r}: {count!r}"
            )
        lengths.append(count)
        if count > limit:
            removed.append(
                {
                    "candidate_id": row.get("candidate_id"),
                    "demo": row.get("demo"),
                    "episode": row.get("episode"),
                    "role": row.get("candidate_role", row.get("situation")),
                    "split": row.get("split"),
                    "target_tokens": count,
                }
            )
        else:
            kept.append(dict(row))
            kept_lengths.append(count)
    report = {
        "base_model": base_model,
        "limit": limit,
        "operator": "<=",
        "rows_checked": len(rows),
        "rows_removed": len(removed),
        "max_before_filter": max(lengths, default=0),
        "max_after_filter": max(kept_lengths, default=0),
        "removed_by_split": dict(
            sorted(Counter(str(row["split"]) for row in removed).items())
        ),
        "removed_by_demo": dict(
            sorted(Counter(str(row["demo"]) for row in removed).items())
        ),
        "removed_rows": removed,
        "passed": all(length <= limit for length in kept_lengths),
    }
    if not kept or not report["passed"]:
        raise G1FullBuildError("Tinker context filtering did not produce a safe dataset")
    return kept, report


def _inspection_markdown(rows: Sequence[Mapping[str, Any]], integrity: Mapping[str, Any]) -> str:
    lines = [
        "# g1 full-dataset inspection samples",
        "",
        f"- Input rows: {integrity['input_rows']}",
        f"- Published rows: {integrity['output_rows']}",
        f"- Exact duplicates removed: {integrity['exact_pair_duplicates_removed']}",
        "- Train/dev prompt overlap: 0",
        "",
    ]
    for demo in DEMOS:
        lines.extend((f"## {demo}", ""))
        demo_rows = [row for row in rows if row["demo"] == demo]
        by_action: dict[str, Mapping[str, Any]] = {}
        for row in demo_rows:
            by_action.setdefault(str(row["expected_class"]), row)
        for expected_class, row in sorted(by_action.items()):
            lines.extend(
                (
                    f"### {expected_class}",
                    "",
                    f"- candidate: `{row['candidate_id']}`",
                    f"- split: `{row['split']}`",
                    "",
                    "```text",
                    str(row["prompt"]),
                    "```",
                    "",
                    f"Gold: `{row['completion']}`",
                    "",
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def publish_full_artifacts(
    *,
    rows_by_demo: Mapping[str, Sequence[Mapping[str, Any]]],
    demo_manifests: Mapping[str, Mapping[str, Any]],
    train_base_path: Path = DEFAULT_TRAIN_BASE_PATH,
    dev_path: Path = DEFAULT_DEV_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    minimum_train_shards: int = DEFAULT_TRAIN_SHARDS,
    max_train_shard_bytes: int = MAX_TRAIN_SHARD_BYTES,
    target_token_counter: Callable[[Mapping[str, Any]], int],
    tinker_target_token_limit: int = TINKER_MAX_TARGET_TOKENS,
    base_model: str = G1_BASE_MODEL,
) -> dict[str, Any]:
    rows, integrity = merge_demo_rows(rows_by_demo)
    rows_before_context_filter = len(rows)
    rows, context_gate = _apply_tinker_context_filter(
        rows,
        target_token_counter=target_token_counter,
        limit=tinker_target_token_limit,
        base_model=base_model,
    )
    integrity["rows_before_context_filter"] = rows_before_context_filter
    integrity["tinker_target_token_gate"] = context_gate
    integrity["output_rows"] = len(rows)
    output_by_demo = Counter(str(row["demo"]) for row in rows)
    integrity["output_by_demo"] = dict(sorted(output_by_demo.items()))
    integrity["removed_by_demo"] = {
        demo: int(integrity["input_by_demo"][demo]) - output_by_demo[demo]
        for demo in DEMOS
    }
    integrity["output_action_distribution"] = dict(
        sorted(Counter(str(row["expected_class"]) for row in rows).items())
    )
    integrity["output_splits"] = dict(
        sorted(Counter(str(row["split"]) for row in rows).items())
    )
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    integrity["g1_eval_support"] = _g1_eval_support(dev_rows)
    train_shards = _train_shard_payloads(
        train_rows,
        base_path=train_base_path,
        minimum_shards=minimum_train_shards,
        max_shard_bytes=max_train_shard_bytes,
    )
    dev_payload = _jsonl_payload(dev_rows)
    train_digest = hashlib.sha256()
    train_bytes = 0
    train_entries: list[dict[str, Any]] = []
    for path, shard_rows, payload, payload_bytes in train_shards:
        train_digest.update(payload.encode("utf-8"))
        train_bytes += payload_bytes
        train_entries.append(
            {
                "path": _path_label(path),
                "rows": len(shard_rows),
                "bytes": payload_bytes,
                "sha256": _sha256(payload),
            }
        )
    files = {
        "train": {
            "format": "ordered-jsonl-shards",
            "rows": len(train_rows),
            "bytes": train_bytes,
            "aggregate_sha256": train_digest.hexdigest(),
            "max_shard_bytes": max_train_shard_bytes,
            "shards": train_entries,
        },
        "dev": {
            "format": "jsonl",
            "path": _path_label(dev_path),
            "rows": len(dev_rows),
            "bytes": len(dev_payload.encode("utf-8")),
            "sha256": _sha256(dev_payload),
        },
    }
    demo_summaries = {
        demo: _manifest_summary(demo_manifests[demo]) for demo in DEMOS
    }
    chinese_review = demo_summaries["demo-3"].get("chinese_reference_review")
    manifest = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "dataset_schema": DATASET_SCHEMA_VERSION,
        "row_counts": {
            "input": integrity["input_rows"],
            "total": len(rows),
            "train": len(train_rows),
            "dev": len(dev_rows),
        },
        "global_integrity": integrity,
        "files": files,
        "demos": demo_summaries,
        "chinese_reference_review": chinese_review,
        "artifacts": {
            "manifest": _path_label(artifact_dir / "manifest.json"),
            "coverage": _path_label(artifact_dir / "coverage.json"),
            "inspection_samples": _path_label(artifact_dir / "inspection_samples.md"),
        },
    }
    coverage = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "dataset_schema": DATASET_SCHEMA_VERSION,
        "row_counts": manifest["row_counts"],
        "global_integrity": integrity,
        "files": files,
    }
    payloads = {
        **{path: payload for path, _, payload, _ in train_shards},
        dev_path: dev_payload,
        artifact_dir / "manifest.json": _json(manifest, indent=2) + "\n",
        artifact_dir / "coverage.json": _json(coverage, indent=2) + "\n",
        artifact_dir / "inspection_samples.md": _inspection_markdown(rows, integrity),
    }
    current_shards = {path for path, _, _, _ in train_shards}
    stale_shards = [
        path
        for path in _existing_train_shard_paths(train_base_path)
        if path not in current_shards
    ]
    _stage_and_publish(payloads, obsolete_paths=(train_base_path, *stale_shards))
    return manifest


def _build_individual_demos(
    *, authored_root: Path, temporary_root: Path, fail_on_warnings: bool
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Mapping[str, Any]]]:
    data_dir = temporary_root / "data"
    artifact_root = temporary_root / "artifacts"
    rows_by_demo: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, Mapping[str, Any]] = {}
    builders = {
        "demo-1": build_demo1_artifacts,
        "demo-2": build_demo2_artifacts,
        "demo-3": build_demo3_artifacts,
        "demo-4": build_demo4_artifacts,
        "demo-5": build_demo5_artifacts,
    }
    for demo, builder in builders.items():
        kwargs: dict[str, Any] = {
            "authored_root": authored_root,
            "train_base_path": data_dir / f"train_{demo}.jsonl",
            "dev_path": data_dir / f"dev_{demo}.jsonl",
            "artifact_dir": artifact_root / demo,
            "fail_on_warnings": fail_on_warnings,
        }
        if demo == "demo-1":
            kwargs["enforce_selection_distribution"] = True
            kwargs["targets"] = DEMO1_TARGET_PROFILES[DEFAULT_DEMO1_PROFILE]
        manifest = builder(**kwargs)
        manifests[demo] = manifest
        rows_by_demo[demo] = _rows_from_manifest(manifest)
    return rows_by_demo, manifests


def build_g1_full_artifacts(
    *,
    authored_root: Path = DEFAULT_AUTHORED_ROOT,
    train_base_path: Path = DEFAULT_TRAIN_BASE_PATH,
    dev_path: Path = DEFAULT_DEV_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    minimum_train_shards: int = DEFAULT_TRAIN_SHARDS,
    max_train_shard_bytes: int = MAX_TRAIN_SHARD_BYTES,
    fail_on_warnings: bool = False,
    base_model: str = G1_BASE_MODEL,
    tinker_target_token_limit: int = TINKER_MAX_TARGET_TOKENS,
) -> dict[str, Any]:
    tokenizer = get_tokenizer(base_model)
    with tempfile.TemporaryDirectory(prefix="g1-full-build-") as temporary:
        rows_by_demo, manifests = _build_individual_demos(
            authored_root=authored_root,
            temporary_root=Path(temporary),
            fail_on_warnings=fail_on_warnings,
        )
        return publish_full_artifacts(
            rows_by_demo=rows_by_demo,
            demo_manifests=manifests,
            train_base_path=train_base_path,
            dev_path=dev_path,
            artifact_dir=artifact_dir,
            minimum_train_shards=minimum_train_shards,
            max_train_shard_bytes=max_train_shard_bytes,
            target_token_counter=lambda row: g1_target_token_count(tokenizer, row),
            tinker_target_token_limit=tinker_target_token_limit,
            base_model=base_model,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authored-root", type=Path, default=DEFAULT_AUTHORED_ROOT)
    parser.add_argument("--train-base", type=Path, default=DEFAULT_TRAIN_BASE_PATH)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--train-shards", type=int, default=DEFAULT_TRAIN_SHARDS)
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Treat per-demo semantic warnings as release failures.",
    )
    args = parser.parse_args()
    try:
        manifest = build_g1_full_artifacts(
            authored_root=args.authored_root,
            train_base_path=args.train_base,
            dev_path=args.dev,
            artifact_dir=args.artifact_dir,
            minimum_train_shards=args.train_shards,
            fail_on_warnings=args.fail_on_warnings,
        )
    except (G1FullBuildError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"g1 full build failed: {exc}\n")
    counts = manifest["row_counts"]
    integrity = manifest["global_integrity"]
    print(
        f"g1 full: {counts['total']} rows from {counts['input']} inputs "
        f"(train={counts['train']}, dev={counts['dev']})"
    )
    print(
        f"Resolved {integrity['duplicate_prompt_groups']} duplicate prompt groups; "
        f"removed {integrity['exact_pair_duplicates_removed']} exact duplicates; "
        "train/dev overlap=0"
    )
    context_gate = integrity["tinker_target_token_gate"]
    print(
        f"Tinker limit: removed {context_gate['rows_removed']} rows above "
        f"{context_gate['limit']} target tokens; max published="
        f"{context_gate['max_after_filter']}"
    )
    print(f"Manifest: {manifest['artifacts']['manifest']}")


if __name__ == "__main__":
    main()
