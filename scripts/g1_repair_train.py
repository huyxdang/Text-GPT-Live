"""Run one five-step g1 Demo 5 micro-SFT repair and staged evaluation.

The paid path fails closed: it verifies the exact 80-card corpus and tokenizer
limits locally, restores the epoch-1 optimizer state, performs five updates,
then runs a 65-case focused gate. The full 675-case dev evaluation runs only
when every focused restraint/obligation gate passes.

    .venv/bin/python -m scripts.g1_repair_train --dry-run
    .venv/bin/python -m scripts.g1_repair_train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.g1_causal_probe import build_probe_cases
from train.g1_evaluation import evaluate_g1_predictions
from train.g1_repair import pack_repair_batches, score_focused_repair
from train.tinker_run import (
    G1_ADAM_BETA1,
    G1_ADAM_BETA2,
    G1_ADAM_EPS,
    G1_GRAD_CLIP_NORM,
    G1_WEIGHT_DECAY,
    REMOTE_TIMEOUT_SECONDS,
    TINKER_MAX_TARGET_TOKENS,
    build_datum,
    get_tokenizer,
    g1_target_token_count,
    load_env_key,
    sample_outputs,
    save_state_json,
)


ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "data" / "tinker" / "run_state.json"
REPAIR_PATH = ROOT / "data" / "tinker" / "g1-repair-train.jsonl"
REPAIR_MANIFEST = ROOT / "artifacts" / "g1-repair" / "manifest.json"
DEV_PATH = ROOT / "data" / "dev_g1.jsonl"
BASELINE_REPORT = ROOT / "data" / "tinker" / "dev-g1-epoch1_eval.json"
FOCUSED_REPORT = ROOT / "data" / "tinker" / "dev-g1-repair-focused_eval.json"
FULL_REPORT = ROOT / "data" / "tinker" / "dev-g1-repair-full_eval.json"
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
STEPS = 5
SEED = 651


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_focused_cases(dev_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    causal, _ = build_probe_cases(dev_rows)
    cases = [
        {**dict(case), "focus_group": "fire" if case["role"] == "fire" else "matched_wait"}
        for case in causal
    ]
    silence = [
        row
        for row in dev_rows
        if row.get("current_content_empty") is True and row.get("obligation") == "none"
    ]
    if len(silence) != 25:
        raise ValueError(f"Expected 25 frozen ordinary-silence rows; found {len(silence)}.")
    cases.extend({**dict(row), "focus_group": "silence"} for row in silence)
    reserved_prompts = {str(case["prompt"]) for case in cases}

    restraint_pool = [
        row
        for row in dev_rows
        if row.get("demo") == "demo-5"
        and row.get("reminder_eval_kind") == "wait"
        and str(row["prompt"]) not in reserved_prompts
    ]
    restraint_pool.sort(key=lambda row: hashlib.sha256(str(row["candidate_id"]).encode()).digest())
    restraint: list[Mapping[str, Any]] = []
    used: set[str] = set()
    for row in restraint_pool:
        if str(row["episode"]) in used:
            continue
        restraint.append(row)
        used.add(str(row["episode"]))
        if len(restraint) == 8:
            break
    if len(restraint) != 8:
        raise ValueError("Expected eight unique-schedule reminder-restraint rows.")
    cases.extend({**dict(row), "focus_group": "reminder_restraint"} for row in restraint)

    collisions = [
        row
        for row in dev_rows
        if row.get("demo") == "demo-5" and row.get("situation") == "address-positive"
    ]
    if len(collisions) != 4:
        raise ValueError(f"Expected four Demo 5 collision rows; found {len(collisions)}.")
    cases.extend({**dict(row), "focus_group": "collision"} for row in collisions)

    prompts = [str(case["prompt"]) for case in cases]
    if len(cases) != 65 or len(set(prompts)) != len(prompts):
        raise ValueError("Focused repair suite must contain 65 unique prompts.")
    return cases


def _full_comparison(
    pairs: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_rows = baseline["rows"]
    candidate_rows = candidate["rows"]
    if len(pairs) != len(baseline_rows) or len(pairs) != len(candidate_rows):
        raise ValueError("Full comparison inputs do not align.")

    def demo_kind(rows: Sequence[Mapping[str, Any]], demo: str) -> float:
        selected = [row for row in rows if row.get("demo") == demo]
        return sum(bool(row["kind_match"]) for row in selected) / len(selected)

    demos: dict[str, Any] = {}
    for demo in ("demo-1", "demo-2", "demo-3", "demo-4"):
        old = demo_kind(baseline_rows, demo)
        new = demo_kind(candidate_rows, demo)
        demos[demo] = {
            "baseline_kind_accuracy": round(old, 6),
            "candidate_kind_accuracy": round(new, 6),
            "delta": round(new - old, 6),
            "passed": new >= old - 0.02,
        }

    recurring_indexes = [
        index
        for index, pair in enumerate(pairs)
        if pair.get("should_fire") and pair.get("schedule_kind") == "every"
    ]
    recurring_exact = sum(candidate_rows[index]["row_pass"] for index in recurring_indexes)
    summary = candidate["summary"]
    baseline_strict = float(baseline["summary"]["strict_row_accuracy"])
    candidate_strict = float(summary["strict_row_accuracy"])
    gates = {
        "format_validity": summary["format_validity"] == 1.0,
        "canonical_exact_rate": summary["canonical_exact_rate"] == 1.0,
        "recurring_fire": recurring_exact >= 13,
        "reminder_wait": float(summary["reminder_wait_accuracy"]) >= 0.90,
        "ordinary_silence": summary["ordinary_silence_idle_accuracy"] == 1.0,
        "strict_regression": candidate_strict >= baseline_strict - 0.02,
        "demo_1_to_4_regression": all(value["passed"] for value in demos.values()),
    }
    comparison = {
        "recurring_fire_support": len(recurring_indexes),
        "recurring_fire_exact": recurring_exact,
        "baseline_strict": baseline_strict,
        "candidate_strict": candidate_strict,
        "strict_delta": round(candidate_strict - baseline_strict, 6),
        "demos": demos,
        "gates": gates,
        "deterministic_gates_passed": all(gates.values()),
    }
    return finalize_repair_comparison(
        comparison,
        semantic_regression_status="pending_separate_review",
    )


def finalize_repair_comparison(
    comparison: Mapping[str, Any],
    *,
    semantic_regression_status: str,
) -> dict[str, Any]:
    """Fail closed until deterministic gates and semantic review both pass."""

    if semantic_regression_status not in {"pending_separate_review", "passed", "failed"}:
        raise ValueError(f"Unsupported semantic regression status: {semantic_regression_status}")
    result = dict(comparison)
    deterministic_passed = bool(result.get("deterministic_gates_passed", False))
    semantic_passed = semantic_regression_status == "passed"
    result.update(
        {
            "semantic_regression_status": semantic_regression_status,
            "semantic_regression_passed": semantic_passed,
            "passed": deterministic_passed and semantic_passed,
            "promotion_eligible": deterministic_passed and semantic_passed,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repair_rows = _load_jsonl(REPAIR_PATH)
    manifest = _load_json(REPAIR_MANIFEST)
    dev_rows = _load_jsonl(DEV_PATH)
    if len(repair_rows) != 80 or manifest.get("rows") != 80:
        raise SystemExit("Repair corpus must contain exactly 80 rows.")
    if _fingerprint(REPAIR_PATH) != manifest.get("sha256"):
        raise SystemExit("Repair corpus SHA does not match its manifest.")
    batches = pack_repair_batches(repair_rows)
    focused_cases = build_focused_cases(dev_rows)

    state = _load_json(STATE_PATH)
    source_state = state.get("g1:state_epoch1")
    base_model = state.get("g1:base_model")
    if not isinstance(source_state, str) or not source_state:
        raise SystemExit("No g1 epoch-1 optimizer state is recorded.")
    if not isinstance(base_model, str) or not base_model:
        raise SystemExit("No g1 base model is recorded.")
    if state.get("g1-repair:candidate_sampler_path") and not args.force:
        raise SystemExit("A g1 repair candidate already exists; use --force only for an intentional rerun.")

    tokenizer = get_tokenizer(base_model)
    counts = [g1_target_token_count(tokenizer, row) for row in repair_rows]
    if max(counts) > TINKER_MAX_TARGET_TOKENS:
        raise SystemExit("Repair corpus exceeds Tinker's context limit.")
    preflight = {
        "rows": len(repair_rows),
        "batches": len(batches),
        "batch_sizes": [len(batch) for batch in batches],
        "learning_rate": LEARNING_RATE,
        "steps": STEPS,
        "source_state": source_state,
        "corpus_sha256": manifest["sha256"],
        "target_tokens": {
            "total": sum(counts),
            "max": max(counts),
            "per_batch": [
                sum(g1_target_token_count(tokenizer, row) for row in batch) for batch in batches
            ],
        },
        "focused_cases": len(focused_cases),
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if args.dry_run:
        return

    load_env_key()
    import tinker

    service = tinker.ServiceClient(user_metadata={"project": "smol-g1-repair"})
    client = service.create_training_client_from_state_with_optimizer(
        path=source_state,
        user_metadata={"project": "smol-g1-repair"},
    )
    adam = tinker.AdamParams(
        learning_rate=LEARNING_RATE,
        beta1=G1_ADAM_BETA1,
        beta2=G1_ADAM_BETA2,
        eps=G1_ADAM_EPS,
        weight_decay=G1_WEIGHT_DECAY,
        grad_clip_norm=G1_GRAD_CLIP_NORM,
    )
    losses: list[dict[str, Any]] = []
    started = time.monotonic()
    for step, batch in enumerate(batches, start=1):
        data = [build_datum(tinker, tokenizer, row) for row in batch]
        step_started = time.monotonic()
        fb_future = client.forward_backward(data, "cross_entropy")
        optim_future = client.optim_step(adam)
        fb_result = fb_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
        optim_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
        metrics = getattr(fb_result, "metrics", None) or {}
        loss = metrics.get("loss:sum") if isinstance(metrics, dict) else None
        weight = metrics.get("loss_fn_output_weight:sum") if isinstance(metrics, dict) else None
        mean_loss = loss / weight if loss is not None and weight else None
        row = {
            "step": step,
            "batch": len(batch),
            "mean_loss": mean_loss,
            "seconds": round(time.monotonic() - step_started, 3),
        }
        losses.append(row)
        print(f"[repair] {json.dumps(row, sort_keys=True)}", flush=True)

    state_result = client.save_state(name="smol-g1-repair-5step").result(
        timeout=REMOTE_TIMEOUT_SECONDS
    )
    sampler_result = client.save_weights_for_sampler(name="smol-g1-repair-5step").result(
        timeout=REMOTE_TIMEOUT_SECONDS
    )
    candidate_state = str(getattr(state_result, "path", state_result))
    candidate_sampler_path = str(getattr(sampler_result, "path", sampler_result))
    save_state_json(
        {
            "g1-repair:source_state": source_state,
            "g1-repair:corpus_sha256": manifest["sha256"],
            "g1-repair:rows": 80,
            "g1-repair:steps": STEPS,
            "g1-repair:learning_rate": LEARNING_RATE,
            "g1-repair:candidate_state": candidate_state,
            "g1-repair:candidate_sampler_path": candidate_sampler_path,
        }
    )
    sampler = service.create_sampling_client(model_path=candidate_sampler_path)

    focused_outputs = sample_outputs(
        sampler,
        tokenizer,
        focused_cases,
        label="g1-repair-focused",
        seed=SEED,
    )
    focused = score_focused_repair(focused_cases, focused_outputs)
    focused["provenance"] = {
        **preflight,
        "candidate_state": candidate_state,
        "candidate_sampler_path": candidate_sampler_path,
        "losses": losses,
        "wall_seconds_through_focused": round(time.monotonic() - started, 3),
    }
    _write_json(FOCUSED_REPORT, focused)
    print(f"[repair] focused={json.dumps(focused['summary'], sort_keys=True)}", flush=True)

    accepted = False
    deterministic_gates_passed = False
    if focused["summary"]["passed"]:
        full_outputs = sample_outputs(
            sampler,
            tokenizer,
            dev_rows,
            label="g1-repair-full",
            seed=SEED,
        )
        full = evaluate_g1_predictions(dev_rows, full_outputs, label="dev-g1-repair-full")
        full["comparison"] = _full_comparison(
            dev_rows,
            _load_json(BASELINE_REPORT),
            full,
        )
        full["provenance"] = {
            **focused["provenance"],
            "focused_report": str(FOCUSED_REPORT),
            "wall_seconds_total": round(time.monotonic() - started, 3),
        }
        deterministic_gates_passed = bool(full["comparison"]["deterministic_gates_passed"])
        accepted = bool(full["comparison"]["passed"])
        _write_json(FULL_REPORT, full)
        print(f"[repair] full={json.dumps(full['comparison'], sort_keys=True)}", flush=True)
    else:
        print("[repair] focused gate failed; full 675-case eval skipped.", flush=True)

    save_state_json(
        {
            "g1-repair:focused_passed": bool(focused["summary"]["passed"]),
            "g1-repair:full_ran": bool(focused["summary"]["passed"]),
            "g1-repair:deterministic_gates_passed": deterministic_gates_passed,
            "g1-repair:semantic_regression_status": (
                full["comparison"]["semantic_regression_status"]
                if focused["summary"]["passed"]
                else "not_run"
            ),
            "g1-repair:accepted": accepted,
            "g1-repair:accepted_deterministic": False,
        }
    )
    print(
        f"[repair] candidate={candidate_sampler_path} accepted={accepted}",
        flush=True,
    )


if __name__ == "__main__":
    main()
