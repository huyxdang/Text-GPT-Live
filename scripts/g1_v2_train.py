"""Run the one-epoch g1-v2 continuation and rigorous before/after evals."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from train.g1_evaluation import evaluate_g1_predictions
from train.g1_v2_evaluation import evaluate_g1_v2
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
MANIFEST_PATH = ROOT / "artifacts" / "g1-v2" / "manifest.json"
STATE_PATH = ROOT / "data" / "tinker" / "run_state.json"
ORIGINAL_DEV = ROOT / "data" / "dev_g1.jsonl"
OUT = ROOT / "data" / "tinker"
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
SEED = 702
RETAINED_DEMO_FLOORS = {
    "demo-1": {"kind_accuracy": 0.98, "strict_accuracy": 0.80},
    "demo-2": {"kind_accuracy": 0.98, "strict_accuracy": 0.85},
    "demo-4": {"kind_accuracy": 0.98, "strict_accuracy": 0.70},
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _demo_1_to_4(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["rows"]
    result: dict[str, Any] = {}
    for demo in ("demo-1", "demo-2", "demo-3", "demo-4"):
        selected = [row for row in rows if row["demo"] == demo]
        result[demo] = {
            "support": len(selected),
            "kind_accuracy": round(sum(row["kind_match"] for row in selected) / len(selected), 6),
            "strict_accuracy": round(sum(row["row_pass"] for row in selected) / len(selected), 6),
        }
    return result


def _promotion_gate(
    candidate_report: dict[str, Any],
    original_demos: dict[str, Any],
) -> dict[str, Any]:
    retained = {
        demo: {
            metric: original_demos[demo][metric] >= floor
            for metric, floor in floors.items()
        }
        for demo, floors in RETAINED_DEMO_FLOORS.items()
    }
    checks = {
        "candidate_gates": bool(candidate_report["gates"]["passed"]),
        "retained_demo_floors": all(
            passed
            for demo_checks in retained.values()
            for passed in demo_checks.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "retained_demo_floors": RETAINED_DEMO_FLOORS,
        "retained_demo_results": retained,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    train_path = ROOT / manifest["train"]["path"]
    dev_path = ROOT / manifest["dev"]["path"]
    if _sha(train_path) != manifest["train"]["sha256"] or _sha(dev_path) != manifest["dev"]["sha256"]:
        raise SystemExit("g1-v2 dataset fingerprints do not match the manifest")
    train_rows = _load_jsonl(train_path)
    dev_rows = _load_jsonl(dev_path)
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    source_state = state["g1:state_epoch1"]
    source_sampler = state["g1:sampler_path"]
    base_model = state["g1:base_model"]
    if state.get("g1-v2:candidate_sampler_path") and not args.force:
        raise SystemExit("g1-v2 candidate already exists; pass --force only for an intentional rerun")

    tokenizer = get_tokenizer(base_model)
    token_counts = [g1_target_token_count(tokenizer, row) for row in train_rows]
    if max(token_counts) > TINKER_MAX_TARGET_TOKENS:
        raise SystemExit("g1-v2 contains an over-limit training row")
    batches = (len(train_rows) + BATCH_SIZE - 1) // BATCH_SIZE
    preflight = {
        "source_state": source_state,
        "source_sampler": source_sampler,
        "rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "batches": batches,
        "epochs": 1,
        "learning_rate": LEARNING_RATE,
        "max_target_tokens": max(token_counts),
        "total_target_tokens": sum(token_counts),
        "manifest_sha256": _sha(MANIFEST_PATH),
    }
    print(json.dumps(preflight, indent=2), flush=True)
    _write(ROOT / "artifacts" / "g1-v2" / "preflight.json", preflight)
    if args.dry_run:
        return

    load_env_key()
    import tinker

    service = tinker.ServiceClient(user_metadata={"project": "smol-g1-v2"})
    baseline_sampler = service.create_sampling_client(model_path=source_sampler)
    baseline_outputs = sample_outputs(
        baseline_sampler, tokenizer, dev_rows, label="g1-v2-baseline", seed=SEED
    )
    baseline_report = evaluate_g1_v2(dev_rows, baseline_outputs, label="g1-v2-baseline")
    _write(OUT / "dev-g1-v2-baseline_eval.json", baseline_report)
    print(json.dumps(baseline_report["summary"], indent=2), flush=True)

    client = service.create_training_client_from_state_with_optimizer(
        path=source_state,
        user_metadata={"project": "smol-g1-v2"},
    )
    adam = tinker.AdamParams(
        learning_rate=LEARNING_RATE,
        beta1=G1_ADAM_BETA1,
        beta2=G1_ADAM_BETA2,
        eps=G1_ADAM_EPS,
        weight_decay=G1_WEIGHT_DECAY,
        grad_clip_norm=G1_GRAD_CLIP_NORM,
    )
    shuffled = list(train_rows)
    random.Random(SEED).shuffle(shuffled)
    losses: list[dict[str, Any]] = []
    started = time.monotonic()
    for start in range(0, len(shuffled), BATCH_SIZE):
        batch = shuffled[start : start + BATCH_SIZE]
        step = start // BATCH_SIZE + 1
        step_started = time.monotonic()
        data = [build_datum(tinker, tokenizer, row) for row in batch]
        fb_future = client.forward_backward(data, "cross_entropy")
        optim_future = client.optim_step(adam)
        fb = fb_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
        optim_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
        metrics = getattr(fb, "metrics", None) or {}
        total = metrics.get("loss:sum") if isinstance(metrics, dict) else None
        weight = metrics.get("loss_fn_output_weight:sum") if isinstance(metrics, dict) else None
        record = {
            "step": step,
            "batch": len(batch),
            "mean_loss": total / weight if total is not None and weight else None,
            "seconds": round(time.monotonic() - step_started, 3),
        }
        losses.append(record)
        print(f"[g1-v2] {json.dumps(record, sort_keys=True)}", flush=True)

    state_result = client.save_state(name="smol-g1-v2-epoch1").result(timeout=REMOTE_TIMEOUT_SECONDS)
    sampler_result = client.save_weights_for_sampler(name="smol-g1-v2-epoch1").result(timeout=REMOTE_TIMEOUT_SECONDS)
    candidate_state = str(getattr(state_result, "path", state_result))
    candidate_sampler = str(getattr(sampler_result, "path", sampler_result))
    save_state_json(
        {
            "g1-v2:source_state": source_state,
            "g1-v2:candidate_state": candidate_state,
            "g1-v2:candidate_sampler_path": candidate_sampler,
            "g1-v2:learning_rate": LEARNING_RATE,
            "g1-v2:steps": len(losses),
            "g1-v2:train_rows": len(train_rows),
        }
    )

    sampler = service.create_sampling_client(model_path=candidate_sampler)
    candidate_outputs = sample_outputs(
        sampler, tokenizer, dev_rows, label="g1-v2-candidate", seed=SEED
    )
    candidate_report = evaluate_g1_v2(dev_rows, candidate_outputs, label="g1-v2-candidate")
    candidate_report["training"] = {
        "seconds": round(time.monotonic() - started, 3),
        "losses": losses,
        "candidate_state": candidate_state,
        "candidate_sampler": candidate_sampler,
    }
    _write(OUT / "dev-g1-v2-candidate_eval.json", candidate_report)
    print(json.dumps(candidate_report["summary"], indent=2), flush=True)

    original_pairs = _load_jsonl(ORIGINAL_DEV)
    original_outputs = sample_outputs(
        sampler, tokenizer, original_pairs, label="g1-v2-original-dev", seed=SEED + 1
    )
    original_report = evaluate_g1_predictions(
        original_pairs, original_outputs, label="g1-v2-original-dev"
    )
    original_report["demo_1_to_4"] = _demo_1_to_4(original_report)
    _write(OUT / "dev-g1-v2-original-dev_eval.json", original_report)
    print(json.dumps(original_report["demo_1_to_4"], indent=2), flush=True)

    comparison = {
        "baseline": baseline_report["summary"],
        "candidate": candidate_report["summary"],
        "candidate_gates": candidate_report["gates"],
        "original_dev_demo_1_to_4": original_report["demo_1_to_4"],
    }
    promotion = _promotion_gate(candidate_report, original_report["demo_1_to_4"])
    comparison["promotion"] = promotion
    _write(ROOT / "artifacts" / "g1-v2" / "training-comparison.json", comparison)
    if not promotion["passed"]:
        raise SystemExit("g1-v2 candidate failed automated promotion gates")
    save_state_json(
        {
            "g1-v2:selected_state": candidate_state,
            "g1-v2:selected_sampler_path": candidate_sampler,
        }
    )


if __name__ == "__main__":
    main()
