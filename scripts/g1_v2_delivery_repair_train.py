"""Train and gate the focused foreground-dialogue/search-delivery repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from scripts.g1_v2_train import _demo_1_to_4, _promotion_gate
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
MANIFEST = ROOT / "artifacts" / "g1-v2" / "delivery-repair-manifest.json"
STATE = ROOT / "data" / "tinker" / "run_state.json"
FULL_DEV = ROOT / "data" / "g1_v2" / "dev.jsonl"
ORIGINAL_DEV = ROOT / "data" / "dev_g1.jsonl"
OUT = ROOT / "data" / "tinker"
BATCH = 16
LR = 1e-5
SEED = 704


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _role_kind(report: dict[str, Any], role: str) -> float:
    return float(report["roles"].get(role, {}).get("kind_accuracy") or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train_path = ROOT / manifest["train"]["path"]
    dev_path = ROOT / manifest["dev"]["path"]
    if _sha(train_path) != manifest["train"]["sha256"] or _sha(dev_path) != manifest["dev"]["sha256"]:
        raise SystemExit("Delivery-repair dataset fingerprints do not match the manifest")
    train_rows = _load(train_path)
    dev_rows = _load(dev_path)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("g1-v2-delivery:candidate_sampler_path") and not args.force:
        raise SystemExit("Delivery-repair candidate already exists; use --force only intentionally")
    source_state = str(manifest["source_state"])
    source_sampler = str(manifest["source_sampler"])
    if source_state == state.get("g1-v2-delivery:candidate_state"):
        raise SystemExit("Delivery repair source must be the immutable pre-repair checkpoint")
    tokenizer = get_tokenizer(state["g1:base_model"])
    counts = [g1_target_token_count(tokenizer, row) for row in train_rows]
    if max(counts) > TINKER_MAX_TARGET_TOKENS:
        raise SystemExit("Delivery-repair row exceeds the target-token limit")
    preflight = {
        "source_state": source_state,
        "rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "steps": (len(train_rows) + BATCH - 1) // BATCH,
        "epochs": 1,
        "learning_rate": LR,
        "max_target_tokens": max(counts),
    }
    _write(ROOT / "artifacts" / "g1-v2" / "delivery-repair-preflight.json", preflight)
    print(json.dumps(preflight, indent=2), flush=True)
    if args.dry_run:
        return

    load_env_key()
    import tinker

    service = tinker.ServiceClient(user_metadata={"project": "smol-g1-v2-delivery"})
    baseline_sampler = service.create_sampling_client(model_path=source_sampler)
    baseline_outputs = sample_outputs(
        baseline_sampler, tokenizer, dev_rows, label="g1-v2-delivery-baseline", seed=SEED
    )
    baseline_report = evaluate_g1_v2(
        dev_rows, baseline_outputs, label="g1-v2-delivery-baseline"
    )
    _write(OUT / "dev-g1-v2-delivery-baseline_eval.json", baseline_report)
    print(json.dumps(baseline_report["summary"], indent=2), flush=True)

    client = service.create_training_client_from_state_with_optimizer(
        path=source_state,
        user_metadata={"project": "smol-g1-v2-delivery"},
    )
    adam = tinker.AdamParams(
        learning_rate=LR,
        beta1=G1_ADAM_BETA1,
        beta2=G1_ADAM_BETA2,
        eps=G1_ADAM_EPS,
        weight_decay=G1_WEIGHT_DECAY,
        grad_clip_norm=G1_GRAD_CLIP_NORM,
    )
    shuffled = list(train_rows)
    random.Random(SEED).shuffle(shuffled)
    started = time.monotonic()
    for start in range(0, len(shuffled), BATCH):
        batch = shuffled[start : start + BATCH]
        step = start // BATCH + 1
        fb = client.forward_backward(
            [build_datum(tinker, tokenizer, row) for row in batch], "cross_entropy"
        )
        optim = client.optim_step(adam)
        fb.result(timeout=REMOTE_TIMEOUT_SECONDS)
        optim.result(timeout=REMOTE_TIMEOUT_SECONDS)
        print(f"[delivery] step={step}/{preflight['steps']} batch={len(batch)}", flush=True)

    state_result = client.save_state(name="smol-g1-v2-delivery-repair").result(
        timeout=REMOTE_TIMEOUT_SECONDS
    )
    sampler_result = client.save_weights_for_sampler(name="smol-g1-v2-delivery-repair").result(
        timeout=REMOTE_TIMEOUT_SECONDS
    )
    candidate_state = str(getattr(state_result, "path", state_result))
    candidate_sampler = str(getattr(sampler_result, "path", sampler_result))
    save_state_json(
        {
            "g1-v2-delivery:source_state": source_state,
            "g1-v2-delivery:candidate_state": candidate_state,
            "g1-v2-delivery:candidate_sampler_path": candidate_sampler,
            "g1-v2-delivery:steps": preflight["steps"],
            "g1-v2-delivery:learning_rate": LR,
        }
    )
    sampler = service.create_sampling_client(model_path=candidate_sampler)

    delivery_outputs = sample_outputs(
        sampler, tokenizer, dev_rows, label="g1-v2-delivery-candidate", seed=SEED
    )
    delivery_report = evaluate_g1_v2(
        dev_rows, delivery_outputs, label="g1-v2-delivery-candidate"
    )
    _write(OUT / "dev-g1-v2-delivery-candidate_eval.json", delivery_report)

    full_dev = _load(FULL_DEV)
    full_outputs = sample_outputs(
        sampler, tokenizer, full_dev, label="g1-v2-delivery-full-dev", seed=SEED + 1
    )
    full_report = evaluate_g1_v2(full_dev, full_outputs, label="g1-v2-delivery-full-dev")
    _write(OUT / "dev-g1-v2-delivery-full_eval.json", full_report)

    original = _load(ORIGINAL_DEV)
    original_outputs = sample_outputs(
        sampler, tokenizer, original, label="g1-v2-delivery-original", seed=SEED + 2
    )
    original_report = evaluate_g1_predictions(
        original, original_outputs, label="g1-v2-delivery-original"
    )
    original_report["demo_1_to_4"] = _demo_1_to_4(original_report)
    _write(OUT / "dev-g1-v2-delivery-original_eval.json", original_report)

    delivery_checks = {
        "overall_kind_accuracy": float(delivery_report["summary"]["kind_accuracy"] or 0) >= 0.98,
        "foreground_response": _role_kind(delivery_report, "foreground-response") == 1.0,
        "search_delivery": _role_kind(delivery_report, "search-completed") == 1.0,
        "delegate_before_idle": _role_kind(delivery_report, "delegate-completed-before-delivery") >= 0.95,
        "delegate_after_idle": _role_kind(delivery_report, "delegate-completed-after-delivery") >= 0.95,
        "delivered_idle": _role_kind(delivery_report, "delivered-idle") >= 0.95,
    }
    retained = _promotion_gate(full_report, original_report["demo_1_to_4"])
    promotion = {
        "passed": all(delivery_checks.values()) and retained["passed"],
        "delivery_checks": delivery_checks,
        "full_v2_gates": full_report["gates"],
        "retained_original": retained,
    }
    comparison = {
        "baseline_delivery": baseline_report["summary"],
        "candidate_delivery": delivery_report["summary"],
        "candidate_delivery_roles": delivery_report["roles"],
        "candidate_full_v2": full_report["summary"],
        "candidate_original_demo_1_to_4": original_report["demo_1_to_4"],
        "promotion": promotion,
        "training_seconds": round(time.monotonic() - started, 3),
    }
    _write(ROOT / "artifacts" / "g1-v2" / "delivery-repair-comparison.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)
    if not promotion["passed"]:
        raise SystemExit("Delivery-repair candidate failed automated promotion gates")
    save_state_json(
        {
            "g1-v2:selected_state": candidate_state,
            "g1-v2:selected_sampler_path": candidate_sampler,
            "g1-v2-delivery:selected": True,
        }
    )


if __name__ == "__main__":
    main()
