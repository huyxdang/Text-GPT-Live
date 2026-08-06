"""Run the cost-controlled recovery continuation and both regression evals."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from scripts.g1_v2_train import _demo_1_to_4
from train.g1_evaluation import evaluate_g1_predictions
from train.g1_v2_evaluation import evaluate_g1_v2
from train.tinker_run import (
    G1_ADAM_BETA1,
    G1_ADAM_BETA2,
    G1_ADAM_EPS,
    G1_GRAD_CLIP_NORM,
    G1_WEIGHT_DECAY,
    REMOTE_TIMEOUT_SECONDS,
    build_datum,
    get_tokenizer,
    load_env_key,
    sample_outputs,
    save_state_json,
)


ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "data" / "tinker" / "run_state.json"
CORPUS = ROOT / "data" / "g1_v2" / "recovery.jsonl"
MANIFEST = ROOT / "artifacts" / "g1-v2" / "recovery-manifest.json"
DEV = ROOT / "data" / "g1_v2" / "dev.jsonl"
ORIGINAL_DEV = ROOT / "data" / "dev_g1.jsonl"
OUT = ROOT / "data" / "tinker"
LR = 2e-5
BATCH = 16
SEED = 703


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if hashlib.sha256(CORPUS.read_bytes()).hexdigest() != manifest["sha256"]:
        raise SystemExit("Recovery corpus does not match manifest")
    rows = _load(CORPUS)
    state = json.loads(STATE_PATH.read_text())
    source = state["g1-v2:candidate_state"]
    tokenizer = get_tokenizer(state["g1:base_model"])
    preflight = {
        "source": source,
        "rows": len(rows),
        "steps": (len(rows) + BATCH - 1) // BATCH,
        "learning_rate": LR,
        "epochs": 1,
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if args.dry_run:
        return

    load_env_key()
    import tinker

    service = tinker.ServiceClient(user_metadata={"project": "smol-g1-v2-recovery"})
    client = service.create_training_client_from_state_with_optimizer(
        path=source, user_metadata={"project": "smol-g1-v2-recovery"}
    )
    adam = tinker.AdamParams(
        learning_rate=LR,
        beta1=G1_ADAM_BETA1,
        beta2=G1_ADAM_BETA2,
        eps=G1_ADAM_EPS,
        weight_decay=G1_WEIGHT_DECAY,
        grad_clip_norm=G1_GRAD_CLIP_NORM,
    )
    random.Random(SEED).shuffle(rows)
    started = time.monotonic()
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        step = start // BATCH + 1
        fb = client.forward_backward([build_datum(tinker, tokenizer, row) for row in batch], "cross_entropy")
        optim = client.optim_step(adam)
        fb.result(timeout=REMOTE_TIMEOUT_SECONDS)
        optim.result(timeout=REMOTE_TIMEOUT_SECONDS)
        print(f"[recovery] step={step}/{manifest['steps_at_batch_16']} batch={len(batch)}", flush=True)

    saved_state = client.save_state(name="smol-g1-v2-recovery").result(timeout=REMOTE_TIMEOUT_SECONDS)
    saved_sampler = client.save_weights_for_sampler(name="smol-g1-v2-recovery").result(timeout=REMOTE_TIMEOUT_SECONDS)
    state_path = str(getattr(saved_state, "path", saved_state))
    sampler_path = str(getattr(saved_sampler, "path", saved_sampler))
    save_state_json(
        {
            "g1-v2-recovery:source_state": source,
            "g1-v2-recovery:candidate_state": state_path,
            "g1-v2-recovery:candidate_sampler_path": sampler_path,
            "g1-v2-recovery:steps": manifest["steps_at_batch_16"],
            "g1-v2-recovery:learning_rate": LR,
        }
    )
    sampler = service.create_sampling_client(model_path=sampler_path)
    dev = _load(DEV)
    outputs = sample_outputs(sampler, tokenizer, dev, label="g1-v2-recovery", seed=SEED)
    report = evaluate_g1_v2(dev, outputs, label="g1-v2-recovery")
    report["training"] = {"seconds": round(time.monotonic() - started, 3), "sampler": sampler_path}
    _write(OUT / "dev-g1-v2-recovery_eval.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)

    original = _load(ORIGINAL_DEV)
    original_outputs = sample_outputs(
        sampler, tokenizer, original, label="g1-v2-recovery-original", seed=SEED + 1
    )
    original_report = evaluate_g1_predictions(
        original, original_outputs, label="g1-v2-recovery-original"
    )
    original_report["demo_1_to_4"] = _demo_1_to_4(original_report)
    _write(OUT / "dev-g1-v2-recovery-original_eval.json", original_report)
    print(json.dumps(original_report["demo_1_to_4"], indent=2), flush=True)


if __name__ == "__main__":
    main()
