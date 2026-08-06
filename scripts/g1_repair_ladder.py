"""Cost-capped continuation ladder after the first g1 repair micro-epoch.

The first 5 low-rate steps were intentionally conservative and produced zero
held-out recurring fires. This runner resumes that optimizer state, restores
the original g1 learning rate, and evaluates 14 matched fire/wait pairs from
unused train-split schedules after every five updates. It stops after two
zero-fire rungs, or after four rungs total. The frozen 65-case focused suite and
675-case full dev suite each run only once, after the validation set selects a
rung. Semantic regression review remains a separate, mandatory promotion gate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.g1_repair_train import (
    BASELINE_REPORT,
    DEV_PATH,
    FULL_REPORT,
    REPAIR_MANIFEST,
    REPAIR_PATH,
    STATE_PATH,
    _fingerprint,
    _full_comparison,
    _load_json,
    _load_jsonl,
    _write_json,
    build_focused_cases,
)
from train.g1_evaluation import evaluate_g1_predictions, score_g1_prediction
from train.g1_repair import (
    build_ladder_validation_cases,
    pack_repair_batches,
    score_focused_repair,
)
from train.tinker_run import (
    G1_ADAM_BETA1,
    G1_ADAM_BETA2,
    G1_ADAM_EPS,
    G1_GRAD_CLIP_NORM,
    G1_LEARNING_RATE,
    G1_WEIGHT_DECAY,
    REMOTE_TIMEOUT_SECONDS,
    build_datum,
    get_tokenizer,
    load_env_key,
    sample_outputs,
    save_state_json,
)


ROOT = Path(__file__).resolve().parent.parent
MAX_RUNGS = 4
STEPS_PER_RUNG = 5
SEED = 652
LADDER_DIR = ROOT / "data" / "tinker" / "g1-repair-ladder"
TRAIN_GLOB = "train_g1-*-of-*.jsonl"


def _selection_score(cases: Sequence[Mapping[str, Any]], outputs: Sequence[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, (case, output) in enumerate(zip(cases, outputs, strict=True)):
        scored = score_g1_prediction(case, output, row_index=index)
        rows.append({**scored, "role": case["role"]})
    fire = [row for row in rows if row["role"] == "fire"]
    wait = [row for row in rows if row["role"] == "wait"]
    summary = {
        "fire_exact": sum(row["row_pass"] for row in fire),
        "fire_support": len(fire),
        "wait_exact": sum(row["row_pass"] for row in wait),
        "wait_support": len(wait),
        "format_valid": sum(row["format_valid"] for row in rows),
        "support": len(rows),
    }
    summary["passed"] = (
        summary["fire_exact"] >= 13
        and summary["wait_exact"] >= 13
        and summary["format_valid"] == summary["support"]
    )
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state = _load_json(STATE_PATH)
    source_state = state.get("g1-repair:candidate_state")
    base_model = state.get("g1:base_model")
    if not isinstance(source_state, str) or not source_state:
        raise SystemExit("The rejected five-step repair state is not recorded.")
    if state.get("g1-repair-ladder:finished"):
        raise SystemExit("The repair ladder already finished; refusing to spend twice.")
    manifest = _load_json(REPAIR_MANIFEST)
    if _fingerprint(REPAIR_PATH) != manifest.get("sha256"):
        raise SystemExit("Repair corpus SHA does not match its manifest.")
    repair_rows = _load_jsonl(REPAIR_PATH)
    batches = pack_repair_batches(repair_rows)
    dev_rows = _load_jsonl(DEV_PATH)
    train_paths = sorted((ROOT / "data").glob(TRAIN_GLOB))
    if not train_paths:
        raise SystemExit(f"No g1 training shards matched data/{TRAIN_GLOB}.")
    train_rows = [row for path in train_paths for row in _load_jsonl(path)]
    validation_cases, validation_audit = build_ladder_validation_cases(
        train_rows, repair_rows, dev_rows
    )
    focused_cases = build_focused_cases(dev_rows)
    preflight = {
        "source_state": source_state,
        "learning_rate": G1_LEARNING_RATE,
        "max_rungs": MAX_RUNGS,
        "steps_per_rung": STEPS_PER_RUNG,
        "max_additional_steps": MAX_RUNGS * STEPS_PER_RUNG,
        "selection_validation_cases_per_rung": len(validation_cases),
        "selection_validation_audit": validation_audit,
        "frozen_acceptance_policy": "focused and full run once after rung selection",
        "promotion_policy": "deterministic gates plus separate semantic regression review",
        "futility_stop": "stop after two ladder rungs if fire recall remains 0/14",
        "corpus_sha256": manifest["sha256"],
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if args.dry_run:
        return

    load_env_key()
    import tinker

    tokenizer = get_tokenizer(base_model)
    service = tinker.ServiceClient(user_metadata={"project": "smol-g1-repair-ladder"})
    client = service.create_training_client_from_state_with_optimizer(
        path=source_state,
        user_metadata={"project": "smol-g1-repair-ladder"},
    )
    adam = tinker.AdamParams(
        learning_rate=G1_LEARNING_RATE,
        beta1=G1_ADAM_BETA1,
        beta2=G1_ADAM_BETA2,
        eps=G1_ADAM_EPS,
        weight_decay=G1_WEIGHT_DECAY,
        grad_clip_norm=G1_GRAD_CLIP_NORM,
    )
    started = time.monotonic()
    rungs: list[dict[str, Any]] = []
    winning_sampler = None
    winning_sampler_path = ""
    winning_state_path = ""

    for rung in range(1, MAX_RUNGS + 1):
        rung_started = time.monotonic()
        losses: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(batches, start=1):
            step_started = time.monotonic()
            data = [build_datum(tinker, tokenizer, row) for row in batch]
            fb = client.forward_backward(data, "cross_entropy")
            opt = client.optim_step(adam)
            result = fb.result(timeout=REMOTE_TIMEOUT_SECONDS)
            opt.result(timeout=REMOTE_TIMEOUT_SECONDS)
            metrics = getattr(result, "metrics", None) or {}
            row = {
                "batch": batch_index,
                "seconds": round(time.monotonic() - step_started, 3),
                "metrics": metrics if isinstance(metrics, dict) else str(metrics),
            }
            losses.append(row)
            print(f"[ladder] rung={rung} step={batch_index}/5 seconds={row['seconds']}", flush=True)

        name = f"smol-g1-repair-ladder-r{rung}"
        state_result = client.save_state(name=name).result(timeout=REMOTE_TIMEOUT_SECONDS)
        sampler_result = client.save_weights_for_sampler(name=name).result(
            timeout=REMOTE_TIMEOUT_SECONDS
        )
        state_path = str(getattr(state_result, "path", state_result))
        sampler_path = str(getattr(sampler_result, "path", sampler_result))
        sampler = service.create_sampling_client(model_path=sampler_path)
        outputs = sample_outputs(
            sampler,
            tokenizer,
            list(validation_cases),
            label=f"g1-repair-ladder-r{rung}",
            seed=SEED,
        )
        validation = _selection_score(validation_cases, outputs)
        rung_report = {
            "rung": rung,
            "additional_steps": rung * STEPS_PER_RUNG,
            "state_path": state_path,
            "sampler_path": sampler_path,
            "selection_validation": validation,
            "losses": losses,
            "wall_seconds": round(time.monotonic() - rung_started, 3),
        }
        LADDER_DIR.mkdir(parents=True, exist_ok=True)
        _write_json(LADDER_DIR / f"rung-{rung}.json", rung_report)
        rungs.append(rung_report)
        print(
            f"[ladder] rung={rung} "
            f"selection_validation={json.dumps(validation['summary'], sort_keys=True)}",
            flush=True,
        )
        if validation["summary"]["passed"]:
            winning_sampler = sampler
            winning_sampler_path = sampler_path
            winning_state_path = state_path
            break
        if rung == 2 and all(
            item["selection_validation"]["summary"]["fire_exact"] == 0 for item in rungs
        ):
            print("[ladder] futility stop: two rungs remain at 0/14 fire.", flush=True)
            break

    focused = None
    full = None
    accepted = False
    deterministic_gates_passed = False
    if winning_sampler is not None:
        outputs = sample_outputs(
            winning_sampler,
            tokenizer,
            focused_cases,
            label="g1-repair-ladder-focused",
            seed=SEED,
        )
        focused = score_focused_repair(focused_cases, outputs)
        _write_json(LADDER_DIR / "focused.json", focused)
        print(f"[ladder] focused={json.dumps(focused['summary'], sort_keys=True)}", flush=True)
        if focused["summary"]["passed"]:
            outputs = sample_outputs(
                winning_sampler,
                tokenizer,
                dev_rows,
                label="g1-repair-ladder-full",
                seed=SEED,
            )
            full = evaluate_g1_predictions(dev_rows, outputs, label="dev-g1-repair-ladder-full")
            full["comparison"] = _full_comparison(dev_rows, _load_json(BASELINE_REPORT), full)
            _write_json(FULL_REPORT, full)
            deterministic_gates_passed = bool(full["comparison"]["deterministic_gates_passed"])
            accepted = bool(full["comparison"]["passed"])
            print(f"[ladder] full={json.dumps(full['comparison'], sort_keys=True)}", flush=True)

    summary = {
        **preflight,
        "rungs_run": len(rungs),
        "additional_steps_run": len(rungs) * STEPS_PER_RUNG,
        "selection_validation_summaries": [
            item["selection_validation"]["summary"] for item in rungs
        ],
        "winning_state": winning_state_path or None,
        "winning_sampler": winning_sampler_path or None,
        "focused_passed": bool(focused and focused["summary"]["passed"]),
        "full_ran": full is not None,
        "deterministic_gates_passed": deterministic_gates_passed,
        "semantic_regression_status": (
            full["comparison"]["semantic_regression_status"] if full is not None else "not_run"
        ),
        "accepted": accepted,
        "accepted_deterministic": False,
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    _write_json(LADDER_DIR / "summary.json", summary)
    save_state_json(
        {
            "g1-repair-ladder:finished": True,
            "g1-repair-ladder:rungs_run": len(rungs),
            "g1-repair-ladder:additional_steps": len(rungs) * STEPS_PER_RUNG,
            "g1-repair-ladder:winning_state": winning_state_path or None,
            "g1-repair-ladder:winning_sampler_path": winning_sampler_path or None,
            "g1-repair-ladder:deterministic_gates_passed": deterministic_gates_passed,
            "g1-repair-ladder:semantic_regression_status": (
                full["comparison"]["semantic_regression_status"]
                if full is not None
                else "not_run"
            ),
            "g1-repair-ladder:accepted": accepted,
            "g1-repair-ladder:accepted_deterministic": False,
        }
    )
    print(f"[ladder] summary={json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
