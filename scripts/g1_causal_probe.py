"""Run the staged, maximum-84-call causal probe for g1 recurring reminders.

The stages distinguish raw-history clock extraction, action binding when due
state is supplied diagnostically, and elicitation from one minimal prompt rule.
Vague one-shot reminders are audited and excluded from exact timing claims.

    .venv/bin/python -m scripts.g1_causal_probe --dry-run
    .venv/bin/python -m scripts.g1_causal_probe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.policy import SYSTEM_PROMPT_G1
from datagen.g1_authored_demo5 import authored_paths, load_authored_batches
from datagen.g1_demo5 import (
    DEMO,
    Demo5Candidate,
    Demo5Targets,
    compile_demo5_dataset,
    demo5_bank_from_batches,
    derive_expected_class,
    plan_demo5_schedule,
    render_demo5_card,
)
from train.g1_causal_probe import (
    ClockState,
    MATH_SYSTEM_PROMPT,
    ORACLE_SYSTEM_SUFFIX,
    PROMPT_RULE_SUFFIX,
    extract_clock_state,
    make_oracle_prompt,
    score_action_outputs,
    score_math_outputs,
)


ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "data" / "tinker" / "run_state.json"
DEFAULT_DATA = ROOT / "data" / "dev_g1.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "tinker" / "dev-g1-epoch1_causal_probe.json"
DEFAULT_AUTHORED_ROOT = ROOT / "data" / "g1_authored"


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _case_fingerprint(cases: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _all_compiled_schedules():
    paths = authored_paths(DEFAULT_AUTHORED_ROOT, DEMO)
    bank = demo5_bank_from_batches(load_authored_batches(paths))
    schedule_ids = [f"demo5-sched-{index:05d}" for index in range(Demo5Targets().schedules)]
    configs = [plan_demo5_schedule(schedule_id, bank) for schedule_id in schedule_ids]
    return compile_demo5_dataset(configs, targets=Demo5Targets(), dev_fraction=0.1).schedules


def build_probe_cases(pairs: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_fires = [pair for pair in pairs if pair.get("demo") == DEMO and pair.get("should_fire")]
    once_fires = [pair for pair in all_fires if pair.get("schedule_kind") == "once"]
    recurring_fires = [pair for pair in all_fires if pair.get("schedule_kind") == "every"]
    if len(all_fires) != 20 or len(once_fires) != 6 or len(recurring_fires) != 14:
        raise ValueError(
            "Frozen dev reminder mix drifted; expected 20 fires = 6 once + 14 recurring."
        )

    schedules = {schedule.schedule_id: schedule for schedule in _all_compiled_schedules()}
    cases: list[dict[str, Any]] = []
    for fire in recurring_fires:
        schedule = schedules[str(fire["episode"])]
        fire_index = int(fire["fire_index"])
        compiled_fires = [
            candidate
            for candidate in schedule.candidates
            if candidate.role in {"fire-typing", "fire-silent"}
            and candidate.fire_index == fire_index
        ]
        if len(compiled_fires) != 1:
            raise ValueError(f"No unique compiled fire for {fire['candidate_id']}.")
        compiled_fire = compiled_fires[0]
        wait = None
        for turn_offset in range(compiled_fire.turn_offset - 1, -1, -1):
            wait_candidate = Demo5Candidate(
                candidate_id=f"{fire['candidate_id']}:probe-wait",
                schedule_id=schedule.schedule_id,
                turn_offset=turn_offset,
                role="fire-before",
                fire_index=fire_index,
                silent=compiled_fire.silent,
                alignment=compiled_fire.alignment,
            )
            try:
                candidate_row = render_demo5_card(schedule, wait_candidate)
            except ValueError:
                continue
            if candidate_row["completion"] == "<action>idle()</action>":
                wait = candidate_row
                break
        if wait is None:
            raise ValueError(f"No observable matched wait for {fire['candidate_id']}.")
        for role, row in (("fire", dict(fire)), ("wait", wait)):
            state = extract_clock_state(row)
            expected = derive_expected_class(
                str(row["prompt"]),
                schedule_kind="every",
                interval_s=int(row["interval_s"]),
                fire_message=str(row["fire_message"]),
                cancel_ack_text=row.get("cancel_ack_text"),
            )
            expected_class = "fire" if role == "fire" else "idle"
            if expected != expected_class or state.due != (role == "fire"):
                raise ValueError(f"Independent clock audit failed for {row['candidate_id']}.")
            cases.append(
                {
                    "case_id": f"{fire['candidate_id']}::{role}",
                    "pair_id": str(fire["candidate_id"]),
                    "role": role,
                    "episode": str(row["episode"]),
                    "candidate_id": str(row["candidate_id"]),
                    "prompt": str(row["prompt"]),
                    "completion": str(row["completion"]),
                    "clock_state": {
                        "interval_ms": state.interval_ms,
                        "anchor_ms": state.anchor_ms,
                        "now_ms": state.now_ms,
                        "elapsed_ms": state.elapsed_ms,
                        "due": state.due,
                    },
                    "current_event_index": state.current_event_index,
                    "fire_message": state.message,
                    "silent": bool(row.get("silent")),
                    "source": "frozen-dev" if role == "fire" else "reconstructed-heldout-neighbor",
                }
            )

    # Interleave the paired fire/wait cases instead of presenting a class block.
    cases.sort(key=lambda case: (case["pair_id"], case["role"] == "fire"))
    audit = {
        "all_dev_should_fire": len(all_fires),
        "excluded_unverifiable_once_fires": len(once_fires),
        "observable_recurring_fires": len(recurring_fires),
        "recurring_schedules": len({case["episode"] for case in cases}),
        "matched_waits": sum(case["role"] == "wait" for case in cases),
        "total_probe_cases": len(cases),
        "once_candidate_ids": [str(pair["candidate_id"]) for pair in once_fires],
    }
    if audit["recurring_schedules"] != 7 or len(cases) != 28:
        raise ValueError("Expected 28 cases from seven held-out recurring schedules.")
    return cases, audit


def _render_ids(tokenizer, *, system_prompt: str, user_prompt: str) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    return list(ids)


def _sample(
    *,
    sampler,
    tokenizer,
    requests: Sequence[tuple[str, str]],
    max_tokens: int,
    concurrency: int,
    seed: int,
    label: str,
) -> list[str]:
    import tinker

    params = tinker.types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        stop=["\n"],
        seed=seed,
    )
    outputs = [""] * len(requests)
    for start in range(0, len(requests), concurrency):
        chunk = list(enumerate(requests[start : start + concurrency], start=start))
        futures = [
            (
                index,
                sampler.sample(
                    prompt=tinker.ModelInput.from_ints(
                        _render_ids(tokenizer, system_prompt=system, user_prompt=user)
                    ),
                    num_samples=1,
                    sampling_params=params,
                ),
            )
            for index, (system, user) in chunk
        ]
        for index, future in futures:
            response = future.result(timeout=600)
            outputs[index] = tokenizer.decode(
                response.sequences[0].tokens,
                skip_special_tokens=True,
            ).strip()
        print(f"[{label}] sampled {min(start + concurrency, len(requests))}/{len(requests)}", flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tag", default="g1")
    parser.add_argument("--seed", type=int, default=650)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive.")

    cases, audit = build_probe_cases(_load_jsonl(args.data))
    fingerprint = _case_fingerprint(cases)
    print(f"[causal] audit={json.dumps(audit, sort_keys=True)}", flush=True)
    print(f"[causal] case_fingerprint={fingerprint}", flush=True)
    if args.dry_run:
        return

    state = _load_json(STATE_PATH)
    sampler_path = state.get(f"{args.tag}:sampler_path")
    base_model = state.get(f"{args.tag}:base_model")
    if not isinstance(sampler_path, str) or not sampler_path:
        raise SystemExit(f"No {args.tag}:sampler_path in {STATE_PATH}.")
    if not isinstance(base_model, str) or not base_model:
        raise SystemExit(f"No {args.tag}:base_model in {STATE_PATH}.")

    report: dict[str, Any] = {
        "schema_version": "g1-causal-probe-1",
        "status": "in_progress",
        "audit": audit,
        "case_fingerprint": fingerprint,
        "provenance": {
            "sampler_path": sampler_path,
            "base_model": base_model,
            "data": str(args.data),
            "seed": args.seed,
            "maximum_paid_calls": 84,
        },
        "stages": {},
    }
    if args.output.exists() and not args.restart:
        existing = _load_json(args.output)
        if (
            existing.get("schema_version") == report["schema_version"]
            and existing.get("case_fingerprint") == fingerprint
            and existing.get("provenance", {}).get("sampler_path") == sampler_path
        ):
            report = existing
            print(f"[causal] resuming {args.output}", flush=True)
            if report.get("status") == "complete":
                for name, stage in report.get("stages", {}).items():
                    print(
                        f"[causal] {name}="
                        f"{json.dumps(stage['summary'], sort_keys=True)}",
                        flush=True,
                    )
                print(
                    f"[causal] decision={report.get('decision')} "
                    f"paid_calls={report.get('paid_calls')}",
                    flush=True,
                )
                print(f"[causal] report={args.output}", flush=True)
                return

    _load_env()
    import tinker
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    service = tinker.ServiceClient(user_metadata={"project": "smol-g1-causal-probe"})
    sampler = service.create_sampling_client(model_path=sampler_path)
    started = time.monotonic()

    if "math" not in report["stages"]:
        requests = [(MATH_SYSTEM_PROMPT, case["prompt"]) for case in cases]
        outputs = _sample(
            sampler=sampler,
            tokenizer=tokenizer,
            requests=requests,
            max_tokens=128,
            concurrency=args.concurrency,
            seed=args.seed,
            label="causal-math",
        )
        report["stages"]["math"] = score_math_outputs(cases, outputs)
        _write_json(args.output, report)

    if "oracle_binding" not in report["stages"]:
        requests = []
        for case in cases:
            state_payload = ClockState(
                **case["clock_state"],
                current_event_index=int(case["current_event_index"]),
                message=str(case["fire_message"]),
            )
            requests.append(
                (
                    SYSTEM_PROMPT_G1 + ORACLE_SYSTEM_SUFFIX,
                    make_oracle_prompt(case["prompt"], state_payload),
                )
            )
        outputs = _sample(
            sampler=sampler,
            tokenizer=tokenizer,
            requests=requests,
            max_tokens=96,
            concurrency=args.concurrency,
            seed=args.seed,
            label="causal-oracle",
        )
        report["stages"]["oracle_binding"] = score_action_outputs(
            cases, outputs, gate_kind="oracle"
        )
        _write_json(args.output, report)

    math_passed = bool(report["stages"]["math"]["summary"]["gate_passed"])
    oracle_passed = bool(report["stages"]["oracle_binding"]["summary"]["gate_passed"])
    if math_passed and oracle_passed:
        if "minimal_prompt" not in report["stages"]:
            requests = [
                (SYSTEM_PROMPT_G1 + PROMPT_RULE_SUFFIX, case["prompt"])
                for case in cases
            ]
            outputs = _sample(
                sampler=sampler,
                tokenizer=tokenizer,
                requests=requests,
                max_tokens=96,
                concurrency=args.concurrency,
                seed=args.seed,
                label="causal-prompt",
            )
            report["stages"]["minimal_prompt"] = score_action_outputs(
                cases, outputs, gate_kind="prompt"
            )
            _write_json(args.output, report)
        prompt_passed = bool(report["stages"]["minimal_prompt"]["summary"]["gate_passed"])
        decision = (
            "minimal_prompt_candidate_requires_regression_eval"
            if prompt_passed
            else "self_trigger_gap_consider_single_micro_sft_or_explicit_runtime_scheduler"
        )
    elif oracle_passed:
        decision = "clock_extraction_gap_no_dpo_consider_narrow_math_repair_or_runtime_state"
    else:
        decision = "action_binding_gap_no_dpo_consider_micro_sft_or_runtime_scheduler"

    paid_calls = sum(stage["summary"]["support"] for stage in report["stages"].values())
    report.update(
        {
            "status": "complete",
            "decision": decision,
            "paid_calls": paid_calls,
            "stopped_before_prompt": "minimal_prompt" not in report["stages"],
            "wall_seconds_this_run": round(time.monotonic() - started, 3),
        }
    )
    _write_json(args.output, report)
    for name, stage in report["stages"].items():
        print(f"[causal] {name}={json.dumps(stage['summary'], sort_keys=True)}", flush=True)
    print(f"[causal] decision={decision} paid_calls={paid_calls}", flush=True)
    print(f"[causal] report={args.output}", flush=True)


if __name__ == "__main__":
    main()
