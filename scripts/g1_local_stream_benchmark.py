"""Benchmark controlled g1 prompt growth with generated-output parity.

This benchmark never substitutes a deterministic action. Every recorded output
comes from greedy MLX decoding. It additionally compares selected cached
outputs with fresh-cache model generations on the exact same tokenized prompt.

Historical actions in these synthetic prompts are deliberately fixed to
``idle()`` so prompt growth is reproducible. This is not a closed-loop app
session; use ``g1_local_app_smoke`` for model-generated history.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.policy import SYSTEM_PROMPT_G1
from app.stream import parse_g1_action
from scripts.g1_context_growth import build_live_prompt, ticks_at
from scripts.local_model import G1_QUANTIZED_DIR, LocalMLXPolicy


DEFAULT_OUTPUT = ROOT / "artifacts" / "g1-v2" / "local-stream-benchmark-8bit.json"
DEADLINE_MS = 650.0
SCENARIOS = ("empty_silence", "steady_short_text")
END_TICK = ticks_at(60)
PARITY_TICKS = {1, ticks_at(10), ticks_at(30), END_TICK}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    steady = [row for row in rows if row["tick"] > 1]
    latencies = [float(row["latency_ms"]) for row in steady]
    return {
        "samples": len(rows),
        "steady_samples": len(steady),
        "initial_latency_ms": rows[0]["latency_ms"],
        "steady_latency_ms": {
            "min": round(min(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3),
        },
        "steady_missed_deadlines": sum(value > DEADLINE_MS for value in latencies),
        "steady_missed_tick_rate": round(
            sum(value > DEADLINE_MS for value in latencies) / len(latencies), 6
        ),
        "valid_actions": sum(bool(row["valid_action"]) for row in rows),
        "valid_action_rate": round(sum(bool(row["valid_action"]) for row in rows) / len(rows), 6),
        "minimum_reused_tokens_after_first": min(int(row["reused_tokens"]) for row in steady),
    }


def _valid(output: str) -> bool:
    return bool(parse_g1_action(output).valid)


def _uncached(policy: LocalMLXPolicy, prompt: str) -> tuple[str, float]:
    started = time.perf_counter()
    output = policy._complete_uncached(prompt)
    return output, round((time.perf_counter() - started) * 1000, 3)


def run(model_path: Path) -> dict[str, Any]:
    import mlx.core as mx
    import mlx_lm

    startup_started = time.perf_counter()
    policy = LocalMLXPolicy(model_path=str(model_path), system_prompt=SYSTEM_PROMPT_G1)
    startup_seconds = time.perf_counter() - startup_started
    provenance_path = model_path / "MERGE_PROVENANCE.json"
    weight_label = "MLX checkpoint"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        quantization = provenance.get("quantization", {})
        if quantization.get("bits"):
            weight_label = (
                f"{quantization['bits']}-bit "
                f"{quantization.get('mode', 'quantized')} MLX"
            )
    scenarios: list[dict[str, Any]] = []
    parity: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        policy.reset_stream_cache()
        rows: list[dict[str, Any]] = []
        selected: dict[int, tuple[str, str]] = {}
        for tick in range(1, END_TICK + 1):
            prompt = build_live_prompt(scenario, tick)
            output = policy.complete(prompt)
            row = {
                "tick": tick,
                **policy.last_metrics,
                "output": output,
                "valid_action": _valid(output),
                "missed_deadline": policy.last_metrics["latency_ms"] > DEADLINE_MS,
            }
            rows.append(row)
            if tick in PARITY_TICKS:
                selected[tick] = (prompt, output)
            if tick == 1 or tick % 10 == 0 or tick == END_TICK:
                print(
                    f"[persistent] {scenario} tick={tick}/{END_TICK} "
                    f"latency={row['latency_ms']:.1f}ms reused={row['reused_tokens']}",
                    flush=True,
                )
        scenarios.append({"scenario": scenario, "summary": _summary(rows), "rows": rows})
        for tick, (prompt, cached_output) in selected.items():
            uncached_output, latency = _uncached(policy, prompt)
            item = {
                "scenario": scenario,
                "tick": tick,
                "cached_output": cached_output,
                "uncached_output": uncached_output,
                "exact_match": cached_output == uncached_output,
                "uncached_latency_ms": latency,
            }
            parity.append(item)
            print(
                f"[parity] {scenario} tick={tick} exact={item['exact_match']} "
                f"uncached={latency:.1f}ms",
                flush=True,
            )
    all_steady = [
        float(row["latency_ms"])
        for scenario in scenarios
        for row in scenario["rows"]
        if row["tick"] > 1
    ]
    return {
        "schema_version": "g1-controlled-prefix-latency-2",
        "measured_at": datetime.now(UTC).isoformat(),
        "model_path": str(model_path.resolve()),
        "weights": weight_label,
        "model_startup_seconds": round(startup_seconds, 3),
        "runtime": {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__},
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": mx.device_info(),
        },
        "contract": {
            "deadline_ms": DEADLINE_MS,
            "tick_ms": 650,
            "system_prompt": "SYSTEM_PROMPT_G1",
            "enable_thinking": False,
            "decoding": "greedy model generation",
            "deterministic_action_substitution": False,
            "cache_boundary": "exact token prefix through current stream_event",
            "startup_cache": "invariant system and user-role prefix prefilled before sessions",
            "startup_cache_tokens": len(policy._static_tokens),
            "generation_cache": "structural copy; generated recurrent state discarded",
            "history_mode": "controlled synthetic history with scripted prior idle actions",
            "closed_loop": False,
        },
        "overall_steady": {
            "samples": len(all_steady),
            "median_ms": round(statistics.median(all_steady), 3),
            "p95_ms": round(_percentile(all_steady, 0.95), 3),
            "missed_tick_rate": round(
                sum(value > DEADLINE_MS for value in all_steady) / len(all_steady), 6
            ),
        },
        "parity": {
            "support": len(parity),
            "exact": sum(item["exact_match"] for item in parity),
            "passed": all(item["exact_match"] for item in parity),
            "rows": parity,
        },
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=G1_QUANTIZED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory not found: {args.model}")
    report = run(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall_steady": report["overall_steady"], "parity": report["parity"]}, indent=2), flush=True)
    print(f"report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
