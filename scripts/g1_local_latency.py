"""Benchmark the downloaded Qwen3.5 checkpoint against the g1 tick contract.

The runner uses the exact g1 system prompt, header-free stream compiler, native
Qwen chat template, and non-thinking mode. It reports complete-action latency,
time to first token, missed 650 ms deadlines, prompt throughput, output grammar,
and MLX peak memory.

Two lanes are deliberately separate:

* ``uncached`` matches the current ``LocalMLXPolicy.complete`` implementation,
  which creates a fresh cache for every decision.
* ``prefix_cache`` exercises MLX-LM's nearest-prefix cache across consecutive
  full-snapshot ticks. This measures an available optimization; it is not a
  claim that the current app already uses it.

Run on macOS outside a GPU-restricted sandbox:

    .venv-mlx/bin/python -m scripts.g1_local_latency
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.policy import SYSTEM_PROMPT_G1
from app.stream import parse_g1_action
from scripts.g1_context_growth import build_live_prompt, ticks_at


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "Qwen3.5-4B"
DEFAULT_REPORT = ROOT / "artifacts" / "g1-local-latency" / "report.json"
DEADLINE_MS = 650.0
MAX_OUTPUT_TOKENS = 32


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    scenario: str
    ticks: int
    trials: int


UNCACHED_CASES = (
    BenchmarkCase("initial-empty", "empty_silence", 1, 10),
    BenchmarkCase("empty-at-10s", "empty_silence", ticks_at(10), 5),
    BenchmarkCase("empty-at-30s", "empty_silence", ticks_at(30), 3),
    BenchmarkCase("empty-at-60s", "empty_silence", ticks_at(60), 5),
    BenchmarkCase("steady-text-at-60s", "steady_short_text", ticks_at(60), 3),
    BenchmarkCase("empty-at-5m", "empty_silence", ticks_at(300), 1),
    BenchmarkCase("empty-at-7m30s", "empty_silence", ticks_at(450), 1),
)

CACHE_SEQUENCES = (
    ("empty-around-60s", "empty_silence", ticks_at(60) - 20, ticks_at(60)),
)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a deterministic nearest-rank percentile."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    first_tokens = [float(row["first_token_ms"]) for row in rows]
    missed = sum(bool(row["missed_deadline"]) for row in rows)
    valid = sum(bool(row["valid_action"]) for row in rows)
    return {
        "samples": len(rows),
        "latency_ms": {
            "min": round(min(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3),
        },
        "first_token_ms": {
            "median": round(statistics.median(first_tokens), 3),
            "p95": round(percentile(first_tokens, 0.95), 3),
        },
        "missed_deadlines": missed,
        "missed_tick_rate": round(missed / len(rows), 6),
        "valid_actions": valid,
        "valid_action_rate": round(valid / len(rows), 6),
        "peak_memory_gb": round(max(float(row["peak_memory_gb"]) for row in rows), 3),
    }


def render_prompt(tokenizer: Any, stream: str) -> list[int]:
    value = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT_G1},
            {"role": "user", "content": stream},
        ],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(value, dict):
        value = value["input_ids"]
    return list(value)


def _valid_action(output: str) -> bool:
    try:
        parse_g1_action(output.strip())
    except ValueError:
        return False
    return True


def run_generation(
    *,
    model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    sampler: Any,
    prompt_cache: Any | None = None,
    cached_tokens: int = 0,
) -> tuple[dict[str, Any], list[int]]:
    from mlx_lm import stream_generate

    started = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    generated_tokens: list[int] = []
    final_response = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt_tokens,
        max_tokens=MAX_OUTPUT_TOKENS,
        sampler=sampler,
        prompt_cache=prompt_cache,
    ):
        now = time.perf_counter()
        if first_token_at is None:
            first_token_at = now
        pieces.append(response.text)
        generated_tokens.append(int(response.token))
        final_response = response
        if "\n" in "".join(pieces):
            break
    finished = time.perf_counter()
    if final_response is None or first_token_at is None:
        raise RuntimeError("MLX generation yielded no response")
    output = "".join(pieces).split("\n", 1)[0]
    latency_ms = (finished - started) * 1000
    measured_prompt_tokens = len(prompt_tokens) + cached_tokens
    return (
        {
            "prompt_tokens": measured_prompt_tokens,
            "processed_prompt_tokens": len(prompt_tokens),
            "cached_tokens": cached_tokens,
            "cache_hit_ratio": round(cached_tokens / measured_prompt_tokens, 6),
            "output_tokens": len(generated_tokens),
            "first_token_ms": round((first_token_at - started) * 1000, 3),
            "latency_ms": round(latency_ms, 3),
            "missed_deadline": latency_ms > DEADLINE_MS,
            "prompt_tps": round(float(final_response.prompt_tps), 3),
            "generation_tps": round(float(final_response.generation_tps), 3),
            "peak_memory_gb": round(float(final_response.peak_memory), 3),
            "output": output,
            "valid_action": _valid_action(output),
        },
        generated_tokens,
    )


def run_uncached(model: Any, tokenizer: Any, sampler: Any) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for case in UNCACHED_CASES:
        prompt = render_prompt(tokenizer, build_live_prompt(case.scenario, case.ticks))
        rows: list[dict[str, Any]] = []
        for trial in range(1, case.trials + 1):
            row, _ = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt,
                sampler=sampler,
            )
            row["trial"] = trial
            rows.append(row)
            all_rows.append(row)
            print(
                f"uncached {case.name} {trial}/{case.trials}: "
                f"{row['prompt_tokens']:,} tok, {row['latency_ms']:.1f} ms, "
                f"{'MISS' if row['missed_deadline'] else 'pass'}",
                flush=True,
            )
        cases.append({**asdict(case), "summary": summarize(rows), "rows": rows})
    return {"summary": summarize(all_rows), "cases": cases}


def run_cache_sequence(
    model: Any,
    tokenizer: Any,
    sampler: Any,
    *,
    name: str,
    scenario: str,
    seed_tick: int,
    end_tick: int,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache

    model_key = "qwen3.5-4b"
    lru = LRUPromptCache(max_size=1)
    rows: list[dict[str, Any]] = []
    terminated_reason: str | None = None
    for tick in range(seed_tick, end_tick + 1):
        full_prompt = render_prompt(tokenizer, build_live_prompt(scenario, tick))
        cache, rest = lru.fetch_nearest_cache(model_key, full_prompt)
        if cache is None:
            cache = make_prompt_cache(model)
        cached_tokens = len(full_prompt) - len(rest)
        row, generated = run_generation(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=rest,
            sampler=sampler,
            prompt_cache=cache,
            cached_tokens=cached_tokens,
        )
        row["tick"] = tick
        row["seed"] = tick == seed_tick
        lru.insert_cache(model_key, full_prompt + generated, cache)
        print(
            f"cached {name} tick {tick}: {row['prompt_tokens']:,} tok, "
            f"{row['cached_tokens']:,} cached, {row['latency_ms']:.1f} ms, "
            f"{'MISS' if row['missed_deadline'] else 'pass'}",
            flush=True,
        )
        if tick != seed_tick:
            rows.append(row)
            if len(rows) >= 3 and all(item["cached_tokens"] == 0 for item in rows[-3:]):
                terminated_reason = (
                    "Stopped after three consecutive target ticks reused zero tokens; "
                    "continuing would only repeat uncached inference."
                )
                break
    result = {
        "name": name,
        "scenario": scenario,
        "seed_tick": seed_tick,
        "end_tick": end_tick,
        "summary": summarize(rows),
        "rows": rows,
        "cache_bytes_after_sequence": lru.nbytes,
        "terminated_reason": terminated_reason,
    }
    del lru
    mx.clear_cache()
    return result


def run_prefix_cache(model: Any, tokenizer: Any, sampler: Any) -> dict[str, Any]:
    sequences = [
        run_cache_sequence(
            model,
            tokenizer,
            sampler,
            name=name,
            scenario=scenario,
            seed_tick=seed_tick,
            end_tick=end_tick,
        )
        for name, scenario, seed_tick, end_tick in CACHE_SEQUENCES
    ]
    all_rows = [row for sequence in sequences for row in sequence["rows"]]
    return {"summary": summarize(all_rows), "sequences": sequences}


def run_benchmark(model_path: Path) -> dict[str, Any]:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    load_started = time.perf_counter()
    model, tokenizer = load(str(model_path))
    load_seconds = time.perf_counter() - load_started
    sampler = make_sampler(temp=0.0)

    warm_prompt = render_prompt(tokenizer, build_live_prompt("empty_silence", 1))
    warmup, _ = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_tokens=warm_prompt,
        sampler=sampler,
    )
    print(
        f"warmup: {warmup['prompt_tokens']:,} tok, {warmup['latency_ms']:.1f} ms",
        flush=True,
    )

    uncached = run_uncached(model, tokenizer, sampler)
    prefix_cache = run_prefix_cache(model, tokenizer, sampler)
    return {
        "schema_version": "g1-local-latency-1",
        "measured_at": datetime.now(UTC).isoformat(),
        "model_path": str(model_path),
        "weights": "BF16 source checkpoint",
        "runtime": {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__},
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": mx.device_info(),
        },
        "contract": {
            "system_prompt": "SYSTEM_PROMPT_G1",
            "stream_format": "g1",
            "enable_thinking": False,
            "deadline_ms": DEADLINE_MS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "model_load_seconds": round(load_seconds, 3),
        "warmup": warmup,
        "uncached": uncached,
        "prefix_cache": prefix_cache,
        "interpretation_constraints": [
            "The uncached lane matches the current LocalMLXPolicy cache behavior.",
            "The prefix-cache lane measures MLX-LM nearest-prefix reuse but is not integrated into the current app.",
            "This is the untrained base checkpoint, so action correctness is diagnostic rather than a trained-model quality evaluation.",
            "The run is single-request and single-process; no overlapping inference was tested.",
        ],
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def print_final(report: dict[str, Any], output: Path) -> None:
    for lane in ("uncached", "prefix_cache"):
        summary = report[lane]["summary"]
        latency = summary["latency_ms"]
        print(
            f"{lane}: n={summary['samples']}, median={latency['median']:.1f} ms, "
            f"p95={latency['p95']:.1f} ms, missed={summary['missed_tick_rate']:.1%}",
            flush=True,
        )
    print(f"report: {output}", flush=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.model.is_dir():
        parser.error(f"model directory not found: {args.model}")
    report = run_benchmark(args.model.resolve())
    write_report(report, args.output)
    print_final(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
