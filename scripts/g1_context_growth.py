"""Measure live g1 prompt growth with the real Qwen3.5 tokenizer.

This is the weight-free half of the local feasibility gate. It uses the exact
g1 stream compiler, system prompt, Qwen chat template, and thinking-disabled
rendering used by serving. It does not load model weights or claim inference
latency/cache results.

Run:
    .venv/bin/python -m scripts.g1_context_growth
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from app.domain import Action, ActionKind, CompletedTurn, EventSource, StreamEvent, UserState
from app.policy import SYSTEM_PROMPT_G1
from app.stream import compile_stream


ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen3.5-4B"
TICK_MS = 650
DEFAULT_CHECKPOINT_SECONDS = (1, 10, 30, 60, 120, 300, 600)
DEFAULT_REPORT = ROOT / "artifacts" / "g1-context-growth" / "report.json"
TYPING_CHARS_PER_TICK = 4

STEADY_TEXT = (
    "I am drafting a short launch note for the lighthouse dashboard. "
    "It should explain the main signal, name the audience, and end with a "
    "clear next step without sounding like marketing copy."
)
TYPING_SOURCE = (
    "I am drafting a note for the team about the lighthouse dashboard. "
    "The first paragraph should explain why the signal matters and who should "
    "act on it. The second paragraph should summarize the current pattern, "
    "including the quiet periods where no intervention is needed. The final "
    "paragraph should name one concrete next step and avoid vague timing. "
)

SCENARIOS: dict[str, str] = {
    "empty_silence": "The foreground app keeps ticking while the textbox remains empty.",
    "steady_short_text": "A short completed textbox remains unchanged while the user pauses.",
    "continuous_typing": (
        "The textbox grows by four characters per tick, approximately the observed "
        "human typing cadence, and every historical tick retains its full snapshot."
    ),
}


class TokenizerLike(Protocol):
    model_max_length: int

    def __call__(
        self, text: str, *, add_special_tokens: bool = False
    ) -> dict[str, list[int]]: ...

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int] | dict[str, list[int]]: ...


def ticks_at(seconds: int) -> int:
    """Number of 650 ms decisions whose timestamps fall within `seconds`."""
    if seconds <= 0:
        raise ValueError("Checkpoint seconds must be positive.")
    return max(1, math.floor(seconds * 1000 / TICK_MS))


def _repeat_prefix(source: str, length: int) -> str:
    if length <= 0:
        return ""
    repetitions = math.ceil(length / len(source))
    return (source * repetitions)[:length]


def content_for_tick(scenario: str, tick: int) -> str:
    if scenario == "empty_silence":
        return ""
    if scenario == "steady_short_text":
        return STEADY_TEXT
    if scenario == "continuous_typing":
        return _repeat_prefix(TYPING_SOURCE, tick * TYPING_CHARS_PER_TICK)
    raise ValueError(f"Unknown context-growth scenario: {scenario}")


def build_live_prompt(scenario: str, ticks: int) -> str:
    """Build the exact live g1 user prompt at a given decision count."""
    if ticks < 1:
        raise ValueError("At least one tick is required.")
    state = UserState.ACTIVE if scenario == "continuous_typing" else UserState.IDLE
    history: list[CompletedTurn] = []
    for index in range(1, ticks):
        event = StreamEvent(
            index=index,
            source=EventSource.USER,
            content=content_for_tick(scenario, index),
            state=state,
            elapsed_ms=index * TICK_MS,
        )
        history.append(CompletedTurn(event=event, action=Action(ActionKind.IDLE)))
    current = StreamEvent(
        index=ticks,
        source=EventSource.USER,
        content=content_for_tick(scenario, ticks),
        state=state,
        elapsed_ms=ticks * TICK_MS,
    )
    return compile_stream(history, current, fmt="g1")


def _token_ids(value: list[int] | dict[str, list[int]]) -> list[int]:
    return value if isinstance(value, list) else value["input_ids"]


def token_counts(tokenizer: TokenizerLike, prompt: str) -> tuple[int, int]:
    stream_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    rendered_ids = _token_ids(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT_G1},
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    return len(stream_ids), len(rendered_ids)


def measure_scenario(
    tokenizer: TokenizerLike,
    scenario: str,
    checkpoint_seconds: tuple[int, ...],
    context_window: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    previous_ticks = 0
    previous_tokens = 0
    for seconds in checkpoint_seconds:
        ticks = ticks_at(seconds)
        prompt = build_live_prompt(scenario, ticks)
        stream_tokens, rendered_tokens = token_counts(tokenizer, prompt)
        added_ticks = ticks - previous_ticks
        row = {
            "checkpoint_seconds": seconds,
            "ticks": ticks,
            "current_text_chars": len(content_for_tick(scenario, ticks)),
            "compiled_stream_chars": len(prompt),
            "stream_tokens": stream_tokens,
            "rendered_tokens": rendered_tokens,
            "context_utilization": round(rendered_tokens / context_window, 6),
            "within_context_window": rendered_tokens <= context_window,
            "tokens_per_added_tick": round(
                (rendered_tokens - previous_tokens) / added_ticks, 3
            ),
        }
        rows.append(row)
        previous_ticks = ticks
        previous_tokens = rendered_tokens
    first_over = next((row for row in rows if not row["within_context_window"]), None)
    first_tick_over = None
    if first_over is not None:
        low = 1
        high = int(first_over["ticks"])
        while low < high:
            middle = (low + high) // 2
            _, rendered_tokens = token_counts(
                tokenizer, build_live_prompt(scenario, middle)
            )
            if rendered_tokens > context_window:
                high = middle
            else:
                low = middle + 1
        prompt = build_live_prompt(scenario, low)
        stream_tokens, rendered_tokens = token_counts(tokenizer, prompt)
        first_tick_over = {
            "ticks": low,
            "elapsed_seconds": round(low * TICK_MS / 1000, 3),
            "current_text_chars": len(content_for_tick(scenario, low)),
            "stream_tokens": stream_tokens,
            "rendered_tokens": rendered_tokens,
            "context_utilization": round(rendered_tokens / context_window, 6),
        }
    return {
        "name": scenario,
        "description": SCENARIOS[scenario],
        "measurements": rows,
        "first_measured_over_context": first_over,
        "first_tick_over_context": first_tick_over,
    }


def run_measurement(
    tokenizer: TokenizerLike,
    checkpoint_seconds: tuple[int, ...] = DEFAULT_CHECKPOINT_SECONDS,
) -> dict[str, Any]:
    context_window = int(tokenizer.model_max_length)
    if context_window <= 0 or context_window > 10_000_000:
        raise ValueError(f"Tokenizer exposed an implausible context window: {context_window}")
    checkpoints = tuple(sorted(set(checkpoint_seconds)))
    if not checkpoints:
        raise ValueError("At least one checkpoint is required.")
    return {
        "schema_version": "g1-context-growth-1",
        "measured_at": datetime.now(UTC).isoformat(),
        "model": MODEL_ID,
        "context_window_tokens": context_window,
        "tick_ms": TICK_MS,
        "weights_loaded": False,
        "rendering": {
            "system_prompt": "SYSTEM_PROMPT_G1",
            "stream_format": "g1",
            "chat_template": "model-native",
            "enable_thinking": False,
            "add_generation_prompt": True,
        },
        "scenarios": [
            measure_scenario(tokenizer, scenario, checkpoints, context_window)
            for scenario in SCENARIOS
        ],
        "limitations": [
            "No model weights were loaded.",
            "This does not measure inference latency or prefix-cache reuse.",
            "Historical snapshots are retained exactly as the current live compiler emits them.",
        ],
    }


def print_summary(report: dict[str, Any]) -> None:
    print(
        f"{report['model']} · {report['context_window_tokens']:,}-token context · "
        f"{report['tick_ms']} ms ticks"
    )
    for scenario in report["scenarios"]:
        print(f"\n{scenario['name']}")
        for row in scenario["measurements"]:
            print(
                f"  {row['checkpoint_seconds']:>4}s  {row['ticks']:>4} ticks  "
                f"{row['rendered_tokens']:>8,} tokens  "
                f"{row['context_utilization'] * 100:>6.1f}%  "
                f"text={row['current_text_chars']:,} chars"
            )
        crossing = scenario["first_tick_over_context"]
        if crossing is not None:
            print(
                f"  first over context: tick {crossing['ticks']} at "
                f"{crossing['elapsed_seconds']:.1f}s "
                f"({crossing['rendered_tokens']:,} tokens)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--checkpoint-seconds",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHECKPOINT_SECONDS),
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    report = run_measurement(tokenizer, tuple(args.checkpoint_seconds))
    report["model"] = args.model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_summary(report)
    print(f"\nreport: {args.output}")


if __name__ == "__main__":
    main()
