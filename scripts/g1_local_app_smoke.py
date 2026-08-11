"""Run durable closed-loop g1 app smokes with genuine model history."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "artifacts" / "g1-local-latency" / "app-closed-loop-smoke.json"
UNCHANGED_TEXT = (
    "I am drafting a short launch note for the lighthouse dashboard. "
    "It should explain the main signal, name the audience, and end with a "
    "clear next step without sounding like marketing copy."
)
TYPING_SOURCE = (
    "I am drafting a note for the team about the lighthouse dashboard. "
    "The first paragraph should explain why the signal matters and who should "
    "act on it. The second paragraph should summarize the current pattern, "
    "including quiet periods where no intervention is needed. The final "
    "paragraph should name one concrete next step and avoid vague timing. "
)
TICK_SECONDS = 0.650


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    latencies = [int(row["decision_ms"]) for row in rows]
    return {
        "ticks": len(rows),
        "median_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        "max_ms": max(latencies),
        "misses": sum(value > 650 for value in latencies),
        "valid_actions": sum(bool(row["valid"]) for row in rows),
    }


def _source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _run_session(
    runtime,
    *,
    content_for_tick,
    state,
    ticks: int,
    paced: bool = False,
) -> dict[str, object]:
    from app.domain import EventSource, UserState

    session = runtime.create_session()
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for tick in range(1, ticks + 1):
        if paced:
            deadline = started + (tick - 1) * TICK_SECONDS
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        content = content_for_tick(tick)
        turn = await runtime.process_event(
            session.id,
            source=EventSource.USER,
            content=content,
            state=state,
        )
        row = {
            "tick": tick,
            "event_index": turn.event.index,
            "decision_ms": turn.decision_ms,
            "raw": turn.action.raw_output,
            "valid": turn.action.valid,
            "content_chars": len(content),
            "elapsed_wall_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        rows.append(row)
        print(f"[app-smoke] tick={tick} {turn.decision_ms}ms {turn.action.raw_output}", flush=True)
    return {"summary": _summary(rows), "rows": rows}


async def run(ticks: int, typing_ticks: int) -> dict[str, object]:
    os.environ.setdefault("POLICY_MODE", "local")
    os.environ.setdefault("POLICY_PROMPT", "g1")
    os.environ.setdefault("STREAM_FORMAT", "g1")
    os.environ.setdefault("SEARCH_MODE", "demo")
    os.environ.setdefault("TRACE_PATH", "/tmp/text-gpt-live-app-smoke-trace.jsonl")

    from app.main import runtime

    runtime.policy.warm_up()
    from app.domain import UserState

    empty = await _run_session(
        runtime,
        content_for_tick=lambda _tick: "",
        state=UserState.IDLE,
        ticks=ticks,
    )
    unchanged = await _run_session(
        runtime,
        content_for_tick=lambda _tick: UNCHANGED_TEXT,
        state=UserState.IDLE,
        ticks=ticks,
    )
    continuous_typing = await _run_session(
        runtime,
        content_for_tick=lambda tick: TYPING_SOURCE[: tick * 4],
        state=UserState.ACTIVE,
        ticks=typing_ticks,
        paced=True,
    )
    from scripts.local_model import G1_QUANTIZED_DIR

    model_path = Path(os.environ.get("LOCAL_MODEL_PATH", str(G1_QUANTIZED_DIR)))
    weights = sorted(model_path.glob("*.safetensors"))
    return {
        "schema_version": "g1-local-app-smoke-2",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_revision": _source_revision(),
        "source_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        ),
        "policy": runtime.policy.display_name,
        "model_path": str(model_path.resolve()),
        "weight_sha256": {path.name: _sha256(path) for path in weights},
        "host": platform.platform(),
        "stream_format": runtime.stream_format,
        "tick_budget_ms": 650,
        "history": "closed loop; every prior action is the model output parsed by InteractionRuntime",
        "deterministic_action_substitution": False,
        "empty_silence": empty,
        "unchanged_text": unchanged,
        "continuous_typing": continuous_typing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=10)
    parser.add_argument("--typing-ticks", type=int, default=92)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = asyncio.run(run(args.ticks, args.typing_ticks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]["summary"]
                for key in ("empty_silence", "unchanged_text", "continuous_typing")
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
