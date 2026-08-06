"""Fresh closed-loop acceptance for the four user-facing g1-v2 demos."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _verb(turn) -> str:
    action = turn.action
    if not action.valid:
        return "invalid"
    return action.tool_name if action.kind.value == "tool" else action.kind.value


def _search_answer_is_grounded(message: str, payload: dict[str, Any]) -> bool:
    source = " ".join(
        str(value)
        for result in payload.get("results", [])
        if isinstance(result, dict)
        for value in (result.get("title", ""), result.get("snippet", ""))
    )
    message_tokens = set(re.findall(r"[a-z0-9$]+", message.lower()))
    source_tokens = set(re.findall(r"[a-z0-9$]+", source.lower()))
    numeric_overlap = {
        token for token in message_tokens & source_tokens if any(char.isdigit() for char in token)
    }
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
        "is", "it", "latest", "of", "on", "or", "result", "search", "source", "the",
        "this", "to", "was", "with",
    }
    content_overlap = (message_tokens & source_tokens) - stop
    return bool(numeric_overlap) or len(content_overlap) >= 2


async def _wait_until(predicate, *, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the expected live acceptance event.")
        await asyncio.sleep(0.02)


async def _tick(runtime, session_id: str, content: str, idle: bool = False):
    from app.domain import EventSource, UserState

    return await runtime.process_event(
        session_id,
        source=EventSource.USER,
        content=content,
        state=UserState.IDLE if idle else UserState.ACTIVE,
    )


async def _run(args) -> dict[str, Any]:
    os.environ["POLICY_MODE"] = args.policy
    os.environ["POLICY_PROMPT"] = "g1"
    os.environ["STREAM_FORMAT"] = "g1"
    os.environ["SEARCH_MODE"] = args.search
    os.environ.setdefault("UIGEN_MODE", "demo")
    os.environ.setdefault("WRITER_MODE", "demo")
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.write_text("", encoding="utf-8")
    os.environ["TRACE_PATH"] = str(args.trace)
    if args.model:
        if args.policy == "local":
            os.environ["SMOL_LOCAL_MODEL"] = args.model
        else:
            os.environ["TINKER_MODEL_PATH"] = args.model

    from app.main import build_runtime

    runtime = build_runtime()
    runtime.policy.warm_up()
    checks: list[dict[str, Any]] = []

    def check(demo: str, label: str, expected: str, turn, *, detail: bool = True) -> None:
        actual = _verb(turn)
        checks.append(
            {
                "demo": demo,
                "label": label,
                "expected": expected,
                "actual": actual,
                "passed": actual == expected and detail,
                "event": turn.event.to_dict(),
                "action": turn.action.to_dict(),
            }
        )

    def fact(demo: str, label: str, actual: Any, expected: Any = True) -> None:
        checks.append(
            {
                "demo": demo,
                "label": label,
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
            }
        )

    # Demo 1: narration remains silent; direct address gets one response.
    d1 = runtime.create_session(mode="g1")
    check("demo-1", "empty silence", "idle", await _tick(runtime, d1.id, "", True))
    check("demo-1", "unfinished thought", "idle", await _tick(runtime, d1.id, "I was thinking about", False))
    check(
        "demo-1",
        "reported question",
        "idle",
        await _tick(runtime, d1.id, 'I was thinking about when my friend asked, "are you ready?"', True),
    )
    d1_address = await _tick(runtime, d1.id, "Anyway, are you still there?", True)
    check(
        "demo-1",
        "direct address",
        "respond",
        d1_address,
        detail=d1_address.action.target == d1_address.event.index
        and bool(d1_address.action.message.strip()),
    )
    d1_repeat = await _tick(runtime, d1.id, "Anyway, are you still there?", True)
    check("demo-1", "no duplicate answer", "idle", d1_repeat)

    # Demo 2: only the already-demonstrated simple highlight surface is gated.
    d2 = runtime.create_session(mode="g1")
    d2_instruction = "Please highlight every color word as I type."
    d2_ack = await _tick(runtime, d2.id, d2_instruction, True)
    check(
        "demo-2",
        "highlight instruction ack",
        "respond",
        d2_ack,
        detail=d2_ack.action.target == d2_ack.event.index,
    )
    check(
        "demo-2",
        "partial color",
        "idle",
        await _tick(runtime, d2.id, d2_instruction + "\nThe door looked blu", False),
    )
    d2_blue = await _tick(runtime, d2.id, d2_instruction + "\nThe door looked blue.", False)
    check(
        "demo-2",
        "first color",
        "highlight",
        d2_blue,
        detail=d2_blue.action.arguments == {"occurrence": 1, "quote": "blue"},
    )
    d2_red = await _tick(
        runtime,
        d2.id,
        d2_instruction + "\nThe door looked blue, beside a red chair.",
        True,
    )
    check(
        "demo-2",
        "second color",
        "highlight",
        d2_red,
        detail=d2_red.action.arguments == {"occurrence": 1, "quote": "red"},
    )

    # Demo 3: commits stable sense units, not arbitrary punctuation fragments.
    d3 = runtime.create_session(mode="g1")
    d3_instruction = "Translate what I type into Chinese as I go."
    d3_ack = await _tick(runtime, d3.id, d3_instruction, True)
    check(
        "demo-3",
        "translation instruction ack",
        "respond",
        d3_ack,
        detail=d3_ack.action.target == d3_ack.event.index,
    )
    check(
        "demo-3",
        "unstable prefix",
        "idle",
        await _tick(runtime, d3.id, d3_instruction + "\nAfter the meeting", False),
    )
    d3_first = await _tick(runtime, d3.id, d3_instruction + "\nAfter the meeting,", False)
    check(
        "demo-3",
        "first sense unit",
        "translate_commit",
        d3_first,
        detail=d3_first.action.arguments.get("for") == d3_first.event.index
        and bool(str(d3_first.action.arguments.get("message", "")).strip()),
    )
    d3_second = await _tick(
        runtime,
        d3.id,
        d3_instruction + "\nAfter the meeting, we walked to the station.",
        True,
    )
    first_message = str(d3_first.action.arguments.get("message", ""))
    second_message = str(d3_second.action.arguments.get("message", ""))
    check(
        "demo-3",
        "second sense unit",
        "translate_commit",
        d3_second,
        detail=d3_second.action.arguments.get("for") == d3_second.event.index
        and bool(second_message.strip()),
    )
    if _verb(d3_second) == "translate_commit":
        fact(
            "demo-3",
            "second commit does not repeat the first sense unit",
            bool(first_message) and first_message not in second_message,
        )
        fact(
            "demo-3",
            "rendered translation is exactly both append-only deltas",
            runtime.get_session(d3.id).to_dict()["translation"],
            first_message + second_message,
        )

    # Demo 4: delegate and real search are both model decisions. Tool results
    # re-enter the same policy loop; no deterministic action substitution.
    d4 = runtime.create_session(mode="g1")
    combined = (
        "Build a small dashboard about reusable rockets. While that runs, "
        "search for the latest SpaceX valuation."
    )
    check(
        "demo-4",
        "unfinished combined request",
        "idle",
        await _tick(runtime, d4.id, combined[:-11], False),
    )
    d4_delegate = await _tick(runtime, d4.id, combined, True)
    check(
        "demo-4",
        "delegate first",
        "delegate",
        d4_delegate,
        detail="dashboard" in str(d4_delegate.action.arguments.get("task", "")).lower()
        and "rocket" in str(d4_delegate.action.arguments.get("task", "")).lower(),
    )
    await _wait_until(
        lambda: any(
            turn.event.tool_name == "delegate"
            and json.loads(turn.event.content).get("status") == "accepted"
            for turn in d4.history
            if turn.event.source.value == "tool"
        )
    )
    delegate_still_running = not any(
        turn.event.tool_name == "delegate"
        and json.loads(turn.event.content).get("status") in {"completed", "failed"}
        for turn in d4.history
        if turn.event.source.value == "tool"
    )
    foreground_text = combined + "\nWhile that runs, what does reusable mean here?"
    d4_foreground = await _tick(runtime, d4.id, foreground_text, True)
    check(
        "demo-4",
        "foreground conversation while delegate is running",
        "respond",
        d4_foreground,
        detail=delegate_still_running
        and d4_foreground.action.target == d4_foreground.event.index
        and bool(d4_foreground.action.message.strip()),
    )
    await runtime.wait_for_background(d4.id, timeout=30.0)
    tool_turns = [turn for turn in d4.history if turn.event.source.value == "tool"]
    delegate_accepted = next(
        (
            turn
            for turn in tool_turns
            if turn.event.tool_name == "delegate"
            and json.loads(turn.event.content).get("status") == "accepted"
        ),
        None,
    )
    search_result = next(
        (
            turn
            for turn in tool_turns
            if turn.event.tool_name == "web_search"
            and json.loads(turn.event.content).get("status") in {"completed", "failed"}
        ),
        None,
    )
    delegate_terminal = next(
        (
            turn
            for turn in tool_turns
            if turn.event.tool_name == "delegate"
            and json.loads(turn.event.content).get("status") in {"completed", "failed"}
        ),
        None,
    )
    if delegate_accepted is not None:
        search_query = str(delegate_accepted.action.arguments.get("query", "")).lower()
        check(
            "demo-4",
            "search launched after delegate accepted",
            "web_search",
            delegate_accepted,
            detail="spacex" in search_query and "valuation" in search_query,
        )
    else:
        checks.append({"demo": "demo-4", "label": "delegate accepted event", "passed": False})
    if search_result is not None:
        payload = json.loads(search_result.event.content)
        check(
            "demo-4",
            "search result surfaced once and grounded",
            "respond",
            search_result,
            detail=search_result.action.target == search_result.event.index
            and _search_answer_is_grounded(search_result.action.message, payload),
        )
        fact(
            "demo-4",
            "real search returned sources",
            payload.get("status") == "completed" and bool(payload.get("results")),
        )
        terminal_responses = [
            turn
            for turn in d4.history
            if turn.action.valid
            and _verb(turn) == "respond"
            and turn.action.target == search_result.event.index
        ]
        fact("demo-4", "exactly one response targets the search result", len(terminal_responses), 1)
    else:
        checks.append({"demo": "demo-4", "label": "search completion event", "passed": False})

    if delegate_terminal is None:
        checks.append({"demo": "demo-4", "label": "delegate terminal event", "passed": False})
    else:
        delegate_payload = json.loads(delegate_terminal.event.content)
        fact(
            "demo-4",
            "delegate completed with a renderable UI",
            delegate_payload.get("status") == "completed"
            and delegate_terminal.event.job_id in d4.job_specs,
        )
        fact(
            "demo-4",
            "foreground turn happened before delegate completion",
            d4_foreground.event.index < delegate_terminal.event.index,
        )

    await runtime.shutdown()
    demos: dict[str, Any] = {}
    for demo in ("demo-1", "demo-2", "demo-3", "demo-4"):
        selected = [item for item in checks if item["demo"] == demo]
        demos[demo] = {
            "passed": all(item["passed"] for item in selected),
            "checks_passed": sum(item["passed"] for item in selected),
            "checks": len(selected),
        }
    return {
        "schema_version": "g1-v2-live-acceptance-1",
        "policy": args.policy,
        "model": args.model,
        "search": args.search,
        "model_outputs_only": True,
        "session_ids": [d1.id, d2.id, d3.id, d4.id],
        "overall_passed": all(value["passed"] for value in demos.values()),
        "demos": demos,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("local", "tinker"), default="local")
    parser.add_argument("--model", default="")
    parser.add_argument("--search", choices=("ddgs", "demo"), default="ddgs")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "g1-v2" / "live-acceptance.json",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=ROOT / "artifacts" / "g1-v2" / "live-acceptance-trace.jsonl",
    )
    args = parser.parse_args()
    report = asyncio.run(_run(args))
    trace_bytes = args.trace.read_bytes()
    report["trace"] = {
        "path": str(args.trace),
        "sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "turns": sum(1 for line in trace_bytes.splitlines() if line.strip()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
