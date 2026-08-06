"""Closed-loop behavioural eval for the g1 policy.

The existing app smoke test measures latency and format validity: it asks
whether an action *parses*, never whether it was *right*. Every green g1
number so far has been either teacher-forced (dev cards hand the model a
gold history) or format-only. Neither predicts whether a real typing
session works, because in closed loop the model's own mistakes become the
history it reasons from, and one miss cascades.

This harness drives the real `InteractionRuntime` with scripted typing at
human cadence, then scores the action *class* on every tick against what
the training contract says should happen. Message wording is not scored --
that needs a judge -- but firing at all, firing on the right tick, and
staying silent everywhere else are all scored, and those are the failures
that actually break a session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = "g1-closed-loop-eval-1"
DEFAULT_OUTPUT = ROOT / "artifacts" / "g1-closed-loop" / "report.json"
CHUNK_CHARS = 5
VERB_RE = re.compile(r"<action>\s*(\w+)")

Expect = Literal["idle", "respond", "suggest_edit", "highlight", "delegate", "any"]


@dataclass(frozen=True, slots=True)
class Step:
    """One authored beat. `expect` is graded on the tick the beat completes.

    `force` overwrites what the model actually did with a gold action, in the
    session history only, on the tick this beat completes. That isolates the
    cascade question: if the model fires correctly once a *clean* history is
    restored, the failure is self-consistent silence rather than an inability
    to recognise the trigger.
    """

    text: str = ""
    expect: Expect = "idle"
    kind: Literal["type", "backtrack", "pause"] = "type"
    count: int = 0
    note: str = ""
    force: str = ""


def typing(text: str, expect: Expect = "idle", note: str = "", force: str = "") -> Step:
    return Step(text=text, expect=expect, kind="type", note=note, force=force)


def backtrack(count: int, note: str = "") -> Step:
    return Step(kind="backtrack", count=count, note=note)


def pause(ticks: int, expect: Expect = "idle", note: str = "") -> Step:
    return Step(kind="pause", count=ticks, expect=expect, note=note)


@dataclass(frozen=True, slots=True)
class Session:
    name: str
    demo: str
    steps: tuple[Step, ...]
    why: str = ""


@dataclass
class Tick:
    content: str
    expect: Expect
    note: str
    idle_state: bool = False
    force: str = ""


def compile_ticks(session: Session) -> list[Tick]:
    """Expand a session into per-tick textbox snapshots at human cadence.

    Every tick inside a beat is graded `idle` (the model must stay silent
    while text is still arriving); the beat's own `expect` lands on the tick
    where its final character appears -- exactly how the training compiler
    places a positive card.
    """

    ticks: list[Tick] = []
    current = ""
    for step in session.steps:
        if step.kind == "pause":
            for _ in range(step.count):
                ticks.append(Tick(current, "idle", step.note, idle_state=True))
            if step.expect != "idle" and ticks:
                ticks[-1].expect = step.expect
            continue
        if step.kind == "backtrack":
            for _ in range(max(1, step.count // CHUNK_CHARS)):
                current = current[: max(0, len(current) - CHUNK_CHARS)]
                ticks.append(Tick(current, "idle", step.note))
            continue
        addition = ("\n" if current else "") + step.text
        cursor = 0
        while cursor < len(addition):
            cursor = min(len(addition), cursor + CHUNK_CHARS)
            snapshot = current + addition[:cursor]
            complete = cursor == len(addition)
            ticks.append(
                Tick(
                    snapshot,
                    step.expect if complete else "idle",
                    step.note,
                    force=step.force if complete else "",
                )
            )
        current += addition
    return ticks


def verb_of(raw: str) -> str:
    match = VERB_RE.search(raw or "")
    return match.group(1) if match else "unparseable"


async def run_session(runtime, session: Session) -> dict:
    from app.domain import EventSource, UserState

    ticks = compile_ticks(session)
    handle = runtime.create_session()
    rows: list[dict] = []
    graded = 0
    correct = 0
    false_fires = 0
    missed_fires = 0

    for index, tick in enumerate(ticks, start=1):
        turn = await runtime.process_event(
            handle.id,
            source=EventSource.USER,
            content=tick.content,
            state=UserState.IDLE if tick.idle_state else UserState.ACTIVE,
        )
        actual = verb_of(turn.action.raw_output)
        ok = tick.expect == "any" or actual == tick.expect

        forced = False
        if tick.force:
            from app.stream import parse_g1_action

            gold = parse_g1_action(tick.force)
            if not gold.valid:
                raise ValueError(f"forced action is not canonical: {tick.force}")
            history = runtime.sessions[handle.id].history
            history[-1] = type(history[-1])(
                event=history[-1].event,
                action=gold,
                decision_ms=history[-1].decision_ms,
            )
            forced = True
            print(
                f"  [FORCE] tick={index:3d} history overwritten with {tick.force}",
                flush=True,
            )
        if tick.expect != "any":
            graded += 1
            correct += int(ok)
            if tick.expect == "idle" and actual != "idle":
                false_fires += 1
            elif tick.expect != "idle" and actual == "idle":
                missed_fires += 1
        rows.append(
            {
                "tick": index,
                "expect": tick.expect,
                "actual": actual,
                "ok": ok,
                "decision_ms": turn.decision_ms,
                "note": tick.note,
                "raw": turn.action.raw_output,
                "forced": forced,
                "text_tail": tick.content[-60:],
            }
        )
        if tick.expect != "idle" or actual != "idle":
            mark = "OK " if ok else "MISS"
            print(
                f"  [{mark}] tick={index:3d} expect={tick.expect:12s} "
                f"got={actual:12s} {turn.action.raw_output[:90]}",
                flush=True,
            )

    fire_ticks = [r for r in rows if r["expect"] not in ("idle", "any")]
    return {
        "session": session.name,
        "demo": session.demo,
        "why": session.why,
        "ticks": len(rows),
        "graded": graded,
        "correct": correct,
        "accuracy": round(correct / graded, 4) if graded else None,
        "fire_support": len(fire_ticks),
        "fire_hits": sum(1 for r in fire_ticks if r["ok"]),
        "false_fires": false_fires,
        "missed_fires": missed_fires,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# The scripted sessions. Each one isolates a behaviour that a real take needs.
# ---------------------------------------------------------------------------

SESSIONS: tuple[Session, ...] = (
    Session(
        name="d2-grammar-direct",
        demo="demo-2",
        why="Baseline: instruction phrased like the training data, three planted errors.",
        steps=(
            pause(3),
            typing("Fix any grammar slips as I type.", expect="respond", note="ack"),
            pause(2),
            typing("Today was a beautiful day, it is very hot.", expect="suggest_edit", note="tense"),
            pause(2),
            typing("I west to the cinema.", expect="suggest_edit", note="west->went"),
            pause(2),
            typing("But only seen a few birds.", expect="suggest_edit", note="seen->saw"),
            pause(4),
        ),
    ),
    Session(
        name="d2-grammar-filler-wrapped",
        demo="demo-2",
        why="The instruction that failed live: request buried in conversational filler.",
        steps=(
            pause(3),
            typing("Okay, please fix my grammar as I type, okay?", expect="respond", note="ack"),
            pause(2),
            typing("Today was a beautiful day, it is very hot.", expect="suggest_edit", note="tense"),
            pause(2),
            typing("I west to the cinema.", expect="suggest_edit", note="west->went"),
            pause(4),
        ),
    ),
    Session(
        name="d3-translate-direct",
        demo="demo-3",
        why="Translation with comma clauses: tests commit granularity.",
        steps=(
            pause(3),
            typing("Translate what I type into Chinese as I go.", expect="respond", note="ack"),
            pause(2),
            typing("The market was crowded this morning,", expect="translate_commit", note="comma clause"),
            typing(" and the noodle stall had a long line.", expect="translate_commit", note="sentence clause"),
            pause(4),
        ),
    ),
    Session(
        name="d5-reminder-direct",
        demo="demo-5",
        why="Reminder firing: the behaviour that scored 0.0 recall on dev.",
        steps=(
            pause(3),
            typing("Remind me every 5 seconds to drink water!", expect="respond", note="ack"),
            pause(8, expect="respond", note="first fire due"),
            pause(8, expect="respond", note="second fire due"),
        ),
    ),
    Session(
        name="d2-highlight-natural",
        demo="demo-2",
        why="Highlight with no help: scores 14/14 teacher-forced, so this isolates closed loop.",
        steps=(
            pause(3),
            typing("Please highlight every animal word as I type.", expect="respond", note="ack"),
            pause(2),
            typing("At sunrise, a fox crossed the garden.", expect="highlight", note="fox"),
            pause(2),
            typing("A rabbit hid beneath the rosemary.", expect="highlight", note="rabbit"),
            pause(2),
            typing("A heron stood by the pond.", expect="highlight", note="heron"),
            pause(3),
        ),
    ),
    Session(
        name="d2-highlight-first-forced",
        demo="demo-2",
        why="THE experiment: force only the FIRST highlight into history, then see if the rest follow unaided.",
        steps=(
            pause(3),
            typing("Please highlight every animal word as I type.", expect="respond", note="ack"),
            pause(2),
            typing(
                "At sunrise, a fox crossed the garden.",
                expect="any",
                note="fox (FORCED into history)",
                force='<action>highlight({"occurrence":1,"quote":"fox"})</action>',
            ),
            pause(2),
            typing("A rabbit hid beneath the rosemary.", expect="highlight", note="rabbit (unaided)"),
            pause(2),
            typing("A heron stood by the pond.", expect="highlight", note="heron (unaided)"),
            pause(3),
        ),
    ),
    Session(
        name="d2-edit-first-forced",
        demo="demo-2",
        why="Does the cold-start unlock generalise from highlight to suggest_edit?",
        steps=(
            pause(3),
            typing("Fix any grammar slips as I type.", expect="respond", note="ack"),
            pause(2),
            typing(
                "I west to the cinema.",
                expect="any",
                note="west->went (FORCED into history)",
                force='<action>suggest_edit({"quote":"I west","replacement":"I went"})</action>',
            ),
            pause(2),
            typing("But only seen a few birds.", expect="suggest_edit", note="seen->saw (unaided)"),
            pause(2),
            typing("We was very tired after.", expect="suggest_edit", note="was->were (unaided)"),
            pause(3),
        ),
    ),
    Session(
        name="d2-highlight-in-distribution",
        demo="demo-2",
        why="Trained instruction phrasing + trained category + trained quote words. No seed.",
        steps=(
            pause(3),
            typing("Flag any container word you spot while I write.", expect="respond", note="ack"),
            pause(2),
            typing("The garage was full after the weekend sort out.", note="clean stretch"),
            pause(2),
            typing("A box sat by the door.", expect="highlight", note="box"),
            pause(2),
            typing("Beside it a crate held loose screws.", expect="highlight", note="crate"),
            pause(2),
            typing("An old jar caught the light.", expect="highlight", note="jar"),
            pause(3),
        ),
    ),
    Session(
        name="d2-highlight-colour",
        demo="demo-2",
        why="Colour category fired unaided live. Does it reproduce, and past two marks?",
        steps=(
            pause(3),
            typing("Please highlight everytime I say a color.", expect="respond", note="ack"),
            pause(2),
            typing("I love the color blue", expect="highlight", note="blue"),
            typing(" because it gives me joy, like red", expect="highlight", note="red"),
            typing(". The green door was open too", expect="highlight", note="green"),
            typing(", beside a yellow crate", expect="highlight", note="yellow"),
            pause(3),
        ),
    ),
    Session(
        name="restraint-quoted-question",
        demo="demo-1",
        why="Restraint: quoted question and reported speech must not draw a reply.",
        steps=(
            pause(3),
            typing("So my aunt calls me and asks, \"can you keep a secret?\""),
            pause(3),
            typing("She told me the venue was already booked."),
            pause(3),
            typing("Anyway, are you still there?", expect="respond", note="real address"),
            pause(3),
        ),
    ),
)


async def run(selected: tuple[str, ...] | None) -> dict:
    os.environ.setdefault("POLICY_MODE", "local")
    os.environ.setdefault("POLICY_PROMPT", "g1")
    os.environ.setdefault("STREAM_FORMAT", "g1")
    os.environ.setdefault("SEARCH_MODE", "demo")
    os.environ.setdefault("WRITER_MODE", "demo")
    os.environ.setdefault("TRACE_PATH", "/tmp/smol-g1-closed-loop-trace.jsonl")

    from app.main import runtime

    runtime.policy.warm_up()

    sessions = [s for s in SESSIONS if not selected or s.name in selected]
    started = time.perf_counter()
    results = []
    for session in sessions:
        print(f"\n=== {session.name} ({session.demo}) — {session.why}", flush=True)
        results.append(await run_session(runtime, session))

    graded = sum(r["graded"] for r in results)
    correct = sum(r["correct"] for r in results)
    fire_support = sum(r["fire_support"] for r in results)
    fire_hits = sum(r["fire_hits"] for r in results)
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": os.environ.get("LOCAL_MODEL_PATH", "local MLX"),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "summary": {
            "sessions": len(results),
            "graded_ticks": graded,
            "tick_accuracy": round(correct / graded, 4) if graded else None,
            "fire_support": fire_support,
            "fire_recall": round(fire_hits / fire_support, 4) if fire_support else None,
            "false_fires": sum(r["false_fires"] for r in results),
            "missed_fires": sum(r["missed_fires"] for r in results),
        },
        "sessions": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="session names to run")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--list", action="store_true", help="list sessions and exit")
    args = parser.parse_args()

    if args.list:
        for session in SESSIONS:
            print(f"{session.name:32s} {session.demo}  {session.why}")
        return

    report = asyncio.run(run(tuple(args.only) if args.only else None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    s = report["summary"]
    print("\n=== summary ===")
    print(f"  tick accuracy : {s['tick_accuracy']}  ({s['graded_ticks']} graded)")
    print(f"  fire recall   : {s['fire_recall']}  ({s['fire_support']} expected fires)")
    print(f"  false fires   : {s['false_fires']}")
    print(f"  missed fires  : {s['missed_fires']}")
    for r in report["sessions"]:
        print(
            f"  {r['session']:32s} acc={r['accuracy']} "
            f"fires={r['fire_hits']}/{r['fire_support']} false={r['false_fires']}"
        )
    print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()
