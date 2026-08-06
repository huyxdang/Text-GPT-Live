"""Pure construction and scoring for the minimum-cost Demo 5 causal probe."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from app.domain import ActionKind
from app.stream import parse_g1_action


MATH_SYSTEM_PROMPT = """You are diagnosing clock-state extraction, not acting in the stream.
For the final event of an explicit recurring reminder, locate the requested interval and the
anchor timestamp: the most recent matching reminder response, or the reminder acknowledgment
when no reminder has fired. Compute elapsed_ms = now_ms - anchor_ms and whether the reminder is
due. Return exactly one JSON object on one line with integer fields interval_ms, anchor_ms,
now_ms, elapsed_ms and boolean field due. Return nothing else."""

ORACLE_SYSTEM_SUFFIX = """

Diagnostic rule: a final <diagnostic_obligation> record, when present, is authoritative.
If it has active=true and due=true, respond now to the current event using its exact message.
If due=false, idle unless a higher-priority current action is independently required."""

PROMPT_RULE_SUFFIX = """

Before choosing idle for an active recurring reminder, locate its acknowledgment or most recent
matching reminder response and compute current event time minus that time; when the elapsed time
is at least the requested interval, respond with the reminder now."""

_EVENT_RE = re.compile(
    r'<stream_event\b[^>]*\bindex="(\d+)"[^>]*\btime="t\+(\d+)ms"[^>]*>'
)
_ACTION_RE = re.compile(r"^<action>.*</action>$")


@dataclass(frozen=True, slots=True)
class ClockState:
    interval_ms: int
    anchor_ms: int
    now_ms: int
    elapsed_ms: int
    due: bool
    current_event_index: int
    message: str

    def diagnostic_payload(self) -> dict[str, Any]:
        return {"active": True, **asdict(self)}


def extract_clock_state(row: Mapping[str, Any]) -> ClockState:
    """Recompute the recurring reminder state from rendered history and row metadata."""

    if row.get("schedule_kind") != "every":
        raise ValueError("Clock-state probes require an explicit recurring reminder.")
    interval_s = row.get("interval_s")
    if not isinstance(interval_s, int) or isinstance(interval_s, bool) or interval_s <= 0:
        raise ValueError("Recurring reminder row has no positive integer interval_s.")
    fire_message = row.get("fire_message")
    if not isinstance(fire_message, str) or not fire_message:
        raise ValueError("Recurring reminder row has no fire_message.")
    cancel_ack = row.get("cancel_ack_text")

    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.endswith("<PREDICT_THIS_ACTION>"):
        raise ValueError("Probe row has no canonical g1 prompt.")
    lines = prompt.splitlines()
    body = lines[:-1]
    if not body or len(body) % 2 == 0:
        raise ValueError("Probe prompt has no final event.")
    history = body[:-1]
    if len(history) % 2:
        raise ValueError("Probe history does not contain event/action pairs.")

    def event_fields(line: str) -> tuple[int, int]:
        match = _EVENT_RE.search(line)
        if not match:
            raise ValueError(f"Event line has no index/time: {line!r}")
        return int(match.group(1)), int(match.group(2))

    anchor_ms: int | None = None
    cancelled = False
    for offset in range(0, len(history), 2):
        event_line, action_line = history[offset : offset + 2]
        if not _ACTION_RE.match(action_line):
            raise ValueError(f"Invalid action line: {action_line!r}")
        action = parse_g1_action(action_line)
        if not action.valid:
            raise ValueError(f"History contains invalid action: {action.diagnostic}")
        if action.kind is not ActionKind.RESPOND or action.message is None:
            continue
        _, time_ms = event_fields(event_line)
        if isinstance(cancel_ack, str) and cancel_ack and action.message == cancel_ack:
            cancelled = True
            continue
        if action.message == fire_message:
            anchor_ms = time_ms
            continue
        if anchor_ms is None:
            anchor_ms = time_ms

    if anchor_ms is None:
        raise ValueError("No reminder acknowledgment or prior fire was found.")
    current_index, now_ms = event_fields(body[-1])
    interval_ms = interval_s * 1_000
    elapsed_ms = now_ms - anchor_ms
    return ClockState(
        interval_ms=interval_ms,
        anchor_ms=anchor_ms,
        now_ms=now_ms,
        elapsed_ms=elapsed_ms,
        due=not cancelled and elapsed_ms >= interval_ms,
        current_event_index=current_index,
        message=fire_message,
    )


def make_oracle_prompt(prompt: str, state: ClockState) -> str:
    marker = "<PREDICT_THIS_ACTION>"
    if not prompt.endswith(marker):
        raise ValueError("Oracle source prompt has no prediction marker.")
    payload = json.dumps(state.diagnostic_payload(), ensure_ascii=False, separators=(",", ":"))
    return prompt[: -len(marker)] + f"<diagnostic_obligation>{payload}</diagnostic_obligation>\n" + marker


def parse_math_output(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    required = ("interval_ms", "anchor_ms", "now_ms", "elapsed_ms", "due")
    if not isinstance(value, dict) or set(value) != set(required):
        return None
    if any(not isinstance(value[key], int) or isinstance(value[key], bool) for key in required[:-1]):
        return None
    if not isinstance(value["due"], bool):
        return None
    return value


def score_math_outputs(
    cases: Sequence[Mapping[str, Any]], outputs: Sequence[str]
) -> dict[str, Any]:
    if len(cases) != len(outputs):
        raise ValueError("Math case/output counts do not match.")
    rows: list[dict[str, Any]] = []
    for case, raw in zip(cases, outputs, strict=True):
        expected = case["clock_state"]
        parsed = parse_math_output(raw)
        field_correct = {
            field: parsed is not None and parsed[field] == expected[field]
            for field in ("interval_ms", "anchor_ms", "now_ms", "elapsed_ms", "due")
        }
        rows.append(
            {
                "case_id": case["case_id"],
                "expected_due": expected["due"],
                "raw": raw,
                "parsed": parsed,
                "format_valid": parsed is not None,
                "field_correct": field_correct,
                "exact": all(field_correct.values()),
            }
        )
    support = len(rows)
    exact = sum(row["exact"] for row in rows)
    due = sum(row["field_correct"]["due"] for row in rows)
    return {
        "summary": {
            "support": support,
            "format_valid": sum(row["format_valid"] for row in rows),
            "exact": exact,
            "exact_rate": round(exact / support, 6),
            "due_correct": due,
            "due_accuracy": round(due / support, 6),
            "gate_passed": exact >= 26,
        },
        "rows": rows,
    }


def score_action_outputs(
    cases: Sequence[Mapping[str, Any]], outputs: Sequence[str], *, gate_kind: str
) -> dict[str, Any]:
    if len(cases) != len(outputs):
        raise ValueError("Action case/output counts do not match.")
    rows: list[dict[str, Any]] = []
    for case, raw in zip(cases, outputs, strict=True):
        parsed = parse_g1_action(raw.strip())
        expected = parse_g1_action(str(case["completion"]))
        valid = parsed.valid
        kind_correct = valid and parsed.kind is expected.kind
        exact = raw.strip() == case["completion"]
        rows.append(
            {
                "case_id": case["case_id"],
                "expected_due": case["clock_state"]["due"],
                "raw": raw,
                "format_valid": valid,
                "kind_correct": kind_correct,
                "exact": exact,
                "diagnostic": parsed.diagnostic,
            }
        )
    fires = [row for row in rows if row["expected_due"]]
    waits = [row for row in rows if not row["expected_due"]]
    fire_correct = sum(row["exact"] for row in fires)
    wait_correct = sum(row["exact"] for row in waits)
    valid = sum(row["format_valid"] for row in rows)
    fire_floor = 13 if gate_kind == "oracle" else 12
    passed = fire_correct >= fire_floor and wait_correct >= 13 and valid == len(rows)
    return {
        "summary": {
            "support": len(rows),
            "format_valid": valid,
            "fire_support": len(fires),
            "fire_exact": fire_correct,
            "wait_support": len(waits),
            "wait_exact": wait_correct,
            "gate_kind": gate_kind,
            "gate_passed": passed,
        },
        "rows": rows,
    }
