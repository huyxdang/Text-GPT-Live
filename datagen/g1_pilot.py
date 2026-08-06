"""Deterministic g1 pilot cards and coverage guards.

This is deliberately a pilot, not the full ~20k-card generator. It proves the
new header-free contract, silence-aware situation schema, Demo 5 class balance,
and grader wiring before authored data is scaled.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from app.domain import Action, ActionKind, CompletedTurn, EventSource, StreamEvent, UserState
from app.policy import SYSTEM_PROMPT_G1
from app.stream import compile_stream, g1_action_completion, parse_g1_action


G1_SCHEMA_VERSION = "g1"
PILOT_FIRE_FLOOR = 8
PILOT_WAIT_FLOOR = 8
REQUIRED_DEMOS = {"demo-1", "demo-2", "demo-3", "demo-4", "demo-5"}
REQUIRED_EMPTY_KINDS = {"initial", "cleared", "unchanged"}
REQUIRED_TIMING_BOUNDARIES = {"before", "at", "after", "already-fired"}
REQUIRED_COLLISION_PRIORITIES = {
    "direct-address",
    "overdue-schedule",
    "standing-target",
}
APPROVED_REFERENCE_REVIEWS = {"human-reviewed", "codex-reviewed"}


@dataclass(frozen=True)
class PilotCard:
    row: dict[str, Any]
    history: tuple[CompletedTurn, ...]
    current: StreamEvent
    expected: Action


def idle() -> Action:
    return Action(ActionKind.IDLE)


def respond(target: int, message: str) -> Action:
    return Action(ActionKind.RESPOND, target=target, message=message)


def tool(name: str, arguments: dict[str, Any]) -> Action:
    return Action(ActionKind.TOOL, tool_name=name, arguments=arguments)


def user(index: int, content: str, state: UserState, elapsed_ms: int) -> StreamEvent:
    return StreamEvent(index, EventSource.USER, content, state, elapsed_ms)


def tool_event(index: int, content: str, elapsed_ms: int, job_id: str) -> StreamEvent:
    return StreamEvent(
        index,
        EventSource.TOOL,
        content,
        elapsed_ms=elapsed_ms,
        tool_name="delegate",
        job_id=job_id,
    )


def turn(event: StreamEvent, action: Action) -> CompletedTurn:
    return CompletedTurn(event=event, action=action)


def action_label(action: Action) -> str:
    if action.kind is ActionKind.IDLE:
        return "idle"
    if action.kind is ActionKind.RESPOND:
        return "respond"
    return str(action.tool_name)


def make_card(
    *,
    episode: str,
    demo: str,
    situation: str,
    history: Iterable[CompletedTurn],
    current: StreamEvent,
    expected: Action,
    empty_kind: str | None = None,
    obligation: str = "none",
    should_fire: bool = False,
    timing_boundary: str | None = None,
    clause_state: str | None = None,
    chinese_reference: str | None = None,
    reference_review: str | None = None,
    source_text: str | None = None,
    ambiguity_intentional: bool = False,
    collision_priority: str | None = None,
) -> PilotCard:
    history_tuple = tuple(history)
    prompt = compile_stream(list(history_tuple), current, fmt="g1")
    completion = g1_action_completion(expected)
    parsed = parse_g1_action(completion)
    if not parsed.valid:
        raise ValueError(f"{episode}: non-canonical g1 gold: {parsed.diagnostic}")
    row = {
        "schema_version": G1_SCHEMA_VERSION,
        "split": "pilot",
        "episode": episode,
        "demo": demo,
        "situation": situation,
        "bucket": situation,
        "prompt": prompt,
        "completion": completion,
        "expected_class": action_label(expected),
        "current_event_index": current.index,
        "current_content_empty": current.content == "",
        "empty_kind": empty_kind,
        "obligation": obligation,
        "should_fire": should_fire,
        "timing_boundary": timing_boundary,
        "clause_state": clause_state,
        "chinese_reference": chinese_reference,
        "reference_review": reference_review,
        "source_text": source_text,
        "ambiguity_intentional": ambiguity_intentional,
        "collision_priority": collision_priority,
    }
    return PilotCard(row=row, history=history_tuple, current=current, expected=expected)


def _demo_1_cards() -> list[PilotCard]:
    address = user(1, "hey, are you there?", UserState.IDLE, 700)
    answered = respond(1, "I'm here.")
    return [
        make_card(
            episode="g1-pilot-d1-initial-empty",
            demo="demo-1",
            situation="ordinary-silence",
            history=[],
            current=user(1, "", UserState.IDLE, 650),
            expected=idle(),
            empty_kind="initial",
        ),
        make_card(
            episode="g1-pilot-d1-unchanged-empty",
            demo="demo-1",
            situation="ordinary-silence",
            history=[turn(user(1, "", UserState.IDLE, 650), idle())],
            current=user(2, "", UserState.IDLE, 1300),
            expected=idle(),
            empty_kind="unchanged",
        ),
        make_card(
            episode="g1-pilot-d1-cleared-empty",
            demo="demo-1",
            situation="ordinary-silence",
            history=[turn(address, answered)],
            current=user(2, "", UserState.IDLE, 1600),
            expected=idle(),
            empty_kind="cleared",
        ),
        make_card(
            episode="g1-pilot-d1-address",
            demo="demo-1",
            situation="direct-address",
            history=[],
            current=address,
            expected=answered,
        ),
    ]


def _demo_2_cards() -> list[PilotCard]:
    instruction = user(
        1,
        "Fix any grammar slips you catch while I'm typing.",
        UserState.IDLE,
        2100,
    )
    ack = respond(1, "Got it — I'll flag slips as you go.")
    return [
        make_card(
            episode="g1-pilot-d2-empty-under-instruction",
            demo="demo-2",
            situation="ordinary-silence",
            history=[turn(instruction, ack)],
            current=user(2, "", UserState.IDLE, 3200),
            expected=idle(),
            empty_kind="cleared",
        ),
        make_card(
            episode="g1-pilot-d2-suggest",
            demo="demo-2",
            situation="correction-ready",
            history=[turn(instruction, ack)],
            current=user(2, "Last night I tried make ramen.", UserState.ACTIVE, 5400),
            expected=tool(
                "suggest_edit",
                {"quote": "tried make ramen", "replacement": "tried making ramen"},
            ),
        ),
        make_card(
            episode="g1-pilot-d2-highlight",
            demo="demo-2",
            situation="highlight-ready",
            history=[
                turn(
                    user(1, "Highlight every animal I mention.", UserState.IDLE, 1700),
                    respond(1, "Got it — I'll highlight each animal."),
                )
            ],
            current=user(2, "The fox path behind the barn was muddy.", UserState.ACTIVE, 4900),
            expected=tool("highlight", {"occurrence": 1, "quote": "fox"}),
        ),
    ]


def _demo_3_cards() -> list[PilotCard]:
    instruction = user(1, "Translate what I type into Chinese as I go.", UserState.IDLE, 800)
    ack = respond(1, "Okay — I'll translate complete clauses as they arrive.")
    partial = user(2, "The market was crowded this morn", UserState.ACTIVE, 2900)
    first_clause = user(
        3,
        "The market was crowded this morning, and I",
        UserState.ACTIVE,
        3900,
    )
    first_translation = respond(3, "今天早上市场很拥挤，")
    return [
        make_card(
            episode="g1-pilot-d3-empty",
            demo="demo-3",
            situation="ordinary-silence",
            history=[turn(instruction, ack)],
            current=user(2, "", UserState.IDLE, 1500),
            expected=idle(),
            empty_kind="cleared",
        ),
        make_card(
            episode="g1-pilot-d3-partial-clause",
            demo="demo-3",
            situation="translation-partial-clause",
            history=[turn(instruction, ack)],
            current=partial,
            expected=idle(),
            clause_state="partial",
            source_text=partial.content,
        ),
        make_card(
            episode="g1-pilot-d3-first-clause-complete",
            demo="demo-3",
            situation="translation-complete-clause",
            history=[turn(instruction, ack), turn(partial, idle())],
            current=first_clause,
            expected=first_translation,
            clause_state="complete",
            chinese_reference=first_translation.message,
            reference_review="codex-reviewed",
            source_text=first_clause.content,
        ),
        make_card(
            episode="g1-pilot-d3-second-clause-complete",
            demo="demo-3",
            situation="translation-incremental-clause",
            history=[
                turn(instruction, ack),
                turn(partial, idle()),
                turn(first_clause, first_translation),
            ],
            current=user(
                4,
                "The market was crowded this morning, and I left before lunch.",
                UserState.ACTIVE,
                4600,
            ),
            expected=respond(4, "我在午饭前离开了。"),
            clause_state="complete",
            chinese_reference="我在午饭前离开了。",
            reference_review="codex-reviewed",
            source_text="The market was crowded this morning, and I left before lunch.",
        ),
    ]


def _demo_4_cards() -> list[PilotCard]:
    request = user(
        1,
        "Can you build a little dashboard about lighthouses?",
        UserState.IDLE,
        1200,
    )
    delegated = tool(
        "delegate",
        {"task": "generate a UI to visualize lighthouse statistics"},
    )
    accepted = tool_event(2, '{"status":"accepted"}', 1300, "job-1")
    completed = tool_event(
        3,
        '{"status":"completed","task":"generate a UI to visualize lighthouse statistics"}',
        5400,
        "job-1",
    )
    return [
        make_card(
            episode="g1-pilot-d4-delegate",
            demo="demo-4",
            situation="delegate-request",
            history=[],
            current=request,
            expected=delegated,
        ),
        make_card(
            episode="g1-pilot-d4-empty-pending",
            demo="demo-4",
            situation="ordinary-silence",
            history=[turn(request, delegated), turn(accepted, idle())],
            current=user(3, "", UserState.IDLE, 2600),
            expected=idle(),
            empty_kind="cleared",
        ),
        make_card(
            episode="g1-pilot-d4-empty-completed",
            demo="demo-4",
            situation="ordinary-silence",
            history=[
                turn(request, delegated),
                turn(accepted, idle()),
                turn(completed, idle()),
            ],
            current=user(4, "", UserState.IDLE, 6100),
            expected=idle(),
            empty_kind="cleared",
        ),
    ]


def _reminder_cards(interval_ms: int) -> list[PilotCard]:
    episode = f"g1-pilot-d5-{interval_ms}ms"
    request_time = 1000
    request = user(
        1,
        f"Remind me every {interval_ms // 1000} seconds to drink water.",
        UserState.IDLE,
        request_time,
    )
    ack = respond(1, f"Got it — water every {interval_ms // 1000} seconds.")
    due = request_time + interval_ms
    base = [turn(request, ack)]
    cards = [
        make_card(
            episode=f"{episode}-before",
            demo="demo-5",
            situation="reminder-wait",
            history=base,
            current=user(2, "", UserState.IDLE, due - 1),
            expected=idle(),
            empty_kind="cleared",
            obligation="reminder-scheduled",
            timing_boundary="before",
        ),
        make_card(
            episode=f"{episode}-at",
            demo="demo-5",
            situation="reminder-fire",
            history=base,
            current=user(2, "", UserState.IDLE, due),
            expected=respond(2, "Drink water!"),
            empty_kind="cleared",
            obligation="reminder-due",
            should_fire=True,
            timing_boundary="at",
        ),
        make_card(
            episode=f"{episode}-after",
            demo="demo-5",
            situation="reminder-fire",
            history=base,
            current=user(2, "", UserState.IDLE, due + 1),
            expected=respond(2, "Drink water!"),
            empty_kind="cleared",
            obligation="reminder-due",
            should_fire=True,
            timing_boundary="after",
        ),
    ]
    fired_event = user(2, "", UserState.IDLE, due)
    cards.append(
        make_card(
            episode=f"{episode}-already-fired",
            demo="demo-5",
            situation="reminder-wait",
            history=[*base, turn(fired_event, respond(2, "Drink water!"))],
            current=user(3, "", UserState.IDLE, due + 650),
            expected=idle(),
            empty_kind="unchanged",
            obligation="reminder-scheduled",
            timing_boundary="already-fired",
        )
    )
    return cards


def _ordinary_silence_card(demo: str, empty_kind: str) -> PilotCard:
    if empty_kind == "initial":
        history: list[CompletedTurn] = []
        current = user(1, "", UserState.IDLE, 650)
    elif empty_kind == "cleared":
        history = [
            turn(user(1, "Temporary text.", UserState.ACTIVE, 650), idle())
        ]
        current = user(2, "", UserState.IDLE, 1300)
    elif empty_kind == "unchanged":
        history = [turn(user(1, "", UserState.IDLE, 650), idle())]
        current = user(2, "", UserState.IDLE, 1300)
    else:
        raise ValueError(f"Unknown empty kind: {empty_kind}")
    return make_card(
        episode=f"g1-pilot-{demo}-{empty_kind}-empty",
        demo=demo,
        situation="ordinary-silence",
        history=history,
        current=current,
        expected=idle(),
        empty_kind=empty_kind,
    )


def _collision_cards() -> list[PilotCard]:
    reminder = user(1, "Remind me every 3 seconds to drink water.", UserState.IDLE, 1000)
    reminder_ack = respond(1, "Got it — water every 3 seconds.")
    direct_address = user(2, "Are you still there?", UserState.IDLE, 4000)
    direct_reply = respond(2, "Still here.")

    correction_instruction = user(
        2,
        "Fix any grammar slips you catch while I'm typing.",
        UserState.IDLE,
        1500,
    )
    correction_ack = respond(2, "Got it — I'll flag slips as you go.")
    error_text = user(3, "Last night I tried make ramen.", UserState.ACTIVE, 4000)
    reminder_fire = respond(3, "Drink water!")

    return [
        make_card(
            episode="g1-pilot-collision-direct-over-reminder",
            demo="demo-1",
            situation="collision-direct-address-priority",
            history=[turn(reminder, reminder_ack)],
            current=direct_address,
            expected=direct_reply,
            obligation="reminder-overdue",
            collision_priority="direct-address",
        ),
        make_card(
            episode="g1-pilot-collision-deferred-reminder",
            demo="demo-5",
            situation="collision-deferred-reminder",
            history=[turn(reminder, reminder_ack), turn(direct_address, direct_reply)],
            current=user(3, "Are you still there?", UserState.IDLE, 4650),
            expected=respond(3, "Drink water!"),
            obligation="reminder-due",
            should_fire=True,
            collision_priority="overdue-schedule",
        ),
        make_card(
            episode="g1-pilot-collision-reminder-over-standing-target",
            demo="demo-5",
            situation="collision-reminder-priority",
            history=[
                turn(reminder, reminder_ack),
                turn(correction_instruction, correction_ack),
            ],
            current=error_text,
            expected=reminder_fire,
            obligation="reminder-due-and-correction-ready",
            should_fire=True,
            collision_priority="overdue-schedule",
        ),
        make_card(
            episode="g1-pilot-collision-deferred-standing-target",
            demo="demo-2",
            situation="collision-deferred-standing-target",
            history=[
                turn(reminder, reminder_ack),
                turn(correction_instruction, correction_ack),
                turn(error_text, reminder_fire),
            ],
            current=user(4, error_text.content, UserState.IDLE, 4650),
            expected=tool(
                "suggest_edit",
                {"quote": "tried make ramen", "replacement": "tried making ramen"},
            ),
            obligation="correction-ready",
            collision_priority="standing-target",
        ),
    ]


def build_pilot_cards() -> list[PilotCard]:
    cards = (
        _demo_1_cards()
        + _demo_2_cards()
        + _demo_3_cards()
        + _demo_4_cards()
        + _collision_cards()
    )
    for interval_ms in (3000, 5000, 7000, 9000):
        cards.extend(_reminder_cards(interval_ms))
    existing = {
        (card.row["demo"], card.row["empty_kind"])
        for card in cards
        if card.row["situation"] == "ordinary-silence" and card.row["empty_kind"]
    }
    for demo in sorted(REQUIRED_DEMOS):
        for empty_kind in sorted(REQUIRED_EMPTY_KINDS):
            if (demo, empty_kind) not in existing:
                cards.append(_ordinary_silence_card(demo, empty_kind))
    return cards


_VAGUE_TIME = re.compile(
    r"\bat\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2})\b",
    re.IGNORECASE,
)
_TIME_QUALIFIER = re.compile(
    r"\b(?:a\.?m\.?|p\.?m\.?|morning|afternoon|evening|tonight|tomorrow|today)\b",
    re.IGNORECASE,
)
_VAGUE_RELATIVE_TIME = re.compile(
    r"\b(?:later|soon|sometime|in a while|after a while)\b",
    re.IGNORECASE,
)
_VAGUE_QUANTITY = re.compile(
    r"\b(?:a few|a couple(?:\s+of)?|several|some|many|a bunch(?:\s+of)?)\b",
    re.IGNORECASE,
)


def has_vague_time_reference(text: str) -> bool:
    if _VAGUE_RELATIVE_TIME.search(text):
        return True
    for match in _VAGUE_TIME.finditer(text):
        nearby = text[match.start() : match.end() + 24]
        if not _TIME_QUALIFIER.search(nearby):
            return True
    return False


def has_vague_quantity_reference(text: str) -> bool:
    return bool(_VAGUE_QUANTITY.search(text))


def _user_texts(card: PilotCard) -> list[str]:
    texts = [
        completed.event.content
        for completed in card.history
        if completed.event.source is EventSource.USER
    ]
    if card.current.source is EventSource.USER:
        texts.append(card.current.content)
    return texts


def _actual_empty_kind(card: PilotCard) -> str | None:
    if card.current.source is not EventSource.USER or card.current.content != "":
        return None
    previous_user_events = [
        completed.event
        for completed in card.history
        if completed.event.source is EventSource.USER
    ]
    if not previous_user_events:
        return "initial"
    return "unchanged" if previous_user_events[-1].content == "" else "cleared"


def validate_pilot_coverage(
    cards: Iterable[PilotCard],
    *,
    min_fire_examples: int = PILOT_FIRE_FLOOR,
    min_wait_examples: int = PILOT_WAIT_FLOOR,
    require_reference_review: bool = False,
) -> dict[str, Any]:
    items = list(cards)
    rows = [card.row for card in items]
    errors: list[str] = []
    warnings: list[str] = []
    empty_matrix = {
        demo: {
            row["empty_kind"]
            for row in rows
            if (
                row["demo"] == demo
                and row["situation"] == "ordinary-silence"
                and row["current_content_empty"]
                and row["empty_kind"]
            )
        }
        for demo in REQUIRED_DEMOS
    }
    empty_demos = {demo for demo, kinds in empty_matrix.items() if kinds}
    empty_kinds = {kind for kinds in empty_matrix.values() for kind in kinds}
    fire_rows = [row for row in rows if row["should_fire"]]
    wait_rows = [
        row
        for row in rows
        if row["demo"] == "demo-5" and row["situation"] == "reminder-wait"
    ]
    boundaries = {
        row["timing_boundary"] for row in rows if row["timing_boundary"] is not None
    }
    collision_priorities = {
        row["collision_priority"] for row in rows if row["collision_priority"] is not None
    }

    if empty_demos != REQUIRED_DEMOS:
        errors.append(f"empty-text demo coverage is {sorted(empty_demos)}, expected all five demos")
    for demo, kinds in sorted(empty_matrix.items()):
        missing = REQUIRED_EMPTY_KINDS - kinds
        if missing:
            errors.append(f"{demo} is missing empty-text kinds: {sorted(missing)}")
    if len(fire_rows) < min_fire_examples:
        errors.append(f"should-fire support {len(fire_rows)} is below floor {min_fire_examples}")
    if len(wait_rows) < min_wait_examples:
        errors.append(f"reminder-wait support {len(wait_rows)} is below floor {min_wait_examples}")
    if not REQUIRED_TIMING_BOUNDARIES.issubset(boundaries):
        errors.append(
            f"missing timing boundaries: {sorted(REQUIRED_TIMING_BOUNDARIES - boundaries)}"
        )
    if not REQUIRED_COLLISION_PRIORITIES.issubset(collision_priorities):
        errors.append(
            "missing collision priorities: "
            f"{sorted(REQUIRED_COLLISION_PRIORITIES - collision_priorities)}"
        )

    for card, row in zip(items, rows, strict=True):
        action = parse_g1_action(row["completion"])
        if not action.valid:
            errors.append(f"{row['episode']}: invalid completion: {action.diagnostic}")
        actual_empty_kind = _actual_empty_kind(card)
        if row["empty_kind"] != actual_empty_kind:
            errors.append(
                f"{row['episode']}: empty_kind {row['empty_kind']!r} "
                f"does not match stream history {actual_empty_kind!r}"
            )
        if (
            row["current_content_empty"]
            and row["obligation"] == "none"
            and row["expected_class"] != "idle"
        ):
            errors.append(f"{row['episode']}: ordinary empty text must grade idle")
        if row["should_fire"] and row["expected_class"] != "respond":
            errors.append(f"{row['episode']}: should-fire row must grade respond")
        ambiguous_texts = [
            text
            for text in _user_texts(card)
            if has_vague_time_reference(text) or has_vague_quantity_reference(text)
        ]
        if ambiguous_texts and not row.get("ambiguity_intentional"):
            errors.append(
                f"{row['episode']}: vague time or quantity reference is not marked intentional"
            )

    clause_states = {row["clause_state"] for row in rows if row["clause_state"]}
    if not {"partial", "complete"}.issubset(clause_states):
        errors.append("translation pilot must cover both partial and complete clauses")
    for row in rows:
        if not row.get("chinese_reference"):
            continue
        status = row.get("reference_review")
        if status not in APPROVED_REFERENCE_REVIEWS:
            message = f"{row['episode']}: Chinese reference is pending language review"
            (errors if require_reference_review else warnings).append(message)

    action_counts = Counter(row["expected_class"] for row in rows)
    situation_counts = Counter(row["situation"] for row in rows)
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "rows": len(rows),
            "actions": dict(sorted(action_counts.items())),
            "situations": dict(sorted(situation_counts.items())),
            "should_fire": len(fire_rows),
            "reminder_wait": len(wait_rows),
            "empty_demos": sorted(empty_demos),
            "empty_kinds": sorted(empty_kinds),
            "empty_matrix": {
                demo: sorted(kinds) for demo, kinds in sorted(empty_matrix.items())
            },
            "timing_boundaries": sorted(boundaries),
            "collision_priorities": sorted(collision_priorities),
        },
        "system_prompt": SYSTEM_PROMPT_G1,
    }
