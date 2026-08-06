"""Deterministic compiler for Demo 4 (Talk-while-task) g1 situations.

Demo 4 teaches the ``delegate`` async-tool discipline: fire once on a
completed visual request, then hold silence through acceptance, progress
narration, and completion/failure -- the window is the announcement, not the
model's voice.

Unlike Demos 1-3, agents do **not** author whole episodes here.  They author
two flat banks of words (see ``datagen.g1_authored_demo4``):

* ``requests`` -- varied visual-request phrasings, each with the ``task``
  string that should ride inside ``delegate({"task": ...})``.
* ``progress_pairs`` -- progress-question / honest-reply pairs (``kind":
  "check"``) and receipt-nudge phrasings that get silently ignored while a job
  is already running (``kind": "nudge"``).

This module is what turns those two banks into ~130 structurally different
job-dialog episodes: ``expand_demo4_episodes`` cross-products the banks into
episode plans (which request, which progress content, success vs failure,
episode skeleton -- every one of those choices is a hash of the episode id,
never an RNG draw), and ``compile_demo4_record`` renders one plan into the
live app's full-textbox + tool-event stream, exactly as ``compile_stream(...,
fmt="g1")`` and ``g1_action_completion`` would at serving time.

A few short phrase banks below (acknowledgement wording, failure narration,
narration filler) are **generator-owned, not authored**: the brief is that
"authors supply words; timing is yours," and these are the connective tissue
the splicer needs to keep a pending job's dialog feeling alive without
inventing new authored content per episode. They are deliberately small,
generic, and documented as such -- production content variety is expected to
come from the two authored banks, not from these filler pools.

Tool-event payloads are built with the exact key order the live runtime uses
(``app/runtime.py:_run_job`` / ``_inject_job_event``): insertion order, not
alphabetical -- ``{"status":"accepted"}``, ``{"status":"completed","task":...}``,
``{"status":"failed","task":...,"error":...}``. ``job_id`` lives on the
``<stream_event>`` tag only, never inside the payload, matching the
2026-07-31 one-place decision recorded in the spec.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.domain import Action, ActionKind, CompletedTurn, EventSource, StreamEvent, UserState
from app.stream import compile_stream, g1_action_completion, parse_g1_action


G1_SCHEMA_VERSION = "g1"
DEMO = "demo-4"

DEMO4_PROVENANCE_FIELDS = (
    "persona",
    "domain",
    "register",
    "author_slot",
    "author_model",
    "author_tranche",
)

CONTENT_KINDS = ("check", "nudge", "narration")
OUTCOMES = ("success", "failure")

DEMO4_ROLES = (
    "request-before",
    "request-positive",
    "accepted-idle",
    "ack-positive",
    "check-before",
    "check-positive",
    "check-after",
    "nudge-idle",
    "narration-idle",
    "completed-idle",
    "failed-idle",
    "failure-check-positive",
    "empty-initial",
    "empty-unchanged",
    "empty-cleared",
)
TRAP_ROLES = ("nudge-idle", "narration-idle")
EMPTY_KINDS = ("initial", "unchanged", "cleared")
NEIGHBOR_ROLES = ("request-before", "check-before", "check-after")

PAUSE_MS = {
    "none": (540, 820),
    "short": (1_000, 3_000),
    "medium": (4_000, 9_000),
    "long": (10_000, 30_000),
}

# Demo 3's insight applied here: a constant episode shape teaches structure as
# furniture. Opening/closing skeleton is a hash of the episode id.
OPENING_SHAPES = ("immediate", "idle", "idle-idle", "idle-long-idle")
CLOSING_SHAPES = ("clear", "clear-hold", "hold", "hold-clear")

# Job pendency, generated not authored (spec, "Timing requirements (global)").
PENDENCY_SUCCESS_RANGE = (2, 8)
PENDENCY_FAILURE_RANGE = (5, 10)

MIN_REQUEST_CHARS = 16
MIN_QUESTION_CHARS = 8
MAX_REQUEST_CHARS = 90
# Progress-window text (check/nudge/narration/failure-follow-up questions) is
# typed *inside* the sampled pendency budget (2-8 ticks success, 5-10 ticks
# failure) alongside the one-shot ack beat, so it must stay short enough that
# even a worst-case (all-4-char) chunking cannot blow the budget: 16 chars is
# at most 5 ticks, leaving headroom for the 1-2 tick ack beat.
MAX_QUESTION_CHARS = 16
MAX_REPLY_CHARS = 90
MAX_TASK_CHARS = 90

# -- generator-owned filler banks -------------------------------------------
# Deliberately small and generic. These are connective tissue the splicer uses
# to keep a pending job's dialog alive between authored beats; they are not a
# substitute for the two authored banks and contribute no authored-content
# variety of their own. Documented in the Demo 4 report as a scope choice.
ACK_TEMPLATES: tuple[str, ...] = (
    "On it — I'll have that ready shortly.",
    "Got it, working on that now.",
    "Sure thing — building that for you now.",
    "Starting that now, one sec.",
    "Okay, kicking that off now.",
    "Working on it — shouldn't take long.",
    "Sure — I'm on that now.",
    "Alright, that's in progress now.",
    "Got it — putting that together now.",
    "On it now, give me a moment.",
)
BEAT_FILLERS: tuple[str, ...] = (
    "cool, and",
    "nice, so",
    "okay, and",
    "alright",
    "got it",
    "sure, so",
    "nice one",
    "cool then",
    "okay then",
    "right, so",
)
NARRATION_FILLER: tuple[str, ...] = (
    "anyway, so",
    "let me think",
    "one sec",
    "moving on",
    "hold on",
    "let's see",
    "hmm, okay",
    "back to it",
    "one more sec",
    "okay, next",
    "right, so",
    "let me check",
    "just a sec",
    "one moment",
    "anyway",
    "okay then",
)
FAILURE_REPLY_TEMPLATES: tuple[str, ...] = (
    "That one didn't go through — want me to try again?",
    "Hmm, that failed on my end. Should I retry it?",
    "That didn't work out — I can give it another shot if you want.",
    "Sorry, that one failed. Want me to try it again?",
    "No luck on that one — happy to retry if you'd like.",
    "That job failed on my side. I can re-run it if you want.",
    "Didn't work that time — want another attempt?",
    "That one broke on my end, sorry. I can try again.",
)


# --------------------------------------------------------------------------
# deterministic primitives (hashed over UTF-8 bytes; no RNG anywhere)
# --------------------------------------------------------------------------


def _digest_bytes(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _stable_rank(value: str) -> tuple[bytes, str]:
    return (_digest_bytes(value), value)


def _stable_int(key: str, low: int, high: int) -> int:
    if low > high:
        raise ValueError("low must not exceed high")
    span = high - low + 1
    return low + int.from_bytes(_digest_bytes(key)[:8], "big") % span


def _stable_choice(key: str, options: Sequence[str]) -> str:
    if not options:
        raise ValueError("cannot choose from an empty option list")
    return options[_stable_int(key, 0, len(options) - 1)]


def opening_shape(record_id: str) -> str:
    return _stable_choice(f"{record_id}:opening-shape", OPENING_SHAPES)


def closing_shape(record_id: str) -> str:
    return _stable_choice(f"{record_id}:closing-shape", CLOSING_SHAPES)


def _chunk_sizes(length: int, key: str) -> tuple[int, ...]:
    """Partition text into deterministic 4-7 character typing deltas."""

    if length <= 0:
        return ()
    if length < 4:
        return (length,)
    sizes: list[int] = []
    remaining = length
    step = 0
    while remaining:
        feasible = [
            size
            for size in range(4, 8)
            if size <= remaining and (remaining - size == 0 or remaining - size >= 4)
        ]
        if not feasible:
            raise ValueError(f"cannot partition {length} characters into 4-7 character ticks")
        choice = _stable_int(f"{key}:chunk:{step}", 0, len(feasible) - 1)
        size = feasible[choice]
        sizes.append(size)
        remaining -= size
        step += 1
    return tuple(sizes)


def _pause_delta(record_id: str, key: str, pause_after: str) -> int:
    if pause_after not in PAUSE_MS:
        raise ValueError(f"{record_id}: invalid pause_after {pause_after!r}")
    low, high = PAUSE_MS[pause_after]
    return _stable_int(f"{record_id}:{key}:pause", low, high)


def _typing_delta(record_id: str, key: str, tick_index: int) -> int:
    return _stable_int(f"{record_id}:{key}:tick:{tick_index}:time", 520, 820)


def _idle() -> Action:
    return Action(ActionKind.IDLE)


def _respond(target: int, message: str) -> Action:
    return Action(ActionKind.RESPOND, target=target, message=message)


def _delegate(task: str) -> Action:
    return Action(ActionKind.TOOL, tool_name="delegate", arguments={"task": task})


def _user_event(index: int, content: str, state: UserState, elapsed_ms: int) -> StreamEvent:
    return StreamEvent(index, EventSource.USER, content, state, elapsed_ms)


def _tool_event(
    index: int, content: str, elapsed_ms: int, *, job_id: str
) -> StreamEvent:
    return StreamEvent(
        index,
        EventSource.TOOL,
        content,
        state=None,
        elapsed_ms=elapsed_ms,
        tool_name="delegate",
        job_id=job_id,
    )


def _turn(event: StreamEvent, action: Action) -> CompletedTurn:
    return CompletedTurn(event=event, action=action)


def _tool_payload(pairs: Sequence[tuple[str, str]]) -> str:
    """Serialize a tool-event payload with the runtime's exact key order.

    ``app/runtime.py`` builds these dicts by literal insertion order and
    serializes with ``json.dumps(..., separators=(",", ":"))`` -- no
    ``sort_keys``. Matching that byte-for-byte (not the alphabetical order
    ``app/stream.py`` uses for action arguments) is what "train equals serve"
    means for tool events.
    """

    return json.dumps(dict(pairs), ensure_ascii=False, separators=(",", ":"))


def accepted_payload() -> str:
    return _tool_payload((("status", "accepted"),))


def completed_payload(task: str) -> str:
    return _tool_payload((("status", "completed"), ("task", task)))


def failed_payload(task: str, error: str) -> str:
    return _tool_payload((("status", "failed"), ("task", task), ("error", error)))


# --------------------------------------------------------------------------
# episode expansion: the two authored banks -> ~N episode plans
# --------------------------------------------------------------------------


def _require_str(value: Any, context: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {field} must be a non-empty string")
    return value


def expand_demo4_episodes(
    requests: Sequence[Mapping[str, Any]],
    progress_pairs: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    """Cross the two authored banks into ``count`` deterministic episode plans.

    This is the fan-out the spec asks for: "a few hundred authored lines fan
    out into structurally different episodes." Requests are cycled
    round-robin (sorted by id, so the expansion is a pure function of the
    bank contents); each full cycle bumps a variant counter that seeds every
    other per-episode choice (content kind, job outcome, which progress items
    are used), so the same request phrasing produces different episodes each
    time it recurs. Nothing here is random: every choice is
    ``_stable_int``/``_stable_choice`` over the episode id.
    """

    if count <= 0:
        raise ValueError("Demo 4 episode count must be positive")
    if not requests:
        raise ValueError("Demo 4 requires at least one authored request")
    checks = [item for item in progress_pairs if item.get("kind") == "check"]
    nudges = [item for item in progress_pairs if item.get("kind") == "nudge"]
    if not checks:
        raise ValueError("Demo 4 requires at least one 'check' progress pair")
    if not nudges:
        raise ValueError("Demo 4 requires at least one 'nudge' progress pair")

    ordered_requests = sorted(requests, key=lambda item: str(item.get("id")))
    ordered_checks = sorted(checks, key=lambda item: str(item.get("id")))
    ordered_nudges = sorted(nudges, key=lambda item: str(item.get("id")))

    episodes: list[dict[str, Any]] = []
    for index in range(count):
        request = ordered_requests[index % len(ordered_requests)]
        variant = index // len(ordered_requests)
        request_id = _require_str(request.get("id"), f"request[{index}]", "id")
        episode_id = f"demo4-{request_id}-v{variant}"

        content_kind = CONTENT_KINDS[index % 3]
        outcome = "failure" if index % 5 == 0 else "success"

        check_item = ordered_checks[_stable_int(f"{episode_id}:check-pick", 0, len(ordered_checks) - 1)]
        nudge_item = ordered_nudges[_stable_int(f"{episode_id}:nudge-pick", 0, len(ordered_nudges) - 1)]
        failure_item = ordered_checks[
            _stable_int(f"{episode_id}:failure-pick", 0, len(ordered_checks) - 1)
        ]

        plan: dict[str, Any] = {
            "id": episode_id,
            "persona": request.get("persona"),
            "domain": request.get("domain"),
            "register": request.get("register"),
            "author_slot": request.get("author_slot"),
            "author_model": request.get("author_model"),
            "author_tranche": request.get("author_tranche"),
            "request_id": request_id,
            "request_text": request.get("text"),
            "request_task": request.get("task"),
            "content_kind": content_kind,
            "outcome": outcome,
        }
        if content_kind == "check":
            plan["check_id"] = check_item.get("id")
            plan["check_question"] = check_item.get("question")
            plan["check_reply"] = check_item.get("reply")
        elif content_kind == "nudge":
            plan["nudge_id"] = nudge_item.get("id")
            plan["nudge_question"] = nudge_item.get("question")
        if outcome == "failure":
            plan["failure_id"] = failure_item.get("id")
            plan["failure_question"] = failure_item.get("question")
        episodes.append(plan)
    return episodes


# --------------------------------------------------------------------------
# candidates and compiled records
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Demo4Candidate:
    candidate_id: str
    record_id: str
    turn_offset: int
    role: str
    pause_after: str | None = None
    empty_kind: str | None = None

    def __post_init__(self) -> None:
        if self.role not in DEMO4_ROLES:
            raise ValueError(f"unknown Demo 4 candidate role: {self.role!r}")


@dataclass(frozen=True, slots=True)
class CompiledDemo4Record:
    record_id: str
    split: str
    turns: tuple[CompletedTurn, ...]
    candidates: tuple[Demo4Candidate, ...]
    opening_shape: str
    closing_shape: str
    content_kind: str
    outcome: str
    request_task: str
    job_id: str
    source_metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Demo4Build:
    records: tuple[CompiledDemo4Record, ...]
    selected: tuple[Demo4Candidate, ...]
    rows: tuple[dict[str, Any], ...]
    coverage: dict[str, Any]


def _validated_provenance_metadata(
    record: Mapping[str, Any], *, context: str
) -> tuple[tuple[str, str], ...]:
    metadata: list[tuple[str, str]] = []
    for field in DEMO4_PROVENANCE_FIELDS:
        if field not in record:
            raise ValueError(f"{context}: missing required provenance field {field!r}")
        value = record[field]
        if value is None or not str(value).strip():
            raise ValueError(f"{context}: provenance field {field!r} must be non-blank")
        metadata.append((field, str(value)))
    return tuple(sorted(metadata))


def assign_demo4_splits(
    records: Iterable[Mapping[str, Any]], *, dev_fraction: float = 0.1
) -> dict[str, str]:
    items = list(records)
    if not 0 <= dev_fraction <= 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    record_ids = [record.get("id") for record in items]
    if any(not isinstance(record_id, str) or not record_id for record_id in record_ids):
        raise ValueError("every Demo 4 record requires a non-empty string id")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Demo 4 record ids must be unique")
    dev_count = round(len(record_ids) * dev_fraction)
    ranked = sorted(record_ids, key=lambda record_id: _stable_rank(f"split:{record_id}"))
    dev_ids = set(ranked[:dev_count])
    return {record_id: "dev" if record_id in dev_ids else "train" for record_id in record_ids}


def _candidate(
    record_id: str,
    turn_offset: int,
    role: str,
    *,
    key: str,
    pause_after: str | None = None,
    empty_kind: str | None = None,
) -> Demo4Candidate:
    return Demo4Candidate(
        candidate_id=f"{record_id}:{role}:{key}",
        record_id=record_id,
        turn_offset=turn_offset,
        role=role,
        pause_after=pause_after,
        empty_kind=empty_kind,
    )


def compile_demo4_record(
    record: Mapping[str, Any], *, split: str = "train"
) -> CompiledDemo4Record:
    """Splice one episode plan into the full-textbox + tool-event stream."""

    record_id = _require_str(record.get("id"), "Demo 4 record", "id")
    if split not in {"train", "dev"}:
        raise ValueError("Demo 4 split must be 'train' or 'dev'")
    source_metadata = _validated_provenance_metadata(record, context=record_id)

    request_text = _require_str(record.get("request_text"), record_id, "request_text")
    task = _require_str(record.get("request_task"), record_id, "request_task")
    if not MIN_REQUEST_CHARS <= len(request_text) <= MAX_REQUEST_CHARS:
        raise ValueError(
            f"{record_id}: request_text must be {MIN_REQUEST_CHARS}-{MAX_REQUEST_CHARS} "
            f"characters; got {len(request_text)}"
        )
    if len(task) > MAX_TASK_CHARS:
        raise ValueError(f"{record_id}: request_task must be at most {MAX_TASK_CHARS} characters")
    content_kind = record.get("content_kind")
    if content_kind not in CONTENT_KINDS:
        raise ValueError(f"{record_id}: invalid content_kind {content_kind!r}")
    outcome = record.get("outcome")
    if outcome not in OUTCOMES:
        raise ValueError(f"{record_id}: invalid outcome {outcome!r}")

    open_shape = opening_shape(record_id)
    close_shape = closing_shape(record_id)
    job_id = "job-1"

    turns: list[CompletedTurn] = []
    candidates: list[Demo4Candidate] = []
    elapsed_ms = _stable_int(f"{record_id}:initial-time", 520, 780)

    def emit_user(content: str, state: UserState, action: Action) -> int:
        offset = len(turns)
        turns.append(_turn(_user_event(offset + 1, content, state, elapsed_ms), action))
        return offset

    def emit_tool(content: str, action: Action) -> int:
        offset = len(turns)
        turns.append(
            _turn(_tool_event(offset + 1, content, elapsed_ms, job_id=job_id), action)
        )
        return offset

    # -- opening skeleton --------------------------------------------------
    if open_shape != "immediate":
        offset = emit_user("", UserState.IDLE, _idle())
        candidates.append(
            _candidate(record_id, offset, "empty-initial", key="initial", empty_kind="initial")
        )
        if open_shape in {"idle-idle", "idle-long-idle"}:
            gap = "long" if open_shape == "idle-long-idle" else "none"
            elapsed_ms += _pause_delta(record_id, "open-hold", gap)
            offset = emit_user("", UserState.IDLE, _idle())
            candidates.append(
                _candidate(
                    record_id, offset, "empty-unchanged", key="open-unchanged", empty_kind="unchanged"
                )
            )

    # -- the visual request, typed at human cadence; fires on completion ---
    current_text = ""
    sizes = _chunk_sizes(len(request_text), f"{record_id}:request")
    cursor = 0
    request_offsets: list[int] = []
    for tick_index, size in enumerate(sizes):
        cursor += size
        elapsed_ms += _typing_delta(record_id, "request", tick_index)
        complete = cursor == len(request_text)
        action = _delegate(task) if complete else _idle()
        request_offsets.append(emit_user(request_text[:cursor], UserState.ACTIVE, action))
    if len(request_offsets) < 2:
        raise ValueError(f"{record_id}: request is too short for a before-neighbour")
    current_text = request_text
    candidates.append(
        _candidate(record_id, request_offsets[-2], "request-before", key="request")
    )
    fire_offset = request_offsets[-1]
    candidates.append(_candidate(record_id, fire_offset, "request-positive", key="request"))

    # -- acceptance: lands almost immediately, never announced -------------
    elapsed_ms += _stable_int(f"{record_id}:accept-gap", 120, 420)
    accepted_offset = emit_tool(accepted_payload(), _idle())
    candidates.append(_candidate(record_id, accepted_offset, "accepted-idle", key="accept"))

    # -- the one-shot ack: the first user tick after acceptance -------------
    beat = _stable_choice(f"{record_id}:beat", BEAT_FILLERS)
    ack_addition = " " + beat
    elapsed_ms += _typing_delta(record_id, "beat", 0)
    sizes = _chunk_sizes(len(ack_addition), f"{record_id}:beat")
    cursor = 0
    ack_ticks = 0
    ack_offset = -1
    for tick_index, size in enumerate(sizes):
        cursor += size
        if tick_index:
            elapsed_ms += _typing_delta(record_id, "beat", tick_index)
        complete = cursor == len(ack_addition)
        ack_message = _stable_choice(f"{record_id}:ack", ACK_TEMPLATES)
        snapshot = current_text + ack_addition[:cursor]
        offset = len(turns)
        action = _respond(offset + 1, ack_message) if complete else _idle()
        emit_user(snapshot, UserState.ACTIVE, action)
        ack_ticks += 1
        if complete:
            ack_offset = offset
    current_text += ack_addition
    candidates.append(_candidate(record_id, ack_offset, "ack-positive", key="ack"))

    # -- pending-window content: check / nudge / narration ------------------
    content_ticks = 0
    check_offset: int | None = None
    if content_kind == "check":
        question = _require_str(record.get("check_question"), record_id, "check_question")
        reply = _require_str(record.get("check_reply"), record_id, "check_reply")
        if not MIN_QUESTION_CHARS <= len(question) <= MAX_QUESTION_CHARS:
            raise ValueError(f"{record_id}: check_question length out of bounds")
        if len(reply) > MAX_REPLY_CHARS:
            raise ValueError(f"{record_id}: check_reply too long")
        addition = " " + question
        sizes = _chunk_sizes(len(addition), f"{record_id}:check")
        cursor = 0
        offsets: list[int] = []
        for tick_index, size in enumerate(sizes):
            cursor += size
            elapsed_ms += _typing_delta(record_id, "check", tick_index)
            complete = cursor == len(addition)
            snapshot = current_text + addition[:cursor]
            offset = len(turns)
            action = _respond(offset + 1, reply) if complete else _idle()
            emit_user(snapshot, UserState.ACTIVE, action)
            offsets.append(offset)
            content_ticks += 1
        current_text += addition
        if len(offsets) < 2:
            raise ValueError(f"{record_id}: check_question is too short for a before-neighbour")
        candidates.append(_candidate(record_id, offsets[-2], "check-before", key="check"))
        candidates.append(_candidate(record_id, offsets[-1], "check-positive", key="check"))
        check_offset = offsets[-1]
    elif content_kind == "nudge":
        question = _require_str(record.get("nudge_question"), record_id, "nudge_question")
        if not MIN_QUESTION_CHARS <= len(question) <= MAX_QUESTION_CHARS:
            raise ValueError(f"{record_id}: nudge_question length out of bounds")
        addition = " " + question
        sizes = _chunk_sizes(len(addition), f"{record_id}:nudge")
        cursor = 0
        last_offset = -1
        for tick_index, size in enumerate(sizes):
            cursor += size
            elapsed_ms += _typing_delta(record_id, "nudge", tick_index)
            complete = cursor == len(addition)
            snapshot = current_text + addition[:cursor]
            offset = len(turns)
            emit_user(snapshot, UserState.ACTIVE, _idle())
            content_ticks += 1
            if complete:
                last_offset = offset
        current_text += addition
        candidates.append(_candidate(record_id, last_offset, "nudge-idle", key="nudge"))
    else:  # narration
        filler = _stable_choice(f"{record_id}:narration", NARRATION_FILLER)
        addition = " " + filler
        sizes = _chunk_sizes(len(addition), f"{record_id}:narration")
        cursor = 0
        last_offset = -1
        for tick_index, size in enumerate(sizes):
            cursor += size
            elapsed_ms += _typing_delta(record_id, "narration", tick_index)
            complete = cursor == len(addition)
            snapshot = current_text + addition[:cursor]
            offset = len(turns)
            emit_user(snapshot, UserState.ACTIVE, _idle())
            content_ticks += 1
            if complete:
                last_offset = offset
        current_text += addition
        candidates.append(_candidate(record_id, last_offset, "narration-idle", key="narration"))

    # -- pad to the sampled pendency length ---------------------------------
    low, high = PENDENCY_SUCCESS_RANGE if outcome == "success" else PENDENCY_FAILURE_RANGE
    used = ack_ticks + content_ticks
    pad_low = max(0, low - used)
    pad_high = max(pad_low, high - used)
    pad_ticks = _stable_int(f"{record_id}:pendency-pad", pad_low, pad_high)
    for pad_index in range(pad_ticks):
        elapsed_ms += _pause_delta(record_id, f"pad:{pad_index}", "short")
        emit_user(current_text, UserState.IDLE, _idle())
    pendency = used + pad_ticks
    if not low <= pendency <= high:
        raise AssertionError(
            f"{record_id}: pendency {pendency} outside [{low}, {high}] for outcome {outcome!r}"
        )

    # -- the terminal tool event: never narrated unless asked ---------------
    elapsed_ms += _stable_int(f"{record_id}:terminal-gap", 150, 450)
    if outcome == "success":
        terminal_offset = emit_tool(completed_payload(task), _idle())
        candidates.append(_candidate(record_id, terminal_offset, "completed-idle", key="terminal"))
    else:
        error = _stable_choice(
            f"{record_id}:error",
            (
                "render timed out",
                "invalid layout produced",
                "data source unavailable",
                "spec failed validation",
            ),
        )
        terminal_offset = emit_tool(failed_payload(task, error), _idle())
        candidates.append(_candidate(record_id, terminal_offset, "failed-idle", key="terminal"))

    # -- check-after: the first later idle tick with the check still visible
    if check_offset is not None:
        for later in range(check_offset + 1, len(turns)):
            turn = turns[later]
            if turn.action.kind is ActionKind.IDLE and turn.event.source is EventSource.USER:
                candidates.append(_candidate(record_id, later, "check-after", key="check"))
                break

    # -- failure follow-up: never claim the visual exists -------------------
    if outcome == "failure":
        question = _require_str(record.get("failure_question"), record_id, "failure_question")
        reply = _stable_choice(f"{record_id}:failure-reply", FAILURE_REPLY_TEMPLATES)
        addition = " " + question
        sizes = _chunk_sizes(len(addition), f"{record_id}:failure-check")
        cursor = 0
        last_offset = -1
        for tick_index, size in enumerate(sizes):
            cursor += size
            elapsed_ms += _typing_delta(record_id, "failure-check", tick_index)
            complete = cursor == len(addition)
            snapshot = current_text + addition[:cursor]
            offset = len(turns)
            action = _respond(offset + 1, reply) if complete else _idle()
            emit_user(snapshot, UserState.ACTIVE, action)
            if complete:
                last_offset = offset
        current_text += addition
        candidates.append(
            _candidate(record_id, last_offset, "failure-check-positive", key="failure")
        )

    # -- closing skeleton ----------------------------------------------------
    if close_shape in {"hold", "hold-clear"}:
        elapsed_ms += _pause_delta(record_id, "close-hold", "medium")
        emit_user(current_text, UserState.IDLE, _idle())
    if close_shape != "hold":
        elapsed_ms += _pause_delta(record_id, "close-clear", "short")
        offset = emit_user("", UserState.IDLE, _idle())
        candidates.append(
            _candidate(record_id, offset, "empty-cleared", key="cleared", empty_kind="cleared")
        )
        if close_shape == "clear-hold":
            elapsed_ms += _pause_delta(record_id, "close-clear-hold", "medium")
            offset = emit_user("", UserState.IDLE, _idle())
            candidates.append(
                _candidate(
                    record_id, offset, "empty-unchanged", key="close-unchanged", empty_kind="unchanged"
                )
            )

    return CompiledDemo4Record(
        record_id=record_id,
        split=split,
        turns=tuple(turns),
        candidates=tuple(candidates),
        opening_shape=open_shape,
        closing_shape=close_shape,
        content_kind=content_kind,
        outcome=outcome,
        request_task=task,
        job_id=job_id,
        source_metadata=source_metadata,
    )


def compile_demo4_records(
    records: Iterable[Mapping[str, Any]], *, dev_fraction: float = 0.1
) -> tuple[CompiledDemo4Record, ...]:
    items = list(records)
    split_by_id = assign_demo4_splits(items, dev_fraction=dev_fraction)
    return tuple(
        compile_demo4_record(record, split=split_by_id[str(record["id"])])
        for record in sorted(items, key=lambda item: str(item["id"]))
    )


def demo4_banks_from_batches(
    batches: Iterable[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Flatten validated authored batches into (requests, progress_pairs)."""

    requests: list[dict[str, Any]] = []
    progress_pairs: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches):
        author = batch.get("author")
        if not isinstance(author, Mapping):
            raise ValueError(f"batch {batch_index}: author object required")
        for bank_name, sink in (("requests", requests), ("progress_pairs", progress_pairs)):
            raw_items = batch.get(bank_name)
            if not isinstance(raw_items, list):
                raise ValueError(f"batch {batch_index}: {bank_name} must be an array")
            for item_index, raw_item in enumerate(raw_items):
                if not isinstance(raw_item, Mapping):
                    raise ValueError(
                        f"batch {batch_index}: {bank_name}[{item_index}] must be an object"
                    )
                item = dict(raw_item)
                item["author_slot"] = str(author.get("slot"))
                item["author_model"] = str(author.get("model"))
                item["author_tranche"] = str(author.get("tranche"))
                _validated_provenance_metadata(
                    item, context=f"batch {batch_index}: {bank_name}[{item_index}]"
                )
                sink.append(item)
    return tuple(requests), tuple(progress_pairs)


# --------------------------------------------------------------------------
# targets, selection
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Demo4Targets:
    """Source and selected-card counts for a Demo 4 build.

    Defaults reach 870 cards from 130 generated job-dialog episodes, built
    from 50 request phrasings and 50 progress Q/A pairs -- the 3x-cut scale
    named in the brief. Every number is a flag on the build CLI.
    """

    requests: int = 50
    progress_pairs: int = 50
    episodes: int = 130
    cards: int = 870
    empty_per_kind: int = 40
    min_check_positive: int = 30
    min_nudge_idle: int = 30
    min_narration_idle: int = 30
    min_failure_check: int = 20
    neighbor_weights: tuple[tuple[str, int], ...] = (
        ("request-before", 40),
        ("check-before", 30),
        ("check-after", 30),
    )

    def __post_init__(self) -> None:
        if self.requests <= 0 or self.progress_pairs <= 0 or self.episodes <= 0 or self.cards <= 0:
            raise ValueError("Demo 4 requests, progress_pairs, episodes, and cards must be positive")
        if self.empty_per_kind < 0:
            raise ValueError("empty_per_kind must not be negative")
        for name, value in (
            ("min_check_positive", self.min_check_positive),
            ("min_nudge_idle", self.min_nudge_idle),
            ("min_narration_idle", self.min_narration_idle),
            ("min_failure_check", self.min_failure_check),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        roles = [role for role, _ in self.neighbor_weights]
        if sorted(roles) != sorted(NEIGHBOR_ROLES):
            raise ValueError(f"neighbor_weights must cover exactly {sorted(NEIGHBOR_ROLES)}")
        if any(weight <= 0 for _, weight in self.neighbor_weights):
            raise ValueError("neighbor weights must be positive")


def _all_candidates(records: Iterable[CompiledDemo4Record]) -> tuple[Demo4Candidate, ...]:
    return tuple(candidate for record in records for candidate in record.candidates)


def _apportion(total: int, weights: Sequence[tuple[str, int]]) -> dict[str, int]:
    """Largest-remainder apportionment; deterministic ties broken by name."""

    weight_sum = sum(weight for _, weight in weights)
    base: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    assigned = 0
    for role, weight in weights:
        exact = total * weight
        share = exact // weight_sum
        base[role] = share
        assigned += share
        remainders.append((exact % weight_sum, role))
    for _, role in sorted(remainders, key=lambda item: (-item[0], item[1]))[: total - assigned]:
        base[role] += 1
    return base


def select_demo4_candidates(
    records: Iterable[CompiledDemo4Record],
    *,
    targets: Demo4Targets = Demo4Targets(),
) -> tuple[Demo4Candidate, ...]:
    """Select the exact production card mix or fail with a count diagnosis."""

    record_items = tuple(records)
    if len(record_items) != targets.episodes:
        raise ValueError(
            f"Demo 4 source requires exactly {targets.episodes} episodes; found {len(record_items)}"
        )
    candidates = _all_candidates(record_items)
    by_role: dict[str, list[Demo4Candidate]] = {role: [] for role in DEMO4_ROLES}
    for candidate in candidates:
        by_role[candidate.role].append(candidate)

    mandatory_roles = ("request-positive", "accepted-idle", "ack-positive")
    for role in mandatory_roles:
        if len(by_role[role]) != targets.episodes:
            raise ValueError(
                f"Demo 4 requires one {role} per episode ({targets.episodes}); "
                f"found {len(by_role[role])}"
            )
    terminal = by_role["completed-idle"] + by_role["failed-idle"]
    if len(terminal) != targets.episodes:
        raise ValueError(
            f"Demo 4 requires one terminal tool tick per episode ({targets.episodes}); "
            f"found {len(terminal)}"
        )

    selected: list[Demo4Candidate] = [
        *by_role["request-positive"],
        *by_role["accepted-idle"],
        *by_role["ack-positive"],
        *terminal,
    ]

    floors = {
        "check-positive": targets.min_check_positive,
        "nudge-idle": targets.min_nudge_idle,
        "narration-idle": targets.min_narration_idle,
        "failure-check-positive": targets.min_failure_check,
    }
    for role, floor in floors.items():
        pool = by_role[role]
        if len(pool) < floor:
            raise ValueError(f"Demo 4 requires at least {floor} {role} cards; found {len(pool)}")
        selected.extend(pool)

    for empty_kind in EMPTY_KINDS:
        pool = [candidate for candidate in candidates if candidate.empty_kind == empty_kind]
        if len(pool) < targets.empty_per_kind:
            raise ValueError(
                f"Demo 4 needs {targets.empty_per_kind} {empty_kind} empty cards; found {len(pool)}"
            )
        selected.extend(
            sorted(pool, key=lambda item: _stable_rank(f"empty:{item.candidate_id}"))[
                : targets.empty_per_kind
            ]
        )

    remaining = targets.cards - len(selected)
    if remaining < 0:
        raise ValueError(
            f"Demo 4 mandatory cards ({len(selected)}) already exceed the "
            f"{targets.cards}-card target; raise --cards or reduce source"
        )
    pools = {
        role: sorted(by_role[role], key=lambda item: _stable_rank(f"{item.role}:{item.candidate_id}"))
        for role in NEIGHBOR_ROLES
    }
    available = sum(len(pool) for pool in pools.values())
    if available < remaining:
        raise ValueError(
            f"Demo 4 needs {remaining} neighbour cards to reach {targets.cards}; "
            f"only {available} are available"
        )
    allocation = _apportion(remaining, targets.neighbor_weights)
    for _ in range(len(NEIGHBOR_ROLES) + 1):
        overflow = 0
        for role in NEIGHBOR_ROLES:
            if allocation[role] > len(pools[role]):
                overflow += allocation[role] - len(pools[role])
                allocation[role] = len(pools[role])
        if not overflow:
            break
        for role in NEIGHBOR_ROLES:
            if not overflow:
                break
            headroom = len(pools[role]) - allocation[role]
            take = min(headroom, overflow)
            allocation[role] += take
            overflow -= take
    if sum(allocation.values()) != remaining:
        raise ValueError(
            f"Demo 4 could not apportion {remaining} neighbour cards across {sorted(NEIGHBOR_ROLES)}"
        )
    for role in NEIGHBOR_ROLES:
        selected.extend(pools[role][: allocation[role]])

    if len(selected) != targets.cards:
        raise AssertionError(
            f"internal Demo 4 selector error: selected {len(selected)} cards, expected {targets.cards}"
        )
    unique = {candidate.candidate_id for candidate in selected}
    if len(unique) != len(selected):
        raise AssertionError("internal Demo 4 selector error: duplicate candidate ids")
    return tuple(sorted(selected, key=lambda item: item.candidate_id))


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_demo4_card(
    record: CompiledDemo4Record, candidate: Demo4Candidate
) -> dict[str, Any]:
    """Render one card, refusing anything the serving parser cannot read."""

    if candidate.record_id != record.record_id:
        raise ValueError("candidate does not belong to the supplied Demo 4 record")
    if not 0 <= candidate.turn_offset < len(record.turns):
        raise ValueError("candidate turn offset is outside its compiled record")
    current_turn = record.turns[candidate.turn_offset]
    history = record.turns[: candidate.turn_offset]
    prompt = compile_stream(list(history), current_turn.event, fmt="g1")
    completion = g1_action_completion(current_turn.action)
    parsed = parse_g1_action(completion)
    if not parsed.valid:
        raise ValueError(
            f"{candidate.candidate_id}: non-canonical g1 completion: {parsed.diagnostic}"
        )
    if parsed.kind is ActionKind.RESPOND:
        if parsed.message != current_turn.action.message:
            raise ValueError(
                f"{candidate.candidate_id}: respond message did not survive the "
                "compile -> parse round trip byte-exactly"
            )
        if parsed.target != current_turn.event.index:
            raise ValueError(f"{candidate.candidate_id}: respond target is not the current event")
    if parsed.kind is ActionKind.TOOL and parsed.tool_name == "delegate":
        if parsed.arguments.get("task") != record.request_task:
            raise ValueError(f"{candidate.candidate_id}: delegate task drifted from the source")
    # No respond may claim completion before the completion/failure event
    # exists in this candidate's own visible history.
    if candidate.role in {"check-positive", "failure-check-positive"}:
        terminal_seen = any(
            turn.event.source is EventSource.TOOL
            and json.loads(turn.event.content).get("status") in {"completed", "failed"}
            for turn in history
        )
        if candidate.role == "check-positive" and terminal_seen:
            raise ValueError(
                f"{candidate.candidate_id}: check-positive answered after the job already resolved"
            )
        if candidate.role == "failure-check-positive" and not terminal_seen:
            raise ValueError(
                f"{candidate.candidate_id}: failure-check-positive fired before the job failed"
            )

    if current_turn.action.kind is ActionKind.TOOL:
        expected_class = str(current_turn.action.tool_name)
    else:
        expected_class = current_turn.action.kind.value
    metadata = dict(record.source_metadata)
    obligation = "job-pending" if candidate.role not in {"request-before", "empty-initial"} else "none"
    return {
        "schema_version": G1_SCHEMA_VERSION,
        "split": record.split,
        "episode": record.record_id,
        "demo": DEMO,
        "situation": candidate.role,
        "bucket": candidate.role,
        "prompt": prompt,
        "completion": completion,
        "expected_class": expected_class,
        "current_event_index": current_turn.event.index,
        "current_content_empty": current_turn.event.content == "",
        "candidate_id": candidate.candidate_id,
        "candidate_role": candidate.role,
        "source_record_id": record.record_id,
        "source_persona": metadata.get("persona"),
        "source_domain": metadata.get("domain"),
        "source_register": metadata.get("register"),
        "source_author_slot": metadata.get("author_slot"),
        "source_author_model": metadata.get("author_model"),
        "source_author_tranche": metadata.get("author_tranche"),
        "job_id": record.job_id if current_turn.event.source is EventSource.TOOL else None,
        "request_task": record.request_task,
        "content_kind": record.content_kind,
        "outcome": record.outcome,
        "pause_after": candidate.pause_after,
        "empty_kind": candidate.empty_kind,
        "opening_shape": record.opening_shape,
        "closing_shape": record.closing_shape,
        "obligation": obligation,
    }


def demo4_coverage_report(
    records: Iterable[CompiledDemo4Record],
    selected: Iterable[Demo4Candidate],
) -> dict[str, Any]:
    record_items = tuple(records)
    selected_items = tuple(selected)
    record_by_id = {record.record_id: record for record in record_items}
    role_counts = Counter(candidate.role for candidate in selected_items)
    empty_counts = Counter(candidate.empty_kind for candidate in selected_items if candidate.empty_kind)
    split_counts = Counter(record_by_id[candidate.record_id].split for candidate in selected_items)
    source_split_counts = Counter(record.split for record in record_items)
    outcome_counts = Counter(record_by_id[c.record_id].outcome for c in selected_items)
    content_kind_counts = Counter(record_by_id[c.record_id].content_kind for c in selected_items)
    selected_source_counts: dict[str, Counter[str]] = {
        field: Counter() for field in ("persona", "domain", "register", "author_slot")
    }
    for candidate in selected_items:
        metadata = dict(record_by_id[candidate.record_id].source_metadata)
        for field, counts in selected_source_counts.items():
            if field in metadata:
                counts[metadata[field]] += 1

    non_empty_deltas: list[int] = []
    time_gaps: list[int] = []
    for record in record_items:
        prior = ""
        prior_time: int | None = None
        for turn in record.turns:
            current = turn.event.content
            if current and current.startswith(prior) and len(current) > len(prior):
                non_empty_deltas.append(len(current) - len(prior))
            if prior_time is not None and turn.event.elapsed_ms is not None:
                time_gaps.append(turn.event.elapsed_ms - prior_time)
            prior = current
            prior_time = turn.event.elapsed_ms

    return {
        "records": len(record_items),
        "source_splits": dict(sorted(source_split_counts.items())),
        "source_turns": sum(len(record.turns) for record in record_items),
        "opening_shapes": dict(sorted(Counter(record.opening_shape for record in record_items).items())),
        "closing_shapes": dict(sorted(Counter(record.closing_shape for record in record_items).items())),
        "outcomes": dict(sorted(Counter(record.outcome for record in record_items).items())),
        "content_kinds": dict(sorted(Counter(record.content_kind for record in record_items).items())),
        "selected_cards": len(selected_items),
        "selected_roles": dict(sorted(role_counts.items())),
        "selected_empty_kinds": dict(sorted(empty_counts.items())),
        "selected_splits": dict(sorted(split_counts.items())),
        "selected_outcomes": dict(sorted(outcome_counts.items())),
        "selected_content_kinds": dict(sorted(content_kind_counts.items())),
        "selected_source_distribution": {
            field: dict(sorted(counts.items())) for field, counts in selected_source_counts.items()
        },
        "typing_delta_chars": {
            "min": min(non_empty_deltas) if non_empty_deltas else None,
            "max": max(non_empty_deltas) if non_empty_deltas else None,
        },
        "tick_gap_ms": {
            "min": min(time_gaps) if time_gaps else None,
            "max": max(time_gaps) if time_gaps else None,
        },
    }


def compile_demo4_dataset(
    records: Iterable[Mapping[str, Any]],
    *,
    targets: Demo4Targets = Demo4Targets(),
    dev_fraction: float = 0.1,
) -> Demo4Build:
    compiled = compile_demo4_records(records, dev_fraction=dev_fraction)
    selected = select_demo4_candidates(compiled, targets=targets)
    record_by_id = {record.record_id: record for record in compiled}
    rows = tuple(
        render_demo4_card(record_by_id[candidate.record_id], candidate) for candidate in selected
    )
    return Demo4Build(
        records=compiled,
        selected=selected,
        rows=rows,
        coverage=demo4_coverage_report(compiled, selected),
    )
