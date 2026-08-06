"""Deterministic compiler for Demo 5 (reminders / time awareness).

Demo 5 is different in kind from Demos 1 and 3.  Authors do not write whole
episodes; they write three small banks of *words* -- reminder-request
phrasings, cancellation phrasings, and filler narration passages (see
``datagen.g1_authored_demo5`` for the bank schema and its validation gates).
This module is the "generator" the spec describes: it cross-products those
banks with sampled intervals, fire offsets, tick alignments, and cancellation
timings into schedules, and compiles each schedule into the live app's
full-textbox event stream.

The behaviour under training is clock-reading: "Remind me every N seconds to
X" opens a standing obligation; on each later tick the model must fire
``respond`` exactly when the elapsed time since the last fire has reached the
interval, and ``idle`` on every tick before that.  Two things make this
module's job different from every other demo's compiler:

1. **The arithmetic is generated, not authored.**  Every interval, gap, and
   fire offset comes from hashing the schedule id -- there is no author-set
   timestamp to trust or distrust.  ``verify_fire_timing`` recomputes,
   independently and from the rendered prompt text alone, whether a fire was
   due; every schedule's fire cards are checked against it before they may
   become part of a build (see ``scripts/g1_demo5_build.py``).
2. **Empty content does not cancel a standing schedule.**  Some fires are
   deliberately placed inside a silent (empty-textbox) stretch so idle-vs-fire
   is graded correctly during silence, not just during typing -- the demo's
   worked example in the spec is exactly this case.

Everything is derived by hashing UTF-8 bytes.  There is no RNG and no seed:
the same authored banks produce byte-identical cards forever, on any
platform.

The small hashing/cadence helpers are duplicated from ``datagen.g1_demo3``
rather than imported, matching that module's own note: these are frozen while
concurrent demos are in flight, and a later consolidation pass will merge the
copies.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.domain import Action, ActionKind, CompletedTurn, EventSource, StreamEvent, UserState
from app.stream import compile_stream, g1_action_completion, parse_g1_action


G1_SCHEMA_VERSION = "g1"
DEMO = "demo-5"

MIN_PHRASE_CHARS = 12
INTERVAL_PLACEHOLDER = "{interval}"

#: Reminder intervals the generator samples for "every N seconds" schedules,
#: and the nominal check window used for a one-shot's "does it repeat" trap.
INTERVAL_CHOICES_S = (3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30)

SCHEDULE_KINDS = ("once", "every")
CANCEL_VARIANTS = ("none", "mid", "late")
COLLISION_VARIANTS = ("none", "bait", "address", "both")
ALIGNMENTS = ("tight", "medium", "wide")

PAUSE_MS = {
    "none": (540, 820),
    "short": (1_000, 3_000),
    "medium": (4_000, 9_000),
    "long": (10_000, 30_000),
}

#: Demo 3 derives its opening/closing shape from the record id so silence
#: appears in different places across episodes rather than one fixed shape.
#: Demo 5 reuses the same trick for the same reason.
OPENING_SHAPES = ("immediate", "idle", "idle-idle", "idle-long-idle")
CLOSING_SHAPES = ("clear", "clear-hold", "hold", "hold-clear")

DEMO5_ROLES = (
    "request-before",
    "request-ack",
    "request-after",
    "fire-before",
    "fire-typing",
    "fire-silent",
    "fire-after",
    "silence-idle",
    "cancel-before",
    "cancel-ack",
    "post-cancel-idle",
    "once-no-repeat",
    "bait-idle",
    "address-positive",
    "empty-initial",
    "empty-unchanged",
    "empty-cleared",
)
ROLE_PRIORITY = {role: index for index, role in enumerate(DEMO5_ROLES)}

#: All "the reminder is due, fire" candidates -- the mandatory positives.
FIRE_ROLES = ("fire-typing", "fire-silent")
#: Hard-idle traps: hard-graded idle ticks that teach a specific rule, not
#: ordinary typing ballast.
TRAP_ROLES = ("post-cancel-idle", "once-no-repeat", "bait-idle", "silence-idle")
#: A collision positive: direct address answered while the schedule is live.
COLLISION_POSITIVE_ROLES = ("address-positive",)
NEIGHBOR_ROLES = ("request-before", "request-after", "fire-before", "fire-after")
EMPTY_KINDS = ("initial", "unchanged", "cleared")

DEMO5_PROVENANCE_FIELDS = (
    "persona",
    "domain",
    "register",
    "author_slot",
    "author_model",
    "author_tranche",
)

#: Roles whose expected class is independently re-derivable from the
#: rendered prompt's timestamps alone (see ``verify_fire_timing``).  Roles
#: outside this set are not reminder-arithmetic cards (an ack, a cancel, a
#: collision reply, a plain empty tick) and are out of that function's scope.
TIMING_VERIFIABLE_ROLES = (
    "fire-before",
    "fire-typing",
    "fire-silent",
    "fire-after",
    "silence-idle",
    "post-cancel-idle",
    "once-no-repeat",
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


def _stable_choice(key: str, options: Sequence[Any]) -> Any:
    if not options:
        raise ValueError("cannot choose from an empty option list")
    return options[_stable_int(key, 0, len(options) - 1)]


def opening_shape(schedule_id: str) -> str:
    return _stable_choice(f"{schedule_id}:opening-shape", OPENING_SHAPES)


def closing_shape(schedule_id: str) -> str:
    return _stable_choice(f"{schedule_id}:closing-shape", CLOSING_SHAPES)


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


def _pause_delta(schedule_id: str, key: str, pause_after: str) -> int:
    if pause_after not in PAUSE_MS:
        raise ValueError(f"{schedule_id}: invalid pause_after {pause_after!r}")
    low, high = PAUSE_MS[pause_after]
    return _stable_int(f"{schedule_id}:{key}:pause", low, high)


def _typing_delta(schedule_id: str, key: str, tick_index: int) -> int:
    return _stable_int(f"{schedule_id}:{key}:tick:{tick_index}:time", 520, 820)


def _idle() -> Action:
    return Action(ActionKind.IDLE)


def _respond(target: int, message: str) -> Action:
    return Action(ActionKind.RESPOND, target=target, message=message)


# --------------------------------------------------------------------------
# authored bank entries (validated & flattened by g1_authored_demo5)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RequestEntry:
    id: str
    schedule_kind: str  # "once" | "every"
    text_template: str
    gold_ack_template: str
    fire_message: str
    provenance: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.schedule_kind not in SCHEDULE_KINDS:
            raise ValueError(f"{self.id}: invalid schedule_kind {self.schedule_kind!r}")


@dataclass(frozen=True, slots=True)
class CancellationEntry:
    id: str
    text: str
    gold_ack: str
    provenance: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class FillerEntry:
    id: str
    text: str
    trap: str  # "none" | "bait" | "address"
    gold_reply: str | None
    provenance: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class Demo5Bank:
    requests: tuple[RequestEntry, ...]
    cancellations: tuple[CancellationEntry, ...]
    fillers_plain: tuple[FillerEntry, ...]
    fillers_bait: tuple[FillerEntry, ...]
    fillers_address: tuple[FillerEntry, ...]


def render_request_text(request: RequestEntry, interval_s: int | None) -> str:
    if request.schedule_kind == "every":
        if interval_s is None:
            raise ValueError(f"{request.id}: every-N schedules require an interval")
        return request.text_template.replace(INTERVAL_PLACEHOLDER, str(interval_s))
    return request.text_template


def render_ack_text(request: RequestEntry, interval_s: int | None) -> str:
    if request.schedule_kind == "every":
        if interval_s is None:
            raise ValueError(f"{request.id}: every-N schedules require an interval")
        return request.gold_ack_template.replace(INTERVAL_PLACEHOLDER, str(interval_s))
    return request.gold_ack_template


# --------------------------------------------------------------------------
# schedule planning: cross-product banks x sampled arithmetic
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Demo5ScheduleConfig:
    """One fully-resolved schedule: which bank entries, which sampled knobs."""

    schedule_id: str
    request: RequestEntry
    schedule_kind: str
    interval_s: int | None
    fire_cycles: int
    cancel_variant: str
    collision: str
    cancellation: CancellationEntry | None
    fillers: tuple[FillerEntry, ...]
    bait_filler: FillerEntry | None
    address_filler: FillerEntry | None
    once_check_interval_s: int | None


def plan_demo5_schedule(schedule_id: str, bank: Demo5Bank) -> Demo5ScheduleConfig:
    """Deterministically cross-product the authored banks into one schedule.

    Every knob here -- which request, which interval, how many fire cycles,
    whether/when a cancellation lands, which filler passages, whether a
    collision trap is spliced in -- is a hashed choice keyed off
    ``schedule_id``.  No RNG, no author-set timestamp: the arithmetic variety
    is exact by construction.
    """

    if not bank.requests:
        raise ValueError("bank has no request phrasings")
    request = bank.requests[_stable_int(f"{schedule_id}:request", 0, len(bank.requests) - 1)]
    schedule_kind = request.schedule_kind

    if schedule_kind == "every":
        interval_s = _stable_choice(f"{schedule_id}:interval", INTERVAL_CHOICES_S)
        fire_cycles = _stable_int(f"{schedule_id}:fire-cycles", 2, 3)
        cancel_variant = (
            _stable_choice(f"{schedule_id}:cancel-variant", CANCEL_VARIANTS)
            if bank.cancellations
            else "none"
        )
        once_check_interval_s = None
    else:
        interval_s = None
        fire_cycles = 1
        cancel_variant = "none"
        once_check_interval_s = _stable_choice(f"{schedule_id}:once-check", INTERVAL_CHOICES_S)

    cancellation = None
    if cancel_variant != "none":
        if not bank.cancellations:
            raise ValueError(f"{schedule_id}: cancel_variant requires a cancellation bank entry")
        cancellation = bank.cancellations[
            _stable_int(f"{schedule_id}:cancellation", 0, len(bank.cancellations) - 1)
        ]

    if not bank.fillers_plain:
        raise ValueError("bank has no plain filler passages")
    fillers = tuple(
        bank.fillers_plain[
            _stable_int(f"{schedule_id}:filler:{cycle}", 0, len(bank.fillers_plain) - 1)
        ]
        for cycle in range(fire_cycles)
    )

    # Collisions ("schedule running while X happens") are only spliced in for
    # schedules that stay live to the end: inserting one mid-cycle would race
    # its own typing/pause time against the interval countdown, and a
    # cancelled schedule is no longer "live" for the collision to exercise.
    collision_options = ["none"]
    if cancel_variant == "none":
        if bank.fillers_bait:
            collision_options.append("bait")
        if bank.fillers_address:
            collision_options.append("address")
        if bank.fillers_bait and bank.fillers_address:
            collision_options.append("both")
    collision = _stable_choice(f"{schedule_id}:collision", collision_options)

    bait_filler = None
    if collision in ("bait", "both"):
        bait_filler = bank.fillers_bait[
            _stable_int(f"{schedule_id}:bait", 0, len(bank.fillers_bait) - 1)
        ]
    address_filler = None
    if collision in ("address", "both"):
        address_filler = bank.fillers_address[
            _stable_int(f"{schedule_id}:address", 0, len(bank.fillers_address) - 1)
        ]

    return Demo5ScheduleConfig(
        schedule_id=schedule_id,
        request=request,
        schedule_kind=schedule_kind,
        interval_s=interval_s,
        fire_cycles=fire_cycles,
        cancel_variant=cancel_variant,
        collision=collision,
        cancellation=cancellation,
        fillers=fillers,
        bait_filler=bait_filler,
        address_filler=address_filler,
        once_check_interval_s=once_check_interval_s,
    )


# --------------------------------------------------------------------------
# compiled records
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Demo5Candidate:
    candidate_id: str
    schedule_id: str
    turn_offset: int
    role: str
    fire_index: int | None = None
    silent: bool | None = None
    alignment: str | None = None
    empty_kind: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLE_PRIORITY:
            raise ValueError(f"unknown Demo 5 candidate role: {self.role!r}")


@dataclass(frozen=True, slots=True)
class CompiledDemo5Schedule:
    schedule_id: str
    split: str
    turns: tuple[CompletedTurn, ...]
    candidates: tuple[Demo5Candidate, ...]
    config: Demo5ScheduleConfig
    opening_shape: str
    closing_shape: str
    request_text: str
    ack_text: str
    cancel_ack_text: str | None


@dataclass(frozen=True, slots=True)
class Demo5Build:
    schedules: tuple[CompiledDemo5Schedule, ...]
    selected: tuple[Demo5Candidate, ...]
    rows: tuple[dict[str, Any], ...]
    coverage: dict[str, Any]


def _candidate(
    schedule_id: str,
    turn_offset: int,
    role: str,
    *,
    key: str,
    fire_index: int | None = None,
    silent: bool | None = None,
    alignment: str | None = None,
    empty_kind: str | None = None,
) -> Demo5Candidate:
    return Demo5Candidate(
        candidate_id=f"{schedule_id}:{role}:{key}",
        schedule_id=schedule_id,
        turn_offset=turn_offset,
        role=role,
        fire_index=fire_index,
        silent=silent,
        alignment=alignment,
        empty_kind=empty_kind,
    )


def _dedupe_candidates(candidates: Sequence[Demo5Candidate]) -> tuple[Demo5Candidate, ...]:
    best: dict[int, Demo5Candidate] = {}
    for candidate in candidates:
        existing = best.get(candidate.turn_offset)
        if existing is None or ROLE_PRIORITY[candidate.role] < ROLE_PRIORITY[existing.role]:
            best[candidate.turn_offset] = candidate
    return tuple(best[offset] for offset in sorted(best))


class _Ctx:
    """Mutable cursor over one schedule's compiled event stream."""

    __slots__ = ("schedule_id", "turns", "content", "elapsed_ms")

    def __init__(self, schedule_id: str, elapsed_ms: int) -> None:
        self.schedule_id = schedule_id
        self.turns: list[CompletedTurn] = []
        self.content = ""
        self.elapsed_ms = elapsed_ms

    def emit(self, content: str, state: UserState, action: Action) -> int:
        offset = len(self.turns)
        event = StreamEvent(offset + 1, EventSource.USER, content, state, self.elapsed_ms)
        self.turns.append(CompletedTurn(event=event, action=action))
        return offset


def _joined(ctx: _Ctx, text: str) -> str:
    if ctx.content and not ctx.content.endswith((" ", "\n")):
        return " " + text
    return text


class _TypingFeed:
    """Doles out 4-7 character chunks of one filler passage, in order."""

    def __init__(self, schedule_id: str, key: str, text: str) -> None:
        sizes = _chunk_sizes(len(text), f"{schedule_id}:{key}")
        pieces: list[str] = []
        cursor = 0
        for size in sizes:
            pieces.append(text[cursor : cursor + size])
            cursor += size
        self._pieces = pieces
        self._index = 0

    def next(self) -> str | None:
        if self._index >= len(self._pieces):
            return None
        piece = self._pieces[self._index]
        self._index += 1
        return piece


def _type_phrase(
    ctx: _Ctx, schedule_id: str, key: str, text: str, *, message: str | None = None
) -> list[int]:
    """Type ``text`` to completion; the completing tick gets ``respond(message)``.

    Requires at least two ticks so the completion has a real "still typing"
    before-neighbour, mirroring Demo 3's instruction/clause minimum.
    """

    base = ctx.content
    sizes = _chunk_sizes(len(text), f"{schedule_id}:{key}")
    if len(sizes) < 2:
        raise ValueError(f"{schedule_id}: {key} text is too short for a before-neighbour")
    offsets: list[int] = []
    cursor = 0
    for tick_index, size in enumerate(sizes):
        cursor += size
        ctx.elapsed_ms += _typing_delta(schedule_id, key, tick_index)
        complete = cursor == len(text)
        content = base + text[:cursor]
        index = len(ctx.turns) + 1
        action = _respond(index, message) if (complete and message) else _idle()
        offsets.append(ctx.emit(content, UserState.ACTIVE, action))
    ctx.content = base + text
    return offsets


def _walk_to(
    ctx: _Ctx,
    schedule_id: str,
    key: str,
    target_ms: int,
    *,
    mode: str,
    feed: _TypingFeed | None = None,
    final_message: str | None = None,
) -> list[int]:
    """Advance ``ctx.elapsed_ms`` to exactly ``target_ms`` over 1-3 non-uniform ticks.

    ``mode`` is ``"typing"`` (content grows from ``feed``, falling back to an
    unchanged tick once the feed is exhausted) or ``"silent"`` (content stays
    exactly as it is -- the empty-box case demo 5 exists to test).  If
    ``final_message`` is given, the very last emitted tick is a
    ``respond(final_message)`` aimed at itself instead of ``idle()``.
    """

    if target_ms <= ctx.elapsed_ms:
        raise ValueError(f"{schedule_id}: {key} cannot walk to a non-increasing time")
    remaining = target_ms - ctx.elapsed_ms
    steps = 1 if remaining < 1_200 else _stable_int(f"{schedule_id}:{key}:steps", 1, 3)
    gaps: list[int] = []
    cumulative = 0
    for step in range(steps - 1):
        headroom = remaining - cumulative - (steps - 1 - step) * 150
        high = max(200, min(headroom, 9_000))
        low = min(200, high)
        gap = _stable_int(f"{schedule_id}:{key}:gap:{step}", low, high)
        gaps.append(gap)
        cumulative += gap
    gaps.append(remaining - cumulative)

    offsets: list[int] = []
    for step, gap in enumerate(gaps):
        ctx.elapsed_ms += gap
        if mode == "silent":
            content = ctx.content
            state = UserState.IDLE
        else:
            piece = feed.next() if feed is not None else None
            if piece:
                ctx.content += piece
                state = UserState.ACTIVE
            else:
                state = UserState.IDLE
            content = ctx.content
        is_last = step == len(gaps) - 1
        if is_last and final_message is not None:
            index = len(ctx.turns) + 1
            action = _respond(index, final_message)
        else:
            action = _idle()
        offsets.append(ctx.emit(content, state, action))
    if ctx.elapsed_ms != target_ms:
        raise AssertionError(f"{schedule_id}: {key} internal Demo 5 timing drift")
    return offsets


def _first_visible_idle_after(ctx: _Ctx, offset: int) -> int | None:
    for later in range(offset + 1, len(ctx.turns)):
        turn = ctx.turns[later]
        if turn.action.kind is ActionKind.IDLE and turn.event.content:
            return later
    return None


def compile_demo5_schedule(
    config: Demo5ScheduleConfig, *, split: str = "train"
) -> CompiledDemo5Schedule:
    """Compile one planned schedule into the live app's event stream."""

    schedule_id = config.schedule_id
    if split not in {"train", "dev"}:
        raise ValueError("Demo 5 split must be 'train' or 'dev'")
    _validated_provenance_metadata(
        dict(config.request.provenance), context=f"{schedule_id}: request {config.request.id}"
    )
    open_shape = opening_shape(schedule_id)
    close_shape = closing_shape(schedule_id)
    ctx = _Ctx(schedule_id, _stable_int(f"{schedule_id}:initial-time", 520, 780))
    candidates: list[Demo5Candidate] = []

    # -- opening skeleton ---------------------------------------------------
    if open_shape != "immediate":
        offset = ctx.emit("", UserState.IDLE, _idle())
        candidates.append(_candidate(schedule_id, offset, "empty-initial", key="initial", empty_kind="initial"))
        if open_shape in {"idle-idle", "idle-long-idle"}:
            gap = "long" if open_shape == "idle-long-idle" else "none"
            ctx.elapsed_ms += _pause_delta(schedule_id, "open-hold", gap)
            offset = ctx.emit("", UserState.IDLE, _idle())
            candidates.append(
                _candidate(schedule_id, offset, "empty-unchanged", key="open-unchanged", empty_kind="unchanged")
            )

    # -- the standing reminder request and its one-shot ack -----------------
    request_text = render_request_text(config.request, config.interval_s)
    ack_text = render_ack_text(config.request, config.interval_s)
    if len(request_text) < MIN_PHRASE_CHARS:
        raise ValueError(f"{schedule_id}: request text must be at least {MIN_PHRASE_CHARS} characters")
    offsets = _type_phrase(ctx, schedule_id, "request", request_text, message=ack_text)
    candidates.append(_candidate(schedule_id, offsets[-2], "request-before", key="request"))
    ack_offset = offsets[-1]
    candidates.append(_candidate(schedule_id, ack_offset, "request-ack", key="request"))
    anchor_ms = ctx.elapsed_ms

    # -- fire cycles ----------------------------------------------------------
    cancel_ack_text: str | None = None
    cancelled = False
    for cycle in range(config.fire_cycles):
        if config.schedule_kind == "every":
            assert config.interval_s is not None
            interval_ms = config.interval_s * 1_000
            silent = _stable_choice(f"{schedule_id}:fire:{cycle}:silent", (False, True))
            alignment = _stable_choice(f"{schedule_id}:fire:{cycle}:alignment", ALIGNMENTS)
            before_frac = _stable_int(f"{schedule_id}:fire:{cycle}:before-frac", 50, 85)
            before_ms = anchor_ms + before_frac * interval_ms // 100
            overshoot = _overshoot_ms(schedule_id, cycle, alignment, interval_ms)
            fire_ms = anchor_ms + interval_ms + overshoot
        else:
            silent = _stable_choice(f"{schedule_id}:once:silent", (False, True))
            alignment = "medium"
            before_ms = anchor_ms + _stable_int(f"{schedule_id}:once:before", 2_000, 8_000)
            fire_ms = before_ms + _stable_int(f"{schedule_id}:once:overshoot", 300, 3_000)

        mode = "silent" if silent else "typing"
        feed = None
        if mode == "typing":
            filler = config.fillers[cycle]
            feed = _TypingFeed(schedule_id, f"fire:{cycle}", _joined(ctx, filler.text))
        else:
            ctx.content = ""

        before_offsets = _walk_to(ctx, schedule_id, f"fire:{cycle}:before", before_ms, mode=mode, feed=feed)
        if silent and len(before_offsets) >= 2:
            candidates.append(_candidate(schedule_id, before_offsets[0], "silence-idle", key=str(cycle), fire_index=cycle))
        candidates.append(
            _candidate(schedule_id, before_offsets[-1], "fire-before", key=str(cycle), fire_index=cycle, silent=silent)
        )

        fire_offsets = _walk_to(
            ctx,
            schedule_id,
            f"fire:{cycle}:due",
            fire_ms,
            mode=mode,
            feed=feed,
            final_message=config.request.fire_message,
        )
        fire_offset = fire_offsets[-1]
        role = "fire-silent" if silent else "fire-typing"
        candidates.append(
            _candidate(
                schedule_id, fire_offset, role, key=str(cycle), fire_index=cycle, silent=silent, alignment=alignment
            )
        )
        anchor_ms = fire_ms

        after_ms = anchor_ms + _stable_int(f"{schedule_id}:fire:{cycle}:after-gap", 400, 900)
        after_offsets = _walk_to(ctx, schedule_id, f"fire:{cycle}:after", after_ms, mode=mode, feed=feed)
        candidates.append(
            _candidate(schedule_id, after_offsets[-1], "fire-after", key=str(cycle), fire_index=cycle, silent=silent)
        )

        if config.schedule_kind == "once":
            break
        if config.cancel_variant == "mid" and cycle == 0:
            assert config.cancellation is not None
            cancel_ack_text = config.cancellation.gold_ack
            _emit_cancellation(ctx, schedule_id, config, candidates)
            cancelled = True
            break
    else:
        if config.schedule_kind == "every" and config.cancel_variant == "late":
            assert config.cancellation is not None
            cancel_ack_text = config.cancellation.gold_ack
            _emit_cancellation(ctx, schedule_id, config, candidates)
            cancelled = True

    # -- optional collision splice: bait / direct address while still live --
    # Only reachable when the schedule was never cancelled (planning already
    # guarantees collision == "none" whenever cancel_variant != "none"), so
    # this never races the interval countdown of an in-progress fire cycle.
    if not cancelled:
        if config.collision in ("bait", "both"):
            filler = config.bait_filler
            assert filler is not None
            offsets = _type_phrase(ctx, schedule_id, "bait", _joined(ctx, filler.text))
            candidates.append(_candidate(schedule_id, offsets[-1], "bait-idle", key="bait"))
            ctx.elapsed_ms += _pause_delta(schedule_id, "post-bait", "short")
        if config.collision in ("address", "both"):
            filler = config.address_filler
            assert filler is not None and filler.gold_reply is not None
            offsets = _type_phrase(ctx, schedule_id, "address", _joined(ctx, filler.text), message=filler.gold_reply)
            candidates.append(_candidate(schedule_id, offsets[-1], "address-positive", key="address"))
            ctx.elapsed_ms += _pause_delta(schedule_id, "post-address", "short")

    if config.schedule_kind == "once":
        assert config.once_check_interval_s is not None
        ctx.content = ""
        post_ms = ctx.elapsed_ms + config.once_check_interval_s * 1_000 + _stable_int(
            f"{schedule_id}:once-extra", 500, 4_000
        )
        offsets = _walk_to(ctx, schedule_id, "once-no-repeat", post_ms, mode="silent")
        candidates.append(_candidate(schedule_id, offsets[-1], "once-no-repeat", key="once"))
    elif cancelled:
        ctx.content = ""
        interval_ms = config.interval_s * 1_000 if config.interval_s else 8_000
        post_ms = ctx.elapsed_ms + interval_ms + _stable_int(f"{schedule_id}:post-cancel", 500, 4_000)
        offsets = _walk_to(ctx, schedule_id, "post-cancel", post_ms, mode="silent")
        candidates.append(_candidate(schedule_id, offsets[-1], "post-cancel-idle", key="cancel"))

    # -- request-after: first later idle tick that still shows completed text
    after = _first_visible_idle_after(ctx, ack_offset)
    if after is not None:
        candidates.append(_candidate(schedule_id, after, "request-after", key="request"))

    # -- closing skeleton -----------------------------------------------------
    if close_shape in {"hold", "hold-clear"}:
        ctx.elapsed_ms += _pause_delta(schedule_id, "close-hold", "medium")
        ctx.emit(ctx.content, UserState.IDLE, _idle())
    if close_shape != "hold":
        ctx.elapsed_ms += _pause_delta(schedule_id, "close-clear", "short")
        ctx.content = ""
        offset = ctx.emit("", UserState.IDLE, _idle())
        candidates.append(_candidate(schedule_id, offset, "empty-cleared", key="cleared", empty_kind="cleared"))
        if close_shape == "clear-hold":
            ctx.elapsed_ms += _pause_delta(schedule_id, "close-clear-hold", "medium")
            offset = ctx.emit("", UserState.IDLE, _idle())
            candidates.append(
                _candidate(schedule_id, offset, "empty-unchanged", key="close-unchanged", empty_kind="unchanged")
            )

    return CompiledDemo5Schedule(
        schedule_id=schedule_id,
        split=split,
        turns=tuple(ctx.turns),
        candidates=_dedupe_candidates(candidates),
        config=config,
        opening_shape=open_shape,
        closing_shape=close_shape,
        request_text=request_text,
        ack_text=ack_text,
        cancel_ack_text=cancel_ack_text,
    )


def _overshoot_ms(schedule_id: str, cycle: int, alignment: str, interval_ms: int) -> int:
    key = f"{schedule_id}:fire:{cycle}:overshoot"
    if alignment == "tight":
        return _stable_int(key, 50, 300)
    if alignment == "medium":
        return _stable_int(key, 300, 2_000)
    high = max(2_100, min(interval_ms - 100, 15_000))
    return _stable_int(key, 2_000, high)


def _emit_cancellation(
    ctx: _Ctx, schedule_id: str, config: Demo5ScheduleConfig, candidates: list[Demo5Candidate]
) -> None:
    assert config.cancellation is not None
    text = _joined(ctx, config.cancellation.text)
    offsets = _type_phrase(ctx, schedule_id, "cancel", text, message=config.cancellation.gold_ack)
    candidates.append(_candidate(schedule_id, offsets[-2], "cancel-before", key="cancel"))
    candidates.append(_candidate(schedule_id, offsets[-1], "cancel-ack", key="cancel"))


def demo5_bank_from_batches(batches: Iterable[Mapping[str, Any]]) -> Demo5Bank:
    """Flatten validated authored batches into a typed, provenance-carrying bank."""

    requests: list[RequestEntry] = []
    cancellations: list[CancellationEntry] = []
    fillers_plain: list[FillerEntry] = []
    fillers_bait: list[FillerEntry] = []
    fillers_address: list[FillerEntry] = []

    for batch_index, batch in enumerate(batches):
        author = batch.get("author")
        bank_entries = batch.get("bank")
        if not isinstance(author, Mapping) or not isinstance(bank_entries, list):
            raise ValueError(f"batch {batch_index}: author object and bank array required")
        for entry_index, raw_entry in enumerate(bank_entries):
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"batch {batch_index}: entry {entry_index} must be an object")
            entry = dict(raw_entry)
            provenance = _validated_provenance_metadata(
                {
                    **entry,
                    "author_slot": str(author.get("slot")),
                    "author_model": str(author.get("model")),
                    "author_tranche": str(author.get("tranche")),
                },
                context=f"batch {batch_index}: entry {entry_index}",
            )
            kind = entry.get("kind")
            entry_id = str(entry["id"])
            if kind == "request":
                requests.append(
                    RequestEntry(
                        id=entry_id,
                        schedule_kind=str(entry["schedule_kind"]),
                        text_template=str(entry["text_template"]),
                        gold_ack_template=str(entry["gold_ack_template"]),
                        fire_message=str(entry["fire_message"]),
                        provenance=provenance,
                    )
                )
            elif kind == "cancellation":
                cancellations.append(
                    CancellationEntry(
                        id=entry_id,
                        text=str(entry["text"]),
                        gold_ack=str(entry["gold_ack"]),
                        provenance=provenance,
                    )
                )
            elif kind == "filler":
                trap = str(entry.get("trap", "none"))
                filler = FillerEntry(
                    id=entry_id,
                    text=str(entry["text"]),
                    trap=trap,
                    gold_reply=str(entry["gold_reply"]) if entry.get("gold_reply") else None,
                    provenance=provenance,
                )
                if trap == "bait":
                    fillers_bait.append(filler)
                elif trap == "address":
                    fillers_address.append(filler)
                else:
                    fillers_plain.append(filler)
            else:
                raise ValueError(f"batch {batch_index}: entry {entry_index} has unknown kind {kind!r}")

    return Demo5Bank(
        requests=tuple(requests),
        cancellations=tuple(cancellations),
        fillers_plain=tuple(fillers_plain),
        fillers_bait=tuple(fillers_bait),
        fillers_address=tuple(fillers_address),
    )


def compile_demo5_schedules(
    configs: Iterable[Demo5ScheduleConfig], *, dev_fraction: float = 0.1
) -> tuple[CompiledDemo5Schedule, ...]:
    items = list(configs)
    split_by_id = assign_demo5_splits([config.schedule_id for config in items], dev_fraction=dev_fraction)
    return tuple(
        compile_demo5_schedule(config, split=split_by_id[config.schedule_id])
        for config in sorted(items, key=lambda item: item.schedule_id)
    )


def assign_demo5_splits(schedule_ids: Iterable[str], *, dev_fraction: float = 0.1) -> dict[str, str]:
    ids = list(schedule_ids)
    if not 0 <= dev_fraction <= 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    if any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("every Demo 5 schedule requires a non-empty string id")
    if len(set(ids)) != len(ids):
        raise ValueError("Demo 5 schedule ids must be unique")
    dev_count = round(len(ids) * dev_fraction)
    ranked = sorted(ids, key=lambda schedule_id: _stable_rank(f"split:{schedule_id}"))
    dev_ids = set(ranked[:dev_count])
    return {schedule_id: "dev" if schedule_id in dev_ids else "train" for schedule_id in ids}


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Demo5Targets:
    """Source and selected-card counts for a Demo 5 build.

    Defaults reach a 970-card slice from 130 schedules cross-produced out of
    36 authored phrasings (per ``synthetic_data_spec.md``'s 3x-cut Demo 5
    numbers).  Every number is a parameter; the build CLI exposes each as a
    flag.
    """

    schedules: int = 130
    fires: int = 198
    cards: int = 970
    empty_per_kind: int = 25
    min_post_cancel_idle: int = 40
    min_once_no_repeat: int = 55
    min_bait_idle: int = 38
    min_address_positive: int = 36
    min_silence_idle: int = 45
    neighbor_weights: tuple[tuple[str, int], ...] = (
        ("fire-before", 40),
        ("fire-after", 30),
        ("request-before", 15),
        ("request-after", 15),
    )

    def __post_init__(self) -> None:
        if self.schedules <= 0 or self.fires <= 0 or self.cards <= 0:
            raise ValueError("Demo 5 schedule, fire, and card targets must be positive")
        if self.empty_per_kind < 0:
            raise ValueError("empty_per_kind must not be negative")
        roles = [role for role, _ in self.neighbor_weights]
        if sorted(roles) != sorted(NEIGHBOR_ROLES):
            raise ValueError(f"neighbor_weights must cover exactly {sorted(NEIGHBOR_ROLES)}")
        if any(weight <= 0 for _, weight in self.neighbor_weights):
            raise ValueError("neighbor weights must be positive")


def _all_candidates(schedules: Iterable[CompiledDemo5Schedule]) -> tuple[Demo5Candidate, ...]:
    return tuple(candidate for schedule in schedules for candidate in schedule.candidates)


def _apportion(total: int, weights: Sequence[tuple[str, int]]) -> dict[str, int]:
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


def select_demo5_candidates(
    schedules: Iterable[CompiledDemo5Schedule], *, targets: Demo5Targets = Demo5Targets()
) -> tuple[Demo5Candidate, ...]:
    """Select the exact production card mix or fail with a count diagnosis."""

    schedule_items = tuple(schedules)
    candidates = _all_candidates(schedule_items)
    by_role: dict[str, list[Demo5Candidate]] = {role: [] for role in DEMO5_ROLES}
    for candidate in candidates:
        by_role[candidate.role].append(candidate)

    if len(schedule_items) != targets.schedules:
        raise ValueError(
            f"Demo 5 source requires exactly {targets.schedules} schedules; found {len(schedule_items)}"
        )
    fires = by_role["fire-typing"] + by_role["fire-silent"]
    if len(fires) != targets.fires:
        raise ValueError(f"Demo 5 source requires exactly {targets.fires} fires; found {len(fires)}")
    acks = by_role["request-ack"]
    if len(acks) != targets.schedules:
        raise ValueError(
            f"Demo 5 requires one acknowledgment per schedule ({targets.schedules}); found {len(acks)}"
        )

    selected: list[Demo5Candidate] = [*fires, *acks]

    floors = {
        "post-cancel-idle": targets.min_post_cancel_idle,
        "once-no-repeat": targets.min_once_no_repeat,
        "bait-idle": targets.min_bait_idle,
        "silence-idle": targets.min_silence_idle,
    }
    for role, floor in floors.items():
        pool = by_role[role]
        if len(pool) < floor:
            raise ValueError(f"Demo 5 requires at least {floor} {role} cards; found {len(pool)}")
        selected.extend(pool)

    address_pool = by_role["address-positive"]
    if len(address_pool) < targets.min_address_positive:
        raise ValueError(
            f"Demo 5 requires at least {targets.min_address_positive} address-positive cards; "
            f"found {len(address_pool)}"
        )
    selected.extend(address_pool)

    for empty_kind in EMPTY_KINDS:
        pool = [candidate for candidate in candidates if candidate.empty_kind == empty_kind]
        if len(pool) < targets.empty_per_kind:
            raise ValueError(
                f"Demo 5 needs {targets.empty_per_kind} {empty_kind} empty cards; found {len(pool)}"
            )
        selected.extend(
            sorted(pool, key=lambda item: _stable_rank(f"empty:{item.candidate_id}"))[: targets.empty_per_kind]
        )

    remaining = targets.cards - len(selected)
    if remaining < 0:
        raise ValueError(
            f"Demo 5 mandatory cards ({len(selected)}) already exceed the {targets.cards}-card target; "
            "raise --cards or reduce source"
        )
    pools = {
        role: sorted(by_role[role], key=lambda item: _stable_rank(f"{item.role}:{item.candidate_id}"))
        for role in NEIGHBOR_ROLES
    }
    available = sum(len(pool) for pool in pools.values())
    if available < remaining:
        raise ValueError(
            f"Demo 5 needs {remaining} neighbour cards to reach {targets.cards}; only {available} are available"
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
        raise ValueError(f"Demo 5 could not apportion {remaining} neighbour cards across {sorted(NEIGHBOR_ROLES)}")
    for role in NEIGHBOR_ROLES:
        selected.extend(pools[role][: allocation[role]])

    if len(selected) != targets.cards:
        raise AssertionError(f"internal Demo 5 selector error: selected {len(selected)} cards, expected {targets.cards}")
    unique = {candidate.candidate_id for candidate in selected}
    if len(unique) != len(selected):
        raise AssertionError("internal Demo 5 selector error: duplicate candidate ids")
    return tuple(sorted(selected, key=lambda item: item.candidate_id))


# --------------------------------------------------------------------------
# independent fire-timing verification
# --------------------------------------------------------------------------

_EVENT_TIME_RE = re.compile(r'<stream_event[^>]*\btime="t\+(\d+)ms"[^>]*>')
_ACTION_LINE_RE = re.compile(r"^<action>(.*)</action>$")


def derive_expected_class(
    prompt: str,
    *,
    schedule_kind: str,
    interval_s: int | None,
    fire_message: str,
    cancel_ack_text: str | None,
) -> str:
    """Independently recompute idle-vs-fire from the rendered g1 prompt alone.

    This does not consult the compiler's internal state at all: it re-parses
    the ``<stream_event ... time="t+Nms">`` / ``<action>`` line pairs with a
    regex and ``parse_g1_action`` (the same parser serving uses), tracks the
    schedule's anchor time and cancellation state purely from that, and
    answers "idle" or "fire" for the current (final) event in the prompt.

    For ``schedule_kind == "once"`` this can only check the *no-repeat*
    invariant (once fired, must never fire again) -- there is no interval to
    recompute an exact one-shot due instant from, so a pre-fire "once" card
    returns ``"unverifiable"`` rather than a guess.  Callers must skip those.
    """

    lines = prompt.split("\n")
    if lines[-1] != "<PREDICT_THIS_ACTION>":
        raise ValueError("prompt does not end with <PREDICT_THIS_ACTION>")
    body = lines[:-1]
    if not body or len(body) % 2 == 0:
        raise ValueError("prompt has no current event")
    current_line = body[-1]
    history = body[:-1]
    if len(history) % 2 != 0:
        raise ValueError("prompt history is not event/action line pairs")

    def event_time(line: str) -> int:
        match = _EVENT_TIME_RE.search(line)
        if not match:
            raise ValueError(f"event line missing time attribute: {line!r}")
        return int(match.group(1))

    def action_message(line: str) -> str | None:
        wrapper = _ACTION_LINE_RE.match(line)
        if not wrapper:
            raise ValueError(f"not an action line: {line!r}")
        parsed = parse_g1_action(f"<action>{wrapper.group(1)}</action>")
        if not parsed.valid:
            raise ValueError(f"prompt history contains a non-canonical action: {line!r}")
        return parsed.message if parsed.kind is ActionKind.RESPOND else None

    anchor_ms: int | None = None
    fired_once = False
    cancelled = False
    for index in range(0, len(history), 2):
        message = action_message(history[index + 1])
        if message is None:
            continue
        time_ms = event_time(history[index])
        if cancel_ack_text is not None and message == cancel_ack_text:
            cancelled = True
            continue
        if message == fire_message:
            anchor_ms = time_ms
            fired_once = True
            continue
        if anchor_ms is None:
            anchor_ms = time_ms

    current_ms = event_time(current_line)
    if cancelled:
        return "idle"
    if schedule_kind == "once":
        return "idle" if fired_once else "unverifiable"
    if anchor_ms is None or interval_s is None:
        return "unverifiable"
    return "fire" if current_ms - anchor_ms >= interval_s * 1_000 else "idle"


def verify_fire_timing(row: Mapping[str, Any]) -> str | None:
    """Cross-check one rendered row's gold action against ``derive_expected_class``.

    Returns ``None`` when the row's role is out of scope for this check (an
    ack, a cancel line, a collision reply, a plain empty tick) or the
    independent recomputation is a documented "unverifiable" case (a
    pre-fire one-shot).  Returns an error string on disagreement, or ``""``
    (falsy) when the check ran and agreed.
    """

    if row.get("candidate_role") not in TIMING_VERIFIABLE_ROLES:
        return None
    expected = derive_expected_class(
        str(row["prompt"]),
        schedule_kind=str(row["schedule_kind"]),
        interval_s=row.get("interval_s"),
        fire_message=str(row["fire_message"]),
        cancel_ack_text=row.get("cancel_ack_text"),
    )
    if expected == "unverifiable":
        return None
    completion = str(row["completion"])
    actual = "idle" if completion == "<action>idle()</action>" else "fire"
    if actual != expected:
        return (
            f"{row['candidate_id']}: independent timing check says {expected!r} but the "
            f"gold completion is {actual!r}"
        )
    return ""


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


NO_OBLIGATION_ROLES = {"request-before", "post-cancel-idle", "once-no-repeat"}


def _validated_provenance_metadata(record: Mapping[str, Any], *, context: str) -> tuple[tuple[str, str], ...]:
    metadata: list[tuple[str, str]] = []
    for field in DEMO5_PROVENANCE_FIELDS:
        if field not in record:
            raise ValueError(f"{context}: missing required provenance field {field!r}")
        value = record[field]
        if value is None or not str(value).strip():
            raise ValueError(f"{context}: provenance field {field!r} must be non-blank")
        metadata.append((field, str(value)))
    return tuple(sorted(metadata))


def render_demo5_card(schedule: CompiledDemo5Schedule, candidate: Demo5Candidate) -> dict[str, Any]:
    """Render one card, refusing anything the serving parser cannot read."""

    if candidate.schedule_id != schedule.schedule_id:
        raise ValueError("candidate does not belong to the supplied Demo 5 schedule")
    if not 0 <= candidate.turn_offset < len(schedule.turns):
        raise ValueError("candidate turn offset is outside its compiled schedule")
    current_turn = schedule.turns[candidate.turn_offset]
    history = schedule.turns[: candidate.turn_offset]
    prompt = compile_stream(list(history), current_turn.event, fmt="g1")
    completion = g1_action_completion(current_turn.action)
    parsed = parse_g1_action(completion)
    if not parsed.valid:
        raise ValueError(f"{candidate.candidate_id}: non-canonical g1 completion: {parsed.diagnostic}")
    if parsed.kind is ActionKind.RESPOND:
        if parsed.message != current_turn.action.message:
            raise ValueError(
                f"{candidate.candidate_id}: respond message did not survive the compile -> parse round trip"
            )
        if parsed.target != current_turn.event.index:
            raise ValueError(f"{candidate.candidate_id}: respond target is not the current event")
    expected_class = "respond" if current_turn.action.kind is ActionKind.RESPOND else "idle"
    metadata = dict(schedule.config.request.provenance)
    obligation = "none" if candidate.role in NO_OBLIGATION_ROLES else "reminder-active"
    should_fire = candidate.role in FIRE_ROLES
    reminder_eval_kind = (
        "fire"
        if should_fire
        else "wait"
        if candidate.role
        in {
            "fire-before",
            "fire-after",
            "silence-idle",
            "post-cancel-idle",
            "once-no-repeat",
            "bait-idle",
        }
        else None
    )
    timing_boundary = (
        "before"
        if candidate.role == "fire-before"
        else "at"
        if should_fire
        else "after"
        if candidate.role == "fire-after"
        else "already-fired"
        if candidate.role == "once-no-repeat"
        else None
    )
    row = {
        "schema_version": G1_SCHEMA_VERSION,
        "split": schedule.split,
        "episode": schedule.schedule_id,
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
        "source_record_id": schedule.schedule_id,
        "source_persona": metadata.get("persona"),
        "source_domain": metadata.get("domain"),
        "source_register": metadata.get("register"),
        "source_author_slot": metadata.get("author_slot"),
        "source_author_model": metadata.get("author_model"),
        "source_author_tranche": metadata.get("author_tranche"),
        "schedule_kind": schedule.config.schedule_kind,
        "interval_s": schedule.config.interval_s,
        "fire_message": schedule.config.request.fire_message,
        "cancel_ack_text": schedule.cancel_ack_text,
        "fire_index": candidate.fire_index,
        "silent": candidate.silent,
        "alignment": candidate.alignment,
        "empty_kind": candidate.empty_kind,
        "opening_shape": schedule.opening_shape,
        "closing_shape": schedule.closing_shape,
        "obligation": obligation,
        "should_fire": should_fire,
        "reminder_eval_kind": reminder_eval_kind,
        "timing_boundary": timing_boundary,
    }
    timing_error = verify_fire_timing(row)
    if timing_error:
        raise ValueError(timing_error)
    return row


def demo5_coverage_report(
    schedules: Iterable[CompiledDemo5Schedule], selected: Iterable[Demo5Candidate]
) -> dict[str, Any]:
    schedule_items = tuple(schedules)
    selected_items = tuple(selected)
    schedule_by_id = {schedule.schedule_id: schedule for schedule in schedule_items}
    role_counts = Counter(candidate.role for candidate in selected_items)
    empty_counts = Counter(candidate.empty_kind for candidate in selected_items if candidate.empty_kind)
    split_counts = Counter(schedule_by_id[candidate.schedule_id].split for candidate in selected_items)
    source_split_counts = Counter(schedule.split for schedule in schedule_items)
    kind_counts = Counter(
        schedule_by_id[candidate.schedule_id].config.schedule_kind for candidate in selected_items
    )
    alignment_counts = Counter(candidate.alignment for candidate in selected_items if candidate.alignment)
    selected_source_counts: dict[str, Counter[str]] = {
        field: Counter() for field in ("persona", "domain", "register", "author_slot")
    }
    for candidate in selected_items:
        metadata = dict(schedule_by_id[candidate.schedule_id].config.request.provenance)
        for field, counts in selected_source_counts.items():
            if field in metadata:
                counts[metadata[field]] += 1

    non_empty_deltas: list[int] = []
    time_gaps: list[int] = []
    for schedule in schedule_items:
        prior = ""
        prior_time: int | None = None
        for turn in schedule.turns:
            current = turn.event.content
            if current and current.startswith(prior) and len(current) > len(prior):
                non_empty_deltas.append(len(current) - len(prior))
            if prior_time is not None and turn.event.elapsed_ms is not None:
                time_gaps.append(turn.event.elapsed_ms - prior_time)
            prior = current
            prior_time = turn.event.elapsed_ms

    return {
        "schedules": len(schedule_items),
        "source_splits": dict(sorted(source_split_counts.items())),
        "source_fires": sum(
            1 for schedule in schedule_items for candidate in schedule.candidates if candidate.role in FIRE_ROLES
        ),
        "source_turns": sum(len(schedule.turns) for schedule in schedule_items),
        "opening_shapes": dict(sorted(Counter(schedule.opening_shape for schedule in schedule_items).items())),
        "closing_shapes": dict(sorted(Counter(schedule.closing_shape for schedule in schedule_items).items())),
        "schedule_kinds": dict(sorted(Counter(schedule.config.schedule_kind for schedule in schedule_items).items())),
        "cancel_variants": dict(sorted(Counter(schedule.config.cancel_variant for schedule in schedule_items).items())),
        "collisions": dict(sorted(Counter(schedule.config.collision for schedule in schedule_items).items())),
        "selected_cards": len(selected_items),
        "selected_roles": dict(sorted(role_counts.items())),
        "selected_empty_kinds": dict(sorted(empty_counts.items())),
        "selected_splits": dict(sorted(split_counts.items())),
        "selected_schedule_kinds": dict(sorted(kind_counts.items())),
        "selected_alignments": dict(sorted(alignment_counts.items())),
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


def compile_demo5_dataset(
    configs: Iterable[Demo5ScheduleConfig],
    *,
    targets: Demo5Targets = Demo5Targets(),
    dev_fraction: float = 0.1,
) -> Demo5Build:
    compiled = compile_demo5_schedules(configs, dev_fraction=dev_fraction)
    selected = select_demo5_candidates(compiled, targets=targets)
    schedule_by_id = {schedule.schedule_id: schedule for schedule in compiled}
    rows = tuple(render_demo5_card(schedule_by_id[candidate.schedule_id], candidate) for candidate in selected)
    return Demo5Build(
        schedules=compiled,
        selected=selected,
        rows=rows,
        coverage=demo5_coverage_report(compiled, selected),
    )
