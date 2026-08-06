"""Deterministic compiler for agent-authored g1 Demo 2 situations.

Demo 2 is *interjection*: 2.1 suggest_edit under a standing correction
instruction, 2.2 highlight under a standing category instruction.  Every
episode opens with the instruction being typed and acknowledged, then compiles a
positive+neighbour cluster per planted error / literal match, four kinds of
graded hard-idle trap, and silence.

Guarantees this module enforces in code, not in review:

* **Train equals serve.**  Prompts come from ``compile_stream(..., fmt="g1")``,
  completions from ``g1_action_completion``, and every completion is round
  tripped through ``parse_g1_action``.  A card the serving parser cannot read
  cannot leave this module.
* **Content correctness.**  A ``suggest_edit`` quote is widened until it occurs
  exactly once in the visible textbox *at that tick*, never crosses into the
  instruction line, never repeats a fix, and always differs from its
  replacement.  A ``highlight`` occurrence is recomputed from the snapshot with
  the app's own search walk and must land on the intended word.
* **No RNG.**  Every choice is a SHA-256 of a stable key.  Same source, same
  bytes, forever.
* **No constant skeleton.**  Leading silence, settle-tick counts, mid-episode
  pauses and the ending (clear / clear+idle / no clear) vary per record.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.domain import Action, ActionKind, CompletedTurn, EventSource, StreamEvent, UserState
from app.stream import compile_stream, g1_action_completion, parse_g1_action
from datagen.g1_authored_demo2 import (
    MIN_TICK_CHARS,
    PARTIAL_BEFORE_MIN_TRIGGER_CHARS,
    occurrence_count,
    occurrence_start,
    segment_trigger_layout,
)


G1_SCHEMA_VERSION = "g1"
DEMO2_DEMO = "demo-2"
DEMO2_PROVENANCE_FIELDS = (
    "mode",
    "persona",
    "domain",
    "register",
    "author_slot",
    "author_model",
    "author_tranche",
)
POSITIVE_ROLES = ("instruction-ack", "error-positive", "match-positive")
BEFORE_ROLES = ("instruction-before", "error-before", "match-before")
AFTER_ROLES = ("instruction-after", "error-after", "match-after")
TRAP_ROLES = (
    "trap-clean",
    "trap-selfcorrect",
    "trap-instruction-mention",
    "trap-non-literal",
)
EMPTY_ROLES = ("empty-initial", "empty-unchanged", "empty-cleared")
DEMO2_ROLES = frozenset(
    POSITIVE_ROLES + BEFORE_ROLES + AFTER_ROLES + TRAP_ROLES + EMPTY_ROLES
    + ("ballast-idle",)
)
EMPTY_KINDS = ("initial", "unchanged", "cleared")
TRIGGER_KINDS = ("instruction", "error", "match")
PAUSE_MS = {
    "none": (540, 820),
    "short": (1_000, 3_000),
    "medium": (4_000, 9_000),
    "long": (10_000, 30_000),
}
#: The final tick of a partially-typed before-neighbour.  Four characters is the
#: smallest legal human tick, so it is the widest trigger window we can promise.
PARTIAL_TAIL_CHARS = 4


@dataclass(frozen=True, slots=True)
class Demo2Targets:
    """Source and selected-card counts for a Demo 2 build.

    Defaults are the 1,800-card production slice.  ``errors``/``matches``/
    ``episodes`` are *exact source* requirements; the remaining fields describe
    how the 1,800 cards are apportioned.  See ``select_demo2_candidates`` for the
    arithmetic, which is checked at build time rather than assumed.
    """

    errors: int = 285
    matches: int = 165
    episodes: int = 90
    cards: int = 1_800
    empty_per_kind: int = 20
    hard_idle_cards: int = 191


@dataclass(frozen=True, slots=True)
class Demo2Candidate:
    """A gradable turn in a compiled episode; padding turns are not candidates."""

    candidate_id: str
    record_id: str
    turn_offset: int
    role: str
    mode: str
    segment_index: int | None = None
    trigger_key: str | None = None
    trigger_kind: str | None = None
    trap: str | None = None
    empty_kind: str | None = None
    before_kind: str | None = None
    subtype: str | None = None
    category: str | None = None
    pause_after: str | None = None

    def __post_init__(self) -> None:
        if self.role not in DEMO2_ROLES:
            raise ValueError(f"unknown Demo 2 candidate role: {self.role!r}")


@dataclass(frozen=True, slots=True)
class CompiledDemo2Record:
    record_id: str
    split: str
    mode: str
    turns: tuple[CompletedTurn, ...]
    candidates: tuple[Demo2Candidate, ...]
    source_metadata: tuple[tuple[str, str], ...] = ()
    skeleton: str = ""


@dataclass(frozen=True, slots=True)
class Demo2Build:
    records: tuple[CompiledDemo2Record, ...]
    selected: tuple[Demo2Candidate, ...]
    rows: tuple[dict[str, Any], ...]
    coverage: dict[str, Any]


# --------------------------------------------------------------------------
# determinism primitives
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


def _chunk_sizes(length: int, key: str) -> tuple[int, ...]:
    """Partition ``length`` characters into deterministic 4-7 character ticks.

    A source shorter than four characters is represented faithfully as one
    unavoidable short tick rather than padded with invented text.
    """

    if length <= 0:
        return ()
    if length < MIN_TICK_CHARS:
        return (length,)
    sizes: list[int] = []
    remaining = length
    step = 0
    while remaining:
        feasible = [
            size
            for size in range(MIN_TICK_CHARS, 8)
            if size <= remaining and (remaining - size == 0 or remaining - size >= MIN_TICK_CHARS)
        ]
        if not feasible:
            raise ValueError(f"cannot partition {length} characters into 4-7 character ticks")
        sizes.append(feasible[_stable_int(f"{key}:chunk:{step}", 0, len(feasible) - 1)])
        remaining -= sizes[-1]
        step += 1
    return tuple(sizes)


def _typing_delta(record_id: str, key: str, tick_index: int) -> int:
    return _stable_int(f"{record_id}:{key}:tick:{tick_index}:time", 520, 820)


def _pause_delta(record_id: str, key: str, pause_after: str) -> int:
    if pause_after not in PAUSE_MS:
        raise ValueError(f"{record_id}: invalid pause_after {pause_after!r}")
    low, high = PAUSE_MS[pause_after]
    return _stable_int(f"{record_id}:{key}:pause", low, high)


def episode_skeleton(record_id: str) -> dict[str, Any]:
    """Deterministic, per-record episode shape.

    Demo 1 gave every scene an identical skeleton, so structure carried no
    information.  Demo 2 varies the opening silence, the settle rhythm, the
    mid-episode long pause and the ending.
    """

    variant = _stable_int(f"{record_id}:skeleton", 0, 3)
    return {
        "variant": f"v{variant}",
        "lead_unchanged": variant in (1, 2, 3),
        "clear": variant in (0, 1, 2),
        "tail_unchanged": variant == 2,
        "mid_long_pause": _stable_int(f"{record_id}:midpause", 0, 2) == 0,
    }


# --------------------------------------------------------------------------
# quote arithmetic
# --------------------------------------------------------------------------


def unique_edit_quote(
    snapshot: str, start: int, end: int, replacement_text: str
) -> tuple[str, str]:
    """Widen ``snapshot[start:end]`` leftwards until it occurs exactly once.

    Widening never crosses a newline, so an edit quote can never swallow the
    standing-instruction line.  The edit's outcome is invariant under widening:
    the neighbouring words ride along unchanged in the replacement.
    """

    if not 0 <= start < end <= len(snapshot):
        raise ValueError("edit span is outside the snapshot")
    line_start = snapshot.rfind("\n", 0, start) + 1
    cursor = start
    while True:
        quote = snapshot[cursor:end]
        if occurrence_count(snapshot, quote) == 1:
            replacement = snapshot[cursor:start] + replacement_text
            if replacement == quote:
                raise ValueError("suggest_edit replacement is identical to its quote")
            if not quote.strip() or not replacement.strip():
                raise ValueError("suggest_edit quote and replacement must be non-blank")
            return quote, replacement
        if cursor <= line_start:
            raise ValueError(
                f"cannot make the edit quote unique within its line: {snapshot[line_start:end]!r}"
            )
        probe = cursor
        while probe > line_start and snapshot[probe - 1] == " ":
            probe -= 1
        while probe > line_start and snapshot[probe - 1] != " ":
            probe -= 1
        if probe == cursor:
            raise ValueError("edit quote widening made no progress")
        cursor = probe


def highlight_occurrence(snapshot: str, quote: str, start: int) -> int:
    """1-based occurrence of ``quote`` whose match begins at ``start``."""

    if " " in quote or not quote:
        raise ValueError("a highlight quotes exactly one word")
    occurrence = 0
    cursor = -1
    while True:
        cursor = snapshot.find(quote, cursor + 1)
        if cursor < 0:
            raise ValueError(f"highlight quote {quote!r} is not in the snapshot")
        occurrence += 1
        if cursor == start:
            if occurrence_start(snapshot, quote, occurrence) != start:
                raise ValueError("highlight occurrence index does not resolve to its word")
            return occurrence


# --------------------------------------------------------------------------
# episode builder
# --------------------------------------------------------------------------


def _idle() -> Action:
    return Action(ActionKind.IDLE)


class _Episode:
    def __init__(self, record_id: str) -> None:
        self.record_id = record_id
        self.turns: list[CompletedTurn] = []
        self.current_text = ""
        self.elapsed_ms = _stable_int(f"{record_id}:initial-time", 520, 780)

    def emit(
        self, content: str, state: UserState, gap_ms: int, action: Action | None = None
    ) -> int:
        self.elapsed_ms += gap_ms
        event = StreamEvent(
            len(self.turns) + 1, EventSource.USER, content, state, self.elapsed_ms
        )
        offset = len(self.turns)
        self.turns.append(CompletedTurn(event=event, action=action or _idle()))
        self.current_text = content
        return offset

    def set_action(self, offset: int, action: Action) -> None:
        turn = self.turns[offset]
        self.turns[offset] = CompletedTurn(event=turn.event, action=action)

    def type_text(
        self,
        addition: str,
        *,
        key: str,
        forced_cuts: Iterable[int] = (),
        optional_cuts: Iterable[int] = (),
    ) -> dict[int, int]:
        """Type ``addition`` onto the current text; return boundary -> turn offset.

        ``forced_cuts`` are trigger completions: a tick *must* end there or the
        card would grade the wrong moment, so an infeasible one is an error.
        ``optional_cuts`` are the partially-typed before-neighbour boundaries;
        they are dropped silently when the surrounding text is too short to hold
        a legal 4-character tick.
        """

        base = self.current_text
        total = len(addition)
        cuts = sorted({cut for cut in forced_cuts if 0 < cut < total} | {total})
        previous = 0
        for cut in cuts:
            span = cut - previous
            if span < MIN_TICK_CHARS and total >= MIN_TICK_CHARS:
                raise ValueError(
                    f"{self.record_id}: {key}: a tick boundary at {previous} leaves "
                    f"only {span} characters; keep 4+ characters before, between "
                    "and after triggers"
                )
            previous = cut
        for candidate in sorted({cut for cut in optional_cuts if 0 < cut < total}):
            if candidate in cuts:
                continue
            position = 0
            while position < len(cuts) and cuts[position] < candidate:
                position += 1
            lower = cuts[position - 1] if position else 0
            upper = cuts[position]
            if candidate - lower >= MIN_TICK_CHARS and upper - candidate >= MIN_TICK_CHARS:
                cuts.insert(position, candidate)
        boundaries: list[int] = []
        cursor = 0
        for span_index, cut in enumerate(cuts):
            span = cut - cursor
            if span <= 0:
                continue
            for size in _chunk_sizes(span, f"{key}:span:{span_index}"):
                cursor += size
                boundaries.append(cursor)
        offsets: dict[int, int] = {}
        for tick_index, boundary in enumerate(boundaries):
            offsets[boundary] = self.emit(
                base + addition[:boundary],
                UserState.ACTIVE,
                _typing_delta(self.record_id, key, tick_index),
            )
        return offsets

    def delete_to(self, target_text: str, *, key: str) -> list[int]:
        base = self.current_text
        if not base.startswith(target_text) or len(target_text) >= len(base):
            raise ValueError(f"{self.record_id}: {key}: deletion target is not a prefix")
        offsets: list[int] = []
        removed = 0
        for tick_index, size in enumerate(
            _chunk_sizes(len(base) - len(target_text), f"{key}:delete")
        ):
            removed += size
            offsets.append(
                self.emit(
                    base[: len(base) - removed],
                    UserState.ACTIVE,
                    _typing_delta(self.record_id, f"{key}:delete", tick_index),
                )
            )
        return offsets

    def settle(self, *, key: str, pause_after: str, count: int) -> list[int]:
        offsets: list[int] = []
        for index in range(count):
            offsets.append(
                self.emit(
                    self.current_text,
                    UserState.IDLE,
                    _pause_delta(self.record_id, f"{key}:settle:{index}", pause_after),
                )
            )
        return offsets


# --------------------------------------------------------------------------
# compilation
# --------------------------------------------------------------------------


def _validated_provenance_metadata(
    record: Mapping[str, Any], *, context: str
) -> tuple[tuple[str, str], ...]:
    metadata: list[tuple[str, str]] = []
    for field in DEMO2_PROVENANCE_FIELDS:
        if field not in record:
            raise ValueError(f"{context}: missing required provenance field {field!r}")
        value = record[field]
        if value is None or not str(value).strip():
            raise ValueError(f"{context}: provenance field {field!r} must be non-blank")
        metadata.append((field, str(value)))
    return tuple(sorted(metadata))


def assign_demo2_splits(
    records: Iterable[Mapping[str, Any]], *, dev_fraction: float = 0.1
) -> dict[str, str]:
    items = list(records)
    if not 0 <= dev_fraction <= 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    record_ids = [record.get("id") for record in items]
    if any(not isinstance(record_id, str) or not record_id for record_id in record_ids):
        raise ValueError("every Demo 2 record requires a non-empty string id")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Demo 2 record ids must be unique")
    dev_count = round(len(record_ids) * dev_fraction)
    ranked = sorted(record_ids, key=lambda record_id: _stable_rank(f"d2-split:{record_id}"))
    dev_ids = set(ranked[:dev_count])
    return {
        str(record_id): "dev" if record_id in dev_ids else "train"
        for record_id in record_ids
    }


class _CandidateBook:
    """Assigns at most one graded role per turn, by explicit precedence."""

    def __init__(self, record_id: str, mode: str) -> None:
        self.record_id = record_id
        self.mode = mode
        self._by_offset: dict[int, Demo2Candidate] = {}

    def claim(self, offset: int, role: str, *, suffix: str, **fields: Any) -> bool:
        if offset in self._by_offset:
            return False
        self._by_offset[offset] = Demo2Candidate(
            candidate_id=f"{self.record_id}:{role}:{suffix}",
            record_id=self.record_id,
            turn_offset=offset,
            role=role,
            mode=self.mode,
            **fields,
        )
        return True

    def taken(self, offset: int) -> bool:
        return offset in self._by_offset

    def finalize(self) -> tuple[Demo2Candidate, ...]:
        return tuple(
            self._by_offset[offset] for offset in sorted(self._by_offset)
        )


def _trigger_cuts(start: int, end: int) -> tuple[list[int], list[int]]:
    """Required (trigger completes) and optional (trigger half-typed) boundaries."""

    optional = (
        [end - PARTIAL_TAIL_CHARS]
        if end - start >= PARTIAL_BEFORE_MIN_TRIGGER_CHARS
        else []
    )
    return [end], optional


def compile_demo2_record(
    record: Mapping[str, Any], *, split: str = "train"
) -> CompiledDemo2Record:
    """Compile one authored Demo 2 episode into stream turns and candidates."""

    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("Demo 2 record requires a non-empty string id")
    if split not in {"train", "dev"}:
        raise ValueError("Demo 2 split must be 'train' or 'dev'")
    source_metadata = _validated_provenance_metadata(record, context=record_id)
    mode = str(record["mode"])
    if mode not in {"corrections", "highlights"}:
        raise ValueError(f"{record_id}: unknown Demo 2 mode {mode!r}")
    instruction = record.get("instruction")
    if not isinstance(instruction, Mapping):
        raise ValueError(f"{record_id}: instruction must be an object")
    instruction_text = str(instruction.get("text", ""))
    ack = str(instruction.get("ack", ""))
    if not instruction_text.strip() or not ack.strip():
        raise ValueError(f"{record_id}: instruction requires text and ack")
    segments = record.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{record_id}: segments must be a non-empty array")

    skeleton = episode_skeleton(record_id)
    episode = _Episode(record_id)
    book = _CandidateBook(record_id, mode)
    category = str(record.get("category")) if mode == "highlights" else None

    # --- opening silence -------------------------------------------------
    initial_offset = episode.emit("", UserState.IDLE, 0)
    book.claim(initial_offset, "empty-initial", suffix="initial", empty_kind="initial")
    if skeleton["lead_unchanged"]:
        unchanged_offset = episode.emit(
            "", UserState.IDLE, _stable_int(f"{record_id}:lead-gap", 540, 4_000)
        )
        book.claim(
            unchanged_offset, "empty-unchanged", suffix="lead", empty_kind="unchanged"
        )

    # --- the standing instruction and its one-shot acknowledgment --------
    required, optional = _trigger_cuts(0, len(instruction_text))
    instruction_offsets = episode.type_text(
        instruction_text,
        key="instruction",
        forced_cuts=required,
        optional_cuts=optional,
    )
    ack_offset = instruction_offsets[len(instruction_text)]
    ack_event_index = episode.turns[ack_offset].event.index
    episode.set_action(
        ack_offset, Action(ActionKind.RESPOND, target=ack_event_index, message=ack)
    )
    book.claim(
        ack_offset,
        "instruction-ack",
        suffix="instruction",
        trigger_key=f"{record_id}:instruction",
        trigger_kind="instruction",
    )
    pending: list[dict[str, Any]] = [
        {
            "key": f"{record_id}:instruction",
            "kind": "instruction",
            "positive_offset": ack_offset,
            "positive_role": "instruction-ack",
            "before_role": "instruction-before",
            "after_role": "instruction-after",
            "segment_index": None,
            "trigger_start": 0,
            "subtype": None,
            "suffix": "instruction",
        }
    ]
    episode.settle(
        key="instruction",
        pause_after="short",
        count=1 + _stable_int(f"{record_id}:instruction-settle", 0, 1),
    )

    # --- passages --------------------------------------------------------
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ValueError(f"{record_id}: segment {segment_index} must be an object")
        kind = segment.get("kind")
        text = str(segment.get("text", ""))
        pause_after = str(segment.get("pause_after", "none"))
        if not text:
            raise ValueError(f"{record_id}: segment {segment_index} text must be non-empty")
        key = f"segment:{segment_index}"
        prefix = "\n" if episode.current_text else ""

        if kind == "repair":
            _compile_repair_segment(
                episode, book, record_id, segment_index, segment, prefix, text
            )
        elif kind == "aside":
            offsets = episode.type_text(prefix + text, key=key)
            completion = offsets[len(prefix) + len(text)]
            book.claim(
                completion,
                "trap-instruction-mention",
                suffix=str(segment_index),
                segment_index=segment_index,
                trap="instruction_mention",
                pause_after=pause_after,
            )
        elif kind == "passage":
            _compile_passage_segment(
                episode,
                book,
                pending,
                record=record,
                record_id=record_id,
                segment_index=segment_index,
                segment=segment,
                prefix=prefix,
                text=text,
                pause_after=pause_after,
                category=category,
            )
        else:
            raise ValueError(f"{record_id}: segment {segment_index}: invalid kind {kind!r}")

        settle_pause = pause_after
        if skeleton["mid_long_pause"] and segment_index == len(segments) // 2:
            settle_pause = "long"
        episode.settle(
            key=key,
            pause_after=settle_pause,
            count=1 + _stable_int(f"{record_id}:{key}:settle-count", 0, 1),
        )

    # --- ending ----------------------------------------------------------
    if skeleton["clear"]:
        cleared_offset = episode.emit(
            "", UserState.IDLE, _stable_int(f"{record_id}:clear-gap", 560, 1_100)
        )
        book.claim(
            cleared_offset, "empty-cleared", suffix="cleared", empty_kind="cleared"
        )
        if skeleton["tail_unchanged"]:
            tail_offset = episode.emit(
                "", UserState.IDLE, _stable_int(f"{record_id}:tail-gap", 600, 9_000)
            )
            book.claim(
                tail_offset, "empty-unchanged", suffix="tail", empty_kind="unchanged"
            )
    else:
        episode.emit(
            episode.current_text,
            UserState.IDLE,
            _stable_int(f"{record_id}:tail-hold", 10_000, 30_000),
        )

    # --- neighbours, then ballast ---------------------------------------
    total_turns = len(episode.turns)
    for entry in pending:
        positive_offset = int(entry["positive_offset"])
        after_offset = positive_offset + 1
        if after_offset >= total_turns:
            raise ValueError(
                f"{record_id}: trigger at turn {positive_offset} has no following tick"
            )
        book.claim(
            after_offset,
            str(entry["after_role"]),
            suffix=str(entry["suffix"]),
            segment_index=entry["segment_index"],
            trigger_key=str(entry["key"]),
            trigger_kind=str(entry["kind"]),
            subtype=entry["subtype"],
            category=category,
        )
    for entry in pending:
        positive_offset = int(entry["positive_offset"])
        before_offset = positive_offset - 1
        if before_offset < 0:
            continue
        before_content = episode.turns[before_offset].event.content
        before_kind = (
            "partial"
            if len(before_content) > int(entry["trigger_start"])
            else "absent"
        )
        book.claim(
            before_offset,
            str(entry["before_role"]),
            suffix=str(entry["suffix"]),
            segment_index=entry["segment_index"],
            trigger_key=str(entry["key"]),
            trigger_kind=str(entry["kind"]),
            before_kind=before_kind,
            subtype=entry["subtype"],
            category=category,
        )
    for offset in range(total_turns):
        if book.taken(offset):
            continue
        if not episode.turns[offset].event.content:
            continue
        book.claim(offset, "ballast-idle", suffix=str(offset), category=category)

    candidates = book.finalize()
    seen_edits: set[str] = set()
    seen_marks: set[tuple[str, int]] = set()
    for turn in episode.turns:
        action = turn.action
        if action.kind is not ActionKind.TOOL:
            continue
        if action.tool_name == "suggest_edit":
            quote = str(action.arguments["quote"])
            if quote in seen_edits:
                raise ValueError(f"{record_id}: repeated suggest_edit quote {quote!r}")
            seen_edits.add(quote)
        elif action.tool_name == "highlight":
            mark = (str(action.arguments["quote"]), int(action.arguments["occurrence"]))
            if mark in seen_marks:
                raise ValueError(f"{record_id}: repeated highlight {mark[0]!r}#{mark[1]}")
            seen_marks.add(mark)
    for candidate in candidates:
        turn = episode.turns[candidate.turn_offset]
        if candidate.role in POSITIVE_ROLES:
            if turn.action.kind is ActionKind.IDLE:
                raise ValueError(f"{candidate.candidate_id}: positive graded as idle")
        elif turn.action.kind is not ActionKind.IDLE:
            raise ValueError(f"{candidate.candidate_id}: neighbour graded as an action")

    return CompiledDemo2Record(
        record_id=record_id,
        split=split,
        mode=mode,
        turns=tuple(episode.turns),
        candidates=candidates,
        source_metadata=source_metadata,
        skeleton=str(skeleton["variant"]),
    )


def _compile_passage_segment(
    episode: _Episode,
    book: _CandidateBook,
    pending: list[dict[str, Any]],
    *,
    record: Mapping[str, Any],
    record_id: str,
    segment_index: int,
    segment: Mapping[str, Any],
    prefix: str,
    text: str,
    pause_after: str,
    category: str | None,
) -> None:
    key = f"segment:{segment_index}"
    base_length = len(episode.current_text)
    shift = base_length + len(prefix)
    triggers = segment_trigger_layout(record, segment)
    forced: list[int] = []
    optional: list[int] = []
    for trigger in triggers:
        required_cuts, optional_cuts = _trigger_cuts(
            len(prefix) + int(trigger["start"]), len(prefix) + int(trigger["end"])
        )
        forced.extend(required_cuts)
        optional.extend(optional_cuts)
    offsets = episode.type_text(
        prefix + text, key=key, forced_cuts=forced, optional_cuts=optional
    )
    completion_offset = offsets[len(prefix) + len(text)]

    emitted_edits: set[str] = set()
    emitted_marks: set[tuple[str, int]] = set()
    for trigger in triggers:
        boundary = len(prefix) + int(trigger["end"])
        positive_offset = offsets[boundary]
        snapshot = episode.turns[positive_offset].event.content
        absolute_start = shift + int(trigger["start"])
        absolute_end = shift + int(trigger["end"])
        if absolute_end != len(snapshot):
            raise ValueError(
                f"{record_id}: segment {segment_index}: trigger does not end the snapshot"
            )
        role = str(trigger["role"])
        if role == "error":
            quote, replacement = unique_edit_quote(
                snapshot, absolute_start, absolute_end, str(trigger["right"])
            )
            if quote in emitted_edits:
                raise ValueError(f"{record_id}: repeated suggest_edit quote {quote!r}")
            emitted_edits.add(quote)
            episode.set_action(
                positive_offset,
                Action(
                    ActionKind.TOOL,
                    tool_name="suggest_edit",
                    arguments={"quote": quote, "replacement": replacement},
                ),
            )
            book.claim(
                positive_offset,
                "error-positive",
                suffix=f"{segment_index}:{trigger['start']}",
                segment_index=segment_index,
                trigger_key=f"{record_id}:error:{segment_index}:{trigger['start']}",
                trigger_kind="error",
                subtype=str(trigger["subtype"]),
                pause_after=pause_after,
            )
            pending.append(
                {
                    "key": f"{record_id}:error:{segment_index}:{trigger['start']}",
                    "kind": "error",
                    "positive_offset": positive_offset,
                    "positive_role": "error-positive",
                    "before_role": "error-before",
                    "after_role": "error-after",
                    "segment_index": segment_index,
                    "trigger_start": absolute_start,
                    "subtype": str(trigger["subtype"]),
                    "suffix": f"{segment_index}:{trigger['start']}",
                }
            )
        elif role == "match":
            quote = str(trigger["word"])
            occurrence = highlight_occurrence(snapshot, quote, absolute_start)
            if (quote, occurrence) in emitted_marks:
                raise ValueError(f"{record_id}: repeated highlight {quote!r}#{occurrence}")
            emitted_marks.add((quote, occurrence))
            episode.set_action(
                positive_offset,
                Action(
                    ActionKind.TOOL,
                    tool_name="highlight",
                    arguments={"occurrence": occurrence, "quote": quote},
                ),
            )
            book.claim(
                positive_offset,
                "match-positive",
                suffix=f"{segment_index}:{trigger['start']}",
                segment_index=segment_index,
                trigger_key=f"{record_id}:match:{segment_index}:{trigger['start']}",
                trigger_kind="match",
                category=category,
                pause_after=pause_after,
            )
            pending.append(
                {
                    "key": f"{record_id}:match:{segment_index}:{trigger['start']}",
                    "kind": "match",
                    "positive_offset": positive_offset,
                    "positive_role": "match-positive",
                    "before_role": "match-before",
                    "after_role": "match-after",
                    "segment_index": segment_index,
                    "trigger_start": absolute_start,
                    "subtype": None,
                    "suffix": f"{segment_index}:{trigger['start']}",
                }
            )
        elif role == "bait":
            book.claim(
                positive_offset,
                "trap-non-literal",
                suffix=f"{segment_index}:{trigger['start']}",
                segment_index=segment_index,
                trap="non_literal_match",
                category=category,
                pause_after=pause_after,
            )
        else:  # pragma: no cover - segment_trigger_layout emits no other role
            raise ValueError(f"{record_id}: unknown trigger role {role!r}")

    if segment.get("trap") == "clean_text":
        book.claim(
            completion_offset,
            "trap-clean",
            suffix=str(segment_index),
            segment_index=segment_index,
            trap="clean_text",
            category=category,
            pause_after=pause_after,
        )


def _compile_repair_segment(
    episode: _Episode,
    book: _CandidateBook,
    record_id: str,
    segment_index: int,
    segment: Mapping[str, Any],
    prefix: str,
    text: str,
) -> None:
    """Type an error, then watch the typist repair it before we ever fire.

    The tick where ``wrong`` completes is deliberately *ungraded* padding: at
    that instant the model cannot know a repair is coming, so grading it either
    way would teach a lie.  The graded hard idle lands on the tick where the
    text is correct again — the lesson is "that error is gone; do nothing".
    """

    repair = segment.get("repair")
    if not isinstance(repair, Mapping):
        raise ValueError(f"{record_id}: segment {segment_index}: repair object required")
    wrong = str(repair["wrong"])
    right = str(repair["right"])
    occurrence = int(repair.get("occurrence", 1))
    key = f"segment:{segment_index}"

    start = occurrence_start(text, right, occurrence)
    if start < 0:
        raise ValueError(
            f"{record_id}: segment {segment_index}: repair target {right!r} not found"
        )
    if len(right) < MIN_TICK_CHARS:
        raise ValueError(
            f"{record_id}: segment {segment_index}: repair.right must be 4+ characters"
        )
    tail = text[start + len(right) :]
    # An overshoot of 1-3 characters would force an illegal short tick, so the
    # typist either stops at the typo or runs a full tick past it.
    widest_overshoot = min(6, len(tail))
    overshoot = 0
    if widest_overshoot >= MIN_TICK_CHARS and _stable_int(
        f"{record_id}:{key}:overshoot-flag", 0, 1
    ):
        overshoot = _stable_int(
            f"{record_id}:{key}:overshoot", MIN_TICK_CHARS, widest_overshoot
        )
    draft = prefix + text[:start] + wrong + tail[:overshoot]
    episode.type_text(
        draft, key=f"{key}:draft", forced_cuts=[len(prefix) + start + len(wrong)]
    )
    correct_prefix = episode.current_text[: len(episode.current_text) - len(wrong) - overshoot]
    episode.delete_to(correct_prefix, key=key)
    remainder = text[start:]
    repair_required, repair_optional = _trigger_cuts(0, len(right))
    repaired = episode.type_text(
        remainder,
        key=f"{key}:repair",
        forced_cuts=repair_required,
        optional_cuts=repair_optional,
    )
    book.claim(
        repaired[len(right)],
        "trap-selfcorrect",
        suffix=str(segment_index),
        segment_index=segment_index,
        trap="self_correction",
        subtype=str(repair.get("subtype")),
    )


def compile_demo2_records(
    records: Iterable[Mapping[str, Any]], *, dev_fraction: float = 0.1
) -> tuple[CompiledDemo2Record, ...]:
    items = list(records)
    split_by_id = assign_demo2_splits(items, dev_fraction=dev_fraction)
    return tuple(
        compile_demo2_record(record, split=split_by_id[str(record["id"])])
        for record in sorted(items, key=lambda item: str(item["id"]))
    )


def demo2_records_from_batches(
    batches: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches):
        author = batch.get("author")
        raw_records = batch.get("records")
        if not isinstance(author, Mapping) or not isinstance(raw_records, list):
            raise ValueError(f"batch {batch_index}: author object and records array required")
        for record_index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, Mapping):
                raise ValueError(
                    f"batch {batch_index}: record {record_index} must be an object"
                )
            record = dict(raw_record)
            record["author_slot"] = author.get("slot")
            record["author_model"] = author.get("model")
            record["author_tranche"] = author.get("tranche")
            _validated_provenance_metadata(
                record, context=f"batch {batch_index}: record {record_index}"
            )
            record["author_slot"] = str(record["author_slot"])
            record["author_model"] = str(record["author_model"])
            record["author_tranche"] = str(record["author_tranche"])
            records.append(record)
    return tuple(records)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def _allocate_round_robin(budget: int, pools: Mapping[str, int]) -> dict[str, int]:
    """Spread ``budget`` as evenly as possible over pools, capped by capacity."""

    allocation = {name: 0 for name in pools}
    remaining = budget
    while remaining > 0:
        progressed = False
        for name in sorted(pools):
            if remaining == 0:
                break
            if allocation[name] < pools[name]:
                allocation[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return allocation


def _take(pool: Sequence[Demo2Candidate], count: int, salt: str) -> list[Demo2Candidate]:
    return sorted(pool, key=lambda item: _stable_rank(f"{salt}:{item.candidate_id}"))[:count]


def select_demo2_candidates(
    records: Iterable[CompiledDemo2Record],
    *,
    targets: Demo2Targets = Demo2Targets(),
) -> tuple[Demo2Candidate, ...]:
    """Select the exact production card mix, or fail with a count diagnosis.

    Order of claim, strongest guarantee first:

    1. every positive (instruction ack, suggest_edit, highlight) ships;
    2. ``empty_per_kind`` cards of each empty-text kind;
    3. ``hard_idle_cards`` trap cards, spread round-robin over the four traps;
    4. before/after neighbour *pairs*, spread over the three trigger kinds, so a
       positive either gets both of its neighbours or neither;
    5. ordinary ballast idles fill any remainder.
    """

    record_items = tuple(records)
    candidates = [candidate for record in record_items for candidate in record.candidates]
    by_role: dict[str, list[Demo2Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_role[candidate.role].append(candidate)

    source_checks = (
        ("episodes", targets.episodes, len(by_role["instruction-ack"])),
        ("errors", targets.errors, len(by_role["error-positive"])),
        ("matches", targets.matches, len(by_role["match-positive"])),
    )
    for name, expected, actual in source_checks:
        if expected != actual:
            raise ValueError(
                f"Demo 2 source requires exactly {expected} {name}; found {actual}"
            )

    selected: list[Demo2Candidate] = []
    for role in POSITIVE_ROLES:
        selected.extend(by_role[role])

    for empty_kind in EMPTY_KINDS:
        pool = [item for item in candidates if item.empty_kind == empty_kind]
        if len(pool) < targets.empty_per_kind:
            raise ValueError(
                f"Demo 2 needs {targets.empty_per_kind} {empty_kind} empty cards; "
                f"found {len(pool)}"
            )
        selected.extend(_take(pool, targets.empty_per_kind, "empty"))

    trap_pools = {role: by_role[role] for role in TRAP_ROLES}
    missing_traps = [role for role, pool in trap_pools.items() if not pool]
    if missing_traps:
        raise ValueError(f"Demo 2 requires every trap class; missing {missing_traps}")
    trap_allocation = _allocate_round_robin(
        targets.hard_idle_cards, {role: len(pool) for role, pool in trap_pools.items()}
    )
    if sum(trap_allocation.values()) != targets.hard_idle_cards:
        raise ValueError(
            f"Demo 2 needs {targets.hard_idle_cards} hard-idle trap cards; "
            f"only {sum(len(pool) for pool in trap_pools.values())} are available"
        )
    for role, count in sorted(trap_allocation.items()):
        selected.extend(_take(trap_pools[role], count, "trap"))

    remaining = targets.cards - len(selected)
    if remaining < 0:
        raise ValueError(
            f"Demo 2 mandatory cards ({len(selected)}) exceed the target of {targets.cards}"
        )
    pairs_by_kind: dict[str, list[tuple[Demo2Candidate, Demo2Candidate]]] = {
        kind: [] for kind in TRIGGER_KINDS
    }
    before_by_key = {item.trigger_key: item for item in candidates if item.role in BEFORE_ROLES}
    after_by_key = {item.trigger_key: item for item in candidates if item.role in AFTER_ROLES}
    for key, before in sorted(before_by_key.items(), key=lambda item: str(item[0])):
        after = after_by_key.get(key)
        if after is None or before.trigger_kind not in pairs_by_kind:
            continue
        pairs_by_kind[str(before.trigger_kind)].append((before, after))
    pair_budget = remaining // 2
    pair_allocation = _allocate_round_robin(
        pair_budget, {kind: len(pairs) for kind, pairs in pairs_by_kind.items()}
    )
    for kind, count in sorted(pair_allocation.items()):
        chosen = sorted(
            pairs_by_kind[kind],
            key=lambda pair: _stable_rank(f"pair:{pair[0].candidate_id}"),
        )[:count]
        for before, after in chosen:
            selected.extend((before, after))

    ballast_needed = targets.cards - len(selected)
    if ballast_needed < 0:
        raise ValueError("Demo 2 neighbour selection overshot the card target")
    ballast_pool = by_role["ballast-idle"]
    if len(ballast_pool) < ballast_needed:
        raise ValueError(
            f"Demo 2 needs {ballast_needed} ballast idle cards; found {len(ballast_pool)}"
        )
    selected.extend(_take(ballast_pool, ballast_needed, "ballast"))

    if len(selected) != targets.cards:
        raise AssertionError(
            f"internal Demo 2 selector error: selected {len(selected)} cards, "
            f"expected {targets.cards}"
        )
    unique_ids = {candidate.candidate_id for candidate in selected}
    if len(unique_ids) != len(selected):
        raise AssertionError("internal Demo 2 selector error: duplicate candidate")
    return tuple(sorted(selected, key=lambda item: item.candidate_id))


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _expected_class(action: Action) -> str:
    if action.kind is ActionKind.TOOL:
        return str(action.tool_name)
    return action.kind.value


def render_demo2_card(
    record: CompiledDemo2Record, candidate: Demo2Candidate
) -> dict[str, Any]:
    """Render one card, proving it round-trips through the serving parser."""

    if candidate.record_id != record.record_id:
        raise ValueError("candidate does not belong to the supplied Demo 2 record")
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

    visible = current_turn.event.content
    if parsed.kind is ActionKind.TOOL and parsed.tool_name == "suggest_edit":
        quote = str(parsed.arguments["quote"])
        if occurrence_count(visible, quote) != 1:
            raise ValueError(
                f"{candidate.candidate_id}: suggest_edit quote is not unique in the textbox"
            )
        if str(parsed.arguments["replacement"]) == quote:
            raise ValueError(f"{candidate.candidate_id}: suggest_edit is a no-op")
    if parsed.kind is ActionKind.TOOL and parsed.tool_name == "highlight":
        quote = str(parsed.arguments["quote"])
        occurrence = int(parsed.arguments["occurrence"])
        if occurrence_start(visible, quote, occurrence) < 0:
            raise ValueError(
                f"{candidate.candidate_id}: highlight occurrence {occurrence} does not exist"
            )
    if parsed.kind is ActionKind.RESPOND and parsed.target != current_turn.event.index:
        raise ValueError(f"{candidate.candidate_id}: respond target is not the current event")

    metadata = dict(record.source_metadata)
    return {
        "schema_version": G1_SCHEMA_VERSION,
        "split": record.split,
        "episode": record.record_id,
        "demo": DEMO2_DEMO,
        "situation": candidate.role,
        "bucket": candidate.role,
        "prompt": prompt,
        "completion": completion,
        "expected_class": _expected_class(current_turn.action),
        "current_event_index": current_turn.event.index,
        "current_content_empty": current_turn.event.content == "",
        "candidate_id": candidate.candidate_id,
        "candidate_role": candidate.role,
        "source_record_id": record.record_id,
        "source_mode": record.mode,
        "source_skeleton": record.skeleton,
        "source_persona": metadata.get("persona"),
        "source_domain": metadata.get("domain"),
        "source_register": metadata.get("register"),
        "source_author_slot": metadata.get("author_slot"),
        "source_author_model": metadata.get("author_model"),
        "source_author_tranche": metadata.get("author_tranche"),
        "segment_index": candidate.segment_index,
        "trigger_key": candidate.trigger_key,
        "trigger_kind": candidate.trigger_kind,
        "trap": candidate.trap,
        "before_kind": candidate.before_kind,
        "error_subtype": candidate.subtype,
        "category": candidate.category,
        "pause_after": candidate.pause_after,
        "empty_kind": candidate.empty_kind,
        "obligation": "standing-instruction",
    }


def demo2_coverage_report(
    records: Iterable[CompiledDemo2Record], selected: Iterable[Demo2Candidate]
) -> dict[str, Any]:
    record_items = tuple(records)
    selected_items = tuple(selected)
    record_by_id = {record.record_id: record for record in record_items}
    selected_source: dict[str, Counter[str]] = {
        field: Counter() for field in ("mode", "persona", "domain", "register", "author_slot")
    }
    for candidate in selected_items:
        metadata = dict(record_by_id[candidate.record_id].source_metadata)
        for field, counts in selected_source.items():
            if field in metadata:
                counts[metadata[field]] += 1

    non_empty_deltas: list[int] = []
    time_gaps: list[int] = []
    for record in record_items:
        prior_text = ""
        prior_time = 0
        for turn in record.turns:
            content = turn.event.content
            if content and content.startswith(prior_text) and len(content) > len(prior_text):
                non_empty_deltas.append(len(content) - len(prior_text))
            if prior_time:
                time_gaps.append(int(turn.event.elapsed_ms) - prior_time)
            prior_text = content
            prior_time = int(turn.event.elapsed_ms)

    return {
        "records": len(record_items),
        "source_splits": dict(sorted(Counter(record.split for record in record_items).items())),
        "source_modes": dict(sorted(Counter(record.mode for record in record_items).items())),
        "source_skeletons": dict(
            sorted(Counter(record.skeleton for record in record_items).items())
        ),
        "selected_cards": len(selected_items),
        "selected_roles": dict(
            sorted(Counter(candidate.role for candidate in selected_items).items())
        ),
        "selected_traps": dict(
            sorted(
                Counter(
                    candidate.trap for candidate in selected_items if candidate.trap
                ).items()
            )
        ),
        "selected_empty_kinds": dict(
            sorted(
                Counter(
                    candidate.empty_kind
                    for candidate in selected_items
                    if candidate.empty_kind
                ).items()
            )
        ),
        "selected_before_kinds": dict(
            sorted(
                Counter(
                    candidate.before_kind
                    for candidate in selected_items
                    if candidate.before_kind
                ).items()
            )
        ),
        "selected_error_subtypes": dict(
            sorted(
                Counter(
                    candidate.subtype for candidate in selected_items if candidate.subtype
                ).items()
            )
        ),
        "selected_categories": dict(
            sorted(
                Counter(
                    candidate.category for candidate in selected_items if candidate.category
                ).items()
            )
        ),
        "selected_splits": dict(
            sorted(
                Counter(
                    record_by_id[candidate.record_id].split for candidate in selected_items
                ).items()
            )
        ),
        "selected_source_distribution": {
            field: dict(sorted(counts.items())) for field, counts in selected_source.items()
        },
        "source_turns": sum(len(record.turns) for record in record_items),
        "typing_delta_chars": {
            "min": min(non_empty_deltas) if non_empty_deltas else None,
            "max": max(non_empty_deltas) if non_empty_deltas else None,
        },
        "time_gap_ms": {
            "min": min(time_gaps) if time_gaps else None,
            "max": max(time_gaps) if time_gaps else None,
            "distinct": len(set(time_gaps)),
        },
    }


def compile_demo2_dataset(
    records: Iterable[Mapping[str, Any]],
    *,
    targets: Demo2Targets = Demo2Targets(),
    dev_fraction: float = 0.1,
) -> Demo2Build:
    compiled = compile_demo2_records(records, dev_fraction=dev_fraction)
    selected = select_demo2_candidates(compiled, targets=targets)
    record_by_id = {record.record_id: record for record in compiled}
    rows = tuple(
        render_demo2_card(record_by_id[candidate.record_id], candidate)
        for candidate in selected
    )
    return Demo2Build(
        records=compiled,
        selected=selected,
        rows=rows,
        coverage=demo2_coverage_report(compiled, selected),
    )
