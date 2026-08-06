"""Deterministic compiler for agent-authored Demo 3 situations.

Demo 3 is live English -> Chinese translation.  An authored record is a typing
*script*: the operator types a standing translation instruction, then types an
English passage in which clause boundaries are marked explicitly.  This module
turns that script into the live app's full-textbox event stream, keeps every
ungraded typing tick in history, and selects the production card slice.

Two properties this module exists to guarantee:

1. **The clause-commit boundary is exact.**  A clause commits on the tick whose
   snapshot ends at the authored boundary offset -- never a tick earlier, never
   a tick later.  Boundaries are authored as an in-text marker *and* as the
   clause's English span; the planner cross-checks the two and refuses to
   compile if they disagree.
2. **Train equals serve.**  Prompts come from the same ``compile_stream(...,
   fmt="g1")`` the runtime uses, completions from ``g1_action_completion``, and
   every completion is round-tripped through ``parse_g1_action`` before it may
   become a card.  Chinese survives that round trip byte-exactly or the build
   fails loudly.

Everything is derived by hashing UTF-8 bytes.  There is no RNG and no seed:
the same authored source produces byte-identical cards forever, on any
platform.

The small hashing/cadence helpers are duplicated from ``datagen.g1_demo1``
rather than imported: Demo 1 is frozen while a concurrent session works in it,
and a later consolidation pass will merge the two copies.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.domain import Action, ActionKind, CompletedTurn, EventSource, StreamEvent, UserState
from app.stream import compile_stream, g1_action_completion, parse_g1_action


G1_SCHEMA_VERSION = "g1"
DEMO = "demo-3"

#: The clause-commit marker.  U+2016 DOUBLE VERTICAL LINE: one character, not
#: produced by any ordinary keyboard, and never legitimate inside authored
#: English prose or a Chinese reference.  It marks the position *immediately
#: after* the clause's final character and is stripped before compilation, so
#: it never reaches a prompt.
CLAUSE_MARKER = "‖"

#: A committing clause piece must be long enough to chunk into at least two
#: human-cadence ticks, so that the tick before the commit is a real
#: "clause not complete yet" neighbour rather than the previous clause's
#: respond.  Seven characters is the largest single tick, so twelve is safe.
MIN_COMMIT_PIECE_CHARS = 12
MIN_INSTRUCTION_CHARS = 12

DEMO3_PROVENANCE_FIELDS = (
    "persona",
    "domain",
    "register",
    "author_slot",
    "author_model",
    "author_tranche",
    "reference_review_method",
    "reference_review_reviewer",
)

STEP_TRAPS = ("none", "half_clause", "prequoted_chinese")

DEMO3_ROLES = (
    "instruction-before",
    "instruction-ack",
    "instruction-after",
    "clause-before",
    "clause-positive",
    "clause-after",
    "backtrack-idle",
    "prequoted-idle",
    "partial-idle",
    "empty-initial",
    "empty-unchanged",
    "empty-cleared",
)

#: When two candidate roles land on the same compiled turn, the higher-priority
#: role wins and the other is dropped.  Two cards with an identical prompt would
#: otherwise be emitted under different candidate ids.
ROLE_PRIORITY = {role: index for index, role in enumerate(DEMO3_ROLES)}

TRAP_ROLES = ("backtrack-idle", "prequoted-idle", "partial-idle")
NEIGHBOR_ROLES = (
    "clause-before",
    "clause-after",
    "instruction-before",
    "instruction-after",
)
EMPTY_KINDS = ("initial", "unchanged", "cleared")

PAUSE_MS = {
    "none": (540, 820),
    "short": (1_000, 3_000),
    "medium": (4_000, 9_000),
    "long": (10_000, 30_000),
}
PAUSE_VALUES = tuple(PAUSE_MS)

#: Demo 1 gave every scene an identical skeleton, which made structure a
#: constant across the whole demo.  Demo 3 derives its opening and closing
#: shapes from the record id, so silence appears in four different places
#: rather than always the same two.
OPENING_SHAPES = ("immediate", "idle", "idle-idle", "idle-long-idle")
CLOSING_SHAPES = ("clear", "clear-hold", "hold", "hold-clear")


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
    """Deterministic opening skeleton for one episode."""

    return _stable_choice(f"{record_id}:opening-shape", OPENING_SHAPES)


def closing_shape(record_id: str) -> str:
    """Deterministic closing skeleton for one episode."""

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
            raise ValueError(
                f"cannot partition {length} characters into 4-7 character ticks"
            )
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
    # The app normally ticks at 650ms.  Small deterministic jitter keeps the
    # clock from becoming decorative furniture in training histories.
    return _stable_int(f"{record_id}:{key}:tick:{tick_index}:time", 520, 820)


def _idle() -> Action:
    return Action(ActionKind.IDLE)


def _respond(target: int, message: str) -> Action:
    return Action(ActionKind.RESPOND, target=target, message=message)


def _user_event(index: int, content: str, state: UserState, elapsed_ms: int) -> StreamEvent:
    return StreamEvent(index, EventSource.USER, content, state, elapsed_ms)


def _turn(event: StreamEvent, action: Action) -> CompletedTurn:
    return CompletedTurn(event=event, action=action)


# --------------------------------------------------------------------------
# script planning: clause boundaries as exact character offsets
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScriptPiece:
    """One contiguous run of appended passage text between clause markers."""

    step_index: int
    piece_index: int
    text: str
    clause_index: int | None
    ends_step: bool


@dataclass(frozen=True, slots=True)
class ClauseSpan:
    """An authored clause resolved to exact offsets in the passage text."""

    clause_index: int
    start_offset: int
    end_offset: int
    span: str
    english: str
    reference: str


@dataclass(frozen=True, slots=True)
class ScriptStep:
    index: int
    kind: str
    text: str = ""
    delete: int = 0
    trap: str = "none"
    pause_after: str = "none"


@dataclass(frozen=True, slots=True)
class Demo3Plan:
    """A fully resolved typing script: pieces, deletions, and clause offsets."""

    record_id: str
    steps: tuple[ScriptStep, ...]
    pieces: tuple[ScriptPiece, ...]
    clause_spans: tuple[ClauseSpan, ...]
    final_passage: str


def _require_str(value: Any, context: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: {field} must be a non-empty string")
    return value


def normalize_demo3_steps(
    record_id: str, script: Any
) -> tuple[ScriptStep, ...]:
    """Validate the raw authored script into typed steps."""

    if not isinstance(script, list) or not script:
        raise ValueError(f"{record_id}: script must be a non-empty array")
    steps: list[ScriptStep] = []
    for index, raw in enumerate(script):
        context = f"{record_id}: script[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{context}: step must be an object")
        kind = raw.get("kind")
        pause_after = raw.get("pause_after", "none")
        if pause_after not in PAUSE_MS:
            raise ValueError(f"{context}: invalid pause_after {pause_after!r}")
        if kind == "type":
            text = _require_str(raw.get("text"), context, "text")
            trap = raw.get("trap", "none")
            if trap not in STEP_TRAPS:
                raise ValueError(f"{context}: invalid trap {trap!r}")
            if len(text.replace(CLAUSE_MARKER, "")) < 4:
                raise ValueError(f"{context}: a type step must add at least 4 characters")
            if trap == "half_clause" and text.endswith(CLAUSE_MARKER):
                raise ValueError(
                    f"{context}: a half_clause step must end mid-clause, "
                    "so its text must not end with a clause marker"
                )
            if trap == "prequoted_chinese":
                if CLAUSE_MARKER in text:
                    raise ValueError(
                        f"{context}: prequoted_chinese text already contains Chinese "
                        "and is never a committable clause; remove the clause marker"
                    )
                if not any(_is_han(character) for character in text):
                    raise ValueError(
                        f"{context}: prequoted_chinese requires Han characters in the text"
                    )
            steps.append(
                ScriptStep(
                    index=index,
                    kind="type",
                    text=text,
                    trap=trap,
                    pause_after=pause_after,
                )
            )
        elif kind == "backtrack":
            delete = raw.get("delete")
            if not isinstance(delete, int) or isinstance(delete, bool) or delete <= 0:
                raise ValueError(f"{context}: delete must be a positive integer")
            steps.append(
                ScriptStep(
                    index=index,
                    kind="backtrack",
                    delete=delete,
                    trap="backtrack",
                    pause_after=pause_after,
                )
            )
        else:
            raise ValueError(f"{context}: invalid kind {kind!r}")
    if steps[0].kind != "type":
        raise ValueError(f"{record_id}: script must open with a type step")
    return tuple(steps)


def _is_han(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def plan_demo3_script(record: Mapping[str, Any]) -> Demo3Plan:
    """Resolve an authored script into pieces and exact clause boundaries.

    The author states each boundary twice: once positionally, as a
    ``CLAUSE_MARKER`` in the script text, and once textually, as the clause's
    ``english`` span in the ``clauses`` array.  Disagreement is a hard error --
    that redundancy is what makes the commit tick unambiguous.
    """

    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("Demo 3 record requires a non-empty string id")
    steps = normalize_demo3_steps(record_id, record.get("script"))

    clauses = record.get("clauses")
    if not isinstance(clauses, list) or not clauses:
        raise ValueError(f"{record_id}: clauses must be a non-empty array")

    passage = ""
    last_commit_offset = 0
    clause_cursor = 0
    pieces: list[ScriptPiece] = []
    spans: list[ClauseSpan] = []

    for step in steps:
        context = f"{record_id}: script[{step.index}]"
        if step.kind == "backtrack":
            pending = len(passage) - last_commit_offset
            if pending <= 0:
                raise ValueError(
                    f"{context}: nothing uncommitted to backtrack over"
                )
            if step.delete > pending:
                raise ValueError(
                    f"{context}: backtrack of {step.delete} characters would delete "
                    f"committed text; at most {pending} characters are uncommitted"
                )
            passage = passage[: len(passage) - step.delete]
            continue

        if step.trap == "prequoted_chinese" and len(passage) != last_commit_offset:
            raise ValueError(
                f"{context}: a prequoted_chinese run must begin at a clause boundary, "
                "so that the untranslatable text is never folded into a clause span; "
                f"{len(passage) - last_commit_offset} characters are still uncommitted"
            )

        parts = step.text.split(CLAUSE_MARKER)
        for part_index, part in enumerate(parts):
            commits = part_index < len(parts) - 1
            if not part:
                if commits:
                    raise ValueError(f"{context}: empty clause piece before a marker")
                continue
            if commits and len(part) < MIN_COMMIT_PIECE_CHARS:
                raise ValueError(
                    f"{context}: the {MIN_COMMIT_PIECE_CHARS}-character minimum before a "
                    f"clause marker is not met (got {len(part)}); a shorter run cannot "
                    "produce a 'not yet' neighbour on its own clause"
                )
            passage += part
            clause_index: int | None = None
            if commits:
                if clause_cursor >= len(clauses):
                    raise ValueError(
                        f"{context}: more clause markers than authored clauses"
                    )
                raw_clause = clauses[clause_cursor]
                if not isinstance(raw_clause, Mapping):
                    raise ValueError(
                        f"{record_id}: clauses[{clause_cursor}] must be an object"
                    )
                english = _require_str(
                    raw_clause.get("english"),
                    f"{record_id}: clauses[{clause_cursor}]",
                    "english",
                )
                reference = _require_str(
                    raw_clause.get("reference"),
                    f"{record_id}: clauses[{clause_cursor}]",
                    "reference",
                )
                span = passage[last_commit_offset:]
                if span != span.rstrip():
                    raise ValueError(
                        f"{context}: a clause marker must follow the clause's last "
                        "character, not trailing whitespace"
                    )
                if span.strip() != english:
                    raise ValueError(
                        f"{record_id}: clauses[{clause_cursor}].english does not match the "
                        f"marked span; script says {span.strip()!r}, clause says {english!r}"
                    )
                spans.append(
                    ClauseSpan(
                        clause_index=clause_cursor,
                        start_offset=last_commit_offset,
                        end_offset=len(passage),
                        span=span,
                        english=english,
                        reference=reference,
                    )
                )
                clause_index = clause_cursor
                clause_cursor += 1
                last_commit_offset = len(passage)
            pieces.append(
                ScriptPiece(
                    step_index=step.index,
                    piece_index=part_index,
                    text=part,
                    clause_index=clause_index,
                    ends_step=part_index == len(parts) - 1
                    or (part_index == len(parts) - 2 and parts[-1] == ""),
                )
            )

        if step.trap == "prequoted_chinese":
            # The English already quotes Chinese: this run is explicitly not
            # translatable, so it is excluded from the next clause's span
            # rather than silently swallowed by it.
            last_commit_offset = len(passage)

    if clause_cursor != len(clauses):
        raise ValueError(
            f"{record_id}: {len(clauses)} clauses authored but {clause_cursor} clause "
            "markers found in the script"
        )
    return Demo3Plan(
        record_id=record_id,
        steps=steps,
        pieces=tuple(pieces),
        clause_spans=tuple(spans),
        final_passage=passage,
    )


# --------------------------------------------------------------------------
# targets, candidates, compiled records
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Demo3Targets:
    """Source and selected-card counts for a Demo 3 build.

    Defaults reach the 1,300-card Demo 3 slice from 532 authored clause
    commits across 102 episodes.  Every number is a parameter: the build CLI
    exposes each as a flag so scaling is a command-line change, not an edit.
    """

    episodes: int = 102
    clauses: int = 532
    cards: int = 1_300
    empty_per_kind: int = 40
    min_backtrack_idles: int = 37
    min_prequoted_idles: int = 40
    min_partial_idles: int = 39
    neighbor_weights: tuple[tuple[str, int], ...] = (
        ("clause-before", 50),
        ("clause-after", 20),
        ("instruction-before", 15),
        ("instruction-after", 15),
    )

    def __post_init__(self) -> None:
        if self.episodes <= 0 or self.clauses <= 0 or self.cards <= 0:
            raise ValueError("Demo 3 episode, clause, and card targets must be positive")
        if self.empty_per_kind < 0:
            raise ValueError("empty_per_kind must not be negative")
        roles = [role for role, _ in self.neighbor_weights]
        if sorted(roles) != sorted(NEIGHBOR_ROLES):
            raise ValueError(
                f"neighbor_weights must cover exactly {sorted(NEIGHBOR_ROLES)}"
            )
        if any(weight <= 0 for _, weight in self.neighbor_weights):
            raise ValueError("neighbor weights must be positive")


@dataclass(frozen=True, slots=True)
class Demo3Candidate:
    """A gradable turn in a compiled record; padding turns are not candidates."""

    candidate_id: str
    record_id: str
    turn_offset: int
    role: str
    clause_index: int | None = None
    step_index: int | None = None
    trap: str | None = None
    pause_after: str | None = None
    empty_kind: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLE_PRIORITY:
            raise ValueError(f"unknown Demo 3 candidate role: {self.role!r}")


@dataclass(frozen=True, slots=True)
class CompiledDemo3Record:
    record_id: str
    split: str
    turns: tuple[CompletedTurn, ...]
    candidates: tuple[Demo3Candidate, ...]
    plan: Demo3Plan
    opening_shape: str
    closing_shape: str
    source_metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Demo3Build:
    records: tuple[CompiledDemo3Record, ...]
    selected: tuple[Demo3Candidate, ...]
    rows: tuple[dict[str, Any], ...]
    coverage: dict[str, Any]


def _validated_provenance_metadata(
    record: Mapping[str, Any], *, context: str
) -> tuple[tuple[str, str], ...]:
    metadata: list[tuple[str, str]] = []
    for field in DEMO3_PROVENANCE_FIELDS:
        if field not in record:
            raise ValueError(f"{context}: missing required provenance field {field!r}")
        value = record[field]
        if value is None or not str(value).strip():
            raise ValueError(f"{context}: provenance field {field!r} must be non-blank")
        metadata.append((field, str(value)))
    return tuple(sorted(metadata))


def assign_demo3_splits(
    records: Iterable[Mapping[str, Any]], *, dev_fraction: float = 0.1
) -> dict[str, str]:
    """Assign whole records to a deterministic, exact-count train/dev split."""

    items = list(records)
    if not 0 <= dev_fraction <= 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    record_ids = [record.get("id") for record in items]
    if any(not isinstance(record_id, str) or not record_id for record_id in record_ids):
        raise ValueError("every Demo 3 record requires a non-empty string id")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Demo 3 record ids must be unique")
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
    clause_index: int | None = None,
    step_index: int | None = None,
    trap: str | None = None,
    pause_after: str | None = None,
    empty_kind: str | None = None,
) -> Demo3Candidate:
    return Demo3Candidate(
        candidate_id=f"{record_id}:{role}:{key}",
        record_id=record_id,
        turn_offset=turn_offset,
        role=role,
        clause_index=clause_index,
        step_index=step_index,
        trap=trap,
        pause_after=pause_after,
        empty_kind=empty_kind,
    )


def _dedupe_candidates(
    candidates: Sequence[Demo3Candidate],
) -> tuple[Demo3Candidate, ...]:
    """Keep one candidate per compiled turn, highest role priority wins.

    Two candidates on one turn would emit two cards with byte-identical prompts
    under different ids.  The skeleton varies per episode, so occasional
    collisions (an after-neighbour landing on a trap tick, say) are expected
    rather than exceptional.
    """

    best: dict[int, Demo3Candidate] = {}
    for candidate in candidates:
        existing = best.get(candidate.turn_offset)
        if existing is None or ROLE_PRIORITY[candidate.role] < ROLE_PRIORITY[existing.role]:
            best[candidate.turn_offset] = candidate
    return tuple(best[offset] for offset in sorted(best))


def compile_demo3_record(
    record: Mapping[str, Any], *, split: str = "train"
) -> CompiledDemo3Record:
    """Compile one authored translation episode into a stream and candidates."""

    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("Demo 3 record requires a non-empty string id")
    if split not in {"train", "dev"}:
        raise ValueError("Demo 3 split must be 'train' or 'dev'")
    source_metadata = _validated_provenance_metadata(record, context=record_id)
    plan = plan_demo3_script(record)

    instruction = record.get("instruction")
    if not isinstance(instruction, Mapping):
        raise ValueError(f"{record_id}: instruction must be an object")
    instruction_text = _require_str(
        instruction.get("text"), f"{record_id}: instruction", "text"
    )
    gold_ack = _require_str(
        instruction.get("gold_ack"), f"{record_id}: instruction", "gold_ack"
    )
    instruction_pause = instruction.get("pause_after", "short")
    if instruction_pause not in PAUSE_MS:
        raise ValueError(f"{record_id}: instruction has invalid pause_after")
    if len(instruction_text) < MIN_INSTRUCTION_CHARS:
        raise ValueError(
            f"{record_id}: instruction text must be at least "
            f"{MIN_INSTRUCTION_CHARS} characters so it has a 'still typing' neighbour"
        )
    if CLAUSE_MARKER in instruction_text or CLAUSE_MARKER in gold_ack:
        raise ValueError(f"{record_id}: instruction text must not carry a clause marker")

    open_shape = opening_shape(record_id)
    close_shape = closing_shape(record_id)

    turns: list[CompletedTurn] = []
    candidates: list[Demo3Candidate] = []
    elapsed_ms = _stable_int(f"{record_id}:initial-time", 520, 780)

    def emit(content: str, state: UserState, action: Action) -> int:
        offset = len(turns)
        turns.append(_turn(_user_event(offset + 1, content, state, elapsed_ms), action))
        return offset

    # -- opening skeleton -------------------------------------------------
    if open_shape != "immediate":
        offset = emit("", UserState.IDLE, _idle())
        candidates.append(
            _candidate(record_id, offset, "empty-initial", key="initial", empty_kind="initial")
        )
        if open_shape in {"idle-idle", "idle-long-idle"}:
            gap = "long" if open_shape == "idle-long-idle" else "none"
            elapsed_ms += _pause_delta(record_id, "open-hold", gap)
            offset = emit("", UserState.IDLE, _idle())
            candidates.append(
                _candidate(
                    record_id,
                    offset,
                    "empty-unchanged",
                    key="open-unchanged",
                    empty_kind="unchanged",
                )
            )

    # -- the standing translation instruction and its one-shot ack --------
    instruction_offsets: list[int] = []
    cursor = 0
    for tick_index, size in enumerate(_chunk_sizes(len(instruction_text), f"{record_id}:instruction")):
        cursor += size
        elapsed_ms += _typing_delta(record_id, "instruction", tick_index)
        complete = cursor == len(instruction_text)
        offset = len(turns)
        action = _respond(offset + 1, gold_ack) if complete else _idle()
        instruction_offsets.append(emit(instruction_text[:cursor], UserState.ACTIVE, action))
    if len(instruction_offsets) < 2:
        raise ValueError(f"{record_id}: instruction is too short for a before-neighbour")
    candidates.append(
        _candidate(
            record_id,
            instruction_offsets[-2],
            "instruction-before",
            key="instruction",
            pause_after=instruction_pause,
        )
    )
    ack_offset = instruction_offsets[-1]
    candidates.append(
        _candidate(
            record_id,
            ack_offset,
            "instruction-ack",
            key="instruction",
            pause_after=instruction_pause,
        )
    )

    # The passage starts on a new line below the instruction.  That newline is
    # part of the first typed chunk, not a free extra character: counting it
    # separately would put an 8-character delta on one tick and break the
    # 4-7 char/tick human cadence.
    base_text = instruction_text
    passage_lead = "\n"
    if instruction_pause != "none":
        elapsed_ms += _pause_delta(record_id, "instruction-hold", instruction_pause)
        emit(instruction_text, UserState.IDLE, _idle())

    # -- the English passage ----------------------------------------------
    passage = ""
    clause_offsets: dict[int, int] = {}
    clause_steps: dict[int, int] = {
        piece.clause_index: piece.step_index
        for piece in plan.pieces
        if piece.clause_index is not None
    }
    step_pause = {step.index: step.pause_after for step in plan.steps}
    pieces_by_step: dict[int, list[ScriptPiece]] = {}
    for piece in plan.pieces:
        pieces_by_step.setdefault(piece.step_index, []).append(piece)

    for step in plan.steps:
        step_offsets: list[int] = []
        if step.kind == "backtrack":
            passage = passage[: len(passage) - step.delete]
            elapsed_ms += _typing_delta(record_id, f"backtrack:{step.index}", 0)
            offset = emit(base_text + passage, UserState.ACTIVE, _idle())
            step_offsets.append(offset)
            candidates.append(
                _candidate(
                    record_id,
                    offset,
                    "backtrack-idle",
                    key=str(step.index),
                    step_index=step.index,
                    trap="backtrack",
                    pause_after=step.pause_after,
                )
            )
        else:
            for piece in pieces_by_step.get(step.index, ()):
                piece_key = f"piece:{piece.step_index}:{piece.piece_index}"
                unit = passage_lead + piece.text
                passage_lead = ""
                sizes = _chunk_sizes(len(unit), f"{record_id}:{piece_key}")
                cursor = 0
                offsets: list[int] = []
                for tick_index, size in enumerate(sizes):
                    cursor += size
                    elapsed_ms += _typing_delta(record_id, piece_key, tick_index)
                    visible = passage + unit[:cursor]
                    complete = cursor == len(unit)
                    offset = len(turns)
                    if complete and piece.clause_index is not None:
                        action = _respond(
                            offset + 1, plan.clause_spans[piece.clause_index].reference
                        )
                    else:
                        action = _idle()
                    offsets.append(emit(base_text + visible, UserState.ACTIVE, action))
                passage += unit
                step_offsets.extend(offsets)

                if piece.clause_index is None:
                    continue
                clause_offsets[piece.clause_index] = offsets[-1]
                candidates.append(
                    _candidate(
                        record_id,
                        offsets[-1],
                        "clause-positive",
                        key=str(piece.clause_index),
                        clause_index=piece.clause_index,
                        step_index=step.index,
                        pause_after=step.pause_after,
                    )
                )
                # The "not ready" neighbour is the tick immediately before the
                # commit: same clause, one chunk short of complete.
                if len(offsets) >= 2 and turns[offsets[-2]].action.kind is ActionKind.IDLE:
                    candidates.append(
                        _candidate(
                            record_id,
                            offsets[-2],
                            "clause-before",
                            key=str(piece.clause_index),
                            clause_index=piece.clause_index,
                            step_index=step.index,
                            pause_after=step.pause_after,
                        )
                    )

            if step.trap == "half_clause":
                candidates.append(
                    _candidate(
                        record_id,
                        step_offsets[-1],
                        "partial-idle",
                        key=str(step.index),
                        step_index=step.index,
                        trap="half_clause",
                        pause_after=step.pause_after,
                    )
                )
            elif step.trap == "prequoted_chinese":
                candidates.append(
                    _candidate(
                        record_id,
                        step_offsets[-1],
                        "prequoted-idle",
                        key=str(step.index),
                        step_index=step.index,
                        trap="prequoted_chinese",
                        pause_after=step.pause_after,
                    )
                )

        if step.pause_after != "none":
            elapsed_ms += _pause_delta(record_id, f"step:{step.index}", step.pause_after)
            emit(base_text + passage, UserState.IDLE, _idle())

    if passage != "\n" + plan.final_passage:
        raise AssertionError(
            f"{record_id}: compiled passage diverged from the planned passage"
        )

    # -- closing skeleton --------------------------------------------------
    if close_shape in {"hold", "hold-clear"}:
        elapsed_ms += _pause_delta(record_id, "close-hold", "medium")
        emit(base_text + passage, UserState.IDLE, _idle())
    if close_shape != "hold":
        elapsed_ms += _pause_delta(record_id, "close-clear", "short")
        offset = emit("", UserState.IDLE, _idle())
        candidates.append(
            _candidate(record_id, offset, "empty-cleared", key="cleared", empty_kind="cleared")
        )
        if close_shape == "clear-hold":
            elapsed_ms += _pause_delta(record_id, "close-clear-hold", "medium")
            offset = emit("", UserState.IDLE, _idle())
            candidates.append(
                _candidate(
                    record_id,
                    offset,
                    "empty-unchanged",
                    key="close-unchanged",
                    empty_kind="unchanged",
                )
            )

    # -- after-neighbours: the first later idle tick that still shows it ----
    def first_visible_idle_after(offset: int) -> int | None:
        for later in range(offset + 1, len(turns)):
            turn = turns[later]
            if turn.action.kind is ActionKind.IDLE and turn.event.content:
                return later
        return None

    after_offset = first_visible_idle_after(ack_offset)
    if after_offset is not None:
        candidates.append(
            _candidate(
                record_id,
                after_offset,
                "instruction-after",
                key="instruction",
                pause_after=instruction_pause,
            )
        )
    for clause_index, offset in sorted(clause_offsets.items()):
        later = first_visible_idle_after(offset)
        if later is None:
            continue
        candidates.append(
            _candidate(
                record_id,
                later,
                "clause-after",
                key=str(clause_index),
                clause_index=clause_index,
                step_index=clause_steps[clause_index],
                pause_after=step_pause[clause_steps[clause_index]],
            )
        )

    return CompiledDemo3Record(
        record_id=record_id,
        split=split,
        turns=tuple(turns),
        candidates=_dedupe_candidates(candidates),
        plan=plan,
        opening_shape=open_shape,
        closing_shape=close_shape,
        source_metadata=source_metadata,
    )


def compile_demo3_records(
    records: Iterable[Mapping[str, Any]], *, dev_fraction: float = 0.1
) -> tuple[CompiledDemo3Record, ...]:
    items = list(records)
    split_by_id = assign_demo3_splits(items, dev_fraction=dev_fraction)
    return tuple(
        compile_demo3_record(record, split=split_by_id[str(record["id"])])
        for record in sorted(items, key=lambda item: str(item["id"]))
    )


def demo3_records_from_batches(
    batches: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Flatten validated authored batches while preserving author provenance."""

    records: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches):
        author = batch.get("author")
        review = batch.get("reference_review")
        raw_records = batch.get("records")
        if not isinstance(author, Mapping) or not isinstance(raw_records, list):
            raise ValueError(
                f"batch {batch_index}: author object and records array required"
            )
        if not isinstance(review, Mapping):
            raise ValueError(
                f"batch {batch_index}: reference_review object required "
                "(Chinese provenance is not optional)"
            )
        for record_index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, Mapping):
                raise ValueError(
                    f"batch {batch_index}: record {record_index} must be an object"
                )
            record = dict(raw_record)
            record["author_slot"] = str(author.get("slot"))
            record["author_model"] = str(author.get("model"))
            record["author_tranche"] = str(author.get("tranche"))
            record["reference_review_method"] = str(review.get("method"))
            record["reference_review_reviewer"] = str(review.get("reviewer"))
            record["reference_review_date"] = str(review.get("reviewed_at"))
            _validated_provenance_metadata(
                record, context=f"batch {batch_index}: record {record_index}"
            )
            records.append(record)
    return tuple(records)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def _all_candidates(records: Iterable[CompiledDemo3Record]) -> tuple[Demo3Candidate, ...]:
    return tuple(candidate for record in records for candidate in record.candidates)


def _apportion(total: int, weights: Sequence[tuple[str, int]]) -> dict[str, int]:
    """Largest-remainder apportionment; deterministic ties broken by name."""

    weight_sum = sum(weight for _, weight in weights)
    base: dict[str, int] = {}
    remainders: list[tuple[int, str, str]] = []
    assigned = 0
    for role, weight in weights:
        exact = total * weight
        share = exact // weight_sum
        base[role] = share
        assigned += share
        remainders.append((exact % weight_sum, role, role))
    for _, role, _ in sorted(remainders, key=lambda item: (-item[0], item[1]))[
        : total - assigned
    ]:
        base[role] += 1
    return base


def select_demo3_candidates(
    records: Iterable[CompiledDemo3Record],
    *,
    targets: Demo3Targets = Demo3Targets(),
) -> tuple[Demo3Candidate, ...]:
    """Select the exact production card mix or fail with a count diagnosis."""

    record_items = tuple(records)
    candidates = _all_candidates(record_items)
    by_role: dict[str, list[Demo3Candidate]] = {role: [] for role in DEMO3_ROLES}
    for candidate in candidates:
        by_role[candidate.role].append(candidate)

    if len(record_items) != targets.episodes:
        raise ValueError(
            f"Demo 3 source requires exactly {targets.episodes} episodes; "
            f"found {len(record_items)}"
        )
    positives = by_role["clause-positive"]
    if len(positives) != targets.clauses:
        raise ValueError(
            f"Demo 3 source requires exactly {targets.clauses} clause commits; "
            f"found {len(positives)}"
        )
    acks = by_role["instruction-ack"]
    if len(acks) != targets.episodes:
        raise ValueError(
            f"Demo 3 requires one acknowledgment per episode ({targets.episodes}); "
            f"found {len(acks)}"
        )

    selected: list[Demo3Candidate] = [*positives, *acks]

    floors = {
        "backtrack-idle": targets.min_backtrack_idles,
        "prequoted-idle": targets.min_prequoted_idles,
        "partial-idle": targets.min_partial_idles,
    }
    for role in TRAP_ROLES:
        pool = by_role[role]
        if len(pool) < floors[role]:
            raise ValueError(
                f"Demo 3 requires at least {floors[role]} {role} cards; found {len(pool)}"
            )
        selected.extend(pool)

    for empty_kind in EMPTY_KINDS:
        pool = [
            candidate for candidate in candidates if candidate.empty_kind == empty_kind
        ]
        if len(pool) < targets.empty_per_kind:
            raise ValueError(
                f"Demo 3 needs {targets.empty_per_kind} {empty_kind} empty cards; "
                f"found {len(pool)}"
            )
        selected.extend(
            sorted(pool, key=lambda item: _stable_rank(f"empty:{item.candidate_id}"))[
                : targets.empty_per_kind
            ]
        )

    remaining = targets.cards - len(selected)
    if remaining < 0:
        raise ValueError(
            f"Demo 3 mandatory cards ({len(selected)}) already exceed the "
            f"{targets.cards}-card target; raise --cards or reduce source"
        )
    pools = {
        role: sorted(
            by_role[role], key=lambda item: _stable_rank(f"{item.role}:{item.candidate_id}")
        )
        for role in NEIGHBOR_ROLES
    }
    available = sum(len(pool) for pool in pools.values())
    if available < remaining:
        raise ValueError(
            f"Demo 3 needs {remaining} neighbour cards to reach {targets.cards}; "
            f"only {available} are available"
        )
    allocation = _apportion(remaining, targets.neighbor_weights)
    # Spill any allocation a role cannot cover onto the roles that can, in a
    # fixed order, until the budget is met exactly.
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
            f"Demo 3 could not apportion {remaining} neighbour cards across "
            f"{sorted(NEIGHBOR_ROLES)}"
        )
    for role in NEIGHBOR_ROLES:
        selected.extend(pools[role][: allocation[role]])

    if len(selected) != targets.cards:
        raise AssertionError(
            f"internal Demo 3 selector error: selected {len(selected)} cards, "
            f"expected {targets.cards}"
        )
    unique = {candidate.candidate_id for candidate in selected}
    if len(unique) != len(selected):
        raise AssertionError("internal Demo 3 selector error: duplicate candidate ids")
    return tuple(sorted(selected, key=lambda item: item.candidate_id))


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_demo3_card(
    record: CompiledDemo3Record, candidate: Demo3Candidate
) -> dict[str, Any]:
    """Render one card, refusing anything the serving parser cannot read."""

    if candidate.record_id != record.record_id:
        raise ValueError("candidate does not belong to the supplied Demo 3 record")
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
        # Non-ASCII is the whole point of Demo 3: the Chinese that goes in must
        # be the Chinese that comes back out, byte for byte, through the same
        # parser the runtime uses.
        if parsed.message != current_turn.action.message:
            raise ValueError(
                f"{candidate.candidate_id}: respond message did not survive the "
                "compile -> parse round trip byte-exactly"
            )
        if parsed.target != current_turn.event.index:
            raise ValueError(
                f"{candidate.candidate_id}: respond target is not the current event"
            )
    expected_class = (
        "respond" if current_turn.action.kind is ActionKind.RESPOND else "idle"
    )
    metadata = dict(record.source_metadata)
    clause = (
        record.plan.clause_spans[candidate.clause_index]
        if candidate.clause_index is not None
        else None
    )
    obligation = (
        "translate-active"
        if candidate.role not in {"instruction-before"}
        else "none"
    )
    clause_state = (
        "complete"
        if candidate.role == "clause-positive"
        else "partial"
        if candidate.role in {"clause-before", "partial-idle"}
        else None
    )
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
        "reference_review_method": metadata.get("reference_review_method"),
        "reference_review_reviewer": metadata.get("reference_review_reviewer"),
        "clause_index": candidate.clause_index,
        "clause_english": clause.english if clause else None,
        "clause_reference": clause.reference if clause else None,
        "clause_end_offset": clause.end_offset if clause else None,
        "step_index": candidate.step_index,
        "trap": candidate.trap,
        "pause_after": candidate.pause_after,
        "empty_kind": candidate.empty_kind,
        "opening_shape": record.opening_shape,
        "closing_shape": record.closing_shape,
        "obligation": obligation,
        "clause_state": clause_state,
    }


def demo3_coverage_report(
    records: Iterable[CompiledDemo3Record],
    selected: Iterable[Demo3Candidate],
) -> dict[str, Any]:
    record_items = tuple(records)
    selected_items = tuple(selected)
    record_by_id = {record.record_id: record for record in record_items}
    role_counts = Counter(candidate.role for candidate in selected_items)
    empty_counts = Counter(
        candidate.empty_kind for candidate in selected_items if candidate.empty_kind
    )
    trap_counts = Counter(
        candidate.trap for candidate in selected_items if candidate.trap
    )
    split_counts = Counter(
        record_by_id[candidate.record_id].split for candidate in selected_items
    )
    source_split_counts = Counter(record.split for record in record_items)
    selected_source_counts: dict[str, Counter[str]] = {
        field: Counter()
        for field in (
            "persona",
            "domain",
            "register",
            "author_slot",
            "reference_review_method",
        )
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
        "source_clauses": sum(len(record.plan.clause_spans) for record in record_items),
        "source_turns": sum(len(record.turns) for record in record_items),
        "opening_shapes": dict(
            sorted(Counter(record.opening_shape for record in record_items).items())
        ),
        "closing_shapes": dict(
            sorted(Counter(record.closing_shape for record in record_items).items())
        ),
        "selected_cards": len(selected_items),
        "selected_roles": dict(sorted(role_counts.items())),
        "selected_empty_kinds": dict(sorted(empty_counts.items())),
        "selected_traps": dict(sorted(trap_counts.items())),
        "selected_splits": dict(sorted(split_counts.items())),
        "selected_source_distribution": {
            field: dict(sorted(counts.items()))
            for field, counts in selected_source_counts.items()
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


def compile_demo3_dataset(
    records: Iterable[Mapping[str, Any]],
    *,
    targets: Demo3Targets = Demo3Targets(),
    dev_fraction: float = 0.1,
) -> Demo3Build:
    compiled = compile_demo3_records(records, dev_fraction=dev_fraction)
    selected = select_demo3_candidates(compiled, targets=targets)
    record_by_id = {record.record_id: record for record in compiled}
    rows = tuple(
        render_demo3_card(record_by_id[candidate.record_id], candidate)
        for candidate in selected
    )
    return Demo3Build(
        records=compiled,
        selected=selected,
        rows=rows,
        coverage=demo3_coverage_report(compiled, selected),
    )
