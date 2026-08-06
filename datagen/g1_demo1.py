"""Deterministic compiler for agent-authored Demo 1 situations.

Authored records describe semantic segments.  This module turns those segments
into the live app's full-textbox event stream, keeps every ungraded typing tick
in history, and selects a fixed-size production slice from them.

Two selection profiles ship here.  ``DEMO1_FULL_TARGETS`` reproduces the
original 5,400-card build byte for byte.  ``DEMO1_SCALED_TARGETS`` is the
1,800-card build used after the project scale was cut 3x; it keeps the same
role proportions but subsamples deterministically and with stratified spread.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from app.domain import Action, ActionKind, CompletedTurn, EventSource, StreamEvent, UserState
from app.stream import compile_stream, g1_action_completion, parse_g1_action

# The authored validator owns the canonical bucket definitions.  Importing them
# keeps the selection strata and the authored distribution gates in lock-step
# instead of quietly drifting apart behind two copies of the same thresholds.
from datagen.g1_authored import (
    _event_count_bucket,
    _length_bucket,
    _trigger_position_bucket,
)


G1_SCHEMA_VERSION = "g1"
DEMO1_PROVENANCE_FIELDS = (
    "persona",
    "domain",
    "register",
    "author_slot",
    "author_model",
    "author_tranche",
)
# Derived, record-level buckets appended to each compiled record's metadata.
# They are what the subsampler stratifies on and what the selection-level
# distribution gate reports, so a smaller build cannot drift into one author,
# one length, or one trigger position.
DEMO1_DERIVED_FIELDS = ("length_bucket", "event_count_bucket", "trigger_position")
DEMO1_STRATUM_FIELDS = ("author_slot", *DEMO1_DERIVED_FIELDS)
DEMO1_DISTRIBUTION_FIELDS = (
    "author_slot",
    "persona",
    "domain",
    "register",
    *DEMO1_DERIVED_FIELDS,
)
DEMO1_ROLES = {
    "address-before",
    "address-positive",
    "address-after",
    "narration-hard-idle",
    "narration-ballast",
    "empty-initial",
    "empty-unchanged",
    "empty-cleared",
}
EMPTY_KINDS = ("initial", "unchanged", "cleared")
PAUSE_MS = {
    "none": (540, 820),
    "short": (1_000, 3_000),
    "medium": (4_000, 9_000),
    "long": (10_000, 30_000),
}


SELECTION_STRATEGIES = ("hash", "stratified")


@dataclass(frozen=True, slots=True)
class Demo1Targets:
    """Source requirements and selected-card counts for a Demo 1 build.

    ``addresses`` and ``narrations`` are *source* requirements: the authored
    corpus must contain exactly that many.  ``address_sites``, ``hard_idles``
    and ``empty_per_kind`` are *selection* counts, so shrinking a build never
    silently accepts a shrunken source.

    ``selected_addresses`` and ``hard_idles`` default to "take every one that
    exists", which is what the original 5,400-card build did.
    """

    addresses: int = 700
    narrations: int = 4_000
    cards: int = 5_400
    empty_per_kind: int = 100
    selected_addresses: int | None = None
    hard_idles: int | None = None
    strategy: str = "hash"

    def __post_init__(self) -> None:
        if self.strategy not in SELECTION_STRATEGIES:
            raise ValueError(
                f"unknown Demo 1 selection strategy {self.strategy!r}; "
                f"expected one of {list(SELECTION_STRATEGIES)}"
            )
        for field, value in (
            ("addresses", self.addresses),
            ("narrations", self.narrations),
            ("cards", self.cards),
            ("empty_per_kind", self.empty_per_kind),
        ):
            if value <= 0:
                raise ValueError(f"Demo 1 target {field} must be positive")
        if self.selected_addresses is not None:
            if not 0 < self.selected_addresses <= self.addresses:
                raise ValueError(
                    "Demo 1 selected_addresses must be between 1 and the source address count"
                )
        if self.hard_idles is not None and self.hard_idles < 0:
            raise ValueError("Demo 1 hard_idles must not be negative")
        fixed = 3 * self.address_sites + len(EMPTY_KINDS) * self.empty_per_kind
        if fixed > self.cards:
            raise ValueError(
                f"Demo 1 address and empty cards ({fixed}) exceed the card target "
                f"of {self.cards}"
            )
        if self.hard_idles is not None and self.hard_idles > self.cards - fixed:
            raise ValueError(
                f"Demo 1 hard_idles ({self.hard_idles}) exceed the "
                f"{self.cards - fixed} available narration slots"
            )

    @property
    def address_sites(self) -> int:
        """Number of address *sites*; each contributes before/positive/after."""

        return self.addresses if self.selected_addresses is None else self.selected_addresses

    @property
    def narration_cards(self) -> int:
        return self.cards - 3 * self.address_sites - len(EMPTY_KINDS) * self.empty_per_kind


# The original production build: every address, every hard idle, plain hash
# subsampling of the empty and ballast pools.
DEMO1_FULL_TARGETS = Demo1Targets()

# The 3x-scaled build.  Role shares are held to the 5,400 mix except that the
# rare-but-critical empty-silence classes keep a 50-card floor per kind instead
# of their proportional 33, per the spec's "fix absolute counts of rare actions,
# not percentages" balance principle.  Hard idles take exactly one third of the
# 615 originals (205).  Ballast, the deliberately elastic filler class, absorbs
# the remainder:
#   3 x 233 addresses (699) + 3 x 50 empties (150) + 205 hard idle + 746 ballast
#   = 1,800 cards.
DEMO1_SCALED_TARGETS = Demo1Targets(
    cards=1_800,
    empty_per_kind=50,
    selected_addresses=233,
    hard_idles=205,
    strategy="stratified",
)

DEMO1_TARGET_PROFILES: dict[str, Demo1Targets] = {
    "full-5400": DEMO1_FULL_TARGETS,
    "scaled-1800": DEMO1_SCALED_TARGETS,
}
DEFAULT_DEMO1_PROFILE = "scaled-1800"


@dataclass(frozen=True, slots=True)
class Demo1Candidate:
    """A gradable turn in a compiled record; padding turns are not candidates."""

    candidate_id: str
    record_id: str
    turn_offset: int
    role: str
    segment_index: int | None = None
    traps: tuple[str, ...] = ()
    pause_after: str | None = None
    empty_kind: str | None = None

    def __post_init__(self) -> None:
        if self.role not in DEMO1_ROLES:
            raise ValueError(f"unknown Demo 1 candidate role: {self.role!r}")


@dataclass(frozen=True, slots=True)
class CompiledDemo1Record:
    record_id: str
    split: str
    turns: tuple[CompletedTurn, ...]
    candidates: tuple[Demo1Candidate, ...]
    source_metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Demo1Build:
    records: tuple[CompiledDemo1Record, ...]
    selected: tuple[Demo1Candidate, ...]
    rows: tuple[dict[str, Any], ...]
    coverage: dict[str, Any]


def _digest_bytes(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _stable_rank(value: str) -> tuple[bytes, str]:
    return (_digest_bytes(value), value)


def _stable_int(key: str, low: int, high: int) -> int:
    if low > high:
        raise ValueError("low must not exceed high")
    span = high - low + 1
    return low + int.from_bytes(_digest_bytes(key)[:8], "big") % span


def _validated_provenance_metadata(
    record: Mapping[str, Any], *, context: str
) -> tuple[tuple[str, str], ...]:
    """Return normalized metadata only after every provenance field is present."""

    metadata: list[tuple[str, str]] = []
    for field in DEMO1_PROVENANCE_FIELDS:
        if field not in record:
            raise ValueError(f"{context}: missing required provenance field {field!r}")
        value = record[field]
        if value is None or not str(value).strip():
            raise ValueError(f"{context}: provenance field {field!r} must be non-blank")
        metadata.append((field, str(value)))
    return tuple(sorted(metadata))


def _derived_bucket_metadata(
    segments: Sequence[Any],
) -> tuple[tuple[str, str], ...]:
    """Derive the record-level strata the subsampler and gates rely on."""

    mappings = [segment for segment in segments if isinstance(segment, Mapping)]
    characters = sum(len(str(segment.get("text", ""))) for segment in mappings)
    address_positions = [
        index
        for index, segment in enumerate(segments)
        if isinstance(segment, Mapping) and segment.get("kind") == "address"
    ]
    if len(address_positions) == 1 and segments:
        trigger = _trigger_position_bucket(address_positions[0], len(segments))
    else:
        # Authored validation already rejects anything but exactly one address.
        # A build that somehow gets here still needs a total stratum label.
        trigger = "unknown"
    return (
        ("length_bucket", _length_bucket(characters)),
        ("event_count_bucket", _event_count_bucket(len(segments))),
        ("trigger_position", trigger),
    )


def _chunk_sizes(length: int, key: str) -> tuple[int, ...]:
    """Partition text into deterministic 4-7 character typing deltas.

    Authored production segments are comfortably longer than four characters.
    A shorter source is still represented faithfully as one unavoidable short
    tick rather than padded with invented text.
    """

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
            # This is unreachable for integers >= 4, but keeps failure explicit
            # if the cadence range changes later.
            raise ValueError(f"cannot partition {length} characters into 4-7 character ticks")
        choice = _stable_int(f"{key}:chunk:{step}", 0, len(feasible) - 1)
        size = feasible[choice]
        sizes.append(size)
        remaining -= size
        step += 1
    return tuple(sizes)


def _idle() -> Action:
    return Action(ActionKind.IDLE)


def _respond(target: int, message: str) -> Action:
    return Action(ActionKind.RESPOND, target=target, message=message)


def _user_event(index: int, content: str, state: UserState, elapsed_ms: int) -> StreamEvent:
    return StreamEvent(index, EventSource.USER, content, state, elapsed_ms)


def _turn(event: StreamEvent, action: Action) -> CompletedTurn:
    return CompletedTurn(event=event, action=action)


def assign_demo1_splits(
    records: Iterable[Mapping[str, Any]], *, dev_fraction: float = 0.1
) -> dict[str, str]:
    """Assign whole records to a deterministic, exact-count train/dev split."""

    items = list(records)
    if not 0 <= dev_fraction <= 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    record_ids = [record.get("id") for record in items]
    if any(not isinstance(record_id, str) or not record_id for record_id in record_ids):
        raise ValueError("every Demo 1 record requires a non-empty string id")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Demo 1 record ids must be unique")

    dev_count = round(len(record_ids) * dev_fraction)
    ranked = sorted(record_ids, key=lambda record_id: _stable_rank(f"split:{record_id}"))
    dev_ids = set(ranked[:dev_count])
    return {record_id: "dev" if record_id in dev_ids else "train" for record_id in record_ids}


def _pause_delta(record_id: str, segment_index: int, pause_after: str) -> int:
    if pause_after not in PAUSE_MS:
        raise ValueError(
            f"{record_id}: segment {segment_index}: invalid pause_after {pause_after!r}"
        )
    low, high = PAUSE_MS[pause_after]
    return _stable_int(f"{record_id}:segment:{segment_index}:pause", low, high)


def _typing_delta(record_id: str, segment_index: int, tick_index: int) -> int:
    # The app normally ticks at 650ms. Small deterministic jitter prevents the
    # clock from becoming a decorative constant in training histories.
    return _stable_int(
        f"{record_id}:segment:{segment_index}:tick:{tick_index}:time", 520, 820
    )


def _candidate(
    record_id: str,
    turn_offset: int,
    role: str,
    *,
    segment_index: int | None = None,
    traps: Sequence[str] = (),
    pause_after: str | None = None,
    empty_kind: str | None = None,
) -> Demo1Candidate:
    suffix = str(segment_index) if segment_index is not None else str(empty_kind)
    return Demo1Candidate(
        candidate_id=f"{record_id}:{role}:{suffix}",
        record_id=record_id,
        turn_offset=turn_offset,
        role=role,
        segment_index=segment_index,
        traps=tuple(sorted(trap for trap in traps if trap != "none")),
        pause_after=pause_after,
        empty_kind=empty_kind,
    )


def compile_demo1_record(
    record: Mapping[str, Any], *, split: str = "train"
) -> CompiledDemo1Record:
    """Compile one authored mini-scene into an event stream and candidate refs."""

    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("Demo 1 record requires a non-empty string id")
    source_metadata = _validated_provenance_metadata(record, context=record_id)
    if split not in {"train", "dev"}:
        raise ValueError("Demo 1 split must be 'train' or 'dev'")
    segments = record.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{record_id}: segments must be a non-empty array")
    source_metadata = tuple(
        sorted(source_metadata + _derived_bucket_metadata(segments))
    )

    turns: list[CompletedTurn] = []
    candidates: list[Demo1Candidate] = []
    current_text = ""
    elapsed_ms = _stable_int(f"{record_id}:initial-time", 520, 780)

    # Silence is part of the same chronological episode, not a separate canned
    # fixture. This lets selected empty cards inherit realistic record identity.
    turns.append(_turn(_user_event(1, "", UserState.IDLE, elapsed_ms), _idle()))
    candidates.append(_candidate(record_id, 0, "empty-initial", empty_kind="initial"))
    elapsed_ms += _stable_int(f"{record_id}:unchanged-gap", 540, 850)
    turns.append(_turn(_user_event(2, "", UserState.IDLE, elapsed_ms), _idle()))
    candidates.append(
        _candidate(record_id, 1, "empty-unchanged", empty_kind="unchanged")
    )

    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ValueError(f"{record_id}: segment {segment_index} must be an object")
        kind = segment.get("kind")
        text = segment.get("text")
        pause_after = segment.get("pause_after", "none")
        if kind not in {"narration", "address"}:
            raise ValueError(f"{record_id}: segment {segment_index}: invalid kind {kind!r}")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{record_id}: segment {segment_index}: text must be non-empty")
        if pause_after not in PAUSE_MS:
            raise ValueError(
                f"{record_id}: segment {segment_index}: invalid pause_after {pause_after!r}"
            )

        addition = ("\n" if current_text else "") + text
        sizes = _chunk_sizes(len(addition), f"{record_id}:segment:{segment_index}")
        cursor = 0
        segment_turn_offsets: list[int] = []
        for tick_index, size in enumerate(sizes):
            cursor += size
            elapsed_ms += _typing_delta(record_id, segment_index, tick_index)
            snapshot = current_text + addition[:cursor]
            event = _user_event(len(turns) + 1, snapshot, UserState.ACTIVE, elapsed_ms)
            is_complete = cursor == len(addition)
            if kind == "address" and is_complete:
                reply = segment.get("gold_reply")
                if not isinstance(reply, str) or not reply.strip():
                    raise ValueError(
                        f"{record_id}: segment {segment_index}: address requires gold_reply"
                    )
                action = _respond(event.index, reply)
            else:
                action = _idle()
            segment_turn_offsets.append(len(turns))
            turns.append(_turn(event, action))

        current_text += addition
        completion_offset = segment_turn_offsets[-1]
        if kind == "address":
            if len(segment_turn_offsets) < 2:
                raise ValueError(
                    f"{record_id}: segment {segment_index}: address is too short for a before-neighbor"
                )
            candidates.append(
                _candidate(
                    record_id,
                    segment_turn_offsets[-2],
                    "address-before",
                    segment_index=segment_index,
                    pause_after=pause_after,
                )
            )
            candidates.append(
                _candidate(
                    record_id,
                    completion_offset,
                    "address-positive",
                    segment_index=segment_index,
                    pause_after=pause_after,
                )
            )
        else:
            raw_traps = segment.get("traps", [])
            if not isinstance(raw_traps, list) or any(
                not isinstance(trap, str) for trap in raw_traps
            ):
                raise ValueError(f"{record_id}: segment {segment_index}: traps must be strings")
            traps = tuple(trap for trap in raw_traps if trap != "none")
            # The authored contract makes trap labels the reason a completed
            # narration is a graded hard idle. Pause markers shape the visible
            # elapsed-time history, but do not silently relabel ordinary prose.
            hard_idle = bool(traps)
            candidates.append(
                _candidate(
                    record_id,
                    completion_offset,
                    "narration-hard-idle" if hard_idle else "narration-ballast",
                    segment_index=segment_index,
                    traps=traps,
                    pause_after=pause_after,
                )
            )

        # The unchanged pause tick remains in history. For an address this is
        # also the after-neighbor, and its history contains the prior respond.
        elapsed_ms += _pause_delta(record_id, segment_index, pause_after)
        pause_event = _user_event(
            len(turns) + 1, current_text, UserState.IDLE, elapsed_ms
        )
        pause_offset = len(turns)
        turns.append(_turn(pause_event, _idle()))
        if kind == "address":
            candidates.append(
                _candidate(
                    record_id,
                    pause_offset,
                    "address-after",
                    segment_index=segment_index,
                    pause_after=pause_after,
                )
            )

    # Clearing is a real empty snapshot. Its history retains the entire scene,
    # including any response action, so the model can tell cleared from initial.
    elapsed_ms += _stable_int(f"{record_id}:clear-gap", 560, 1_100)
    clear_offset = len(turns)
    turns.append(
        _turn(
            _user_event(len(turns) + 1, "", UserState.IDLE, elapsed_ms),
            _idle(),
        )
    )
    candidates.append(
        _candidate(record_id, clear_offset, "empty-cleared", empty_kind="cleared")
    )
    return CompiledDemo1Record(
        record_id=record_id,
        split=split,
        turns=tuple(turns),
        candidates=tuple(candidates),
        source_metadata=source_metadata,
    )


def compile_demo1_records(
    records: Iterable[Mapping[str, Any]], *, dev_fraction: float = 0.1
) -> tuple[CompiledDemo1Record, ...]:
    items = list(records)
    split_by_id = assign_demo1_splits(items, dev_fraction=dev_fraction)
    return tuple(
        compile_demo1_record(record, split=split_by_id[str(record["id"])])
        for record in sorted(items, key=lambda item: str(item["id"]))
    )


def demo1_records_from_batches(
    batches: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Flatten validated authored batches while preserving author provenance."""

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


def _all_candidates(
    records: Iterable[CompiledDemo1Record],
) -> tuple[Demo1Candidate, ...]:
    return tuple(candidate for record in records for candidate in record.candidates)


def _proportional_allocation(
    sizes: Mapping[str, int], count: int, *, salt: str
) -> dict[str, int]:
    """Split ``count`` across strata by largest remainder, exactly and stably."""

    total = sum(sizes.values())
    if count > total:
        raise ValueError("cannot allocate more items than exist")
    allocation = {name: (count * size) // total for name, size in sizes.items()}
    remainders = {name: (count * size) % total for name, size in sizes.items()}
    leftover = count - sum(allocation.values())
    order = sorted(
        sizes,
        key=lambda name: (
            -remainders[name],
            _stable_rank(f"{salt}:stratum:{name}"),
        ),
    )
    for name in order:
        if leftover <= 0:
            break
        if allocation[name] < sizes[name]:
            allocation[name] += 1
            leftover -= 1
    if leftover:  # pragma: no cover - guarded by the count <= total check above
        raise AssertionError("proportional allocation could not place every card")
    return allocation


def _record_spread_order(
    items: Sequence[Demo1Candidate], *, salt: str
) -> list[Demo1Candidate]:
    """Order candidates round-robin over their source records.

    Taking a prefix of this order therefore touches as many distinct authored
    records as possible before it takes a second card from any one of them.
    """

    by_record: dict[str, list[Demo1Candidate]] = {}
    for item in items:
        by_record.setdefault(item.record_id, []).append(item)
    keyed: list[tuple[int, tuple[bytes, str], Demo1Candidate]] = []
    for record_id in sorted(by_record):
        ordered = sorted(
            by_record[record_id],
            key=lambda item: _stable_rank(f"{salt}:{item.candidate_id}"),
        )
        for round_index, item in enumerate(ordered):
            keyed.append(
                (round_index, _stable_rank(f"{salt}:{item.candidate_id}"), item)
            )
    keyed.sort(key=lambda entry: (entry[0], entry[1]))
    return [entry[2] for entry in keyed]


def _subsample(
    items: Sequence[Demo1Candidate],
    count: int,
    *,
    salt: str,
    strategy: str,
    stratum_of: Callable[[Demo1Candidate], str],
    label: str,
) -> list[Demo1Candidate]:
    """Deterministically take ``count`` candidates from ``items``.

    ``hash`` reproduces the original 5,400-card build exactly.  ``stratified``
    holds every stratum's share of the draw to its share of the pool and then
    spreads the draw across as many source records as possible.  Neither uses
    an RNG: both are pure functions of the candidate ids.
    """

    if count < 0:
        raise ValueError(f"Demo 1 {label} target must not be negative")
    if count > len(items):
        raise ValueError(
            f"Demo 1 needs {count} {label} cards; found {len(items)}"
        )
    if count == len(items):
        return list(items)
    if strategy == "hash":
        return sorted(
            items, key=lambda item: _stable_rank(f"{salt}:{item.candidate_id}")
        )[:count]
    if strategy != "stratified":  # pragma: no cover - Demo1Targets validates this
        raise ValueError(f"unknown Demo 1 selection strategy {strategy!r}")

    groups: dict[str, list[Demo1Candidate]] = {}
    for item in items:
        groups.setdefault(stratum_of(item), []).append(item)
    allocation = _proportional_allocation(
        {name: len(group) for name, group in groups.items()}, count, salt=salt
    )
    taken: list[Demo1Candidate] = []
    for name in sorted(groups):
        share = allocation[name]
        if share:
            taken.extend(_record_spread_order(groups[name], salt=salt)[:share])
    if len(taken) != count:  # pragma: no cover - allocation is exact
        raise AssertionError(
            f"internal Demo 1 subsample error: took {len(taken)} {label} cards, "
            f"expected {count}"
        )
    return taken


def select_demo1_candidates(
    records: Iterable[CompiledDemo1Record],
    *,
    targets: Demo1Targets = DEMO1_FULL_TARGETS,
) -> tuple[Demo1Candidate, ...]:
    """Select the exact production card mix or fail with a count diagnosis."""

    record_items = tuple(records)
    candidates = _all_candidates(record_items)
    stratum_by_record = {
        record.record_id: "|".join(
            dict(record.source_metadata).get(field, "unknown")
            for field in DEMO1_STRATUM_FIELDS
        )
        for record in record_items
    }

    def stratum_of(candidate: Demo1Candidate) -> str:
        return stratum_by_record.get(candidate.record_id, "unknown")

    addresses = [candidate for candidate in candidates if candidate.role == "address-positive"]
    narrations = [candidate for candidate in candidates if candidate.role.startswith("narration-")]
    if len(addresses) != targets.addresses:
        raise ValueError(
            f"Demo 1 source requires exactly {targets.addresses} addresses; found {len(addresses)}"
        )
    if len(narrations) != targets.narrations:
        raise ValueError(
            f"Demo 1 source requires exactly {targets.narrations} narrations; found {len(narrations)}"
        )

    # An address is one coherent site: pick the sites once, then take the
    # before/positive/after triple for each.  A positive can never be selected
    # without the neighbours that give it its meaning.
    chosen_sites = {
        (candidate.record_id, candidate.segment_index)
        for candidate in _subsample(
            addresses,
            targets.address_sites,
            salt="address-site",
            strategy=targets.strategy,
            stratum_of=stratum_of,
            label="address site",
        )
    }
    selected: list[Demo1Candidate] = []
    for role in ("address-before", "address-positive", "address-after"):
        role_cards = [candidate for candidate in candidates if candidate.role == role]
        if len(role_cards) != targets.addresses:
            raise ValueError(
                f"Demo 1 requires {targets.addresses} {role} cards; found {len(role_cards)}"
            )
        neighbours = [
            candidate
            for candidate in role_cards
            if (candidate.record_id, candidate.segment_index) in chosen_sites
        ]
        if len(neighbours) != targets.address_sites:
            raise AssertionError(
                f"internal Demo 1 selector error: {len(neighbours)} {role} cards "
                f"for {targets.address_sites} selected address sites"
            )
        selected.extend(neighbours)

    for empty_kind in EMPTY_KINDS:
        available = [
            candidate for candidate in candidates if candidate.empty_kind == empty_kind
        ]
        if len(available) < targets.empty_per_kind:
            raise ValueError(
                f"Demo 1 needs {targets.empty_per_kind} {empty_kind} empty cards; "
                f"found {len(available)}"
            )
        selected.extend(
            _subsample(
                available,
                targets.empty_per_kind,
                salt="empty",
                strategy=targets.strategy,
                stratum_of=stratum_of,
                label=f"{empty_kind} empty",
            )
        )

    narration_slots = targets.cards - len(selected)
    hard_idles = [candidate for candidate in narrations if candidate.role == "narration-hard-idle"]
    ballast = [candidate for candidate in narrations if candidate.role == "narration-ballast"]
    if narration_slots < 0:
        raise ValueError(
            f"Demo 1 fixed address/empty cards exceed target of {targets.cards} cards"
        )
    hard_idle_target = (
        len(hard_idles) if targets.hard_idles is None else targets.hard_idles
    )
    if hard_idle_target > len(hard_idles):
        raise ValueError(
            f"Demo 1 needs {hard_idle_target} hard-idle narrations; found {len(hard_idles)}"
        )
    if hard_idle_target > narration_slots:
        raise ValueError(
            f"Demo 1 has {hard_idle_target} hard-idle narrations but only "
            f"{narration_slots} narration slots"
        )
    ballast_needed = narration_slots - hard_idle_target
    if len(ballast) < ballast_needed:
        raise ValueError(
            f"Demo 1 needs {ballast_needed} ordinary narration ballast cards; "
            f"found {len(ballast)}"
        )
    selected.extend(
        _subsample(
            hard_idles,
            hard_idle_target,
            salt="hard-idle",
            strategy=targets.strategy,
            stratum_of=stratum_of,
            label="hard-idle narration",
        )
    )
    selected.extend(
        _subsample(
            ballast,
            ballast_needed,
            salt="ballast",
            strategy=targets.strategy,
            stratum_of=stratum_of,
            label="ballast narration",
        )
    )
    if len(selected) != targets.cards:
        raise AssertionError(
            f"internal Demo 1 selector error: selected {len(selected)} cards, expected {targets.cards}"
        )
    return tuple(sorted(selected, key=lambda item: item.candidate_id))


def render_demo1_card(
    record: CompiledDemo1Record, candidate: Demo1Candidate
) -> dict[str, Any]:
    if candidate.record_id != record.record_id:
        raise ValueError("candidate does not belong to the supplied Demo 1 record")
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
    expected_class = (
        "respond" if current_turn.action.kind is ActionKind.RESPOND else "idle"
    )
    narration_idle_kind = None
    if candidate.role.startswith("narration-"):
        narration_idle_kind = (
            "hard-idle" if candidate.role == "narration-hard-idle" else "ballast"
        )
    source_metadata = dict(record.source_metadata)
    return {
        "schema_version": G1_SCHEMA_VERSION,
        "split": record.split,
        "episode": record.record_id,
        "demo": "demo-1",
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
        "source_persona": source_metadata.get("persona"),
        "source_domain": source_metadata.get("domain"),
        "source_register": source_metadata.get("register"),
        "source_author_slot": source_metadata.get("author_slot"),
        "source_author_model": source_metadata.get("author_model"),
        "source_author_tranche": source_metadata.get("author_tranche"),
        "segment_index": candidate.segment_index,
        "traps": list(candidate.traps),
        "pause_after": candidate.pause_after,
        "narration_idle_kind": narration_idle_kind,
        "empty_kind": candidate.empty_kind,
        "obligation": "none",
    }


def demo1_coverage_report(
    records: Iterable[CompiledDemo1Record],
    selected: Iterable[Demo1Candidate],
) -> dict[str, Any]:
    record_items = tuple(records)
    selected_items = tuple(selected)
    record_by_id = {record.record_id: record for record in record_items}
    role_counts = Counter(candidate.role for candidate in selected_items)
    empty_counts = Counter(
        candidate.empty_kind for candidate in selected_items if candidate.empty_kind
    )
    split_counts = Counter(record_by_id[candidate.record_id].split for candidate in selected_items)
    source_split_counts = Counter(record.split for record in record_items)
    selected_source_counts: dict[str, Counter[str]] = {
        field: Counter() for field in DEMO1_DISTRIBUTION_FIELDS
    }
    for candidate in selected_items:
        metadata = dict(record_by_id[candidate.record_id].source_metadata)
        for field, counts in selected_source_counts.items():
            if field in metadata:
                counts[metadata[field]] += 1
    all_turns = [turn for record in record_items for turn in record.turns]
    non_empty_deltas: list[int] = []
    for record in record_items:
        prior = ""
        for turn in record.turns:
            current = turn.event.content
            if current and current.startswith(prior) and len(current) > len(prior):
                non_empty_deltas.append(len(current) - len(prior))
            prior = current
    return {
        "records": len(record_items),
        "source_splits": dict(sorted(source_split_counts.items())),
        "selected_cards": len(selected_items),
        "selected_roles": dict(sorted(role_counts.items())),
        "selected_empty_kinds": dict(sorted(empty_counts.items())),
        "selected_splits": dict(sorted(split_counts.items())),
        "selected_source_distribution": {
            field: dict(sorted(counts.items()))
            for field, counts in selected_source_counts.items()
        },
        "selected_distinct_records": len(
            {candidate.record_id for candidate in selected_items}
        ),
        "source_turns": len(all_turns),
        "typing_delta_chars": {
            "min": min(non_empty_deltas) if non_empty_deltas else None,
            "max": max(non_empty_deltas) if non_empty_deltas else None,
        },
    }


def demo1_selection_distribution_errors(
    coverage: Mapping[str, Any]
) -> list[str]:
    """Apply the authored distribution gates to the *selected* card mix.

    The authored gates guard the 700-record source.  Subsampling can only make
    a build narrower, so the same thresholds are re-checked on the cards that
    actually ship: author share 20-40%, no persona/domain/register above 35%,
    and every length/event-count/trigger bucket present with none above 60%.
    """

    total = int(coverage.get("selected_cards", 0))
    distribution = coverage.get("selected_source_distribution", {})
    errors: list[str] = []
    if not total:
        return ["selection distribution: no cards were selected"]

    def share(count: int) -> float:
        return count / total

    author_counts = distribution.get("author_slot", {})
    if not 3 <= len(author_counts) <= 4:
        errors.append("selection distribution: each demo requires 3-4 authors")
    for author, count in sorted(author_counts.items()):
        if not 0.20 <= share(count) <= 0.40:
            errors.append(
                f"selection distribution: author {author!r} share "
                f"{share(count):.3f} must be 20-40%"
            )
    for field, minimum in (("persona", 5), ("domain", 5), ("register", 4)):
        counts = distribution.get(field, {})
        if len(counts) < minimum:
            errors.append(
                f"selection distribution: {field} requires at least {minimum} buckets"
            )
        for name, count in sorted(counts.items()):
            if share(count) > 0.35:
                errors.append(
                    f"selection distribution: {field} {name!r} exceeds 35% "
                    f"({share(count):.3f})"
                )
    for field in DEMO1_DERIVED_FIELDS:
        counts = distribution.get(field, {})
        required = (
            {"early", "middle", "late"}
            if field == "trigger_position"
            else {"short", "medium", "long"}
        )
        missing = required - set(counts)
        if missing:
            errors.append(
                f"selection distribution: {field} missing {sorted(missing)}"
            )
        for name, count in sorted(counts.items()):
            if share(count) > 0.60:
                errors.append(
                    f"selection distribution: {field} {name!r} exceeds 60% "
                    f"({share(count):.3f})"
                )
    return errors


def compile_demo1_dataset(
    records: Iterable[Mapping[str, Any]],
    *,
    targets: Demo1Targets = DEMO1_FULL_TARGETS,
    dev_fraction: float = 0.1,
) -> Demo1Build:
    compiled = compile_demo1_records(records, dev_fraction=dev_fraction)
    selected = select_demo1_candidates(compiled, targets=targets)
    record_by_id = {record.record_id: record for record in compiled}
    rows = tuple(
        render_demo1_card(record_by_id[candidate.record_id], candidate)
        for candidate in selected
    )
    return Demo1Build(
        records=compiled,
        selected=selected,
        rows=rows,
        coverage=demo1_coverage_report(compiled, selected),
    )
