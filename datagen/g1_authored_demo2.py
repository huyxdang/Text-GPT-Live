"""Authored-source contract for g1 Demo 2 (interjection: corrections + highlights).

Demo 2 records are *situations*, never cards.  Each record is one episode: a
standing instruction the typist types first, then passages that carry planted
typo/grammar errors (mode ``corrections``) or literal category matches and
non-literal bait (mode ``highlights``), plus the four graded trap kinds.

This module owns two things the compiler must not re-invent:

1. the schema/distribution validator (the acceptance gate for an authored
   tranche), and
2. the *text primitives* — occurrence counting, whole-word category matching,
   error spans — that decide what a ``quote`` actually means.  The compiler
   imports them so validation and compilation can never disagree about where a
   trigger is.

Occurrence semantics deliberately mirror :func:`app.domain.find_occurrence`
(plain, case-sensitive, step-by-one substring search) because that is what the
live app uses to place a highlight.  Train equals serve includes arithmetic.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from datagen.g1_authored import RESERVED_DOMAINS, RESERVED_PERSONAS


DEMO2_SCHEMA_VERSION = "g1-authored-demo2-1"
DEMO2_DEMO = "demo-2"
DEMO2_MODES = ("corrections", "highlights")
DEMO2_SEGMENT_KINDS = ("passage", "aside", "repair")
DEMO2_TRAPS = (
    "clean_text",
    "self_correction",
    "instruction_mention",
    "non_literal_match",
)
PAUSE_VALUES = ("none", "short", "medium", "long")

TYPO_SUBTYPES = (
    "letter-swap",
    "missing-letter",
    "doubled-letter",
    "spacing",
    "apostrophe",
    "capitalization",
)
GRAMMAR_SUBTYPES = (
    "tense",
    "agreement",
    "article",
    "preposition",
    "word-form",
    "plural",
)
ERROR_SUBTYPES = frozenset(TYPO_SUBTYPES + GRAMMAR_SUBTYPES)

# The g1 prompt renders event content through html.escape(..., quote=False).
# If authored text contained & < >, the bytes the model reads would differ from
# the bytes a quote is copied from, and every uniqueness proof would be a lie.
UNSAFE_TEXT_RE = re.compile(r"[<>&]")
MARKUP_RE = re.compile(
    r"(?:<action|</action>|<stream_event|<PREDICT_THIS_ACTION>"
    r"|\b(?:idle|respond|suggest_edit|highlight|delegate)\s*\()",
    re.IGNORECASE,
)
CATEGORY_WORD_RE = re.compile(r"^[a-z]+$")

MIN_SEGMENT_CHARS = 24
MIN_INSTRUCTION_CHARS = 16
MIN_TICK_CHARS = 4
#: A before-neighbour can only show a *partially typed* trigger when the trigger
#: is long enough to hold a 4-character final tick strictly inside it.
PARTIAL_BEFORE_MIN_TRIGGER_CHARS = 5

#: Total authored characters (instruction + every segment) per record.  Demo 2
#: passages are shorter than Demo 1 narrations — an episode is a handful of
#: sentences carrying triggers, not a story — so the edges sit lower than
#: Demo 1's 500/900.
LENGTH_BUCKET_EDGES = (400, 700)
EVENT_COUNT_BUCKET_EDGES = (4, 6)

#: Below this many passages in a batch, opener-share percentages are noise —
#: one or two passages sharing a first word can swing the share past any
#: sane threshold.  The gate does not run at all below the floor.
NARRATIVE_OPENER_FLOOR = 20
#: Deliberately soft: natural English genuinely favours "the" as a sentence
#: opener (roughly 10% of the time).  Forcing authors away from it produces
#: contrived prose, which is worse than a slightly skewed distribution.
NARRATIVE_OPENER_WARN_SHARE = 0.25
NARRATIVE_OPENER_ERROR_SHARE = 0.40
#: The top-four-openers cap is evaluated at two units.  Per file it is only a
#: warning (a canary): each author is deliberately assigned a distinct
#: persona/register, so a single author's openers concentrating is partly the
#: intended design, not a defect.  The build-blocking check is the merged
#: corpus across every batch, because that merged corpus is what the model
#: actually trains on.
NARRATIVE_OPENER_TOP4_ERROR_SHARE = 0.75
#: The corpus-level top-four check needs enough merged passages that a share
#: is meaningful; below this the check does not run at all (same rationale as
#: NARRATIVE_OPENER_FLOOR, just at corpus scale).
NARRATIVE_OPENER_CORPUS_FLOOR = 60


# --------------------------------------------------------------------------
# text primitives (shared with the compiler)
# --------------------------------------------------------------------------


def occurrence_start(text: str, quote: str, occurrence: int) -> int:
    """Start index of the ``occurrence``-th ``quote`` in ``text``, else ``-1``.

    Byte-for-byte the same walk as :func:`app.domain.find_occurrence`.
    """

    if not quote or occurrence < 1:
        return -1
    start = -1
    for _ in range(occurrence):
        start = text.find(quote, start + 1)
        if start < 0:
            return -1
    return start


def occurrence_count(text: str, quote: str) -> int:
    """How many times ``quote`` occurs in ``text`` under app search semantics."""

    if not quote:
        return 0
    total = 0
    start = -1
    while True:
        start = text.find(quote, start + 1)
        if start < 0:
            return total
        total += 1


def category_pattern(words: Sequence[str]) -> re.Pattern[str]:
    ordered = sorted({str(word) for word in words}, key=lambda word: (-len(word), word))
    if not ordered:
        raise ValueError("category_words must not be empty")
    body = "|".join(re.escape(word) for word in ordered)
    return re.compile(rf"\b(?:{body})\b", re.IGNORECASE)


def category_matches(text: str, words: Sequence[str]) -> list[tuple[int, int, str]]:
    """Whole-word, case-insensitive occurrences of any category word, in order."""

    return [
        (match.start(), match.end(), match.group(0))
        for match in category_pattern(words).finditer(text)
    ]


def error_span(text: str, wrong: str, occurrence: int) -> tuple[int, int]:
    start = occurrence_start(text, wrong, occurrence)
    if start < 0:
        raise ValueError(f"planted error {wrong!r} has no occurrence {occurrence}")
    return start, start + len(wrong)


# --------------------------------------------------------------------------
# small local helpers (duplicated from datagen.g1_authored on purpose; the
# shared module belongs to a concurrent session and must not be refactored)
# --------------------------------------------------------------------------


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _word_prefix(value: str, words: int) -> str:
    normalized = re.sub(r"[^a-z0-9']+", " ", value.lower())
    return " ".join(normalized.split()[:words])


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def _length_bucket(characters: int) -> str:
    short_edge, medium_edge = LENGTH_BUCKET_EDGES
    if characters < short_edge:
        return "short"
    if characters < medium_edge:
        return "medium"
    return "long"


def _event_count_bucket(segments: int) -> str:
    short_edge, medium_edge = EVENT_COUNT_BUCKET_EDGES
    if segments <= short_edge:
        return "short"
    if segments <= medium_edge:
        return "medium"
    return "long"


def _trigger_position_bucket(position: int, total: int) -> str:
    ratio = position / total if total else 1.0
    if ratio <= 1 / 3:
        return "early"
    if ratio <= 2 / 3:
        return "middle"
    return "late"


def load_authored_batches(paths: Iterable[Path]) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for path in sorted(paths):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: cannot load authored batch: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}: authored batch must be one JSON object")
        value["_path"] = str(path)
        batches.append(value)
    return batches


def demo2_authored_paths(root: Path) -> list[Path]:
    return sorted((root / "demo2").glob("*.json"))


# --------------------------------------------------------------------------
# per-record structural analysis (also consumed by the compiler's own guards)
# --------------------------------------------------------------------------


def segment_trigger_layout(
    record: Mapping[str, Any], segment: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Ordered trigger descriptors for one segment, in character order.

    Every descriptor has ``start``/``end`` offsets *within the segment text*,
    a ``role`` the compiler will grade, and the payload the action needs.
    Repair segments are handled separately: their draft/repair phases are not a
    linear slice of the final text.
    """

    text = str(segment.get("text", ""))
    kind = segment.get("kind")
    triggers: list[dict[str, Any]] = []
    if kind == "passage" and record.get("mode") == "corrections":
        for error in segment.get("errors") or []:
            start, end = error_span(
                text, str(error["wrong"]), int(error.get("occurrence", 1))
            )
            triggers.append(
                {
                    "start": start,
                    "end": end,
                    "role": "error",
                    "wrong": str(error["wrong"]),
                    "right": str(error["right"]),
                    "subtype": str(error["subtype"]),
                }
            )
    elif kind == "passage" and record.get("mode") == "highlights":
        found = category_matches(text, record.get("category_words") or [])
        marks = list(segment.get("marks") or [])
        for (start, end, surface), mark in zip(found, marks, strict=False):
            triggers.append(
                {
                    "start": start,
                    "end": end,
                    "role": "match" if mark.get("literal") else "bait",
                    "word": surface,
                }
            )
    triggers.sort(key=lambda trigger: (trigger["start"], trigger["end"]))
    return triggers


def _check_tick_spacing(location: str, text: str, cuts: Sequence[int]) -> list[str]:
    """Forced tick boundaries must leave 4+ characters between them.

    The trailing remainder counts too: a trigger that ends one character before
    the full stop would force a one-character tick, and the 4-7 character
    cadence guarantee is only worth having if it is exact.
    """

    errors: list[str] = []
    ordered = sorted({cut for cut in cuts if 0 < cut <= len(text)})
    previous = 0
    for cut in [*ordered, len(text)]:
        span = cut - previous
        if span == 0:
            continue
        if span < MIN_TICK_CHARS:
            errors.append(
                f"{location}: a tick boundary at character {previous} leaves a "
                f"{span}-character tick; keep 4+ characters before, between and "
                "after triggers"
            )
        previous = cut
    return errors


def _text_safety_errors(location: str, text: str) -> list[str]:
    errors: list[str] = []
    if UNSAFE_TEXT_RE.search(text):
        errors.append(
            f"{location}: text may not contain '<', '>' or '&'; stream escaping "
            "would desynchronise quotes from the visible textbox"
        )
    if MARKUP_RE.search(text):
        errors.append(f"{location}: authored text may not contain g1 action markup")
    return errors


def _validate_correction_segment(
    location: str, record: Mapping[str, Any], segment: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    text = str(segment.get("text", ""))
    kind = segment.get("kind")
    raw_errors = segment.get("errors")
    if kind != "passage" and raw_errors:
        errors.append(f"{location}: only passage segments may plant errors")
        return errors
    if raw_errors is None:
        raw_errors = []
    if not isinstance(raw_errors, list):
        errors.append(f"{location}: errors must be an array")
        return errors

    spans: list[tuple[int, int]] = []
    for error_index, error in enumerate(raw_errors):
        error_location = f"{location}:errors[{error_index}]"
        if not isinstance(error, Mapping):
            errors.append(f"{error_location}: must be an object")
            continue
        wrong = error.get("wrong")
        right = error.get("right")
        subtype = error.get("subtype")
        occurrence = error.get("occurrence", 1)
        if not isinstance(wrong, str) or not wrong.strip():
            errors.append(f"{error_location}: wrong must be a non-empty string")
            continue
        if not isinstance(right, str) or not right.strip():
            errors.append(f"{error_location}: right must be a non-empty string")
            continue
        if wrong == right:
            errors.append(f"{error_location}: right must differ from wrong")
        if subtype not in ERROR_SUBTYPES:
            errors.append(
                f"{error_location}: subtype {subtype!r} is not a typo/grammar subtype"
            )
        if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 1:
            errors.append(f"{error_location}: occurrence must be a positive integer")
            continue
        errors.extend(_text_safety_errors(error_location, f"{wrong} {right}"))
        start = occurrence_start(text, wrong, occurrence)
        if start < 0:
            errors.append(
                f"{error_location}: wrong {wrong!r} does not occur {occurrence} time(s) "
                "in the segment text"
            )
            continue
        spans.append((start, start + len(wrong)))

    ordered = sorted(spans)
    if ordered != spans:
        errors.append(f"{location}: errors must be listed in text order")
    for (start, end), (next_start, _) in zip(ordered, ordered[1:]):
        if next_start < end:
            errors.append(f"{location}: planted errors overlap at character {next_start}")
    errors.extend(_check_tick_spacing(location, text, [end for _, end in ordered]))
    return errors


def _validate_repair_segment(location: str, segment: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    text = str(segment.get("text", ""))
    repair = segment.get("repair")
    if not isinstance(repair, Mapping):
        errors.append(f"{location}: repair segments require a repair object")
        return errors
    wrong = repair.get("wrong")
    right = repair.get("right")
    subtype = repair.get("subtype")
    occurrence = repair.get("occurrence", 1)
    if not isinstance(wrong, str) or not wrong.strip():
        errors.append(f"{location}: repair.wrong must be a non-empty string")
        return errors
    if not isinstance(right, str) or not right.strip():
        errors.append(f"{location}: repair.right must be a non-empty string")
        return errors
    if wrong == right:
        errors.append(f"{location}: repair.right must differ from repair.wrong")
    if subtype not in ERROR_SUBTYPES:
        errors.append(f"{location}: repair.subtype {subtype!r} is not a typo/grammar subtype")
    if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 1:
        errors.append(f"{location}: repair.occurrence must be a positive integer")
        return errors
    errors.extend(_text_safety_errors(location, f"{wrong} {right}"))
    if wrong in text:
        errors.append(
            f"{location}: repair.wrong {wrong!r} still appears in the final text; "
            "a self-correction must leave the passage clean"
        )
    start = occurrence_start(text, right, occurrence)
    if start < 0:
        errors.append(
            f"{location}: repair.right {right!r} does not occur {occurrence} time(s)"
        )
        return errors
    if start < MIN_TICK_CHARS:
        errors.append(
            f"{location}: repair target begins at character {start}; keep 4+ "
            "characters of clean lead-in before the self-correction"
        )
    # After the deletion the typist retypes text[start:], so the repaired word
    # itself has to be a legal tick.
    if len(right) < MIN_TICK_CHARS:
        errors.append(
            f"{location}: repair.right {right!r} must be at least "
            f"{MIN_TICK_CHARS} characters"
        )
    if len(wrong) < MIN_TICK_CHARS:
        errors.append(
            f"{location}: repair.wrong {wrong!r} must be at least "
            f"{MIN_TICK_CHARS} characters"
        )
    # Phase three retypes text[start:], so the repaired word and whatever
    # follows it must both be legal ticks.
    errors.extend(_check_tick_spacing(location, text[start:], [len(right)]))
    return errors


def _validate_highlight_segment(
    location: str, record: Mapping[str, Any], segment: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    text = str(segment.get("text", ""))
    words = record.get("category_words") or []
    marks = segment.get("marks")
    if marks is None:
        marks = []
    if not isinstance(marks, list):
        errors.append(f"{location}: marks must be an array")
        return errors
    try:
        found = category_matches(text, words)
    except ValueError:
        return errors
    if len(found) != len(marks):
        errors.append(
            f"{location}: {len(found)} category word(s) occur in the text but "
            f"{len(marks)} mark(s) are declared; declare every occurrence in order"
        )
        return errors
    for mark_index, ((_, _, surface), mark) in enumerate(zip(found, marks, strict=True)):
        mark_location = f"{location}:marks[{mark_index}]"
        if not isinstance(mark, Mapping):
            errors.append(f"{mark_location}: must be an object")
            continue
        if mark.get("word") != surface:
            errors.append(
                f"{mark_location}: declared word {mark.get('word')!r} does not match "
                f"the {mark_index + 1}th category occurrence {surface!r}"
            )
        if not isinstance(mark.get("literal"), bool):
            errors.append(f"{mark_location}: literal must be a boolean")
    errors.extend(_check_tick_spacing(location, text, [end for _, end, _ in found]))
    return errors


def _validate_record(
    location: str, record: Mapping[str, Any], seen_text: dict[str, str],
    text_prefixes: dict[str, list[str]],
) -> tuple[list[str], dict[str, Any]]:
    """Validate one authored record and return errors plus derived facets."""

    errors: list[str] = []
    facets: dict[str, Any] = {}

    mode = record.get("mode")
    if mode not in DEMO2_MODES:
        errors.append(f"{location}: mode must be one of {list(DEMO2_MODES)}")
    for field in ("persona", "domain", "register"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            errors.append(f"{location}: {field} must be a non-empty string")
    if isinstance(record.get("persona"), str) and _slug(record["persona"]) in RESERVED_PERSONAS:
        errors.append(f"{location}: reserved persona leaked into training")
    if isinstance(record.get("domain"), str) and _slug(record["domain"]) in RESERVED_DOMAINS:
        errors.append(f"{location}: reserved domain leaked into training")

    instruction = record.get("instruction")
    if not isinstance(instruction, Mapping):
        errors.append(f"{location}: instruction must be an object")
        return errors, facets
    instruction_text = instruction.get("text")
    ack = instruction.get("ack")
    if not isinstance(instruction_text, str) or len(instruction_text.strip()) < MIN_INSTRUCTION_CHARS:
        errors.append(
            f"{location}: instruction.text must be at least {MIN_INSTRUCTION_CHARS} characters"
        )
        instruction_text = ""
    if not isinstance(ack, str) or not ack.strip():
        errors.append(f"{location}: instruction.ack must be a non-empty string")
        ack = ""
    if instruction.get("mentions_error") is not None and not isinstance(
        instruction.get("mentions_error"), bool
    ):
        errors.append(f"{location}: instruction.mentions_error must be a boolean")
    errors.extend(_text_safety_errors(f"{location}:instruction", instruction_text))
    errors.extend(_text_safety_errors(f"{location}:ack", ack))

    if mode == "highlights":
        category = record.get("category")
        words = record.get("category_words")
        if not isinstance(category, str) or not category.strip():
            errors.append(f"{location}: highlight records require a non-empty category")
        if not isinstance(words, list) or not words or any(
            not isinstance(word, str) or not CATEGORY_WORD_RE.fullmatch(word)
            for word in words
        ):
            errors.append(
                f"{location}: category_words must be a non-empty array of lowercase words"
            )
            words = []
        for word in words:
            if re.search(rf"\b{re.escape(word)}\b", instruction_text, re.IGNORECASE):
                errors.append(
                    f"{location}: category word {word!r} appears in the instruction; "
                    "occurrence 1 would point at instruction text"
                )
    else:
        if record.get("category") is not None or record.get("category_words") is not None:
            errors.append(f"{location}: category fields belong to highlight records only")
        words = []

    segments = record.get("segments")
    if not isinstance(segments, list) or not segments:
        errors.append(f"{location}: segments must be a non-empty array")
        return errors, facets
    if not 3 <= len(segments) <= 9:
        errors.append(f"{location}: segment count must be 3-9")

    passages = 0
    error_count = 0
    literal_marks = 0
    bait_marks = 0
    repairs = 0
    asides = 0
    clean_passages = 0
    subtypes: Counter[str] = Counter()
    first_trigger_segment: int | None = None
    trigger_lengths: list[int] = []
    passage_openers: list[str] = []

    for segment_index, segment in enumerate(segments, start=1):
        segment_location = f"{location}:segments[{segment_index - 1}]"
        if not isinstance(segment, Mapping):
            errors.append(f"{segment_location}: segment must be an object")
            continue
        kind = segment.get("kind")
        text = segment.get("text")
        if kind not in DEMO2_SEGMENT_KINDS:
            errors.append(f"{segment_location}: invalid kind {kind!r}")
            continue
        if not isinstance(text, str) or len(text.strip()) < MIN_SEGMENT_CHARS:
            errors.append(
                f"{segment_location}: text must be at least {MIN_SEGMENT_CHARS} characters"
            )
            continue
        if segment.get("pause_after", "none") not in PAUSE_VALUES:
            errors.append(f"{segment_location}: invalid pause_after")
        errors.extend(_text_safety_errors(segment_location, text))

        normalized = _normalized(text)
        earlier = seen_text.get(normalized)
        if earlier is not None:
            errors.append(f"{segment_location}: exact text duplicates {earlier}")
        else:
            seen_text[normalized] = segment_location
        prefix = _word_prefix(text, 7)
        if len(prefix.split()) == 7:
            text_prefixes.setdefault(prefix, []).append(segment_location)

        if kind == "repair":
            if mode != "corrections":
                errors.append(f"{segment_location}: repair segments are corrections-only")
            else:
                repair_errors = _validate_repair_segment(segment_location, segment)
                errors.extend(repair_errors)
                if not repair_errors:
                    repairs += 1
                    subtypes[str(segment["repair"]["subtype"])] += 1
            continue
        if kind == "aside":
            mentions = segment.get("mentions")
            if not isinstance(mentions, list) or not mentions or any(
                not isinstance(item, str) or not item.strip() for item in mentions
            ):
                errors.append(
                    f"{segment_location}: aside segments require a non-empty mentions array"
                )
            else:
                for mention in mentions:
                    if mention not in text:
                        errors.append(
                            f"{segment_location}: mention {mention!r} does not appear in the text"
                        )
            if segment.get("errors") or segment.get("marks"):
                errors.append(f"{segment_location}: aside segments carry no triggers")
            if mode == "highlights" and words and category_matches(text, words):
                # An unmarked category word inside an aside is a silent
                # false-negative: the compiler grades no trigger there.
                errors.append(
                    f"{segment_location}: aside segments may not contain category words"
                )
            asides += 1
            continue

        passages += 1
        opener = _word_prefix(text, 1)
        if opener:
            passage_openers.append(opener)
        trap = segment.get("trap")
        if trap is not None and trap != "clean_text":
            errors.append(f"{segment_location}: passage trap must be 'clean_text' or absent")

        if mode == "corrections":
            segment_errors = _validate_correction_segment(segment_location, record, segment)
            errors.extend(segment_errors)
            planted = segment.get("errors") or []
            if not segment_errors:
                error_count += len(planted)
                for error in planted:
                    subtypes[str(error["subtype"])] += 1
                    trigger_lengths.append(len(str(error["wrong"])))
                if planted and first_trigger_segment is None:
                    first_trigger_segment = segment_index
            if trap == "clean_text":
                if planted:
                    errors.append(
                        f"{segment_location}: a clean_text trap may not plant errors"
                    )
                else:
                    clean_passages += 1
            elif not planted:
                errors.append(
                    f"{segment_location}: a passage with no planted error must be "
                    "labelled trap='clean_text'"
                )
        elif mode == "highlights":
            segment_errors = _validate_highlight_segment(segment_location, record, segment)
            errors.extend(segment_errors)
            marks = segment.get("marks") or []
            if not segment_errors:
                for mark in marks:
                    if mark.get("literal"):
                        literal_marks += 1
                        trigger_lengths.append(len(str(mark["word"])))
                    else:
                        bait_marks += 1
                if any(mark.get("literal") for mark in marks) and first_trigger_segment is None:
                    first_trigger_segment = segment_index
            if trap == "clean_text":
                if marks:
                    errors.append(
                        f"{segment_location}: a clean_text trap may not contain category words"
                    )
                else:
                    clean_passages += 1
            elif not marks:
                errors.append(
                    f"{segment_location}: a passage with no category word must be "
                    "labelled trap='clean_text'"
                )

    if passages < 2:
        errors.append(f"{location}: at least two passage segments are required")

    if mode == "corrections" and error_count == 0:
        errors.append(f"{location}: correction records require at least one planted error")
    if mode == "highlights":
        if literal_marks == 0:
            errors.append(f"{location}: highlight records require at least one literal match")
        # Substring bleed makes the app's occurrence index ambiguous: quoting
        # "fox" when "foxes" appeared earlier marks the wrong characters.
        full_text = "\n".join(
            [str(instruction_text)]
            + [
                str(segment.get("text", ""))
                for segment in segments
                if isinstance(segment, Mapping)
            ]
        )
        surfaces = {
            str(mark.get("word"))
            for segment in segments
            if isinstance(segment, Mapping)
            for mark in (segment.get("marks") or [])
            if isinstance(mark, Mapping) and isinstance(mark.get("word"), str)
        }
        for surface in sorted(surfaces):
            whole = len(re.findall(rf"\b{re.escape(surface)}\b", full_text))
            if occurrence_count(full_text, surface) != whole:
                errors.append(
                    f"{location}: {surface!r} also appears inside a longer word; "
                    "the highlight occurrence index would be ambiguous"
                )

    total_characters = len(str(instruction_text)) + sum(
        len(str(segment.get("text", "")))
        for segment in segments
        if isinstance(segment, Mapping)
    )
    facets = {
        "mode": str(mode),
        "length_bucket": _length_bucket(total_characters),
        "event_count_bucket": _event_count_bucket(len(segments)),
        "trigger_position": _trigger_position_bucket(
            first_trigger_segment if first_trigger_segment else len(segments),
            len(segments),
        ),
        "errors": error_count,
        "literal_marks": literal_marks,
        "bait_marks": bait_marks,
        "repairs": repairs,
        "asides": asides,
        "clean_passages": clean_passages,
        "subtypes": dict(subtypes),
        "category": str(record.get("category")) if mode == "highlights" else None,
        "instruction_opener": _word_prefix(str(instruction_text), 2),
        "instruction_first_word": _word_prefix(str(instruction_text), 1),
        "ack_opener": _word_prefix(str(ack), 2),
        "ack_first_word": _word_prefix(str(ack), 1),
        "passage_openers": passage_openers,
        "partial_before_triggers": sum(
            1 for length in trigger_lengths if length >= PARTIAL_BEFORE_MIN_TRIGGER_CHARS
        ),
        "total_triggers": len(trigger_lengths),
        "characters": total_characters,
    }
    return errors, facets


# --------------------------------------------------------------------------
# batch validation + distribution gates
# --------------------------------------------------------------------------


def validate_demo2_batches(
    batches: Iterable[Mapping[str, Any]],
    *,
    enforce_distribution: bool = False,
) -> dict[str, Any]:
    batch_items = list(batches)
    errors: list[str] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    text_prefixes: dict[str, list[str]] = {}
    passage_openers_by_file: dict[str, Counter[str]] = {}

    for batch_index, batch in enumerate(batch_items):
        label = str(batch.get("_path", f"batch-{batch_index}"))
        if batch.get("schema_version") != DEMO2_SCHEMA_VERSION:
            errors.append(f"{label}: schema_version must be {DEMO2_SCHEMA_VERSION!r}")
        if batch.get("demo") != DEMO2_DEMO:
            errors.append(f"{label}: demo must be {DEMO2_DEMO!r}")
        author = batch.get("author")
        if not isinstance(author, Mapping):
            errors.append(f"{label}: author must be an object")
            continue
        if not all(
            isinstance(author.get(key), str) and author[key].strip()
            for key in ("model", "slot")
        ):
            errors.append(f"{label}: author requires non-empty model and slot")
        if not isinstance(author.get("tranche"), int) or isinstance(
            author.get("tranche"), bool
        ):
            errors.append(f"{label}: author.tranche must be an integer")
        authored_records = batch.get("records")
        if not isinstance(authored_records, list):
            errors.append(f"{label}: records must be an array")
            continue

        batch_records: list[dict[str, Any]] = []
        for record_index, raw_record in enumerate(authored_records):
            location = f"{label}:records[{record_index}]"
            if not isinstance(raw_record, Mapping):
                errors.append(f"{location}: record must be an object")
                continue
            record = dict(raw_record)
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id.strip():
                errors.append(f"{location}: id must be a non-empty string")
                continue
            if record_id in record_ids:
                errors.append(f"{location}: duplicate record id {record_id!r}")
            record_ids.add(record_id)
            record["agent"] = author.get("slot")
            record_errors, facets = _validate_record(
                location, record, seen_text, text_prefixes
            )
            errors.extend(record_errors)
            record.update(facets)
            records.append(record)
            batch_records.append(record)

        if len(batch_records) >= 20:
            errors.extend(_batch_opener_gates(label, batch_records))

        batch_passage_openers: Counter[str] = Counter()
        for record in batch_records:
            batch_passage_openers.update(record.get("passage_openers") or [])
        passage_openers_by_file[label] = batch_passage_openers
        opener_warnings, opener_errors = _narrative_opener_gate(
            label, batch_passage_openers, "narrative passage"
        )
        warnings.extend(opener_warnings)
        errors.extend(opener_errors)

    for prefix, locations in text_prefixes.items():
        if len(locations) >= 3:
            errors.append(
                f"template: 7-word prefix {prefix!r} repeats {len(locations)} times "
                f"({', '.join(locations[:3])})"
            )

    counters = {
        field: Counter(str(record.get(field)) for record in records)
        for field in (
            "agent",
            "persona",
            "domain",
            "register",
            "mode",
            "length_bucket",
            "event_count_bucket",
            "trigger_position",
        )
    }
    subtype_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    ack_openers: Counter[str] = Counter()
    ack_first_words: Counter[str] = Counter()
    instruction_openers: Counter[str] = Counter()
    passage_openers: Counter[str] = Counter()
    joint: dict[str, Counter[str]] = {
        "mode_x_trigger": Counter(),
        "event_count_x_trigger": Counter(),
        "length_x_trigger": Counter(),
        "domain_x_mode": Counter(),
        "persona_x_mode": Counter(),
    }
    totals = Counter()
    records_with_bait = 0
    records_with_repair = 0
    records_with_aside = 0
    records_with_clean = 0
    for record in records:
        subtype_counts.update(record.get("subtypes") or {})
        if record.get("category"):
            category_counts[str(record["category"])] += 1
        ack_openers[str(record.get("ack_opener"))] += 1
        ack_first_words[str(record.get("ack_first_word"))] += 1
        instruction_openers[str(record.get("instruction_opener"))] += 1
        passage_openers.update(record.get("passage_openers") or [])
        for key in ("errors", "literal_marks", "bait_marks", "repairs", "asides", "clean_passages"):
            totals[key] += int(record.get(key, 0) or 0)
        totals["partial_before_triggers"] += int(record.get("partial_before_triggers", 0) or 0)
        totals["total_triggers"] += int(record.get("total_triggers", 0) or 0)
        records_with_bait += 1 if int(record.get("bait_marks", 0) or 0) else 0
        records_with_repair += 1 if int(record.get("repairs", 0) or 0) else 0
        records_with_aside += 1 if int(record.get("asides", 0) or 0) else 0
        records_with_clean += 1 if int(record.get("clean_passages", 0) or 0) else 0
        mode = str(record.get("mode"))
        trigger = str(record.get("trigger_position"))
        joint["mode_x_trigger"][f"{mode}|{trigger}"] += 1
        joint["event_count_x_trigger"][
            f"{record.get('event_count_bucket')}|{trigger}"
        ] += 1
        joint["length_x_trigger"][f"{record.get('length_bucket')}|{trigger}"] += 1
        joint["domain_x_mode"][f"{record.get('domain')}|{mode}"] += 1
        joint["persona_x_mode"][f"{record.get('persona')}|{mode}"] += 1

    errors.extend(
        _narrative_opener_corpus_gate(passage_openers, "narrative passage")
    )

    if enforce_distribution and records:
        errors.extend(
            _distribution_gates(
                records,
                counters=counters,
                subtype_counts=subtype_counts,
                category_counts=category_counts,
                ack_openers=ack_openers,
                ack_first_words=ack_first_words,
                instruction_openers=instruction_openers,
                joint=joint,
                records_with_bait=records_with_bait,
                records_with_repair=records_with_repair,
                records_with_aside=records_with_aside,
                records_with_clean=records_with_clean,
                totals=totals,
            )
        )

    return {
        "passed": not errors,
        "distribution_enforced": bool(enforce_distribution and records),
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "batches": len(batch_items),
            "records": len(records),
            "errors_planted": totals["errors"],
            "literal_marks": totals["literal_marks"],
            "bait_marks": totals["bait_marks"],
            "repairs": totals["repairs"],
            "asides": totals["asides"],
            "clean_passages": totals["clean_passages"],
            "partial_before_triggers": totals["partial_before_triggers"],
            "total_triggers": totals["total_triggers"],
            "records_with_bait": records_with_bait,
            "records_with_repair": records_with_repair,
            "records_with_aside": records_with_aside,
            "records_with_clean_passage": records_with_clean,
            "error_subtypes": dict(sorted(subtype_counts.items())),
            "categories": dict(sorted(category_counts.items())),
            "ack_openers": dict(sorted(ack_openers.items())),
            "instruction_openers": dict(sorted(instruction_openers.items())),
            "passage_openers": dict(sorted(passage_openers.items())),
            "passage_openers_by_file": {
                file_label: dict(sorted(counter.items()))
                for file_label, counter in sorted(passage_openers_by_file.items())
            },
            "joint_distributions": {
                name: dict(sorted(counts.items())) for name, counts in joint.items()
            },
            **{field: dict(sorted(counts.items())) for field, counts in counters.items()},
        },
    }


def _batch_opener_gates(label: str, batch_records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Per-batch wording caps so one author cannot canned-phrase its tranche."""

    errors: list[str] = []
    total = len(batch_records)
    ack_first = Counter(str(record.get("ack_first_word")) for record in batch_records)
    ack_two = Counter(str(record.get("ack_opener")) for record in batch_records)
    instruction_two = Counter(
        str(record.get("instruction_opener")) for record in batch_records
    )
    for opener, count in ack_first.items():
        if _share(count, total) > 0.25:
            errors.append(
                f"{label}: acknowledgment first word {opener!r} exceeds 25%; vary the ack"
            )
    for opener, count in ack_two.items():
        if _share(count, total) > 0.40:
            errors.append(
                f"{label}: acknowledgment opener {opener!r} exceeds 40%; vary the ack"
            )
    for opener, count in instruction_two.items():
        if _share(count, total) > 0.50:
            errors.append(
                f"{label}: instruction opener {opener!r} exceeds 50%; vary the instruction"
            )
    return errors


def _narrative_opener_gate(
    label: str, openers: Counter[str], kind_label: str
) -> tuple[list[str], list[str]]:
    """Soft caps on the first word of narrative body text within one batch.

    Instruction and acknowledgment openers are already gated (see
    ``_batch_opener_gates``); this closes the same gap on the passages
    themselves, where nothing previously watched the distribution and it
    quietly converged on "The" in the real authored corpus.  Thresholds are
    deliberately soft — natural English favours "the" as an opener roughly
    10% of the time, and pushing authors away from it produces contrived
    prose, which is worse than a somewhat skewed distribution.

    The single-opener caps (warn/error) are a genuine per-file quality bar:
    a file where 40%+ of passages open the same way is defective regardless
    of what the rest of the corpus looks like.  The top-four cap is
    different — each author is deliberately given a distinct persona and
    register, so one author's openers concentrating is partly the intended
    design.  It is therefore only a per-file warning here; the build-blocking
    top-four check runs once, across the merged corpus, in
    ``_narrative_opener_corpus_gate``.
    """

    warnings: list[str] = []
    errors: list[str] = []
    total = sum(openers.values())
    if total < NARRATIVE_OPENER_FLOOR:
        return warnings, errors
    for opener, count in openers.items():
        share = _share(count, total)
        if share > NARRATIVE_OPENER_ERROR_SHARE:
            errors.append(
                f"{label}: {kind_label} opener {opener!r} is {share:.0%} of the batch "
                f"(>{NARRATIVE_OPENER_ERROR_SHARE:.0%}); vary the narrative opening"
            )
        elif share > NARRATIVE_OPENER_WARN_SHARE:
            warnings.append(
                f"{label}: {kind_label} opener {opener!r} is {share:.0%} of the batch "
                f"(>{NARRATIVE_OPENER_WARN_SHARE:.0%}); vary the narrative opening"
            )
    top_four = sum(count for _, count in openers.most_common(4))
    top_four_share = _share(top_four, total)
    if top_four_share > NARRATIVE_OPENER_TOP4_ERROR_SHARE:
        warnings.append(
            f"{label}: the top four {kind_label} openers cover {top_four_share:.0%} "
            f"of the batch (>{NARRATIVE_OPENER_TOP4_ERROR_SHARE:.0%}); this is a "
            "per-file canary only, not a build blocker — the merged-corpus top-four "
            "check is what gates acceptance"
        )
    return warnings, errors


def _narrative_opener_corpus_gate(
    openers: Counter[str], kind_label: str
) -> list[str]:
    """Build-blocking top-four cap, evaluated across the merged corpus.

    The model trains on all authored batches merged together, not on any one
    author file, so this is the unit where a real opener-concentration defect
    should be caught.  Per-file concentration is expected — see
    ``_narrative_opener_gate`` — because each author is deliberately given a
    distinct persona/register; it is only a warning there.
    """

    errors: list[str] = []
    total = sum(openers.values())
    if total < NARRATIVE_OPENER_CORPUS_FLOOR:
        return errors
    top_four = sum(count for _, count in openers.most_common(4))
    top_four_share = _share(top_four, total)
    if top_four_share > NARRATIVE_OPENER_TOP4_ERROR_SHARE:
        errors.append(
            f"corpus: the top four {kind_label} openers cover {top_four_share:.0%} "
            f"of the merged corpus (>{NARRATIVE_OPENER_TOP4_ERROR_SHARE:.0%}); vary "
            "the narrative openings across authors"
        )
    return errors


def _distribution_gates(
    records: Sequence[Mapping[str, Any]],
    *,
    counters: Mapping[str, Counter[str]],
    subtype_counts: Counter[str],
    category_counts: Counter[str],
    ack_openers: Counter[str],
    ack_first_words: Counter[str],
    instruction_openers: Counter[str],
    joint: Mapping[str, Counter[str]],
    records_with_bait: int,
    records_with_repair: int,
    records_with_aside: int,
    records_with_clean: int,
    totals: Counter[str],
) -> list[str]:
    errors: list[str] = []
    total = len(records)

    agent_counts = counters["agent"]
    if not 3 <= len(agent_counts) <= 4:
        errors.append("distribution: Demo 2 requires 3-4 authors")
    for agent, count in agent_counts.items():
        if not 0.20 <= _share(count, total) <= 0.40:
            errors.append(f"distribution: author {agent!r} share must be 20-40%")

    for field, minimum in (("persona", 5), ("domain", 5), ("register", 4)):
        counts = counters[field]
        if len(counts) < minimum:
            errors.append(f"distribution: {field} requires at least {minimum} buckets")
        for name, count in counts.items():
            if _share(count, total) > 0.35:
                errors.append(f"distribution: {field} {name!r} exceeds 35%")

    for field in ("length_bucket", "event_count_bucket", "trigger_position"):
        counts = counters[field]
        required = (
            {"early", "middle", "late"}
            if field == "trigger_position"
            else {"short", "medium", "long"}
        )
        missing = required - set(counts)
        if missing:
            errors.append(f"distribution: {field} missing {sorted(missing)}")
        for name, count in counts.items():
            if _share(count, total) > 0.60:
                errors.append(f"distribution: {field} {name!r} exceeds 60%")

    mode_counts = counters["mode"]
    for mode in DEMO2_MODES:
        share = _share(mode_counts.get(mode, 0), total)
        if not 0.20 <= share <= 0.80:
            errors.append(
                f"distribution: mode {mode!r} share {share:.2f} must be 20-80%"
            )

    subtype_total = sum(subtype_counts.values())
    if len(subtype_counts) < 6:
        errors.append("distribution: at least 6 distinct error subtypes are required")
    for subtype, count in subtype_counts.items():
        if _share(count, subtype_total) > 0.35:
            errors.append(f"distribution: error subtype {subtype!r} exceeds 35%")
    if not {subtype for subtype in subtype_counts} & set(TYPO_SUBTYPES):
        errors.append("distribution: no typo subtypes present")
    if not {subtype for subtype in subtype_counts} & set(GRAMMAR_SUBTYPES):
        errors.append("distribution: no grammar subtypes present")

    highlight_records = mode_counts.get("highlights", 0)
    if highlight_records:
        if len(category_counts) < 4:
            errors.append("distribution: at least 4 highlight categories are required")
        for category, count in category_counts.items():
            if _share(count, highlight_records) > 0.40:
                errors.append(f"distribution: highlight category {category!r} exceeds 40%")
        if _share(records_with_bait, highlight_records) < 0.20:
            errors.append(
                "distribution: fewer than 20% of highlight records carry non-literal bait"
            )
    correction_records = mode_counts.get("corrections", 0)
    if correction_records and _share(records_with_repair, correction_records) < 0.15:
        errors.append(
            "distribution: fewer than 15% of correction records carry a self-correction"
        )
    if _share(records_with_clean, total) < 0.15:
        errors.append("distribution: fewer than 15% of records carry a clean-text trap")
    if _share(records_with_aside, total) < 0.10:
        errors.append(
            "distribution: fewer than 10% of records carry an instruction-mention aside"
        )
    if _share(totals["partial_before_triggers"], totals["total_triggers"]) < 0.50:
        errors.append(
            "distribution: fewer than 50% of triggers are long enough for a "
            "partially-typed before-neighbour"
        )

    for opener, count in ack_first_words.items():
        if _share(count, total) > 0.20:
            errors.append(f"distribution: acknowledgment first word {opener!r} exceeds 20%")
    for opener, count in ack_openers.items():
        if _share(count, total) > 0.30:
            errors.append(f"distribution: acknowledgment opener {opener!r} exceeds 30%")
    for opener, count in instruction_openers.items():
        if _share(count, total) > 0.35:
            errors.append(f"distribution: instruction opener {opener!r} exceeds 35%")

    for name, counts in joint.items():
        if not counts:
            continue
        bucket_totals: Counter[str] = Counter()
        for key, count in counts.items():
            bucket_totals[key.split("|", 1)[0]] += count
        for prefix, prefix_total in bucket_totals.items():
            if prefix_total < 5:
                continue
            dominant, dominant_count = max(
                (
                    (key, count)
                    for key, count in counts.items()
                    if key.split("|", 1)[0] == prefix
                ),
                key=lambda item: (item[1], item[0]),
            )
            if _share(dominant_count, prefix_total) > 0.80:
                errors.append(
                    f"distribution: {name} {dominant!r} holds "
                    f"{dominant_count}/{prefix_total}; cap joint dominance at 80%"
                )
    return errors
