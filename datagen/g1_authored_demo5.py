"""Validation and distribution reporting for agent-authored Demo 5 banks.

Demo 5 authors do not write episodes.  They write three small banks of
*words*: reminder-request phrasings, cancellation phrasings, and filler
narration passages (some tagged as quoted-question bait or a direct-address
utterance).  ``datagen.g1_demo5`` is the generator that cross-products these
banks with sampled intervals, fire offsets, tick alignments, and cancellation
timings into schedules; this module is the gate on the banks themselves --
their grammar (placeholders match schedule kind), their mechanical hygiene
(no markup leakage, no duplicate phrasings), and their distribution (author
share, persona/domain/register breadth, bucket coverage, opener variety),
mirroring the gates ``g1_authored_demo3`` runs on Demo 3's clauses.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datagen.g1_demo5 import (
    INTERVAL_PLACEHOLDER,
    MIN_PHRASE_CHARS,
)


AUTHORED_SCHEMA_VERSION = "g1-authored-demo5-1"
DEMO = "demo-5"

RESERVED_PERSONAS = {
    "product-reviewer",
    "letter-to-a-friend",
    "technical-writeup",
}
RESERVED_DOMAINS = {"sport", "health", "personal-finance"}

#: Markup that must never appear in authored text of any kind.
MARKUP_RE = re.compile(
    r"(?:<action>|</action>|<stream_event|<PREDICT_THIS_ACTION>|"
    r"\bidle\(\)|\brespond\(\{|\bsuggest_edit\(|\bhighlight\(|\bdelegate\()"
)

ENTRY_KINDS = ("request", "cancellation", "filler")
SCHEDULE_KINDS = ("once", "every")
FILLER_TRAPS = ("none", "bait", "address")

#: Author share, persona breadth, and bucket coverage are scale-dependent:
#: they cannot hold on a handful of records, and asserting them there would
#: only teach people to disable the gate.  Enforcement therefore requires a
#: bank of at least this many entries; a smaller bank must say so out loud.
DISTRIBUTION_FLOOR = 24

#: Below this many entries of one kind, opener-share percentages are noise.
NARRATIVE_OPENER_FLOOR = 20
NARRATIVE_OPENER_WARN_SHARE = 0.25
NARRATIVE_OPENER_ERROR_SHARE = 0.40
NARRATIVE_OPENER_TOP4_ERROR_SHARE = 0.75
NARRATIVE_OPENER_CORPUS_FLOOR = 60
COORDINATING_CONJUNCTIONS = frozenset({"and", "but", "so", "then", "or"})


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _word_prefix(value: str, words: int) -> str:
    normalized = re.sub(r"[^a-z0-9']+", " ", value.lower())
    return " ".join(normalized.split()[:words])


def _opener(value: str) -> str:
    words = re.sub(r"[^a-z0-9']+", " ", value.lower()).split()
    if words and words[0] in COORDINATING_CONJUNCTIONS:
        words = words[1:]
    return words[0] if words else ""


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def _check_no_markup(text: str, location: str, field: str, errors: list[str]) -> None:
    if MARKUP_RE.search(text):
        errors.append(f"{location}: {field} contains action or stream markup")
    for character in "<>&":
        if character in text:
            errors.append(
                f"{location}: {field} must not contain {character!r}; compile_stream "
                "HTML-escapes event content and the literal character would not "
                "round-trip"
            )
    if text != text.strip():
        errors.append(f"{location}: {field} must not have surrounding whitespace")


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


def authored_paths(root: Path, demo: str = DEMO) -> list[Path]:
    return sorted((root / demo.replace("-", "")).glob("*.json"))


# --------------------------------------------------------------------------
# batch validation
# --------------------------------------------------------------------------


def _validate_request(entry: dict[str, Any], location: str, errors: list[str]) -> None:
    schedule_kind = entry.get("schedule_kind")
    if schedule_kind not in SCHEDULE_KINDS:
        errors.append(f"{location}: schedule_kind must be one of {SCHEDULE_KINDS}")
        return
    text = entry.get("text_template")
    ack = entry.get("gold_ack_template")
    fire_message = entry.get("fire_message")
    for field, value in (("text_template", text), ("gold_ack_template", ack), ("fire_message", fire_message)):
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{location}: {field} must be a non-empty string")
    if not isinstance(text, str) or not isinstance(ack, str):
        return
    _check_no_markup(text, location, "text_template", errors)
    _check_no_markup(ack, location, "gold_ack_template", errors)
    if len(text.replace(INTERVAL_PLACEHOLDER, "000")) < MIN_PHRASE_CHARS:
        errors.append(f"{location}: text_template must be at least {MIN_PHRASE_CHARS} characters")
    if schedule_kind == "every":
        if INTERVAL_PLACEHOLDER not in text:
            errors.append(f"{location}: every-N text_template must contain {INTERVAL_PLACEHOLDER!r}")
        if INTERVAL_PLACEHOLDER not in ack:
            errors.append(f"{location}: every-N gold_ack_template must contain {INTERVAL_PLACEHOLDER!r}")
    else:
        if INTERVAL_PLACEHOLDER in text:
            errors.append(f"{location}: a once-kind text_template must not contain {INTERVAL_PLACEHOLDER!r}")
        if INTERVAL_PLACEHOLDER in ack:
            errors.append(f"{location}: a once-kind gold_ack_template must not contain {INTERVAL_PLACEHOLDER!r}")


def _validate_cancellation(entry: dict[str, Any], location: str, errors: list[str]) -> None:
    for field in ("text", "gold_ack"):
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{location}: {field} must be a non-empty string")
            continue
        _check_no_markup(value, location, field, errors)
    text = entry.get("text")
    if isinstance(text, str) and len(text) < MIN_PHRASE_CHARS:
        errors.append(f"{location}: text must be at least {MIN_PHRASE_CHARS} characters")


def _validate_filler(entry: dict[str, Any], location: str, errors: list[str]) -> None:
    text = entry.get("text")
    if not isinstance(text, str) or not text.strip():
        errors.append(f"{location}: text must be a non-empty string")
        text = None
    trap = entry.get("trap", "none")
    if trap not in FILLER_TRAPS:
        errors.append(f"{location}: trap must be one of {FILLER_TRAPS}")
    gold_reply = entry.get("gold_reply")
    if trap == "address":
        if not isinstance(gold_reply, str) or not gold_reply.strip():
            errors.append(f"{location}: an address filler requires a non-empty gold_reply")
        elif isinstance(text, str):
            _check_no_markup(gold_reply, location, "gold_reply", errors)
    elif gold_reply is not None:
        errors.append(f"{location}: gold_reply is only valid on an address filler")
    if trap == "bait" and isinstance(text, str) and '"' not in text and "'" not in text:
        errors.append(f"{location}: a bait filler must contain a quoted span to be genuine bait")
    if text is not None:
        _check_no_markup(text, location, "text", errors)
        if len(text) < MIN_PHRASE_CHARS:
            errors.append(f"{location}: text must be at least {MIN_PHRASE_CHARS} characters")


def validate_demo5_batches(
    batches: Iterable[dict[str, Any]],
    *,
    enforce_distribution: bool = False,
) -> dict[str, Any]:
    """Validate authored Demo 5 bank batches and report every distribution."""

    batch_items = list(batches)
    errors: list[str] = []
    warnings: list[str] = []
    all_entries: list[dict[str, Any]] = []
    entry_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    openers_by_file: dict[str, Counter[str]] = {}

    for batch_index, batch in enumerate(batch_items):
        label = str(batch.get("_path", f"batch-{batch_index}"))
        if batch.get("schema_version") != AUTHORED_SCHEMA_VERSION:
            errors.append(f"{label}: schema_version must be {AUTHORED_SCHEMA_VERSION!r}")
        if batch.get("demo") != DEMO:
            errors.append(f"{label}: demo must be {DEMO!r}")
        author = batch.get("author")
        if not isinstance(author, dict):
            errors.append(f"{label}: author must be an object")
            continue
        if not all(isinstance(author.get(key), str) and author[key].strip() for key in ("model", "slot")):
            errors.append(f"{label}: author requires non-empty model and slot")
        if not isinstance(author.get("tranche"), int) or isinstance(author.get("tranche"), bool):
            errors.append(f"{label}: author.tranche must be an integer")

        bank_entries = batch.get("bank")
        if not isinstance(bank_entries, list):
            errors.append(f"{label}: bank must be an array")
            continue

        batch_openers: Counter[str] = Counter()
        for entry_index, raw_entry in enumerate(bank_entries):
            location = f"{label}:bank[{entry_index}]"
            if not isinstance(raw_entry, dict):
                errors.append(f"{location}: entry must be an object")
                continue
            entry = dict(raw_entry)
            entry_id = entry.get("id")
            if not isinstance(entry_id, str) or not entry_id.strip():
                errors.append(f"{location}: id must be a non-empty string")
                continue
            if entry_id in entry_ids:
                errors.append(f"{location}: duplicate entry id {entry_id!r}")
            entry_ids.add(entry_id)
            entry["agent"] = author.get("slot")

            kind = entry.get("kind")
            if kind not in ENTRY_KINDS:
                errors.append(f"{location}: kind must be one of {ENTRY_KINDS}")
                continue

            for field in ("persona", "domain", "register"):
                if not isinstance(entry.get(field), str) or not entry[field].strip():
                    errors.append(f"{location}: {field} must be a non-empty string")
            if isinstance(entry.get("persona"), str) and _slug(entry["persona"]) in RESERVED_PERSONAS:
                errors.append(f"{location}: reserved persona leaked into training")
            if isinstance(entry.get("domain"), str) and _slug(entry["domain"]) in RESERVED_DOMAINS:
                errors.append(f"{location}: reserved domain leaked into training")

            if kind == "request":
                _validate_request(entry, location, errors)
                dedupe_text = entry.get("text_template")
                opener_text = entry.get("text_template", "")
            elif kind == "cancellation":
                _validate_cancellation(entry, location, errors)
                dedupe_text = entry.get("text")
                opener_text = entry.get("text", "")
            else:
                _validate_filler(entry, location, errors)
                dedupe_text = entry.get("text")
                opener_text = entry.get("text", "")

            if isinstance(dedupe_text, str) and dedupe_text:
                normalized = _normalized(dedupe_text)
                earlier = seen_text.get(normalized)
                if earlier is not None:
                    errors.append(f"{location}: text duplicates {earlier}; vary the phrasing")
                else:
                    seen_text[normalized] = location

            opener = _opener(opener_text) if isinstance(opener_text, str) else ""
            if opener:
                batch_openers[opener] += 1
            entry["opener"] = opener
            entry["length_bucket"] = _length_bucket(len(opener_text) if isinstance(opener_text, str) else 0)

            all_entries.append(entry)

        openers_by_file[label] = batch_openers
        opener_warnings, opener_errors = _narrative_opener_gate(label, batch_openers)
        warnings.extend(opener_warnings)
        errors.extend(opener_errors)

    corpus_openers: Counter[str] = Counter()
    for counter in openers_by_file.values():
        corpus_openers.update(counter)
    errors.extend(_narrative_opener_corpus_gate(corpus_openers))

    by_kind: dict[str, list[dict[str, Any]]] = {kind: [] for kind in ENTRY_KINDS}
    for entry in all_entries:
        if entry.get("kind") in by_kind:
            by_kind[entry["kind"]].append(entry)

    counters = {
        field: Counter(str(entry.get(field)) for entry in all_entries)
        for field in ("agent", "persona", "domain", "register", "length_bucket")
    }
    kind_counts = Counter(str(entry.get("kind")) for entry in all_entries)
    schedule_kind_counts = Counter(
        str(entry.get("schedule_kind")) for entry in by_kind["request"] if entry.get("schedule_kind")
    )
    trap_counts = Counter(str(entry.get("trap", "none")) for entry in by_kind["filler"])

    if enforce_distribution and len(all_entries) < DISTRIBUTION_FLOOR:
        errors.append(
            f"distribution: {len(all_entries)} bank entries is below the {DISTRIBUTION_FLOOR}-entry "
            "floor for distribution gates; author more source or build with "
            "--allow-small-corpus and accept that the gates did not run"
        )
    elif enforce_distribution and all_entries:
        errors.extend(
            _distribution_gates(
                all_entries,
                by_kind=by_kind,
                counters=counters,
                schedule_kind_counts=schedule_kind_counts,
                trap_counts=trap_counts,
            )
        )

    return {
        "passed": not errors,
        "distribution_enforced": bool(enforce_distribution and len(all_entries) >= DISTRIBUTION_FLOOR),
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "batches": len(batch_items),
            "entries": len(all_entries),
            "requests": len(by_kind["request"]),
            "cancellations": len(by_kind["cancellation"]),
            "fillers": len(by_kind["filler"]),
            "kinds": dict(sorted(kind_counts.items())),
            "schedule_kinds": dict(sorted(schedule_kind_counts.items())),
            "filler_traps": dict(sorted(trap_counts.items())),
            "openers_by_file": {label: dict(sorted(counter.items())) for label, counter in sorted(openers_by_file.items())},
            "corpus_openers": dict(sorted(corpus_openers.items())),
            **{field: dict(sorted(counts.items())) for field, counts in counters.items()},
        },
    }


def _length_bucket(characters: int) -> str:
    if characters < 40:
        return "short"
    if characters < 90:
        return "medium"
    return "long"


def _narrative_opener_gate(label: str, openers: Counter[str]) -> tuple[list[str], list[str]]:
    """Soft per-file caps on the first word of authored phrasings.

    Mirrors ``g1_authored_demo3._narrative_opener_gate``: single-opener caps
    are a real per-file quality bar (warn/error); the top-four cap is only a
    canary here because each author's persona/register is deliberately
    distinct, so some per-file concentration is expected. The build-blocking
    top-four check runs once, across the merged corpus.
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
                f"{label}: opener {opener!r} is {share:.0%} of the batch "
                f"(>{NARRATIVE_OPENER_ERROR_SHARE:.0%}); vary the phrasing"
            )
        elif share > NARRATIVE_OPENER_WARN_SHARE:
            warnings.append(
                f"{label}: opener {opener!r} is {share:.0%} of the batch "
                f"(>{NARRATIVE_OPENER_WARN_SHARE:.0%}); vary the phrasing"
            )
    top_four = sum(count for _, count in openers.most_common(4))
    top_four_share = _share(top_four, total)
    if top_four_share > NARRATIVE_OPENER_TOP4_ERROR_SHARE:
        warnings.append(
            f"{label}: the top four openers cover {top_four_share:.0%} of the batch "
            f"(>{NARRATIVE_OPENER_TOP4_ERROR_SHARE:.0%}); per-file canary only, not a build blocker"
        )
    return warnings, errors


def _narrative_opener_corpus_gate(openers: Counter[str]) -> list[str]:
    errors: list[str] = []
    total = sum(openers.values())
    if total < NARRATIVE_OPENER_CORPUS_FLOOR:
        return errors
    top_four = sum(count for _, count in openers.most_common(4))
    top_four_share = _share(top_four, total)
    if top_four_share > NARRATIVE_OPENER_TOP4_ERROR_SHARE:
        errors.append(
            f"corpus: the top four openers cover {top_four_share:.0%} of the merged corpus "
            f"(>{NARRATIVE_OPENER_TOP4_ERROR_SHARE:.0%}); vary the phrasings across authors"
        )
    return errors


def _distribution_gates(
    entries: list[dict[str, Any]],
    *,
    by_kind: dict[str, list[dict[str, Any]]],
    counters: dict[str, Counter[str]],
    schedule_kind_counts: Counter[str],
    trap_counts: Counter[str],
) -> list[str]:
    errors: list[str] = []
    total = len(entries)

    agent_counts = counters["agent"]
    if not 3 <= len(agent_counts) <= 4:
        errors.append("distribution: each demo requires 3-4 authors")
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

    length_counts = counters["length_bucket"]
    missing = {"short", "medium", "long"} - set(length_counts)
    if missing:
        errors.append(f"distribution: length_bucket missing {sorted(missing)}")
    for name, count in length_counts.items():
        if _share(count, total) > 0.60:
            errors.append(f"distribution: length_bucket {name!r} exceeds 60%")

    if not by_kind["request"]:
        errors.append("distribution: at least one request phrasing is required")
    if not by_kind["cancellation"]:
        errors.append("distribution: at least one cancellation phrasing is required")
    if not by_kind["filler"]:
        errors.append("distribution: at least one filler passage is required")
    for kind in ("once", "every"):
        if not schedule_kind_counts.get(kind):
            errors.append(f"distribution: Demo 5 requires at least one {kind!r} request phrasing")
    for trap in ("bait", "address"):
        if not trap_counts.get(trap):
            errors.append(f"distribution: Demo 5 requires at least one {trap!r} filler passage")

    return errors
