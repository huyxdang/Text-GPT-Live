"""Validation and distribution reporting for agent-authored Demo 3 situations.

Demo 3 is live English -> Chinese translation.  Two things make its authored
source harder to police than Demo 1's:

1. **The clause-commit boundary is the entire lesson.**  Authors state each
   boundary twice -- positionally, as a ``CLAUSE_MARKER`` in the script text,
   and textually, as the clause's ``english`` span.  ``plan_demo3_script``
   refuses to compile when the two disagree, so a mis-placed boundary is a
   build failure rather than a subtly wrong card.

2. **Nobody on this project reads Chinese.**  The decision recorded 2026-08-01
   is that Chinese references get *LLM* review, not human review.  This module
   enforces that the claim is recorded honestly (``method`` must be
   ``"machine"``; claiming ``"human"`` is rejected) and adds every mechanical
   check on the Chinese that does not require reading it: Han content,
   no untranslated Latin runs, no half-width punctuation, no action markup,
   no reference reused across clauses, plausible length ratio.

The Demo 1 review found that mechanical properties were gated while authored
*content* was not.  Section "Chinese reference checks" below is the answer to
that: as much correctness as can be pushed into code, with the residue named
plainly rather than implied to be verified.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datagen.g1_demo3 import (
    CLAUSE_MARKER,
    MIN_INSTRUCTION_CHARS,
    closing_shape,
    opening_shape,
    plan_demo3_script,
)


AUTHORED_SCHEMA_VERSION = "g1-authored-demo3-1"
DEMO = "demo-3"

RESERVED_PERSONAS = {
    "product-reviewer",
    "letter-to-a-friend",
    "technical-writeup",
}
RESERVED_DOMAINS = {"sport", "health", "personal-finance"}

#: The only reference-review method this project can honestly claim.  "human"
#: is rejected on purpose: the owner does not read Chinese, so a human-review
#: claim on this corpus would be false provenance in the manifest.
ACCEPTED_REVIEW_METHODS = {"machine"}
KNOWN_REVIEW_METHODS = {"machine", "human", "none"}

#: Markup that must never appear in authored text of any kind.
MARKUP_RE = re.compile(
    r"(?:<action>|</action>|<stream_event|<PREDICT_THIS_ACTION>|"
    r"\bidle\(\)|\brespond\(\{|\bsuggest_edit\(|\bhighlight\(|\bdelegate\()"
)
HAN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
LATIN_RUN_RE = re.compile(r"[A-Za-z]{4,}")
#: Half-width punctuation inside Chinese prose is a genuine quality defect.
#: Digits are exempt so "3.14" and "1,200" stay legal.
HALFWIDTH_PUNCT_RE = re.compile(r"(?<![0-9])[,.;:?!](?![0-9])")

#: Author share, persona breadth, bucket coverage and skeleton spread are all
#: scale-dependent: they cannot hold on a handful of records and asserting them
#: there would only teach people to disable the gate.  Enforcement therefore
#: requires a corpus of at least this size, and a smaller corpus must say so
#: out loud rather than quietly pass.
DISTRIBUTION_FLOOR = 24

MAX_LATIN_ALLOWLIST = 8
LENGTH_RATIO_MIN = 0.12
LENGTH_RATIO_MAX = 1.30
LENGTH_RATIO_MIN_ENGLISH_CHARS = 20

#: Below this many clauses in a batch, opener-share percentages are noise.
#: The gate does not run at all below the floor.
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
#: The corpus-level top-four check needs enough merged clauses that a share
#: is meaningful; below this the check does not run at all (same rationale as
#: NARRATIVE_OPENER_FLOOR, just at corpus scale).
NARRATIVE_OPENER_CORPUS_FLOOR = 60
#: Clause continuations legitimately open with a coordinating conjunction
#: ("And then she..."); stripping it before taking the first word keeps the
#: gate watching the real narrative distribution instead of "and"/"but".
COORDINATING_CONJUNCTIONS = frozenset({"and", "but", "so", "then", "or"})


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _word_prefix(value: str, words: int) -> str:
    normalized = re.sub(r"[^a-z0-9']+", " ", value.lower())
    return " ".join(normalized.split()[:words])


def _clause_opener(value: str) -> str:
    """First word of a clause's English text, for the narrative opener gate.

    A leading coordinating conjunction is stripped first: clause
    continuations legitimately start with "and"/"but"/"so"/"then"/"or", and
    counting those would mask the real opening-word distribution.
    """

    words = re.sub(r"[^a-z0-9']+", " ", value.lower()).split()
    if words and words[0] in COORDINATING_CONJUNCTIONS:
        words = words[1:]
    return words[0] if words else ""


def _han_prefix(value: str, characters: int) -> str:
    """Leading Han characters of a reference; Chinese has no word breaks."""

    han = "".join(HAN_RE.findall(value))
    return han[:characters] if len(han) >= characters else ""


def _opening_key(value: str, *, words: int, han_characters: int) -> str:
    """A comparable opening for text that may be English, Chinese, or both.

    Demo 3 acknowledgments are routinely bilingual ("好的 — I'll translate as you
    type."), so an ASCII-only word prefix silently reports nothing and the
    canned-phrase cap stops biting.  Chinese openings are keyed by leading Han
    characters instead, since Chinese has no word breaks.
    """

    text = value.strip()
    if text and _is_han_char(text[0]):
        han = _han_prefix(text, han_characters)
        return f"han:{han}" if han else ""
    prefix = _word_prefix(text, words)
    return f"en:{prefix}" if prefix else ""


def _is_han_char(character: str) -> bool:
    return bool(HAN_RE.match(character))


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def _length_bucket(characters: int) -> str:
    if characters < 220:
        return "short"
    if characters < 460:
        return "medium"
    return "long"


def _clause_count_bucket(clauses: int) -> str:
    if clauses <= 3:
        return "short"
    if clauses <= 6:
        return "medium"
    return "long"


def _trigger_position_bucket(position: int, total: int) -> str:
    ratio = position / total if total else 0.0
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


def authored_paths(root: Path, demo: str = DEMO) -> list[Path]:
    return sorted((root / demo.replace("-", "")).glob("*.json"))


# --------------------------------------------------------------------------
# Chinese reference checks -- everything verifiable without reading Chinese
# --------------------------------------------------------------------------


def check_chinese_reference(
    reference: str,
    english: str,
    *,
    location: str,
    latin_allowlist: Iterable[str] = (),
) -> list[str]:
    """Deterministic quality checks on one Chinese clause reference.

    These do not judge meaning -- meaning is what the LLM reviewer is for.
    They catch the failures that do not need a reader: empty or untranslated
    output, English echoed back, markup leakage, half-width punctuation, and
    implausible length.
    """

    errors: list[str] = []
    if reference != reference.strip():
        errors.append(f"{location}: reference must not have leading/trailing whitespace")
    stripped = reference.strip()
    if not stripped:
        errors.append(f"{location}: reference must be non-empty")
        return errors
    if not HAN_RE.search(stripped):
        errors.append(f"{location}: reference contains no Han characters")
    if stripped == english.strip():
        errors.append(f"{location}: reference is byte-identical to its English source")
    if CLAUSE_MARKER in reference:
        errors.append(f"{location}: reference must not contain the clause marker")
    if MARKUP_RE.search(reference):
        errors.append(f"{location}: reference contains action or stream markup")
    for character in "<>&":
        if character in reference:
            errors.append(
                f"{location}: reference must not contain {character!r}; it renders "
                "differently in history than in the graded completion"
            )
    allow = {item.lower() for item in latin_allowlist}
    for run in LATIN_RUN_RE.findall(stripped):
        if run.lower() not in allow:
            errors.append(
                f"{location}: reference contains the Latin run {run!r}, which suggests "
                "untranslated English; declare it in latin_allowlist if intentional"
            )
    for match in HALFWIDTH_PUNCT_RE.finditer(stripped):
        errors.append(
            f"{location}: reference uses half-width punctuation {match.group()!r}; "
            "Chinese prose takes full-width punctuation"
        )
        break
    if len(english.strip()) >= LENGTH_RATIO_MIN_ENGLISH_CHARS:
        ratio = len(stripped) / len(english.strip())
        if not LENGTH_RATIO_MIN <= ratio <= LENGTH_RATIO_MAX:
            errors.append(
                f"{location}: reference/English length ratio {ratio:.2f} is outside "
                f"[{LENGTH_RATIO_MIN}, {LENGTH_RATIO_MAX}]; the clause looks truncated "
                "or padded"
            )
    return errors


# --------------------------------------------------------------------------
# batch validation
# --------------------------------------------------------------------------


def _validate_review(batch: Any, label: str, errors: list[str]) -> dict[str, Any]:
    review = batch.get("reference_review")
    if not isinstance(review, dict):
        errors.append(
            f"{label}: reference_review object is required; Chinese provenance is not optional"
        )
        return {}
    method = review.get("method")
    if method not in KNOWN_REVIEW_METHODS:
        errors.append(
            f"{label}: reference_review.method must be one of {sorted(KNOWN_REVIEW_METHODS)}"
        )
    elif method not in ACCEPTED_REVIEW_METHODS:
        errors.append(
            f"{label}: reference_review.method {method!r} is not accepted for g1. "
            "Chinese references on this project get machine (LLM) review only; "
            "recording human verification would be false provenance"
        )
    for field in ("reviewer", "reviewed_at"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            errors.append(f"{label}: reference_review.{field} must be a non-empty string")
    return review


def validate_demo3_batches(
    batches: Iterable[dict[str, Any]],
    *,
    enforce_distribution: bool = False,
) -> dict[str, Any]:
    """Validate authored Demo 3 batches and report every distribution."""

    batch_items = list(batches)
    errors: list[str] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    seen_english: dict[str, str] = {}
    seen_reference: dict[str, str] = {}
    seen_instruction: dict[str, str] = {}
    english_prefixes: dict[str, list[str]] = {}
    reference_prefixes: dict[str, list[str]] = {}
    clause_openers_by_file: dict[str, Counter[str]] = {}

    for batch_index, batch in enumerate(batch_items):
        label = str(batch.get("_path", f"batch-{batch_index}"))
        if batch.get("schema_version") != AUTHORED_SCHEMA_VERSION:
            errors.append(f"{label}: schema_version must be {AUTHORED_SCHEMA_VERSION!r}")
        if batch.get("demo") != DEMO:
            errors.append(f"{label}: demo must be {DEMO!r}")
        review = _validate_review(batch, label, errors)
        author = batch.get("author")
        if not isinstance(author, dict):
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
            if not isinstance(raw_record, dict):
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
            record["reference_review_method"] = review.get("method")

            for field in ("persona", "domain", "register"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    errors.append(f"{location}: {field} must be a non-empty string")
            if (
                isinstance(record.get("persona"), str)
                and _slug(record["persona"]) in RESERVED_PERSONAS
            ):
                errors.append(f"{location}: reserved persona leaked into training")
            if (
                isinstance(record.get("domain"), str)
                and _slug(record["domain"]) in RESERVED_DOMAINS
            ):
                errors.append(f"{location}: reserved domain leaked into training")

            latin_allowlist = record.get("latin_allowlist", [])
            if not isinstance(latin_allowlist, list) or any(
                not isinstance(item, str) or not item.strip() for item in latin_allowlist
            ):
                errors.append(f"{location}: latin_allowlist must be an array of strings")
                latin_allowlist = []
            elif len(latin_allowlist) > MAX_LATIN_ALLOWLIST:
                errors.append(
                    f"{location}: latin_allowlist may name at most "
                    f"{MAX_LATIN_ALLOWLIST} terms"
                )

            instruction = record.get("instruction")
            if not isinstance(instruction, dict):
                errors.append(f"{location}: instruction must be an object")
                continue
            instruction_text = instruction.get("text")
            gold_ack = instruction.get("gold_ack")
            if not isinstance(instruction_text, str) or not instruction_text.strip():
                errors.append(f"{location}: instruction.text must be non-empty")
                instruction_text = ""
            elif len(instruction_text) < MIN_INSTRUCTION_CHARS:
                errors.append(
                    f"{location}: instruction.text must be at least "
                    f"{MIN_INSTRUCTION_CHARS} characters"
                )
            if not isinstance(gold_ack, str) or not gold_ack.strip():
                errors.append(f"{location}: instruction.gold_ack must be non-empty")
                gold_ack = ""
            for field, text in (("text", instruction_text), ("gold_ack", gold_ack)):
                if text and MARKUP_RE.search(text):
                    errors.append(f"{location}: instruction.{field} contains action markup")
                if text and CLAUSE_MARKER in text:
                    errors.append(
                        f"{location}: instruction.{field} must not carry a clause marker"
                    )
                if text != text.strip():
                    errors.append(
                        f"{location}: instruction.{field} must not have surrounding whitespace"
                    )
            if instruction_text:
                normalized = _normalized(instruction_text)
                earlier = seen_instruction.get(normalized)
                if earlier is not None:
                    errors.append(
                        f"{location}: instruction text duplicates {earlier}; "
                        "vary the standing-instruction phrasing"
                    )
                else:
                    seen_instruction[normalized] = location

            try:
                plan = plan_demo3_script(record)
            except ValueError as exc:
                errors.append(f"{location}: {exc}")
                continue

            clauses = record.get("clauses")
            clause_openers: list[str] = []
            for clause_index, clause in enumerate(clauses):
                clause_location = f"{location}:clauses[{clause_index}]"
                english = str(clause.get("english", ""))
                reference = str(clause.get("reference", ""))
                opener = _clause_opener(english)
                if opener:
                    clause_openers.append(opener)
                if MARKUP_RE.search(english):
                    errors.append(f"{clause_location}: english contains action markup")
                if HAN_RE.search(english):
                    errors.append(
                        f"{clause_location}: a committable clause must be English; "
                        "text that already quotes Chinese belongs on a "
                        "prequoted_chinese step"
                    )
                errors.extend(
                    check_chinese_reference(
                        reference,
                        english,
                        location=clause_location,
                        latin_allowlist=latin_allowlist,
                    )
                )
                normalized_english = _normalized(english)
                earlier = seen_english.get(normalized_english)
                if earlier is not None:
                    errors.append(
                        f"{clause_location}: English clause duplicates {earlier}"
                    )
                else:
                    seen_english[normalized_english] = clause_location
                earlier = seen_reference.get(reference)
                if earlier is not None:
                    errors.append(
                        f"{clause_location}: Chinese reference is byte-identical to "
                        f"{earlier}"
                    )
                else:
                    seen_reference[reference] = clause_location
                prefix = _word_prefix(english, 7)
                if len(prefix.split()) == 7:
                    english_prefixes.setdefault(prefix, []).append(clause_location)
                han_prefix = _han_prefix(reference, 6)
                if han_prefix:
                    reference_prefixes.setdefault(han_prefix, []).append(clause_location)

            step_traps = [step.trap for step in plan.steps]
            record_traps = {trap for trap in step_traps if trap != "none"}
            if not 2 <= len(plan.clause_spans) <= 12:
                errors.append(
                    f"{location}: a record must commit 2-12 clauses; "
                    f"found {len(plan.clause_spans)}"
                )
            if not any(step.pause_after != "none" for step in plan.steps):
                errors.append(
                    f"{location}: at least one script step must carry a real pause; "
                    "a uniform-cadence episode teaches that the clock is furniture"
                )

            passage_chars = len(plan.final_passage)
            first_commit = (
                plan.clause_spans[0].end_offset if plan.clause_spans else 0
            )
            record["clause_count"] = len(plan.clause_spans)
            record["step_count"] = len(plan.steps)
            record["passage_chars"] = passage_chars
            record["length_bucket"] = _length_bucket(passage_chars)
            record["clause_count_bucket"] = _clause_count_bucket(len(plan.clause_spans))
            record["trigger_position"] = _trigger_position_bucket(
                first_commit, passage_chars
            )
            record["traps"] = sorted(record_traps)
            record["opening_shape"] = opening_shape(record_id)
            record["closing_shape"] = closing_shape(record_id)
            record["instruction_text"] = instruction_text
            record["gold_ack"] = gold_ack
            record["clause_openers"] = clause_openers
            records.append(record)
            batch_records.append(record)

        if len(batch_records) >= 20:
            errors.extend(_batch_wording_gates(label, batch_records))

        batch_clause_openers: Counter[str] = Counter()
        for record in batch_records:
            batch_clause_openers.update(record.get("clause_openers") or [])
        clause_openers_by_file[label] = batch_clause_openers
        opener_warnings, opener_errors = _narrative_opener_gate(
            label, batch_clause_openers, "clause"
        )
        warnings.extend(opener_warnings)
        errors.extend(opener_errors)

    for prefix, locations in english_prefixes.items():
        if len(locations) >= 3:
            errors.append(
                f"template: English 7-word prefix {prefix!r} repeats {len(locations)} "
                f"times ({', '.join(locations[:3])})"
            )
    for prefix, locations in reference_prefixes.items():
        if len(locations) >= 4:
            errors.append(
                f"template: Chinese 6-character opening {prefix!r} repeats "
                f"{len(locations)} times ({', '.join(locations[:3])})"
            )

    counters = {
        field: Counter(str(record.get(field)) for record in records)
        for field in (
            "agent",
            "persona",
            "domain",
            "register",
            "length_bucket",
            "clause_count_bucket",
            "trigger_position",
            "opening_shape",
            "closing_shape",
            "reference_review_method",
        )
    }
    trap_counts: Counter[str] = Counter()
    trap_signatures: Counter[str] = Counter()
    ack_openers: Counter[str] = Counter()
    instruction_openers: Counter[str] = Counter()
    clause_openers: Counter[str] = Counter()
    trapless_records = 0
    joint_distributions: dict[str, Counter[str]] = {
        "clause_count_x_trigger": Counter(),
        "length_x_trigger": Counter(),
        "domain_x_length": Counter(),
        "persona_x_clause_count": Counter(),
        "trap_x_trigger": Counter(),
        "agent_x_domain": Counter(),
    }
    for record in records:
        record_traps = set(record.get("traps", []))
        trap_counts.update(record_traps)
        if not record_traps:
            trapless_records += 1
        signature = "+".join(sorted(record_traps)) if record_traps else "<none>"
        trap_signatures[signature] += 1
        ack_opener = _opening_key(str(record.get("gold_ack", "")), words=2, han_characters=3)
        if ack_opener:
            ack_openers[ack_opener] += 1
        instruction_opener = _word_prefix(str(record.get("instruction_text", "")), 2)
        if instruction_opener:
            instruction_openers[instruction_opener] += 1
        clause_openers.update(record.get("clause_openers") or [])
        clause_bucket = str(record.get("clause_count_bucket"))
        length = str(record.get("length_bucket"))
        trigger = str(record.get("trigger_position"))
        domain = str(record.get("domain"))
        persona = str(record.get("persona"))
        agent = str(record.get("agent"))
        joint_distributions["clause_count_x_trigger"][f"{clause_bucket}|{trigger}"] += 1
        joint_distributions["length_x_trigger"][f"{length}|{trigger}"] += 1
        joint_distributions["domain_x_length"][f"{domain}|{length}"] += 1
        joint_distributions["persona_x_clause_count"][f"{persona}|{clause_bucket}"] += 1
        joint_distributions["agent_x_domain"][f"{agent}|{domain}"] += 1
        for trap in record_traps:
            joint_distributions["trap_x_trigger"][f"{trap}|{trigger}"] += 1

    errors.extend(_narrative_opener_corpus_gate(clause_openers, "clause"))

    if enforce_distribution and len(records) < DISTRIBUTION_FLOOR:
        errors.append(
            f"distribution: {len(records)} records is below the {DISTRIBUTION_FLOOR}-record "
            "floor for distribution gates; author more source or build with "
            "--allow-small-corpus and accept that the gates did not run"
        )
    elif enforce_distribution and records:
        errors.extend(
            _distribution_gates(
                records,
                counters=counters,
                trap_counts=trap_counts,
                trap_signatures=trap_signatures,
                trapless_records=trapless_records,
                joint_distributions=joint_distributions,
                ack_openers=ack_openers,
            )
        )

    total_clauses = sum(int(record.get("clause_count", 0)) for record in records)
    return {
        "passed": not errors,
        "distribution_enforced": bool(
            enforce_distribution and len(records) >= DISTRIBUTION_FLOOR
        ),
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "batches": len(batch_items),
            "records": len(records),
            "clauses": total_clauses,
            "steps": sum(int(record.get("step_count", 0)) for record in records),
            "trapless_records": trapless_records,
            "traps": dict(sorted(trap_counts.items())),
            "trap_signatures": dict(sorted(trap_signatures.items())),
            "ack_openers": dict(sorted(ack_openers.items())),
            "instruction_openers": dict(sorted(instruction_openers.items())),
            "clause_openers": dict(sorted(clause_openers.items())),
            "clause_openers_by_file": {
                file_label: dict(sorted(counter.items()))
                for file_label, counter in sorted(clause_openers_by_file.items())
            },
            "joint_distributions": {
                name: dict(sorted(counts.items()))
                for name, counts in joint_distributions.items()
            },
            **{field: dict(sorted(counts.items())) for field, counts in counters.items()},
        },
    }


def _batch_wording_gates(label: str, batch_records: list[dict[str, Any]]) -> list[str]:
    """Per-tranche caps so acknowledgment wording cannot collapse to one phrase."""

    errors: list[str] = []
    total = len(batch_records)
    ack_heads: Counter[str] = Counter()
    ack_openers: Counter[str] = Counter()
    instruction_first_words: Counter[str] = Counter()
    for record in batch_records:
        ack = str(record.get("gold_ack", ""))
        head = _opening_key(ack, words=1, han_characters=2)
        if head:
            ack_heads[head] += 1
        opener = _opening_key(ack, words=2, han_characters=3)
        if opener:
            ack_openers[opener] += 1
        instruction_first = _word_prefix(str(record.get("instruction_text", "")), 1)
        if instruction_first:
            instruction_first_words[instruction_first] += 1
    for opener, count in ack_heads.items():
        if _share(count, total) > 0.25:
            errors.append(
                f"{label}: acknowledgment opening {opener!r} exceeds 25%; "
                "vary the ack wording"
            )
    for opener, count in ack_openers.items():
        if _share(count, total) > 0.40:
            errors.append(
                f"{label}: acknowledgment opener {opener!r} exceeds 40%; vary the ack wording"
            )
    for opener, count in instruction_first_words.items():
        if _share(count, total) > 0.35:
            errors.append(
                f"{label}: instruction first word {opener!r} exceeds 35%; "
                "vary how the standing instruction is phrased"
            )
    trap_share: Counter[str] = Counter()
    for record in batch_records:
        trap_share.update(set(record.get("traps", [])))
    for trap, count in trap_share.items():
        if _share(count, total) > 0.70:
            errors.append(f"{label}: trap {trap!r} appears in over 70% of records")
    trapless = sum(1 for record in batch_records if not record.get("traps"))
    if _share(trapless, total) < 0.15:
        errors.append(
            f"{label}: fewer than 15% of records are trap-free; clean straight-through "
            "translation is a real case and must be represented"
        )
    return errors


def _narrative_opener_gate(
    label: str, openers: Counter[str], kind_label: str
) -> tuple[list[str], list[str]]:
    """Soft caps on the first word of clause narrative text within one batch.

    Acknowledgment and instruction openers are already gated (see
    ``_batch_wording_gates``); this closes the same gap on the clauses
    themselves, where nothing previously watched the distribution and it
    quietly converged on "The" in the real authored corpus.  Thresholds are
    deliberately soft — natural English favours "the" as an opener roughly
    10% of the time, and pushing authors away from it produces contrived
    prose, which is worse than a somewhat skewed distribution.

    The single-opener caps (warn/error) are a genuine per-file quality bar:
    a file where 40%+ of clauses open the same way is defective regardless
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
    records: list[dict[str, Any]],
    *,
    counters: dict[str, Counter[str]],
    trap_counts: Counter[str],
    trap_signatures: Counter[str],
    trapless_records: int,
    joint_distributions: dict[str, Counter[str]],
    ack_openers: Counter[str],
) -> list[str]:
    errors: list[str] = []
    total = len(records)

    # The register rule: every standing instruction gets one acknowledgment, and
    # the wording must not collapse into one canned phrase across the corpus.
    for opener, count in ack_openers.items():
        if _share(count, total) > 0.30:
            errors.append(
                f"distribution: acknowledgment opener {opener!r} exceeds 30% of "
                "episodes; vary the ack wording"
            )

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

    for field, required in (
        ("length_bucket", {"short", "medium", "long"}),
        ("clause_count_bucket", {"short", "medium", "long"}),
        ("trigger_position", {"early", "middle", "late"}),
    ):
        counts = counters[field]
        missing = required - set(counts)
        if missing:
            errors.append(f"distribution: {field} missing {sorted(missing)}")
        for name, count in counts.items():
            if _share(count, total) > 0.60:
                errors.append(f"distribution: {field} {name!r} exceeds 60%")

    # All three traps are mandatory for Demo 3; none may dominate.
    for trap in ("half_clause", "prequoted_chinese", "backtrack"):
        if not trap_counts.get(trap):
            errors.append(f"distribution: Demo 3 requires the {trap!r} trap")
    for trap, count in trap_counts.items():
        if _share(count, total) > 0.70:
            errors.append(f"distribution: trap {trap!r} appears in over 70% of records")
    if _share(trapless_records, total) < 0.15:
        errors.append("distribution: fewer than 15% of records are trap-free")
    for signature, count in trap_signatures.items():
        if _share(count, total) > 0.50:
            errors.append(
                f"distribution: trap signature {signature!r} exceeds 50%; "
                "vary scene structure"
            )

    # Skeleton shapes are compiler-derived from the record id.  They cannot be
    # over-concentrated by an author, but an absent shape means a silence class
    # the selector cannot fill, so presence is gated.
    for field, shapes in (
        ("opening_shape", 4),
        ("closing_shape", 4),
    ):
        if len(counters[field]) < shapes:
            errors.append(
                f"distribution: {field} must span all {shapes} skeletons; "
                "the episode structure would otherwise be a constant"
            )

    for name, counts in joint_distributions.items():
        joint_total = sum(counts.values())
        if joint_total < 5 or not counts:
            continue
        dominant, dominant_count = counts.most_common(1)[0]
        if _share(dominant_count, joint_total) > 0.80:
            errors.append(
                f"distribution: {dominant_count}/{joint_total} of {name} sit in "
                f"{dominant!r}; cap joint-distribution dominance at 80%"
            )

    methods = set(counters["reference_review_method"])
    if methods - ACCEPTED_REVIEW_METHODS:
        errors.append(
            "distribution: every record must carry a machine reference-review claim; "
            f"found {sorted(methods)}"
        )
    return errors
