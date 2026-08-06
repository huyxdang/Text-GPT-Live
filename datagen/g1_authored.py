"""Validation and distribution reporting for agent-authored g1 situations."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


AUTHORED_SCHEMA_VERSION = "g1-authored-1"
RESERVED_PERSONAS = {
    "product-reviewer",
    "letter-to-a-friend",
    "technical-writeup",
}
RESERVED_DOMAINS = {"sport", "health", "personal-finance"}
DEMO1_TRAPS = {
    "quoted_question",
    "reported_speech",
    "rhetorical_question",
    "descriptive_command_word",
    "elaborating_question",
    "none",
}
PAUSE_VALUES = {"none", "short", "medium", "long"}
COMMAND_WORD_RE = re.compile(
    r"\b(?:respond|suggest|edit|correct|highlight|delegate|search|remind|translate|wait)\b",
    re.IGNORECASE,
)


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
    if characters < 500:
        return "short"
    if characters < 900:
        return "medium"
    return "long"


def _event_count_bucket(events: int) -> str:
    if events <= 6:
        return "short"
    if events == 7:
        return "medium"
    return "long"


def _trigger_position_bucket(position: int, total: int) -> str:
    ratio = position / total
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


def authored_paths(root: Path, demo: str) -> list[Path]:
    return sorted((root / demo.replace("-", "")).glob("*.json"))


def validate_demo1_batches(
    batches: Iterable[dict[str, Any]],
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

    for batch_index, batch in enumerate(batch_items):
        label = str(batch.get("_path", f"batch-{batch_index}"))
        if batch.get("schema_version") != AUTHORED_SCHEMA_VERSION:
            errors.append(f"{label}: schema_version must be {AUTHORED_SCHEMA_VERSION!r}")
        if batch.get("demo") != "demo-1":
            errors.append(f"{label}: demo must be 'demo-1'")
        author = batch.get("author")
        if not isinstance(author, dict):
            errors.append(f"{label}: author must be an object")
            continue
        if not all(isinstance(author.get(key), str) and author[key].strip() for key in ("model", "slot")):
            errors.append(f"{label}: author requires non-empty model and slot")
        if not isinstance(author.get("tranche"), int) or isinstance(author.get("tranche"), bool):
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

            segments = record.get("segments")
            if not isinstance(segments, list):
                errors.append(f"{location}: segments must be an array")
                continue
            narration_count = 0
            address_positions: list[int] = []
            for segment_index, segment in enumerate(segments, start=1):
                segment_location = f"{location}:segments[{segment_index - 1}]"
                if not isinstance(segment, dict):
                    errors.append(f"{segment_location}: segment must be an object")
                    continue
                kind = segment.get("kind")
                text = segment.get("text")
                if kind not in {"narration", "address"}:
                    errors.append(f"{segment_location}: invalid kind {kind!r}")
                if not isinstance(text, str) or not text.strip():
                    errors.append(f"{segment_location}: text must be non-empty")
                else:
                    normalized = _normalized(text)
                    earlier = seen_text.get(normalized)
                    if earlier is not None:
                        errors.append(
                            f"{segment_location}: exact text duplicates {earlier}"
                        )
                    else:
                        seen_text[normalized] = segment_location
                    prefix = _word_prefix(text, 7)
                    if len(prefix.split()) == 7:
                        text_prefixes.setdefault(prefix, []).append(segment_location)
                if segment.get("pause_after") not in PAUSE_VALUES:
                    errors.append(f"{segment_location}: invalid pause_after")
                if kind == "narration":
                    narration_count += 1
                    traps = segment.get("traps")
                    if not isinstance(traps, list) or any(
                        not isinstance(trap, str) or trap not in DEMO1_TRAPS
                        for trap in traps
                    ):
                        errors.append(f"{segment_location}: invalid traps")
                    elif "none" in traps and len(traps) != 1:
                        errors.append(
                            f"{segment_location}: 'none' cannot accompany another trap"
                        )
                    elif isinstance(text, str):
                        quoted_question = bool(
                            re.search(r"[\"“][^\"”]*\?[^\"”]*[\"”]", text)
                        )
                        if "quoted_question" in traps and not quoted_question:
                            errors.append(
                                f"{segment_location}: quoted_question lacks a quoted question"
                            )
                        if quoted_question and "quoted_question" not in traps:
                            errors.append(
                                f"{segment_location}: quoted question is missing its trap label"
                            )
                        for question_trap in (
                            "rhetorical_question",
                            "elaborating_question",
                        ):
                            if question_trap in traps and "?" not in text:
                                errors.append(
                                    f"{segment_location}: {question_trap} lacks a question"
                                )
                        if (
                            "descriptive_command_word" in traps
                            and not COMMAND_WORD_RE.search(text)
                        ):
                            errors.append(
                                f"{segment_location}: descriptive_command_word lacks a command word"
                            )
                        if (
                            COMMAND_WORD_RE.search(text)
                            and "descriptive_command_word" not in traps
                        ):
                            errors.append(
                                f"{segment_location}: command word is missing its trap label"
                            )
                        if (
                            "?" in text
                            and not {
                                "quoted_question",
                                "rhetorical_question",
                                "elaborating_question",
                            }.intersection(traps)
                        ):
                            errors.append(
                                f"{segment_location}: narration question is missing a question trap label"
                            )
                        if (
                            re.search(
                                r"\b(?:said|told|explained|asked|reported|mentioned|"
                                r"suggested|confirmed|requested|proposed|recommended|"
                                r"warned|noted|replied|claimed|announced)\b",
                                text,
                                re.IGNORECASE,
                            )
                            and "reported_speech" not in traps
                            and "quoted_question" not in traps
                        ):
                            warnings.append(
                                f"{segment_location}: possible unlabelled reported speech"
                            )
                elif kind == "address":
                    address_positions.append(segment_index)
                    if isinstance(text, str):
                        record["address_form"] = "question" if "?" in text else "other"
                    reply = segment.get("gold_reply")
                    if not isinstance(reply, str) or not reply.strip():
                        errors.append(f"{segment_location}: gold_reply must be non-empty")

            if not 5 <= len(segments) <= 9:
                errors.append(f"{location}: segment count must be 5-9")
            if not 4 <= narration_count <= 8:
                errors.append(f"{location}: narration count must be 4-8")
            if len(address_positions) != 1:
                errors.append(f"{location}: exactly one address segment is required")
            if len(address_positions) == 1:
                record["trigger_position"] = _trigger_position_bucket(
                    address_positions[0], len(segments)
                )
            record["event_count_bucket"] = _event_count_bucket(len(segments))
            record["length_bucket"] = _length_bucket(
                sum(len(str(segment.get("text", ""))) for segment in segments if isinstance(segment, dict))
            )
            record["narration_count"] = narration_count
            record["address_count"] = len(address_positions)
            records.append(record)
            batch_records.append(record)

        if len(batch_records) >= 20:
            address_openers: Counter[str] = Counter()
            address_first_words: Counter[str] = Counter()
            reply_openers: Counter[str] = Counter()
            question_addresses = 0
            trap_signatures: Counter[tuple[str, ...]] = Counter()
            per_record_traps: list[set[str]] = []
            for record in batch_records:
                record_traps: set[str] = set()
                for segment in record.get("segments", []):
                    if not isinstance(segment, dict):
                        continue
                    if segment.get("kind") == "address":
                        address_text = str(segment.get("text", ""))
                        first_word = _word_prefix(address_text, 1)
                        if first_word:
                            address_first_words[first_word] += 1
                        opener = _word_prefix(address_text, 2)
                        if opener:
                            address_openers[opener] += 1
                        if "?" in address_text:
                            question_addresses += 1
                        reply_opener = _word_prefix(
                            str(segment.get("gold_reply", "")), 1
                        )
                        if reply_opener:
                            reply_openers[reply_opener] += 1
                    elif segment.get("kind") == "narration":
                        record_traps.update(
                            trap
                            for trap in segment.get("traps", [])
                            if trap != "none"
                        )
                per_record_traps.append(record_traps)
                trap_signatures[tuple(sorted(record_traps))] += 1

            batch_total = len(batch_records)
            for opener, count in address_first_words.items():
                if _share(count, batch_total) > 0.25:
                    errors.append(
                        f"{label}: address first word {opener!r} exceeds 25%; "
                        "vary speech acts"
                    )
            for opener, count in address_openers.items():
                if _share(count, batch_total) > 0.40:
                    errors.append(
                        f"{label}: address opener {opener!r} exceeds 40%; vary speech acts"
                    )
            if _share(question_addresses, batch_total) > 0.80:
                errors.append(
                    f"{label}: question-form addresses exceed 80%; vary speech acts"
                )
            for opener, count in reply_openers.items():
                if _share(count, batch_total) > 0.25:
                    errors.append(
                        f"{label}: reply opener {opener!r} exceeds 25%; vary reply forms"
                    )
            trapless = sum(not traps for traps in per_record_traps)
            if _share(trapless, batch_total) < 0.20:
                errors.append(
                    f"{label}: fewer than 20% of records are trap-free"
                )
            per_trap = Counter(
                trap for record_traps in per_record_traps for trap in record_traps
            )
            for trap, count in per_trap.items():
                if _share(count, batch_total) > 0.50:
                    errors.append(
                        f"{label}: trap {trap!r} appears in over 50% of records"
                    )
            for signature, count in trap_signatures.items():
                if _share(count, batch_total) > 0.50:
                    errors.append(
                        f"{label}: trap signature {signature!r} exceeds 50%; vary scene structure"
                    )
            trigger_positions = Counter(
                str(record.get("trigger_position")) for record in batch_records
            )
            for position, count in trigger_positions.items():
                if _share(count, batch_total) > 0.60:
                    errors.append(
                        f"{label}: trigger position {position!r} exceeds 60%"
                    )
            triggers_by_event_count: dict[str, Counter[str]] = {}
            for record in batch_records:
                event_bucket = str(record.get("event_count_bucket"))
                trigger_bucket = str(record.get("trigger_position"))
                triggers_by_event_count.setdefault(event_bucket, Counter())[
                    trigger_bucket
                ] += 1
            for event_bucket, trigger_counts in triggers_by_event_count.items():
                bucket_total = sum(trigger_counts.values())
                dominant_trigger, dominant_count = trigger_counts.most_common(1)[0]
                if bucket_total >= 5 and _share(dominant_count, bucket_total) > 0.80:
                    errors.append(
                        f"{label}: {dominant_count}/{bucket_total} {event_bucket!r} "
                        f"event-count records use trigger {dominant_trigger!r}; "
                        "cap joint-distribution dominance at 80%"
                    )
            triggers_by_address_form: dict[str, Counter[str]] = {}
            for record in batch_records:
                address_form = str(record.get("address_form"))
                trigger_bucket = str(record.get("trigger_position"))
                triggers_by_address_form.setdefault(address_form, Counter())[
                    trigger_bucket
                ] += 1
            for address_form, trigger_counts in triggers_by_address_form.items():
                form_total = sum(trigger_counts.values())
                dominant_trigger, dominant_count = trigger_counts.most_common(1)[0]
                if form_total >= 5 and _share(dominant_count, form_total) > 0.80:
                    errors.append(
                        f"{label}: {dominant_count}/{form_total} {address_form!r} "
                        f"addresses use trigger {dominant_trigger!r}; "
                        "cap joint-distribution dominance at 80%"
                    )

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
            "length_bucket",
            "event_count_bucket",
            "trigger_position",
        )
    }
    trap_counts: Counter[str] = Counter()
    address_openers: Counter[str] = Counter()
    reply_openers: Counter[str] = Counter()
    address_forms: Counter[str] = Counter()
    trap_signatures: Counter[str] = Counter()
    joint_distributions: dict[str, Counter[str]] = {
        "event_count_x_trigger": Counter(),
        "length_x_trigger": Counter(),
        "address_form_x_trigger": Counter(),
        "domain_x_length": Counter(),
        "domain_x_address_form": Counter(),
        "trap_x_trigger": Counter(),
    }
    trapless_records = 0
    for record in records:
        record_traps: set[str] = set()
        address_form = "missing"
        for segment in record.get("segments", []):
            if not isinstance(segment, dict):
                continue
            if segment.get("kind") == "narration":
                real_traps = {
                    trap for trap in segment.get("traps", []) if trap != "none"
                }
                trap_counts.update(real_traps)
                record_traps.update(real_traps)
            elif segment.get("kind") == "address":
                opener = _word_prefix(str(segment.get("text", "")), 2)
                if opener:
                    address_openers[opener] += 1
                reply_opener = _word_prefix(str(segment.get("gold_reply", "")), 1)
                if reply_opener:
                    reply_openers[reply_opener] += 1
                address_form = (
                    "question" if "?" in str(segment.get("text", "")) else "other"
                )
                address_forms[address_form] += 1
        if not record_traps:
            trapless_records += 1
        signature = "+".join(sorted(record_traps)) if record_traps else "<none>"
        trap_signatures[signature] += 1
        event_count = str(record.get("event_count_bucket"))
        length = str(record.get("length_bucket"))
        trigger = str(record.get("trigger_position"))
        domain = str(record.get("domain"))
        joint_distributions["event_count_x_trigger"][f"{event_count}|{trigger}"] += 1
        joint_distributions["length_x_trigger"][f"{length}|{trigger}"] += 1
        joint_distributions["address_form_x_trigger"][f"{address_form}|{trigger}"] += 1
        joint_distributions["domain_x_length"][f"{domain}|{length}"] += 1
        joint_distributions["domain_x_address_form"][f"{domain}|{address_form}"] += 1
        for trap in record_traps:
            joint_distributions["trap_x_trigger"][f"{trap}|{trigger}"] += 1

    if enforce_distribution and records:
        total = len(records)
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
        for field in ("length_bucket", "event_count_bucket", "trigger_position"):
            counts = counters[field]
            required = {"short", "medium", "long"} if field != "trigger_position" else {"early", "middle", "late"}
            missing = required - set(counts)
            if missing:
                errors.append(f"distribution: {field} missing {sorted(missing)}")
            for name, count in counts.items():
                if _share(count, total) > 0.60:
                    errors.append(f"distribution: {field} {name!r} exceeds 60%")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "batches": len(batch_items),
            "records": len(records),
            "narration": sum(int(record.get("narration_count", 0)) for record in records),
            "addresses": sum(int(record.get("address_count", 0)) for record in records),
            "trapless_records": trapless_records,
            "traps": dict(sorted(trap_counts.items())),
            "trap_signatures": dict(sorted(trap_signatures.items())),
            "address_openers": dict(sorted(address_openers.items())),
            "address_forms": dict(sorted(address_forms.items())),
            "reply_openers": dict(sorted(reply_openers.items())),
            "joint_distributions": {
                name: dict(sorted(counts.items()))
                for name, counts in joint_distributions.items()
            },
            **{
                field: dict(sorted(counts.items()))
                for field, counts in counters.items()
            },
        },
    }
