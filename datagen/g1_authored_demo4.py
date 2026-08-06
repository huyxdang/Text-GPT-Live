"""Validation and distribution reporting for agent-authored Demo 4 banks.

Demo 4 authors do not write episodes. They write two flat *banks* of words
(see the module docstring in ``datagen.g1_demo4`` for why): varied
visual-request phrasings (``requests``) and progress-question/reply pairs
(``progress_pairs``, split into ``kind": "check"`` -- an honest not-yet
answer -- and ``kind": "nudge"`` -- a receipt check-in that gets silently
ignored while the job is already running). The generator (``g1_demo4.py``)
is what turns those banks into structurally different job-dialog episodes.

Because a bank item is a phrase, not an episode, this module cannot reuse
Demo 3's clause-boundary machinery. What it does reuse, adapted:

* the reserved persona/domain list and the author/persona/domain/register
  distribution gate, at the same 20-40% / >=5 / >=5 / >=4 / 35% thresholds;
* the narrative-opener soft-cap / hard-cap scheme from
  ``datagen.g1_authored_demo3`` (``_narrative_opener_gate`` /
  ``_narrative_opener_corpus_gate``), applied to request-text and
  check-reply openers so wording cannot collapse to one canned phrase;
* a length-bucket and trigger-bucket gate. There is no in-episode "trigger
  position" for a flat bank item, so this module defines trigger bucket as
  the item's ordinal position within its own authored batch (early/middle/
  late), which is the only position concept that exists at authoring time.
  This is a deliberate reinterpretation of the Demo 3 gate, documented here
  rather than silently reused past its original meaning.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datagen.g1_demo4 import (
    MAX_QUESTION_CHARS,
    MAX_REPLY_CHARS,
    MAX_REQUEST_CHARS,
    MAX_TASK_CHARS,
    MIN_QUESTION_CHARS,
    MIN_REQUEST_CHARS,
)


AUTHORED_SCHEMA_VERSION = "g1-authored-demo4-1"
DEMO = "demo-4"

RESERVED_PERSONAS = {
    "product-reviewer",
    "letter-to-a-friend",
    "technical-writeup",
}
RESERVED_DOMAINS = {"sport", "health", "personal-finance"}

PROGRESS_KINDS = {"check", "nudge"}

MARKUP_RE = re.compile(
    r"(?:<action>|</action>|<stream_event|<PREDICT_THIS_ACTION>|"
    r"\bidle\(\)|\brespond\(\{|\bsuggest_edit\(|\bhighlight\(|\bdelegate\()"
)

DISTRIBUTION_FLOOR = 24
NARRATIVE_OPENER_FLOOR = 20
NARRATIVE_OPENER_WARN_SHARE = 0.25
NARRATIVE_OPENER_ERROR_SHARE = 0.40
NARRATIVE_OPENER_TOP4_ERROR_SHARE = 0.75
NARRATIVE_OPENER_CORPUS_FLOOR = 40


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
    if characters < 25:
        return "short"
    if characters < 45:
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
    import json

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


def _check_forbidden_chars(text: str, location: str, field: str, errors: list[str]) -> None:
    for character in "<>&":
        if character in text:
            errors.append(
                f"{location}: {field} must not contain {character!r}; compile_stream "
                "HTML-escapes event content and this would break exact-span guarantees"
            )
    if MARKUP_RE.search(text):
        errors.append(f"{location}: {field} contains action or stream markup")


def _validate_common_fields(
    item: dict[str, Any], location: str, errors: list[str]
) -> None:
    for field in ("persona", "domain", "register"):
        if not isinstance(item.get(field), str) or not item[field].strip():
            errors.append(f"{location}: {field} must be a non-empty string")
    if isinstance(item.get("persona"), str) and _slug(item["persona"]) in RESERVED_PERSONAS:
        errors.append(f"{location}: reserved persona leaked into training")
    if isinstance(item.get("domain"), str) and _slug(item["domain"]) in RESERVED_DOMAINS:
        errors.append(f"{location}: reserved domain leaked into training")


def validate_demo4_batches(
    batches: Iterable[dict[str, Any]],
    *,
    enforce_distribution: bool = False,
) -> dict[str, Any]:
    """Validate authored Demo 4 banks and report every distribution."""

    batch_items = list(batches)
    errors: list[str] = []
    warnings: list[str] = []
    requests: list[dict[str, Any]] = []
    progress_pairs: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    progress_ids: set[str] = set()
    seen_request_text: dict[str, str] = {}
    seen_question: dict[str, str] = {}
    request_prefixes: dict[str, list[str]] = {}
    request_openers_by_file: dict[str, Counter[str]] = {}
    reply_openers_by_file: dict[str, Counter[str]] = {}

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
        if not all(
            isinstance(author.get(key), str) and author[key].strip()
            for key in ("model", "slot")
        ):
            errors.append(f"{label}: author requires non-empty model and slot")
        if not isinstance(author.get("tranche"), int) or isinstance(author.get("tranche"), bool):
            errors.append(f"{label}: author.tranche must be an integer")

        raw_requests = batch.get("requests")
        raw_progress = batch.get("progress_pairs")
        if not isinstance(raw_requests, list):
            errors.append(f"{label}: requests must be an array")
            raw_requests = []
        if not isinstance(raw_progress, list):
            errors.append(f"{label}: progress_pairs must be an array")
            raw_progress = []

        batch_request_openers: Counter[str] = Counter()
        batch_reply_openers: Counter[str] = Counter()
        batch_requests: list[dict[str, Any]] = []
        batch_progress: list[dict[str, Any]] = []

        for item_index, raw_item in enumerate(raw_requests):
            location = f"{label}:requests[{item_index}]"
            if not isinstance(raw_item, dict):
                errors.append(f"{location}: item must be an object")
                continue
            item = dict(raw_item)
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id.strip():
                errors.append(f"{location}: id must be a non-empty string")
                continue
            if item_id in request_ids:
                errors.append(f"{location}: duplicate request id {item_id!r}")
            request_ids.add(item_id)
            item["agent"] = author.get("slot")
            _validate_common_fields(item, location, errors)

            text = item.get("text")
            task = item.get("task")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{location}: text must be non-empty")
                text = ""
            elif not MIN_REQUEST_CHARS <= len(text) <= MAX_REQUEST_CHARS:
                errors.append(
                    f"{location}: text must be {MIN_REQUEST_CHARS}-{MAX_REQUEST_CHARS} "
                    f"characters; got {len(text)}"
                )
            if not isinstance(task, str) or not task.strip():
                errors.append(f"{location}: task must be non-empty")
                task = ""
            elif len(task) > MAX_TASK_CHARS:
                errors.append(f"{location}: task must be at most {MAX_TASK_CHARS} characters")
            for field, value in (("text", text), ("task", task)):
                if value:
                    _check_forbidden_chars(value, location, field, errors)
                    if value != value.strip():
                        errors.append(f"{location}: {field} must not have surrounding whitespace")

            if text:
                normalized = _normalized(text)
                earlier = seen_request_text.get(normalized)
                if earlier is not None:
                    errors.append(f"{location}: request text duplicates {earlier}")
                else:
                    seen_request_text[normalized] = location
                prefix = _word_prefix(text, 7)
                if len(prefix.split()) == 7:
                    request_prefixes.setdefault(prefix, []).append(location)
                opener = _word_prefix(text, 2)
                if opener:
                    batch_request_openers[opener] += 1

            item["length_bucket"] = _length_bucket(len(text))
            item["record_kind"] = "request"
            requests.append(item)
            batch_requests.append(item)

        for item_index, raw_item in enumerate(raw_progress):
            location = f"{label}:progress_pairs[{item_index}]"
            if not isinstance(raw_item, dict):
                errors.append(f"{location}: item must be an object")
                continue
            item = dict(raw_item)
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id.strip():
                errors.append(f"{location}: id must be a non-empty string")
                continue
            if item_id in progress_ids:
                errors.append(f"{location}: duplicate progress_pairs id {item_id!r}")
            progress_ids.add(item_id)
            item["agent"] = author.get("slot")
            _validate_common_fields(item, location, errors)

            kind = item.get("kind")
            if kind not in PROGRESS_KINDS:
                errors.append(f"{location}: kind must be one of {sorted(PROGRESS_KINDS)}")
                kind = None
            question = item.get("question")
            if not isinstance(question, str) or not question.strip():
                errors.append(f"{location}: question must be non-empty")
                question = ""
            elif not MIN_QUESTION_CHARS <= len(question) <= MAX_QUESTION_CHARS:
                errors.append(
                    f"{location}: question must be {MIN_QUESTION_CHARS}-{MAX_QUESTION_CHARS} "
                    f"characters; got {len(question)}"
                )
            reply = item.get("reply")
            if kind == "check":
                if not isinstance(reply, str) or not reply.strip():
                    errors.append(f"{location}: check items require a non-empty reply")
                    reply = ""
                elif len(reply) > MAX_REPLY_CHARS:
                    errors.append(f"{location}: reply must be at most {MAX_REPLY_CHARS} characters")
            elif kind == "nudge":
                if reply not in (None, ""):
                    errors.append(
                        f"{location}: nudge items must not carry a reply; the graded "
                        "answer to a nudge is always idle(), never authored text"
                    )
                reply = ""
            for field, value in (("question", question), ("reply", reply)):
                if value:
                    _check_forbidden_chars(value, location, field, errors)
                    if value != value.strip():
                        errors.append(f"{location}: {field} must not have surrounding whitespace")

            if question:
                normalized = _normalized(question)
                earlier = seen_question.get(normalized)
                if earlier is not None:
                    errors.append(f"{location}: question duplicates {earlier}")
                else:
                    seen_question[normalized] = location
            if kind == "check" and reply:
                opener = _word_prefix(reply, 2)
                if opener:
                    batch_reply_openers[opener] += 1

            item["length_bucket"] = _length_bucket(len(question))
            item["record_kind"] = f"progress-{kind}" if kind else "progress-invalid"
            progress_pairs.append(item)
            batch_progress.append(item)

        batch_all = batch_requests + batch_progress
        total_batch = len(batch_all)
        for index, item in enumerate(batch_all):
            item["trigger_position"] = _trigger_position_bucket(index, total_batch)

        if len(batch_all) >= 20:
            errors.extend(_batch_wording_gates(label, batch_requests, batch_progress))

        request_openers_by_file[label] = batch_request_openers
        reply_openers_by_file[label] = batch_reply_openers
        for openers, kind_label in (
            (batch_request_openers, "request"),
            (batch_reply_openers, "reply"),
        ):
            opener_warnings, opener_errors = _narrative_opener_gate(label, openers, kind_label)
            warnings.extend(opener_warnings)
            errors.extend(opener_errors)

    for prefix, locations in request_prefixes.items():
        if len(locations) >= 3:
            errors.append(
                f"template: request 7-word prefix {prefix!r} repeats {len(locations)} "
                f"times ({', '.join(locations[:3])})"
            )

    records = requests + progress_pairs
    counters = {
        field: Counter(str(record.get(field)) for record in records)
        for field in (
            "agent",
            "persona",
            "domain",
            "register",
            "length_bucket",
            "trigger_position",
            "record_kind",
        )
    }
    request_openers: Counter[str] = Counter()
    reply_openers: Counter[str] = Counter()
    for record in requests:
        opener = _word_prefix(str(record.get("text", "")), 2)
        if opener:
            request_openers[opener] += 1
    for record in progress_pairs:
        if record.get("record_kind") == "progress-check":
            opener = _word_prefix(str(record.get("reply", "")), 2)
            if opener:
                reply_openers[opener] += 1

    joint_distributions: dict[str, Counter[str]] = {
        "domain_x_length": Counter(),
        "persona_x_record_kind": Counter(),
        "agent_x_domain": Counter(),
        "record_kind_x_trigger": Counter(),
    }
    for record in records:
        domain = str(record.get("domain"))
        length = str(record.get("length_bucket"))
        persona = str(record.get("persona"))
        agent = str(record.get("agent"))
        kind = str(record.get("record_kind"))
        trigger = str(record.get("trigger_position"))
        joint_distributions["domain_x_length"][f"{domain}|{length}"] += 1
        joint_distributions["persona_x_record_kind"][f"{persona}|{kind}"] += 1
        joint_distributions["agent_x_domain"][f"{agent}|{domain}"] += 1
        joint_distributions["record_kind_x_trigger"][f"{kind}|{trigger}"] += 1

    errors.extend(_narrative_opener_corpus_gate(request_openers, "request"))
    errors.extend(_narrative_opener_corpus_gate(reply_openers, "reply"))

    total_records = len(records)
    if enforce_distribution and total_records < DISTRIBUTION_FLOOR:
        errors.append(
            f"distribution: {total_records} records is below the {DISTRIBUTION_FLOOR}-record "
            "floor for distribution gates; author more source or build with "
            "--allow-small-corpus and accept that the gates did not run"
        )
    elif enforce_distribution and records:
        errors.extend(
            _distribution_gates(
                records,
                counters=counters,
                request_openers=request_openers,
                reply_openers=reply_openers,
            )
        )

    checks = sum(1 for record in progress_pairs if record.get("record_kind") == "progress-check")
    nudges = sum(1 for record in progress_pairs if record.get("record_kind") == "progress-nudge")
    return {
        "passed": not errors,
        "distribution_enforced": bool(enforce_distribution and total_records >= DISTRIBUTION_FLOOR),
        "errors": errors,
        "warnings": warnings,
        "requests": requests,
        "progress_pairs": progress_pairs,
        "counts": {
            "batches": len(batch_items),
            "requests": len(requests),
            "progress_pairs": len(progress_pairs),
            "checks": checks,
            "nudges": nudges,
            "request_openers": dict(sorted(request_openers.items())),
            "reply_openers": dict(sorted(reply_openers.items())),
            "request_openers_by_file": {
                file_label: dict(sorted(counter.items()))
                for file_label, counter in sorted(request_openers_by_file.items())
            },
            "joint_distributions": {
                name: dict(sorted(counts.items())) for name, counts in joint_distributions.items()
            },
            **{field: dict(sorted(counts.items())) for field, counts in counters.items()},
        },
    }


def _batch_wording_gates(
    label: str, batch_requests: list[dict[str, Any]], batch_progress: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    total = len(batch_requests) + len(batch_progress)
    request_first_words: Counter[str] = Counter()
    for record in batch_requests:
        first = _word_prefix(str(record.get("text", "")), 1)
        if first:
            request_first_words[first] += 1
    for opener, count in request_first_words.items():
        if _share(count, max(total, 1)) > 0.30:
            errors.append(
                f"{label}: request first word {opener!r} exceeds 30%; vary request phrasing"
            )
    kind_counts = Counter(str(record.get("record_kind")) for record in batch_progress)
    for kind, count in kind_counts.items():
        if _share(count, max(len(batch_progress), 1)) > 0.80:
            errors.append(
                f"{label}: progress kind {kind!r} exceeds 80% of the batch's progress pairs; "
                "both 'check' and 'nudge' must be represented"
            )
    return errors


def _narrative_opener_gate(
    label: str, openers: Counter[str], kind_label: str
) -> tuple[list[str], list[str]]:
    """Per-file soft caps on wording openers, mirroring Demo 3's scheme."""

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
                f"(>{NARRATIVE_OPENER_ERROR_SHARE:.0%}); vary the wording"
            )
        elif share > NARRATIVE_OPENER_WARN_SHARE:
            warnings.append(
                f"{label}: {kind_label} opener {opener!r} is {share:.0%} of the batch "
                f"(>{NARRATIVE_OPENER_WARN_SHARE:.0%}); vary the wording"
            )
    top_four = sum(count for _, count in openers.most_common(4))
    top_four_share = _share(top_four, total)
    if top_four_share > NARRATIVE_OPENER_TOP4_ERROR_SHARE:
        warnings.append(
            f"{label}: the top four {kind_label} openers cover {top_four_share:.0%} "
            f"of the batch (>{NARRATIVE_OPENER_TOP4_ERROR_SHARE:.0%}); per-file canary only, "
            "the merged-corpus top-four check is what gates acceptance"
        )
    return warnings, errors


def _narrative_opener_corpus_gate(openers: Counter[str], kind_label: str) -> list[str]:
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
            "the wording across authors"
        )
    return errors


def _distribution_gates(
    records: list[dict[str, Any]],
    *,
    counters: dict[str, Counter[str]],
    request_openers: Counter[str],
    reply_openers: Counter[str],
) -> list[str]:
    errors: list[str] = []
    total = len(records)

    for openers, kind_label, cap in ((request_openers, "request", 0.30), (reply_openers, "reply", 0.30)):
        opener_total = sum(openers.values())
        for opener, count in openers.items():
            if _share(count, max(opener_total, 1)) > cap:
                errors.append(
                    f"distribution: {kind_label} opener {opener!r} exceeds {cap:.0%}; "
                    "vary the wording"
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
        ("trigger_position", {"early", "middle", "late"}),
    ):
        counts = counters[field]
        missing = required - set(counts)
        if missing:
            errors.append(f"distribution: {field} missing {sorted(missing)}")
        for name, count in counts.items():
            if _share(count, total) > 0.60:
                errors.append(f"distribution: {field} {name!r} exceeds 60%")

    record_kinds = counters["record_kind"]
    if "progress-check" not in record_kinds or "progress-nudge" not in record_kinds:
        errors.append("distribution: both 'check' and 'nudge' progress kinds are required")

    return errors
