"""Pure, strict evaluation for the Version 4 interaction protocol.

The evaluator deliberately keeps sampling and file I/O out of this module.  A
caller supplies dataset pairs and decoded model outputs, then persists the
returned JSON-compatible report wherever it wants.

The most important invariant is that an invalid completion is its own
``invalid`` prediction.  ``app.stream.parse_action`` represents parse failures
with an invalid ``ActionKind.IDLE`` value for runtime safety; evaluation must
never turn that fallback into an idle true-positive.
"""

from __future__ import annotations

import html
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from app.domain import Action, ActionKind, DEFAULT_PERMISSIONS
from app.stream import action_completion, parse_action


EVALUATED_CLASSES = (
    "idle",
    "highlight",
    "search",
    "respond",
    # v5 editor classes
    "underline",
    "pause",
    "write",
    "resume",
    "revise",
    # v6 surface classes
    "suggest_edit",
    "generate_ui",
)

# These are acceptance gates, rather than targets used to tune on the hidden
# test set.  Dev can be evaluated repeatedly; hidden test should be run once on
# the checkpoint selected by dev.
DEFAULT_HARD_GATES: dict[str, dict[str, float | str]] = {
    "format_validity": {"operator": "min", "threshold": 0.995},
    "permission_violation_rate": {"operator": "max", "threshold": 0.0},
    "macro_f1": {"operator": "min", "threshold": 0.90},
    "highlight_payload_exact": {"operator": "min", "threshold": 0.90},
    "highlight_quote_token_f1": {"operator": "min", "threshold": 0.97},
    "search_precision": {"operator": "min", "threshold": 0.95},
    "search_recall": {"operator": "min", "threshold": 0.90},
    "search_query_term_coverage": {"operator": "min", "threshold": 0.95},
    "respond_target_accuracy": {"operator": "min", "threshold": 0.98},
    "respond_message_grounding": {"operator": "min", "threshold": 0.95},
    "respond_forbidden_claim_rate": {"operator": "max", "threshold": 0.0},
    "episode_success_rate": {"operator": "min", "threshold": 0.85},
}

# Gate set for the v6 surface (suggest_edit / generate_ui / web_search / respond).
# Gates whose metric is absent from a split report as "skipped", not failed.
V6_HARD_GATES: dict[str, dict[str, float | str]] = {
    "format_validity": {"operator": "min", "threshold": 0.995},
    "permission_violation_rate": {"operator": "max", "threshold": 0.0},
    "macro_f1": {"operator": "min", "threshold": 0.90},
    "suggest_payload_exact": {"operator": "min", "threshold": 0.90},
    "suggest_quote_token_f1": {"operator": "min", "threshold": 0.95},
    "suggest_replacement_exact": {"operator": "min", "threshold": 0.90},
    "generate_ui_request_coverage": {"operator": "min", "threshold": 0.95},
    "search_query_term_coverage": {"operator": "min", "threshold": 0.95},
    "respond_target_accuracy": {"operator": "min", "threshold": 0.98},
    "respond_message_grounding": {"operator": "min", "threshold": 0.95},
    "respond_forbidden_claim_rate": {"operator": "max", "threshold": 0.0},
    "episode_success_rate": {"operator": "min", "threshold": 0.85},
}

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_PERMISSIONS_RE = re.compile(r"<permissions>(.*?)</permissions>", re.DOTALL)


def action_class(action: Action) -> str:
    """Map an action to one of the four v4 labels (or a diagnostic label).

    Checking ``valid`` first is intentional and prevents parse failures from
    being scored as idle.
    """

    if not action.valid:
        return "invalid"
    if action.kind is ActionKind.IDLE:
        return "idle"
    if action.kind is ActionKind.RESPOND:
        return "respond"
    if action.kind is ActionKind.TOOL:
        if action.tool_name == "web_search":
            return "search"
        if action.tool_name == "ui" and action.arguments.get("operation") == "highlight":
            return "highlight"
        if action.tool_name == "ui" and action.arguments.get("operation") == "underline":
            return "underline"
        if action.tool_name == "writer":
            operation = action.arguments.get("operation")
            if operation in ("write", "pause", "resume", "revise"):
                return str(operation)
        if action.tool_name in ("suggest_edit", "generate_ui"):
            return str(action.tool_name)
        return f"tool:{action.tool_name or 'missing'}"
    return "invalid"


def _normalise_text(value: str | None) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_WORD_RE.findall(value))


def _tokens(value: str | None) -> list[str]:
    normalised = _normalise_text(value)
    return normalised.split() if normalised else []


def _token_scores(expected: str | None, predicted: str | None) -> dict[str, float]:
    expected_tokens = Counter(_tokens(expected))
    predicted_tokens = Counter(_tokens(predicted))
    overlap = sum((expected_tokens & predicted_tokens).values())
    expected_n = sum(expected_tokens.values())
    predicted_n = sum(predicted_tokens.values())
    precision = overlap / predicted_n if predicted_n else (1.0 if not expected_n else 0.0)
    recall = overlap / expected_n if expected_n else (1.0 if not predicted_n else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _as_terms(value: Any) -> list[str]:
    if isinstance(value, str):
        return _tokens(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        terms: list[str] = []
        for item in value:
            if isinstance(item, str):
                terms.extend(_tokens(item))
        return terms
    return []


def _evaluation_metadata(pair: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("expected", "evaluation", "eval"):
        value = pair.get(key)
        if isinstance(value, Mapping):
            metadata.update(value)
    return metadata


def _metadata_terms(pair: Mapping[str, Any], *keys: str) -> list[str]:
    metadata = _evaluation_metadata(pair)
    for key in keys:
        terms = _as_terms(metadata.get(key))
        if terms:
            return terms
    return []


def _metadata_strings(pair: Mapping[str, Any], *keys: str) -> list[str]:
    metadata = _evaluation_metadata(pair)
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            strings = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            if strings:
                return strings
    return []


def _metadata_concept_groups(pair: Mapping[str, Any], *keys: str) -> list[list[str]]:
    """Return semantic concepts expressed as alternative surface forms.

    ``[["aurora", "northern lights"], ["when", "best time"]]`` means the
    prediction must express one alias from each inner list. A flat list remains
    backwards compatible and treats every string as its own required concept.
    """

    metadata = _evaluation_metadata(pair)
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return [[value.strip()]]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            continue
        groups: list[list[str]] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                groups.append([item.strip()])
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                aliases = [
                    alias.strip()
                    for alias in item
                    if isinstance(alias, str) and alias.strip()
                ]
                if aliases:
                    groups.append(aliases)
        if groups:
            return groups
    return []


def _term_coverage(required: Sequence[str], predicted: str | None) -> float:
    if not required:
        return 1.0
    required_counts = Counter(_normalise_text(term) for term in required)
    required_counts.pop("", None)
    if not required_counts:
        return 1.0
    predicted_counts = Counter(_tokens(predicted))
    covered = sum(min(count, predicted_counts.get(term, 0)) for term, count in required_counts.items())
    return covered / sum(required_counts.values())


def _claim_coverage(claims: Sequence[str], predicted: str | None) -> float:
    if not claims:
        return 1.0
    message = _normalise_text(predicted)
    matched = sum(bool(claim := _normalise_text(item)) and claim in message for item in claims)
    return matched / len(claims)


def _concept_coverage(groups: Sequence[Sequence[str]], predicted: str | None) -> float:
    """Score how many authored concepts appear through any accepted alias."""

    if not groups:
        return 1.0
    message = _normalise_text(predicted)
    matched = 0
    for group in groups:
        aliases = [_normalise_text(alias) for alias in group]
        if any(alias and alias in message for alias in aliases):
            matched += 1
    return matched / len(groups)


def _permissions_from_prompt(prompt: str) -> dict[str, list[str]] | None:
    match = _PERMISSIONS_RE.search(prompt)
    if not match:
        return None
    try:
        value = json.loads(html.unescape(match.group(1)))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    return {
        str(name): [str(operation) for operation in operations]
        for name, operations in value.items()
        if isinstance(operations, list)
    }


def pair_permissions(pair: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return server permissions attached to a pair.

    Generated v4 rows carry a structured ``permissions`` field.  Parsing the
    stable prompt prefix is a compatibility fallback for hand-written evals.
    If neither is present, the runtime defaults are used.
    """

    value = pair.get("permissions")
    if isinstance(value, Mapping):
        return {
            str(name): [str(operation) for operation in operations]
            for name, operations in value.items()
            if isinstance(operations, list)
        }
    prompt_permissions = _permissions_from_prompt(str(pair.get("prompt", "")))
    if prompt_permissions is not None:
        return prompt_permissions
    return {name: list(operations) for name, operations in DEFAULT_PERMISSIONS.items()}


def permission_violation(action: Action, permissions: Mapping[str, Sequence[str]]) -> bool:
    """Return whether a syntactically valid action exceeds its tool permissions."""

    if not action.valid or action.kind is not ActionKind.TOOL:
        return False
    if "tools" in permissions:
        # v6 flat allow-list: {"tools": ["generate_ui", "suggest_edit", "web_search"]}
        return (action.tool_name or "") not in permissions.get("tools", ())
    if action.tool_name == "web_search":
        operation = "search"
    else:
        operation = action.arguments.get("operation")
    allowed = permissions.get(action.tool_name or "", ())
    return not isinstance(operation, str) or operation not in allowed


def action_schema_valid(action: Action) -> bool:
    """Check the MVP's semantic schema after envelope/JSON parsing succeeds."""

    if not action.valid:
        return False
    if action.kind is ActionKind.IDLE:
        return True
    if action.kind is ActionKind.RESPOND:
        return (
            isinstance(action.target, int)
            and not isinstance(action.target, bool)
            and action.target > 0
            and isinstance(action.message, str)
            and bool(action.message.strip())
        )
    if action.kind is not ActionKind.TOOL:
        return False
    if action.tool_name == "web_search":
        query = action.arguments.get("query")
        return isinstance(query, str) and bool(query.strip())
    if action.tool_name == "ui" and action.arguments.get("operation") in ("highlight", "underline"):
        target = action.arguments.get("target")
        if not isinstance(target, Mapping):
            return False
        quote = target.get("quote")
        occurrence = target.get("occurrence")
        return (
            isinstance(quote, str)
            and bool(quote)
            and isinstance(occurrence, int)
            and not isinstance(occurrence, bool)
            and occurrence > 0
        )
    if action.tool_name == "writer":
        operation = action.arguments.get("operation")
        if operation in ("pause", "resume"):
            return True
        if operation in ("write", "revise"):
            instruction = action.arguments.get("instruction")
            return isinstance(instruction, str) and bool(instruction.strip())
        return False
    if action.tool_name == "suggest_edit":
        target = action.arguments.get("target")
        replacement = action.arguments.get("replacement")
        if not isinstance(target, Mapping):
            return False
        quote = target.get("quote")
        occurrence = target.get("occurrence")
        return (
            isinstance(quote, str)
            and bool(quote)
            and isinstance(occurrence, int)
            and not isinstance(occurrence, bool)
            and occurrence > 0
            and isinstance(replacement, str)
            and bool(replacement.strip())
        )
    if action.tool_name == "generate_ui":
        request = action.arguments.get("request")
        return isinstance(request, str) and bool(request.strip())
    return False


def _expected_action(pair: Mapping[str, Any]) -> Action:
    completion = pair.get("completion")
    if not isinstance(completion, str):
        raise ValueError("Every evaluation pair needs a string 'completion'.")
    action = parse_action(completion)
    label = action_class(action)
    if not action.valid or label not in EVALUATED_CLASSES or not action_schema_valid(action):
        episode = pair.get("episode", "unknown")
        raise ValueError(f"Evaluation pair {episode!r} has a non-canonical completion: {completion!r}")
    return action


def _payload_metrics(
    pair: Mapping[str, Any], expected: Action, predicted: Action, predicted_label: str
) -> dict[str, Any]:
    expected_label = action_class(expected)
    result: dict[str, Any] = {"payload_pass": expected_label == "idle"}

    if expected_label == "highlight":
        expected_target = expected.arguments.get("target", {})
        predicted_target = (
            predicted.arguments.get("target", {}) if predicted_label == "highlight" else {}
        )
        expected_quote = expected_target.get("quote")
        predicted_quote = predicted_target.get("quote")
        quote_scores = _token_scores(expected_quote, predicted_quote)
        quote_exact = isinstance(predicted_quote, str) and predicted_quote == expected_quote
        occurrence_exact = predicted_target.get("occurrence") == expected_target.get("occurrence")
        result.update(
            {
                "highlight_quote_exact": quote_exact,
                "highlight_occurrence_exact": occurrence_exact,
                "highlight_quote_token_precision": quote_scores["precision"],
                "highlight_quote_token_recall": quote_scores["recall"],
                "highlight_quote_token_f1": quote_scores["f1"],
                "highlight_payload_exact": bool(quote_exact and occurrence_exact),
                "payload_pass": bool(quote_exact and occurrence_exact),
            }
        )
        return result

    if expected_label == "search":
        expected_query = expected.query or ""
        predicted_query = predicted.query if predicted_label == "search" else ""
        query_scores = _token_scores(expected_query, predicted_query)
        concept_groups = _metadata_concept_groups(
            pair, "required_query_concepts", "search_query_concepts", "query_concepts"
        )
        required_terms = _metadata_terms(
            pair, "required_query_terms", "search_query_terms", "query_terms"
        ) or _tokens(expected_query)
        coverage = (
            _concept_coverage(concept_groups, predicted_query)
            if concept_groups
            else _term_coverage(required_terms, predicted_query)
        )
        query_exact = _normalise_text(predicted_query) == _normalise_text(expected_query)
        result.update(
            {
                "search_query_exact": query_exact,
                "search_query_token_precision": query_scores["precision"],
                "search_query_token_recall": query_scores["recall"],
                "search_query_token_f1": query_scores["f1"],
                "search_query_term_coverage": coverage,
                # Required-term coverage allows harmless reordering or extra
                # specificity while still demanding every authored concept.
                "payload_pass": bool(predicted_label == "search" and coverage == 1.0),
            }
        )
        return result

    if expected_label == "underline":
        expected_target = expected.arguments.get("target", {})
        predicted_target = (
            predicted.arguments.get("target", {}) if predicted_label == "underline" else {}
        )
        expected_quote = expected_target.get("quote")
        predicted_quote = predicted_target.get("quote")
        quote_scores = _token_scores(expected_quote, predicted_quote)
        quote_exact = isinstance(predicted_quote, str) and predicted_quote == expected_quote
        occurrence_exact = predicted_target.get("occurrence") == expected_target.get("occurrence")
        result.update(
            {
                "underline_quote_exact": quote_exact,
                "underline_occurrence_exact": occurrence_exact,
                "underline_quote_token_f1": quote_scores["f1"],
                "underline_payload_exact": bool(quote_exact and occurrence_exact),
                "payload_pass": bool(quote_exact and occurrence_exact),
            }
        )
        return result

    if expected_label == "suggest_edit":
        expected_target = expected.arguments.get("target", {})
        predicted_target = (
            predicted.arguments.get("target", {}) if predicted_label == "suggest_edit" else {}
        )
        expected_quote = expected_target.get("quote")
        predicted_quote = predicted_target.get("quote")
        quote_scores = _token_scores(expected_quote, predicted_quote)
        quote_exact = isinstance(predicted_quote, str) and predicted_quote == expected_quote
        occurrence_exact = predicted_target.get("occurrence") == expected_target.get("occurrence")
        expected_replacement = str(expected.arguments.get("replacement") or "")
        predicted_replacement = (
            str(predicted.arguments.get("replacement") or "")
            if predicted_label == "suggest_edit"
            else ""
        )
        replacement_scores = _token_scores(expected_replacement, predicted_replacement)
        replacement_exact = predicted_replacement == expected_replacement
        payload_exact = bool(quote_exact and occurrence_exact and replacement_exact)
        result.update(
            {
                "suggest_quote_exact": quote_exact,
                "suggest_occurrence_exact": occurrence_exact,
                "suggest_quote_token_f1": quote_scores["f1"],
                "suggest_replacement_exact": replacement_exact,
                "suggest_replacement_token_f1": replacement_scores["f1"],
                "suggest_payload_exact": payload_exact,
                "payload_pass": payload_exact,
            }
        )
        return result

    if expected_label == "generate_ui":
        expected_request = str(expected.arguments.get("request") or "")
        predicted_request = (
            str(predicted.arguments.get("request") or "")
            if predicted_label == "generate_ui"
            else ""
        )
        request_scores = _token_scores(expected_request, predicted_request)
        required_terms = _metadata_terms(
            pair, "required_request_terms", "request_terms"
        ) or _tokens(expected_request)
        coverage = _term_coverage(required_terms, predicted_request)
        result.update(
            {
                "generate_ui_request_token_f1": request_scores["f1"],
                "generate_ui_request_coverage": coverage,
                "payload_pass": bool(predicted_label == "generate_ui" and coverage == 1.0),
            }
        )
        return result

    if expected_label in ("write", "revise"):
        expected_instruction = str(expected.arguments.get("instruction") or "")
        predicted_instruction = (
            str(predicted.arguments.get("instruction") or "")
            if predicted_label == expected_label
            else ""
        )
        instruction_scores = _token_scores(expected_instruction, predicted_instruction)
        required_terms = _metadata_terms(
            pair, "required_instruction_terms", "instruction_terms"
        ) or _tokens(expected_instruction)
        coverage = _term_coverage(required_terms, predicted_instruction)
        result.update(
            {
                f"{expected_label}_instruction_token_f1": instruction_scores["f1"],
                f"{expected_label}_instruction_coverage": coverage,
                "payload_pass": bool(predicted_label == expected_label and coverage == 1.0),
            }
        )
        return result

    if expected_label in ("pause", "resume"):
        # These carry no payload: matching the class is the whole decision.
        result["payload_pass"] = predicted_label == expected_label
        return result

    if expected_label == "respond":
        predicted_message = predicted.message if predicted_label == "respond" else ""
        message_scores = _token_scores(expected.message, predicted_message)
        claim_groups = _metadata_concept_groups(
            pair, "required_response_claims", "response_claims", "required_claims"
        )
        claims = _metadata_strings(
            pair, "required_response_claims", "response_claims", "required_claims"
        )
        if claim_groups:
            grounding = _concept_coverage(claim_groups, predicted_message)
        elif claims:
            grounding = _claim_coverage(claims, predicted_message)
        else:
            # Without explicit atomic claims, expected-token recall is the
            # deterministic proxy.  Production eval data should provide claims
            # whenever paraphrases are acceptable.
            grounding = message_scores["recall"]
        forbidden_groups = _metadata_concept_groups(
            pair, "forbidden_response_claims", "forbidden_claims"
        )
        forbidden_rate = (
            _concept_coverage(forbidden_groups, predicted_message)
            if forbidden_groups
            else 0.0
        )
        target_exact = predicted_label == "respond" and predicted.target == expected.target
        message_exact = _normalise_text(predicted_message) == _normalise_text(expected.message)
        result.update(
            {
                "respond_target_exact": target_exact,
                "respond_message_exact": message_exact,
                "respond_message_token_precision": message_scores["precision"],
                "respond_message_token_recall": message_scores["recall"],
                "respond_message_token_f1": message_scores["f1"],
                "respond_message_grounding": grounding,
                "respond_forbidden_claim_rate": forbidden_rate,
                "payload_pass": bool(
                    target_exact and grounding == 1.0 and forbidden_rate == 0.0
                ),
            }
        )
        return result

    return result


def score_prediction(
    pair: Mapping[str, Any], raw_output: str, *, row_index: int = 0
) -> dict[str, Any]:
    """Score one decoded output against one dataset pair."""

    expected = _expected_action(pair)
    predicted = parse_action(raw_output)
    expected_label = action_class(expected)
    predicted_label = action_class(predicted)
    permissions = pair_permissions(pair)
    violation = permission_violation(predicted, permissions)
    schema_valid = action_schema_valid(predicted)
    kind_match = predicted.valid and predicted_label == expected_label
    payload = _payload_metrics(pair, expected, predicted, predicted_label)
    row_pass = bool(
        predicted.valid
        and schema_valid
        and kind_match
        and not violation
        and payload["payload_pass"]
    )

    return {
        "row_index": row_index,
        "episode": str(pair.get("episode", f"row-{row_index}")),
        "resample": pair.get("resample"),
        "event_index": pair.get("event_index"),
        "bucket": pair.get("bucket"),
        "expected_class": expected_label,
        "predicted_class": predicted_label,
        "format_valid": predicted.valid,
        "canonical_exact": bool(
            predicted.valid and raw_output.strip() == action_completion(predicted)
        ),
        "schema_valid": schema_valid,
        "kind_match": kind_match,
        "permission_violation": violation,
        "diagnostic": predicted.diagnostic,
        "raw": raw_output,
        "row_pass": row_pass,
        **payload,
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _class_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for label in EVALUATED_CLASSES:
        support = sum(row["expected_class"] == label for row in rows)
        predicted = sum(row["predicted_class"] == label for row in rows)
        true_positive = sum(
            row["expected_class"] == label and row["predicted_class"] == label for row in rows
        )
        false_positive = predicted - true_positive
        false_negative = support - true_positive
        precision = true_positive / predicted if predicted else (0.0 if support else None)
        recall = true_positive / support if support else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else (0.0 if support else None)
        )
        report[label] = {
            "support": support,
            "predicted": predicted,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
        }
    return report


def _episode_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["episode"])].append(row)
    return {
        episode: {
            "n": len(episode_rows),
            "passed": all(bool(row["row_pass"]) for row in episode_rows),
            "row_accuracy": _round(_mean(episode_rows, "row_pass")),
            "format_validity": _round(_mean(episode_rows, "format_valid")),
            "failed_rows": [
                int(row["row_index"]) for row in episode_rows if not row["row_pass"]
            ],
        }
        for episode, episode_rows in sorted(grouped.items())
    }


def _bucket_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("bucket") or "unknown")].append(row)
    return {
        bucket: {
            "n": len(bucket_rows),
            "kind_accuracy": _round(_mean(bucket_rows, "kind_match")),
            "strict_row_accuracy": _round(_mean(bucket_rows, "row_pass")),
        }
        for bucket, bucket_rows in sorted(grouped.items())
    }


def _wilson_95(successes: int, total: int) -> list[float] | None:
    """95% Wilson score interval for a binomial proportion.

    Attached to the two headline pass/fail rates so a 6-episode or 32-target
    difference is read with its uncertainty, not as a bare point estimate.
    """
    if total <= 0:
        return None
    z = 1.959964
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def _overall_score(metrics: Mapping[str, Any]) -> float:
    # v6 datasets are scored on the v6 decision set. The presence of
    # suggest_edit rows is the discriminator: only v6 data contains them.
    if metrics.get("suggest_payload_exact") is not None:
        v6_components = {
            "format_validity": (0.15, metrics.get("format_validity")),
            "macro_f1": (0.25, metrics.get("macro_f1")),
            "permission_compliance": (
                0.10,
                1.0 - float(metrics.get("permission_violation_rate") or 0.0),
            ),
            "suggest_payload_exact": (0.15, metrics.get("suggest_payload_exact")),
            "generate_ui_request_coverage": (
                0.05,
                metrics.get("generate_ui_request_coverage"),
            ),
            "search_query_term_coverage": (
                0.05,
                metrics.get("search_query_term_coverage"),
            ),
            "respond_target_accuracy": (0.05, metrics.get("respond_target_accuracy")),
            "respond_message_grounding": (0.05, metrics.get("respond_message_grounding")),
            "episode_success_rate": (0.15, metrics.get("episode_success_rate")),
        }
        return sum(weight * float(value or 0.0) for weight, value in v6_components.values())
    # Editor (v5) datasets are scored on the editor decision set. The presence
    # of pause rows is the discriminator: only cowrite data contains them.
    if metrics.get("pause_accuracy") is not None:
        editor_components = {
            "format_validity": (0.15, metrics.get("format_validity")),
            "macro_f1": (0.25, metrics.get("macro_f1")),
            "permission_compliance": (
                0.10,
                1.0 - float(metrics.get("permission_violation_rate") or 0.0),
            ),
            "pause_accuracy": (0.10, metrics.get("pause_accuracy")),
            "underline_payload_exact": (0.15, metrics.get("underline_payload_exact")),
            "revise_instruction_coverage": (
                0.08,
                metrics.get("revise_instruction_coverage"),
            ),
            "respond_target_accuracy": (0.03, metrics.get("respond_target_accuracy")),
            "respond_message_grounding": (0.04, metrics.get("respond_message_grounding")),
            "episode_success_rate": (0.10, metrics.get("episode_success_rate")),
        }
        return sum(weight * float(value or 0.0) for weight, value in editor_components.values())
    components = {
        "format_validity": (0.15, metrics.get("format_validity")),
        "macro_f1": (0.25, metrics.get("macro_f1")),
        "permission_compliance": (
            0.10,
            1.0 - float(metrics.get("permission_violation_rate") or 0.0),
        ),
        "highlight_payload_exact": (0.15, metrics.get("highlight_payload_exact")),
        "search_query_term_coverage": (
            0.10,
            metrics.get("search_query_term_coverage"),
        ),
        "respond_target_accuracy": (0.05, metrics.get("respond_target_accuracy")),
        "respond_message_grounding": (0.08, metrics.get("respond_message_grounding")),
        "respond_forbidden_claim_compliance": (
            0.02,
            1.0 - float(metrics.get("respond_forbidden_claim_rate") or 0.0),
        ),
        "episode_success_rate": (0.10, metrics.get("episode_success_rate")),
    }
    # Missing action classes intentionally contribute zero.  A supposedly
    # comprehensive eval cannot earn a high score by omitting a hard behavior.
    return sum(weight * float(value or 0.0) for weight, value in components.values())


def hard_gate_report(
    metrics: Mapping[str, Any],
    gates: Mapping[str, Mapping[str, Any] | float] | None = None,
) -> dict[str, Any]:
    """Evaluate named metrics against min/max acceptance thresholds."""

    gates = DEFAULT_HARD_GATES if gates is None else gates
    results: dict[str, dict[str, Any]] = {}
    for metric, specification in gates.items():
        if isinstance(specification, (int, float)):
            operator, threshold = "min", float(specification)
        else:
            operator = str(specification.get("operator", "min"))
            threshold = float(specification["threshold"])
        if operator not in {"min", "max"}:
            raise ValueError(f"Unknown hard-gate operator {operator!r} for {metric!r}")
        raw_value = metrics.get(metric)
        value = float(raw_value) if isinstance(raw_value, (int, float)) else None
        passed = (
            value is not None
            and math.isfinite(value)
            and (value >= threshold if operator == "min" else value <= threshold)
        )
        # A gate whose metric is absent from this split is skipped, not failed:
        # a cowrite-only split cannot fail a highlight gate (2026-07-15 finding).
        results[metric] = {
            "value": _round(value),
            "operator": ">=" if operator == "min" else "<=",
            "threshold": threshold,
            "passed": passed,
            "status": "pass" if passed else ("skipped" if value is None else "fail"),
        }
    passed_count = sum(result["passed"] for result in results.values())
    skipped_count = sum(result["status"] == "skipped" for result in results.values())
    evaluated = len(results) - skipped_count
    return {
        "passed": evaluated > 0 and passed_count == evaluated,
        "passed_count": passed_count,
        "skipped_count": skipped_count,
        "total": len(results),
        "gates": results,
    }


def evaluate_predictions(
    pairs: Sequence[Mapping[str, Any]],
    outputs: Sequence[str],
    *,
    label: str | None = None,
    hard_gates: Mapping[str, Mapping[str, Any] | float] | None = None,
) -> dict[str, Any]:
    """Return a JSON-compatible v4 evaluation report.

    ``pairs`` and ``outputs`` must be aligned one-to-one.  Expected completions
    are validated before scoring so a corrupt gold row cannot silently improve
    or degrade model metrics.
    """

    if len(pairs) != len(outputs):
        raise ValueError(f"Expected {len(pairs)} outputs, received {len(outputs)}.")
    if not pairs:
        raise ValueError("Evaluation requires at least one pair.")

    rows = [
        score_prediction(pair, raw_output, row_index=index)
        for index, (pair, raw_output) in enumerate(zip(pairs, outputs, strict=True))
    ]
    classes = _class_report(rows)
    episodes = _episode_report(rows)
    highlight_rows = [row for row in rows if row["expected_class"] == "highlight"]
    search_rows = [row for row in rows if row["expected_class"] == "search"]
    respond_rows = [row for row in rows if row["expected_class"] == "respond"]
    underline_rows = [row for row in rows if row["expected_class"] == "underline"]
    revise_rows = [row for row in rows if row["expected_class"] == "revise"]
    pause_rows = [row for row in rows if row["expected_class"] == "pause"]
    suggest_rows = [row for row in rows if row["expected_class"] == "suggest_edit"]
    genui_rows = [row for row in rows if row["expected_class"] == "generate_ui"]
    action_rows = [row for row in rows if row["expected_class"] != "idle"]
    macro_values = [
        float(item["f1"])
        for item in classes.values()
        if item["support"] and item["f1"] is not None
    ]

    metrics: dict[str, Any] = {
        "label": label,
        "n": len(rows),
        "episode_count": len(episodes),
        "format_validity": _round(_mean(rows, "format_valid")),
        "canonical_exact_rate": _round(_mean(rows, "canonical_exact")),
        "schema_validity": _round(_mean(rows, "schema_valid")),
        "kind_accuracy": _round(_mean(rows, "kind_match")),
        "macro_f1": _round(sum(macro_values) / len(macro_values) if macro_values else None),
        "strict_row_accuracy": _round(_mean(rows, "row_pass")),
        "action_payload_accuracy": _round(_mean(action_rows, "payload_pass")),
        "permission_violations": sum(bool(row["permission_violation"]) for row in rows),
        "permission_violation_rate": _round(_mean(rows, "permission_violation")),
        "highlight_quote_exact": _round(_mean(highlight_rows, "highlight_quote_exact")),
        "highlight_occurrence_accuracy": _round(
            _mean(highlight_rows, "highlight_occurrence_exact")
        ),
        "highlight_payload_exact": _round(
            _mean(highlight_rows, "highlight_payload_exact")
        ),
        "highlight_quote_token_f1": _round(
            _mean(highlight_rows, "highlight_quote_token_f1")
        ),
        "search_query_exact": _round(_mean(search_rows, "search_query_exact")),
        "search_query_token_f1": _round(_mean(search_rows, "search_query_token_f1")),
        "search_query_term_coverage": _round(
            _mean(search_rows, "search_query_term_coverage")
        ),
        "respond_target_accuracy": _round(_mean(respond_rows, "respond_target_exact")),
        "respond_message_exact": _round(_mean(respond_rows, "respond_message_exact")),
        "respond_message_token_f1": _round(
            _mean(respond_rows, "respond_message_token_f1")
        ),
        "respond_message_grounding": _round(
            _mean(respond_rows, "respond_message_grounding")
        ),
        "respond_forbidden_claim_rate": _round(
            _mean(respond_rows, "respond_forbidden_claim_rate")
        ),
        "underline_quote_exact": _round(_mean(underline_rows, "underline_quote_exact")),
        "underline_payload_exact": _round(_mean(underline_rows, "underline_payload_exact")),
        "underline_quote_token_f1": _round(_mean(underline_rows, "underline_quote_token_f1")),
        "revise_instruction_coverage": _round(
            _mean(revise_rows, "revise_instruction_coverage")
        ),
        "pause_accuracy": _round(_mean(pause_rows, "kind_match")),
        "suggest_quote_exact": _round(_mean(suggest_rows, "suggest_quote_exact")),
        "suggest_occurrence_accuracy": _round(_mean(suggest_rows, "suggest_occurrence_exact")),
        "suggest_quote_token_f1": _round(_mean(suggest_rows, "suggest_quote_token_f1")),
        "suggest_replacement_exact": _round(_mean(suggest_rows, "suggest_replacement_exact")),
        "suggest_replacement_token_f1": _round(
            _mean(suggest_rows, "suggest_replacement_token_f1")
        ),
        "suggest_payload_exact": _round(_mean(suggest_rows, "suggest_payload_exact")),
        "generate_ui_request_coverage": _round(
            _mean(genui_rows, "generate_ui_request_coverage")
        ),
        "generate_ui_request_token_f1": _round(
            _mean(genui_rows, "generate_ui_request_token_f1")
        ),
        "episode_success_rate": _round(
            sum(bool(item["passed"]) for item in episodes.values()) / len(episodes)
        ),
        "episode_success_ci95": _wilson_95(
            sum(bool(item["passed"]) for item in episodes.values()), len(episodes)
        ),
        "strict_row_ci95": _wilson_95(
            sum(bool(row["row_pass"]) for row in rows), len(rows)
        ),
        "search_precision": classes["search"]["precision"],
        "search_recall": classes["search"]["recall"],
        "confusion": dict(
            sorted(Counter(f"{row['expected_class']}->{row['predicted_class']}" for row in rows).items())
        ),
        "per_class": classes,
        "per_bucket": _bucket_report(rows),
    }
    metrics["overall_score"] = _round(_overall_score(metrics))
    metrics["overall_score_percent"] = _round(100 * metrics["overall_score"], 3)
    gate_report = hard_gate_report(metrics, hard_gates)
    return {
        "summary": metrics,
        "hard_gates": gate_report,
        "episodes": episodes,
        "rows": rows,
    }


# A natural alias for callers that refer to decoded samples as outputs.
evaluate_outputs = evaluate_predictions


__all__ = [
    "DEFAULT_HARD_GATES",
    "V6_HARD_GATES",
    "EVALUATED_CLASSES",
    "action_class",
    "action_schema_valid",
    "evaluate_outputs",
    "evaluate_predictions",
    "hard_gate_report",
    "pair_permissions",
    "permission_violation",
    "score_prediction",
]
