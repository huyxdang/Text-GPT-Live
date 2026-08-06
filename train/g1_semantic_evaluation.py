"""Hybrid deterministic and LLM-judged evaluation for the g1 policy.

The strict evaluator remains the source of truth for grammar, action choice,
timing, response targets, and exact source spans.  This module adds semantic
judgment only where wording is legitimately open-ended: response messages,
delegate task descriptions, and suggest-edit replacements.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from app.domain import Action, ActionKind
from app.stream import parse_g1_action


SEMANTIC_PAYLOAD_CLASSES = frozenset({"respond", "delegate", "suggest_edit"})
G1_HYBRID_GATES = {
    "format_validity": 1.0,
    "canonical_exact_rate": 1.0,
    "hybrid_row_accuracy": 1.0,
    "should_fire_recall": 0.90,
    "reminder_wait_accuracy": 0.90,
    "ordinary_silence_idle_accuracy": 1.0,
    "hybrid_clause_boundary_accuracy": 1.0,
}


def _action_class(action: Action) -> str:
    if not action.valid:
        return "invalid"
    if action.kind is ActionKind.TOOL:
        return str(action.tool_name)
    return action.kind.value


def _semantic_value(action: Action, action_class: str) -> str | None:
    if action_class == "respond":
        return action.message
    if action_class == "delegate":
        task = action.arguments.get("task")
        return task if isinstance(task, str) else None
    if action_class == "suggest_edit":
        replacement = action.arguments.get("replacement")
        return replacement if isinstance(replacement, str) else None
    return None


def _deterministic_anchors_pass(expected: Action, predicted: Action, action_class: str) -> bool:
    """Keep routing and exact source-location contracts outside the LLM judge."""

    if not predicted.valid or _action_class(predicted) != action_class:
        return False
    if action_class == "respond":
        return expected.target == predicted.target
    if action_class == "suggest_edit":
        return expected.arguments.get("quote") == predicted.arguments.get("quote")
    if action_class == "highlight":
        return expected.arguments == predicted.arguments
    return True


def _row_pair(
    pairs: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(pairs):
        raise ValueError("Strict report rows must align one-for-one with evaluation pairs.")
    aligned: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for index, (pair, row) in enumerate(zip(pairs, rows, strict=True)):
        if not isinstance(row, Mapping):
            raise ValueError(f"Strict report row {index} is not an object.")
        if row.get("row_index") != index or row.get("episode") != pair.get("episode"):
            raise ValueError(f"Strict report row {index} does not match the evaluation pair.")
        aligned.append((pair, row))
    return aligned


def _context_tail(pair: Mapping[str, Any], *, max_chars: int) -> str:
    prompt = pair.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError(f"Evaluation pair {pair.get('episode')!r} has no prompt text.")
    if max_chars <= 0:
        raise ValueError("Semantic judge context length must be positive.")
    return prompt if len(prompt) <= max_chars else "[earlier stream omitted]\n" + prompt[-max_chars:]


def build_semantic_cases(
    pairs: Sequence[Mapping[str, Any]],
    strict_report: Mapping[str, Any],
    *,
    context_chars: int = 12_000,
) -> list[dict[str, Any]]:
    """Return only non-exact rows whose remaining difference is semantic wording."""

    cases: list[dict[str, Any]] = []
    for pair, row in _row_pair(pairs, strict_report):
        action_class = str(row.get("expected_class"))
        if action_class not in SEMANTIC_PAYLOAD_CLASSES or row.get("payload_exact") is True:
            continue
        expected = parse_g1_action(str(pair.get("completion", "")))
        predicted = parse_g1_action(str(row.get("raw", "")))
        if not expected.valid:
            raise ValueError(f"Invalid gold action for {pair.get('episode')!r}.")
        if not _deterministic_anchors_pass(expected, predicted, action_class):
            continue
        reference = _semantic_value(expected, action_class)
        candidate = _semantic_value(predicted, action_class)
        if not reference or not candidate:
            continue
        cases.append(
            {
                "row_index": int(row["row_index"]),
                "episode": str(row["episode"]),
                "demo": row.get("demo"),
                "situation": row.get("situation"),
                "payload_kind": action_class,
                "reference": reference,
                "candidate": candidate,
                "context": _context_tail(pair, max_chars=context_chars),
            }
        )
    return cases


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(bool(row.get(key)) for row in rows) / len(rows), 6)


def apply_semantic_judgments(
    pairs: Sequence[Mapping[str, Any]],
    strict_report: Mapping[str, Any],
    judgments: Mapping[int | str, Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Merge semantic verdicts while failing closed on deterministic mistakes."""

    normalized: dict[int, Mapping[str, Any]] = {}
    for key, value in judgments.items():
        try:
            row_index = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Semantic judgment key {key!r} is not a row index.") from exc
        if not isinstance(value, Mapping) or not isinstance(value.get("pass"), bool):
            raise ValueError(f"Semantic judgment {key!r} must contain a boolean pass field.")
        normalized[row_index] = value

    merged_rows: list[dict[str, Any]] = []
    missing: list[int] = []
    semantic_support = 0
    semantic_passes = 0
    judged_support = 0
    exact_semantic_support = 0

    for pair, strict_row in _row_pair(pairs, strict_report):
        row = dict(strict_row)
        action_class = str(row.get("expected_class"))
        expected = parse_g1_action(str(pair.get("completion", "")))
        predicted = parse_g1_action(str(row.get("raw", "")))
        anchors_pass = _deterministic_anchors_pass(expected, predicted, action_class)
        row["deterministic_anchors_pass"] = anchors_pass
        row["semantic_judgment"] = None

        if action_class in SEMANTIC_PAYLOAD_CLASSES and anchors_pass:
            semantic_support += 1
            if row.get("payload_exact") is True:
                semantic_pass = True
                exact_semantic_support += 1
                row["semantic_evaluation"] = "exact"
            else:
                judgment = normalized.get(int(row["row_index"]))
                if judgment is None:
                    missing.append(int(row["row_index"]))
                    semantic_pass = False
                    row["semantic_evaluation"] = "missing"
                else:
                    judged_support += 1
                    semantic_pass = bool(judgment["pass"])
                    row["semantic_evaluation"] = "judge"
                    row["semantic_judgment"] = dict(judgment)
            semantic_passes += int(semantic_pass)
            row["semantic_payload_pass"] = semantic_pass
            row["hybrid_row_pass"] = bool(anchors_pass and semantic_pass)
        else:
            row["semantic_evaluation"] = "not_applicable"
            row["semantic_payload_pass"] = None
            row["hybrid_row_pass"] = bool(row.get("payload_exact"))
        merged_rows.append(row)

    if missing and require_complete:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(
            f"Missing {len(missing)} required semantic judgments (row indexes: {preview})."
        )

    clause_rows = [
        row for row in merged_rows if row.get("clause_state") in {"partial", "complete"}
    ]
    summary = dict(strict_report.get("summary") or {})
    summary.update(
        {
            "hybrid_row_accuracy": _rate(merged_rows, "hybrid_row_pass") if not missing else None,
            "semantic_payload_support": semantic_support,
            "semantic_payload_accuracy": (
                round(semantic_passes / semantic_support, 6) if semantic_support else None
            ),
            "semantic_judged_support": judged_support,
            "semantic_exact_support": exact_semantic_support,
            "semantic_pending_support": len(missing),
            "hybrid_clause_boundary_accuracy": (
                _rate(clause_rows, "hybrid_row_pass") if not missing else None
            ),
            "hybrid_confusion": dict(
                sorted(
                    Counter(
                        f"{row['expected_class']}->{'pass' if row['hybrid_row_pass'] else 'fail'}"
                        for row in merged_rows
                    ).items()
                )
            ),
        }
    )
    gates: dict[str, dict[str, Any]] = {}
    for metric, threshold in G1_HYBRID_GATES.items():
        value = summary.get(metric)
        gates[metric] = {
            "value": value,
            "operator": ">=",
            "threshold": threshold,
            "passed": isinstance(value, (int, float)) and not isinstance(value, bool) and value >= threshold,
        }
    return {
        "summary": summary,
        "hard_gates": strict_report.get("hard_gates"),
        "hybrid_hard_gates": {
            "passed": not missing and all(gate["passed"] for gate in gates.values()),
            "gates": gates,
        },
        "semantic_judge": {
            "complete": not missing,
            "required": judged_support + len(missing),
            "received": judged_support,
            "missing_row_indexes": missing,
        },
        "rows": merged_rows,
    }
