"""Strict g1 pilot scoring with failure-resistant reminder metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from app.domain import Action, ActionKind
from app.stream import g1_action_completion, parse_g1_action
from datagen.g1_pilot import action_label


G1_PILOT_GATES = {
    "format_validity": 1.0,
    "canonical_exact_rate": 1.0,
    "strict_row_accuracy": 1.0,
    "should_fire_recall": 0.90,
    "reminder_wait_accuracy": 0.90,
    "ordinary_silence_idle_accuracy": 1.0,
    "clause_boundary_accuracy": 1.0,
}


def _payload_exact(expected: Action, predicted: Action) -> bool:
    if action_label(expected) != action_label(predicted):
        return False
    if expected.kind is ActionKind.IDLE:
        return True
    if expected.kind is ActionKind.RESPOND:
        return expected.target == predicted.target and expected.message == predicted.message
    return (
        expected.tool_name == predicted.tool_name
        and expected.arguments == predicted.arguments
    )


def score_g1_prediction(
    pair: Mapping[str, Any],
    raw_output: str,
    *,
    row_index: int = 0,
) -> dict[str, Any]:
    expected = parse_g1_action(str(pair.get("completion", "")))
    if not expected.valid:
        raise ValueError(
            f"Evaluation pair {pair.get('episode', row_index)!r} has invalid g1 gold: "
            f"{expected.diagnostic}"
        )
    predicted = parse_g1_action(raw_output)
    expected_class = action_label(expected)
    predicted_class = action_label(predicted) if predicted.valid else "invalid"
    kind_match = predicted.valid and expected_class == predicted_class
    payload_exact = predicted.valid and _payload_exact(expected, predicted)
    return {
        "row_index": row_index,
        "episode": str(pair.get("episode", f"row-{row_index}")),
        "demo": pair.get("demo"),
        "situation": pair.get("situation"),
        "expected_class": expected_class,
        "predicted_class": predicted_class,
        "format_valid": predicted.valid,
        "canonical_exact": bool(
            predicted.valid and raw_output == g1_action_completion(predicted)
        ),
        "kind_match": kind_match,
        "payload_exact": payload_exact,
        "row_pass": bool(kind_match and payload_exact),
        "should_fire": bool(pair.get("should_fire")),
        "timing_boundary": pair.get("timing_boundary"),
        "reminder_eval_kind": pair.get("reminder_eval_kind"),
        "current_content_empty": bool(pair.get("current_content_empty")),
        "obligation": pair.get("obligation"),
        "clause_state": pair.get("clause_state"),
        "diagnostic": predicted.diagnostic,
        "raw": raw_output,
    }


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(bool(row.get(key)) for row in rows) / len(rows), 6)


def evaluate_g1_predictions(
    pairs: Sequence[Mapping[str, Any]],
    outputs: Sequence[str],
    *,
    label: str = "g1",
) -> dict[str, Any]:
    if len(pairs) != len(outputs):
        raise ValueError(f"Expected {len(pairs)} outputs, received {len(outputs)}.")
    if not pairs:
        raise ValueError("g1 evaluation requires at least one pair.")

    rows = [
        score_g1_prediction(pair, output, row_index=index)
        for index, (pair, output) in enumerate(zip(pairs, outputs, strict=True))
    ]
    fire_rows = [row for row in rows if row["should_fire"]]
    reminder_wait_rows = [
        row
        for row in rows
        if row["demo"] == "demo-5"
        and (
            row["reminder_eval_kind"] == "wait"
            or row["situation"] == "reminder-wait"
        )
    ]
    ordinary_silence_rows = [
        row
        for row in rows
        if row["current_content_empty"] and row["obligation"] == "none"
    ]
    clause_rows = [row for row in rows if row["clause_state"] in {"partial", "complete"}]
    fire_recall = (
        round(
            sum(row["predicted_class"] == "respond" for row in fire_rows) / len(fire_rows),
            6,
        )
        if fire_rows
        else None
    )
    wait_accuracy = _rate(reminder_wait_rows, "row_pass")
    silence_accuracy = _rate(ordinary_silence_rows, "row_pass")
    clause_accuracy = _rate(clause_rows, "row_pass")
    metrics = {
        "label": label,
        "n": len(rows),
        "format_validity": _rate(rows, "format_valid"),
        "canonical_exact_rate": _rate(rows, "canonical_exact"),
        "strict_row_accuracy": _rate(rows, "row_pass"),
        "should_fire_support": len(fire_rows),
        "should_fire_recall": fire_recall,
        "reminder_wait_support": len(reminder_wait_rows),
        "reminder_wait_accuracy": wait_accuracy,
        "ordinary_silence_support": len(ordinary_silence_rows),
        "ordinary_silence_idle_accuracy": silence_accuracy,
        "clause_boundary_support": len(clause_rows),
        "clause_boundary_accuracy": clause_accuracy,
        "confusion": dict(
            sorted(
                Counter(
                    f"{row['expected_class']}->{row['predicted_class']}"
                    for row in rows
                ).items()
            )
        ),
    }
    gates: dict[str, dict[str, Any]] = {}
    for metric, threshold in G1_PILOT_GATES.items():
        value = metrics.get(metric)
        passed = isinstance(value, (int, float)) and value >= threshold
        gates[metric] = {
            "value": value,
            "operator": ">=",
            "threshold": threshold,
            "passed": passed,
        }
    return {
        "summary": metrics,
        "hard_gates": {
            "passed": all(gate["passed"] for gate in gates.values()),
            "gates": gates,
        },
        "rows": rows,
    }
