"""Contract-focused scoring for the g1-v2 adaptation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from train.g1_evaluation import score_g1_prediction


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(bool(row.get(key)) for row in rows) / len(rows), 6)


def evaluate_g1_v2(
    pairs: Sequence[Mapping[str, Any]], outputs: Sequence[str], *, label: str
) -> dict[str, Any]:
    if len(pairs) != len(outputs) or not pairs:
        raise ValueError("g1-v2 pairs and outputs must be non-empty and aligned")
    rows: list[dict[str, Any]] = []
    for index, (pair, output) in enumerate(zip(pairs, outputs, strict=True)):
        scored = score_g1_prediction(pair, output, row_index=index)
        scored["v2_group"] = pair.get("v2_group")
        scored["candidate_role"] = pair.get("candidate_role")
        rows.append(scored)

    by_group: dict[str, Any] = {}
    for group in sorted({str(row["v2_group"]) for row in rows}):
        selected = [row for row in rows if row["v2_group"] == group]
        by_group[group] = {
            "support": len(selected),
            "kind_accuracy": _rate(selected, "kind_match"),
            "strict_accuracy": _rate(selected, "row_pass"),
        }

    by_role: dict[str, Any] = {}
    for role in sorted({str(row["candidate_role"]) for row in rows}):
        selected = [row for row in rows if row["candidate_role"] == role]
        by_role[role] = {
            "support": len(selected),
            "kind_accuracy": _rate(selected, "kind_match"),
            "strict_accuracy": _rate(selected, "row_pass"),
        }

    translation_positive = [row for row in rows if row["expected_class"] == "translate_commit"]
    search_call = [row for row in rows if row["expected_class"] == "web_search"]
    search_delivery = [
        row
        for row in rows
        if row["candidate_role"] in {"search-completed", "search-failed"}
    ]
    delivered_idle = [row for row in rows if row["candidate_role"] == "delivered-idle"]
    replay = [row for row in rows if row["v2_group"] == "replay"]
    summary = {
        "label": label,
        "n": len(rows),
        "format_validity": _rate(rows, "format_valid"),
        "canonical_exact_rate": _rate(rows, "canonical_exact"),
        "kind_accuracy": _rate(rows, "kind_match"),
        "strict_accuracy": _rate(rows, "row_pass"),
        "translation_commit_support": len(translation_positive),
        "translation_commit_recall": _rate(translation_positive, "kind_match"),
        "translation_commit_exact": _rate(translation_positive, "row_pass"),
        "web_search_support": len(search_call),
        "web_search_recall": _rate(search_call, "kind_match"),
        "web_search_exact": _rate(search_call, "row_pass"),
        "search_delivery_support": len(search_delivery),
        "search_delivery_recall": _rate(search_delivery, "kind_match"),
        "search_delivery_exact": _rate(search_delivery, "row_pass"),
        "delivered_idle_support": len(delivered_idle),
        "delivered_idle_accuracy": _rate(delivered_idle, "row_pass"),
        "replay_support": len(replay),
        "replay_kind_accuracy": _rate(replay, "kind_match"),
        "replay_strict_accuracy": _rate(replay, "row_pass"),
        "confusion": dict(
            sorted(Counter(f"{row['expected_class']}->{row['predicted_class']}" for row in rows).items())
        ),
    }
    gates = {
        "format_validity": summary["format_validity"] == 1.0,
        "canonical_exact_rate": summary["canonical_exact_rate"] == 1.0,
        "translation_commit_recall": float(summary["translation_commit_recall"] or 0) >= 0.80,
        "web_search_recall": float(summary["web_search_recall"] or 0) >= 0.85,
        "search_delivery_recall": float(summary["search_delivery_recall"] or 0) >= 0.90,
        "delivered_idle_accuracy": float(summary["delivered_idle_accuracy"] or 0) >= 0.95,
        "replay_kind_accuracy": float(summary["replay_kind_accuracy"] or 0) >= 0.85,
    }
    return {
        "summary": summary,
        "groups": by_group,
        "roles": by_role,
        "gates": {"passed": all(gates.values()), "values": gates},
        "rows": rows,
    }
