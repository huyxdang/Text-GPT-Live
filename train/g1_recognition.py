"""Pure construction and scoring for the g1 fire-now recognition probe."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from random import Random
import re
from typing import Any

from app.stream import parse_g1_action


RECOGNITION_SYSTEM_PROMPT = """You are diagnosing an interaction policy, not acting in the stream.
Read the chronological stream and compare the two proposed next actions. Select the action that
correctly follows the standing reminder and event timestamps. Answer with exactly A or B."""

_COMPLETED_TURN_RE = re.compile(
    r"(<stream_event\b.*?</stream_event>)\n(<action>.*?</action>)",
    re.DOTALL,
)
_EVENT_RE = re.compile(r"<stream_event\b.*?</stream_event>", re.DOTALL)


def _head_tail(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    head = limit // 2
    tail = limit - head
    return value[:head] + "\n[repeated snapshot text omitted]\n" + value[-tail:]


def recognition_stream_evidence(prompt: str, *, max_chars: int = 12_000) -> str:
    """Keep the current snapshot and timestamped prior responses, not every keystroke."""

    if max_chars < 2_000:
        raise ValueError("Recognition context must allow at least 2,000 characters.")
    current_events = _EVENT_RE.findall(prompt)
    if not current_events or not prompt.rstrip().endswith("<PREDICT_THIS_ACTION>"):
        raise ValueError("Recognition source prompt has no current stream event.")
    prior_responses = [
        f"{event}\n{action}"
        for event, action in _COMPLETED_TURN_RE.findall(prompt)
        if action.startswith("<action>respond(")
    ]
    prior_budget = min(max_chars // 3, len(prior_responses) * 1_500)
    prior = "\n".join(prior_responses)
    current_budget = max_chars - min(len(prior), prior_budget)
    pieces = []
    if prior:
        pieces.append("<prior_responses>\n" + _head_tail(prior, prior_budget) + "\n</prior_responses>")
    pieces.append(_head_tail(current_events[-1], current_budget))
    return "\n".join(pieces)


def build_fire_recognition_cases(
    pairs: Sequence[Mapping[str, Any]],
    *,
    eligible_row_indexes: set[int] | None = None,
    seed: int = 650,
    presentations_per_item: int = 2,
    context_chars: int = 12_000,
) -> list[dict[str, Any]]:
    if presentations_per_item not in {1, 2}:
        raise ValueError("Recognition presentations per item must be one or two.")
    cases: list[dict[str, Any]] = []
    rng = Random(seed)
    for row_index, pair in enumerate(pairs):
        if not pair.get("should_fire"):
            continue
        if eligible_row_indexes is not None and row_index not in eligible_row_indexes:
            continue
        gold = str(pair.get("completion", ""))
        action = parse_g1_action(gold)
        if not action.valid or action.message is None:
            raise ValueError(f"Should-fire row {row_index} does not have a valid respond action.")
        orders = [rng.choice([True, False])]
        if presentations_per_item == 2:
            orders.append(not orders[0])
        for presentation, gold_first in enumerate(orders):
            choices = [gold, "<action>idle()</action>"] if gold_first else ["<action>idle()</action>", gold]
            expected_choice = "A" if gold_first else "B"
            evidence = recognition_stream_evidence(
                str(pair.get("prompt", "")),
                max_chars=context_chars,
            )
            prompt = (
                "<interaction_stream>\n"
                f"{evidence}\n"
                "</interaction_stream>\n\n"
                "Which proposed next action is correct at this tick?\n"
                f"A: {choices[0]}\n"
                f"B: {choices[1]}\n"
                "Answer exactly A or B."
            )
            cases.append(
                {
                    "row_index": row_index,
                    "episode": str(pair.get("episode", f"row-{row_index}")),
                    "candidate_id": pair.get("candidate_id"),
                    "situation": pair.get("situation"),
                    "timing_boundary": pair.get("timing_boundary"),
                    "presentation": presentation,
                    "correct_action": gold,
                    "rejected_action": "<action>idle()</action>",
                    "correct_position": expected_choice,
                    "prompt": prompt,
                }
            )
    if not cases:
        raise ValueError("Recognition probe found no eligible should-fire rows.")
    return cases


def score_fire_recognition(
    cases: Sequence[Mapping[str, Any]],
    outputs: Sequence[str],
) -> dict[str, Any]:
    if len(cases) != len(outputs):
        raise ValueError(f"Expected {len(cases)} recognition outputs, received {len(outputs)}.")
    rows: list[dict[str, Any]] = []
    by_item: dict[int, list[bool]] = defaultdict(list)
    for case, raw in zip(cases, outputs, strict=True):
        choice = raw.strip() if raw.strip() in {"A", "B"} else "invalid"
        correct = choice == case.get("correct_position")
        row = dict(case)
        row.pop("prompt", None)
        row.update({"raw": raw, "choice": choice, "valid": choice != "invalid", "correct": correct})
        rows.append(row)
        by_item[int(case["row_index"])].append(correct)

    valid = sum(row["valid"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    consistent = sum(all(results) for results in by_item.values())
    accuracy = correct / len(rows)
    consistency = consistent / len(by_item)
    if accuracy >= 0.90 and consistency >= 0.80:
        diagnosis = "preference_gap"
        next_training_action = "short_symmetric_dpo_with_sft_replay"
    elif accuracy <= 0.60:
        diagnosis = "capability_gap"
        next_training_action = "targeted_fire_wait_sft_then_reprobe"
    else:
        diagnosis = "mixed_gap"
        next_training_action = "targeted_fire_wait_sft_then_short_symmetric_dpo"
    return {
        "summary": {
            "items": len(by_item),
            "presentations": len(rows),
            "valid_choice_rate": round(valid / len(rows), 6),
            "recognition_accuracy": round(accuracy, 6),
            "order_consistent_item_rate": round(consistency, 6),
            "choice_counts": dict(sorted(Counter(row["choice"] for row in rows).items())),
            "diagnosis": diagnosis,
            "next_training_action": next_training_action,
        },
        "rows": rows,
    }
