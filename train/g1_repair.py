"""Deterministic construction and focused evaluation for the g1 Demo 5 repair."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from train.g1_evaluation import score_g1_prediction


REPAIR_COUNTS = {
    "fire": 24,
    "matched_wait": 24,
    "replay_action": 16,
    "replay_silence": 8,
    "replay_reminder": 8,
}

LADDER_VALIDATION_PAIRS_PER_MODE = 7


def _rank(value: str) -> tuple[bytes, str]:
    return hashlib.sha256(value.encode("utf-8")).digest(), value


def _tag(row: Mapping[str, Any], group: str) -> dict[str, Any]:
    return {**dict(row), "repair_group": group}


def _take_ranked(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    *,
    key: str,
    used_episodes: set[str] | None = None,
) -> list[dict[str, Any]]:
    used = used_episodes if used_episodes is not None else set()
    selected: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: _rank(f"{key}:{item['candidate_id']}")):
        episode = str(row["episode"])
        if episode in used:
            continue
        selected.append(dict(row))
        used.add(episode)
        if len(selected) == count:
            return selected
    raise ValueError(f"Could not select {count} unique-episode rows for {key}.")


def _select_matched_pairs(train_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_cycle: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in train_rows:
        if (
            row.get("demo") == "demo-5"
            and row.get("schedule_kind") == "every"
            and isinstance(row.get("fire_index"), int)
        ):
            by_cycle[(str(row["episode"]), int(row["fire_index"]))][
                str(row["candidate_role"])
            ] = row

    pools: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {
        "fire-silent": [],
        "fire-typing": [],
    }
    for roles in by_cycle.values():
        wait = roles.get("fire-before")
        if wait is None:
            continue
        for fire_role in pools:
            fire = roles.get(fire_role)
            if fire is not None:
                pools[fire_role].append((fire, wait))

    used_episodes: set[str] = set()
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for fire_role in ("fire-silent", "fire-typing"):
        pool = sorted(
            pools[fire_role],
            key=lambda pair: (
                int(pair[0]["interval_s"]),
                _rank(f"repair-pair:{pair[0]['candidate_id']}"),
            ),
        )
        chosen: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        # Cover as many interval values as possible before filling the 12 slots.
        for interval in sorted({int(pair[0]["interval_s"]) for pair in pool}):
            match = next(
                (
                    pair
                    for pair in pool
                    if int(pair[0]["interval_s"]) == interval
                    and str(pair[0]["episode"]) not in used_episodes
                ),
                None,
            )
            if match is not None and len(chosen) < 12:
                chosen.append(match)
                used_episodes.add(str(match[0]["episode"]))
        for pair in pool:
            if len(chosen) == 12:
                break
            if str(pair[0]["episode"]) in used_episodes:
                continue
            chosen.append(pair)
            used_episodes.add(str(pair[0]["episode"]))
        if len(chosen) != 12:
            raise ValueError(f"Need 12 unique-schedule {fire_role} repair pairs.")
        selected.extend(chosen)

    rows: list[dict[str, Any]] = []
    for fire, wait in selected:
        rows.extend((_tag(fire, "fire"), _tag(wait, "matched_wait")))
    return rows


def build_repair_rows(
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the exact 80-card repair corpus and a leakage audit."""

    paired = _select_matched_pairs(train_rows)
    used_episodes = {str(row["episode"]) for row in paired}

    action_specs = (
        ("demo-1", {"address-positive"}, 4, "d1"),
        ("demo-2", {"error-positive"}, 2, "d2-edit"),
        ("demo-2", {"match-positive"}, 2, "d2-highlight"),
        ("demo-3", {"clause-positive"}, 4, "d3"),
        ("demo-4", {"request-positive"}, 2, "d4-delegate"),
        ("demo-4", {"check-positive", "failure-check-positive"}, 2, "d4-respond"),
    )
    action_replay: list[dict[str, Any]] = []
    for demo, situations, count, key in action_specs:
        pool = [
            row
            for row in train_rows
            if row.get("demo") == demo and row.get("situation") in situations
        ]
        action_replay.extend(
            _tag(row, "replay_action")
            for row in _take_ranked(pool, count, key=f"repair-action:{key}", used_episodes=used_episodes)
        )

    silence_replay: list[dict[str, Any]] = []
    for demo in ("demo-1", "demo-2", "demo-3", "demo-4"):
        pool = [
            row
            for row in train_rows
            if row.get("demo") == demo
            and row.get("current_content_empty") is True
            and row.get("expected_class") == "idle"
        ]
        silence_replay.extend(
            _tag(row, "replay_silence")
            for row in _take_ranked(
                pool,
                2,
                key=f"repair-silence:{demo}",
                used_episodes=used_episodes,
            )
        )

    reminder_pool = [
        row
        for row in train_rows
        if row.get("demo") == "demo-5"
        and row.get("situation") in {"post-cancel-idle", "once-no-repeat", "fire-after"}
        and row.get("expected_class") == "idle"
    ]
    reminder_replay = [
        _tag(row, "replay_reminder")
        for row in _take_ranked(
            reminder_pool,
            8,
            key="repair-reminder",
            used_episodes=used_episodes,
        )
    ]

    rows = [*paired, *action_replay, *silence_replay, *reminder_replay]
    counts = Counter(str(row["repair_group"]) for row in rows)
    if dict(counts) != REPAIR_COUNTS:
        raise ValueError(f"Repair composition drifted: {dict(counts)}")
    prompts = [str(row["prompt"]) for row in rows]
    candidate_ids = [str(row["candidate_id"]) for row in rows]
    dev_episodes = {str(row["episode"]) for row in dev_rows}
    dev_prompts = {str(row["prompt"]) for row in dev_rows}
    leaked_episodes = sorted({str(row["episode"]) for row in rows} & dev_episodes)
    leaked_prompts = sorted(set(prompts) & dev_prompts)
    if len(set(prompts)) != len(prompts) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Repair corpus contains duplicate prompts or candidate ids.")
    if leaked_episodes or leaked_prompts:
        raise ValueError("Repair corpus leaks frozen dev episodes or prompts.")
    if any(row.get("schedule_kind") == "once" for row in rows if row["repair_group"] == "fire"):
        raise ValueError("Repair fire cards must not include vague one-shot schedules.")

    manifest = {
        "schema_version": "g1-repair-corpus-1",
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "fire_modes": dict(
            sorted(Counter(str(row["candidate_role"]) for row in rows if row["repair_group"] == "fire").items())
        ),
        "fire_intervals_s": dict(
            sorted(Counter(int(row["interval_s"]) for row in rows if row["repair_group"] == "fire").items())
        ),
        "unique_episodes": len({str(row["episode"]) for row in rows}),
        "dev_episode_overlap": len(leaked_episodes),
        "dev_prompt_overlap": len(leaked_prompts),
        "duplicate_prompts": len(prompts) - len(set(prompts)),
        "duplicate_candidate_ids": len(candidate_ids) - len(set(candidate_ids)),
    }
    return rows, manifest


def build_ladder_validation_cases(
    train_rows: Sequence[Mapping[str, Any]],
    repair_rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a train-split-only fire/wait set for choosing a repair rung.

    The validation schedules are disjoint from both the repair corpus and the
    frozen acceptance split. They may be used repeatedly while selecting a
    checkpoint, unlike the frozen focused and full suites.
    """

    excluded_episodes = {str(row["episode"]) for row in repair_rows}
    frozen_episodes = {str(row["episode"]) for row in frozen_rows}
    frozen_prompts = {str(row["prompt"]) for row in frozen_rows}
    by_cycle: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in train_rows:
        if (
            row.get("demo") == "demo-5"
            and row.get("schedule_kind") == "every"
            and isinstance(row.get("fire_index"), int)
            and str(row["episode"]) not in excluded_episodes
        ):
            by_cycle[(str(row["episode"]), int(row["fire_index"]))][
                str(row.get("candidate_role"))
            ] = row

    pools: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {
        "fire-silent": [],
        "fire-typing": [],
    }
    for roles in by_cycle.values():
        wait = roles.get("fire-before")
        if wait is None:
            continue
        for fire_role in pools:
            fire = roles.get(fire_role)
            if fire is not None:
                pools[fire_role].append((fire, wait))

    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    used_episodes: set[str] = set()
    for fire_role in ("fire-silent", "fire-typing"):
        pool = sorted(
            pools[fire_role],
            key=lambda pair: (
                int(pair[0]["interval_s"]),
                _rank(f"repair-validation:{pair[0]['candidate_id']}"),
            ),
        )
        chosen: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for interval in sorted({int(pair[0]["interval_s"]) for pair in pool}):
            match = next(
                (
                    pair
                    for pair in pool
                    if int(pair[0]["interval_s"]) == interval
                    and str(pair[0]["episode"]) not in used_episodes
                ),
                None,
            )
            if match is not None and len(chosen) < LADDER_VALIDATION_PAIRS_PER_MODE:
                chosen.append(match)
                used_episodes.add(str(match[0]["episode"]))
        for pair in pool:
            if len(chosen) == LADDER_VALIDATION_PAIRS_PER_MODE:
                break
            if str(pair[0]["episode"]) in used_episodes:
                continue
            chosen.append(pair)
            used_episodes.add(str(pair[0]["episode"]))
        if len(chosen) != LADDER_VALIDATION_PAIRS_PER_MODE:
            raise ValueError(
                f"Need {LADDER_VALIDATION_PAIRS_PER_MODE} unused {fire_role} "
                "schedules for ladder validation."
            )
        selected.extend(chosen)

    cases: list[dict[str, Any]] = []
    for fire, wait in selected:
        pair_id = str(fire["candidate_id"])
        for role, row in (("wait", wait), ("fire", fire)):
            cases.append(
                {
                    **dict(row),
                    "case_id": f"{pair_id}::{role}",
                    "pair_id": pair_id,
                    "role": role,
                    "source": "train-only-ladder-validation",
                }
            )
    cases.sort(key=lambda case: (str(case["pair_id"]), str(case["role"])))

    prompts = [str(case["prompt"]) for case in cases]
    validation_episodes = {str(case["episode"]) for case in cases}
    repair_prompts = {str(row["prompt"]) for row in repair_rows}
    audit = {
        "schema_version": "g1-repair-ladder-validation-1",
        "support": len(cases),
        "pairs": len(selected),
        "schedules": len(validation_episodes),
        "fire_silent": sum(case.get("candidate_role") == "fire-silent" for case in cases),
        "fire_typing": sum(case.get("candidate_role") == "fire-typing" for case in cases),
        "repair_episode_overlap": len(validation_episodes & excluded_episodes),
        "repair_prompt_overlap": len(set(prompts) & repair_prompts),
        "frozen_episode_overlap": len(validation_episodes & frozen_episodes),
        "frozen_prompt_overlap": len(set(prompts) & frozen_prompts),
        "duplicate_prompts": len(prompts) - len(set(prompts)),
    }
    if audit != {
        **audit,
        "repair_episode_overlap": 0,
        "repair_prompt_overlap": 0,
        "frozen_episode_overlap": 0,
        "frozen_prompt_overlap": 0,
        "duplicate_prompts": 0,
    }:
        raise ValueError(f"Ladder validation leakage audit failed: {audit}")
    if len(cases) != 28 or len(selected) != 14 or len(validation_episodes) != 14:
        raise ValueError(f"Ladder validation composition drifted: {audit}")
    return cases, audit


def pack_repair_batches(rows: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    """Create five deterministic 16-card batches with fire/wait balance."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["repair_group"])].append(dict(row))
    for group in groups:
        groups[group].sort(key=lambda row: _rank(f"repair-batch:{row['candidate_id']}"))

    batches: list[list[dict[str, Any]]] = [[] for _ in range(5)]
    for index, row in enumerate(groups["fire"]):
        batches[index % 5].append(row)
    # Rotate the 24 waits so the four-item remainder lands on another batch.
    for index, row in enumerate(groups["matched_wait"]):
        batches[(index + 4) % 5].append(row)
    replay = [
        row
        for group in ("replay_action", "replay_silence", "replay_reminder")
        for row in groups[group]
    ]
    replay.sort(key=lambda row: _rank(f"repair-replay:{row['candidate_id']}"))
    for row in replay:
        destination = min(range(5), key=lambda index: (len(batches[index]), index))
        batches[destination].append(row)
    for batch in batches:
        batch.sort(key=lambda row: _rank(f"repair-within:{row['candidate_id']}"))
    if [len(batch) for batch in batches] != [16] * 5:
        raise ValueError(f"Repair batches are not exactly five by 16: {[len(batch) for batch in batches]}")
    if any(sum(row["repair_group"] == "fire" for row in batch) not in {4, 5} for batch in batches):
        raise ValueError("Every repair batch must contain four or five fires.")
    if any(sum(row["repair_group"] == "matched_wait" for row in batch) not in {4, 5} for batch in batches):
        raise ValueError("Every repair batch must contain four or five matched waits.")
    return batches


def score_focused_repair(
    cases: Sequence[Mapping[str, Any]], outputs: Sequence[str]
) -> dict[str, Any]:
    if len(cases) != len(outputs):
        raise ValueError("Focused repair case/output counts do not match.")
    rows = [
        {**score_g1_prediction(case, output, row_index=index), "focus_group": case["focus_group"]}
        for index, (case, output) in enumerate(zip(cases, outputs, strict=True))
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["focus_group"])].append(row)

    def exact(group: str) -> int:
        return sum(row["row_pass"] for row in grouped[group])

    summary = {
        "support": len(rows),
        "format_valid": sum(row["format_valid"] for row in rows),
        "canonical_exact": sum(row["canonical_exact"] for row in rows),
        "fire_support": len(grouped["fire"]),
        "fire_exact": exact("fire"),
        "matched_wait_support": len(grouped["matched_wait"]),
        "matched_wait_exact": exact("matched_wait"),
        "silence_support": len(grouped["silence"]),
        "silence_exact": exact("silence"),
        "reminder_restraint_support": len(grouped["reminder_restraint"]),
        "reminder_restraint_exact": exact("reminder_restraint"),
        "collision_support": len(grouped["collision"]),
        "collision_kind_correct": sum(row["kind_match"] for row in grouped["collision"]),
    }
    summary["passed"] = (
        summary["format_valid"] == len(rows)
        and summary["canonical_exact"] == len(rows)
        and summary["fire_exact"] >= 13
        and summary["matched_wait_exact"] >= 13
        and summary["silence_exact"] == summary["silence_support"] == 25
        and summary["reminder_restraint_exact"] == summary["reminder_restraint_support"]
        and summary["collision_kind_correct"] == summary["collision_support"]
    )
    return {"summary": summary, "rows": rows}
