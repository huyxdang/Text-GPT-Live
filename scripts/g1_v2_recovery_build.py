"""Build a small recovery-heavy continuation after the round-1 live miss."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "g1_v2" / "recovery.jsonl"
MANIFEST = ROOT / "artifacts" / "g1-v2" / "recovery-manifest.json"


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _rank(row: dict[str, Any], key: str) -> tuple[bytes, str]:
    value = f"{key}:{row['candidate_id']}"
    return hashlib.sha256(value.encode()).digest(), value


def _take(rows: list[dict[str, Any]], n: int, key: str, group: str) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: _rank(row, key))[:n]
    if len(selected) != n:
        raise ValueError(f"Need {n} rows for {group}, found {len(selected)}")
    return [
        {
            **row,
            "schema_version": "g1-v2",
            "candidate_id": f"recovery:{group}:{row['candidate_id']}",
            "v2_group": f"recovery_{group}",
        }
        for row in selected
    ]


def main() -> None:
    v2 = _load(ROOT / "data" / "g1_v2" / "train.jsonl")
    original = [
        row
        for index in range(1, 5)
        for row in _load(ROOT / "data" / f"train_g1-{index:05d}-of-00004.jsonl")
    ]
    rows: list[dict[str, Any]] = []
    rows += _take(
        [
            row
            for row in v2
            if row["expected_class"] == "translate_commit"
            and row["prompt"].count("<action>translate_commit(") >= 1
        ],
        160,
        "translation-positive",
        "translation_positive",
    )
    rows += _take(
        [
            row
            for row in v2
            if row["expected_class"] == "idle"
            and row["prompt"].count("<action>translate_commit(") >= 1
        ],
        100,
        "translation-idle",
        "translation_idle",
    )
    rows += _take(
        [
            row
            for row in original
            if row["demo"] == "demo-2"
            and row["expected_class"] == "highlight"
            and row["prompt"].count("<action>highlight(") >= 1
        ],
        100,
        "highlight-positive",
        "highlight_positive",
    )
    rows += _take(
        [
            row
            for row in original
            if row["demo"] == "demo-2"
            and row["expected_class"] == "idle"
            and row["prompt"].count("<action>highlight(") >= 1
        ],
        60,
        "highlight-idle",
        "highlight_idle",
    )
    search = [row for row in v2 if row.get("v2_group") == "web_search"]
    rows += _take(
        [row for row in search if row["expected_class"] == "web_search"],
        20,
        "search-call",
        "search_call",
    )
    rows += _take(
        [row for row in search if row["candidate_role"] in {"search-completed", "search-failed"}],
        20,
        "search-delivery",
        "search_delivery",
    )
    rows += _take(
        [row for row in search if row["expected_class"] == "idle"],
        40,
        "search-idle",
        "search_idle",
    )
    rows += _take(
        [row for row in original if row["demo"] in {"demo-1", "demo-4"}],
        80,
        "dialog-task-replay",
        "dialog_task_replay",
    )

    rows.sort(key=lambda row: _rank(row, "final"))
    prompts = [str(row["prompt"]) for row in rows]
    if len(prompts) != len(set(prompts)):
        raise ValueError("Recovery set contains duplicate prompts")
    dev_prompts = {str(row["prompt"]) for row in _load(ROOT / "data" / "g1_v2" / "dev.jsonl")}
    if set(prompts) & dev_prompts:
        raise ValueError("Recovery set overlaps frozen g1-v2 dev")
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    OUT.write_text(payload, encoding="utf-8")
    manifest = {
        "schema_version": "g1-v2-recovery-1",
        "source_checkpoint": "g1-v2:candidate_state",
        "rows": len(rows),
        "steps_at_batch_16": (len(rows) + 15) // 16,
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "groups": dict(sorted(Counter(str(row["v2_group"]) for row in rows).items())),
        "classes": dict(sorted(Counter(str(row["expected_class"]) for row in rows).items())),
        "frozen_dev_overlap": 0,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
