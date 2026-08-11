"""Build a focused repair corpus for search delivery after foreground dialogue."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.domain import Action, ActionKind, CompletedTurn, EventSource
from scripts.g1_v2_build import (
    ROOT,
    SEARCH_TOPICS,
    UI_TASKS,
    _card,
    _event,
    _load_jsonl,
    _rank,
    _write_jsonl,
)


OUTPUT = ROOT / "data" / "g1_v2" / "delivery_repair"
MANIFEST = ROOT / "artifacts" / "g1-v2" / "delivery-repair-manifest.json"
STATE_PATH = ROOT / "data" / "tinker" / "run_state.json"
FOREGROUND = (
    ("what does reusable mean here?", "It means the rocket can fly again after landing instead of being discarded."),
    ("can I keep talking while it builds?", "Yes. The delegated job keeps running while we continue this conversation."),
    ("what will the panel show?", "It will show the requested topic in a compact visual panel."),
    ("are you still there?", "Yes, I am still here while the background job runs."),
)


def _payload(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _episode(number: int, split: str, *, varied_indices: bool) -> list[dict[str, Any]]:
    query, title, source_fact = SEARCH_TOPICS[number % len(SEARCH_TOPICS)]
    task = UI_TASKS[number % len(UI_TASKS)]
    question, foreground_answer = FOREGROUND[number % len(FOREGROUND)]
    episode = f"g1v2-delivery-{split}-{number:03d}"
    complete = f"Please {task}. While that runs, look up {query} for delivery case {number}."
    history: list[CompletedTurn] = []
    idle = Action(ActionKind.IDLE)

    # Real typing sessions contain a variable number of earlier ticks. Vary the
    # absolute indices so the model must copy the CURRENT result index rather
    # than memorize fixed positions from a synthetic episode skeleton.
    pad = 1 + number % 7 if varied_indices else 0
    for offset in range(pad):
        end = max(1, len(complete) * (offset + 1) // (pad + 1))
        partial = _event(offset + 1, EventSource.USER, complete[:end])
        history.append(CompletedTurn(partial, idle))

    request_index = pad + 1
    delegate_accepted_index = request_index + 1
    search_accepted_index = request_index + 2
    foreground_index = request_index + 3
    first_terminal_index = request_index + 4
    second_terminal_index = request_index + 5
    final_index = request_index + 6
    delegate_job = f"job-{request_index}"
    search_job = f"job-{delegate_accepted_index}"

    request = _event(request_index, EventSource.USER, complete)
    delegate = Action(ActionKind.TOOL, "delegate", {"task": task})
    history.append(CompletedTurn(request, delegate))

    delegate_accepted = _event(
        delegate_accepted_index,
        EventSource.TOOL,
        _payload({"status": "accepted"}),
        tool="delegate",
        job=delegate_job,
    )
    search = Action(ActionKind.TOOL, "web_search", {"query": query})
    history.append(CompletedTurn(delegate_accepted, search))

    search_accepted = _event(
        search_accepted_index,
        EventSource.TOOL,
        _payload({"job_id": search_job, "status": "accepted"}),
        tool="web_search",
        job=search_job,
    )
    history.append(CompletedTurn(search_accepted, idle))

    foreground_text = complete + "\nWhile that runs, " + question
    foreground = _event(foreground_index, EventSource.USER, foreground_text)
    foreground_response = Action(
        ActionKind.RESPOND,
        target=foreground_index,
        message=foreground_answer,
    )
    rows = [_card(episode, "foreground-response", history, foreground, foreground_response, split)]
    history.append(CompletedTurn(foreground, foreground_response))

    search_event = _event(
        first_terminal_index if number % 2 == 0 else second_terminal_index,
        EventSource.TOOL,
        _payload(
            {
                "job_id": search_job,
                "status": "completed",
                "query": query,
                "results": [
                    {
                        "title": title,
                        "url": "https://example.com/source",
                        "snippet": source_fact,
                    }
                ],
            }
        ),
        tool="web_search",
        job=search_job,
    )
    search_response = Action(
        ActionKind.RESPOND,
        target=search_event.index,
        message=f"According to {title}, {source_fact[0].lower() + source_fact[1:]}",
    )
    delegate_event = _event(
        second_terminal_index if number % 2 == 0 else first_terminal_index,
        EventSource.TOOL,
        _payload({"status": "completed", "task": task}),
        tool="delegate",
        job=delegate_job,
    )

    if number % 2 == 0:
        rows.append(_card(episode, "search-completed", history, search_event, search_response, split))
        history.append(CompletedTurn(search_event, search_response))
        rows.append(_card(episode, "delegate-completed-after-delivery", history, delegate_event, idle, split))
        history.append(CompletedTurn(delegate_event, idle))
    else:
        rows.append(_card(episode, "delegate-completed-before-delivery", history, delegate_event, idle, split))
        history.append(CompletedTurn(delegate_event, idle))
        rows.append(_card(episode, "search-completed", history, search_event, search_response, split))
        history.append(CompletedTurn(search_event, search_response))

    unchanged = _event(final_index, EventSource.USER, foreground_text)
    rows.append(_card(episode, "delivered-idle", history, unchanged, idle, split))
    return rows


def _replay_rows() -> list[dict[str, Any]]:
    source = _load_jsonl(ROOT / "data" / "g1_v2" / "train.jsonl")
    groups = {
        "translation": 60,
        "web_search": 50,
        "replay": 50,
    }
    selected: list[dict[str, Any]] = []
    for group, count in groups.items():
        candidates = [row for row in source if row.get("v2_group") == group]
        candidates.sort(key=lambda row: _rank(f"delivery-replay:{row['candidate_id']}"))
        selected.extend({**row, "delivery_repair_role": "replay"} for row in candidates[:count])
    return selected


def _source_paths(
    state_path: Path,
    source_state: str | None,
    source_sampler: str | None,
) -> tuple[str, str]:
    """Resolve the stage-2 checkpoint without embedding one user's Tinker path."""

    if source_state and source_sampler:
        return source_state, source_sampler
    if source_state or source_sampler:
        raise SystemExit("Pass --source-state and --source-sampler together.")
    if not state_path.exists():
        raise SystemExit(
            "No promoted stage-2 run state found. Run scripts.g1_v2_train and "
            "pass its promotion gates, or pass --source-state and --source-sampler."
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    resolved_state = state.get("g1-v2:selected_state")
    resolved_sampler = state.get("g1-v2:selected_sampler_path")
    if not resolved_state or not resolved_sampler:
        raise SystemExit(
            "run_state.json does not contain a promoted stage-2 checkpoint. "
            "Run scripts.g1_v2_train and pass its promotion gates first."
        )
    return str(resolved_state), str(resolved_sampler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-mode", choices=("fixed", "varied"), default="fixed")
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    parser.add_argument("--source-state")
    parser.add_argument("--source-sampler")
    args = parser.parse_args()
    source_state, source_sampler = _source_paths(
        args.state_path,
        args.source_state,
        args.source_sampler,
    )
    varied = args.index_mode == "varied"
    train_focus = [
        row for number in range(96) for row in _episode(number, "train", varied_indices=varied)
    ]
    dev = [
        row for number in range(96, 120) for row in _episode(number, "dev", varied_indices=varied)
    ]
    train = [*train_focus, *_replay_rows()]
    train.sort(key=lambda row: _rank(f"delivery-train:{row['candidate_id']}"))
    dev.sort(key=lambda row: _rank(f"delivery-dev:{row['candidate_id']}"))
    train_prompts = {str(row["prompt"]) for row in train}
    dev_prompts = {str(row["prompt"]) for row in dev}
    if len(train_prompts) != len(train) or len(dev_prompts) != len(dev):
        raise ValueError("Duplicate delivery-repair prompts")
    if train_prompts & dev_prompts:
        raise ValueError("Delivery-repair train/dev prompt overlap")

    train_path = OUTPUT / "train.jsonl"
    dev_path = OUTPUT / "dev.jsonl"
    train_sha = _write_jsonl(train_path, train)
    dev_sha = _write_jsonl(dev_path, dev)
    manifest = {
        "schema_version": "g1-v2-delivery-repair-1",
        "source_state": source_state,
        "source_sampler": source_sampler,
        "index_mode": args.index_mode,
        "train": {
            "path": str(train_path.relative_to(ROOT)),
            "rows": len(train),
            "sha256": train_sha,
            "classes": dict(sorted(Counter(row["expected_class"] for row in train).items())),
        },
        "dev": {
            "path": str(dev_path.relative_to(ROOT)),
            "rows": len(dev),
            "sha256": dev_sha,
            "classes": dict(sorted(Counter(row["expected_class"] for row in dev).items())),
        },
        "train_dev_prompt_overlap": 0,
        "steps_at_batch_16": (len(train) + 15) // 16,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
