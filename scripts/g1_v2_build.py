"""Build the deterministic g1-v2 adaptation and frozen evaluation sets.

The adaptation changes two contracts only:

* Demo 3 clause responses become append-only ``translate_commit`` actions.
* Demo 4 gains real ``web_search`` jobs whose completed results must be
  surfaced exactly once by the model.

All original Demo 3 causal histories are converted, then mixed with new
closed-loop search episodes and deterministic replay from passing Demos 1, 2,
and 4. Demo 5 is intentionally absent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from app.domain import Action, ActionKind, CompletedTurn, EventSource, StreamEvent, UserState
from app.stream import compile_stream, g1_action_completion, parse_g1_action


ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATHS = tuple(ROOT / "data" / f"train_g1-{index:05d}-of-00004.jsonl" for index in range(1, 5))
DEV_PATH = ROOT / "data" / "dev_g1.jsonl"
OUTPUT_DIR = ROOT / "data" / "g1_v2"
ARTIFACT_DIR = ROOT / "artifacts" / "g1-v2"
ACTION_RE = re.compile(r'^<action>respond\((\{.*\})\)</action>$')

SEARCH_TOPICS = (
    ("latest SpaceX private-market valuation", "Space market report", "The report lists a recent private-market valuation estimate for SpaceX."),
    ("current price of gold per ounce", "Commodities snapshot", "The snapshot lists the current quoted gold price per ounce."),
    ("weather in Hanoi today", "Hanoi forecast", "The forecast gives today's temperature and rain outlook for Hanoi."),
    ("latest Python stable release", "Python release notes", "The release page identifies the latest stable Python version."),
    ("current NVIDIA share price", "Market quote", "The quote page provides the latest available NVIDIA share price."),
    ("next Formula One race date", "Formula One calendar", "The calendar identifies the next scheduled Formula One race."),
    ("latest population estimate for Tokyo", "Population estimate", "The source provides a recent population estimate for Tokyo."),
    ("current USD to VND exchange rate", "Currency quote", "The quote shows the current indicative USD to VND rate."),
    ("latest Qwen model release", "Qwen release update", "The update describes the newest published Qwen model release."),
    ("today's air quality in Singapore", "Singapore air quality", "The report provides today's air-quality reading for Singapore."),
    ("current Bitcoin price", "Crypto market quote", "The quote provides the latest available Bitcoin price."),
    ("next public holiday in Vietnam", "Vietnam holiday calendar", "The calendar identifies the next public holiday in Vietnam."),
    ("latest Premier League table", "Premier League standings", "The table shows the latest published league standings."),
    ("current opening hours for the Louvre", "Louvre visitor information", "The visitor page lists the museum's current opening hours."),
    ("latest macOS version", "Apple software update", "The update page identifies the latest generally available macOS version."),
    ("current coffee futures price", "Coffee futures quote", "The quote lists the latest available coffee futures price."),
    ("next lunar eclipse date", "Eclipse calendar", "The calendar identifies the next scheduled lunar eclipse."),
    ("latest unemployment rate in the United States", "Labor statistics release", "The release reports the latest published unemployment rate."),
    ("current time in London", "London local time", "The page provides the current local time in London."),
    ("latest OpenAI API model release", "OpenAI model update", "The update identifies a recently released API model."),
)

SEARCH_PREFIXES = (
    "Please search for",
    "Can you look up",
    "Find me",
    "Check",
)

UI_TASKS = (
    "build a clean dashboard about reusable rockets",
    "make a compact market card",
    "create a visual travel brief",
    "design a small status panel",
    "generate a simple comparison UI",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _rank(value: str) -> tuple[bytes, str]:
    return hashlib.sha256(value.encode()).digest(), value


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def _dedupe(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    kept: dict[str, dict[str, Any]] = {}
    dropped = 0
    for row in rows:
        prompt = str(row["prompt"])
        previous = kept.get(prompt)
        if previous is None:
            kept[prompt] = row
        elif previous["completion"] != row["completion"]:
            raise ValueError("Conflicting labels for an identical g1-v2 prompt")
        else:
            dropped += 1
    return list(kept.values()), dropped


def _convert_prompt(prompt: str) -> str:
    converted: list[str] = []
    seen_response = False
    for line in prompt.splitlines():
        match = ACTION_RE.fullmatch(line)
        if not match:
            converted.append(line)
            continue
        payload = json.loads(match.group(1))
        if seen_response:
            action = Action(
                ActionKind.TOOL,
                "translate_commit",
                {"for": payload["for"], "message": payload["message"]},
            )
            converted.append(g1_action_completion(action))
        else:
            converted.append(line)
            seen_response = True
    return "\n".join(converted)


def convert_demo3(row: dict[str, Any]) -> dict[str, Any]:
    result = {**row, "prompt": _convert_prompt(str(row["prompt"])), "v2_group": "translation"}
    if row.get("situation") == "clause-positive":
        parsed = parse_g1_action(str(row["completion"]))
        if not parsed.valid or parsed.kind is not ActionKind.RESPOND:
            raise ValueError(f"Unexpected Demo 3 clause label: {row['candidate_id']}")
        result["completion"] = g1_action_completion(
            Action(
                ActionKind.TOOL,
                "translate_commit",
                {"for": parsed.target, "message": parsed.message},
            )
        )
        result["expected_class"] = "translate_commit"
    if not parse_g1_action(str(result["completion"])).valid:
        raise ValueError(f"Converted label is invalid: {row['candidate_id']}")
    return result


def _event(index: int, source: EventSource, content: str, *, tool: str | None = None, job: str | None = None) -> StreamEvent:
    return StreamEvent(
        index=index,
        source=source,
        content=content,
        state=UserState.ACTIVE if source is EventSource.USER else None,
        elapsed_ms=index * 650,
        tool_name=tool,
        job_id=job,
    )


def _card(episode: str, role: str, history: list[CompletedTurn], event: StreamEvent, action: Action, split: str) -> dict[str, Any]:
    completion = g1_action_completion(action)
    if not parse_g1_action(completion).valid:
        raise ValueError(completion)
    expected = action.tool_name if action.kind is ActionKind.TOOL else action.kind.value
    return {
        "schema_version": "g1-v2",
        "demo": "demo-4",
        "episode": episode,
        "candidate_id": f"{episode}:{role}",
        "candidate_role": role,
        "situation": role,
        "bucket": role,
        "split": split,
        "expected_class": expected,
        "current_event_index": event.index,
        "current_content_empty": not bool(event.content),
        "obligation": "search-delivery" if role in {"search-completed", "search-failed"} else "none",
        "v2_group": "web_search",
        "prompt": compile_stream(history, event, fmt="g1"),
        "completion": completion,
    }


def build_search_episode(number: int, split: str, combined: bool) -> list[dict[str, Any]]:
    query, title, fact = SEARCH_TOPICS[number % len(SEARCH_TOPICS)]
    prefix = SEARCH_PREFIXES[(number // len(SEARCH_TOPICS)) % len(SEARCH_PREFIXES)]
    episode = f"g1v2-search-{split}-{number:03d}-{'combined' if combined else 'direct'}"
    job_search = "job-3" if combined else "job-2"
    history: list[CompletedTurn] = []
    rows: list[dict[str, Any]] = []

    if combined:
        task = UI_TASKS[number % len(UI_TASKS)]
        complete = (
            f"Please {task}. While that runs, {prefix.lower()} {query} "
            f"for lookup {number}."
        )
    else:
        task = ""
        complete = f"{prefix} {query} for lookup {number}."
    partial = complete[:-max(4, len(complete) // 6)]
    first = _event(1, EventSource.USER, partial)
    idle = Action(ActionKind.IDLE)
    rows.append(_card(episode, "request-partial", history, first, idle, split))
    history.append(CompletedTurn(first, idle))

    second = _event(2, EventSource.USER, complete)
    first_action = (
        Action(ActionKind.TOOL, "delegate", {"task": task})
        if combined
        else Action(ActionKind.TOOL, "web_search", {"query": query})
    )
    rows.append(_card(episode, "request-complete", history, second, first_action, split))
    history.append(CompletedTurn(second, first_action))

    next_index = 3
    if combined:
        accepted_delegate = _event(
            next_index,
            EventSource.TOOL,
            json.dumps({"status": "accepted"}, separators=(",", ":")),
            tool="delegate",
            job="job-2",
        )
        search = Action(ActionKind.TOOL, "web_search", {"query": query})
        rows.append(_card(episode, "delegate-accepted-search", history, accepted_delegate, search, split))
        history.append(CompletedTurn(accepted_delegate, search))
        next_index += 1

    accepted_search = _event(
        next_index,
        EventSource.TOOL,
        json.dumps({"job_id": job_search, "status": "accepted"}, separators=(",", ":")),
        tool="web_search",
        job=job_search,
    )
    rows.append(_card(episode, "search-accepted", history, accepted_search, idle, split))
    history.append(CompletedTurn(accepted_search, idle))
    next_index += 1

    if combined:
        completed_delegate = _event(
            next_index,
            EventSource.TOOL,
            json.dumps({"status": "completed", "task": task}, separators=(",", ":")),
            tool="delegate",
            job="job-2",
        )
        rows.append(_card(episode, "delegate-completed", history, completed_delegate, idle, split))
        history.append(CompletedTurn(completed_delegate, idle))
        next_index += 1

    failed = number % 11 == 0
    payload: dict[str, Any]
    role: str
    if failed:
        payload = {
            "job_id": job_search,
            "status": "failed",
            "query": query,
            "error": "Search provider temporarily unavailable",
        }
        message = "I couldn't complete that search because the provider was unavailable."
        role = "search-failed"
    else:
        payload = {
            "job_id": job_search,
            "status": "completed",
            "query": query,
            "results": [{"title": title, "url": "https://example.com/source", "snippet": fact}],
        }
        message = f"According to {title}, {fact[0].lower() + fact[1:]}"
        role = "search-completed"
    completed_search = _event(
        next_index,
        EventSource.TOOL,
        json.dumps(payload, separators=(",", ":")),
        tool="web_search",
        job=job_search,
    )
    respond = Action(ActionKind.RESPOND, target=next_index, message=message)
    rows.append(_card(episode, role, history, completed_search, respond, split))
    history.append(CompletedTurn(completed_search, respond))
    next_index += 1

    unchanged = _event(next_index, EventSource.USER, complete)
    rows.append(_card(episode, "delivered-idle", history, unchanged, idle, split))
    return rows


def main() -> None:
    train = [row for path in TRAIN_PATHS for row in _load_jsonl(path)]
    dev = _load_jsonl(DEV_PATH)

    train_translation = [convert_demo3(row) for row in train if row.get("demo") == "demo-3"]
    dev_translation = [convert_demo3(row) for row in dev if row.get("demo") == "demo-3"]

    replay_pool = [row for row in train if row.get("demo") in {"demo-1", "demo-2", "demo-4"}]
    replay_pool.sort(key=lambda row: _rank(f"replay:{row['candidate_id']}"))
    train_replay = [{**row, "v2_group": "replay"} for row in replay_pool[:800]]
    dev_replay_pool = [row for row in dev if row.get("demo") in {"demo-1", "demo-2", "demo-4"}]
    dev_replay_pool.sort(key=lambda row: _rank(f"dev-replay:{row['candidate_id']}"))
    dev_replay = [{**row, "v2_group": "replay"} for row in dev_replay_pool[:180]]

    train_search = [
        row
        for number in range(160)
        for row in build_search_episode(number, "train", combined=number % 2 == 1)
    ]
    dev_search = [
        row
        for number in range(160, 200)
        for row in build_search_episode(number, "dev", combined=number % 2 == 1)
    ]

    train_rows, dropped_train_duplicates = _dedupe(
        [*train_translation, *train_search, *train_replay]
    )
    dev_rows, dropped_dev_duplicates = _dedupe(
        [*dev_translation, *dev_search, *dev_replay]
    )
    train_rows.sort(key=lambda row: _rank(f"train:{row['candidate_id']}"))
    dev_rows.sort(key=lambda row: _rank(f"dev:{row['candidate_id']}"))

    train_prompts = {str(row["prompt"]) for row in train_rows}
    dropped_dev_overlap = sum(str(row["prompt"]) in train_prompts for row in dev_rows)
    dev_rows = [row for row in dev_rows if str(row["prompt"]) not in train_prompts]
    dev_prompts = {str(row["prompt"]) for row in dev_rows}
    overlap = train_prompts & dev_prompts
    if overlap:
        raise ValueError(f"Train/dev prompt overlap: {len(overlap)}")
    if len(train_prompts) != len(train_rows) or len(dev_prompts) != len(dev_rows):
        raise ValueError("Duplicate prompts in g1-v2 build")

    train_path = OUTPUT_DIR / "train.jsonl"
    dev_path = OUTPUT_DIR / "dev.jsonl"
    train_sha = _write_jsonl(train_path, train_rows)
    dev_sha = _write_jsonl(dev_path, dev_rows)
    manifest = {
        "schema_version": "g1-v2-adaptation-1",
        "source_checkpoint": "g1:state_epoch1",
        "demo_5_included": False,
        "train": {
            "path": str(train_path.relative_to(ROOT)),
            "rows": len(train_rows),
            "sha256": train_sha,
            "groups": dict(sorted(Counter(str(row["v2_group"]) for row in train_rows).items())),
            "classes": dict(sorted(Counter(str(row["expected_class"]) for row in train_rows).items())),
        },
        "dev": {
            "path": str(dev_path.relative_to(ROOT)),
            "rows": len(dev_rows),
            "sha256": dev_sha,
            "groups": dict(sorted(Counter(str(row["v2_group"]) for row in dev_rows).items())),
            "classes": dict(sorted(Counter(str(row["expected_class"]) for row in dev_rows).items())),
        },
        "train_dev_prompt_overlap": 0,
        "dropped_dev_rows_after_contract_conversion": dropped_dev_overlap,
        "dropped_duplicate_train_prompts": dropped_train_duplicates,
        "dropped_duplicate_dev_prompts": dropped_dev_duplicates,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
