"""Probe whether the selected g1 checkpoint recognizes fire-now when shown A/B.

The normal policy generation already chose idle on every epoch-1 fire row.
This diagnostic changes the task to recognition and presents each row twice
with reversed candidate order.  Strong order-invariant recognition supports a
short symmetric DPO pass; weak recognition calls for targeted SFT first.

    .venv/bin/python -m scripts.g1_fire_recognition_probe
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from train.g1_recognition import (
    RECOGNITION_SYSTEM_PROMPT,
    build_fire_recognition_cases,
    score_fire_recognition,
)


ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "data" / "tinker" / "run_state.json"
DEFAULT_DATA = ROOT / "data" / "dev_g1.jsonl"
DEFAULT_SOURCE = ROOT / "data" / "tinker" / "dev-g1-epoch1_eval.json"
DEFAULT_OUTPUT = ROOT / "data" / "tinker" / "dev-g1-epoch1_fire_recognition.json"


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read JSON from {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read JSONL from {path}: {exc}") from exc


def _failed_fire_indexes(source: Path, pairs: list[dict[str, Any]]) -> set[int]:
    report = _load_json(source)
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(pairs):
        raise SystemExit("Source report does not align with the probe dataset.")
    failed: set[int] = set()
    for index, (row, pair) in enumerate(zip(rows, pairs, strict=True)):
        if row.get("row_index") != index or row.get("episode") != pair.get("episode"):
            raise SystemExit(f"Source report row {index} does not align with the probe dataset.")
        if pair.get("should_fire") and row.get("predicted_class") != "respond":
            failed.add(index)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tag", default="g1")
    parser.add_argument("--seed", type=int, default=650)
    parser.add_argument("--presentations", type=int, choices=(1, 2), default=2)
    parser.add_argument("--context-chars", type=int, default=12_000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pairs = _load_jsonl(args.data)
    failed = _failed_fire_indexes(args.source_report, pairs)
    cases = build_fire_recognition_cases(
        pairs,
        eligible_row_indexes=failed,
        seed=args.seed,
        presentations_per_item=args.presentations,
        context_chars=args.context_chars,
    )
    print(
        f"[recognition] failed_fire_items={len(failed)} presentations={len(cases)}",
        flush=True,
    )
    if args.dry_run:
        return
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive.")

    _load_env()
    state = _load_json(STATE_PATH)
    sampler_path = state.get(f"{args.tag}:sampler_path")
    base_model = state.get(f"{args.tag}:base_model")
    if not isinstance(sampler_path, str) or not sampler_path:
        raise SystemExit(f"No {args.tag}:sampler_path in {STATE_PATH}.")
    if not isinstance(base_model, str) or not base_model:
        raise SystemExit(f"No {args.tag}:base_model in {STATE_PATH}.")

    import tinker
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    service = tinker.ServiceClient(user_metadata={"project": "smol-g1-fire-recognition"})
    sampler = service.create_sampling_client(model_path=sampler_path)
    params = tinker.types.SamplingParams(
        max_tokens=4,
        temperature=0.0,
        stop=["\n"],
        seed=args.seed,
    )
    outputs = [""] * len(cases)
    started = time.monotonic()
    for start in range(0, len(cases), args.concurrency):
        chunk = list(enumerate(cases[start : start + args.concurrency], start=start))
        futures = []
        for index, case in chunk:
            messages = [
                {"role": "system", "content": RECOGNITION_SYSTEM_PROMPT},
                {"role": "user", "content": case["prompt"]},
            ]
            ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            futures.append(
                (
                    index,
                    sampler.sample(
                        prompt=tinker.ModelInput.from_ints(list(ids)),
                        num_samples=1,
                        sampling_params=params,
                    ),
                )
            )
        for index, future in futures:
            response = future.result(timeout=600)
            outputs[index] = tokenizer.decode(
                response.sequences[0].tokens,
                skip_special_tokens=True,
            ).strip()
        print(f"[recognition] sampled {min(start + args.concurrency, len(cases))}/{len(cases)}", flush=True)

    report = score_fire_recognition(cases, outputs)
    report["provenance"] = {
        "sampler_path": sampler_path,
        "base_model": base_model,
        "source_report": str(args.source_report),
        "seed": args.seed,
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"[recognition] report={args.output}", flush=True)


if __name__ == "__main__":
    main()
