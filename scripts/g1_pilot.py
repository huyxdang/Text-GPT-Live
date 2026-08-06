"""Build and validate the deterministic g1 pilot batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.policy import SYSTEM_PROMPT_G1
from app.stream import compile_stream
from datagen.g1_pilot import build_pilot_cards, validate_pilot_coverage
from train.g1_evaluation import evaluate_g1_predictions
from train.tinker_run import completion_class, example_weight_for_pair, system_prompt_for_pair


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "pilot_g1.jsonl"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "g1-pilot"


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=indent,
        separators=(",", ":") if indent is None else None,
        sort_keys=True,
    )


def _inspection_markdown(rows: list[dict[str, Any]]) -> str:
    samples: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["situation"] == "ordinary-silence":
            key = f"ordinary-silence/{row['demo']}/{row['empty_kind']}"
        elif row["demo"] == "demo-5":
            key = f"{row['situation']}/{row['timing_boundary']}"
        elif row["situation"].startswith("translation-"):
            key = str(row["episode"])
        else:
            key = str(row["situation"])
        samples.setdefault(key, row)
    lines = [
        "# g1 pilot inspection samples",
        "",
        "One deterministic representative per meaningful coverage class.",
        "",
    ]
    for coverage_class, row in sorted(samples.items()):
        lines.extend(
            [
                f"## {coverage_class}",
                "",
                f"- Episode: `{row['episode']}`",
                f"- Demo: `{row['demo']}`",
                f"- Empty current text: `{str(row['current_content_empty']).lower()}`",
                f"- Expected: `{row['expected_class']}`",
                "",
                "```text",
                row["prompt"],
                "```",
                "",
                "```text",
                row["completion"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def run_pilot(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    out_dir: Path = DEFAULT_OUT_DIR,
    require_reference_review: bool = False,
    inspection_ack: str | None = None,
) -> dict[str, Any]:
    if not inspection_ack or not inspection_ack.strip():
        raise ValueError("A non-empty manual inspection acknowledgement is required.")
    cards = build_pilot_cards()
    coverage = validate_pilot_coverage(
        cards,
        require_reference_review=require_reference_review,
    )
    if not coverage["passed"]:
        raise RuntimeError("g1 pilot coverage failed: " + "; ".join(coverage["errors"]))

    rows = [card.row for card in cards]
    for card in cards:
        if card.row["prompt"] != compile_stream(
            list(card.history), card.current, fmt="g1"
        ):
            raise RuntimeError(f"{card.row['episode']}: generator/runtime prompt mismatch")
        if system_prompt_for_pair(card.row) != SYSTEM_PROMPT_G1:
            raise RuntimeError(f"{card.row['episode']}: training system prompt mismatch")
        if completion_class(card.row) != card.row["expected_class"]:
            raise RuntimeError(f"{card.row['episode']}: training completion parser mismatch")
        if example_weight_for_pair(card.row) != 1.0:
            raise RuntimeError(f"{card.row['episode']}: g1 card weight is not 1.0")

    payload = "".join(_json(row) + "\n" for row in rows)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(payload, encoding="utf-8")

    gold_report = evaluate_g1_predictions(
        rows,
        [row["completion"] for row in rows],
        label="g1-pilot-gold",
    )
    if not gold_report["hard_gates"]["passed"]:
        raise RuntimeError("g1 pilot gold answers failed their own grader")

    always_idle_report = evaluate_g1_predictions(
        rows,
        ["<action>idle()</action>"] * len(rows),
        label="g1-pilot-always-idle",
    )
    if always_idle_report["hard_gates"]["passed"]:
        raise RuntimeError("always-idle baseline incorrectly passed the g1 pilot")
    if always_idle_report["summary"]["should_fire_recall"] != 0.0:
        raise RuntimeError("always-idle baseline should have zero should-fire recall")

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        dataset_label = str(dataset_path.relative_to(ROOT))
    except ValueError:
        dataset_label = str(dataset_path)
    manifest = {
        "schema_version": "g1-pilot-1",
        "dataset": dataset_label,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "rows": len(rows),
        "coverage": coverage,
        "gold_summary": gold_report["summary"],
        "always_idle_summary": always_idle_report["summary"],
        "reference_review_required_for_release": True,
        "reference_review_enforced": require_reference_review,
        "manual_inspection": {
            "status": "acknowledged",
            "acknowledgement": inspection_ack.strip(),
        },
    }
    (out_dir / "manifest.json").write_text(_json(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "gold_eval.json").write_text(
        _json(gold_report, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "always_idle_eval.json").write_text(
        _json(always_idle_report, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "inspection_samples.md").write_text(
        _inspection_markdown(rows), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--require-reference-review",
        action="store_true",
        help="Fail if any Chinese reference lacks an approved language review.",
    )
    parser.add_argument(
        "--inspection-ack",
        required=True,
        help="Who inspected the generated representative samples, with date or run ID.",
    )
    args = parser.parse_args()
    manifest = run_pilot(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        require_reference_review=args.require_reference_review,
        inspection_ack=args.inspection_ack,
    )
    print(_json(manifest["coverage"]["counts"], indent=2))
    print(
        "gold strict_row_accuracy="
        f"{manifest['gold_summary']['strict_row_accuracy']:.3f}"
    )
    print(
        "always_idle should_fire_recall="
        f"{manifest['always_idle_summary']['should_fire_recall']:.3f}"
    )
    for warning in manifest["coverage"]["warnings"]:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
