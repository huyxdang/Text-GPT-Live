"""Merge two blind Codex-agent semantic reviews and root adjudication.

This is an offline report assembly step: it performs no model or network calls.
The strict evaluator remains authoritative for action kind, routing, timing, and
source-span anchors through ``apply_semantic_judgments``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from train.g1_semantic_evaluation import apply_semantic_judgments


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BATCH_DIR = ROOT / "data" / "tinker" / "semantic-review-batches"
DEFAULT_SOURCE = ROOT / "data" / "tinker" / "dev-g1-epoch1_eval.json"
DEFAULT_DATA = ROOT / "data" / "dev_g1.jsonl"
DEFAULT_ADJUDICATION = DEFAULT_BATCH_DIR / "adjudication.json"
DEFAULT_JUDGMENTS = ROOT / "data" / "tinker" / "dev-g1-epoch1_codex_judgments.json"
DEFAULT_OUTPUT = ROOT / "data" / "tinker" / "dev-g1-epoch1_hybrid_eval.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_review(path: Path) -> tuple[str, dict[int, Mapping[str, Any]]]:
    value = _load_json(path)
    reviewer = value.get("reviewer")
    raw_judgments = value.get("judgments")
    if not isinstance(reviewer, str) or not isinstance(raw_judgments, Mapping):
        raise ValueError(f"Invalid review file: {path}")
    judgments: dict[int, Mapping[str, Any]] = {}
    for key, judgment in raw_judgments.items():
        if not isinstance(judgment, Mapping) or not isinstance(judgment.get("pass"), bool):
            raise ValueError(f"Invalid judgment {key!r} in {path}")
        judgments[int(key)] = judgment
    return reviewer, judgments


def _collect_reviews(batch_dir: Path) -> tuple[
    set[int],
    dict[int, tuple[str, Mapping[str, Any]]],
    dict[int, tuple[str, Mapping[str, Any]]],
    list[str],
]:
    expected: set[int] = set()
    first: dict[int, tuple[str, Mapping[str, Any]]] = {}
    second: dict[int, tuple[str, Mapping[str, Any]]] = {}
    source_files: list[str] = []

    for batch_number in (1, 2, 3):
        batch_path = batch_dir / f"batch-{batch_number}.json"
        batch = _load_json(batch_path)
        if not isinstance(batch, list):
            raise ValueError(f"Invalid case batch: {batch_path}")
        expected.update(int(case["row_index"]) for case in batch)

        first_path = batch_dir / f"batch-{batch_number}-judgments.json"
        cross_paths = sorted(batch_dir.glob(f"batch-{batch_number}-cross-by-*.json"))
        if len(cross_paths) != 1:
            raise ValueError(f"Expected one cross-review for batch {batch_number}.")

        first_reviewer, first_judgments = _load_review(first_path)
        second_reviewer, second_judgments = _load_review(cross_paths[0])
        source_files.extend((str(first_path), str(cross_paths[0])))
        for row_index, judgment in first_judgments.items():
            if row_index in first:
                raise ValueError(f"Duplicate first-pass row {row_index}.")
            first[row_index] = (first_reviewer, judgment)
        for row_index, judgment in second_judgments.items():
            if row_index in second:
                raise ValueError(f"Duplicate second-pass row {row_index}.")
            second[row_index] = (second_reviewer, judgment)

    if set(first) != expected or set(second) != expected:
        raise ValueError("Both review passes must cover every semantic case exactly once.")
    return expected, first, second, source_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_ADJUDICATION)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    expected, first, second, source_files = _collect_reviews(args.batch_dir)
    adjudication_document = _load_json(args.adjudication)
    adjudications = adjudication_document.get("judgments")
    if not isinstance(adjudications, Mapping):
        raise ValueError("Adjudication file has no judgments object.")

    disagreements = {
        row_index
        for row_index in expected
        if first[row_index][1]["pass"] != second[row_index][1]["pass"]
    }
    if {int(key) for key in adjudications} != disagreements:
        raise ValueError("Adjudications must match the disagreement rows exactly.")

    final_judgments: dict[str, dict[str, Any]] = {}
    final_passes = 0
    for row_index in sorted(expected):
        first_reviewer, first_judgment = first[row_index]
        second_reviewer, second_judgment = second[row_index]
        agreed = first_judgment["pass"] == second_judgment["pass"]
        adjudication = adjudications.get(str(row_index))
        final_pass = bool(first_judgment["pass"] if agreed else adjudication["pass"])
        final_passes += int(final_pass)
        final_judgments[str(row_index)] = {
            "pass": final_pass,
            "reviewer_agreement": agreed,
            "votes": [
                {"reviewer": first_reviewer, **dict(first_judgment)},
                {"reviewer": second_reviewer, **dict(second_judgment)},
            ],
            "adjudication": None if agreed else dict(adjudication),
        }

    judgments_document = {
        "schema_version": "g1-semantic-codex-review-1",
        "provider": "codex-subagents",
        "review_design": "two blind reviews per case with root adjudication",
        "case_count": len(expected),
        "agreement_count": len(expected) - len(disagreements),
        "agreement_rate": round((len(expected) - len(disagreements)) / len(expected), 6),
        "disagreement_count": len(disagreements),
        "semantic_pass_count": final_passes,
        "semantic_fail_count": len(expected) - final_passes,
        "source_files": source_files,
        "adjudication_path": str(args.adjudication),
        "judgments": final_judgments,
    }
    _write_json(args.judgments, judgments_document)

    report = apply_semantic_judgments(
        _load_jsonl(args.data),
        _load_json(args.source_report),
        final_judgments,
    )
    report["semantic_judge"].update(
        {
            "provider": "codex-subagents",
            "review_design": judgments_document["review_design"],
            "reviewers_per_case": 2,
            "agreement_count": judgments_document["agreement_count"],
            "agreement_rate": judgments_document["agreement_rate"],
            "disagreement_count": judgments_document["disagreement_count"],
            "adjudicator": adjudication_document.get("adjudicator"),
            "calibration_status": "provisional-human-unreviewed",
            "source_report": str(args.source_report),
            "judgments_path": str(args.judgments),
        }
    )
    _write_json(args.output, report)

    summary = report["summary"]
    print(
        f"[codex-semantic] cases={len(expected)} agreement="
        f"{judgments_document['agreement_count']}/{len(expected)} "
        f"semantic_pass={final_passes}/{len(expected)}"
    )
    print(
        f"[codex-semantic] strict={summary['strict_row_accuracy']} "
        f"hybrid={summary['hybrid_row_accuracy']} "
        f"clause_hybrid={summary['hybrid_clause_boundary_accuracy']} "
        f"gates={report['hybrid_hard_gates']['passed']}"
    )
    print(f"[codex-semantic] report={args.output}")


if __name__ == "__main__":
    main()
