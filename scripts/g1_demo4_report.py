"""Validate agent-authored Demo 4 banks and print a distribution report.

    .venv/bin/python -m scripts.g1_demo4_report --enforce-distribution

Exits non-zero when validation fails, so it doubles as the tranche-acceptance
gate for the authoring fleet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datagen.g1_authored_demo4 import (  # noqa: E402
    DEMO,
    authored_paths,
    load_authored_batches,
    validate_demo4_batches,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "g1_authored")
    parser.add_argument("--demo", default=DEMO, choices=[DEMO])
    parser.add_argument("--enforce-distribution", action="store_true")
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Treat heuristic semantic warnings as acceptance failures.",
    )
    args = parser.parse_args()

    paths = authored_paths(args.root, args.demo)
    if not paths:
        raise SystemExit(f"No authored JSON files found for {args.demo} under {args.root}")
    report = validate_demo4_batches(
        load_authored_batches(paths),
        enforce_distribution=args.enforce_distribution,
    )
    # requests/progress_pairs carry full authored records for the build CLI;
    # they are large and not meant for eyeballing in the report output.
    printable = {key: value for key, value in report.items() if key not in {"requests", "progress_pairs"}}
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed"] or (args.fail_on_warnings and report["warnings"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
