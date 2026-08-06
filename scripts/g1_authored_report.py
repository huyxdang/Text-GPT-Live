"""Validate agent-authored g1 situation batches and print a distribution report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datagen.g1_authored import (  # noqa: E402
    authored_paths,
    load_authored_batches,
    validate_demo1_batches,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "g1_authored")
    parser.add_argument("--demo", default="demo-1", choices=["demo-1"])
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
    report = validate_demo1_batches(
        load_authored_batches(paths),
        enforce_distribution=args.enforce_distribution,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed"] or (args.fail_on_warnings and report["warnings"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
