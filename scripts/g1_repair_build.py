"""Build the exact 80-card g1 Demo 5 micro-SFT repair corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from train.g1_repair import build_repair_rows, pack_repair_batches


ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATHS = tuple(ROOT / "data" / f"train_g1-{index:05d}-of-00004.jsonl" for index in range(1, 5))
DEV_PATH = ROOT / "data" / "dev_g1.jsonl"
OUTPUT = ROOT / "data" / "tinker" / "g1-repair-train.jsonl"
MANIFEST = ROOT / "artifacts" / "g1-repair" / "manifest.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    train_rows = [row for path in TRAIN_PATHS for row in _load_jsonl(path)]
    rows, manifest = build_repair_rows(train_rows, _load_jsonl(DEV_PATH))
    batches = pack_repair_batches(rows)
    ordered = [row for batch in batches for row in batch]
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered)
    manifest.update(
        {
            "output": str(OUTPUT),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "batches": [
                {
                    "index": index,
                    "rows": len(batch),
                    "groups": {
                        group: sum(row["repair_group"] == group for row in batch)
                        for group in sorted({str(row["repair_group"]) for row in batch})
                    },
                }
                for index, batch in enumerate(batches, start=1)
            ],
        }
    )
    _write(OUTPUT, payload)
    _write(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
