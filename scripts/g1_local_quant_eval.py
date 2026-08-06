"""Evaluate one local quantized model on the frozen 43-card g1 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from app.policy import SYSTEM_PROMPT_G1
from scripts.local_model import LocalMLXPolicy
from train.g1_evaluation import evaluate_g1_predictions


ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "data" / "pilot_g1.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in PILOT.read_text(encoding="utf-8").splitlines() if line]
    policy = LocalMLXPolicy(model_path=str(args.model), system_prompt=SYSTEM_PROMPT_G1)
    outputs: list[str] = []
    latencies: list[float] = []
    for index, row in enumerate(rows, start=1):
        started = time.perf_counter()
        outputs.append(policy.complete(row["prompt"]))
        latencies.append(round((time.perf_counter() - started) * 1000, 3))
        print(
            f"[local-pilot] {index}/{len(rows)} {row['demo']} {row['situation']} "
            f"{latencies[-1]:.1f}ms {outputs[-1]}",
            flush=True,
        )
    report = evaluate_g1_predictions(rows, outputs, label=f"local-pilot:{args.model.name}")
    for scored, latency in zip(report["rows"], latencies, strict=True):
        scored["latency_ms"] = latency
    report["local_provenance"] = {
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        ),
        "model_path": str(args.model.resolve()),
        "weight_sha256": {
            path.name: _sha256(path) for path in sorted(args.model.glob("*.safetensors"))
        },
        "pilot_path": str(PILOT.resolve()),
        "pilot_sha256": _sha256(PILOT),
        "decoding": "greedy model generation",
        "persistent_cache": True,
        "deterministic_action_substitution": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
