"""Evaluate the selected Tinker g1 sampler on the frozen 43-card pilot."""

from __future__ import annotations

import json
from pathlib import Path

from train.g1_evaluation import evaluate_g1_predictions
from train.tinker_run import get_tokenizer, load_env_key, sample_outputs


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    state = json.loads((ROOT / "data" / "tinker" / "run_state.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (ROOT / "data" / "pilot_g1.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    load_env_key()
    import tinker

    service = tinker.ServiceClient(user_metadata={"project": "smol-g1-pilot-eval"})
    sampler = service.create_sampling_client(model_path=state["g1:sampler_path"])
    tokenizer = get_tokenizer(state["g1:base_model"])
    outputs = sample_outputs(sampler, tokenizer, rows, label="g1-epoch1-pilot", seed=650)
    report = evaluate_g1_predictions(rows, outputs, label="g1-epoch1-pilot")
    report["provenance"] = {
        "sampler_path": state["g1:sampler_path"],
        "base_model": state["g1:base_model"],
        "seed": 650,
        "pilot_rows": len(rows),
    }
    output = ROOT / "data" / "tinker" / "pilot-g1-epoch1_eval.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"report: {output}", flush=True)


if __name__ == "__main__":
    main()
