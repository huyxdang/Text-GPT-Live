"""LLM-judge the open-ended payloads in a saved g1 evaluation report.

Exact matching remains in the report as a diagnostic.  This command asks an
LLM only about non-exact response text, translation text, delegate task text,
and edit replacements after all deterministic action/routing/span checks pass.

    .venv/bin/python -m scripts.g1_semantic_eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from train.g1_semantic_evaluation import apply_semantic_judgments, build_semantic_cases


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "data" / "tinker" / "dev-g1-epoch1_eval.json"
DEFAULT_DATA = ROOT / "data" / "dev_g1.jsonl"
DEFAULT_JUDGMENTS = ROOT / "data" / "tinker" / "dev-g1-epoch1_semantic_judgments.json"
DEFAULT_OUTPUT = ROOT / "data" / "tinker" / "dev-g1-epoch1_hybrid_eval.json"
DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
JUDGE_SYSTEM_PROMPT = """You are a strict semantic evaluator for a streaming interaction policy.
The deterministic evaluator has already established that the candidate chose the correct action
kind and all required routing/source-span anchors. Judge only the open-ended payload wording.

Exact wording is not required. Pass a candidate only when it preserves the reference meaning,
fulfills the instruction visible in the stream, is grounded without invented or contradictory
facts, uses the required language and suitable register, and is concise enough for this action.
For suggest_edit, the candidate replacement must actually correct the quoted text without changing
its intended meaning. Do not judge whether the action should have happened; that is deterministic.

Return one JSON object with exactly these fields:
{"pass":boolean,"meaning_preserved":boolean,"instruction_fulfilled":boolean,
"grounded":boolean,"language_register":boolean,"reason":"brief explanation"}"""


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
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read JSONL from {path}: {exc}") from exc


def _fingerprint(case: Mapping[str, Any], *, judge_identity: str) -> str:
    payload = json.dumps(
        {"case": case, "judge_identity": judge_identity},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validated_verdict(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Judge response is not a JSON object.")
    boolean_fields = (
        "pass",
        "meaning_preserved",
        "instruction_fulfilled",
        "grounded",
        "language_register",
    )
    if any(not isinstance(value.get(field), bool) for field in boolean_fields):
        raise ValueError("Judge response is missing a required boolean field.")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Judge response is missing a reason.")
    return {field: bool(value[field]) for field in boolean_fields} | {"reason": reason.strip()[:500]}


class OpenAISemanticJudge:
    def __init__(self, *, model: str, api_url: str, timeout_seconds: float = 90.0) -> None:
        import httpx

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required for semantic judging.")
        self.model = model
        self.resolved_model: str | None = None
        self.client = httpx.Client(
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.api_url = api_url

    def judge(self, case: Mapping[str, Any]) -> dict[str, Any]:
        import httpx

        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "payload_kind": case["payload_kind"],
                            "stream_context": case["context"],
                            "reference_payload": case["reference"],
                            "candidate_payload": case["candidate"],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.post(self.api_url, json=payload)
                response.raise_for_status()
                body = response.json()
                self.resolved_model = str(body.get("model") or self.resolved_model or self.model)
                content = body["choices"][0]["message"]["content"]
                return _validated_verdict(json.loads(content))
            except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Semantic judge failed after three attempts: {last_error}")

    def close(self) -> None:
        self.client.close()


class LocalMLXSemanticJudge:
    """On-device fallback that never exports evaluation streams."""

    def __init__(self, *, model_path: Path, max_tokens: int = 240) -> None:
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        if not model_path.is_dir():
            raise SystemExit(f"Local judge model does not exist: {model_path}")
        self.model_path = model_path
        self.model, self.tokenizer = load(str(model_path))
        self.sampler = make_sampler(temp=0.0)
        self.max_tokens = max_tokens
        self.resolved_model = str(model_path)

    def judge(self, case: Mapping[str, Any]) -> dict[str, Any]:
        from mlx_lm import stream_generate

        user_content = json.dumps(
            {
                "payload_kind": case["payload_kind"],
                "stream_context": case["context"],
                "reference_payload": case["reference"],
                "candidate_payload": case["candidate"],
            },
            ensure_ascii=False,
        )
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        pieces: list[str] = []
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=self.max_tokens,
            sampler=self.sampler,
        ):
            pieces.append(response.text)
            text = "".join(pieces).strip()
            try:
                return _validated_verdict(json.loads(text))
            except (json.JSONDecodeError, ValueError):
                continue
        text = "".join(pieces).strip()
        raise RuntimeError(f"Local semantic judge did not return valid JSON: {text[:300]!r}")

    def close(self) -> None:
        return None


def _load_saved_judgments(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": "g1-semantic-judge-1", "judgments": {}}
    value = _load_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != "g1-semantic-judge-1":
        raise SystemExit(f"Unsupported semantic judgment file: {path}")
    if not isinstance(value.get("judgments"), dict):
        raise SystemExit(f"Semantic judgment file has no judgments object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provider", choices=("local-mlx", "openai"), default="local-mlx")
    parser.add_argument("--model", default=os.getenv("G1_JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--local-model",
        type=Path,
        default=ROOT / "models" / "Qwen3.5-4B",
    )
    parser.add_argument("--api-url", default=os.getenv("OPENAI_API_URL", DEFAULT_API_URL))
    parser.add_argument("--context-chars", type=int, default=12_000)
    parser.add_argument("--votes", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    strict_report = _load_json(args.source_report)
    pairs = _load_jsonl(args.data)
    cases = build_semantic_cases(pairs, strict_report, context_chars=args.context_chars)
    requested_model = args.model if args.provider == "openai" else str(args.local_model)
    judge_identity = f"{args.provider}:{requested_model}"
    saved = _load_saved_judgments(args.judgments)
    saved_judgments: dict[str, Any] = saved["judgments"]
    reusable = sum(
        1
        for case in cases
        if (entry := saved_judgments.get(str(case["row_index"])))
        and entry.get("fingerprint") == _fingerprint(case, judge_identity=judge_identity)
        and len(entry.get("votes") or []) >= args.votes
    )
    print(
        f"[semantic] cases={len(cases)} reusable={reusable} provider={args.provider} "
        f"model={args.model if args.provider == 'openai' else args.local_model} votes={args.votes}",
        flush=True,
    )
    if args.dry_run:
        return

    _load_env()
    judge = (
        OpenAISemanticJudge(model=args.model, api_url=args.api_url)
        if args.provider == "openai"
        else LocalMLXSemanticJudge(model_path=args.local_model)
    )
    try:
        processed = 0
        for case in cases:
            key = str(case["row_index"])
            fingerprint = _fingerprint(case, judge_identity=judge_identity)
            entry = saved_judgments.get(key)
            votes = list(entry.get("votes") or []) if entry and entry.get("fingerprint") == fingerprint else []
            while len(votes) < args.votes:
                votes.append(judge.judge(case))
            passing_votes = sum(bool(vote["pass"]) for vote in votes[: args.votes])
            # A tie with two judges fails closed and is exposed as disagreement.
            semantic_pass = passing_votes > args.votes / 2
            saved_judgments[key] = {
                "row_index": case["row_index"],
                "episode": case["episode"],
                "payload_kind": case["payload_kind"],
                "fingerprint": fingerprint,
                "pass": semantic_pass,
                "disagreement": 0 < passing_votes < args.votes,
                "votes": votes,
            }
            saved.update(
                {
                    "requested_model": requested_model,
                    "resolved_model": judge.resolved_model,
                    "provider": args.provider,
                    "votes_per_case": args.votes,
                    "judgments": saved_judgments,
                }
            )
            _write_json(args.judgments, saved)
            processed += 1
            if processed % 10 == 0 or processed == len(cases):
                print(f"[semantic] judged {processed}/{len(cases)}", flush=True)
            if args.max_cases is not None and processed >= args.max_cases:
                print("[semantic] stopped at --max-cases; no complete hybrid report written.")
                return
    finally:
        judge.close()

    verdicts = {
        str(case["row_index"]): saved_judgments[str(case["row_index"])]
        for case in cases
        if str(case["row_index"]) in saved_judgments
        and saved_judgments[str(case["row_index"])].get("fingerprint")
        == _fingerprint(case, judge_identity=judge_identity)
    }
    report = apply_semantic_judgments(pairs, strict_report, verdicts)
    report["semantic_judge"].update(
        {
            "requested_model": requested_model,
            "resolved_model": saved.get("resolved_model"),
            "provider": args.provider,
            "votes_per_case": args.votes,
            "calibration_status": "uncalibrated",
            "source_report": str(args.source_report),
            "judgments_path": str(args.judgments),
        }
    )
    _write_json(args.output, report)
    summary = report["summary"]
    print(
        f"[semantic] strict={summary['strict_row_accuracy']} "
        f"hybrid={summary['hybrid_row_accuracy']} "
        f"clause_hybrid={summary['hybrid_clause_boundary_accuracy']} "
        f"gates={report['hybrid_hard_gates']['passed']}",
        flush=True,
    )
    print(f"[semantic] report={args.output}", flush=True)


if __name__ == "__main__":
    main()
