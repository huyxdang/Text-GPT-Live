"""Train and evaluate the legacy and g1 interaction models through Tinker.

The normal performance loop is:

    python -m datagen.generate
    python -m train.tinker_run --stage loop --tag v4

`loop` checks connectivity, measures the stock model on the reusable dev set,
trains LoRA, evaluates dev after every epoch, selects the best epoch, and only
then evaluates that checkpoint on the frozen dense test set. Every remote path
and every report is persisted as soon as it exists.

The fresh g1 lineage is a separate, fail-closed path:

    python -m scripts.g1_full_build
    python -m train.tinker_run --stage g1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from random import Random
from typing import Any, Mapping

from app.tls import maybe_use_system_certs

maybe_use_system_certs()

from app.policy import SYSTEM_PROMPT_G1, SYSTEM_PROMPT_V4, SYSTEM_PROMPT_V5, SYSTEM_PROMPT_V6
from app.stream import parse_action, parse_g1_action
from train.evaluation import V6_HARD_GATES, action_class, evaluate_predictions
from train.g1_evaluation import evaluate_g1_predictions

# The v5/v6 stages render every prompt (training and evaluation) under their
# own system prompt so serving matches training; earlier stages keep v4 exactly.
ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT_V4

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "tinker"
STATE_PATH = OUT_DIR / "run_state.json"
BASE_MODEL = "Qwen/Qwen3-8B"
LORA_RANK = 32
LEARNING_RATE = 2e-4
ADAPT_LEARNING_RATE = 7.5e-5
EPOCHS = 3
BATCH_SIZE = 16
SEED = 650
EVAL_CONCURRENCY = 8
MAX_TOKENS = 192
REMOTE_TIMEOUT_SECONDS = 600

# g1 lineage — round-1 configuration, locked in scripts/g1_runbook.md
# (2026-07-31). Every value below is quoted straight from that runbook; do
# not improvise on top of it. `--base-model` promotes the base model choice
# to a CLI flag (per the runbook's "BASE_MODEL... promote it to a CLI flag"
# instruction) without touching the legacy BASE_MODEL constant the v4-v6
# stages still use.
G1_BASE_MODEL = "Qwen/Qwen3.5-4B"
G1_LORA_RANK = 32
G1_TRAIN_ATTN = True
G1_TRAIN_MLP = True
G1_TRAIN_UNEMBED = True
G1_SEED = 650
G1_LEARNING_RATE = 2e-4
G1_ADAM_BETA1 = 0.9
G1_ADAM_BETA2 = 0.95
G1_ADAM_EPS = 1e-12
G1_WEIGHT_DECAY = 0.0
# tinker.AdamParams.grad_clip_norm is a float field (default 0.0 = disabled),
# not Optional — 0.0 is the "off" value the runbook calls for.
G1_GRAD_CLIP_NORM = 0.0
G1_EPOCHS = 3
G1_BATCH_SIZE = 16
TINKER_MAX_TARGET_TOKENS = 65_536
# Random(650 + 20 + epoch): the runbook's deterministic per-epoch shuffle.
# The +20 offset mirrors the existing v6-stage convention (SEED + 20 + epoch)
# so g1's shuffle sequence never collides with any other stage's.
G1_SHUFFLE_BASE_SEED = G1_SEED + 20

G1_DEMO1_MANIFEST_RELATIVE_PATH = Path("artifacts/g1-demo1/manifest.json")
G1_DEMO1_MANIFEST_SCHEMA = "g1-demo1-build-1"
G1_DEMO1_DATASET_NAMES = {
    "train_g1_demo1": "train",
    "dev_g1_demo1": "dev",
}

# The merged five-demo dataset another agent publishes under artifacts/g1-full/
# once the g1 dataset build finishes. Same manifest contract as Demo 1 (ordered
# jsonl train shards with per-shard + aggregate SHA256s, a single dev jsonl,
# byte/row counts everywhere) so both are validated by the same machinery
# below. `--stage g1` requires this manifest and fails closed when it is absent;
# the Demo 1 manifest remains available only through its explicit dataset names.
G1_FULL_MANIFEST_RELATIVE_PATH = Path("artifacts/g1-full/manifest.json")
G1_FULL_MANIFEST_SCHEMA = "g1-full-build-1"
G1_FULL_DATASET_NAMES = {
    "train_g1": "train",
    "dev_g1": "dev",
}

# Per-example loss is normalized by completion length first. The v4 baseline
# already has perfect action recall and poor precision, so equal class weights
# preserve the dataset's restraint signal instead of amplifying over-action.
# The v6 mix is far idler (82 % idle; generate_ui/web_search ~30 rows each), so
# the v6 action classes carry modest upweights — a round-1 knob, revisited from
# dev per-class F1. search/respond upweights also touch v4/v41 replay rows in
# the v6 stage; earlier tags are trained and never re-run.
CLASS_WEIGHTS = {
    "idle": 1.0,
    "highlight": 1.0,
    "search": 3.0,
    "respond": 2.0,
    "underline": 1.0,
    "pause": 1.0,
    "write": 1.0,
    "resume": 1.0,
    "revise": 1.0,
    "suggest_edit": 2.0,
    "generate_ui": 3.0,
    "delegate": 1.0,
    "web_search": 1.0,
    "translate_commit": 1.0,
}


def load_env_key() -> None:
    if os.environ.get("TINKER_API_KEY"):
        return
    env_path = ROOT / ".env"
    matches: list[str] = []
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("TINKER_API_KEY=", "TINER_API_KEY=")):
                matches.append(stripped.split("=", 1)[1].strip().strip("'\""))
    if matches:
        # dotenv semantics: the last declaration is authoritative.
        os.environ["TINKER_API_KEY"] = matches[-1]
        if len(matches) > 1:
            print(f"[env] warning: found {len(matches)} Tinker key declarations; using the last one.")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("No TINKER_API_KEY found in environment or .env")


def dataset_path(name: str) -> Path:
    filename = {
        "train": "train_v4.jsonl",
        "dev": "dev_v4.jsonl",
        "dev_dense": "dev_v4_dense.jsonl",
        "test": "test_v4.jsonl",
        "train_v41": "train_v41_challenge.jsonl",
        "dev_v41": "dev_v41_challenge.jsonl",
        "dev_v41_dense": "dev_v41_challenge_dense.jsonl",
        "test_v41": "test_v41_frozen.jsonl",
        "train_v5": "train_v5_cowrite.jsonl",
        "dev_v5": "dev_v5_cowrite.jsonl",
        "test_v5": "test_v5_frozen.jsonl",
        "train_v6": "train_v6.jsonl",
        "dev_v6": "dev_v6.jsonl",
        "train_v6_dagger": "train_v6_dagger.jsonl",
        "train_v6_dagger2": "train_v6_dagger2.jsonl",
        "train_v6_dagger3": "train_v6_dagger3.jsonl",
        "test_v6_restraint": "test_v6_restraint_frozen.jsonl",
        "pilot_g1": "pilot_g1.jsonl",
    }.get(name)
    if filename is None:
        raise ValueError(f"Unknown dataset split: {name}")
    return ROOT / "data" / filename


class _G1ManifestSpec:
    """One manifest contract: relative path, expected schema tag, the dataset
    names it answers for, and the label/build-hint used in error messages.

    Demo 1 and the merged five-demo build (artifacts/g1-full/) share this
    exact contract, so both are validated by the same functions below.
    """

    __slots__ = ("relative_path", "schema_version", "dataset_names", "label", "build_hint")

    def __init__(
        self,
        *,
        relative_path: Path,
        schema_version: str,
        dataset_names: Mapping[str, str],
        label: str,
        build_hint: str,
    ) -> None:
        self.relative_path = relative_path
        self.schema_version = schema_version
        self.dataset_names = dataset_names
        self.label = label
        self.build_hint = build_hint


_G1_DEMO1_SPEC = _G1ManifestSpec(
    relative_path=G1_DEMO1_MANIFEST_RELATIVE_PATH,
    schema_version=G1_DEMO1_MANIFEST_SCHEMA,
    dataset_names=G1_DEMO1_DATASET_NAMES,
    label="Demo 1",
    build_hint="python -m scripts.g1_demo1_build",
)

_G1_FULL_SPEC = _G1ManifestSpec(
    relative_path=G1_FULL_MANIFEST_RELATIVE_PATH,
    schema_version=G1_FULL_MANIFEST_SCHEMA,
    dataset_names=G1_FULL_DATASET_NAMES,
    label="merged g1",
    build_hint="the merged five-demo g1 dataset build (not published yet)",
)

_G1_MANIFEST_SPECS = (_G1_DEMO1_SPEC, _G1_FULL_SPEC)


def _g1_manifest_spec_for(name: str) -> "_G1ManifestSpec | None":
    for spec in _G1_MANIFEST_SPECS:
        if name in spec.dataset_names:
            return spec
    return None


def _g1_manifest(spec: _G1ManifestSpec) -> Mapping[str, Any]:
    path = ROOT / spec.relative_path
    if not path.exists():
        raise SystemExit(
            f"Missing {spec.label} manifest: {path}. Run: {spec.build_hint}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {spec.label} manifest {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SystemExit(f"{spec.label} manifest {path} must contain a JSON object.")
    if (
        value.get("schema_version") != spec.schema_version
        or value.get("dataset_schema") != "g1"
    ):
        raise SystemExit(
            f"Stale or unsupported {spec.label} manifest {path}: expected "
            f"schema_version={spec.schema_version!r} and dataset_schema='g1'."
        )
    return value


def _manifest_entry(
    spec: _G1ManifestSpec, manifest: Mapping[str, Any], name: str
) -> tuple[str, Mapping[str, Any]]:
    split = spec.dataset_names[name]
    files = manifest.get("files")
    entry = files.get(split) if isinstance(files, Mapping) else None
    if not isinstance(entry, Mapping):
        raise SystemExit(f"{spec.label} manifest is missing files.{split} metadata.")
    expected_format = "ordered-jsonl-shards" if split == "train" else "jsonl"
    if entry.get("format") != expected_format:
        raise SystemExit(
            f"{spec.label} manifest files.{split}.format must be {expected_format!r}."
        )
    return split, entry


def _manifest_file_path(value: Any, *, location: str, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{label} manifest {location}.path must be non-blank.")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _manifest_file_entries(
    spec: _G1ManifestSpec, manifest: Mapping[str, Any], name: str
) -> tuple[str, Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    split, split_entry = _manifest_entry(spec, manifest, name)
    if split == "dev":
        return split, split_entry, (split_entry,)
    raw_shards = split_entry.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise SystemExit(
            f"{spec.label} manifest files.train.shards must be a non-empty array."
        )
    if not all(isinstance(shard, Mapping) for shard in raw_shards):
        raise SystemExit(
            f"Every {spec.label} train shard manifest entry must be an object."
        )
    paths = [
        _manifest_file_path(
            shard.get("path"), location=f"files.train.shards[{index}]", label=spec.label
        )
        for index, shard in enumerate(raw_shards)
    ]
    if len(set(paths)) != len(paths):
        raise SystemExit(f"{spec.label} manifest lists a duplicate train shard path.")
    return split, split_entry, tuple(raw_shards)


def dataset_paths(name: str) -> tuple[Path, ...]:
    """Resolve one legacy file or the ordered files in a generated g1 split."""

    spec = _g1_manifest_spec_for(name)
    if spec is None:
        return (dataset_path(name),)
    manifest = _g1_manifest(spec)
    split, _, entries = _manifest_file_entries(spec, manifest, name)
    return tuple(
        _manifest_file_path(
            entry.get("path"),
            location=(
                f"files.{split}"
                if split == "dev"
                else f"files.train.shards[{index}]"
            ),
            label=spec.label,
        )
        for index, entry in enumerate(entries)
    )


def _manifest_nonnegative_int(
    entry: Mapping[str, Any], key: str, *, location: str, label: str
) -> int:
    value = entry.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SystemExit(
            f"{label} manifest {location}.{key} must be a non-negative integer."
        )
    return value


def _manifest_sha256(
    entry: Mapping[str, Any], key: str, *, location: str, label: str
) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SystemExit(
            f"{label} manifest {location}.{key} must be a SHA256 hex digest."
        )
    return value


def _verified_manifest_paths(name: str) -> tuple[Path, ...]:
    spec = _g1_manifest_spec_for(name)
    assert spec is not None
    manifest = _g1_manifest(spec)
    split, split_entry, entries = _manifest_file_entries(spec, manifest, name)
    verified: list[Path] = []
    total_bytes = 0
    total_rows = 0
    aggregate = hashlib.sha256()
    for index, entry in enumerate(entries):
        location = (
            f"files.{split}"
            if split == "dev"
            else f"files.train.shards[{index}]"
        )
        path = _manifest_file_path(entry.get("path"), location=location, label=spec.label)
        if not path.is_file():
            raise SystemExit(f"Missing {spec.label} {split} dataset file: {path}")
        digest = hashlib.sha256()
        actual_bytes = 0
        actual_rows = 0
        try:
            with path.open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    aggregate.update(line)
                    actual_bytes += len(line)
                    actual_rows += bool(line.strip())
        except OSError as exc:
            raise SystemExit(f"Cannot read {spec.label} dataset file {path}: {exc}") from exc
        expected_bytes = _manifest_nonnegative_int(
            entry, "bytes", location=location, label=spec.label
        )
        if actual_bytes != expected_bytes:
            raise SystemExit(
                f"{spec.label} dataset file {path} byte count mismatch: "
                f"manifest={expected_bytes}, actual={actual_bytes}."
            )
        expected_sha = _manifest_sha256(entry, "sha256", location=location, label=spec.label)
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            raise SystemExit(
                f"{spec.label} dataset file {path} SHA256 mismatch: "
                f"manifest={expected_sha}, actual={actual_sha}."
            )
        expected_rows = _manifest_nonnegative_int(
            entry, "rows", location=location, label=spec.label
        )
        if actual_rows != expected_rows:
            raise SystemExit(
                f"{spec.label} dataset file {path} row count mismatch: "
                f"manifest={expected_rows}, actual={actual_rows}."
            )
        verified.append(path)
        total_bytes += actual_bytes
        total_rows += actual_rows

    split_bytes = _manifest_nonnegative_int(
        split_entry, "bytes", location=f"files.{split}", label=spec.label
    )
    split_rows = _manifest_nonnegative_int(
        split_entry, "rows", location=f"files.{split}", label=spec.label
    )
    if total_bytes != split_bytes:
        raise SystemExit(
            f"{spec.label} {split} aggregate byte count mismatch: "
            f"manifest={split_bytes}, actual={total_bytes}."
        )
    if total_rows != split_rows:
        raise SystemExit(
            f"{spec.label} {split} aggregate row count mismatch: "
            f"manifest={split_rows}, actual={total_rows}."
        )
    if split == "train":
        expected_aggregate = _manifest_sha256(
            split_entry, "aggregate_sha256", location="files.train", label=spec.label
        )
        actual_aggregate = aggregate.hexdigest()
        if actual_aggregate != expected_aggregate:
            raise SystemExit(
                f"{spec.label} train aggregate SHA256 mismatch: "
                f"manifest={expected_aggregate}, actual={actual_aggregate}."
            )
    return tuple(verified)


def _load_manifest_pairs(name: str) -> list[dict[str, Any]]:
    spec = _g1_manifest_spec_for(name)
    assert spec is not None
    pairs: list[dict[str, Any]] = []
    for path in _verified_manifest_paths(name):
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        pair = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SystemExit(
                            f"Invalid JSON in {spec.label} dataset file {path} "
                            f"at line {line_number}: {exc}"
                        ) from exc
                    if not isinstance(pair, dict):
                        raise SystemExit(
                            f"{spec.label} dataset file {path} line {line_number} "
                            "must be a JSON object."
                        )
                    pairs.append(pair)
        except UnicodeDecodeError as exc:
            raise SystemExit(
                f"{spec.label} dataset file {path} is not UTF-8: {exc}"
            ) from exc
        except OSError as exc:
            raise SystemExit(
                f"Cannot read {spec.label} dataset file {path}: {exc}"
            ) from exc
    return pairs


def load_pairs(name: str) -> list[dict[str, Any]]:
    spec = _g1_manifest_spec_for(name)
    if spec is not None:
        pairs = _load_manifest_pairs(name)
        source = ROOT / spec.relative_path
        if not pairs or any(
            pair.get("schema_version") not in (4, 5, 6, "g1") for pair in pairs
        ):
            raise SystemExit(
                f"{source} split {name!r} is empty or uses an unsupported "
                "dataset schema."
            )
        return pairs
    path = dataset_path(name)
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python -m datagen.generate")
    with path.open(encoding="utf-8") as handle:
        pairs = [json.loads(line) for line in handle if line.strip()]
    if not pairs or any(pair.get("schema_version") not in (4, 5, 6, "g1") for pair in pairs):
        raise SystemExit(f"{path} is empty or uses an unsupported dataset schema.")
    return pairs


def g1_train_dev_dataset_names() -> tuple[str, str]:
    """Resolve the only datasets a paid g1 run is allowed to consume."""

    full_manifest_path = ROOT / G1_FULL_MANIFEST_RELATIVE_PATH
    if not full_manifest_path.is_file():
        raise SystemExit(
            f"Missing merged g1 manifest: {full_manifest_path}. "
            "Run: python -m scripts.g1_full_build. Paid training never falls "
            "back to a partial one-demo dataset."
        )
    return "train_g1", "dev_g1"


def get_tokenizer(base_model: str | None = None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base_model or BASE_MODEL)


def render_prompt_ids(tokenizer, pair: dict[str, Any]) -> list[int]:
    messages = [
        {"role": "system", "content": system_prompt_for_pair(pair)},
        {"role": "user", "content": pair["prompt"]},
    ]
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    return list(ids)


def system_prompt_for_pair(pair: dict[str, Any]) -> str:
    schema = str(pair.get("schema_version", ""))
    return SYSTEM_PROMPT_G1 if schema.startswith("g1") else ACTIVE_SYSTEM_PROMPT


def completion_class(pair: dict[str, Any]) -> str:
    if str(pair.get("schema_version", "")).startswith("g1"):
        action = parse_g1_action(str(pair["completion"]))
        if not action.valid:
            label = "invalid"
        elif action.kind.value == "tool":
            label = str(action.tool_name)
        else:
            label = action.kind.value
    else:
        action = parse_action(str(pair["completion"]))
        label = action_class(action)
    if label not in CLASS_WEIGHTS:
        raise ValueError(f"Unsupported gold completion class {label!r}: {pair['completion']!r}")
    return label


def example_weight_for_pair(pair: dict[str, Any]) -> float:
    if str(pair.get("schema_version", "")).startswith("g1"):
        completion_class(pair)
        return 1.0
    return CLASS_WEIGHTS[completion_class(pair)]


def build_datum(tinker, tokenizer, pair: dict[str, Any]):
    prompt_ids = render_prompt_ids(tokenizer, pair)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end, int) or im_end < 0:
        raise RuntimeError("Qwen tokenizer does not expose <|im_end|>.")
    completion_ids = tokenizer(pair["completion"], add_special_tokens=False)["input_ids"] + [im_end]

    full = prompt_ids + completion_ids
    inputs, targets = full[:-1], full[1:]
    example_weight = example_weight_for_pair(pair)
    token_weight = example_weight / len(completion_ids)
    weights = [0.0] * (len(prompt_ids) - 1) + [token_weight] * len(completion_ids)
    assert len(inputs) == len(targets) == len(weights)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(inputs),
        loss_fn_inputs={"weights": weights, "target_tokens": targets},
    )


def g1_target_token_count(tokenizer: Any, pair: Mapping[str, Any]) -> int:
    """Return the exact target-token length Tinker validates for one g1 row."""

    prompt_ids = render_prompt_ids(tokenizer, dict(pair))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end, int) or isinstance(im_end, bool) or im_end < 0:
        raise RuntimeError("Qwen tokenizer does not expose <|im_end|>.")
    completion = pair.get("completion")
    if not isinstance(completion, str):
        raise ValueError("g1 row completion must be text before token counting.")
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    return len(prompt_ids) + len(completion_ids)


def save_state_json(update: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state.update(update)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[state] saved {', '.join(update)}", flush=True)


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "eval"


def _path_of(value: Any) -> str:
    return str(getattr(value, "path", value))


def git_commit_short() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip()
            or "unknown"
        )
    except OSError:
        return "unknown"


def pairs_sha(pairs: list[dict[str, Any]]) -> str:
    """Content fingerprint of the exact rows being scored.

    Reports carry this so a regenerated dataset can never silently produce
    incomparable numbers under the same filename; the beat-previous gate keys
    on the same value.
    """
    canonical = "\n".join(json.dumps(pair, sort_keys=True, ensure_ascii=False) for pair in pairs)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def report_provenance(
    pairs: list[dict[str, Any]],
    label: str,
    *,
    base_model: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "git_commit": git_commit_short(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_rows": len(pairs),
        "dataset_sha": pairs_sha(pairs),
        "base_model": base_model or BASE_MODEL,
        "seed": SEED if seed is None else seed,
    }


def sample_outputs(
    sampling_client,
    tokenizer,
    pairs: list[dict[str, Any]],
    *,
    label: str,
    seed: int | None = None,
) -> list[str]:
    import tinker

    params = tinker.types.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        stop=["\n"],
        seed=SEED if seed is None else seed,
    )
    outputs = [""] * len(pairs)
    for start in range(0, len(pairs), EVAL_CONCURRENCY):
        chunk = list(enumerate(pairs[start : start + EVAL_CONCURRENCY], start=start))
        futures = [
            (
                position,
                sampling_client.sample(
                    prompt=tinker.ModelInput.from_ints(render_prompt_ids(tokenizer, pair)),
                    num_samples=1,
                    sampling_params=params,
                ),
            )
            for position, pair in chunk
        ]
        for position, future in futures:
            response = future.result(timeout=REMOTE_TIMEOUT_SECONDS)
            outputs[position] = tokenizer.decode(
                response.sequences[0].tokens,
                skip_special_tokens=True,
            ).strip()
        print(f"[{label}] sampled {min(start + EVAL_CONCURRENCY, len(pairs))}/{len(pairs)}", flush=True)
    return outputs


def run_eval(
    sampling_client,
    tokenizer,
    pairs: list[dict[str, Any]],
    label: str,
    hard_gates: dict[str, Any] | None = None,
    *,
    base_model: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    outputs = sample_outputs(sampling_client, tokenizer, pairs, label=label, seed=seed)
    report = evaluate_pair_outputs(
        pairs,
        outputs,
        label=label,
        hard_gates=hard_gates,
    )
    report["provenance"] = report_provenance(pairs, label, base_model=base_model, seed=seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{_safe_label(label)}_eval.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_eval_summary(report, label=label)
    return report


def evaluate_pair_outputs(
    pairs: list[dict[str, Any]],
    outputs: list[str],
    *,
    label: str,
    hard_gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schemas = {pair.get("schema_version") for pair in pairs}
    if len(schemas) == 1 and str(next(iter(schemas))).startswith("g1"):
        if hard_gates is not None:
            raise ValueError("g1 evaluation uses its fixed pilot hard gates.")
        return evaluate_g1_predictions(pairs, outputs, label=label)
    if any(str(schema).startswith("g1") for schema in schemas):
        raise ValueError("Cannot evaluate mixed g1 and legacy schemas in one split.")
    return evaluate_predictions(pairs, outputs, label=label, hard_gates=hard_gates)


def _print_eval_summary(report: dict[str, Any], *, label: str, prefix: str = "") -> None:
    summary = report["summary"]
    if "should_fire_recall" in summary:
        message = (
            f"[{label}] {prefix}strict={summary['strict_row_accuracy']} "
            f"fire_recall={summary['should_fire_recall']} "
            f"wait={summary['reminder_wait_accuracy']} "
            f"gates={report['hard_gates']['passed']}"
        )
    else:
        message = (
            f"[{label}] {prefix}score={summary['overall_score_percent']} "
            f"macro_f1={summary['macro_f1']} strict={summary['strict_row_accuracy']} "
            f"episodes={summary['episode_success_rate']} "
            f"gates={report['hard_gates']['passed']}"
        )
    print(message, flush=True)


def rescore_report(source: Path, split: str, label: str) -> dict[str, Any]:
    """Re-run local scoring from a saved report without new remote inference."""

    if not source.exists():
        raise SystemExit(f"Missing saved report: {source}")
    previous = json.loads(source.read_text(encoding="utf-8"))
    previous_rows = previous.get("rows")
    if not isinstance(previous_rows, list):
        raise SystemExit(f"{source} does not contain row-level outputs.")
    pairs = load_pairs(split)
    if len(previous_rows) != len(pairs):
        raise SystemExit(
            f"Saved report has {len(previous_rows)} rows but {split} has {len(pairs)} pairs."
        )
    outputs: list[str] = []
    for index, (row, pair) in enumerate(zip(previous_rows, pairs, strict=True)):
        if row.get("episode") != pair.get("episode") or row.get("event_index") != pair.get(
            "event_index"
        ):
            raise SystemExit(f"Saved report and {split} diverge at row {index}.")
        outputs.append(str(row.get("raw", "")))
    report = evaluate_pair_outputs(pairs, outputs, label=label)
    report["provenance"] = {
        **report_provenance(pairs, label),
        "split": split,
        "rescored_from": str(source),
        "source_provenance": previous.get("provenance"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{_safe_label(label)}_eval.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_eval_summary(report, label=label, prefix="local rescore ")
    return report


def stage_preflight() -> None:
    import tinker

    print(f"[preflight] connecting to Tinker for {BASE_MODEL}", flush=True)
    service = tinker.ServiceClient(user_metadata={"project": "smol-interactions-v4-preflight"})
    sampler = service.create_sampling_client(base_model=BASE_MODEL)
    tokenizer = get_tokenizer()
    # Never export a project-authored row merely to test connectivity.
    pair = {
        "prompt": (
            "<interaction_context>\n"
            "<instruction></instruction>\n"
            "<permissions>{}</permissions>\n"
            "</interaction_context>\n"
            '<stream_event index="1" source="user" state="active" time="t+0ms">hello</stream_event>\n'
            "<PREDICT_THIS_ACTION>"
        )
    }
    output = sample_outputs(sampler, tokenizer, [pair], label="preflight")[0]
    parsed = parse_action(output)
    print(
        f"[preflight] connected; parser_valid={parsed.valid}; predicted={action_class(parsed)}",
        flush=True,
    )


def stage_baseline(split: str = "dev") -> dict[str, Any]:
    import tinker

    service = tinker.ServiceClient(user_metadata={"project": "smol-interactions-v4-baseline"})
    sampler = service.create_sampling_client(base_model=BASE_MODEL)
    return run_eval(sampler, get_tokenizer(), load_pairs(split), f"baseline-v4-{split}")


def stage_train(
    tag: str,
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
) -> dict[str, Any]:
    import tinker

    service = tinker.ServiceClient(user_metadata={"project": f"smol-interactions-{tag}"})
    client = service.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
        seed=SEED,
        user_metadata={"project": f"smol-interactions-{tag}"},
    )
    save_state_json({f"{tag}:model_id": str(getattr(client, "model_id", "unknown"))})
    tokenizer = get_tokenizer()
    train_pairs = load_pairs("train")
    dev_pairs = load_pairs("dev")
    data = [build_datum(tinker, tokenizer, pair) for pair in train_pairs]
    print(
        f"[train] {len(data)} datums, {epochs} epochs, batch {batch_size}, lr {learning_rate}",
        flush=True,
    )

    rng = Random(SEED)
    step = 0
    best_score = -1.0
    best_epoch = 0
    best_sampler_path = ""
    best_report: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        order = list(range(len(data)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = [data[index] for index in order[start : start + batch_size]]
            started = time.monotonic()
            fb_future = client.forward_backward(batch, "cross_entropy")
            optim_future = client.optim_step(tinker.AdamParams(learning_rate=learning_rate))
            fb_result = fb_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
            optim_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
            step += 1
            metrics = getattr(fb_result, "metrics", None) or {}
            loss = metrics.get("loss:sum") if isinstance(metrics, dict) else None
            weight = metrics.get("loss_fn_output_weight:sum") if isinstance(metrics, dict) else None
            mean_loss = loss / weight if loss is not None and weight else None
            print(
                f"[train] epoch={epoch} step={step} batch={len(batch)} "
                f"loss={mean_loss if mean_loss is not None else metrics} "
                f"seconds={time.monotonic() - started:.1f}",
                flush=True,
            )

        state_result = client.save_state(name=f"smol-{tag}-epoch{epoch}").result(
            timeout=REMOTE_TIMEOUT_SECONDS
        )
        save_state_json({f"{tag}:state_epoch{epoch}": _path_of(state_result)})

        epoch_sampler = client.save_weights_and_get_sampling_client()
        report = run_eval(epoch_sampler, tokenizer, dev_pairs, f"dev-{tag}-epoch{epoch}")
        score = float(report["summary"]["overall_score"])
        if score > best_score:
            sampler_result = client.save_weights_for_sampler(
                name=f"smol-{tag}-best-epoch{epoch}"
            ).result(timeout=REMOTE_TIMEOUT_SECONDS)
            best_sampler_path = _path_of(sampler_result)
            best_score = score
            best_epoch = epoch
            best_report = report
            save_state_json(
                {
                    f"{tag}:best_epoch": epoch,
                    f"{tag}:best_dev_score": score,
                    f"{tag}:sampler_path": best_sampler_path,
                    # Only the canonical "v4" tag may update the legacy
                    # v4:sampler_path fallback key; a fresh lineage tag
                    # (e.g. v4f) must not clobber the historical pointer.
                    **({"v4:sampler_path": best_sampler_path} if tag == "v4" else {}),
                }
            )

    if not best_sampler_path or best_report is None:
        raise RuntimeError("Training completed without a selectable dev checkpoint.")
    print(
        f"[train] selected epoch {best_epoch}: score={best_score:.6f} path={best_sampler_path}",
        flush=True,
    )
    return {
        "tag": tag,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "sampler_path": best_sampler_path,
        "dev_report": best_report,
    }


def stage_resume(
    tag: str,
    *,
    from_epoch: int,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
) -> dict[str, Any]:
    """Resume after a completed epoch using its optimizer-inclusive Tinker state."""
    import tinker

    if from_epoch < 1 or from_epoch >= epochs:
        raise SystemExit("--from-epoch must be at least 1 and lower than --epochs.")
    if not STATE_PATH.exists():
        raise SystemExit(f"Missing {STATE_PATH}; there is no state to resume.")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state_path = state.get(f"{tag}:state_epoch{from_epoch}")
    if not state_path:
        raise SystemExit(f"No saved optimizer state for {tag} epoch {from_epoch}.")

    service = tinker.ServiceClient(user_metadata={"project": f"smol-interactions-{tag}-resume"})
    client = service.create_training_client_from_state_with_optimizer(
        path=str(state_path),
        user_metadata={"project": f"smol-interactions-{tag}-resume"},
    )
    tokenizer = get_tokenizer()
    train_pairs = load_pairs("train")
    dev_pairs = load_pairs("dev")
    data = [build_datum(tinker, tokenizer, pair) for pair in train_pairs]

    rng = Random(SEED)
    # Consume the same earlier shuffles so the resumed epoch is reproducible.
    for _ in range(from_epoch):
        prior_order = list(range(len(data)))
        rng.shuffle(prior_order)

    step = from_epoch * ((len(data) + batch_size - 1) // batch_size)
    best_score = float(state.get(f"{tag}:best_dev_score", -1.0))
    best_epoch = int(state.get(f"{tag}:best_epoch", from_epoch))
    best_sampler_path = str(state.get(f"{tag}:sampler_path", ""))
    print(
        f"[resume] epoch {from_epoch + 1} from {state_path}; "
        f"current best epoch={best_epoch} score={best_score}",
        flush=True,
    )

    for epoch in range(from_epoch + 1, epochs + 1):
        order = list(range(len(data)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = [data[index] for index in order[start : start + batch_size]]
            started = time.monotonic()
            fb_future = client.forward_backward(batch, "cross_entropy")
            optim_future = client.optim_step(tinker.AdamParams(learning_rate=learning_rate))
            fb_result = fb_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
            optim_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
            step += 1
            metrics = getattr(fb_result, "metrics", None) or {}
            loss = metrics.get("loss:sum") if isinstance(metrics, dict) else None
            weight = metrics.get("loss_fn_output_weight:sum") if isinstance(metrics, dict) else None
            mean_loss = loss / weight if loss is not None and weight else None
            print(
                f"[train] epoch={epoch} step={step} batch={len(batch)} "
                f"loss={mean_loss if mean_loss is not None else metrics} "
                f"seconds={time.monotonic() - started:.1f}",
                flush=True,
            )

        state_result = client.save_state(name=f"smol-{tag}-epoch{epoch}").result(
            timeout=REMOTE_TIMEOUT_SECONDS
        )
        save_state_json({f"{tag}:state_epoch{epoch}": _path_of(state_result)})
        epoch_sampler = client.save_weights_and_get_sampling_client()
        report = run_eval(epoch_sampler, tokenizer, dev_pairs, f"dev-{tag}-epoch{epoch}")
        score = float(report["summary"]["overall_score"])
        if score > best_score:
            sampler_result = client.save_weights_for_sampler(
                name=f"smol-{tag}-best-epoch{epoch}"
            ).result(timeout=REMOTE_TIMEOUT_SECONDS)
            best_sampler_path = _path_of(sampler_result)
            best_score = score
            best_epoch = epoch
            save_state_json(
                {
                    f"{tag}:best_epoch": epoch,
                    f"{tag}:best_dev_score": score,
                    f"{tag}:sampler_path": best_sampler_path,
                    # Same guard as stage_train: never let a fresh lineage
                    # tag clobber the legacy v4:sampler_path fallback.
                    **({"v4:sampler_path": best_sampler_path} if tag == "v4" else {}),
                }
            )

    print(
        f"[resume] selected epoch {best_epoch}: score={best_score:.6f} "
        f"path={best_sampler_path}",
        flush=True,
    )
    return {
        "tag": tag,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "sampler_path": best_sampler_path,
    }


def stage_adapt(
    tag: str,
    *,
    source_tag: str = "v4",
    source_epoch: int = 2,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = ADAPT_LEARNING_RATE,
) -> dict[str, Any]:
    """Run one replay-protected v4.1 adaptation from a selected v4 state.

    The natural 1,344 base / 560 challenge mixture is approximately 70/30.
    Selection rewards challenge improvement but rejects more than two points of
    base-dev regression.
    """

    import tinker

    if not STATE_PATH.exists():
        raise SystemExit(f"Missing {STATE_PATH}; no source checkpoint is available.")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    source_state_path = state.get(f"{source_tag}:state_epoch{source_epoch}")
    source_sampler_path = state.get(f"{source_tag}:sampler_path")
    if not source_state_path or not source_sampler_path:
        raise SystemExit(
            f"Missing {source_tag} epoch-{source_epoch} optimizer state or sampler path."
        )

    service = tinker.ServiceClient(user_metadata={"project": f"smol-interactions-{tag}-adapt"})
    tokenizer = get_tokenizer()
    base_dev = load_pairs("dev")
    challenge_dev = load_pairs("dev_v41")

    print(f"[adapt] measuring source checkpoint {source_sampler_path}", flush=True)
    source_sampler = service.create_sampling_client(model_path=str(source_sampler_path))
    source_base_report = run_eval(
        source_sampler, tokenizer, base_dev, f"dev-{tag}-source-base"
    )
    source_challenge_report = run_eval(
        source_sampler, tokenizer, challenge_dev, f"dev-{tag}-source-challenge"
    )

    client = service.create_training_client_from_state_with_optimizer(
        path=str(source_state_path),
        user_metadata={"project": f"smol-interactions-{tag}-adapt"},
    )
    base_train = load_pairs("train")
    challenge_train = load_pairs("train_v41")
    train_pairs = [*base_train, *challenge_train]
    data = [build_datum(tinker, tokenizer, pair) for pair in train_pairs]
    print(
        f"[adapt] {len(base_train)} base + {len(challenge_train)} challenge = "
        f"{len(data)} datums, one epoch, batch {batch_size}, lr {learning_rate}",
        flush=True,
    )

    order = list(range(len(data)))
    Random(SEED + 1).shuffle(order)
    for step, start in enumerate(range(0, len(order), batch_size), start=1):
        batch = [data[index] for index in order[start : start + batch_size]]
        started = time.monotonic()
        fb_future = client.forward_backward(batch, "cross_entropy")
        optim_future = client.optim_step(tinker.AdamParams(learning_rate=learning_rate))
        fb_result = fb_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
        optim_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
        metrics = getattr(fb_result, "metrics", None) or {}
        loss = metrics.get("loss:sum") if isinstance(metrics, dict) else None
        weight = metrics.get("loss_fn_output_weight:sum") if isinstance(metrics, dict) else None
        mean_loss = loss / weight if loss is not None and weight else None
        print(
            f"[adapt] step={step} batch={len(batch)} "
            f"loss={mean_loss if mean_loss is not None else metrics} "
            f"seconds={time.monotonic() - started:.1f}",
            flush=True,
        )

    state_result = client.save_state(name=f"smol-{tag}-adapt1").result(
        timeout=REMOTE_TIMEOUT_SECONDS
    )
    save_state_json({f"{tag}:state_adapt1": _path_of(state_result)})
    candidate_sampler = client.save_weights_and_get_sampling_client()
    adapted_base_report = run_eval(
        candidate_sampler, tokenizer, base_dev, f"dev-{tag}-adapt1-base"
    )
    adapted_challenge_report = run_eval(
        candidate_sampler, tokenizer, challenge_dev, f"dev-{tag}-adapt1-challenge"
    )

    source_base_score = float(source_base_report["summary"]["overall_score"])
    source_challenge_score = float(source_challenge_report["summary"]["overall_score"])
    adapted_base_score = float(adapted_base_report["summary"]["overall_score"])
    adapted_challenge_score = float(adapted_challenge_report["summary"]["overall_score"])
    accepted = (
        adapted_challenge_score > source_challenge_score
        and adapted_base_score >= source_base_score - 0.02
    )
    if accepted:
        sampler_result = client.save_weights_for_sampler(name=f"smol-{tag}-selected").result(
            timeout=REMOTE_TIMEOUT_SECONDS
        )
        selected_path = _path_of(sampler_result)
        selected = "adapt1"
    else:
        selected_path = str(source_sampler_path)
        selected = "source"

    save_state_json(
        {
            f"{tag}:source_tag": source_tag,
            f"{tag}:source_epoch": source_epoch,
            f"{tag}:source_base_dev_score": source_base_score,
            f"{tag}:source_challenge_dev_score": source_challenge_score,
            f"{tag}:adapted_base_dev_score": adapted_base_score,
            f"{tag}:adapted_challenge_dev_score": adapted_challenge_score,
            f"{tag}:adapted_accepted": accepted,
            f"{tag}:selected": selected,
            f"{tag}:sampler_path": selected_path,
        }
    )
    print(
        f"[adapt] selected={selected} accepted={accepted}; "
        f"base {source_base_score:.6f}->{adapted_base_score:.6f}; "
        f"challenge {source_challenge_score:.6f}->{adapted_challenge_score:.6f}",
        flush=True,
    )
    return {
        "tag": tag,
        "accepted": accepted,
        "selected": selected,
        "sampler_path": selected_path,
        "source_base_report": source_base_report,
        "source_challenge_report": source_challenge_report,
        "adapted_base_report": adapted_base_report,
        "adapted_challenge_report": adapted_challenge_report,
    }


def stage_v5(
    tag: str = "v5",
    *,
    source_tag: str = "v41",
    epochs: int = 1,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = ADAPT_LEARNING_RATE,
) -> dict[str, Any]:
    """Adapt the selected v4.1 checkpoint into the v5 co-writing policy.

    Everything in this stage — training and evaluation of the adapted model —
    renders under SYSTEM_PROMPT_V5 so serving matches training. The source
    checkpoint is measured under its native v4 prompt for a fair regression
    gate. Selection requires a v5-dev win with at most two points of
    regression on the v4 base dev and the v4.1 challenge dev.
    """

    global ACTIVE_SYSTEM_PROMPT
    import tinker

    if not STATE_PATH.exists():
        raise SystemExit(f"Missing {STATE_PATH}; no source checkpoint is available.")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state.get(f"{source_tag}:selected") == "adapt1":
        source_state_path = state.get(f"{source_tag}:state_adapt1")
    else:
        fallback_tag = state.get(f"{source_tag}:source_tag", "v4")
        fallback_epoch = state.get(f"{source_tag}:source_epoch", 2)
        source_state_path = state.get(f"{fallback_tag}:state_epoch{fallback_epoch}")
    source_sampler_path = state.get(f"{source_tag}:sampler_path")
    if not source_state_path or not source_sampler_path:
        raise SystemExit(f"Missing {source_tag} optimizer state or sampler path.")

    service = tinker.ServiceClient(user_metadata={"project": f"smol-interactions-{tag}"})
    tokenizer = get_tokenizer()
    base_dev = load_pairs("dev")
    challenge_dev = load_pairs("dev_v41")
    cowrite_dev = load_pairs("dev_v5")

    print(f"[v5] measuring source checkpoint {source_sampler_path}", flush=True)
    source_sampler = service.create_sampling_client(model_path=str(source_sampler_path))
    source_base = run_eval(source_sampler, tokenizer, base_dev, f"dev-{tag}-source-base")
    source_challenge = run_eval(
        source_sampler, tokenizer, challenge_dev, f"dev-{tag}-source-challenge"
    )
    ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT_V5
    try:
        source_cowrite = run_eval(
            source_sampler, tokenizer, cowrite_dev, f"dev-{tag}-source-cowrite"
        )

        client = service.create_training_client_from_state_with_optimizer(
            path=str(source_state_path),
            user_metadata={"project": f"smol-interactions-{tag}"},
        )
        train_pairs = [*load_pairs("train"), *load_pairs("train_v41"), *load_pairs("train_v5")]
        data = [build_datum(tinker, tokenizer, pair) for pair in train_pairs]
        approx_chars = sum(len(str(pair["prompt"])) for pair in train_pairs)
        print(
            f"[v5] {len(train_pairs)} pairs (~{approx_chars / 4e6:.2f}M prompt tokens est.), "
            f"{epochs} epoch(s), batch {batch_size}, lr {learning_rate}",
            flush=True,
        )

        for epoch in range(1, epochs + 1):
            order = list(range(len(data)))
            Random(SEED + 10 + epoch).shuffle(order)
            for step, start in enumerate(range(0, len(order), batch_size), start=1):
                batch = [data[index] for index in order[start : start + batch_size]]
                started = time.monotonic()
                fb_future = client.forward_backward(batch, "cross_entropy")
                optim_future = client.optim_step(tinker.AdamParams(learning_rate=learning_rate))
                fb_result = fb_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
                optim_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
                metrics = getattr(fb_result, "metrics", None) or {}
                loss = metrics.get("loss:sum") if isinstance(metrics, dict) else None
                weight = metrics.get("loss_fn_output_weight:sum") if isinstance(metrics, dict) else None
                mean_loss = loss / weight if loss is not None and weight else None
                print(
                    f"[v5] epoch={epoch} step={step} loss={mean_loss if mean_loss is not None else metrics} "
                    f"seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )

        state_result = client.save_state(name=f"smol-{tag}-adapt1").result(
            timeout=REMOTE_TIMEOUT_SECONDS
        )
        save_state_json({f"{tag}:state_adapt1": _path_of(state_result)})
        candidate = client.save_weights_and_get_sampling_client()

        adapted_base = run_eval(candidate, tokenizer, base_dev, f"dev-{tag}-adapt1-base")
        adapted_challenge = run_eval(
            candidate, tokenizer, challenge_dev, f"dev-{tag}-adapt1-challenge"
        )
        adapted_cowrite = run_eval(candidate, tokenizer, cowrite_dev, f"dev-{tag}-adapt1-cowrite")

        scores = {
            "source_base": float(source_base["summary"]["overall_score"]),
            "source_challenge": float(source_challenge["summary"]["overall_score"]),
            "source_cowrite": float(source_cowrite["summary"]["overall_score"]),
            "adapted_base": float(adapted_base["summary"]["overall_score"]),
            "adapted_challenge": float(adapted_challenge["summary"]["overall_score"]),
            "adapted_cowrite": float(adapted_cowrite["summary"]["overall_score"]),
        }
        accepted = (
            scores["adapted_cowrite"] > scores["source_cowrite"]
            and scores["adapted_base"] >= scores["source_base"] - 0.02
            and scores["adapted_challenge"] >= scores["source_challenge"] - 0.02
        )
        if accepted:
            sampler_result = client.save_weights_for_sampler(name=f"smol-{tag}-selected").result(
                timeout=REMOTE_TIMEOUT_SECONDS
            )
            selected_path = _path_of(sampler_result)
            test_report = run_eval(
                service.create_sampling_client(model_path=selected_path),
                tokenizer,
                load_pairs("test_v5"),
                f"final-{tag}-test",
            )
            test_score = float(test_report["summary"]["overall_score"])
        else:
            selected_path = str(source_sampler_path)
            test_score = None

        save_state_json(
            {
                f"{tag}:source_tag": source_tag,
                **{f"{tag}:{name}_dev_score": value for name, value in scores.items()},
                f"{tag}:adapted_accepted": accepted,
                f"{tag}:selected": "adapt1" if accepted else "source",
                f"{tag}:sampler_path": selected_path,
                **({f"{tag}:test_score": test_score} if test_score is not None else {}),
            }
        )
        print(f"[v5] accepted={accepted} scores={scores} test={test_score}", flush=True)
        return {"tag": tag, "accepted": accepted, "sampler_path": selected_path, "scores": scores}
    finally:
        ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT_V4


def stage_v6(
    tag: str = "v6",
    *,
    source_tag: str = "v41",
    epochs: int = 1,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = ADAPT_LEARNING_RATE,
    train_only: bool = False,
) -> dict[str, Any]:
    """Adapt the selected v4.1 checkpoint into the v6 surface policy.

    Everything in this stage renders under SYSTEM_PROMPT_V6 so serving matches
    training. The source checkpoint is measured under its native prompts for
    fair regression gates. Selection requires a v6-dev win, at most two points
    of regression on the v4 base dev and the v4.1 challenge dev, and a clean
    sweep of the frozen restraint episodes (the trigger-happiness guard: the
    base model is 6/6 silent, and training must not sell that).
    """

    global ACTIVE_SYSTEM_PROMPT
    import tinker

    if not STATE_PATH.exists():
        raise SystemExit(f"Missing {STATE_PATH}; no source checkpoint is available.")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state.get(f"{source_tag}:selected") == "adapt1":
        source_state_path = state.get(f"{source_tag}:state_adapt1")
    else:
        fallback_tag = state.get(f"{source_tag}:source_tag", "v4")
        fallback_epoch = state.get(f"{source_tag}:source_epoch", 2)
        source_state_path = state.get(f"{fallback_tag}:state_epoch{fallback_epoch}")
    source_sampler_path = state.get(f"{source_tag}:sampler_path")
    if not source_state_path or not source_sampler_path:
        raise SystemExit(f"Missing {source_tag} optimizer state or sampler path.")

    service = tinker.ServiceClient(user_metadata={"project": f"smol-interactions-{tag}"})
    tokenizer = get_tokenizer()
    base_dev = load_pairs("dev")
    challenge_dev = load_pairs("dev_v41")
    v6_dev = load_pairs("dev_v6")
    restraint_frozen = load_pairs("test_v6_restraint")

    if not train_only:
        print(f"[v6] measuring source checkpoint {source_sampler_path}", flush=True)
        source_sampler = service.create_sampling_client(model_path=str(source_sampler_path))
        source_base = run_eval(source_sampler, tokenizer, base_dev, f"dev-{tag}-source-base")
        source_challenge = run_eval(
            source_sampler, tokenizer, challenge_dev, f"dev-{tag}-source-challenge"
        )
    ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT_V6
    try:
        if not train_only:
            source_v6 = run_eval(
                source_sampler, tokenizer, v6_dev, f"dev-{tag}-source-v6", hard_gates=V6_HARD_GATES
            )

        client = service.create_training_client_from_state_with_optimizer(
            path=str(source_state_path),
            user_metadata={"project": f"smol-interactions-{tag}"},
        )
        # v4/v4.1 replay guards against forgetting the surfaces the regression
        # gates measure; the retired v5 co-writing surface is not replayed.
        # DAgger rows (round 2+) join automatically when the harvest exists.
        train_pairs = [*load_pairs("train"), *load_pairs("train_v41"), *load_pairs("train_v6")]
        for harvest in ("train_v6_dagger", "train_v6_dagger2", "train_v6_dagger3"):
            if dataset_path(harvest).exists():
                train_pairs.extend(load_pairs(harvest))
        data = [build_datum(tinker, tokenizer, pair) for pair in train_pairs]
        approx_chars = sum(len(str(pair["prompt"])) for pair in train_pairs)
        print(
            f"[v6] {len(train_pairs)} pairs (~{approx_chars / 4e6:.2f}M prompt tokens est.), "
            f"{epochs} epoch(s), batch {batch_size}, lr {learning_rate}",
            flush=True,
        )

        for epoch in range(1, epochs + 1):
            order = list(range(len(data)))
            Random(SEED + 20 + epoch).shuffle(order)
            for step, start in enumerate(range(0, len(order), batch_size), start=1):
                batch = [data[index] for index in order[start : start + batch_size]]
                started = time.monotonic()
                fb_future = client.forward_backward(batch, "cross_entropy")
                optim_future = client.optim_step(tinker.AdamParams(learning_rate=learning_rate))
                fb_result = fb_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
                optim_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
                metrics = getattr(fb_result, "metrics", None) or {}
                loss = metrics.get("loss:sum") if isinstance(metrics, dict) else None
                weight = metrics.get("loss_fn_output_weight:sum") if isinstance(metrics, dict) else None
                mean_loss = loss / weight if loss is not None and weight else None
                print(
                    f"[v6] epoch={epoch} step={step} loss={mean_loss if mean_loss is not None else metrics} "
                    f"seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )

        state_result = client.save_state(name=f"smol-{tag}-adapt1").result(
            timeout=REMOTE_TIMEOUT_SECONDS
        )
        adapt_state_path = _path_of(state_result)

        if train_only:
            # Evals and gates run LOCALLY on the downloaded adapter; nothing
            # here may touch the accepted keys (sampler_path stays the last
            # accepted round's). The sampler name carries the epoch count so
            # sweep variants never overwrite each other.
            sampler_result = client.save_weights_for_sampler(
                name=f"smol-{tag}-pending-e{epochs}"
            ).result(timeout=REMOTE_TIMEOUT_SECONDS)
            pending_sampler = _path_of(sampler_result)
            dev_sha = pairs_sha(v6_dev)
            save_state_json(
                {
                    f"{tag}:pending_e{epochs}_state_adapt1": adapt_state_path,
                    f"{tag}:pending_e{epochs}_sampler_path": pending_sampler,
                    f"{tag}:pending_e{epochs}_dev_sha": dev_sha,
                }
            )
            print(f"[v6] train-only done: state={adapt_state_path}", flush=True)
            print(f"[v6] pending sampler={pending_sampler}", flush=True)
            return

        candidate = client.save_weights_and_get_sampling_client()

        adapted_v6 = run_eval(
            candidate, tokenizer, v6_dev, f"dev-{tag}-adapt1-v6", hard_gates=V6_HARD_GATES
        )
        adapted_restraint = run_eval(
            candidate,
            tokenizer,
            restraint_frozen,
            f"frozen-{tag}-adapt1-restraint",
            hard_gates=V6_HARD_GATES,
        )
    finally:
        ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT_V4

    adapted_base = run_eval(candidate, tokenizer, base_dev, f"dev-{tag}-adapt1-base")
    adapted_challenge = run_eval(
        candidate, tokenizer, challenge_dev, f"dev-{tag}-adapt1-challenge"
    )

    scores = {
        "source_base": float(source_base["summary"]["overall_score"]),
        "source_challenge": float(source_challenge["summary"]["overall_score"]),
        "source_v6": float(source_v6["summary"]["overall_score"]),
        "adapted_base": float(adapted_base["summary"]["overall_score"]),
        "adapted_challenge": float(adapted_challenge["summary"]["overall_score"]),
        "adapted_v6": float(adapted_v6["summary"]["overall_score"]),
        "adapted_restraint_clean": float(
            adapted_restraint["summary"]["episode_success_rate"]
        ),
    }
    accepted = (
        scores["adapted_v6"] > scores["source_v6"]
        and scores["adapted_base"] >= scores["source_base"] - 0.02
        and scores["adapted_challenge"] >= scores["source_challenge"] - 0.02
        and scores["adapted_restraint_clean"] == 1.0
    )
    # Beat-previous-round gate: a new round must exceed the last accepted
    # round's dev score — but dev scores are only comparable when the dev set
    # is identical, so the check keys on the dev content fingerprint (the same
    # value run_eval stamps into each report's provenance block).
    dev_sha = pairs_sha(v6_dev)
    previous = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}
    previous_score = previous.get(f"{tag}:adapted_v6_dev_score")
    has_baseline = bool(previous.get(f"{tag}:adapted_accepted")) and previous_score is not None
    comparable = has_baseline and previous.get(f"{tag}:dev_sha") == dev_sha
    if accepted and comparable and scores["adapted_v6"] <= float(previous_score):
        accepted = False
        print(
            f"[v6] rejected by beat-previous gate: {scores['adapted_v6']:.4f} "
            f"<= previous round {float(previous_score):.4f}",
            flush=True,
        )
    elif accepted and has_baseline and not comparable:
        # Never fail silent: a regenerated dev set (or a pre-provenance accept)
        # resets the baseline, and the operator must see that happen.
        print(
            f"[v6] beat-previous gate NOT ENFORCED: no recorded dev_sha matches the "
            f"current dev_v6 ({dev_sha[:12]}). Previous score {previous_score} was "
            f"measured on a different dev set; this round becomes the new baseline.",
            flush=True,
        )
    if accepted:
        sampler_result = client.save_weights_for_sampler(name=f"smol-{tag}-selected").result(
            timeout=REMOTE_TIMEOUT_SECONDS
        )
        selected_path = _path_of(sampler_result)
        save_state_json(
            {
                f"{tag}:source_tag": source_tag,
                **{f"{tag}:{name}_dev_score": value for name, value in scores.items()},
                f"{tag}:adapted_accepted": accepted,
                f"{tag}:selected": "adapt1",
                f"{tag}:sampler_path": selected_path,
                f"{tag}:state_adapt1": adapt_state_path,
                f"{tag}:dev_sha": dev_sha,
            }
        )
    else:
        # A rejected round must not clobber the previous accepted round's state.
        selected_path = str(previous.get(f"{tag}:sampler_path", source_sampler_path))
        print(f"[v6] keeping previous checkpoint: {selected_path}", flush=True)
    print(f"[v6] accepted={accepted} scores={scores}", flush=True)
    return {"tag": tag, "accepted": accepted, "sampler_path": selected_path, "scores": scores}


def stage_v6_promote(tag: str = "v6", *, epochs: int = 1) -> None:
    """Promote a train-only pending checkpoint after LOCAL gates passed.

    Run this only after the downloaded adapter cleared the local acceptance
    gates (frozen restraint 6/6 clean, corrections >= previous verified column,
    concurrency, dev). Copies pending paths to the active serving keys.
    """

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    prefix = f"{tag}:pending_e{epochs}_"
    state_path = state.get(prefix + "state_adapt1")
    sampler_path = state.get(prefix + "sampler_path")
    dev_sha = state.get(prefix + "dev_sha")
    if not state_path or not sampler_path:
        raise SystemExit(f"No pending e{epochs} checkpoint recorded in run_state.json.")
    save_state_json(
        {
            f"{tag}:state_adapt1": state_path,
            f"{tag}:sampler_path": sampler_path,
            f"{tag}:selected": "adapt1",
            f"{tag}:adapted_accepted": True,
            f"{tag}:dev_sha": dev_sha,
            f"{tag}:promoted_from": f"pending_e{epochs} (local gates)",
        }
    )
    print(f"[v6] promoted pending e{epochs}: {sampler_path}")


def stage_v6_evals(tag: str = "v6", *, source_tag: str = "v41") -> dict[str, Any]:
    """Finish an interrupted stage_v6: reload the saved adapter state, derive
    sampler weights, and run the candidate evals + acceptance gates without
    retraining. Source scores are read from the eval reports the interrupted
    run already wrote, so nothing is re-sampled twice."""

    global ACTIVE_SYSTEM_PROMPT
    import tinker

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    adapt_state = state.get(f"{tag}:state_adapt1")
    source_sampler_path = state.get(f"{source_tag}:sampler_path")
    if not adapt_state or not source_sampler_path:
        raise SystemExit(f"Missing {tag}:state_adapt1 or {source_tag}:sampler_path.")

    def saved_score(label: str) -> float:
        path = OUT_DIR / f"{_safe_label(label)}_eval.json"
        if not path.exists():
            raise SystemExit(f"Missing source report {path}; rerun --stage v6 instead.")
        return float(json.loads(path.read_text(encoding="utf-8"))["summary"]["overall_score"])

    scores = {
        "source_base": saved_score(f"dev-{tag}-source-base"),
        "source_challenge": saved_score(f"dev-{tag}-source-challenge"),
        "source_v6": saved_score(f"dev-{tag}-source-v6"),
    }

    service = tinker.ServiceClient(user_metadata={"project": f"smol-interactions-{tag}"})
    client = service.create_training_client_from_state_with_optimizer(
        path=str(adapt_state), user_metadata={"project": f"smol-interactions-{tag}"}
    )
    candidate = client.save_weights_and_get_sampling_client()
    tokenizer = get_tokenizer()

    v6_dev = load_pairs("dev_v6")
    ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT_V6
    try:
        adapted_v6 = run_eval(
            candidate, tokenizer, v6_dev, f"dev-{tag}-adapt1-v6",
            hard_gates=V6_HARD_GATES,
        )
        adapted_restraint = run_eval(
            candidate, tokenizer, load_pairs("test_v6_restraint"),
            f"frozen-{tag}-adapt1-restraint", hard_gates=V6_HARD_GATES,
        )
    finally:
        ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT_V4

    adapted_base = run_eval(candidate, tokenizer, load_pairs("dev"), f"dev-{tag}-adapt1-base")
    adapted_challenge = run_eval(
        candidate, tokenizer, load_pairs("dev_v41"), f"dev-{tag}-adapt1-challenge"
    )
    scores.update(
        {
            "adapted_base": float(adapted_base["summary"]["overall_score"]),
            "adapted_challenge": float(adapted_challenge["summary"]["overall_score"]),
            "adapted_v6": float(adapted_v6["summary"]["overall_score"]),
            "adapted_restraint_clean": float(
                adapted_restraint["summary"]["episode_success_rate"]
            ),
        }
    )
    accepted = (
        scores["adapted_v6"] > scores["source_v6"]
        and scores["adapted_base"] >= scores["source_base"] - 0.02
        and scores["adapted_challenge"] >= scores["source_challenge"] - 0.02
        and scores["adapted_restraint_clean"] == 1.0
    )
    # Same beat-previous gate as stage_v6: the resume path must not be a
    # side door around it.
    dev_sha = pairs_sha(v6_dev)
    previous = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}
    previous_score = previous.get(f"{tag}:adapted_v6_dev_score")
    has_baseline = bool(previous.get(f"{tag}:adapted_accepted")) and previous_score is not None
    comparable = has_baseline and previous.get(f"{tag}:dev_sha") == dev_sha
    if accepted and comparable and scores["adapted_v6"] <= float(previous_score):
        accepted = False
        print(
            f"[v6-evals] rejected by beat-previous gate: {scores['adapted_v6']:.4f} "
            f"<= previous round {float(previous_score):.4f}",
            flush=True,
        )
    elif accepted and has_baseline and not comparable:
        print(
            f"[v6-evals] beat-previous gate NOT ENFORCED: no recorded dev_sha matches "
            f"the current dev_v6 ({dev_sha[:12]}); this round becomes the new baseline.",
            flush=True,
        )
    if accepted:
        sampler_result = client.save_weights_for_sampler(name=f"smol-{tag}-selected").result(
            timeout=REMOTE_TIMEOUT_SECONDS
        )
        selected_path = _path_of(sampler_result)
    else:
        selected_path = str(source_sampler_path)

    save_state_json(
        {
            f"{tag}:source_tag": source_tag,
            **{f"{tag}:{name}_dev_score": value for name, value in scores.items()},
            f"{tag}:adapted_accepted": accepted,
            f"{tag}:selected": "adapt1" if accepted else "source",
            f"{tag}:sampler_path": selected_path,
            **({f"{tag}:dev_sha": dev_sha} if accepted else {}),
        }
    )
    print(f"[v6-evals] accepted={accepted} scores={scores}", flush=True)
    return {"tag": tag, "accepted": accepted, "sampler_path": selected_path, "scores": scores}


_G1_TAG_PATTERN = re.compile(r"g1(?:-[a-z0-9][a-z0-9-]*)?")


def g1_checkpoint_selection_key(report: Mapping[str, Any]) -> tuple[float, ...]:
    """Rank epochs without letting an always-idle model hide in aggregate accuracy.

    The weakest obligation/restraint slice is compared before strict row
    accuracy. Passing every hard gate remains the strongest signal. Remaining
    slice metrics provide deterministic tie-breakers.
    """

    summary = report.get("summary")
    hard_gates = report.get("hard_gates")
    if not isinstance(summary, Mapping) or not isinstance(hard_gates, Mapping):
        raise ValueError("g1 evaluation report is missing summary or hard_gates")
    metric_names = (
        "strict_row_accuracy",
        "should_fire_recall",
        "reminder_wait_accuracy",
        "ordinary_silence_idle_accuracy",
        "clause_boundary_accuracy",
        "canonical_exact_rate",
        "format_validity",
    )
    metrics: dict[str, float] = {}
    for name in metric_names:
        value = summary.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"g1 evaluation metric {name!r} is missing or non-numeric")
        metrics[name] = float(value)
    behavior_floor = min(
        metrics["should_fire_recall"],
        metrics["reminder_wait_accuracy"],
        metrics["ordinary_silence_idle_accuracy"],
        metrics["clause_boundary_accuracy"],
    )
    return (
        float(bool(hard_gates.get("passed"))),
        behavior_floor,
        metrics["strict_row_accuracy"],
        metrics["should_fire_recall"],
        metrics["reminder_wait_accuracy"],
        metrics["ordinary_silence_idle_accuracy"],
        metrics["clause_boundary_accuracy"],
        metrics["canonical_exact_rate"],
        metrics["format_validity"],
    )


def _validate_g1_local_inputs(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    base_model: str,
    seed: int,
) -> None:
    """Reject invalid paid-run inputs before any Tinker service is created."""

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise SystemExit("g1 epochs must be a positive integer.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise SystemExit("g1 batch size must be a positive integer.")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise SystemExit("g1 learning rate must be a finite positive number.")
    if not isinstance(base_model, str) or not base_model.strip():
        raise SystemExit("g1 base model must be a non-empty model identifier.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SystemExit("g1 seed must be a non-negative integer.")


def _validate_g1_dev_contract(dev_pairs: list[dict[str, Any]]) -> None:
    """Prove gold dev scoring and every selection slice locally."""

    gold_outputs = [str(pair.get("completion", "")) for pair in dev_pairs]
    report = evaluate_g1_predictions(dev_pairs, gold_outputs, label="g1-local-gold")
    g1_checkpoint_selection_key(report)
    if not report["hard_gates"]["passed"]:
        raise SystemExit("g1 gold dev rows fail their own evaluation gates.")


def _validate_g1_context_limit(
    tokenizer: Any,
    train_pairs: list[dict[str, Any]],
    dev_pairs: list[dict[str, Any]],
) -> None:
    """Fail locally before Tinker sees an over-limit train or dev row."""

    over: list[tuple[str, str, int]] = []
    for split, pairs in (("train", train_pairs), ("dev", dev_pairs)):
        for pair in pairs:
            count = g1_target_token_count(tokenizer, pair)
            if count > TINKER_MAX_TARGET_TOKENS:
                over.append((split, str(pair.get("candidate_id", "unknown")), count))
    if over:
        examples = ", ".join(
            f"{split}:{candidate_id}={count}" for split, candidate_id, count in over[:5]
        )
        raise SystemExit(
            f"g1 dataset has {len(over)} rows above Tinker's "
            f"{TINKER_MAX_TARGET_TOKENS}-target-token limit: {examples}. "
            "Rebuild with python -m scripts.g1_full_build."
        )


def stage_g1(
    tag: str = "g1",
    *,
    epochs: int = G1_EPOCHS,
    batch_size: int = G1_BATCH_SIZE,
    learning_rate: float = G1_LEARNING_RATE,
    base_model: str = G1_BASE_MODEL,
    seed: int = G1_SEED,
) -> dict[str, Any]:
    """Generation 1: stock `base_model` + ONE LoRA fine-tune on the g1 dataset.

    One stage, no curriculum — the g1 lineage never adapts from any v4-v6
    checkpoint (scripts/g1_runbook.md, "What g1 is"). Every configuration
    value here is the round-1 locked recipe from that runbook, "Round-1
    configuration — locked 2026-07-31":

      * adapter creation: rank 32, train_attn/train_mlp/train_unembed all on
        (passed explicitly), seed 650
      * optimizer (AdamParams, every step): lr 2e-4 constant/no schedule,
        beta1/beta2/eps = 0.9/0.95/1e-12, weight_decay 0, grad_clip_norm off
      * training loop: 3 epochs with per-epoch dev eval + best-epoch
        selection, batch size 16, shuffle Random(650 + 20 + epoch)
      * loss: cross-entropy with per-token weights — prompt tokens weight 0,
        completion tokens 1/len(completion) each so every card totals 1.0,
        end token graded. This is exactly what `build_datum` already does
        for schema_version == "g1" pairs: `example_weight_for_pair` returns
        1.0 (class weights are dead for g1 — "the count-engineered mix does
        the balancing"), `build_datum` divides by len(completion_ids), and
        completion_ids always ends with the appended <|im_end|> token, so
        the end token carries the same graded weight as the rest.
      * rendering: g1 system prompt + card as the user message, thinking
        disabled (system_prompt_for_pair/render_prompt_ids already do this
        for schema_version == "g1" pairs; nothing g1-specific to add here)
      * eval-time sampling: greedy, stop on newline, max 192 tokens,
        concurrency 8, timeout 600s (module-level MAX_TOKENS/EVAL_CONCURRENCY/
        REMOTE_TIMEOUT_SECONDS, unchanged from the other stages)

    Bookkeeping: state is written under the `g1:*` namespace ONLY (`--tag`
    is rejected if it would land in a historical v4:/v41:/v5:/v6: namespace);
    every report carries the provenance block (git commit, dataset sha + row
    count, base model, seed) via `run_eval`'s `base_model=`/`seed=` overrides;
    per-epoch and total wall-clock are recorded under `g1:*` keys — the old
    rounds never did this and it was missed (runbook, "Bookkeeping").
    """

    if _G1_TAG_PATTERN.fullmatch(tag) is None:
        raise SystemExit(
            f"--tag {tag!r} is outside the g1 namespace; use 'g1' or a "
            "lowercase g1-* tag."
        )

    _validate_g1_local_inputs(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        base_model=base_model,
        seed=seed,
    )

    import tinker

    train_name, dev_name = g1_train_dev_dataset_names()
    train_pairs = load_pairs(train_name)
    dev_pairs = load_pairs(dev_name)
    if any(pair.get("schema_version") != "g1" for pair in (*train_pairs, *dev_pairs)):
        raise SystemExit(
            f"{train_name}/{dev_name} must contain only schema_version='g1' rows."
        )
    if not train_pairs or not dev_pairs:
        raise SystemExit(f"{train_name}/{dev_name} must both contain at least one row.")
    _validate_g1_dev_contract(dev_pairs)

    tokenizer = get_tokenizer(base_model)
    _validate_g1_context_limit(tokenizer, train_pairs, dev_pairs)
    data = [build_datum(tinker, tokenizer, pair) for pair in train_pairs]
    service = tinker.ServiceClient(user_metadata={"project": f"smol-interactions-{tag}"})
    client = service.create_lora_training_client(
        base_model=base_model,
        rank=G1_LORA_RANK,
        seed=seed,
        train_attn=G1_TRAIN_ATTN,
        train_mlp=G1_TRAIN_MLP,
        train_unembed=G1_TRAIN_UNEMBED,
        user_metadata={"project": f"smol-interactions-{tag}"},
    )
    save_state_json(
        {
            f"{tag}:model_id": str(getattr(client, "model_id", "unknown")),
            f"{tag}:base_model": base_model,
            f"{tag}:seed": seed,
            f"{tag}:train_dataset": train_name,
            f"{tag}:dev_dataset": dev_name,
            f"{tag}:train_rows": len(train_pairs),
            f"{tag}:dev_rows": len(dev_pairs),
        }
    )

    print(
        f"[g1] base_model={base_model} rank={G1_LORA_RANK} seed={seed} "
        f"{len(data)} datums, {epochs} epochs, batch {batch_size}, lr {learning_rate}",
        flush=True,
    )

    adam_kwargs = dict(
        learning_rate=learning_rate,
        beta1=G1_ADAM_BETA1,
        beta2=G1_ADAM_BETA2,
        eps=G1_ADAM_EPS,
        weight_decay=G1_WEIGHT_DECAY,
        grad_clip_norm=G1_GRAD_CLIP_NORM,
    )

    step = 0
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_sampler_path = ""
    best_report: dict[str, Any] | None = None
    stage_started = time.monotonic()
    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        rng = Random(G1_SHUFFLE_BASE_SEED + epoch)
        order = list(range(len(data)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = [data[index] for index in order[start : start + batch_size]]
            started = time.monotonic()
            fb_future = client.forward_backward(batch, "cross_entropy")
            optim_future = client.optim_step(tinker.AdamParams(**adam_kwargs))
            fb_result = fb_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
            optim_future.result(timeout=REMOTE_TIMEOUT_SECONDS)
            step += 1
            metrics = getattr(fb_result, "metrics", None) or {}
            loss = metrics.get("loss:sum") if isinstance(metrics, dict) else None
            weight = metrics.get("loss_fn_output_weight:sum") if isinstance(metrics, dict) else None
            mean_loss = loss / weight if loss is not None and weight else None
            print(
                f"[g1] epoch={epoch} step={step} batch={len(batch)} "
                f"loss={mean_loss if mean_loss is not None else metrics} "
                f"seconds={time.monotonic() - started:.1f}",
                flush=True,
            )

        state_result = client.save_state(name=f"smol-{tag}-epoch{epoch}").result(
            timeout=REMOTE_TIMEOUT_SECONDS
        )
        save_state_json({f"{tag}:state_epoch{epoch}": _path_of(state_result)})

        epoch_sampler = client.save_weights_and_get_sampling_client()
        report = run_eval(
            epoch_sampler,
            tokenizer,
            dev_pairs,
            f"dev-{tag}-epoch{epoch}",
            base_model=base_model,
            seed=seed,
        )
        selection_key = g1_checkpoint_selection_key(report)
        strict_accuracy = float(report["summary"]["strict_row_accuracy"])
        epoch_wall_seconds = time.monotonic() - epoch_started
        save_state_json({f"{tag}:epoch{epoch}_wall_seconds": epoch_wall_seconds})
        print(f"[g1] epoch={epoch} wall_seconds={epoch_wall_seconds:.1f}", flush=True)
        if best_key is None or selection_key > best_key:
            sampler_result = client.save_weights_for_sampler(
                name=f"smol-{tag}-best-epoch{epoch}"
            ).result(timeout=REMOTE_TIMEOUT_SECONDS)
            best_sampler_path = _path_of(sampler_result)
            best_key = selection_key
            best_epoch = epoch
            best_report = report
            save_state_json(
                {
                    f"{tag}:best_epoch": epoch,
                    f"{tag}:best_dev_score": strict_accuracy,
                    f"{tag}:best_dev_selection_key": list(selection_key),
                    f"{tag}:sampler_path": best_sampler_path,
                }
            )

    if not best_sampler_path or best_report is None or best_key is None:
        raise RuntimeError("g1 training completed without a selectable dev checkpoint.")
    total_wall_seconds = time.monotonic() - stage_started
    save_state_json({f"{tag}:total_wall_seconds": total_wall_seconds})
    print(
        f"[g1] selected epoch {best_epoch}: strict="
        f"{best_report['summary']['strict_row_accuracy']} key={best_key} "
        f"path={best_sampler_path} total_wall_seconds={total_wall_seconds:.1f}",
        flush=True,
    )
    return {
        "tag": tag,
        "best_epoch": best_epoch,
        "best_score": float(best_report["summary"]["strict_row_accuracy"]),
        "selection_key": best_key,
        "sampler_path": best_sampler_path,
        "dev_report": best_report,
        "total_wall_seconds": total_wall_seconds,
    }


def _sampler_path(tag: str) -> str:
    if not STATE_PATH.exists():
        raise SystemExit(f"Missing {STATE_PATH}; no trained v4 checkpoint is available.")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    path = state.get(f"{tag}:sampler_path")
    if not path:
        raise SystemExit(f"No sampler path stored for {tag!r}.")
    return str(path)


def _g1_checkpoint_provenance(tag: str) -> tuple[str, int]:
    """Load the exact tokenizer/model provenance saved by ``stage_g1``."""

    if not STATE_PATH.exists():
        raise SystemExit(f"Missing {STATE_PATH}; no trained g1 checkpoint is available.")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    base_model = state.get(f"{tag}:base_model")
    seed = state.get(f"{tag}:seed")
    if not isinstance(base_model, str) or not base_model.strip():
        raise SystemExit(f"Missing or invalid {tag}:base_model in {STATE_PATH}.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SystemExit(f"Missing or invalid {tag}:seed in {STATE_PATH}.")
    return base_model, seed


def stage_final(tag: str, split: str = "test") -> dict[str, Any]:
    global ACTIVE_SYSTEM_PROMPT
    sampler_path = _sampler_path(tag)
    is_g1 = _G1_TAG_PATTERN.fullmatch(tag) is not None
    g1_base_model, g1_seed = (
        _g1_checkpoint_provenance(tag) if is_g1 else (None, None)
    )
    tokenizer = get_tokenizer(g1_base_model)
    pairs = load_pairs(split)

    import tinker

    print(f"[final] evaluating selected checkpoint on {split}: {sampler_path}", flush=True)
    service = tinker.ServiceClient(user_metadata={"project": f"smol-interactions-{tag}-final"})
    sampler = service.create_sampling_client(model_path=sampler_path)
    previous_prompt = ACTIVE_SYSTEM_PROMPT
    gates = None
    if tag.startswith("v5"):
        ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT_V5
    elif tag.startswith("v6"):
        ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT_V6
        gates = V6_HARD_GATES
    try:
        report = run_eval(
            sampler,
            tokenizer,
            pairs,
            f"final-{tag}-{split}",
            hard_gates=gates,
            base_model=g1_base_model,
            seed=g1_seed,
        )
        score = (
            report["summary"]["strict_row_accuracy"]
            if is_g1
            else report["summary"]["overall_score"]
        )
        save_state_json(
            {
                f"{tag}:final_{split}_score": score,
                f"{tag}:final_{split}_gates_passed": report["hard_gates"]["passed"],
            }
        )
        return report
    finally:
        ACTIVE_SYSTEM_PROMPT = previous_prompt


def stage_evalpath(checkpoint: str, label: str, split: str) -> dict[str, Any]:
    import tinker

    if not checkpoint:
        raise SystemExit("--checkpoint is required for --stage evalpath")
    service = tinker.ServiceClient(user_metadata={"project": "smol-interactions-v4-evalpath"})
    sampler = service.create_sampling_client(model_path=checkpoint)
    return run_eval(sampler, get_tokenizer(), load_pairs(split), label)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "preflight",
            "baseline",
            "train",
            "resume",
            "adapt",
            "v5",
            "v6",
            "v6-evals",
            "v6-promote",
            "g1",
            "rescore",
            "final",
            "evalpath",
            "loop",
        ],
        required=True,
    )
    parser.add_argument("--tag", default="v4")
    parser.add_argument("--checkpoint", help="tinker:// path for --stage evalpath")
    parser.add_argument("--report", help="saved evaluation report for --stage rescore")
    parser.add_argument("--label", default="manual-v4-eval")
    parser.add_argument(
        "--split",
        choices=[
            "dev", "dev_dense", "test", "dev_v41", "dev_v41_dense", "test_v41",
            "dev_v5", "test_v5", "train_v6", "dev_v6", "test_v6_restraint",
            "train_g1_demo1", "dev_g1_demo1", "train_g1", "dev_g1",
        ],
        default="dev",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-only", action="store_true",
                        help="v6 stage: train + save pending checkpoint, no Tinker evals/gates")
    parser.add_argument("--from-epoch", type=int, default=2)
    parser.add_argument("--source-tag", default="v4")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--base-model", default=G1_BASE_MODEL)
    parser.add_argument("--seed", type=int, default=G1_SEED)
    return parser


_LEGACY_MUTATING_STAGES = frozenset(
    {"train", "resume", "adapt", "v5", "v6", "v6-evals", "v6-promote", "loop"}
)
_LEGACY_SOURCE_TAG_STAGES = frozenset({"adapt", "v5", "v6", "v6-evals"})


def validate_cli_namespace(arguments: argparse.Namespace) -> None:
    """Fail before credentials/network when a CLI stage could cross g1 lineages."""

    stage = str(arguments.stage)
    tag = str(arguments.tag)
    source_tag = str(arguments.source_tag)
    is_g1_tag = _G1_TAG_PATTERN.fullmatch(tag) is not None
    is_g1_source = _G1_TAG_PATTERN.fullmatch(source_tag) is not None

    if stage == "g1":
        effective_tag = "g1" if tag == "v4" else tag
        if _G1_TAG_PATTERN.fullmatch(effective_tag) is None:
            raise SystemExit(
                f"--stage g1 requires --tag 'g1' or a lowercase g1-* tag, got {tag!r}."
            )
        return

    if stage in _LEGACY_MUTATING_STAGES and is_g1_tag:
        raise SystemExit(
            f"--stage {stage} cannot use reserved g1 tag {tag!r}; use --stage g1."
        )
    if stage in _LEGACY_SOURCE_TAG_STAGES and is_g1_source:
        raise SystemExit(
            f"--stage {stage} cannot use reserved g1 source tag {source_tag!r}."
        )


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()

    validate_cli_namespace(arguments)
    load_env_key()
    if arguments.stage == "preflight":
        stage_preflight()
    elif arguments.stage == "baseline":
        stage_baseline(arguments.split)
    elif arguments.stage == "train":
        stage_train(
            arguments.tag,
            epochs=arguments.epochs or EPOCHS,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate or LEARNING_RATE,
        )
    elif arguments.stage == "resume":
        stage_resume(
            arguments.tag,
            from_epoch=arguments.from_epoch,
            epochs=arguments.epochs or EPOCHS,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate or LEARNING_RATE,
        )
    elif arguments.stage == "adapt":
        stage_adapt(
            arguments.tag,
            source_tag=arguments.source_tag,
            source_epoch=arguments.from_epoch,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate or ADAPT_LEARNING_RATE,
        )
    elif arguments.stage == "v5":
        stage_v5(
            arguments.tag if arguments.tag != "v4" else "v5",
            source_tag=arguments.source_tag if arguments.source_tag != "v4" else "v41",
            epochs=arguments.epochs or 1,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate or ADAPT_LEARNING_RATE,
        )
    elif arguments.stage == "v6":
        stage_v6(
            arguments.tag if arguments.tag != "v4" else "v6",
            source_tag=arguments.source_tag if arguments.source_tag != "v4" else "v41",
            epochs=arguments.epochs or 1,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate or ADAPT_LEARNING_RATE,
            train_only=arguments.train_only,
        )
    elif arguments.stage == "v6-promote":
        stage_v6_promote(
            arguments.tag if arguments.tag != "v4" else "v6",
            epochs=arguments.epochs or 1,
        )
    elif arguments.stage == "v6-evals":
        stage_v6_evals(
            arguments.tag if arguments.tag != "v4" else "v6",
            source_tag=arguments.source_tag if arguments.source_tag != "v4" else "v41",
        )
    elif arguments.stage == "g1":
        stage_g1(
            arguments.tag if arguments.tag != "v4" else "g1",
            epochs=arguments.epochs if arguments.epochs is not None else G1_EPOCHS,
            batch_size=arguments.batch_size,
            learning_rate=(
                arguments.learning_rate
                if arguments.learning_rate is not None
                else G1_LEARNING_RATE
            ),
            base_model=arguments.base_model,
            seed=arguments.seed,
        )
    elif arguments.stage == "rescore":
        source = Path(arguments.report) if arguments.report else OUT_DIR / "final-v4-test_eval.json"
        rescore_report(source, arguments.split, arguments.label)
    elif arguments.stage == "final":
        stage_final(arguments.tag, arguments.split)
    elif arguments.stage == "evalpath":
        stage_evalpath(arguments.checkpoint or "", arguments.label, arguments.split)
    elif arguments.stage == "loop":
        stage_preflight()
        stage_baseline("dev")
        stage_train(
            arguments.tag,
            epochs=arguments.epochs or EPOCHS,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate or LEARNING_RATE,
        )
        stage_final(arguments.tag, "test")


if __name__ == "__main__":
    main()
