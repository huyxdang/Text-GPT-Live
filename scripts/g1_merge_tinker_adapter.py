"""Merge a Tinker PEFT adapter into Qwen3.5 and save a quantized MLX model.

Tinker's Qwen3.5 adapter exposes separate linear-attention q/k/v LoRA
matrices, while MLX-LM stores the base projection as one qkv matrix. This
converter applies the exact B@A deltas, concatenates q/k/v in model order,
preserves the input embedding, materializes a distinct adapted output head,
then quantizes the genuinely merged model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = ROOT / "models" / "Qwen3.5-4B"
DEFAULT_ADAPTER = ROOT / "models" / "smol-g1-v2-delivery-repair-lora"
DEFAULT_OUTPUT = ROOT / "models" / "smol-g1-v2-delivery-repair-fixed-mlx-8bit"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge(base: Path, adapter: Path, output: Path, *, bits: int, group_size: int) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"Output already exists: {output}")

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.utils import quantize_model, save

    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = int(adapter_config["r"])
    scale = float(adapter_config["lora_alpha"]) / rank
    if adapter_config.get("use_rslora"):
        raise ValueError("RS-LoRA adapters require a different scaling rule.")

    print(f"[merge] loading {base}", flush=True)
    model, tokenizer, model_config = load(str(base), return_config=True, lazy=False)
    tensors = mx.load(str(weights_path))
    pairs: dict[str, dict[str, Any]] = {}
    prefix = "base_model.model.model."
    for key, value in tensors.items():
        if not key.startswith(prefix) or not key.endswith(".weight"):
            raise ValueError(f"Unexpected adapter key: {key}")
        short = key[len(prefix) : -len(".weight")]
        if short.endswith(".lora_A"):
            module = short[: -len(".lora_A")]
            pairs.setdefault(module, {})["A"] = value
        elif short.endswith(".lora_B"):
            module = short[: -len(".lora_B")]
            pairs.setdefault(module, {})["B"] = value
        else:
            raise ValueError(f"Unexpected adapter tensor: {key}")
    if any(set(value) != {"A", "B"} for value in pairs.values()):
        raise ValueError("Every LoRA module must have exactly A and B tensors.")

    modules = dict(model.named_modules())

    def delta(module: str):
        value = pairs[module]
        return scale * (value["B"] @ value["A"])

    applied: set[str] = set()
    qkv_groups = 0
    standard_modules = 0
    for layer in range(len(model.layers)):
        names = [f"layers.{layer}.linear_attn.in_proj_{name}" for name in ("q", "k", "v")]
        present = [name in pairs for name in names]
        if any(present) and not all(present):
            raise ValueError(f"Incomplete q/k/v adapter group at layer {layer}.")
        if not all(present):
            continue
        path = f"language_model.model.layers.{layer}.linear_attn.in_proj_qkv"
        module = modules[path]
        update = mx.concatenate([delta(name) for name in names], axis=0)
        if update.shape != module.weight.shape:
            raise ValueError(f"qkv delta shape mismatch at layer {layer}: {update.shape} != {module.weight.shape}")
        module.weight = (module.weight + update).astype(module.weight.dtype)
        mx.eval(module.weight)
        applied.update(names)
        qkv_groups += 1
        print(f"[merge] qkv layer {layer + 1}/{len(model.layers)}", flush=True)

    unembed = "unembed_tokens"
    if unembed not in pairs:
        raise ValueError("Tinker adapter is missing the trained output head.")
    embedding = model.language_model.model.embed_tokens.weight
    output_head = nn.Linear(embedding.shape[1], embedding.shape[0], bias=False)
    output_head.weight = (embedding + delta(unembed)).astype(embedding.dtype)
    mx.eval(output_head.weight)
    model.language_model.lm_head = output_head
    model.language_model.args.tie_word_embeddings = False
    model_config["tie_word_embeddings"] = False
    if isinstance(model_config.get("text_config"), dict):
        model_config["text_config"]["tie_word_embeddings"] = False
    applied.add(unembed)
    print("[merge] materialized adapted output head", flush=True)

    for name in sorted(pairs):
        if name in applied:
            continue
        path = "language_model.model." + name
        module = modules.get(path)
        if module is None or not hasattr(module, "weight"):
            raise ValueError(f"No MLX linear module for adapter target {name!r} ({path!r}).")
        update = delta(name)
        if update.shape != module.weight.shape:
            raise ValueError(f"Delta shape mismatch for {name}: {update.shape} != {module.weight.shape}")
        module.weight = (module.weight + update).astype(module.weight.dtype)
        mx.eval(module.weight)
        applied.add(name)
        standard_modules += 1
    if applied != set(pairs):
        raise ValueError(f"Unapplied adapter modules: {sorted(set(pairs) - applied)}")

    print(f"[merge] quantizing {bits}-bit group={group_size}", flush=True)
    model, model_config = quantize_model(model, model_config, group_size, bits)
    mx.eval(model.parameters())
    save(output, base, model, tokenizer, model_config)
    provenance = {
        "schema_version": "g1-tinker-to-mlx-merge-1",
        "base": str(base.resolve()),
        "adapter": str(adapter.resolve()),
        "adapter_sha256": _sha256(weights_path),
        "rank": rank,
        "scale": scale,
        "adapter_modules": len(pairs),
        "qkv_groups": qkv_groups,
        "standard_modules": standard_modules,
        "separate_output_head": True,
        "quantization": {"bits": bits, "group_size": group_size, "mode": "affine"},
        "output": str(output.resolve()),
    }
    (output / "MERGE_PROVENANCE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2), flush=True)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    merge(args.base, args.adapter, args.output, bits=args.bits, group_size=args.group_size)


if __name__ == "__main__":
    main()
