"""Local MLX serving for legacy checkpoints and the trained g1 policy.

Pipeline (run once, in order):
    python scripts/local_model.py merge      # PEFT LoRA -> merged bf16 HF model
    python scripts/local_model.py convert    # merged HF -> quantized MLX (4-bit)
    python scripts/local_model.py smoke      # one prompt through the local model

For g1, `scripts.g1_merge_tinker_adapter` produces the selected 8-bit model.
`LocalMLXPolicy` then uses the exact training template, greedy decoding, an
invariant startup prefix, and a persistent per-stream cache. Decoding stops
only after the model generates a complete action envelope.

Quantized local weights are not the weights the Tinker column was measured on.
Local scores get their own column and are never written over the Tinker ones.
"""

from __future__ import annotations

import copy
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE_MODEL = "Qwen/Qwen3-8B"
# Which local checkpoint this process serves/builds; rounds slot in via env.
LOCAL_NAME = os.environ.get("LOCAL_BUILD_NAME", "text-gpt-live")
ADAPTER_DIR = ROOT / "models" / LOCAL_NAME
MERGED_DIR = ROOT / "models" / f"{LOCAL_NAME}-merged"
MLX_BITS = int(os.environ.get("LOCAL_QUANT_BITS", "4"))
MLX_DIR = ROOT / "models" / f"{LOCAL_NAME}-mlx-{MLX_BITS}bit"
MAX_TOKENS = 192
G1_MAX_TOKENS = 192
G1_PREDICTION_MARKER = "<PREDICT_THIS_ACTION>"
G1_QUANTIZED_DIR = ROOT / "models" / "text-gpt-live-mlx-8bit"
DEFAULT_G1_SESSION_CACHE_LIMIT = 8


@dataclass
class _G1StreamState:
    stable_cache: Any | None
    stable_tokens: list[int]


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def g1_prompt_segments(tokenizer: Any, system_prompt: str, user_prompt: str) -> tuple[list[int], list[int]]:
    """Return the stable-through-current-event prefix and generation suffix.

    The current event is the durable boundary. The prediction marker and chat
    assistant prefix are temporary: the next tick replaces them with the
    model's previous action and a new event. Computing the boundary from the
    rendered template, then proving it is an exact token prefix, prevents an
    unsafe cache hit at a tokenizer merge boundary.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    marker = rendered.rfind(G1_PREDICTION_MARKER)
    if marker < 0 or not user_prompt.endswith(G1_PREDICTION_MARKER):
        raise ValueError("g1 prompt must end with <PREDICT_THIS_ACTION>.")
    stable_text = rendered[:marker].rstrip("\n")
    stable = _token_ids(tokenizer.encode(stable_text, add_special_tokens=False))
    full = _token_ids(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if full[: len(stable)] != stable:
        raise RuntimeError("The g1 cache boundary is not an exact tokenizer prefix.")
    suffix = full[len(stable) :]
    if not suffix:
        raise RuntimeError("The g1 generation suffix is empty.")
    return stable, suffix


def g1_static_prefix(tokenizer: Any, system_prompt: str) -> list[int]:
    """Return the invariant system and user-role prefix for every g1 stream."""

    sentinel = "__G1_STREAM_CONTENT_SENTINEL_9f7c__"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sentinel},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    position = rendered.find(sentinel)
    if position < 0:
        raise RuntimeError("The chat template did not preserve the g1 user-content sentinel.")
    prefix = _token_ids(tokenizer.encode(rendered[:position], add_special_tokens=False))
    full = _token_ids(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if full[: len(prefix)] != prefix:
        raise RuntimeError("The static g1 prefix ends inside a tokenizer merge boundary.")
    return prefix


def clone_prompt_cache(cache: Any) -> Any:
    """Clone cache containers while sharing immutable MLX array snapshots.

    MLX arrays are immutable values; cache updates assign new arrays. A deep
    copy therefore wastes time proportional to the full KV history. ArraysCache
    is the one container with a mutable Python list, so that list alone must be
    copied to keep generated recurrent state out of the stable cache.
    """

    cloned = []
    for entry in cache:
        item = copy.copy(entry)
        if hasattr(entry, "cache") and isinstance(entry.cache, list):
            item.cache = list(entry.cache)
        cloned.append(item)
    return cloned


def completed_action(pieces: list[str]) -> str | None:
    """Return a model-generated complete envelope as soon as its tag closes."""

    value = "".join(pieces)
    end = value.find("</action>")
    if end >= 0:
        return value[: end + len("</action>")]
    return None


def stage_merge() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge] loading {BASE_MODEL} (bf16, low mem)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cpu"
    )
    print("[merge] applying adapter", flush=True)
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR), device_map="cpu")
    model = model.merge_and_unload()
    print(f"[merge] saving merged model -> {MERGED_DIR}", flush=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR), max_shard_size="4GB")
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(str(MERGED_DIR))
    print("[merge] done", flush=True)


def stage_convert() -> None:
    from mlx_lm import convert

    print(f"[convert] {MERGED_DIR} -> {MLX_DIR} ({MLX_BITS}-bit)", flush=True)
    convert(str(MERGED_DIR), mlx_path=str(MLX_DIR), quantize=True, q_bits=MLX_BITS, q_group_size=64)
    print("[convert] done", flush=True)


class LocalMLXPolicy:
    """Greedy local sampler over the quantized merged checkpoint.

    Mirrors BaseModelTinkerPolicy's contract: __call__(prompt) -> raw action
    line. Rendering matches training exactly (system prompt + chat template,
    thinking disabled); generation is temp-0 and cut at the first newline.
    """

    mode = "local-mlx"
    display_name = f"{LOCAL_NAME} (local mlx {MLX_BITS}-bit)"
    availability = "ready"
    availability_message = ""

    def __init__(
        self,
        model_path: str | None = None,
        system_prompt: str | None = None,
        adapter_path: str | None = None,
        kv_bits: int | None = None,
    ) -> None:
        from mlx_lm import load

        from app.policy import SYSTEM_PROMPT_G1, SYSTEM_PROMPT_V6

        self.system_prompt = system_prompt or SYSTEM_PROMPT_V6
        self.action_schema = "g1" if self.system_prompt == SYSTEM_PROMPT_G1 else "legacy"
        import os as _os

        adapter = adapter_path or _os.environ.get("LOCAL_ADAPTER_PATH") or None
        configured_model = _os.environ.get("LOCAL_MODEL_PATH")
        default_model = G1_QUANTIZED_DIR if self.action_schema == "g1" else MLX_DIR
        selected_model = model_path or configured_model or str(default_model)
        self.model, self.tokenizer = load(
            selected_model,
            adapter_path=adapter,
        )
        self.display_name = f"{Path(selected_model).name} (local MLX)"
        if adapter:
            self.display_name += f" + {Path(adapter).name}"
        self._inference_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="text-gpt-live-mlx")
        self._closed = False
        self._stream_caches: OrderedDict[str, _G1StreamState] = OrderedDict()
        self._session_cache_limit = int(
            _os.environ.get("LOCAL_SESSION_CACHE_LIMIT", DEFAULT_G1_SESSION_CACHE_LIMIT)
        )
        if self._session_cache_limit < 1:
            raise ValueError("LOCAL_SESSION_CACHE_LIMIT must be at least 1.")
        self._static_cache: Any | None = None
        self._static_tokens: list[int] = []
        configured_kv_bits = _os.environ.get("LOCAL_KV_BITS")
        self.kv_bits = kv_bits if kv_bits is not None else (
            int(configured_kv_bits) if configured_kv_bits else None
        )
        if self.kv_bits not in {None, 4, 8}:
            raise ValueError("LOCAL_KV_BITS must be unset, 4, or 8.")
        self.last_metrics: dict[str, Any] = {}
        if self.action_schema == "g1":
            self._static_tokens = g1_static_prefix(self.tokenizer, self.system_prompt)
            static_state = _G1StreamState(stable_cache=None, stable_tokens=[])
            self._append_to_stable_cache(static_state, self._static_tokens)
            self._static_cache = clone_prompt_cache(static_state.stable_cache)

    def warm_up(self) -> None:
        if self.action_schema == "g1":
            prompt = (
                '<stream_event index="1" source="user" state="idle" '
                'time="t+650ms"></stream_event>\n<PREDICT_THIS_ACTION>'
            )
            self._executor.submit(self.complete, prompt, "__warmup__").result()
            self.reset_stream_cache()
        else:
            self._executor.submit(self.complete, "warm up").result()

    def reset_stream_cache(self, session_id: str | None = None) -> None:
        """Forget one live stream, or every stream, without unloading weights."""
        with self._inference_lock:
            if session_id is None:
                self._stream_caches.clear()
            else:
                self._stream_caches.pop(session_id, None)
            self.last_metrics = {}

    def _new_stream_state(self) -> _G1StreamState:
        return _G1StreamState(
            stable_cache=(
                clone_prompt_cache(self._static_cache)
                if self._static_cache is not None
                else None
            ),
            stable_tokens=list(self._static_tokens),
        )

    def _stream_state(self, session_id: str) -> _G1StreamState:
        state = self._stream_caches.pop(session_id, None)
        if state is None:
            state = self._new_stream_state()
        self._stream_caches[session_id] = state
        while len(self._stream_caches) > self._session_cache_limit:
            self._stream_caches.popitem(last=False)
        return state

    def complete(self, user_prompt: str, session_id: str = "__default__") -> str:
        with self._inference_lock:
            if self.action_schema == "g1":
                return self._complete_g1_cached(user_prompt, session_id)
            return self._complete_uncached(user_prompt)

    def _complete_uncached(self, user_prompt: str) -> str:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        pieces: list[str] = []
        output: str | None = None
        for response in stream_generate(
            self.model, self.tokenizer, prompt,
            max_tokens=MAX_TOKENS, sampler=make_sampler(temp=0.0),
        ):
            pieces.append(response.text)
            if self.action_schema == "g1":
                output = completed_action(pieces)
                if output is not None:
                    break
            elif "\n" in "".join(pieces):
                output = "".join(pieces).split("\n", 1)[0]
                break
        if output is None:
            output = (
                "".join(pieces)
                if self.action_schema == "g1"
                else "".join(pieces).split("\n", 1)[0]
            )
        return output if self.action_schema == "g1" else output.strip()

    def _append_to_stable_cache(self, state: _G1StreamState, tokens: list[int]) -> None:
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache

        if state.stable_cache is None:
            state.stable_cache = make_prompt_cache(self.model)
            if self.kv_bits is not None:
                state.stable_cache = [
                    QuantizedKVCache(group_size=64, bits=self.kv_bits)
                    if isinstance(entry, KVCache)
                    else entry
                    for entry in state.stable_cache
                ]
        for start in range(0, len(tokens), 256):
            chunk = mx.array(tokens[start : start + 256])
            self.model(chunk[None], cache=state.stable_cache)
            mx.eval([entry.state for entry in state.stable_cache])
            mx.clear_cache()

    def _complete_g1_cached(self, user_prompt: str, session_id: str) -> str:
        """Generate from a copy of a token-exact, persistent stream cache.

        Qwen3.5 mixes KV attention with recurrent linear-attention state. The
        recurrent state cannot be trimmed after generation, so each decision
        runs on a cheap structural copy while the stable prefix cache remains
        untouched. Every returned character still comes from greedy model
        decoding; the cache only avoids recomputing unchanged history.
        """

        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        started = time.perf_counter()
        stable, suffix = g1_prompt_segments(self.tokenizer, self.system_prompt, user_prompt)
        state = self._stream_state(session_id)
        cache_reset = stable[: len(state.stable_tokens)] != state.stable_tokens
        if cache_reset:
            state = self._new_stream_state()
            self._stream_caches[session_id] = state
            if stable[: len(state.stable_tokens)] != state.stable_tokens:
                state = _G1StreamState(stable_cache=None, stable_tokens=[])
                self._stream_caches[session_id] = state
        delta = stable[len(state.stable_tokens) :]
        prefill_started = time.perf_counter()
        self._append_to_stable_cache(state, delta)
        prefill_finished = time.perf_counter()
        state.stable_tokens = stable

        generation_cache = clone_prompt_cache(state.stable_cache)
        first_token_at: float | None = None
        pieces: list[str] = []
        output_tokens = 0
        output: str | None = None
        for response in stream_generate(
            self.model,
            self.tokenizer,
            suffix,
            max_tokens=G1_MAX_TOKENS,
            sampler=make_sampler(temp=0.0),
            prompt_cache=generation_cache,
        ):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            pieces.append(response.text)
            output_tokens += 1
            output = completed_action(pieces)
            if output is not None:
                break
        finished = time.perf_counter()
        if first_token_at is None:
            raise RuntimeError("MLX generation yielded no tokens.")
        if output is None:
            output = "".join(pieces)
        self.last_metrics = {
            "latency_ms": round((finished - started) * 1000, 3),
            "first_token_ms": round((first_token_at - started) * 1000, 3),
            "stable_tokens": len(stable),
            "reused_tokens": len(stable) - len(delta),
            "appended_tokens": len(delta),
            "suffix_tokens": len(suffix),
            "output_tokens": output_tokens,
            "cache_reset": cache_reset,
            "kv_bits": self.kv_bits,
        }
        del generation_cache
        mx.clear_cache()
        return output

    # rollout_eval's layer-1 restraint path calls the Tinker policy's private
    # sampler directly; expose the same name.
    def _sample_sync(self, compiled_stream: str) -> str:
        return self.complete(compiled_stream)

    async def predict(self, compiled_stream, current, history) -> str:
        import asyncio

        del current, history
        return await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self.complete,
            compiled_stream,
        )

    async def predict_session(self, session_id, compiled_stream, current, history) -> str:
        """Run one decision against the cache owned by ``session_id``."""
        import asyncio

        del current, history
        return await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self.complete,
            compiled_stream,
            session_id,
        )

    async def aclose(self) -> None:
        """Release the dedicated inference worker and all session caches."""
        import asyncio

        if self._closed:
            return
        self._closed = True
        self.reset_stream_cache()
        await asyncio.to_thread(self._executor.shutdown, True, cancel_futures=True)


def stage_smoke() -> None:
    policy = LocalMLXPolicy()
    out = policy.complete(
        '<stream_event index="1" source="user" state="active" time="t+650ms">The quick brown fox</stream_event>\n'
        "<PREDICT_THIS_ACTION>"
    )
    print("smoke output:", out)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    {"merge": stage_merge, "convert": stage_convert, "smoke": stage_smoke}[stage]()
