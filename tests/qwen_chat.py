"""Launch a standalone local Qwen chatbot with llama.cpp."""

from __future__ import annotations

import argparse
import os
import shutil
import sys


DEFAULT_MODEL = "ggml-org/Qwen3-0.6B-GGUF:Q4_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face GGUF repo and quant selector",
    )
    parser.add_argument("--context-size", type=int, default=8192)
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful, concise assistant. /no_think",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    llama_cli = shutil.which("llama-cli")
    if llama_cli is None:
        print(
            "llama-cli is not installed. On macOS, run: brew install llama.cpp",
            file=sys.stderr,
        )
        return 2

    command = [
        llama_cli,
        "-hf",
        args.model,
        "--conversation",
        "--ctx-size",
        str(args.context_size),
        "--n-gpu-layers",
        "all",
        "--system-prompt",
        args.system_prompt,
        "--temperature",
        "0.7",
        "--top-p",
        "0.8",
        "--top-k",
        "20",
        "--min-p",
        "0",
    ]
    print(f"Loading {args.model} (the first run downloads and caches the GGUF)...")
    os.execv(llama_cli, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
