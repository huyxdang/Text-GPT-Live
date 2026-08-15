# Text GPT-Live

<img src="static/demo_vid.gif" alt="Text GPT-Live demo" width="760" />

The goal: post-train a model for real-time interactivity, with an affordable
budget.

We trained Qwen3.5-4B to process a continuous stream of text. This is in
contrast to normal chatbots like ChatGPT or Claude.ai, which only takes in the
completed text as input and waits for its turn. Our model acts in 650ms
micro-turns, which allows for real-time interactions.

The model reads the current text and takes one of the following options:

- stay silent
- respond
- translate
- highlight
- search (tool)
- delegate work (tool)

The whole post-training process, consisted of three LoRA stages, cost almost
$50 in total.

You can read the full blog post [here](https://huyxdang.com/text-gpt-live).

The released [model](https://huggingface.co/huyxdang/text-gpt-live) and
[dataset](https://huggingface.co/datasets/huyxdang/text-gpt-live-dataset) are
available on Hugging Face.

## What's included

- Training pipeline for reproducing Text GPT-Live on Tinker
- Browser UI and continuous interaction runtime for running the model locally

## Run locally

You need:

- An Apple silicon Mac. The released checkpoint uses MLX.
- Python 3.11 or newer
- Git and the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
- About 6 GB of free disk space for the model
- An internet connection for the download and live web search

Install the Hugging Face CLI if you do not already have it:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
```

Then clone and run the project:

```bash
git clone https://github.com/huyxdang/Text-GPT-Live.git
cd Text-GPT-Live

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.txt
hf download huyxdang/text-gpt-live \
  --local-dir models/text-gpt-live-mlx-8bit

cp .env.example .env
.venv/bin/python -m uvicorn app.main:app
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

No API key is required for this local path. The model and dataset are public,
live search uses DDGS, and generative UI falls back to a deterministic demo
provider. To generate the UI with OpenAI instead, add `OPENAI_API_KEY` to
`.env` and set `UIGEN_MODE=openai`.

## Reproduce the training

Training is separate from local inference. It requires:

- Python 3.11 or newer
- A [Tinker](https://tinker-docs.thinkingmachines.ai/) account
- A funded `TINKER_API_KEY`; Tinker training is a paid service
- An internet connection

The released model used three sequential one-epoch LoRA stages. The original
runs cost approximately $50 in total, but current pricing and exact cost may
vary.

```bash
git clone https://github.com/huyxdang/Text-GPT-Live.git
cd Text-GPT-Live

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-tinker.txt
cp .env.example .env
```

Add your key to `.env`:

```dotenv
TINKER_API_KEY=tml-...
```

Run the stages in order:

```bash
# Stage 1: core interaction behavior
.venv/bin/python -m scripts.g1_full_build
.venv/bin/python -m train.tinker_run --stage g1 --epochs 1

# Stage 2: translation and search
.venv/bin/python -m scripts.g1_v2_build
.venv/bin/python -m scripts.g1_v2_train

# Stage 3: final delivery repair
.venv/bin/python -m scripts.g1_v2_delivery_repair_build --index-mode fixed
.venv/bin/python -m scripts.g1_v2_delivery_repair_train
```

Each stage evaluates its candidate before promoting it, and the later stages
resume from the accepted Tinker state produced by the previous stage. See the
[training runbook](docs/training.md) for the locked recipe, expected dataset
sizes, evaluation gates, artifacts, and local export instructions.
