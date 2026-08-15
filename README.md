# Text GPT-Live

<img src="static/demo_vid.gif" alt="Text GPT-Live demo" width="760" />

**TL;DR:** We trained Qwen3.5-4B to process a continuous stream of text instead
of waiting for completed turns. Every 650 ms, it reads the current textbox and
chooses whether to stay silent, respond, edit, highlight, translate, search, or
delegate work—allowing it to act while the user is still typing. The three-stage
LoRA training cost approximately $50.

You can read the full blog post [here](https://huyxdang.com/text-gpt-live).

The released [model](https://huggingface.co/huyxdang/text-gpt-live) and
[dataset](https://huggingface.co/datasets/huyxdang/text-gpt-live-dataset) are
available on Hugging Face.

## Run the code

Local inference requires an Apple silicon Mac and Python 3.11 or newer.

```bash
git clone https://github.com/huyxdang/Text-GPT-Live.git
cd Text-GPT-Live

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.txt
.venv/bin/hf download huyxdang/text-gpt-live \
  --local-dir models/text-gpt-live-mlx-8bit

POLICY_MODE=local \
POLICY_PROMPT=g1 \
LOCAL_MODEL_PATH=models/text-gpt-live-mlx-8bit \
.venv/bin/python -m uvicorn app.main:app
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).
