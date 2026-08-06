# Text GPT-Live

A small language model trained to act *while* you type, instead of waiting for
you to finish.

Every 650 ms the browser sends the current contents of the textbox. The model
reads that stream — including pauses, corrections, and an empty box — and
predicts exactly one action: stay silent, reply, highlight a word, translate a
committed clause, start a background task, or run a search. Nothing in the
runtime decides whether the user has finished talking. The model does.

- **Write-up:** [Can you train a GPT-Live for $50?](https://huyxdang.com/text-gpt-live)
- **Model:** [huyxdang/text-gpt-live](https://huggingface.co/huyxdang/text-gpt-live)
- **Dataset:** [huyxdang/text-gpt-live-dataset](https://huggingface.co/datasets/huyxdang/text-gpt-live-dataset)

Base model is `Qwen/Qwen3.5-4B` with a rank-32 LoRA, trained in three
sequential SFT stages. All three stages cost about $50 on Tinker.

## The action grammar

The model emits one line per tick, from a closed set:

```
idle()                    stay silent — the default, and ~68.5% of stage-1 targets
respond({...})            reply, acknowledge, or fire a reminder
highlight({...})          mark a specific occurrence of a word
suggest_edit({...})       replace an exact span in the current text
delegate({...})           hand a slow task to a background model
web_search({...})         issue a query; results arrive as a later event
translate_commit({...})   append a stable unit of translation mid-sentence
```

Because the set is closed, every prediction can be validated and executed
without putting any interaction policy in the runtime.

## Run the app

Python 3.11+.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-tinker.txt
cp .env.example .env      # set TINKER_API_KEY
POLICY_PROMPT=g1 .venv/bin/python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. To serve the released weights locally instead of
through Tinker, download the model from Hugging Face and set:

```bash
export POLICY_MODE=local
export SMOL_LOCAL_MODEL=models/text-gpt-live-mlx-8bit
```

If no model is reachable the UI stays open and says **No model available** —
it never quietly substitutes a fallback.

## Reproduce the dataset

The authored source records are in [`data/g1_authored/`](data/g1_authored) —
42 JSON files across five demos. These are the *inputs*, not the training
cards. Authoring models wrote scenarios under a fixed schema; a deterministic
compiler turns each one into a 650 ms event stream and assigns every target.
No model judges anything, and the validators reject any authored text that
contains action syntax.

```bash
.venv/bin/python -m scripts.g1_full_build
```

This validates all five demos against their distribution gates, compiles them,
removes exact duplicate prompt/answer pairs globally, and requires zero prompt
overlap between train and dev. On a clean checkout it produces:

```
g1 full: 6688 rows from 6740 inputs (train=6013, dev=675)
Resolved 42 duplicate prompt groups; removed 47 exact duplicates; train/dev overlap=0
Tinker limit: removed 5 rows above 65536 target tokens; max published=62223
```

Those are the numbers in the write-up.

**This step is required.** Training runs in three stages, and Hugging Face
carries only the last two — `train.jsonl` (2,903 cards, stage 2) and
`delivery_repair_train.jsonl` (544, stage 3). The 6,013-card stage that
teaches the core interaction behavior is not published there; it is rebuilt
from the authored records above.

| stage | cards | source |
|---|---|---|
| 1 — core interaction behavior | 6,013 train / 675 dev | `scripts.g1_full_build` |
| 2 — translation, web search | 2,903 | Hugging Face |
| 3 — evaluation repairs | 544 | Hugging Face |

## Train

```bash
.venv/bin/python -m train.tinker_run --stage g1
```

Each stage resumes from the previous checkpoint at a lower learning rate
(2e-4, 5e-5, 1e-5). The full locked configuration — rank, seed, optimizer
settings, per-stage step counts — is in [`docs/training.md`](docs/training.md).

## Layout

```
app/          runtime, policy, stream format, search, background delegation
datagen/      authored-record validators and the deterministic compilers
train/        Tinker training loop and evaluation
scripts/      build, eval, probe, and local-serving entry points
tests/        distribution gates and contract tests
static/       the browser client that ticks every 650 ms
data/         authored source records
docs/         architecture, data spec, training recipe, limitations
```

`app/stream.py` is worth reading first. Training and serving compile prompts
through the same `compile_stream`, so a reproduction cannot silently diverge
from what the model was trained on.

## What is not in this repo

- **Weights** — on Hugging Face. `models/` is gitignored.
- **Compiled training cards** — stages 2 and 3 are on Hugging Face; stage 1
  rebuilds from `data/g1_authored` with the command above.
  `data/**/*.jsonl` is gitignored.
- **The V4–V6 lineage.** This project went through several earlier
  generations with an incompatible card format. `app/policy.py` still carries
  those grammars, but the datagen and training paths here are g1 only.

## Limitations

[`docs/limitations.md`](docs/limitations.md) documents what does not work,
including a recurring-reminder failure the shipped checkpoint does not solve
and a highlighting behavior coupled to an artifact of the generator. It is
worth reading before trusting any number here.

## License

MIT. See [LICENSE](LICENSE).
