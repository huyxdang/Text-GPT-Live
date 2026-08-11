# Training runbook

**Status:** the released checkpoint was trained in three sequential one-epoch
SFT stages on Tinker and exported as a merged 8-bit MLX model.

**Companions:** [data.md](data.md) defines the corpus and
[architecture.md](architecture.md) defines the released serving contract.

## Locked recipe

| Stage | Rows | Epochs | Batch | Learning rate | Seed | Purpose |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 6,013 | 1 | 16 | `2e-4` | 650 | Core silence, response, highlighting, editing, delegation, and initial task behavior |
| 2 | 2,903 | 1 | 16 | `5e-5` | 702 | Append-only translation commits, web search, and replay of retained behavior |
| 3 | 544 | 1 | 16 | `1e-5` | 704 | Search-result delivery after foreground dialogue while delegation continues |

All stages use `Qwen/Qwen3.5-4B`, rank-32 LoRA, a 65,536 target-token
ceiling, deterministic shuffling, and Adam with the optimizer settings in
`train/tinker_run.py`. Each continuation resumes from the previous stage's
optimizer-inclusive Tinker state.

The budget was deliberately capped at one epoch per stage. The training scripts
save fingerprints, loss records, evaluation reports, checkpoint paths, and
promotion decisions under the ignored `artifacts/` and `data/tinker/`
directories. The three accepted stages cost approximately $50 in total.

## Prerequisites

Training requires Python 3.11+, a Tinker account, and `TINKER_API_KEY` in the
environment or a local `.env` file:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-tinker.txt
cp .env.example .env
```

Do not commit `.env` or `data/tinker/run_state.json`; both can contain private
provider state.

## Stage 1: core interaction behavior

Build the deterministic corpus and train one epoch:

```bash
.venv/bin/python -m scripts.g1_full_build
.venv/bin/python -m train.tinker_run --stage g1 --epochs 1
```

The build must report 6,688 total rows: 6,013 train and 675 dev, with zero
train/dev prompt overlap. The trainer records the accepted state and sampler
paths in `data/tinker/run_state.json`.

## Stage 2: translation and search

The stage-2 builder converts Demo 3 to `translate_commit`, creates asynchronous
search episodes, and replays retained Demo 1, 2, and 4 behavior:

```bash
.venv/bin/python -m scripts.g1_v2_build
.venv/bin/python -m scripts.g1_v2_train
```

The trainer evaluates the candidate on the new dev set and the original
Demo 1-4 holdout. It refuses promotion when the new gates fail or retained
behavior falls below its per-demo floor.

## Stage 3: delivery repair

The final repair teaches the model to answer foreground dialogue while jobs are
running, then surface the completed search result exactly once:

```bash
.venv/bin/python -m scripts.g1_v2_delivery_repair_build --index-mode fixed
.venv/bin/python -m scripts.g1_v2_delivery_repair_train
```

The builder reads the stage-2 checkpoint paths from
`data/tinker/run_state.json`; it does not embed another operator's private
Tinker path. For a deliberate external checkpoint, pass both `--source-state`
and `--source-sampler`.

The release uses the fixed-index repair selected by the recorded evaluation.
The trainer gates foreground responses, search delivery, both delegation/result
orders, delivered-idle behavior, the complete stage-2 dev set, and retained
Demo 1-4 behavior.

## Export the local checkpoint

The selected Tinker adapter can be merged into an MLX copy of the base model
with:

```bash
.venv/bin/python -m scripts.g1_merge_tinker_adapter \
  --base models/Qwen3.5-4B-mlx \
  --adapter models/text-gpt-live-adapter \
  --output models/text-gpt-live-mlx-8bit \
  --bits 8 \
  --group-size 64
```

The exporter materializes the adapted output head, quantizes the merged model,
and writes `MERGE_PROVENANCE.json`. The released 8-bit checkpoint was selected
because the 4-bit candidate changed a required search-result target index.

## Evaluation boundary

Exact action structure, grounded spans, target indices, and one-shot job
lifecycle behavior are deterministic checks. Open-ended response text is
evaluated semantically rather than by exact wording. Promotion remains gated by
per-capability results so a majority-idle policy cannot pass on aggregate
accuracy alone.

The release is a research artifact. It does not reliably perform the reminder
demo, long-stream highlighting remains brittle, translation is limited, and
the measured local decision latency remains above the 650 ms target.
