# g1 training runbook

**Status:** implemented; the user-approved one-epoch round-1 Tinker run completed 2026-08-01.
**Companions:** [../synthetic_data_spec.md](data.md) (the data),
[../acceptance_criteria.md](acceptance-criteria.md) (the finish line).

## What g1 is

**Generation 1 of a fresh lineage: stock `Qwen3.5-4B` + one LoRA fine-tune on
the g1 dataset. One stage. No curriculum.**

The earlier from-scratch plan (this file's previous edition) chained the old
curriculum under fresh tags — `v4f` → `v41f` → a final adapt on the v6-era
data. That design is **dead, deliberately** (2026-07-31): the g1 card format is
incompatible with every old-stage dataset — header-free prompts, `delegate` and
`highlight` verbs, no `web_search`, human typing cadence, non-uniform gaps, the
conversational register. Training through the old stages would teach the model
things the final stage immediately unteaches. One coherent dataset → one stage.

Naming: `g1`, not `v7` — this is not round seven of the old line; the `v1`–`v6`
names stay historical.

## Base model — Qwen3.5-4B (decided 2026-07-30)

1. **Latency is the product.** A decision is due every 650 ms; the old 8B
   measured 1.0–1.6 s per decision. A ~4B dense model roughly halves that.
2. **The task is small.** One constrained single-line action per tick — mostly
   judgment plus exact span copying.
3. **Newer generation** — the parameter drop is not expected to cost quality
   outright; that assumption is under test.
4. **Chinese** (demo 3) is a Qwen strength.
5. **Local serving** on the Mac for filming, no per-call cost.

Fallback: `Qwen/Qwen3.5-9B` (also confirmed live). Rejected: 27B+ — those are
what `delegate` hands work *to*, not the per-tick reflex. The g1 trainer exposes
`--base-model`, while the legacy stages retain their historical constant.

## The run

```
.venv/bin/python -m scripts.g1_full_build
.venv/bin/python -m train.tinker_run --stage g1
```

The first command validates and compiles all five demos, removes exact duplicate
prompt/answer pairs globally, and requires zero prompt overlap between train and
dev. The second creates a fresh LoRA client from `--base-model`, loads only the
verified `artifacts/g1-full/manifest.json`, evaluates dev after every epoch, and
writes state under the **`g1:*` namespace only**. Missing or stale full-dataset
metadata is fatal; paid training never falls back to the Demo-1-only artifact.

The full recipe is the locked configuration below.

## Round-1 configuration — locked 2026-07-31

Everything on **our** side of the Tinker line. Tinker abstracts the GPUs, the
distributed step, and checkpoint storage; every value below is ours, and the
provenance block stamps the actual values used into every report, so no
forgotten flag can masquerade as the default recipe.

**Adapter creation** (passed once):

| Setting | Value | Note |
|---|---|---|
| `base_model` | `Qwen/Qwen3.5-4B` | ID confirmed against the live model list, 2026-07-31 |
| `rank` | 32 | the escalation lever, parked |
| `train_attn` / `train_mlp` / `train_unembed` | on / on / on | Tinker defaults, passed explicitly so the choice is visible |
| `seed` | 650 | NEW — adapter-init seed was never set in the old runs |

**Optimizer** (`AdamParams`, passed by us on every step):

| Setting | Value | Note |
|---|---|---|
| `learning_rate` | 2e-4, constant | from-base recipe; no schedule, frozen-on-purpose |
| `beta1` / `beta2` / `eps` | 0.9 / 0.95 / 1e-12 | Tinker defaults, accepted deliberately |
| `weight_decay` / `grad_clip_norm` | 0 / off | round-2 levers, only if the curves ask |

**Training loop** (entirely our code):

| Setting | Value | Note |
|---|---|---|
| epochs | 3, per-epoch dev eval, best-epoch selection | 376 steps/epoch at 6,013 train cards |
| batch size | 16 | |
| shuffle | `Random(650 + 20 + epoch)` | deterministic per epoch |
| loss | cross-entropy, per-token weights | prompt tokens 0; completion tokens `1/len(completion)`; end-token graded |
| class weights | **all 1.0** | CHANGED — multipliers die; the count-engineered mix does the balancing |

**Rendering** (where train-equals-serve lives): g1 system prompt + card as the
user message, thinking disabled; header-free flat-grammar card format; the
completion is one `<action>` line + `<|im_end|>`.

**Eval-time sampling:** greedy (temperature 0, top-k/top-p off), stop on
newline, max 192 tokens, concurrency 8, timeout 600 s.

**Bookkeeping:** `g1:*` state namespace only; provenance on every report;
Wilson intervals on headline rates; `respond_message_grounding` disabled;
**record wall-clock per stage** — the old rounds never did, and it was missed.

**Epoch selection:** pass/fail of all g1 hard gates ranks first. Before aggregate
strict accuracy, the selector compares the weakest of should-fire recall,
reminder-wait accuracy, ordinary-silence accuracy, and clause-boundary accuracy.
This prevents a deceptively accurate always-idle epoch from being selected.

### The loss, unpacked

One long token sequence per card: system prompt + event stream + answer.
Cross-entropy grades every token, but the weights decide what counts:

- **Prompt tokens → weight 0.** The card is reading material; the exam starts
  at `<action>`. Otherwise most of the signal is "predict the user's diary."
- **Answer tokens → `1/len` each, so every card totals weight 1.0.** One card,
  one vote: a 5-token `idle()` and a 40-token Chinese clause are equal
  decisions.
- **The end-token is graded.** Saying one line *and then stopping* is part of
  the answer — omit it and outputs dribble.

### Why per-card weighting survived the audit

Challenged 2026-07-31; kept, for three reasons and one caveat:

1. **Asymmetric failure modes.** The alternative (per-token) hands responds
   ~8× the gradient of idle purely for being wordier — a verbosity bias
   trained into a model whose flagship demo is shutting up. Our scheme's
   downside (diluted per-token signal inside long answers) is a nuisance; the
   alternative's is a project-killer.
2. **Unit coherence.** The mix table was engineered in card units ("~42%
   actions"). The loss must count in the same currency or the table lies —
   under per-token loss the effective mix becomes ~80% respond-signal.
3. **Coupling with the coverage guards.** Dropping multipliers is only safe
   because the guards enforce real counts. Flat weights and the guards are
   one decision in two files.

**Caveat + named diagnostic:** long-answer content gets diluted per-token
supervision. If round-1 translations come out *fluent-but-wrong* while all
else passes, data is suspect #1 and diluted content signal is **suspect #2**
— the round-2 lever is a modest content-token boost, taken only then.

### Escalation tells (read the round-1 curves, then touch knobs)

| Symptom | Meaning | Lever |
|---|---|---|
| Train loss low; dev poor on specific families | data gaps / diversity | more situations (the usual suspect) |
| Train loss plateaus high while data scales | adapter capacity | raise `rank` 32 → 64 |
| Translations fluent-but-wrong, all else passing | diluted content signal | content-token boost (suspect #2 only) |
| Overfitting (dev degrades across epochs) | — | earlier epoch (selection already handles), then `weight_decay` |

Everything not listed as a lever is **frozen on purpose**: the base model and
the whole dataset already changed this round, and each additional knob touched
blurs attribution of whatever round 1 shows.

## Evals

### Round-1 measured result

The actual budget was reduced to **one epoch** before round 1 finished. All 376
epoch-1 steps completed; one epoch-2 batch slipped through during shutdown, but
no epoch-2 checkpoint or evaluation exists. The saved epoch-1 report is
`data/tinker/dev-g1-epoch1_eval.json`.

Strict exact scoring measured 512/675 rows (75.8519%), with 100% format and
canonical validity. Reminder-wait and ordinary-silence accuracy were both 100%,
but nominal should-fire recall was 0/20 and exact clause-boundary accuracy was
25/77. A later timing audit found that six of those 20 fires were vague one-shot
reminders with no due time observable in the stream; the defensible explicit
recurring subset is 0/14.
The exact payload metric is retained as a reproducible diagnostic; it is not an
adequate semantic acceptance metric for open-ended replies or translations.

The hybrid evaluator keeps action kind, time, response target, edit/highlight
source spans, syntax, and canonical bytes deterministic. Only a correctly
routed non-exact response message, delegate task, or edit replacement reaches
the semantic judge:

```bash
# Show how many rows require semantic judgment without inference.
.venv/bin/python -m scripts.g1_semantic_eval --dry-run

# Private on-device provisional judge. Its verdicts remain uncalibrated.
.venv-mlx/bin/python -m scripts.g1_semantic_eval --provider local-mlx

# Higher-capability remote judge; requires explicit approval to export the
# selected private dev cases to the configured OpenAI endpoint.
.venv/bin/python -m scripts.g1_semantic_eval --provider openai
```

Judgments are checkpointed per row in
`data/tinker/dev-g1-epoch1_semantic_judgments.json`; the completed hybrid report
is written to `data/tinker/dev-g1-epoch1_hybrid_eval.json`. Exact-match and
hybrid metrics coexist in that report. A wrong action such as `idle()` on a
fire-now row is never sent to the judge and cannot be rescued by wording.

The fire-now generation/recognition diagnostic runs against the selected
checkpoint and presents every failed fire item twice with reversed A/B order:

```bash
.venv/bin/python -m scripts.g1_fire_recognition_probe
```

Epoch 1 scored 31/40 presentations (77.5%), but only 11/20 items were correct
under both orders (55%). All nine mistakes chose A when the correct action was
in B. That is not clean evidence for DPO.

Before authorizing more training, the causal probe reconstructs 14 matched
pre-boundary waits for the 14 observable recurring fires and runs at most three
28-call stages:

```bash
# Audit labels and case construction locally; no sampler calls.
.venv/bin/python -m scripts.g1_causal_probe --dry-run

# Run staged extraction/math, diagnostic due-state binding, and (only when the
# first two pass) a one-sentence prompt intervention.
.venv/bin/python -m scripts.g1_causal_probe
```

The epoch-1 run stopped after 56 calls. In the extraction stage, all 28 outputs
were `<action>idle()</action>` instead of the requested JSON, so that stage
shows strong action-format lock and does not provide a clean arithmetic score.
In the oracle-binding stage, every output was again idle: all 14 waits passed,
but all 14 fires failed even when an authoritative `due=true` record supplied
the exact message and target rule. The minimal-prompt stage was skipped, saving
28 calls. This makes prompt-only repair and DPO poor next bets. The remaining
honest choice is product-level: deterministic runtime scheduling for zero
training cost, or one tightly gated micro-SFT continuation if model-owned clock
reasoning is still a required research claim.

Any future repair ladder has two deliberately separate evaluation layers:

1. **Rung selection:** 14 recurring fire/wait pairs from unused train-split
   schedules, with zero episode or prompt overlap against the repair corpus and
   frozen dev split. This validation set may be checked after each rung.
2. **Acceptance:** after a rung is selected, run the frozen 65-case focused
   suite once, then the 675-case full dev suite once if focused passes. Passing
   those deterministic gates records diagnostic success only. The checkpoint
   remains ineligible for promotion until a separate semantic regression review
   also passes.

This separation prevents repeated checkpoint selection against the frozen
acceptance cases and prevents exact-match gates from silently standing in for
semantic quality.

- **Dev** (`dev_g1`, same-batch holdout): per-epoch curves and epoch selection.
  Internal only.
- **After the checkpoint freezes:** fresh acceptance fixtures (10/demo) and the
  report eval (100–200/demo) per acceptance_criteria.md's held-out policy —
  confirmed 2026-07-31 as THE evaluation plan (tier 2).
- **Old frozen suites** (v4/v41/v6): **retired from g1's story entirely**
  (corrected 2026-07-31 — earlier "weak regression" was too generous). The g1
  model speaks a different dialect: it cannot emit `tool(web_search,…)`, the
  header is gone, the grammar is flat — old episodes fail on grammar, not
  behavior. Zero signal. They remain solely to score old-lineage models.
  g1 regression lives in dev + the fresh sets.
- Comparisons vs the prompted stock baseline run through
  `scripts/rollout_eval.py`-style harnessing with `compare_reports.py`
  significance testing, per the criteria doc.

## Obsolete wiring removed

The `--stage v7` CLI path is gone. Historical v4-v6 functions remain available
for reproducibility, but there is exactly one live path to a fresh g1 checkpoint.

## Prerequisites

The full list lives in synthetic_data_spec.md. The five authored corpora, live
g1 app contract, full compiler, manifest integrity checks, and trainer dry-run
tests are complete. The local full build currently contains 6,688 trainable
cards (6,013 train / 675 dev). The compiler removed five late Demo 1 snapshots
above Tinker's 65,536-target-token limit, records them in the manifest, and the
trainer independently enforces the same limit before remote client creation.
Chinese references are machine-reviewed with a focused semantic correction
pass, not human-certified.
