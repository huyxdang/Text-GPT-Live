# Synthetic data limitations

## Demo 2: highlighting is coupled to an artificial word-end tick

**Status:** Confirmed from the generator and published datasets on 2026-08-03.

### What the data actually teaches

Demo 2 does not intentionally teach the model to wait for a period. Instead,
the compiler forces a synthetic tick to end at the exact final character of
every literal category match. The highlight action is supervised on that exact
tick, and the compiler rejects the example unless the matched word ends the
visible textbox snapshot.

This makes a typical positive look like:

```text
...today I used a wrench
<action>highlight({"occurrence":1,"quote":"wrench"})</action>
```

The original published Demo 2 data contains 165 highlight-positive cards:

- 165 snapshots end exactly on the target word;
- 0 snapshots end on a period;
- 0 snapshots end on a comma.

The authored source prose is less artificial: 150 of those target words are
followed by a space and 15 by a comma. The compiler inserts a forced observation
boundary before that following character. The g1-v2 replay slice preserves the
same pattern in all 38 of its highlight-positive cards.

### Why this can fail in the live application

Real observation ticks are independent of word boundaries. If the user types
quickly, the runtime may first observe `a lion sleeping` rather than a snapshot
ending at `a lion`. The training data does not positively supervise highlighting
a newly appeared target when additional text already follows it.

The neighbouring after-target cards are not missed-action recovery examples.
They are teacher-forced histories in which the correct highlight already
appears on the preceding exact-boundary tick. Therefore, when the live model
misses that artificial tick, its subsequent history is outside the trajectory
it was trained to follow.

The observed behavior of highlighting only after a period is consequently not
an intentional period rule in the labels. It is most likely a delayed-confidence
or recovery artifact caused by the real runtime skipping the exact word-end
snapshot.

### Category coverage gap

The authored production corpus includes categories such as `bird`, `insect`,
`color`, `tool`, and `plant`, but contains no general `animal` category record.
An instruction such as `Please highlight any animal you see` therefore requires
category-level generalization that the training corpus never directly teaches.

### Required data repair

- Decouple observation boundaries from target positions and typing speed.
- Mix unchanged ticks, 1-3 character increments, 4-7 character increments,
  8-15 character jumps, and occasional larger jumps.
- Supervise a highlight on the first observed snapshot containing a newly
  completed, not-yet-highlighted target, even when later text already follows
  the target.
- Add explicit recovery trajectories where a target crossed between ticks and
  no highlight action exists in history.
- Include snapshots that introduce multiple targets, with a defined policy for
  emitting the remaining highlights across subsequent decisions.
- Add the broad `animal` category and varied natural paraphrases, alongside
  specific subordinate categories such as birds and insects.
- Evaluate highlight recall separately across typing speeds, characters per
  observed tick, target position, punctuation, and number of newly visible
  targets.

### Evidence

- Trigger-boundary compilation: `datagen/g1_demo2.py`
- Authored category sources: `data/g1_authored/demo2/*.json`
- Original published cards: `data/train_g1-*.jsonl` and `data/dev_g1.jsonl`
- g1-v2 replay cards: `data/g1_v2/train.jsonl` and `data/g1_v2/dev.jsonl`

---

# Demo 5 limitation: recurring reminders

## Status

Demo 5 is not solved by the selected epoch-one checkpoint. The model reliably
waits before a reminder is due, but it does not reliably switch from
`idle()` to `respond(...)` when the same schedule crosses its firing boundary.
No repair checkpoint was promoted.

## Round one: the original training data

The first g1 dataset contained 6,688 published decision cards: 6,013 train and
675 dev. Demo 5 began with 36 authored reminder-bank entries that the generator
expanded into 130 schedules and 970 cards; global deduplication left 961 Demo 5
cards. The final split contained 862 Demo 5 train cards and 99 dev cards.

The 862 training cards were not missing positive reminder examples:

- 178 cards required a reminder to fire now.
- 388 cards were explicit reminder waits.
- 335 cards expected `respond(...)`; 527 expected `idle()`.
- The generator included recurring and one-shot schedules, typing-time and
  silent fires, pre-boundary waits, exact-boundary and post-boundary cases,
  already-fired and post-cancellation restraint, and initial, cleared, and
  unchanged empty-text ticks.
- Before/action/after neighbors taught “not yet,” “fire now,” and “already
  handled” as separate decisions. Personas, domains, registers, intervals, and
  filler traps were varied rather than cloned from one reminder template.

Qwen3.5-4B was then trained for one complete epoch: 376 optimizer steps with a
rank-32 LoRA. Training stopped after one epoch for budget; no epoch-two
checkpoint or evaluation exists.

## What round one revealed

On the 675-card dev set, epoch one produced valid, canonical action syntax on
every row. Strict accuracy was 512/675 (75.85%); the later meaning-aware score
was 595/675 (88.15%). Those aggregate numbers hid the central failure:

- reminder waits: 41/41 correct;
- ordinary silence: 25/25 correct;
- nominal fire-now cases: 0/20 correct.

A timing audit then found that six of the 20 nominal fires were one-shot
examples whose precise deadline was not observable from the serialized event
history. Those six should not have supported a hard clock-reasoning claim. The
clean result is therefore **0/14 on observable recurring fires**, not 0/20.

This was also not a formatting or local-serving failure. The model emitted
valid `idle()` actions, and the final local 4-bit build reproduced the hosted
Tinker checkpoint byte-for-byte on all 43 frozen pilot rows. The model was
making the wrong policy decision.

## Why DPO was not the right diagnosis

DPO would make sense if the model already recognized the correct fire and wait
answers but free generation preferred the safer `idle()` response. The probes
did not show that clean preference gap.

First, every failed fire item was presented as a balanced A/B choice twice,
with answer order reversed. The model scored 31/40 individual presentations,
but only 11/20 items were correct in both orders. It selected slot A 29 times
and slot B 11 times; every error occurred when the correct answer was in B.
That is strong position bias, not stable recognition of the preferred action.

Second, a causal probe reconstructed 14 observable recurring fires and 14
matched waits from the same schedules:

1. In the extraction/arithmetic stage, the model returned `idle()` on all 28
   prompts instead of the requested structured clock answer. This exposed
   action-format lock, so it did **not** provide a clean measurement of the
   model's arithmetic.
2. In the oracle-binding stage, the prompt supplied authoritative `due=true`
   or `due=false` state and the exact reminder message. The model still scored
   0/14 fires and 14/14 waits, returning `idle()` every time.

The model therefore did not reliably bind an explicit due state to the fire
action. Preference training at this point could amplify an order bias or make
the model generally trigger-happy without teaching the missing causal binding.
The precondition for DPO was not met: strong, order-invariant recognition while
free generation alone remained biased toward waiting.

## Round two: the targeted SFT repair

Because model-owned scheduling remained part of the research claim, a small
supervised repair was attempted instead of DPO. The repair corpus contained 80
train-only cards:

- 24 observable recurring fire decisions;
- 24 matched, same-cycle pre-boundary waits;
- 16 cross-demo action replays;
- 8 cross-demo ordinary-silence guards;
- 8 reminder-restraint guards.

The corpus covered 56 unique episodes, had zero exact prompt or episode overlap
with dev, and was packed into five balanced 16-card updates. The replay and
restraint rows were included so that learning to fire would not erase the
model's working silence, waiting, and Demos 1–4 behavior.

The first continuation resumed the saved epoch-one optimizer and ran five
steps at a conservative `5e-5` learning rate. Its focused 65-case result was:

- recurring fires: 0/14;
- matched waits: 14/14;
- ordinary silence: 25/25;
- reminder restraint: 8/8;
- collision action kind: 4/4;
- valid and canonical format: 65/65.

Since the candidate still never fired, the full 675-card evaluation was
skipped.

A cost-capped second continuation restored the original `2e-4` learning rate
and evaluated after each additional five-step rung. Fire/wait exact results,
each out of 14, were:

| Additional rung | Fire | Wait |
|---|---:|---:|
| 1 | 5 | 13 |
| 2 | 8 | 7 |
| 3 | 7 | 11 |
| 4 | 8 | 11 |

All 28 outputs remained valid at every rung, but none met the precommitted gate
of at least 13/14 fires **and** 13/14 waits. More willingness to fire arrived
only with unstable damage to waiting. That is a trade-off, not a repair.

## Why we know the repair did not work

The conclusion does not depend on exact wording or an LLM judge. Fire versus
wait is a deterministic action-and-timing decision, and all candidates failed
that focused gate. No candidate was allowed to hide behind aggregate accuracy,
majority-idle performance, or semantically similar prose. The full dev run was
not spent after a causal failure, no repair checkpoint was promoted, and epoch
one remains selected.

This result does **not** prove that Qwen3.5-4B can never learn recurring
reminders. It shows that this one-epoch model plus this 80-card continuation did
not learn the capability cleanly. Another repair needs a broader objective or
data redesign, an independent train-only checkpoint-selection set, one frozen
acceptance pass, and semantic regression approval before promotion. DPO should
only be reconsidered if a future checkpoint first demonstrates strong,
order-invariant fire/wait recognition while still failing in free generation.

## Evidence

- Original dataset: `artifacts/g1-full/manifest.json` and
  `artifacts/g1-full/coverage.json`
- Epoch-one evaluation: `data/tinker/dev-g1-epoch1_eval.json`
- Fire-recognition probe: `data/tinker/dev-g1-epoch1_fire_recognition.json`
- Causal probe: `data/tinker/dev-g1-epoch1_causal_probe.json`
- First repair gate: `data/tinker/dev-g1-repair-focused_eval.json`
- Repair ladder: `data/tinker/g1-repair-ladder/summary.json`
