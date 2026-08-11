# Demo 3 sample source — NOT training data

Four hand-written records that exist to prove the Demo 3 machinery compiles,
not to teach the model anything. The real authored corpus goes in
`data/g1_authored/demo3/` and is written by the authoring fleet.

Everything here is deliberately tiny, so the scale-dependent distribution gates
(author share, persona breadth, bucket coverage, skeleton spread) cannot run.
Builds against this root must pass `--allow-small-corpus`, and the manifest
records `distribution_gates_enforced: false` when they do.

Validate:

    .venv/bin/python -m scripts.g1_demo3_report --root data/g1_authored_demo3_sample

Build (writes only to a scratch directory):

    .venv/bin/python -m scripts.g1_demo3_build \
        --authored-root data/g1_authored_demo3_sample \
        --allow-small-corpus \
        --episodes 4 --clauses 12 --cards 60 \
        --empty-per-kind 1 --min-backtrack 1 --min-prequoted 1 --min-partial 1 \
        --train-shards 2 \
        --train-base /tmp/g1-demo3-sample/train.jsonl \
        --dev /tmp/g1-demo3-sample/dev.jsonl \
        --artifact-dir /tmp/g1-demo3-sample/artifacts

## What each record exercises

| Record | Opening / closing skeleton | Situation classes |
|---|---|---|
| `demo3-sample-01` | `idle-idle` / `hold-clear` | all three traps in one episode: half-finished clause, backtrack, pre-quoted Chinese |
| `demo3-sample-02` | `idle-long-idle` / `clear-hold` | half-finished clause spanning two script steps; clause commits mid-step |
| `demo3-sample-12` | `immediate` / `hold` | no empty prelude; backtrack that rewrites an unfinished clause |
| `demo3-sample-13` | `idle` / `hold` | pre-quoted Chinese mid-passage; clean commits either side |

Chinese references carry **machine (LLM) review only**. No human on this
project reads Chinese, and the manifest says so rather than implying otherwise.

## The clause marker

`‖` (U+2016) marks the position immediately *after* a clause's last character.
Each clause is stated twice — positionally by the marker, textually by
`clauses[i].english` — and the compiler refuses to build when the two disagree.
