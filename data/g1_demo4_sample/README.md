# Demo 4 sample source — NOT training data

Six hand-written request phrasings and six progress-question/reply pairs
(three `check`, three `nudge`, split across two "authors") that exist to prove
the Demo 4 machinery compiles, not to teach the model anything. The real
authored banks go in `data/g1_authored/demo4/` and are written by the
authoring fleet.

Everything here is deliberately tiny, so the scale-dependent distribution
gates (author share, persona breadth, bucket coverage, opener spread) cannot
run. Builds against this root must pass `--allow-small-corpus`, and the
manifest records `distribution_gates_enforced: false` when they do.

Validate:

    .venv/bin/python -m scripts.g1_demo4_report --root data/g1_demo4_sample

Build (writes only to a scratch directory; these targets are the exact
reachable numbers for this fixture at 15 expanded episodes — see
`tests/test_g1_demo4_build.py`):

    .venv/bin/python -m scripts.g1_demo4_build \
        --authored-root data/g1_demo4_sample \
        --allow-small-corpus \
        --requests 6 --progress-pairs 6 --episodes 15 --cards 128 \
        --empty-per-kind 10 --min-check 5 --min-nudge 5 \
        --min-narration 5 --min-failure-check 3 \
        --train-shards 2 \
        --train-base /tmp/g1-demo4-sample/train.jsonl \
        --dev /tmp/g1-demo4-sample/dev.jsonl \
        --artifact-dir /tmp/g1-demo4-sample/artifacts

## How episodes are generated from these banks

`expand_demo4_episodes` cycles the 6 requests round-robin; each full cycle
bumps a variant counter, so 15 expanded episodes reuse requests 1-6 across
variants 0-2 with a different progress-window content kind, job outcome, and
episode skeleton each time (all hashed off the episode id — no RNG). At 15
episodes: content kind cycles check/nudge/narration roughly evenly (5 each),
and every 5th episode (indices 0, 5, 10) fails, giving 3 failed jobs.

## What the fixture exercises (the five required traps)

| Trap | Bank item(s) | Compiled role |
|---|---|---|
| 1. Nudge while pending -> `idle()`, never re-fires `delegate` | `demo4-sample-nudge-01/02/03` | `nudge-idle` |
| 2. "is it done yet?" while pending -> honest not-yet `respond` | `demo4-sample-check-01/02/03` | `check-positive` (with `check-before`/`check-after` neighbours) |
| 3. Completion arriving unasked -> `idle()` (the window is the announcement) | structural — every tool tick is graded `idle()` by construction, since `respond`'s `for` must target the current event and a tool event can never be that target | `accepted-idle`, `completed-idle`, `failed-idle` |
| 4. Ordinary narration during the job -> silence | generator-owned filler (see `NARRATION_FILLER` in `datagen/g1_demo4.py`) | `narration-idle` |
| 5. A failed job never claims the visual exists | reuses a `check` question as the post-failure follow-up; the reply is drawn from `FAILURE_REPLY_TEMPLATES`, which never asserts the deliverable exists | `failed-idle` + `failure-check-positive` |

## The two authored banks are words only

Authors write phrasings, not episodes: `requests[].text` / `.task` and
`progress_pairs[].question` / `.reply`. Everything else — pendency length (2-8
ticks success, 5-10 ticks failure), the fire tick, acknowledgement wording,
narration filler, episode opening/closing skeleton, and which bank items land
in which episode — is generated deterministically by `datagen/g1_demo4.py`
from a SHA-256 hash of the episode id. Same banks in, byte-identical cards
out, forever.
