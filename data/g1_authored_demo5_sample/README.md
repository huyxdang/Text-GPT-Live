# Demo 5 sample source — NOT training data

A hand-written bank that exists to prove the Demo 5 machinery compiles, not to
teach the model anything. The real authored corpus goes in
`data/g1_authored/demo5/` and is written by the authoring fleet, per
`docs/data.md`'s "Demo 5 — Reminders (time awareness)" section.

Everything here is deliberately tiny (14 bank entries across 3 authors), so
the scale-dependent distribution gates (author share, persona breadth, bucket
coverage) cannot run. Builds against this root must pass
`--allow-small-corpus`, and the manifest records
`distribution_gates_enforced: false` when they do.

Validate:

    .venv/bin/python -m scripts.g1_demo5_report --root data/g1_authored_demo5_sample

Build (writes only to a scratch directory):

    .venv/bin/python -m scripts.g1_demo5_build \
        --authored-root data/g1_authored_demo5_sample \
        --allow-small-corpus \
        --schedules 24 --fires 35 --cards 190 \
        --empty-per-kind 10 \
        --min-post-cancel 5 --min-once-no-repeat 10 --min-bait 4 \
        --min-address 5 --min-silence-idle 12 \
        --train-shards 2 \
        --train-base /tmp/g1-demo5-sample/train.jsonl \
        --dev /tmp/g1-demo5-sample/dev.jsonl \
        --artifact-dir /tmp/g1-demo5-sample/artifacts

Exact `--fires`/`--cards` counts are bank- and `--schedules`-dependent (the
generator cross-products banks with sampled arithmetic, so the fire count is
an emergent property, not something the fixture author sets). If a build
fails with an exact-target mismatch, read the printed actual count and pass
it back in — that is the intended workflow, mirroring Demo 3's `--clauses`.

## What the bank exercises

| Bank entry | Kind | Trap / role it feeds |
|---|---|---|
| `demo5-sample-req-every-01/02/03` | request, `every` | the "every N seconds" schedule kind; N is generator-sampled per schedule from `INTERVAL_CHOICES_S` |
| `demo5-sample-req-once-01/02` | request, `once` | the "remind me once" schedule kind; must never repeat (`once-no-repeat` trap) |
| `demo5-sample-cancel-01/02` | cancellation | "stop reminding me" → nothing fires after (`post-cancel-idle` trap) |
| `demo5-sample-filler-01..04` | filler, `trap: none` | ordinary typed narration filling the gaps between fires (`fire-before`/`fire-after`/`fire-typing` neighbours) |
| `demo5-sample-bait-01/02` | filler, `trap: bait` | a quoted question while the reminder schedule is live → idle, not a reply (`bait-idle`) |
| `demo5-sample-address-01` | filler, `trap: address` | a direct address while the schedule is live → answered, schedule keeps ticking (`address-positive`) |

## Boundary coverage

The generator (`datagen.g1_demo5.plan_demo5_schedule`) hashes each generated
schedule id to an alignment bucket (`tight` / `medium` / `wide` overshoot past
the due instant) independently of the bank content, so building enough
schedules from this same small bank already exercises ticks landing just
before, essentially at, and well after the due moment — see
`tests/test_g1_demo5.py::Demo5TimingTests` for a direct assertion that all
three alignments, both schedule kinds, and a fire that lands during genuine
silence (`fire-silent`, empty textbox) all appear.

## Independent timing verification

Every fire/idle card produced from this bank is re-checked by
`datagen.g1_demo5.verify_fire_timing`, which recomputes idle-vs-fire from the
rendered prompt's own `time="t+Nms"` timestamps — not from the compiler's
internal state. `scripts/g1_demo5_build.py` fails the build if any gold
action disagrees with that independent recomputation.
