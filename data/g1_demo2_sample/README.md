# Demo 2 machinery fixture — NOT training data

Twelve hand-written Demo 2 situations used to exercise and prove the Demo 2
machinery (`datagen/g1_authored_demo2.py`, `datagen/g1_demo2.py`,
`scripts/g1_demo2_build.py`) end to end before the authoring fleet runs.

This tree is **not** part of any dataset. Real authored tranches land in
`data/g1_authored/demo2/`, are written by the fleet, and are built with the
production `Demo2Targets` defaults. These twelve records exist only so the build
CLI has something to compile, and they deliberately cover every situation class:

| class | where |
|---|---|
| instruction + one-shot ack | every record |
| planted typo/grammar error (`suggest_edit`) | every `corrections` record |
| literal category match (`highlight`) | every `highlights` record |
| trap: clean text under an active instruction | `trap: "clean_text"` passages |
| trap: typist self-corrects before we fire | `kind: "repair"` segments |
| trap: text that merely mentions errors | `kind: "aside"` segments |
| trap: category word used non-literally | `marks` with `"literal": false` |
| silence (initial / unchanged / cleared) | compiled from the episode skeleton |

Run the build against it with:

```
.venv/bin/python -m scripts.g1_demo2_build \
  --authored-root data/g1_demo2_sample \
  --train-base /tmp/demo2/train.jsonl --dev /tmp/demo2/dev.jsonl \
  --artifact-dir /tmp/demo2/artifacts \
  --errors <N> --matches <M> --episodes 12 --cards <K> \
  --empty-per-kind 2 --hard-idle-cards <H>
```
