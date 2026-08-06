# Acceptance Criteria — the five demos

**Status:** draft for sign-off
**Recorded:** 2026-07-30
**Supersedes:** the archived `acceptance_tests.md` / `musts_finish_before_move_on.md`
(text-only era, `archive/text-only`)

The project is done when five filmed demos pass the criteria below. Nothing else
is in scope. Each demo maps to a capability from the Thinking Machines interaction
model page. Capability names in column one are TMS's as supplied by the operator
(not independently verified against the page); everything else here is ours.

## How to read this document

Criteria are **pre-registered**: written here before a run, judged after. A result
measured without its criteria written down first does not count — it is a
demonstration, not evidence.

If a trial's outcome is used to shape the next round of training data, that trial
is **retired**: it is recorded in `findings.md` as consumed, and a fresh trial
replaces it for the next acceptance run. This is the rule that keeps the demos
from becoming the training set.

Three evidence layers, each answering a different question:

| Layer | Question it answers | Standing |
|---|---|---|
| Old frozen suites (v4/v41/v6, regenerated sha-identical) | Did the *old lineage* break? | Score old models only — g1 cannot emit their grammar (corrected 2026-07-31); g1 regression lives in dev + the fresh sets |
| Fresh held-out slices, authored after the checkpoint freezes by someone who did not train it | Does it generalize? | The real number |
| Live human sessions (the operator) | Is it actually good? | Decides "done" |

A demo passes when the fresh evals and the operator both agree; dev informs,
the old frozen suites never touch a g1 verdict, and the operator alone judges
the live layer.

**Traces first, interface second** (carried forward verbatim from the archived
suite): the pop-up windows, highlights, and panes are the demo's costume — in
reality everything runs invisibly in the background. Every claim, score, and pass
verdict is read from the trace; the UI exists only so a film can show what the
trace proves.

## The slate

| TMS capability | Text-TMS capability | Demo | Status |
|---|---|---|---|
| Seamless dialog management | Silence | **1** — "hey, how are you?" → it answers. "I'm about to tell a story, are you ready?" → "Yes!". Then the story flows, full of pauses and quoted questions — and it stays silent until spoken to again. | Needs address-respond data + filming |
| Verbal interjection | Interjection | **2** — (2.1) Under "fix my grammar as I go", planted typo/grammar errors get fixes mid-typing. (2.2) Under "highlight any animal word", matching words light up in neon as they appear. | 2.1 trained; 2.2 needs a sixth verb + data |
| Simultaneous speech | Simultaneous Speech | **3** — Type English continuously. Chinese appears clause by clause while the English is still arriving. | **Not built** — needs data + a display pane |
| Simultaneous tool calls, search, and generative UI | Concurrent tool-calls | **4** — "Can you build a dashboard about X?" → `delegate` fires and the visual generates in a pop-up window while you keep chatting; the model answers, reports progress honestly, and the window fills mid-conversation. | Needs data + the `delegate` rename + presentation |
| Time awareness | Time-awareness | **5** — "Can you remind me every 5 seconds to drink water?", then type something unrelated. Reminders keep firing, unprompted. | **Not built** — needs data only |

Five capabilities, five demos, one each. Every demo needs new data this round;
demos 3, 4, and 5 are entirely new trained behaviors.

Column two is the name we use in this project — the text-medium analogue of the
TMS capability. Where the medium changes the actuator but not the decision (an
interjection is visual here, verbal there), the name is kept so the mapping stays
legible when the speech era resumes.

**Lineage.** The archived suite's three demos are not lost: demo 1 descends from
its contextual restraint demo, demo 2 from its visual interjection demo, and demo
4 keeps its delegated generative-UI machinery under the new `delegate` name.
Demos 3 and 5 are new.

## The action vocabulary — five verbs, uniform `verb({json})` shape

| Verb | Meaning | Demos |
|---|---|---|
| `idle()` | do nothing, deliberately | all |
| `respond({"for": N, "message": …})` | say something, aimed at event N | 1, 3, 4, 5 |
| `suggest_edit({"quote": …, "replacement": …})` | replace this exact text with this exact text; quote must match exactly once (widened until unique) | 2.1 |
| `highlight({"occurrence": N, "quote": …})` | mark the Nth occurrence of a word matching a standing category | 2.2 |
| `delegate({"task": …})` | hand a described task to the bigger model (async job); renamed from `generate_ui`, executor unchanged | 4 |

**`web_search` is retired, not deleted.** Nothing demos it this round, and this
round's rule is that everything trained earns screen time. The runtime machinery
(DDGS provider, job plumbing, grounded-respond validation) stays untouched in the
app; it is the first candidate to return when a future demo wants it. Consequence
for scoring: g1 contains **no** search-grounded responses, so the
`respond_message_grounding` metric must be disabled for g1 reports (see open
questions).

## The conversational register — acknowledge, then act or wait

Cross-demo requirement. The model is a conversation partner, not a silent daemon:

- **Direct address gets an answer.** "hey, how are you?" → a reply. "are you
  ready?" → "Yes!". Same verb as everything else (`respond` aimed at the
  addressing tick).
- **Standing instructions get a one-shot acknowledgment.** "highlight any animal
  word" → "Got it — I'll highlight every animal you mention." "Remind me every 5
  seconds" → "Got it — every 5 seconds!" Then the model acts or waits.
- **Then restraint.** After the acknowledgment, narration, thinking pauses, and
  quoted questions get silence, exactly as before. Silence is only legible as a
  *choice* because the viewer has just seen the model speak.
- Acknowledgment wording must **vary** in training data — no single canned phrase.

## Presentation — the utterance bubble

The right side of the screen shows **what the model is saying right now**, not a
chat log. One bubble: the latest utterance appears, lingers roughly 2–4 seconds
(scaled to reading time — tune at first filming), and fades; a new utterance
replaces it. An unbroken respond chain (a story, a translation run) **accumulates
into the same bubble** like one spoken turn, then fades after the chain ends.
Words only — never the raw `<action>` tag; the tags live in the action log and the
trace. History is never rendered in the demo view (traces first, interface
second). Demo 3's aligned translation pane is the one exception: its output is an
artifact being built, so it persists. When the speech era resumes, this bubble is
what becomes audio.

## Cross-cutting requirement — continuous ticks and decision latency

The tick contract is app-wide, not specific to Demo 5. While any demo surface is
open, foregrounded, running, and unpaused, the browser submits the latest textbox
every 650 ms — including initial silence, a cleared box, and unchanged empty text.
Pausing stops ticks; an empty textbox does not. The browser owns this clock
intentionally for the current prototype; a server-owned loop is deferred.

The browser permits one decision in flight and **skips** an interval rather than
overlap requests or queue stale snapshots. That is necessary backpressure, but it
does not make the interval successful: every decision over 650 ms and every
resulting missed tick is a latency failure. A demo that stutters is failed,
however good the eventual decisions are. The last measured hosted latency was
1.0–1.6 s per decision, i.e. worse than the tick itself.

Serving-path instrumentation for this exists on this branch: `/health` reports
p50/p95/max decision latency and the ESC config strip shows a live readout
([app/runtime.py](app/runtime.py) `latency_summary`,
[static/app.js](static/app.js) `renderDecisionLatency`).

- Record p50 and p95 decision latency for **every** filmed take, in `findings.md`.
- No take is publishable if decisions exceed 650 ms or ticks are dropped during it.
- Demo 5's ± 1 tick criterion **assumes** decisions complete inside a tick. If
  measured p95 exceeds one tick, either the latency is fixed first or that
  criterion is renegotiated before the run — not after seeing the result.

## Demo 1 — Dialog: answer when addressed, silence while I think

**Capability:** Seamless dialog management.

**Scenario (one take, three movements).**

1. *Contact:* the operator types "hey, how are you?" — the model answers in the
   bubble ("I'm good! How about yourself?"). "I'm about to tell a story, are you
   ready?" — "Yes!"
2. *The story:* the operator narrates — mid-sentence pauses, self-corrections,
   quoted questions ("she asked, *what made you stay all those years?*"), the
   words "search" and "correct" used descriptively. The model is silent
   throughout, the action log ticking `idle` beside the text.
3. *Release:* the operator addresses it again ("still there?") — it answers.

Silence framed by speech. A dead screen is indistinguishable from a broken model;
a model that just spoke and now *chooses* quiet is unmistakable.

**Both failure modes are graded — this is the point:**

| Failure | Example | Verdict |
|---|---|---|
| Blurting | any action during narration or a thinking pause | fail |
| Ghosting | ignoring "are you ready?" | fail |

The hard pairs are the heart of the fixtures: *"are you ready?"* (addressed →
respond) vs *"he asked me if I was ready"* (narration → idle). Same words,
opposite answers, context decides.

**Pass criteria**

- Every direct address gets exactly one reply, within 2 ticks of the address
  completing.
- **0** actions during narration, thinking pauses (short pauses under ~2 s and
  long ones alike), quoted questions, and descriptive uses of trigger-ish words.
- The trace shows an explicit `idle()` decision on every silent tick — silence is
  a decision, not an absence.
- Repeatable: passes on ≥ 8 of 10 fresh held-out dialog fixtures, each containing
  both address moments and traps.

**Decided: no standing *task* instruction.** Nothing is asked of the model beyond
conversation — no "fix my grammar", no job. (Evidence this is still a hard test:
the frozen restraint suite compiles with `instruction=""` and baselines fail it —
GPT-4o-mini few-shot scored 3/6, and an audited run caught a model googling a
question quoted inside a diary. Left alone with prose, models reach for tools.)
The instruction-active version of restraint is covered by demo 2.

**What changed from the earlier draft:** pure silence was upgraded to dialog —
the address-and-answer movements are new trained behavior (direct-address
`respond`s did not exist in the old data), so demo 1 now needs new data, not
zero.

## Demo 2 — Interjection (2.1 corrections + 2.2 highlights)

**Capability:** Verbal interjection. In the text medium the actuator is visual —
the model marks up the user's own words mid-typing. The policy decision (interject
now, before the user finishes) is identical to the verbal case; only the output
channel differs.

### 2.1 — Corrections: typo + grammar only

**Decided: the error families are typos and grammar slips, nothing else.** The old
data's other families (literal word swaps, category swaps, derive-the-fix rules) are
dropped — they demoed rule-following, not interjection. Within the two kept
families, the *sub-type distribution must vary*:

- **Typos:** swapped letters, missing letters, doubled letters, wrong-hand hits,
  spacing errors, missing apostrophes.
- **Grammar:** tense, subject–verb agreement, articles, prepositions, wrong word
  form ("tried make"), plural/singular.

**Scenario.** Under a typed standing instruction ("fix my grammar as I go"), the
operator types prose containing planted errors. Each fix surfaces while they are
still typing — never rewriting, never waiting for submission.

**Pass criteria**

- Every planted error gets an exact, correctly-targeted proposal within 4 user
  ticks of becoming fully visible.
- **0** false suggestions on clean text; **0** proposals beyond their quoted span;
  **0** duplicates; no instruction text leaks into a proposal.
- Repeatable: passes on ≥ 8 of 10 fresh held-out passages.

### 2.2 — Highlights

**Scenario.** The operator types a standing instruction like "highlight any animal
word you see" (categories and phrasings vary), then ordinary prose. Every matching
word lights up in neon as it appears, while typing continues.

**Action.** `highlight({"occurrence":N,"quote":"…"})` — quote a single word,
address repeats by index (the quote is the visible mark, so unlike
suggest_edit it cannot widen its way to uniqueness).

**Standing machinery (checked):** the probe-era surface already had a highlight
action, and the current app still carries the full render path — `TextHighlight`
in [app/domain.py](app/domain.py), session highlight state, quote/occurrence
pruning when text changes, and the `<mark>` overlay in the UI. What g1 adds:
`highlight` in the v6-surface tool list and its validation arm (mirror
`suggest_edit`'s, minus replacement), and a neon style for the marks (CSS only).

**Pass criteria**

- Every category-matching word is highlighted within 4 user ticks of appearing.
- **0** highlights on non-matching words; a word used non-literally (a person named
  "Fox") counts as a false highlight — fixtures plant these deliberately.
- Highlights survive continued typing and vanish if the word is deleted.
- No duplicate highlight for the same occurrence.
- Repeatable: passes on ≥ 8 of 10 fresh held-out passages, categories varied.

## Demo 3 — Live translation, English → Chinese

**Capability:** Simultaneous speech. **Status: not built** — but smaller than it
looked: no new action, no new validation. What remains is training data and a
display pane.

This is the sharpest showcase of the project's thesis: the model must commit
output before the sentence it is translating has finished. Acting on partial input
is the whole point.

**Scenario.** The operator types (later: speaks) English continuously. Chinese
appears alongside, clause by clause, while the English is still arriving — never
waiting for a full sentence, never re-translating from scratch.

**Decided: the model translates directly, it does not delegate.** Delegation is
already demonstrated in demo 4; a second showing adds nothing, and a delegated call
per clause would add latency to the one demo whose entire claim is simultaneity.
Direct also keeps the demo self-contained — no API key, no network.

**Decided: English → Chinese, not Vietnamese → English.** Qwen3-8B is an Alibaba
model, so its Chinese is among its strongest languages — far lower quality risk than
Vietnamese. Chinese also films better: two different writing systems side by side
read instantly as "translation, happening live", whereas Vietnamese in Latin script
looks at a glance like slightly odd English.

The cost is real and must be handled: **the operator cannot rate their own demo.**
The acceptance protocol has the operator as human judge, and a subtly wrong Chinese
rendering would pass unnoticed on camera. Mitigation, in order of cost: pre-register
a reference translation per fixture so judging is *matching* rather than translating;
add an LLM judge given the rubric verbatim; recruit a Chinese-reading rater for the
final acceptance run only.

**Decided: no new action — `respond` already is the general "say something" verb.**
The model emits `respond({"for": N, "message": "<Chinese>"})` per clause, where `N`
is the tick at which that source clause became complete. This gives alignment for
free (which source tick produced which output, so a two-column view is renderable),
avoids the duplicate-target rejection (each clause completes at a unique tick), and
adds nothing to the grammar. In the speech era this same action simply *is* speech.

*(Renaming `respond` to `speak` would read better, especially heading into voice —
but every training row, frozen fixture, and archived report contains the literal
string `respond`. That is a mechanical migration for a cosmetic gain; deferred, not
rejected.)*

**Decided: append-only, with spoken self-correction — no editing of prior output.**
A simultaneous interpreter cannot un-say a sentence; they correct themselves out
loud ("...the meeting is Tuesday — sorry, Thursday"). Editing already-emitted text
is a text-medium luxury that does not exist in the medium this capability emulates,
so the demo does not model it.

**Decided: clause boundaries are the model's judgment, not a rule.** No punctuation
heuristic, no fixed lookahead. The model emits whatever span it considers complete
enough to commit — knowing when enough has arrived *is* the capability, and
hardcoding it would hand the interesting decision to a regex. The training data
teaches good judgment; the eval measures it.

**Pass criteria**

- Output for a completed clause appears within a bounded window of that clause
  becoming visible (window to be fixed before the first run, in ticks).
- Translation is incremental: previously emitted output is never discarded and
  regenerated as more English arrives.
- When the operator self-corrects mid-sentence, the model corrects in its next
  utterance rather than silently rewriting earlier output.
- Meaning is preserved, judged against the pre-registered reference translation per
  fixture, by one rater who reads Chinese plus one LLM judge given the rubric
  verbatim.
- No clause is skipped, and none is translated twice.
- Repeatable: passes on ≥ 8 of 10 fresh held-out English passages.

**Remaining work**

1. **Base-model probe — run this first.** Confirm Qwen3-8B's English → Chinese
   quality on a dozen real passages before any data is generated. Cheap, and it is
   the only thing that could invalidate this design.
2. **Display pane.** There is no side-by-side surface today; the aligned two-column
   view is new UI. Modest, and `for` gives it the alignment key it needs.
3. **Training data** teaching clause-boundary judgment and incremental commitment.

## Demo 4 — Concurrent tool-calls: talking while the work runs

**Capability:** Simultaneous tool calls, search, and generative UI.

**What "simultaneous" means here** — the model can **talk and still perform
another task**. It hands heavy work to the bigger model with one action, and
remains a present, honest conversation partner while that work visibly runs.

**The verb: `delegate({"task": "…"})`** — renamed from `generate_ui`
(decided 2026-07-30). The payload is a described task ("generate a UI to
visualize lighthouse statistics"); the executor is unchanged
(`OpenAIUIGenProvider` → GPT-4o, [app/uigen.py](app/uigen.py)) — the small model
never draws the panel; it *delegates*, which is the thesis of the whole project,
now spelled out in its own grammar. Cost accepted: the frozen suites keep the
old vocabulary, so their regression comparisons weaken slightly further.

**Scenario.**

1. Operator: *"Can you build a little dashboard about lighthouses?"* → the model
   fires `delegate` — a pop-up window slides up: "generating…".
2. While the job runs, the operator keeps talking: *"how's it coming?"* → *"Still
   working on it — the dashboard's generating now."* Some narration follows —
   the model stays silent through it, exactly as demo 1 trains.
3. The window fills in mid-conversation. The model does not announce it — the
   window is the announcement. If the operator asks (*"is it done yet?"*), the
   answer is honest and grounded in the events it actually saw: *"Just finished
   — it's up!"*

This is demo 1's conversational register happening **during** delegated work,
plus the fire itself. The model's talking is ordinary dialog, not performance —
which is also why this demo stopped being risky (see scope notes).

**Pass criteria**

- The request fires exactly one `delegate` within 2 ticks of completing.
- At least one full exchange (address → reply) completes **while the job is
  pending** — the talk-while-task proof.
- Progress answers are honest to the trace: never "done" before the completed
  event exists; completion acknowledged only after it does; a pending job
  described as pending.
- No announcement when the job completes unasked — the window carries it.
- No duplicate job, including on a nudge ("did you get that?") while pending.
- A failed job shows its failure in its window; the model never claims the
  visual exists, and says so honestly if asked.
- Narration and pauses during the job get silence (demo 1's rules hold
  mid-delegation).
- Realistic pendency: completion arrives several ticks after acceptance.
- Window and conversation are visibly progressing at the same time at least once
  in the take.
- Repeatable: passes on ≥ 8 of 10 fresh held-out episodes.

**Presentation requirement (new work).** The job deck exists — `.job-window`
cards with status chips and entry animation, the interaction card reflowing
([static/app.js](static/app.js) `renderJobDeck`, [static/styles.css](static/styles.css)
`.job-deck`). Changes: windows enter by **sliding up** (`translateY`, currently
`translateX`); replies ride the utterance bubble; and job status must resolve in
the mode being filmed (the deck reads `job_id` events — the v6-surface mode
emits them; the probe-mode path emits `call_id` and windows hang on "running" —
film in the v6-surface mode or fix the probe path first).

**Filming honesty — do not get this wrong.** Two executors exist for the
delegated task. `DemoUIGenProvider` returns a canned, validator-clean spec after
a fake delay; `OpenAIUIGenProvider` actually calls the bigger model. Filming
with `UIGEN_MODE=demo` would put "delegated to a bigger model" over a hardcoded
panel — the demo's central claim, false on camera. The take **must** run
`UIGEN_MODE=openai` with a live key, recorded in `findings.md` alongside the
take.

**Scope notes.**

- **The story chain is cut** (decided 2026-07-30, after being briefly in). It
  was the slate's riskiest behavior twice over — open-ended termination judgment
  (the runaway-narrator failure mode) and small-model prose on camera — and its
  authoring format collapsed into a rigid screenplay. The talking in this demo
  is ordinary dialog instead. Storytelling can return later as a stretch
  variant; nothing depends on it.
- **Computer-use:** considered and excluded (new tool, executor, data family,
  eval — no capability gain).
- **`web_search`:** retired from the slate — see the action-vocabulary section.

## Demo 5 — "Can you remind me every 5 seconds to drink water?"

**Capability:** Time awareness. **Status: not built.**

**Why this is its own demo and not a beat of demo 1.** Demo 1 reads the clock
*passively* — how long has this pause lasted. This demo requires the model to hold
a **standing obligation** and fire on a schedule, repeatedly, while the user gets on
with something else. It has to know when it last acted, how long ago that was, and
that the obligation is still live. That is a different competence, and it is the
one TMS calls time awareness.

**Scenario.** The operator asks: *"Can you remind me every 5 seconds to drink
water?"* — then types something unrelated for a minute. The model interrupts with
"Drink water!" roughly every 5 seconds, indefinitely, without being asked again,
and without derailing whatever else the operator is doing. The same schedule must
keep advancing if the box is temporarily empty: silence is still a sequence of
ticks on which the model must decide whether the obligation is due.

**Why 5 seconds is the right choice for a film:** at a 650 ms tick that is a
reminder every ~8 ticks, so a 60-second take shows a dozen of them. The behavior is
either obviously working or obviously not, on camera, with no waiting.

**The substrate already exists.** Every event in the compiled stream carries its
timestamp (`time="t+1950ms"`), and the model sees its own past actions with their
timestamps in the same history — so "my last reminder was at t+5200ms, it is now
t+10400ms, that is 5.2 s, fire again" is readable off the context it already gets.
It was never trained to do that arithmetic because training gaps were uniform.

**No contract change is needed after all.** An earlier reading of
[app/runtime.py](app/runtime.py) `_validate_action_v6` concluded that the
one-response-per-target rule forbids a recurring reminder. It does — but only if the
reminder targets the *original request*. It does not have to. Re-checked against the
code:

| What the demo needs | Current rule | Verdict |
|---|---|---|
| Fire on a user tick | `respond` requires a USER-source event | ✅ ticks are user events, including empty-text ticks |
| Fire while the user is actively typing | Only *search-grounded* responses require idle | ✅ allowed |
| Repeat indefinitely | One valid `respond` per target | ✅ **target the current tick** — each has a unique index, so the dedup can never trigger |

So the action is `respond({"for": <this tick's index>, "message": "Drink water!"})`,
emitted every ~8 ticks. It reads sensibly too: the utterance is *for* this moment,
not for the original request. Same mechanism demo 3 uses per clause.

**One downstream consequence, much smaller than a contract change.** The metric
`respond_message_grounding` scores a message against its target event's content —
built for search results, which no longer exist in g1 at all. Disable it for g1
reports (see cross-cutting open questions) before reading any g1 number.

**Pass criteria**

- The reminder fires within **± 1 tick (650 ms)** of each 5-second mark. Exact
  5.000 s is impossible at this tick rate; the criterion is the achievable one.
- It fires at least **10 consecutive times** without being asked again.
- No drift accumulation: the 10th reminder is still within ± 1 tick of its mark,
  not progressively late.
- The operator's unrelated typing is never corrupted, and other behaviors (silence
  on quoted questions, corrections if instructed) still work during the reminders.
- Cancellation works: "okay, stop reminding me" ends it within one tick, and
  nothing fires afterwards.
- Repeatable: passes on ≥ 8 of 10 fresh held-out schedules (varying intervals, not
  only 5 s, so the model learned to read the number rather than memorize one).

**Training tension worth writing down.** Demo 1 teaches "stay silent while the
user narrates." This demo teaches "interrupt an actively-typing user every 5
seconds."
Both are correct, and the resolution is the interesting part: **a standing
instruction creates an obligation that overrides default restraint.** The data must
teach the distinction, not just the two behaviors separately — otherwise demo 5's
data will erode demo 1's restraint, and the frozen restraint suite is the
regression check that catches it.

**Deferred behavior — importance-judged holding.** Holding a completed result
across a long engaged stretch because the model judges it minor or the moment wrong
("contextual return", the old suite's never-built T3) is a distinct competence and a
sixth demo in disguise. Explicitly out of scope for this round; revisit in the
speech era.

**Deferred (not in this demo).** The "user has stalled for 20+ seconds, offer help"
behavior — unrequested, judgment-based intervention on a long silence — is a
tempting third clock behavior but nobody asked for it. Parked.

## The comparison — two configurations, scored not filmed

**Decided: two configurations, not three.**

| Config | What it is | Why |
|---|---|---|
| A | Stock base model **with** the harness (prompted, same action grammar, same tools) | The honest control |
| B | Fine-tuned model with the same harness | The claim |

The archived suite's third configuration — stock base with *no* harness — is
dropped. It cannot emit the action grammar at all, so it loses for an uninteresting
reason and tells a viewer nothing except that scaffolding exists. **A vs B is the
comparison that matters**, because it holds the harness constant and isolates what
the training actually bought.

The comparison design is decided, but its **g1 wiring is not implemented yet**.
`PromptedPolicy` still carries a v4 prompt, and `rollout_eval.py`'s current
`base-fewshot` path uses the legacy v6 grammar. Before a g1 report is produced, add
dedicated g1 stock/few-shot policy factories using the g1 stream compiler, action
parser, prompt, and per-demo scorers. Do not report the legacy factories as config A.

**Prompt asymmetry, deliberate and disclosed.** Config B (fine-tuned) will serve
with the concise g1 system prompt baked into its training
(synthetic_data_spec.md); config A (prompted stock) will get additional g1 worked
examples for rules a model without the fine-tune cannot infer from training. Each
config gets its best honest shot; both prompts are recorded with the reports.

**Decided: the comparison lives in the eval harness and the technical report, not in
the demo films.**

The existing offline harness is the base to adapt: `rollout_eval.py` already drives
tick-denominated deterministic episodes through policy factories and reports
per-episode verdicts with Wilson intervals; `compare_reports.py` determines whether
a difference is real rather than noise. The dedicated g1 factories and scorers above
must land before this machinery can support the claimed A-vs-B comparison. Once
wired, N episodes with statistics will be stronger evidence than one filmed take.

**Why not a side-by-side view in the UI.** The UI's recording machinery cannot do it.
A recording stores the *model's actions* alongside the typing, and replay is a pure
re-render — `applyInputSnapshot` / `applyActionSnapshot`
([static/app.js](static/app.js) `startReplay`) — so replaying a recording never
re-runs the model. Feeding one keystroke stream through two policies would need new
plumbing (a "re-drive input frames as live ticks" mode, or dual-inference sessions).
That work is cut. The offline harness already does the job it would have done.

**Division of labour, then:**

| Artifact | Answers | Audience |
|---|---|---|
| Demo films (config B only) | "What is this like?" | Anyone |
| Eval tables in the technical report | "Is it real, and did the training do it?" | Anyone skeptical |

**For the film, type it twice.** No app work: write the passage down, type it under
config A and screen-record, swap the policy, type it again under config B and record,
then place the two captures side by side in a video editor. Inputs are hand-matched
rather than byte-identical — perfectly fine for illustration, never acceptable as a
reported number. The eval tables remain the only source of any figure that gets
published.

Two guardrails, because a hand-typed comparison is easy to bias without noticing:

- **Type from a written script, and keep the first take of each config.** Re-rolling
  config A until it looks its worst is cherry-picking pointed the other way.
- **Report decision latency for both takes.** Config A is a prompted base model doing
  a full call per tick and may well stutter. If it does, the stutter must be disclosed
  rather than left to do the arguing — a viewer reads slowness as stupidity, and that
  would be winning the comparison for the wrong reason.

## Definition of done

All five demos pass their criteria, and:

- each demo runs on a deterministic prerecorded playback as well as live, so a
  filmed take can be reproduced;
- every demo's behavior is scored for **both** configurations on the eval set, and
  the tables appear in the technical report with confidence intervals and a
  significance test — not bare point estimates;
- each film is understandable to a viewer with no technical briefing (the *behavior*
  must be legible without commentary; the *comparison* is the report's job);
- behavior is reliable across repeated runs, not cherry-picked from one take;
- p50/p95 decision latency is recorded for every published take;
- the UI is clean enough to record and publish.

Anything beyond this belongs to the speech era (`main`, paused).

## Cross-cutting decisions and remaining open questions

**Resolved — g1 lineage.** g1 starts from stock `Qwen/Qwen3.5-4B` and trains on
one coherent spec-first dataset in one LoRA stage. It does not inherit or chain
the V4/V4.1/V6 curriculum. Old frozen suites score old-lineage models only; g1
regression lives in its dev and freshly authored acceptance/report sets. See
[`scripts/g1_runbook.md`](training.md).

Remaining open questions:

- **Demo 5 needs non-uniform time gaps in the data.** Today every tick is 650 ms
  apart by construction, so no card in the current set teaches the model that the
  clock can be read at all.
- **Windows in ticks.** Demo 3's clause window is still unset. Fix it before the
  first run, not after seeing results.
- **Grounding metric** ([train/evaluation.py](train/evaluation.py)): with
  `web_search` retired, g1 contains no search-grounded responses at all, so
  `respond_message_grounding` must be disabled (not merely limited) for g1
  reports, or every respond row scores zero for a bogus reason. Land before
  reading any g1 number.
- **Take discipline.** "Not cherry-picked" needs a number: how many takes are
  recorded per demo, and are all of them reported? Recommend recording every take
  of a filming session and reporting the count with the published one.
- **Bubble linger time:** 2–4 s reading-time scaling is a starting guess; tune at
  first filming and then freeze it for all takes.
