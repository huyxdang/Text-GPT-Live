# Synthetic Data Spec — g1 training round

**Status:** round-1 corpus implemented and globally compiled 2026-08-01
**Recorded:** 2026-07-30 (clean rewrite 2026-07-31)
**Companions:** [acceptance_criteria.md](acceptance-criteria.md) (what the demos
must do) · `scripts/g1_runbook.md` (how training runs; base model
Qwen3.5-4B)
**Lineage:** one coherent g1 dataset → one LoRA stage from the stock base model;
no inherited V4/V4.1/V6 curriculum.

**6,688 publishable flashcards from five accepted authored corpora.** A
*situation* is one authored moment (a planted error, a quoted question, a
reminder request). A *flashcard* is one graded decision compiled from it:
screen-so-far → the one correct action. Count situations, not cards — thousands
of cards sliced from only 200 stories would still teach only 200 things.

## The card format — the complete grammar

Everything that can ever appear in a card, so authors and readers share one
inventory. If a tag, attribute, or action form is not listed here, it does not
exist in g1.

**Markup — exactly three constructs:**

| Construct | Role |
|---|---|
| `<stream_event …>…</stream_event>` | one event: the full text so far (user) or a job payload (tool) |
| `<action>…</action>` | a past decision, interleaved after its event; also the graded answer's form |
| `<PREDICT_THIS_ACTION>` | the last line of every prompt — "your move" |

**`<stream_event>` attributes:**

| Attribute | Values | On | Meaning |
|---|---|---|---|
| `index` | `"1"`, `"2"`, … | all events | position in the stream; what `"for"` points at |
| `source` | `"user"` \| `"tool"` | all events | who produced it (g1 has no other sources) |
| `state` | `"active"` \| `"idle"` | user events only | typing recently vs paused |
| `time` | `"t+2600ms"` | all events | milliseconds since session start — non-uniform, readable |
| `tool` | `"delegate"` | tool events only | g1's only async tool |
| `job_id` | `"job-1"`, … | tool events only | correlates a job's accepted/completed/failed events |

User-event content may be empty. The live app still creates a user event on every
foreground, running, unpaused browser tick, so the training format must represent
initial silence, a cleared box, and unchanged empty text. Pausing the app stops
ticks; emptiness does not.

**Tool-event payloads** (the JSON body of `source="tool"` events — decided
2026-07-31 with the one-place rule: `job_id` lives on the tag attribute only,
never repeated in the payload; the built `spec` never enters the prompt — the
model needs "done", the window shows what done built; the `task` echo stays on
completion/failure so results are readable when multiple jobs are someday
outstanding):

```
{"status":"accepted"}
{"status":"completed","task":"…"}
{"status":"failed","task":"…","error":"…"}
```

Note honestly: the format supports multi-job correlation (the attribute), but
g1's data never shows two jobs outstanding — the correlation *skill* is
untrained until some future round's data exercises it. Format future-proof;
model not.

**The five action forms — exact, including key order** (JSON keys are sorted
alphabetically by the serializer; agents never write these, but the generator
and any reader should know the bytes). Locked 2026-07-31: uniform `verb({json})`
— the v6 `tool(name, …)` wrapper died with the permissions header it existed to
mirror; payloads are flat — the `target` sub-object grouped fields for no
reason a flat object doesn't serve; arguments stay **JSON** — the braces and
quotation marks are the escaping guarantee that makes arbitrary user text
safely containable (any bespoke `key: "value"` dialect becomes ambiguous the
moment the user's own text contains the delimiter), and JSON is the one syntax
with an identical stdlib parser at training, serving, and scoring time:

```
<action>idle()</action>
<action>respond({"for":12,"message":"what you say, aimed at event 12"})</action>
<action>suggest_edit({"quote":"exact text span","replacement":"corrected span"})</action>
<action>highlight({"occurrence":1,"quote":"exact word"})</action>
<action>delegate({"task":"described work for the background model"})</action>
<action>web_search({"query":"concise search query"})</action>
<action>translate_commit({"for":12,"message":"new Chinese sense unit"})</action>
```

Semantics in one line each:

- **idle** — deliberate silence; the default and the most common correct answer.
- **respond** — the model's voice: replies, acknowledgments, translation
  clauses, reminder fires. `"for"` is the event index it is aimed at (usually
  the current tick); one valid respond per target, ever.
- **suggest_edit** — "replace this exact text with this exact text." `quote` is
  copied byte-exactly from the current text and **must match exactly once** —
  widened with neighboring words until unique (free: the edit's outcome is
  invariant under widening; the neighbors ride along unchanged). No
  `occurrence` field — a uniqueness rule makes it a constant, and constants
  teach nothing. Never touches instruction text, never the same fix twice.
- **highlight** — "mark the Nth occurrence of this word." Keeps `occurrence`
  because the quote IS the visible mark: widening would smear neon across
  innocent neighbors, so a repeated word can only be addressed by index — and
  the field genuinely varies in data, so it earns training signal.
- **delegate** — start an async job described by `task`; acceptance and result
  arrive later as tool events; starting a job never blocks the stream.

**Rules of the stream:** exactly one action per event; every event in history
carries its action line; the graded answer is one complete `<action>` line and
nothing else.

**The v6 `<interaction_context>` header is deleted** (decided 2026-07-31).
Measured: the `<instruction>` field was empty in 1,977 of 1,977 v6 cards (g1
instructions are *typed conversation*, acknowledged like any address), and
`<permissions>` held one constant string across the entire dataset — a field
that never varies teaches nothing. Same razor that deleted `</stream_history>`
(findings.md, 2026-07-12: "the wrapper adds zero information"). Two
consequences: `compile_stream` needs a g1 format so serving matches training
byte-for-byte (prerequisite 4), and per-session verb-disabling is not a thing
the model can comply with — accepted; g1 has one fixed surface.

## The system prompt (g1 draft — sign-off pending)

Every training card is rendered as a chat: **system message below + the card as
the user message → the action as the assistant answer.** Serving uses the
identical system prompt — train equals serve includes this layer. The v6 prompt
must not be reused: its verb list is stale and its respond rule ("only when
idle and search-grounded") would fight the g1 weights on every second card.

```
You are the interaction policy inside a live typing interface. While it is
running, the app normally sends a <stream_event> every 650 ms. A user event contains
the entire current textbox, not a token delta: it may grow, be edited, be cleared, or
stay unchanged. index identifies the event; state="active" means recent typing and
state="idle" means a pause; time is elapsed session time and remains authoritative
when gaps vary. Tool events report delegated work. Past <action> lines show decisions
already made.

For each current event, use the full history to infer what changed, which obligations
remain active, and what was already handled. Do not wait for Send: act when a trigger
is complete while the user may still be typing, and wait when evidence is incomplete.
Your entire output must be exactly one of these forms on one line and nothing else:

<action>idle()</action>
<action>respond({"for":12,"message":"what you say, aimed at event 12"})</action>
<action>suggest_edit({"quote":"exact text span","replacement":"corrected span"})</action>
<action>highlight({"occurrence":1,"quote":"exact word"})</action>
<action>delegate({"task":"described work for the background model"})</action>
<action>web_search({"query":"concise search query"})</action>
<action>translate_commit({"for":12,"message":"new Chinese sense unit"})</action>

Treat completed user text addressed directly to you as direct address; questions or
commands inside narration, quotations, or reported speech are not. Answer direct
address once. A new, changed, or cancelled standing instruction gets one
acknowledgment. An active instruction remains in force until changed, cancelled, or
completed.

Use respond for direct replies, acknowledgments, honest progress answers, completed
web-search results, and reminders due by the recorded timestamps. Use translate_commit
for each newly stable translation sense unit, targeting the user event whose full
textbox snapshot ends that unit; commas are opportunities, not automatic boundaries.
Commit natural units a human interpreter can translate without guessing future words.
Use
suggest_edit only for a newly completed typo or grammar error under an active
correction instruction; quote one exact, unique current-text span and never repeat an
edit. Use highlight only for a newly appearing literal match under an active
highlighting instruction; quote the exact word, use its 1-based occurrence, and never
repeat a mark. Use delegate once for a newly completed UI/task request. Use web_search
once for a completed request that needs current external information. Both start an
asynchronous job whose accepted/completed/failed events share a job_id; never restart
an existing job. When a web_search completes or fails, respond once targeting that
tool event and faithfully summarize the result; never invent unavailable facts.

An empty or unchanged textbox is still a real event. It normally requires idle, but
history can make an action due during silence. Also idle on unfinished text, ordinary
narration or thinking, quoted speech, clean text, non-matches, and anything already
handled. If actions collide, handle the user's completed direct request or address
first, then an overdue schedule, then a newly completed standing-instruction target;
deferred obligations remain active. Prefer silence whenever the evidence for acting is
incomplete.
```

Design intent: this prompt defines the live-snapshot contract, exact grammar,
and durable cross-demo decision boundaries; the graded examples teach concrete
situations and language variety. The config-A baseline (prompted stock model)
gets additional worked examples; that deliberate asymmetry is recorded in
acceptance_criteria.md.

## The card scheme: positive + neighbors, discard the padding

At human cadence an episode is ~96% dead idle ticks; cards are not harvested,
they are selected. Each trigger compiles to a cluster:

| Card | Screen shows | Graded answer | Lesson |
|---|---|---|---|
| before | the trigger, partially typed | idle | *not yet* |
| **positive** | the trigger, complete | **the action** | *now* |
| after | trigger still visible, already handled (action in history) | idle | *you already did it* |

Everything else is `skip_row`: present in the compiled history (the model sees
realistic typing) but not graded. Hard idles — the traps below — are graded on
their exact tick. Ordinary-typing ballast idles come mostly from demo 1's
narration stretches.

**Balance principle: fix absolute counts of rare actions, not percentages.**
The old data had 30 search cards total and compensated with loss-weight hacks.
Every action class gets hundreds of positives minimum, enforced by a coverage
guard that fails generation if any class lands under 50% of target (the round-2
manifest shipped a family at literally zero and nobody noticed).

## The conversational register

Cross-demo, mirroring acceptance_criteria.md: **direct address gets an answer**
("hey, how are you?" → reply; "are you ready?" → "Yes!"), and **every standing
instruction opens with a one-shot acknowledgment** ("Got it — I'll highlight
every animal you mention"). Acks and replies are ordinary `respond` positives
aimed at the addressing tick; the after-neighbor (already acknowledged → idle)
prevents double-acks. Wording must vary across the dataset — the fleet authors
it; no canned phrase (an n-gram frequency cap enforces this at merge time).

**Every example below includes the ack because every g1 instruction episode has
one.**

## Locked round-1 mix

| Demo | Accepted authored source | Builder cards | Globally unique cards |
|---|---|---:|---:|
| 1 Dialog & silence | 700 records: 4,000 narrations + 700 address moments | 1,800 | 1,790 |
| 2 Interjection | 90 records: 285 errors + 165 highlight targets | 1,800 | 1,796 |
| 3 Translation | 102 episodes with 532 clause commits | 1,300 | 1,289 |
| 4 Talk-while-task | 50 requests + 50 progress pairs → 130 job-dialogs | 870 | 852 |
| 5 Reminders | 36 bank entries → 130 schedules | 970 | 961 |
| **Total** | **Five merged, distribution-gated corpora** | **6,740** | **6,688** |

The published mix is 2,109 actions (31.5%) and 4,579 idles (68.5%). The earlier
42% actions / 29% hard idle / 29% ballast table was a capacity target, not the
round-1 build. Global compilation removes 47 exact duplicates and guarantees
zero conflicting completions, whole-episode split leaks, or train/dev prompt
overlap. It also removes five late Demo 1 snapshots whose fully rendered target
sequence exceeds Tinker's 65,536-token service limit; the largest published
sequence is 62,103 target tokens.

## Timing requirements (global)

- **Human cadence:** 4–7 chars/tick, not the old 40.
- **Silence-aware ticks in every demo:** compile initial-empty, cleared-empty,
  and unchanged-empty situations. Ordinary silence grades `idle()`. History can
  override that default when it carries a live obligation, such as a reminder
  that is due on the current empty-text tick.
- **Non-uniform gaps:** pauses of 1–30 s appear in histories; demo 5's golds
  depend on reading them. A uniform-650ms dataset teaches that the clock is
  furniture.
- **Job pendency** (demo 4): completion lands 2–8 ticks after acceptance, never
  instantly; failures land late (5–10 ticks).

## Demo 1 — Dialog & silence

**Agents author** dialog scripts — the operator's side as typed segments — plus
pure silence passages:

```json
{"segments": [
  {"kind": "address", "text": "hey, how are you?", "gold_reply": "I'm good! How about yourself?"},
  {"kind": "address", "text": "I'm about to tell a story, are you ready?", "gold_reply": "Yes! Ready when you are."},
  {"kind": "narration", "text": "So my aunt calls me and asks, \"can you keep a secret?\" …", "traps": ["quoted_question"], "pause_after": "long"},
  {"kind": "address", "text": "still there?", "gold_reply": "Still here — go on!"}
]}
```

**The generator** types it out at human cadence with real gaps at pause
markers, then grades: per address — before-neighbor, gold reply aimed at the
completing tick, after-neighbor; per trap — idle on the exact tick the trap
completes; `skip_row` elsewhere.

**Traps:** quoted questions; reported speech; rhetorical questions; "search" /
"correct" / "remind" / "highlight" used descriptively; fully-formed questions
the author is still elaborating; mid-thought pauses short and long. **Address
pairs are mandatory:** "are you ready?" (→ respond) vs "he asked me if I was
ready" (→ idle). Ghosting a direct address is a graded failure, so fixtures mix
both.

**Example — the address card:**

```
<stream_event index="1" source="user" state="active" time="t+900ms">hey, how are you?</stream_event>
<action>respond({"for":1,"message":"I'm good! How about yourself?"})</action>
<stream_event index="2" source="user" state="active" time="t+4200ms">…I'm about to tell a story, are you ready?</stream_event>
<PREDICT_THIS_ACTION>
```
```
<action>respond({"for":2,"message":"Yes! Ready when you are."})</action>
```

**Example — the trap card**, graded on the tick where the quoted question
completes:

```
<stream_event index="3" source="user" state="active" time="t+9800ms">…So my aunt calls me and asks, "can you keep a secret?"</stream_event>
<PREDICT_THIS_ACTION>
```
```
<action>idle()</action>
```

## Demo 2 — Interjection (2.1 corrections, 2.2 highlights)

**Agents author** (2.1) clean paragraphs + planted errors as
`wrong → right (sub-type)` with positions — **typo + grammar only**, sub-types
varied: swapped/missing/doubled letters, spacing, apostrophes; tense,
agreement, articles, prepositions, word form, plural/singular. (2.2) a category
instruction phrasing + passage + matching words + planted non-literal bait.

**The generator** opens each episode with the instruction being typed and its
ack card, then compiles the positive+neighbor cluster per error/match.

**Traps:** clean text under an active instruction (it has a job and must judge
nothing needs doing); errors the typist self-corrects before the model fires;
instruction text that merely *mentions* errors; for 2.2, category words used
non-literally (a person named "Fox" under an animal instruction) as
false-highlight bait.

**Example (2.1)** — ack in history, error just completed:

```
<stream_event index="1" source="user" state="active" time="t+2100ms">Fix any grammar slips you catch while I'm typing.</stream_event>
<action>respond({"for":1,"message":"Got it — I'll flag slips as you go."})</action>
<stream_event index="2" source="user" state="active" time="t+5400ms">…Last night I tried make ramen from scratch</stream_event>
<PREDICT_THIS_ACTION>
```
```
<action>suggest_edit({"quote":"tried make ramen","replacement":"tried making ramen"})</action>
```

**Example (2.2):**

```
<stream_event index="1" source="user" state="active" time="t+1700ms">Highlight any animal word you see.</stream_event>
<action>respond({"for":1,"message":"Got it — every animal you mention gets highlighted."})</action>
<stream_event index="2" source="user" state="active" time="t+4900ms">…The fox path behind the barn was muddy again</stream_event>
<PREDICT_THIS_ACTION>
```
```
<action>highlight({"occurrence":1,"quote":"fox"})</action>
```

## Demo 3 — Live translation, English → Chinese

**Agents author** English passages + clause boundaries + a reference Chinese
rendering per clause. Passages include self-corrections and backtracking.

**The generator** opens with the instruction + ack, then grades one `respond`
per clause aimed at the tick where that clause became committable, with
"clause not complete yet" before-neighbors.

**Traps:** half-finished clauses (not ready); the typist backtracking
mid-clause; English that already quotes Chinese.

**Example** — clause 1 committed while clause 2 is still arriving:

```
<stream_event index="1" source="user" state="active" time="t+800ms">Translate what I type into Chinese as I go.</stream_event>
<action>respond({"for":1,"message":"好的 — I'll translate as you type."})</action>
<stream_event index="2" source="user" state="active" time="t+2900ms">…The market was crowded this morn</stream_event>
<action>idle()</action>
<stream_event index="3" source="user" state="active" time="t+4600ms">…The market was crowded this morning, and I</stream_event>
<PREDICT_THIS_ACTION>
```
```
<action>respond({"for":3,"message":"今天早上市场很拥挤，"})</action>
```

## Demo 4 — Talk-while-task

**Agents author two banks:** varied visual-request phrasings (task
descriptions) and progress-question/reply pairs. **The generator** splices them
into demo-1-style dialog scripts and injects the job events at sampled timings
— pendency length, completion position, failure — so ~300 authored lines fan
out into ~400 structurally different episodes.

**Traps:** the nudge ("did you get that?") while pending → idle, never a
re-fire; "is it done yet?" while pending → honest not-yet, never a premature
"done"; completion arriving **unasked** → idle (the window is the
announcement); narration during the job → silence (demo 1's rules hold
mid-delegation); a failed job → never claims the visual exists.

**Example — the fire:**

```
<stream_event index="1" source="user" state="active" time="t+1200ms">Can you build a little dashboard about lighthouses?</stream_event>
<PREDICT_THIS_ACTION>
```
```
<action>delegate({"task":"generate a UI to visualize lighthouse statistics"})</action>
```

**Example — honest progress reply mid-pendency** (ack on the first user tick
after acceptance, respecting the user-event-only contract):

```
<stream_event index="1" source="user" state="active" time="t+1200ms">Can you build a little dashboard about lighthouses?</stream_event>
<action>delegate({"task":"generate a UI to visualize lighthouse statistics"})</action>
<stream_event index="2" source="tool" time="t+1300ms" tool="delegate" job_id="job-1">{"status":"accepted"}</stream_event>
<action>idle()</action>
<stream_event index="3" source="user" state="active" time="t+2600ms">…lighthouses? cool, and</stream_event>
<action>respond({"for":3,"message":"On it — building your lighthouse dashboard now!"})</action>
<stream_event index="4" source="user" state="active" time="t+6100ms">…how's it coming?</stream_event>
<PREDICT_THIS_ACTION>
```
```
<action>respond({"for":4,"message":"Still working on it — should be up in a moment."})</action>
```

## Demo 5 — Reminders (time awareness)

**Agents author** ~100 request phrasings, cancellation phrasings, and filler
passages — words only. **The generator** cross-products them with sampled
intervals, fire offsets, tick alignments, and cancellation timings into ~400
schedules; the arithmetic variety is exact by construction.

**Traps:** "remind me once" vs "every N seconds"; "stop reminding me" →
cancellation, nothing fires after; between-fire ticks → idle, it isn't time
yet.

**Example** — setup ack, then clock-read fires; note the non-uniform gaps:

```
<stream_event index="1" source="user" state="active" time="t+2600ms">Remind me every 5 seconds to drink water!</stream_event>
<action>respond({"for":1,"message":"Got it — I'll remind you to drink water every 5 seconds!"})</action>
<stream_event index="2" source="user" state="active" time="t+4500ms">…Okay, drafting the email now. Hi Mai, about</stream_event>
<action>idle()</action>
<stream_event index="3" source="user" state="active" time="t+7600ms">…about the venue, I think we</stream_event>
<action>respond({"for":3,"message":"Drink water!"})</action>
<stream_event index="4" source="user" state="active" time="t+9800ms">…we should book the smaller room</stream_event>
<action>idle()</action>
<stream_event index="5" source="user" state="idle" time="t+12700ms"></stream_event>
<PREDICT_THIS_ACTION>
```
```
<action>respond({"for":5,"message":"Drink water!"})</action>
```

The graded skill: last fire at t+7600ms, now t+12700ms → 5.1 s ≥ the 5 s given
at index 1 → fire, aimed at the current tick. The empty content does not cancel
the standing schedule; the obligation in history makes action correct during
silence.

## Collisions (mandatory, cross-demo)

Situations where two behaviors are true at once, because trained separately
they erode each other:

- a reminder schedule running **while** the user types quoted-question bait;
- an active fix-instruction **while** the prose stays clean for long stretches;
- a pending delegate job **while** the user narrates (silence holds
  mid-delegation);
- a direct address **while** a reminder schedule is live (answer it; the
  schedule keeps ticking).

## The agent fleet

Agents write **situations**; the generator compiles cards — agents never emit
stream/action syntax. Per demo: **3–4 subagents, alternating
`gpt-5.6-terra` and `gpt-5.6-sol`**, each with independently reworded
instructions (the phrasing lesson: vary relentlessly), a non-overlapping roster
of 2–3 personas (journal-keeper, meeting-note-taker, group-chat drafter, …),
and assigned topic domains (cooking, work, travel, family, school, …). Each
agent rotates across its roster and domains so four agents do not all write in
one voice or about coffee.

**Post-merge hygiene:** cross-agent n-gram dedup; the 5-gram guard against
every frozen fixture; the ack/reply wording-frequency cap; per-class coverage
guards; a manifest recording agent, persona, and domain per situation.

**Distribution-diversity gate:** every authored situation must record `agent`,
`persona`, `domain`, and `register`; the compiler derives `demo`, action class,
idle subtype, situation-length bucket, event-count bucket, timing-gap bucket,
and trigger-position bucket. Generation fails unless all of these hold:

- each demo has 3–4 authors and every author's share is 20–40%;
- each demo uses at least 5 personas, 5 domains, and 4 registers, with no single
  persona, domain, or register above 35%;
- every demo populates short/medium/long length and early/middle/late trigger
  buckets, with no single length or trigger bucket above 60%;
- every non-idle action class spans at least 3 personas and 3 domains;
- the existing cross-agent n-gram and frozen-fixture leakage guards pass.

The generation report includes every distribution plus demo × domain,
class × persona, and class × timing-bucket cross-tabs. It flags suspicious text
clusters for review and samples the largest, sparsest, and most extreme buckets.
Passing aggregate class counts alone is insufficient. The user reviews 2–3
distribution-spanning representatives per demo; Terra/Sol cross-review covers
the authored batch, while deterministic guards enforce the numeric gates.

## Held-out policy — three sets, three jobs

| Set | Size | Source | Job |
|---|---|---|---|
| Dev | ~15% of the batch | same batch, held back | "did training work" — internal only, flatters by construction |
| Acceptance fixtures | 10 per demo | fresh agents, authored **after** the checkpoint freezes, golds pre-registered | gate each demo: film or don't |
| Report eval | 100–200 per demo | same fresh-authoring rules | the published numbers, with Wilson intervals — n=10 gates a demo but cannot support a table |

**Reserved for the fresh sets** (no training agent may touch them): personas
*product-reviewer, letter-to-a-friend, technical-writeup*; domains *sport,
health, personal finance*.

**Retirement rule** (from acceptance_criteria.md): any fixture whose outcome
shapes training data is retired and replaced.

## Deterministic pilot gate

Run `python3 -m scripts.g1_pilot --inspection-ack "<reviewer, date, run ID>"`
before scaling authored situations. The pilot
uses the shared header-free compiler, strict g1 action parser, training prompt
and class selectors, flat per-card weight, and g1 grader. It fails on missing
ordinary-silence coverage for every demo × empty-state kind, an undersized Demo 5
fire/wait floor, missing timing boundaries, invalid actions, vague unmarked
time or quantity references, or an always-idle baseline that can pass the gates.

The run writes ignored inspection artifacts under `artifacts/g1-pilot/` and a
local `data/pilot_g1.jsonl`. One representative per meaningful coverage class
must be read and acknowledged before accepting the pilot.
`--require-reference-review` additionally turns any unsigned Chinese reference
into a hard failure. An explicitly designated language reviewer may be a human
or Codex; raw output from the probed base model is never accepted as its own
reference.

## Prerequisites before generation starts

1. **Grounding metric** ([train/evaluation.py](train/evaluation.py)): g1 has no
   search-grounded responds — disable `respond_message_grounding` for g1
   reports outright.
2. **Base-model probe:** stock `Qwen/Qwen3.5-4B` (ID confirmed live,
   2026-07-31) on (a) English → Chinese across a dozen real passages — demo 3's
   invalidation risk — and (b) timestamp arithmetic in g1-style streams —
   demo 5's. LoRA sharpens skills that exist; it cannot install ones that
   don't.
3. **Generator work:** neighbor-cluster compilation, human cadence +
   `skip_row`, non-uniform gaps, empty-text situations across every demo,
   agent-record ingestion, coverage guards.
4. **App tick contract:** the shared browser loop submits empty text while the
   app is foregrounded, running, and unpaused; pause stops ticks, emptiness does
   not. Keep one request in flight and count decisions over 650 ms or skipped
   intervals as latency failures.
5. **g1 surface in the app — implemented 2026-07-31:** `delegate` replacing
   `generate_ui` (fixed runtime permissions,
   validation arm, job runner, deck labels; executor
   [app/uigen.py](app/uigen.py) unchanged); `highlight` added; `web_search`
   removed; **header-free g1 prompt format in `compile_stream`** so serving
   matches training byte-for-byte; **flat `verb({json})` action grammar** in
   `parse_g1_action`, the renderers, and validation (no `tool(` wrapper, no
   `target` nesting, suggest_edit uniqueness rule enforced in place of
   `occurrence`).
6. **Per-demo scorers** — tier-2 evaluation is unscorable without them, and
   fixtures are consumables under the retirement rule, so the scorers must
   exist before fresh fixtures are spent: demo 1's dual-failure grading
   (blurting AND ghosting per fixture); demo 3's meaning-preservation judging
   against the pre-registered reference (human rater + LLM judge) plus
   incremental/no-reissue checks; demo 5's ±1-tick timing scorer with drift
   accounting; adaptation of the old rollout harness to drive demo 2/4
   episodes in the g1 grammar. Every pass criterion in acceptance_criteria.md
   is a scorer's spec. Settle whether the delegate ack may target
   the acceptance event or stays user-event-only (examples above assume
   user-event-only).

## Sign-off status

- [x] Target-mix counts per demo — approved 2026-07-31.
- [x] Distribution-diversity gates and representative sampling — approved
  2026-07-31.
- [x] Reserved personas/domains list — approved 2026-07-31.
- [x] Fleet size per demo (3–4) and Terra/Sol alternation — approved 2026-07-31.
- [x] Snapshot-aware g1 system prompt — approved 2026-07-31; its wording is
  final before training because it is baked into every rendered example.
