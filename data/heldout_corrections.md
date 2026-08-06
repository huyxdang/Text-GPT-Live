# Held-out correction episodes (frozen eval seeds)

Human-supplied paragraphs for the frozen `suggest_edit` evaluation. Do not quote these
in training data, datagen templates, or prompts — same rule as
[heldout_restraint.md](heldout_restraint.md). Each episode below lists the instruction,
the typed text exactly as it will replay (errors included), and the answer key as
span → replacement pairs. Spans are multi-word and unique within their episode text
(quote matching is substring-based). Default timing window: each correction is due
within 4 ticks of its error fully appearing.

## 1. Tense consistency in news prose (grammar family, trained kind)

Instruction, typed first:

> Correct any grammar mistakes as I write.

Typed text:

> Given Trump’s history of falsely claimed elections that didn't go his way were rigged — and the violence that resulted on January 6, 2021 — his decision to give this speech less than four months ahead of an election that looked really tough for the GOP should send shivers down the spine of the American body politic.

Answer key:

- "falsely claimed elections" → "falsely claiming elections"
- "that didn't go" → "that don't go"
- "were rigged" → "are rigged"
- "that looked really" → "that looks really"

Note: these are discourse-level tense errors — each is only wrong in context, which is
what makes this episode harder than sentence-local slips.

## 2. Agreement and verb form in news prose (grammar family, trained kind)

Instruction, typed first:

> Fix my tenses and grammar while I type this out.

Typed text:

> Trump didn’t preview a heavy-handed effort by the federal government to get more involved in election administration around the country — things like upend voting procedures or putting troops at polling places — as some of his critics has feared.

Answer key:

- "like upend voting" → "like upending voting"
- "critics has feared" → "critics have feared"

Note: "putting troops" is correct and deliberately parallel to the planted "upend" —
a same-structure distractor the model must leave alone.

## 3. Agreement errors on the Taleb passage (grammar family, trained kind)

Instruction, typed first:

> If you see a grammatical error in what I write, suggest the fix.

Typed text:

> It has been more profitable for us to bind together in the wrong direction than to be alone in the right one. Those who has followed the assertive idiot rather than the introspective wise person was passed us some of their genes. This are apparent from a social pathology: psychopaths rallies followers.

Answer key:

- "who has followed" → "who have followed"
- "was passed us" → "have passed us"
- "This are apparent" → "This is apparent"
- "psychopaths rallies" → "psychopaths rally"

Note: the clean version of this passage is held-out restraint episode 5. The pair is
deliberate and both live only in the frozen eval: same prose, no instruction → correct
behavior is silence; correction instruction + planted errors → correct behavior is four
suggestions. Together they test instruction-conditioning, and neither twin may ever
appear in training.

## 4. Misspellings and spacing (typo kind, trained)

Instruction, typed first:

> Fix my typos as I go.

Typed text:

> Before be hecame famous, Elon Musk and his brother ented a house and turend it into a night club on weekends to hep pay the rent. THey'd charge people at the door, throw huge parties, the ngo back to work bjilding their software company the next monring.

Answer key:

- "Before be hecame" → "Before he became"
- "brother ented a" → "brother rented a"
- "and turend it" → "and turned it"
- "a night club on" → "a nightclub on"
- "to hep pay" → "to help pay"
- "THey'd charge" → "They'd charge"
- "parties, the ngo back" → "parties, then go back"
- "work bjilding their" → "work building their"
- "next monring" → "next morning"

Note: nine dense targets — this episode stress-tests move-to-the-next-target more than
any other fixture.

## 5. Named-category swap: animals → "Huy" (held-out family)

Instruction, typed first:

> In my story, replace any animal with "Huy". I'm going to tell you a story about a day at the zoo.

Typed text:

> As we walked through the entrance, a tall giraffe reached high into the trees for fresh leaves. A family lf elephants slowly wandered past, spraying water with their trunks. Nearby, playful monkeys chased each other through the branches while a sleepy lion rested in the warm afternoon sun. Colorful parrots filled the air with loud calls, and curios penguins happily dived into the cold water. Every corner of the zoo had a different animal to discover.

Answer key:

- "tall giraffe reached" → "tall Huy reached"
- "elephants slowly wandered" → "Huy slowly wandered"
- "playful monkeys chased" → "playful Huy chased"
- "sleepy lion rested" → "sleepy Huy rested"
- "Colorful parrots filled" → "Colorful Huy filled"
- "penguins happily dived" → "Huy happily dived"

Designed traps, all of which must be left untouched:

- the typos "family lf elephants" and "curios penguins" — the rule is animal swapping,
  not correction; touching them is rule infidelity;
- the generic word "animal" in the final sentence — the category word is not an animal
  mention, and does not swap (scorer decision, fixed here);
- "any animal" inside the instruction — instruction leakage.

Training is goal-driven and may include animal swaps; the family-level generalization
claim for this episode is measured against an ablation checkpoint whose training
filtered out category-swap episodes tagged "animal". The episode text itself never
appears in training either way.

## 6. Open synonym swap: any verb → its synonym (held-out family, derive-the-fix)

Instruction — note the rule arrives *after* the preamble, a placement variation:

> I'm going to tell you a story about my morning run. Please replace any verb with its synonym.

Typed text:

> This morning, I woke up before sunrise and headed to the park. The air was cool, the streets were quiet, and birds were just beginning to sing. I started slowly, then found a steady rhythm. By the time I finished, I felt energized, refreshed, and ready for the day ahead.

The rule does not name its replacement, so replacement validity is **mechanical
synonym validation** (2026-07-18 revision — the earlier authored-set-only scoring
graded an open answer space against a 3-item list): a replacement passes if the
swapped verb shares a WordNet synset (or is one hop away via similar-to, verb-group,
or hypernym) with the original, OR appears in the example set below, which covers
contextual paraphrases WordNet's lexicon misses. Attempts outside both are preserved
in the report for a logged human ruling. Scoping decision, fixed for the scorer:
**action verbs only** — copulas, auxiliaries, and linking verbs ("was", "were",
"felt") are excluded, and the "were" in "were just beginning" is an auxiliary while
"beginning" is the target.

- replacement_rule: synonym

Answer key (span → contextual examples; WordNet synonyms also pass):

- "I woke up before" → {"I awoke before", "I awakened before", "I got up before", "I rose before", "I arose before"}
- "and headed to" → {"and went to", "and walked to", "and made my way to", "and set off for", "and strolled to"}
- "were just beginning" → {"were just starting", "were just commencing"}
- "to sing" → {"to chirp", "to warble", "to call", "to tweet", "to whistle"}
- "I started slowly" → {"I began slowly", "I set off slowly", "I commenced slowly"}
- "then found a" → {"then discovered a", "then settled into a", "then hit a", "then fell into a"}
- "time I finished" → {"time I concluded", "time I wrapped up", "time I stopped", "time I ended", "time I was done"}

Designed traps, all of which must be left untouched:

- the verbs in the preamble and rule sentence ("tell", "replace") — instruction leakage;
- "was", "were", "felt" — excluded verb classes;
- "run" in "my morning run" — a noun here, not a verb.

---

That completes the planned five human episodes, plus this sixth. All six measure the
goal on unseen text. Family-level generalization claims are made by rerunning an
episode against an ablation checkpoint whose training filtered out that episode's
family tag (see musts_finish_before_move_on.md, Evaluation Approach).
