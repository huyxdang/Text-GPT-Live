# g1 authored-source contract

These JSON files are source situations, not training cards. Deterministic code
validates them and later compiles them into full-textbox stream snapshots and
canonical g1 actions.

## Global reservations

- Held-out personas: `product-reviewer`, `letter-to-a-friend`,
  `technical-writeup`
- Held-out domains: `sport`, `health`, `personal-finance`
- Never include action markup or g1 action syntax in authored text.

## Demo 1

Each record is one coherent mini-scene with 4–8 narration segments and exactly
one direct address plus its natural `gold_reply`. The address must be answerable
from text already visible at that point. Later narration may elaborate, but it
must not supply facts required by the earlier reply.

Trap labels describe why a completed narration segment is a graded hard idle:

- `quoted_question`: a literal question inside quotation marks
- `reported_speech`: another person's speech, request, proposal, or statement
  is being recounted rather than addressed to the assistant
- `rhetorical_question`: a question used as narration, not a request for an
  assistant answer
- `descriptive_command_word`: an action-looking word such as `search`,
  `correct`, `remind`, or `highlight` is mentioned descriptively
- `elaborating_question`: a complete question is immediately followed by the
  author's continued elaboration, such as `Could X? I am checking Y.`
- `none` or `[]`: no trap is present

Validation intentionally checks more than exact duplicates. A tranche fails on
collapsed sentence openings, address/reply openings, speech acts, trap patterns,
trigger positions, or length buckets. A deterministic pass is necessary but not
sufficient: another agent must also review chronology, coherence, reply quality,
trap completeness, and hidden interaction templates record by record.

## Exact Demo 1 production schedule

The accepted source target is 700 records, 4,000 narration passages, and 700
address moments across three author slots.

- Tranche 1: 30 records and 180 narrations per slot.
- Tranches 2–3: two 30-record tranches per slot; each has ten records with five
  narrations and twenty with six (170 narrations).
- Tranche 4: 30 records / 170 narrations per slot, rebalanced to three records
  with four narrations, seven with five, seventeen with six, and three with
  seven. This keeps the cumulative event-count distribution under its 60% cap.
- Tranches 5–7: three 30-record tranches per slot; each has fourteen records
  with five narrations, eleven with six, and five with seven (171 narrations).
- Final tranche: Sol-A writes 20 records / 112 narrations (10×5, 8×6, 2×7);
  Sol-B writes the same; Terra-C writes 30 / 167 (14×5, 15×6, 1×7).

This yields 230 Sol-A records, 230 Sol-B records, 240 Terra-C records, exactly
4,000 narration passages, and exactly 700 direct-address moments.
