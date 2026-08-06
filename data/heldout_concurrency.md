# Held-out concurrency episodes (frozen eval fixtures)

Frozen layer-2 fixtures for the concurrency behavior. Do not quote these in training
data, datagen templates, or prompts — same rule as the other heldout files. Schedules
are **tick-denominated**: a job completes N user ticks after its action was accepted,
enforced by the rollout runner's scheduled providers, so provider latency and model
latency can never decide an outcome. The runner uses the simulated clock (tick × 650 ms)
so every configuration sees byte-identical timestamps.

Fixture-validity rule: every question must be **search-shaped** — public information a
web lookup could plausibly return (news, schedules, hours, event results, conditions).
Questions are fictional because the provider is simulated; they must never require a
judgment call about whether searching is appropriate. (S2 originally violated this
with an internal-office question and was replaced on 2026-07-18.)

Pass conditions for every episode (window = 4 user ticks unless noted):

- each expected tool fires exactly once, on or after the tick its request settles and
  within the window — firing on a partial utterance is a premature-fire failure;
- no duplicate valid job for the same request (validator-rejected retries are counted
  separately and do not fail the user-visible episode);
- dual episodes: both jobs genuinely outstanding during an overlapping span of events;
- respond: exactly one, only after the search completion event, targeting it, with a
  message grounded in the canned result (shares content tokens with title + snippet);
- failure episodes: the failed job earns no respond and no retry;
- nothing else — no spurious tools, no reaction to acknowledgements.

## D1. dual — search completes first, question arrives immediately (frozen Demo 1 script)

- request: Can you create a visual brief about Vietnam's 2025 economic report for me?
- question: I think Thinking Machine just released a new product. Can you tell me about what it is?
- arrival_idle_ticks: 0
- ui_delay_ticks: 8
- search_delay_ticks: 2
- result_title: Thinking Machine ships a new lab product
- result_snippet: The release landed this week and focuses on developer tooling.

## D2. dual — UI completes first, question arrives late

- request: Would you draw up a visual snapshot of the harbor festival budget?
- question: Also, did the night ferry timetable get updated this week?
- arrival_idle_ticks: 3
- ui_delay_ticks: 7
- search_delay_ticks: 4
- result_title: Night ferry timetable revised
- result_snippet: Departures now run every forty minutes after ten.

## D3. dual — UI completes first, question arrives immediately

- request: make me a little chart panel showing our jiu jitsu club signups.
- question: hey wait, who won the city chess open yesterday?
- arrival_idle_ticks: 0
- ui_delay_ticks: 5
- search_delay_ticks: 5
- result_title: Local teacher takes the city chess open
- result_snippet: The final round ended in a queen sacrifice and resignation.

## D4. dual — search completes first, question arrives late

- request: Please assemble a visual digest of my spring garden log.
- question: Is the flower market open tomorrow morning?
- arrival_idle_ticks: 3
- ui_delay_ticks: 9
- search_delay_ticks: 2
- result_title: Flower market keeps weekend hours
- result_snippet: Stalls open at seven and close by early afternoon.

## S1. search only

- question: What time does the observatory open tonight?
- search_delay_ticks: 3
- result_title: Observatory opens at dusk for the meteor watch
- result_snippet: Doors open at eight with telescopes on the roof terrace.

## S2. search only — casual phrasing, no question mark

- question: is the road up the mountain pass open yet
- search_delay_ticks: 2
- result_title: Mountain pass reopens after rockfall clearing
- result_snippet: Crews reopened both lanes late this morning.

## U1. ui only

- request: Turn my study hours from last month into a small visual card.
- ui_delay_ticks: 4

## U2. ui only — casual phrasing

- request: can you visualize the bake sale totals real quick
- ui_delay_ticks: 3

## F1. search failure — silence after the failed job

- question: When does the tide come in at the north pier?
- search_delay_ticks: 2
- fail: search

## F2. ui failure — silence after the failed job

- request: Give me a visual rundown of the robotics team budget.
- ui_delay_ticks: 3
- fail: ui

## T1. duplicate temptation — nudge must not re-fire the pending job

- request: Build a visual summary of my French vocabulary progress.
- nudge: hello?? did that go through, still want the visual
- ui_delay_ticks: 9

## T2. duplicate temptation — polite nudge

- request: show my cycling distances this month as a chart please
- nudge: you got that right?
- ui_delay_ticks: 8
