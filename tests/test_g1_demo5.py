from __future__ import annotations

import json
import unittest

from app.domain import ActionKind, UserState
from app.stream import g1_action_completion, parse_g1_action
from datagen.g1_demo5 import (
    ALIGNMENTS,
    CancellationEntry,
    Demo5Bank,
    Demo5Targets,
    FillerEntry,
    RequestEntry,
    closing_shape,
    compile_demo5_dataset,
    compile_demo5_schedule,
    compile_demo5_schedules,
    demo5_bank_from_batches,
    derive_expected_class,
    opening_shape,
    plan_demo5_schedule,
    render_demo5_card,
    select_demo5_candidates,
    verify_fire_timing,
)


PROVENANCE = (
    ("persona", "journal-keeper"),
    ("domain", "cooking"),
    ("register", "casual-reflective"),
    ("author_slot", "test-author"),
    ("author_model", "test-model"),
    ("author_tranche", "1"),
)


def make_bank() -> Demo5Bank:
    requests = (
        RequestEntry(
            "r-every",
            "every",
            "Remind me every {interval} seconds to drink water!",
            "Got it — I'll remind you to drink water every {interval} seconds!",
            "Drink water!",
            PROVENANCE,
        ),
        RequestEntry(
            "r-once",
            "once",
            "Remind me once to check the oven in a little while!",
            "Got it — I'll remind you to check the oven once.",
            "Check the oven!",
            PROVENANCE,
        ),
    )
    cancellations = (
        CancellationEntry(
            "c-1",
            "Okay, you can stop reminding me about the water now, thanks.",
            "Got it — I'll stop the water reminders.",
            PROVENANCE,
        ),
    )
    fillers_plain = (
        FillerEntry(
            "f-1",
            "Okay, drafting the email now. Hi Mai, about the venue, I think we should book the smaller room.",
            "none",
            None,
            PROVENANCE,
        ),
        FillerEntry(
            "f-2",
            "Back to the report, I need to check last quarter numbers again before sending it out this afternoon.",
            "none",
            None,
            PROVENANCE,
        ),
    )
    fillers_bait = (
        FillerEntry(
            "fb-1",
            'So my aunt calls me and asks, "can you keep a secret?" and I just laughed and kept going.',
            "bait",
            None,
            PROVENANCE,
        ),
    )
    fillers_address = (
        FillerEntry(
            "fa-1",
            "hey, are you still with me on this?",
            "address",
            "Still here — go ahead!",
            PROVENANCE,
        ),
    )
    return Demo5Bank(requests, cancellations, fillers_plain, fillers_bait, fillers_address)


class Demo5PlannerTests(unittest.TestCase):
    def test_plan_is_deterministic(self) -> None:
        bank = make_bank()
        first = plan_demo5_schedule("demo5-x", bank)
        second = plan_demo5_schedule("demo5-x", bank)
        self.assertEqual(first, second)

    def test_every_kind_gets_an_interval_and_once_does_not(self) -> None:
        bank = make_bank()
        seen_kinds = set()
        for index in range(60):
            config = plan_demo5_schedule(f"demo5-plan-{index:03d}", bank)
            seen_kinds.add(config.schedule_kind)
            if config.schedule_kind == "every":
                self.assertIsNotNone(config.interval_s)
                self.assertIn(config.interval_s, range(1, 31))
                self.assertEqual(config.once_check_interval_s, None)
            else:
                self.assertIsNone(config.interval_s)
                self.assertEqual(config.fire_cycles, 1)
                self.assertIsNotNone(config.once_check_interval_s)
        self.assertEqual(seen_kinds, {"once", "every"})

    def test_collision_never_chosen_alongside_a_cancellation(self) -> None:
        bank = make_bank()
        for index in range(120):
            config = plan_demo5_schedule(f"demo5-collide-{index:03d}", bank)
            if config.cancel_variant != "none":
                self.assertEqual(config.collision, "none")


class Demo5CompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = make_bank()

    def _compiled(self, schedule_id: str):
        config = plan_demo5_schedule(schedule_id, self.bank)
        return compile_demo5_schedule(config)

    def test_every_role_appears_across_enough_schedules(self) -> None:
        roles: set[str] = set()
        for index in range(150):
            compiled = self._compiled(f"demo5-roles-{index:03d}")
            roles.update(c.role for c in compiled.candidates)
        self.assertEqual(
            roles,
            {
                "request-before",
                "request-ack",
                "request-after",
                "fire-before",
                "fire-typing",
                "fire-silent",
                "fire-after",
                "silence-idle",
                "cancel-before",
                "cancel-ack",
                "post-cancel-idle",
                "once-no-repeat",
                "bait-idle",
                "address-positive",
                "empty-initial",
                "empty-unchanged",
                "empty-cleared",
            },
        )

    def test_all_three_alignments_appear(self) -> None:
        seen: set[str] = set()
        for index in range(150):
            compiled = self._compiled(f"demo5-align-{index:03d}")
            seen.update(c.alignment for c in compiled.candidates if c.alignment)
        self.assertEqual(seen, set(ALIGNMENTS))

    def test_request_ack_is_a_respond_aimed_at_the_completing_tick(self) -> None:
        compiled = self._compiled("demo5-ack-01")
        ack = next(c for c in compiled.candidates if c.role == "request-ack")
        turn = compiled.turns[ack.turn_offset]
        self.assertIs(turn.action.kind, ActionKind.RESPOND)
        self.assertEqual(turn.action.target, turn.event.index)
        self.assertEqual(turn.event.content, compiled.request_text)

    def test_request_before_is_one_tick_short_and_idles(self) -> None:
        compiled = self._compiled("demo5-ack-01")
        before = next(c for c in compiled.candidates if c.role == "request-before")
        ack = next(c for c in compiled.candidates if c.role == "request-ack")
        self.assertEqual(before.turn_offset + 1, ack.turn_offset)
        turn = compiled.turns[before.turn_offset]
        self.assertIs(turn.action.kind, ActionKind.IDLE)
        self.assertLess(len(turn.event.content), len(compiled.turns[ack.turn_offset].event.content))

    def test_fire_message_matches_request_and_target_is_the_current_event(self) -> None:
        found = 0
        for index in range(60):
            compiled = self._compiled(f"demo5-fire-{index:03d}")
            for candidate in compiled.candidates:
                if candidate.role in ("fire-typing", "fire-silent"):
                    turn = compiled.turns[candidate.turn_offset]
                    self.assertIs(turn.action.kind, ActionKind.RESPOND)
                    self.assertEqual(turn.action.message, compiled.config.request.fire_message)
                    self.assertEqual(turn.action.target, turn.event.index)
                    found += 1
        self.assertGreater(found, 0)

    def test_fire_before_is_idle_and_earlier_than_the_fire(self) -> None:
        found = 0
        for index in range(60):
            compiled = self._compiled(f"demo5-fb-{index:03d}")
            by_cycle_before = {c.fire_index: c for c in compiled.candidates if c.role == "fire-before"}
            by_cycle_fire = {
                c.fire_index: c for c in compiled.candidates if c.role in ("fire-typing", "fire-silent")
            }
            for cycle, before in by_cycle_before.items():
                fire = by_cycle_fire[cycle]
                self.assertLess(before.turn_offset, fire.turn_offset)
                before_turn = compiled.turns[before.turn_offset]
                fire_turn = compiled.turns[fire.turn_offset]
                self.assertIs(before_turn.action.kind, ActionKind.IDLE)
                self.assertLess(before_turn.event.elapsed_ms, fire_turn.event.elapsed_ms)
                found += 1
        self.assertGreater(found, 0)

    def test_silent_fire_has_empty_content_and_idle_state(self) -> None:
        found = 0
        for index in range(60):
            compiled = self._compiled(f"demo5-silent-{index:03d}")
            for candidate in compiled.candidates:
                if candidate.role == "fire-silent":
                    turn = compiled.turns[candidate.turn_offset]
                    self.assertEqual(turn.event.content, "")
                    self.assertEqual(turn.event.state, UserState.IDLE)
                    found += 1
        self.assertGreater(found, 0)

    def test_nothing_fires_after_a_cancellation(self) -> None:
        found = 0
        for index in range(120):
            compiled = self._compiled(f"demo5-cancel-{index:03d}")
            if compiled.config.cancel_variant == "none":
                continue
            cancel_ack = next((c for c in compiled.candidates if c.role == "cancel-ack"), None)
            if cancel_ack is None:
                continue
            found += 1
            cancel_offset = cancel_ack.turn_offset
            for turn in compiled.turns[cancel_offset + 1 :]:
                self.assertIsNot(turn.action.kind, ActionKind.RESPOND)
            post = next(c for c in compiled.candidates if c.role == "post-cancel-idle")
            self.assertIs(compiled.turns[post.turn_offset].action.kind, ActionKind.IDLE)
        self.assertGreater(found, 0)

    def test_once_kind_never_fires_twice(self) -> None:
        found = 0
        for index in range(60):
            compiled = self._compiled(f"demo5-once-{index:03d}")
            if compiled.config.schedule_kind != "once":
                continue
            found += 1
            fire_count = sum(
                1
                for turn in compiled.turns
                if turn.action.kind is ActionKind.RESPOND
                and turn.action.message == compiled.config.request.fire_message
            )
            self.assertEqual(fire_count, 1)
            no_repeat = next(c for c in compiled.candidates if c.role == "once-no-repeat")
            self.assertIs(compiled.turns[no_repeat.turn_offset].action.kind, ActionKind.IDLE)
        self.assertGreater(found, 0)

    def test_human_cadence_and_non_uniform_gaps(self) -> None:
        deltas: list[int] = []
        gaps: list[int] = []
        for index in range(20):
            compiled = self._compiled(f"demo5-cadence-{index:03d}")
            prior = ""
            prior_time = None
            for turn in compiled.turns:
                content = turn.event.content
                if content and content.startswith(prior) and len(content) > len(prior):
                    deltas.append(len(content) - len(prior))
                if prior_time is not None:
                    gaps.append(turn.event.elapsed_ms - prior_time)
                prior = content
                prior_time = turn.event.elapsed_ms
        self.assertTrue(deltas)
        self.assertGreaterEqual(min(deltas), 4)
        self.assertLessEqual(max(deltas), 7)
        self.assertGreaterEqual(max(gaps), 1_000)
        self.assertGreater(len(set(gaps)), 10)

    def test_event_indices_are_contiguous(self) -> None:
        compiled = self._compiled("demo5-contig-01")
        self.assertEqual(
            [turn.event.index for turn in compiled.turns],
            list(range(1, len(compiled.turns) + 1)),
        )

    def test_exactly_one_candidate_per_turn(self) -> None:
        compiled = self._compiled("demo5-dedupe-01")
        offsets = [c.turn_offset for c in compiled.candidates]
        self.assertEqual(len(offsets), len(set(offsets)))


class Demo5SkeletonTests(unittest.TestCase):
    def test_opening_and_closing_shapes_vary(self) -> None:
        openings = {opening_shape(f"demo5-shape-{i:03d}") for i in range(200)}
        closings = {closing_shape(f"demo5-shape-{i:03d}") for i in range(200)}
        self.assertEqual(len(openings), 4)
        self.assertEqual(len(closings), 4)

    def test_shape_selection_is_deterministic(self) -> None:
        self.assertEqual(opening_shape("demo5-x"), opening_shape("demo5-x"))
        self.assertEqual(closing_shape("demo5-x"), closing_shape("demo5-x"))


class Demo5TimingVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = make_bank()

    def test_worked_example_matches_the_spec(self) -> None:
        """Reproduces synthetic_data_spec.md's own worked arithmetic exactly:

        last fire at t+7600ms, current tick t+12700ms, interval 5s ->
        5.1s >= 5s -> fire, aimed at the current (empty, idle) tick.
        """

        prompt = (
            '<stream_event index="1" source="user" state="active" time="t+2600ms">'
            "Remind me every 5 seconds to drink water!</stream_event>\n"
            '<action>respond({"for":1,"message":"Got it — I\'ll remind you to drink water '
            'every 5 seconds!"})</action>\n'
            '<stream_event index="2" source="user" state="active" time="t+4500ms">'
            "…Okay, drafting the email now. Hi Mai, about</stream_event>\n"
            "<action>idle()</action>\n"
            '<stream_event index="3" source="user" state="active" time="t+7600ms">'
            "…about the venue, I think we</stream_event>\n"
            '<action>respond({"for":3,"message":"Drink water!"})</action>\n'
            '<stream_event index="4" source="user" state="active" time="t+9800ms">'
            "…we should book the smaller room</stream_event>\n"
            "<action>idle()</action>\n"
            '<stream_event index="5" source="user" state="idle" time="t+12700ms"></stream_event>\n'
            "<PREDICT_THIS_ACTION>"
        )
        expected = derive_expected_class(
            prompt,
            schedule_kind="every",
            interval_s=5,
            fire_message="Drink water!",
            cancel_ack_text=None,
        )
        self.assertEqual(expected, "fire")

    def test_before_the_interval_elapses_is_idle(self) -> None:
        prompt = (
            '<stream_event index="1" source="user" state="active" time="t+2600ms">'
            "Remind me every 5 seconds to drink water!</stream_event>\n"
            '<action>respond({"for":1,"message":"ack"})</action>\n'
            '<stream_event index="2" source="user" state="active" time="t+6000ms">'
            "…still typing</stream_event>\n"
            "<PREDICT_THIS_ACTION>"
        )
        expected = derive_expected_class(
            prompt, schedule_kind="every", interval_s=5, fire_message="Drink water!", cancel_ack_text=None
        )
        self.assertEqual(expected, "idle")

    def test_cancellation_forces_idle_regardless_of_elapsed_time(self) -> None:
        prompt = (
            '<stream_event index="1" source="user" state="active" time="t+2600ms">'
            "Remind me every 5 seconds to drink water!</stream_event>\n"
            '<action>respond({"for":1,"message":"ack"})</action>\n'
            '<stream_event index="2" source="user" state="active" time="t+9000ms">'
            "stop reminding me</stream_event>\n"
            '<action>respond({"for":2,"message":"cancel-ack"})</action>\n'
            '<stream_event index="3" source="user" state="idle" time="t+90000ms"></stream_event>\n'
            "<PREDICT_THIS_ACTION>"
        )
        expected = derive_expected_class(
            prompt,
            schedule_kind="every",
            interval_s=5,
            fire_message="Drink water!",
            cancel_ack_text="cancel-ack",
        )
        self.assertEqual(expected, "idle")

    def test_pre_fire_once_kind_is_unverifiable(self) -> None:
        prompt = (
            '<stream_event index="1" source="user" state="active" time="t+2600ms">'
            "Remind me once to check the oven!</stream_event>\n"
            '<action>respond({"for":1,"message":"ack"})</action>\n'
            '<stream_event index="2" source="user" state="idle" time="t+9000ms"></stream_event>\n'
            "<PREDICT_THIS_ACTION>"
        )
        expected = derive_expected_class(
            prompt, schedule_kind="once", interval_s=None, fire_message="Check the oven!", cancel_ack_text=None
        )
        self.assertEqual(expected, "unverifiable")

    def test_post_fire_once_kind_never_refires(self) -> None:
        prompt = (
            '<stream_event index="1" source="user" state="active" time="t+2600ms">'
            "Remind me once to check the oven!</stream_event>\n"
            '<action>respond({"for":1,"message":"ack"})</action>\n'
            '<stream_event index="2" source="user" state="idle" time="t+9000ms"></stream_event>\n'
            '<action>respond({"for":2,"message":"Check the oven!"})</action>\n'
            '<stream_event index="3" source="user" state="idle" time="t+90000ms"></stream_event>\n'
            "<PREDICT_THIS_ACTION>"
        )
        expected = derive_expected_class(
            prompt, schedule_kind="once", interval_s=None, fire_message="Check the oven!", cancel_ack_text=None
        )
        self.assertEqual(expected, "idle")

    def test_every_rendered_row_passes_its_own_independent_check(self) -> None:
        checked = 0
        for index in range(40):
            config = plan_demo5_schedule(f"demo5-verify-{index:03d}", self.bank)
            compiled = compile_demo5_schedule(config)
            for candidate in compiled.candidates:
                row = render_demo5_card(compiled, candidate)
                error = verify_fire_timing(row)
                self.assertNotEqual(bool(error), True, error)
                if error is not None:
                    checked += 1
        self.assertGreater(checked, 0)


class Demo5RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = make_bank()
        config = plan_demo5_schedule("demo5-render-01", self.bank)
        self.compiled = compile_demo5_schedule(config)

    def test_every_card_round_trips_through_the_serving_parser(self) -> None:
        for candidate in self.compiled.candidates:
            row = render_demo5_card(self.compiled, candidate)
            parsed = parse_g1_action(row["completion"])
            self.assertTrue(parsed.valid, row["candidate_id"])
            self.assertTrue(row["prompt"].endswith("<PREDICT_THIS_ACTION>"))

    def test_row_carries_reminder_provenance(self) -> None:
        candidate = next(c for c in self.compiled.candidates if c.role in ("fire-typing", "fire-silent"))
        row = render_demo5_card(self.compiled, candidate)
        self.assertEqual(row["source_persona"], "journal-keeper")
        self.assertEqual(row["fire_message"], self.compiled.config.request.fire_message)
        self.assertEqual(row["schedule_kind"], self.compiled.config.schedule_kind)

    def test_rows_label_reminder_evaluation_state(self) -> None:
        fire = next(
            c for c in self.compiled.candidates if c.role in ("fire-typing", "fire-silent")
        )
        fire_row = render_demo5_card(self.compiled, fire)
        self.assertTrue(fire_row["should_fire"])
        self.assertEqual(fire_row["reminder_eval_kind"], "fire")
        self.assertEqual(fire_row["timing_boundary"], "at")

        before = next(c for c in self.compiled.candidates if c.role == "fire-before")
        before_row = render_demo5_card(self.compiled, before)
        self.assertFalse(before_row["should_fire"])
        self.assertEqual(before_row["reminder_eval_kind"], "wait")
        self.assertEqual(before_row["timing_boundary"], "before")

        no_repeat = next(
            (c for c in self.compiled.candidates if c.role == "once-no-repeat"),
            None,
        )
        if no_repeat is not None:
            no_repeat_row = render_demo5_card(self.compiled, no_repeat)
            self.assertFalse(no_repeat_row["should_fire"])
            self.assertEqual(no_repeat_row["reminder_eval_kind"], "wait")
            self.assertEqual(no_repeat_row["timing_boundary"], "already-fired")

    def test_row_survives_jsonl_serialization(self) -> None:
        candidate = self.compiled.candidates[0]
        row = render_demo5_card(self.compiled, candidate)
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
        restored = json.loads(payload)
        self.assertEqual(restored["completion"], row["completion"])
        self.assertTrue(parse_g1_action(restored["completion"]).valid)

    def test_action_bytes_have_sorted_json_keys(self) -> None:
        candidate = next(c for c in self.compiled.candidates if c.role in ("fire-typing", "fire-silent"))
        row = render_demo5_card(self.compiled, candidate)
        body = row["completion"].removeprefix("<action>").removesuffix("</action>")
        self.assertTrue(body.startswith('respond({"for":'))
        self.assertLess(body.index('"for"'), body.index('"message"'))

    def test_missing_provenance_is_refused(self) -> None:
        bad_request = RequestEntry(
            "r-bad",
            "every",
            "Remind me every {interval} seconds to drink water!",
            "Got it — I'll remind you every {interval} seconds!",
            "Drink water!",
            (),
        )
        bank = Demo5Bank((bad_request,), self.bank.cancellations, self.bank.fillers_plain, (), ())
        config = plan_demo5_schedule("demo5-badprov", bank)
        with self.assertRaises(ValueError):
            compile_demo5_schedule(config)


class Demo5DeterminismTests(unittest.TestCase):
    def test_same_source_compiles_byte_identically(self) -> None:
        bank = make_bank()
        config1 = plan_demo5_schedule("demo5-det-01", bank)
        config2 = plan_demo5_schedule("demo5-det-01", bank)
        first = compile_demo5_schedule(config1)
        second = compile_demo5_schedule(config2)
        self.assertEqual(
            [render_demo5_card(first, c) for c in first.candidates],
            [render_demo5_card(second, c) for c in second.candidates],
        )

    def test_utf8_hashing_is_stable_for_non_ascii_ids(self) -> None:
        bank = make_bank()
        config = plan_demo5_schedule("demo5-记录-01", bank)
        first = compile_demo5_schedule(config)
        second = compile_demo5_schedule(config)
        self.assertEqual(first.opening_shape, second.opening_shape)
        self.assertEqual(
            [turn.event.elapsed_ms for turn in first.turns],
            [turn.event.elapsed_ms for turn in second.turns],
        )


def _batch(slot: str, index: int) -> dict:
    return {
        "schema_version": "g1-authored-demo5-1",
        "demo": "demo-5",
        "author": {"model": "test-model", "slot": slot, "tranche": 1},
        "bank": [
            {
                "id": f"{slot}-req-every-{index}",
                "kind": "request",
                "schedule_kind": "every",
                "text_template": f"Remind me every {{interval}} seconds to do task {index}!",
                "gold_ack_template": f"Got it — task {index} every {{interval}} seconds!",
                "fire_message": f"Do task {index}!",
                "persona": f"persona-{index % 6}",
                "domain": f"domain-{index % 5}",
                "register": f"register-{index % 4}",
            },
            {
                "id": f"{slot}-req-once-{index}",
                "kind": "request",
                "schedule_kind": "once",
                "text_template": f"Remind me once to do a special task number {index} today!",
                "gold_ack_template": f"Got it — one reminder for task {index}.",
                "fire_message": f"Do special task {index}!",
                "persona": f"persona-{index % 6}",
                "domain": f"domain-{index % 5}",
                "register": f"register-{index % 4}",
            },
            {
                "id": f"{slot}-cancel-{index}",
                "kind": "cancellation",
                "text": f"Okay, you can stop reminding me about task {index} now, please.",
                "gold_ack": f"Understood — task {index} reminders are off.",
                "persona": f"persona-{index % 6}",
                "domain": f"domain-{index % 5}",
                "register": f"register-{index % 4}",
            },
            {
                "id": f"{slot}-filler-{index}",
                "kind": "filler",
                "trap": "none",
                "text": f"Back to drafting notes about project {index}, there is quite a lot left to cover today.",
                "persona": f"persona-{index % 6}",
                "domain": f"domain-{index % 5}",
                "register": f"register-{index % 4}",
            },
            {
                "id": f"{slot}-bait-{index}",
                "kind": "filler",
                "trap": "bait",
                "text": f'Someone asked me, "are you done with task {index} yet?" and I just shrugged.',
                "persona": f"persona-{index % 6}",
                "domain": f"domain-{index % 5}",
                "register": f"register-{index % 4}",
            },
            {
                "id": f"{slot}-address-{index}",
                "kind": "filler",
                "trap": "address",
                "text": f"hey, are you keeping track of task {index} for me?",
                "gold_reply": f"Yes, tracking task {index}!",
                "persona": f"persona-{index % 6}",
                "domain": f"domain-{index % 5}",
                "register": f"register-{index % 4}",
            },
        ],
    }


class Demo5SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.batches = [_batch(f"slot-{slot}", index) for slot, index in zip("abc" * 3, range(9))]
        self.bank = demo5_bank_from_batches(self.batches)

    def test_bank_flattening_attaches_provenance(self) -> None:
        self.assertGreater(len(self.bank.requests), 0)
        self.assertEqual(dict(self.bank.requests[0].provenance)["author_slot"], "slot-a")

    def test_selection_hits_the_exact_card_target(self) -> None:
        schedule_ids = [f"demo5-sel-{i:04d}" for i in range(40)]
        configs = [plan_demo5_schedule(schedule_id, self.bank) for schedule_id in schedule_ids]
        compiled = compile_demo5_schedules(configs, dev_fraction=0.25)
        from collections import Counter

        counts: Counter[str] = Counter()
        for schedule in compiled:
            for candidate in schedule.candidates:
                counts[candidate.role] += 1
        fires = counts["fire-typing"] + counts["fire-silent"]
        # Trap-role and address-positive pools are selected in full (not just
        # down to their floor), mirroring Demo 3's "fix absolute counts of
        # rare actions" selector -- so the mandatory total includes every
        # trap-pool candidate actually generated, not the floor minimums.
        mandatory = (
            fires
            + 40
            + counts["post-cancel-idle"]
            + counts["once-no-repeat"]
            + counts["bait-idle"]
            + counts["silence-idle"]
            + counts["address-positive"]
            + 3
        )
        targets = Demo5Targets(
            schedules=40,
            fires=fires,
            cards=mandatory + 20,
            empty_per_kind=1,
            min_post_cancel_idle=1,
            min_once_no_repeat=1,
            min_bait_idle=1,
            min_address_positive=1,
            min_silence_idle=1,
        )
        selected = select_demo5_candidates(compiled, targets=targets)
        self.assertEqual(len(selected), targets.cards)
        self.assertEqual(sum(1 for c in selected if c.role in ("fire-typing", "fire-silent")), fires)
        self.assertEqual(sum(1 for c in selected if c.role == "request-ack"), 40)

    def test_wrong_source_size_is_a_hard_failure(self) -> None:
        configs = [plan_demo5_schedule(f"demo5-wrong-{i:03d}", self.bank) for i in range(10)]
        compiled = compile_demo5_schedules(configs)
        with self.assertRaises(ValueError) as caught:
            select_demo5_candidates(compiled, targets=Demo5Targets(schedules=99, fires=1, cards=1))
        self.assertIn("99 schedules", str(caught.exception))

    def test_dataset_build_is_deterministic_and_parseable(self) -> None:
        schedule_ids = [f"demo5-build-{i:04d}" for i in range(30)]
        configs = [plan_demo5_schedule(schedule_id, self.bank) for schedule_id in schedule_ids]
        compiled_once = compile_demo5_schedules(configs, dev_fraction=0.2)
        from collections import Counter

        counts: Counter[str] = Counter()
        for schedule in compiled_once:
            for candidate in schedule.candidates:
                counts[candidate.role] += 1
        fires = counts["fire-typing"] + counts["fire-silent"]
        mandatory = (
            fires
            + 30
            + counts["post-cancel-idle"]
            + counts["once-no-repeat"]
            + counts["bait-idle"]
            + counts["silence-idle"]
            + counts["address-positive"]
            + 3
        )
        targets = Demo5Targets(
            schedules=30,
            fires=fires,
            cards=mandatory + 20,
            empty_per_kind=1,
            min_post_cancel_idle=1,
            min_once_no_repeat=1,
            min_bait_idle=1,
            min_address_positive=1,
            min_silence_idle=1,
        )
        first = compile_demo5_dataset(configs, targets=targets, dev_fraction=0.2)
        second = compile_demo5_dataset(configs, targets=targets, dev_fraction=0.2)
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(len(first.rows), targets.cards)
        self.assertEqual(len({row["prompt"] for row in first.rows}), targets.cards)
        for row in first.rows:
            self.assertTrue(parse_g1_action(row["completion"]).valid)
        splits = first.coverage["selected_splits"]
        self.assertTrue(set(splits) <= {"train", "dev"})


if __name__ == "__main__":
    unittest.main()
