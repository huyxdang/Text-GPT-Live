from __future__ import annotations

import json
import unittest

from app.domain import ActionKind, UserState
from app.stream import g1_action_completion, parse_g1_action
from datagen.g1_demo3 import (
    CLAUSE_MARKER as M,
    Demo3Targets,
    closing_shape,
    compile_demo3_dataset,
    compile_demo3_record,
    compile_demo3_records,
    demo3_records_from_batches,
    opening_shape,
    plan_demo3_script,
    render_demo3_card,
    select_demo3_candidates,
)


PROVENANCE = {
    "persona": "journal-keeper",
    "domain": "cooking",
    "register": "casual-reflective",
    "author_slot": "test-author",
    "author_model": "test-model",
    "author_tranche": "1",
    "reference_review_method": "machine",
    "reference_review_reviewer": "test-reviewer",
}


def record_all_traps(index: int = 1) -> dict:
    """One episode exercising every Demo 3 situation class at once."""

    return {
        "id": f"demo3-test-{index:02d}",
        **PROVENANCE,
        "instruction": {
            "text": f"Translate what I type into Chinese as I go, run {index}.",
            "gold_ack": f"好的，我会边打边翻译，第{index}次。",
            "pause_after": "short",
        },
        "script": [
            {
                "kind": "type",
                "text": f"The market was crowded this morning in run {index},{M}",
                "pause_after": "none",
            },
            {
                "kind": "type",
                "text": " and I bought some greeen",
                "trap": "half_clause",
                "pause_after": "short",
            },
            {"kind": "backtrack", "delete": 6, "pause_after": "none"},
            {"kind": "type", "text": f"green tea for my mother.{M}", "pause_after": "medium"},
            {
                "kind": "type",
                "text": " She always says 谢谢 when I hand it over.",
                "trap": "prequoted_chinese",
                "pause_after": "long",
            },
            {"kind": "type", "text": f" I laughed and went back outside.{M}", "pause_after": "short"},
        ],
        "clauses": [
            {
                "english": f"The market was crowded this morning in run {index},",
                "reference": f"第{index}次时，今天早上市场很拥挤，",
            },
            {
                "english": "and I bought some green tea for my mother.",
                "reference": "我给母亲买了一些绿茶。",
            },
            {
                "english": "I laughed and went back outside.",
                "reference": "我笑了笑，又走回外面。",
            },
        ],
    }


class Demo3PlannerTests(unittest.TestCase):
    def test_marker_and_english_span_resolve_to_exact_offsets(self) -> None:
        plan = plan_demo3_script(record_all_traps())
        self.assertEqual(len(plan.clause_spans), 3)
        for span in plan.clause_spans:
            self.assertEqual(span.span.strip(), span.english)
            self.assertEqual(
                plan.final_passage[span.start_offset : span.end_offset], span.span
            )
            # A clause ends on its final character, never on trailing space.
            self.assertEqual(span.span, span.span.rstrip())
        self.assertNotIn(M, plan.final_passage)

    def test_backtracked_text_never_reaches_the_committed_span(self) -> None:
        plan = plan_demo3_script(record_all_traps())
        self.assertNotIn("greeen", plan.final_passage)
        self.assertIn("green tea for my mother.", plan.clause_spans[1].span)

    def test_marker_disagreeing_with_english_is_rejected(self) -> None:
        record = record_all_traps()
        record["clauses"][0]["english"] = "The market was crowded"
        with self.assertRaises(ValueError) as caught:
            plan_demo3_script(record)
        self.assertIn("does not match the marked span", str(caught.exception))

    def test_backtrack_may_not_delete_committed_text(self) -> None:
        record = record_all_traps()
        record["script"][2]["delete"] = 400
        with self.assertRaises(ValueError) as caught:
            plan_demo3_script(record)
        self.assertIn("would delete", str(caught.exception))

    def test_prequoted_chinese_run_must_start_at_a_clause_boundary(self) -> None:
        record = record_all_traps()
        record["script"][4]["trap"] = "none"
        record["script"].insert(
            5,
            {
                "kind": "type",
                "text": " He replied 不客气 right away.",
                "trap": "prequoted_chinese",
                "pause_after": "short",
            },
        )
        with self.assertRaises(ValueError) as caught:
            plan_demo3_script(record)
        self.assertIn("must begin at a clause boundary", str(caught.exception))

    def test_prequoted_run_is_excluded_from_the_next_clause_span(self) -> None:
        plan = plan_demo3_script(record_all_traps())
        self.assertNotIn("谢谢", plan.clause_spans[2].span)
        self.assertIn("谢谢", plan.final_passage)

    def test_short_run_before_a_marker_is_rejected(self) -> None:
        record = record_all_traps()
        record["script"][3]["text"] = f"green.{M}"
        with self.assertRaises(ValueError) as caught:
            plan_demo3_script(record)
        self.assertIn("minimum before a", str(caught.exception))

    def test_marker_count_must_match_clause_count(self) -> None:
        record = record_all_traps()
        record["clauses"].append({"english": "extra", "reference": "多余"})
        with self.assertRaises(ValueError) as caught:
            plan_demo3_script(record)
        self.assertIn("clause markers found", str(caught.exception))

    def test_clause_marker_after_whitespace_is_rejected(self) -> None:
        record = record_all_traps()
        record["script"][5]["text"] = f" I laughed and went back outside. {M}"
        with self.assertRaises(ValueError) as caught:
            plan_demo3_script(record)
        self.assertIn("last character", str(caught.exception))


class Demo3CompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compiled = compile_demo3_record(record_all_traps())

    def test_every_situation_class_is_produced(self) -> None:
        roles = {candidate.role for candidate in self.compiled.candidates}
        self.assertLessEqual(
            {
                "instruction-before",
                "instruction-ack",
                "instruction-after",
                "clause-before",
                "clause-positive",
                "clause-after",
                "backtrack-idle",
                "prequoted-idle",
                "partial-idle",
            },
            roles,
        )

    def test_instruction_ack_is_a_respond_aimed_at_the_instruction_tick(self) -> None:
        ack = next(c for c in self.compiled.candidates if c.role == "instruction-ack")
        turn = self.compiled.turns[ack.turn_offset]
        self.assertIs(turn.action.kind, ActionKind.RESPOND)
        self.assertEqual(turn.action.target, turn.event.index)
        self.assertEqual(turn.event.content, self.compiled.turns[ack.turn_offset].event.content)
        # The ack fires on the tick the instruction completes, not later.
        self.assertTrue(turn.event.content.endswith("as I go, run 1."))

    def test_clause_positive_fires_on_the_exact_commit_tick(self) -> None:
        positives = [c for c in self.compiled.candidates if c.role == "clause-positive"]
        self.assertEqual(len(positives), 3)
        for candidate in sorted(positives, key=lambda item: item.clause_index or 0):
            span = self.compiled.plan.clause_spans[candidate.clause_index]
            turn = self.compiled.turns[candidate.turn_offset]
            self.assertIs(turn.action.kind, ActionKind.RESPOND)
            self.assertEqual(turn.action.message, span.reference)
            self.assertEqual(turn.action.target, turn.event.index)
            # The visible textbox ends exactly at the clause boundary.
            self.assertTrue(turn.event.content.endswith(span.span.rstrip()))
            passage = turn.event.content.split("\n", 1)[1]
            self.assertEqual(len(passage), span.end_offset)

    def test_clause_before_neighbour_is_one_tick_short_and_idles(self) -> None:
        befores = {
            c.clause_index: c
            for c in self.compiled.candidates
            if c.role == "clause-before"
        }
        positives = {
            c.clause_index: c
            for c in self.compiled.candidates
            if c.role == "clause-positive"
        }
        self.assertTrue(befores)
        for clause_index, before in befores.items():
            positive = positives[clause_index]
            self.assertEqual(before.turn_offset + 1, positive.turn_offset)
            before_turn = self.compiled.turns[before.turn_offset]
            self.assertIs(before_turn.action.kind, ActionKind.IDLE)
            self.assertLess(
                len(before_turn.event.content),
                len(self.compiled.turns[positive.turn_offset].event.content),
            )

    def test_traps_are_graded_idles_on_their_own_ticks(self) -> None:
        for role, expectation in (
            ("backtrack-idle", lambda text: "greeen" not in text),
            ("prequoted-idle", lambda text: "谢谢" in text),
            ("partial-idle", lambda text: text.endswith("greeen")),
        ):
            candidate = next(c for c in self.compiled.candidates if c.role == role)
            turn = self.compiled.turns[candidate.turn_offset]
            self.assertIs(turn.action.kind, ActionKind.IDLE, role)
            self.assertTrue(expectation(turn.event.content), role)

    def test_backtrack_tick_shrinks_the_textbox(self) -> None:
        candidate = next(
            c for c in self.compiled.candidates if c.role == "backtrack-idle"
        )
        previous = self.compiled.turns[candidate.turn_offset - 1].event.content
        current = self.compiled.turns[candidate.turn_offset].event.content
        self.assertLess(len(current), len(previous))
        self.assertTrue(previous.startswith(current))

    def test_human_cadence_and_non_uniform_gaps(self) -> None:
        deltas: list[int] = []
        gaps: list[int] = []
        prior = ""
        prior_time = None
        for turn in self.compiled.turns:
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
        self.assertGreaterEqual(max(gaps), 10_000)
        self.assertGreater(len(set(gaps)), 5)

    def test_event_indices_are_contiguous(self) -> None:
        self.assertEqual(
            [turn.event.index for turn in self.compiled.turns],
            list(range(1, len(self.compiled.turns) + 1)),
        )

    def test_exactly_one_candidate_per_turn(self) -> None:
        offsets = [c.turn_offset for c in self.compiled.candidates]
        self.assertEqual(len(offsets), len(set(offsets)))


class Demo3SkeletonTests(unittest.TestCase):
    def test_opening_and_closing_shapes_vary_across_episodes(self) -> None:
        openings = {opening_shape(f"demo3-shape-{index:03d}") for index in range(200)}
        closings = {closing_shape(f"demo3-shape-{index:03d}") for index in range(200)}
        self.assertEqual(len(openings), 4)
        self.assertEqual(len(closings), 4)

    def test_shape_selection_is_deterministic(self) -> None:
        self.assertEqual(opening_shape("demo3-x"), opening_shape("demo3-x"))
        self.assertEqual(closing_shape("demo3-x"), closing_shape("demo3-x"))

    def test_immediate_opening_emits_no_empty_prelude(self) -> None:
        record_id = next(
            f"demo3-shape-{index:03d}"
            for index in range(200)
            if opening_shape(f"demo3-shape-{index:03d}") == "immediate"
        )
        record = record_all_traps()
        record["id"] = record_id
        compiled = compile_demo3_record(record)
        self.assertNotEqual(compiled.turns[0].event.content, "")
        self.assertFalse(
            any(c.role == "empty-initial" for c in compiled.candidates)
        )

    def test_silence_classes_appear_across_the_shape_space(self) -> None:
        kinds: set[str] = set()
        states: set[UserState] = set()
        for index in range(24):
            record = record_all_traps(index)
            record["id"] = f"demo3-shape-{index:03d}"
            compiled = compile_demo3_record(record)
            kinds.update(
                c.empty_kind for c in compiled.candidates if c.empty_kind
            )
            states.update(turn.event.state for turn in compiled.turns)
        self.assertEqual(kinds, {"initial", "unchanged", "cleared"})
        self.assertEqual(states, {UserState.ACTIVE, UserState.IDLE})


class Demo3RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compiled = compile_demo3_record(record_all_traps())

    def test_every_card_round_trips_through_the_serving_parser(self) -> None:
        for candidate in self.compiled.candidates:
            row = render_demo3_card(self.compiled, candidate)
            parsed = parse_g1_action(row["completion"])
            self.assertTrue(parsed.valid, row["candidate_id"])
            self.assertTrue(row["prompt"].endswith("<PREDICT_THIS_ACTION>"))

    def test_chinese_survives_compile_parse_byte_exactly(self) -> None:
        candidate = next(
            c for c in self.compiled.candidates if c.role == "clause-positive"
        )
        row = render_demo3_card(self.compiled, candidate)
        reference = self.compiled.plan.clause_spans[candidate.clause_index].reference
        self.assertIn(reference, row["completion"])
        parsed = parse_g1_action(row["completion"])
        self.assertEqual(parsed.message, reference)
        # No \uXXXX escaping: the literal characters must be present.
        self.assertNotIn("\\u", row["completion"])
        self.assertEqual(
            row["completion"], g1_action_completion(parsed)
        )

    def test_chinese_survives_jsonl_serialization(self) -> None:
        candidate = next(
            c for c in self.compiled.candidates if c.role == "clause-positive"
        )
        row = render_demo3_card(self.compiled, candidate)
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
        restored = json.loads(payload)
        self.assertEqual(restored["completion"], row["completion"])
        self.assertTrue(parse_g1_action(restored["completion"]).valid)
        # ensure_ascii=True must survive too: the bytes differ, the text does not.
        escaped = json.loads(json.dumps(row, ensure_ascii=True))
        self.assertEqual(escaped["completion"], row["completion"])

    def test_action_bytes_have_sorted_json_keys(self) -> None:
        candidate = next(
            c for c in self.compiled.candidates if c.role == "clause-positive"
        )
        row = render_demo3_card(self.compiled, candidate)
        body = row["completion"].removeprefix("<action>").removesuffix("</action>")
        self.assertTrue(body.startswith('respond({"for":'))
        self.assertIn('"message":', body)
        self.assertLess(body.index('"for"'), body.index('"message"'))

    def test_respond_target_is_the_current_event(self) -> None:
        for candidate in self.compiled.candidates:
            row = render_demo3_card(self.compiled, candidate)
            parsed = parse_g1_action(row["completion"])
            if parsed.kind is ActionKind.RESPOND:
                self.assertEqual(parsed.target, row["current_event_index"])

    def test_prompt_history_carries_the_earlier_respond(self) -> None:
        after = next(c for c in self.compiled.candidates if c.role == "clause-after")
        row = render_demo3_card(self.compiled, after)
        reference = self.compiled.plan.clause_spans[after.clause_index].reference
        self.assertIn(reference, row["prompt"])
        self.assertEqual(row["completion"], "<action>idle()</action>")

    def test_rows_carry_reference_review_provenance(self) -> None:
        candidate = self.compiled.candidates[0]
        row = render_demo3_card(self.compiled, candidate)
        self.assertEqual(row["reference_review_method"], "machine")
        self.assertEqual(row["reference_review_reviewer"], "test-reviewer")

    def test_rows_label_clause_boundary_evaluation_state(self) -> None:
        expected = {
            "clause-positive": "complete",
            "clause-before": "partial",
            "partial-idle": "partial",
            "instruction-ack": None,
        }
        for role, clause_state in expected.items():
            candidate = next(c for c in self.compiled.candidates if c.role == role)
            row = render_demo3_card(self.compiled, candidate)
            self.assertEqual(row["clause_state"], clause_state, role)

    def test_missing_provenance_is_refused(self) -> None:
        record = record_all_traps()
        del record["reference_review_method"]
        with self.assertRaises(ValueError):
            compile_demo3_record(record)


class Demo3DeterminismTests(unittest.TestCase):
    def test_same_source_compiles_byte_identically(self) -> None:
        first = compile_demo3_record(record_all_traps())
        second = compile_demo3_record(record_all_traps())
        self.assertEqual(
            [render_demo3_card(first, c) for c in first.candidates],
            [render_demo3_card(second, c) for c in second.candidates],
        )

    def test_utf8_hashing_is_stable_for_non_ascii_ids(self) -> None:
        record = record_all_traps()
        record["id"] = "demo3-记录-01"
        first = compile_demo3_record(record)
        second = compile_demo3_record(record)
        self.assertEqual(first.opening_shape, second.opening_shape)
        self.assertEqual(
            [turn.event.elapsed_ms for turn in first.turns],
            [turn.event.elapsed_ms for turn in second.turns],
        )


def _batch(slot: str, records: list[dict]) -> dict:
    return {
        "schema_version": "g1-authored-demo3-1",
        "demo": "demo-3",
        "author": {"model": "test-model", "slot": slot, "tranche": 1},
        "reference_review": {
            "method": "machine",
            "reviewer": "test-reviewer",
            "reviewed_at": "2026-08-01",
        },
        "records": records,
    }


def _authored(index: int) -> dict:
    record = record_all_traps(index)
    for field in (
        "author_slot",
        "author_model",
        "author_tranche",
        "reference_review_method",
        "reference_review_reviewer",
    ):
        record.pop(field)
    record["persona"] = f"persona-{index % 6}"
    record["domain"] = f"domain-{index % 5}"
    record["register"] = f"register-{index % 4}"
    return record


class Demo3SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.batches = [
            _batch("slot-a", [_authored(index) for index in range(0, 4)]),
            _batch("slot-b", [_authored(index) for index in range(4, 8)]),
            _batch("slot-c", [_authored(index) for index in range(8, 12)]),
        ]
        self.records = demo3_records_from_batches(self.batches)

    def test_flattening_attaches_author_and_review_provenance(self) -> None:
        self.assertEqual(len(self.records), 12)
        self.assertEqual(self.records[0]["author_slot"], "slot-a")
        self.assertEqual(self.records[0]["reference_review_method"], "machine")

    def test_selection_hits_the_exact_card_target(self) -> None:
        compiled = compile_demo3_records(self.records, dev_fraction=0.25)
        targets = Demo3Targets(
            episodes=12,
            clauses=36,
            cards=120,
            empty_per_kind=2,
            min_backtrack_idles=2,
            min_prequoted_idles=2,
            min_partial_idles=2,
        )
        selected = select_demo3_candidates(compiled, targets=targets)
        self.assertEqual(len(selected), 120)
        roles = {candidate.role for candidate in selected}
        self.assertIn("clause-positive", roles)
        self.assertEqual(
            sum(1 for c in selected if c.role == "clause-positive"), 36
        )
        self.assertEqual(
            sum(1 for c in selected if c.role == "instruction-ack"), 12
        )
        for kind in ("initial", "unchanged", "cleared"):
            self.assertEqual(sum(1 for c in selected if c.empty_kind == kind), 2)

    def test_wrong_source_size_is_a_hard_failure(self) -> None:
        compiled = compile_demo3_records(self.records)
        with self.assertRaises(ValueError) as caught:
            select_demo3_candidates(
                compiled, targets=Demo3Targets(episodes=99, clauses=36, cards=120)
            )
        self.assertIn("99 episodes", str(caught.exception))

    def test_unreachable_card_target_is_diagnosed(self) -> None:
        compiled = compile_demo3_records(self.records)
        with self.assertRaises(ValueError) as caught:
            select_demo3_candidates(
                compiled,
                targets=Demo3Targets(
                    episodes=12,
                    clauses=36,
                    cards=10_000,
                    empty_per_kind=2,
                    min_backtrack_idles=2,
                    min_prequoted_idles=2,
                    min_partial_idles=2,
                ),
            )
        self.assertIn("neighbour cards", str(caught.exception))

    def test_dataset_build_is_deterministic_and_parseable(self) -> None:
        targets = Demo3Targets(
            episodes=12,
            clauses=36,
            cards=120,
            empty_per_kind=2,
            min_backtrack_idles=2,
            min_prequoted_idles=2,
            min_partial_idles=2,
        )
        first = compile_demo3_dataset(self.records, targets=targets, dev_fraction=0.25)
        second = compile_demo3_dataset(self.records, targets=targets, dev_fraction=0.25)
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(len(first.rows), 120)
        self.assertEqual(len({row["prompt"] for row in first.rows}), 120)
        for row in first.rows:
            self.assertTrue(parse_g1_action(row["completion"]).valid)
        self.assertEqual(
            set(first.coverage["selected_empty_kinds"]),
            {"initial", "unchanged", "cleared"},
        )
        splits = first.coverage["selected_splits"]
        self.assertEqual(set(splits), {"train", "dev"})

    def test_production_scale_source_reaches_exactly_1300_cards(self) -> None:
        """The headline number must be reachable, not merely declared.

        100 episodes x ~5 clauses, with each trap class present in roughly 45%
        of episodes, is the shape the authoring fleet is being asked for.
        """

        from tests.test_g1_authored_demo3 import _record as authored_record

        records = []
        for index in range(100):
            record = authored_record(index)
            if index % 20 >= 11:
                record["script"].append(
                    {
                        "kind": "type",
                        "text": f" A card near it read 欢迎 on day {index}.",
                        "trap": "prequoted_chinese",
                        "pause_after": "short",
                    }
                )
            if index % 20 >= 13:
                record["script"].insert(
                    1, {"kind": "type", "text": " a stray phrase", "pause_after": "none"}
                )
                record["script"].insert(
                    2,
                    {"kind": "backtrack", "delete": 15, "pause_after": "short"},
                )
            if index % 20 >= 8:
                first = record["script"][0]["text"]
                cut = len(first) // 2
                record["script"][0] = {
                    "kind": "type",
                    "text": first[:cut],
                    "trap": "half_clause",
                    "pause_after": "medium",
                }
                record["script"].insert(
                    1, {"kind": "type", "text": first[cut:], "pause_after": "short"}
                )
            record.update(
                author_slot=f"slot-{index % 3}",
                author_model="test-model",
                author_tranche="1",
                reference_review_method="machine",
                reference_review_reviewer="test-reviewer",
            )
            records.append(record)

        compiled = compile_demo3_records(records)
        clauses = sum(len(record.plan.clause_spans) for record in compiled)
        targets = Demo3Targets(episodes=100, clauses=clauses)
        selected = select_demo3_candidates(compiled, targets=targets)
        self.assertEqual(len(selected), 1_300)

        counts: dict[str, int] = {}
        for candidate in selected:
            counts[candidate.role] = counts.get(candidate.role, 0) + 1
        actions = counts["clause-positive"] + counts["instruction-ack"]
        traps = sum(
            counts.get(role, 0)
            for role in ("backtrack-idle", "prequoted-idle", "partial-idle")
        )
        silence = sum(
            1 for candidate in selected if candidate.empty_kind is not None
        )
        neighbours = 1_300 - actions - traps - silence
        self.assertEqual(actions + traps + silence + neighbours, 1_300)
        self.assertEqual(counts["clause-positive"], clauses)
        self.assertEqual(counts["instruction-ack"], 100)
        self.assertEqual(silence, 120)
        # ~46% graded actions, the rest hard idles, silence and neighbours.
        self.assertGreater(actions / 1_300, 0.40)
        self.assertLess(actions / 1_300, 0.50)
        for role in ("backtrack-idle", "prequoted-idle", "partial-idle"):
            self.assertGreaterEqual(counts[role], 40, role)

    def test_default_targets_reach_1300_cards(self) -> None:
        targets = Demo3Targets()
        self.assertEqual(targets.cards, 1_300)
        self.assertEqual(targets.clauses, 532)
        self.assertEqual(targets.episodes, 102)
        mandatory = (
            targets.clauses
            + targets.episodes
            + len(("initial", "unchanged", "cleared")) * targets.empty_per_kind
        )
        # 532 positives + 102 acks + 120 empties = 754 before trap idles, which
        # leaves 546 slots shared by the three trap classes and neighbours.
        self.assertEqual(mandatory, 754)
        self.assertEqual(targets.cards - mandatory, 546)


if __name__ == "__main__":
    unittest.main()
