from __future__ import annotations

import re
import unittest

from app.domain import ActionKind, UserState
from app.stream import parse_g1_action
from collections import Counter

from datagen.g1_demo1 import (
    DEMO1_FULL_TARGETS,
    DEMO1_SCALED_TARGETS,
    DEMO1_TARGET_PROFILES,
    EMPTY_KINDS,
    Demo1Targets,
    compile_demo1_dataset,
    compile_demo1_record,
    compile_demo1_records,
    demo1_coverage_report,
    demo1_records_from_batches,
    demo1_selection_distribution_errors,
    render_demo1_card,
    select_demo1_candidates,
)


def authored_record(index: int) -> dict:
    suffix = f"scene {index}"
    return {
        "id": f"demo1-record-{index:02d}",
        "persona": "journal-keeper",
        "domain": "cooking",
        "register": "casual-reflective",
        "author_slot": "test-author",
        "author_model": "test-model",
        "author_tranche": 1,
        "segments": [
            {
                "kind": "narration",
                "text": f"I set the mixing bowl beside the window for {suffix}.",
                "traps": [],
                "pause_after": "none",
            },
            {
                "kind": "narration",
                "text": f'Mara asked, "will the bread rise?" during {suffix}.',
                "traps": ["quoted_question"],
                "pause_after": "short",
            },
            {
                "kind": "address",
                "text": f"Does the sequence make sense in {suffix}?",
                "gold_reply": f"Yes—the bowl came first and Mara's question followed in {suffix}.",
                "pause_after": "medium",
            },
            {
                "kind": "narration",
                "text": f"The dough rested under a striped towel throughout {suffix}.",
                "traps": [],
                "pause_after": "medium",
            },
            {
                "kind": "narration",
                "text": f"I left the finished loaf untouched until morning after {suffix}.",
                "traps": [],
                "pause_after": "long",
            },
        ],
    }


AUTHORS = ("slot-a", "slot-b", "slot-c")
PERSONAS = tuple(f"persona-{name}" for name in "abcdefg")
DOMAINS = tuple(f"domain-{name}" for name in "hijklm")
REGISTERS = tuple(f"register-{name}" for name in "nopqr")
# Segment counts and address positions chosen so the fixture corpus lands in
# every length, event-count, and trigger-position bucket the gates care about.
SHAPES = (
    (5, 0, 300),
    (6, 2, 520),
    (7, 3, 700),
    (8, 6, 950),
    (9, 8, 1_200),
    (6, 4, 480),
)


def diverse_record(index: int) -> dict:
    """One authored record from a deliberately heterogeneous fixture corpus."""

    segment_count, address_index, characters = SHAPES[index % len(SHAPES)]
    filler = characters // segment_count
    segments: list[dict] = []
    for segment_index in range(segment_count):
        body = f"record {index} segment {segment_index} " + (
            chr(ord("a") + (index + segment_index) % 26) * filler
        )
        if segment_index == address_index:
            segments.append(
                {
                    "kind": "address",
                    "text": f"Does record {index} still read clearly? {body}",
                    "gold_reply": f"Yes, record {index} keeps its sequence at {segment_index}.",
                    "pause_after": ("none", "short", "medium", "long")[segment_index % 4],
                }
            )
            continue
        # Roughly a third of narrations carry a trap, so the fixture has a
        # meaningful hard-idle pool to subsample from.
        traps = ["rhetorical_question"] if (index + segment_index) % 3 == 0 else []
        segments.append(
            {
                "kind": "narration",
                "text": f"Narration {body}.",
                "traps": traps,
                "pause_after": ("none", "short", "medium", "long")[segment_index % 4],
            }
        )
    return {
        "id": f"diverse-{index:03d}",
        "persona": PERSONAS[index % len(PERSONAS)],
        "domain": DOMAINS[index % len(DOMAINS)],
        "register": REGISTERS[index % len(REGISTERS)],
        "author_slot": AUTHORS[index % len(AUTHORS)],
        "author_model": "fixture-model",
        "author_tranche": str(index % 4 + 1),
        "segments": segments,
    }


def diverse_corpus(count: int = 90) -> list[dict]:
    return [diverse_record(index) for index in range(count)]


def role_pool(records, role: str) -> list:
    return [
        candidate
        for record in records
        for candidate in record.candidates
        if candidate.role == role
    ]


class Demo1CompilerTests(unittest.TestCase):
    def test_full_snapshots_cadence_pause_and_address_neighbors(self) -> None:
        source = authored_record(1)
        compiled = compile_demo1_record(source)
        positive = next(
            candidate
            for candidate in compiled.candidates
            if candidate.role == "address-positive"
        )
        before = next(
            candidate
            for candidate in compiled.candidates
            if candidate.role == "address-before"
        )
        after = next(
            candidate
            for candidate in compiled.candidates
            if candidate.role == "address-after"
        )

        expected_address_snapshot = "\n".join(
            segment["text"] for segment in source["segments"][:3]
        )
        positive_turn = compiled.turns[positive.turn_offset]
        before_turn = compiled.turns[before.turn_offset]
        after_turn = compiled.turns[after.turn_offset]
        self.assertEqual(positive_turn.event.content, expected_address_snapshot)
        self.assertTrue(expected_address_snapshot.startswith(before_turn.event.content))
        self.assertLess(len(before_turn.event.content), len(expected_address_snapshot))
        self.assertEqual(after_turn.event.content, expected_address_snapshot)
        self.assertIs(positive_turn.action.kind, ActionKind.RESPOND)
        self.assertEqual(positive_turn.action.target, positive_turn.event.index)
        self.assertIs(after_turn.action.kind, ActionKind.IDLE)
        self.assertIs(before_turn.event.state, UserState.ACTIVE)
        self.assertIs(after_turn.event.state, UserState.IDLE)
        narration_by_segment = {
            candidate.segment_index: candidate.role
            for candidate in compiled.candidates
            if candidate.role.startswith("narration-")
        }
        self.assertEqual(narration_by_segment[1], "narration-hard-idle")
        self.assertEqual(narration_by_segment[4], "narration-ballast")

        prior = ""
        typing_deltas: list[int] = []
        time_gaps: list[int] = []
        prior_time = 0
        for turn in compiled.turns:
            content = turn.event.content
            if content and content.startswith(prior) and len(content) > len(prior):
                typing_deltas.append(len(content) - len(prior))
            if prior_time:
                time_gaps.append(turn.event.elapsed_ms - prior_time)
            prior = content
            prior_time = turn.event.elapsed_ms
        self.assertTrue(typing_deltas)
        self.assertGreaterEqual(min(typing_deltas), 4)
        self.assertLessEqual(max(typing_deltas), 7)
        self.assertGreater(len(set(time_gaps)), 5)
        self.assertTrue(any(gap >= 10_000 for gap in time_gaps))

    def test_small_exact_selection_is_deterministic_and_keeps_all_hard_idles(self) -> None:
        sources = [authored_record(index) for index in range(10)]
        targets = Demo1Targets(
            addresses=10,
            narrations=40,
            cards=66,
            empty_per_kind=2,
        )
        first = compile_demo1_dataset(sources, targets=targets)
        second = compile_demo1_dataset(reversed(sources), targets=targets)

        self.assertEqual(first.rows, second.rows)
        self.assertEqual(len(first.rows), 66)
        self.assertEqual(first.coverage["selected_roles"]["address-before"], 10)
        self.assertEqual(first.coverage["selected_roles"]["address-positive"], 10)
        self.assertEqual(first.coverage["selected_roles"]["address-after"], 10)
        self.assertEqual(
            first.coverage["selected_empty_kinds"],
            {"cleared": 2, "initial": 2, "unchanged": 2},
        )
        hard_available = {
            candidate.candidate_id
            for record in first.records
            for candidate in record.candidates
            if candidate.role == "narration-hard-idle"
        }
        selected = {candidate.candidate_id for candidate in first.selected}
        self.assertTrue(hard_available)
        self.assertTrue(hard_available <= selected)

    def test_rows_are_header_free_parseable_and_after_history_shows_response(self) -> None:
        compiled = compile_demo1_record(authored_record(4))
        for candidate in compiled.candidates:
            row = render_demo1_card(compiled, candidate)
            self.assertNotIn("<interaction_context>", row["prompt"])
            self.assertTrue(row["prompt"].endswith("<PREDICT_THIS_ACTION>"))
            self.assertTrue(parse_g1_action(row["completion"]).valid)

        after = next(
            candidate
            for candidate in compiled.candidates
            if candidate.role == "address-after"
        )
        after_row = render_demo1_card(compiled, after)
        self.assertRegex(after_row["prompt"], re.compile(r"<action>respond\(.+\)</action>"))
        self.assertEqual(after_row["completion"], "<action>idle()</action>")
        self.assertEqual(
            after_row["prompt"].count("<stream_event "),
            after_row["current_event_index"],
        )
        self.assertEqual(
            after_row["prompt"].count("<action>"),
            after_row["current_event_index"] - 1,
        )

        initial = next(
            candidate
            for candidate in compiled.candidates
            if candidate.role == "empty-initial"
        )
        cleared = next(
            candidate
            for candidate in compiled.candidates
            if candidate.role == "empty-cleared"
        )
        self.assertEqual(render_demo1_card(compiled, initial)["empty_kind"], "initial")
        self.assertIn("</stream_event>\n<action>", render_demo1_card(compiled, cleared)["prompt"])
        self.assertEqual(
            render_demo1_card(compiled, initial)["source_author_slot"],
            "test-author",
        )

    def test_split_is_exact_whole_record_and_stable(self) -> None:
        sources = [authored_record(index) for index in range(10)]
        compiled = compile_demo1_records(sources)
        reversed_compiled = compile_demo1_records(reversed(sources))
        split_by_record = {record.record_id: record.split for record in compiled}
        self.assertEqual(
            split_by_record,
            {record.record_id: record.split for record in reversed_compiled},
        )
        self.assertEqual(list(split_by_record.values()).count("dev"), 1)
        self.assertEqual(list(split_by_record.values()).count("train"), 9)
        for record in compiled:
            for candidate in record.candidates:
                self.assertEqual(render_demo1_card(record, candidate)["split"], record.split)

    def test_selection_fails_clearly_when_source_counts_are_wrong(self) -> None:
        compiled = compile_demo1_records([authored_record(1), authored_record(2)])
        with self.assertRaisesRegex(ValueError, "requires exactly 3 addresses; found 2"):
            select_demo1_candidates(
                compiled,
                targets=Demo1Targets(
                    addresses=3,
                    narrations=8,
                    cards=20,
                    empty_per_kind=1,
                ),
            )

    def test_batch_flattening_preserves_author_provenance(self) -> None:
        source = authored_record(1)
        flattened = demo1_records_from_batches(
            [
                {
                    "author": {
                        "slot": "sol-a",
                        "model": "gpt-5.6-sol",
                        "tranche": 2,
                    },
                    "records": [source],
                }
            ]
        )

        self.assertEqual(flattened[0]["author_slot"], "sol-a")
        self.assertEqual(flattened[0]["author_model"], "gpt-5.6-sol")
        self.assertEqual(flattened[0]["author_tranche"], "2")

    def test_each_required_provenance_field_rejects_missing_and_blank_values(self) -> None:
        fields = (
            "persona",
            "domain",
            "register",
            "author_slot",
            "author_model",
            "author_tranche",
        )
        for field in fields:
            with self.subTest(field=field, case="missing"):
                source = authored_record(1)
                del source[field]
                with self.assertRaisesRegex(
                    ValueError, rf"provenance field ['\"]{field}['\"]"
                ):
                    compile_demo1_record(source)
            with self.subTest(field=field, case="blank"):
                source = authored_record(1)
                source[field] = "   "
                with self.assertRaisesRegex(
                    ValueError, rf"provenance field ['\"]{field}['\"]"
                ):
                    compile_demo1_record(source)

    def test_valid_provenance_survives_compilation_and_rendering(self) -> None:
        source = authored_record(1)
        compiled = compile_demo1_record(source)
        row = render_demo1_card(compiled, compiled.candidates[0])

        self.assertEqual(row["source_persona"], source["persona"])
        self.assertEqual(row["source_domain"], source["domain"])
        self.assertEqual(row["source_register"], source["register"])
        self.assertEqual(row["source_author_slot"], source["author_slot"])
        self.assertEqual(row["source_author_model"], source["author_model"])
        self.assertEqual(row["source_author_tranche"], str(source["author_tranche"]))

    def test_batch_flattening_rejects_blank_author_provenance(self) -> None:
        source = authored_record(1)
        for bad_value in (None, " "):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(
                    ValueError, "provenance field 'author_slot'"
                ):
                    demo1_records_from_batches(
                        [
                            {
                                "author": {
                                    "slot": bad_value,
                                    "model": "gpt-5.6-sol",
                                    "tranche": 2,
                                },
                                "records": [source],
                            }
                        ]
                    )


class Demo1ScaledTargetTests(unittest.TestCase):
    """The 1,800-card mix is arithmetic, not a magic constant."""

    def test_scaled_profile_arithmetic_is_exact_and_third_scale(self) -> None:
        scaled = DEMO1_SCALED_TARGETS
        full = DEMO1_FULL_TARGETS

        # Source requirements are unchanged: no re-authoring, only selection.
        self.assertEqual(scaled.addresses, full.addresses)
        self.assertEqual(scaled.narrations, full.narrations)

        self.assertEqual(scaled.cards, 1_800)
        self.assertEqual(scaled.cards * 3, full.cards)
        self.assertEqual(scaled.address_sites, 233)
        self.assertEqual(scaled.empty_per_kind, 50)
        self.assertEqual(scaled.hard_idles, 205)
        self.assertEqual(scaled.narration_cards, 951)
        ballast = scaled.narration_cards - scaled.hard_idles
        self.assertEqual(ballast, 746)
        self.assertEqual(
            3 * scaled.address_sites
            + len(EMPTY_KINDS) * scaled.empty_per_kind
            + scaled.hard_idles
            + ballast,
            1_800,
        )
        # Addresses take the floor of an exact third; hard idles take an exact
        # third; empties are held above their proportional share on purpose.
        self.assertEqual(scaled.address_sites, full.addresses // 3)
        self.assertEqual(scaled.hard_idles, 615 // 3)
        self.assertGreater(scaled.empty_per_kind, full.empty_per_kind // 3)
        self.assertGreaterEqual(scaled.empty_per_kind, 50)
        self.assertEqual(scaled.strategy, "stratified")
        self.assertEqual(DEMO1_TARGET_PROFILES["scaled-1800"], scaled)
        self.assertEqual(DEMO1_TARGET_PROFILES["full-5400"], full)

    def test_targets_reject_card_budgets_that_cannot_hold_the_fixed_roles(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceed the card target"):
            Demo1Targets(cards=1_800)
        with self.assertRaisesRegex(ValueError, "hard_idles"):
            Demo1Targets(cards=1_800, selected_addresses=233, empty_per_kind=50, hard_idles=1_000)
        with self.assertRaisesRegex(ValueError, "selected_addresses"):
            Demo1Targets(selected_addresses=701)
        with self.assertRaisesRegex(ValueError, "selection strategy"):
            Demo1Targets(strategy="random")


class Demo1SubsampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = diverse_corpus()
        self.records = compile_demo1_records(self.sources)
        self.hard_available = len(role_pool(self.records, "narration-hard-idle"))
        self.ballast_available = len(role_pool(self.records, "narration-ballast"))
        self.address_sites = 30
        self.empty_per_kind = 8
        self.hard_idles = self.hard_available // 3
        self.ballast = self.ballast_available // 3
        self.targets = Demo1Targets(
            addresses=len(self.sources),
            narrations=self.hard_available + self.ballast_available,
            cards=3 * self.address_sites
            + len(EMPTY_KINDS) * self.empty_per_kind
            + self.hard_idles
            + self.ballast,
            empty_per_kind=self.empty_per_kind,
            selected_addresses=self.address_sites,
            hard_idles=self.hard_idles,
            strategy="stratified",
        )

    def select(self, records=None):
        return select_demo1_candidates(records or self.records, targets=self.targets)

    def test_subsample_hits_the_exact_count_and_role_mix(self) -> None:
        selected = self.select()
        counts = Counter(candidate.role for candidate in selected)

        self.assertEqual(len(selected), self.targets.cards)
        self.assertEqual(counts["address-before"], self.address_sites)
        self.assertEqual(counts["address-positive"], self.address_sites)
        self.assertEqual(counts["address-after"], self.address_sites)
        self.assertEqual(counts["narration-hard-idle"], self.hard_idles)
        self.assertEqual(counts["narration-ballast"], self.ballast)
        self.assertEqual(
            Counter(
                candidate.empty_kind
                for candidate in selected
                if candidate.empty_kind
            ),
            {kind: self.empty_per_kind for kind in EMPTY_KINDS},
        )
        self.assertLess(len(selected), len(list(role_pool(self.records, "narration-ballast"))) * 4)

    def test_address_triples_always_travel_together(self) -> None:
        selected = self.select()
        sites: dict[tuple[str, int | None], set[str]] = {}
        for candidate in selected:
            if candidate.role.startswith("address-"):
                key = (candidate.record_id, candidate.segment_index)
                sites.setdefault(key, set()).add(candidate.role)

        self.assertEqual(len(sites), self.address_sites)
        for key, roles in sites.items():
            self.assertEqual(
                roles,
                {"address-before", "address-positive", "address-after"},
                f"incoherent address site {key}",
            )

    def test_subsample_is_deterministic_under_input_reordering(self) -> None:
        first = self.select()
        second = self.select(compile_demo1_records(list(reversed(self.sources))))
        third = self.select()

        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertEqual(
            [candidate.candidate_id for candidate in first],
            sorted(candidate.candidate_id for candidate in first),
        )

    def test_subsample_spreads_across_records_authors_and_buckets(self) -> None:
        selected = self.select()
        coverage = demo1_coverage_report(self.records, selected)
        full = demo1_coverage_report(
            self.records,
            select_demo1_candidates(
                self.records,
                targets=Demo1Targets(
                    addresses=len(self.sources),
                    narrations=self.hard_available + self.ballast_available,
                    cards=3 * len(self.sources)
                    + len(EMPTY_KINDS) * self.empty_per_kind
                    + self.hard_available
                    + self.ballast_available,
                    empty_per_kind=self.empty_per_kind,
                ),
            ),
        )

        # Ballast is the largest draw; it should touch nearly every record
        # rather than pile several cards onto a few.
        ballast_records = {
            candidate.record_id
            for candidate in selected
            if candidate.role == "narration-ballast"
        }
        self.assertGreaterEqual(
            len(ballast_records),
            int(min(self.ballast, len(self.sources)) * 0.9),
        )
        self.assertGreaterEqual(
            coverage["selected_distinct_records"], int(len(self.sources) * 0.9)
        )

        selected_total = coverage["selected_cards"]
        full_total = full["selected_cards"]
        for field in ("author_slot", "length_bucket", "trigger_position"):
            selected_shares = coverage["selected_source_distribution"][field]
            full_shares = full["selected_source_distribution"][field]
            self.assertEqual(set(selected_shares), set(full_shares))
            for name, count in selected_shares.items():
                self.assertAlmostEqual(
                    count / selected_total,
                    full_shares[name] / full_total,
                    delta=0.03,
                    msg=f"{field} {name} drifted from the full-build share",
                )

    def test_distribution_gates_still_pass_at_the_smaller_size(self) -> None:
        coverage = demo1_coverage_report(self.records, self.select())
        self.assertEqual(demo1_selection_distribution_errors(coverage), [])

    def test_distribution_gate_rejects_a_narrow_selection(self) -> None:
        narrow = {
            "selected_cards": 100,
            "selected_source_distribution": {
                "author_slot": {"slot-a": 60, "slot-b": 25, "slot-c": 15},
                "persona": {f"persona-{index}": 20 for index in range(5)},
                "domain": {f"domain-{index}": 20 for index in range(5)},
                "register": {"domi": 40, "b": 20, "c": 20, "d": 20},
                "length_bucket": {"short": 70, "medium": 20, "long": 10},
                "event_count_bucket": {"short": 40, "medium": 40, "long": 20},
                "trigger_position": {"early": 60, "middle": 40},
            },
        }
        errors = demo1_selection_distribution_errors(narrow)
        joined = " | ".join(errors)

        self.assertIn("author 'slot-a'", joined)
        self.assertIn("author 'slot-c'", joined)
        self.assertIn("register 'domi' exceeds 35%", joined)
        self.assertIn("length_bucket 'short' exceeds 60%", joined)
        self.assertIn("trigger_position missing ['late']", joined)
        self.assertEqual(demo1_selection_distribution_errors({}), [
            "selection distribution: no cards were selected"
        ])

    def test_hash_strategy_still_takes_every_address_and_hard_idle(self) -> None:
        legacy = Demo1Targets(
            addresses=len(self.sources),
            narrations=self.hard_available + self.ballast_available,
            cards=3 * len(self.sources)
            + len(EMPTY_KINDS) * self.empty_per_kind
            + self.hard_available
            + self.ballast,
            empty_per_kind=self.empty_per_kind,
        )
        selected = select_demo1_candidates(self.records, targets=legacy)
        counts = Counter(candidate.role for candidate in selected)

        self.assertEqual(legacy.strategy, "hash")
        self.assertEqual(counts["address-positive"], len(self.sources))
        self.assertEqual(counts["narration-hard-idle"], self.hard_available)
        self.assertEqual(counts["narration-ballast"], self.ballast)

    def test_subsample_reports_a_shortfall_instead_of_silently_shrinking(self) -> None:
        with self.assertRaisesRegex(ValueError, "hard-idle narrations; found"):
            select_demo1_candidates(
                self.records,
                targets=Demo1Targets(
                    addresses=len(self.sources),
                    narrations=self.hard_available + self.ballast_available,
                    cards=3 * self.address_sites
                    + len(EMPTY_KINDS) * self.empty_per_kind
                    + self.hard_available
                    + 1_000,
                    empty_per_kind=self.empty_per_kind,
                    selected_addresses=self.address_sites,
                    hard_idles=self.hard_available + 1,
                ),
            )

    def test_scaled_rows_still_round_trip_through_the_g1_parser(self) -> None:
        build = compile_demo1_dataset(self.sources, targets=self.targets)

        self.assertEqual(len(build.rows), self.targets.cards)
        for row in build.rows:
            self.assertEqual(row["schema_version"], "g1")
            self.assertTrue(row["prompt"].endswith("<PREDICT_THIS_ACTION>"))
            self.assertTrue(parse_g1_action(row["completion"]).valid)
        splits_by_episode: dict[str, set[str]] = {}
        for row in build.rows:
            splits_by_episode.setdefault(row["episode"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in splits_by_episode.values()))

    def test_compiled_records_carry_the_derived_selection_strata(self) -> None:
        metadata = dict(self.records[0].source_metadata)

        self.assertIn(metadata["length_bucket"], {"short", "medium", "long"})
        self.assertIn(metadata["event_count_bucket"], {"short", "medium", "long"})
        self.assertIn(metadata["trigger_position"], {"early", "middle", "late"})
        buckets = {
            field: {
                dict(record.source_metadata)[field] for record in self.records
            }
            for field in ("length_bucket", "event_count_bucket", "trigger_position")
        }
        self.assertEqual(buckets["length_bucket"], {"short", "medium", "long"})
        self.assertEqual(buckets["trigger_position"], {"early", "middle", "late"})


if __name__ == "__main__":
    unittest.main()
