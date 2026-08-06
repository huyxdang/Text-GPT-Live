from __future__ import annotations

import re
import unittest
from pathlib import Path

from app.domain import ActionKind, find_occurrence
from app.stream import parse_g1_action
from datagen.g1_authored_demo2 import (
    demo2_authored_paths,
    load_authored_batches,
    occurrence_count,
)
from datagen.g1_demo2 import (
    Demo2Targets,
    compile_demo2_dataset,
    compile_demo2_record,
    compile_demo2_records,
    demo2_records_from_batches,
    episode_skeleton,
    highlight_occurrence,
    render_demo2_card,
    select_demo2_candidates,
    unique_edit_quote,
)

SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "data" / "g1_demo2_sample"


def sample_records() -> tuple[dict, ...]:
    return demo2_records_from_batches(
        load_authored_batches(demo2_authored_paths(SAMPLE_ROOT))
    )


def provenance(**overrides: object) -> dict:
    base = {
        "persona": "journal-keeper",
        "domain": "cooking",
        "register": "casual-reflective",
        "author_slot": "test-author",
        "author_model": "test-model",
        "author_tranche": 1,
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def correction_source(identifier: str = "unit-corr-1") -> dict:
    return {
        "id": identifier,
        "mode": "corrections",
        **provenance(),
        "instruction": {
            "text": "Fix any grammar slips you catch while I am typing.",
            "ack": "Got it, I will flag slips as you go.",
        },
        "segments": [
            {
                "kind": "passage",
                "text": "On Monday they was late for the standing meeting again.",
                "errors": [
                    {
                        "wrong": "they was",
                        "right": "they were",
                        "subtype": "agreement",
                        "occurrence": 1,
                    }
                ],
                "pause_after": "short",
            },
            {
                "kind": "passage",
                "text": "By Friday they was early for once in their whole lives.",
                "errors": [
                    {
                        "wrong": "they was",
                        "right": "they were",
                        "subtype": "agreement",
                        "occurrence": 1,
                    }
                ],
                "pause_after": "medium",
            },
            {
                "kind": "passage",
                "text": "Nothing else about the week was worth writing down.",
                "trap": "clean_text",
                "pause_after": "long",
            },
            {
                "kind": "repair",
                "text": "The kitchen calendar finally showed the right dates.",
                "repair": {
                    "wrong": "calender",
                    "right": "calendar",
                    "subtype": "letter-swap",
                    "occurrence": 1,
                },
                "pause_after": "short",
            },
            {
                "kind": "aside",
                "text": "I always type recieve the wrong way round in my notes.",
                "mentions": ["recieve"],
                "pause_after": "none",
            },
        ],
    }


def highlight_source(identifier: str = "unit-high-1") -> dict:
    return {
        "id": identifier,
        "mode": "highlights",
        "category": "animal",
        "category_words": ["heron", "otter"],
        **provenance(persona="letter-writer", domain="family", register="warm"),
        "instruction": {
            "text": "Mark every creature I name while I write this down.",
            "ack": "Happy to, each creature gets marked as it arrives.",
        },
        "segments": [
            {
                "kind": "passage",
                "text": "A heron stood in the shallow water beside the reeds.",
                "marks": [{"word": "heron", "literal": True}],
                "pause_after": "short",
            },
            {
                "kind": "passage",
                "text": "Another heron landed on the same bank an hour later.",
                "marks": [{"word": "heron", "literal": True}],
                "pause_after": "none",
            },
            {
                "kind": "passage",
                "text": "Our neighbour Otter never explained where that name came from.",
                "marks": [{"word": "Otter", "literal": False}],
                "pause_after": "medium",
            },
            {
                "kind": "passage",
                "text": "The rest of the afternoon passed without anything notable.",
                "trap": "clean_text",
                "pause_after": "long",
            },
        ],
    }


def action_at(record, role: str):
    candidate = next(item for item in record.candidates if item.role == role)
    return candidate, record.turns[candidate.turn_offset]


class QuoteArithmeticTests(unittest.TestCase):
    def test_a_quote_is_widened_only_until_it_is_unique(self) -> None:
        snapshot = "they was late\nby friday they was"
        quote, replacement = unique_edit_quote(snapshot, 24, 32, "they were")
        self.assertEqual(quote, "friday they was")
        self.assertEqual(replacement, "friday they were")
        self.assertEqual(occurrence_count(snapshot, quote), 1)

    def test_widening_never_crosses_into_the_instruction_line(self) -> None:
        snapshot = "fix my slips they was\nthey was"
        with self.assertRaisesRegex(ValueError, "unique within its line"):
            unique_edit_quote(snapshot, 22, 30, "they were")

    def test_highlight_occurrence_matches_the_app_search(self) -> None:
        snapshot = "a heron then another heron"
        self.assertEqual(highlight_occurrence(snapshot, "heron", 2), 1)
        self.assertEqual(highlight_occurrence(snapshot, "heron", 21), 2)
        self.assertEqual(find_occurrence(snapshot, "heron", 2), 21)


class Demo2CompilerTests(unittest.TestCase):
    def test_human_cadence_and_non_uniform_gaps(self) -> None:
        compiled = compile_demo2_records(sample_records())
        deltas: list[int] = []
        gaps: list[int] = []
        for record in compiled:
            prior_text = ""
            prior_time = 0
            for turn in record.turns:
                content = turn.event.content
                if content and content.startswith(prior_text) and len(content) > len(prior_text):
                    deltas.append(len(content) - len(prior_text))
                if prior_time:
                    gaps.append(int(turn.event.elapsed_ms) - prior_time)
                prior_text = content
                prior_time = int(turn.event.elapsed_ms)
        self.assertGreaterEqual(min(deltas), 4)
        self.assertLessEqual(max(deltas), 7)
        self.assertGreater(len(set(gaps)), 50)
        self.assertTrue(any(gap >= 10_000 for gap in gaps))
        self.assertTrue(all(gap >= 500 for gap in gaps))

    def test_the_episode_opens_with_the_instruction_and_one_ack(self) -> None:
        compiled = compile_demo2_record(correction_source())
        candidate, turn = action_at(compiled, "instruction-ack")
        self.assertIs(turn.action.kind, ActionKind.RESPOND)
        self.assertEqual(turn.action.target, turn.event.index)
        self.assertEqual(turn.event.content, correction_source()["instruction"]["text"])
        responds = [
            item for item in compiled.turns if item.action.kind is ActionKind.RESPOND
        ]
        self.assertEqual(len(responds), 1)
        _, after = action_at(compiled, "instruction-after")
        self.assertIs(after.action.kind, ActionKind.IDLE)
        _, before = action_at(compiled, "instruction-before")
        self.assertIs(before.action.kind, ActionKind.IDLE)
        self.assertLess(len(before.event.content), len(turn.event.content))

    def test_error_clusters_carry_a_unique_widened_quote(self) -> None:
        compiled = compile_demo2_record(correction_source())
        positives = [item for item in compiled.candidates if item.role == "error-positive"]
        self.assertEqual(len(positives), 2)
        quotes = []
        for candidate in positives:
            turn = compiled.turns[candidate.turn_offset]
            self.assertEqual(turn.action.tool_name, "suggest_edit")
            quote = turn.action.arguments["quote"]
            quotes.append(quote)
            self.assertEqual(occurrence_count(turn.event.content, quote), 1)
            self.assertNotEqual(turn.action.arguments["replacement"], quote)
            self.assertNotIn(quote, turn.event.content.split("\n", 1)[0])
        self.assertEqual(quotes[0], "they was")
        # The second identical slip must widen; the first stays minimal.
        self.assertEqual(quotes[1], "Friday they was")

    def test_before_neighbour_shows_a_partially_typed_trigger(self) -> None:
        compiled = compile_demo2_record(correction_source())
        before = next(item for item in compiled.candidates if item.role == "error-before")
        positive = next(
            item
            for item in compiled.candidates
            if item.role == "error-positive" and item.trigger_key == before.trigger_key
        )
        after = next(
            item
            for item in compiled.candidates
            if item.role == "error-after" and item.trigger_key == before.trigger_key
        )
        before_turn = compiled.turns[before.turn_offset]
        positive_turn = compiled.turns[positive.turn_offset]
        after_turn = compiled.turns[after.turn_offset]
        self.assertEqual(before.before_kind, "partial")
        self.assertTrue(positive_turn.event.content.startswith(before_turn.event.content))
        self.assertTrue(before_turn.event.content.endswith("they"))
        self.assertIs(before_turn.action.kind, ActionKind.IDLE)
        self.assertIs(after_turn.action.kind, ActionKind.IDLE)
        self.assertIn(
            positive_turn.action.arguments["quote"], after_turn.event.content
        )

    def test_highlight_occurrence_counts_the_visible_textbox(self) -> None:
        compiled = compile_demo2_record(highlight_source())
        positives = [item for item in compiled.candidates if item.role == "match-positive"]
        self.assertEqual(len(positives), 2)
        occurrences = []
        for candidate in positives:
            turn = compiled.turns[candidate.turn_offset]
            self.assertEqual(turn.action.tool_name, "highlight")
            quote = turn.action.arguments["quote"]
            occurrence = turn.action.arguments["occurrence"]
            occurrences.append(occurrence)
            self.assertEqual(quote, "heron")
            self.assertGreaterEqual(find_occurrence(turn.event.content, quote, occurrence), 0)
        self.assertEqual(occurrences, [1, 2])

    def test_all_four_traps_compile_as_graded_hard_idles(self) -> None:
        traps: dict[str, str] = {}
        for source in (correction_source(), highlight_source()):
            compiled = compile_demo2_record(source)
            for candidate in compiled.candidates:
                if candidate.trap:
                    traps[candidate.trap] = candidate.role
                    self.assertIs(
                        compiled.turns[candidate.turn_offset].action.kind, ActionKind.IDLE
                    )
        self.assertEqual(
            sorted(traps),
            ["clean_text", "instruction_mention", "non_literal_match", "self_correction"],
        )

    def test_a_self_correction_is_typed_wrong_then_repaired(self) -> None:
        compiled = compile_demo2_record(correction_source())
        candidate, turn = action_at(compiled, "trap-selfcorrect")
        history = [item.event.content for item in compiled.turns[: candidate.turn_offset]]
        self.assertTrue(any("calender" in content for content in history))
        self.assertNotIn("calender", turn.event.content)
        self.assertTrue(turn.event.content.endswith("calendar"))
        shrinks = [
            index
            for index in range(1, len(compiled.turns))
            if len(compiled.turns[index].event.content)
            < len(compiled.turns[index - 1].event.content)
            and compiled.turns[index].event.content
        ]
        self.assertTrue(shrinks)

    def test_silence_covers_initial_unchanged_and_cleared(self) -> None:
        kinds: set[str] = set()
        for record in compile_demo2_records(sample_records()):
            for candidate in record.candidates:
                if candidate.empty_kind:
                    turn = record.turns[candidate.turn_offset]
                    self.assertEqual(turn.event.content, "")
                    self.assertIs(turn.action.kind, ActionKind.IDLE)
                    kinds.add(candidate.empty_kind)
        self.assertEqual(kinds, {"initial", "unchanged", "cleared"})

    def test_the_episode_skeleton_is_not_a_constant(self) -> None:
        shapes = {
            episode_skeleton(f"record-{index}")["variant"] for index in range(40)
        }
        self.assertGreaterEqual(len(shapes), 4)
        compiled = compile_demo2_records(sample_records())
        self.assertGreater(len({record.skeleton for record in compiled}), 1)
        openings = {
            tuple(turn.event.content == "" for turn in record.turns[:3])
            for record in compiled
        }
        self.assertGreater(len(openings), 1)

    def test_rows_are_header_free_canonical_and_parseable(self) -> None:
        compiled = compile_demo2_record(highlight_source())
        for candidate in compiled.candidates:
            row = render_demo2_card(compiled, candidate)
            self.assertNotIn("<interaction_context>", row["prompt"])
            self.assertTrue(row["prompt"].endswith("<PREDICT_THIS_ACTION>"))
            parsed = parse_g1_action(row["completion"])
            self.assertTrue(parsed.valid, row["completion"])
            self.assertEqual(
                row["prompt"].count("<stream_event "), row["current_event_index"]
            )
            self.assertEqual(
                row["prompt"].count("<action>"), row["current_event_index"] - 1
            )
        after = next(
            item for item in compiled.candidates if item.role == "instruction-after"
        )
        after_row = render_demo2_card(compiled, after)
        self.assertRegex(after_row["prompt"], re.compile(r"<action>respond\(.+\)</action>"))
        self.assertEqual(after_row["completion"], "<action>idle()</action>")

    def test_a_match_after_neighbour_shows_the_highlight_in_history(self) -> None:
        compiled = compile_demo2_record(highlight_source())
        after = next(item for item in compiled.candidates if item.role == "match-after")
        row = render_demo2_card(compiled, after)
        self.assertIn("<action>highlight(", row["prompt"])
        self.assertEqual(row["completion"], "<action>idle()</action>")

    def test_provenance_is_required_and_survives_rendering(self) -> None:
        for field in (
            "mode",
            "persona",
            "domain",
            "register",
            "author_slot",
            "author_model",
            "author_tranche",
        ):
            for bad in (None, "   "):
                source = correction_source()
                source[field] = bad
                with self.subTest(field=field, bad=bad):
                    with self.assertRaisesRegex(
                        ValueError, rf"provenance field ['\"]{field}['\"]"
                    ):
                        compile_demo2_record(source)
            source = correction_source()
            del source[field]
            with self.assertRaisesRegex(ValueError, rf"provenance field ['\"]{field}['\"]"):
                compile_demo2_record(source)
        compiled = compile_demo2_record(correction_source())
        row = render_demo2_card(compiled, compiled.candidates[0])
        self.assertEqual(row["source_persona"], "journal-keeper")
        self.assertEqual(row["source_mode"], "corrections")
        self.assertEqual(row["source_author_tranche"], "1")
        self.assertEqual(row["obligation"], "standing-instruction")


class Demo2SelectionTests(unittest.TestCase):
    TARGETS = Demo2Targets(
        errors=17,
        matches=8,
        episodes=12,
        cards=140,
        empty_per_kind=2,
        hard_idle_cards=24,
    )

    def test_the_exact_mix_is_deterministic_under_input_reordering(self) -> None:
        records = list(sample_records())
        first = compile_demo2_dataset(records, targets=self.TARGETS)
        second = compile_demo2_dataset(list(reversed(records)), targets=self.TARGETS)
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(len(first.rows), 140)
        roles = first.coverage["selected_roles"]
        self.assertEqual(roles["instruction-ack"], 12)
        self.assertEqual(roles["error-positive"], 17)
        self.assertEqual(roles["match-positive"], 8)
        self.assertEqual(
            first.coverage["selected_empty_kinds"],
            {"cleared": 2, "initial": 2, "unchanged": 2},
        )
        self.assertEqual(sum(first.coverage["selected_traps"].values()), 24)
        self.assertEqual(len(first.coverage["selected_traps"]), 4)

    def test_neighbours_are_selected_as_pairs(self) -> None:
        build = compile_demo2_dataset(list(sample_records()), targets=self.TARGETS)
        roles = build.coverage["selected_roles"]
        for before, after in (
            ("instruction-before", "instruction-after"),
            ("error-before", "error-after"),
            ("match-before", "match-after"),
        ):
            self.assertEqual(roles.get(before, 0), roles.get(after, 0), before)
        before_keys = {
            item.trigger_key for item in build.selected if item.role.endswith("-before")
        }
        after_keys = {
            item.trigger_key for item in build.selected if item.role.endswith("-after")
        }
        self.assertEqual(before_keys, after_keys)

    def test_selection_fails_with_a_count_diagnosis(self) -> None:
        compiled = compile_demo2_records(sample_records())
        with self.assertRaisesRegex(ValueError, "exactly 99 errors; found 17"):
            select_demo2_candidates(
                compiled, targets=Demo2Targets(errors=99, matches=8, episodes=12)
            )
        with self.assertRaisesRegex(ValueError, "hard-idle trap cards"):
            select_demo2_candidates(
                compiled,
                targets=Demo2Targets(
                    errors=17,
                    matches=8,
                    episodes=12,
                    cards=400,
                    empty_per_kind=2,
                    hard_idle_cards=999,
                ),
            )

    def test_the_production_targets_close_exactly(self) -> None:
        """1,800 cards must be reachable with no ballast slack and no overshoot."""

        targets = Demo2Targets()
        positives = targets.episodes + targets.errors + targets.matches
        empties = targets.empty_per_kind * 3
        mandatory = positives + empties + targets.hard_idle_cards
        remaining = targets.cards - mandatory
        self.assertEqual(positives, 540)
        self.assertEqual(mandatory, 791)
        self.assertEqual(remaining, 1_009)
        # Neighbours ship in pairs; an odd remainder deliberately leaves one
        # ordinary ballast idle after the maximum 504 pairs.
        self.assertEqual(remaining % 2, 1)
        self.assertLessEqual(remaining // 2, positives)
        self.assertEqual(remaining - 2 * (remaining // 2), 1)
        self.assertEqual(mandatory + remaining, targets.cards)

    def test_splits_are_whole_episode_and_stable(self) -> None:
        build = compile_demo2_dataset(list(sample_records()), targets=self.TARGETS)
        split_by_episode: dict[str, set[str]] = {}
        for row in build.rows:
            split_by_episode.setdefault(str(row["episode"]), set()).add(str(row["split"]))
        self.assertTrue(all(len(splits) == 1 for splits in split_by_episode.values()))
        self.assertEqual(build.coverage["source_splits"], {"dev": 1, "train": 11})


if __name__ == "__main__":
    unittest.main()
