from __future__ import annotations

import copy
import unittest
from pathlib import Path

from datagen.g1_authored_demo2 import (
    category_matches,
    demo2_authored_paths,
    load_authored_batches,
    occurrence_count,
    occurrence_start,
    validate_demo2_batches,
)

SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "data" / "g1_demo2_sample"


def correction_record(index: int, **overrides: object) -> dict:
    record = {
        "id": f"corr-{index:03d}",
        "mode": "corrections",
        "persona": f"persona-{index}",
        "domain": f"domain-{index}",
        "register": f"register-{index % 4}",
        "instruction": {
            "text": f"Please repair slip number {index} the moment you notice it.",
            "ack": f"Fine, slip number {index} will be repaired quickly.",
        },
        "segments": [
            {
                "kind": "passage",
                "text": f"Fixture {index} opened with a perfectly ordinary clean sentence.",
                "trap": "clean_text",
                "pause_after": "short",
            },
            {
                "kind": "passage",
                "text": f"Fixture {index} recorded that the parcels was left beside the door.",
                "errors": [
                    {
                        "wrong": "parcels was",
                        "right": "parcels were",
                        "subtype": "agreement",
                        "occurrence": 1,
                    }
                ],
                "pause_after": "none",
            },
            {
                "kind": "repair",
                "text": f"Fixture {index} noted the delivery arrived before the rain started.",
                "repair": {
                    "wrong": "delivary",
                    "right": "delivery",
                    "subtype": "letter-swap",
                    "occurrence": 1,
                },
                "pause_after": "medium",
            },
            {
                "kind": "aside",
                "text": f"Fixture {index} always writes seperate the wrong way round.",
                "mentions": ["seperate"],
                "pause_after": "long",
            },
        ],
    }
    record.update(overrides)  # type: ignore[arg-type]
    return record


def highlight_record(index: int, **overrides: object) -> dict:
    record = {
        "id": f"high-{index:03d}",
        "mode": "highlights",
        "category": f"category-{index}",
        "category_words": ["heron", "otter"],
        "persona": f"hpersona-{index}",
        "domain": f"hdomain-{index}",
        "register": f"register-{index % 4}",
        "instruction": {
            "text": f"Mark creature number {index} whenever one turns up here.",
            "ack": f"Right, creature number {index} will be marked on sight.",
        },
        "segments": [
            {
                "kind": "passage",
                "text": f"Scene {index} began with a heron standing in the shallow river.",
                "marks": [{"word": "heron", "literal": True}],
                "pause_after": "short",
            },
            {
                "kind": "passage",
                "text": f"Scene {index} introduced a neighbour named Otter who never explained.",
                "marks": [{"word": "Otter", "literal": False}],
                "pause_after": "none",
            },
            {
                "kind": "passage",
                "text": f"Scene {index} ended with nothing at all worth marking down.",
                "trap": "clean_text",
                "pause_after": "long",
            },
        ],
    }
    record.update(overrides)  # type: ignore[arg-type]
    return record


def batch(slot: str, records: list[dict]) -> dict:
    return {
        "schema_version": "g1-authored-demo2-1",
        "demo": "demo-2",
        "author": {"model": "test-model", "slot": slot, "tranche": 1},
        "records": records,
    }


def errors_for(records: list[dict]) -> list[str]:
    return validate_demo2_batches([batch("author-a", records)])["errors"]


def narrative_records(openers: list[str]) -> list[dict]:
    """Correction records whose two passages open with chosen words, in order.

    Each record contributes exactly two passages -- ``openers[2*i]`` and
    ``openers[2*i + 1]`` -- so a test can put an exact number of passages
    behind any given opening word regardless of how records happen to group.
    """

    records: list[dict] = []
    for pair_index in range(0, len(openers), 2):
        word_a = openers[pair_index]
        word_b = openers[pair_index + 1] if pair_index + 1 < len(openers) else "clean"
        index = pair_index // 2
        records.append(
            {
                "id": f"nar-{index:03d}",
                "mode": "corrections",
                "persona": f"persona-{index}",
                "domain": f"domain-{index}",
                "register": f"register-{index % 4}",
                "instruction": {
                    "text": f"Please repair slip number {index} the moment you notice it.",
                    "ack": f"Fine, slip number {index} will be repaired quickly.",
                },
                "segments": [
                    {
                        "kind": "passage",
                        "text": f"{word_a} fixture {index} opened with a perfectly ordinary sentence.",
                        "trap": "clean_text",
                        "pause_after": "short",
                    },
                    {
                        "kind": "passage",
                        "text": f"{word_b} fixture {index} recorded that the parcels was left beside the door.",
                        "errors": [
                            {
                                "wrong": "parcels was",
                                "right": "parcels were",
                                "subtype": "agreement",
                                "occurrence": 1,
                            }
                        ],
                        "pause_after": "none",
                    },
                ],
            }
        )
    return records


def offset_narrative_records(index_offset: int, openers: list[str]) -> list[dict]:
    """Like ``narrative_records``, but with the record index shifted.

    Needed whenever a test builds more than one batch: each record's id and
    passage/instruction text is derived from its index, so two batches that
    each start counting from zero would collide on duplicate ids and
    duplicate text.  A large, distinct offset per batch keeps every batch's
    records unique.
    """

    records: list[dict] = []
    for pair_index in range(0, len(openers), 2):
        word_a = openers[pair_index]
        word_b = openers[pair_index + 1] if pair_index + 1 < len(openers) else "clean"
        index = index_offset + pair_index // 2
        records.append(
            {
                "id": f"nar-{index:05d}",
                "mode": "corrections",
                "persona": f"persona-{index}",
                "domain": f"domain-{index}",
                "register": f"register-{index % 4}",
                "instruction": {
                    "text": f"Please repair slip number {index} the moment you notice it.",
                    "ack": f"Fine, slip number {index} will be repaired quickly.",
                },
                "segments": [
                    {
                        "kind": "passage",
                        "text": f"{word_a} fixture {index} opened with a perfectly ordinary sentence.",
                        "trap": "clean_text",
                        "pause_after": "short",
                    },
                    {
                        "kind": "passage",
                        "text": f"{word_b} fixture {index} recorded that the parcels was left beside the door.",
                        "errors": [
                            {
                                "wrong": "parcels was",
                                "right": "parcels were",
                                "subtype": "agreement",
                                "occurrence": 1,
                            }
                        ],
                        "pause_after": "none",
                    },
                ],
            }
        )
    return records


class TextPrimitiveTests(unittest.TestCase):
    def test_occurrence_walk_matches_the_app_search(self) -> None:
        text = "aaa fox and fox again"
        self.assertEqual(occurrence_count(text, "fox"), 2)
        self.assertEqual(occurrence_start(text, "fox", 2), 12)
        self.assertEqual(occurrence_start(text, "fox", 3), -1)
        # Overlapping matches are counted the way find_occurrence walks them.
        self.assertEqual(occurrence_count("aaaa", "aa"), 3)

    def test_category_matching_is_whole_word_and_case_preserving(self) -> None:
        found = category_matches("A Fox, a foxglove and one fox.", ["fox"])
        self.assertEqual([surface for _, _, surface in found], ["Fox", "fox"])


class Demo2SchemaTests(unittest.TestCase):
    def test_a_well_formed_pair_of_records_validates(self) -> None:
        self.assertEqual(errors_for([correction_record(1), highlight_record(1)]), [])

    def test_reserved_personas_and_domains_are_rejected(self) -> None:
        for field, value in (("persona", "product-reviewer"), ("domain", "personal-finance")):
            with self.subTest(field=field):
                errors = errors_for([correction_record(1, **{field: value})])
                self.assertTrue(any("reserved" in error for error in errors), errors)

    def test_markup_and_escapable_characters_are_rejected(self) -> None:
        record = correction_record(1)
        record["segments"][0]["text"] = "Fixture one used a < character in prose text."
        self.assertTrue(any("'<'" in error for error in errors_for([record])))
        record = correction_record(2)
        record["segments"][0]["text"] = "Fixture two wrote highlight({}) into its prose."
        self.assertTrue(any("action markup" in error for error in errors_for([record])))

    def test_a_planted_error_must_really_be_in_the_text(self) -> None:
        record = correction_record(1)
        record["segments"][1]["errors"][0]["wrong"] = "parcels where"
        self.assertTrue(any("does not occur" in error for error in errors_for([record])))

    def test_a_planted_error_must_actually_change_something(self) -> None:
        record = correction_record(1)
        record["segments"][1]["errors"][0]["right"] = "parcels was"
        self.assertTrue(any("must differ" in error for error in errors_for([record])))

    def test_error_subtypes_are_restricted_to_typo_and_grammar(self) -> None:
        record = correction_record(1)
        record["segments"][1]["errors"][0]["subtype"] = "style"
        self.assertTrue(any("not a typo/grammar subtype" in e for e in errors_for([record])))

    def test_triggers_must_leave_room_for_a_legal_tick(self) -> None:
        record = correction_record(1)
        record["segments"][1] = {
            "kind": "passage",
            "text": "The parcells wa left beside the front door this morning.",
            "errors": [
                {"wrong": "parcells", "right": "parcels", "subtype": "doubled-letter", "occurrence": 1},
                {"wrong": "wa", "right": "was", "subtype": "missing-letter", "occurrence": 1},
            ],
            "pause_after": "none",
        }
        self.assertTrue(any("4+ characters" in error for error in errors_for([record])))

    def test_a_passage_without_a_trigger_must_be_labelled_clean(self) -> None:
        record = correction_record(1)
        del record["segments"][1]["errors"]
        self.assertTrue(any("clean_text" in error for error in errors_for([record])))

    def test_a_clean_trap_may_not_carry_a_trigger(self) -> None:
        record = correction_record(1)
        record["segments"][1]["trap"] = "clean_text"
        self.assertTrue(any("may not plant errors" in e for e in errors_for([record])))

    def test_self_correction_must_leave_the_passage_clean(self) -> None:
        record = correction_record(1)
        record["segments"][2]["text"] = "Fixture noted the delivary arrived and the delivery too."
        self.assertTrue(any("still appears" in error for error in errors_for([record])))

    def test_self_correction_words_must_be_long_enough_to_type(self) -> None:
        record = correction_record(1)
        record["segments"][2]["repair"] = {
            "wrong": "teh",
            "right": "the",
            "subtype": "letter-swap",
            "occurrence": 1,
        }
        errors = errors_for([record])
        self.assertTrue(any("must differ" not in e and "4" in e for e in errors), errors)

    def test_asides_must_actually_mention_what_they_claim(self) -> None:
        record = correction_record(1)
        record["segments"][3]["mentions"] = ["definately"]
        self.assertTrue(any("does not appear" in error for error in errors_for([record])))

    def test_every_category_occurrence_must_be_declared_in_order(self) -> None:
        record = highlight_record(1)
        record["segments"][0]["marks"] = []
        errors = errors_for([record])
        self.assertTrue(any("declare every occurrence" in error for error in errors), errors)

        record = highlight_record(2)
        record["segments"][0]["marks"] = [{"word": "otter", "literal": True}]
        self.assertTrue(any("does not match" in error for error in errors_for([record])))

    def test_a_category_word_inside_a_longer_word_is_rejected(self) -> None:
        record = highlight_record(1)
        record["segments"][2] = {
            "kind": "passage",
            "text": "Scene one mentioned herons in the plural and nothing else.",
            "trap": "clean_text",
            "pause_after": "long",
        }
        errors = errors_for([record])
        self.assertTrue(any("inside a longer word" in error for error in errors), errors)

    def test_the_instruction_may_not_contain_a_category_word(self) -> None:
        record = highlight_record(1)
        record["instruction"]["text"] = "Mark every heron I happen to write down here."
        errors = errors_for([record])
        self.assertTrue(any("appears in the instruction" in error for error in errors), errors)

    def test_an_aside_may_not_hide_an_ungraded_category_word(self) -> None:
        record = highlight_record(1)
        record["segments"].append(
            {
                "kind": "aside",
                "text": "I always mistype otter in my own private shorthand.",
                "mentions": ["mistype"],
                "pause_after": "none",
            }
        )
        errors = errors_for([record])
        self.assertTrue(
            any("may not contain category words" in error for error in errors), errors
        )


class Demo2DistributionTests(unittest.TestCase):
    """Gates are proved by breaking a corpus that otherwise passes them.

    The healthy baseline is the checked-in sample fixture (see
    ``Demo2SampleFixtureTests``); each test below mutates exactly one axis and
    asserts the corresponding gate fires.
    """

    def _sample(self) -> list[dict]:
        return [
            {key: value for key, value in item.items() if key != "_path"}
            for item in load_authored_batches(demo2_authored_paths(SAMPLE_ROOT))
        ]

    def _report(self, batches: list[dict]) -> dict:
        return validate_demo2_batches(batches, enforce_distribution=True)

    def test_a_single_author_fails_the_author_share_gate(self) -> None:
        batches = self._sample()
        merged = batch(
            "only-author", [record for item in batches for record in item["records"]]
        )
        self.assertTrue(
            any("3-4 authors" in error for error in self._report([merged])["errors"])
        )

    def test_a_single_mode_fails_the_mode_share_gate(self) -> None:
        batches = self._sample()
        for item in batches:
            item["records"] = [
                record for record in item["records"] if record["mode"] == "corrections"
            ]
        errors = self._report(batches)["errors"]
        self.assertTrue(any("mode 'highlights'" in error for error in errors), errors)

    def test_canned_acknowledgments_fail_the_wording_cap(self) -> None:
        batches = self._sample()
        for item in batches:
            for record in item["records"]:
                record["instruction"]["ack"] = "Got it, that is now handled for you."
        errors = self._report(batches)["errors"]
        self.assertTrue(any("acknowledgment" in error for error in errors), errors)

    def test_one_error_subtype_everywhere_fails_the_subtype_gate(self) -> None:
        batches = self._sample()
        for item in batches:
            for record in item["records"]:
                for segment in record["segments"]:
                    for error in segment.get("errors") or []:
                        error["subtype"] = "agreement"
                    if segment.get("kind") == "repair":
                        segment["repair"]["subtype"] = "agreement"
        errors = self._report(batches)["errors"]
        self.assertTrue(any("error subtype" in error for error in errors), errors)

    def test_one_persona_everywhere_fails_the_persona_gate(self) -> None:
        batches = self._sample()
        for item in batches:
            for record in item["records"]:
                record["persona"] = "one-voice"
        errors = self._report(batches)["errors"]
        self.assertTrue(any("persona" in error for error in errors), errors)

    def test_repeated_seven_word_openings_are_flagged_as_a_template(self) -> None:
        records = [correction_record(index) for index in range(3)]
        for record in records:
            record["segments"][0]["text"] = (
                "The very same seven word opening appears here " f"{record['id']}."
            )
        errors = errors_for(records)
        self.assertTrue(any(error.startswith("template:") for error in errors), errors)


class Demo2NarrativeOpenerGateTests(unittest.TestCase):
    """The passage-opener gate (Fix 1): soft caps on narrative body text.

    Instruction and acknowledgment openers already had caps; nothing watched
    the passages themselves, and the real authored corpus quietly converged
    on "The".  Each test below builds a batch with exactly 24 passages (two
    per record, 12 records) via ``narrative_records`` and puts a precise
    share behind one or more openers.
    """

    def _report(self, openers: list[str]) -> dict:
        return validate_demo2_batches([batch("author-a", narrative_records(openers))])

    def test_a_well_varied_batch_passes_cleanly(self) -> None:
        # Six distinct openers, four passages each (16.7% each, 66.7% top four).
        openers = ["Yesterday", "Marcus", "Priya", "Later", "Quietly", "Somehow"] * 4
        report = self._report(openers)
        self.assertEqual(
            [w for w in report["warnings"] if "narrative passage opener" in w], []
        )
        self.assertEqual(
            [e for e in report["errors"] if "narrative passage opener" in e], []
        )
        self.assertEqual(report["counts"]["passage_openers"]["yesterday"], 4)

    def test_a_share_over_25_percent_warns(self) -> None:
        # "the" covers 9/24 = 37.5% (warn range); top four stay at 62.5%.
        others = [f"opener{i}" for i in range(9)]
        openers = ["The"] * 9 + ["b", "b", "c", "c", "d", "d"] + others
        report = self._report(openers)
        warnings = [w for w in report["warnings"] if "narrative passage opener" in w]
        self.assertTrue(any("'the'" in w for w in warnings), report["warnings"])
        self.assertEqual(
            [e for e in report["errors"] if "narrative passage opener" in e], []
        )

    def test_a_share_over_40_percent_errors(self) -> None:
        # "the" covers 11/24 = 45.8%, over the single-opener error cap.
        others = [f"opener{i}" for i in range(13)]
        openers = ["The"] * 11 + others
        report = self._report(openers)
        errors = [e for e in report["errors"] if "narrative passage opener" in e]
        self.assertTrue(any("'the'" in e and ">40%" in e for e in errors), report["errors"])

    def test_top_four_openers_over_75_percent_is_a_per_file_warning(self) -> None:
        # Four openers evenly split the whole batch (25% each, 100% top four).
        # Per-file top-four concentration is expected -- each author has a
        # distinct persona/register -- so this is only a canary warning, not
        # a build-blocking error; see _narrative_opener_corpus_gate for the
        # error-level check, which runs on the merged corpus instead.
        openers = (["The"] * 6) + (["A"] * 6) + (["It"] * 6) + (["We"] * 6)
        report = self._report(openers)
        warnings = [w for w in report["warnings"] if "top four" in w]
        self.assertTrue(warnings, report["warnings"])
        self.assertEqual([e for e in report["errors"] if "top four" in e], [])

    def test_corpus_wide_top_four_errors_even_when_every_file_only_warns(self) -> None:
        # Two files, each 90% concentrated in the same four openers (a
        # per-file warning, not an error, since no single opener exceeds the
        # 40% error cap). The merged corpus inherits the same 90%
        # concentration and must error -- the top-four cap is a build
        # blocker at corpus scale even though it never errors per file.
        dominant = (["Alpha"] * 9) + (["Bravo"] * 9) + (["Charlie"] * 9) + (["Delta"] * 9)
        file_a = dominant + ["p1", "p2", "p3", "p4"]
        file_b = dominant + ["q1", "q2", "q3", "q4"]
        report = validate_demo2_batches(
            [
                batch("author-a", offset_narrative_records(0, file_a)),
                batch("author-b", offset_narrative_records(1000, file_b)),
            ]
        )
        per_file_errors = [e for e in report["errors"] if "top four" in e and "corpus" not in e]
        per_file_warnings = [w for w in report["warnings"] if "top four" in w]
        corpus_errors = [e for e in report["errors"] if e.startswith("corpus:")]
        self.assertEqual(per_file_errors, [])
        self.assertEqual(len(per_file_warnings), 2, report["warnings"])
        self.assertTrue(corpus_errors, report["errors"])
        self.assertFalse(report["passed"])

    def test_corpus_wide_top_four_stays_clear_when_the_merge_dilutes_it(self) -> None:
        # One file is heavily concentrated (a per-file warning) but the other
        # file's openers are all distinct, so once merged into the corpus the
        # concentrated four no longer cover more than 75% of the whole. The
        # corpus-wide check must not error just because one author file did.
        dominant = (["Alpha"] * 9) + (["Bravo"] * 9) + (["Charlie"] * 9) + (["Delta"] * 9)
        concentrated_file = dominant + ["p1", "p2", "p3", "p4"]
        diverse_file = [f"unique{i}" for i in range(40)]
        report = validate_demo2_batches(
            [
                batch("author-a", offset_narrative_records(0, concentrated_file)),
                batch("author-b", offset_narrative_records(1000, diverse_file)),
            ]
        )
        per_file_warnings = [w for w in report["warnings"] if "top four" in w]
        corpus_errors = [e for e in report["errors"] if e.startswith("corpus:")]
        self.assertEqual(len(per_file_warnings), 1, report["warnings"])
        self.assertEqual(corpus_errors, [])

    def test_corpus_wide_top_four_floor_exempts_a_small_merged_corpus(self) -> None:
        # A single 40-passage file, 90% concentrated -- large enough to run
        # the per-file gate (warns), but the merged corpus (also 40 passages)
        # sits below NARRATIVE_OPENER_CORPUS_FLOOR (60), so the corpus-wide
        # check must not run at all, let alone error.
        dominant = (["Alpha"] * 9) + (["Bravo"] * 9) + (["Charlie"] * 9) + (["Delta"] * 9)
        openers = dominant + ["p1", "p2", "p3", "p4"]
        report = self._report(openers)
        self.assertTrue(
            [w for w in report["warnings"] if "top four" in w], report["warnings"]
        )
        self.assertEqual([e for e in report["errors"] if e.startswith("corpus:")], [])

    def test_below_the_twenty_passage_floor_is_exempt(self) -> None:
        # 9 records = 18 passages, all "The" -- below the floor, gate silent.
        openers = ["The"] * 18
        report = self._report(openers)
        self.assertEqual(
            [w for w in report["warnings"] if "opener" in w], []
        )
        self.assertEqual(
            [e for e in report["errors"] if "opener" in e and "narrative" in e], []
        )


class Demo2SampleFixtureTests(unittest.TestCase):
    def test_the_checked_in_sample_passes_every_gate(self) -> None:
        paths = demo2_authored_paths(SAMPLE_ROOT)
        self.assertTrue(paths)
        report = validate_demo2_batches(
            load_authored_batches(paths), enforce_distribution=True
        )
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["warnings"], [])
        counts = report["counts"]
        self.assertEqual(counts["records"], 12)
        self.assertGreater(counts["errors_planted"], 0)
        self.assertGreater(counts["literal_marks"], 0)
        for key in ("bait_marks", "repairs", "asides", "clean_passages"):
            self.assertGreater(counts[key], 0, key)

    def test_validation_does_not_mutate_the_authored_source(self) -> None:
        batches = load_authored_batches(demo2_authored_paths(SAMPLE_ROOT))
        before = copy.deepcopy(batches)
        validate_demo2_batches(batches, enforce_distribution=True)
        self.assertEqual(batches, before)


if __name__ == "__main__":
    unittest.main()
