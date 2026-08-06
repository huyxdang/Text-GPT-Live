from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from datagen.g1_authored_demo3 import (
    ACCEPTED_REVIEW_METHODS,
    AUTHORED_SCHEMA_VERSION,
    DISTRIBUTION_FLOOR,
    authored_paths,
    check_chinese_reference,
    load_authored_batches,
    validate_demo3_batches,
)
from datagen.g1_demo3 import CLAUSE_MARKER as M


HAN_DIGITS = "零一二三四五六七八九"
NOUNS = (
    ("market", "市场"),
    ("garden", "花园"),
    ("station", "车站"),
    ("kitchen", "厨房"),
    ("harbour", "港口"),
    ("gallery", "画廊"),
    ("workshop", "工坊"),
    ("orchard", "果园"),
)
ADJECTIVES = (
    ("quiet", "安静"),
    ("busy", "热闹"),
    ("bright", "明亮"),
    ("damp", "潮湿"),
    ("empty", "空荡"),
    ("warm", "温暖"),
)
OPENERS_ZH = (
    "随后某处响起铃声",
    "旁边一扇门被推开",
    "有人重重放下木箱",
    "风忽然大了起来",
    "远处传来低低的说话声",
    "台阶上落满了细碎的光",
    "水管在墙后缓缓作响",
    "屋檐下有鸟一直不肯走",
)
INSTRUCTION_VERBS = (
    "Translate",
    "Render",
    "Turn",
    "Convert",
)
ACK_OPENERS_ZH = (
    "好的，",
    "明白，",
    "收到，",
    "没问题，",
    "可以，",
    "了解，",
    "行，",
    "清楚了，",
)
TAIL_EN = (
    ", and the whole street kept its slow and steady rhythm "
    "without any sudden interruption at all"
)
TAIL_ZH = "，整条街依旧保持着缓慢而稳定的节奏，没有任何突然的打断"
#: Sentence-opener leads for the English half of each clause.  Spread across
#: eight distinct first words so that large fixtures (e.g. the 30-record
#: ``_wide_corpus``) do not themselves collapse onto a single narrative
#: opener and trip the opener-distribution gate under test.
ENGLISH_LEADS = ("On", "By", "Near", "After", "Before", "Around", "Since", "Toward")


def _clause(index: int, clause_no: int, *, front_loaded: bool) -> tuple[str, str]:
    """One English clause and its Chinese reference, unique across the corpus."""

    noun, noun_zh = NOUNS[(index + clause_no) % len(NOUNS)]
    adjective, adjective_zh = ADJECTIVES[(index * 2 + clause_no) % len(ADJECTIVES)]
    opener_zh = OPENERS_ZH[(index * 3 + clause_no * 5) % len(OPENERS_ZH)]
    lead = ENGLISH_LEADS[(index + clause_no) % len(ENGLISH_LEADS)]
    english = (
        f"{lead} day {index} at stop {clause_no} the {noun} "
        f"looked {adjective} to everyone nearby"
    )
    reference = (
        f"第{HAN_DIGITS[index % 10]}{HAN_DIGITS[clause_no % 10]}号，"
        f"{opener_zh}，那里显得{adjective_zh}"
    )
    if front_loaded:
        english += TAIL_EN
        reference += TAIL_ZH
    return english + ".", reference + "。"


def _record(index: int) -> dict:
    """A distinct, valid authored episode; every gated field varies with `index`.

    This fixture is deliberately built to *pass* the full distribution gate at
    scale, which is the only way the negative tests below prove anything: a
    corpus that failed for unrelated reasons would make every assertion vacuous.
    """

    clause_count = (2, 5, 8)[index % 3]
    front_loaded = index % 2 == 1
    trap = ("none", "half_clause", "backtrack", "prequoted_chinese")[index % 4]
    noun, noun_zh = NOUNS[index % len(NOUNS)]

    script: list[dict] = []
    clauses: list[dict] = []
    for clause_no in range(clause_count):
        english, reference = _clause(
            index, clause_no, front_loaded=front_loaded and clause_no == 0
        )
        script.append(
            {
                "kind": "type",
                "text": ("" if clause_no == 0 else " ") + english + M,
                "pause_after": ("short", "medium", "long", "none")[clause_no % 4],
            }
        )
        clauses.append({"english": english, "reference": reference})

    if trap == "half_clause":
        # Split the final clause across two steps, leaving one mid-clause.
        final = script[-1]["text"]
        cut = len(final) // 2
        script[-1] = {
            "kind": "type",
            "text": final[:cut],
            "trap": "half_clause",
            "pause_after": "medium",
        }
        script.append({"kind": "type", "text": final[cut:], "pause_after": "short"})
    elif trap == "backtrack":
        # A false start after the first commit, deleted before it can commit.
        script.insert(
            1, {"kind": "type", "text": " a false start here", "pause_after": "none"}
        )
        script.insert(
            2, {"kind": "backtrack", "delete": len(" a false start here"), "pause_after": "short"}
        )
    elif trap == "prequoted_chinese":
        script.append(
            {
                "kind": "type",
                "text": f" A sign by the {noun} read 欢迎 on day {index}.",
                "trap": "prequoted_chinese",
                "pause_after": "short",
            }
        )

    verb = INSTRUCTION_VERBS[index % len(INSTRUCTION_VERBS)]
    return {
        "id": f"demo3-fixture-{index:03d}",
        "persona": f"persona-{index % 6}",
        "domain": f"domain-{index % 5}",
        "register": f"register-{index % 4}",
        "instruction": {
            "text": f"{verb} what I write about the {noun} on day {index} into Chinese.",
            "gold_ack": (
                f"{ACK_OPENERS_ZH[index % len(ACK_OPENERS_ZH)]}"
                f"第{index}天的{noun_zh}内容我会边写边译。"
            ),
            "pause_after": ("none", "short", "medium", "long")[index % 4],
        },
        "script": script,
        "clauses": clauses,
    }


def _batch(slot: str, records: list[dict], **overrides) -> dict:
    batch = {
        "schema_version": AUTHORED_SCHEMA_VERSION,
        "demo": "demo-3",
        "author": {"model": "test-model", "slot": slot, "tranche": 1},
        "reference_review": {
            "method": "machine",
            "reviewer": "test-reviewer",
            "reviewed_at": "2026-08-01",
        },
        "records": records,
    }
    batch.update(overrides)
    return batch


def _corpus(count: int = 3) -> list[dict]:
    return [_batch("slot-a", [_record(0), _record(1)]), _batch("slot-b", [_record(2)])][
        : max(1, count - 1)
    ]


def _narrative_clause(index: int, clause_no: int, lead: str) -> tuple[str, str]:
    """A two-clause fixture's English/Chinese pair with an exact opening word."""

    noun, noun_zh = NOUNS[(index + clause_no) % len(NOUNS)]
    adjective, adjective_zh = ADJECTIVES[(index * 2 + clause_no) % len(ADJECTIVES)]
    opener_zh = OPENERS_ZH[(index * 3 + clause_no * 5) % len(OPENERS_ZH)]
    english = (
        f"{lead} day {index} at stop {clause_no} the {noun} "
        f"looked {adjective} to everyone nearby"
    )
    reference = (
        f"第{HAN_DIGITS[index % 10]}{HAN_DIGITS[clause_no % 10]}号，"
        f"{opener_zh}，那里显得{adjective_zh}"
    )
    return english + ".", reference + "。"


def _narrative_record(index: int, leads: tuple[str, str]) -> dict:
    """A minimal, valid two-clause record whose clause openers are exact."""

    clauses: list[dict] = []
    script: list[dict] = []
    for clause_no, lead in enumerate(leads):
        english, reference = _narrative_clause(index, clause_no, lead)
        script.append(
            {
                "kind": "type",
                "text": ("" if clause_no == 0 else " ") + english + M,
                "pause_after": "short" if clause_no == 0 else "none",
            }
        )
        clauses.append({"english": english, "reference": reference})
    noun, noun_zh = NOUNS[index % len(NOUNS)]
    verb = INSTRUCTION_VERBS[index % len(INSTRUCTION_VERBS)]
    return {
        "id": f"nar3-{index:04d}",
        "persona": f"persona-{index % 6}",
        "domain": f"domain-{index % 5}",
        "register": f"register-{index % 4}",
        "instruction": {
            "text": f"{verb} what I write about the {noun} number {index} into Chinese.",
            "gold_ack": (
                f"{ACK_OPENERS_ZH[index % len(ACK_OPENERS_ZH)]}"
                f"第{index}天的{noun_zh}内容我会边写边译。"
            ),
            "pause_after": "short",
        },
        "script": script,
        "clauses": clauses,
    }


def narrative_clause_records(openers: list[str]) -> list[dict]:
    """Two-clause records whose clauses open with chosen words, in order.

    Mirrors ``narrative_records`` in ``test_g1_authored_demo2.py``: each
    record contributes exactly two clauses -- ``openers[2*i]`` and
    ``openers[2*i + 1]`` -- so a test can put an exact number of clauses
    behind any given opening word regardless of how records happen to group.
    """

    records: list[dict] = []
    for pair_index in range(0, len(openers), 2):
        lead_a = openers[pair_index]
        lead_b = openers[pair_index + 1] if pair_index + 1 < len(openers) else "Later"
        index = pair_index // 2
        records.append(_narrative_record(index, (lead_a, lead_b)))
    return records


def _offset_narrative_records(index_offset: int, openers: list[str]) -> list[dict]:
    """Like ``narrative_clause_records``, but with the record index shifted.

    Needed whenever a test builds more than one batch: ``_narrative_record``
    derives both the record id and the clause/instruction text from its
    index, so two batches that each start counting from zero would collide
    on duplicate ids, duplicate clause text, and duplicate instruction
    wording.  A large, distinct offset per batch keeps every batch's records
    unique.
    """

    records: list[dict] = []
    for pair_index in range(0, len(openers), 2):
        lead_a = openers[pair_index]
        lead_b = openers[pair_index + 1] if pair_index + 1 < len(openers) else "Later"
        index = index_offset + pair_index // 2
        records.append(_narrative_record(index, (lead_a, lead_b)))
    return records


class ChineseReferenceCheckTests(unittest.TestCase):
    def test_a_good_reference_passes(self) -> None:
        self.assertEqual(
            check_chinese_reference(
                "今天早上市场很拥挤，",
                "The market was crowded this morning,",
                location="ref",
            ),
            [],
        )

    def test_missing_han_is_rejected(self) -> None:
        errors = check_chinese_reference("...", "Some English text here.", location="ref")
        self.assertTrue(any("no Han characters" in error for error in errors))

    def test_english_echoed_back_is_rejected(self) -> None:
        english = "The market was crowded this morning,"
        errors = check_chinese_reference(english, english, location="ref")
        self.assertTrue(any("byte-identical" in error for error in errors))

    def test_untranslated_latin_run_is_rejected_unless_allowlisted(self) -> None:
        errors = check_chinese_reference(
            "我打开了Excel表格。", "I opened the spreadsheet program.", location="ref"
        )
        self.assertTrue(any("Latin run" in error for error in errors))
        self.assertEqual(
            check_chinese_reference(
                "我打开了Excel表格。",
                "I opened the spreadsheet program.",
                location="ref",
                latin_allowlist=["Excel"],
            ),
            [],
        )

    def test_half_width_punctuation_is_rejected(self) -> None:
        errors = check_chinese_reference(
            "今天早上市场很拥挤,", "The market was crowded this morning,", location="ref"
        )
        self.assertTrue(any("half-width" in error for error in errors))

    def test_digits_keep_their_ascii_punctuation(self) -> None:
        self.assertEqual(
            check_chinese_reference(
                "圆周率约为3.14。", "Pi is roughly three point one four.", location="ref"
            ),
            [],
        )

    def test_action_markup_and_angle_brackets_are_rejected(self) -> None:
        errors = check_chinese_reference(
            "<action>idle()</action>今天很好。",
            "Everything went fine today, honestly.",
            location="ref",
        )
        self.assertTrue(any("markup" in error for error in errors))
        self.assertTrue(any("renders" in error for error in errors))

    def test_implausible_length_ratio_is_rejected(self) -> None:
        errors = check_chinese_reference(
            "好。",
            "This is a very long English clause that carries a great deal of meaning.",
            location="ref",
        )
        self.assertTrue(any("length ratio" in error for error in errors))

    def test_surrounding_whitespace_is_rejected(self) -> None:
        errors = check_chinese_reference(
            " 今天早上市场很拥挤，", "The market was crowded this morning,", location="ref"
        )
        self.assertTrue(any("whitespace" in error for error in errors))


class Demo3BatchValidationTests(unittest.TestCase):
    def test_a_clean_corpus_passes(self) -> None:
        report = validate_demo3_batches(_corpus())
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["counts"]["records"], 3)
        # Records 0/1/2 commit 2, 5 and 8 clauses respectively.
        self.assertEqual(report["counts"]["clauses"], 15)
        self.assertFalse(report["distribution_enforced"])

    def test_reserved_personas_and_domains_are_rejected(self) -> None:
        for field, value in (
            ("persona", "product-reviewer"),
            ("persona", "letter-to-a-friend"),
            ("persona", "technical-writeup"),
            ("domain", "sport"),
            ("domain", "health"),
            ("domain", "personal finance"),
        ):
            batches = _corpus()
            batches[0]["records"][0][field] = value
            report = validate_demo3_batches(batches)
            self.assertFalse(report["passed"], f"{field}={value}")
            self.assertTrue(
                any("reserved" in error for error in report["errors"]),
                f"{field}={value}",
            )

    def test_human_review_claim_is_rejected(self) -> None:
        batches = _corpus()
        batches[0]["reference_review"]["method"] = "human"
        report = validate_demo3_batches(batches)
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("false provenance" in error for error in report["errors"]),
            report["errors"],
        )
        self.assertEqual(ACCEPTED_REVIEW_METHODS, {"machine"})

    def test_missing_reference_review_is_rejected(self) -> None:
        batches = _corpus()
        del batches[0]["reference_review"]
        report = validate_demo3_batches(batches)
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("not optional" in error for error in report["errors"]), report["errors"]
        )

    def test_duplicate_reference_across_clauses_is_rejected(self) -> None:
        batches = _corpus()
        batches[0]["records"][1]["clauses"][0]["reference"] = batches[0]["records"][0][
            "clauses"
        ][0]["reference"]
        report = validate_demo3_batches(batches)
        self.assertTrue(
            any("byte-identical to" in error for error in report["errors"]),
            report["errors"],
        )

    def test_duplicate_english_clause_is_rejected(self) -> None:
        batches = _corpus()
        batches[0]["records"][1]["clauses"][0]["english"] = batches[0]["records"][0][
            "clauses"
        ][0]["english"]
        report = validate_demo3_batches(batches)
        self.assertFalse(report["passed"])

    def test_han_in_a_committable_clause_is_rejected(self) -> None:
        batches = _corpus()
        record = batches[0]["records"][0]
        record["script"][0]["text"] = f"The sign said 欢迎 by the gate number nine,{M}"
        record["clauses"][0]["english"] = "The sign said 欢迎 by the gate number nine,"
        report = validate_demo3_batches(batches)
        self.assertTrue(
            any("prequoted_chinese step" in error for error in report["errors"]),
            report["errors"],
        )

    def test_planner_failures_surface_as_validation_errors(self) -> None:
        batches = _corpus()
        batches[0]["records"][0]["clauses"][0]["english"] = "mismatched"
        report = validate_demo3_batches(batches)
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("does not match the marked span" in error for error in report["errors"])
        )

    def test_uniform_cadence_episode_is_rejected(self) -> None:
        batches = _corpus()
        for step in batches[0]["records"][0]["script"]:
            step["pause_after"] = "none"
        report = validate_demo3_batches(batches)
        self.assertTrue(
            any("clock is furniture" in error for error in report["errors"]),
            report["errors"],
        )

    def test_duplicate_instruction_phrasing_is_rejected(self) -> None:
        batches = _corpus()
        batches[0]["records"][1]["instruction"]["text"] = batches[0]["records"][0][
            "instruction"
        ]["text"]
        report = validate_demo3_batches(batches)
        self.assertTrue(
            any("duplicates" in error for error in report["errors"]), report["errors"]
        )

    def test_markup_in_instruction_is_rejected(self) -> None:
        batches = _corpus()
        batches[0]["records"][0]["instruction"]["gold_ack"] = "<action>idle()</action>"
        report = validate_demo3_batches(batches)
        self.assertTrue(any("action markup" in error for error in report["errors"]))

    def test_repeated_english_prefix_is_flagged_as_a_template(self) -> None:
        batches = _corpus()
        records = [_record(index) for index in range(3, 7)]
        for index, record in enumerate(records):
            record["clauses"][0]["english"] = (
                f"One quiet afternoon in the long summer of {index}"
            )
            record["script"][0]["text"] = (
                f"One quiet afternoon in the long summer of {index}{M}"
            )
        batches.append(_batch("slot-c", records))
        report = validate_demo3_batches(batches)
        self.assertTrue(
            any("7-word prefix" in error for error in report["errors"]), report["errors"]
        )


class Demo3DistributionGateTests(unittest.TestCase):
    def _wide_corpus(self, count: int = 30) -> list[dict]:
        records = [_record(index) for index in range(count)]
        per_slot = len(records) // 3
        return [
            _batch("slot-a", records[:per_slot]),
            _batch("slot-b", records[per_slot : 2 * per_slot]),
            _batch("slot-c", records[2 * per_slot :]),
        ]

    def test_the_full_gate_is_jointly_satisfiable(self) -> None:
        """A corpus that satisfies every gate at once must exist, or the fleet
        is being sent at an impossible target."""

        report = validate_demo3_batches(self._wide_corpus(), enforce_distribution=True)
        self.assertTrue(report["passed"], report["errors"])
        self.assertTrue(report["distribution_enforced"])
        self.assertEqual(report["warnings"], [])

    def test_small_corpus_cannot_silently_pass_the_gates(self) -> None:
        report = validate_demo3_batches(_corpus(), enforce_distribution=True)
        self.assertFalse(report["passed"])
        self.assertFalse(report["distribution_enforced"])
        self.assertTrue(
            any("below the" in error for error in report["errors"]), report["errors"]
        )

    def test_single_author_corpus_is_rejected(self) -> None:
        records = [_record(index) for index in range(DISTRIBUTION_FLOOR + 6)]
        report = validate_demo3_batches(
            [_batch("only-slot", records)], enforce_distribution=True
        )
        self.assertTrue(
            any("3-4 authors" in error for error in report["errors"]), report["errors"]
        )

    def test_dominant_persona_is_rejected(self) -> None:
        batches = self._wide_corpus()
        for batch in batches:
            for record in batch["records"]:
                record["persona"] = "one-voice"
        report = validate_demo3_batches(batches, enforce_distribution=True)
        self.assertTrue(
            any("persona" in error and "exceeds 35%" in error for error in report["errors"])
            or any("at least 5 buckets" in error for error in report["errors"]),
            report["errors"],
        )

    def test_missing_trap_class_is_rejected(self) -> None:
        batches = self._wide_corpus()
        for batch in batches:
            for record in batch["records"]:
                record["script"] = [
                    step
                    for step in record["script"]
                    if step.get("trap") != "prequoted_chinese"
                ]
        report = validate_demo3_batches(batches, enforce_distribution=True)
        self.assertTrue(
            any("'prequoted_chinese' trap" in error for error in report["errors"]),
            report["errors"],
        )

    def test_canned_acknowledgment_wording_is_rejected(self) -> None:
        batches = self._wide_corpus()
        for batch in batches:
            for record in batch["records"]:
                record["instruction"]["gold_ack"] = "好的，我会边打边翻译。"
        report = validate_demo3_batches(batches, enforce_distribution=True)
        self.assertTrue(
            any("acknowledgment" in error for error in report["errors"]),
            report["errors"],
        )

    def test_report_covers_every_required_distribution(self) -> None:
        report = validate_demo3_batches(self._wide_corpus(), enforce_distribution=True)
        counts = report["counts"]
        for field in (
            "agent",
            "persona",
            "domain",
            "register",
            "length_bucket",
            "clause_count_bucket",
            "trigger_position",
            "opening_shape",
            "closing_shape",
            "traps",
            "trap_signatures",
            "ack_openers",
            "instruction_openers",
            "joint_distributions",
            "reference_review_method",
        ):
            self.assertIn(field, counts)
        self.assertTrue(counts["ack_openers"], "Chinese acks must still be counted")
        self.assertEqual(set(counts["reference_review_method"]), {"machine"})


class Demo3NarrativeOpenerGateTests(unittest.TestCase):
    """The clause-opener gate (Fix 1): soft caps on narrative clause text.

    Acknowledgment and instruction openers already had caps; nothing watched
    the clauses themselves, and the real authored corpus quietly converged
    on "The".  Each test below builds a batch with exactly 24 clauses (two
    per record, 12 records) via ``narrative_clause_records`` and puts a
    precise share behind one or more openers.
    """

    def _report(self, openers: list[str]) -> dict:
        return validate_demo3_batches([_batch("slot-a", narrative_clause_records(openers))])

    def test_a_well_varied_batch_passes_cleanly(self) -> None:
        # Six distinct openers, four clauses each (16.7% each, 66.7% top four).
        openers = ["Yesterday", "Marcus", "Priya", "Later", "Quietly", "Somehow"] * 4
        report = self._report(openers)
        self.assertEqual([w for w in report["warnings"] if "clause opener" in w], [])
        self.assertEqual([e for e in report["errors"] if "clause opener" in e], [])
        self.assertEqual(report["counts"]["clause_openers"]["yesterday"], 4)

    def test_a_share_over_25_percent_warns(self) -> None:
        # "the" covers 9/24 = 37.5% (warn range); top four stay at 62.5%.
        others = [f"opener{i}" for i in range(9)]
        openers = ["The"] * 9 + ["b", "b", "c", "c", "d", "d"] + others
        report = self._report(openers)
        warnings = [w for w in report["warnings"] if "clause opener" in w]
        self.assertTrue(any("'the'" in w for w in warnings), report["warnings"])
        self.assertEqual([e for e in report["errors"] if "clause opener" in e], [])

    def test_a_share_over_40_percent_errors(self) -> None:
        # "the" covers 11/24 = 45.8%, over the single-opener error cap.
        others = [f"opener{i}" for i in range(13)]
        openers = ["The"] * 11 + others
        report = self._report(openers)
        errors = [e for e in report["errors"] if "clause opener" in e]
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
        report = validate_demo3_batches(
            [
                _batch("slot-a", _offset_narrative_records(0, file_a)),
                _batch("slot-b", _offset_narrative_records(1000, file_b)),
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
        report = validate_demo3_batches(
            [
                _batch("slot-a", _offset_narrative_records(0, concentrated_file)),
                _batch("slot-b", _offset_narrative_records(1000, diverse_file)),
            ]
        )
        per_file_warnings = [w for w in report["warnings"] if "top four" in w]
        corpus_errors = [e for e in report["errors"] if e.startswith("corpus:")]
        self.assertEqual(len(per_file_warnings), 1, report["warnings"])
        self.assertEqual(corpus_errors, [])

    def test_corpus_wide_top_four_floor_exempts_a_small_merged_corpus(self) -> None:
        # A single 40-clause file, 90% concentrated -- large enough to run
        # the per-file gate (warns), but the merged corpus (also 40 clauses)
        # sits below NARRATIVE_OPENER_CORPUS_FLOOR (60), so the corpus-wide
        # check must not run at all, let alone error.
        dominant = (["Alpha"] * 9) + (["Bravo"] * 9) + (["Charlie"] * 9) + (["Delta"] * 9)
        openers = dominant + ["p1", "p2", "p3", "p4"]
        report = self._report(openers)
        self.assertEqual(report["counts"]["clauses"], 40)
        self.assertTrue(
            [w for w in report["warnings"] if "top four" in w], report["warnings"]
        )
        self.assertEqual([e for e in report["errors"] if e.startswith("corpus:")], [])

    def test_below_the_twenty_clause_floor_is_exempt(self) -> None:
        # 9 records = 18 clauses, all "The" -- below the floor, gate silent.
        openers = ["The"] * 18
        report = self._report(openers)
        self.assertEqual([w for w in report["warnings"] if "opener" in w], [])
        self.assertEqual(
            [e for e in report["errors"] if "opener" in e and "clause" in e], []
        )

    def test_a_leading_coordinating_conjunction_is_stripped(self) -> None:
        # "And" is not itself counted; the real word after it is.
        openers = ["And market", "And market"] * 12
        report = self._report(openers)
        self.assertNotIn("and", report["counts"]["clause_openers"])
        self.assertEqual(report["counts"]["clause_openers"].get("market"), 24)
        errors = [e for e in report["errors"] if "clause opener" in e]
        self.assertTrue(any("'market'" in e for e in errors), errors)


class AuthoredPathTests(unittest.TestCase):
    def test_paths_and_loading_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "demo3").mkdir()
            payload = _batch("slot-a", [_record(0)])
            (root / "demo3" / "one.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            paths = authored_paths(root)
            self.assertEqual(len(paths), 1)
            batches = load_authored_batches(paths)
            self.assertEqual(batches[0]["author"]["slot"], "slot-a")
            self.assertIn("_path", batches[0])


class SampleCorpusTests(unittest.TestCase):
    """The checked-in hand-written fixture must stay valid and compilable."""

    ROOT = Path(__file__).resolve().parent.parent / "data" / "g1_authored_demo3_sample"

    def test_sample_corpus_validates(self) -> None:
        paths = authored_paths(self.ROOT)
        self.assertTrue(paths, "sample corpus is missing")
        report = validate_demo3_batches(load_authored_batches(paths))
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["warnings"], [])
        self.assertEqual(report["counts"]["records"], 4)
        self.assertEqual(
            set(report["counts"]["traps"]),
            {"half_clause", "backtrack", "prequoted_chinese"},
        )
        shapes = Counter(report["counts"]["opening_shape"])
        self.assertEqual(len(shapes), 4, "fixture must exercise every opening skeleton")

    def test_sample_corpus_is_not_mistaken_for_production_source(self) -> None:
        """The hand-written fixture batches carry unmistakable markers --
        ``author.model == "hand-written-fixture"`` and record ids prefixed
        ``"demo3-sample-"`` (see the files under ``g1_authored_demo3_sample``).
        Production authored source must never carry either marker; that is
        what would happen if a fixture batch were copied/published into the
        production tree instead of staying a machinery fixture.

        This used to key off a generic ``note`` field, which was a false
        proxy: real authored tranches legitimately use ``note`` for their own
        descriptions (e.g. ``d3_slot_b_01.json``'s "Tranche 1 of author slot
        d3-slot-b."), so once real production data existed the old assertion
        failed on ordinary, non-fixture files. The fixture-specific markers
        below are what actually distinguish a leaked sample from real source.
        """
        fixture_paths = authored_paths(self.ROOT)
        self.assertTrue(fixture_paths, "sample corpus is missing")
        fixture_record_ids: set[str] = set()
        for path in fixture_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload.get("author", {}).get("model"),
                "hand-written-fixture",
                f"{path} no longer carries the fixture author marker this "
                "test relies on",
            )
            for record in payload.get("records", []):
                record_id = record.get("id")
                if isinstance(record_id, str):
                    fixture_record_ids.add(record_id)
        self.assertTrue(fixture_record_ids)
        self.assertTrue(
            all(rid.startswith("demo3-sample-") for rid in fixture_record_ids)
        )

        production = (
            Path(__file__).resolve().parent.parent / "data" / "g1_authored" / "demo3"
        )
        if not production.exists():
            return
        for path in production.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            author = payload.get("author") or {}
            self.assertNotEqual(
                author.get("model"),
                "hand-written-fixture",
                f"{path} carries the fixture author marker; a sample batch "
                "appears to have been published into the production source tree",
            )
            leaked = {
                record.get("id")
                for record in payload.get("records", [])
                if isinstance(record.get("id"), str)
                and record["id"].startswith("demo3-sample-")
            } & fixture_record_ids
            self.assertFalse(
                leaked,
                f"{path} contains fixture record ids {sorted(leaked)}; a "
                "sample batch appears to have been published into the "
                "production source tree",
            )


if __name__ == "__main__":
    unittest.main()
