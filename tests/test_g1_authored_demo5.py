from __future__ import annotations

import unittest

from datagen.g1_authored_demo5 import (
    AUTHORED_SCHEMA_VERSION,
    DEMO,
    validate_demo5_batches,
)
from datagen.g1_demo5 import demo5_bank_from_batches, plan_demo5_schedule, compile_demo5_schedule


def _entry(kind: str, index: int, **overrides) -> dict:
    base = {
        "id": f"{kind}-{index}",
        "kind": kind,
        "persona": f"persona-{index % 6}",
        "domain": f"domain-{index % 5}",
        "register": f"register-{index % 4}",
    }
    if kind == "request":
        base.update(
            schedule_kind="every",
            text_template=f"Remind me every {{interval}} seconds to do task {index}!",
            gold_ack_template=f"Got it — task {index} every {{interval}} seconds!",
            fire_message=f"Do task {index}!",
        )
    elif kind == "cancellation":
        base.update(
            text=f"Okay, you can stop reminding me about task {index} now, please.",
            gold_ack=f"Understood — task {index} reminders are off.",
        )
    else:
        base.update(
            trap="none",
            text=f"Back to drafting notes about project {index}, quite a lot is left to cover today.",
        )
    base.update(overrides)
    return base


def _batch(slot: str, entries: list[dict], *, tranche: int = 1) -> dict:
    return {
        "schema_version": AUTHORED_SCHEMA_VERSION,
        "demo": DEMO,
        "author": {"model": "test-model", "slot": slot, "tranche": tranche},
        "bank": entries,
    }


class Demo5AuthoredValidationTests(unittest.TestCase):
    def test_minimal_bank_passes(self) -> None:
        batch = _batch(
            "slot-a",
            [
                _entry("request", 1),
                _entry("request", 2, schedule_kind="once", text_template="Remind me once to text my brother back!", gold_ack_template="Got it — one reminder.", fire_message="Text your brother!"),
                _entry("cancellation", 1),
                _entry("filler", 1),
                _entry("filler", 2, trap="bait", text='Someone asked me, "are you coming or not?" and I just laughed.'),
                _entry("filler", 3, trap="address", text="hey, are you still tracking this for me?", gold_reply="Yes, still tracking!"),
            ],
        )
        report = validate_demo5_batches([batch])
        self.assertTrue(report["passed"], report["errors"])

    def test_every_kind_without_interval_placeholder_is_rejected(self) -> None:
        entry = _entry("request", 1)
        entry["text_template"] = "Remind me every 5 seconds to do task 1!"
        report = validate_demo5_batches([_batch("slot-a", [entry])])
        self.assertFalse(report["passed"])
        self.assertTrue(any("interval" in error for error in report["errors"]))

    def test_once_kind_with_interval_placeholder_is_rejected(self) -> None:
        entry = _entry("request", 1, schedule_kind="once", text_template="Remind me once every {interval} to do it!", gold_ack_template="Got it.", fire_message="Do it!")
        report = validate_demo5_batches([_batch("slot-a", [entry])])
        self.assertFalse(report["passed"])
        self.assertTrue(any("must not contain" in error for error in report["errors"]))

    def test_address_filler_requires_gold_reply(self) -> None:
        entry = _entry("filler", 1, trap="address", text="hey, are you there right now still?")
        report = validate_demo5_batches([_batch("slot-a", [entry])])
        self.assertFalse(report["passed"])
        self.assertTrue(any("gold_reply" in error for error in report["errors"]))

    def test_bait_filler_requires_a_quoted_span(self) -> None:
        entry = _entry("filler", 1, trap="bait", text="Someone asked if I was coming or not and I laughed.")
        report = validate_demo5_batches([_batch("slot-a", [entry])])
        self.assertFalse(report["passed"])
        self.assertTrue(any("quoted span" in error for error in report["errors"]))

    def test_markup_characters_are_rejected(self) -> None:
        entry = _entry("filler", 1)
        entry["text"] = entry["text"] + " <b>bold</b>"
        report = validate_demo5_batches([_batch("slot-a", [entry])])
        self.assertFalse(report["passed"])
        self.assertTrue(any("must not contain '<'" in error for error in report["errors"]))

    def test_ampersand_is_rejected(self) -> None:
        entry = _entry("filler", 1)
        entry["text"] = entry["text"] + " AT&T called."
        report = validate_demo5_batches([_batch("slot-a", [entry])])
        self.assertFalse(report["passed"])
        self.assertTrue(any("&" in error for error in report["errors"]))

    def test_duplicate_phrasing_is_rejected(self) -> None:
        entries = [_entry("filler", 1), _entry("filler", 2)]
        entries[1]["text"] = entries[0]["text"]
        report = validate_demo5_batches([_batch("slot-a", entries)])
        self.assertFalse(report["passed"])
        self.assertTrue(any("duplicates" in error for error in report["errors"]))

    def test_reserved_persona_is_rejected(self) -> None:
        entry = _entry("filler", 1, persona="product-reviewer")
        report = validate_demo5_batches([_batch("slot-a", [entry])])
        self.assertFalse(report["passed"])
        self.assertTrue(any("reserved persona" in error for error in report["errors"]))

    def test_reserved_domain_is_rejected(self) -> None:
        entry = _entry("filler", 1, domain="health")
        report = validate_demo5_batches([_batch("slot-a", [entry])])
        self.assertFalse(report["passed"])
        self.assertTrue(any("reserved domain" in error for error in report["errors"]))

    def test_short_corpus_skips_distribution_gates(self) -> None:
        batch = _batch(
            "slot-a",
            [_entry("request", 1), _entry("cancellation", 1), _entry("filler", 1)],
        )
        report = validate_demo5_batches([batch], enforce_distribution=True)
        self.assertFalse(report["distribution_enforced"])
        self.assertTrue(any("below the" in error for error in report["errors"]))


def _distribution_bank(count: int) -> list[dict]:
    entries = []
    for index in range(count):
        entries.append(_entry("request", index, schedule_kind="every" if index % 2 else "once"))
        if entries[-1]["schedule_kind"] == "once":
            entries[-1].update(
                text_template=f"Remind me once to handle task {index} today!",
                gold_ack_template=f"Got it — task {index} once.",
            )
        entries.append(_entry("cancellation", index))
        entries.append(
            _entry(
                "filler",
                index,
                text=(
                    f"Back to drafting notes about project {index}, there is quite a lot left to "
                    "cover today and I keep losing my train of thought whenever the phone buzzes."
                ),
            )
        )
        entries.append(_entry("filler", index + 1000, trap="bait", text=f'Someone asked, "is task {index} done?" and I shrugged.'))
        entries.append(
            _entry("filler", index + 2000, trap="address", text=f"hey, tracking task {index} still?", gold_reply=f"Yes, tracking task {index}!")
        )
    return entries


class Demo5DistributionGateTests(unittest.TestCase):
    def test_well_spread_bank_passes_distribution_gates(self) -> None:
        entries = _distribution_bank(6)
        third = len(entries) // 3
        batches = [
            _batch("slot-a", entries[:third], tranche=1),
            _batch("slot-b", entries[third : 2 * third], tranche=1),
            _batch("slot-c", entries[2 * third :], tranche=1),
        ]
        report = validate_demo5_batches(batches, enforce_distribution=True)
        self.assertTrue(report["distribution_enforced"])
        self.assertTrue(report["passed"], report["errors"])

    def test_author_share_outside_band_is_rejected(self) -> None:
        entries = _distribution_bank(5)
        batches = [
            _batch("slot-a", entries[:2], tranche=1),
            _batch("slot-b", entries[2:23], tranche=1),
            _batch("slot-c", entries[23:], tranche=1),
        ]
        report = validate_demo5_batches(batches, enforce_distribution=True)
        self.assertFalse(report["passed"])
        self.assertTrue(any("share must be" in error for error in report["errors"]))

    def test_missing_schedule_kind_variety_is_rejected(self) -> None:
        entries = []
        for index in range(30):
            entries.append(
                _entry("request", index, persona=f"persona-{index % 6}", domain=f"domain-{index % 5}", register=f"register-{index % 4}")
            )
        batches = [
            _batch("slot-a", entries[:10], tranche=1),
            _batch("slot-b", entries[10:20], tranche=1),
            _batch("slot-c", entries[20:], tranche=1),
        ]
        report = validate_demo5_batches(batches, enforce_distribution=True)
        self.assertFalse(report["passed"])
        self.assertTrue(any("once" in error for error in report["errors"]))


class Demo5BankFlattenTests(unittest.TestCase):
    def test_flattened_bank_compiles(self) -> None:
        batch = _batch(
            "slot-a",
            [
                _entry("request", 1),
                _entry(
                    "request",
                    2,
                    schedule_kind="once",
                    text_template="Remind me once to text my brother back tonight!",
                    gold_ack_template="Got it — one reminder to text your brother.",
                    fire_message="Text your brother!",
                ),
                _entry("cancellation", 1),
                _entry("filler", 1),
                _entry("filler", 2, trap="bait", text='Someone asked me, "are you coming or not?" and I just laughed.'),
                _entry("filler", 3, trap="address", text="hey, are you still tracking this for me?", gold_reply="Yes, still tracking!"),
            ],
        )
        report = validate_demo5_batches([batch])
        self.assertTrue(report["passed"], report["errors"])
        bank = demo5_bank_from_batches([batch])
        self.assertEqual(len(bank.requests), 2)
        self.assertEqual(len(bank.cancellations), 1)
        self.assertEqual(len(bank.fillers_plain), 1)
        self.assertEqual(len(bank.fillers_bait), 1)
        self.assertEqual(len(bank.fillers_address), 1)
        config = plan_demo5_schedule("demo5-flat-01", bank)
        compiled = compile_demo5_schedule(config)
        self.assertTrue(compiled.turns)


if __name__ == "__main__":
    unittest.main()
