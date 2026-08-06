from __future__ import annotations

import unittest

from datagen.g1_authored import validate_demo1_batches


def segment(kind: str, text: str, *, position: int) -> dict[str, object]:
    if kind == "address":
        return {
            "kind": "address",
            "text": text,
            "gold_reply": f"Reply {position}",
            "pause_after": "short",
        }
    return {
        "kind": "narration",
        "text": text,
        "traps": ["none"],
        "pause_after": "none",
    }


def record(
    identifier: str,
    *,
    persona: str,
    domain: str,
    register: str,
    events: int,
    address_position: int,
    text_size: int,
) -> dict[str, object]:
    segments = []
    for position in range(1, events + 1):
        kind = "address" if position == address_position else "narration"
        text = f"{identifier} segment {position} " + ("x" * text_size)
        segments.append(segment(kind, text, position=position))
    return {
        "id": identifier,
        "persona": persona,
        "domain": domain,
        "register": register,
        "segments": segments,
    }


def batch(slot: str, records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "g1-authored-1",
        "demo": "demo-1",
        "author": {"model": "test", "slot": slot, "tranche": 1},
        "records": records,
    }


class G1AuthoredTests(unittest.TestCase):
    def test_validates_common_schema_and_distribution(self) -> None:
        personas = [f"persona-{index}" for index in range(6)]
        domains = [f"domain-{index}" for index in range(6)]
        registers = [f"register-{index}" for index in range(4)]
        records = []
        shapes = [(5, 1, 20), (7, 4, 80), (9, 9, 140)]
        for index in range(30):
            events, address_position, text_size = shapes[index % len(shapes)]
            records.append(
                record(
                    f"record-{index}",
                    persona=personas[index % len(personas)],
                    domain=domains[index % len(domains)],
                    register=registers[index % len(registers)],
                    events=events,
                    address_position=address_position,
                    text_size=text_size,
                )
            )
        report = validate_demo1_batches(
            [
                batch("author-a", records[:10]),
                batch("author-b", records[10:20]),
                batch("author-c", records[20:]),
            ],
            enforce_distribution=True,
        )

        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["counts"]["records"], 30)
        self.assertEqual(report["counts"]["addresses"], 30)
        joints = report["counts"]["joint_distributions"]
        self.assertEqual(sum(joints["event_count_x_trigger"].values()), 30)
        self.assertEqual(sum(joints["address_form_x_trigger"].values()), 30)

    def test_rejects_reserved_categories_and_duplicates(self) -> None:
        first = record(
            "record-a",
            persona="product-reviewer",
            domain="sport",
            register="casual",
            events=5,
            address_position=1,
            text_size=20,
        )
        second = record(
            "record-b",
            persona="journal-keeper",
            domain="cooking",
            register="casual",
            events=5,
            address_position=1,
            text_size=20,
        )
        second["segments"][1]["text"] = first["segments"][1]["text"]  # type: ignore[index]

        report = validate_demo1_batches([batch("author-a", [first, second])])

        self.assertFalse(report["passed"])
        self.assertTrue(any("reserved persona" in error for error in report["errors"]))
        self.assertTrue(any("reserved domain" in error for error in report["errors"]))
        self.assertTrue(any("duplicates" in error for error in report["errors"]))

    def test_rejects_invalid_demo1_shape(self) -> None:
        value = record(
            "record-a",
            persona="journal-keeper",
            domain="cooking",
            register="casual",
            events=5,
            address_position=1,
            text_size=20,
        )
        value["segments"].append(segment("address", "second address", position=6))  # type: ignore[union-attr]

        report = validate_demo1_batches([batch("author-a", [value])])

        self.assertFalse(report["passed"])
        self.assertTrue(any("exactly one address" in error for error in report["errors"]))

    def test_rejects_template_and_trap_distribution_collapse(self) -> None:
        records = []
        for index in range(20):
            value = record(
                f"record-{index}",
                persona=f"persona-{index % 5}",
                domain=f"domain-{index % 5}",
                register=f"register-{index % 4}",
                events=5,
                address_position=5,
                text_size=20,
            )
            for segment_value in value["segments"]:  # type: ignore[union-attr]
                if segment_value["kind"] == "address":
                    segment_value["text"] = f"Would you choose option {index}?"
                else:
                    segment_value["traps"] = ["rhetorical_question"]
                    segment_value["text"] += " Why would anyone disagree?"
            records.append(value)

        report = validate_demo1_batches([batch("author-a", records)])

        self.assertFalse(report["passed"])
        joined = "\n".join(report["errors"])
        self.assertIn("address opener", joined)
        self.assertIn("question-form addresses", joined)
        self.assertIn("reply opener", joined)
        self.assertIn("fewer than 20%", joined)
        self.assertIn("appears in over 50%", joined)
        self.assertIn("trap signature", joined)

    def test_rejects_first_word_speech_act_collapse(self) -> None:
        records = []
        for index in range(20):
            value = record(
                f"record-{index}",
                persona=f"persona-{index % 5}",
                domain=f"domain-{index % 5}",
                register=f"register-{index % 4}",
                events=5,
                address_position=(index % 4) + 1,
                text_size=20,
            )
            address = next(
                segment_value
                for segment_value in value["segments"]  # type: ignore[union-attr]
                if segment_value["kind"] == "address"
            )
            if index < 6:
                address["text"] = f"Keep variation {index} grounded."
            else:
                address["text"] = f"Verb{index} variation {index}."
            records.append(value)

        report = validate_demo1_batches([batch("author-a", records)])

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("address first word 'keep' exceeds 25%" in error for error in report["errors"])
        )

    def test_rejects_event_count_trigger_collapse(self) -> None:
        records = []
        for index in range(20):
            events = 5 if index < 10 else 7
            address_position = 1 if index < 10 else (4 if index % 2 else 7)
            records.append(
                record(
                    f"record-{index}",
                    persona=f"persona-{index % 5}",
                    domain=f"domain-{index % 5}",
                    register=f"register-{index % 4}",
                    events=events,
                    address_position=address_position,
                    text_size=20,
                )
            )

        report = validate_demo1_batches([batch("author-a", records)])

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("event-count records use trigger" in error for error in report["errors"])
        )

    def test_rejects_address_form_trigger_collapse(self) -> None:
        records = []
        for index in range(20):
            address_position = 1 if index < 10 else (4 if index % 2 else 7)
            value = record(
                f"record-{index}",
                persona=f"persona-{index % 5}",
                domain=f"domain-{index % 5}",
                register=f"register-{index % 4}",
                events=7,
                address_position=address_position,
                text_size=20,
            )
            address = next(
                segment_value
                for segment_value in value["segments"]  # type: ignore[union-attr]
                if segment_value["kind"] == "address"
            )
            punctuation = "?" if index < 10 else "."
            address["text"] = f"Verb{index} option {index}{punctuation}"
            records.append(value)

        report = validate_demo1_batches([batch("author-a", records)])

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("'question' addresses use trigger" in error for error in report["errors"])
        )

    def test_flags_missing_trap_labels(self) -> None:
        value = record(
            "record-a",
            persona="journal-keeper",
            domain="cooking",
            register="casual",
            events=5,
            address_position=5,
            text_size=20,
        )
        value["segments"][0].update(  # type: ignore[index]
            {"text": 'Mira asked, "Can we leave now?"', "traps": []}
        )
        value["segments"][1].update(  # type: ignore[index]
            {"text": "The note used highlight as a heading.", "traps": []}
        )
        value["segments"][2].update(  # type: ignore[index]
            {"text": "Sam said the door was open.", "traps": []}
        )

        report = validate_demo1_batches([batch("author-a", [value])])

        self.assertFalse(report["passed"])
        joined = "\n".join(report["errors"])
        self.assertIn("quoted question is missing", joined)
        self.assertIn("command word is missing", joined)
        self.assertTrue(any("reported speech" in item for item in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
