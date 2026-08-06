from __future__ import annotations

import unittest

from datagen.g1_authored_demo4 import validate_demo4_batches


REQUEST_TEMPLATES = [
    "Topic {i}: chart please?",
    "Item {i}: quick chart please, thanks!",
    "Board for {i}, could you make one soon please?",
    "Item {i} — could you sketch me a quick view of it and its recent trend?",
    "Thing {i} needs a small panel, would you mind building one whenever you get a chance today?",
    "Number {i} could use a tiny board showing how it has been trending lately please.",
]
CHECK_TEMPLATES = [
    "how's it going {j}",
    "is it done {j}",
    "any progress {j}",
    "status check {j}",
]
CHECK_REPLIES = [
    "Still working on it ({i}).",
    "Almost there, hang tight ({i}).",
    "Not yet — a bit longer ({i}).",
    "Working on that now ({i}).",
]
NUDGE_TEMPLATES = [
    "got that {j}?",
    "you there {j}?",
    "did it start {j}?",
    "still around {j}?",
]


def _request(index: int, *, persona: str = "journal-keeper", domain: str = "cooking",
             register: str = "casual") -> dict:
    text = REQUEST_TEMPLATES[index % len(REQUEST_TEMPLATES)].format(i=index)
    return {
        "id": f"req-{index:03d}",
        "text": text,
        "task": f"generate a UI to visualize topic {index}",
        "persona": persona,
        "domain": domain,
        "register": register,
    }


def _check(index: int, *, persona: str = "journal-keeper", domain: str = "cooking",
           register: str = "casual") -> dict:
    return {
        "id": f"check-{index:03d}",
        "kind": "check",
        "question": CHECK_TEMPLATES[index % len(CHECK_TEMPLATES)].format(j=index % 9),
        "reply": CHECK_REPLIES[index % len(CHECK_REPLIES)].format(i=index),
        "persona": persona,
        "domain": domain,
        "register": register,
    }


def _nudge(index: int, *, persona: str = "journal-keeper", domain: str = "cooking",
           register: str = "casual") -> dict:
    return {
        "id": f"nudge-{index:03d}",
        "kind": "nudge",
        "question": NUDGE_TEMPLATES[index % len(NUDGE_TEMPLATES)].format(j=index % 9),
        "persona": persona,
        "domain": domain,
        "register": register,
    }


def _batch(slot: str, requests: list[dict], progress: list[dict], tranche: int = 1) -> dict:
    return {
        "schema_version": "g1-authored-demo4-1",
        "demo": "demo-4",
        "author": {"model": "test-model", "slot": slot, "tranche": tranche},
        "requests": requests,
        "progress_pairs": progress,
    }


class BasicValidationTests(unittest.TestCase):
    def test_a_minimal_valid_batch_passes(self) -> None:
        batch = _batch("slot-a", [_request(0)], [_check(0), _nudge(0)])
        report = validate_demo4_batches([batch])
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["counts"]["requests"], 1)
        self.assertEqual(report["counts"]["checks"], 1)
        self.assertEqual(report["counts"]["nudges"], 1)

    def test_wrong_schema_version_is_rejected(self) -> None:
        batch = _batch("slot-a", [_request(0)], [_check(0), _nudge(0)])
        batch["schema_version"] = "wrong"
        report = validate_demo4_batches([batch])
        self.assertFalse(report["passed"])
        self.assertTrue(any("schema_version" in error for error in report["errors"]))

    def test_reserved_persona_is_rejected(self) -> None:
        batch = _batch("slot-a", [_request(0, persona="product-reviewer")], [_check(0), _nudge(0)])
        report = validate_demo4_batches([batch])
        self.assertFalse(report["passed"])
        self.assertTrue(any("reserved persona" in error for error in report["errors"]))

    def test_reserved_domain_is_rejected(self) -> None:
        batch = _batch("slot-a", [_request(0, domain="health")], [_check(0), _nudge(0)])
        report = validate_demo4_batches([batch])
        self.assertFalse(report["passed"])
        self.assertTrue(any("reserved domain" in error for error in report["errors"]))

    def test_duplicate_request_text_is_rejected(self) -> None:
        first = _request(0)
        second = _request(1)
        second["text"] = first["text"]
        batch = _batch("slot-a", [first, second], [_check(0), _nudge(0)])
        report = validate_demo4_batches([batch])
        self.assertFalse(report["passed"])
        self.assertTrue(any("duplicates" in error for error in report["errors"]))

    def test_nudge_with_a_reply_is_rejected(self) -> None:
        item = _nudge(0)
        item["reply"] = "should not be here"
        batch = _batch("slot-a", [_request(0)], [_check(0), item])
        report = validate_demo4_batches([batch])
        self.assertFalse(report["passed"])
        self.assertTrue(any("must not carry a reply" in error for error in report["errors"]))

    def test_check_without_a_reply_is_rejected(self) -> None:
        item = _check(0)
        del item["reply"]
        batch = _batch("slot-a", [_request(0)], [item, _nudge(0)])
        report = validate_demo4_batches([batch])
        self.assertFalse(report["passed"])
        self.assertTrue(any("require a non-empty reply" in error for error in report["errors"]))

    def test_forbidden_characters_are_rejected(self) -> None:
        item = _request(0)
        item["text"] = item["text"] + " <b>"
        batch = _batch("slot-a", [item], [_check(0), _nudge(0)])
        report = validate_demo4_batches([batch])
        self.assertFalse(report["passed"])
        self.assertTrue(any("must not contain" in error for error in report["errors"]))

    def test_too_long_request_text_is_rejected(self) -> None:
        item = _request(0)
        item["text"] = "x" * 200
        batch = _batch("slot-a", [item], [_check(0), _nudge(0)])
        report = validate_demo4_batches([batch])
        self.assertFalse(report["passed"])

    def test_too_short_question_is_rejected(self) -> None:
        item = _check(0)
        item["question"] = "hi?"
        batch = _batch("slot-a", [_request(0)], [item, _nudge(0)])
        report = validate_demo4_batches([batch])
        self.assertFalse(report["passed"])


class DistributionGateTests(unittest.TestCase):
    def _wide_batches(self, per_author: int = 10) -> list[dict]:
        personas = ["journal-keeper", "meeting-note-taker", "group-chat-drafter", "seed-diary-writer", "student-planner", "family-organizer"]
        domains = ["cooking", "work", "travel", "gardening", "school", "family"]
        registers = ["casual", "brisk", "warm", "plain"]
        batches = []
        counter = 0
        check_counter = 0
        nudge_counter = 0
        for author_index, slot in enumerate(("slot-a", "slot-b", "slot-c")):
            requests = []
            progress = []
            for local in range(per_author):
                persona = personas[counter % len(personas)]
                domain = domains[counter % len(domains)]
                register = registers[counter % len(registers)]
                requests.append(_request(counter, persona=persona, domain=domain, register=register))
                if counter % 2 == 0:
                    progress.append(
                        _check(check_counter, persona=persona, domain=domain, register=register)
                    )
                    check_counter += 1
                else:
                    progress.append(
                        _nudge(nudge_counter, persona=persona, domain=domain, register=register)
                    )
                    nudge_counter += 1
                counter += 1
            batches.append(_batch(slot, requests, progress, tranche=author_index))
        return batches

    def test_wide_corpus_passes_distribution_gates(self) -> None:
        batches = self._wide_batches()
        report = validate_demo4_batches(batches, enforce_distribution=True)
        self.assertTrue(report["passed"], report["errors"])
        self.assertTrue(report["distribution_enforced"])

    def test_small_corpus_without_enforcement_passes(self) -> None:
        batch = _batch("slot-a", [_request(0)], [_check(0), _nudge(0)])
        report = validate_demo4_batches([batch], enforce_distribution=False)
        self.assertTrue(report["passed"], report["errors"])
        self.assertFalse(report["distribution_enforced"])

    def test_small_corpus_with_enforcement_fails_loudly(self) -> None:
        """Mirrors Demo 3: enforcing the gates on a too-small corpus is a hard
        failure that names --allow-small-corpus, not a silent skip."""

        batch = _batch("slot-a", [_request(0)], [_check(0), _nudge(0)])
        report = validate_demo4_batches([batch], enforce_distribution=True)
        self.assertFalse(report["passed"])
        self.assertFalse(report["distribution_enforced"])
        self.assertTrue(any("allow-small-corpus" in error for error in report["errors"]))

    def test_lopsided_author_share_fails_distribution(self) -> None:
        batches = self._wide_batches()
        # Give one author almost everything so its share exceeds 40%.
        extra_requests = [
            _request(1000 + i, persona="journal-keeper", domain="cooking", register="casual")
            for i in range(80)
        ]
        extra_progress = [
            _check(1000 + i, persona="journal-keeper", domain="cooking", register="casual")
            if i % 2 == 0
            else _nudge(1000 + i, persona="journal-keeper", domain="cooking", register="casual")
            for i in range(20)
        ]
        batches[0] = _batch("slot-a", batches[0]["requests"] + extra_requests, batches[0]["progress_pairs"] + extra_progress)
        report = validate_demo4_batches(batches, enforce_distribution=True)
        self.assertFalse(report["passed"])
        self.assertTrue(any("author" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
