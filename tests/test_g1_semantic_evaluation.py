from __future__ import annotations

import unittest

from scripts.g1_semantic_eval import _fingerprint
from train.g1_evaluation import evaluate_g1_predictions
from train.g1_semantic_evaluation import apply_semantic_judgments, build_semantic_cases


def pair(
    completion: str,
    *,
    episode: str,
    demo: str = "demo-1",
    situation: str = "address-positive",
    clause_state: str | None = None,
    should_fire: bool = False,
) -> dict:
    return {
        "schema_version": "g1",
        "episode": episode,
        "demo": demo,
        "situation": situation,
        "prompt": (
            '<stream_event index="1" source="user" state="idle" time="t+650ms">'
            "Please help with this.</stream_event>\n<PREDICT_THIS_ACTION>"
        ),
        "completion": completion,
        "clause_state": clause_state,
        "should_fire": should_fire,
    }


class G1SemanticEvaluationTests(unittest.TestCase):
    def test_judgment_cache_is_scoped_to_provider_and_model(self) -> None:
        case = {"row_index": 4, "reference": "yes", "candidate": "sure"}

        local = _fingerprint(case, judge_identity="local-mlx:qwen")
        remote = _fingerprint(case, judge_identity="openai:judge-a")
        other_remote = _fingerprint(case, judge_identity="openai:judge-b")

        self.assertNotEqual(local, remote)
        self.assertNotEqual(remote, other_remote)

    def test_paraphrase_is_judged_but_wrong_target_is_not(self) -> None:
        pairs = [
            pair(
                '<action>respond({"for":1,"message":"I will remind you in five minutes."})</action>',
                episode="paraphrase",
            ),
            pair(
                '<action>respond({"for":1,"message":"The first clause in Chinese."})</action>',
                episode="wrong-target",
                demo="demo-3",
                situation="clause-positive",
                clause_state="complete",
            ),
        ]
        outputs = [
            '<action>respond({"for":1,"message":"Sure, five minutes from now."})</action>',
            '<action>respond({"for":2,"message":"The first clause in Chinese."})</action>',
        ]
        strict = evaluate_g1_predictions(pairs, outputs)

        cases = build_semantic_cases(pairs, strict)

        self.assertEqual([case["row_index"] for case in cases], [0])
        hybrid = apply_semantic_judgments(pairs, strict, {0: {"pass": True}})
        self.assertTrue(hybrid["rows"][0]["hybrid_row_pass"])
        self.assertFalse(hybrid["rows"][1]["hybrid_row_pass"])
        self.assertFalse(hybrid["rows"][1]["deterministic_anchors_pass"])
        self.assertEqual(hybrid["summary"]["strict_row_accuracy"], 0.0)
        self.assertEqual(hybrid["summary"]["hybrid_row_accuracy"], 0.5)

    def test_edit_replacement_is_semantic_but_quote_stays_exact(self) -> None:
        pairs = [
            pair(
                '<action>suggest_edit({"quote":"tries make","replacement":"tries to make"})</action>',
                episode="replacement",
                demo="demo-2",
                situation="error-positive",
            ),
            pair(
                '<action>suggest_edit({"quote":"tries make","replacement":"tries to make"})</action>',
                episode="wrong-quote",
                demo="demo-2",
                situation="error-positive",
            ),
        ]
        outputs = [
            '<action>suggest_edit({"quote":"tries make","replacement":"tries making"})</action>',
            '<action>suggest_edit({"quote":"make","replacement":"making"})</action>',
        ]
        strict = evaluate_g1_predictions(pairs, outputs)

        cases = build_semantic_cases(pairs, strict)

        self.assertEqual([case["row_index"] for case in cases], [0])
        hybrid = apply_semantic_judgments(pairs, strict, {0: {"pass": True}})
        self.assertTrue(hybrid["rows"][0]["hybrid_row_pass"])
        self.assertFalse(hybrid["rows"][1]["hybrid_row_pass"])

    def test_wrong_action_cannot_be_rescued_by_semantic_judge(self) -> None:
        pairs = [
            pair(
                '<action>respond({"for":1,"message":"Drink water!"})</action>',
                episode="fire-now",
                demo="demo-5",
                situation="fire-silent",
                should_fire=True,
            )
        ]
        strict = evaluate_g1_predictions(pairs, ["<action>idle()</action>"])

        self.assertEqual(build_semantic_cases(pairs, strict), [])
        hybrid = apply_semantic_judgments(pairs, strict, {})
        self.assertFalse(hybrid["rows"][0]["hybrid_row_pass"])
        self.assertEqual(hybrid["summary"]["should_fire_recall"], 0.0)

    def test_missing_required_judgment_fails_closed(self) -> None:
        pairs = [
            pair(
                '<action>delegate({"task":"build a chart of lighthouse statistics"})</action>',
                episode="delegate",
                demo="demo-4",
                situation="request-positive",
            )
        ]
        strict = evaluate_g1_predictions(
            pairs,
            ['<action>delegate({"task":"make a lighthouse statistics chart"})</action>'],
        )

        with self.assertRaisesRegex(ValueError, "Missing 1 required semantic judgment"):
            apply_semantic_judgments(pairs, strict, {})


if __name__ == "__main__":
    unittest.main()
