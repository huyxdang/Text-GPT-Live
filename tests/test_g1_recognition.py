from __future__ import annotations

import unittest

from train.g1_recognition import (
    build_fire_recognition_cases,
    recognition_stream_evidence,
    score_fire_recognition,
)


def fire_pair(index: int) -> dict:
    return {
        "episode": f"fire-{index}",
        "candidate_id": f"fire-{index}:at",
        "situation": "fire-silent",
        "timing_boundary": "at",
        "should_fire": True,
        "prompt": (
            f'<stream_event index="{index}" source="user" state="idle" '
            f'time="t+5000ms"></stream_event>\n<PREDICT_THIS_ACTION>'
        ),
        "completion": (
            f'<action>respond({{"for":{index},"message":"Drink water!"}})</action>'
        ),
    }


class G1RecognitionTests(unittest.TestCase):
    def test_evidence_keeps_prior_response_and_current_snapshot_without_every_tick(self) -> None:
        prompt = (
            '<stream_event index="1" source="user" state="active" time="t+1000ms">'
            + "x" * 3_000
            + "</stream_event>\n"
            + '<action>respond({"for":1,"message":"Schedule acknowledged."})</action>\n'
            + '<stream_event index="2" source="user" state="idle" time="t+5000ms">'
            + "schedule at the beginning "
            + "y" * 4_000
            + " latest text"
            + "</stream_event>\n<PREDICT_THIS_ACTION>"
        )

        evidence = recognition_stream_evidence(prompt, max_chars=4_000)

        self.assertIn("Schedule acknowledged", evidence)
        self.assertIn('time="t+5000ms"', evidence)
        self.assertIn("schedule at the beginning", evidence)
        self.assertIn("latest text", evidence)
        self.assertLess(len(evidence), 4_300)

    def test_two_presentations_reverse_candidate_order(self) -> None:
        cases = build_fire_recognition_cases([fire_pair(1)], presentations_per_item=2)

        self.assertEqual(len(cases), 2)
        self.assertEqual({case["correct_position"] for case in cases}, {"A", "B"})
        self.assertTrue(all("<interaction_stream>" in case["prompt"] for case in cases))

    def test_position_bias_cannot_look_like_recognition(self) -> None:
        cases = build_fire_recognition_cases(
            [fire_pair(1), fire_pair(2)],
            presentations_per_item=2,
        )

        report = score_fire_recognition(cases, ["A"] * len(cases))

        self.assertEqual(report["summary"]["recognition_accuracy"], 0.5)
        self.assertEqual(report["summary"]["order_consistent_item_rate"], 0.0)
        self.assertEqual(report["summary"]["diagnosis"], "capability_gap")

    def test_order_invariant_recognition_recommends_dpo(self) -> None:
        cases = build_fire_recognition_cases(
            [fire_pair(1), fire_pair(2)],
            presentations_per_item=2,
        )
        outputs = [case["correct_position"] for case in cases]

        report = score_fire_recognition(cases, outputs)

        self.assertEqual(report["summary"]["recognition_accuracy"], 1.0)
        self.assertEqual(report["summary"]["order_consistent_item_rate"], 1.0)
        self.assertEqual(report["summary"]["diagnosis"], "preference_gap")
        self.assertEqual(
            report["summary"]["next_training_action"],
            "short_symmetric_dpo_with_sft_replay",
        )


if __name__ == "__main__":
    unittest.main()
