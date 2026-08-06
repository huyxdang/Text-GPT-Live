from __future__ import annotations

import json
import unittest

from train.g1_causal_probe import (
    extract_clock_state,
    make_oracle_prompt,
    parse_math_output,
    score_action_outputs,
    score_math_outputs,
)


def recurring_row(*, now_ms: int, completion: str) -> dict[str, object]:
    prompt = (
        '<stream_event index="1" source="user" state="active" time="t+1000ms">'
        'Remind me every 5 seconds to drink water.</stream_event>\n'
        '<action>respond({"for":1,"message":"Set for every 5 seconds."})</action>\n'
        f'<stream_event index="2" source="user" state="idle" time="t+{now_ms}ms"></stream_event>\n'
        '<PREDICT_THIS_ACTION>'
    )
    return {
        "schema_version": "g1",
        "episode": "demo5-test",
        "candidate_id": f"demo5-test:{now_ms}",
        "prompt": prompt,
        "completion": completion,
        "schedule_kind": "every",
        "interval_s": 5,
        "fire_message": "Drink water!",
        "cancel_ack_text": None,
    }


class ClockStateTests(unittest.TestCase):
    def test_extracts_ack_anchor_and_due_boundary(self) -> None:
        row = recurring_row(
            now_ms=6000,
            completion='<action>respond({"for":2,"message":"Drink water!"})</action>',
        )
        state = extract_clock_state(row)
        self.assertEqual(state.interval_ms, 5000)
        self.assertEqual(state.anchor_ms, 1000)
        self.assertEqual(state.elapsed_ms, 5000)
        self.assertTrue(state.due)

    def test_oracle_record_precedes_prediction_marker(self) -> None:
        row = recurring_row(now_ms=5900, completion="<action>idle()</action>")
        prompt = make_oracle_prompt(str(row["prompt"]), extract_clock_state(row))
        self.assertIn('<diagnostic_obligation>{"active":true', prompt)
        self.assertTrue(prompt.endswith("</diagnostic_obligation>\n<PREDICT_THIS_ACTION>"))


class CausalScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        fire = recurring_row(
            now_ms=6000,
            completion='<action>respond({"for":2,"message":"Drink water!"})</action>',
        )
        wait = recurring_row(now_ms=5900, completion="<action>idle()</action>")
        self.cases = []
        for role, row in (("fire", fire), ("wait", wait)):
            state = extract_clock_state(row)
            self.cases.append(
                {
                    "case_id": role,
                    "completion": row["completion"],
                    "clock_state": {
                        "interval_ms": state.interval_ms,
                        "anchor_ms": state.anchor_ms,
                        "now_ms": state.now_ms,
                        "elapsed_ms": state.elapsed_ms,
                        "due": state.due,
                    },
                }
            )

    def test_math_parser_requires_exact_schema(self) -> None:
        self.assertIsNone(parse_math_output('{"due":true}'))
        self.assertIsNone(parse_math_output("```json\n{}\n```"))
        parsed = parse_math_output(
            '{"interval_ms":5000,"anchor_ms":1000,"now_ms":6000,"elapsed_ms":5000,"due":true}'
        )
        self.assertIsNotNone(parsed)

    def test_math_and_action_scores(self) -> None:
        math_outputs = [
            json.dumps(case["clock_state"], separators=(",", ":")) for case in self.cases
        ]
        math = score_math_outputs(self.cases, math_outputs)
        self.assertEqual(math["summary"]["exact"], 2)

        action = score_action_outputs(
            self.cases,
            [str(case["completion"]) for case in self.cases],
            gate_kind="oracle",
        )
        self.assertEqual(action["summary"]["fire_exact"], 1)
        self.assertEqual(action["summary"]["wait_exact"], 1)


if __name__ == "__main__":
    unittest.main()
