from __future__ import annotations

import unittest

from train.evaluation import evaluate_predictions, score_prediction


def pair(completion: str, *, episode: str, permissions: dict | None = None) -> dict:
    return {
        "episode": episode,
        "bucket": "fixture",
        "completion": completion,
        "permissions": permissions
        if permissions is not None
        else {"web_search": ["search"], "ui": ["highlight"]},
    }


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs = [
            pair("<action>idle()</action>", episode="idle"),
            pair(
                '<action>tool(ui,{"operation":"highlight","target":{"occurrence":1,"quote":"otter"}})</action>',
                episode="highlight",
            ),
            pair(
                '<action>tool(web_search,{"query":"otter habitat"})</action>',
                episode="search",
            ),
            pair(
                '<action>respond({"for":7,"message":"Otters live near rivers and coasts."})</action>',
                episode="respond",
            ),
        ]

    def test_perfect_predictions_pass_every_gate(self) -> None:
        outputs = [item["completion"] for item in self.pairs]
        report = evaluate_predictions(self.pairs, outputs, label="perfect")

        self.assertEqual(report["summary"]["macro_f1"], 1.0)
        self.assertEqual(report["summary"]["strict_row_accuracy"], 1.0)
        self.assertEqual(report["summary"]["episode_success_rate"], 1.0)
        self.assertTrue(report["hard_gates"]["passed"])

    def test_invalid_output_is_not_counted_as_idle(self) -> None:
        row = score_prediction(self.pairs[0], "<idle/>")

        self.assertEqual(row["expected_class"], "idle")
        self.assertEqual(row["predicted_class"], "invalid")
        self.assertFalse(row["format_valid"])
        self.assertFalse(row["kind_match"])
        self.assertFalse(row["row_pass"])

    def test_permission_violation_is_a_strict_failure(self) -> None:
        denied = pair(
            '<action>idle()</action>',
            episode="denied",
            permissions={},
        )
        output = '<action>tool(ui,{"operation":"highlight","target":{"occurrence":1,"quote":"otter"}})</action>'
        row = score_prediction(denied, output)

        self.assertTrue(row["permission_violation"])
        self.assertFalse(row["row_pass"])

    def test_payload_mismatch_fails_even_when_class_matches(self) -> None:
        output = '<action>tool(ui,{"operation":"highlight","target":{"occurrence":2,"quote":"otter"}})</action>'
        row = score_prediction(self.pairs[1], output)

        self.assertTrue(row["kind_match"])
        self.assertFalse(row["highlight_occurrence_exact"])
        self.assertFalse(row["row_pass"])

    def test_search_concept_aliases_accept_a_semantic_paraphrase(self) -> None:
        search = pair(
            '<action>tool(web_search,{"query":"best time of year to see northern lights"})</action>',
            episode="semantic-search",
        )
        search["evaluation"] = {
            "required_query_concepts": [
                ["aurora", "northern lights"],
                ["best time", "when", "season"],
            ]
        }
        row = score_prediction(
            search,
            '<action>tool(web_search,{"query":"when is aurora season"})</action>',
        )

        self.assertFalse(row["search_query_exact"])
        self.assertEqual(row["search_query_term_coverage"], 1.0)
        self.assertTrue(row["row_pass"])

    def test_search_concepts_still_reject_an_underspecified_query(self) -> None:
        search = pair(
            '<action>tool(web_search,{"query":"Riverside Library District 3 Sunday hours"})</action>',
            episode="specific-search",
        )
        search["evaluation"] = {
            "required_query_concepts": [
                ["riverside library"],
                ["district 3"],
                ["sunday"],
                ["hours", "open"],
            ]
        }
        row = score_prediction(
            search,
            '<action>tool(web_search,{"query":"library hours"})</action>',
        )

        self.assertLess(row["search_query_term_coverage"], 1.0)
        self.assertFalse(row["row_pass"])

    def test_response_claim_aliases_allow_paraphrase_but_forbid_result_merging(self) -> None:
        response = pair(
            '<action>respond({"for":7,"message":"Mercury orbits the Sun in about 88 days."})</action>',
            episode="routed-response",
        )
        response["evaluation"] = {
            "required_response_claims": [["mercury"], ["88"], ["day", "days"]],
            "forbidden_response_claims": [["venus"], ["225"]],
        }
        clean = score_prediction(
            response,
            '<action>respond({"for":7,"message":"A Mercury year lasts roughly 88 days."})</action>',
        )
        merged = score_prediction(
            response,
            '<action>respond({"for":7,"message":"Mercury takes 88 days; Venus takes 225 days."})</action>',
        )

        self.assertTrue(clean["row_pass"])
        self.assertEqual(clean["respond_forbidden_claim_rate"], 0.0)
        self.assertFalse(merged["row_pass"])
        self.assertGreater(merged["respond_forbidden_claim_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
