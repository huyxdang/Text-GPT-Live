from __future__ import annotations

import unittest

from scripts.g1_context_growth import (
    TICK_MS,
    build_live_prompt,
    content_for_tick,
    run_measurement,
    ticks_at,
)


class FakeTokenizer:
    model_max_length = 10_000

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": list(range(len(text)))}

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ):
        self.asserted = (add_generation_prompt, enable_thinking)
        length = sum(len(message["content"]) for message in messages) + 7
        return list(range(length))


class G1ContextGrowthTests(unittest.TestCase):
    def test_ticks_at_uses_real_650_ms_boundaries(self) -> None:
        self.assertEqual(TICK_MS, 650)
        self.assertEqual(ticks_at(1), 1)
        self.assertEqual(ticks_at(60), 92)
        self.assertEqual(ticks_at(600), 923)
        with self.assertRaises(ValueError):
            ticks_at(0)

    def test_live_prompt_uses_exact_g1_history_shape(self) -> None:
        prompt = build_live_prompt("empty_silence", 3)

        self.assertEqual(prompt.count("<stream_event "), 3)
        self.assertEqual(prompt.count("<action>idle()</action>"), 2)
        self.assertIn('index="3" source="user" state="idle" time="t+1950ms"', prompt)
        self.assertNotIn("<interaction_context>", prompt)
        self.assertTrue(prompt.endswith("<PREDICT_THIS_ACTION>"))

    def test_continuous_typing_retains_full_growing_snapshot(self) -> None:
        self.assertEqual(len(content_for_tick("continuous_typing", 1)), 4)
        self.assertEqual(len(content_for_tick("continuous_typing", 92)), 368)
        self.assertTrue(
            content_for_tick("continuous_typing", 2).startswith(
                content_for_tick("continuous_typing", 1)
            )
        )

    def test_measurement_is_monotonic_and_marks_weight_free_rendering(self) -> None:
        tokenizer = FakeTokenizer()
        report = run_measurement(tokenizer, (1, 10))

        self.assertFalse(report["weights_loaded"])
        self.assertFalse(report["rendering"]["enable_thinking"])
        self.assertEqual(report["tick_ms"], 650)
        for scenario in report["scenarios"]:
            rows = scenario["measurements"]
            self.assertLess(rows[0]["rendered_tokens"], rows[1]["rendered_tokens"])
            self.assertGreater(rows[1]["tokens_per_added_tick"], 0)
            self.assertIsNone(scenario["first_tick_over_context"])
        self.assertEqual(tokenizer.asserted, (True, False))


if __name__ == "__main__":
    unittest.main()
