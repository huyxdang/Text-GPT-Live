from __future__ import annotations

import asyncio
import threading
import unittest
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from scripts.g1_local_latency import percentile, summarize
from scripts.local_model import (
    LocalMLXPolicy,
    _G1StreamState,
    clone_prompt_cache,
    completed_action,
    g1_prompt_segments,
    g1_static_prefix,
)


class _FakeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, enable_thinking, tokenize=True):
        del add_generation_prompt, enable_thinking
        rendered = f"SYSTEM:{messages[0]['content']}\nUSER:{messages[1]['content']}\nASSISTANT:"
        return [ord(value) for value in rendered] if tokenize else rendered

    def encode(self, value, *, add_special_tokens):
        del add_special_tokens
        return [ord(character) for character in value]


class _FakeArraysCache:
    def __init__(self) -> None:
        self.cache = ["stable-a", "stable-b"]


class G1LocalLatencyTests(unittest.TestCase):
    def test_g1_prompt_segments_use_exact_pre_marker_prefix(self) -> None:
        prompt = "<stream_event>hello</stream_event>\n<PREDICT_THIS_ACTION>"

        stable, suffix = g1_prompt_segments(_FakeTokenizer(), "policy", prompt)

        self.assertEqual("".join(map(chr, stable)), "SYSTEM:policy\nUSER:<stream_event>hello</stream_event>")
        self.assertEqual(
            "".join(map(chr, suffix)),
            "\n<PREDICT_THIS_ACTION>\nASSISTANT:",
        )

    def test_g1_static_prefix_ends_before_user_stream_content(self) -> None:
        prefix = g1_static_prefix(_FakeTokenizer(), "policy")

        self.assertEqual("".join(map(chr, prefix)), "SYSTEM:policy\nUSER:")

    def test_close_tag_stops_without_waiting_for_newline(self) -> None:
        pieces = ["<action>idle()", "</action>", "\nignored"]

        self.assertEqual(completed_action(pieces[:2]), "<action>idle()</action>")
        self.assertEqual(completed_action(pieces), "<action>idle()</action>")

    def test_newline_does_not_end_an_incomplete_g1_action(self) -> None:
        self.assertIsNone(completed_action(["<action>idle()\n"]))

    def test_cache_clone_copies_mutable_container_only(self) -> None:
        original = _FakeArraysCache()

        cloned = clone_prompt_cache([original])[0]
        cloned.cache[0] = "generated"

        self.assertEqual(original.cache, ["stable-a", "stable-b"])
        self.assertEqual(cloned.cache, ["generated", "stable-b"])

    def test_interleaved_session_caches_are_isolated_and_lru_bounded(self) -> None:
        policy = LocalMLXPolicy.__new__(LocalMLXPolicy)
        policy._static_cache = [_FakeArraysCache()]
        policy._static_tokens = [1, 2]
        policy._stream_caches = OrderedDict()
        policy._session_cache_limit = 2

        first = policy._stream_state("session-a")
        first.stable_tokens.append(3)
        second = policy._stream_state("session-b")
        second.stable_tokens.append(4)
        self.assertEqual(policy._stream_state("session-a").stable_tokens, [1, 2, 3])
        self.assertEqual(policy._stream_state("session-b").stable_tokens, [1, 2, 4])

        policy._stream_state("session-c")
        self.assertNotIn("session-a", policy._stream_caches)
        self.assertEqual(list(policy._stream_caches), ["session-b", "session-c"])

    def test_reset_clears_only_requested_session_cache(self) -> None:
        policy = LocalMLXPolicy.__new__(LocalMLXPolicy)
        policy._inference_lock = threading.Lock()
        policy._stream_caches = OrderedDict(
            [("a", _G1StreamState(None, [1])), ("b", _G1StreamState(None, [2]))]
        )
        policy.last_metrics = {"latency_ms": 10}

        policy.reset_stream_cache("a")

        self.assertEqual(list(policy._stream_caches), ["b"])
        self.assertEqual(policy.last_metrics, {})

    def test_aclose_shuts_down_executor_idempotently(self) -> None:
        policy = LocalMLXPolicy.__new__(LocalMLXPolicy)
        policy._closed = False
        policy._inference_lock = threading.Lock()
        policy._stream_caches = OrderedDict()
        policy.last_metrics = {}
        policy._executor = ThreadPoolExecutor(max_workers=1)

        asyncio.run(policy.aclose())
        asyncio.run(policy.aclose())

        with self.assertRaises(RuntimeError):
            policy._executor.submit(lambda: None)

    def test_percentile_uses_nearest_rank(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(percentile(values, 0.5), 30.0)
        self.assertEqual(percentile(values, 0.95), 50.0)

    def test_summary_reports_deadline_and_action_rates(self) -> None:
        rows = [
            {
                "latency_ms": 500.0,
                "first_token_ms": 400.0,
                "missed_deadline": False,
                "valid_action": True,
                "peak_memory_gb": 9.0,
            },
            {
                "latency_ms": 800.0,
                "first_token_ms": 700.0,
                "missed_deadline": True,
                "valid_action": False,
                "peak_memory_gb": 10.0,
            },
        ]

        result = summarize(rows)

        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["latency_ms"]["median"], 650.0)
        self.assertEqual(result["latency_ms"]["p95"], 800.0)
        self.assertEqual(result["missed_tick_rate"], 0.5)
        self.assertEqual(result["valid_action_rate"], 0.5)
        self.assertEqual(result["peak_memory_gb"], 10.0)


if __name__ == "__main__":
    unittest.main()
