import json
import unittest

import httpx

from app.domain import Action, ActionKind, CompletedTurn, EventSource, StreamEvent, UserState
from app.policy import SYSTEM_PROMPT_G1, DemoPolicy, PromptedPolicy, TinkerPolicy
from app.stream import parse_action


class PromptedPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_v4_system_prompt_and_stream_as_chat_messages(self) -> None:
        captured: dict[str, object] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "<action>idle()</action>"}}]},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            policy = PromptedPolicy(
                model="test-model",
                base_url="http://policy.local/v1",
                api_key="test-key",
                client=client,
            )
            output = await policy.predict(
                "<interaction_context></interaction_context>",
                StreamEvent(1, EventSource.USER, "hello", UserState.ACTIVE),
                [],
            )

        self.assertEqual(output, "<action>idle()</action>")
        self.assertEqual(captured["model"], "test-model")
        messages = captured["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("<action>tool(web_search", messages[0]["content"])
        self.assertIn("<action>tool(ui", messages[0]["content"])
        self.assertEqual(
            messages[1],
            {"role": "user", "content": "<interaction_context></interaction_context>"},
        )
        self.assertEqual(captured["temperature"], 0)


class DemoPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_does_not_repeat_while_search_is_pending(self) -> None:
        policy = DemoPolicy()
        previous = CompletedTurn(
            event=StreamEvent(1, EventSource.USER, "What was the result?", UserState.ACTIVE),
            action=Action(
                ActionKind.TOOL,
                tool_name="web_search",
                arguments={"query": "What was the result?"},
                raw_output='<action>tool(web_search,{"query":"What was the result?"})</action>',
            ),
        )

        output = await policy.predict(
            "prompt",
            StreamEvent(2, EventSource.USER, "What was the result? Today is", UserState.ACTIVE),
            [previous],
        )

        self.assertEqual(output, "<action>idle()</action>")

    async def test_search_decision_is_a_valid_generic_v4_tool_call(self) -> None:
        output = await DemoPolicy().predict(
            "prompt",
            StreamEvent(1, EventSource.USER, "What is the weather today?", UserState.ACTIVE),
            [],
        )

        action = parse_action(output)
        self.assertTrue(action.valid)
        self.assertEqual(action.kind, ActionKind.TOOL)
        self.assertEqual(action.tool_name, "web_search")
        self.assertEqual(action.query, "What is the weather today?")

    async def test_tool_result_event_is_always_idle(self) -> None:
        output = await DemoPolicy().predict(
            "prompt",
            StreamEvent(
                2,
                EventSource.TOOL,
                '{"query":"weather","results":[]}',
                tool_name="web_search",
                call_id="call-1",
            ),
            [],
        )
        self.assertEqual(output, "<action>idle()</action>")


class TinkerPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_checkpoint_reports_no_model_without_connecting(self) -> None:
        policy = TinkerPolicy(model_path="")

        with self.assertRaisesRegex(RuntimeError, "No model available"):
            await policy.predict(
                "prompt",
                StreamEvent(1, EventSource.USER, "hello", UserState.ACTIVE),
                [],
            )

    def test_default_output_budget_can_fit_structured_v4_actions(self) -> None:
        policy = TinkerPolicy(model_path="unused")
        self.assertGreaterEqual(policy._max_tokens, 160)

    def test_g1_decode_preserves_outer_whitespace_for_strict_runtime_parser(self) -> None:
        class Tokenizer:
            @staticmethod
            def decode(tokens, skip_special_tokens=True):
                del tokens, skip_special_tokens
                return " <action>idle()</action>"

        policy = TinkerPolicy(
            model_path="unused",
            system_prompt=SYSTEM_PROMPT_G1,
            action_schema="g1",
        )
        policy._tokenizer = Tokenizer()

        self.assertEqual(
            policy._decode_tokens([1]),
            " <action>idle()</action>",
        )


if __name__ == "__main__":
    unittest.main()
