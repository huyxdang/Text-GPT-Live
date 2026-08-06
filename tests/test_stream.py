import unittest

from app.domain import Action, ActionKind, CompletedTurn, EventSource, StreamEvent, UserState
from app.stream import (
    action_completion,
    compile_stream,
    g1_action_completion,
    parse_action,
    parse_g1_action,
)


class StreamCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instruction = "Highlight every animal I mention."
        self.permissions = {"ui": ["highlight"], "web_search": ["search"]}

    def test_wrapped_compiler_keeps_context_history_and_current_event_separate(self) -> None:
        history = [
            CompletedTurn(
                event=StreamEvent(
                    index=1,
                    source=EventSource.USER,
                    content="I saw an otter",
                    state=UserState.ACTIVE,
                    elapsed_ms=650,
                ),
                action=Action(kind=ActionKind.IDLE),
            )
        ]
        current = StreamEvent(
            index=2,
            source=EventSource.USER,
            content="I saw an otter near the river",
            state=UserState.ACTIVE,
            elapsed_ms=1300,
        )

        compiled = compile_stream(
            history,
            current,
            instruction=self.instruction,
            permissions=self.permissions,
            fmt="wrapped",
        )

        self.assertTrue(compiled.startswith("<interaction_context>"))
        self.assertIn(f"<instruction>{self.instruction}</instruction>", compiled)
        self.assertIn("<stream_history>", compiled)
        self.assertIn('<stream_event index="1" source="user" state="active" time="t+650ms">', compiled)
        self.assertIn('<action source="assistant">idle()</action>', compiled)
        self.assertLess(compiled.index("</stream_history>"), compiled.index('index="2"'))
        self.assertTrue(compiled.endswith("<PREDICT_THIS_ACTION>"))

    def test_flat_stream_is_append_only_and_context_is_stable(self) -> None:
        first_event = StreamEvent(1, EventSource.USER, "I saw an ot", UserState.ACTIVE, 650)
        first_prompt = compile_stream(
            [],
            first_event,
            instruction=self.instruction,
            permissions=self.permissions,
        )
        first_turn = CompletedTurn(first_event, Action(ActionKind.IDLE))
        second_event = StreamEvent(2, EventSource.USER, "I saw an otter", UserState.ACTIVE, 1300)
        second_prompt = compile_stream(
            [first_turn],
            second_event,
            instruction=self.instruction,
            permissions=self.permissions,
        )

        context = (
            "<interaction_context>\n"
            f"<instruction>{self.instruction}</instruction>\n"
            '<permissions>{"ui":["highlight"],"web_search":["search"]}</permissions>\n'
            "</interaction_context>"
        )
        self.assertTrue(first_prompt.startswith(context))
        self.assertTrue(second_prompt.startswith(context))
        first_without_prediction = first_prompt.removesuffix("<PREDICT_THIS_ACTION>")
        self.assertTrue(second_prompt.startswith(first_without_prediction + "<action>idle()</action>"))

    def test_compiler_serializes_generic_tool_events_and_actions(self) -> None:
        history = [
            CompletedTurn(
                StreamEvent(1, EventSource.USER, "Who won?", UserState.ACTIVE, 650),
                Action(ActionKind.TOOL, "web_search", {"query": "who won"}),
            ),
            CompletedTurn(
                StreamEvent(
                    2,
                    EventSource.TOOL,
                    '{"results":[]}',
                    elapsed_ms=975,
                    tool_name="web_search",
                    call_id="call-1",
                ),
                Action(ActionKind.IDLE),
            ),
        ]

        compiled = compile_stream(
            history,
            StreamEvent(3, EventSource.USER, "Who won?", UserState.IDLE, 1300),
            instruction="",
            permissions=self.permissions,
        )

        self.assertIn('<action>tool(web_search,{"query":"who won"})</action>', compiled)
        self.assertIn('source="tool" time="t+975ms" tool="web_search" call_id="call-1"', compiled)

    def test_compiler_escapes_context_user_text_and_action_payloads(self) -> None:
        history = [
            CompletedTurn(
                StreamEvent(1, EventSource.USER, "a < b & c", UserState.ACTIVE),
                Action(ActionKind.TOOL, "web_search", {"query": "a < b & c"}),
            )
        ]
        current = StreamEvent(2, EventSource.USER, "still > typing", UserState.ACTIVE)

        compiled = compile_stream(
            history,
            current,
            instruction="Highlight <animal> & friends",
            permissions=self.permissions,
        )

        self.assertIn("Highlight &lt;animal&gt; &amp; friends", compiled)
        self.assertIn("a &lt; b &amp; c", compiled)
        self.assertIn("still &gt; typing", compiled)
        self.assertNotIn("a < b & c", compiled)

    def test_unknown_stream_format_is_rejected(self) -> None:
        current = StreamEvent(1, EventSource.USER, "hello", UserState.ACTIVE)
        with self.assertRaises(ValueError):
            compile_stream([], current, fmt="v3")


class ActionSerializationTests(unittest.TestCase):
    def test_all_canonical_action_forms_round_trip(self) -> None:
        actions = [
            Action(ActionKind.IDLE),
            Action(ActionKind.TOOL, "web_search", {"query": "latest weather"}),
            Action(
                ActionKind.TOOL,
                "ui",
                {"operation": "highlight", "target": {"occurrence": 2, "quote": "otter"}},
            ),
            Action(ActionKind.RESPOND, target=17, message="It is raining."),
        ]

        for expected in actions:
            with self.subTest(action=expected):
                completion = action_completion(expected)
                parsed = parse_action(completion)
                self.assertTrue(parsed.valid, parsed.diagnostic)
                self.assertEqual(parsed.kind, expected.kind)
                self.assertEqual(parsed.tool_name, expected.tool_name)
                self.assertEqual(parsed.arguments, expected.arguments)
                self.assertEqual(parsed.target, expected.target)
                self.assertEqual(parsed.message, expected.message)

    def test_all_g1_action_forms_round_trip_canonically(self) -> None:
        actions = [
            Action(ActionKind.IDLE),
            Action(ActionKind.RESPOND, target=12, message="Drink water!"),
            Action(
                ActionKind.TOOL,
                "suggest_edit",
                {"quote": "tried make ramen", "replacement": "tried making ramen"},
            ),
            Action(ActionKind.TOOL, "highlight", {"occurrence": 2, "quote": "fox"}),
            Action(ActionKind.TOOL, "delegate", {"task": "build a lighthouse dashboard"}),
            Action(ActionKind.TOOL, "web_search", {"query": "latest SpaceX valuation"}),
            Action(
                ActionKind.TOOL,
                "translate_commit",
                {"for": 19, "message": "会议结束后，"},
            ),
        ]

        for expected in actions:
            with self.subTest(action=expected):
                completion = g1_action_completion(expected)
                parsed = parse_g1_action(completion)
                self.assertTrue(parsed.valid, parsed.diagnostic)
                self.assertEqual(parsed.kind, expected.kind)
                self.assertEqual(parsed.tool_name, expected.tool_name)
                self.assertEqual(parsed.arguments, expected.arguments)
                self.assertEqual(parsed.target, expected.target)
                self.assertEqual(parsed.message, expected.message)

    def test_g1_stream_is_header_free_and_keeps_empty_current_events(self) -> None:
        history = [
            CompletedTurn(
                user_event := StreamEvent(
                    1,
                    EventSource.USER,
                    "Remind me every 5 seconds.",
                    UserState.IDLE,
                    1000,
                ),
                Action(ActionKind.RESPOND, target=1, message="Got it."),
            )
        ]
        current = StreamEvent(2, EventSource.USER, "", UserState.IDLE, 6000)

        compiled = compile_stream(history, current, fmt="g1")

        self.assertNotIn("<interaction_context>", compiled)
        self.assertIn('time="t+1000ms"', compiled)
        self.assertIn(g1_action_completion(history[0].action), compiled)
        self.assertIn(
            '<stream_event index="2" source="user" state="idle" time="t+6000ms"></stream_event>',
            compiled,
        )
        self.assertTrue(compiled.endswith("<PREDICT_THIS_ACTION>"))
        self.assertEqual(user_event.index, 1)

    def test_g1_parser_rejects_old_or_noncanonical_forms(self) -> None:
        rejected = [
            '<action>tool(delegate,{"task":"build it"})</action>',
            '<action>delegate({"task":"build it","extra":true})</action>',
            '<action>suggest_edit({"replacement":"right","quote":"wrong"})</action>',
            '<action>suggest_edit({"quote":"   ","replacement":"right"})</action>',
            '<action>suggest_edit({"quote":"wrong","replacement":"   "})</action>',
            '<action>respond({"message":"hello","for":1})</action>',
            '<action>web_search({"query":""})</action>',
            '<action>translate_commit({"for":true,"message":"你好"})</action>',
            '<action>translate_commit({"for":2,"message":""})</action>',
            " <action>idle()</action>",
            "<action>idle()</action>\n",
        ]
        for output in rejected:
            with self.subTest(output=output):
                self.assertFalse(parse_g1_action(output).valid)

    def test_g1_completion_round_trips_literal_entities_and_unicode(self) -> None:
        expected = Action(
            kind=ActionKind.RESPOND,
            target=7,
            message="AT&amp;T <测试>",
        )
        parsed = parse_g1_action(g1_action_completion(expected))

        self.assertTrue(parsed.valid, parsed.diagnostic)
        self.assertEqual(parsed.message, expected.message)


class ActionParserTests(unittest.TestCase):
    def test_accepts_canonical_idle(self) -> None:
        action = parse_action("  <action>idle()</action>\n")
        self.assertEqual(action.kind, ActionKind.IDLE)
        self.assertTrue(action.valid)

    def test_accepts_web_search_with_json_escaping(self) -> None:
        action = parse_action(r'<action>tool(web_search,{"query":"cats & dogs \"today\""})</action>')
        self.assertEqual(action.kind, ActionKind.TOOL)
        self.assertEqual(action.tool_name, "web_search")
        self.assertEqual(action.arguments, {"query": 'cats & dogs "today"'})
        self.assertTrue(action.valid)

    def test_accepts_ui_highlight_and_respond(self) -> None:
        highlight = parse_action(
            '<action>tool(ui,{"operation":"highlight","target":{"occurrence":2,"quote":"otter"}})</action>'
        )
        response = parse_action(
            '<action>respond({"for":48,"message":"It is called poutine."})</action>'
        )

        self.assertEqual(highlight.kind, ActionKind.TOOL)
        self.assertEqual(highlight.tool_name, "ui")
        self.assertEqual(highlight.arguments["target"]["occurrence"], 2)
        self.assertEqual(response.kind, ActionKind.RESPOND)
        self.assertEqual(response.target, 48)
        self.assertEqual(response.message, "It is called poutine.")

    def test_rejects_legacy_markdown_explanations_and_multiple_actions(self) -> None:
        outputs = [
            "<idle/>",
            "<web_search>cats</web_search>",
            "I would wait. <action>idle()</action>",
            "```xml\n<action>idle()</action>\n```",
            "<action>idle()</action><action>idle()</action>",
            "<action><action>idle()</action></action>",
            "<wait/>",
        ]
        for output in outputs:
            with self.subTest(output=output):
                action = parse_action(output)
                self.assertEqual(action.kind, ActionKind.IDLE)
                self.assertFalse(action.valid)

    def test_rejects_malformed_and_non_object_payloads(self) -> None:
        outputs = [
            '<action>tool(web_search,{"query":})</action>',
            '<action>tool(web_search,["query"])</action>',
            '<action>respond({"for":"48","message":"answer"})</action>',
            '<action>respond({"for":true,"message":"answer"})</action>',
            '<action>respond({"for":0,"message":"answer"})</action>',
            '<action>respond({"for":-1,"message":"answer"})</action>',
            '<action>respond({"for":48,"message":"   "})</action>',
            '<action>tool(web_search,{"query":"unterminated})</action>',
        ]
        for output in outputs:
            with self.subTest(output=output):
                action = parse_action(output)
                self.assertFalse(action.valid)

    def test_python_like_payload_is_data_never_executed(self) -> None:
        output = '<action>tool(ui,{"operation":__import__("os").system("touch /tmp/nope")})</action>'
        action = parse_action(output)
        self.assertFalse(action.valid)
        self.assertIn("JSON", action.diagnostic)


if __name__ == "__main__":
    unittest.main()
