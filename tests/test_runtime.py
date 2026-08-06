import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from app.domain import ActionKind, EventSource, UserState
from app.runtime import InteractionRuntime, JsonlTraceWriter


class SequencePolicy:
    mode = "test"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    async def predict(self, compiled_stream, current, history) -> str:
        del compiled_stream, current, history
        if not self.outputs:
            raise AssertionError("Test policy ran out of configured outputs")
        return self.outputs.pop(0)


class SessionAwarePolicy:
    mode = "test"

    def __init__(self) -> None:
        self.predicted_sessions: list[str] = []
        self.reset_sessions: list[str] = []

    async def predict_session(self, session_id, compiled_stream, current, history) -> str:
        del compiled_stream, current, history
        self.predicted_sessions.append(session_id)
        return "<action>idle()</action>"

    def reset_stream_cache(self, session_id: str) -> None:
        self.reset_sessions.append(session_id)


class BlockingSearch:
    mode = "test"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def search(self, query: str):
        self.query = query
        self.started.set()
        await self.release.wait()
        return {
            "query": query,
            "results": [{"title": "Result", "url": "https://example.com", "snippet": "Found"}],
        }


class ImmediateSearch:
    mode = "test"

    async def search(self, query: str):
        return {"query": query, "results": []}


class FailingSearch:
    mode = "test"

    async def search(self, query: str):
        raise RuntimeError(f"search unavailable for {query}")


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def make_runtime(self, directory: str, policy, search) -> InteractionRuntime:
        runtime = InteractionRuntime(
            policy=policy,
            search_provider=search,
            trace_writer=JsonlTraceWriter(Path(directory) / "trace.jsonl"),
        )
        self.addAsyncCleanup(runtime.shutdown)
        return runtime

    async def test_session_aware_policy_receives_interleaved_ids_and_reset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = SessionAwarePolicy()
            runtime = self.make_runtime(directory, policy, ImmediateSearch())
            first = runtime.create_session()
            second = runtime.create_session()

            await runtime.process_event(
                first.id, source=EventSource.USER, content="one", state=UserState.ACTIVE
            )
            await runtime.process_event(
                second.id, source=EventSource.USER, content="two", state=UserState.ACTIVE
            )
            await runtime.process_event(
                first.id, source=EventSource.USER, content="three", state=UserState.ACTIVE
            )
            await runtime.reset_session(first.id)

            self.assertEqual(policy.predicted_sessions, [first.id, second.id, first.id])
            self.assertEqual(policy.reset_sessions, [first.id])

    async def test_search_is_async_and_result_reenters_as_generic_tool_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            search = BlockingSearch()
            runtime = self.make_runtime(
                directory,
                SequencePolicy(
                    [
                        '<action>tool(web_search,{"query":"Morocco Netherlands result"})</action>',
                        "<action>idle()</action>",
                        "<action>idle()</action>",
                    ]
                ),
                search,
            )
            session = runtime.create_session()

            first = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="What was the result?",
                state=UserState.ACTIVE,
            )
            await asyncio.wait_for(search.started.wait(), timeout=1)

            second = await asyncio.wait_for(
                runtime.process_event(
                    session.id,
                    source=EventSource.USER,
                    content="What was the result? Today is",
                    state=UserState.ACTIVE,
                ),
                timeout=1,
            )
            self.assertEqual(first.action.kind, ActionKind.TOOL)
            self.assertEqual(first.action.tool_name, "web_search")
            self.assertEqual(second.action.kind, ActionKind.IDLE)
            self.assertEqual(runtime.pending_count(session.id), 1)

            search.release.set()
            await runtime.wait_for_background(session.id)

            history = runtime.get_session(session.id).history
            self.assertEqual([turn.event.index for turn in history], [1, 2, 3])
            self.assertEqual(history[2].event.source, EventSource.TOOL)
            self.assertEqual(history[2].event.tool_name, "web_search")
            self.assertEqual(history[2].event.call_id, "call-1")
            self.assertEqual(history[2].action.kind, ActionKind.IDLE)
            payload = json.loads(history[2].event.content)
            self.assertEqual(payload["query"], "Morocco Netherlands result")

            records = [
                json.loads(line)
                for line in (Path(directory) / "trace.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 3)
            self.assertEqual(records[2]["event"]["source"], "tool")
            self.assertEqual(records[2]["event"]["tool_name"], "web_search")

    async def test_ui_highlight_is_applied_deduplicated_and_reconciled_after_edits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                directory,
                SequencePolicy(
                    [
                        '<action>tool(ui,{"operation":"highlight","target":{"occurrence":1,"quote":"elephant"}})</action>',
                        '<action>tool(ui,{"operation":"highlight","target":{"occurrence":1,"quote":"elephant"}})</action>',
                        "<action>idle()</action>",
                        '<action>tool(ui,{"operation":"highlight","target":{"occurrence":2,"quote":"otter"}})</action>',
                    ]
                ),
                ImmediateSearch(),
            )
            session = runtime.create_session(instruction="Highlight every animal I mention.")

            first = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="I saw an elephant",
                state=UserState.ACTIVE,
            )
            self.assertTrue(first.action.valid)
            self.assertEqual(first.action.tool_name, "ui")
            self.assertEqual(session.highlights[0].quote, "elephant")

            duplicate = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="I saw an elephant",
                state=UserState.ACTIVE,
            )
            self.assertFalse(duplicate.action.valid)
            self.assertIn("already highlighted", duplicate.action.diagnostic)
            self.assertEqual(len(session.highlights), 1)

            await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="I changed the sentence",
                state=UserState.ACTIVE,
            )
            self.assertEqual(session.highlights, [])

            second_occurrence = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="otter and otter",
                state=UserState.ACTIVE,
            )
            self.assertTrue(second_occurrence.action.valid)
            self.assertEqual(session.highlights[0].occurrence, 2)

    async def test_runtime_enforces_permissions_and_registered_tool_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                directory,
                SequencePolicy(
                    [
                        '<action>tool(web_search,{"query":"current score"})</action>',
                        '<action>tool(ui,{"operation":"highlight","target":{"occurrence":1,"quote":"cat"}})</action>',
                        '<action>tool(filesystem,{"path":"/tmp"})</action>',
                        '<action>tool(ui,{"operation":"highlight","target":{"occurrence":true,"quote":"cat"}})</action>',
                    ]
                ),
                ImmediateSearch(),
            )
            session = runtime.create_session(
                instruction="Highlight animals.",
                permissions={"web_search": [], "ui": []},
            )

            denied_search = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="What is the current score?",
                state=UserState.ACTIVE,
            )
            denied_highlight = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="cat",
                state=UserState.ACTIVE,
            )
            unknown_tool = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="cat",
                state=UserState.ACTIVE,
            )

            self.assertFalse(denied_search.action.valid)
            self.assertIn("permission", denied_search.action.diagnostic)
            self.assertFalse(denied_highlight.action.valid)
            self.assertIn("permission", denied_highlight.action.diagnostic)
            self.assertFalse(unknown_tool.action.valid)
            self.assertIn("not registered", unknown_tool.action.diagnostic)
            self.assertEqual(runtime.pending_count(session.id), 0)

            # Enable UI only, then verify booleans are not accepted as integer occurrences.
            session.permissions["ui"] = ["highlight"]
            invalid_occurrence = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="cat",
                state=UserState.ACTIVE,
            )
            self.assertFalse(invalid_occurrence.action.valid)
            self.assertIn("positive integer", invalid_occurrence.action.diagnostic)

    async def test_explicit_empty_permissions_do_not_restore_default_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            search = BlockingSearch()
            runtime = self.make_runtime(
                directory,
                SequencePolicy(
                    ['<action>tool(web_search,{"query":"current score"})</action>']
                ),
                search,
            )
            session = runtime.create_session(permissions={})
            turn = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="What is the current score?",
                state=UserState.ACTIVE,
            )

            self.assertEqual(session.permissions, {})
            self.assertFalse(turn.action.valid)
            self.assertIn("permission", turn.action.diagnostic)
            self.assertEqual(runtime.pending_count(session.id), 0)

    async def test_equivalent_search_is_rejected_while_first_call_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            search = BlockingSearch()
            runtime = self.make_runtime(
                directory,
                SequencePolicy(
                    [
                        '<action>tool(web_search,{"query":"Current Score"})</action>',
                        '<action>tool(web_search,{"query":"  current   score  "})</action>',
                        "<action>idle()</action>",
                    ]
                ),
                search,
            )
            session = runtime.create_session()
            first = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="What is the current score?",
                state=UserState.ACTIVE,
            )
            await asyncio.wait_for(search.started.wait(), timeout=1)
            duplicate = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="What is the current score?",
                state=UserState.ACTIVE,
            )

            self.assertTrue(first.action.valid)
            self.assertFalse(duplicate.action.valid)
            self.assertIn("already pending or complete", duplicate.action.diagnostic)
            self.assertEqual(runtime.pending_count(session.id), 1)

            search.release.set()
            await runtime.wait_for_background(session.id)

    async def test_ui_tool_cannot_mutate_text_from_a_tool_result_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                directory,
                SequencePolicy(
                    [
                        '<action>tool(ui,{"operation":"highlight","target":{"occurrence":1,"quote":"otter"}})</action>'
                    ]
                ),
                ImmediateSearch(),
            )
            session = runtime.create_session()
            turn = await runtime.process_event(
                session.id,
                source=EventSource.TOOL,
                content='{"snippet":"otter"}',
                tool_name="web_search",
                call_id="call-1",
            )

            self.assertFalse(turn.action.valid)
            self.assertIn("user event", turn.action.diagnostic)
            self.assertEqual(session.highlights, [])

    async def test_respond_requires_idle_user_and_unanswered_search_result_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                directory,
                SequencePolicy(
                    [
                        "<action>idle()</action>",
                        '<action>respond({"for":1,"message":"The result says Found."})</action>',
                        '<action>respond({"for":99,"message":"Unsupported."})</action>',
                        '<action>respond({"for":1,"message":"The result says Found."})</action>',
                        '<action>respond({"for":1,"message":"Repeating the answer."})</action>',
                    ]
                ),
                ImmediateSearch(),
            )
            session = runtime.create_session()
            result = await runtime.process_event(
                session.id,
                source=EventSource.TOOL,
                content='{"query":"test","results":[{"snippet":"Found"}]}',
                tool_name="web_search",
                call_id="call-0",
            )
            self.assertEqual(result.event.index, 1)

            while_active = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Still typing",
                state=UserState.ACTIVE,
            )
            missing_target = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Done",
                state=UserState.IDLE,
            )
            accepted = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Done",
                state=UserState.IDLE,
            )
            duplicate = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Done",
                state=UserState.IDLE,
            )

            self.assertFalse(while_active.action.valid)
            self.assertIn("idle", while_active.action.diagnostic)
            self.assertFalse(missing_target.action.valid)
            self.assertIn("target", missing_target.action.diagnostic)
            self.assertTrue(accepted.action.valid)
            self.assertEqual(accepted.action.kind, ActionKind.RESPOND)
            self.assertEqual(accepted.action.target, 1)
            self.assertFalse(duplicate.action.valid)
            self.assertIn("already", duplicate.action.diagnostic)

    async def test_invalid_policy_output_becomes_inspectable_idle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                directory,
                SequencePolicy(["Here is my answer: <action>idle()</action>"]),
                ImmediateSearch(),
            )
            session = runtime.create_session()
            turn = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="hello",
                state=UserState.ACTIVE,
            )

            self.assertEqual(turn.action.kind, ActionKind.IDLE)
            self.assertFalse(turn.action.valid)
            self.assertEqual(turn.action.raw_output, "Here is my answer: <action>idle()</action>")
            self.assertIn("not exactly", turn.action.diagnostic)

    async def test_reset_clears_history_highlights_and_cancels_stale_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            search = BlockingSearch()
            runtime = self.make_runtime(
                directory,
                SequencePolicy(
                    ['<action>tool(web_search,{"query":"query"})</action>']
                ),
                search,
            )
            session = runtime.create_session()
            await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="search this?",
                state=UserState.ACTIVE,
            )
            await asyncio.wait_for(search.started.wait(), timeout=1)

            reset = await runtime.reset_session(session.id)
            await asyncio.sleep(0)

            self.assertEqual(reset.history, [])
            self.assertEqual(reset.highlights, [])
            self.assertEqual(reset.next_index, 1)
            self.assertEqual(runtime.pending_count(session.id), 0)

    async def test_search_failure_returns_as_inspectable_generic_tool_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                directory,
                SequencePolicy(
                    [
                        '<action>tool(web_search,{"query":"current score"})</action>',
                        "<action>idle()</action>",
                    ]
                ),
                FailingSearch(),
            )
            session = runtime.create_session()
            await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="What is the current score?",
                state=UserState.ACTIVE,
            )
            await runtime.wait_for_background(session.id)

            result_turn = runtime.get_session(session.id).history[1]
            payload = json.loads(result_turn.event.content)
            self.assertEqual(result_turn.event.source, EventSource.TOOL)
            self.assertEqual(result_turn.event.tool_name, "web_search")
            self.assertIn("RuntimeError", payload["error"])
            self.assertEqual(result_turn.action.kind, ActionKind.IDLE)


if __name__ == "__main__":
    unittest.main()
