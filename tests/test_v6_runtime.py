"""Acceptance tests for the v6 surface: async jobs with identity, flat tools.

These assert at the trace level (musts_finish_before_move_on.md, "Traces first,
interface second"): the Demo 1 choreography must be fully expressed in the recorded
events — jobs genuinely outstanding together, results routed by job_id in either
completion order — with simulated deterministic providers and no model.
"""

import asyncio
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path

from app.domain import ActionKind, EventSource, UserState, derive_jobs
from app.policy import ScriptedV6Policy
from app.runtime import InteractionRuntime, JsonlTraceWriter
from app.uigen import DemoUIGenProvider, validate_ui_spec

UI_REQUEST_TEXT = "Can you create a visual brief about Vietnam's 2025 economic report for me?"
QUESTION_TEXT = (
    "I think Thinking Machine just released a new product. Can you tell me about what it is?"
)

VALID_SPEC = {
    "title": "Vietnam 2025",
    "components": [
        {"type": "stat_card", "label": "GDP growth", "value": "6.9%"},
        {"type": "sources", "links": [{"title": "Source", "url": "https://example.com"}]},
    ],
}


class SequencePolicy:
    mode = "test"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    async def predict(self, compiled_stream, current, history) -> str:
        del compiled_stream, current, history
        if not self.outputs:
            raise AssertionError("Test policy ran out of configured outputs")
        return self.outputs.pop(0)


class BlockingSearch:
    mode = "test"

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def search(self, query: str):
        await self.release.wait()
        return {
            "query": query,
            "results": [
                {
                    "title": "Thinking Machines launches Inkling",
                    "url": "https://example.com/inkling",
                    "snippet": "Open-weights model released this week.",
                }
            ],
        }


class ImmediateSearch:
    mode = "test"

    async def search(self, query: str):
        return {
            "query": query,
            "results": [
                {
                    "title": "Thinking Machines launches Inkling",
                    "url": "https://example.com/inkling",
                    "snippet": "Open-weights model released this week.",
                }
            ],
        }


class BlockingUIGen:
    mode = "test"

    def __init__(self, spec=None) -> None:
        self.release = asyncio.Event()
        self.spec = spec if spec is not None else VALID_SPEC

    async def generate(self, request: str):
        del request
        await self.release.wait()
        return copy.deepcopy(self.spec)


class FailingUIGen:
    mode = "test"

    async def generate(self, request: str):
        raise RuntimeError(f"generator unavailable for {request}")


async def wait_for_history(runtime, session_id: str, length: int, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while len(runtime.get_session(session_id).history) < length:
        if time.monotonic() > deadline:
            raise AssertionError(
                f"History never reached {length} events "
                f"(got {len(runtime.get_session(session_id).history)})"
            )
        await asyncio.sleep(0.01)


class V6RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def make_runtime(self, directory: str, policy, search, uigen) -> InteractionRuntime:
        runtime = InteractionRuntime(
            policy=policy,
            search_provider=search,
            trace_writer=JsonlTraceWriter(Path(directory) / "trace.jsonl"),
            uigen_provider=uigen,
        )
        self.addAsyncCleanup(runtime.shutdown)
        return runtime

    async def start_demo1_jobs(self, runtime, session_id: str) -> None:
        """Run the shared Demo 1 opening: UI job accepted, then search accepted."""
        await runtime.process_event(
            session_id,
            source=EventSource.USER,
            content=UI_REQUEST_TEXT,
            state=UserState.ACTIVE,
        )
        await wait_for_history(runtime, session_id, 2)  # generate_ui ack
        await runtime.process_event(
            session_id,
            source=EventSource.USER,
            content=f"{UI_REQUEST_TEXT} {QUESTION_TEXT}",
            state=UserState.ACTIVE,
        )
        await wait_for_history(runtime, session_id, 4)  # web_search ack
        # Both jobs genuinely outstanding during an overlapping period.
        self.assertEqual(runtime.pending_count(session_id), 2)

    async def test_demo1_search_finishes_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            search, uigen = BlockingSearch(), BlockingUIGen()
            runtime = self.make_runtime(
                directory,
                SequencePolicy(
                    [
                        f'<action>tool(generate_ui,{{"request":"{UI_REQUEST_TEXT[:-1]}"}})</action>',
                        "<action>idle()</action>",  # generate_ui ack
                        '<action>tool(web_search,{"query":"Thinking Machines new product"})</action>',
                        "<action>idle()</action>",  # web_search ack
                        "<action>idle()</action>",  # web_search completion
                        '<action>respond({"for":5,"message":"Inkling, an open-weights model."})</action>',
                        "<action>idle()</action>",  # generate_ui completion
                    ]
                ),
                search,
                uigen,
            )
            session = runtime.create_session(mode="v6")
            await self.start_demo1_jobs(runtime, session.id)

            search.release.set()
            await wait_for_history(runtime, session.id, 5)
            respond_turn = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content=f"{UI_REQUEST_TEXT} {QUESTION_TEXT}",
                state=UserState.IDLE,
            )
            uigen.release.set()
            await runtime.wait_for_background(session.id)

            history = runtime.get_session(session.id).history
            self.assertEqual(len(history), 7)
            self.assertEqual(history[1].event.job_id, "job-1")
            self.assertEqual(json.loads(history[1].event.content)["status"], "accepted")
            self.assertEqual(history[3].event.job_id, "job-3")
            self.assertEqual(history[4].event.tool_name, "web_search")
            self.assertEqual(history[4].event.job_id, "job-3")
            self.assertEqual(json.loads(history[4].event.content)["status"], "completed")
            # The answer routes to the search result, not the still-running UI job.
            self.assertEqual(respond_turn.action.kind, ActionKind.RESPOND)
            self.assertTrue(respond_turn.action.valid)
            self.assertEqual(respond_turn.action.target, 5)
            # The UI job completes last and carries its validated spec.
            self.assertEqual(history[6].event.tool_name, "generate_ui")
            self.assertEqual(history[6].event.job_id, "job-1")
            payload = json.loads(history[6].event.content)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(validate_ui_spec(payload["spec"]), [])

            jobs = {job["job_id"]: job for job in derive_jobs(history)}
            self.assertEqual(jobs["job-1"]["status"], "completed")
            self.assertEqual(jobs["job-3"]["status"], "completed")
            panels = runtime.get_session(session.id).to_dict()["panels"]
            self.assertEqual(len(panels), 1)

            # Traces first: the whole choreography is expressed in the trace file.
            records = [
                json.loads(line)
                for line in (Path(directory) / "trace.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 7)
            self.assertEqual(
                [record["event"].get("job_id") for record in records],
                [None, "job-1", None, "job-3", "job-3", None, "job-1"],
            )

    async def test_demo1_ui_finishes_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            search, uigen = BlockingSearch(), BlockingUIGen()
            runtime = self.make_runtime(
                directory,
                SequencePolicy(
                    [
                        f'<action>tool(generate_ui,{{"request":"{UI_REQUEST_TEXT[:-1]}"}})</action>',
                        "<action>idle()</action>",  # generate_ui ack
                        '<action>tool(web_search,{"query":"Thinking Machines new product"})</action>',
                        "<action>idle()</action>",  # web_search ack
                        "<action>idle()</action>",  # generate_ui completion
                        "<action>idle()</action>",  # web_search completion
                        '<action>respond({"for":6,"message":"Inkling, an open-weights model."})</action>',
                    ]
                ),
                search,
                uigen,
            )
            session = runtime.create_session(mode="v6")
            await self.start_demo1_jobs(runtime, session.id)

            uigen.release.set()
            await wait_for_history(runtime, session.id, 5)
            search.release.set()
            await wait_for_history(runtime, session.id, 6)
            respond_turn = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content=f"{UI_REQUEST_TEXT} {QUESTION_TEXT}",
                state=UserState.IDLE,
            )
            await runtime.wait_for_background(session.id)

            history = runtime.get_session(session.id).history
            self.assertEqual(history[4].event.tool_name, "generate_ui")
            self.assertEqual(history[4].event.job_id, "job-1")
            self.assertEqual(history[5].event.tool_name, "web_search")
            self.assertEqual(history[5].event.job_id, "job-3")
            # Reversed completion order still routes the answer to the search result.
            self.assertTrue(respond_turn.action.valid)
            self.assertEqual(respond_turn.action.target, 6)
            jobs = {job["job_id"]: job for job in derive_jobs(history)}
            self.assertEqual(jobs["job-1"]["status"], "completed")
            self.assertEqual(jobs["job-3"]["status"], "completed")

    async def test_duplicate_generate_ui_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            uigen = BlockingUIGen()
            runtime = self.make_runtime(
                directory,
                SequencePolicy(
                    [
                        '<action>tool(generate_ui,{"request":"visual brief of Vietnam"})</action>',
                        "<action>idle()</action>",
                        '<action>tool(generate_ui,{"request":"Visual brief of Vietnam"})</action>',
                    ]
                ),
                ImmediateSearch(),
                uigen,
            )
            session = runtime.create_session(mode="v6")
            await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content=UI_REQUEST_TEXT,
                state=UserState.ACTIVE,
            )
            await wait_for_history(runtime, session.id, 2)
            duplicate = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content=UI_REQUEST_TEXT,
                state=UserState.ACTIVE,
            )
            self.assertFalse(duplicate.action.valid)
            self.assertIn("generate_ui", duplicate.action.diagnostic)
            self.assertEqual(runtime.pending_count(session.id), 1)
            uigen.release.set()
            await runtime.wait_for_background(session.id)

    async def test_invalid_spec_and_provider_error_become_failed_jobs(self) -> None:
        for uigen in (BlockingUIGen(spec={"title": "x", "components": [{"type": "video"}]}), FailingUIGen()):
            with tempfile.TemporaryDirectory() as directory:
                runtime = self.make_runtime(
                    directory,
                    SequencePolicy(
                        [
                            '<action>tool(generate_ui,{"request":"visual brief"})</action>',
                            "<action>idle()</action>",
                            "<action>idle()</action>",
                        ]
                    ),
                    ImmediateSearch(),
                    uigen,
                )
                session = runtime.create_session(mode="v6")
                await runtime.process_event(
                    session.id,
                    source=EventSource.USER,
                    content=UI_REQUEST_TEXT,
                    state=UserState.ACTIVE,
                )
                if isinstance(uigen, BlockingUIGen):
                    uigen.release.set()
                await runtime.wait_for_background(session.id)
                history = runtime.get_session(session.id).history
                payload = json.loads(history[-1].event.content)
                self.assertEqual(payload["status"], "failed")
                self.assertTrue(payload["error"])
                jobs = derive_jobs(history)
                self.assertEqual(jobs[0]["status"], "failed")
                self.assertEqual(
                    runtime.get_session(session.id).to_dict()["panels"], []
                )

    async def test_suggest_edit_works_and_legacy_tools_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                directory,
                SequencePolicy(
                    [
                        '<action>tool(suggest_edit,{"target":{"occurrence":1,"quote":"was quiet"},"replacement":"was loud"})</action>',
                        '<action>tool(suggest_edit,{"target":{"occurrence":1,"quote":"was quiet"},"replacement":"was loud"})</action>',
                        '<action>tool(highlight,{"target":{"occurrence":1,"quote":"quiet"}})</action>',
                        '<action>tool(ui,{"operation":"highlight","target":{"occurrence":1,"quote":"quiet"}})</action>',
                    ]
                ),
                ImmediateSearch(),
                DemoUIGenProvider(),
            )
            session = runtime.create_session(mode="v6")
            content = 'Change every "quiet" to "loud" for me. The reading room was quiet.'
            first = await runtime.process_event(
                session.id, source=EventSource.USER, content=content, state=UserState.ACTIVE
            )
            self.assertTrue(first.action.valid)
            suggestions = runtime.get_session(session.id).suggestions
            self.assertEqual(len(suggestions), 1)
            self.assertEqual(suggestions[0].quote, "was quiet")
            self.assertEqual(suggestions[0].replacement, "was loud")
            self.assertEqual(
                runtime.get_session(session.id).to_dict()["suggestions"][0]["replacement"],
                "was loud",
            )

            duplicate = await runtime.process_event(
                session.id, source=EventSource.USER, content=content, state=UserState.ACTIVE
            )
            self.assertFalse(duplicate.action.valid)
            self.assertIn("already has a live suggestion", duplicate.action.diagnostic)

            for expected in ("not part of this surface", "not part of this surface"):
                legacy = await runtime.process_event(
                    session.id, source=EventSource.USER, content=content, state=UserState.ACTIVE
                )
                self.assertFalse(legacy.action.valid)
                self.assertIn(expected, legacy.action.diagnostic)

    async def test_reset_clears_suggestions_so_matching_text_cannot_resurface_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                directory,
                SequencePolicy(
                    [
                        '<action>tool(suggest_edit,{"target":{"occurrence":1,"quote":"was quiet"},"replacement":"was loud"})</action>',
                        "<action>idle()</action>",
                    ]
                ),
                ImmediateSearch(),
                DemoUIGenProvider(),
            )
            session = runtime.create_session(mode="v6")
            content = "The reading room was quiet."

            await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content=content,
                state=UserState.ACTIVE,
            )
            self.assertEqual(len(session.suggestions), 1)

            reset = await runtime.reset_session(session.id)
            self.assertEqual(reset.suggestions, [])
            self.assertEqual(reset.to_dict()["suggestions"], [])

            await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content=content,
                state=UserState.IDLE,
            )
            self.assertEqual(session.suggestions, [])


class ScriptedV6PolicyTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end shakedown: the scripted policy drives the full demo loops."""

    def make_runtime(self, directory: str) -> InteractionRuntime:
        runtime = InteractionRuntime(
            policy=ScriptedV6Policy(),
            search_provider=ImmediateSearch(),
            trace_writer=JsonlTraceWriter(Path(directory) / "trace.jsonl"),
            uigen_provider=DemoUIGenProvider(),
        )
        self.addAsyncCleanup(runtime.shutdown)
        return runtime

    async def test_scripted_demo1_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(directory)
            session = runtime.create_session(mode="v6")

            first = await runtime.process_event(
                session.id, source=EventSource.USER, content=UI_REQUEST_TEXT, state=UserState.ACTIVE
            )
            self.assertEqual(first.action.tool_name, "generate_ui")
            await runtime.wait_for_background(session.id)

            second = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content=f"{UI_REQUEST_TEXT} {QUESTION_TEXT}",
                state=UserState.ACTIVE,
            )
            self.assertEqual(second.action.tool_name, "web_search")
            await runtime.wait_for_background(session.id)

            third = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content=f"{UI_REQUEST_TEXT} {QUESTION_TEXT}",
                state=UserState.IDLE,
            )
            self.assertEqual(third.action.kind, ActionKind.RESPOND)
            self.assertTrue(third.action.valid)
            self.assertIn("Inkling", third.action.message)

            state = runtime.get_session(session.id).to_dict()
            self.assertEqual(len(state["panels"]), 1)
            self.assertEqual(
                {job["status"] for job in state["jobs"]}, {"completed"}
            )

    async def test_scripted_swap_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(directory)
            session = runtime.create_session(mode="v6")
            instruction = 'Please replace every "quiet" with "loud" in what I write next.'

            first = await runtime.process_event(
                session.id, source=EventSource.USER, content=instruction, state=UserState.ACTIVE
            )
            # The occurrence inside the instruction is never touched.
            self.assertEqual(first.action.kind, ActionKind.IDLE)

            full_text = (
                f"{instruction} The library stayed open late. A few people worked near "
                "the windows, but the reading room was quiet."
            )
            second = await runtime.process_event(
                session.id, source=EventSource.USER, content=full_text, state=UserState.ACTIVE
            )
            self.assertEqual(second.action.tool_name, "suggest_edit")
            self.assertTrue(second.action.valid)
            suggestions = runtime.get_session(session.id).suggestions
            self.assertEqual(len(suggestions), 1)
            self.assertEqual(suggestions[0].occurrence, 2)
            self.assertEqual(suggestions[0].replacement, "loud")

            third = await runtime.process_event(
                session.id, source=EventSource.USER, content=full_text, state=UserState.IDLE
            )
            # No duplicates on later ticks.
            self.assertEqual(third.action.kind, ActionKind.IDLE)
            self.assertEqual(len(runtime.get_session(session.id).suggestions), 1)


class UISpecValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_demo_provider_spec_is_valid(self) -> None:
        spec = await DemoUIGenProvider().generate("visual brief of Vietnam")
        self.assertEqual(validate_ui_spec(spec), [])

    def test_rejects_malformed_specs(self) -> None:
        self.assertTrue(validate_ui_spec("not a dict"))
        self.assertTrue(validate_ui_spec({"title": "x", "components": []}))
        self.assertTrue(validate_ui_spec({"title": "", "components": [{"type": "stat_card"}]}))
        self.assertTrue(
            validate_ui_spec({"title": "x", "components": [{"type": "hologram"}]})
        )
        two_charts = {
            "title": "x",
            "components": [
                {"type": "chart", "chart": "bar", "series": [{"label": "a", "value": 1}]},
                {"type": "chart", "chart": "line", "series": [{"label": "b", "value": 2}]},
            ],
        }
        self.assertTrue(validate_ui_spec(two_charts))
        bad_url = {
            "title": "x",
            "components": [{"type": "sources", "links": [{"title": "s", "url": "ftp://x"}]}],
        }
        self.assertTrue(validate_ui_spec(bad_url))


if __name__ == "__main__":
    unittest.main()
