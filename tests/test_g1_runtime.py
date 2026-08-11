from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main_module
from app.domain import ActionKind, EventSource, UserState, derive_jobs
from app.policy import SYSTEM_PROMPT_G1, TinkerPolicy
from app.runtime import InteractionRuntime, JsonlTraceWriter
from app.search import DemoSearchProvider
from app.stream import compile_stream
from app.uigen import DemoUIGenProvider


class QueuePolicy:
    mode = "test-g1"
    display_name = "queued g1 policy"
    availability = "ready"
    availability_message = ""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def warm_up(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def predict(self, compiled_stream, current, history) -> str:
        del current, history
        self.prompts.append(compiled_stream)
        if not self.outputs:
            raise AssertionError("QueuePolicy ran out of outputs.")
        return self.outputs.pop(0)


class FailingUIGenProvider:
    async def generate(self, request: str) -> dict:
        raise RuntimeError(f"cannot build {request}")


class ProgressiveUIGenProvider:
    """A provider fixture that proves partial UI state is render-only."""

    mode = "test"

    def __init__(self) -> None:
        self.title_published = asyncio.Event()
        self.release_component = asyncio.Event()

    async def generate(self, request: str) -> dict:
        del request
        return {
            "title": "Fallback should not run",
            "components": [{"type": "stat_card", "label": "Fallback", "value": "0"}],
        }

    async def stream_build(self, request: str):
        del request
        yield {"kind": "title", "title": "Rocket dashboard"}
        self.title_published.set()
        await self.release_component.wait()
        yield {"kind": "component", "component": {"type": "stat_card", "label": "Reused", "value": "10x"}}


def make_runtime(
    root: Path,
    outputs: list[str],
    *,
    uigen_delay: float = 0.0,
) -> tuple[InteractionRuntime, QueuePolicy]:
    policy = QueuePolicy(outputs)
    runtime = InteractionRuntime(
        policy=policy,
        search_provider=DemoSearchProvider(),
        trace_writer=JsonlTraceWriter(root / "trace.jsonl"),
        stream_format="g1",
        uigen_provider=DemoUIGenProvider(delay_seconds=uigen_delay),
        simulated_tick_ms=650,
    )
    return runtime, policy


class G1RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_g1_delegate_publishes_partial_ui_without_new_model_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = ProgressiveUIGenProvider()
            policy = QueuePolicy(
                [
                    '<action>delegate({"task":"build a rocket dashboard"})</action>',
                    "<action>idle()</action>",  # accepted
                    "<action>idle()</action>",  # completed
                ]
            )
            runtime = InteractionRuntime(
                policy=policy,
                search_provider=DemoSearchProvider(),
                trace_writer=JsonlTraceWriter(Path(directory) / "trace.jsonl"),
                stream_format="g1",
                uigen_provider=provider,
                simulated_tick_ms=650,
            )
            self.addAsyncCleanup(runtime.shutdown)
            session = runtime.create_session(mode="g1")

            await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Build a rocket dashboard.",
                state=UserState.IDLE,
            )
            await provider.title_published.wait()

            partial_job = session.to_dict()["jobs"][0]
            self.assertEqual(partial_job["status"], "running")
            self.assertEqual(
                partial_job["spec"],
                {"title": "Rocket dashboard", "components": []},
            )
            self.assertEqual(len(session.history), 2)
            self.assertEqual(session.history[-1].event.tool_name, "delegate")
            self.assertEqual(json.loads(session.history[-1].event.content)["status"], "accepted")

            provider.release_component.set()
            await runtime.wait_for_background(session.id)

            completed_job = session.to_dict()["jobs"][0]
            self.assertEqual(completed_job["status"], "completed")
            self.assertEqual(completed_job["spec"]["components"][0]["value"], "10x")
            self.assertEqual(len(session.history), 3)

    async def test_g1_web_search_runs_and_result_is_answered_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = make_runtime(
                Path(directory),
                [
                    '<action>web_search({"query":"latest SpaceX valuation"})</action>',
                    '<action>idle()</action>',
                    '<action>respond({"for":3,"message":"The returned source reports the latest valuation."})</action>',
                ],
            )
            session = runtime.create_session(mode="g1")

            called = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Search for the latest SpaceX valuation.",
                state=UserState.IDLE,
            )
            await runtime.wait_for_background(session.id)

            self.assertTrue(called.action.valid, called.action.diagnostic)
            self.assertEqual(called.action.tool_name, "web_search")
            jobs = derive_jobs(session.history)
            self.assertEqual(jobs[0]["status"], "completed")
            completion = session.history[-1]
            self.assertEqual(completion.event.tool_name, "web_search")
            self.assertEqual(completion.action.kind, ActionKind.RESPOND)
            self.assertEqual(completion.action.target, completion.event.index)

            duplicate = runtime._validate_action_g1(
                session,
                completion.event,
                completion.action,
            )
            self.assertFalse(duplicate.valid)

    async def test_g1_delegate_acceptance_can_launch_a_second_search_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = make_runtime(
                Path(directory),
                [
                    '<action>delegate({"task":"build a rocket dashboard"})</action>',
                    '<action>web_search({"query":"latest SpaceX valuation"})</action>',
                    '<action>idle()</action>',
                    '<action>idle()</action>',
                    '<action>respond({"for":5,"message":"The search result is ready."})</action>',
                ],
                uigen_delay=0.01,
            )
            session = runtime.create_session(mode="g1")
            await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Build a rocket dashboard and look up the latest SpaceX valuation.",
                state=UserState.IDLE,
            )
            await runtime.wait_for_background(session.id)

            jobs = derive_jobs(session.history)
            self.assertEqual({job["tool"] for job in jobs}, {"delegate", "web_search"})
            self.assertEqual({job["status"] for job in jobs}, {"completed"})
            responses = [
                turn
                for turn in session.history
                if turn.action.kind is ActionKind.RESPOND and turn.action.valid
            ]
            self.assertEqual(len(responses), 1)
            self.assertEqual(responses[0].event.tool_name, "web_search")

    async def test_g1_tool_event_cannot_answer_a_historical_user_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = make_runtime(
                Path(directory),
                [
                    '<action>delegate({"task":"build a rocket dashboard"})</action>',
                    '<action>respond({"for":1,"message":"An unsolicited lifecycle reply."})</action>',
                    '<action>idle()</action>',
                ],
            )
            session = runtime.create_session(mode="g1")
            await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Build a rocket dashboard.",
                state=UserState.IDLE,
            )
            await runtime.wait_for_background(session.id)

            accepted = next(
                turn
                for turn in session.history
                if turn.event.tool_name == "delegate"
                and json.loads(turn.event.content).get("status") == "accepted"
            )
            self.assertFalse(accepted.action.valid)
            self.assertIn("user event", accepted.action.diagnostic)

    async def test_g1_translation_commits_accumulate_and_recover_an_earlier_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = make_runtime(
                Path(directory),
                [
                    '<action>translate_commit({"for":1,"message":"会议结束后，"})</action>',
                    '<action>idle()</action>',
                    '<action>translate_commit({"for":2,"message":"我们去吃了午饭。"})</action>',
                ],
            )
            session = runtime.create_session(mode="g1")

            first = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="After the meeting,",
                state=UserState.ACTIVE,
            )
            second = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="After the meeting, we had lunch.",
                state=UserState.ACTIVE,
            )
            recovery = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="After the meeting, we had lunch. ",
                state=UserState.IDLE,
            )

            self.assertTrue(first.action.valid, first.action.diagnostic)
            self.assertEqual(second.action.kind, ActionKind.IDLE)
            self.assertTrue(recovery.action.valid, recovery.action.diagnostic)
            self.assertEqual(recovery.action.arguments["for"], second.event.index)
            self.assertEqual(session.to_dict()["translation"], "会议结束后，我们去吃了午饭。")

            await runtime.reset_session(session.id)
            self.assertEqual(session.translation_commits, [])

    async def test_g1_translation_edit_invalidates_affected_commits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = make_runtime(
                Path(directory),
                [
                    '<action>translate_commit({"for":1,"message":"会议结束后，"})</action>',
                    '<action>idle()</action>',
                ],
            )
            session = runtime.create_session(mode="g1")
            await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="After the meeting,",
                state=UserState.ACTIVE,
            )
            await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Before the meeting,",
                state=UserState.ACTIVE,
            )

            self.assertEqual(session.translation_commits, [])

    async def test_g1_translation_cannot_recommit_stale_text_after_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = make_runtime(
                Path(directory),
                [
                    '<action>translate_commit({"for":1,"message":"会议结束后，"})</action>',
                    '<action>idle()</action>',
                    '<action>translate_commit({"for":1,"message":"会议结束后，"})</action>',
                ],
            )
            session = runtime.create_session(mode="g1")
            first = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="After the meeting,",
                state=UserState.ACTIVE,
            )
            await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Before the meeting,",
                state=UserState.ACTIVE,
            )
            stale = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Before the meeting, we waited.",
                state=UserState.IDLE,
            )

            self.assertTrue(first.action.valid)
            self.assertFalse(stale.action.valid)
            self.assertIn("no longer extends", stale.action.diagnostic)
            self.assertEqual(session.translation_commits, [])

    async def test_live_runtime_uses_header_free_compiler_strict_parser_and_g1_tools(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, policy = make_runtime(
                root,
                [
                    '<action>suggest_edit({"quote":"tried make","replacement":"tried making"})</action>',
                    '<action>highlight({"occurrence":1,"quote":"fox"})</action>',
                    '<action>respond({"for":3,"message":"Still here."})</action>',
                ],
            )
            session = runtime.create_session(mode="g1")

            suggestion = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Last night I tried make ramen.",
                state=UserState.ACTIVE,
            )
            suggestion_replacement = session.suggestions[0].replacement
            highlight = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="The fox crossed the field.",
                state=UserState.ACTIVE,
            )
            highlight_quote = session.highlights[0].quote
            response = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="",
                state=UserState.IDLE,
            )

            self.assertTrue(suggestion.action.valid, suggestion.action.diagnostic)
            self.assertEqual(suggestion.action.tool_name, "suggest_edit")
            self.assertEqual(suggestion_replacement, "tried making")
            self.assertTrue(highlight.action.valid, highlight.action.diagnostic)
            self.assertEqual(highlight_quote, "fox")
            self.assertTrue(response.action.valid, response.action.diagnostic)
            self.assertEqual(response.action.kind, ActionKind.RESPOND)
            self.assertEqual(response.action.target, response.event.index)

            expected_prompt = compile_stream(
                session.history[:-1],
                session.history[-1].event,
                fmt="g1",
            )
            self.assertEqual(session.latest_prompt, expected_prompt)
            self.assertEqual(policy.prompts[-1], expected_prompt)
            self.assertNotIn("<interaction_context>", expected_prompt)
            self.assertIn(
                '<action>suggest_edit({"quote":"tried make","replacement":"tried making"})</action>',
                expected_prompt,
            )

            trace_rows = [
                json.loads(line)
                for line in (root / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual({row["action_schema"] for row in trace_rows}, {"g1"})
            self.assertEqual(trace_rows[-1]["compiled_prompt"], expected_prompt)

    async def test_legacy_wrapper_is_invalid_on_live_g1_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = make_runtime(
                Path(directory),
                ['<action>tool(delegate,{"task":"build it"})</action>'],
            )
            session = runtime.create_session(mode="g1")

            turn = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Build it.",
                state=UserState.IDLE,
            )

            self.assertFalse(turn.action.valid)
            self.assertEqual(turn.action.kind, ActionKind.IDLE)
            self.assertIn("g1", turn.action.diagnostic or "")

    async def test_delegate_runs_asynchronous_job_and_reenters_g1_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = make_runtime(
                Path(directory),
                [
                    '<action>delegate({"task":"generate a lighthouse dashboard"})</action>',
                    "<action>idle()</action>",
                    "<action>idle()</action>",
                ],
                uigen_delay=0.01,
            )
            session = runtime.create_session(mode="g1")

            delegated = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Build a dashboard about lighthouses.",
                state=UserState.IDLE,
            )
            await runtime.wait_for_background(session.id)

            self.assertTrue(delegated.action.valid, delegated.action.diagnostic)
            self.assertEqual(delegated.action.tool_name, "delegate")
            jobs = derive_jobs(session.history)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["tool"], "delegate")
            self.assertEqual(jobs[0]["status"], "completed")
            state = session.to_dict()
            self.assertEqual(len(state["panels"]), 1)
            self.assertIn("spec", state["jobs"][0])
            tool_events = [
                turn.event
                for turn in session.history
                if turn.event.source is EventSource.TOOL
            ]
            self.assertEqual([event.tool_name for event in tool_events], ["delegate", "delegate"])
            self.assertEqual({event.job_id for event in tool_events}, {"job-1"})
            accepted_payload, completed_payload = [
                json.loads(event.content) for event in tool_events
            ]
            self.assertEqual(accepted_payload, {"status": "accepted"})
            self.assertEqual(
                completed_payload,
                {
                    "status": "completed",
                    "task": "generate a lighthouse dashboard",
                },
            )
            self.assertNotIn("spec", completed_payload)
            self.assertNotIn("job_id", completed_payload)
            self.assertNotIn('"spec"', session.latest_prompt)

    async def test_g1_suggest_edit_preserves_canonical_replacement_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = make_runtime(
                Path(directory),
                [
                    '<action>suggest_edit({"quote":"wrong","replacement":" right "})</action>',
                ],
            )
            session = runtime.create_session(mode="g1")

            turn = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="This is wrong.",
                state=UserState.ACTIVE,
            )

            self.assertTrue(turn.action.valid, turn.action.diagnostic)
            self.assertEqual(turn.action.arguments["replacement"], " right ")
            self.assertEqual(session.suggestions[0].replacement, " right ")

    async def test_g1_delegate_failure_echoes_task_without_job_id_or_spec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = make_runtime(
                Path(directory),
                [
                    '<action>delegate({"task":"build a broken panel"})</action>',
                    "<action>idle()</action>",
                    "<action>idle()</action>",
                ],
            )
            runtime.uigen_provider = FailingUIGenProvider()
            session = runtime.create_session(mode="g1")

            await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Build a broken panel.",
                state=UserState.IDLE,
            )
            await runtime.wait_for_background(session.id)

            payloads = [
                json.loads(turn.event.content)
                for turn in session.history
                if turn.event.source is EventSource.TOOL
            ]
            self.assertEqual(payloads[0], {"status": "accepted"})
            self.assertEqual(
                payloads[1],
                {
                    "status": "failed",
                    "task": "build a broken panel",
                    "error": "RuntimeError: cannot build build a broken panel",
                },
            )
            self.assertEqual(session.job_specs, {})
            state = session.to_dict()
            self.assertEqual(state["jobs"][0]["status"], "failed")
            self.assertEqual(state["panels"], [])

    async def test_g1_session_rejects_hidden_header_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = make_runtime(Path(directory), ["<action>idle()</action>"])

            with self.assertRaisesRegex(ValueError, "event stream"):
                runtime.create_session(instruction="Hidden instruction", mode="g1")
            with self.assertRaisesRegex(ValueError, "only creates g1"):
                runtime.create_session(mode="v6")


class G1AppConfigurationTests(unittest.TestCase):
    def test_build_runtime_locks_g1_prompt_stream_parser_and_base_model_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                "POLICY_MODE": "tinker",
                "POLICY_PROMPT": "g1",
                "TINKER_MODEL_PATH": "",
                "SEARCH_MODE": "demo",
                "UIGEN_MODE": "demo",
                "TRACE_PATH": str(Path(directory) / "trace.jsonl"),
            }
            with patch.dict(os.environ, environment, clear=True):
                runtime = main_module.build_runtime()

            self.assertEqual(runtime.stream_format, "g1")
            self.assertEqual(runtime.action_schema, "g1")
            self.assertIsInstance(runtime.policy, TinkerPolicy)
            self.assertEqual(runtime.policy.system_prompt, SYSTEM_PROMPT_G1)
            self.assertEqual(runtime.policy.action_schema, "g1")
            self.assertEqual(runtime.policy.base_model, "Qwen/Qwen3.5-4B")

    def test_build_runtime_rejects_mixed_g1_and_legacy_contract(self) -> None:
        environment = {
            "POLICY_MODE": "tinker",
            "POLICY_PROMPT": "g1",
            "STREAM_FORMAT": "flat",
            "TINKER_MODEL_PATH": "",
            "SEARCH_MODE": "demo",
            "UIGEN_MODE": "demo",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(ValueError, "select the g1 contract together"):
                main_module.build_runtime()

    def test_api_auto_creates_g1_session_and_ticks_empty_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = make_runtime(
                Path(directory),
                ["<action>idle()</action>"],
            )
            with patch.object(main_module, "runtime", runtime):
                with TestClient(main_module.app) as client:
                    health = client.get("/health").json()
                    created = client.post(
                        "/api/sessions",
                        json={"instruction": "", "mode": "probe"},
                    ).json()
                    tick = client.post(
                        f"/api/sessions/{created['id']}/tick",
                        json={"text": "", "state": "idle"},
                    )
                    rejected = client.post(
                        "/api/sessions",
                        json={"instruction": "Hidden instruction", "mode": "g1"},
                    )

            self.assertEqual(health["action_schema"], "g1")
            self.assertEqual(health["stream_format"], "g1")
            self.assertEqual(created["mode"], "g1")
            self.assertEqual(created["instruction"], "")
            self.assertEqual(tick.status_code, 200)
            self.assertEqual(tick.json()["turn"]["event"]["content"], "")
            self.assertNotIn("<interaction_context>", tick.json()["session"]["latest_prompt"])
            self.assertEqual(rejected.status_code, 400)


if __name__ == "__main__":
    unittest.main()
