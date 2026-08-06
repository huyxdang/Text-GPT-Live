import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from app.domain import ActionKind, Document, EventSource, UserState
from app.policy import ScriptedEditorPolicy
from app.runtime import InteractionRuntime, JsonlTraceWriter
from app.writer import DemoWriterProvider, split_complete_sentences


class SequencePolicy:
    mode = "test"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    async def predict(self, compiled_stream, current, history) -> str:
        del compiled_stream, current, history
        if not self.outputs:
            return "<action>idle()</action>"
        return self.outputs.pop(0)


class ImmediateSearch:
    mode = "test"

    async def search(self, query: str):
        return {"query": query, "results": []}


class DocumentTests(unittest.TestCase):
    def test_undo_restores_exact_prior_text(self) -> None:
        document = Document()
        document.append("The first sentence.")
        document.append("The second sentence.")
        before = document.text
        document.apply_patch("The second sentence.", 1, "A calmer second sentence.")
        self.assertIn("A calmer second sentence.", document.text)
        self.assertTrue(document.undo())
        self.assertEqual(document.text, before)
        self.assertFalse(document.undo())

    def test_apply_patch_rejects_missing_target(self) -> None:
        document = Document()
        document.append("Only one sentence.")
        with self.assertRaises(ValueError):
            document.apply_patch("Nonexistent.", 1, "x")

    def test_sentence_splitter_handles_partial_buffers(self) -> None:
        sentences, rest = split_complete_sentences("One. Two! Thr")
        self.assertEqual(sentences, ["One.", "Two!"])
        self.assertEqual(rest, "Thr")


class EditorRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def make_runtime(self, policy) -> InteractionRuntime:
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        runtime = InteractionRuntime(
            policy=policy,
            search_provider=ImmediateSearch(),
            trace_writer=JsonlTraceWriter(Path(directory) / "trace.jsonl"),
            writer_provider=DemoWriterProvider(sentence_delay_seconds=0.01, sentence_count=4),
        )
        self.addAsyncCleanup(runtime.shutdown)
        return runtime

    async def wait_for(self, condition, timeout: float = 2.0) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while not condition():
            if asyncio.get_event_loop().time() > deadline:
                raise AssertionError("Timed out waiting for condition")
            await asyncio.sleep(0.01)

    async def test_write_pause_underline_revise_resume_loop(self) -> None:
            runtime = self.make_runtime(
            SequencePolicy(
                    [
                        '<action>tool(writer,{"operation":"write","instruction":"a story about a lighthouse"})</action>',
                    ]
                ),
            )
            session = runtime.create_session(mode="editor")

            turn = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Write me a story about a lighthouse.",
                state=UserState.IDLE,
            )
            self.assertTrue(turn.action.valid, turn.action.diagnostic)
            self.assertEqual(session.document.status, "writing")

            await self.wait_for(
                lambda: sum(
                    1 for t in session.history if t.event.source is EventSource.WRITER
                ) >= 2
            )

            # Pause: no further writer events after the pause is processed.
            runtime.policy.outputs.append('<action>tool(writer,{"operation":"pause"})</action>')
            pause_turn = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Stop — I don't like the second sentence.",
                state=UserState.ACTIVE,
            )
            self.assertTrue(pause_turn.action.valid, pause_turn.action.diagnostic)
            self.assertEqual(session.document.status, "paused")
            writer_count = sum(
                1 for t in session.history if t.event.source is EventSource.WRITER
            )
            await asyncio.sleep(0.08)
            self.assertEqual(
                writer_count,
                sum(1 for t in session.history if t.event.source is EventSource.WRITER),
            )

            sentences = [
                t.event.content for t in session.history if t.event.source is EventSource.WRITER
            ]
            second = sentences[1]

            # Underline the second sentence (a proposal, not a mutation).
            target = json.dumps({"operation": "underline", "target": {"occurrence": 1, "quote": second}})
            runtime.policy.outputs.append(f"<action>tool(ui,{target})</action>")
            document_before = session.document.text
            underline_turn = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Stop — I don't like the second sentence.",
                state=UserState.IDLE,
            )
            self.assertTrue(underline_turn.action.valid, underline_turn.action.diagnostic)
            self.assertEqual(session.proposal.quote, second)
            self.assertEqual(session.document.text, document_before)

            # Confirmation question targets the user's objection event.
            payload = json.dumps({"for": pause_turn.event.index, "message": "The second sentence — this one?"})
            runtime.policy.outputs.append(f"<action>respond({payload})</action>")
            confirm_turn = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Stop — I don't like the second sentence.",
                state=UserState.IDLE,
            )
            self.assertTrue(confirm_turn.action.valid, confirm_turn.action.diagnostic)
            self.assertEqual(confirm_turn.action.kind, ActionKind.RESPOND)

            # Confirm → revise: only the proposed span changes; undo restores it.
            runtime.policy.outputs.append(
                '<action>tool(writer,{"operation":"revise","instruction":"less dramatic"})</action>'
            )
            revise_turn = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Yes — make it less dramatic.",
                state=UserState.IDLE,
            )
            self.assertTrue(revise_turn.action.valid, revise_turn.action.diagnostic)
            await self.wait_for(
                lambda: any(
                    t.event.source is EventSource.TOOL and t.event.tool_name == "writer"
                    for t in session.history
                )
            )
            result = json.loads(
                next(
                    t.event.content
                    for t in session.history
                    if t.event.source is EventSource.TOOL and t.event.tool_name == "writer"
                )
            )
            self.assertNotIn("error", result)
            self.assertNotIn(second, session.document.text)
            self.assertIn("less dramatic", session.document.text)
            self.assertIsNone(session.proposal)
            unchanged = [s for s in sentences if s != second]
            for sentence in unchanged:
                self.assertIn(sentence, session.document.text)

            # Resume after the successful revision result.
            runtime.policy.outputs.append('<action>tool(writer,{"operation":"resume"})</action>')
            resumed = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="Looks good.",
                state=UserState.IDLE,
            )
            self.assertTrue(resumed.action.valid, resumed.action.diagnostic)
            self.assertEqual(session.document.status, "writing")

            # Undo restores the pre-revision text exactly.
            await runtime.reset_session(session.id)

    async def test_editor_guardrails(self) -> None:
            runtime = self.make_runtime(
            SequencePolicy(
                    [
                        '<action>tool(writer,{"operation":"pause"})</action>',
                        '<action>tool(writer,{"operation":"revise","instruction":"softer"})</action>',
                        '<action>tool(ui,{"operation":"underline","target":{"occurrence":1,"quote":"ghost"}})</action>',
                        '<action>tool(writer,{"operation":"write","instruction":"a tale"})</action>',
                    ]
                ),
            )
            session = runtime.create_session(mode="editor")

            paused = await runtime.process_event(
                session.id, source=EventSource.USER, content="stop", state=UserState.IDLE
            )
            self.assertFalse(paused.action.valid)
            self.assertIn("no active writing", paused.action.diagnostic.lower())

            revise = await runtime.process_event(
                session.id, source=EventSource.USER, content="yes fix it", state=UserState.IDLE
            )
            self.assertFalse(revise.action.valid)
            self.assertIn("proposal", revise.action.diagnostic)

            underline = await runtime.process_event(
                session.id, source=EventSource.USER, content="that ghost part", state=UserState.IDLE
            )
            self.assertFalse(underline.action.valid)
            self.assertIn("does not exist in the document", underline.action.diagnostic)

            # Probe sessions have no document: writer actions are rejected.
            probe = runtime.create_session()
            runtime.policy.outputs.append(
                '<action>tool(writer,{"operation":"write","instruction":"a tale"})</action>'
            )
            denied = await runtime.process_event(
                probe.id, source=EventSource.USER, content="write a tale", state=UserState.IDLE
            )
            self.assertFalse(denied.action.valid)
            self.assertIn("document session", denied.action.diagnostic)

    async def test_respond_may_target_user_event_for_confirmation(self) -> None:
            runtime = self.make_runtime(
            SequencePolicy(['<action>respond({"for":1,"message":"This part?"})</action>']),
            )
            session = runtime.create_session(mode="editor")
            turn = await runtime.process_event(
                session.id,
                source=EventSource.USER,
                content="I don't like that part.",
                state=UserState.ACTIVE,
            )
            self.assertTrue(turn.action.valid, turn.action.diagnostic)
            self.assertEqual(turn.action.kind, ActionKind.RESPOND)


class ScriptedPolicyLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_scripted_policy_drives_full_t1_loop(self) -> None:
            directory = tempfile.mkdtemp()
            self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
            runtime = InteractionRuntime(
                policy=ScriptedEditorPolicy(),
                search_provider=ImmediateSearch(),
                trace_writer=JsonlTraceWriter(Path(directory) / "trace.jsonl"),
                writer_provider=DemoWriterProvider(sentence_delay_seconds=0.01, sentence_count=4),
            )
            self.addAsyncCleanup(runtime.shutdown)
            session = runtime.create_session(mode="editor")

            async def user(text: str, state: UserState = UserState.IDLE):
                return await runtime.process_event(
                    session.id, source=EventSource.USER, content=text, state=state
                )

            start = await user("Write me a short story about a lighthouse.")
            self.assertEqual(start.action.arguments.get("operation"), "write")

            deadline = asyncio.get_event_loop().time() + 2
            while (
                sum(1 for t in session.history if t.event.source is EventSource.WRITER) < 2
            ):
                self.assertLess(asyncio.get_event_loop().time(), deadline)
                await asyncio.sleep(0.01)

            stop = await user("Stop — I don't like the second sentence.", UserState.ACTIVE)
            self.assertEqual(stop.action.arguments.get("operation"), "pause")
            self.assertEqual(session.document.status, "paused")

            propose = await user("Stop — I don't like the second sentence.")
            self.assertEqual(propose.action.arguments.get("operation"), "underline")
            sentences = [
                t.event.content for t in session.history if t.event.source is EventSource.WRITER
            ]
            self.assertEqual(session.proposal.quote, sentences[1])

            confirm = await user("Stop — I don't like the second sentence.")
            self.assertEqual(confirm.action.kind, ActionKind.RESPOND)

            revise = await user("Yes — make it less dramatic.")
            self.assertEqual(revise.action.arguments.get("operation"), "revise")

            deadline = asyncio.get_event_loop().time() + 2
            while session.proposal is not None:
                self.assertLess(asyncio.get_event_loop().time(), deadline)
                await asyncio.sleep(0.01)
            self.assertIn("less dramatic", session.document.text)

            # The policy resumes on the revision result event without user prompting.
            deadline = asyncio.get_event_loop().time() + 2
            while session.document.status != "writing":
                self.assertLess(asyncio.get_event_loop().time(), deadline)
                await asyncio.sleep(0.01)
            resume_turn = next(
                t
                for t in session.history
                if t.action.valid
                and t.action.tool_name == "writer"
                and t.action.arguments.get("operation") == "resume"
            )
            self.assertEqual(resume_turn.event.source, EventSource.TOOL)

            # A repeated snapshot of the confirmation must not re-fire revise.
            repeat = await user("Yes — make it less dramatic.")
            self.assertTrue(repeat.action.valid, repeat.action.diagnostic)
            self.assertEqual(repeat.action.kind, ActionKind.IDLE)


if __name__ == "__main__":
    unittest.main()
