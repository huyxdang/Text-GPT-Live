from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.domain import (
    Action,
    ActionKind,
    CompletedTurn,
    DEFAULT_PERMISSIONS,
    Document,
    EDITOR_PERMISSIONS,
    EventSource,
    G1_PERMISSIONS,
    Session,
    StreamEvent,
    TextHighlight,
    TextSuggestion,
    TranslationCommit,
    UserState,
    V6_PERMISSIONS,
    WRITER_STATUS_IDLE,
    WRITER_STATUS_PAUSED,
    WRITER_STATUS_WRITING,
)
from app.policy import Policy
from app.search import SearchProvider
from app.stream import compile_stream, parse_action, parse_g1_action
from app.uigen import UI_ACCENTS, UIGenProvider, validate_ui_spec
from app.writer import WriterProvider


class JsonlTraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        session_id: str,
        turn: CompletedTurn,
        compiled_prompt: str,
        action_schema: str = "legacy",
    ) -> None:
        record = {
            "recorded_at": time.time(),
            "session_id": session_id,
            "event": turn.event.to_dict(),
            "action": turn.action.to_dict(),
            "decision_ms": turn.decision_ms,
            "compiled_prompt": compiled_prompt,
            "action_schema": action_schema,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class InteractionRuntime:
    def __init__(
        self,
        *,
        policy: Policy,
        search_provider: SearchProvider,
        trace_writer: JsonlTraceWriter,
        stream_format: str = "flat",
        writer_provider: WriterProvider | None = None,
        uigen_provider: UIGenProvider | None = None,
        simulated_tick_ms: int | None = None,
    ) -> None:
        self.policy = policy
        self.search_provider = search_provider
        self.writer_provider = writer_provider
        self.uigen_provider = uigen_provider
        self.trace_writer = trace_writer
        self.stream_format = stream_format
        self.action_schema = "g1" if stream_format == "g1" else "legacy"
        # When set, event timestamps are index * simulated_tick_ms instead of
        # wall clock, so compiled prompts are identical across policies no
        # matter how long each decision takes (evaluation determinism).
        self.simulated_tick_ms = simulated_tick_ms
        self.sessions: dict[str, Session] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._tasks: defaultdict[str, set[asyncio.Task[None]]] = defaultdict(set)
        self._writer_tasks: dict[str, asyncio.Task[None]] = {}
        # Rolling window of policy.predict durations (ms) across all sessions,
        # feeding /health percentiles. The acceptance budget is one tick (650 ms).
        self._decision_times_ms: deque[int] = deque(maxlen=200)

    def create_session(
        self,
        *,
        instruction: str = "",
        permissions: dict[str, list[str]] | None = None,
        mode: str = "probe",
    ) -> Session:
        session_id = uuid.uuid4().hex[:12]
        if self.action_schema == "g1":
            if mode == "probe":
                mode = "g1"
            if mode != "g1":
                raise ValueError("g1 runtime only creates g1 sessions.")
            if instruction.strip():
                raise ValueError(
                    "g1 standing instructions belong in the event stream, not session headers."
                )
            if permissions is not None and permissions != G1_PERMISSIONS:
                raise ValueError("g1 sessions use one fixed runtime permission surface.")
            permissions = G1_PERMISSIONS
        elif permissions is None:
            if mode == "editor":
                permissions = EDITOR_PERMISSIONS
            elif mode == "v6":
                permissions = V6_PERMISSIONS
            else:
                permissions = DEFAULT_PERMISSIONS
        session = Session(
            id=session_id,
            instruction=instruction.strip(),
            mode=mode,
            permissions={
                name: list(operations) for name, operations in permissions.items()
            },
            document=Document() if mode == "editor" else None,
            started_monotonic=time.monotonic(),
        )
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown session: {session_id}") from exc

    def pending_count(self, session_id: str) -> int:
        return sum(not task.done() for task in self._tasks[session_id])

    def latency_summary(self) -> dict[str, int | None]:
        """Percentiles over the rolling window of policy decision times.

        Decision latency is the project's acceptance metric (budget: one 650 ms
        tick), so the serving path must always be able to report it.
        """
        times = sorted(self._decision_times_ms)
        if not times:
            return {"n": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
        return {
            "n": len(times),
            "p50_ms": times[(len(times) - 1) // 2],
            "p95_ms": times[min(len(times) - 1, (len(times) * 95) // 100)],
            "max_ms": times[-1],
        }

    async def reset_session(self, session_id: str) -> Session:
        async with self._locks[session_id]:
            session = self.get_session(session_id)
            session.generation += 1
            self._cancel_writer(session_id)
            for task in list(self._tasks[session_id]):
                task.cancel()
            self._tasks[session_id].clear()
            session.history.clear()
            session.highlights.clear()
            session.suggestions.clear()
            session.translation_commits.clear()
            session.job_specs.clear()
            session.proposal = None
            if session.document is not None:
                session.document = Document()
            session.next_index = 1
            session.started_monotonic = time.monotonic()
            session.latest_prompt = ""
            reset_stream_cache = getattr(self.policy, "reset_stream_cache", None)
            if reset_stream_cache is not None:
                reset_stream_cache(session_id)
            return session

    async def undo_document(self, session_id: str) -> Session:
        """User-initiated undo: restores the exact prior document text."""
        async with self._locks[session_id]:
            session = self.get_session(session_id)
            if session.document is None:
                raise ValueError("This session has no document.")
            session.document.undo()
            if session.proposal is not None and not _quote_occurrence_exists(
                session.document.text, session.proposal.quote, session.proposal.occurrence
            ):
                session.proposal = None
            return session

    def _cancel_writer(self, session_id: str) -> None:
        task = self._writer_tasks.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def process_event(
        self,
        session_id: str,
        *,
        source: EventSource,
        content: str,
        state: UserState | None = None,
        tool_name: str | None = None,
        call_id: str | None = None,
        job_id: str | None = None,
    ) -> CompletedTurn:
        search_to_start: tuple[str, int, str] | None = None
        job_to_start: tuple[str, str, dict[str, Any], int] | None = None
        async with self._locks[session_id]:
            session = self.get_session(session_id)
            if source is EventSource.USER:
                session.highlights = [
                    highlight
                    for highlight in session.highlights
                    if _quote_occurrence_exists(content, highlight.quote, highlight.occurrence)
                ]
                session.suggestions = [
                    suggestion
                    for suggestion in session.suggestions
                    if _quote_occurrence_exists(content, suggestion.quote, suggestion.occurrence)
                ]
                # Translation is append-only only while the current textbox still
                # extends every committed source snapshot. Editing or clearing an
                # earlier span invalidates that commit and every later one.
                kept_commits: list[TranslationCommit] = []
                for commit in session.translation_commits:
                    if content.startswith(commit.source_snapshot):
                        kept_commits.append(commit)
                    else:
                        break
                session.translation_commits = kept_commits
            event = StreamEvent(
                index=session.next_index,
                source=source,
                content=content,
                state=state,
                elapsed_ms=(
                    session.next_index * self.simulated_tick_ms
                    if self.simulated_tick_ms is not None
                    else round((time.monotonic() - session.started_monotonic) * 1000)
                ),
                tool_name=tool_name,
                call_id=call_id,
                job_id=job_id,
            )
            compile_options: dict[str, Any] = {"fmt": self.stream_format}
            if self.action_schema != "g1":
                compile_options.update(
                    instruction=session.instruction,
                    permissions=session.permissions,
                )
            compiled_prompt = compile_stream(session.history, event, **compile_options)
            session.latest_prompt = compiled_prompt
            predict_started = time.perf_counter()
            try:
                predict_session = getattr(self.policy, "predict_session", None)
                if predict_session is None:
                    raw_output = await self.policy.predict(
                        compiled_prompt, event, session.history.copy()
                    )
                else:
                    raw_output = await predict_session(
                        session_id, compiled_prompt, event, session.history.copy()
                    )
            except Exception as exc:  # The trace must survive a provider failure.
                raw_output = f"<policy_error>{type(exc).__name__}: {exc}</policy_error>"
            decision_ms = round((time.perf_counter() - predict_started) * 1000)
            self._decision_times_ms.append(decision_ms)
            action = (
                parse_g1_action(raw_output)
                if self.action_schema == "g1"
                else parse_action(raw_output)
            )
            action = self._validate_action(session, event, action)
            turn = CompletedTurn(event=event, action=action, decision_ms=decision_ms)
            session.history.append(turn)
            session.next_index += 1
            self.trace_writer.write(
                session_id=session_id,
                turn=turn,
                compiled_prompt=compiled_prompt,
                action_schema=self.action_schema,
            )
            writer_to_start: tuple[str, int, dict[str, Any]] | None = None
            if session.mode == "g1":
                if action.kind is ActionKind.TOOL and action.tool_name == "delegate":
                    job_to_start = (
                        "delegate",
                        f"job-{event.index}",
                        {"task": str(action.arguments.get("task", ""))},
                        session.generation,
                    )
                elif action.kind is ActionKind.TOOL and action.tool_name == "suggest_edit":
                    session.suggestions.append(
                        TextSuggestion(
                            quote=str(action.arguments["quote"]),
                            occurrence=1,
                            replacement=str(action.arguments["replacement"]),
                            event_index=event.index,
                        )
                    )
                elif action.kind is ActionKind.TOOL and action.tool_name == "highlight":
                    session.highlights.append(
                        TextHighlight(
                            quote=str(action.arguments["quote"]),
                            occurrence=int(action.arguments["occurrence"]),
                            event_index=event.index,
                        )
                    )
                elif action.kind is ActionKind.TOOL and action.tool_name == "web_search":
                    job_to_start = (
                        "web_search",
                        f"job-{event.index}",
                        {"query": str(action.arguments.get("query", ""))},
                        session.generation,
                    )
                elif action.kind is ActionKind.TOOL and action.tool_name == "translate_commit":
                    target_index = int(action.arguments["for"])
                    target_event = event if target_index == event.index else next(
                        turn.event
                        for turn in session.history
                        if turn.event.index == target_index
                    )
                    session.translation_commits.append(
                        TranslationCommit(
                            target_event_index=target_index,
                            source_snapshot=target_event.content,
                            message=str(action.arguments["message"]),
                            action_event_index=event.index,
                        )
                    )
            elif session.mode == "v6":
                if action.kind is ActionKind.TOOL and action.tool_name == "web_search" and action.query:
                    job_to_start = (
                        "web_search",
                        f"job-{event.index}",
                        {"query": action.query},
                        session.generation,
                    )
                elif action.kind is ActionKind.TOOL and action.tool_name == "generate_ui":
                    job_to_start = (
                        "generate_ui",
                        f"job-{event.index}",
                        {"request": str(action.arguments.get("request", ""))},
                        session.generation,
                    )
                elif action.kind is ActionKind.TOOL and action.tool_name == "suggest_edit":
                    target = action.arguments["target"]
                    session.suggestions.append(
                        TextSuggestion(
                            quote=target["quote"],
                            occurrence=target["occurrence"],
                            replacement=action.arguments["replacement"],
                            event_index=event.index,
                        )
                    )
            elif action.kind is ActionKind.TOOL and action.tool_name == "web_search" and action.query:
                search_to_start = (action.query, session.generation, f"call-{event.index}")
            elif action.kind is ActionKind.TOOL and action.tool_name == "ui":
                operation = action.arguments.get("operation")
                target = action.arguments["target"]
                if operation == "underline":
                    # A proposal, never content: one live underline, replaced by the next.
                    session.proposal = TextHighlight(
                        quote=target["quote"],
                        occurrence=target["occurrence"],
                        event_index=event.index,
                    )
                else:
                    session.highlights.append(
                        TextHighlight(
                            quote=target["quote"],
                            occurrence=target["occurrence"],
                            event_index=event.index,
                        )
                    )
            elif action.kind is ActionKind.TOOL and action.tool_name == "writer":
                writer_to_start = self._execute_writer_action(session, event, action)

        if search_to_start:
            query, generation, call_id = search_to_start
            task = asyncio.create_task(self._run_search(session_id, query, generation, call_id))
            self._tasks[session_id].add(task)
            task.add_done_callback(self._tasks[session_id].discard)
        if job_to_start:
            tool, new_job_id, arguments, generation = job_to_start
            task = asyncio.create_task(
                self._run_job(session_id, tool, new_job_id, arguments, generation)
            )
            self._tasks[session_id].add(task)
            task.add_done_callback(self._tasks[session_id].discard)
        if writer_to_start:
            kind, generation, spec = writer_to_start
            if kind == "stream":
                task = asyncio.create_task(self._run_writer(session_id, generation))
                self._writer_tasks[session_id] = task
            else:
                task = asyncio.create_task(self._run_revision(session_id, generation, spec))
            self._tasks[session_id].add(task)
            task.add_done_callback(self._tasks[session_id].discard)
        return turn

    def _execute_writer_action(
        self,
        session: Session,
        event: StreamEvent,
        action: Action,
    ) -> tuple[str, int, dict[str, Any]] | None:
        """Apply a validated writer action to session state. Runs under the session lock.

        Returns (task_kind, generation, spec) when a background task must start
        after the lock is released.
        """
        del event
        document = session.document
        assert document is not None
        operation = action.arguments.get("operation")
        if operation == "pause":
            # The reflex: stop requesting content immediately. At most the sentence
            # already committed by the writer task renders (T1 pass criterion).
            document.status = WRITER_STATUS_PAUSED
            self._cancel_writer(session.id)
            return None
        if operation == "write":
            document.task = str(action.arguments.get("instruction", "")).strip()
            document.status = WRITER_STATUS_WRITING
            return ("stream", session.generation, {})
        if operation == "resume":
            document.status = WRITER_STATUS_WRITING
            return ("stream", session.generation, {})
        if operation == "revise":
            assert session.proposal is not None  # guaranteed by validation
            instruction = str(action.arguments.get("instruction", "")).strip()
            document.preferences.append(instruction)
            spec = {
                "instruction": instruction,
                "quote": session.proposal.quote,
                "occurrence": session.proposal.occurrence,
                "revision": document.revision,
            }
            return ("revise", session.generation, spec)
        return None

    def _validate_action(self, session: Session, event: StreamEvent, action: Action) -> Action:
        if not action.valid:
            return action
        if session.mode == "g1":
            return self._validate_action_g1(session, event, action)
        if session.mode == "v6":
            return self._validate_action_v6(session, event, action)

        if action.kind is ActionKind.RESPOND:
            if event.source is not EventSource.USER:
                return _rejected(action, "Responses may only be surfaced on a user event.")
            target_turn = next(
                (turn for turn in session.history if turn.event.index == action.target),
                None,
            )
            target_event = target_turn.event if target_turn is not None else None
            if action.target == event.index:
                target_event = event
            is_search_result = (
                target_event is not None
                and target_event.source is EventSource.TOOL
                and target_event.tool_name == "web_search"
            )
            is_user_target = target_event is not None and target_event.source is EventSource.USER
            if is_search_result and event.state is not UserState.IDLE:
                return _rejected(action, "Responses may only be surfaced when the user is idle.")
            if not is_search_result and not is_user_target:
                return _rejected(
                    action,
                    "Respond target must be a web_search result or a user event.",
                )
            if any(
                turn.action.kind is ActionKind.RESPOND
                and turn.action.valid
                and turn.action.target == action.target
                for turn in session.history
            ):
                return _rejected(action, "That event has already been answered.")
            return action

        if action.kind is not ActionKind.TOOL:
            return action

        tool_name = action.tool_name or ""
        if tool_name == "web_search":
            if event.source is not EventSource.USER:
                return _rejected(action, "web_search may only be called from a user event.")
            if "search" not in session.permissions.get("web_search", []):
                return _rejected(action, "web_search permission is not available.")
            query = action.arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return _rejected(action, 'web_search requires a non-empty string "query".')
            normalized_query = _normalize_query(query)
            if any(
                turn.action.valid
                and turn.action.kind is ActionKind.TOOL
                and turn.action.tool_name == "web_search"
                and _normalize_query(turn.action.query or "") == normalized_query
                for turn in session.history
            ):
                return _rejected(action, "An equivalent web_search is already pending or complete.")
            return replace(action, arguments={"query": query.strip()})

        if tool_name == "ui":
            if event.source is not EventSource.USER:
                return _rejected(action, "UI actions may only be taken on a user event.")
            operation = action.arguments.get("operation")
            if operation not in ("highlight", "underline"):
                return _rejected(action, "The UI tool supports highlight and underline operations.")
            if operation not in session.permissions.get("ui", []):
                return _rejected(action, f"ui.{operation} permission is not available.")
            target = action.arguments.get("target")
            if not isinstance(target, dict):
                return _rejected(action, f'ui.{operation} requires a "target" object.')
            quote = target.get("quote")
            occurrence = target.get("occurrence", 1)
            if not isinstance(quote, str) or not quote:
                return _rejected(action, f'ui.{operation} target requires a non-empty "quote".')
            if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 1:
                return _rejected(action, f'ui.{operation} target "occurrence" must be a positive integer.')
            if operation == "underline":
                # Underlines are proposals over the shared document.
                if session.document is None:
                    return _rejected(action, "ui.underline requires a document session.")
                if not _quote_occurrence_exists(session.document.text, quote, occurrence):
                    return _rejected(action, "ui.underline target does not exist in the document.")
                if (
                    session.proposal is not None
                    and session.proposal.quote == quote
                    and session.proposal.occurrence == occurrence
                ):
                    return _rejected(action, "That span is already the live proposal.")
            else:
                if not _quote_occurrence_exists(event.content, quote, occurrence):
                    return _rejected(action, "ui.highlight target does not exist in the current text.")
                if any(
                    highlight.quote == quote and highlight.occurrence == occurrence
                    for highlight in session.highlights
                ):
                    return _rejected(action, "ui.highlight target is already highlighted.")
            return replace(
                action,
                arguments={
                    "operation": operation,
                    "target": {"quote": quote, "occurrence": occurrence},
                },
            )

        if tool_name == "writer":
            document = session.document
            if document is None:
                return _rejected(action, "Writer actions require a document session.")
            if self.writer_provider is None:
                return _rejected(action, "No writer provider is configured.")
            operation = action.arguments.get("operation")
            allowed = session.permissions.get("writer", [])
            if not isinstance(operation, str) or operation not in ("write", "pause", "resume", "revise"):
                return _rejected(action, "The writer tool supports write, pause, resume, and revise.")
            if operation not in allowed:
                return _rejected(action, f"writer.{operation} permission is not available.")
            if operation == "write":
                instruction = action.arguments.get("instruction")
                if not isinstance(instruction, str) or not instruction.strip():
                    return _rejected(action, 'writer.write requires a non-empty "instruction".')
                if document.status == WRITER_STATUS_WRITING:
                    return _rejected(action, "The writer is already writing.")
                return replace(
                    action,
                    arguments={"operation": "write", "instruction": instruction.strip()},
                )
            if operation == "pause":
                if document.status != WRITER_STATUS_WRITING:
                    return _rejected(action, "There is no active writing to pause.")
                return replace(action, arguments={"operation": "pause"})
            if operation == "resume":
                if document.status != WRITER_STATUS_PAUSED:
                    return _rejected(action, "The writer is not paused.")
                if not document.task:
                    return _rejected(action, "There is no writing task to resume.")
                return replace(action, arguments={"operation": "resume"})
            # revise
            instruction = action.arguments.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                return _rejected(action, 'writer.revise requires a non-empty "instruction".')
            if session.proposal is None:
                return _rejected(action, "writer.revise requires a live underline proposal.")
            if document.status == WRITER_STATUS_WRITING:
                return _rejected(action, "Pause the writer before revising.")
            if not _quote_occurrence_exists(
                document.text, session.proposal.quote, session.proposal.occurrence
            ):
                return _rejected(action, "The proposed span no longer exists in the document.")
            return replace(
                action,
                arguments={"operation": "revise", "instruction": instruction.strip()},
            )

        return _rejected(action, f"Tool {tool_name!r} is not registered.")

    def _validate_action_g1(
        self,
        session: Session,
        event: StreamEvent,
        action: Action,
    ) -> Action:
        """Validate the exact flat g1 grammar used by training and serving."""
        if action.kind is ActionKind.RESPOND:
            target_event = next(
                (turn.event for turn in session.history if turn.event.index == action.target),
                None,
            )
            if action.target == event.index:
                target_event = event
            is_user_target = target_event is not None and target_event.source is EventSource.USER
            is_search_result = (
                target_event is not None
                and target_event.source is EventSource.TOOL
                and target_event.tool_name == "web_search"
                and _payload_status(target_event) in ("completed", "failed")
            )
            if not is_user_target and not is_search_result:
                return _rejected(
                    action,
                    "g1 respond target must be a user event or completed web_search result.",
                )
            if is_user_target and event.source is not EventSource.USER:
                return _rejected(
                    action,
                    "g1 user responses may only run while processing a user event.",
                )
            if is_search_result and (
                event.source is not EventSource.TOOL
                or target_event is not event
                or action.target != event.index
            ):
                return _rejected(
                    action,
                    "g1 search responses must target the current terminal web_search event.",
                )
            if any(
                turn.action.valid
                and turn.action.kind is ActionKind.RESPOND
                and turn.action.target == action.target
                for turn in session.history
            ):
                return _rejected(action, "That user event has already been answered.")
            return action

        if action.kind is not ActionKind.TOOL:
            return action
        tool_name = action.tool_name or ""
        if tool_name not in session.permissions.get("tools", []):
            return _rejected(action, f"Tool {tool_name!r} is not permitted in g1.")

        delegate_accepted = (
            event.source is EventSource.TOOL
            and event.tool_name == "delegate"
            and _payload_status(event) == "accepted"
        )
        if event.source is not EventSource.USER and not (
            tool_name == "web_search" and delegate_accepted
        ):
            return _rejected(action, "g1 tools may only run on a user event.")

        if tool_name == "highlight":
            quote = str(action.arguments["quote"])
            occurrence = int(action.arguments["occurrence"])
            if not _quote_occurrence_exists(event.content, quote, occurrence):
                return _rejected(action, "g1 highlight target does not exist in current text.")
            if any(
                highlight.quote == quote and highlight.occurrence == occurrence
                for highlight in session.highlights
            ):
                return _rejected(action, "That span is already highlighted.")
            return action

        if tool_name == "suggest_edit":
            quote = str(action.arguments["quote"])
            replacement = str(action.arguments["replacement"])
            if replacement == quote:
                return _rejected(action, "g1 suggestion must change the quoted text.")
            if event.content.count(quote) != 1:
                return _rejected(
                    action,
                    "g1 suggestion quote must identify exactly one current-text span.",
                )
            return action

        if tool_name == "delegate":
            if self.uigen_provider is None:
                return _rejected(action, "No delegate provider is configured.")
            task = str(action.arguments["task"])
            normalized_task = _normalize_query(task)
            if any(
                turn.action.valid
                and turn.action.kind is ActionKind.TOOL
                and turn.action.tool_name == "delegate"
                and _normalize_query(str(turn.action.arguments.get("task", "")))
                == normalized_task
                for turn in session.history
            ):
                return _rejected(action, "An equivalent delegate job already exists.")
            return action

        if tool_name == "web_search":
            query = str(action.arguments["query"]).strip()
            normalized_query = _normalize_query(query)
            if any(
                turn.action.valid
                and turn.action.kind is ActionKind.TOOL
                and turn.action.tool_name == "web_search"
                and _normalize_query(turn.action.query or "") == normalized_query
                for turn in session.history
            ):
                return _rejected(action, "An equivalent web_search already exists.")
            return replace(action, arguments={"query": query})

        if tool_name == "translate_commit":
            target_index = int(action.arguments["for"])
            target_event = event if target_index == event.index else next(
                (
                    turn.event
                    for turn in session.history
                    if turn.event.index == target_index
                ),
                None,
            )
            if target_event is None or target_event.source is not EventSource.USER:
                return _rejected(action, "Translation commit must target a user event.")
            if not event.content.startswith(target_event.content):
                return _rejected(
                    action,
                    "Current translation source no longer extends the target snapshot.",
                )
            if any(
                commit.target_event_index == target_index
                for commit in session.translation_commits
            ):
                return _rejected(action, "That source snapshot is already translated.")
            if session.translation_commits:
                previous = session.translation_commits[-1]
                if target_index <= previous.target_event_index:
                    return _rejected(action, "Translation commits must move forward.")
                if not target_event.content.startswith(previous.source_snapshot):
                    return _rejected(action, "Translation source no longer extends the prior commit.")
            return action

        return _rejected(action, f"Tool {tool_name!r} is not part of g1.")

    def _validate_action_v6(self, session: Session, event: StreamEvent, action: Action) -> Action:
        """Validation for the v6 surface: flat intent-named tools with job identity."""
        if action.kind is ActionKind.RESPOND:
            if event.source is not EventSource.USER:
                return _rejected(action, "Responses may only be surfaced on a user event.")
            target_event = next(
                (turn.event for turn in session.history if turn.event.index == action.target),
                None,
            )
            if action.target == event.index:
                target_event = event
            is_search_completion = (
                target_event is not None
                and target_event.source is EventSource.TOOL
                and target_event.tool_name == "web_search"
                and _payload_status(target_event) == "completed"
            )
            is_user_target = target_event is not None and target_event.source is EventSource.USER
            if is_search_completion and event.state is not UserState.IDLE:
                return _rejected(action, "Responses may only be surfaced when the user is idle.")
            if not is_search_completion and not is_user_target:
                return _rejected(
                    action,
                    "Respond target must be a completed web_search result or a user event.",
                )
            if any(
                turn.action.kind is ActionKind.RESPOND
                and turn.action.valid
                and turn.action.target == action.target
                for turn in session.history
            ):
                return _rejected(action, "That event has already been answered.")
            return action

        if action.kind is not ActionKind.TOOL:
            return action

        tool_name = action.tool_name or ""
        if tool_name not in ("web_search", "generate_ui", "suggest_edit"):
            return _rejected(action, f"Tool {tool_name!r} is not part of this surface.")
        if tool_name not in session.permissions.get("tools", []):
            return _rejected(action, f"Tool {tool_name!r} is not permitted.")
        if event.source is not EventSource.USER:
            return _rejected(action, "Tools may only be called from a user event.")

        if tool_name == "web_search":
            query = action.arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return _rejected(action, 'web_search requires a non-empty string "query".')
            normalized_query = _normalize_query(query)
            if any(
                turn.action.valid
                and turn.action.kind is ActionKind.TOOL
                and turn.action.tool_name == "web_search"
                and _normalize_query(turn.action.query or "") == normalized_query
                for turn in session.history
            ):
                return _rejected(action, "An equivalent web_search is already pending or complete.")
            return replace(action, arguments={"query": query.strip()})

        if tool_name == "generate_ui":
            if self.uigen_provider is None:
                return _rejected(action, "No UI generation provider is configured.")
            request = action.arguments.get("request")
            if not isinstance(request, str) or not request.strip():
                return _rejected(action, 'generate_ui requires a non-empty string "request".')
            normalized_request = _normalize_query(request)
            if any(
                turn.action.valid
                and turn.action.kind is ActionKind.TOOL
                and turn.action.tool_name == "generate_ui"
                and _normalize_query(str(turn.action.arguments.get("request", ""))) == normalized_request
                for turn in session.history
            ):
                return _rejected(action, "An equivalent generate_ui job is already pending or complete.")
            return replace(action, arguments={"request": request.strip()})

        # suggest_edit
        target = action.arguments.get("target")
        if not isinstance(target, dict):
            return _rejected(action, 'suggest_edit requires a "target" object.')
        quote = target.get("quote")
        occurrence = target.get("occurrence", 1)
        replacement = action.arguments.get("replacement")
        if not isinstance(quote, str) or not quote:
            return _rejected(action, 'suggest_edit target requires a non-empty "quote".')
        if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 1:
            return _rejected(action, 'suggest_edit target "occurrence" must be a positive integer.')
        if not isinstance(replacement, str) or not replacement.strip():
            return _rejected(action, 'suggest_edit requires a non-empty string "replacement".')
        if replacement.strip() == quote:
            return _rejected(action, "suggest_edit replacement must differ from the quoted span.")
        if not _quote_occurrence_exists(event.content, quote, occurrence):
            return _rejected(action, "suggest_edit target does not exist in the current text.")
        if any(
            suggestion.quote == quote and suggestion.occurrence == occurrence
            for suggestion in session.suggestions
        ):
            return _rejected(action, "That span already has a live suggestion.")
        return replace(
            action,
            arguments={
                "target": {"quote": quote, "occurrence": occurrence},
                "replacement": replacement.strip(),
            },
        )

    async def _run_job(
        self,
        session_id: str,
        tool: str,
        job_id: str,
        arguments: dict[str, Any],
        generation: int,
    ) -> None:
        """Run one async v6/g1 job: acknowledge, execute, then inject the result.

        Failures re-enter the stream as failed-job events, never as crashes, and
        every event carries the job identity so results route in any completion order.
        g1 keeps generated UI specs out of model-visible event bodies and stores
        them separately for the browser renderer.
        """
        try:
            accepted = (
                {"status": "accepted"}
                if tool == "delegate"
                else {"job_id": job_id, "status": "accepted"}
            )
            await self._inject_job_event(
                session_id, generation, tool, job_id, accepted
            )
            try:
                if tool == "web_search":
                    result: dict[str, Any] = await self.search_provider.search(arguments["query"])
                    if isinstance(result, dict) and result.get("error"):
                        payload = {
                            "job_id": job_id,
                            "status": "failed",
                            "query": arguments["query"],
                            "error": str(result["error"]),
                        }
                    else:
                        payload = {
                            "query": str(result.get("query", arguments["query"])),
                            "results": result.get("results", []),
                            "job_id": job_id,
                            "status": "completed",
                        }
                else:
                    assert self.uigen_provider is not None  # guaranteed by validation
                    request_key = "task" if tool == "delegate" else "request"
                    request = arguments[request_key]
                    spec = await self._build_ui_progressively(
                        session_id=session_id,
                        generation=generation,
                        job_id=job_id,
                        request=request,
                    )
                    problems = validate_ui_spec(spec)
                    if problems:
                        payload = {"status": "failed", request_key: request}
                        if tool != "delegate":
                            payload["job_id"] = job_id
                        payload["error"] = "Invalid UI spec: " + " ".join(problems)
                    else:
                        payload = {"status": "completed", request_key: request}
                        if tool == "delegate":
                            stored = await self._store_job_spec(
                                session_id, generation, job_id, spec
                            )
                            if not stored:
                                return
                        else:
                            payload.update({"job_id": job_id, "spec": spec})
            except Exception as exc:
                if tool == "delegate":
                    payload = {
                        "status": "failed",
                        "task": str(arguments["task"]),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                else:
                    payload = {
                        "job_id": job_id,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            await self._inject_job_event(session_id, generation, tool, job_id, payload)
        except asyncio.CancelledError:
            raise

    async def _build_ui_progressively(
        self,
        *,
        session_id: str,
        generation: int,
        job_id: str,
        request: str,
    ) -> dict[str, Any]:
        """Publish render-only UI build snapshots without changing model history.

        Partial build state belongs to the runtime/session API. The interaction
        policy still sees only the accepted and final lifecycle events, so the
        g1 grammar and training contract do not gain a new tool event type.
        """
        assert self.uigen_provider is not None
        stream_build = getattr(self.uigen_provider, "stream_build", None)
        if stream_build is None:
            # Compatibility path for old test/custom providers. Production
            # providers publish real build steps through stream_build.
            spec = await self.uigen_provider.generate(request)
            stored = await self._store_job_spec(session_id, generation, job_id, spec)
            if not stored:
                raise asyncio.CancelledError
            return spec

        draft: dict[str, Any] = {"title": "", "components": []}
        async for update in stream_build(request):
            kind = update.get("kind") if isinstance(update, dict) else None
            if kind == "title":
                title = update.get("title")
                if not isinstance(title, str) or not title.strip():
                    raise ValueError("UI build title must be a non-empty string.")
                if draft["title"]:
                    raise ValueError("UI build may publish its title only once.")
                draft["title"] = title.strip()
                # The panel accent is chosen once, with the plan; it only joins
                # the draft when the provider actually supplies one.
                accent = update.get("accent")
                if accent is not None:
                    if accent not in UI_ACCENTS:
                        raise ValueError(f"UI build accent must be one of {UI_ACCENTS}.")
                    draft["accent"] = accent
            elif kind == "component":
                component = update.get("component")
                if not draft["title"]:
                    raise ValueError("UI build must publish a title before components.")
                if not isinstance(component, dict):
                    raise ValueError("UI build component must be a JSON object.")
                candidate = {
                    "title": draft["title"],
                    "components": [*draft["components"], component],
                }
                problems = validate_ui_spec(candidate)
                if problems:
                    raise ValueError("Invalid UI build component: " + " ".join(problems))
                draft["components"].append(component)
            else:
                raise ValueError("UI build update must be a title or component.")

            stored = await self._store_job_spec(session_id, generation, job_id, draft)
            if not stored:
                raise asyncio.CancelledError

        return draft

    async def _store_job_spec(
        self,
        session_id: str,
        generation: int,
        job_id: str,
        spec: dict[str, Any],
    ) -> bool:
        """Store a render-only partial or final build without model-history events."""
        async with self._locks[session_id]:
            session = self.get_session(session_id)
            if session.generation != generation:
                return False
            session.job_specs[job_id] = spec
            return True

    async def _inject_job_event(
        self,
        session_id: str,
        generation: int,
        tool: str,
        job_id: str,
        payload: dict[str, Any],
    ) -> None:
        session = self.get_session(session_id)
        if session.generation != generation:
            return
        await self.process_event(
            session_id,
            source=EventSource.TOOL,
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            tool_name=tool,
            job_id=job_id,
        )

    async def _run_search(
        self,
        session_id: str,
        query: str,
        generation: int,
        call_id: str,
    ) -> None:
        try:
            try:
                result: dict[str, Any] = await self.search_provider.search(query)
            except Exception as exc:
                result = {
                    "query": query,
                    "error": f"{type(exc).__name__}: {exc}",
                    "results": [],
                }
            session = self.get_session(session_id)
            if session.generation != generation:
                return
            await self.process_event(
                session_id,
                source=EventSource.TOOL,
                content=json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                tool_name="web_search",
                call_id=call_id,
            )
        except asyncio.CancelledError:
            raise

    async def _run_writer(self, session_id: str, generation: int) -> None:
        """Stream committed sentences from the writer into the document and the event stream.

        Each sentence is committed to the document under the session lock, then
        re-enters the chronological stream as a writer event. A pause lands between
        sentences: the committed sentence may render, but nothing further is
        requested from the provider (the task is cancelled).
        """
        if self.writer_provider is None:
            return
        session = self.get_session(session_id)
        document = session.document
        if document is None:
            return
        try:
            iterator = self.writer_provider.stream_sentences(
                task=document.task,
                document_text=document.text,
                preferences=list(document.preferences),
            )
            async for sentence in iterator:
                async with self._locks[session_id]:
                    if (
                        session.generation != generation
                        or document.status != WRITER_STATUS_WRITING
                    ):
                        return
                    document.append(sentence)
                await self.process_event(
                    session_id,
                    source=EventSource.WRITER,
                    content=sentence,
                )
            async with self._locks[session_id]:
                if session.generation == generation and document.status == WRITER_STATUS_WRITING:
                    document.status = WRITER_STATUS_IDLE
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._inject_writer_result(
                session_id,
                generation,
                {"operation": "write", "error": f"{type(exc).__name__}: {exc}"},
            )

    async def _run_revision(self, session_id: str, generation: int, spec: dict[str, Any]) -> None:
        """Fetch a scoped rewrite for the confirmed proposal and apply it as a patch."""
        if self.writer_provider is None:
            return
        session = self.get_session(session_id)
        document = session.document
        if document is None:
            return
        result: dict[str, Any]
        try:
            replacement = await self.writer_provider.revise(
                task=document.task,
                document_text=document.text,
                span=str(spec["quote"]),
                instruction=str(spec["instruction"]),
                preferences=list(document.preferences),
            )
            async with self._locks[session_id]:
                if session.generation != generation:
                    return
                if document.revision != spec["revision"]:
                    result = {
                        "operation": "revise",
                        "error": "The document changed while the revision was in flight.",
                    }
                else:
                    document.apply_patch(
                        str(spec["quote"]), int(spec["occurrence"]), replacement
                    )
                    session.proposal = None
                    result = {
                        "operation": "revise",
                        "replaced": spec["quote"],
                        "replacement": replacement,
                    }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result = {"operation": "revise", "error": f"{type(exc).__name__}: {exc}"}
        await self._inject_writer_result(session_id, generation, result)

    async def _inject_writer_result(
        self,
        session_id: str,
        generation: int,
        result: dict[str, Any],
    ) -> None:
        session = self.get_session(session_id)
        if session.generation != generation:
            return
        await self.process_event(
            session_id,
            source=EventSource.TOOL,
            content=json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            tool_name="writer",
        )

    async def wait_for_background(self, session_id: str, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while self.pending_count(session_id):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Background search did not finish in time.")
            await asyncio.sleep(min(0.02, remaining))

    async def shutdown(self) -> None:
        tasks = [task for session_tasks in self._tasks.values() for task in session_tasks]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        close = getattr(self.policy, "aclose", None)
        if close is not None:
            await close()
        writer_close = getattr(self.writer_provider, "aclose", None)
        if writer_close is not None:
            await writer_close()


def _payload_status(event: StreamEvent) -> str | None:
    try:
        payload = json.loads(event.content)
    except ValueError:
        return None
    status = payload.get("status") if isinstance(payload, dict) else None
    return status if isinstance(status, str) else None


def _quote_occurrence_exists(text: str, quote: str, occurrence: int) -> bool:
    start = -1
    for _ in range(occurrence):
        start = text.find(quote, start + 1)
        if start < 0:
            return False
    return True


def _normalize_query(query: str) -> str:
    return " ".join(query.casefold().split()).rstrip("?.!")


def _rejected(action: Action, diagnostic: str) -> Action:
    return Action(
        kind=ActionKind.IDLE,
        valid=False,
        raw_output=action.raw_output,
        diagnostic=diagnostic,
    )
