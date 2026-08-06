from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EventSource(StrEnum):
    USER = "user"
    TOOL = "tool"
    WRITER = "writer"


class UserState(StrEnum):
    ACTIVE = "active"
    IDLE = "idle"


class ActionKind(StrEnum):
    IDLE = "idle"
    TOOL = "tool"
    RESPOND = "respond"


DEFAULT_PERMISSIONS: dict[str, list[str]] = {
    "web_search": ["search"],
    "ui": ["highlight"],
}

# Editor sessions add the co-writing verbs: the shared document, reversible
# underline proposals, and pause/resume/revise control over the background writer.
EDITOR_PERMISSIONS: dict[str, list[str]] = {
    "web_search": ["search"],
    "ui": ["highlight", "underline"],
    "writer": ["write", "pause", "resume", "revise"],
}

# The v6 surface uses flat, intent-named tools; async tools carry job identity.
V6_PERMISSIONS: dict[str, list[str]] = {
    "tools": ["generate_ui", "suggest_edit", "web_search"],
}

# g1 permissions are runtime-only guardrails. They are fixed for every g1
# session and deliberately never serialized into the header-free model prompt.
G1_PERMISSIONS: dict[str, list[str]] = {
    "tools": [
        "delegate",
        "highlight",
        "suggest_edit",
        "translate_commit",
        "web_search",
    ],
}

WRITER_STATUS_IDLE = "idle"
WRITER_STATUS_WRITING = "writing"
WRITER_STATUS_PAUSED = "paused"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    index: int
    source: EventSource
    content: str
    state: UserState | None = None
    elapsed_ms: int | None = None
    tool_name: str | None = None
    call_id: str | None = None
    job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "index": self.index,
            "source": self.source.value,
            "content": self.content,
        }
        if self.state is not None:
            value["state"] = self.state.value
        if self.elapsed_ms is not None:
            value["elapsed_ms"] = self.elapsed_ms
        if self.tool_name is not None:
            value["tool_name"] = self.tool_name
        if self.call_id is not None:
            value["call_id"] = self.call_id
        if self.job_id is not None:
            value["job_id"] = self.job_id
        return value


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    raw_output: str = ""
    diagnostic: str | None = None
    target: int | None = None
    message: str | None = None

    @property
    def query(self) -> str | None:
        if self.kind is not ActionKind.TOOL or self.tool_name != "web_search":
            return None
        query = self.arguments.get("query")
        return query if isinstance(query, str) else None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "kind": self.kind.value,
            "valid": self.valid,
            "raw_output": self.raw_output,
        }
        if self.tool_name is not None:
            value["tool_name"] = self.tool_name
            value["arguments"] = self.arguments
        if self.query is not None:
            # Kept as a convenience for existing UI and trace consumers.
            value["query"] = self.query
        if self.diagnostic is not None:
            value["diagnostic"] = self.diagnostic
        if self.target is not None:
            value["target"] = self.target
        if self.message is not None:
            value["message"] = self.message
        return value


@dataclass(frozen=True, slots=True)
class TextHighlight:
    quote: str
    occurrence: int
    event_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "quote": self.quote,
            "occurrence": self.occurrence,
            "event_index": self.event_index,
        }


@dataclass(frozen=True, slots=True)
class TextSuggestion:
    """A proposed correction: an overlay, never a mutation of the user's text."""

    quote: str
    occurrence: int
    replacement: str
    event_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "quote": self.quote,
            "occurrence": self.occurrence,
            "replacement": self.replacement,
            "event_index": self.event_index,
        }


@dataclass(frozen=True, slots=True)
class TranslationCommit:
    """One append-only translated sense unit grounded in a user snapshot."""

    target_event_index: int
    source_snapshot: str
    message: str
    action_event_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_event_index": self.target_event_index,
            "source_snapshot": self.source_snapshot,
            "message": self.message,
            "action_event_index": self.action_event_index,
        }


@dataclass(slots=True)
class Document:
    """The shared artifact the background writer produces and the user negotiates over.

    The runtime owns this state; the policy only ever proposes operations on it.
    Undo restores exact prior text (acceptance criterion), so snapshots are kept whole.
    """

    text: str = ""
    revision: int = 0
    task: str = ""
    preferences: list[str] = field(default_factory=list)
    status: str = WRITER_STATUS_IDLE
    undo_stack: list[str] = field(default_factory=list)

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        if self.text and not self.text.endswith(("\n", " ")) and not chunk.startswith(("\n", " ")):
            self.text += " "
        self.text += chunk
        self.revision += 1

    def apply_patch(self, quote: str, occurrence: int, replacement: str) -> None:
        start = find_occurrence(self.text, quote, occurrence)
        if start < 0:
            raise ValueError("Patch target no longer exists in the document.")
        self.undo_stack.append(self.text)
        self.text = self.text[:start] + replacement + self.text[start + len(quote):]
        self.revision += 1

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self.text = self.undo_stack.pop()
        self.revision += 1
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "revision": self.revision,
            "task": self.task,
            "preferences": list(self.preferences),
            "status": self.status,
            "undo_available": bool(self.undo_stack),
        }


def find_occurrence(text: str, quote: str, occurrence: int) -> int:
    """Return the start index of the nth occurrence of quote, or -1."""
    if not quote or occurrence < 1:
        return -1
    start = -1
    for _ in range(occurrence):
        start = text.find(quote, start + 1)
        if start < 0:
            return -1
    return start


@dataclass(frozen=True, slots=True)
class CompletedTurn:
    event: StreamEvent
    action: Action
    # Wall-clock milliseconds spent inside policy.predict for this turn. None for
    # turns built outside the serving path (datagen, teacher-forced replays).
    decision_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"event": self.event.to_dict(), "action": self.action.to_dict()}
        if self.decision_ms is not None:
            value["decision_ms"] = self.decision_ms
        return value


@dataclass(slots=True)
class Session:
    id: str
    instruction: str = ""
    mode: str = "probe"
    permissions: dict[str, list[str]] = field(
        default_factory=lambda: {name: list(operations) for name, operations in DEFAULT_PERMISSIONS.items()}
    )
    history: list[CompletedTurn] = field(default_factory=list)
    highlights: list[TextHighlight] = field(default_factory=list)
    suggestions: list[TextSuggestion] = field(default_factory=list)
    translation_commits: list[TranslationCommit] = field(default_factory=list)
    document: Document | None = None
    proposal: TextHighlight | None = None
    next_index: int = 1
    started_monotonic: float = 0.0
    generation: int = 0
    latest_prompt: str = ""
    # Render-only delegate build snapshots. g1 tool events tell the model only
    # that the task completed; partial and final UI state stays out of the
    # model-visible history.
    job_specs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self, pending_searches: int = 0) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.id,
            "instruction": self.instruction,
            "mode": self.mode,
            "permissions": self.permissions,
            "history": [turn.to_dict() for turn in self.history],
            "highlights": [highlight.to_dict() for highlight in self.highlights],
            "next_index": self.next_index,
            "pending_searches": pending_searches,
            "latest_prompt": self.latest_prompt,
        }
        if self.document is not None:
            value["document"] = self.document.to_dict()
            value["proposal"] = self.proposal.to_dict() if self.proposal else None
        if self.mode in ("v6", "g1"):
            value["suggestions"] = [suggestion.to_dict() for suggestion in self.suggestions]
            value["translation_commits"] = [
                commit.to_dict() for commit in self.translation_commits
            ]
            value["translation"] = "".join(
                commit.message for commit in self.translation_commits
            )
            jobs = derive_jobs(self.history)
            for job in jobs:
                spec = self.job_specs.get(job["job_id"])
                if spec is not None:
                    job["spec"] = spec
            value["jobs"] = jobs
            value["panels"] = [
                job["spec"]
                for job in jobs
                if job["tool"] in ("generate_ui", "delegate")
                and job["status"] == "completed"
                and "spec" in job
            ]
        return value


def derive_jobs(history: list[CompletedTurn]) -> list[dict[str, Any]]:
    """Reconstruct background-job state purely from the event stream.

    The trace is the source of truth (traces first, interface second): job status
    is never stored separately, only derived from acknowledgement and result events.
    """
    jobs: dict[str, dict[str, Any]] = {}
    for turn in history:
        event = turn.event
        if event.source is not EventSource.TOOL or event.job_id is None:
            continue
        try:
            payload = json.loads(event.content)
        except ValueError:
            payload = {}
        entry = jobs.setdefault(
            event.job_id,
            {"job_id": event.job_id, "tool": event.tool_name, "status": "pending"},
        )
        status = payload.get("status")
        if status == "accepted":
            entry["status"] = "running"
            entry["accepted_index"] = event.index
        elif status in ("completed", "failed"):
            entry["status"] = status
            entry["result_index"] = event.index
            if status == "failed" and payload.get("error"):
                entry["error"] = payload["error"]
            if event.tool_name == "generate_ui" and status == "completed":
                entry["spec"] = payload.get("spec")
    return list(jobs.values())
