from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.domain import EventSource, UserState
from app.policy import (
    SYSTEM_PROMPT_G1,
    SYSTEM_PROMPT_V5,
    SYSTEM_PROMPT_V6,
    ScriptedEditorPolicy,
    ScriptedV6Policy,
    TinkerPolicy,
)
from app.runtime import InteractionRuntime, JsonlTraceWriter
from app.search import DDGSSearchProvider, DemoSearchProvider
from app.stream import STREAM_FORMATS
from app.uigen import DemoUIGenProvider, OpenAIUIGenProvider
from app.writer import AnthropicWriterProvider, DemoWriterProvider, OpenAIWriterProvider


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
TICK_MS = 650


def _load_dotenv() -> None:
    """Best-effort .env loader (no overwrite) so provider keys reach the app
    even when the launching shell cannot source the file."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()


def build_runtime() -> InteractionRuntime:
    policy_mode = os.getenv("POLICY_MODE", "tinker").lower()
    prompt_version = os.getenv(
        "POLICY_PROMPT",
        "v6" if policy_mode == "local" else "v4",
    ).lower()
    uses_g1_contract = False
    if policy_mode == "scripted":
        # Deterministic T1-loop driver for runtime/UI development without a model.
        policy = ScriptedEditorPolicy()
    elif policy_mode == "scripted-v6":
        # Deterministic v6 demo-loop driver: concurrency + highlight, no model.
        policy = ScriptedV6Policy()
    elif policy_mode == "tinker":
        if prompt_version not in ("v4", "v5", "v6", "g1"):
            raise ValueError("POLICY_PROMPT must be 'v4', 'v5', 'v6', or 'g1'.")
        uses_g1_contract = prompt_version == "g1"
        model_path = os.getenv("TINKER_MODEL_PATH", "")
        if not model_path:
            state_path = ROOT / "data" / "tinker" / "run_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                model_path = (
                    state.get("g1-v2:selected_sampler_path", "")
                    if prompt_version == "g1"
                    else ""
                ) or state.get(f"{prompt_version}:sampler_path", "")
        prompt_overrides = {
            "v5": SYSTEM_PROMPT_V5,
            "v6": SYSTEM_PROMPT_V6,
            "g1": SYSTEM_PROMPT_G1,
        }
        policy = TinkerPolicy(
            model_path=model_path,
            base_model=os.getenv(
                "TINKER_BASE_MODEL",
                "Qwen/Qwen3.5-4B" if uses_g1_contract else "Qwen/Qwen3-8B",
            ),
            system_prompt=prompt_overrides.get(prompt_version),
            action_schema="g1" if uses_g1_contract else "legacy",
        )
    elif policy_mode == "local":
        from scripts.local_model import LocalMLXPolicy

        if prompt_version not in ("v6", "g1"):
            raise ValueError("Local POLICY_PROMPT must be 'v6' or 'g1'.")
        uses_g1_contract = prompt_version == "g1"
        policy = LocalMLXPolicy(
            system_prompt=SYSTEM_PROMPT_G1 if uses_g1_contract else SYSTEM_PROMPT_V6
        )
    else:
        raise ValueError("POLICY_MODE must be 'tinker', 'scripted', 'scripted-v6', or 'local'.")

    writer_mode = os.getenv("WRITER_MODE", "").lower()
    if not writer_mode:
        if os.getenv("ANTHROPIC_API_KEY"):
            writer_mode = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            writer_mode = "openai"
        else:
            writer_mode = "demo"
    if writer_mode == "anthropic":
        writer_provider = AnthropicWriterProvider()
    elif writer_mode == "openai":
        writer_provider = OpenAIWriterProvider()
    elif writer_mode == "demo":
        writer_provider = DemoWriterProvider()
    else:
        raise ValueError("WRITER_MODE must be 'anthropic', 'openai', or 'demo'.")

    search_mode = os.getenv("SEARCH_MODE", "ddgs").lower()
    if search_mode == "ddgs":
        search_provider = DDGSSearchProvider(
            max_results=int(os.getenv("SEARCH_MAX_RESULTS", "5")),
            timeout_seconds=float(os.getenv("SEARCH_TIMEOUT_SECONDS", "15")),
        )
    elif search_mode == "demo":
        search_provider = DemoSearchProvider()
    else:
        raise ValueError("SEARCH_MODE must be 'ddgs' or 'demo'.")

    stream_format = os.getenv(
        "STREAM_FORMAT",
        "g1" if uses_g1_contract else "flat",
    ).lower()
    if stream_format not in STREAM_FORMATS:
        raise ValueError(f"STREAM_FORMAT must be one of {STREAM_FORMATS}.")
    if (stream_format == "g1") != uses_g1_contract:
        raise ValueError(
            "POLICY_PROMPT and STREAM_FORMAT must select the g1 contract together."
        )

    uigen_mode = os.getenv("UIGEN_MODE", "auto").lower()
    if uigen_mode == "auto":
        uigen_mode = "openai" if os.getenv("OPENAI_API_KEY") else "demo"
    if uigen_mode == "demo":
        uigen_provider = DemoUIGenProvider(
            delay_seconds=float(os.getenv("UIGEN_DELAY_SECONDS", "4"))
        )
    elif uigen_mode == "openai":
        uigen_provider = OpenAIUIGenProvider()
    else:
        raise ValueError("UIGEN_MODE must be 'auto', 'openai', or 'demo'.")

    trace_path = Path(os.getenv("TRACE_PATH", str(ROOT / "data" / "sessions.jsonl")))
    return InteractionRuntime(
        policy=policy,
        search_provider=search_provider,
        trace_writer=JsonlTraceWriter(trace_path),
        stream_format=stream_format,
        writer_provider=writer_provider,
        uigen_provider=uigen_provider,
    )


runtime = build_runtime()


@asynccontextmanager
async def lifespan(application: FastAPI):
    del application
    runtime.policy.warm_up()
    yield
    await runtime.shutdown()


app = FastAPI(title="Text GPT-Live", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TickRequest(BaseModel):
    text: str = Field(max_length=50_000)
    state: Literal["active", "idle"]


class SessionCreateRequest(BaseModel):
    instruction: str = Field(default="", max_length=2_000)
    mode: Literal["probe", "editor", "v6", "g1"] = "probe"


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/editor", include_in_schema=False)
async def editor() -> FileResponse:
    return FileResponse(STATIC_DIR / "editor.html")


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "protocol_version": 4,
        "tick_ms": TICK_MS,
        "model_name": runtime.policy.display_name,
        "model_status": runtime.policy.availability,
        "model_message": runtime.policy.availability_message,
        "search_mode": runtime.search_provider.mode,
        "uigen_mode": runtime.uigen_provider.mode if runtime.uigen_provider else "none",
        "stream_format": runtime.stream_format,
        "action_schema": runtime.action_schema,
        "decision_latency": runtime.latency_summary(),
    }


class ConfigRequest(BaseModel):
    search_mode: Literal["ddgs", "demo"] | None = None
    uigen_mode: Literal["openai", "demo"] | None = None


@app.post("/api/config")
async def set_config(request: ConfigRequest) -> dict[str, object]:
    """Live provider swaps: search and UI generation only — the policy is a
    loaded model and needs a restart to change."""
    if request.search_mode == "ddgs":
        runtime.search_provider = DDGSSearchProvider(
            max_results=int(os.getenv("SEARCH_MAX_RESULTS", "5")),
            timeout_seconds=float(os.getenv("SEARCH_TIMEOUT_SECONDS", "15")),
        )
    elif request.search_mode == "demo":
        runtime.search_provider = DemoSearchProvider()
    if request.uigen_mode == "openai":
        runtime.uigen_provider = OpenAIUIGenProvider()
    elif request.uigen_mode == "demo":
        runtime.uigen_provider = DemoUIGenProvider(
            delay_seconds=float(os.getenv("UIGEN_DELAY_SECONDS", "4"))
        )
    return await health()


@app.post("/api/sessions")
async def create_session(request: SessionCreateRequest | None = None) -> dict[str, object]:
    instruction = request.instruction if request else ""
    mode = request.mode if request else "probe"
    if runtime.action_schema == "g1":
        if instruction.strip():
            raise HTTPException(
                status_code=400,
                detail="g1 instructions must be typed into the chronological event stream.",
            )
        if mode not in ("probe", "g1"):
            raise HTTPException(status_code=400, detail="g1 runtime only creates g1 sessions.")
        mode = "g1"
    try:
        session = runtime.create_session(instruction=instruction, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return session.to_dict(pending_searches=0)


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, object]:
    try:
        session = runtime.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return session.to_dict(pending_searches=runtime.pending_count(session_id))


@app.post("/api/sessions/{session_id}/tick")
async def tick(session_id: str, request: TickRequest) -> dict[str, object]:
    try:
        turn = await runtime.process_event(
            session_id,
            source=EventSource.USER,
            content=request.text,
            state=UserState(request.state),
        )
        session = runtime.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "turn": turn.to_dict(),
        "session": session.to_dict(pending_searches=runtime.pending_count(session_id)),
    }


@app.post("/api/sessions/{session_id}/undo")
async def undo_document(session_id: str) -> dict[str, object]:
    try:
        session = await runtime.undo_document(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return session.to_dict(pending_searches=runtime.pending_count(session_id))


@app.post("/api/sessions/{session_id}/reset")
async def reset_session(session_id: str) -> dict[str, object]:
    try:
        session = await runtime.reset_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return session.to_dict(pending_searches=0)
