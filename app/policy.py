from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Protocol

import httpx

from app.domain import ActionKind, CompletedTurn, EventSource, StreamEvent, UserState


# Short compatibility prompt kept for archived experiments. Current serving and
# evaluation paths use SYSTEM_PROMPT_V4 below.
SYSTEM_PROMPT = (
    "You watch a live interaction stream. Output exactly one machine-readable "
    "<action>...</action> decision and nothing else."
)

SYSTEM_PROMPT_V4 = """You are a machine-parsed interaction policy watching a chronological stream.
Your entire output must be exactly one of these forms on one line and nothing else:

<action>idle()</action>
<action>tool(web_search,{"query":"actual query"})</action>
<action>tool(ui,{"operation":"highlight","target":{"occurrence":1,"quote":"exact text"}})</action>
<action>respond({"for":17,"message":"answer grounded in tool result 17"})</action>

The interaction context gives the standing instruction and permissions. Never call a tool or
operation that is not permitted. Apply these rules to the CURRENT final event:

- ui.highlight: call only for a current user event and only when an exact span in the current text
  clearly matches the standing instruction. `quote` must be copied exactly from the current text;
  `occurrence` is 1-based. Do not highlight partial words, uncertain matches, instruction examples,
  or a target already highlighted by an earlier equivalent UI action.
- web_search: call only when the current user text contains a complete, clear factual question that
  requires external information. Use a concise query grounded in the user's words. Never repeat an
  equivalent search already in history, whether pending or returned.
- tool result events: never call another tool merely because a result arrived. Stay idle until the
  user's next event gives an appropriate action or opening.
- respond: use only when the current event is a user event with state="idle" and history contains a
  relevant web_search result not referenced by an earlier respond action. `for` is that result event
  index. The message must use only facts in that result. Never respond while the user is active.
- idle: use for partial text, non-matches, ordinary statements, pending work, results arriving while
  the user is active, duplicate actions, and every situation where no permitted action is justified.

When search and highlight first become valid on the same event, search first; highlight may run on
the next event. Prefer restraint: an unnecessary intervention is worse than waiting one tick."""


# Historical training-only prompt retained until the mixed V4-V6 stages are
# extracted from train/tinker_run.py. The app no longer serves a V5 editor.
SYSTEM_PROMPT_V5 = """You are a machine-parsed interaction policy watching a chronological stream.
Your entire output must be exactly one of these forms on one line and nothing else:

<action>idle()</action>
<action>tool(web_search,{"query":"actual query"})</action>
<action>tool(ui,{"operation":"highlight","target":{"occurrence":1,"quote":"exact text"}})</action>
<action>tool(ui,{"operation":"underline","target":{"occurrence":1,"quote":"exact document text"}})</action>
<action>tool(writer,{"operation":"write","instruction":"the writing task"})</action>
<action>tool(writer,{"operation":"pause"})</action>
<action>tool(writer,{"operation":"resume"})</action>
<action>tool(writer,{"operation":"revise","instruction":"the revision instruction"})</action>
<action>respond({"for":17,"message":"grounded answer or confirmation question"})</action>

The interaction context gives the standing instruction and permissions. Never call a tool or
operation that is not permitted. Events with source="writer" are sentences the background
writer has committed to the shared document, in order. Apply these rules to the CURRENT final
event:

- writer.write: call when the current user event asks for a document to be written and no
  writing task exists yet. The instruction restates the user's request.
- writer.pause: call immediately when the current user event tells the writing to stop
  ("stop", "wait", "hold on") while writer events are still arriving. The pause is a reflex:
  never delay it to ask a question first. Do not pause for the word "stop" inside story
  content or inside an ordinary statement.
- ui.underline: after a pause, when the user objects to a part of the document ("I don't like
  the second sentence", "that part about the storm"), underline the exact sentence they mean.
  `quote` must be copied exactly from one writer event. Ordinals count writer events: the
  second sentence is the second writer event. "The last sentence" is the most recent one. If
  the user rejects a proposal with a relative reference ("the sentence before that"), underline
  the writer event immediately before the currently proposed one.
- respond for confirmation: after proposing an underline, confirm your interpretation on the
  next tick: respond targeting the user's objection event with a short question restating the
  choice ("The second sentence — this one?"). Also use respond for grounded answers to
  web_search results, exactly as before.
- writer.revise: when the user confirms the proposal and gives a change ("yes — make it less
  dramatic"), call revise with only the instruction. The runtime applies it to the underlined
  span.
- writer.resume: when the current event is a writer tool result reporting a successful
  revision, resume the original writing task.
- ui.highlight and web_search keep their previous rules on user text.
- idle: for everything else — writer sentences that need no action, partial text, ordinary
  talk while writing continues, pending work, and every situation where no permitted action is
  justified.

Prefer restraint: an unnecessary intervention is worse than waiting one tick."""


SYSTEM_PROMPT_G1 = """You are the interaction policy inside a live typing interface. While it is
running, the app normally sends a <stream_event> every 650 ms. A user event contains
the entire current textbox, not a token delta: it may grow, be edited, be cleared, or
stay unchanged. index identifies the event; state="active" means recent typing and
state="idle" means a pause; time is elapsed session time and remains authoritative
when gaps vary. Tool events report delegated work. Past <action> lines show decisions
already made.

For each current event, use the full history to infer what changed, which obligations
remain active, and what was already handled. Do not wait for Send: act when a trigger
is complete while the user may still be typing, and wait when evidence is incomplete.
Your entire output must be exactly one of these forms on one line and nothing else:

<action>idle()</action>
<action>respond({"for":12,"message":"what you say, aimed at event 12"})</action>
<action>suggest_edit({"quote":"exact text span","replacement":"corrected span"})</action>
<action>highlight({"occurrence":1,"quote":"exact word"})</action>
<action>delegate({"task":"described work for the background model"})</action>
<action>web_search({"query":"concise search query"})</action>
<action>translate_commit({"for":12,"message":"new Chinese sense unit"})</action>

Treat completed user text addressed directly to you as direct address; questions or
commands inside narration, quotations, or reported speech are not. Answer direct
address once. A new, changed, or cancelled standing instruction gets one
acknowledgment. An active instruction remains in force until changed, cancelled, or
completed.

Use respond for direct replies, acknowledgments, honest progress answers, completed
web-search results, and reminders due by the recorded timestamps. Use translate_commit
for each newly stable translation sense unit, targeting the user event whose full
textbox snapshot ends that unit; commas are opportunities, not automatic boundaries.
Commit natural units a human interpreter can translate without guessing future words.
Use
suggest_edit only for a newly completed typo or grammar error under an active
correction instruction; quote one exact, unique current-text span and never repeat an
edit. Use highlight only for a newly appearing literal match under an active
highlighting instruction; quote the exact word, use its 1-based occurrence, and never
repeat a mark. Use delegate once for a newly completed UI/task request. Use web_search
once for a completed request that needs current external information. Both start an
asynchronous job whose accepted/completed/failed events share a job_id; never restart
an existing job. When a web_search completes or fails, respond once targeting that
tool event and faithfully summarize the result; never invent unavailable facts.

An empty or unchanged textbox is still a real event. It normally requires idle, but
history can make an action due during silence. Also idle on unfinished text, ordinary
narration or thinking, quoted speech, clean text, non-matches, and anything already
handled. If actions collide, handle the user's completed direct request or address
first, then an overdue schedule, then a newly completed standing-instruction target;
deferred obligations remain active. Prefer silence whenever the evidence for acting is
incomplete."""


SYSTEM_PROMPT_V6 = """You are a machine-parsed interaction policy watching a chronological stream.
Your entire output must be exactly one of these forms on one line and nothing else:

<action>idle()</action>
<action>tool(web_search,{"query":"actual query"})</action>
<action>tool(generate_ui,{"request":"what the user asked to be built"})</action>
<action>tool(suggest_edit,{"target":{"occurrence":1,"quote":"exact text span"},"replacement":"corrected span"})</action>
<action>respond({"for":17,"message":"answer grounded in tool result 17"})</action>

The interaction context lists the permitted tools. web_search and generate_ui are
asynchronous jobs: the action starts a job, a tool event with status "accepted" confirms
it, and a later tool event with the same job_id carries the result or an error. Several
jobs can be outstanding at once; starting one never blocks the stream. Apply these rules
to the CURRENT final event:

- generate_ui: call once when the user asks for something visual to be built. The request
  restates what they asked for. Never repeat a request that is already accepted or running.
- web_search: call once when the user asks something that needs looked-up information.
  Never repeat an equivalent pending or completed query.
- suggest_edit: call when a rule the user stated makes a span of their current text need
  fixing or swapping. The quote is copied exactly from their text, multi-word and unique;
  the replacement is that span with the change applied. Never touch text inside the rule
  itself, never suggest the same span twice, and move on to the next target after acting.
- respond: only when the user is idle and the relevant web_search job has completed;
  "for" is the index of that result event and the message must be grounded in it.
- idle: for everything else — partial text, job acknowledgements, results that need no
  reply, and every situation where no permitted action is justified.

Prefer restraint: an unnecessary intervention is worse than waiting one tick."""


# The spec-mandated "best honest prompt" for UN-FINE-TUNED configurations
# (configuration 2 in the comparison). Worked examples demonstrate the exact
# action format — the tool( wrapper, occurrence discipline, idle on partial
# text, respond only from completed results. The fine-tuned model never uses
# this variant; its formats come from training.
SYSTEM_PROMPT_V6_FEWSHOT = SYSTEM_PROMPT_V6 + """

Worked examples — the format is exact, including the tool( wrapper:

<stream_event index="1" source="user" state="active" time="t+650ms">Fix my typos please. The repor</stream_event>
<action>idle()</action>
<stream_event index="2" source="user" state="active" time="t+1300ms">Fix my typos please. The reprot was late.</stream_event>
<action>tool(suggest_edit,{"replacement":"The report was","target":{"occurrence":1,"quote":"The reprot was"}})</action>
<stream_event index="3" source="user" state="active" time="t+1950ms">Fix my typos please. The reprot was late. Can you build a small graph of my sleep log?</stream_event>
<action>tool(generate_ui,{"request":"build a small graph of my sleep log"})</action>
<stream_event index="4" source="tool" tool="generate_ui" job_id="job-3" time="t+2600ms">{"job_id":"job-3","status":"accepted"}</stream_event>
<action>idle()</action>
<stream_event index="5" source="user" state="active" time="t+3250ms">Fix my typos please. The reprot was late. Can you build a small graph of my sleep log? Also is the gym open right now?</stream_event>
<action>tool(web_search,{"query":"is the gym open right now"})</action>
<stream_event index="6" source="tool" tool="web_search" job_id="job-5" time="t+3900ms">{"job_id":"job-5","status":"accepted"}</stream_event>
<action>idle()</action>
<stream_event index="7" source="tool" tool="web_search" job_id="job-5" time="t+4550ms">{"job_id":"job-5","status":"completed","query":"is the gym open right now","results":[{"title":"Gym open until ten","url":"https://example.com/g","snippet":"Evening hours run to ten tonight."}]}</stream_event>
<action>idle()</action>
<stream_event index="8" source="user" state="idle" time="t+5200ms">Fix my typos please. The reprot was late. Can you build a small graph of my sleep log? Also is the gym open right now?</stream_event>
<action>respond({"for":7,"message":"From the search: Gym open until ten — evening hours run to ten tonight."})</action>

Note tick 1: the sentence was incomplete, so the only correct action was idle()."""


class Policy(Protocol):
    mode: str

    async def predict(
        self,
        compiled_stream: str,
        current: StreamEvent,
        history: list[CompletedTurn],
    ) -> str: ...


class PromptedPolicy:
    mode = "prompted"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        enable_thinking: bool = False,
        finetuned: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("POLICY_MODEL is required in prompted mode.")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.enable_thinking = enable_thinking
        # A fine-tuned checkpoint uses its training-aligned prompt with the plain chat
        # template; the anti-thinking shims below would break train/serve prompt parity.
        self.finetuned = finetuned
        self._client = client or httpx.AsyncClient(timeout=self.timeout_seconds)
        self._owns_client = client is None

    async def predict(
        self,
        compiled_stream: str,
        current: StreamEvent,
        history: list[CompletedTurn],
    ) -> str:
        del current, history
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        system_prompt = SYSTEM_PROMPT_V4
        payload: dict[str, object] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 192,
        }
        if not self.enable_thinking and not self.finetuned:
            # Qwen3 and similar hybrid-reasoning models otherwise spend the entire
            # token budget in <think> and never emit the action tag. Disable it via
            # the chat-template control and the inline token, so `content` is the tag.
            # Skipped for fine-tuned checkpoints: they were trained on the bare prompt
            # and learned to emit the action directly.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            system_prompt = f"{SYSTEM_PROMPT_V4} /no_think"
        payload["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": compiled_stream},
        ]
        response = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        if not content.strip():
            # Some servers place a no-think model's answer in reasoning_content.
            content = message.get("reasoning_content") or content
        return str(content)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class TinkerPolicy:
    """Serves a fine-tuned Tinker checkpoint (e.g. smol-v4-final) via the Tinker sampling API.

    Reproduces the training-time rendering exactly: SYSTEM_PROMPT_V4, the base model's chat
    template with thinking disabled, temperature 0, stop at newline.
    """

    mode = "tinker"

    def __init__(
        self,
        *,
        model_path: str,
        base_model: str = "Qwen/Qwen3-8B",
        max_tokens: int = 192,
        system_prompt: str | None = None,
        action_schema: str = "legacy",
    ) -> None:

        self.system_prompt = system_prompt or SYSTEM_PROMPT_V4
        self.action_schema = action_schema
        self.model = model_path
        self.base_model = base_model
        checkpoint = model_path.rsplit("/", maxsplit=1)[-1] if model_path else "unconfigured"
        self.display_name = f"{base_model} · {checkpoint}"
        self._max_tokens = max_tokens
        self._tinker = None
        self._tokenizer = None
        self._client = None
        self._params = None
        self._initialization_task: asyncio.Task[None] | None = None
        self._initialization_error: str | None = None

    @property
    def availability(self) -> str:
        if self._client is not None:
            return "ready"
        if not self.model or not os.environ.get("TINKER_API_KEY") or self._initialization_error:
            return "unavailable"
        if self._initialization_task is not None:
            return "connecting"
        return "configured"

    @property
    def availability_message(self) -> str:
        if self._client is not None:
            return ""
        if not self.model:
            return "No Tinker sampler checkpoint is configured."
        if not os.environ.get("TINKER_API_KEY"):
            return "Tinker API key is missing."
        if self._initialization_error:
            return self._initialization_error
        if self._initialization_task is not None:
            return "Tinker is connecting."
        return ""

    def warm_up(self) -> None:
        """Start the one-time Tinker connection before the first typing event arrives."""
        if (
            self._client is None
            and self._initialization_task is None
            and self.model
            and os.environ.get("TINKER_API_KEY")
        ):
            self._initialization_task = asyncio.create_task(
                asyncio.to_thread(self._initialize_sync)
            )

    def _initialize_sync(self) -> None:
        try:
            import tinker
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            client = tinker.ServiceClient().create_sampling_client(model_path=self.model)
            params = tinker.types.SamplingParams(
                max_tokens=self._max_tokens, temperature=0.0, stop=["\n"]
            )
        except ImportError as exc:
            self._initialization_error = "Tinker dependencies are not installed."
            raise RuntimeError(self._initialization_error) from exc
        except Exception as exc:
            self._initialization_error = "Tinker could not be reached."
            raise RuntimeError(self._initialization_error) from exc

        self._tinker = tinker
        self._tokenizer = tokenizer
        self._client = client
        self._params = params

    async def _ensure_ready(self) -> None:
        if self._client is not None:
            return
        if not self.model or not os.environ.get("TINKER_API_KEY"):
            raise RuntimeError(f"No model available: {self.availability_message}")
        if self._initialization_error:
            raise RuntimeError(f"No model available: {self._initialization_error}")
        self.warm_up()
        assert self._initialization_task is not None
        try:
            await asyncio.wait_for(asyncio.shield(self._initialization_task), timeout=30)
        except TimeoutError as exc:
            raise RuntimeError("No model available: Tinker is still connecting.") from exc
        except Exception as exc:
            raise RuntimeError(f"No model available: {self.availability_message}") from exc

    async def predict(
        self,
        compiled_stream: str,
        current: StreamEvent,
        history: list[CompletedTurn],
    ) -> str:
        del current, history
        await self._ensure_ready()
        assert self._tokenizer is not None
        assert self._client is not None
        assert self._tinker is not None
        assert self._params is not None
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": compiled_stream},
        ]
        ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        future = self._client.sample(
            prompt=self._tinker.ModelInput.from_ints(list(ids)),
            num_samples=1,
            sampling_params=self._params,
        )
        try:
            response = await asyncio.wait_for(asyncio.wrap_future(future), timeout=30)
        except TimeoutError as exc:
            raise RuntimeError("No model available: Tinker did not return a decision.") from exc
        tokens = response.sequences[0].tokens
        return self._decode_tokens(tokens)

    def _decode_tokens(self, tokens: list[int]) -> str:
        assert self._tokenizer is not None
        decoded = self._tokenizer.decode(tokens, skip_special_tokens=True)
        return decoded if self.action_schema == "g1" else decoded.strip()


class ScriptedV6Policy:
    """Deterministic demo-loop driver for the v6 surface — not a model substitute.

    Implements just enough rules to exercise the concurrency choreography
    (generate_ui + web_search outstanding together, grounded respond) and a
    literal swap rule, so the harness can be verified end-to-end at the
    trace level before any checkpoint exists. State is reconstructed purely
    from the event stream, exactly as a trained policy must.
    """

    mode = "scripted"
    display_name = "scripted v6 rules (dev)"
    availability = "ready"
    availability_message = ""

    def warm_up(self) -> None:
        return None

    @staticmethod
    def _has_pending_job(history: list[CompletedTurn]) -> bool:
        status: dict[str, str] = {}
        for turn in history:
            event = turn.event
            if event.source is EventSource.TOOL and event.job_id:
                try:
                    payload = json.loads(event.content)
                except ValueError:
                    continue
                status[event.job_id] = str(payload.get("status"))
        return any(value == "accepted" for value in status.values())

    _UI_REQUEST = re.compile(
        r"(?:\b(?:create|make|generate|build|draw|sketch|assemble|turn|show|give)\b"
        r".{0,80}?\b(?:visual|brief|briefing|dashboard|chart|ui|panel|card|digest"
        r"|snapshot|summary|rundown|graphic)\b|\bvisualize\b)",
        re.IGNORECASE | re.DOTALL,
    )
    _QUESTION_HINT = re.compile(
        r"\b(what|which|who|when|did|does|is|are|latest|released|tell me)\b", re.IGNORECASE
    )
    _QUESTION_START = re.compile(r"(?i)^(did|does|is|are|who|what|when|can|will|has|have)\b")
    _SWAP_RULE = re.compile(
        r'(?:replace|change|switch)\s+(?:every|any|all)\s+["“]([^"”]+)["”]\s+(?:with|to)\s+["“]([^"”]+)["”]',
        re.IGNORECASE,
    )
    _SENTENCES = re.compile(r"[^.?!]+[.?!]")

    async def predict(
        self,
        compiled_stream: str,
        current: StreamEvent,
        history: list[CompletedTurn],
    ) -> str:
        del compiled_stream

        def action_line(body: str) -> str:
            return f"<action>{body}</action>"

        def dumps(payload: dict) -> str:
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

        if current.source is not EventSource.USER:
            return action_line("idle()")

        taken = [turn.action for turn in history if turn.action.valid]
        ui_requests = {
            str(action.arguments.get("request", ""))
            for action in taken
            if action.tool_name == "generate_ui"
        }
        search_queries = {
            _normalize(action.arguments.get("query", ""))
            for action in taken
            if action.tool_name == "web_search"
        }
        sentences = [chunk.strip() for chunk in self._SENTENCES.findall(current.content)]
        # An un-punctuated tail only counts once the user has gone idle on it,
        # and only while it is fresh (not text we already chose to ignore).
        trailing = self._SENTENCES.sub("", current.content).strip()
        same_content_before = sum(
            1
            for turn in history
            if turn.event.source is EventSource.USER and turn.event.content == current.content
        )
        trailing_fresh = (
            bool(trailing) and current.state is UserState.IDLE and same_content_before <= 2
        )

        # 1. A completed request for something visual starts a UI job.
        ui_candidates = sentences + ([trailing] if trailing_fresh else [])
        for sentence in ui_candidates:
            if self._UI_REQUEST.search(sentence):
                request = sentence.rstrip("?.! ").strip()
                already_requested = request in ui_requests or any(
                    prev and prev in request for prev in ui_requests
                )
                if request and not already_requested:
                    return action_line(f"tool(generate_ui,{dumps({'request': request})})")

        # 2. Any other completed question starts a search. Un-punctuated tail
        # questions additionally require that no job is pending — chatter about
        # an in-flight job ("did that go through?") is not a new request.
        pending_job = self._has_pending_job(history)
        for sentence in sentences:
            if (
                sentence.endswith("?")
                and not self._UI_REQUEST.search(sentence)
                and self._QUESTION_HINT.search(sentence)
            ):
                query = sentence.rstrip("? ").strip()
                if query and _normalize(query) not in search_queries:
                    return action_line(f"tool(web_search,{dumps({'query': query})})")
        if (
            trailing_fresh
            and not pending_job
            and not self._UI_REQUEST.search(trailing)
            and self._QUESTION_START.match(trailing)
            and self._QUESTION_HINT.search(trailing)
        ):
            query = trailing.rstrip("?.! ").strip()
            if query and _normalize(query) not in search_queries:
                return action_line(f"tool(web_search,{dumps({'query': query})})")

        # 3. When the user is idle, answer the newest unanswered completed search.
        if current.state is UserState.IDLE:
            answered = {
                turn.action.target
                for turn in history
                if turn.action.valid and turn.action.kind is ActionKind.RESPOND
            }
            for turn in reversed(history):
                event = turn.event
                if (
                    event.source is not EventSource.TOOL
                    or event.tool_name != "web_search"
                    or event.index in answered
                ):
                    continue
                try:
                    payload = json.loads(event.content)
                except ValueError:
                    continue
                if payload.get("status") != "completed":
                    continue
                results = payload.get("results") or []
                if results:
                    first = results[0]
                    message = f"From the search: {first.get('title', '')} — {first.get('snippet', '')}".strip(" —")
                else:
                    message = "The search returned no results."
                return action_line(f'respond({dumps({"for": event.index, "message": message})})')

        # 4. A stated literal swap rule applies to text after the instruction.
        rule = self._SWAP_RULE.search(current.content)
        if rule:
            word, replacement = rule.group(1), rule.group(2)
            suggested = {
                (turn.action.arguments.get("target") or {}).get("occurrence")
                for turn in history
                if turn.action.valid
                and turn.action.tool_name == "suggest_edit"
                and (turn.action.arguments.get("target") or {}).get("quote") == word
            }
            occurrence, start = 0, -1
            while True:
                start = current.content.find(word, start + 1)
                if start < 0:
                    break
                occurrence += 1
                if start >= rule.end() and occurrence not in suggested:
                    payload = {
                        "target": {"occurrence": occurrence, "quote": word},
                        "replacement": replacement,
                    }
                    return action_line(f"tool(suggest_edit,{dumps(payload)})")

        return action_line("idle()")


def _normalize(query: object) -> str:
    return " ".join(str(query).casefold().split()).rstrip("?.!")


class DemoPolicy:
    """Visible no-key fallback for exercising the product loop, not a model substitute."""

    mode = "demo"
    _QUESTION = re.compile(
        r"(?:\?|\bwhat\b|\bwho\b|\bwhen\b|\bwhere\b|\bwhich\b|\bhow many\b|"
        r"\bi wonder\b|\blook up\b|\bsearch for\b)",
        re.IGNORECASE,
    )

    async def predict(
        self,
        compiled_stream: str,
        current: StreamEvent,
        history: list[CompletedTurn],
    ) -> str:
        del compiled_stream
        if current.source is not EventSource.USER or not self._QUESTION.search(current.content):
            return "<action>idle()</action>"

        normalized_text = " ".join(current.content.split())
        pending_searches = 0
        for turn in history:
            if turn.action.query:
                pending_searches += 1
            if (
                turn.event.source is EventSource.TOOL
                and turn.event.tool_name == "web_search"
                and pending_searches
            ):
                pending_searches -= 1
        if pending_searches:
            return "<action>idle()</action>"

        previous_queries = [
            " ".join((turn.action.query or "").casefold().split())
            for turn in history
            if turn.action.query
        ]
        query = normalized_text[-240:].strip(" ,.;:")
        normalized_query = " ".join(query.casefold().split())
        if not query or any(previous in normalized_query for previous in previous_queries):
            return "<action>idle()</action>"
        return f'<action>tool(web_search,{{"query":{json.dumps(query)}}})</action>'
