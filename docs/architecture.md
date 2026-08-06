# Current architecture

**Status:** implemented V4 probe, V5 editor, and V6 demo surface; g1 not implemented
**Source of truth:** `app/`, `static/`, and `data/tinker/run_state.json`

Smol Interactions is a single-process FastAPI application with a static browser UI. The browser
sends a full textbox snapshot every 650 ms while an app surface is open, foregrounded, running,
and unpaused. This includes initial, cleared, and unchanged empty text: silence is an event, not
a reason to stop the clock. Pausing stops ticks; emptiness does not. The runtime serializes
events per session, calls one policy, validates its decision, and runs permitted tools.

The browser intentionally owns scheduling for the current prototype. It allows only one request
in flight, so slow decisions do not overlap or build a stale queue. A decision that runs longer
than 650 ms causes a missed interval and is a latency failure; skipping that interval is
backpressure, not successful on-time behavior. A server-owned session clock is deferred.

## Runtime flow

```text
Browser textarea + standing instruction
  -> POST /api/sessions/{id}/tick
  -> InteractionRuntime
  -> compile context + stream + policy decision
  -> validate permissions and action schema
  -> append event/action trace
  -> optional async web_search or immediate ui.highlight
  -> tool result re-enters the chronological stream
```

`app/runtime.py` owns sessions, locks, trace persistence, permission enforcement, highlight
reconciliation, reset, and asynchronous search reinjection. `app/stream.py` renders prompts and
strictly parses model output. Invalid output becomes an inspectable invalid idle decision rather
than executing an unsafe or malformed action.

## Context and action grammar

Each prompt begins with server-owned standing context:

```xml
<interaction_context>
<instruction>Highlight every animal I mention.</instruction>
<permissions>{"ui":["highlight"],"web_search":["search"]}</permissions>
</interaction_context>
```

The model must emit exactly one canonical v4 action:

```xml
<action>idle()</action>
<action>tool(web_search,{"query":"actual query"})</action>
<action>tool(ui,{"operation":"highlight","target":{"occurrence":1,"quote":"exact text"}})</action>
<action>respond({"for":17,"message":"grounded answer"})</action>
```

The runtime rejects unknown tools, missing permissions, invalid highlight targets, duplicate
highlights/searches, active-user responses, nonexistent response targets, and repeated responses.
Highlight anchors use an exact quote plus a one-based occurrence, avoiding fragile character
offsets as text changes.

Search results return as generic events:

```xml
<stream_event index="17" source="tool" tool="web_search" call_id="call-12">{...}</stream_event>
```

## Editor sessions (v5)

`/editor` sessions add a runtime-owned `Document` (revision counter, full-snapshot undo, one
live underline proposal) and a background `WriterProvider` (Anthropic streaming or offline
demo). New grammar, all inside the existing envelope:

```xml
<action>tool(writer,{"operation":"write","instruction":"..."})</action>
<action>tool(writer,{"operation":"pause"})</action>
<action>tool(writer,{"operation":"resume"})</action>
<action>tool(writer,{"operation":"revise","instruction":"..."})</action>
<action>tool(ui,{"operation":"underline","target":{"occurrence":1,"quote":"..."}})</action>
```

Writer sentences enter the stream as `source="writer"` events; revision results re-enter as
`tool="writer"` events. `respond` may additionally target a user event to ask a confirmation
question. Underlines are reversible proposals over the document, never mutations; `revise`
applies only to the confirmed proposal and rejects stale document revisions. The
`ScriptedEditorPolicy` (POLICY_MODE=scripted) drives the full loop deterministically for
development; the trained policy serves under `SYSTEM_PROMPT_V5` (POLICY_PROMPT=v5).

## V6 demo sessions

The implemented V6 surface is selected with `POLICY_PROMPT=v6` and browser query
`?mode=v6`. It adds flat intent-named `generate_ui` and `suggest_edit` tools, job identity for
asynchronous work, and direct responses to user events while retaining the V4 action envelope.
`POLICY_MODE=scripted-v6` drives the same runtime surface deterministically for development.

V6 is the latest implemented lineage, but it is not g1. The planned g1 surface changes the
action grammar, removes the old interaction header, renames `generate_ui` to `delegate`, adds
`highlight`, and retires `web_search` from its training slate. Those changes remain
prerequisites in the g1 data specification.

## Model and prompt parity

The hosted production path is `TinkerPolicy`. `POLICY_PROMPT` selects the implemented V4, V5,
or V6 system prompt and the matching sampler namespace in `run_state.json`; V4 is the default.
The base tokenizer is `Qwen/Qwen3-8B`. Each implemented training stage and serving mode share
their system prompt, the Qwen chat template with thinking disabled, the flat compiled stream,
and a 192-token output budget.

Tinker connects lazily so the UI can load while the provider is unavailable. A missing key,
checkpoint, connection, or decision stops streaming and shows **No model available**.

## UI behavior

The demo view contains the writing surface and current assistant action. Details contains the
standing instruction, start/pause/clear/record controls, active checkpoint, readable history,
exact model input, and copy control.

Highlights render through a synchronized, non-interactive backdrop behind the textarea; the
textarea remains the real editing surface. Recording is browser-local. Version-2 recordings
capture text, highlight state, and visible assistant state, then replay without a blinking caret.
