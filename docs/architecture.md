# Current architecture

**Status:** g1 runtime, local MLX serving, asynchronous delegation/search, and the static browser UI are implemented.

Text GPT-Live is a single-process FastAPI application with a static browser UI.
While the foreground demo is running, the browser sends the full current
textbox every 650 ms. It sends initial, cleared, and unchanged empty snapshots
too: silence is an event, not a reason to stop the clock.

The browser owns scheduling in the current prototype and permits only one tick
request in flight. A slow decision therefore skips an interval instead of
building a stale queue. This is backpressure, not successful 650 ms inference.

## Runtime flow

```text
Browser textarea
  -> POST /api/sessions/{id}/tick
  -> InteractionRuntime serializes the event
  -> compile_stream renders the complete chronological history
  -> policy emits exactly one g1 action
  -> runtime parses and validates the action
  -> immediate UI/response effect or asynchronous job
  -> accepted/completed tool events re-enter the same stream
```

`app/runtime.py` owns sessions, per-session locks, traces, action validation,
reset, and asynchronous-result reinjection. `app/stream.py` is shared by data
generation, evaluation, and serving; it renders prompts and parses the closed
action grammar. Invalid output is recorded as an invalid idle decision and is
never executed.

## Stream contract

Each user event contains the entire textbox, not a token delta:

```xml
<stream_event index="12" source="user" state="active" time="t+7800ms">full current textbox</stream_event>
```

Tool events use the same chronological history and carry a stable `job_id`:

```xml
<stream_event index="13" source="tool" tool="web_search" job_id="job-12" time="t+8450ms">{"status":"accepted"}</stream_event>
```

Every prediction is one canonical action line:

```xml
<action>idle()</action>
<action>respond({"for":12,"message":"..."})</action>
<action>suggest_edit({"quote":"...","replacement":"..."})</action>
<action>highlight({"occurrence":1,"quote":"..."})</action>
<action>delegate({"task":"..."})</action>
<action>web_search({"query":"..."})</action>
<action>translate_commit({"for":12,"message":"..."})</action>
```

The runtime checks syntax, grounded targets, duplicate actions, valid event
references, and asynchronous job identity. It does not decide whether the user
has finished, whether silence is appropriate, or which valid action to choose.
Those decisions remain in the model.

## Policies

The released path is `LocalMLXPolicy`, which loads the merged 8-bit
`huyxdang/text-gpt-live` checkpoint. It uses the same g1 system prompt and Qwen
chat template used during training, disables thinking, decodes greedily, and
stops after the first complete action envelope.

Local serving maintains a persistent cache per session. The stable prefix is
extended only through the latest durable stream event; the prediction marker
and generated action are temporary. Cache access is serialized because MLX
generation mutates cache containers, and the process keeps a bounded LRU of
session caches.

`TinkerPolicy` is retained for training-time evaluation and serving a sampler
path owned by the operator. `scripted-v6` is a deterministic UI-development
driver, not a model substitute.

## Asynchronous work

`delegate` and `web_search` return an accepted event immediately. Their jobs
run in the background and later inject a completed or failed event with the
same `job_id`. Multiple jobs can remain outstanding while ordinary user ticks
continue. The model sees those lifecycle events and decides when and how to
weave results back into the conversation.

The default search provider uses DDGS. UI generation uses a configured OpenAI
provider when credentials are available and a deterministic demo provider
otherwise.

## UI behavior

The main view presents the human and model as two sides of one live
interaction. Responses stream visually; tool actions appear immediately;
delegated work and search results move into a separate work area so background
activity does not read as conversation. A details view exposes the exact event
history and model input for inspection.

Reset clears runtime session state and the corresponding model cache. Recording
and replay are browser-local.

## Known boundary

This is browser-driven continuous interaction, not a server-owned always-on
session. Background tabs may throttle timers, closing the page ends the loop,
and the released checkpoint currently runs slower than the 650 ms target on the
measured local path.
