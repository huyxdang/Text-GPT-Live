const TICK_MS = 650;
const ACTIVE_WINDOW_MS = 1300;
const ACTION_STAGE_MS = 2800;
const RESPONSE_STAGE_HOLD_MS = 3500;
const ACTION_STAGE_EXIT_MS = 220;
const TRANSLATION_APPEND_MS_PER_CHARACTER = 42;
const SEARCH_RESULT_HOLD_MS = 3500;
const WORK_THREAD_EXIT_MS = 320;
const RECORDING_STORAGE_KEY = "smol-interactions:last-recording";
const SESSION_MODE = new URLSearchParams(window.location.search).get("mode") || "probe";

const elements = {
  text: document.querySelector("#live-text"),
  highlightLayer: document.querySelector("#highlight-layer"),
  instruction: document.querySelector("#standing-instruction"),
  run: document.querySelector("#run-button"),
  reset: document.querySelector("#reset-button"),
  record: document.querySelector("#record-button"),
  replay: document.querySelector("#replay-button"),
  exportRecording: document.querySelector("#export-button"),
  importRecording: document.querySelector("#import-button"),
  importInput: document.querySelector("#import-input"),
  recordingStatus: document.querySelector("#recording-status"),
  errorLine: document.querySelector("#error-line"),
  decidePulse: document.querySelector("#decide-pulse"),
  trace: document.querySelector("#trace-list"),
  pulse: document.querySelector("#clock-pulse"),
  runtimeLight: document.querySelector("#runtime-light"),
  runtimeLabel: document.querySelector("#runtime-label"),
  modeLabel: document.querySelector("#mode-label"),
  modelName: document.querySelector("#model-name"),
  currentPanel: document.querySelector("#current-panel"),
  detailsPanel: document.querySelector("#details-panel"),
  currentAction: document.querySelector("#current-action"),
  currentActionTitle: document.querySelector("#current-action-title"),
  currentActionValue: document.querySelector("#current-action-value"),
  currentActionRaw: document.querySelector("#current-action-raw"),
  currentActionIndex: document.querySelector("#current-action-index"),
  eventCount: document.querySelector("#event-count"),
  idleCount: document.querySelector("#idle-count"),
  searchCount: document.querySelector("#search-count"),
  pendingCount: document.querySelector("#pending-count"),
  readableHistory: document.querySelector("#readable-history-button"),
  modelInputButton: document.querySelector("#model-input-button"),
  streamViewButton: document.querySelector("#stream-view-button"),
  modelActionsViewButton: document.querySelector("#model-actions-view-button"),
  modelInput: document.querySelector("#model-input"),
  copyHistory: document.querySelector("#copy-history-button"),
  actionLog: document.querySelector("#action-log"),
  actionStage: document.querySelector("#action-stage"),
  actionStageAnnouncement: document.querySelector("#action-stage-announcement"),
  actionStageLabel: document.querySelector("#action-stage-label"),
  actionStageValue: document.querySelector("#action-stage-value"),
  translationOutput: document.querySelector("#translation-output"),
  translationText: document.querySelector("#translation-text"),
  paneStatus: document.querySelector("#pane-status"),
  copyHistoryStatus: document.querySelector("#copy-history-status"),
  liveStage: document.querySelector("#live-stage"),
  externalWorkbench: document.querySelector("#external-workbench"),
  workbenchCount: document.querySelector("#workbench-count"),
  workSurface: document.querySelector("#work-surface"),
};

let sessionId = null;
let timer = null;
let running = false;
let loadingModel = false;
let modelStatus = "connecting";
let actionSchema = "legacy";
let requestInFlight = false;
let lastInputAt = 0;
let lastRenderedIndex = 0;
let pendingPollTimer = null;
let renderedHistory = [];
let latestModelInput = "";
let latestPendingSearches = 0;
let historyFormat = new URLSearchParams(window.location.search).get("history") === "model"
  ? "model"
  : "readable";
let copyFeedbackTimer = null;
let activeRecording = null;
let savedRecording = null;
let recordingStartedAt = 0;
let replayAnimationFrame = null;
let replayRunning = false;
let sessionInstruction = "";
let renderedHighlights = [];
let latestSessionHighlights = [];
let latestSessionSuggestions = [];
let actionStageTimer = null;
let actionStageStreamFrame = null;
let lastStagedActionKey = "";
let actionPanelView = "stream";
let translationAnimationFrame = null;
let translationAnimationTarget = "";
let translationHydrated = false;

function setView(view) {
  const showCurrent = view === "current";
  elements.currentPanel.hidden = !showCurrent;
  elements.detailsPanel.hidden = showCurrent;
  document.body.dataset.view = view;
}

function handleOperatorShortcut(event) {
  if (event.key === "Escape") {
    event.preventDefault();
    setView(document.body.dataset.view === "current" ? "details" : "current");
    return;
  }
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    void start().catch(showError);
  }
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status}: ${body}`);
  }
  return response.json();
}

async function createSessionForInstruction({ preserveText = true } = {}) {
  const text = preserveText ? elements.text.value : "";
  const instruction = actionSchema === "g1" ? "" : elements.instruction.value.trim();
  const mode = actionSchema === "g1" ? "g1" : SESSION_MODE;
  const session = await request("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ instruction, mode }),
  });
  sessionId = session.id;
  sessionInstruction = session.instruction || "";
  elements.instruction.value = sessionInstruction;
  elements.text.value = text;
  lastRenderedIndex = 0;
  renderSession(session);
  return session;
}

async function ensureSessionInstruction({ preserveText = true } = {}) {
  if (elements.instruction.value.trim() === sessionInstruction) return false;
  await createSessionForInstruction({ preserveText });
  return true;
}

function setRuntimeState(label, kind = "quiet") {
  elements.runtimeLabel.textContent = label;
  elements.runtimeLight.classList.toggle("is-live", kind === "live");
  elements.runtimeLight.classList.toggle("is-error", kind === "error");
}

function updateInstructionAvailability() {
  elements.instruction.disabled = actionSchema === "g1"
    || running
    || loadingModel
    || activeRecording !== null
    || document.body.dataset.playback === "true";
}

function setRunButton(label, { loading = false } = {}) {
  elements.run.textContent = label;
  elements.run.classList.toggle("is-loading", loading);
  elements.run.disabled = loading;
  elements.run.setAttribute("aria-busy", String(loading));
  updateInstructionAvailability();
}

function setRecordButton(recording) {
  elements.record.textContent = recording ? "Stop recording" : "Record";
  elements.record.classList.toggle("is-recording", recording);
  elements.record.setAttribute("aria-pressed", String(recording));
  elements.recordingStatus.textContent = recording ? "Recording started" : "Recording stopped";
  updateInstructionAvailability();
}

function updateReplayButton() {
  elements.replay.textContent = replayRunning ? "Stop replay" : "Replay";
  elements.replay.disabled = activeRecording !== null || savedRecording === null;
  elements.exportRecording.disabled = savedRecording === null;
  elements.importRecording.disabled = activeRecording !== null || replayRunning;
  updateInstructionAvailability();
}

function isValidRecording(value) {
  return [1, 2].includes(value?.version) && Array.isArray(value.frames) && value.frames.length > 0;
}

function exportRecording() {
  if (savedRecording === null) return;
  const stamp = String(savedRecording.recorded_at || "recording").replaceAll(/[:.]/g, "-");
  const blob = new Blob([JSON.stringify(savedRecording, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `smol-recording-${stamp}.json`;
  link.click();
  URL.revokeObjectURL(url);
  elements.recordingStatus.textContent = "Recording exported";
}

async function importRecording(file) {
  if (!file) return;
  try {
    const value = JSON.parse(await file.text());
    if (!isValidRecording(value)) throw new Error("Not a Text GPT-Live recording.");
    savedRecording = value;
    try {
      window.localStorage.setItem(RECORDING_STORAGE_KEY, JSON.stringify(value));
    } catch (error) {
      console.warn("Could not persist the recording in this browser.", error);
    }
    elements.recordingStatus.textContent = "Recording loaded";
    updateReplayButton();
  } catch (error) {
    showError(error);
  }
}

function loadSavedRecording() {
  try {
    const value = JSON.parse(window.localStorage.getItem(RECORDING_STORAGE_KEY) || "null");
    if (isValidRecording(value)) {
      savedRecording = value;
    }
  } catch (error) {
    console.warn("Could not load the previous recording.", error);
  }
  updateReplayButton();
}

function recordingElapsed() {
  return Math.max(0, Math.round(performance.now() - recordingStartedAt));
}

function appendRecordingFrame(type, value) {
  if (activeRecording === null || replayRunning) return;
  const key = JSON.stringify(value);
  const previousKey = type === "input"
    ? activeRecording.lastInputKey
    : activeRecording.lastActionKey;
  if (key === previousKey) return;
  if (type === "input") activeRecording.lastInputKey = key;
  else activeRecording.lastActionKey = key;
  activeRecording.frames.push({ at: recordingElapsed(), type, value });
}

function captureInputFrame() {
  appendRecordingFrame("input", {
    text: elements.text.value,
    highlights: renderedHighlights,
  });
}

function quoteRange(text, quote, occurrence) {
  if (!quote || !Number.isInteger(occurrence) || occurrence < 1) return null;
  let start = -1;
  let cursor = 0;
  for (let match = 0; match < occurrence; match += 1) {
    start = text.indexOf(quote, cursor);
    if (start < 0) return null;
    cursor = start + 1;
  }
  return { start, end: start + quote.length };
}

function renderHighlights(text, highlights = [], suggestions = []) {
  renderedHighlights = Array.isArray(highlights)
    ? highlights
      .filter((highlight) => typeof highlight?.quote === "string")
      .map((highlight) => ({
        quote: highlight.quote,
        occurrence: Number(highlight.occurrence),
        event_index: highlight.event_index,
      }))
    : [];

  const suggestionRanges = (Array.isArray(suggestions) ? suggestions : [])
    .filter((s) => typeof s?.quote === "string")
    .map((s) => {
      const range = quoteRange(text, s.quote, Number(s.occurrence));
      return range ? { ...range, replacement: s.replacement } : null;
    })
    .filter(Boolean)
    .sort((left, right) => left.start - right.start);

  const ranges = renderedHighlights
    .map((highlight) => quoteRange(text, highlight.quote, highlight.occurrence))
    .filter(Boolean)
    .sort((left, right) => left.start - right.start || right.end - left.end);
  const mergedRanges = [];
  for (const range of ranges) {
    const previous = mergedRanges.at(-1);
    if (previous && range.start <= previous.end) {
      previous.end = Math.max(previous.end, range.end);
    } else {
      mergedRanges.push({ ...range });
    }
  }

  const markSegments = mergedRanges
    .filter((m) => !suggestionRanges.some((s) => m.start < s.end && s.start < m.end))
    .map((m) => ({ ...m, kind: "mark" }));
  const segments = [...markSegments, ...suggestionRanges.map((s) => ({ ...s, kind: "suggest" }))]
    .sort((left, right) => left.start - right.start);

  const fragment = document.createDocumentFragment();
  let cursor = 0;
  for (const segment of segments) {
    if (segment.start < cursor) continue;
    fragment.append(document.createTextNode(text.slice(cursor, segment.start)));
    if (segment.kind === "mark") {
      const mark = document.createElement("mark");
      mark.textContent = text.slice(segment.start, segment.end);
      fragment.append(mark);
    } else {
      const wrap = document.createElement("span");
      wrap.className = "suggest";
      const wrong = document.createElement("s");
      wrong.textContent = text.slice(segment.start, segment.end);
      const arrow = document.createElement("span");
      arrow.className = "arrow";
      arrow.textContent = "\u2192";
      const fix = document.createElement("ins");
      fix.textContent = segment.replacement;
      wrap.append(wrong, arrow, fix);
      fragment.append(wrap);
    }
    cursor = segment.end;
  }
  fragment.append(document.createTextNode(text.slice(cursor)));
  elements.highlightLayer.replaceChildren(fragment);
}

function applyInputSnapshot(snapshot) {
  const isLegacy = typeof snapshot === "string";
  const text = isLegacy ? snapshot : String(snapshot?.text || "");
  const highlights = isLegacy ? [] : (snapshot?.highlights || []);
  elements.text.value = text;
  renderHighlights(text, highlights);
}

function setCurrentAction({ state, title = "", value, raw = "", index }) {
  if (elements.paneStatus && state !== "idle" && state !== "waiting") {
    elements.paneStatus.textContent = "acting\u2026";
    elements.paneStatus.dataset.state = state;
  }
  elements.currentAction.dataset.action = state;
  elements.currentActionTitle.textContent = title;
  elements.currentActionValue.textContent = value;
  elements.currentActionRaw.textContent = raw;
  elements.currentActionRaw.hidden = !raw;
  elements.currentActionIndex.textContent = index;
}

function currentActionSnapshot() {
  return {
    state: elements.currentAction.dataset.action || "idle",
    title: elements.currentActionTitle.textContent,
    value: elements.currentActionValue.textContent,
    raw: elements.currentActionRaw.textContent,
    index: elements.currentActionIndex.textContent,
  };
}

function captureActionFrame() {
  appendRecordingFrame("action", currentActionSnapshot());
}

function applyActionSnapshot(snapshot) {
  setCurrentAction({
    state: snapshot.state || "idle",
    // Recordings made before the raw line existed carry the verb in both
    // fields; suppress the duplicate title instead of showing it twice.
    title: snapshot.title === snapshot.value ? "" : (snapshot.title || ""),
    value: snapshot.value || "Model is silent",
    raw: snapshot.raw || "",
    index: snapshot.index || "No decision",
  });
  if (actionPanelView === "stream" && !["idle", "waiting", "connecting"].includes(snapshot.state)) {
    showActionStage({
      key: `${snapshot.index || "replay"}:${snapshot.state}:${snapshot.value || ""}`,
      state: snapshot.state || "tool",
      label: snapshot.title || "Model action",
      value: snapshot.value || "Action completed",
      stream: snapshot.state === "respond",
    });
  }
}

function setActionPanelView(view) {
  actionPanelView = view === "model-actions" ? "model-actions" : "stream";
  const showsStream = actionPanelView === "stream";
  document.body.dataset.actionPanel = actionPanelView;
  elements.streamViewButton?.classList.toggle("is-active", showsStream);
  elements.modelActionsViewButton?.classList.toggle("is-active", !showsStream);
  elements.streamViewButton?.setAttribute("aria-pressed", String(showsStream));
  elements.modelActionsViewButton?.setAttribute("aria-pressed", String(!showsStream));
  if (!showsStream) dismissActionStage();
}

function dismissActionStage({ resetKey = false } = {}) {
  if (actionStageTimer !== null) {
    window.clearTimeout(actionStageTimer);
    actionStageTimer = null;
  }
  if (actionStageStreamFrame !== null) {
    window.cancelAnimationFrame(actionStageStreamFrame);
    actionStageStreamFrame = null;
  }
  if (elements.actionStage) {
    elements.actionStage.classList.remove("is-visible");
    elements.actionStage.classList.remove("is-streaming");
    elements.actionStage.classList.remove("is-dismissing");
    elements.actionStage.hidden = true;
  }
  if (resetKey) lastStagedActionKey = "";
}

function scheduleActionStageDismiss(delay = ACTION_STAGE_MS) {
  if (actionStageTimer !== null) window.clearTimeout(actionStageTimer);
  actionStageTimer = window.setTimeout(() => {
    actionStageTimer = null;
    elements.actionStage.classList.add("is-dismissing");
    actionStageTimer = window.setTimeout(
      () => dismissActionStage(),
      ACTION_STAGE_EXIT_MS,
    );
  }, delay);
}

function streamResponseText(value, onComplete) {
  const chunks = String(value).match(/\S+\s*/g) || [String(value)];
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reducedMotion || chunks.length <= 1) {
    elements.actionStageValue.textContent = value;
    onComplete();
    return;
  }

  const duration = Math.min(1500, Math.max(650, chunks.length * 80));
  let startedAt = null;
  let visibleChunks = 0;
  elements.actionStage.classList.add("is-streaming");

  const reveal = (timestamp) => {
    if (startedAt === null) startedAt = timestamp;
    const progress = Math.min(1, (timestamp - startedAt) / duration);
    const targetChunks = Math.max(1, Math.ceil(chunks.length * progress));
    if (targetChunks !== visibleChunks) {
      visibleChunks = targetChunks;
      elements.actionStageValue.textContent = chunks.slice(0, visibleChunks).join("");
    }
    if (progress < 1) {
      actionStageStreamFrame = window.requestAnimationFrame(reveal);
    } else {
      actionStageStreamFrame = null;
      elements.actionStage.classList.remove("is-streaming");
      onComplete();
    }
  };
  actionStageStreamFrame = window.requestAnimationFrame(reveal);
}

function showActionStage({ key, state, label = "", value, stream = false }) {
  if (actionPanelView !== "stream" || !elements.actionStage || !value || key === lastStagedActionKey) return;
  lastStagedActionKey = key;
  if (actionStageTimer !== null) window.clearTimeout(actionStageTimer);
  if (actionStageStreamFrame !== null) {
    window.cancelAnimationFrame(actionStageStreamFrame);
    actionStageStreamFrame = null;
  }

  elements.actionStage.dataset.action = state;
  elements.actionStageLabel.textContent = label;
  elements.actionStageValue.textContent = stream ? "" : value;
  elements.actionStage.classList.remove("is-streaming");
  elements.actionStage.classList.remove("is-dismissing");
  if (elements.actionStageAnnouncement) {
    elements.actionStageAnnouncement.textContent = label ? `${label}: ${value}` : value;
  }
  elements.actionStage.hidden = false;
  elements.actionStage.classList.remove("is-visible");
  void elements.actionStage.offsetWidth;
  elements.actionStage.classList.add("is-visible");
  if (stream) {
    streamResponseText(value, () => scheduleActionStageDismiss(RESPONSE_STAGE_HOLD_MS));
  } else {
    scheduleActionStageDismiss();
  }
}

function stageTurnAction(turn) {
  const action = turn?.action;
  if (actionPanelView !== "stream" || !action?.valid || action.kind === "idle") return;

  const key = `${turn.event.index}:${actionText(action)}`;
  if (action.kind === "respond") {
    showActionStage({
      key,
      state: "respond",
      value: action.message || "Response ready",
      stream: true,
    });
    return;
  }

  const name = action.tool_name || "tool";
  const argumentsValue = action.arguments || {};
  // Translation is its own continuous channel in the Model pane, not a
  // notification. The channel is updated from the committed session state.
  if (name === "translate_commit") return;

  // Search is a sustained background activity, so retain its compact status
  // treatment instead of presenting it as an utterance from the Model.
  if (name === "web_search") {
    showActionStage({
      key,
      state: "web_search",
      label: "Searching the web",
      value: String(argumentsValue.query || "Search query"),
    });
    return;
  }

  if (name === "highlight" || (name === "ui" && argumentsValue.operation === "highlight")) {
    const target = argumentsValue.target || argumentsValue;
    const quote = String(target.quote || "");
    showActionStage({
      key,
      state: "highlight",
      value: `Highlight: “${quote}”`,
    });
    return;
  }

  const toolDisplays = {
    delegate: ["Working in background", argumentsValue.task],
    generate_ui: ["Building UI", argumentsValue.request],
    suggest_edit: ["Suggested edit", argumentsValue.replacement],
  };
  const [verb, payload] = toolDisplays[name] || ["Model action", `Using ${name}`];
  showActionStage({ key, state: name, value: `${verb}: ${payload || `Using ${name}`}` });
}

function stopTranslationAnimation() {
  if (translationAnimationFrame !== null) {
    window.cancelAnimationFrame(translationAnimationFrame);
    translationAnimationFrame = null;
  }
  translationAnimationTarget = "";
  elements.translationOutput?.removeAttribute("data-state");
}

function renderTranslation(commits = []) {
  if (!elements.translationOutput || !elements.translationText) return;
  const translation = commits.map((commit) => String(commit.message || "")).join("");
  elements.translationOutput.hidden = !translation;

  if (!translation) {
    stopTranslationAnimation();
    elements.translationText.textContent = "";
    translationHydrated = false;
    return;
  }

  const rendered = elements.translationText.textContent || "";
  if (!translationHydrated) {
    stopTranslationAnimation();
    elements.translationText.textContent = translation;
    translationHydrated = true;
    return;
  }

  if (translation === translationAnimationTarget && translationAnimationFrame !== null) return;
  if (translation === rendered) return;

  stopTranslationAnimation();
  if (!translation.startsWith(rendered)
    || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    elements.translationText.textContent = translation;
    return;
  }

  const committedCharacters = Array.from(rendered);
  const appendedCharacters = Array.from(translation).slice(committedCharacters.length);
  if (!appendedCharacters.length) return;

  translationAnimationTarget = translation;
  elements.translationOutput.dataset.state = "committing";
  const duration = Math.min(
    900,
    Math.max(220, appendedCharacters.length * TRANSLATION_APPEND_MS_PER_CHARACTER),
  );
  let startedAt = null;
  let visibleCharacters = 0;

  const reveal = (timestamp) => {
    if (startedAt === null) startedAt = timestamp;
    const progress = Math.min(1, (timestamp - startedAt) / duration);
    const targetCharacters = Math.max(1, Math.ceil(appendedCharacters.length * progress));
    if (targetCharacters !== visibleCharacters) {
      visibleCharacters = targetCharacters;
      elements.translationText.textContent = [
        ...committedCharacters,
        ...appendedCharacters.slice(0, visibleCharacters),
      ].join("");
    }
    if (progress < 1) {
      translationAnimationFrame = window.requestAnimationFrame(reveal);
      return;
    }
    translationAnimationFrame = null;
    translationAnimationTarget = "";
    elements.translationOutput.removeAttribute("data-state");
  };
  translationAnimationFrame = window.requestAnimationFrame(reveal);
}

function renderDecisionLatency(ms) {
  const readout = document.querySelector("#config-latency");
  if (!readout || typeof ms !== "number") return;
  readout.textContent = `${ms} ms`;
  readout.classList.toggle("is-over-budget", ms > TICK_MS);
}

function applyConfigStrip(health) {
  const model = document.querySelector("#config-model");
  if (model) model.textContent = health.model_name || "";
  const latency = health.decision_latency;
  if (latency && typeof latency.p50_ms === "number") renderDecisionLatency(latency.p50_ms);
  document.querySelectorAll("[data-config]").forEach((button) => {
    const active = health[button.dataset.config] === button.dataset.value;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

document.querySelectorAll("[data-config]").forEach((button) => {
  button.addEventListener("click", async () => {
    try {
      const health = await request("/api/config", {
        method: "POST",
        body: JSON.stringify({ [button.dataset.config]: button.dataset.value }),
      });
      applyConfigStrip(health);
    } catch (error) {
      showError(error);
    }
  });
});

function applyModelHealth(health) {
  applyConfigStrip(health);
  actionSchema = health.action_schema || "legacy";
  if (actionSchema === "g1") {
    elements.instruction.value = "";
    sessionInstruction = "";
    elements.instruction.placeholder = "For g1, type instructions into the live text stream";
  }
  updateInstructionAvailability();
  modelStatus = health.model_status || "unavailable";
  elements.modelName.textContent = health.model_name || "Model unavailable";
  elements.modelName.dataset.status = modelStatus;
  elements.modelName.title = health.model_message || health.model_name || "No model available";
  elements.modeLabel.textContent = health.model_name || "Model unavailable";
}

function pulse() {
  for (const dot of [elements.pulse, elements.decidePulse]) {
    dot.classList.remove("tick");
    void dot.offsetWidth;
    dot.classList.add("tick");
  }
}

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    const entries = Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`);
    return `{${entries.join(",")}}`;
  }
  return JSON.stringify(value);
}

function actionBody(action) {
  if (action.kind === "tool" && action.tool_name) {
    if (
      actionSchema === "g1"
      && ["delegate", "highlight", "suggest_edit", "translate_commit", "web_search"].includes(action.tool_name)
    ) {
      return `${action.tool_name}(${stableJson(action.arguments || {})})`;
    }
    return `tool(${action.tool_name},${stableJson(action.arguments || {})})`;
  }
  if (action.kind === "respond" && Number.isInteger(action.target) && action.message) {
    return `respond(${stableJson({ for: action.target, message: action.message })})`;
  }
  return "idle()";
}

function actionText(action) {
  return `<action>${actionBody(action)}</action>`;
}

function actionFlavor(action) {
  if (action.kind === "respond") return "respond";
  if (action.kind !== "tool") return "idle";
  if (action.tool_name === "web_search") return "search";
  if (
    action.tool_name === "highlight"
    || (action.tool_name === "ui" && action.arguments?.operation === "highlight")
  ) return "highlight";
  return "tool";
}

function isModelUnavailable(action) {
  return !action.valid && String(action.raw_output || "").startsWith("<policy_error>");
}

function showNoModelAvailable(message = "") {
  setCurrentAction({ state: "invalid", value: "No model available", index: "Model unavailable" });
  setDecideActivity(false);
  setRuntimeState("No model available", "error");
  if (message) elements.modelName.title = message;
  captureActionFrame();
}

function showModelConnecting() {
  setCurrentAction({ state: "connecting", value: "Connecting to model…", index: "Loading model" });
  setRuntimeState("Connecting to model", "live");
  captureActionFrame();
}

function showSessionPreparing() {
  setCurrentAction({
    state: "connecting",
    value: "Preparing Qwen…",
    index: "Building session cache",
  });
  setRuntimeState("Preparing Qwen", "live");
  captureActionFrame();
}

function setDecideActivity(active) {
  elements.decidePulse.hidden = !active;
  if (!active) elements.decidePulse.classList.remove("is-busy");
}

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatHistoryForCopy(history) {
  const lines = ["<stream_history>"];
  for (const turn of history) {
    const attributes = [
      `index="${turn.event.index}"`,
      `source="${turn.event.source}"`,
    ];
    if (turn.event.state) attributes.push(`state="${turn.event.state}"`);
    if (turn.event.elapsed_ms !== undefined) attributes.push(`time="t+${turn.event.elapsed_ms}ms"`);
    if (turn.event.tool_name) attributes.push(`tool="${escapeXml(turn.event.tool_name)}"`);
    if (turn.event.call_id) attributes.push(`call_id="${escapeXml(turn.event.call_id)}"`);
    lines.push(`  <stream_event ${attributes.join(" ")}>${escapeXml(turn.event.content)}</stream_event>`);
    lines.push(`  <action>${escapeXml(actionBody(turn.action))}</action>`, "");
  }
  if (lines.at(-1) === "") lines.pop();
  lines.push("</stream_history>");
  return lines.join("\n");
}

function updateHistoryView() {
  const showReadable = historyFormat === "readable";
  const hasContent = showReadable ? renderedHistory.length > 0 : latestModelInput.length > 0;

  elements.readableHistory.classList.toggle("is-active", showReadable);
  elements.modelInputButton.classList.toggle("is-active", !showReadable);
  elements.readableHistory.setAttribute("aria-pressed", String(showReadable));
  elements.modelInputButton.setAttribute("aria-pressed", String(!showReadable));
  elements.trace.hidden = !showReadable || !hasContent;
  elements.modelInput.hidden = showReadable || !hasContent;
  elements.copyHistory.disabled = !hasContent;
}

function setHistoryFormat(format) {
  historyFormat = format === "model" ? "model" : "readable";
  updateHistoryView();
}

function fallbackCopy(text) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.append(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) throw new Error("Clipboard copy was rejected.");
}

async function copyHistory() {
  const text = historyFormat === "model"
    ? latestModelInput
    : formatHistoryForCopy(renderedHistory);
  if (!text) return;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      fallbackCopy(text);
    }
    elements.copyHistory.dataset.copyState = "copied";
    elements.copyHistory.setAttribute("aria-label", "History copied");
    elements.copyHistory.title = "History copied";
    elements.copyHistoryStatus.textContent = "History copied";
  } catch (error) {
    console.error(error);
    elements.copyHistory.dataset.copyState = "failed";
    elements.copyHistory.setAttribute("aria-label", "Copy failed");
    elements.copyHistory.title = "Copy failed";
    elements.copyHistoryStatus.textContent = "Copy failed";
  }
  if (copyFeedbackTimer !== null) window.clearTimeout(copyFeedbackTimer);
  copyFeedbackTimer = window.setTimeout(() => {
    delete elements.copyHistory.dataset.copyState;
    elements.copyHistory.setAttribute("aria-label", "Copy history");
    elements.copyHistory.title = "Copy history";
    elements.copyHistoryStatus.textContent = "";
    copyFeedbackTimer = null;
  }, 1600);
}

function parseToolContent(content) {
  try {
    return JSON.parse(content);
  } catch {
    return null;
  }
}

function buildToolResults(event) {
  if (event.source !== "tool" || event.tool_name !== "web_search") return null;
  const data = parseToolContent(event.content);
  if (!data) return null;

  const container = document.createElement("div");
  container.className = "tool-results";
  if (data.error) {
    const error = document.createElement("p");
    error.textContent = data.error;
    container.append(error);
    return container;
  }
  for (const result of (data.results || []).slice(0, 3)) {
    const item = document.createElement("div");
    item.className = "tool-result";
    const title = document.createElement("strong");
    title.textContent = result.title || "Untitled result";
    const snippet = document.createElement("p");
    snippet.textContent = result.snippet || result.url || "No summary returned.";
    item.append(title, snippet);
    container.append(item);
  }
  return container;
}

// --- Generated UI panel -----------------------------------------------------
// A completed generate_ui job carries a declarative spec; render it into the
// slide-in stage. Keyed by job_id so polling re-renders don't re-animate, and
// a closed panel stays closed.

const UI_ACCENTS = new Set(["blue", "violet", "green", "amber", "red", "slate"]);

// Orientation follows the data, not a toggle: wordy category labels need room
// to the left (horizontal), while short sequential labels like years read
// better left-to-right along an axis (vertical). Long series stay vertical so
// the chart never grows into a tall column.
function chartOrientation(series) {
  const longest = Math.max(...series.map((point) => String(point.label).length));
  return longest > 6 && series.length <= 7 ? "horizontal" : "vertical";
}

function buildChart(component) {
  const figure = document.createElement("figure");
  figure.className = "ui-chart";
  if (component.label) {
    const caption = document.createElement("figcaption");
    caption.textContent = component.label;
    figure.append(caption);
  }

  const series = component.series || [];
  const values = series.map((point) => Math.max(0, Number(point.value) || 0));
  const max = Math.max(...values, 0.0001);
  // One accent, three weights: the peak carries it, the rest are a tint.
  const peak = values.indexOf(Math.max(...values));

  if (chartOrientation(series) === "horizontal") {
    const rows = document.createElement("div");
    rows.className = "ui-chart__rows";
    series.forEach((point, index) => {
      const label = document.createElement("span");
      label.className = "ui-chart__row-label";
      label.textContent = point.label;
      const track = document.createElement("div");
      track.className = "ui-chart__track";
      const bar = document.createElement("div");
      bar.className = `ui-chart__hbar${index === peak ? " is-peak" : ""}`;
      bar.style.width = `${Math.max(2, (values[index] / max) * 100)}%`;
      bar.style.animationDelay = `${index * 70}ms`;
      track.append(bar);
      const value = document.createElement("span");
      value.className = `ui-chart__row-value${index === peak ? " is-peak" : ""}`;
      value.textContent = point.value;
      rows.append(label, track, value);
    });
    figure.append(rows);
    return figure;
  }

  const bars = document.createElement("div");
  bars.className = "ui-chart__bars";
  for (const fraction of [50, 100]) {
    const line = document.createElement("div");
    line.className = "ui-chart__grid";
    line.style.bottom = `${fraction}%`;
    bars.append(line);
  }
  series.forEach((point, index) => {
    const slot = document.createElement("div");
    slot.className = "ui-chart__slot";
    const bar = document.createElement("div");
    bar.className = `ui-chart__bar${index === peak ? " is-peak" : ""}`;
    bar.style.height = `${Math.max(2, (values[index] / max) * 100)}%`;
    bar.style.animationDelay = `${index * 70}ms`;
    const value = document.createElement("i");
    value.textContent = point.value;
    const label = document.createElement("u");
    label.textContent = point.label;
    bar.append(value, label);
    slot.append(bar);
    bars.append(slot);
  });
  figure.append(bars);
  return figure;
}

// The model expresses layout intent (span, emphasis) and semantic tone; the
// renderer keeps ownership of every pixel those choices resolve to.
function uiBlock(component) {
  const block = document.createElement("div");
  block.className = "ui-block";
  if (component.span === "half") block.classList.add("is-half");
  if (component.emphasis === "lead") block.classList.add("is-lead");
  if (component.tone === "positive") block.classList.add("is-positive");
  if (component.tone === "negative") block.classList.add("is-negative");
  return block;
}

function buildHero(component) {
  const hero = document.createElement("div");
  hero.className = "ui-hero";
  if (component.image_url) {
    const image = document.createElement("img");
    image.src = component.image_url;
    image.alt = component.name || "";
    image.loading = "lazy";
    hero.append(image);
  } else {
    const initials = document.createElement("div");
    initials.className = "ui-hero__initials";
    initials.textContent = String(component.name || "?")
      .split(/\s+/)
      .map((word) => word[0] || "")
      .slice(0, 2)
      .join("");
    hero.append(initials);
  }
  const text = document.createElement("div");
  const name = document.createElement("h3");
  name.textContent = component.name || "";
  const tagline = document.createElement("p");
  tagline.textContent = component.tagline || "";
  text.append(name, tagline);
  hero.append(text);
  return hero;
}

function buildTimeline(component) {
  const wrap = document.createElement("div");
  wrap.className = "ui-timeline";
  if (component.label) {
    const heading = document.createElement("h4");
    heading.textContent = component.label;
    wrap.append(heading);
  }
  (component.events || []).forEach((event, index) => {
    const item = document.createElement("div");
    item.className = "ui-timeline__event";
    item.style.animationDelay = `${index * 60}ms`;
    const date = document.createElement("b");
    date.textContent = event.date;
    const text = document.createElement("span");
    text.textContent = event.text;
    item.append(date, text);
    wrap.append(item);
  });
  return wrap;
}

function buildFacts(component) {
  const wrap = document.createElement("div");
  wrap.className = "ui-facts";
  if (component.label) {
    const heading = document.createElement("h4");
    heading.textContent = component.label;
    wrap.append(heading);
  }
  for (const fact of component.facts || []) {
    const row = document.createElement("div");
    row.className = "ui-facts__row";
    const label = document.createElement("span");
    label.textContent = fact.label;
    const value = document.createElement("b");
    value.textContent = fact.value;
    row.append(label, value);
    wrap.append(row);
  }
  return wrap;
}

function buildComparison(component) {
  const fragment = document.createDocumentFragment();
  if (component.label) {
    const heading = document.createElement("h4");
    heading.className = "ui-compare__title";
    heading.textContent = component.label;
    fragment.append(heading);
  }
  const wrap = document.createElement("div");
  wrap.className = "ui-compare";
  for (const column of component.columns || []) {
    const cell = document.createElement("div");
    const heading = document.createElement("h4");
    heading.textContent = column.title;
    const list = document.createElement("ul");
    for (const item of column.items || []) {
      const entry = document.createElement("li");
      entry.textContent = item;
      list.append(entry);
    }
    cell.append(heading, list);
    wrap.append(cell);
  }
  fragment.append(wrap);
  return fragment;
}

function buildList(component) {
  const wrap = document.createElement("div");
  wrap.className = "ui-list";
  if (component.label) {
    const heading = document.createElement("h3");
    heading.textContent = component.label;
    wrap.append(heading);
  }
  const list = document.createElement("ul");
  (component.items || []).forEach((item, index) => {
    const entry = document.createElement("li");
    entry.textContent = item;
    entry.style.animationDelay = `${index * 50}ms`;
    list.append(entry);
  });
  wrap.append(list);
  return wrap;
}

function buildSources(component) {
  const wrap = document.createElement("p");
  wrap.className = "ui-sources";
  wrap.append("Sources: ");
  (component.links || []).forEach((link, index) => {
    if (index > 0) wrap.append(" · ");
    const anchor = document.createElement("a");
    anchor.href = link.url;
    anchor.textContent = link.title;
    anchor.target = "_blank";
    anchor.rel = "noreferrer noopener";
    wrap.append(anchor);
  });
  return wrap;
}

function createUIPanel(panelTitle = "") {
  const root = document.createElement("article");
  root.className = "ui-panel";
  const heading = document.createElement("h2");
  heading.className = "ui-panel__title";
  heading.textContent = panelTitle;
  const grid = document.createElement("div");
  grid.className = "ui-panel__grid";
  root.append(heading, grid);
  root.dataset.componentCount = "0";
  return root;
}

function appendUIPanelComponent(grid, component, index) {
  if (component.type === "stat_card") {
    // Cards share a grid only while they stay consecutive, so the model's
    // ordering is never silently rearranged around them.
    let host = grid.lastElementChild;
    if (!host || !host.classList.contains("ui-block--cards")) {
      host = document.createElement("div");
      host.className = "ui-block ui-block--cards";
      if (component.span === "half") host.classList.add("is-half");
      const cards = document.createElement("div");
      cards.className = "ui-cards";
      host.append(cards);
      grid.append(host);
    }
    const cards = host.querySelector(".ui-cards");
    const card = document.createElement("div");
    card.className = "ui-card";
    if (component.tone === "positive") card.classList.add("is-positive");
    if (component.tone === "negative") card.classList.add("is-negative");
    card.style.animationDelay = `${index * 60}ms`;
    const value = document.createElement("span");
    value.className = "ui-card__value";
    value.textContent = component.value;
    const label = document.createElement("span");
    label.className = "ui-card__label";
    label.textContent = component.label;
    card.append(value, label);
    if (typeof component.sub === "string" && component.sub.trim()) {
      const sub = document.createElement("span");
      sub.className = "ui-card__sub";
      sub.textContent = component.sub;
      card.append(sub);
    }
    cards.append(card);
    cards.dataset.count = String(Math.min(cards.childElementCount, 4));
    return;
  }

  const builders = {
    hero: buildHero,
    chart: buildChart,
    timeline: buildTimeline,
    fact: buildFacts,
    comparison: buildComparison,
    list: buildList,
    sources: buildSources,
  };
  const block = uiBlock(component);
  if (component.type === "callout") {
    block.classList.add("ui-callout");
    block.textContent = component.text;
  } else {
    const build = builders[component.type];
    if (!build) return;
    block.append(build(component));
  }
  grid.append(block);
}

function syncUIPanel(root, spec) {
  const title = root.querySelector(".ui-panel__title");
  if (title && spec.title) title.textContent = spec.title;
  if (UI_ACCENTS.has(spec.accent)) root.dataset.accent = spec.accent;
  const grid = root.querySelector(".ui-panel__grid");
  const components = spec.components || [];
  const rendered = Number(root.dataset.componentCount || 0);
  if (components.length < rendered) {
    const replacement = buildUIPanel(spec);
    root.replaceWith(replacement);
    return replacement;
  }
  components.slice(rendered).forEach((component, offset) => {
    appendUIPanelComponent(grid, component, rendered + offset);
  });
  root.dataset.componentCount = String(components.length);

  return root;
}

function buildUIPanel(spec) {
  const root = createUIPanel(spec.title || "");
  syncUIPanel(root, spec);
  return root;
}

const workThreads = new Map(); // job_id -> {node, status, specKey, body, panel}
// Jobs whose thread has already left the pane; polling must not rebuild them.
const retiredJobs = new Set();

function reduceWorkMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function captureWorkLayout() {
  const layout = new Map();
  for (const [id, entry] of workThreads) {
    if (entry.node.isConnected) layout.set(id, entry.node.getBoundingClientRect());
  }
  return layout;
}

function syncWorkSurfaceCount() {
  if (!elements.workSurface) return;
  const count = elements.workSurface.childElementCount;
  elements.workSurface.dataset.jobCount = String(count);
  elements.liveStage?.classList.toggle("has-work", count > 0);
  elements.externalWorkbench?.setAttribute("aria-hidden", String(count === 0));
  if (elements.workbenchCount) {
    elements.workbenchCount.textContent = count ? `${count} active` : "";
  }
}

// FLIP the surviving cards when a concurrent job splits the surface or leaves
// it. The work itself stays in normal document flow; only the transition is
// inverted, so the final layout remains responsive and content-sized.
function animateWorkLayout(beforeLayout) {
  if (reduceWorkMotion() || !beforeLayout.size) return;
  window.requestAnimationFrame(() => {
    for (const [id, before] of beforeLayout) {
      const entry = workThreads.get(id);
      if (!entry?.node.isConnected || entry.node.classList.contains("is-leaving")) continue;
      const after = entry.node.getBoundingClientRect();
      const deltaX = before.left - after.left;
      const deltaY = before.top - after.top;
      const scaleX = before.width / Math.max(after.width, 1);
      if (Math.abs(deltaX) < 1 && Math.abs(deltaY) < 1 && Math.abs(scaleX - 1) < 0.01) continue;
      entry.layoutAnimation?.cancel();
      entry.layoutAnimation = entry.node.animate(
        [
          { transform: `translate(${deltaX}px, ${deltaY}px) scaleX(${scaleX})` },
          { transform: "none" },
        ],
        { duration: 520, easing: "cubic-bezier(0.22, 1, 0.36, 1)" },
      );
    }
  });
}

// Keep one card identity while its body changes from a launch placeholder to
// progressive UI, search results, or an error. Animating the measured height
// makes that state change read as a morph rather than a replacement.
function morphWorkThread(entry, update, { animate = true } = {}) {
  entry.morphAnimation?.cancel();
  const beforeHeight = entry.node.getBoundingClientRect().height;
  update();
  if (!animate || reduceWorkMotion() || !entry.node.isConnected) return;
  const afterHeight = entry.node.getBoundingClientRect().height;
  entry.node.classList.add("is-morphing");
  entry.morphAnimation = entry.node.animate(
    [
      { height: `${beforeHeight}px` },
      { height: `${afterHeight}px` },
    ],
    { duration: 560, easing: "cubic-bezier(0.22, 1, 0.36, 1)" },
  );
  entry.body.animate(
    [
      { opacity: 0.3, transform: "translateY(-7px)" },
      { opacity: 1, transform: "none" },
    ],
    { duration: 420, easing: "cubic-bezier(0.22, 1, 0.36, 1)" },
  );
  const animation = entry.morphAnimation;
  animation.addEventListener("finish", () => {
    if (entry.morphAnimation !== animation) return;
    entry.morphAnimation = null;
    entry.node.classList.remove("is-morphing");
  });
}

function retireWorkThread(id) {
  const entry = workThreads.get(id);
  retiredJobs.add(id);
  if (!entry) return;
  entry.node.classList.add("is-leaving");
  window.setTimeout(() => {
    const beforeLayout = captureWorkLayout();
    entry.node.remove();
    workThreads.delete(id);
    syncWorkSurfaceCount();
    animateWorkLayout(beforeLayout);
  }, WORK_THREAD_EXIT_MS);
}

function collectJobs(history, sessionJobs = []) {
  const jobs = new Map(); // id -> {tool, request, status, payload, startIndex}
  for (const turn of history) {
    const action = turn.action;
    if (
      action.valid
      && action.kind === "tool"
      && ["delegate", "generate_ui", "web_search"].includes(action.tool_name)
    ) {
      const id = `job-${turn.event.index}`;
      jobs.set(id, {
        tool: action.tool_name,
        request: action.arguments?.task || action.arguments?.request || action.arguments?.query || "",
        status: "starting",
        payload: null,
      });
    }
    if (turn.event.source === "tool" && turn.event.job_id) {
      const data = parseToolContent(turn.event.content);
      if (!data) continue;
      const job = jobs.get(turn.event.job_id) || jobs.set(turn.event.job_id, {
        tool: turn.event.tool_name, request: "", status: "starting", payload: null,
      }).get(turn.event.job_id);
      if (data.status === "accepted") job.status = "running";
      if (data.status === "completed") {
        job.status = "completed";
        job.payload = data;
        job.resultIndex = turn.event.index;
      }
      if (data.status === "failed") {
        job.status = "failed";
        job.payload = data;
        job.resultIndex = turn.event.index;
      }
      // A web_search result that carries its payload inline lands on the same
      // event, so record where it arrived.
      if (data.results && job.resultIndex === undefined) {
        job.status = "completed";
        job.payload = data;
        job.resultIndex = turn.event.index;
      }
    }
  }
  // g1 deliberately keeps generated UI specs out of model-visible tool events.
  // The session API returns those render-only specs on the derived job instead.
  for (const apiJob of sessionJobs) {
    const job = jobs.get(apiJob.job_id);
    if (!job) continue;
    if (apiJob.spec) {
      job.payload = { ...(job.payload || {}), spec: apiJob.spec };
    }
    if (apiJob.error && !job.payload?.error) {
      job.payload = { ...(job.payload || {}), error: apiJob.error };
    }
  }
  return jobs;
}

function buildSearchResults(payload) {
  const wrap = document.createElement("div");
  wrap.className = "job-search";
  const results = (payload.results || []).slice(0, 4);
  for (const [index, result] of results.entries()) {
    const item = document.createElement("div");
    item.className = "job-search__result";
    if (index > 1) item.hidden = true;
    const title = document.createElement("a");
    title.textContent = result.title || "Untitled";
    if (result.url) { title.href = result.url; title.target = "_blank"; title.rel = "noreferrer noopener"; }
    const snippet = document.createElement("p");
    snippet.textContent = result.snippet || "";
    item.append(title, snippet);
    wrap.append(item);
  }
  if (results.length > 2) {
    const more = document.createElement("button");
    more.className = "job-search__more";
    more.type = "button";
    more.textContent = `${results.length - 2} more results`;
    more.setAttribute("aria-expanded", "false");
    more.addEventListener("click", () => {
      const expanded = more.getAttribute("aria-expanded") === "true";
      more.setAttribute("aria-expanded", String(!expanded));
      for (const item of wrap.querySelectorAll(".job-search__result:nth-of-type(n + 3)")) {
        item.hidden = expanded;
      }
      more.textContent = expanded ? `${results.length - 2} more results` : "Show fewer results";
    });
    wrap.append(more);
  }
  if (!wrap.childNodes.length) wrap.textContent = "No results.";
  return wrap;
}

function makeWorkThread(job) {
  const node = document.createElement("section");
  node.className = "work-thread is-entering";
  node.dataset.tool = job.tool;
  const head = document.createElement("header");
  head.className = "work-thread__head";
  const signal = document.createElement("span");
  signal.className = "work-thread__signal";
  signal.setAttribute("aria-hidden", "true");
  const label = document.createElement("span");
  label.className = "work-thread__label";
  label.textContent = job.tool === "web_search" ? "Searching the web" : "Building interface";
  const state = document.createElement("span");
  state.className = "work-thread__state";
  head.append(signal, label, state);
  const body = document.createElement("div");
  body.className = "work-thread__body";
  node.append(head, body);
  return {
    node,
    state,
    body,
    status: null,
    specKey: "",
    panel: null,
    retiring: false,
    morphAnimation: null,
    layoutAnimation: null,
  };
}

function buildWorkPlaceholder(job) {
  const wrap = document.createElement("div");
  wrap.className = "work-thread__placeholder";
  const motion = document.createElement("div");
  motion.className = job.tool === "web_search" ? "work-thread__scan" : "work-thread__shimmer";
  motion.setAttribute("aria-hidden", "true");
  if (job.tool === "web_search") {
    const request = document.createElement("p");
    request.textContent = job.request;
    wrap.append(request, motion);
  } else {
    wrap.append(motion);
  }
  return wrap;
}

function renderWorkSurface(history, sessionJobs = []) {
  if (!elements.workSurface) return;
  const beforeLayout = captureWorkLayout();
  let layoutChanged = false;
  const jobs = collectJobs(history, sessionJobs);
  const visibleJobs = [...jobs.entries()].filter(([id, job]) => (
    ["delegate", "generate_ui", "web_search"].includes(job.tool) && !retiredJobs.has(id)
  ));
  const activeIds = new Set(visibleJobs.map(([id]) => id));
  for (const [id, entry] of workThreads) {
    if (activeIds.has(id)) continue;
    entry.node.remove();
    workThreads.delete(id);
    layoutChanged = true;
  }

  for (const [id, job] of visibleJobs) {
    let entry = workThreads.get(id);
    let isNew = false;
    if (!entry) {
      entry = makeWorkThread(job);
      elements.workSurface.append(entry.node);
      requestAnimationFrame(() => entry.node.classList.remove("is-entering"));
      workThreads.set(id, entry);
      isNew = true;
      layoutChanged = true;
    }

    const hasUI = ["delegate", "generate_ui"].includes(job.tool);
    const spec = job.payload?.spec;
    const specKey = spec ? JSON.stringify(spec) : "";
    const changedStatus = entry.status !== job.status;
    entry.status = job.status;
    entry.node.dataset.status = job.status;
    entry.state.textContent = job.status === "completed" ? "Ready"
      : job.status === "failed" ? "Failed" : "";

    // Search is transient evidence, not a deliverable. Hold completed results
    // long enough to read, then return the full workbench to generated UI.
    const finished = job.tool === "web_search"
      && job.status === "completed"
      && job.resultIndex !== undefined;
    if (finished && !entry.retiring) {
      entry.retiring = true;
      window.setTimeout(() => retireWorkThread(id), SEARCH_RESULT_HOLD_MS);
    }

    if (job.status === "failed") {
      if (!changedStatus) continue;
      const error = document.createElement("p");
      error.className = "work-thread__error";
      error.textContent = job.payload?.error || "Job failed.";
      morphWorkThread(entry, () => entry.body.replaceChildren(error), { animate: !isNew });
      continue;
    }

    if (hasUI && spec) {
      const needsUpdate = !entry.panel || entry.specKey !== specKey;
      if (needsUpdate) {
        morphWorkThread(entry, () => {
          if (!entry.panel) {
            entry.body.replaceChildren();
            entry.panel = buildUIPanel(spec);
            entry.body.append(entry.panel);
          } else if (entry.specKey !== specKey) {
            entry.panel = syncUIPanel(entry.panel, spec);
          }
        }, { animate: !isNew });
        entry.specKey = specKey;
      }
      continue;
    }

    if (job.status === "completed" && !hasUI) {
      if (changedStatus) {
        const results = buildSearchResults(job.payload);
        morphWorkThread(entry, () => entry.body.replaceChildren(results), { animate: !isNew });
      }
      continue;
    }

    if (changedStatus || !entry.body.childNodes.length) {
      const placeholder = buildWorkPlaceholder(job);
      morphWorkThread(entry, () => entry.body.replaceChildren(placeholder), { animate: !isNew });
    }
  }
  syncWorkSurfaceCount();
  if (layoutChanged) animateWorkLayout(beforeLayout);
}

function renderCurrentAction(history, pendingSearches) {
  const turn = history.at(-1);

  if (!turn) {
    setCurrentAction({ state: "idle", value: "Model is silent", index: "No decision" });
    return;
  }

  const action = turn.action;
  if (isModelUnavailable(action)) {
    setCurrentAction({ state: "invalid", value: "No model available", index: "Model unavailable" });
    return;
  }

  const raw = actionText(action);
  const index = `Decision ${String(turn.event.index).padStart(2, "0")}`;
  if (!action.valid) {
    setCurrentAction({
      state: "invalid",
      title: "Fallback",
      value: action.diagnostic || "Invalid action",
      raw,
      index,
    });
    return;
  }

  // The big line carries the human payload; the raw serialization stays
  // beneath it in small type as the ground truth of what the model emitted.
  const flavor = actionFlavor(action);
  const displays = {
    idle: { title: "", value: "Model is silent", raw: "" },
    search: {
      title: pendingSearches > 0 ? "Searching" : "Search requested",
      value: action.arguments?.query || raw,
      raw,
    },
    respond: { title: "Responding", value: action.message || raw, raw },
    highlight: {
      title: "Highlighting",
      value: action.arguments?.quote ? `“${action.arguments.quote}”` : raw,
      raw,
    },
    tool: {
      title: action.tool_name ? `Using ${action.tool_name}` : "Using tool",
      value: raw,
      raw: "",
    },
  };
  setCurrentAction({ state: flavor, index, ...displays[flavor] });
  stageTurnAction(turn);
}

function renderSession(session) {
  const history = session.history || [];
  renderedHistory = history;
  latestSessionHighlights = session.highlights || [];
  latestSessionSuggestions = session.suggestions || [];
  renderHighlights(elements.text.value, latestSessionHighlights, latestSessionSuggestions);
  latestModelInput = session.latest_prompt || "";
  latestPendingSearches = session.pending_searches || 0;
  renderWorkSurface(history, session.jobs || []);
  renderTranslation(session.translation_commits || []);
  elements.modelInput.textContent = latestModelInput;
  if (document.body.dataset.playback !== "true") {
    renderCurrentAction(history, latestPendingSearches);
    captureActionFrame();
  }
  elements.trace.replaceChildren();

  for (const turn of [...history].reverse()) {
    const item = document.createElement("li");
    item.className = "trace-item";
    if (turn.event.index > lastRenderedIndex) item.classList.add("is-new");

    const index = document.createElement("span");
    index.className = "trace-index";
    index.textContent = String(turn.event.index).padStart(2, "0");

    const content = document.createElement("div");
    content.className = "trace-content";

    const meta = document.createElement("div");
    meta.className = "trace-meta";
    const source = document.createElement("span");
    source.className = `source-${turn.event.source}`;
    source.textContent = turn.event.source === "tool"
      ? (turn.event.tool_name || "tool")
      : turn.event.source;
    const timing = document.createElement("span");
    timing.className = "event-state";
    timing.textContent = [
      turn.event.state || "result",
      turn.event.elapsed_ms !== undefined ? `t+${turn.event.elapsed_ms}ms` : "",
    ].filter(Boolean).join(" · ");
    meta.append(source, timing);

    const eventText = document.createElement("p");
    eventText.className = "event-text";
    eventText.textContent = turn.event.source === "tool" && turn.event.tool_name === "web_search"
      ? (parseToolContent(turn.event.content)?.query || "Search returned")
      : (turn.event.content || "(empty text)");
    eventText.title = eventText.textContent;

    const action = document.createElement("div");
    const kindClass = actionFlavor(turn.action);
    action.className = `action-line ${kindClass} ${turn.action.valid ? "" : "invalid"}`;
    const actionLabel = document.createElement("span");
    actionLabel.className = "action-label";
    const modelUnavailable = isModelUnavailable(turn.action);
    actionLabel.textContent = modelUnavailable ? "model" : (turn.action.valid ? "action" : "fallback");
    const actionValue = document.createElement("span");
    actionValue.className = "action-value";
    actionValue.textContent = modelUnavailable
      ? "No model available"
      : (turn.action.valid
        ? actionText(turn.action)
        : `${actionText(turn.action)} - ${turn.action.diagnostic}`);
    action.append(actionLabel, actionValue);

    content.append(meta, eventText);
    const results = buildToolResults(turn.event);
    if (results) content.append(results);
    if (!turn.action.valid || turn.action.kind !== "idle") content.append(action);
    item.append(index, content);
    elements.trace.append(item);
  }

  if (elements.actionLog) {
    const log = document.createDocumentFragment();
    for (const turn of history) {
      const action = turn.action;
      if (!action.valid || action.kind === "idle") continue;
      const item = document.createElement("li");
      if (action.kind === "tool") item.classList.add("is-tool");
      if (action.tool_name === "delegate") item.classList.add("is-delegate");
      if (action.kind === "tool" && action.tool_name === "suggest_edit") {
        const target = action.arguments?.target || action.arguments || {};
        item.textContent = `fixed \u201c${target.quote}\u201d \u2192 \u201c${action.arguments?.replacement}\u201d`;
      } else if (action.kind === "tool" && action.tool_name === "web_search") {
        item.textContent = `searched \u201c${action.arguments?.query || ""}\u201d`;
      } else if (action.kind === "tool" && action.tool_name === "generate_ui") {
        item.textContent = `generated a panel: ${action.arguments?.request || ""}`;
      } else if (action.kind === "tool" && action.tool_name === "delegate") {
        item.textContent = `delegated: ${action.arguments?.task || ""}`;
      } else if (action.kind === "tool" && action.tool_name === "translate_commit") {
        item.textContent = `translated: ${action.arguments?.message || ""}`;
      } else if (action.kind === "respond") {
        item.textContent = `replied: ${action.message || ""}`;
      } else if (action.kind === "tool") {
        item.textContent = `used ${action.tool_name}`;
      } else {
        continue;
      }
      log.append(item);
    }
    elements.actionLog.replaceChildren(log);
  }

  const idles = history.filter((turn) => turn.action.kind === "idle").length;
  const tools = history.filter((turn) => turn.action.kind === "tool").length;
  elements.eventCount.textContent = history.length;
  elements.idleCount.textContent = idles;
  elements.searchCount.textContent = tools;
  elements.pendingCount.textContent = session.pending_searches || 0;
  updateHistoryView();
  lastRenderedIndex = history.at(-1)?.event.index || 0;
  captureInputFrame();
}

async function refreshSession() {
  if (!sessionId) return;
  const session = await request(`/api/sessions/${sessionId}`);
  renderSession(session);
  if (session.pending_searches > 0) {
    schedulePendingRefresh();
  }
}

function schedulePendingRefresh() {
  if (pendingPollTimer !== null) return;
  pendingPollTimer = window.setTimeout(async () => {
    pendingPollTimer = null;
    try {
      await refreshSession();
    } catch (error) {
      showError(error);
    }
  }, 250);
}

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function pauseStreaming() {
  running = false;
  delete document.body.dataset.running;
  if (timer !== null) {
    window.clearInterval(timer);
    timer = null;
  }
  setDecideActivity(false);
  setRunButton("Resume");
  setRuntimeState("Paused");
}

function restoreLiveDisplay() {
  const latestUserEvent = [...renderedHistory]
    .reverse()
    .find((turn) => turn.event.source === "user");
  elements.text.value = latestUserEvent?.event.content || "";
  renderHighlights(elements.text.value, latestSessionHighlights, latestSessionSuggestions);
  renderCurrentAction(renderedHistory, latestPendingSearches);
}

function stopReplay({ restore = true } = {}) {
  if (replayAnimationFrame !== null) {
    window.cancelAnimationFrame(replayAnimationFrame);
    replayAnimationFrame = null;
  }
  replayRunning = false;
  delete document.body.dataset.playback;
  elements.text.readOnly = false;
  if (restore) restoreLiveDisplay();
  updateReplayButton();
}

function finishReplay() {
  replayAnimationFrame = null;
  replayRunning = false;
  elements.recordingStatus.textContent = "Replay complete";
  updateReplayButton();
}

function startReplay() {
  if (savedRecording === null || activeRecording !== null) return;
  if (replayRunning) {
    stopReplay();
    return;
  }

  loadingModel = false;
  pauseStreaming();
  setView("current");
  document.body.dataset.playback = "true";
  elements.text.readOnly = true;
  elements.text.blur();
  applyInputSnapshot({ text: "", highlights: [] });
  renderCurrentAction([], 0);
  replayRunning = true;
  updateReplayButton();
  elements.recordingStatus.textContent = "Replay started";

  const frames = savedRecording.frames;
  let cursor = 0;
  const startedAt = performance.now();
  const renderFrame = (now) => {
    const elapsed = now - startedAt;
    while (cursor < frames.length && frames[cursor].at <= elapsed) {
      const frame = frames[cursor];
      if (frame.type === "input") applyInputSnapshot(frame.value);
      if (frame.type === "action") applyActionSnapshot(frame.value || {});
      cursor += 1;
    }
    if (cursor < frames.length) {
      replayAnimationFrame = window.requestAnimationFrame(renderFrame);
    } else {
      finishReplay();
    }
  };
  replayAnimationFrame = window.requestAnimationFrame(renderFrame);
}

function stopRecording() {
  if (activeRecording === null) return;
  loadingModel = false;
  pauseStreaming();
  const finalFrame = activeRecording.frames.at(-1);
  const recording = {
    version: 2,
    recorded_at: activeRecording.recorded_at,
    duration_ms: finalFrame?.at || 0,
    frames: activeRecording.frames,
  };
  activeRecording = null;
  savedRecording = recording;
  setRecordButton(false);
  updateReplayButton();
  try {
    window.localStorage.setItem(RECORDING_STORAGE_KEY, JSON.stringify(recording));
  } catch (error) {
    console.warn("Could not persist the recording in this browser.", error);
  }
}

async function startRecording() {
  if (activeRecording !== null) {
    stopRecording();
    return;
  }
  stopReplay({ restore: false });
  await reset();
  activeRecording = {
    recorded_at: new Date().toISOString(),
    frames: [],
    lastInputKey: null,
    lastActionKey: null,
  };
  recordingStartedAt = performance.now();
  setRecordButton(true);
  updateReplayButton();
  captureInputFrame();
  captureActionFrame();
  setView("current");
  if (modelStatus === "ready") await beginStreaming();
  else void waitForModel();
  elements.text.focus();
}

async function beginStreaming() {
  const needsSessionPreparation = renderedHistory.length === 0;
  running = true;
  document.body.dataset.running = "true";
  clearError();
  setDecideActivity(true);
  if (needsSessionPreparation) {
    loadingModel = true;
    setRunButton("Preparing Qwen", { loading: true });
    showSessionPreparing();
    const prepared = await tick();
    loadingModel = false;
    if (!prepared || !running) {
      running = false;
      delete document.body.dataset.running;
      setDecideActivity(false);
      setRunButton("Start");
      return;
    }
  } else {
    void tick();
  }
  setRunButton("Pause");
  setRuntimeState("Streaming", "live");
  timer = window.setInterval(() => void tick(), TICK_MS);
}

async function waitForModel() {
  loadingModel = true;
  setRunButton("Connecting to model", { loading: true });
  showModelConnecting();
  try {
    while (loadingModel) {
      const health = await request("/health");
      applyModelHealth(health);
      if (!loadingModel) return;
      if (modelStatus === "ready") {
        loadingModel = false;
        if (!renderedHistory.length) {
          renderCurrentAction([], 0);
          captureActionFrame();
        }
        await beginStreaming();
        return;
      }
      if (modelStatus === "unavailable") {
        loadingModel = false;
        setRunButton("Start");
        showNoModelAvailable(health.model_message);
        return;
      }
      await wait(350);
    }
  } catch (error) {
    loadingModel = false;
    setRunButton("Start");
    showError(error);
  }
}

async function tick() {
  if (!running || !sessionId || requestInFlight) return false;
  // Empty text is still meaningful: a quiet or cleared input must reach the
  // model on every foreground tick while the session is running.
  requestInFlight = true;
  if (elements.paneStatus) {
    elements.paneStatus.textContent = "processing\u2026";
    elements.paneStatus.dataset.state = "busy";
  }
  pulse();
  elements.decidePulse.classList.add("is-busy");
  setRuntimeState("Deciding", "live");
  try {
    const state = Date.now() - lastInputAt < ACTIVE_WINDOW_MS ? "active" : "idle";
    const data = await request(`/api/sessions/${sessionId}/tick`, {
      method: "POST",
      body: JSON.stringify({ text: elements.text.value, state }),
    });
    clearError();
    renderDecisionLatency(data.turn.decision_ms);
    renderSession(data.session);
    if (isModelUnavailable(data.turn.action)) {
      running = false;
      delete document.body.dataset.running;
      window.clearInterval(timer);
      timer = null;
      elements.run.textContent = "Start";
      showNoModelAvailable();
      return false;
    }
    setRuntimeState(data.session.pending_searches ? "Searching" : "Streaming", "live");
    if (data.session.pending_searches > 0) {
      schedulePendingRefresh();
    }
    return true;
  } catch (error) {
    showError(error);
    return false;
  } finally {
    requestInFlight = false;
    elements.decidePulse.classList.remove("is-busy");
    if (elements.paneStatus && elements.paneStatus.dataset.state === "busy") {
      elements.paneStatus.textContent = "";
    }
  }
}

async function start() {
  if (document.body.dataset.playback === "true") stopReplay();
  if (loadingModel) return;
  if (running) {
    pauseStreaming();
    return;
  }
  await ensureSessionInstruction({ preserveText: true });
  if (modelStatus !== "ready") {
    void waitForModel();
    return;
  }
  await beginStreaming();
}

async function reset() {
  if (!sessionId) return;
  if (activeRecording !== null) stopRecording();
  stopReplay({ restore: false });
  running = false;
  delete document.body.dataset.running;
  loadingModel = false;
  dismissActionStage({ resetKey: true });
  stopTranslationAnimation();
  translationHydrated = false;
  setDecideActivity(false);
  clearError();
  if (timer !== null) {
    window.clearInterval(timer);
    timer = null;
  }
  if (pendingPollTimer !== null) {
    window.clearTimeout(pendingPollTimer);
    pendingPollTimer = null;
  }
  const replacedSession = await ensureSessionInstruction({ preserveText: false });
  const session = replacedSession
    ? null
    : await request(`/api/sessions/${sessionId}/reset`, { method: "POST" });
  lastRenderedIndex = 0;
  lastInputAt = 0;
  elements.text.value = "";
  renderHighlights("", []);
  workThreads.clear();
  retiredJobs.clear();
  if (elements.workSurface) {
    elements.workSurface.replaceChildren();
    syncWorkSurfaceCount();
  }
  setRunButton("Start");
  if (session) renderSession(session);
  setRuntimeState("Ready");
}

function showError(error) {
  console.error(error);
  setRuntimeState("Runtime error", "error");
  elements.modeLabel.textContent = error.message;
  elements.errorLine.textContent = error.message;
  elements.errorLine.hidden = false;
}

function clearError() {
  elements.errorLine.textContent = "";
  elements.errorLine.hidden = true;
}

async function initialize() {
  try {
    setRecordButton(false);
    loadSavedRecording();
    const urlParameters = new URLSearchParams(window.location.search);
    const requestedSession = urlParameters.get("session");
    const initialInstruction = urlParameters.get("instruction") || "";
    const health = await request("/health");
    applyModelHealth(health);
    const session = await (requestedSession
      ? request(`/api/sessions/${encodeURIComponent(requestedSession)}`)
      : request("/api/sessions", {
        method: "POST",
        body: JSON.stringify({
          instruction: actionSchema === "g1" ? "" : initialInstruction,
          mode: actionSchema === "g1" ? "g1" : SESSION_MODE,
        }),
      }));
    sessionId = session.id;
    sessionInstruction = session.instruction || "";
    elements.instruction.value = sessionInstruction;
    const latestUserEvent = [...(session.history || [])]
      .reverse()
      .find((turn) => turn.event.source === "user");
    if (latestUserEvent) elements.text.value = latestUserEvent.event.content;
    const initialText = urlParameters.get("text");
    if (!latestUserEvent && initialText !== null) {
      elements.text.value = initialText;
      lastInputAt = Date.now();
    }
    renderSession(session);
    if (health.model_status === "unavailable") {
      showNoModelAvailable(health.model_message);
    } else {
      setRuntimeState("Ready");
    }
    if (urlParameters.get("autostart") === "1") void start().catch(showError);
  } catch (error) {
    showError(error);
  }
}

document.querySelector("#card-reset")?.addEventListener("click", () => elements.reset.click());

elements.text.addEventListener("input", () => {
  lastInputAt = Date.now();
  renderHighlights(elements.text.value, latestSessionHighlights, latestSessionSuggestions);
  captureInputFrame();
});
elements.run.addEventListener("click", () => start().catch(showError));
elements.reset.addEventListener("click", () => reset().catch(showError));
elements.record.addEventListener("click", () => startRecording().catch(showError));
elements.replay.addEventListener("click", startReplay);
elements.exportRecording.addEventListener("click", exportRecording);
elements.importRecording.addEventListener("click", () => elements.importInput.click());
elements.importInput.addEventListener("change", () => {
  void importRecording(elements.importInput.files?.[0]);
  elements.importInput.value = "";
});
elements.readableHistory.addEventListener("click", () => setHistoryFormat("readable"));
elements.modelInputButton.addEventListener("click", () => setHistoryFormat("model"));
elements.streamViewButton?.addEventListener("click", () => setActionPanelView("stream"));
elements.modelActionsViewButton?.addEventListener("click", () => setActionPanelView("model-actions"));
elements.copyHistory.addEventListener("click", () => copyHistory());
document.addEventListener("keydown", handleOperatorShortcut);
setView(new URLSearchParams(window.location.search).get("view") === "details" ? "details" : "current");
setActionPanelView("stream");
initialize();
