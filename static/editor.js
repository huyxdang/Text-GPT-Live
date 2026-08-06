const TICK_MS = 650;
const POLL_MS = 500;
const ACTIVE_WINDOW_MS = 1300;

const elements = {
  talk: document.querySelector("#talk-input"),
  mic: document.querySelector("#mic-button"),
  run: document.querySelector("#run-button"),
  reset: document.querySelector("#reset-button"),
  undo: document.querySelector("#undo-button"),
  documentCanvas: document.querySelector("#document-canvas"),
  writerStatus: document.querySelector("#writer-status"),
  writerPulse: document.querySelector("#writer-pulse"),
  decidePulse: document.querySelector("#decide-pulse"),
  currentAction: document.querySelector("#current-action"),
  currentActionTitle: document.querySelector("#current-action-title"),
  currentActionValue: document.querySelector("#current-action-value"),
  currentActionRaw: document.querySelector("#current-action-raw"),
  errorLine: document.querySelector("#error-line"),
};

let sessionId = null;
let running = false;
let tickTimer = null;
let pollTimer = null;
let requestInFlight = false;
let lastInputAt = 0;
let lastDocumentRevision = 0;
let completedTicks = 0;

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}

function showError(error) {
  console.error(error);
  elements.errorLine.textContent = error.message;
  elements.errorLine.hidden = false;
}

function clearError() {
  elements.errorLine.hidden = true;
  elements.errorLine.textContent = "";
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

function renderDocument(session) {
  const doc = session.document || { text: "", status: "idle", undo_available: false, revision: 0 };
  const text = doc.text || "";
  const fragment = document.createDocumentFragment();
  const range = session.proposal
    ? quoteRange(text, session.proposal.quote, session.proposal.occurrence)
    : null;

  if (range) {
    fragment.append(document.createTextNode(text.slice(0, range.start)));
    const mark = document.createElement("mark");
    mark.className = "proposal";
    mark.textContent = text.slice(range.start, range.end);
    fragment.append(mark);
    fragment.append(document.createTextNode(text.slice(range.end)));
  } else {
    fragment.append(document.createTextNode(text));
  }
  elements.documentCanvas.replaceChildren(fragment);
  if (doc.revision !== lastDocumentRevision) {
    elements.documentCanvas.classList.remove("revised-flash");
    void elements.documentCanvas.offsetWidth;
    elements.documentCanvas.classList.add("revised-flash");
    lastDocumentRevision = doc.revision;
    elements.documentCanvas.scrollTop = elements.documentCanvas.scrollHeight;
  }

  elements.writerStatus.textContent = doc.status;
  elements.writerStatus.dataset.status = doc.status;
  elements.writerPulse.hidden = doc.status !== "writing";
  elements.undo.disabled = !doc.undo_available;
}

function actionDisplay(turn, session) {
  const action = turn.action;
  const raw = action.raw_output && action.raw_output.startsWith("<action>")
    ? action.raw_output
    : "";
  if (!action.valid) {
    return { state: "invalid", title: "Fallback", value: action.diagnostic || "Invalid action", raw };
  }
  if (action.kind === "respond") {
    return { state: "respond", title: "Asking", value: action.message, raw };
  }
  if (action.kind === "tool" && action.tool_name === "writer") {
    const operation = action.arguments?.operation;
    if (operation === "pause") return { state: "search", title: "Paused", value: "Stopped writing.", raw };
    if (operation === "write") return { state: "search", title: "Writing", value: action.arguments.instruction, raw };
    if (operation === "resume") return { state: "search", title: "Resuming", value: "Continuing the draft.", raw };
    if (operation === "revise") return { state: "highlight", title: "Revising", value: action.arguments.instruction, raw };
  }
  if (action.kind === "tool" && action.tool_name === "ui") {
    const operation = action.arguments?.operation;
    const quote = action.arguments?.target?.quote || "";
    if (operation === "underline") {
      return { state: "highlight", title: "Proposing", value: `“${quote}”`, raw };
    }
    return { state: "highlight", title: "Highlighting", value: `“${quote}”`, raw };
  }
  if (action.kind === "tool" && action.tool_name === "web_search") {
    return { state: "search", title: "Searching", value: action.arguments?.query || "", raw };
  }
  if (session.pending_searches > 0 || (session.document || {}).status === "writing") {
    return null; // keep the previous meaningful display while work continues
  }
  return { state: "idle", title: "", value: "Model is silent", raw: "" };
}

function renderCurrentAction(session) {
  const history = session.history || [];
  // Show the most recent *meaningful* decision; idle ticks should not erase
  // the confirmation question the user is meant to be reading.
  for (let index = history.length - 1; index >= 0; index -= 1) {
    const display = actionDisplay(history[index], session);
    if (display === null) continue;
    if (display.state === "idle" && index !== history.length - 1) continue;
    if (display.state === "idle") {
      const lastMeaningful = [...history].reverse().find(
        (turn) => turn.action.valid && turn.action.kind !== "idle"
      );
      if (lastMeaningful) continue;
    }
    elements.currentAction.dataset.action = display.state;
    elements.currentActionTitle.textContent = display.title;
    elements.currentActionValue.textContent = display.value;
    elements.currentActionRaw.textContent = display.raw;
    elements.currentActionRaw.hidden = !display.raw;
    return;
  }
  elements.currentAction.dataset.action = "idle";
  elements.currentActionTitle.textContent = "";
  elements.currentActionValue.textContent = "Model is silent";
  elements.currentActionRaw.hidden = true;
}

function renderSession(session) {
  renderDocument(session);
  renderCurrentAction(session);
}

async function refresh() {
  if (!sessionId) return;
  const session = await request(`/api/sessions/${sessionId}`);
  renderSession(session);
}

function schedulePolling() {
  if (pollTimer !== null) return;
  pollTimer = window.setInterval(() => {
    refresh().catch(showError);
  }, POLL_MS);
}

function stopPolling() {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function tick() {
  if (!running || !sessionId || requestInFlight) return;
  // Empty text is still meaningful: a quiet or cleared input must reach the
  // model on every foreground tick while the session is running.
  requestInFlight = true;
  elements.decidePulse.classList.remove("tick");
  void elements.decidePulse.offsetWidth;
  elements.decidePulse.classList.add("tick", "is-busy");
  try {
    const state = Date.now() - lastInputAt < ACTIVE_WINDOW_MS ? "active" : "idle";
    const data = await request(`/api/sessions/${sessionId}/tick`, {
      method: "POST",
      body: JSON.stringify({ text: elements.talk.value, state }),
    });
    clearError();
    renderSession(data.session);
  } catch (error) {
    showError(error);
  } finally {
    requestInFlight = false;
    completedTicks += 1;
    elements.decidePulse.classList.remove("is-busy");
  }
}

function start() {
  if (running) {
    running = false;
    window.clearInterval(tickTimer);
    tickTimer = null;
    stopPolling();
    elements.run.textContent = "Resume";
    elements.decidePulse.hidden = true;
    return;
  }
  running = true;
  elements.run.textContent = "Pause";
  elements.decidePulse.hidden = false;
  tick();
  tickTimer = window.setInterval(tick, TICK_MS);
  schedulePolling();
  elements.talk.focus();
}

async function reset() {
  if (!sessionId) return;
  running = false;
  if (tickTimer !== null) window.clearInterval(tickTimer);
  tickTimer = null;
  stopPolling();
  elements.run.textContent = "Start";
  elements.decidePulse.hidden = true;
  elements.talk.value = "";
  lastDocumentRevision = 0;
  const session = await request(`/api/sessions/${sessionId}/reset`, { method: "POST" });
  renderSession(session);
  clearError();
}

async function undo() {
  if (!sessionId) return;
  const session = await request(`/api/sessions/${sessionId}/undo`, { method: "POST" });
  renderSession(session);
}

function setupMicrophone() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    elements.mic.disabled = true;
    elements.mic.title = "Speech recognition is not available in this browser.";
    return;
  }
  const recognition = new Recognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";
  let listening = false;

  recognition.addEventListener("result", (event) => {
    let transcript = "";
    for (const result of event.results) transcript += result[0].transcript;
    elements.talk.value = transcript.trim();
    lastInputAt = Date.now();
  });
  recognition.addEventListener("end", () => {
    if (listening) recognition.start(); // keep listening until toggled off
  });
  recognition.addEventListener("error", (event) => {
    if (event.error !== "no-speech") showError(new Error(`Microphone: ${event.error}`));
  });

  elements.mic.addEventListener("click", () => {
    listening = !listening;
    elements.mic.setAttribute("aria-pressed", String(listening));
    if (listening) {
      recognition.start();
      if (!running) start();
    } else {
      recognition.stop();
    }
  });
}

async function initialize() {
  try {
    const session = await request("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ mode: "editor" }),
    });
    sessionId = session.id;
    renderSession(session);
  } catch (error) {
    showError(error);
  }
}

elements.talk.addEventListener("input", () => {
  lastInputAt = Date.now();
});
elements.talk.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  void submitUtterance();
});

async function submitUtterance() {
  // Enter = "I'm done talking — decide now", then clear for the next utterance.
  if (!elements.talk.value.trim()) {
    if (!running) start();
    return;
  }
  lastInputAt = 0; // the utterance is complete: every tick from here is state=idle
  const baseline = completedTicks;
  if (!running) start();
  // Wait until one tick has actually carried the settled utterance to the model.
  const deadline = Date.now() + 8000;
  while (completedTicks === baseline && Date.now() < deadline) {
    if (!requestInFlight && running) void tick();
    await new Promise((resolve) => window.setTimeout(resolve, 80));
  }
  elements.talk.value = "";
}
elements.run.addEventListener("click", () => start());
elements.reset.addEventListener("click", () => reset().catch(showError));
elements.undo.addEventListener("click", () => undo().catch(showError));
setupMicrophone();
initialize();
