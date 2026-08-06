"""A super simple chat UI for talking to Qwen3-0.6B via llama-server.

Two terminals:

    # 1. start the model server (OpenAI-compatible endpoint on :8001)
    llama-server -hf ggml-org/Qwen3-0.6B-GGUF:Q4_0 --port 8001 --n-gpu-layers all

    # 2. start this chat UI, then open http://127.0.0.1:8010
    python -m tests.qwen_ui

Environment overrides: QWEN_BASE_URL, QWEN_MODEL, QWEN_UI_PORT.
"""

from __future__ import annotations

import os

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

BASE_URL = os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
MODEL = os.getenv("QWEN_MODEL", "Qwen3-0.6B")
UI_PORT = int(os.getenv("QWEN_UI_PORT", "8010"))

app = FastAPI(title="Qwen 0.6B chat")


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]


@app.post("/chat")
async def chat(request: ChatRequest) -> JSONResponse:
    payload = {
        "model": MODEL,
        "messages": request.messages,
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 512,
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{BASE_URL}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        reply = data["choices"][0]["message"]["content"]
        return JSONResponse({"reply": reply})
    except httpx.HTTPError as exc:
        return JSONResponse(
            {"error": f"Could not reach llama-server at {BASE_URL}: {exc}"},
            status_code=502,
        )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Qwen 0.6B chat</title>
<style>
  :root { color-scheme: light dark; }
  body { margin: 0; font: 15px/1.5 system-ui, sans-serif;
         display: flex; flex-direction: column; height: 100vh; }
  header { padding: 10px 16px; border-bottom: 1px solid #8883;
           font-weight: 600; }
  header small { font-weight: 400; opacity: .6; }
  #log { flex: 1; overflow-y: auto; padding: 16px;
         display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 70ch; padding: 8px 12px; border-radius: 12px;
         white-space: pre-wrap; word-wrap: break-word; }
  .user { align-self: flex-end; background: #2f6fed; color: #fff; }
  .assistant { align-self: flex-start; background: #8882; }
  .error { align-self: center; color: #d33; font-size: 13px; }
  form { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid #8883; }
  textarea { flex: 1; resize: none; padding: 8px; border-radius: 8px;
             border: 1px solid #8886; font: inherit; background: transparent;
             color: inherit; }
  button { padding: 0 18px; border: 0; border-radius: 8px; background: #2f6fed;
           color: #fff; font: inherit; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
</style>
</head>
<body>
<header>Qwen 0.6B chat <small>__CFG__</small></header>
<div id="log"></div>
<form id="form">
  <textarea id="input" rows="2" placeholder="Type a message, Enter to send…"></textarea>
  <button id="send" type="submit">Send</button>
</form>
<script>
const log = document.getElementById("log");
const input = document.getElementById("input");
const send = document.getElementById("send");
const form = document.getElementById("form");
const messages = [];

function bubble(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.textContent = text;
  log.append(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

async function sendMessage(text) {
  messages.push({ role: "user", content: text });
  bubble("user", text);
  send.disabled = true;
  const thinking = bubble("assistant", "…");
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.status);
    thinking.textContent = data.reply;
    messages.push({ role: "assistant", content: data.reply });
  } catch (err) {
    thinking.remove();
    bubble("error", String(err.message || err));
  } finally {
    send.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  sendMessage(text);
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
input.focus();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return PAGE.replace("__CFG__", f"· {MODEL} via {BASE_URL}")


def main() -> None:
    import uvicorn

    print(f"Chat UI: http://127.0.0.1:{UI_PORT}  (model {MODEL} via {BASE_URL})")
    uvicorn.run(app, host="127.0.0.1", port=UI_PORT)


if __name__ == "__main__":
    main()
