"""Background writer providers: the delegated content engine for editor sessions.

The interaction model never writes prose. It pauses, proposes, and routes; the
providers here produce the document text (T1's A3 capability) and scoped
rewrites. Both providers speak the same interface so the runtime and tests are
provider-agnostic: `stream_sentences` yields committed sentences, `revise`
returns replacement text for exactly one confirmed span.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import AsyncIterator, Protocol


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")


def split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Split leading complete sentences off the buffer; return (sentences, rest)."""
    sentences: list[str] = []
    rest = buffer
    while True:
        match = _SENTENCE_BOUNDARY.search(rest)
        if not match:
            return sentences, rest
        sentence = rest[: match.end()].strip()
        if sentence:
            sentences.append(sentence)
        rest = rest[match.end():]


WRITE_SYSTEM = (
    "You are the writing engine inside a collaborative document editor. Write the "
    "requested passage as plain prose: no headings, no lists, no meta commentary, no "
    "quotation of these instructions. Write multi-paragraph prose with short, clear "
    "sentences. If the document already contains text, continue seamlessly from it "
    "without repeating it. Honor every stated preference."
)
REVISE_SYSTEM = (
    "You revise exactly one span inside a collaborative document. Rewrite the span "
    "according to the instruction while preserving its meaning, tense, and narrative "
    "role, and keep roughly the same length. Return ONLY the replacement text for the "
    "span: no quotes, no preamble, no commentary."
)


class WriterProvider(Protocol):
    mode: str

    def stream_sentences(
        self,
        *,
        task: str,
        document_text: str,
        preferences: list[str],
    ) -> AsyncIterator[str]: ...

    async def revise(
        self,
        *,
        task: str,
        document_text: str,
        span: str,
        instruction: str,
        preferences: list[str],
    ) -> str: ...


class DemoWriterProvider:
    """Deterministic offline writer for tests and UI work, like DemoSearchProvider."""

    mode = "demo"

    def __init__(self, *, sentence_delay_seconds: float = 0.9, sentence_count: int = 10) -> None:
        self.sentence_delay_seconds = sentence_delay_seconds
        self.sentence_count = sentence_count

    def _topic(self, task: str) -> str:
        words = [word.strip(" ,.!?\"'") for word in task.split()]
        content = [word for word in words if len(word) > 3 and word.isalpha()]
        return content[-1].lower() if content else "the subject"

    def stream_sentences(
        self,
        *,
        task: str,
        document_text: str,
        preferences: list[str],
    ) -> AsyncIterator[str]:
        topic = self._topic(task)
        start = len(split_complete_sentences(document_text + " ")[0])
        tone = " calmly" if any("less dramatic" in p for p in preferences) else " suddenly"

        async def generate() -> AsyncIterator[str]:
            for index in range(start, start + self.sentence_count):
                await asyncio.sleep(self.sentence_delay_seconds)
                yield (
                    f"Sentence {index + 1} of the demo passage about {topic}"
                    f"{tone} moves the story forward."
                )

        return generate()

    async def revise(
        self,
        *,
        task: str,
        document_text: str,
        span: str,
        instruction: str,
        preferences: list[str],
    ) -> str:
        del task, document_text, preferences
        base = span.rstrip(".!?…")
        return f"{base}, quietly revised ({instruction})."


class OpenAIWriterProvider:
    """Streams story prose from the OpenAI chat completions API via SSE.

    Same contract and prompts as the Anthropic provider; used when the deployer
    supplies an OPENAI_API_KEY instead of an Anthropic key.
    """

    mode = "openai"

    def __init__(self, *, model: str | None = None, max_tokens: int = 2048) -> None:
        import httpx

        self.model = model or os.getenv("WRITER_MODEL", "gpt-4o-mini")
        self.max_tokens = max_tokens
        api_key = os.getenv("OPENAI_API_KEY", "").strip().strip("'\"")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for WRITER_MODE=openai.")
        self._client = httpx.AsyncClient(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )

    def _write_prompt(self, task: str, document_text: str, preferences: list[str]) -> str:
        parts = [f"Writing task: {task}"]
        if preferences:
            parts.append("Standing preferences from the user: " + "; ".join(preferences))
        if document_text.strip():
            parts.append(
                "The document so far (continue from its end, do not repeat it):\n" + document_text
            )
        else:
            parts.append("The document is currently empty. Begin the passage.")
        return "\n\n".join(parts)

    def stream_sentences(
        self,
        *,
        task: str,
        document_text: str,
        preferences: list[str],
    ) -> AsyncIterator[str]:
        prompt = self._write_prompt(task, document_text, preferences)
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "stream": True,
            "messages": [
                {"role": "system", "content": WRITE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        }

        async def generate() -> AsyncIterator[str]:
            import json as json_module

            buffer = ""
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode(errors="replace")
                    raise RuntimeError(f"OpenAI writer error {response.status_code}: {body[:200]}")
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):].strip()
                    if data == "[DONE]":
                        break
                    chunk = json_module.loads(data)
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    text = delta.get("content") or ""
                    if not text:
                        continue
                    buffer += text
                    sentences, buffer = split_complete_sentences(buffer)
                    for sentence in sentences:
                        yield sentence
            tail = buffer.strip()
            if tail:
                yield tail

        return generate()

    async def revise(
        self,
        *,
        task: str,
        document_text: str,
        span: str,
        instruction: str,
        preferences: list[str],
    ) -> str:
        parts = [
            f"The document belongs to this writing task: {task}",
            f"Full document for context:\n{document_text}",
            f"Span to rewrite (change nothing else):\n{span}",
            f"Revision instruction: {instruction}",
        ]
        if preferences:
            parts.append("Standing preferences: " + "; ".join(preferences))
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "max_tokens": 512,
                "messages": [
                    {"role": "system", "content": REVISE_SYSTEM},
                    {"role": "user", "content": "\n\n".join(parts)},
                ],
            },
        )
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI writer error {response.status_code}: {response.text[:200]}")
        data = response.json()
        replacement = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        replacement = str(replacement).strip()
        if not replacement:
            raise RuntimeError("The writer returned an empty revision.")
        return replacement

    async def aclose(self) -> None:
        await self._client.aclose()


class AnthropicWriterProvider:
    """Streams story prose from the Anthropic API, one committed sentence at a time."""

    mode = "anthropic"

    def __init__(self, *, model: str | None = None, max_tokens: int = 4096) -> None:
        import anthropic

        self.model = model or os.getenv("WRITER_MODEL", "claude-opus-4-8")
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic()

    def _write_prompt(self, task: str, document_text: str, preferences: list[str]) -> str:
        parts = [f"Writing task: {task}"]
        if preferences:
            parts.append("Standing preferences from the user: " + "; ".join(preferences))
        if document_text.strip():
            parts.append(
                "The document so far (continue from its end, do not repeat it):\n"
                + document_text
            )
        else:
            parts.append("The document is currently empty. Begin the passage.")
        return "\n\n".join(parts)

    def stream_sentences(
        self,
        *,
        task: str,
        document_text: str,
        preferences: list[str],
    ) -> AsyncIterator[str]:
        prompt = self._write_prompt(task, document_text, preferences)

        async def generate() -> AsyncIterator[str]:
            buffer = ""
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=WRITE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    buffer += text
                    sentences, buffer = split_complete_sentences(buffer)
                    for sentence in sentences:
                        yield sentence
            tail = buffer.strip()
            if tail:
                yield tail

        return generate()

    async def revise(
        self,
        *,
        task: str,
        document_text: str,
        span: str,
        instruction: str,
        preferences: list[str],
    ) -> str:
        parts = [
            f"The document belongs to this writing task: {task}",
            f"Full document for context:\n{document_text}",
            f"Span to rewrite (change nothing else):\n{span}",
            f"Revision instruction: {instruction}",
        ]
        if preferences:
            parts.append("Standing preferences: " + "; ".join(preferences))
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=REVISE_SYSTEM,
            messages=[{"role": "user", "content": "\n\n".join(parts)}],
        )
        replacement = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not replacement:
            raise RuntimeError("The writer returned an empty revision.")
        return replacement

    async def aclose(self) -> None:
        await self._client.close()
