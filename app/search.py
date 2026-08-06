from __future__ import annotations

import asyncio
from typing import Any, Protocol


class SearchProvider(Protocol):
    mode: str

    async def search(self, query: str) -> dict[str, Any]: ...


class DDGSSearchProvider:
    mode = "ddgs"

    def __init__(self, max_results: int = 5, timeout_seconds: float = 15.0) -> None:
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str) -> dict[str, Any]:
        return await asyncio.wait_for(
            asyncio.to_thread(self._search_sync, query),
            timeout=self.timeout_seconds,
        )

    def _search_sync(self, query: str) -> dict[str, Any]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise RuntimeError(
                "The ddgs package is required for SEARCH_MODE=ddgs. Install requirements.txt."
            ) from exc

        raw_results = list(DDGS().text(query, max_results=self.max_results))
        results = [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("href", item.get("url", ""))),
                "snippet": str(item.get("body", item.get("snippet", ""))),
            }
            for item in raw_results
        ]
        return {"query": query, "results": results}


class DemoSearchProvider:
    mode = "demo"

    async def search(self, query: str) -> dict[str, Any]:
        await asyncio.sleep(0.35)
        return {
            "query": query,
            "results": [
                {
                    "title": "Demo search result",
                    "url": "",
                    "snippet": (
                        "This is deterministic demo data. Set SEARCH_MODE=ddgs for live web search."
                    ),
                }
            ],
        }
