from __future__ import annotations

import asyncio
import copy
import json
import os
import re
from collections.abc import AsyncIterator
from typing import Any, Protocol


# The generative-UI surface is a constrained declarative spec, not code. The
# harness validates every spec against this allowlist before rendering it.
# The vocabulary is deliberately broad enough that the model does not have to
# force-fit content: a chronology gets a timeline rather than a bar chart, and
# a non-numeric fact gets a fact row rather than a stat card.
UI_COMPONENT_TYPES = (
    "hero",
    "stat_card",
    "chart",
    "timeline",
    "fact",
    "callout",
    "comparison",
    "list",
    "sources",
)
UI_CHART_KINDS = ("bar", "line")

# Layout and color autonomy: the model expresses intent from closed
# vocabularies; the renderer still owns every pixel.
UI_ACCENTS = ("blue", "violet", "green", "amber", "red", "slate")
UI_TONES = ("neutral", "positive", "negative")
UI_SPANS = ("full", "half")
UI_EMPHASIS = ("lead", "normal")

MAX_COMPONENTS = 12
MAX_STAT_CARDS = 6
MAX_SERIES_POINTS = 12
MAX_LIST_ITEMS = 8
MAX_SOURCE_LINKS = 8
MAX_TIMELINE_EVENTS = 8
MAX_FACT_ROWS = 6
MAX_COMPARISON_ITEMS = 6
MAX_CALLOUT_CHARS = 200
MAX_TAGLINE_CHARS = 160

# How many of each component one panel may contain.
UI_TYPE_CAPS: dict[str, int] = {
    "hero": 1,
    "chart": 1,
    "timeline": 1,
    "fact": 1,
    "comparison": 1,
    "sources": 1,
    "callout": 2,
    "list": 2,
    "stat_card": MAX_STAT_CARDS,
}

# A panel that ships "$X.XB" is worse than a panel that omits the number, so
# placeholder-shaped values are rejected rather than rendered.
_PLACEHOLDER_VALUE = re.compile(
    r"(X\.X|XX+|\bTBD\b|\bN/?A\b|\bunknown\b|\bplaceholder\b|\bpending\b|^[\s\-–—.]*$)",
    re.IGNORECASE,
)


class UIGenProvider(Protocol):
    mode: str

    async def generate(self, request: str) -> dict[str, Any]: ...


class ProgressiveUIGenProvider(UIGenProvider, Protocol):
    """A UI provider that publishes only components it has actually produced."""

    async def stream_build(self, request: str) -> AsyncIterator[dict[str, Any]]: ...


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_placeholder(value: Any) -> bool:
    return bool(_PLACEHOLDER_VALUE.search(str(value)))


def _label_value_rows(rows: Any, limit: int) -> bool:
    return (
        isinstance(rows, list)
        and 2 <= len(rows) <= limit
        and all(
            isinstance(row, dict)
            and _non_empty_str(row.get("label"))
            and _non_empty_str(row.get("value"))
            for row in rows
        )
    )


def _validate_presentation(component: dict[str, Any], label: str, errors: list[str]) -> None:
    """span/tone/emphasis are optional on every component."""
    span = component.get("span")
    if span is not None and span not in UI_SPANS:
        errors.append(f"{label} span must be one of {UI_SPANS}.")
    tone = component.get("tone")
    if tone is not None and tone not in UI_TONES:
        errors.append(f"{label} tone must be one of {UI_TONES}.")
    emphasis = component.get("emphasis")
    if emphasis is not None and emphasis not in UI_EMPHASIS:
        errors.append(f"{label} emphasis must be one of {UI_EMPHASIS}.")


def _validate_chart(component: dict[str, Any], label: str, errors: list[str]) -> None:
    if component.get("chart") not in UI_CHART_KINDS:
        errors.append(f"{label} chart kind must be one of {UI_CHART_KINDS}.")
    series = component.get("series")
    well_formed = (
        isinstance(series, list)
        and 1 <= len(series) <= MAX_SERIES_POINTS
        and all(
            isinstance(point, dict)
            and _non_empty_str(point.get("label"))
            and isinstance(point.get("value"), (int, float))
            and not isinstance(point.get("value"), bool)
            for point in series
        )
    )
    if not well_formed:
        errors.append(
            f'{label} chart requires 1-{MAX_SERIES_POINTS} series points of '
            '{"label": str, "value": number}.'
        )
        return

    values = [float(point["value"]) for point in series]
    if all(value == 0 for value in values):
        errors.append(f"{label} chart series is all zeros; supply real figures.")
    if len(values) >= 3:
        if all(1900 <= value <= 2100 for value in values):
            errors.append(
                f"{label} chart series looks like calendar years; chart a real metric instead."
            )
        low, high = min(values), max(values)
        if low > 0 and high / low < 1.05:
            errors.append(
                f"{label} chart series values are nearly identical; "
                "pick a metric with meaningful variation."
            )


def _validate_component_body(component: dict[str, Any], kind: str, label: str, errors: list[str]) -> None:
    if kind == "hero":
        if not _non_empty_str(component.get("name")):
            errors.append(f'{label} hero requires a non-empty "name".')
        if not _non_empty_str(component.get("tagline")):
            errors.append(f'{label} hero requires a non-empty "tagline".')
        elif len(str(component["tagline"])) > MAX_TAGLINE_CHARS:
            errors.append(f"{label} hero tagline exceeds {MAX_TAGLINE_CHARS} characters.")
    elif kind == "stat_card":
        if not _non_empty_str(component.get("label")) or not _non_empty_str(component.get("value")):
            errors.append(f'{label} stat_card requires string "label" and "value".')
        elif _is_placeholder(component["value"]):
            errors.append(
                f"{label} stat_card value is a placeholder; "
                "use a figure you know or choose another component."
            )
    elif kind == "chart":
        _validate_chart(component, label, errors)
    elif kind == "timeline":
        events = component.get("events")
        if not (
            isinstance(events, list)
            and 2 <= len(events) <= MAX_TIMELINE_EVENTS
            and all(
                isinstance(event, dict)
                and _non_empty_str(event.get("date"))
                and _non_empty_str(event.get("text"))
                for event in events
            )
        ):
            errors.append(
                f'{label} timeline requires 2-{MAX_TIMELINE_EVENTS} '
                '{"date": str, "text": str} events.'
            )
    elif kind == "fact":
        if not _label_value_rows(component.get("facts"), MAX_FACT_ROWS):
            errors.append(
                f'{label} fact requires 2-{MAX_FACT_ROWS} {{"label": str, "value": str}} rows.'
            )
        elif any(_is_placeholder(row["value"]) for row in component["facts"]):
            errors.append(f"{label} fact contains a placeholder value; use values you know.")
    elif kind == "callout":
        if not _non_empty_str(component.get("text")):
            errors.append(f'{label} callout requires non-empty "text".')
        elif len(str(component["text"])) > MAX_CALLOUT_CHARS:
            errors.append(f"{label} callout exceeds {MAX_CALLOUT_CHARS} characters.")
    elif kind == "comparison":
        columns = component.get("columns")
        if not (
            isinstance(columns, list)
            and len(columns) == 2
            and all(
                isinstance(column, dict)
                and _non_empty_str(column.get("title"))
                and isinstance(column.get("items"), list)
                and 1 <= len(column["items"]) <= MAX_COMPARISON_ITEMS
                and all(_non_empty_str(item) for item in column["items"])
                for column in columns
            )
        ):
            errors.append(
                f"{label} comparison requires exactly 2 columns of "
                f"{{title, 1-{MAX_COMPARISON_ITEMS} items}}."
            )
    elif kind == "list":
        items = component.get("items")
        if not (
            isinstance(items, list)
            and items
            and len(items) <= MAX_LIST_ITEMS
            and all(_non_empty_str(item) for item in items)
        ):
            errors.append(f"{label} list requires 1-{MAX_LIST_ITEMS} non-empty string items.")
    elif kind == "sources":
        links = component.get("links")
        if not (
            isinstance(links, list)
            and links
            and len(links) <= MAX_SOURCE_LINKS
            and all(
                isinstance(link, dict)
                and _non_empty_str(link.get("title"))
                and isinstance(link.get("url"), str)
                and link["url"].startswith(("http://", "https://"))
                for link in links
            )
        ):
            errors.append(
                f"{label} sources requires 1-{MAX_SOURCE_LINKS} links with a title and an http(s) url."
            )


def validate_ui_spec(spec: Any) -> list[str]:
    """Return a list of problems; an empty list means the spec is renderable."""
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ["Spec must be a JSON object."]
    title = spec.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append('Spec requires a non-empty string "title".')
    accent = spec.get("accent")
    if accent is not None and accent not in UI_ACCENTS:
        errors.append(f"Spec accent must be one of {UI_ACCENTS}.")
    components = spec.get("components")
    if not isinstance(components, list) or not components:
        errors.append('Spec requires a non-empty "components" list.')
        return errors
    if len(components) > MAX_COMPONENTS:
        errors.append(f"Spec has too many components (max {MAX_COMPONENTS}).")

    counts = {name: 0 for name in UI_COMPONENT_TYPES}
    leads = 0
    for position, component in enumerate(components):
        label = f"components[{position}]"
        if not isinstance(component, dict):
            errors.append(f"{label} must be an object.")
            continue
        kind = component.get("type")
        if kind not in UI_COMPONENT_TYPES:
            errors.append(f"{label} type must be one of {UI_COMPONENT_TYPES}.")
            continue
        counts[kind] += 1
        if component.get("emphasis") == "lead":
            leads += 1
        _validate_presentation(component, label, errors)
        _validate_component_body(component, kind, label, errors)

    for kind, cap in UI_TYPE_CAPS.items():
        if counts[kind] > cap:
            errors.append(f"Spec may contain at most {cap} {kind} component(s).")
    if leads > 1:
        errors.append('Spec may mark at most one component as emphasis "lead".')
    return errors


_DEMO_SPEC: dict[str, Any] = {
    "title": "Vietnam — 2025 Economic Brief (demo data)",
    "accent": "green",
    "components": [
        {"type": "stat_card", "label": "GDP growth (2025)", "value": "6.9%", "tone": "positive"},
        {"type": "stat_card", "label": "Inflation (CPI, 2025)", "value": "3.6%"},
        {"type": "stat_card", "label": "Goods exports (2025)", "value": "$405B"},
        {
            "type": "chart",
            "chart": "bar",
            "label": "GDP growth by year (%)",
            "series": [
                {"label": "2021", "value": 2.6},
                {"label": "2022", "value": 8.0},
                {"label": "2023", "value": 5.1},
                {"label": "2024", "value": 7.1},
                {"label": "2025", "value": 6.9},
            ],
        },
        {
            "type": "list",
            "label": "Highlights",
            "items": [
                "Manufacturing and electronics exports led growth.",
                "Foreign direct investment inflows stayed strong.",
                "Services and tourism continued to recover.",
            ],
        },
        {
            "type": "callout",
            "text": "Growth held above 6% while inflation stayed inside the policy ceiling.",
            "tone": "positive",
        },
        {
            "type": "sources",
            "links": [
                {
                    "title": "Demo placeholder — replace with live sources",
                    "url": "https://example.com/vietnam-2025-economy",
                }
            ],
        },
    ],
}


_COMPONENT_SCHEMA_LINES = f"""- {{"type":"hero","name":str,"tagline":str,"image_query":str?}}
   image_query is the Wikipedia article title used to fetch the portrait (e.g. "Elon Musk").
- {{"type":"stat_card","label":str,"value":str,"sub":str?}} — one striking NUMBER with a short
   label; never put plain words in a stat_card.
- {{"type":"chart","chart":"bar"|"line","label":str,"series":[{{"label":str,"value":number}}, ...]}}
   1-{MAX_SERIES_POINTS} points of a real metric; series values must be comparable magnitudes and
   must NEVER be calendar years.
- {{"type":"timeline","label":str?,"events":[{{"date":str,"text":str}}, ...]}} — 2-{MAX_TIMELINE_EVENTS}
   events; the right home for any chronology.
- {{"type":"fact","label":str?,"facts":[{{"label":str,"value":str}}, ...]}} — 2-{MAX_FACT_ROWS} rows of
   short non-numeric facts (role, education, headquarters).
- {{"type":"callout","text":str}} — one takeaway sentence, max {MAX_CALLOUT_CHARS} characters.
- {{"type":"comparison","label":str?,"columns":[{{"title":str,"items":[str, ...]}},{{"title":str,"items":[str, ...]}}]}}
   exactly two columns, 1-{MAX_COMPARISON_ITEMS} items each.
- {{"type":"list","label":str,"items":[str, ...]}} — max {MAX_LIST_ITEMS} items, each one crisp line.
- {{"type":"sources","links":[{{"title":str,"url":"https://..."}}, ...]}} — max {MAX_SOURCE_LINKS} links."""


_UIGEN_SYSTEM = f"""You produce data panels as JSON for a strict renderer. Reply with ONLY a JSON object, no prose.

Schema:
{{"title": str, "accent": "blue"|"violet"|"green"|"amber"|"red"|"slate", "components": [component, ...]}}
(max {MAX_COMPONENTS} components)

Component types:
{_COMPONENT_SCHEMA_LINES}

Optional on any component: "span":"full"|"half" (two consecutive half components sit side by
side), "emphasis":"lead" (at most one per panel), "tone":"neutral"|"positive"|"negative".

Per-panel caps: one each of hero, chart, timeline, fact, comparison, sources; at most two
callouts, two lists, and {MAX_STAT_CARDS} stat cards.

Editorial bar: this panel is shown full-screen in a product demo. Lead with a hero when the
subject is a specific person, company, or place; otherwise lead with the strongest number.
Use real, accurate figures from your knowledge and keep labels tight (2-4 words). Never emit
placeholders such as "$X.XB", "TBD", "N/A" or zeroed series for figures you do not know — use
only values you are confident in. No markdown, no emoji, no filler."""

_UI_BUILD_PLAN_SYSTEM = f"""You art-direct a small, accurate data panel for a strict renderer.
Reply with ONLY this JSON object and nothing else:
{{"title":str,"accent":"blue"|"violet"|"green"|"amber"|"red"|"slate","component_types":[type, ...]}}

Available component types and their per-panel caps:
hero (max 1) — portrait + name + one-line framing; use FIRST when the subject is a specific
  person, company, or place likely to have a Wikipedia article.
stat_card (max {MAX_STAT_CARDS}) — one striking NUMBER with a short label.
chart (max 1) — a bar or line series of a real metric; never years-as-values.
timeline (max 1) — 2-{MAX_TIMELINE_EVENTS} dated events; the right home for any chronology.
fact (max 1) — 2-{MAX_FACT_ROWS} label/value rows for short non-numeric facts.
callout (max 2) — one big takeaway sentence.
comparison (max 1) — two labeled columns contrasting A against B.
list (max 2) — a short bullet list.
sources (max 1) — reference links, last when present.

Pick an accent that suits the subject's character. Order the list exactly as the panel should
read, lead with the strongest element, and choose 4-8 components. Choose only components you can
fill with real content. Do not provide the contents of components yet."""

_UI_BUILD_COMPONENT_SYSTEM = f"""You are adding exactly one component to an in-progress data panel.
Reply with ONLY the JSON object {{"component":{{...}}}} and nothing else. The component's type must
be exactly the requested type and conform to this renderer schema:

{_COMPONENT_SCHEMA_LINES}

Optional on any component: "span":"full"|"half" (two consecutive half components sit side by
side — use half for fact, list, comparison or timeline when a neighbour pairs well),
"emphasis":"lead" (at most one component in the whole panel), and
"tone":"neutral"|"positive"|"negative" for stat cards, callouts and charts.

Use accurate, concise content. Do not repeat a component that is already present. Never emit
placeholders such as "$X.XB", "TBD", "N/A" or zeroed series for figures you do not know: use only
values you are confident in, and otherwise supply the closest fact you are sure of."""


def _validate_build_plan(plan: Any) -> tuple[str, str, list[str]]:
    if not isinstance(plan, dict):
        raise ValueError("UI build plan must be a JSON object.")
    title = plan.get("title")
    accent = plan.get("accent", "blue")
    types = plan.get("component_types")
    if not _non_empty_str(title):
        raise ValueError("UI build plan requires a non-empty title.")
    if accent not in UI_ACCENTS:
        raise ValueError(f"UI build plan accent must be one of {UI_ACCENTS}.")
    if not isinstance(types, list) or not types or len(types) > MAX_COMPONENTS:
        raise ValueError(f"UI build plan requires 1-{MAX_COMPONENTS} component types.")
    if any(kind not in UI_COMPONENT_TYPES for kind in types):
        raise ValueError("UI build plan contains an unsupported component type.")
    for kind, cap in UI_TYPE_CAPS.items():
        if types.count(kind) > cap:
            raise ValueError(f"UI build plan exceeds the cap of {cap} for {kind}.")
    return title.strip(), accent, list(types)


def _validate_component(component: Any) -> dict[str, Any]:
    if not isinstance(component, dict):
        raise ValueError("UI build component must be a JSON object.")
    problems = validate_ui_spec({"title": "Partial build", "components": [component]})
    if problems:
        raise ValueError("Invalid UI build component: " + " ".join(problems))
    return component


class OpenAIUIGenProvider:
    """Live delegated UI model: prompts an OpenAI chat model for a
    validator-clean spec. One retry with the validator's complaints appended;
    a still-invalid spec raises, which the runtime converts to a failed job."""

    mode = "openai"

    def __init__(self, *, model: str | None = None) -> None:
        import httpx

        self.model = model or os.getenv("UIGEN_MODEL", "gpt-4o")
        # Reasoning models take an explicit effort level; the delegated UI job
        # wants the fastest tier, so the demo config sets this to "none".
        self.reasoning_effort = os.getenv("UIGEN_REASONING_EFFORT", "").strip() or None
        api_key = os.getenv("OPENAI_API_KEY", "").strip().strip("'\"")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for UIGEN_MODE=openai.")
        self._client = httpx.AsyncClient(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        # Hero portraits come from Wikipedia's page-image API rather than the
        # model: a generated image URL would be an unverified guess.
        self._wiki = httpx.AsyncClient(
            base_url="https://en.wikipedia.org",
            headers={"User-Agent": "smol-interactions-uigen/0.1"},
            timeout=8.0,
        )

    async def _complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return json.loads(response.json()["choices"][0]["message"]["content"])

    async def hero_image(self, query: str) -> str | None:
        """Best-effort canonical portrait; a miss simply leaves the hero imageless."""
        if not query.strip():
            return None
        try:
            response = await self._wiki.get(
                "/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "redirects": 1,
                    "generator": "search",
                    "gsrsearch": query,
                    "gsrlimit": 1,
                    "prop": "pageimages",
                    "pithumbsize": 640,
                },
            )
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})
            for page in pages.values():
                source = page.get("thumbnail", {}).get("source")
                if isinstance(source, str) and source.startswith("https://"):
                    return source
        except Exception:
            return None
        return None

    async def generate(self, request: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": _UIGEN_SYSTEM},
            {"role": "user", "content": request},
        ]
        spec = await self._complete(messages)
        problems = validate_ui_spec(spec)
        if problems:
            messages.append({"role": "assistant", "content": json.dumps(spec)})
            messages.append(
                {"role": "user", "content": "Fix these problems and resend the full JSON: " + " ".join(problems)}
            )
            spec = await self._complete(messages)
            problems = validate_ui_spec(spec)
            if problems:
                raise ValueError("UI model produced an invalid spec: " + " ".join(problems))
        return spec

    async def _complete_build_plan(self, request: str) -> tuple[str, str, list[str]]:
        messages = [
            {"role": "system", "content": _UI_BUILD_PLAN_SYSTEM},
            {"role": "user", "content": request},
        ]
        plan = await self._complete(messages)
        try:
            return _validate_build_plan(plan)
        except ValueError as exc:
            messages.extend(
                [
                    {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)},
                    {"role": "user", "content": f"Fix this plan: {exc}"},
                ]
            )
            return _validate_build_plan(await self._complete(messages))

    async def _complete_build_component(
        self,
        *,
        context: dict[str, Any],
        component_type: str,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": _UI_BUILD_COMPONENT_SYSTEM},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]

        def validate(result: dict[str, Any]) -> dict[str, Any]:
            component = _validate_component(result.get("component"))
            if component.get("type") != component_type:
                raise ValueError(
                    f"UI model returned {component.get('type')!r}; expected {component_type!r}."
                )
            return component

        result = await self._complete(messages)
        try:
            return validate(result)
        except ValueError as exc:
            messages.extend(
                [
                    {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
                    {"role": "user", "content": f"Fix this component: {exc}"},
                ]
            )
            return validate(await self._complete(messages))

    async def stream_build(self, request: str) -> AsyncIterator[dict[str, Any]]:
        """Build one validated component per model request.

        This intentionally trades a few short background calls for truthful UI
        construction: the browser receives a component only after this provider
        has requested and received that component from the UI model.
        """
        title, accent, component_types = await self._complete_build_plan(request)
        components: list[dict[str, Any]] = []
        lead_used = False
        yield {"kind": "title", "title": title, "accent": accent}

        for position, component_type in enumerate(component_types, start=1):
            context = {
                "request": request,
                "title": title,
                "accent": accent,
                "completed_components": components,
                "next_component_type": component_type,
                "position": position,
                "lead_already_used": lead_used,
            }
            component = await self._complete_build_component(
                context=context,
                component_type=component_type,
            )
            # Only one component may lead; a later claim is demoted rather than
            # failing the whole build.
            if component.get("emphasis") == "lead":
                if lead_used:
                    component.pop("emphasis", None)
                else:
                    lead_used = True
            if component["type"] == "hero":
                image_url = await self.hero_image(
                    str(component.get("image_query") or component.get("name", ""))
                )
                if image_url:
                    component["image_url"] = image_url
            components.append(component)
            yield {"kind": "component", "component": component}

        problems = validate_ui_spec({"title": title, "accent": accent, "components": components})
        if problems:
            raise ValueError("UI model produced an invalid final build: " + " ".join(problems))


class DemoUIGenProvider:
    """Deterministic offline stand-in for the delegated UI model.

    Returns a canned, validator-clean spec after a controlled delay so the
    concurrency choreography (jobs genuinely outstanding, either completion
    order) can be exercised and evaluated without a live model. It is
    intentionally *not* presented as progressive generation: only the live
    provider may reveal a component as it is actually produced.
    """

    mode = "demo"

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds

    async def generate(self, request: str) -> dict[str, Any]:
        del request
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        return copy.deepcopy(_DEMO_SPEC)

    async def stream_build(self, request: str) -> AsyncIterator[dict[str, Any]]:
        """Publish the static demo only after its simulated job has completed."""
        spec = await self.generate(request)
        yield {"kind": "title", "title": spec["title"], "accent": spec["accent"]}
        for component in spec["components"]:
            yield {"kind": "component", "component": component}
