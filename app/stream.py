from __future__ import annotations

import json
import re
from html import escape, unescape
from typing import Any

from app.domain import Action, ActionKind, CompletedTurn, StreamEvent


_ACTION_PATTERN = re.compile(r"^<action>(.+)</action>$", re.DOTALL)
_TOOL_PATTERN = re.compile(r"^tool\(([A-Za-z_][A-Za-z0-9_.-]*),\s*(\{.*\})\)$", re.DOTALL)
_RESPOND_PATTERN = re.compile(r"^respond\((\{.*\})\)$", re.DOTALL)
_G1_TOOL_PATTERN = re.compile(
    r"^(suggest_edit|highlight|delegate|web_search|translate_commit)\((\{.*\})\)$",
    re.DOTALL,
)

STREAM_FORMATS = ("flat", "wrapped", "g1")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _event_attributes(event: StreamEvent) -> str:
    attributes = [f'index="{event.index}"', f'source="{event.source.value}"']
    if event.state is not None:
        attributes.append(f'state="{event.state.value}"')
    if event.elapsed_ms is not None:
        attributes.append(f'time="t+{event.elapsed_ms}ms"')
    if event.tool_name is not None:
        attributes.append(f'tool="{escape(event.tool_name, quote=True)}"')
    if event.call_id is not None:
        attributes.append(f'call_id="{escape(event.call_id, quote=True)}"')
    if event.job_id is not None:
        attributes.append(f'job_id="{escape(event.job_id, quote=True)}"')
    return " ".join(attributes)


def action_body(action: Action) -> str:
    if action.kind is ActionKind.TOOL and action.tool_name:
        return f"tool({action.tool_name},{_json(action.arguments)})"
    if action.kind is ActionKind.RESPOND and action.message and action.target is not None:
        return f'respond({_json({"for": action.target, "message": action.message})})'
    return "idle()"


def action_completion(action: Action) -> str:
    return f"<action>{action_body(action)}</action>"


def g1_action_body(action: Action) -> str:
    if action.kind is ActionKind.IDLE:
        return "idle()"
    if action.kind is ActionKind.RESPOND and action.message and action.target is not None:
        return f'respond({_json({"for": action.target, "message": action.message})})'
    if (
        action.kind is ActionKind.TOOL
        and action.tool_name
        in {"suggest_edit", "highlight", "delegate", "web_search", "translate_commit"}
    ):
        return f"{action.tool_name}({_json(action.arguments)})"
    raise ValueError(f"Action is not part of the g1 grammar: {action!r}")


def g1_action_completion(action: Action) -> str:
    return f"<action>{g1_action_body(action)}</action>"


def render_event(event: StreamEvent, indent: str = "") -> str:
    body = escape(event.content, quote=False)
    return (
        f"{indent}<stream_event {_event_attributes(event)}>\n"
        f"{indent}  {body}\n"
        f"{indent}</stream_event>"
    )


def render_action(action: Action, indent: str = "") -> str:
    return f'{indent}<action source="assistant">{escape(action_body(action), quote=False)}</action>'


def render_event_flat(event: StreamEvent) -> str:
    return f"<stream_event {_event_attributes(event)}>{escape(event.content, quote=False)}</stream_event>"


def render_action_flat(action: Action) -> str:
    return f"<action>{escape(action_body(action), quote=False)}</action>"


def render_action_flat_g1(action: Action) -> str:
    return f"<action>{escape(g1_action_body(action), quote=False)}</action>"


def render_context(instruction: str, permissions: dict[str, list[str]]) -> str:
    return "\n".join(
        [
            "<interaction_context>",
            f"<instruction>{escape(instruction, quote=False)}</instruction>",
            f"<permissions>{escape(_json(permissions), quote=False)}</permissions>",
            "</interaction_context>",
        ]
    )


def compile_stream(
    history: list[CompletedTurn],
    current: StreamEvent,
    *,
    instruction: str = "",
    permissions: dict[str, list[str]] | None = None,
    fmt: str = "flat",
) -> str:
    permissions = permissions or {}
    if fmt == "flat":
        return _compile_flat(history, current, instruction, permissions)
    if fmt == "wrapped":
        return _compile_wrapped(history, current, instruction, permissions)
    if fmt == "g1":
        return _compile_g1(history, current, instruction, permissions)
    raise ValueError(f"Unknown stream format: {fmt!r}. Use one of {STREAM_FORMATS}.")


def _compile_flat(
    history: list[CompletedTurn],
    current: StreamEvent,
    instruction: str,
    permissions: dict[str, list[str]],
) -> str:
    lines: list[str] = [render_context(instruction, permissions)]
    for turn in history:
        lines.append(render_event_flat(turn.event))
        lines.append(render_action_flat(turn.action))
    lines.extend([render_event_flat(current), "<PREDICT_THIS_ACTION>"])
    return "\n".join(lines)


def _compile_wrapped(
    history: list[CompletedTurn],
    current: StreamEvent,
    instruction: str,
    permissions: dict[str, list[str]],
) -> str:
    lines = [render_context(instruction, permissions), "<stream_history>"]
    for turn in history:
        lines.append(render_event(turn.event, indent="  "))
        lines.append("")
        lines.append(render_action(turn.action, indent="  "))
        lines.append("")
    if lines[-1] == "":
        lines.pop()
    lines.extend(["</stream_history>", "", render_event(current), "", "<PREDICT_THIS_ACTION>"])
    return "\n".join(lines)


def _compile_g1(
    history: list[CompletedTurn],
    current: StreamEvent,
    instruction: str,
    permissions: dict[str, list[str]],
) -> str:
    if instruction or permissions:
        raise ValueError("g1 prompts are header-free and do not accept context or permissions.")
    lines: list[str] = []
    for turn in history:
        lines.append(render_event_flat(turn.event))
        lines.append(render_action_flat_g1(turn.action))
    lines.extend([render_event_flat(current), "<PREDICT_THIS_ACTION>"])
    return "\n".join(lines)


def _invalid(raw_output: str, diagnostic: str) -> Action:
    return Action(
        kind=ActionKind.IDLE,
        valid=False,
        raw_output=raw_output,
        diagnostic=diagnostic,
    )


def parse_action(raw_output: str) -> Action:
    cleaned = raw_output.strip()
    wrapper = _ACTION_PATTERN.fullmatch(cleaned)
    if not wrapper:
        return _invalid(raw_output, "Model output was not exactly one <action>...</action> envelope.")

    body = unescape(wrapper.group(1)).strip()
    if body == "idle()":
        return Action(kind=ActionKind.IDLE, raw_output=raw_output)

    tool_match = _TOOL_PATTERN.fullmatch(body)
    if tool_match:
        try:
            arguments = json.loads(tool_match.group(2))
        except json.JSONDecodeError as exc:
            return _invalid(raw_output, f"Tool arguments were not valid JSON: {exc.msg}.")
        if not isinstance(arguments, dict):
            return _invalid(raw_output, "Tool arguments must be a JSON object.")
        return Action(
            kind=ActionKind.TOOL,
            tool_name=tool_match.group(1),
            arguments=arguments,
            raw_output=raw_output,
        )

    respond_match = _RESPOND_PATTERN.fullmatch(body)
    if respond_match:
        try:
            arguments = json.loads(respond_match.group(1))
        except json.JSONDecodeError as exc:
            return _invalid(raw_output, f"Respond arguments were not valid JSON: {exc.msg}.")
        if not isinstance(arguments, dict):
            return _invalid(raw_output, "Respond arguments must be a JSON object.")
        target = arguments.get("for")
        message = arguments.get("message")
        if (
            isinstance(target, int)
            and not isinstance(target, bool)
            and target > 0
            and isinstance(message, str)
            and message.strip()
        ):
            return Action(
                kind=ActionKind.RESPOND,
                target=target,
                message=message.strip(),
                raw_output=raw_output,
            )
        return _invalid(raw_output, 'Respond requires integer "for" and non-empty string "message".')

    return _invalid(
        raw_output,
        "Action body must be idle(), tool(name, JSON object), or respond(JSON object).",
    )


def parse_g1_action(raw_output: str) -> Action:
    if raw_output != raw_output.strip():
        return _invalid(raw_output, "g1 output must contain one action line with no outer whitespace.")
    cleaned = raw_output
    wrapper = _ACTION_PATTERN.fullmatch(cleaned)
    if not wrapper:
        return _invalid(raw_output, "Model output was not exactly one <action>...</action> envelope.")

    # g1 completions are raw action envelopes, not stream markup. Unescaping
    # here would corrupt literal message text such as "AT&amp;T".
    body = wrapper.group(1)
    if body == "idle()":
        action = Action(kind=ActionKind.IDLE, raw_output=raw_output)
    else:
        respond_match = _RESPOND_PATTERN.fullmatch(body)
        tool_match = _G1_TOOL_PATTERN.fullmatch(body)
        if respond_match:
            try:
                arguments = json.loads(respond_match.group(1))
            except json.JSONDecodeError as exc:
                return _invalid(raw_output, f"Respond arguments were not valid JSON: {exc.msg}.")
            if (
                isinstance(arguments, dict)
                and set(arguments) == {"for", "message"}
                and isinstance(arguments.get("for"), int)
                and not isinstance(arguments.get("for"), bool)
                and arguments["for"] > 0
                and isinstance(arguments.get("message"), str)
                and arguments["message"].strip()
            ):
                action = Action(
                    kind=ActionKind.RESPOND,
                    target=arguments["for"],
                    message=arguments["message"],
                    raw_output=raw_output,
                )
            else:
                return _invalid(
                    raw_output,
                    'Respond requires exactly positive integer "for" and non-empty string "message".',
                )
        elif tool_match:
            tool_name = tool_match.group(1)
            try:
                arguments = json.loads(tool_match.group(2))
            except json.JSONDecodeError as exc:
                return _invalid(raw_output, f"{tool_name} arguments were not valid JSON: {exc.msg}.")
            if not isinstance(arguments, dict):
                return _invalid(raw_output, f"{tool_name} arguments must be a JSON object.")
            if tool_name == "suggest_edit":
                valid = (
                    set(arguments) == {"quote", "replacement"}
                    and isinstance(arguments.get("quote"), str)
                    and bool(arguments["quote"].strip())
                    and isinstance(arguments.get("replacement"), str)
                    and bool(arguments["replacement"].strip())
                )
            elif tool_name == "highlight":
                valid = (
                    set(arguments) == {"occurrence", "quote"}
                    and isinstance(arguments.get("occurrence"), int)
                    and not isinstance(arguments.get("occurrence"), bool)
                    and arguments["occurrence"] > 0
                    and isinstance(arguments.get("quote"), str)
                    and bool(arguments["quote"])
                )
            elif tool_name == "delegate":
                valid = (
                    set(arguments) == {"task"}
                    and isinstance(arguments.get("task"), str)
                    and bool(arguments["task"].strip())
                )
            elif tool_name == "web_search":
                valid = (
                    set(arguments) == {"query"}
                    and isinstance(arguments.get("query"), str)
                    and bool(arguments["query"].strip())
                )
            else:
                valid = (
                    set(arguments) == {"for", "message"}
                    and isinstance(arguments.get("for"), int)
                    and not isinstance(arguments.get("for"), bool)
                    and arguments["for"] > 0
                    and isinstance(arguments.get("message"), str)
                    and bool(arguments["message"].strip())
                )
            if not valid:
                return _invalid(raw_output, f"{tool_name} arguments do not match the g1 schema.")
            action = Action(
                kind=ActionKind.TOOL,
                tool_name=tool_name,
                arguments=arguments,
                raw_output=raw_output,
            )
        else:
            return _invalid(
                raw_output,
                "g1 action must be idle, respond, suggest_edit, highlight, delegate, "
                "web_search, or translate_commit.",
            )

    if cleaned != g1_action_completion(action):
        return _invalid(raw_output, "g1 action was valid JSON but not in canonical byte form.")
    return action
