"""Small JSON action protocol used by the constrained ReAct loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .jsonl import decode_jsonl_objects


THOUGHT_MARKER = "✿THOUGHT✿"
ACTION_MARKER = "✿ACTION✿"


@dataclass(frozen=True)
class AgentAction:
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ActionParseResult:
    actions: tuple[AgentAction, ...]
    errors: tuple[str, ...]

    @property
    def action(self) -> AgentAction | None:
        return self.actions[0] if len(self.actions) == 1 else None

    @property
    def error(self) -> str | None:
        return "；".join(self.errors) if self.errors else None


def parse_action(raw: str) -> ActionParseResult:
    """Validate ReAct outer format, then parse JSON tool actions line by line."""
    format_errors = _validate_react_format(raw)
    if format_errors:
        return ActionParseResult((), format_errors)
    content = _extract_action_text(raw)
    items, json_errors = decode_jsonl_objects(content, "工具动作")
    actions: list[AgentAction] = []
    errors = [
        f"第 {error['line']} 行动作 JSON 无法解析：{error['message']}"
        for error in json_errors
    ]
    for line_number, payload in items:
        action, error = _parse_action_item(payload, line_number)
        if action is None:
            errors.append(error or f"第 {line_number} 行动作无效")
        else:
            actions.append(action)
    return ActionParseResult(tuple(actions), tuple(errors))


def _parse_action_item(
    payload: Any,
    index: int,
) -> tuple[AgentAction | None, str | None]:
    if not isinstance(payload, dict):
        return None, f"第 {index} 个动作必须是 JSON 对象"
    unknown_fields = sorted(set(payload) - {"tool", "arguments"})
    if unknown_fields:
        return (
            None,
            f"第 {index} 个动作包含非法顶层字段 {unknown_fields}；"
            "顶层只允许 tool 和 arguments，所有工具参数必须放入 "
            'arguments，例如 {"tool":"工具名","arguments":{"reason":"..."}}',
        )
    tool = payload.get("tool")
    arguments = payload.get("arguments", {})
    if not isinstance(tool, str) or not tool:
        return None, f"第 {index} 个动作缺少非空字符串 tool"
    if not isinstance(arguments, dict):
        return None, f"第 {index} 个动作 arguments 必须是 JSON 对象"
    return AgentAction(tool, arguments), None


def _validate_react_format(raw: str) -> tuple[str, ...]:
    """Validate Thought/Action marker count, order, and payload."""
    content = str(raw or "").strip()
    thought_count = content.count(THOUGHT_MARKER)
    action_count = content.count(ACTION_MARKER)
    errors: list[str] = []
    if thought_count != 1:
        errors.append(
            f"每轮必须且只能输出一个 {THOUGHT_MARKER}，当前为 {thought_count} 个"
        )
    if action_count != 1:
        errors.append(
            f"每轮必须且只能输出一个 {ACTION_MARKER}；多个工具共用该标记，当前为 {action_count} 个"
        )
    if errors:
        return tuple(errors)
    thought_index = content.index(THOUGHT_MARKER)
    action_index = content.index(ACTION_MARKER)
    if thought_index != 0 or action_index <= thought_index:
        return (f"输出必须以 {THOUGHT_MARKER} 开始，并在其后输出 {ACTION_MARKER}",)
    thought = content[
        thought_index + len(THOUGHT_MARKER) : action_index
    ].strip()
    actions = content[action_index + len(ACTION_MARKER) :].strip()
    if not thought:
        errors.append(f"{THOUGHT_MARKER} 后必须包含简短决策摘要")
    if not actions:
        errors.append(f"{ACTION_MARKER} 后必须包含至少一个 JSON 工具动作")
    return tuple(errors)


def _extract_action_text(raw: str) -> str:
    return str(raw).split(ACTION_MARKER, 1)[1].strip()
