"""Validation for untrusted model-generated tool arguments."""

from __future__ import annotations

from typing import Any

from .common import failure
from .specs import TOOL_SPECS


def validate_tool_arguments(
    name: str,
    arguments: Any,
    definitions: list[dict[str, Any]] = TOOL_SPECS,
) -> dict[str, Any] | None:
    """Validate model-generated tool argument names and base JSON types."""
    parameters = next(
        (
            item["function"]["parameters"]
            for item in definitions
            if item["function"]["name"] == name
        ),
        None,
    )
    if parameters is None:
        return failure("UNKNOWN_TOOL", f"未知工具：{name}")
    if not isinstance(arguments, dict):
        return failure("INVALID_ARGUMENTS", "工具 arguments 必须是 JSON 对象")
    properties = parameters["properties"]
    missing = set(parameters["required"]) - arguments.keys()
    if missing:
        return failure("MISSING_ARGUMENT", f"缺少参数：{sorted(missing)}")
    unknown = arguments.keys() - properties.keys()
    if unknown:
        return failure("UNKNOWN_ARGUMENT", f"未知参数：{sorted(unknown)}")
    for key, value in arguments.items():
        error = _validate_value(key, value, properties[key])
        if error is not None:
            return error
    return None


def _validate_value(
    key: str,
    value: Any,
    definition: dict[str, Any],
) -> dict[str, Any] | None:
    expected = definition.get("type")
    if expected == "string" and not isinstance(value, str):
        return failure("INVALID_ARGUMENT_TYPE", f"参数 {key} 必须是字符串")
    if expected == "object" and not isinstance(value, dict):
        return failure("INVALID_ARGUMENT_TYPE", f"参数 {key} 必须是 JSON 对象")
    if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        return failure("INVALID_ARGUMENT_TYPE", f"参数 {key} 必须是整数")
    if expected == "boolean" and not isinstance(value, bool):
        return failure("INVALID_ARGUMENT_TYPE", f"参数 {key} 必须是布尔值")
    if "enum" in definition and value not in definition["enum"]:
        return failure(
            "INVALID_ARGUMENT_VALUE",
            f"参数 {key} 必须是以下值之一：{definition['enum']}",
        )
    if expected == "integer" and value < definition.get("minimum", value):
        return failure("INVALID_ARGUMENT_VALUE", f"参数 {key} 小于允许下限")
    if expected == "integer" and value > definition.get("maximum", value):
        return failure("INVALID_ARGUMENT_VALUE", f"参数 {key} 大于允许上限")
    if expected == "object" and "properties" in definition:
        return _validate_object(key, value, definition)
    return None


def _validate_object(
    key: str,
    value: dict[str, Any],
    definition: dict[str, Any],
) -> dict[str, Any] | None:
    properties = definition["properties"]
    missing = set(definition.get("required", [])) - value.keys()
    if missing:
        return failure("MISSING_ARGUMENT", f"参数 {key} 缺少字段：{sorted(missing)}")
    unknown = value.keys() - properties.keys()
    if unknown:
        return failure("UNKNOWN_ARGUMENT", f"参数 {key} 包含未知字段：{sorted(unknown)}")
    for child_key, child_value in value.items():
        error = _validate_value(
            f"{key}.{child_key}",
            child_value,
            properties[child_key],
        )
        if error is not None:
            return error
    return None
