"""Strict JSONL decoding shared by model-output protocols."""

from __future__ import annotations

import json
from typing import Any


def decode_jsonl_objects(
    raw: str,
    item_label: str,
    skip_blank_lines: bool = False,
) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    """逐行解码 JSON 对象，并按调用方要求处理纯空白行。"""
    content = str(raw or "").strip()
    if not content:
        return [], [
            {
                "line": 1,
                "code": "EMPTY_JSONL",
                "message": f"{item_label}不能为空",
            }
        ]
    items: list[tuple[int, dict[str, Any]]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            if skip_blank_lines:
                continue
            errors.append(
                {
                    "line": line_number,
                    "code": "EMPTY_JSONL_LINE",
                    "message": f"{item_label}不能包含空行",
                }
            )
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as error:
            errors.append(
                {
                    "line": line_number,
                    "code": "INVALID_JSON",
                    "message": str(error),
                }
            )
            continue
        if not isinstance(payload, dict):
            errors.append(
                {
                    "line": line_number,
                    "code": "INVALID_JSON_OBJECT",
                    "message": f"{item_label}每行必须是 JSON 对象，不支持 JSON 数组",
                }
            )
            continue
        items.append((line_number, payload))
    return items, errors
