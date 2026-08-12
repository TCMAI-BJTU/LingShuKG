"""Small response helpers shared by tool implementations."""

from __future__ import annotations

from typing import Any


def success(**payload: Any) -> dict[str, Any]:
    """构造统一的成功工具响应。"""
    return {"ok": True, **payload}


def failure(code: str, message: str, **payload: Any) -> dict[str, Any]:
    """构造统一的失败工具响应。"""
    return {
        "ok": False,
        "error": {"code": code, "message": message, **payload},
    }
