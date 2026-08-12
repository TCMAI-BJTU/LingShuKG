"""Thread-safe logs for operations that exceed the slow threshold."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from time import perf_counter
from typing import Iterator


SLOW_OPERATION_SECONDS = 2.0
_TIMING_FIELDS: ContextVar[dict[str, object]] = ContextVar(
    "timing_fields",
    default={},
)
_PRINT_LOCK = Lock()


@contextmanager
def timing_context(**fields: object) -> Iterator[None]:
    """为当前线程内的嵌套耗时日志附加文件、chunk 等公共字段。"""
    token = _TIMING_FIELDS.set({**_TIMING_FIELDS.get(), **fields})
    try:
        yield
    finally:
        _TIMING_FIELDS.reset(token)


@contextmanager
def log_slow_operation(
    operation: str,
    **fields: object,
) -> Iterator[None]:
    """测量代码块耗时，并仅在超过两秒时向标准错误输出一行日志。"""
    started_at = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - started_at
        if elapsed > SLOW_OPERATION_SECONDS:
            merged_fields = {**_TIMING_FIELDS.get(), **fields}
            details = " ".join(
                f"{name}={_format_field(value)}"
                for name, value in merged_fields.items()
            )
            message = (
                f"[timing] operation={operation} elapsed={elapsed:.2f}s"
            )
            if details:
                message = f"{message} {details}"
            # 多线程只允许整行写入，避免不同 chunk 的耗时日志互相穿插。
            with _PRINT_LOCK:
                print(message, file=sys.stderr, flush=True)


def log_event(event: str, **fields: object) -> None:
    """向标准错误输出一条带线程上下文的原子事件日志。"""
    merged_fields = {**_TIMING_FIELDS.get(), **fields}
    details = " ".join(
        f"{name}={_format_field(value)}"
        for name, value in merged_fields.items()
    )
    message = f"[{event}]"
    if details:
        message = f"{message} {details}"
    with _PRINT_LOCK:
        print(message, file=sys.stderr, flush=True)


def _format_field(value: object) -> str:
    """将日志字段格式化为紧凑且可区分空格的单行文本。"""
    if isinstance(value, (str, dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
