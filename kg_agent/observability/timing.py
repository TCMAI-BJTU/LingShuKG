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
    """Attach shared file/chunk fields to nested timing logs on this thread."""
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
    """Time a block and emit one stderr line only when it exceeds two seconds."""
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
            # Write whole log lines atomically so multi-thread timing logs do not interleave.
            with _PRINT_LOCK:
                print(message, file=sys.stderr, flush=True)


def log_event(event: str, **fields: object) -> None:
    """Write one atomic stderr event line with thread context."""
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
    if isinstance(value, (str, dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
