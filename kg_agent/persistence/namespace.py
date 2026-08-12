"""Stable cache namespaces independent of the extraction Schema."""

from __future__ import annotations

import json
from pathlib import Path


def load_cache_namespace(path: Path | str, model: str) -> str:
    """Load a model's stable cache namespace, or a deterministic default if unset."""
    namespace_path = Path(path)
    if not namespace_path.is_file():
        return _default_cache_namespace(model)
    payload = json.loads(namespace_path.read_text(encoding="utf-8"))
    namespaces = payload["model_namespaces"]
    return str(namespaces.get(model) or _default_cache_namespace(model))


def _default_cache_namespace(model: str) -> str:
    """Build a Schema-independent stable cache namespace for models without prior state."""
    return f"model:{model}"
