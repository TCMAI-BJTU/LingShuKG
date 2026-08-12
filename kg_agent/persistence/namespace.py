"""Stable cache namespaces independent of the extraction Schema."""

from __future__ import annotations

import json
from pathlib import Path


def load_cache_namespace(path: Path | str, model: str) -> str:
    """读取指定模型的稳定缓存命名空间；未配置时使用确定性的模型命名空间。"""
    namespace_path = Path(path)
    if not namespace_path.is_file():
        return _default_cache_namespace(model)
    payload = json.loads(namespace_path.read_text(encoding="utf-8"))
    namespaces = payload["model_namespaces"]
    return str(namespaces.get(model) or _default_cache_namespace(model))


def _default_cache_namespace(model: str) -> str:
    """为没有历史状态的模型生成不依赖 Schema 的稳定缓存命名空间。"""
    return f"model:{model}"
