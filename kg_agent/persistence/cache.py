"""Disk-backed checkpoints for the current chunk workspace."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from diskcache import Cache

from ..core.workspace import ChunkWorkspace
from ..observability import log_slow_operation


class WorkspaceCache:
    def __init__(self, directory: Path | str) -> None:
        """打开指定目录中的 diskcache 缓存。"""
        self._cache = Cache(str(directory))

    @staticmethod
    def _key(cache_namespace: str, source_id: str) -> str:
        """根据稳定缓存命名空间和来源 ID 构造工作区缓存键。"""
        return f"workspace:{cache_namespace}:{source_id}"

    def load(
        self,
        cache_namespace: str,
        source_id: str,
    ) -> ChunkWorkspace | None:
        """读取片段草稿；不存在时返回空值。"""
        with log_slow_operation("cache.workspace_load"):
            payload = self._cache.get(
                self._key(cache_namespace, source_id)
            )
        return ChunkWorkspace.from_dict(payload) if payload is not None else None

    def save(
        self,
        cache_namespace: str,
        workspace: ChunkWorkspace,
    ) -> None:
        """持久化当前片段工作区。"""
        with log_slow_operation("cache.workspace_save"):
            self._cache.set(
                self._key(cache_namespace, workspace.source_id),
                workspace.to_dict(),
            )

    def delete(self, cache_namespace: str, source_id: str) -> None:
        """删除已提交片段的工作区草稿。"""
        with log_slow_operation("cache.workspace_delete"):
            self._cache.delete(self._key(cache_namespace, source_id))

    @staticmethod
    def _entity_review_key(
        cache_namespace: str,
        name: str,
        context: str,
    ) -> str:
        """根据稳定命名空间、实体名称和局部上下文构造审查缓存键。"""
        payload = f"{name}\0{context}".encode("utf-8")
        context_key = hashlib.sha256(payload).hexdigest()
        return f"entity-review:{cache_namespace}:{context_key}"

    def load_entity_review(
        self,
        cache_namespace: str,
        name: str,
        context: str,
    ) -> dict[str, Any] | None:
        """读取指定名称和局部上下文的实体类型审查结果。"""
        with log_slow_operation(
            "cache.entity_review_load",
            entity_name=name,
        ):
            payload = self._cache.get(
                self._entity_review_key(
                    cache_namespace,
                    name,
                    context,
                )
            )
        return dict(payload) if payload is not None else None

    def save_entity_review(
        self,
        cache_namespace: str,
        name: str,
        context: str,
        review: dict[str, Any],
    ) -> None:
        """按名称和局部上下文持久化实体类型审查结果。"""
        with log_slow_operation(
            "cache.entity_review_save",
            entity_name=name,
        ):
            self._cache.set(
                self._entity_review_key(
                    cache_namespace,
                    name,
                    context,
                ),
                review,
            )

    def close(self) -> None:
        """关闭 diskcache 资源。"""
        self._cache.close()
