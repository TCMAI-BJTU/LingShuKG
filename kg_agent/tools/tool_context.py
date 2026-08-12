"""Dependencies and scope shared by all chunk tools."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.schema import GraphSchema
from ..core.workspace import ChunkWorkspace
from ..persistence.cache import WorkspaceCache
from ..persistence.storage import GraphStore, make_source_id


@dataclass
class ToolContext:
    workspace: ChunkWorkspace
    schema: GraphSchema
    store: GraphStore
    cache: WorkspaceCache
    cache_namespace: str
    committed: bool = False
    checkpoint_deferred: bool = False
    checkpoint_dirty: bool = False

    def checkpoint(self) -> None:
        """将当前工作区保存为可恢复草稿。"""
        if self.checkpoint_deferred:
            # 批量工具阶段只标记脏状态，避免每个 add 操作都写一次 diskcache。
            self.checkpoint_dirty = True
            return
        self.cache.save(self.cache_namespace, self.workspace)

    def begin_checkpoint_batch(self) -> None:
        """延迟当前工具批次内的 diskcache 写入。"""
        self.checkpoint_deferred = True

    def flush_checkpoint_batch(self) -> None:
        """结束当前工具批次并将累积修改一次性写入 diskcache。"""
        self.checkpoint_deferred = False
        if not self.checkpoint_dirty:
            return
        self.checkpoint_dirty = False
        self.cache.save(self.cache_namespace, self.workspace)

    def mark_changed(self) -> None:
        """记录图谱修改并使旧校验和空结果确认失效。"""
        self.workspace.revision += 1
        # 校验只对当时的 revision 有效，任何图谱变更都必须重新校验。
        self.workspace.validated_revision = None
        self.workspace.empty_confirmed_revision = None
        self.workspace.empty_reason = None
        self.checkpoint()

    def mark_validated(self, valid: bool) -> None:
        """记录当前工作区版本的校验结果。"""
        self.workspace.validated_revision = (
            self.workspace.revision if valid else None
        )
        self.checkpoint()

    def mark_empty_confirmed(self, reason: str) -> None:
        """记录智能体对当前空工作区的显式确认。"""
        self.workspace.empty_confirmed_revision = self.workspace.revision
        self.workspace.empty_reason = reason
        self.checkpoint()


def open_tool_context(
    source_name: str,
    text: str,
    schema: GraphSchema,
    store: GraphStore,
    cache: WorkspaceCache,
    cache_namespace: str = "default",
    source_key: str | None = None,
) -> ToolContext:
    """为一个固定片段创建或恢复工具上下文。"""
    source_id = make_source_id(source_name, text, source_key)
    committed = store.source_exists(source_id)
    # 已提交结果以 SQLite 为准；只有未提交片段才恢复 diskcache 草稿。
    workspace = (
        None
        if committed
        else cache.load(cache_namespace, source_id)
    )
    if workspace is None:
        workspace = ChunkWorkspace(source_id, source_name, text)
    return ToolContext(
        workspace,
        schema,
        store,
        cache,
        cache_namespace,
        committed,
    )
