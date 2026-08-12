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
        if self.checkpoint_deferred:
            # During batched tools, mark dirty only—avoid a diskcache write on every add.
            self.checkpoint_dirty = True
            return
        self.cache.save(self.cache_namespace, self.workspace)

    def begin_checkpoint_batch(self) -> None:
        """Defer diskcache writes for the current tool batch."""
        self.checkpoint_deferred = True

    def flush_checkpoint_batch(self) -> None:
        """End the tool batch and flush accumulated edits to diskcache once."""
        self.checkpoint_deferred = False
        if not self.checkpoint_dirty:
            return
        self.checkpoint_dirty = False
        self.cache.save(self.cache_namespace, self.workspace)

    def mark_changed(self) -> None:
        """Record a graph change and invalidate stale validation/empty confirmation."""
        self.workspace.revision += 1
        # Validation is revision-scoped; any graph change requires re-validation.
        self.workspace.validated_revision = None
        self.workspace.empty_confirmed_revision = None
        self.workspace.empty_reason = None
        self.checkpoint()

    def mark_validated(self, valid: bool) -> None:
        self.workspace.validated_revision = (
            self.workspace.revision if valid else None
        )
        self.checkpoint()

    def mark_empty_confirmed(self, reason: str) -> None:
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
    """Create or restore tool context for a fixed chunk."""
    source_id = make_source_id(source_name, text, source_key)
    committed = store.source_exists(source_id)
    # Committed results come from SQLite; restore diskcache drafts only for uncommitted chunks.
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
