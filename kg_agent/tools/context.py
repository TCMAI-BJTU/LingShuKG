"""Read-only tools for the bound chunk."""

from __future__ import annotations

from .common import success
from .tool_context import ToolContext


class ContextTools:
    def __init__(self, context: ToolContext) -> None:
        self.context = context

    def get_chunk_context(self) -> dict:
        """Return only the chunk text; hide internal source metadata."""
        return success(text=self.context.workspace.text)

    def get_workspace_summary(self) -> dict:
        workspace = self.context.workspace
        return success(
            entity_count=len(workspace.entities),
            relation_count=len(workspace.relations),
            review_warning_count=len(workspace.review_warnings),
            committed=self.context.committed,
        )
