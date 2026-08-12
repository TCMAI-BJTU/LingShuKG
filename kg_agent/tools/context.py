"""Read-only tools for the bound chunk."""

from __future__ import annotations

from .common import success
from .tool_context import ToolContext


class ContextTools:
    def __init__(self, context: ToolContext) -> None:
        """绑定当前片段工具上下文。"""
        self.context = context

    def get_chunk_context(self) -> dict:
        """仅返回当前片段文本，不暴露内部来源信息。"""
        return success(text=self.context.workspace.text)

    def get_workspace_summary(self) -> dict:
        """返回当前工作区数量和提交状态。"""
        workspace = self.context.workspace
        return success(
            entity_count=len(workspace.entities),
            relation_count=len(workspace.relations),
            review_warning_count=len(workspace.review_warnings),
            committed=self.context.committed,
        )
