"""Validation and atomic submission tools."""

from __future__ import annotations

from typing import Any

from ..core.validation import validate_workspace
from .common import failure, success
from .tool_context import ToolContext


class LifecycleTools:
    def __init__(self, context: ToolContext) -> None:
        """绑定当前片段工具上下文。"""
        self.context = context

    def validate_chunk(self) -> dict[str, Any]:
        """执行提交前完整工作区校验。"""
        issues = validate_workspace(self.context.workspace, self.context.schema)
        self.context.mark_validated(not issues)
        return success(
            valid=not issues,
            errors=[issue.to_dict() for issue in issues],
            review_warnings=list(
                self.context.workspace.review_warnings.values()
            ),
        )

    def confirm_empty_chunk(self, reason: str) -> dict[str, Any]:
        """显式确认当前片段没有可抽取实体和关系。"""
        if self.context.committed:
            return failure("CHUNK_COMMITTED", "当前片段已经提交")
        workspace = self.context.workspace
        if workspace.entities or workspace.relations:
            return failure("CHUNK_NOT_EMPTY", "当前工作区并非空结果")
        normalized_reason = reason.strip()
        if len(normalized_reason) < 4:
            return failure("EMPTY_REASON_TOO_SHORT", "空结果原因至少需要 4 个字符")
        self.context.mark_empty_confirmed(normalized_reason)
        return success(status="empty_confirmed", reason=normalized_reason)

    def submit_chunk(self) -> dict[str, Any]:
        """校验并将当前片段原子提交到数据库。"""
        if self.context.committed:
            return success(status="already_committed")
        workspace = self.context.workspace
        if workspace.review_warnings:
            return failure(
                "REVIEW_WARNINGS_PENDING",
                "提交前必须修正或显式确认全部审查分歧",
                warnings=list(workspace.review_warnings.values()),
            )
        if workspace.validated_revision != workspace.revision:
            # revision 不一致说明 validate 后又发生了修改，旧校验结果不可复用。
            return failure(
                "CHUNK_NOT_VALIDATED",
                "提交前必须对当前工作区版本调用 validate_chunk",
            )
        if (
            not workspace.entities
            and workspace.empty_confirmed_revision != workspace.revision
        ):
            return failure(
                "EMPTY_CHUNK_NOT_CONFIRMED",
                "空结果提交前必须调用 confirm_empty_chunk",
            )
        issues = validate_workspace(workspace, self.context.schema)
        if issues:
            return failure(
                "VALIDATION_FAILED",
                "当前片段未通过校验",
                errors=[issue.to_dict() for issue in issues],
            )
        inserted = self.context.store.commit_chunk(workspace)
        self.context.committed = True
        # SQLite 已成为最终状态，草稿缓存不再需要保留。
        self.context.cache.delete(
            self.context.cache_namespace,
            self.context.workspace.source_id,
        )
        status = "committed" if inserted else "already_committed"
        return success(status=status)
