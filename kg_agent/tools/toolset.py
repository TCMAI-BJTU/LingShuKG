"""Tool aggregation and dispatch for one bound chunk."""

from __future__ import annotations

from typing import Any, Callable

from ..agent.semantic_review import SemanticReviewer
from ..core.schema import GraphSchema
from ..core.workspace import ChunkWorkspace
from ..observability import log_slow_operation
from ..persistence.cache import WorkspaceCache
from ..persistence.storage import GraphStore
from .arguments import validate_tool_arguments
from .common import failure
from .context import ContextTools
from .entities import EntityTools
from .lifecycle import LifecycleTools
from .relations import RelationTools
from .reviews import ReviewTools
from .specs import build_tool_specs
from .tool_context import open_tool_context


class ChunkToolset:
    def __init__(
        self,
        context_tools: ContextTools,
        entity_tools: EntityTools,
        review_tools: ReviewTools,
        relation_tools: RelationTools,
        lifecycle_tools: LifecycleTools,
    ) -> None:
        """聚合一个固定片段可使用的全部工具。"""
        self.context_tools = context_tools
        self.entity_tools = entity_tools
        self.review_tools = review_tools
        self.relation_tools = relation_tools
        self.lifecycle_tools = lifecycle_tools
        schema = context_tools.context.schema
        # 工具参数枚举直接由当前 Schema 生成，在执行前即可拒绝非法类型。
        self._definitions = build_tool_specs(
            list(schema.entity_types),
            list(schema.relation_rules),
        )
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "get_chunk_context": context_tools.get_chunk_context,
            "get_workspace_summary": context_tools.get_workspace_summary,
            "list_entities": entity_tools.list_entities,
            "add_entity": entity_tools.add_entity,
            "update_entity": entity_tools.update_entity,
            "delete_entity": entity_tools.delete_entity,
            "list_review_warnings": review_tools.list_review_warnings,
            "confirm_review_warning": review_tools.confirm_review_warning,
            "list_relations": relation_tools.list_relations,
            "add_relation": relation_tools.add_relation,
            "update_relation": relation_tools.update_relation,
            "delete_relation": relation_tools.delete_relation,
            "confirm_empty_chunk": lifecycle_tools.confirm_empty_chunk,
            "validate_chunk": lifecycle_tools.validate_chunk,
            "submit_chunk": lifecycle_tools.submit_chunk,
        }

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """返回 OpenAI 兼容工具定义。"""
        return self._definitions

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """按名称调用当前片段工具。"""
        handler = self._handlers.get(name)
        if handler is None:
            return failure("UNKNOWN_TOOL", f"未知工具：{name}")
        # 所有模型输入先经过统一参数校验，具体工具只处理已验证的结构。
        argument_error = validate_tool_arguments(name, arguments, self._definitions)
        if argument_error is not None:
            return argument_error
        # 工具计时只覆盖本地逻辑，模型审查时间不计入慢操作日志。
        with log_slow_operation("tool.call", tool=name):
            return handler(**arguments)

    @property
    def review_available(self) -> bool:
        """返回是否可执行自动批量独立审查。"""
        return self.review_tools.available

    def review_entity_types(
        self,
        entities: list[dict[str, str]],
    ) -> dict[str, Any]:
        """在一次独立请求中审查一批已新增实体。"""
        return self.review_tools.review_entity_types(entities)

    def begin_batch(self) -> None:
        """开始一轮工具批处理并延迟草稿检查点写入。"""
        self.context_tools.context.begin_checkpoint_batch()

    def flush_batch(self) -> None:
        """将本轮工具的累积修改一次性写入草稿缓存。"""
        self.context_tools.context.flush_checkpoint_batch()

    @property
    def source_id(self) -> str:
        """返回内部绑定且不能由模型修改的来源 ID。"""
        return self.lifecycle_tools.context.workspace.source_id

    @property
    def committed(self) -> bool:
        """返回当前片段是否已经提交。"""
        return self.lifecycle_tools.context.committed

    @property
    def workspace(self) -> ChunkWorkspace:
        """返回主 Agent 重建有界上下文所需的当前工作区。"""
        return self.context_tools.context.workspace


def open_chunk_toolset(
    source_name: str,
    text: str,
    schema: GraphSchema,
    store: GraphStore,
    cache: WorkspaceCache,
    cache_namespace: str = "default",
    source_key: str | None = None,
    reviewer: SemanticReviewer | None = None,
) -> ChunkToolset:
    """创建绑定单个来源片段的完整工具集。"""
    context = open_tool_context(
        source_name,
        text,
        schema,
        store,
        cache,
        cache_namespace,
        source_key,
    )
    return ChunkToolset(
        ContextTools(context),
        EntityTools(context),
        ReviewTools(context, reviewer),
        RelationTools(context),
        LifecycleTools(context),
    )
