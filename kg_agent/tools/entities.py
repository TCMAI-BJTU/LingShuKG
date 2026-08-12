"""CRUD tools for entities in the current chunk only."""

from __future__ import annotations

from typing import Any

from ..core.evidence import entity_evidence_error, normalize_entity_name
from ..core.models import EntityDraft
from .common import failure, success
from .tool_context import ToolContext


class EntityTools:
    def __init__(self, context: ToolContext) -> None:
        """绑定当前片段工具上下文。"""
        self.context = context

    def list_entities(self, entity_id: str | None = None) -> dict[str, Any]:
        """列出当前片段全部实体或一个指定实体。"""
        entities = self.context.workspace.entities
        if entity_id is not None:
            entity = entities.get(entity_id)
            if entity is None:
                return failure("ENTITY_NOT_FOUND", "当前片段中不存在该实体")
            return success(entities=[entity.to_dict()])
        return success(entities=[entity.to_dict() for entity in entities.values()])

    def add_entity(self, name: str, entity_type: str) -> dict[str, Any]:
        """校验名称证据并向当前工作区添加实体草稿。"""
        if self.context.committed:
            return failure("CHUNK_COMMITTED", "当前片段已经提交")
        normalized_name = normalize_entity_name(name, entity_type)
        if not normalized_name:
            return failure("EMPTY_NAME", "实体名称不能为空")
        if not self.context.schema.has_entity_type(entity_type):
            return failure("UNKNOWN_ENTITY_TYPE", f"未知实体类型：{entity_type}")
        # 症状群是组合实体，校验每个成员；其他实体仍校验完整名称。
        evidence_error = entity_evidence_error(
            normalized_name,
            entity_type,
            self.context.workspace.text,
        )
        if evidence_error is not None:
            return failure("INVALID_EVIDENCE", evidence_error)
        # 关系工具通过 name + type 定位节点，因此当前 chunk 内只保留一条。
        for entity in self.context.workspace.entities.values():
            if (
                entity.name == normalized_name
                and entity.entity_type == entity_type
            ):
                return success(
                    entity_id=entity.entity_id,
                    status="existing",
                )
        entity_id = self.context.workspace.next_entity_id()
        self.context.workspace.entities[entity_id] = EntityDraft(
            entity_id=entity_id,
            name=normalized_name,
            entity_type=entity_type,
        )
        self.context.mark_changed()
        return success(
            entity_id=entity_id,
            status="created",
        )

    def update_entity(self, entity_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """按补丁更新当前工作区中的实体。"""
        if self.context.committed:
            return failure("CHUNK_COMMITTED", "当前片段已经提交")
        entity = self.context.workspace.entities.get(entity_id)
        if entity is None:
            return failure("ENTITY_NOT_FOUND", "当前片段中不存在该实体")
        allowed = {"name", "entity_type"}
        unknown = set(patch) - allowed
        if unknown:
            return failure("INVALID_PATCH", f"不可修改字段：{sorted(unknown)}")
        entity_type = str(patch.get("entity_type", entity.entity_type))
        name = normalize_entity_name(
            str(patch.get("name", entity.name)),
            entity_type,
        )
        if not name:
            return failure("EMPTY_NAME", "实体名称不能为空")
        if not self.context.schema.has_entity_type(entity_type):
            return failure("UNKNOWN_ENTITY_TYPE", f"未知实体类型：{entity_type}")
        evidence_error = entity_evidence_error(
            name,
            entity_type,
            self.context.workspace.text,
        )
        if evidence_error is not None:
            return failure("INVALID_EVIDENCE", evidence_error)
        duplicate = next(
            (
                current.entity_id
                for current in self.context.workspace.entities.values()
                if current.entity_id != entity_id
                and current.name == name
                and current.entity_type == entity_type
            ),
            None,
        )
        if duplicate is not None:
            return failure(
                "DUPLICATE_ENTITY",
                "当前片段已存在相同名称和类型的实体",
                entity_id=duplicate,
            )
        identity_changed = entity.name != name or entity.entity_type != entity_type
        if identity_changed:
            self.context.workspace.clear_review_warnings(entity_id)
        entity.name = name
        entity.entity_type = entity_type
        self.context.mark_changed()
        return success(
            entity_id=entity_id,
            name=entity.name,
            entity_type=entity.entity_type,
            identity_changed=identity_changed,
            status="updated",
        )

    def delete_entity(self, entity_id: str) -> dict[str, Any]:
        """删除未被当前关系引用的实体。"""
        if self.context.committed:
            return failure("CHUNK_COMMITTED", "当前片段已经提交")
        if entity_id not in self.context.workspace.entities:
            return failure("ENTITY_NOT_FOUND", "当前片段中不存在该实体")
        related = [
            relation.relation_id
            for relation in self.context.workspace.relations.values()
            if entity_id
            in {relation.subject_entity_id, relation.object_entity_id}
        ]
        if related:
            return failure(
                "ENTITY_IN_USE",
                "实体仍被当前片段中的关系引用",
                related_relation_ids=related,
            )
        self.context.workspace.clear_review_warnings(entity_id)
        del self.context.workspace.entities[entity_id]
        self.context.mark_changed()
        return success(entity_id=entity_id, status="deleted")
