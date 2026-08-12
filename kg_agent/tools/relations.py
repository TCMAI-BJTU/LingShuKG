"""CRUD tools for relations in the current chunk only."""

from __future__ import annotations

from typing import Any

from ..core.models import EntityDraft, RelationDraft
from .common import failure, success
from .tool_context import ToolContext


class RelationTools:
    def __init__(self, context: ToolContext) -> None:
        self.context = context

    def list_relations(self, relation_id: str | None = None) -> dict[str, Any]:
        relations = self.context.workspace.relations
        if relation_id is not None:
            relation = relations.get(relation_id)
            if relation is None:
                return failure("RELATION_NOT_FOUND", "当前片段中不存在该关系")
            return success(relations=[self._to_tool_dict(relation)])
        return success(
            relations=[self._to_tool_dict(relation) for relation in relations.values()]
        )

    def _check(
        self,
        subject: EntityDraft,
        relation_type: str,
        obj: EntityDraft,
    ) -> dict[str, Any] | None:
        pending = [
            warning
            for warning in self.context.workspace.review_warnings.values()
            if warning.get("entity_id")
            in {subject.entity_id, obj.entity_id}
            and warning.get("kind")
            in {
                "entity_type",
                "entity_not_extractable",
                "entity_type_review_missing",
            }
        ]
        if pending:
            return failure(
                "ENTITY_TYPE_REVIEW_PENDING",
                "关系实体尚未通过类型审查；请先修正或删除实体，不能为凑关系修改类型",
                warning_ids=[
                    warning["warning_id"] for warning in pending
                ],
            )
        rule = self.context.schema.relation_rule(relation_type)
        if rule is None:
            return failure("UNKNOWN_RELATION_TYPE", f"未知关系类型：{relation_type}")
        if not rule.allows(subject.entity_type, obj.entity_type):
            return failure("INVALID_RELATION_PAIR", "关系头尾实体类型配对不符合约束")
        # Only Schema structure is checked here; medical facts still need agent review against the text.
        return None

    def _find_entity(self, name: str, entity_type: str) -> EntityDraft | None:
        return next(
            (
                entity
                for entity in self.context.workspace.entities.values()
                if entity.name == name and entity.entity_type == entity_type
            ),
            None,
        )

    def _to_tool_dict(self, relation: RelationDraft) -> dict[str, str]:
        subject = self.context.workspace.entities[relation.subject_entity_id]
        obj = self.context.workspace.entities[relation.object_entity_id]
        return {
            "relation_id": relation.relation_id,
            "subject_entity_name": subject.name,
            "subject_entity_type": subject.entity_type,
            "relation_type": relation.predicate,
            "object_entity_name": obj.name,
            "object_entity_type": obj.entity_type,
        }

    def add_relation(
        self,
        subject_entity_name: str,
        subject_entity_type: str,
        relation_type: str,
        object_entity_name: str,
        object_entity_type: str,
    ) -> dict[str, Any]:
        """Resolve nodes by name+type and add a relation draft."""
        if self.context.committed:
            return failure("CHUNK_COMMITTED", "当前片段已经提交")
        subject = self._find_entity(subject_entity_name, subject_entity_type)
        obj = self._find_entity(object_entity_name, object_entity_type)
        if subject is None or obj is None:
            return failure("ENTITY_NOT_FOUND", "关系引用的名称和类型在当前片段中不存在")
        error = self._check(subject, relation_type, obj)
        if error is not None:
            return error
        for relation in self.context.workspace.relations.values():
            if (
                relation.subject_entity_id == subject.entity_id
                and relation.predicate == relation_type
                and relation.object_entity_id == obj.entity_id
            ):
                return success(relation_id=relation.relation_id, status="existing")
        relation_id = self.context.workspace.next_relation_id()
        self.context.workspace.relations[relation_id] = RelationDraft(
            relation_id=relation_id,
            subject_entity_id=subject.entity_id,
            predicate=relation_type,
            object_entity_id=obj.entity_id,
        )
        self.context.mark_changed()
        return success(relation_id=relation_id, status="created")

    def update_relation(
        self,
        relation_id: str,
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        if self.context.committed:
            return failure("CHUNK_COMMITTED", "当前片段已经提交")
        relation = self.context.workspace.relations.get(relation_id)
        if relation is None:
            return failure("RELATION_NOT_FOUND", "当前片段中不存在该关系")
        allowed = {
            "subject_entity_name",
            "subject_entity_type",
            "relation_type",
            "object_entity_name",
            "object_entity_type",
        }
        unknown = set(patch) - allowed
        if unknown:
            return failure("INVALID_PATCH", f"不可修改字段：{sorted(unknown)}")
        current_subject = self.context.workspace.entities[relation.subject_entity_id]
        current_object = self.context.workspace.entities[relation.object_entity_id]
        subject = self._find_entity(
            str(patch.get("subject_entity_name", current_subject.name)),
            str(patch.get("subject_entity_type", current_subject.entity_type)),
        )
        obj = self._find_entity(
            str(patch.get("object_entity_name", current_object.name)),
            str(patch.get("object_entity_type", current_object.entity_type)),
        )
        if subject is None or obj is None:
            return failure("ENTITY_NOT_FOUND", "关系引用的名称和类型在当前片段中不存在")
        relation_type = str(patch.get("relation_type", relation.predicate))
        error = self._check(subject, relation_type, obj)
        if error is not None:
            return error
        relation.subject_entity_id = subject.entity_id
        relation.predicate = relation_type
        relation.object_entity_id = obj.entity_id
        self.context.mark_changed()
        return success(relation_id=relation_id, status="updated")

    def delete_relation(self, relation_id: str) -> dict[str, Any]:
        if self.context.committed:
            return failure("CHUNK_COMMITTED", "当前片段已经提交")
        if relation_id not in self.context.workspace.relations:
            return failure("RELATION_NOT_FOUND", "当前片段中不存在该关系")
        del self.context.workspace.relations[relation_id]
        self.context.mark_changed()
        return success(relation_id=relation_id, status="deleted")
