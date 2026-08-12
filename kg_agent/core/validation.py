"""Workspace validation before persistence."""

from __future__ import annotations

from .evidence import entity_evidence_error
from .models import ValidationIssue
from .schema import GraphSchema
from .workspace import ChunkWorkspace


def validate_workspace(
    workspace: ChunkWorkspace,
    schema: GraphSchema,
) -> list[ValidationIssue]:
    """Validate entities and relations in the current workspace."""
    # Structural checks only; medical semantics rely on independent review and the agent.
    issues: list[ValidationIssue] = []
    entity_keys: set[tuple[str, str]] = set()
    for entity in workspace.entities.values():
        if not entity.name.strip():
            issues.append(ValidationIssue("EMPTY_NAME", "实体名称为空", entity.entity_id))
        if not schema.has_entity_type(entity.entity_type):
            issues.append(
                ValidationIssue(
                    "UNKNOWN_ENTITY_TYPE",
                    f"未知实体类型：{entity.entity_type}",
                    entity.entity_id,
                )
            )
        evidence_error = entity_evidence_error(
            entity.name,
            entity.entity_type,
            workspace.text,
        )
        if evidence_error is not None:
            issues.append(
                ValidationIssue(
                    "INVALID_ENTITY_EVIDENCE",
                    evidence_error,
                    entity.entity_id,
                )
            )
        key = (entity.name, entity.entity_type)
        # Tools locate entities by name+type, so uniqueness is required within a chunk.
        if key in entity_keys:
            issues.append(
                ValidationIssue("DUPLICATE_ENTITY", "当前片段存在重复实体", entity.entity_id)
            )
        entity_keys.add(key)

    relation_keys: set[tuple[str, str, str]] = set()
    for relation in workspace.relations.values():
        subject = workspace.entities.get(relation.subject_entity_id)
        obj = workspace.entities.get(relation.object_entity_id)
        if subject is None or obj is None:
            issues.append(
                ValidationIssue(
                    "MISSING_RELATION_ENTITY",
                    "关系引用了不存在的实体",
                    relation.relation_id,
                )
            )
            continue
        rule = schema.relation_rule(relation.predicate)
        if rule is None:
            issues.append(
                ValidationIssue(
                    "UNKNOWN_RELATION_TYPE",
                    f"未知关系类型：{relation.predicate}",
                    relation.relation_id,
                )
            )
        elif not rule.allows(subject.entity_type, obj.entity_type):
            issues.append(
                ValidationIssue(
                    "RELATION_TYPE_MISMATCH",
                    "关系头尾实体类型不符合 Schema",
                    relation.relation_id,
                )
            )
        key = (
            relation.subject_entity_id,
            relation.predicate,
            relation.object_entity_id,
        )
        # Relation dedup uses the persisted (head, predicate, tail) triple.
        if key in relation_keys:
            issues.append(
                ValidationIssue(
                    "DUPLICATE_RELATION",
                    "当前片段存在重复关系",
                    relation.relation_id,
                )
            )
        relation_keys.add(key)
    return issues
