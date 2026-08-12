"""Bounded main-agent context rebuilt from the authoritative workspace."""

from __future__ import annotations

import json
from typing import Any

from ..core.workspace import ChunkWorkspace


WARNING_FIELDS = (
    "warning_id",
    "kind",
    "name",
    "current_entity_type",
    "recommended_entity_type",
    "entity_type",
    "confidence",
    "reason",
)


def build_initial_messages(
    system_prompt: str,
    text: str,
    workspace: ChunkWorkspace,
) -> list[dict[str, str]]:
    """构建首次模型请求，断点恢复时同时告知已有工作区。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    if _workspace_has_state(workspace):
        messages.append(
            {
                "role": "user",
                "content": build_workspace_checkpoint(workspace),
            }
        )
    return messages


def build_next_messages(
    system_prompt: str,
    text: str,
    workspace: ChunkWorkspace,
    assistant_response: str,
    observation: str,
) -> list[dict[str, str]]:
    """仅保留最近一轮交互并附加最新工作区，防止历史无界增长。"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
        {"role": "assistant", "content": assistant_response},
        {
            "role": "user",
            "content": (
                f"{observation}\n"
                f"{build_workspace_checkpoint(workspace)}"
            ),
        },
    ]


def build_workspace_checkpoint(workspace: ChunkWorkspace) -> str:
    """将当前实体、关系和待处理警告序列化为模型可见快照。"""
    entities = [
        {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "entity_type": entity.entity_type,
        }
        for entity in workspace.entities.values()
    ]
    relations = [
        _relation_checkpoint(workspace, relation)
        for relation in workspace.relations.values()
    ]
    payload: dict[str, Any] = {
        "entities": entities,
        "relations": relations,
        "validated": workspace.validated_revision == workspace.revision,
        "empty_confirmed": (
            workspace.empty_confirmed_revision == workspace.revision
        ),
    }
    if workspace.review_warnings:
        payload["review_warnings"] = [
            _compact_warning(warning)
            for warning in workspace.review_warnings.values()
        ]
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"✿WORKSPACE✿\n{content}"


def _workspace_has_state(workspace: ChunkWorkspace) -> bool:
    """判断工作区是否包含需在首轮显式告知模型的恢复状态。"""
    return bool(
        workspace.entities
        or workspace.relations
        or workspace.review_warnings
        or workspace.validated_revision is not None
        or workspace.empty_confirmed_revision is not None
    )


def _relation_checkpoint(workspace: ChunkWorkspace, relation: Any) -> dict:
    """将内部实体 ID 关系转为主 Agent 可直接操作的名称与类型。"""
    subject = workspace.entities[relation.subject_entity_id]
    obj = workspace.entities[relation.object_entity_id]
    return {
        "relation_id": relation.relation_id,
        "subject_entity_name": subject.name,
        "subject_entity_type": subject.entity_type,
        "relation_type": relation.predicate,
        "object_entity_name": obj.name,
        "object_entity_type": obj.entity_type,
    }


def _compact_warning(warning: dict[str, Any]) -> dict[str, Any]:
    """仅保留主 Agent 修正或确认审查警告所需字段。"""
    return {
        field: warning[field]
        for field in WARNING_FIELDS
        if warning.get(field) is not None
    }
