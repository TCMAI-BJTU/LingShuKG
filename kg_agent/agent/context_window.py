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
    """Build the first model request; include existing workspace state on resume."""
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
    """Keep only the latest turn plus the newest workspace to bound history growth."""
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
    """Serialize entities, relations, and pending warnings into a model-visible snapshot."""
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
    return bool(
        workspace.entities
        or workspace.relations
        or workspace.review_warnings
        or workspace.validated_revision is not None
        or workspace.empty_confirmed_revision is not None
    )


def _relation_checkpoint(workspace: ChunkWorkspace, relation: Any) -> dict:
    """Map internal entity-ID relations to name+type structures the agent can edit."""
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
    return {
        field: warning[field]
        for field in WARNING_FIELDS
        if warning.get(field) is not None
    }
