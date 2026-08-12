"""Mutable state for one source chunk."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import EntityDraft, RelationDraft


@dataclass
class ChunkWorkspace:
    source_id: str
    source_name: str
    text: str
    entities: dict[str, EntityDraft] = field(default_factory=dict)
    relations: dict[str, RelationDraft] = field(default_factory=dict)
    entity_sequence: int = 0
    relation_sequence: int = 0
    revision: int = 0
    validated_revision: int | None = None
    empty_confirmed_revision: int | None = None
    empty_reason: str | None = None
    review_warnings: dict[str, dict[str, Any]] = field(default_factory=dict)
    review_sequence: int = 0

    def next_entity_id(self) -> str:
        """生成当前工作区内唯一的实体本地 ID。"""
        self.entity_sequence += 1
        return f"e_{self.entity_sequence}"

    def next_relation_id(self) -> str:
        """生成当前工作区内唯一的关系本地 ID。"""
        self.relation_sequence += 1
        return f"r_{self.relation_sequence}"

    def set_review_warning(
        self,
        key: str,
        warning: dict[str, Any],
    ) -> dict[str, Any]:
        """新建或替换一条稳定键对应的待处理审查警告。"""
        existing = self.review_warnings.get(key)
        if existing is not None:
            # 同一问题重复审查时沿用 warning_id，便于断点恢复后继续处理。
            warning_id = existing["warning_id"]
        else:
            self.review_sequence += 1
            warning_id = f"w_{self.review_sequence}"
        item = {"warning_id": warning_id, **warning}
        self.review_warnings[key] = item
        return item

    def clear_review_warnings(
        self,
        entity_id: str,
        kind: str | None = None,
    ) -> None:
        """清除指定实体的全部或特定类型审查警告。"""
        keys = [
            key
            for key, warning in self.review_warnings.items()
            if warning.get("entity_id") == entity_id
            and (kind is None or warning.get("kind") == kind)
        ]
        for key in keys:
            del self.review_warnings[key]

    def to_dict(self) -> dict[str, Any]:
        """序列化完整工作区以供断点恢复。"""
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "text": self.text,
            "entities": {
                key: entity.to_dict() for key, entity in self.entities.items()
            },
            "relations": {
                key: relation.to_dict() for key, relation in self.relations.items()
            },
            "entity_sequence": self.entity_sequence,
            "relation_sequence": self.relation_sequence,
            "revision": self.revision,
            "validated_revision": self.validated_revision,
            "empty_confirmed_revision": self.empty_confirmed_revision,
            "empty_reason": self.empty_reason,
            "review_warnings": self.review_warnings,
            "review_sequence": self.review_sequence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkWorkspace":
        """从缓存字典恢复工作区。"""
        return cls(
            source_id=payload["source_id"],
            source_name=payload["source_name"],
            text=payload["text"],
            entities={
                key: EntityDraft(**value)
                for key, value in payload["entities"].items()
            },
            relations={
                key: RelationDraft(**value)
                for key, value in payload["relations"].items()
            },
            entity_sequence=int(payload["entity_sequence"]),
            relation_sequence=int(payload["relation_sequence"]),
            revision=int(payload["revision"]),
            validated_revision=payload["validated_revision"],
            empty_confirmed_revision=payload["empty_confirmed_revision"],
            empty_reason=payload["empty_reason"],
            review_warnings=dict(payload["review_warnings"]),
            review_sequence=int(payload["review_sequence"]),
        )
