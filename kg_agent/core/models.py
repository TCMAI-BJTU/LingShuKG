"""In-memory extraction records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EntityDraft:
    entity_id: str
    name: str
    entity_type: str

    def to_dict(self) -> dict[str, Any]:
        """将实体草稿转换为可缓存字典。"""
        return asdict(self)


@dataclass
class RelationDraft:
    relation_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str

    def to_dict(self) -> dict[str, Any]:
        """将关系草稿转换为可缓存字典。"""
        return asdict(self)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    item_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """将校验问题转换为工具响应字典。"""
        return asdict(self)
