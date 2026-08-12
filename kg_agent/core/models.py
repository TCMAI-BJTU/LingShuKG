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
        return asdict(self)


@dataclass
class RelationDraft:
    relation_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    item_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)
