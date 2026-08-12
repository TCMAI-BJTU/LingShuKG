"""Configurable entity and relation constraints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EntityTypeGuidance:
    description: str
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...]


@dataclass(frozen=True)
class RelationRule:
    name: str
    allowed_pairs: frozenset[tuple[str, str]]

    def allows(self, head_type: str, tail_type: str) -> bool:
        """判断一个精确头尾类型配对是否被允许。"""
        return (head_type, tail_type) in self.allowed_pairs


class GraphSchema:
    def __init__(
        self,
        entity_types: tuple[str, ...],
        entity_guidance: dict[str, EntityTypeGuidance],
        relation_rules: dict[str, RelationRule],
    ) -> None:
        """保存实体类型、抽取说明和关系约束。"""
        self.entity_types = tuple(entity_types)
        self.entity_guidance = dict(entity_guidance)
        self.relation_rules = dict(relation_rules)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraphSchema":
        """从配置字典构建并校验 Schema。"""
        allowed_guidance_fields = {
            "description",
            "positive_examples",
            "negative_examples",
        }
        for entity_type, config in payload["entity_types"].items():
            unknown_fields = set(config) - allowed_guidance_fields
            if unknown_fields:
                # Schema 不接受已废弃字段，避免旧配置被静默忽略。
                raise ValueError(
                    f"实体类型 {entity_type!r} 包含未知字段："
                    f"{sorted(unknown_fields)}"
                )
        entity_types = tuple(str(entity_type) for entity_type in payload["entity_types"])
        entity_guidance = {
            entity_type: EntityTypeGuidance(
                description=str(config.get("description", "")),
                positive_examples=tuple(
                    str(example) for example in config.get("positive_examples", [])
                ),
                negative_examples=tuple(
                    str(example) for example in config.get("negative_examples", [])
                ),
            )
            for entity_type, config in payload["entity_types"].items()
        }
        relation_rules = {
            name: RelationRule(
                name=name,
                allowed_pairs=_build_relation_pairs(config),
            )
            for name, config in payload["relation_types"].items()
        }
        schema = cls(entity_types, entity_guidance, relation_rules)
        schema.validate()
        return schema

    @classmethod
    def from_json_file(cls, path: Path | str) -> "GraphSchema":
        """从 UTF-8 JSON 文件加载 Schema。"""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def validate(self) -> None:
        """校验实体类型和关系头尾约束。"""
        if not self.entity_types:
            raise ValueError("entity_types 不能为空")
        for relation in self.relation_rules.values():
            unknown = {
                entity_type
                for pair in relation.allowed_pairs
                for entity_type in pair
            } - set(self.entity_types)
            if unknown:
                raise ValueError(
                    f"关系 {relation.name!r} 使用了未知实体类型：{sorted(unknown)}"
                )

    def has_entity_type(self, entity_type: str) -> bool:
        """判断实体类型是否已在 Schema 中定义。"""
        return entity_type in self.entity_types

    def relation_rule(self, predicate: str) -> RelationRule | None:
        """按谓词名称查找关系约束。"""
        return self.relation_rules.get(predicate)

def _build_relation_pairs(config: dict[str, Any]) -> frozenset[tuple[str, str]]:
    """从 allowed_pairs 读取关系允许的精确头尾类型配对。"""
    return frozenset(
        (str(pair[0]), str(pair[1]))
        for pair in config["allowed_pairs"]
    )
