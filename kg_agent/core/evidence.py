"""Entity-name normalization and source-evidence validation."""

from __future__ import annotations

import unicodedata


SYMPTOM_GROUP_ENTITY_TYPE = "症状群"
SYMPTOM_GROUP_SEPARATOR = "|"


def normalize_entity_name(name: str, entity_type: str) -> str:
    """清理实体名称；症状群额外清理每个竖线分隔成员的首尾空白。"""
    normalized = name.strip()
    if (
        entity_type != SYMPTOM_GROUP_ENTITY_TYPE
        or SYMPTOM_GROUP_SEPARATOR not in normalized
    ):
        return normalized
    return SYMPTOM_GROUP_SEPARATOR.join(
        member.strip()
        for member in normalized.split(SYMPTOM_GROUP_SEPARATOR)
    )


def entity_evidence_error(
    name: str,
    entity_type: str,
    text: str,
) -> str | None:
    """忽略排版空白和零宽字符校验原文证据。"""
    comparable_text = _normalize_evidence_value(text)
    if entity_type != SYMPTOM_GROUP_ENTITY_TYPE:
        comparable_name = _normalize_evidence_value(name)
        return (
            None
            if comparable_name and comparable_name in comparable_text
            else "实体名称不在当前片段中"
        )
    members = name.split(SYMPTOM_GROUP_SEPARATOR)
    if len(members) < 2:
        return "症状群必须使用“症状1|症状2”格式并包含至少两个症状"
    if any(not member for member in members):
        return "症状群名称包含空的症状成员"
    if len(set(members)) != len(members):
        return "症状群不能包含重复的症状成员"
    missing = [
        member
        for member in members
        if _normalize_evidence_value(member) not in comparable_text
    ]
    if missing:
        return f"症状群成员不在当前片段中：{missing}"
    return None


def _normalize_evidence_value(value: str) -> str:
    """统一 Unicode 形式并删除空白及不可见格式字符。"""
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and unicodedata.category(character) != "Cf"
    )
