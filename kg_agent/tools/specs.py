"""OpenAI-compatible schemas for chunk tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    """构造一个 OpenAI 兼容函数工具定义。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


STRING = {"type": "string"}
PATCH = {"type": "object"}

TOOL_SPECS = [
    _tool("get_chunk_context", "读取当前片段文本"),
    _tool("get_workspace_summary", "读取当前工作区统计"),
    _tool(
        "list_entities",
        "查询当前片段实体",
        {"entity_id": STRING},
    ),
    _tool(
        "add_entity",
        "仅添加原文明确且可归入 Schema 类型的实体",
        {"name": STRING, "entity_type": STRING},
        ["name", "entity_type"],
    ),
    _tool(
        "update_entity",
        "更新当前片段实体",
        {"entity_id": STRING, "patch": PATCH},
        ["entity_id", "patch"],
    ),
    _tool(
        "delete_entity",
        "删除当前片段实体",
        {"entity_id": STRING},
        ["entity_id"],
    ),
    _tool("list_review_warnings", "列出当前片段尚未处理的审查分歧"),
    _tool(
        "confirm_review_warning",
        "核对原文后显式确认保留当前结果",
        {"warning_id": STRING, "reason": STRING},
        ["warning_id", "reason"],
    ),
    _tool(
        "list_relations",
        "查询当前片段关系",
        {"relation_id": STRING},
    ),
    _tool(
        "add_relation",
        "仅添加已通过类型审查、原文明示且精确符合 Schema 白名单的关系；实体真实类型不匹配时舍弃关系，不得改类型凑配对；不得硬套近似关系类型",
        {
            "subject_entity_name": STRING,
            "subject_entity_type": STRING,
            "relation_type": STRING,
            "object_entity_name": STRING,
            "object_entity_type": STRING,
        },
        [
            "subject_entity_name",
            "subject_entity_type",
            "relation_type",
            "object_entity_name",
            "object_entity_type",
        ],
    ),
    _tool(
        "update_relation",
        "更新当前片段关系",
        {"relation_id": STRING, "patch": PATCH},
        ["relation_id", "patch"],
    ),
    _tool(
        "delete_relation",
        "删除当前片段关系",
        {"relation_id": STRING},
        ["relation_id"],
    ),
    _tool(
        "confirm_empty_chunk",
        "确认当前片段没有任何可抽取实体或关系",
        {"reason": STRING},
        ["reason"],
    ),
    _tool("validate_chunk", "校验当前片段工作区"),
    _tool("submit_chunk", "提交当前片段"),
]


def build_tool_specs(
    entity_types: list[str],
    predicates: list[str],
) -> list[dict[str, Any]]:
    """构建带当前 Schema 参数类型和枚举值的工具定义。"""
    definitions = deepcopy(TOOL_SPECS)
    by_name = {
        item["function"]["name"]: item["function"]["parameters"]["properties"]
        for item in definitions
    }
    entity_type = {"type": "string", "enum": entity_types}
    predicate = {"type": "string", "enum": predicates}
    by_name["add_entity"]["entity_type"] = entity_type
    by_name["add_relation"]["subject_entity_type"] = entity_type
    by_name["add_relation"]["relation_type"] = predicate
    by_name["add_relation"]["object_entity_type"] = entity_type
    by_name["update_entity"]["patch"] = {
        "type": "object",
        "properties": {
            "name": STRING,
            "entity_type": entity_type,
        },
        "additionalProperties": False,
    }
    by_name["update_relation"]["patch"] = {
        "type": "object",
        "properties": {
            "subject_entity_name": STRING,
            "subject_entity_type": entity_type,
            "relation_type": predicate,
            "object_entity_name": STRING,
            "object_entity_type": entity_type,
        },
        "additionalProperties": False,
    }
    return definitions
