"""Prompts for the constrained ReAct extraction agent."""

from __future__ import annotations

import json

from ..core.schema import GraphSchema
from ..tools.specs import build_tool_specs


# Omit internal get_chunk_context from prompts to reduce irrelevant tool surface.
MODEL_TOOL_NAMES = {
    "get_workspace_summary",
    "list_entities",
    "add_entity",
    "update_entity",
    "delete_entity",
    "list_review_warnings",
    "confirm_review_warning",
    "list_relations",
    "add_relation",
    "update_relation",
    "delete_relation",
    "confirm_empty_chunk",
    "validate_chunk",
    "submit_chunk",
}


def build_system_prompt(schema: GraphSchema) -> str:
    """Build the system prompt with Schema, tools, and extraction workflow only."""
    entities = "\n".join(
        _format_entity_type(schema, entity_type)
        for entity_type in schema.entity_types
    )
    relations = "\n".join(
        f"- {name}: "
        + "；".join(
            f"{head} -> {tail}"
            for head, tail in sorted(rule.allowed_pairs)
        )
        for name, rule in schema.relation_rules.items()
    )
    tools = "\n".join(
        _format_tool(item["function"])
        for item in build_tool_specs(
            list(schema.entity_types),
            list(schema.relation_rules),
        )
        if item["function"]["name"] in MODEL_TOOL_NAMES
    )
    evidence_rule = (
        "实体名称必须逐字出现在片段中。症状群是唯一例外：名称必须使用"
        "“症状1|症状2”格式，完整组合名可以不在原文中，但每个成员都必须"
        "逐字出现在同一语义单元中。"
        if schema.has_entity_type("症状群")
        else "实体名称必须逐字出现在片段中。"
    )
    return f"""你是医学知识图谱抽取智能体。请根据当前片段，通过工具维护实体和关系。

每轮严格使用以下格式。THOUGHT 和 ACTION 标记都必须且只能出现一次；不要输出 Markdown、JSON 数组或完整结果；ACTION 只接受 JSONL，多个工具调用共用一个 ACTION 标记，每个完整 JSON 对象必须压缩在一个物理行内，不得包含空行或行末逗号：
✿THOUGHT✿
一句简短决策摘要，不要展开推理。
✿ACTION✿
{{"tool":"工具名","arguments":{{}}}}
{{"tool":"工具名","arguments":{{}}}}
动作顶层只允许 tool 和 arguments；所有工具参数都必须写在 arguments 对象内。
工具观察中，单次调用使用 result，连续同名调用会合并为 results 数组；整组成功时 ok=true 只在组级出现。
对话历史会被裁剪；每轮 ✿WORKSPACE✿ 是工具执行后的当前完整工作区，应以它为准，不要重复添加已有项。

实体类型：
{entities}

关系约束：
{relations or '- 当前 Schema 不允许关系'}

可用工具：
{tools}

执行规则：
1. Schema 是封闭白名单。先按实体本身含义确定类型，再判断是否存在允许的关系；不得根据想建立的关系反推或修改实体类型。真实类型无法组成白名单配对时舍弃关系，不得为了保留原文信息而硬套最相近的实体或关系类型。
2. 只抽取当前片段明确支持的实体和关系，不使用外部常识补充事实。并列、分型、背景介绍或学术观点不自动构成关系。
3. {evidence_rule}只使用工具定义的参数。患者、病例、受试者、人群以及“无症状”“未见异常”等否定、缺失或正常状态不是医学实体；独立审查返回 should_extract=false 时必须删除该实体。
4. 关系必须引用已添加且已通过独立类型审查的实体，并使用其真实的 name 和 entity_type；没有精确符合 Schema 且有原文依据的关系时直接忽略。禁止为了获得“疾病-表现为-症状”等配对，把另一个疾病、综合征或功能障碍改标为症状。
5. 非空片段依次完成实体和关系处理。空片段先调用 confirm_empty_chunk，reason 必须写在 arguments 内。调用失败时应修正格式或参数；禁止为绕过空片段确认、校验或提交限制而临时添加、修改或删除实体。
6. 提交前重新查看实体、关系和审查警告，再调用 validate_chunk；修改后必须重新校验。不要把结构校验通过视为语义正确。
7. 工具或协议失败后，下一轮先用 get_workspace_summary、list_entities 和 list_relations 核对已成功的结果，再修正。
8. submit_chunk 不能是第一个动作，且必须是本轮最后一个动作；提交成功后停止。"""


def build_observations(results: list[tuple[str, dict]]) -> str:
    """Merge this round's tool responses into one Observation array."""
    payload = _merge_observation_items(
        [
            _observation_item(tool, result)
            for tool, result in results
        ]
    )
    if any(_observation_failed(tool, result) for tool, result in results):
        # JSONL batches may partially succeed; nudge the agent to read the real workspace before recovery.
        payload.append(
            {
                "tool": "recovery_hint",
                "result": {
                    "required": True,
                    "inspect_tools": [
                        "get_workspace_summary",
                        "list_entities",
                        "list_relations",
                    ],
                },
            }
        )
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"✿OBSERVATION✿\n{content}"


def build_parse_feedback(error: str) -> str:
    """Build corrective feedback after action-parse failure."""
    return f"""✿OBSERVATION✿
tool=protocol_error
{error}。
当前工作区可能已有此前成功结果；先调用 get_workspace_summary、list_entities 和 list_relations 查看当前抽取状态。
每轮只能输出一个 ✿THOUGHT✿ 和一个 ✿ACTION✿；多个工具动作紧跟同一个 ✿ACTION✿。
ACTION 仅接受 JSONL：每个完整 JSON 对象必须压缩在一个物理行内，不接受数组、多行 JSON、空行或代码围栏。
请输出：
✿THOUGHT✿
一句简短决策摘要
✿ACTION✿
{{"tool":"工具名","arguments":{{}}}}"""


def build_result(result: dict) -> str:
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    return f"✿RESULT✿\n{payload}"


def _format_tool(function: dict) -> str:
    parameters = function["parameters"]
    properties = parameters["properties"]
    required = set(parameters["required"])
    arguments = [
        f"{name}: {_format_parameter(definition)} "
        f"({'必填' if name in required else '可选'})"
        for name, definition in properties.items()
    ]
    return f"- {function['name']}({', '.join(arguments)}): {function['description']}"


def _observation_failed(tool: str, result: dict) -> bool:
    return not result.get("ok") or (
        tool == "validate_chunk" and not result.get("valid", False)
    )


def _observation_item(tool: str, result: dict) -> dict:
    item = {
        "tool": tool,
        "result": _compact_observation_result(tool, result),
    }
    hint = _agent_hint(tool, result)
    if hint is not None:
        item["agent_hint"] = hint
    return item


def _compact_observation_result(tool: str, result: dict) -> dict:
    """Drop success fields the agent already knows or can infer from the summary."""
    compacted = dict(result)
    if not result.get("ok"):
        return compacted
    if tool in {"list_entities", "list_relations", "list_review_warnings"}:
        # The following WORKSPACE snapshot already includes the same full payload.
        return {"ok": True, "status": "workspace_refreshed"}
    if tool == "review_entity_types":
        reviews = compacted.pop("reviews", [])
        issues = [
            review
            for review in reviews
            if (
                not review.get("should_extract", True)
                or not review.get("consistent", False)
            )
        ]
        if issues:
            compacted["review_issues"] = issues
        else:
            compacted["all_consistent"] = True
        for field in (
            "inconsistent_count",
            "not_extractable_count",
            "not_extractable_names",
        ):
            if not compacted.get(field):
                compacted.pop(field, None)
    if tool == "validate_chunk":
        if not compacted.get("errors"):
            compacted.pop("errors", None)
        # Warning details live in WORKSPACE; Observation keeps only the count.
        compacted.pop("review_warnings", None)
    if tool == "confirm_empty_chunk":
        compacted.pop("reason", None)
    return compacted


def _merge_observation_items(items: list[dict]) -> list[dict]:
    """Merge consecutive same-named tool results and dedupe repeated agent hints."""
    merged: list[dict] = []
    for item in items:
        if not merged or merged[-1]["tool"] != item["tool"]:
            merged.append(item)
            continue
        group = merged[-1]
        if "results" not in group:
            group["results"] = [group.pop("result")]
        group["results"].append(item["result"])
        hints = [
            hint
            for hint in (
                group.pop("agent_hint", None),
                *group.pop("agent_hints", []),
                item.get("agent_hint"),
            )
            if hint is not None
        ]
        unique_hints = list(dict.fromkeys(hints))
        if len(unique_hints) == 1:
            group["agent_hint"] = unique_hints[0]
        elif unique_hints:
            group["agent_hints"] = unique_hints
    return [
        _hoist_group_success(item)
        for item in merged
    ]


def _hoist_group_success(item: dict) -> dict:
    """When every item in a group succeeds, keep ok=true only at group level."""
    results = item.get("results")
    if not results or not all(result.get("ok") is True for result in results):
        return item
    grouped = {
        "tool": item["tool"],
        "ok": True,
        "results": [
            {
                key: value
                for key, value in result.items()
                if key != "ok"
            }
            for result in results
        ],
    }
    if "agent_hint" in item:
        grouped["agent_hint"] = item["agent_hint"]
    if "agent_hints" in item:
        grouped["agent_hints"] = item["agent_hints"]
    return grouped


def _agent_hint(tool: str, result: dict) -> str | None:
    error = result.get("error", {})
    if tool == "review_entity_types" and error.get("model_output"):
        return (
            "独立审查器已自行重试但仍失败；最后一次模型返回的有界前缀位于 "
            "error.model_output。这不是本工具 arguments 格式错误。请处理当前审查警告，"
            "不要原样重复调用。"
        )
    if tool == "list_entities" and result.get("ok"):
        return "请对照原文检查实体是否遗漏或误抽，并确认每个实体的类型符合其真实含义。"
    if tool == "list_relations" and result.get("ok"):
        return "请逐条对照原文和 Schema，删除缺少明确事实依据、方向错误、超出白名单或为保留信息而硬套近似类型的关系。"
    if tool == "validate_chunk" and result.get("review_warnings"):
        return "结构校验已完成，但仍有独立审查分歧；修正结果或核对原文后显式确认，否则不能提交。"
    if tool == "validate_chunk" and result.get("valid"):
        return "结构校验已通过，但不代表医学语义正确；提交前请删除无法归入 Schema 的实体，以及超出白名单或硬套近似类型的关系。"
    if tool == "review_entity_types" and result.get("not_extractable_count"):
        names = "、".join(result.get("not_extractable_names", []))
        return f"独立审查判定以下名称不应作为实体抽取：{names}。请调用 delete_entity 删除，不能强行改成其他类型。"
    if tool == "review_entity_types" and result.get("ok"):
        count = result.get("inconsistent_count", 0)
        if count:
            return f"本批有 {count} 个实体类型与独立审查不一致；请逐个对照原文修正或显式确认。"
        return "本批新增实体的独立类型审查均与当前类型一致。"
    if tool == "list_review_warnings" and result.get("warning_count"):
        return "当前仍有未处理审查分歧，不得提交。"
    return None


def _format_entity_type(schema: GraphSchema, entity_type: str) -> str:
    guidance = schema.entity_guidance[entity_type]
    lines = [f"- {entity_type}"]
    if guidance.description:
        lines.append(f"  定义：{guidance.description}")
    if guidance.positive_examples:
        lines.append(f"  正例：{'、'.join(guidance.positive_examples)}")
    if guidance.negative_examples:
        lines.append(f"  反例：{'、'.join(guidance.negative_examples)}")
    return "\n".join(lines)


def _format_parameter(definition: dict) -> str:
    parts = [definition.get("type", "any")]
    if "enum" in definition:
        values = json.dumps(definition["enum"], ensure_ascii=False)
        parts.append(f"可选值={values}")
    if "minimum" in definition:
        parts.append(f"最小值={definition['minimum']}")
    if "maximum" in definition:
        parts.append(f"最大值={definition['maximum']}")
    if "properties" in definition:
        fields = ", ".join(
            f"{name}: {_format_parameter(field)}"
            for name, field in definition["properties"].items()
        )
        parts.append(f"字段={{{fields}}}")
    return " ".join(parts)
