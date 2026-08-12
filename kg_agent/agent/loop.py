"""Constrained ReAct loop for one source chunk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context_window import build_initial_messages, build_next_messages
from .protocol import parse_action
from .llm import ChatClient
from .prompts import (
    build_observations,
    build_parse_feedback,
    build_result,
    build_system_prompt,
)
from ..core.schema import GraphSchema
from ..tools.toolset import ChunkToolset


@dataclass(frozen=True)
class AgentSettings:
    max_steps: int = 80
    debug: bool = False


@dataclass(frozen=True)
class AgentRunResult:
    status: str
    source_id: str
    steps: int
    tool_calls: int
    parse_errors: int
    last_result: dict[str, Any] | None = None


class ReActChunkAgent:
    def __init__(
        self,
        client: ChatClient,
        schema: GraphSchema,
        settings: AgentSettings | None = None,
    ) -> None:
        """创建绑定模型客户端和 Schema 的片段智能体。"""
        self.client = client
        self.schema = schema
        self.settings = settings or AgentSettings()

    @staticmethod
    def _print_prompt(messages: list[dict[str, str]]) -> None:
        """打印本轮实际发送给模型的完整消息列表。"""
        print("✿PROMPT✿", flush=True)
        for message in messages:
            print(f"[{message['role']}]", flush=True)
            print(message["content"], flush=True)

    def run(self, toolset: ChunkToolset) -> AgentRunResult:
        """迭代调用模型和工具，直到片段提交或达到步数上限。"""
        # 数据库已经存在该 source_id 时直接跳过模型，实现目录级断点续跑。
        if toolset.committed:
            return AgentRunResult(
                status="already_committed",
                source_id=toolset.source_id,
                steps=0,
                tool_calls=0,
                parse_errors=0,
            )
        chunk = toolset.call("get_chunk_context", {})
        system_prompt = build_system_prompt(self.schema)
        user_prompt = chunk["text"]
        messages = build_initial_messages(
            system_prompt,
            user_prompt,
            toolset.workspace,
        )
        tool_calls = 0
        parse_errors = 0
        last_result: dict[str, Any] | None = None
        for step in range(1, self.settings.max_steps + 1):
            if self.settings.debug:
                print(f"\n--- ReAct step {step} ---", flush=True)
                self._print_prompt(messages)
            raw = self.client.complete(messages)
            messages.append({"role": "assistant", "content": raw})
            parsed = parse_action(raw)
            parse_errors += len(parsed.errors)
            if not parsed.actions:
                messages = build_next_messages(
                    system_prompt,
                    user_prompt,
                    toolset.workspace,
                    raw,
                    build_parse_feedback(parsed.error or ""),
                )
                continue
            observation_results = [
                (
                    "protocol_error",
                    {
                        "ok": False,
                        "error": {
                            "code": "INVALID_ACTION_LINE",
                            "message": error,
                        },
                    },
                )
                for error in parsed.errors
            ]
            batch_failed = bool(parsed.errors)
            committed_result = None
            added_entities: dict[tuple[str, str], dict[str, str]] = {}
            # 同一轮的多个修改先写内存，轮次结束时只落一次 diskcache。
            toolset.begin_batch()
            for action in parsed.actions:
                if action.tool in {
                    "add_relation",
                    "update_relation",
                    "submit_chunk",
                }:
                    # 关系和提交动作前先完成本批实体审查，避免先凑关系再倒推类型。
                    batch_failed = _append_entity_type_review(
                        toolset,
                        added_entities,
                        observation_results,
                    ) or batch_failed
                    toolset.flush_batch()
                tool_calls += 1
                if action.tool == "submit_chunk" and batch_failed:
                    # JSONL 中任一动作失败时保留其他成功修改，但禁止本轮继续提交。
                    last_result = {
                        "ok": False,
                        "error": {
                            "code": "BATCH_HAS_FAILURES",
                            "message": "本轮存在失败，查看当前工作区并修正后才能提交",
                        },
                    }
                else:
                    last_result = toolset.call(action.tool, action.arguments)
                observation_results.append((action.tool, last_result))
                if (
                    action.tool == "add_entity"
                    and last_result.get("ok")
                    and last_result.get("status") == "created"
                ):
                    # 收集整批新增实体，稍后合并成一次独立类型审查请求。
                    key = (
                        action.arguments["name"].strip(),
                        action.arguments["entity_type"],
                    )
                    added_entities[key] = {
                        "name": key[0],
                        "entity_type": key[1],
                    }
                if (
                    action.tool == "update_entity"
                    and last_result.get("ok")
                    and last_result.get("identity_changed")
                ):
                    # 改名或改型等同于新实体，后续关系使用前必须重新独立审查。
                    key = (
                        last_result["name"],
                        last_result["entity_type"],
                    )
                    added_entities[key] = {
                        "name": key[0],
                        "entity_type": key[1],
                    }
                batch_failed = batch_failed or _tool_result_failed(
                    action.tool,
                    last_result,
                )
                if action.tool == "submit_chunk":
                    if (
                        last_result.get("ok")
                        and last_result.get("status")
                        in {"committed", "already_committed"}
                    ):
                        committed_result = last_result
                    break
            batch_failed = _append_entity_type_review(
                toolset,
                added_entities,
                observation_results,
            ) or batch_failed
            toolset.flush_batch()
            observation = build_observations(observation_results)
            if self.settings.debug:
                print(observation, flush=True)
            if committed_result is not None:
                if self.settings.debug:
                    print(build_result(committed_result), flush=True)
                return AgentRunResult(
                    status=committed_result["status"],
                    source_id=toolset.source_id,
                    steps=step,
                    tool_calls=tool_calls,
                    parse_errors=parse_errors,
                    last_result=committed_result,
                )
            # 工作区已包含所有成功修改，旧对话不再重复发送。
            messages = build_next_messages(
                system_prompt,
                user_prompt,
                toolset.workspace,
                raw,
                observation,
            )
        return AgentRunResult(
            status="max_steps_exceeded",
            source_id=toolset.source_id,
            steps=self.settings.max_steps,
            tool_calls=tool_calls,
            parse_errors=parse_errors,
            last_result=last_result,
        )


def _tool_result_failed(tool: str, result: dict[str, Any]) -> bool:
    """判断工具结果是否要求模型先检查并修复当前工作区。"""
    return not result.get("ok") or (
        tool == "validate_chunk" and not result.get("valid", False)
    )


def _append_entity_type_review(
    toolset: ChunkToolset,
    added_entities: dict[tuple[str, str], dict[str, str]],
    observation_results: list[tuple[str, dict[str, Any]]],
) -> bool:
    """用一次独立请求审查当前批次新增实体并追加 Observation。"""
    if not added_entities or not toolset.review_available:
        return False
    result = toolset.review_entity_types(list(added_entities.values()))
    observation_results.append(("review_entity_types", result))
    added_entities.clear()
    return _tool_result_failed("review_entity_types", result)
