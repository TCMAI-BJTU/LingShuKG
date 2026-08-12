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
        self.client = client
        self.schema = schema
        self.settings = settings or AgentSettings()

    @staticmethod
    def _print_prompt(messages: list[dict[str, str]]) -> None:
        print("✿PROMPT✿", flush=True)
        for message in messages:
            print(f"[{message['role']}]", flush=True)
            print(message["content"], flush=True)

    def run(self, toolset: ChunkToolset) -> AgentRunResult:
        """Iterate model and tools until the chunk is submitted or the step limit is hit."""
        # Skip the model when this source_id is already committed (directory-level resume).
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
            # Keep in-round edits in memory; flush diskcache once at round end.
            toolset.begin_batch()
            for action in parsed.actions:
                if action.tool in {
                    "add_relation",
                    "update_relation",
                    "submit_chunk",
                }:
                    # Finish entity-type review before relations/submit so types are not reverse-engineered for edges.
                    batch_failed = _append_entity_type_review(
                        toolset,
                        added_entities,
                        observation_results,
                    ) or batch_failed
                    toolset.flush_batch()
                tool_calls += 1
                if action.tool == "submit_chunk" and batch_failed:
                    # Keep successful edits on partial JSONL failure, but block submit in this round.
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
                    # Collect newly added entities for one batched independent type review.
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
                    # Rename/retype is treated as a new entity and must be re-reviewed before relations.
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
            # Workspace already holds successful edits; do not resend old dialogue.
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
    return not result.get("ok") or (
        tool == "validate_chunk" and not result.get("valid", False)
    )


def _append_entity_type_review(
    toolset: ChunkToolset,
    added_entities: dict[tuple[str, str], dict[str, str]],
    observation_results: list[tuple[str, dict[str, Any]]],
) -> bool:
    """Review newly added entities in one independent request and append an Observation."""
    if not added_entities or not toolset.review_available:
        return False
    result = toolset.review_entity_types(list(added_entities.values()))
    observation_results.append(("review_entity_types", result))
    added_entities.clear()
    return _tool_result_failed("review_entity_types", result)
