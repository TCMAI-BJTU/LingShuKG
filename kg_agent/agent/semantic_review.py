"""Context-aware LLM review for entity types."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from threading import Lock
from typing import Any, Iterator, Protocol

from .jsonl import decode_jsonl_objects
from .llm import ChatClient
from ..core.schema import GraphSchema
from ..observability import log_event
from ..persistence.cache import WorkspaceCache


REVIEW_OUTPUT_LIMIT = 8192
REVIEW_AGENT_OUTPUT_LIMIT = 1000
ENTITY_REVIEW_MAX_ATTEMPTS = 3
ENTITY_REVIEW_BATCH_SIZE = 5


@dataclass
class _ReviewLockEntry:
    lock: Any
    users: int = 0


class SemanticReviewer(Protocol):
    def review_entity_types(
        self,
        entities: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Batch-review entity types with local context; retry internally on protocol failure."""
        ...


class ContextualSemanticReviewer:
    def __init__(
        self,
        client: ChatClient,
        schema: GraphSchema,
        debug: bool = False,
        review_cache: WorkspaceCache | None = None,
        cache_namespace: str = "",
        max_review_attempts: int = ENTITY_REVIEW_MAX_ATTEMPTS,
    ) -> None:
        if max_review_attempts < 1:
            raise ValueError("max_review_attempts 必须大于或等于 1")
        self.client = client
        self.schema = schema
        self.debug = debug
        self.review_cache = review_cache
        self.cache_namespace = cache_namespace
        self.max_review_attempts = max_review_attempts
        # Hold the registry lock briefly; model calls lock only matching name+context keys.
        self._entity_review_lock_guard = Lock()
        self._entity_review_locks: dict[
            tuple[str, str],
            _ReviewLockEntry,
        ] = {}

    def review_entity_types(
        self,
        entities: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Batch-classify entity types from local term context and Schema."""
        with self._acquire_entity_review_locks(entities):
            return self._review_entity_types_locked(entities)

    @contextmanager
    def _acquire_entity_review_locks(
        self,
        entities: list[dict[str, str]],
    ) -> Iterator[None]:
        """Lock this batch by name+context and reclaim lock entries afterward."""
        keys = sorted({
            (entity["name"], entity["context"])
            for entity in entities
        })
        with self._entity_review_lock_guard:
            entries: list[_ReviewLockEntry] = []
            for key in keys:
                entry = self._entity_review_locks.setdefault(
                    key,
                    _ReviewLockEntry(Lock()),
                )
                entry.users += 1
                entries.append(entry)
        acquired: list[_ReviewLockEntry] = []
        try:
            # Acquire keys in sorted order to avoid deadlocks between overlapping batches.
            for entry in entries:
                entry.lock.acquire()
                acquired.append(entry)
            yield
        finally:
            for entry in reversed(acquired):
                entry.lock.release()
            with self._entity_review_lock_guard:
                for key, entry in zip(keys, entries, strict=True):
                    entry.users -= 1
                    if entry.users == 0:
                        del self._entity_review_locks[key]

    def _review_entity_types_locked(
        self,
        entities: list[dict[str, str]],
    ) -> dict[str, Any]:
        """While holding cache locks, reuse reviews for matching name+context."""
        cached = {
            (entity["name"], entity["context"]): self._load_entity_review(
                entity["name"],
                entity["context"],
            )
            for entity in entities
        }
        # Reuse reviews only when name and local context match (avoid cross-sense pollution).
        pending = list(
            {
                (entity["name"], entity["context"]): entity
                for entity in entities
                if cached[(entity["name"], entity["context"])] is None
            }.values()
        )
        review_failure: dict[str, Any] | None = None
        for attempt in range(1, self.max_review_attempts + 1):
            if not pending:
                break
            attempt_errors: list[dict[str, Any]] = []
            attempt_outputs: list[dict[str, Any]] = []
            # Cap each review request at 5 entities to keep JSONL responses on protocol.
            for batch in _review_batches(pending):
                requested = self._request_entity_type_reviews(
                    batch,
                    retry=attempt > 1,
                )
                attempt_errors.extend(requested["errors"])
                if "model_output" in requested:
                    attempt_outputs.append(requested["model_output"])
                by_name = {entity["name"]: entity for entity in batch}
                for review in requested["reviews"]:
                    context = by_name[review["name"]]["context"]
                    cached[(review["name"], context)] = review
                    self._save_entity_review(review, context)
            pending = [
                entity
                for entity in pending
                if cached[(entity["name"], entity["context"])] is None
            ]
            if pending:
                review_failure = _final_review_failure(
                    pending,
                    attempt_errors,
                    attempt_outputs,
                    attempt,
                )
                log_event(
                    "entity_review_retry",
                    attempt=attempt,
                    missing_names=[entity["name"] for entity in pending],
                )
        reviews = [
            _materialize_entity_review(
                entity,
                cached[(entity["name"], entity["context"])],
            )
            for entity in entities
            if cached[(entity["name"], entity["context"])] is not None
        ]
        result = {
            "reviews": reviews,
            "errors": review_failure["errors"] if pending else [],
        }
        if pending and review_failure is not None and "model_output" in review_failure:
            result["model_output"] = review_failure["model_output"]
        return result

    def _request_entity_type_reviews(
        self,
        entities: list[dict[str, str]],
        retry: bool = False,
    ) -> dict[str, Any]:
        """Merge uncached entities and local contexts into one independent review request."""
        retry_instruction = (
            "这是协议错误后的内部重试。上一轮输出不是严格 JSONL。"
            "本轮尤其禁止使用方括号包裹结果，禁止在行末添加逗号。"
            if retry
            else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是独立的医学术语分类审查器。请根据每个术语附带的"
                    "局部原文判断其在当前语境中的实际含义。你看不到抽取模型"
                    "选择的当前类型，不得为了让某种关系成立而改变实体类型。"
                    "明确的疾病、综合征和功能障碍诊断"
                    "应归为疾病，不能因其在原文中作为临床表现而归为症状。"
                    "先判断名称本身是否为 Schema 中应抽取的医学实体。患者、"
                    "病人、患儿、病例、受试者、人群及其分组等研究对象不是实体，"
                    "例如“糖尿病患者”应返回 should_extract=false。"
                    "否定、缺失、阴性或正常状态也不是实体，例如“无症状”、"
                    "“未见异常”和“检查阴性”都应返回 should_extract=false，"
                    "不得把否定表达改写成阳性实体。"
                    "症状群名称可以是由竖线分隔的组合编码，例如"
                    "“胸闷|胸痛|气短”；应结合各成员所在的局部原文审查，"
                    "不要因为完整组合字符串未在原文出现而拒绝。"
                    "为每个输入项输出一条结果，不得遗漏或新增名称。"
                    "使用 JSONL，每个实体独占一行 JSON 对象，"
                    "每个对象必须压缩在一行内，不得包含空行。"
                    "不要输出 JSON 数组、Markdown 或其他文本。每行字段"
                    "严格按 reason、name、should_extract、"
                    "recommended_entity_type、confidence 的顺序输出。"
                    "should_extract=false 时 recommended_entity_type 必须为 null；"
                    "should_extract=true 时必须从给定实体类型中选择推荐类型。"
                    "confidence 取 0 到 1。"
                    f"{retry_instruction}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        # Do not send the current type so the reviewer cannot rubber-stamp the extractor.
                        "entities": [
                            {
                                "name": entity["name"],
                                "context": entity["context"],
                            }
                            for entity in entities
                        ],
                        "entity_types": _entity_catalog(self.schema),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        raw = self._complete(messages)
        expected = {entity["name"] for entity in entities}
        required_fields = {
            "reason",
            "name",
            "should_extract",
            "recommended_entity_type",
            "confidence",
        }
        by_name: dict[str, dict[str, Any]] = {}
        review_items, parse_errors = decode_jsonl_objects(
            raw,
            "实体审查结果",
            skip_blank_lines=True,
        )
        errors = _review_jsonl_errors(parse_errors)
        for line_number, review in review_items:
            if set(review) != required_fields:
                errors.append(
                    {
                        "line": line_number,
                        "code": "INVALID_REVIEW_FIELDS",
                        "message": "审查行字段与实体审查协议不一致",
                    }
                )
                continue
            name = review["name"]
            if not isinstance(name, str):
                errors.append(
                    {
                        "line": line_number,
                        "code": "INVALID_REVIEW_NAME_TYPE",
                        "message": "审查行 name 必须是字符串",
                    }
                )
                continue
            if name not in expected or name in by_name:
                errors.append(
                    {
                        "line": line_number,
                        "code": "UNKNOWN_OR_DUPLICATE_REVIEW",
                        "message": "审查行包含未知或重复的实体",
                    }
                )
                continue
            reason = review["reason"]
            if not isinstance(reason, str) or not reason.strip():
                errors.append(
                    {
                        "line": line_number,
                        "code": "INVALID_REVIEW_REASON",
                        "message": "审查行 reason 必须是非空字符串",
                    }
                )
                continue
            should_extract = review.get("should_extract")
            if not isinstance(should_extract, bool):
                errors.append(
                    {
                        "line": line_number,
                        "code": "INVALID_SHOULD_EXTRACT",
                        "message": "审查行 should_extract 必须是布尔值",
                    }
                )
                continue
            recommended = review.get("recommended_entity_type")
            if should_extract and (
                not isinstance(recommended, str)
            ):
                errors.append(
                    {
                        "line": line_number,
                        "code": "INVALID_RECOMMENDED_TYPE",
                        "message": "审查行 recommended_entity_type 必须是字符串",
                    }
                )
                continue
            if should_extract and recommended not in self.schema.entity_types:
                errors.append(
                    {
                        "line": line_number,
                        "code": "UNKNOWN_RECOMMENDED_TYPE",
                        "message": "审查行返回了未知实体类型",
                    }
                )
                continue
            if not should_extract and recommended is not None:
                errors.append(
                    {
                        "line": line_number,
                        "code": "UNEXPECTED_RECOMMENDED_TYPE",
                        "message": "不应抽取的实体必须将 recommended_entity_type 设为 null",
                    }
                )
                continue
            confidence = review.get("confidence")
            if (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not isfinite(float(confidence))
                or not 0 <= confidence <= 1
            ):
                errors.append(
                    {
                        "line": line_number,
                        "code": "INVALID_REVIEW_CONFIDENCE",
                        "message": "审查行 confidence 必须是 0 到 1 的数值",
                    }
                )
                continue
            by_name[name] = {
                "reason": reason.strip(),
                "name": name,
                "should_extract": should_extract,
                "recommended_entity_type": recommended,
                "confidence": round(float(confidence), 2),
            }
        for name in expected - by_name.keys():
            errors.append(
                {
                    "code": "MISSING_ENTITY_REVIEW",
                    "name": name,
                    "message": f"审查模型未成功返回实体：{name}",
                }
            )
        reviews = [
            {
                **by_name[entity["name"]],
                "current_entity_type": entity["current_entity_type"],
            }
            for entity in entities
            if entity["name"] in by_name
        ]
        result = {"reviews": reviews, "errors": errors}
        if errors:
            model_output = _model_output(raw)
            result["model_output"] = model_output
            log_event(
                "review_output_invalid",
                errors=errors,
                model_output=model_output,
            )
        return result

    def _load_entity_review(
        self,
        name: str,
        context: str,
    ) -> dict[str, Any] | None:
        if self.review_cache is None:
            return None
        return self.review_cache.load_entity_review(
            self.cache_namespace,
            name,
            context,
        )

    def _save_entity_review(
        self,
        review: dict[str, Any],
        context: str,
    ) -> None:
        if self.review_cache is None:
            return
        self.review_cache.save_entity_review(
            self.cache_namespace,
            review["name"],
            context,
            {
                "reason": review["reason"],
                "name": review["name"],
                "should_extract": review["should_extract"],
                "recommended_entity_type": review[
                    "recommended_entity_type"
                ],
                "confidence": review["confidence"],
            },
        )

    def _complete(self, messages: list[dict[str, str]]) -> str:
        if self.debug:
            print("✿REVIEW_PROMPT✿", flush=True)
            for message in messages:
                print(f"[{message['role']}]", flush=True)
                print(message["content"], flush=True)
            print("✿REVIEW_RESPONSE✿", flush=True)
        return self.client.complete(messages).strip()


def _entity_catalog(schema: GraphSchema) -> dict[str, dict[str, Any]]:
    return {
        entity_type: {
            "description": guidance.description,
            "positive_examples": list(guidance.positive_examples),
            "negative_examples": list(guidance.negative_examples),
        }
        for entity_type, guidance in schema.entity_guidance.items()
    }


def _review_jsonl_errors(
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    code_map = {
        "INVALID_JSON_OBJECT": "INVALID_REVIEW_ITEM",
        "INVALID_JSON": "INVALID_REVIEW_JSON",
        "EMPTY_JSONL": "INVALID_REVIEW_JSON",
        "EMPTY_JSONL_LINE": "INVALID_REVIEW_JSON",
    }
    return [
        {**error, "code": code_map[error["code"]]}
        for error in errors
    ]


def _model_output(raw: str) -> dict[str, Any]:
    """Keep a bounded prefix of raw independent-review output plus full length."""
    return {
        "content": raw[:REVIEW_OUTPUT_LIMIT],
        "char_count": len(raw),
        "truncated": len(raw) > REVIEW_OUTPUT_LIMIT,
    }


def _review_batches(
    entities: list[dict[str, str]],
) -> list[list[dict[str, str]]]:
    """Split pending entities into fixed batches of at most 5."""
    return [
        entities[offset:offset + ENTITY_REVIEW_BATCH_SIZE]
        for offset in range(0, len(entities), ENTITY_REVIEW_BATCH_SIZE)
    ]


def _final_review_failure(
    pending: list[dict[str, str]],
    errors: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    attempt: int,
) -> dict[str, Any]:
    error_counts: dict[str, int] = {}
    for error in errors:
        code = str(error["code"])
        error_counts[code] = error_counts.get(code, 0) + 1
    result: dict[str, Any] = {
        "errors": [
            {
                "code": "ENTITY_REVIEW_RETRY_EXHAUSTED",
                "message": "独立审查多次返回无效格式",
                "attempts": attempt,
                "missing_names": [entity["name"] for entity in pending],
                "error_counts": error_counts,
            }
        ]
    }
    if outputs:
        last_output = outputs[-1]
        content = str(last_output["content"])[:REVIEW_AGENT_OUTPUT_LIMIT]
        result["model_output"] = {
            "content": content,
            "char_count": last_output["char_count"],
            "truncated": last_output["char_count"] > len(content),
        }
    return result


def _materialize_entity_review(
    entity: dict[str, str],
    cached: dict[str, Any] | None,
) -> dict[str, Any]:
    """Materialize name-cached recommended types into per-current-type review results."""
    if cached is None:
        raise ValueError("实体类型审查结果缺失")
    should_extract = cached["should_extract"]
    recommended = cached["recommended_entity_type"]
    return {
        "reason": cached["reason"],
        "name": entity["name"],
        "should_extract": should_extract,
        "current_entity_type": entity["current_entity_type"],
        "recommended_entity_type": recommended,
        "consistent": (
            should_extract
            and recommended == entity["current_entity_type"]
        ),
        "confidence": cached["confidence"],
    }
