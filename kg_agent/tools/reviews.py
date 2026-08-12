"""Independent semantic review tools for the current chunk draft."""

from __future__ import annotations

from typing import Any

from ..agent.semantic_review import SemanticReviewer
from .common import failure, success
from .tool_context import ToolContext


class ReviewTools:
    def __init__(
        self,
        context: ToolContext,
        reviewer: SemanticReviewer | None,
    ) -> None:
        self.context = context
        self.reviewer = reviewer

    @property
    def available(self) -> bool:
        return self.reviewer is not None

    def review_entity_types(
        self,
        entities: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Batch-review entity types with local text; protocol failures retry inside the reviewer."""
        if not entities:
            return success(
                reviews=[],
                reviewed_count=0,
                inconsistent_count=0,
                not_extractable_count=0,
                not_extractable_names=[],
            )
        found = [
            self._find_entity(entity["name"], entity["entity_type"])
            for entity in entities
        ]
        error = next((entity for entity in found if isinstance(entity, dict)), None)
        if error is not None:
            return error
        if self.reviewer is None:
            return failure("REVIEWER_UNAVAILABLE", "当前未配置独立审查模型")
        requests = [
            {
                "name": entity.name,
                "current_entity_type": entity.entity_type,
                "context": _local_context(
                    self.context.workspace.text,
                    entity.name,
                ),
            }
            for entity in found
        ]
        batch = self.reviewer.review_entity_types(requests)
        by_key = {
            (review["name"], review["current_entity_type"]): review
            for review in batch["reviews"]
        }
        results = [
            self._apply_entity_type_review(
                entity,
                by_key[(entity.name, entity.entity_type)],
            )
            for entity in found
            if (entity.name, entity.entity_type) in by_key
        ]
        for entity in found:
            if (entity.name, entity.entity_type) in by_key:
                continue
            self.context.workspace.set_review_warning(
                f"entity_type:{entity.entity_id}",
                {
                    "kind": "entity_type_review_missing",
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "current_entity_type": entity.entity_type,
                    "reason": "该实体的独立类型审查行未成功解析",
                },
            )
        self.context.checkpoint()
        not_extractable_names = [
            result["name"]
            for result in results
            if not result["should_extract"]
        ]
        if batch["errors"]:
            details = {
                "reviews": results,
                "reviewed_count": len(results),
                "not_extractable_count": len(not_extractable_names),
                "not_extractable_names": not_extractable_names,
                "review_errors": batch["errors"],
            }
            if "model_output" in batch:
                # After internal retries fail, expose only a bounded prefix of the last model output.
                details["model_output"] = batch["model_output"]
            return failure(
                "PARTIAL_ENTITY_REVIEW",
                "独立审查内部重试后仍有部分实体失败，成功结果已保存",
                **details,
            )
        return success(
            reviews=results,
            reviewed_count=len(results),
            inconsistent_count=sum(
                not result["consistent"] for result in results
            ),
            not_extractable_count=len(not_extractable_names),
            not_extractable_names=not_extractable_names,
        )

    def list_review_warnings(self) -> dict[str, Any]:
        warnings = sorted(
            self.context.workspace.review_warnings.values(),
            key=lambda warning: warning["warning_id"],
        )
        return success(warnings=warnings, warning_count=len(warnings))

    def confirm_review_warning(
        self,
        warning_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """After checking the source text, explicitly keep the current entity classification."""
        normalized_reason = reason.strip()
        if len(normalized_reason) < 8:
            return failure(
                "REVIEW_REASON_TOO_SHORT",
                "保留当前结果的理由至少需要 8 个字符",
            )
        match = next(
            (
                (key, warning)
                for key, warning in self.context.workspace.review_warnings.items()
                if warning["warning_id"] == warning_id
            ),
            None,
        )
        if match is None:
            return failure("REVIEW_WARNING_NOT_FOUND", "待处理审查警告不存在")
        key, warning = match
        del self.context.workspace.review_warnings[key]
        self.context.checkpoint()
        return success(
            warning_id=warning_id,
            status="confirmed_current",
            name=warning.get("name"),
            entity_type=(
                warning.get("current_entity_type")
                or warning.get("entity_type")
            ),
        )

    def _find_entity(self, name: str, entity_type: str) -> Any:
        entity = next(
            (
                current
                for current in self.context.workspace.entities.values()
                if current.name == name and current.entity_type == entity_type
            ),
            None,
        )
        if entity is None:
            return failure("ENTITY_NOT_FOUND", "当前片段中不存在该名称和类型的实体")
        return entity

    def _apply_entity_type_review(
        self,
        entity: Any,
        review: dict[str, Any],
    ) -> dict[str, Any]:
        key = f"entity_type:{entity.entity_id}"
        warning = None
        if not review["should_extract"]:
            warning = self.context.workspace.set_review_warning(
                key,
                {
                    "kind": "entity_not_extractable",
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "current_entity_type": entity.entity_type,
                    "recommended_entity_type": None,
                    "confidence": review["confidence"],
                    "reason": review["reason"],
                },
            )
        elif review["consistent"]:
            self.context.workspace.review_warnings.pop(key, None)
        else:
            warning = self.context.workspace.set_review_warning(
                key,
                {
                    "kind": "entity_type",
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "current_entity_type": entity.entity_type,
                    "recommended_entity_type": review[
                        "recommended_entity_type"
                    ],
                    "confidence": review["confidence"],
                    "reason": review["reason"],
                },
            )
        return {
            "reason": review["reason"],
            "entity_id": entity.entity_id,
            "name": entity.name,
            "should_extract": review["should_extract"],
            "current_entity_type": entity.entity_type,
            "recommended_entity_type": review["recommended_entity_type"],
            "consistent": review["consistent"],
            "confidence": review["confidence"],
            "warning_id": warning["warning_id"] if warning else None,
        }

def _local_context(
    text: str,
    name: str,
    radius: int = 160,
    occurrence_limit: int = 3,
) -> str:
    """Return source snippets around the entity's first occurrences for type review."""
    search_names = (
        [member for member in name.split("|") if member]
        if "|" in name and name not in text
        else [name]
    )
    occurrences: list[tuple[int, int]] = []
    for search_name in search_names:
        offset = 0
        member_limit = 1 if len(search_names) > 1 else occurrence_limit
        member_count = 0
        while member_count < member_limit:
            position = text.find(search_name, offset)
            if position < 0:
                break
            occurrences.append((position, len(search_name)))
            member_count += 1
            offset = position + len(search_name)
    occurrences.sort()
    segments = [
        text[
            max(0, position - radius):
            min(len(text), position + name_length + radius)
        ]
        for position, name_length in occurrences
    ]
    # Overlapping windows may be identical; dedupe to shrink batch review prompts.
    return "\n…\n".join(dict.fromkeys(segments))
