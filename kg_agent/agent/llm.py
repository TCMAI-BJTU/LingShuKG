"""Minimal OpenAI-compatible chat client for the extraction agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True)
class LLMSettings:
    # base_url: str = "http://61.163.97.50:8002/qwen/v1"
    # api_key: str = "sk-vllm-8538"
    # model: str = "qwen3.6"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen3.6-27B"
    api_key: str = "EMPTY"
    temperature: float = 0.6
    max_tokens: int = 4096
    timeout: float = 3600.0
    stream: bool = False

    @classmethod
    def from_env(cls) -> "LLMSettings":
        """Load model settings from KG_LLM_* environment variables."""
        return cls(
            base_url=os.environ.get("KG_LLM_BASE_URL", cls.base_url),
            api_key=os.environ.get("KG_LLM_API_KEY", cls.api_key),
            model=os.environ.get("KG_LLM_MODEL", cls.model),
            temperature=float(
                os.environ.get("KG_LLM_TEMPERATURE", cls.temperature)
            ),
            max_tokens=int(os.environ.get("KG_LLM_MAX_TOKENS", cls.max_tokens)),
            timeout=float(os.environ.get("KG_LLM_TIMEOUT", cls.timeout)),
            stream=False,
        )


class OpenAIChatClient:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url.rstrip("/"),
            timeout=settings.timeout,
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        if self.settings.stream:
            return self._complete_stream(messages)
        response = self._client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        )
        message = response.choices[0].message
        return message.content or ""

    def _complete_stream(self, messages: list[dict[str, str]]) -> str:
        stream = self._client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            stream=True,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        )
        parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                parts.append(content)
                print(content, end="", flush=True)
        print(flush=True)
        return "".join(parts)
