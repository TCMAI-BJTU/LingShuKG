"""DeepSeek-OCR-2 eight-port load balancing and output cleanup."""

import base64
import logging
import re
import threading
from dataclasses import dataclass

import httpx
from openai import OpenAI

from . import deepseek_config
from .qwen_ocr import clear_local_proxy
from .text_cleaning import clean_ocr_output


logger = logging.getLogger(__name__)

try:
    from opencc import OpenCC
except ImportError:
    OpenCC = None
    logger.warning(
        "未安装 opencc-python-reimplemented，DeepSeek OCR 文本不会自动转换为简体。"
    )


GROUNDING_TAG_PATTERN = re.compile(
    r"<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>",
    flags=re.DOTALL,
)
GROUNDING_COORD_PATTERN = re.compile(
    r"(?i)\b(?:text|title|sub[_ -]?title|table|image|figure|formula|caption|header|footer)"
    r"\s*\[\[\s*\d+(?:\s*,\s*\d+){3}\s*\]\]"
)
SPECIAL_TAG_PATTERN = re.compile(
    r"<\|(?:/?ref|/?det|endofsentence)\|>"
)


@dataclass
class _Endpoint:
    api_base: str
    client: OpenAI
    semaphore: threading.BoundedSemaphore


def _clean_deepseek_output(text: str) -> tuple[str, bool]:
    """Strip DeepSeek grounding tags; reuse shared degenerate-output cleanup."""
    had_grounding = bool(
        GROUNDING_TAG_PATTERN.search(text)
        or GROUNDING_COORD_PATTERN.search(text)
    )
    text = GROUNDING_TAG_PATTERN.sub("", text)
    text = GROUNDING_COORD_PATTERN.sub("", text)
    text = SPECIAL_TAG_PATTERN.sub("", text)
    text, abnormal = clean_ocr_output(text)
    if OpenCC is not None:
        text = OpenCC("t2s").convert(text)
    return text, abnormal or had_grounding


class DeepSeekOCRPool:
    """Round-robin across vLLM endpoints with per-port concurrency caps."""

    def __init__(
        self,
        api_bases: tuple[str, ...] = deepseek_config.API_BASES,
        model: str = deepseek_config.MODEL,
        api_key: str = deepseek_config.API_KEY,
        per_server_concurrency: int = deepseek_config.PER_SERVER_CONCURRENCY,
    ) -> None:
        if not api_bases:
            raise ValueError("至少需要一个 DeepSeek OCR 服务地址")
        if per_server_concurrency < 1:
            raise ValueError("per_server_concurrency 必须大于或等于 1")

        clear_local_proxy()
        self.model = model
        self._endpoints = [
            _Endpoint(
                api_base=api_base,
                client=OpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    timeout=3600,
                    max_retries=2,
                    http_client=httpx.Client(
                        timeout=3600,
                        limits=httpx.Limits(
                            max_connections=per_server_concurrency,
                            max_keepalive_connections=per_server_concurrency,
                        ),
                    ),
                ),
                semaphore=threading.BoundedSemaphore(per_server_concurrency),
            )
            for api_base in api_bases
        ]
        self._selection_lock = threading.Lock()
        self._next_endpoint = 0

    @property
    def server_count(self) -> int:
        return len(self._endpoints)

    def _select_endpoint(self) -> _Endpoint:
        with self._selection_lock:
            endpoint = self._endpoints[self._next_endpoint]
            self._next_endpoint = (self._next_endpoint + 1) % len(self._endpoints)
        return endpoint

    def recognize(
        self,
        png_bytes: bytes,
        prompt: str,
        max_output_tokens: int,
    ) -> tuple[str, str | None, bool]:
        endpoint = self._select_endpoint()
        image_base64 = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{image_base64}"

        with endpoint.semaphore:
            response = endpoint.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=max_output_tokens,
                temperature=0.0,
                top_p=0.95,
                extra_body={
                    "skip_special_tokens": False,
                    "vllm_xargs": {
                        "ngram_size": deepseek_config.NGRAM_SIZE,
                        "window_size": deepseek_config.WINDOW_SIZE,
                        "whitelist_token_ids": (
                            deepseek_config.WHITELIST_TOKEN_IDS
                        ),
                    },
                },
            )

        choice = response.choices[0]
        text, abnormal = _clean_deepseek_output(
            choice.message.content or ""
        )
        if choice.finish_reason == "length":
            abnormal = True
        return text, choice.finish_reason, abnormal
