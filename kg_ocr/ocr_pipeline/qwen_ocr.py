"""PDF page render, blank detection, and Qwen vision OCR."""

import base64
import os
from collections import Counter

import fitz
import httpx
from openai import OpenAI

from . import config
from .text_cleaning import clean_ocr_output


def clear_local_proxy() -> None:
    """Ensure local vLLM traffic bypasses system proxies."""
    for proxy_key in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        os.environ.pop(proxy_key, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"


def make_client() -> OpenAI:
    clear_local_proxy()
    return OpenAI(
        api_key=config.API_KEY,
        base_url=config.API_BASE,
        timeout=3600,
        max_retries=2,
        http_client=httpx.Client(
            timeout=3600,
            limits=httpx.Limits(
                max_connections=config.MAX_TOTAL_CONCURRENCY,
                max_keepalive_connections=config.MAX_TOTAL_CONCURRENCY,
            ),
        ),
    )


def pdf_page_to_png_bytes(
    page: fitz.Page,
    zoom: float = config.RENDER_ZOOM,
) -> bytes:
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        alpha=False,
    )
    return pixmap.tobytes("png")


def is_blank_page(page: fitz.Page) -> bool:
    """Detect near-blank pages via low-res grayscale ink/contrast."""
    rect = page.rect
    margin_x = rect.width * config.BLANK_MARGIN_RATIO
    margin_y = rect.height * config.BLANK_MARGIN_RATIO
    clip = fitz.Rect(
        rect.x0 + margin_x,
        rect.y0 + margin_y,
        rect.x1 - margin_x,
        rect.y1 - margin_y,
    )
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(config.BLANK_CHECK_ZOOM, config.BLANK_CHECK_ZOOM),
        colorspace=fitz.csGRAY,
        alpha=False,
        clip=clip,
    )
    histogram = Counter(pixmap.samples)
    total = sum(histogram.values())
    if total == 0:
        return True

    target = total * 0.9
    cumulative = 0
    background = 255
    for gray in range(256):
        cumulative += histogram.get(gray, 0)
        if cumulative >= target:
            background = gray
            break

    dark_cutoff = max(0, background - config.BLANK_BACKGROUND_DELTA)
    ink_pixels = sum(
        count for gray, count in histogram.items() if gray < dark_cutoff
    )
    ink_ratio = ink_pixels / total
    mean = sum(gray * count for gray, count in histogram.items()) / total
    variance = sum(
        (gray - mean) ** 2 * count for gray, count in histogram.items()
    ) / total
    contrast = variance**0.5
    return (
        ink_ratio < config.BLANK_INK_RATIO_THRESHOLD
        and contrast < config.BLANK_CONTRAST_THRESHOLD
    )


def ocr_image_png(
    client: OpenAI,
    png_bytes: bytes,
    prompt: str = config.OCR_PROMPT,
    max_output_tokens: int = config.MAX_OUTPUT_TOKENS,
) -> tuple[str, str | None, bool]:
    """OCR one PNG via Qwen3.6-27B; returns (text, finish_reason, abnormal)."""
    image_base64 = base64.b64encode(png_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{image_base64}"
    response = client.chat.completions.create(
        model=config.MODEL,
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
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        presence_penalty=config.PRESENCE_PENALTY,
        seed=1,
        extra_body={
            "top_k": config.TOP_K,
            "min_p": config.MIN_P,
            "repetition_penalty": config.REPETITION_PENALTY,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    choice = response.choices[0]
    text, abnormal = clean_ocr_output(choice.message.content or "")
    if choice.finish_reason == "length":
        abnormal = True
    return text, choice.finish_reason, abnormal
