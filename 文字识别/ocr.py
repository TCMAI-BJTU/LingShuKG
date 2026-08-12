"""使用本地 vLLM 部署的 DeepSeek-OCR-2 解析 PDF。"""
from __future__ import annotations

import base64
import os
import re
from collections import Counter

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["all_proxy"] = ""

from pathlib import Path

import fitz  # pymupdf
from openai import OpenAI

# vLLM 服务
API_BASE = "http://127.0.0.1:8080/v1"
MODEL = "deepseek-ocr-2"
API_KEY = "EMPTY"

PDF_PATH = Path(
    "/home/huarui/pythonProject/data_generate/灵枢数据补充/知识图谱智能体/data/河南书籍/"
    "129、1499类经图翼、类经附翼_可搜索.pdf"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "ocr"

# PDF 渲染分辨率（约 144 dpi）；过大可能触发 vLLM 图像 token 上限
RENDER_ZOOM = 4.0
# 空白页检测使用较低分辨率，避免为无内容页面执行完整渲染和模型推理。
BLANK_CHECK_ZOOM = 0.5
BLANK_INK_RATIO_THRESHOLD = 0.0015
BLANK_CONTRAST_THRESHOLD = 8.0
BLANK_BACKGROUND_DELTA = 25
# None 表示处理全部页面；调试时可设为例如 3
MAX_PAGES: int | None = None

OCR_PROMPT = "<image>\nConvert the document to markdown."

GROUNDING_TAG_PATTERN = re.compile(
    r"<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>",
    flags=re.DOTALL,
)
GROUNDING_COORD_PATTERN = re.compile(
    r"(?i)\b(?:text|title|sub[_ -]?title|table|image|figure|formula|caption|header|footer)"
    r"\s*\[\[\s*\d+(?:\s*,\s*\d+){3}\s*\]\]"
)
REPEATED_BLOCK_PATTERN = re.compile(r"(.{2,256}?)(?:\1){4,}", flags=re.DOTALL)
HAN_PATTERN = re.compile(r"[\u3400-\u9fff]")


def truncate_degenerate_repetition(text: str) -> str:
    """截断模型生成的周期性重复或低字符多样性异常长尾。"""
    for match in REPEATED_BLOCK_PATTERN.finditer(text):
        if len(match.group(0)) >= 80:
            return text[: match.start()].rstrip(" ，,。；;、\n\t")

    # 类似“以、易、人”少数字符持续数百次，但循环并非逐字完全一致。
    han_positions = [
        (match.start(), match.group(0)) for match in HAN_PATTERN.finditer(text)
    ]
    min_tail_chars = 120
    probe_chars = 80
    tail_counts = Counter(char for _, char in han_positions)
    tail_distinct = len(tail_counts)
    for i in range(0, len(han_positions) - min_tail_chars + 1):
        probe = {char for _, char in han_positions[i : i + probe_chars]}
        if len(probe) <= 4 and tail_distinct <= 8:
            start = han_positions[i][0]
            return text[:start].rstrip(" ，,。；;、\n\t")
        current_char = han_positions[i][1]
        tail_counts[current_char] -= 1
        if tail_counts[current_char] == 0:
            tail_distinct -= 1

    return text


def clean_ocr_output(text: str) -> str:
    """清除 DeepSeek-OCR-2 输出中的版面定位标签及坐标。"""
    text = GROUNDING_TAG_PATTERN.sub("", text)
    text = GROUNDING_COORD_PATTERN.sub("", text)
    text = truncate_degenerate_repetition(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ensure_no_proxy() -> None:
    """本地 8080 走直连，避免被代理劫持。"""
    host = "127.0.0.1,localhost"
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [p.strip() for p in current.split(",") if p.strip()]
        for item in host.split(","):
            if item not in parts:
                parts.append(item)
        os.environ[key] = ",".join(parts)


def make_client() -> OpenAI:
    _ensure_no_proxy()
    return OpenAI(api_key=API_KEY, base_url=API_BASE, timeout=3600)


def pdf_page_to_png_bytes(page: fitz.Page, zoom: float = RENDER_ZOOM) -> bytes:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def is_blank_page(page: fitz.Page) -> bool:
    """通过低分辨率灰度图判断页面是否基本空白。"""
    rect = page.rect
    margin_x = rect.width * 0.03
    margin_y = rect.height * 0.03
    clip = fitz.Rect(
        rect.x0 + margin_x,
        rect.y0 + margin_y,
        rect.x1 - margin_x,
        rect.y1 - margin_y,
    )
    pix = page.get_pixmap(
        matrix=fitz.Matrix(BLANK_CHECK_ZOOM, BLANK_CHECK_ZOOM),
        colorspace=fitz.csGRAY,
        alpha=False,
        clip=clip,
    )
    histogram = Counter(pix.samples)
    total = sum(histogram.values())
    if total == 0:
        return True

    # 用第 90 百分位灰度估计纸张背景，兼容偏黄、偏灰的扫描页面。
    target = total * 0.9
    cumulative = 0
    background = 255
    for gray in range(256):
        cumulative += histogram.get(gray, 0)
        if cumulative >= target:
            background = gray
            break

    dark_cutoff = max(0, background - BLANK_BACKGROUND_DELTA)
    ink_pixels = sum(
        count for gray, count in histogram.items() if gray < dark_cutoff
    )
    ink_ratio = ink_pixels / total

    mean = sum(gray * count for gray, count in histogram.items()) / total
    variance = (
        sum((gray - mean) ** 2 * count for gray, count in histogram.items())
        / total
    )
    contrast = variance**0.5

    return (
        ink_ratio < BLANK_INK_RATIO_THRESHOLD
        and contrast < BLANK_CONTRAST_THRESHOLD
    )


def ocr_image_png(client: OpenAI, png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
        max_tokens=4096,
        temperature=0.0,
        top_p=0.95,
        extra_body={
            "skip_special_tokens": False,
            # 配合服务端 NGramPerReqLogitsProcessor；若未启用该 processor 可去掉
            "vllm_xargs": {
                "ngram_size": 20,
                "window_size": 90,
                "whitelist_token_ids": [128821, 128822],
            },
        },
    )
    raw_text = response.choices[0].message.content or ""
    return clean_ocr_output(raw_text)


def parse_pdf(
    pdf_path: Path,
    output_dir: Path | None = None,
    max_pages: int | None = MAX_PAGES,
) -> Path:
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{pdf_path.stem}.md"
    # 每次运行先清空目标文件，之后将各页 OCR 结果依次追加到同一文件。
    md_path.write_text("", encoding="utf-8")

    client = make_client()
    doc = fitz.open(pdf_path)
    total = doc.page_count
    n_pages = total if max_pages is None else min(total, max_pages)

    for i in range(n_pages):
        page_no = i + 1
        print(f"[{page_no}/{n_pages}] OCR ...", flush=True)
        page = doc[i]
        if is_blank_page(page):
            text = ""
            print(f"[{page_no}/{n_pages}] 空白页，跳过 OCR", flush=True)
        else:
            png_bytes = pdf_page_to_png_bytes(page)
            text = ocr_image_png(client, png_bytes)
        page_block = f"## 第 {page_no} 页\n\n{text}\n\n"
        with md_path.open("a", encoding="utf-8") as md_file:
            md_file.write(page_block)
        print(f"[{page_no}/{n_pages}] done ({len(text)} chars)", flush=True)

    doc.close()
    print(f"保存完成: {md_path}")
    return md_path


if __name__ == "__main__":
    parse_pdf(PDF_PATH)
