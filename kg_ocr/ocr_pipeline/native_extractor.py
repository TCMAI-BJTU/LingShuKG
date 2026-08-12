"""Native PDF text extraction and header/footer cleanup."""

import math
import logging
import re
from collections import Counter

import fitz

from . import config
from .text_cleaning import join_plain_lines


logger = logging.getLogger(__name__)

try:
    from opencc import OpenCC
except ImportError:
    OpenCC = None
    logger.warning(
        "未安装 opencc-python-reimplemented，原生 PDF 文本不会自动转换为简体。"
    )


def opencc_available() -> bool:
    return OpenCC is not None


def normalize_margin_line(line: str) -> str:
    """Normalize dates/issue/page numbers in margin lines for matching."""
    normalized = re.sub(r"\d+", "<N>", line.strip())
    return re.sub(r"\s+", "", normalized)


def find_repeated_margin_lines(
    document: fitz.Document,
    start_page: int,
    end_page: int,
) -> set[str]:
    """Find repeated edge text used as headers/footers."""
    occurrences: Counter[str] = Counter()
    page_count = end_page - start_page + 1

    for page_number in range(start_page, end_page + 1):
        page = document[page_number - 1]
        page_height = page.rect.height
        seen_on_page: set[str] = set()
        for block in page.get_text("blocks"):
            if len(block) >= 7 and block[6] != 0:
                continue
            y0, y1, block_text = block[1], block[3], block[4]
            in_margin = (
                y1 <= page_height * config.MARGIN_HEIGHT_RATIO
                or y0 >= page_height * (1 - config.MARGIN_HEIGHT_RATIO)
            )
            if not in_margin:
                continue
            for line in block_text.splitlines():
                normalized = normalize_margin_line(line)
                if normalized:
                    seen_on_page.add(normalized)
        occurrences.update(seen_on_page)

    threshold = max(2, math.ceil(page_count * 0.30))
    return {
        line for line, count in occurrences.items() if count >= threshold
    }


def is_standalone_page_number(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"[\s\-—–·]*[0-9一二三四五六七八九十百千〇○]+[\s\-—–·]*",
            text,
        )
    )


def is_native_heading(line: str) -> bool:
    """Detect short section titles so they stay separate from body text."""
    stripped = line.strip()
    if len(stripped) > 50:
        return False
    if stripped in {"摘要", "Abstract", "参考文献", "关键词", "Key words"}:
        return True
    return bool(re.match(r"^\d+(?:\.\d+)*\s+\S", stripped))


def join_native_block_lines(lines: list[str]) -> str:
    """Join visual lines in a block; keep short headings as paragraph breaks."""
    sections: list[str] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            sections.append(join_plain_lines(paragraph_lines))
            paragraph_lines.clear()

    for line in lines:
        if is_native_heading(line):
            flush_paragraph()
            sections.append(line.strip())
        else:
            paragraph_lines.append(line)
    flush_paragraph()
    return "\n\n".join(sections)


def should_merge_blocks(
    previous: tuple[float, float, float, float, str],
    current: tuple[float, float, float, float, str],
) -> bool:
    """Whether adjacent blocks are visual wraps of the same paragraph."""
    px0, _py0, px1, py1, previous_text = previous
    cx0, cy0, cx1, _cy1, current_text = current
    vertical_gap = cy0 - py1
    if vertical_gap < -1 or vertical_gap > 8:
        return False

    overlap = max(0.0, min(px1, cx1) - max(px0, cx0))
    min_width = max(1.0, min(px1 - px0, cx1 - cx0))
    if overlap / min_width < 0.60:
        return False
    if cx0 > px0 + 10:  # Clear first-line indent → new paragraph.
        return False

    structural_prefixes = (
        "摘要",
        "关键词",
        "Abstract",
        "Key words",
        "Funding",
        "参考文献",
        "基金项目",
        "通信作者",
    )
    if current_text.startswith(structural_prefixes):
        return False
    return len(previous_text) > 12


def extract_native_page_text(
    page: fitz.Page,
    repeated_margin_lines: set[str],
) -> str:
    """Extract native text; drop repeated margins and merge wraps."""
    page_height = page.rect.height
    extracted: list[tuple[float, float, float, float, str]] = []

    # Keep PDF content order; sort=True interleaves two-column text.
    for block in page.get_text("blocks"):
        if len(block) >= 7 and block[6] != 0:
            continue
        y0, y1, block_text = block[1], block[3], block[4]
        in_margin = (
            y1 <= page_height * config.MARGIN_HEIGHT_RATIO
            or y0 >= page_height * (1 - config.MARGIN_HEIGHT_RATIO)
        )

        kept_lines: list[str] = []
        for line in block_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if in_margin and (
                normalize_margin_line(stripped) in repeated_margin_lines
                or is_standalone_page_number(stripped)
            ):
                continue
            kept_lines.append(stripped)

        if kept_lines:
            extracted.append(
                (
                    block[0],
                    block[1],
                    block[2],
                    block[3],
                    join_native_block_lines(kept_lines),
                )
            )

    output_blocks: list[str] = []
    previous: tuple[float, float, float, float, str] | None = None
    for block in extracted:
        if previous is not None and output_blocks and should_merge_blocks(previous, block):
            current_text = block[4]
            separator = ""
            if (
                output_blocks[-1]
                and current_text
                and output_blocks[-1][-1].isascii()
                and output_blocks[-1][-1].isalnum()
                and current_text[0].isascii()
                and current_text[0].isalnum()
            ):
                separator = " "
            output_blocks[-1] += separator + current_text
        else:
            output_blocks.append(block[4])
        previous = block

    text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(output_blocks)).strip()
    if OpenCC is not None:
        text = OpenCC("t2s").convert(text)
    return text
