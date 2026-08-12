"""原生电子排版 PDF 的检测逻辑。"""

import re
import statistics

import fitz

from . import config


def max_image_coverage(page: fitz.Page) -> float:
    """返回页面中单张栅格图片覆盖页面的最大比例。"""
    page_area = page.rect.get_area()
    if page_area <= 0:
        return 0.0

    max_coverage = 0.0
    for image in page.get_images(full=True):
        xref = image[0]
        for image_rect in page.get_image_rects(xref):
            intersection = image_rect & page.rect
            if intersection.is_empty:
                continue
            max_coverage = max(
                max_coverage,
                intersection.get_area() / page_area,
            )
    return min(max_coverage, 1.0)


def detect_native_pdf(
    document: fitz.Document,
    strict: bool = False,
) -> tuple[bool, dict[str, float | int]]:
    """判断整份 PDF 是否为原生排版；扫描和混合版均返回 False。"""
    total_pages = document.page_count
    if total_pages == 0:
        return False, {
            "total_pages": 0,
            "text_pages": 0,
            "full_image_pages": 0,
            "text_page_ratio": 0.0,
            "full_image_page_ratio": 0.0,
        }

    text_pages = 0
    full_image_pages = 0
    total_text_chars = 0
    fullwidth_ascii_chars = 0
    private_use_chars = 0
    replacement_chars = 0
    text_block_counts: list[int] = []
    for page in document:
        text = re.sub(r"\s+", "", page.get_text("text"))
        total_text_chars += len(text)
        fullwidth_ascii_chars += sum(
            0xFF01 <= ord(char) <= 0xFF5E for char in text
        )
        private_use_chars += sum(
            0xE000 <= ord(char) <= 0xF8FF for char in text
        )
        replacement_chars += text.count("\ufffd")
        text_block_counts.append(
            sum(
                1
                for block in page.get_text("blocks")
                if len(block) < 7 or block[6] == 0
            )
        )
        if len(text) >= config.NATIVE_MIN_TEXT_CHARS:
            text_pages += 1
        if max_image_coverage(page) >= config.FULL_PAGE_IMAGE_MIN_COVERAGE:
            full_image_pages += 1

    text_page_ratio = text_pages / total_pages
    full_image_page_ratio = full_image_pages / total_pages
    min_text_page_ratio = (
        config.STRICT_NATIVE_MIN_TEXT_PAGE_RATIO
        if strict
        else config.NATIVE_MIN_TEXT_PAGE_RATIO
    )
    max_full_image_ratio = (
        config.STRICT_NATIVE_MAX_FULL_PAGE_IMAGE_RATIO
        if strict
        else config.NATIVE_MAX_FULL_PAGE_IMAGE_RATIO
    )
    structurally_native = (
        text_page_ratio >= min_text_page_ratio
        and full_image_page_ratio <= max_full_image_ratio
    )
    denominator = max(1, total_text_chars)
    fullwidth_ascii_ratio = fullwidth_ascii_chars / denominator
    private_use_ratio = private_use_chars / denominator
    replacement_char_ratio = replacement_chars / denominator
    median_text_blocks = float(statistics.median(text_block_counts))
    max_fullwidth_ascii_ratio = (
        config.STRICT_NATIVE_MAX_FULLWIDTH_ASCII_RATIO
        if strict
        else config.NATIVE_MAX_FULLWIDTH_ASCII_RATIO
    )
    max_private_use_ratio = (
        config.STRICT_NATIVE_MAX_PRIVATE_USE_RATIO
        if strict
        else config.NATIVE_MAX_PRIVATE_USE_RATIO
    )
    max_replacement_char_ratio = (
        config.STRICT_NATIVE_MAX_REPLACEMENT_CHAR_RATIO if strict else 1.0
    )
    max_median_text_blocks = (
        config.STRICT_NATIVE_MAX_MEDIAN_TEXT_BLOCKS
        if strict
        else config.NATIVE_MAX_MEDIAN_TEXT_BLOCKS
    )
    direct_text_usable = (
        fullwidth_ascii_ratio <= max_fullwidth_ascii_ratio
        and private_use_ratio <= max_private_use_ratio
        and replacement_char_ratio <= max_replacement_char_ratio
        and median_text_blocks <= max_median_text_blocks
    )
    is_native = structurally_native and direct_text_usable
    details: dict[str, float | int] = {
        "total_pages": total_pages,
        "text_pages": text_pages,
        "full_image_pages": full_image_pages,
        "text_page_ratio": text_page_ratio,
        "full_image_page_ratio": full_image_page_ratio,
        "structurally_native": int(structurally_native),
        "direct_text_usable": int(direct_text_usable),
        "fullwidth_ascii_ratio": fullwidth_ascii_ratio,
        "private_use_ratio": private_use_ratio,
        "replacement_char_ratio": replacement_char_ratio,
        "median_text_blocks": median_text_blocks,
    }
    return is_native, details
