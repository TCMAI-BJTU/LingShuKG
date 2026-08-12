"""OCR 输出与普通文本的清洗函数。"""

import re


DOT_LEADER_PATTERN = re.compile(r"(?:[・·\.．…]\s*){12,}")
REPEATED_BLOCK_PATTERN = re.compile(r"(.{2,128}?)(?:\s*\1){5,}", re.DOTALL)
HAN_PATTERN = re.compile(r"[\u3400-\u9fff]")
MARKDOWN_STRUCTURAL_LINE_PATTERN = re.compile(
    r"^(?:#{1,6}\s|[-+*]\s|\d+[.)、]\s|>\s?|```|\|)"
)


def join_plain_lines(lines: list[str]) -> str:
    """拼接同一自然段中由页面栏宽产生的视觉换行。"""
    if not lines:
        return ""

    joined = lines[0].strip()
    for line in lines[1:]:
        current = line.strip()
        if not current:
            continue
        separator = ""
        if (
            joined
            and joined[-1].isascii()
            and joined[-1].isalnum()
            and current[0].isascii()
            and current[0].isalnum()
        ):
            separator = " "
        joined += separator + current
    return joined


def join_visual_line_wraps(text: str) -> str:
    """合并普通正文行，同时保留 Markdown 结构和真正的段落空行。"""
    output_blocks: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        block_lines: list[str] = []
        plain_lines: list[str] = []
        in_code_fence = False

        def flush_plain_lines() -> None:
            if plain_lines:
                block_lines.append(join_plain_lines(plain_lines))
                plain_lines.clear()

        for line in lines:
            stripped = line.strip()
            is_structural = bool(
                MARKDOWN_STRUCTURAL_LINE_PATTERN.match(stripped)
            )
            if stripped.startswith("```"):
                flush_plain_lines()
                block_lines.append(stripped)
                in_code_fence = not in_code_fence
            elif in_code_fence or is_structural:
                flush_plain_lines()
                block_lines.append(stripped)
            else:
                plain_lines.append(stripped)

        flush_plain_lines()
        output_blocks.append("\n".join(block_lines))
    return "\n\n".join(output_blocks)


def clean_ocr_output(text: str) -> tuple[str, bool]:
    """清理明显的退化重复，返回文本及异常标志。"""
    abnormal = False

    dot_match = DOT_LEADER_PATTERN.search(text)
    if dot_match:
        abnormal = len(dot_match.group(0)) >= 200
        text = DOT_LEADER_PATTERN.sub(" ", text)

    repeated_match = REPEATED_BLOCK_PATTERN.search(text)
    if repeated_match and len(repeated_match.group(0)) >= 100:
        text = text[: repeated_match.start()]
        abnormal = True

    han_chars = HAN_PATTERN.findall(text)
    if len(han_chars) >= 160 and len(set(han_chars[-160:])) <= 6:
        text = text[: max(0, len(text) - 320)]
        abnormal = True

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return join_visual_line_wraps(text).strip(), abnormal
