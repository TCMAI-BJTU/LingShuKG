"""Deterministic text chunking for directory inputs."""

from __future__ import annotations


def split_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """按固定字符窗口和重叠长度切分非空文本。"""
    if chunk_size < 1:
        raise ValueError("chunk_size 必须大于或等于 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于或等于 0 且小于 chunk_size")
    if not text.strip():
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == len(text):
            # 已到原文末尾时立即结束，避免额外生成一个只有重叠内容的 chunk。
            break
        # 下一窗口回退 overlap 字符，减少实体或关系被切在边界两侧的概率。
        start = end - overlap
    return chunks
