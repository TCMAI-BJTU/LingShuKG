"""Deterministic text chunking for directory inputs."""

from __future__ import annotations


def split_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """Split non-empty text into fixed-size windows with overlap."""
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
            # Stop at text end to avoid an extra overlap-only chunk.
            break
        # Next window retreats by overlap to reduce entities/relations split across boundaries.
        start = end - overlap
    return chunks
