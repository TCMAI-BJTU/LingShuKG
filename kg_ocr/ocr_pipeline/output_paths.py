"""Safe OCR output path helpers (filename length limits)."""

from hashlib import sha256
from pathlib import Path


MAX_FILENAME_BYTES = 255


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Shorten text without splitting a UTF-8 character."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def build_output_paths(
    pdf_path: Path,
    output_dir: Path,
    output_format: str,
) -> tuple[Path, Path]:
    """Build final and ``.part`` temp paths; hash-suffix if name is too long."""
    output_suffix = f".{output_format}"
    temporary_suffix = f"{output_suffix}.part"
    source_stem = pdf_path.stem
    if len(f"{source_stem}{temporary_suffix}".encode("utf-8")) <= MAX_FILENAME_BYTES:
        output_path = output_dir / f"{source_stem}{output_suffix}"
        return output_path, output_path.with_suffix(temporary_suffix)

    source_hash = sha256(
        str(pdf_path.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    hash_suffix = f"--{source_hash}"
    stem_max_bytes = MAX_FILENAME_BYTES - len(
        f"{hash_suffix}{temporary_suffix}".encode("utf-8")
    )
    shortened_stem = _truncate_utf8(source_stem, stem_max_bytes)
    output_path = output_dir / f"{shortened_stem}{hash_suffix}{output_suffix}"
    return output_path, output_path.with_suffix(temporary_suffix)
