"""OCR 输出文件路径生成工具。"""

from hashlib import sha256
from pathlib import Path


MAX_FILENAME_BYTES = 255


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """在不截断 UTF-8 字符的前提下缩短文本。

    参数：
        text: 待缩短的文本。
        max_bytes: UTF-8 编码后的最大字节数。

    返回：
        不超过 ``max_bytes`` 的文本。
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def build_output_paths(
    pdf_path: Path,
    output_dir: Path,
    output_format: str,
) -> tuple[Path, Path]:
    """为 PDF 生成安全的正式输出路径和临时输出路径。

    超过文件系统单个文件名 255 字节限制时，保留可读前缀并追加源路径的
    SHA-256 短哈希，确保不同长文件名不会映射到同一个输出文件。

    参数：
        pdf_path: 源 PDF 路径。
        output_dir: 输出文件所在目录。
        output_format: 输出后缀，不含开头的点号。

    返回：
        正式输出路径和以 ``.part`` 结尾的临时输出路径。
    """
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
