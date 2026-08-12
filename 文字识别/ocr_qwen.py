"""Qwen 全量 OCR 命令行入口。

运行示例：
    python ocr_qwen.py file.pdf
    python ocr_qwen.py /path/to/pdfs --recursive --workers 128

所有 PDF 页面均强制使用 Qwen OCR，不读取 PDF 原生文本层。
所有文件的页面任务共用同一个线程池；--file-workers 仅用于 --native-only。
"""

from ocr_pipeline import detect_native_pdf, parse_pdf
from ocr_pipeline.cli import main

__all__ = ["detect_native_pdf", "parse_pdf", "main"]


if __name__ == "__main__":
    main()
