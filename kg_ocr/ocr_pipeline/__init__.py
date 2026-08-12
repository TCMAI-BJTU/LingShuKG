"""PDF 原生文本提取与 Qwen OCR 自动分流工具。"""

from .pdf_detection import detect_native_pdf
from .processor import parse_pdf, process_ocr_jobs

__all__ = ["detect_native_pdf", "parse_pdf", "process_ocr_jobs"]
