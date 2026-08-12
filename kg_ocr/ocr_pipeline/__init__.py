"""Native PDF text extraction and Qwen OCR routing."""

from .pdf_detection import detect_native_pdf
from .processor import parse_pdf, process_ocr_jobs

__all__ = ["detect_native_pdf", "parse_pdf", "process_ocr_jobs"]
