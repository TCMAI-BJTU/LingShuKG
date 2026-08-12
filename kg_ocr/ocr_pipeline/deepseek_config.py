"""DeepSeek-OCR-2 multi-server batch configuration."""

from pathlib import Path

from . import config


API_BASES = tuple(
    f"http://127.0.0.1:{port}/v1" for port in range(8080, 8088)
)
MODEL = "deepseek-ocr-2"
API_KEY = "EMPTY"

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "ocr_deepseek"

# 8 servers × max 128 in-flight requests each.
PER_SERVER_CONCURRENCY = 128
FILE_WORKERS = 8
PAGE_WORKERS = 128
MAX_TOTAL_CONCURRENCY = len(API_BASES) * PER_SERVER_CONCURRENCY

RENDER_ZOOM = config.RENDER_ZOOM
MAX_OUTPUT_TOKENS = 4096

# No grounding enabled; tags are only cleaned as abnormal fallback.
OCR_PROMPT = "<image>\nFree OCR."

NGRAM_SIZE = 30
WINDOW_SIZE = 90
WHITELIST_TOKEN_IDS = [128821, 128822]
