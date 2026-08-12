"""Native extraction vs Qwen OCR routing and ordered file output."""

from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass, field
import logging
from pathlib import Path
import threading
from typing import Callable, Protocol

import fitz
from openai import OpenAI

from . import config
from .native_extractor import (
    extract_native_page_text,
    find_repeated_margin_lines,
)
from .output_paths import build_output_paths
from .pdf_detection import detect_native_pdf
from .qwen_ocr import (
    is_blank_page,
    make_client,
    ocr_image_png,
    pdf_page_to_png_bytes,
)


_worker_state = threading.local()
logger = logging.getLogger(__name__)


class OCRBackend(Protocol):
    """Page OCR backend callable by the PDF processor."""

    def recognize(
        self,
        png_bytes: bytes,
        prompt: str,
        max_output_tokens: int,
    ) -> tuple[str, str | None, bool]: ...


@dataclass
class _OCRFileState:
    """Per-PDF scheduling and ordered-write state in the shared page pool."""

    pdf_path: Path
    output_path: Path
    temporary_path: Path
    next_page_to_submit: int
    last_page: int
    next_page_to_write: int
    pending_results: dict[int, str] = field(default_factory=dict)
    in_flight_pages: int = 0
    finished: bool = False


def _get_worker_document(pdf_path: Path) -> fitz.Document:
    """Thread-local PyMuPDF document (not shareable across threads)."""
    path_string = str(pdf_path)
    if getattr(_worker_state, "pdf_path", None) != path_string:
        previous = getattr(_worker_state, "document", None)
        if previous is not None:
            previous.close()
        _worker_state.document = fitz.open(pdf_path)
        _worker_state.pdf_path = path_string
    return _worker_state.document


def _process_ocr_page(
    pdf_path: Path,
    page_number: int,
    client: OpenAI | None,
    render_zoom: float,
    prompt: str,
    max_output_tokens: int,
    ocr_backend: OCRBackend | None,
) -> tuple[int, str, str]:
    """Render and OCR one page in a worker thread."""
    try:
        document = _get_worker_document(pdf_path)
        page = document[page_number - 1]
        if is_blank_page(page):
            return page_number, "", "blank"

        png_bytes = pdf_page_to_png_bytes(page, zoom=render_zoom)
        if ocr_backend is None:
            if client is None:
                raise RuntimeError("Qwen OCR client 未初始化")
            text, finish_reason, abnormal = ocr_image_png(
                client,
                png_bytes,
                prompt=prompt,
                max_output_tokens=max_output_tokens,
            )
        else:
            text, finish_reason, abnormal = ocr_backend.recognize(
                png_bytes,
                prompt,
                max_output_tokens,
            )
        status = finish_reason or "unknown"
        if abnormal:
            status += ", abnormal"
        return page_number, text, status
    except Exception as exc:
        logger.exception(
            "OCR 页面处理失败：PDF=%s，页码=%s",
            pdf_path,
            page_number,
        )
        text = f"[处理失败：{type(exc).__name__}: {exc}]"
        return page_number, text, "error"


def _append_page(
    output_path: Path,
    page_number: int,
    text: str,
    output_format: str,
) -> None:
    page_header = f"## 第 {page_number} 页"
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(f"{page_header}\n\n{text}\n\n")


def _create_ocr_file_state(
    pdf_path: Path,
    output_dir: Path,
    start_page: int,
    page_count: int,
    output_format: str,
) -> _OCRFileState:
    if page_count < 1:
        raise ValueError("page_count 必须大于或等于 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path, temporary_path = build_output_paths(
        pdf_path,
        output_dir,
        output_format,
    )
    temporary_path.write_text("", encoding="utf-8")
    return _OCRFileState(
        pdf_path=pdf_path,
        output_path=output_path,
        temporary_path=temporary_path,
        next_page_to_submit=start_page,
        last_page=start_page + page_count - 1,
        next_page_to_write=start_page,
    )


def _write_ready_ocr_pages(
    state: _OCRFileState,
    output_format: str,
) -> None:
    while state.next_page_to_write in state.pending_results:
        page_number = state.next_page_to_write
        text = state.pending_results.pop(page_number)
        _append_page(
            state.temporary_path,
            page_number,
            text,
            output_format,
        )
        state.next_page_to_write += 1


def process_ocr_jobs(
    jobs: list[tuple[Path, Path, int]],
    start_page: int = config.START_PAGE,
    render_zoom: float = config.RENDER_ZOOM,
    prompt: str = config.OCR_PROMPT,
    max_output_tokens: int = config.MAX_OUTPUT_TOKENS,
    max_workers: int = config.MAX_WORKERS,
    output_format: str = "txt",
    progress_callback: Callable[[], None] | None = None,
    ocr_backend: OCRBackend | None = None,
    activity_callback: Callable[[Path], None] | None = None,
) -> list[Path]:
    """OCR many PDFs on one shared page pool; write each file in page order.

    Submits pages preferring earlier PDFs first (fill the pool from the current
    file before starting the next). Completions may arrive out of order, but
    each output file is still appended strictly by page number.
    """
    if start_page < 1:
        raise ValueError("start_page 必须大于或等于 1")
    if max_workers < 1:
        raise ValueError("max_workers 必须大于或等于 1")
    if output_format not in {"txt", "md"}:
        raise ValueError("output_format 只能是 txt 或 md")
    if not jobs:
        return []

    unstarted_jobs = deque(jobs)
    states: list[_OCRFileState] = []
    client = make_client() if ocr_backend is None else None
    submission_state: _OCRFileState | None = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[
            Future[tuple[int, str, str]],
            tuple[_OCRFileState, int],
        ] = {}

        def submit_page(state: _OCRFileState) -> None:
            page_number = state.next_page_to_submit
            futures[
                executor.submit(
                    _process_ocr_page,
                    state.pdf_path,
                    page_number,
                    client,
                    render_zoom,
                    prompt,
                    max_output_tokens,
                    ocr_backend,
                )
            ] = (state, page_number)
            state.in_flight_pages += 1
            state.next_page_to_submit += 1

        def start_next_file() -> _OCRFileState:
            pdf_path, output_dir, page_count = unstarted_jobs.popleft()
            state = _create_ocr_file_state(
                pdf_path,
                output_dir,
                start_page,
                page_count,
                output_format,
            )
            states.append(state)
            if activity_callback is not None:
                activity_callback(pdf_path)
            return state

        def get_submission_state() -> _OCRFileState | None:
            """Current PDF still has pages to submit; else start the next one."""
            nonlocal submission_state
            while (
                submission_state is None
                or submission_state.next_page_to_submit
                > submission_state.last_page
            ):
                if not unstarted_jobs:
                    return None
                submission_state = start_next_file()
            return submission_state

        def fill_available_workers() -> None:
            while len(futures) < max_workers:
                state = get_submission_state()
                if state is None:
                    break
                submit_page(state)

        def finish_file_if_ready(state: _OCRFileState) -> None:
            if (
                state.finished
                or state.next_page_to_submit <= state.last_page
                or state.in_flight_pages != 0
            ):
                return
            _write_ready_ocr_pages(state, output_format)
            state.temporary_path.replace(state.output_path)
            state.finished = True

        fill_available_workers()
        while futures:
            completed_futures, _ = wait(
                futures,
                return_when=FIRST_COMPLETED,
            )
            for future in completed_futures:
                state, submitted_page = futures.pop(future)
                state.in_flight_pages -= 1
                try:
                    page_number, text, _status = future.result()
                except Exception as exc:
                    logger.exception(
                        "OCR 页面任务异常：PDF=%s，页码=%s",
                        state.pdf_path,
                        submitted_page,
                    )
                    page_number = submitted_page
                    text = f"[处理失败：{type(exc).__name__}: {exc}]"
                state.pending_results[page_number] = text
                _write_ready_ocr_pages(state, output_format)
                finish_file_if_ready(state)
                if progress_callback is not None:
                    progress_callback()
            fill_available_workers()

    for state in states:
        finish_file_if_ready(state)
    return [state.output_path for state in states]


def parse_pdf(
    pdf_path: Path,
    output_dir: Path | None = None,
    start_page: int = config.START_PAGE,
    end_page: int | None = config.END_PAGE,
    render_zoom: float = config.RENDER_ZOOM,
    prompt: str = config.OCR_PROMPT,
    max_output_tokens: int = config.MAX_OUTPUT_TOKENS,
    max_workers: int = config.MAX_WORKERS,
    output_format: str = "txt",
    progress_callback: Callable[[], None] | None = None,
    ocr_backend: OCRBackend | None = None,
    force_ocr: bool = False,
    force_native: bool = False,
) -> Path:
    """Process one PDF (native extract or OCR) into a single ordered file."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 不存在：{pdf_path}")
    if start_page < 1:
        raise ValueError("start_page 必须大于或等于 1")
    if max_workers < 1:
        raise ValueError("max_workers 必须大于或等于 1")
    if output_format not in {"txt", "md"}:
        raise ValueError("output_format 只能是 txt 或 md")
    if force_ocr and force_native:
        raise ValueError("force_ocr 和 force_native 不能同时启用")

    out_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path, temporary_path = build_output_paths(
        pdf_path,
        out_dir,
        output_format,
    )
    temporary_path.write_text("", encoding="utf-8")

    with fitz.open(pdf_path) as document:
        if force_native:
            is_native = True
        elif force_ocr:
            is_native = False
        else:
            is_native, _detection = detect_native_pdf(document)
        total_pages = document.page_count
        last_page = total_pages if end_page is None else min(end_page, total_pages)
        if start_page > last_page:
            raise ValueError(
                f"无可处理页面：start_page={start_page}, end_page={last_page}"
            )

        if is_native:
            repeated_margin_lines = find_repeated_margin_lines(
                document,
                1,
                total_pages,
            )
            for page_number in range(start_page, last_page + 1):
                page = document[page_number - 1]
                try:
                    text = extract_native_page_text(page, repeated_margin_lines)
                except Exception as exc:
                    logger.exception(
                        "原生文本提取失败：PDF=%s，页码=%s",
                        pdf_path,
                        page_number,
                    )
                    text = f"[处理失败：{type(exc).__name__}: {exc}]"
                _append_page(
                    temporary_path,
                    page_number,
                    text,
                    output_format,
                )
                if progress_callback is not None:
                    progress_callback()
        else:
            client = make_client() if ocr_backend is None else None
            pending_results: dict[int, tuple[str, str]] = {}
            next_page_to_write = start_page

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures: list[Future[tuple[int, str, str]]] = []
                for page_number in range(start_page, last_page + 1):
                    futures.append(
                        executor.submit(
                            _process_ocr_page,
                            pdf_path,
                            page_number,
                            client,
                            render_zoom,
                            prompt,
                            max_output_tokens,
                            ocr_backend,
                        )
                    )

                for future in as_completed(futures):
                    page_number, text, status = future.result()
                    pending_results[page_number] = (text, status)
                    if progress_callback is not None:
                        progress_callback()

                    # Completions may be out of order; file append stays page-ordered.
                    while next_page_to_write in pending_results:
                        page_text, _page_status = pending_results.pop(
                            next_page_to_write
                        )
                        _append_page(
                            temporary_path,
                            next_page_to_write,
                            page_text,
                            output_format,
                        )
                        next_page_to_write += 1

    temporary_path.replace(output_path)
    return output_path
