"""CLI entry for batch PDF processing (native extract or Qwen OCR)."""

import argparse
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path

import fitz
from tqdm import tqdm

from . import config
from .output_paths import build_output_paths
from .pdf_detection import detect_native_pdf
from .processor import parse_pdf, process_ocr_jobs


UNSAFE_MUPDF_WARNING_FRAGMENTS = (
    "invalid key in dict",
    "expected object number",
    "cannot find object",
    "object out of range",
    "invalid xref",
)


def _configure_mupdf_logging() -> None:
    """Hide recoverable MuPDF warnings; Python exceptions still raise."""
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)


def _process_native_candidate(
    task: tuple[
        Path,
        Path,
        int,
        int | None,
        float,
        int,
        str,
    ],
) -> tuple[bool, Path]:
    """Strict-detect and extract one high-confidence native PDF in a process."""
    _configure_mupdf_logging()
    fitz.TOOLS.reset_mupdf_warnings()
    (
        pdf_file,
        file_output_dir,
        start_page,
        end_page,
        render_zoom,
        max_output_tokens,
        output_format,
    ) = task
    with fitz.open(pdf_file) as document:
        is_native, _details = detect_native_pdf(document, strict=True)
    mupdf_warnings = fitz.TOOLS.mupdf_warnings(reset=True).lower()
    has_unsafe_warning = any(
        fragment in mupdf_warnings
        for fragment in UNSAFE_MUPDF_WARNING_FRAGMENTS
    )
    if not is_native or has_unsafe_warning:
        return False, pdf_file

    parse_pdf(
        pdf_file,
        output_dir=file_output_dir,
        start_page=start_page,
        end_page=end_page,
        render_zoom=render_zoom,
        max_output_tokens=max_output_tokens,
        output_format=output_format,
        force_native=True,
    )
    return True, pdf_file


def find_pdf_files(inputs: list[Path], recursive: bool = False) -> list[Path]:
    """Expand inputs to unique sorted PDF paths."""
    files: set[Path] = set()
    for input_path in inputs:
        path = input_path.expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() == ".pdf":
                files.add(path)
            else:
                print(f"跳过非 PDF 文件：{path}", flush=True)
        elif path.is_dir():
            pattern = "**/*.pdf" if recursive else "*.pdf"
            files.update(item.resolve() for item in path.glob(pattern))
        else:
            print(f"路径不存在，已跳过：{path}", flush=True)
    return sorted(files)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Qwen3.6-27B OCR on every PDF page.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="PDF files or directories; defaults to the configured sample PDF.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR,
        help=f"Output root directory (default: {config.OUTPUT_DIR})",
    )
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument("--start-page", type=int, default=config.START_PAGE)
    parser.add_argument("--end-page", type=int, default=config.END_PAGE)
    parser.add_argument("--zoom", type=float, default=config.RENDER_ZOOM)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=config.MAX_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.MAX_WORKERS,
        help=f"Shared page OCR threads across PDFs (default: {config.MAX_WORKERS})",
    )
    parser.add_argument(
        "--file-workers",
        type=int,
        default=config.FILE_WORKERS,
        help=(
            "Native-PDF process count for --native-only "
            f"(default: {config.FILE_WORKERS})"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("txt", "md"),
        default="txt",
        dest="output_format",
        help="Output suffix (default: txt); does not change OCR content.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess PDFs that already have complete output files.",
    )
    parser.add_argument(
        "--native-only",
        action="store_true",
        help=(
            "Only extract high-confidence native PDFs; leave others for Qwen OCR."
        ),
    )
    parser.add_argument("--prompt-file", type=Path)
    return parser


def main() -> None:
    _configure_mupdf_logging()
    args = build_argument_parser().parse_args()
    if args.workers < 1 or args.file_workers < 1:
        raise SystemExit("--workers 和 --file-workers 必须大于或等于 1。")
    if args.native_only and args.file_workers > config.MAX_NATIVE_PROCESSES:
        raise SystemExit(
            f"原生 PDF 进程数不能超过 {config.MAX_NATIVE_PROCESSES}。"
        )
    if (
        not args.native_only
        and args.workers > config.MAX_TOTAL_CONCURRENCY
    ):
        raise SystemExit(
            "页面 OCR 线程数过高："
            f"{args.workers}，不能超过 "
            f"{config.MAX_TOTAL_CONCURRENCY}。"
        )

    input_paths = args.inputs or [config.DEFAULT_PDF_PATH]
    pdf_files = find_pdf_files(input_paths, recursive=args.recursive)
    if not pdf_files:
        raise SystemExit("没有找到可处理的 PDF 文件。")

    empty_pdf_files = [
        pdf_file for pdf_file in pdf_files if pdf_file.stat().st_size == 0
    ]
    if empty_pdf_files:
        print(f"发现 {len(empty_pdf_files)} 个空 PDF 文件，已跳过：", flush=True)
        for pdf_file in empty_pdf_files:
            print(f"  {pdf_file}", flush=True)
        pdf_files = [
            pdf_file for pdf_file in pdf_files if pdf_file.stat().st_size > 0
        ]
    if not pdf_files:
        raise SystemExit("除空 PDF 外，没有可处理的 PDF 文件。")

    prompt = config.OCR_PROMPT
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
        if not prompt:
            raise SystemExit("自定义提示词文件为空。")

    directory_roots = sorted(
        (
            path.expanduser().resolve()
            for path in input_paths
            if path.expanduser().resolve().is_dir()
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    jobs: list[tuple[int, Path, Path]] = []
    for file_index, pdf_file in enumerate(pdf_files, start=1):
        relative_parent: Path | None = None
        for directory_root in directory_roots:
            try:
                relative_parent = pdf_file.relative_to(directory_root).parent
                break
            except ValueError:
                continue
        file_output_dir = (
            args.output_dir / relative_parent
            if relative_parent is not None
            else args.output_dir
        )
        completed_output, _temporary_output = build_output_paths(
            pdf_file,
            file_output_dir,
            args.output_format,
        )
        if completed_output.is_file() and not args.overwrite:
            continue
        jobs.append((file_index, pdf_file, file_output_dir))

    if not jobs:
        print("没有需要处理的 PDF。", flush=True)
        return

    if args.native_only:
        native_tasks = [
            (
                pdf_file,
                file_output_dir,
                args.start_page,
                args.end_page,
                args.zoom,
                args.max_output_tokens,
                args.output_format,
            )
            for _, pdf_file, file_output_dir in jobs
        ]

        native_count = 0
        deferred_count = 0
        with tqdm(
            total=len(jobs),
            desc="筛选并读取原生 PDF",
            unit="文件",
            dynamic_ncols=True,
        ) as progress_bar:
            with ProcessPoolExecutor(max_workers=args.file_workers) as executor:
                futures: dict[Future[tuple[bool, Path]], Path] = {
                    executor.submit(_process_native_candidate, task): task[0]
                    for task in native_tasks
                }
                for future in as_completed(futures):
                    try:
                        is_native, _pdf_file = future.result()
                        if is_native:
                            native_count += 1
                        else:
                            deferred_count += 1
                    except Exception as exc:
                        deferred_count += 1
                        progress_bar.write(
                            "原生 PDF 处理失败："
                            f"{futures[future]} "
                            f"[{type(exc).__name__}: {exc}]"
                        )
                    finally:
                        progress_bar.update()

        print(
            f"高置信原生 PDF 已读取：{native_count} 个；"
            f"留给 Qwen OCR：{deferred_count} 个。",
            flush=True,
        )
        return

    def count_pages(pdf_file: Path) -> int:
        with fitz.open(pdf_file) as document:
            last_page = (
                document.page_count
                if args.end_page is None
                else min(args.end_page, document.page_count)
            )
            return max(0, last_page - args.start_page + 1)

    readable_jobs: list[tuple[int, Path, Path]] = []
    page_counts: list[int] = []
    unreadable_files: list[tuple[Path, Exception]] = []
    no_page_files: list[Path] = []
    for job in jobs:
        _, pdf_file, _ = job
        try:
            page_count = count_pages(pdf_file)
        except Exception as exc:
            unreadable_files.append((pdf_file, exc))
            continue
        if page_count < 1:
            no_page_files.append(pdf_file)
            continue
        readable_jobs.append(job)
        page_counts.append(page_count)

    if unreadable_files:
        print(
            f"发现 {len(unreadable_files)} 个无法读取的 PDF 文件，已跳过：",
            flush=True,
        )
        for pdf_file, exc in unreadable_files:
            print(
                f"  {pdf_file} [{type(exc).__name__}: {exc}]",
                flush=True,
            )

    if no_page_files:
        print(
            f"发现 {len(no_page_files)} 个当前页码范围内没有页面的 PDF，"
            "已跳过：",
            flush=True,
        )
        for pdf_file in no_page_files:
            print(f"  {pdf_file}", flush=True)

    jobs = readable_jobs
    if not jobs:
        raise SystemExit("没有可正常读取并处理的 PDF 文件。")
    total_pages = sum(page_counts)

    qwen_jobs = [
        (pdf_file, file_output_dir, page_count)
        for (_, pdf_file, file_output_dir), page_count in zip(
            jobs,
            page_counts,
        )
    ]
    with tqdm(
        total=total_pages,
        desc="处理 PDF 页面",
        unit="页",
        dynamic_ncols=True,
    ) as progress_bar:
        def show_active_file(pdf_path: Path) -> None:
            filename = pdf_path.name
            max_filename_length = 72
            if len(filename) > max_filename_length:
                prefix_length = 35
                suffix_length = max_filename_length - prefix_length - 1
                filename = f"{filename[:prefix_length]}…{filename[-suffix_length:]}"
            progress_bar.set_postfix_str(f"最近启动：{filename}", refresh=True)

        process_ocr_jobs(
            qwen_jobs,
            start_page=args.start_page,
            render_zoom=args.zoom,
            prompt=prompt,
            max_output_tokens=args.max_output_tokens,
            max_workers=args.workers,
            output_format=args.output_format,
            progress_callback=progress_bar.update,
            activity_callback=show_active_file,
        )
