"""CLI for DeepSeek-OCR-2 multi-server batch processing."""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
from tqdm import tqdm

from . import config, deepseek_config
from .cli import find_pdf_files
from .deepseek_ocr import DeepSeekOCRPool
from .output_paths import build_output_paths
from .processor import parse_pdf


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dispatch every PDF page to DeepSeek-OCR-2 services on ports 8080-8087."
        ),
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
        default=deepseek_config.OUTPUT_DIR,
        help=f"Output root directory (default: {deepseek_config.OUTPUT_DIR})",
    )
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument("--start-page", type=int, default=config.START_PAGE)
    parser.add_argument("--end-page", type=int, default=config.END_PAGE)
    parser.add_argument(
        "--zoom",
        type=float,
        default=deepseek_config.RENDER_ZOOM,
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=deepseek_config.MAX_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=deepseek_config.PAGE_WORKERS,
        help=(
            "Page threads per scanned PDF "
            f"(default: {deepseek_config.PAGE_WORKERS})"
        ),
    )
    parser.add_argument(
        "--file-workers",
        type=int,
        default=deepseek_config.FILE_WORKERS,
        help=f"PDF file threads (default: {deepseek_config.FILE_WORKERS})",
    )
    parser.add_argument(
        "--per-server-concurrency",
        type=int,
        default=deepseek_config.PER_SERVER_CONCURRENCY,
        help=(
            "Max in-flight requests per port "
            f"(default: {deepseek_config.PER_SERVER_CONCURRENCY})"
        ),
    )
    parser.add_argument(
        "--model",
        default=deepseek_config.MODEL,
        help=f"Served model name (default: {deepseek_config.MODEL})",
    )
    parser.add_argument(
        "--format",
        choices=("txt", "md"),
        default="txt",
        dest="output_format",
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess PDFs that already have complete output files.",
    )
    parser.add_argument("--prompt-file", type=Path)
    return parser


def _count_pages(
    pdf_file: Path,
    start_page: int,
    end_page: int | None,
) -> int:
    with fitz.open(pdf_file) as document:
        last_page = (
            document.page_count
            if end_page is None
            else min(end_page, document.page_count)
        )
        return max(0, last_page - start_page + 1)


def main() -> None:
    args = build_argument_parser().parse_args()
    if (
        args.workers < 1
        or args.file_workers < 1
        or args.per_server_concurrency < 1
    ):
        raise SystemExit("所有并发参数必须大于或等于 1。")

    max_total_concurrency = (
        len(deepseek_config.API_BASES) * args.per_server_concurrency
    )
    requested_concurrency = args.workers * args.file_workers
    if requested_concurrency > max_total_concurrency:
        raise SystemExit(
            "并发配置过高："
            f"{args.file_workers} 个文件 × {args.workers} 个页面 "
            f"= {requested_concurrency}，超过 "
            f"{len(deepseek_config.API_BASES)} 个服务 × "
            f"{args.per_server_concurrency} = {max_total_concurrency}。"
        )

    input_paths = args.inputs or [config.DEFAULT_PDF_PATH]
    pdf_files = find_pdf_files(input_paths, recursive=args.recursive)
    if not pdf_files:
        raise SystemExit("没有找到可处理的 PDF 文件。")

    empty_pdf_files = [
        pdf_file for pdf_file in pdf_files if pdf_file.stat().st_size == 0
    ]
    if empty_pdf_files:
        print(f"发现 {len(empty_pdf_files)} 个空 PDF 文件，已跳过：")
        for pdf_file in empty_pdf_files:
            print(f"  {pdf_file}")
        empty_set = set(empty_pdf_files)
        pdf_files = [p for p in pdf_files if p not in empty_set]
    if not pdf_files:
        raise SystemExit("除空 PDF 外，没有可处理的 PDF 文件。")

    prompt = deepseek_config.OCR_PROMPT
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
    jobs: list[tuple[Path, Path]] = []
    for pdf_file in pdf_files:
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
        jobs.append((pdf_file, file_output_dir))

    if not jobs:
        print("没有需要处理的 PDF。")
        return

    readable_jobs: list[tuple[Path, Path]] = []
    page_counts: list[int] = []
    unreadable_files: list[tuple[Path, Exception]] = []
    for job in jobs:
        pdf_file, _ = job
        try:
            page_count = _count_pages(
                pdf_file,
                args.start_page,
                args.end_page,
            )
        except Exception as exc:
            unreadable_files.append((pdf_file, exc))
            continue
        readable_jobs.append(job)
        page_counts.append(page_count)

    if unreadable_files:
        print(f"发现 {len(unreadable_files)} 个无法读取的 PDF，已跳过：")
        for pdf_file, exc in unreadable_files:
            print(f"  {pdf_file} [{type(exc).__name__}: {exc}]")

    jobs = readable_jobs
    if not jobs:
        raise SystemExit("没有可正常读取并处理的 PDF 文件。")

    backend = DeepSeekOCRPool(
        model=args.model,
        per_server_concurrency=args.per_server_concurrency,
    )

    def process_file(
        job: tuple[Path, Path],
        progress_callback,
    ) -> Path:
        pdf_file, file_output_dir = job
        return parse_pdf(
            pdf_file,
            output_dir=file_output_dir,
            start_page=args.start_page,
            end_page=args.end_page,
            render_zoom=args.zoom,
            prompt=prompt,
            max_output_tokens=args.max_output_tokens,
            max_workers=args.workers,
            output_format=args.output_format,
            progress_callback=progress_callback,
            ocr_backend=backend,
            force_ocr=True,
        )

    with tqdm(
        total=sum(page_counts),
        desc="DeepSeek OCR",
        unit="页",
        dynamic_ncols=True,
    ) as progress_bar:
        with ThreadPoolExecutor(max_workers=args.file_workers) as executor:
            futures: dict[Future[Path], Path] = {
                executor.submit(process_file, job, progress_bar.update): job[0]
                for job in jobs
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    progress_bar.write(
                        "DeepSeek PDF 处理失败："
                        f"{futures[future]} "
                        f"[{type(exc).__name__}: {exc}]"
                    )
