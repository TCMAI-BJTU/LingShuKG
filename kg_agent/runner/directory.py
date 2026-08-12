"""Recursive directory processing and mirrored JSON export."""

from __future__ import annotations

import json
import sys
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, local
from typing import Any, Iterable, Iterator

from tqdm.auto import tqdm

from ..agent.loop import ReActChunkAgent
from ..agent.semantic_review import ContextualSemanticReviewer
from ..core.schema import GraphSchema
from ..observability import log_slow_operation, timing_context
from ..persistence.cache import WorkspaceCache
from ..persistence.storage import GraphStore, make_source_id
from ..tools.toolset import open_chunk_toolset
from .chunking import split_text


SUPPORTED_EXTENSIONS = frozenset({".txt", ".md"})


@dataclass(frozen=True)
class ChunkTask:
    file_index: int
    source_name: str
    relative_path: Path
    chunk_index: int
    text: str


def find_text_files(data_dir: Path) -> list[Path]:
    """递归查找受支持的文本文件并稳定排序。"""
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def normalize_whitespace(text: str) -> str:
    """将连续空格、制表符和换行统一合并为一个普通空格。"""
    return " ".join(text.split())


class DirectoryRunner:
    def __init__(
        self,
        agent: ReActChunkAgent,
        schema: GraphSchema,
        store: GraphStore,
        cache: WorkspaceCache,
        output_dir: Path,
        cache_namespace: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        workers: int = 1,
    ) -> None:
        """保存目录处理所需依赖和分块参数。"""
        self.agent = agent
        self.schema = schema
        self.store = store
        self.cache = cache
        self.output_dir = output_dir
        self.cache_namespace = cache_namespace
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._worker_local = local()
        self._worker_stores: list[GraphStore] = []
        self._worker_stores_lock = Lock()
        if workers < 1:
            raise ValueError("workers 必须大于或等于 1")
        self.workers = 1 if agent.settings.debug else workers
        self.reviewer = ContextualSemanticReviewer(
            agent.client,
            schema,
            agent.settings.debug,
            cache,
            cache_namespace,
        )

    def run(self, data_dir: Path) -> dict[str, Any]:
        """递归处理目录并返回汇总统计。"""
        with log_slow_operation(
            "directory.scan",
            data_dir=str(data_dir),
        ):
            files = find_text_files(data_dir)
        with log_slow_operation(
            "directory.prepare",
            file_count=len(files),
        ):
            prepared = [
                self._prepare_file(data_dir, path)
                for path in files
            ]
        task_count = sum(len(item["chunks"]) for item in prepared)
        indexed_results = self._run_tasks(
            self._iter_tasks(prepared),
            task_count,
        )
        grouped: list[list[dict[str, Any]]] = [[] for _ in prepared]
        for file_index, result in indexed_results:
            grouped[file_index].append(result)
        with log_slow_operation(
            "directory.write_results",
            file_count=len(prepared),
        ):
            file_results = [
                self._write_file_result(item, grouped[index])
                for index, item in enumerate(prepared)
            ]
        chunks = [
            chunk
            for file_result in file_results
            for chunk in file_result["chunks"]
        ]
        return {
            "data_dir": str(data_dir),
            "output_dir": str(self.output_dir),
            "file_count": len(file_results),
            "chunk_count": len(chunks),
            "committed_count": sum(
                chunk["status"] == "committed" for chunk in chunks
            ),
            "skipped_count": sum(
                chunk["status"] == "already_committed" for chunk in chunks
            ),
            "failed_count": sum(
                chunk["status"] == "failed" for chunk in chunks
            ),
            "incomplete_count": sum(
                chunk["status"] not in {"committed", "already_committed"}
                for chunk in chunks
            ),
        }

    def _iter_tasks(
        self,
        prepared: list[dict[str, Any]],
    ) -> Iterator[ChunkTask]:
        """按文件和 chunk 顺序惰性生成任务，避免一次性创建全部对象。"""
        for file_index, item in enumerate(prepared):
            for chunk_index, chunk in enumerate(item["chunks"]):
                yield ChunkTask(
                    file_index=file_index,
                    source_name=item["source_name"],
                    relative_path=item["relative_path"],
                    chunk_index=chunk_index,
                    text=chunk,
                )

    def _run_tasks(
        self,
        tasks: Iterable[ChunkTask],
        task_count: int,
    ) -> list[tuple[int, dict[str, Any]]]:
        """使用一条全局 tqdm 进度条串行或并发执行全部 chunk。"""
        # 预先按任务序号占位，使并发完成顺序不会改变最终输出顺序。
        results: list[tuple[int, dict[str, Any]] | None] = [None] * task_count
        disabled = self.agent.settings.debug or not sys.stderr.isatty()
        with tqdm(
            total=task_count,
            desc="知识图谱抽取",
            unit="chunk",
            dynamic_ncols=True,
            disable=disabled,
        ) as progress:
            if self.workers == 1:
                self._run_serial_tasks(tasks, results, progress)
            else:
                self._run_parallel_tasks(
                    tasks,
                    task_count,
                    results,
                    progress,
                )
        return [result for result in results if result is not None]

    def _run_serial_tasks(
        self,
        tasks: Iterable[ChunkTask],
        results: list[tuple[int, dict[str, Any]] | None],
        progress: Any,
    ) -> None:
        """串行执行任务，并将单任务异常转换为失败结果后继续。"""
        for index, task in enumerate(tasks):
            try:
                result = self.process_chunk(
                    task.source_name,
                    task.relative_path,
                    task.chunk_index,
                    task.text,
                )
            except Exception as error:
                # 单个 chunk 失败不应中断目录中其余任务。
                result = self._failed_task_result(task, error)
            results[index] = (task.file_index, result)
            progress.update(1)

    def _run_parallel_tasks(
        self,
        tasks: Iterable[ChunkTask],
        task_count: int,
        results: list[tuple[int, dict[str, Any]] | None],
        progress: Any,
    ) -> None:
        """并发执行有限数量 Future，并在结束后关闭各线程复用的连接。"""
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                task_iterator = iter(enumerate(tasks))
                futures = {}
                # 初始只创建 worker 数量的 Future，避免海量 chunk 同时占用内存。
                for _ in range(min(self.workers, task_count)):
                    item = next(task_iterator, None)
                    if item is None:
                        break
                    index, task = item
                    future = executor.submit(
                        self._process_parallel_task,
                        task,
                    )
                    futures[future] = (index, task)
                while futures:
                    completed, _ = wait(
                        futures,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in completed:
                        index, task = futures.pop(future)
                        try:
                            result = future.result()
                        except Exception as error:
                            # 在主线程统一打印 traceback，避免 worker 日志交错。
                            result = self._failed_task_result(task, error)
                        results[index] = (task.file_index, result)
                        progress.update(1)
                        item = next(task_iterator, None)
                        if item is None:
                            continue
                        next_index, next_task = item
                        next_future = executor.submit(
                            self._process_parallel_task,
                            next_task,
                        )
                        futures[next_future] = (next_index, next_task)
        finally:
            self._close_parallel_stores()

    def _failed_task_result(
        self,
        task: ChunkTask,
        error: Exception,
    ) -> dict[str, Any]:
        """打印单个 chunk 的异常，并返回可写入输出文件的失败结果。"""
        print(
            (
                "\n[chunk failed] "
                f"file={task.relative_path.as_posix()} "
                f"chunk={task.chunk_index}"
            ),
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exception(
            type(error),
            error,
            error.__traceback__,
            file=sys.stderr,
        )
        source_key = self._source_key(task.relative_path, task.chunk_index)
        return {
            "chunk_index": task.chunk_index,
            "source_id": make_source_id(
                task.source_name,
                task.text,
                source_key,
            ),
            "status": "failed",
            "steps": 0,
            "entities": [],
            "relations": [],
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }

    def _prepare_file(self, data_dir: Path, path: Path) -> dict[str, Any]:
        """读取、清理并切分一个文件，但不运行模型。"""
        relative_path = path.relative_to(data_dir)
        with timing_context(file=relative_path.as_posix()):
            with log_slow_operation("file.prepare"):
                text = normalize_whitespace(
                    path.read_text(encoding="utf-8", errors="ignore")
                )
                return {
                    "source_name": path.name,
                    "relative_path": relative_path,
                    "chunks": split_text(
                        text,
                        self.chunk_size,
                        self.chunk_overlap,
                    ),
                }

    def _process_parallel_task(self, task: ChunkTask) -> dict[str, Any]:
        """使用当前 worker 线程复用的 SQLite 连接处理一个 chunk。"""
        with timing_context(
            file=task.relative_path.as_posix(),
            chunk=task.chunk_index,
        ):
            store = self._parallel_store()
            return self.process_chunk(
                task.source_name,
                task.relative_path,
                task.chunk_index,
                task.text,
                store,
            )

    def _parallel_store(self) -> GraphStore:
        """获取当前 worker 的复用连接，首次调用时创建并登记。"""
        store = getattr(self._worker_local, "store", None)
        if store is not None:
            return store
        # 连接仍只在所属 worker 内执行查询，但允许线程池结束后由主线程关闭。
        store = GraphStore(self.store.path, check_same_thread=False)
        self._worker_local.store = store
        with self._worker_stores_lock:
            self._worker_stores.append(store)
        return store

    def _close_parallel_stores(self) -> None:
        """在线程池结束后关闭并清空本轮创建的全部 worker 连接。"""
        with self._worker_stores_lock:
            stores = list(self._worker_stores)
            self._worker_stores.clear()
        for store in stores:
            store.close()

    def _write_file_result(
        self,
        prepared: dict[str, Any],
        chunk_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """按输入顺序组装并写出一个文件的镜像 JSON 结果。"""
        relative_path = prepared["relative_path"]
        with timing_context(file=relative_path.as_posix()):
            with log_slow_operation("file.write_result"):
                result = {
                    "source_name": prepared["source_name"],
                    "relative_path": relative_path.as_posix(),
                    "chunks": chunk_results,
                }
                output_path = self.output_path(relative_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return result

    def process_file(self, data_dir: Path, path: Path) -> dict[str, Any]:
        """处理单个文本文件并写出镜像 JSON 结果。"""
        prepared = self._prepare_file(data_dir, path)
        chunk_results = [
            self.process_chunk(
                prepared["source_name"],
                prepared["relative_path"],
                index,
                chunk,
            )
            for index, chunk in enumerate(prepared["chunks"])
        ]
        return self._write_file_result(prepared, chunk_results)

    def process_chunk(
        self,
        source_name: str,
        relative_path: Path,
        chunk_index: int,
        text: str,
        store: GraphStore | None = None,
    ) -> dict[str, Any]:
        """运行一个片段并读取其最终数据库结果。"""
        with timing_context(
            file=relative_path.as_posix(),
            chunk=chunk_index,
        ):
            graph_store = store or self.store
            source_key = self._source_key(relative_path, chunk_index)
            with log_slow_operation("chunk.open_workspace"):
                toolset = open_chunk_toolset(
                    source_name,
                    text,
                    self.schema,
                    graph_store,
                    self.cache,
                    self.cache_namespace,
                    source_key,
                    self.reviewer,
                )
            run_result = self.agent.run(toolset)
            with log_slow_operation("database.read_chunk_result"):
                stored = graph_store.get_source_result(
                    toolset.source_id
                )
            return {
                "chunk_index": chunk_index,
                "source_id": toolset.source_id,
                "status": run_result.status,
                "steps": run_result.steps,
                "entities": (
                    stored["entities"] if stored is not None else []
                ),
                "relations": (
                    stored["relations"] if stored is not None else []
                ),
            }

    def _source_key(self, relative_path: Path, chunk_index: int) -> str:
        """根据相对路径和 chunk 序号生成稳定的内部来源键。"""
        # 内部来源键可区分不同目录下同名、同内容的文件。
        return (
            f"{self.cache_namespace}:"
            f"{relative_path.as_posix()}#chunk_{chunk_index}"
        )

    def output_path(self, relative_path: Path) -> Path:
        """将输入相对路径映射为输出目录中的 JSON 路径。"""
        return self.output_dir / relative_path.with_suffix(".json")
