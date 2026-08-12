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
    """Recursively find supported text files and sort them stably."""
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces, tabs, and newlines into a single space."""
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
        """Process a directory recursively and return summary stats."""
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
        """Lazily yield tasks in file/chunk order without materializing all objects."""
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
        # Pre-slot results by task index so concurrent finish order does not scramble output.
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
        """Run tasks serially; convert per-task exceptions into failure results and continue."""
        for index, task in enumerate(tasks):
            try:
                result = self.process_chunk(
                    task.source_name,
                    task.relative_path,
                    task.chunk_index,
                    task.text,
                )
            except Exception as error:
                # One chunk failure must not stop the rest of the directory run.
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
        """Run a bounded set of Futures and close per-thread reused connections afterward."""
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                task_iterator = iter(enumerate(tasks))
                futures = {}
                # Start only worker-count Futures to avoid holding all chunks in memory.
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
                            # Print tracebacks on the main thread to avoid interleaved worker logs.
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
        """Return this worker's reused store, creating and registering it on first use."""
        store = getattr(self._worker_local, "store", None)
        if store is not None:
            return store
        # Queries stay on the owning worker; the main thread may close connections after the pool ends.
        store = GraphStore(self.store.path, check_same_thread=False)
        self._worker_local.store = store
        with self._worker_stores_lock:
            self._worker_stores.append(store)
        return store

    def _close_parallel_stores(self) -> None:
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
        """Process one text file and write the mirrored JSON result."""
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
        """Build a stable internal source key from relative path and chunk index."""
        # Internal source keys distinguish same-named/same-content files under different paths.
        return (
            f"{self.cache_namespace}:"
            f"{relative_path.as_posix()}#chunk_{chunk_index}"
        )

    def output_path(self, relative_path: Path) -> Path:
        return self.output_dir / relative_path.with_suffix(".json")
