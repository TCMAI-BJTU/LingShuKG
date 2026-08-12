"""Run recursive knowledge-graph extraction for a text directory."""

from __future__ import annotations

import os

# Clear proxy env vars so later HTTP calls are not redirected.
for var in [
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
]:
    os.environ.pop(var, None)


import argparse
import json
from pathlib import Path

from kg_agent import (
    AgentSettings,
    DirectoryRunner,
    GraphSchema,
    GraphStore,
    LLMSettings,
    OpenAIChatClient,
    ReActChunkAgent,
    WorkspaceCache,
    load_cache_namespace,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "state" / "kg.sqlite"
DEFAULT_CACHE_DIR = BASE_DIR / "state" / "cache"
DEFAULT_CACHE_NAMESPACE_PATH = BASE_DIR / "state" / "cache_namespaces.json"


def build_argument_parser() -> argparse.ArgumentParser:
    llm_defaults = LLMSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Recursively process TXT/Markdown under a directory and mirror JSON results."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=BASE_DIR / "schemas" / "example.json",
    )
    parser.add_argument("--base-url", default=llm_defaults.base_url)
    parser.add_argument("--api-key", default=llm_defaults.api_key)
    parser.add_argument("--model", default=llm_defaults.model)
    parser.add_argument("--temperature", type=float, default=llm_defaults.temperature)
    parser.add_argument("--max-tokens", type=int, default=llm_defaults.max_tokens)
    parser.add_argument("--timeout", type=float, default=llm_defaults.timeout)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument(
        "--workers",
        type=int,
        default=128,
        help="Concurrent chunks in normal mode; forced to 1 in --debug",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Stream model responses and tool traces; forces serial execution",
    )
    return parser


def run_directory(args: argparse.Namespace) -> dict:
    if not args.data_dir.is_dir():
        raise NotADirectoryError(f"Data directory does not exist: {args.data_dir}")
    schema = GraphSchema.from_json_file(args.schema)
    store = GraphStore(DEFAULT_DB_PATH)
    cache = WorkspaceCache(DEFAULT_CACHE_DIR)
    llm_settings = LLMSettings(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        stream=args.debug,
    )
    # Cache namespace is model-bound only; schema edits do not invalidate checkpoints.
    cache_namespace = load_cache_namespace(
        DEFAULT_CACHE_NAMESPACE_PATH,
        llm_settings.model,
    )
    agent = ReActChunkAgent(
        OpenAIChatClient(llm_settings),
        schema,
        AgentSettings(max_steps=args.max_steps, debug=args.debug),
    )
    runner = DirectoryRunner(
        agent=agent,
        schema=schema,
        store=store,
        cache=cache,
        output_dir=args.output_dir,
        cache_namespace=cache_namespace,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        workers=1 if args.debug else args.workers,
    )
    result = runner.run(args.data_dir)
    cache.close()
    store.close()
    return result


def main() -> None:
    args = build_argument_parser().parse_args()
    result = run_directory(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
