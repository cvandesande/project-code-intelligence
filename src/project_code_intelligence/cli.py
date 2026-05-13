"""Public command-line entry points."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast

from project_code_intelligence import config, console_ui, ingest_code_intel, mcp_smoke_render, process, progress
from project_code_intelligence.common import default_collection
from project_code_intelligence.embeddings import resolve_embedding_endpoint_model

DEFAULT_EMBED_RECORD_TYPES = (
    "code_chunk,package_definition,config_symbol,patch_hunk,dts_node,"
    "service_entrypoint,security_pattern,static_finding,doc_section"
)


def index_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index code intelligence in pgvector.")
    _ = parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON progress events and reports instead of the pretty TTY display.",
    )
    _ = parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested without writing.")
    _ = parser.add_argument(
        "--collection",
        help=(
            "Collection/workspace name. Defaults to the repo name for one path, "
            "or the common parent directory name for multiple paths."
        ),
    )
    _ = parser.add_argument(
        "--reset-code-intel",
        "--reset",
        action="store_true",
        help="Delete code-intelligence data for the given repository path(s), then exit.",
    )
    _ = parser.add_argument(
        "--reset-all-code-intel",
        "--reset-all",
        action="store_true",
        help="Delete all code-intelligence data in the configured database, then exit.",
    )
    _ = parser.add_argument(
        "--i-know-this-deletes-code-intel-db",
        action="store_true",
        help="Skip interactive confirmation for --reset-code-intel.",
    )
    embedding_group = parser.add_mutually_exclusive_group()
    _ = embedding_group.add_argument(
        "--embed",
        dest="embed",
        action="store_true",
        default=None,
        help="Embed indexed records. This is the pci-index default.",
    )
    _ = embedding_group.add_argument(
        "--no-embed",
        dest="embed",
        action="store_false",
        help="Create a text-only index without semantic embeddings.",
    )
    _ = parser.add_argument(
        "repo_paths",
        nargs="*",
        help="Repository path(s) to index. Use . for the current directory.",
    )
    return parser


class IndexNamespace(argparse.Namespace):
    json: bool
    dry_run: bool
    collection: str | None
    reset_code_intel: bool
    reset_all_code_intel: bool
    i_know_this_deletes_code_intel_db: bool
    embed: bool | None
    repo_paths: list[str]


def normalized_passthrough(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def split_index_argv(argv: list[str] | None) -> tuple[list[str], list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--" not in values:
        return values, []
    separator = values.index("--")
    return values[:separator], values[separator + 1 :]


def repo_paths_to_ingest_args(repo_paths: list[str]) -> list[str]:
    absolute_paths = [Path(path).expanduser().resolve(strict=False) for path in repo_paths]
    if len(absolute_paths) == 1:
        repo_path = absolute_paths[0]
        root = repo_path.parent
        repos = repo_path.name or "."
    else:
        root = Path(os.path.commonpath([str(path) for path in absolute_paths]))
        repos = ",".join(path.relative_to(root).as_posix() or "." for path in absolute_paths)
    return ["--root", str(root), "--repos", repos]


def inferred_collection_for_repo_paths(repo_paths: list[str]) -> str:
    absolute_paths = [Path(path).expanduser().resolve(strict=False) for path in repo_paths]
    if len(absolute_paths) == 1:
        return default_collection(absolute_paths[0])
    root = Path(os.path.commonpath([str(path) for path in absolute_paths]))
    return default_collection(root)


def parse_index_args(argv: list[str] | None = None) -> tuple[IndexNamespace, list[str]]:
    public_argv, passthrough = split_index_argv(argv)
    parser = index_parser()
    parsed = parser.parse_args(public_argv, namespace=IndexNamespace())
    if parsed.reset_code_intel and parsed.reset_all_code_intel:
        parser.error("--reset and --reset-all cannot be combined")
    if parsed.reset_all_code_intel and parsed.repo_paths:
        parser.error("--reset-all does not accept repository paths")
    if parsed.reset_all_code_intel and parsed.collection:
        parser.error("--reset-all does not accept --collection")
    if not parsed.reset_all_code_intel and not parsed.repo_paths:
        parser.error("one or more repository paths are required; use . for the current directory")
    return parsed, normalized_passthrough(passthrough)


def set_index_environment_defaults() -> None:
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_PROFILE", "generic")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_MODE", "incremental")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_EMBED", "1")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_EMBED_ONLY", "0")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_PREEMBED", "1")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_EMBEDDING_BATCH_SIZE", "32")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_EMBEDDING_MAX_CHARS", "3000")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_EMBED_RECORD_TYPES", DEFAULT_EMBED_RECORD_TYPES)


def apply_index_embed_override(parsed: IndexNamespace) -> None:
    if parsed.embed is not None:
        os.environ["PROJECT_CODE_INTELLIGENCE_EMBED"] = "1" if parsed.embed else "0"
        if not parsed.embed:
            os.environ["PROJECT_CODE_INTELLIGENCE_EMBED_ONLY"] = "0"


def forwarded_index_args(parsed: IndexNamespace, passthrough: list[str]) -> list[str]:
    forwarded = passthrough
    if parsed.repo_paths:
        forwarded = [*repo_paths_to_ingest_args(parsed.repo_paths), *forwarded]
        if parsed.collection:
            forwarded = ["--collection", parsed.collection, *forwarded]
        elif not config.env_text("PROJECT_CODE_INTELLIGENCE_COLLECTION"):
            forwarded = ["--collection", inferred_collection_for_repo_paths(parsed.repo_paths), *forwarded]
    if parsed.reset_code_intel:
        forwarded = ["--reset-code-intel", "--reset-only", *forwarded]
    if parsed.reset_all_code_intel:
        forwarded = ["--reset-all-code-intel", "--reset-code-intel", "--reset-only", *forwarded]
    if parsed.i_know_this_deletes_code_intel_db:
        forwarded = ["--i-know-this-deletes-code-intel-db", *forwarded]
    if parsed.dry_run:
        forwarded = [*forwarded, "--dry-run"]
    else:
        os.environ["PROJECT_CODE_INTELLIGENCE_ALLOW_WRITES"] = "1"
    return forwarded


def index_embedding_args(*, embed_only: bool) -> list[str]:
    embedding_args = [
        "--embedding-batch-size",
        os.environ["PROJECT_CODE_INTELLIGENCE_EMBEDDING_BATCH_SIZE"],
        "--embedding-max-chars",
        os.environ["PROJECT_CODE_INTELLIGENCE_EMBEDDING_MAX_CHARS"],
        "--embed-record-types",
        os.environ["PROJECT_CODE_INTELLIGENCE_EMBED_RECORD_TYPES"],
    ]
    endpoint = config.default_embedding_endpoint(local_default=True)
    if endpoint:
        model = resolve_embedding_endpoint_model(
            endpoint,
            config.default_embedding_endpoint_model(endpoint=endpoint),
        )
        embedding_args.extend(["--embedding-endpoint", endpoint, "--embedding-endpoint-model", model])
    else:
        embedding_args.append("--llama-embed")
    if embed_only:
        embedding_args.append("--embed-only")
    else:
        embedding_args.append("--embed")
        if os.environ["PROJECT_CODE_INTELLIGENCE_PREEMBED"] == "0":
            embedding_args.append("--no-preembed")
    return embedding_args


def index_main(argv: list[str] | None = None) -> int:
    parsed, passthrough = parse_index_args(argv)
    set_index_environment_defaults()
    apply_index_embed_override(parsed)
    forwarded = forwarded_index_args(parsed, passthrough)
    embed = config.env_bool("PROJECT_CODE_INTELLIGENCE_EMBED")
    embed_only = config.env_bool("PROJECT_CODE_INTELLIGENCE_EMBED_ONLY")
    if (embed or embed_only) and not parsed.reset_code_intel:
        forwarded = [*index_embedding_args(embed_only=embed_only), *forwarded]
    if parsed.json:
        os.environ["PROJECT_CODE_INTELLIGENCE_OUTPUT"] = "json"
    _ = progress.set_emitter(progress.detect_progress_mode(requested="json" if parsed.json else None))
    return ingest_code_intel.cli_main(forwarded)


class McpSmokeNamespace(argparse.Namespace):
    json: bool
    repo_paths: list[str]


def mcp_smoke_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call code_intel_status through the stdio MCP server.")
    _ = parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the raw JSON-RPC response instead of the pretty TTY display.",
    )
    _ = parser.add_argument(
        "repo_paths",
        nargs="*",
        help="Repository path(s) to scope the status query to. Use . for the current directory.",
    )
    return parser


def _path_to_repo(path: str) -> str:
    resolved = Path(path).expanduser().resolve(strict=False)
    return resolved.name or "."


def _run_mcp_call(tool_name: str, arguments: dict[str, object], request_id: int = 1) -> tuple[int, object | None, str]:
    params: dict[str, object] = {"name": tool_name, "arguments": arguments}
    request: dict[str, object] = {"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": params}
    proc = process.run(
        [sys.executable, "-m", "project_code_intelligence.server"],
        process.RunOptions(input_text=json.dumps(request) + "\n", capture_output=True, timeout=30, check=False),
    )
    if proc.returncode != 0:
        return proc.returncode, None, proc.stderr or ""
    try:
        response_value = cast("object", json.loads(proc.stdout))
    except json.JSONDecodeError:
        return 1, None, proc.stderr or "MCP response was not valid JSON"
    return 0, response_value, proc.stderr or ""


# Tools the smoke run exercises, in order. Each entry is (tool name, kwargs).
# Read-only by design — we never trigger a write_tool here.
# `related_code_intel` requires a record_id or symbol; "main" is a reasonable
# generic probe — any project either has a `main` symbol or returns 0 edges.
_SMOKE_TOOLS: tuple[tuple[str, dict[str, object]], ...] = (
    ("list_code_intel_files", {"limit": 5}),
    ("list_code_intel_parser_failures", {"limit": 5}),
    ("search_code_intel_text", {"limit": 1}),
    ("search_code_intel_semantic", {"query": "main entry point", "limit": 1}),
    ("related_code_intel", {"symbol": "main", "limit": 1}),
)


def _run_smoke_probes(primary_repo: str) -> tuple[list[dict[str, object]], int]:
    """Run a short read-only sequence against the MCP server. Returns (probes, exit_code)."""
    probes: list[dict[str, object]] = []
    exit_code = 0
    for index, (tool_name, base_args) in enumerate(_SMOKE_TOOLS, start=2):
        arguments: dict[str, object] = {"repo": primary_repo, **base_args}
        return_code, response, stderr_text = _run_mcp_call(tool_name, arguments, request_id=index)
        if stderr_text:
            _ = sys.stderr.write(stderr_text)
        probe: dict[str, object] = {"tool": tool_name, "arguments": arguments}
        if return_code != 0:
            probe["status"] = "fail"
            probe["error"] = f"MCP server exited with code {return_code}"
            probes.append(probe)
            exit_code = max(exit_code, return_code)
            continue
        if isinstance(response, dict) and "error" in cast("dict[object, object]", response):
            probe["status"] = "fail"
            probe["response"] = response
            probes.append(probe)
            exit_code = 1
            continue
        probe["status"] = "ok"
        probe["response"] = response
        probes.append(probe)
    return probes, exit_code


def mcp_smoke_main(argv: list[str] | None = None) -> int:
    parser = mcp_smoke_parser()
    parsed = parser.parse_args(argv, namespace=McpSmokeNamespace())
    if not parsed.repo_paths:
        parser.error("one or more repository paths are required; use . for the current directory")

    repos = [_path_to_repo(path) for path in parsed.repo_paths]
    primary_repo = repos[0]
    arguments: dict[str, object] = {"repo": primary_repo}

    return_code, response, stderr_text = _run_mcp_call("code_intel_status", arguments)
    if stderr_text:
        _ = sys.stderr.write(stderr_text)
    if return_code != 0:
        return return_code

    use_pretty = console_ui.should_emit_pretty(sys.stdout, force=False if parsed.json else None)

    has_error = isinstance(response, dict) and "error" in cast("dict[object, object]", response)
    response_for_render = cast("object", response)
    if has_error:
        if use_pretty:
            mcp_smoke_render.render_error(response_for_render)
        else:
            _ = sys.stdout.write(json.dumps(response) + "\n")
        return 1

    probes, probe_exit = _run_smoke_probes(primary_repo)

    if use_pretty:
        mcp_smoke_render.render_status(response_for_render, repo=primary_repo, probes=probes)
    else:
        payload: dict[str, object] = {"status": response, "probes": probes}
        _ = sys.stdout.write(json.dumps(payload) + "\n")
    return probe_exit


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "mcp-smoke":
        return mcp_smoke_main(args[1:])
    return index_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
