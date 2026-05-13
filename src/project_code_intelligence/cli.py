"""Public command-line entry points."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast

from project_code_intelligence import config, ingest_code_intel, process
from project_code_intelligence.embeddings import resolve_embedding_endpoint_model

DEFAULT_EMBED_RECORD_TYPES = (
    "code_chunk,package_definition,config_symbol,patch_hunk,dts_node,"
    "service_entrypoint,security_pattern,static_finding,doc_section"
)


def index_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index code intelligence in pgvector.")
    _ = parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested without writing.")
    _ = parser.add_argument(
        "--reset-code-intel",
        "--reset",
        action="store_true",
        help="Drop and recreate code-intelligence tables, then exit. Prompts unless confirmation flag is set.",
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
    dry_run: bool
    reset_code_intel: bool
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


def parse_index_args(argv: list[str] | None = None) -> tuple[IndexNamespace, list[str]]:
    public_argv, passthrough = split_index_argv(argv)
    parser = index_parser()
    parsed = parser.parse_args(public_argv, namespace=IndexNamespace())
    if parsed.reset_code_intel and parsed.repo_paths:
        parser.error("--reset-code-intel does not accept repository paths")
    if not parsed.reset_code_intel and not parsed.repo_paths:
        parser.error("one or more repository paths are required; use . for the current directory")
    return parsed, normalized_passthrough(passthrough)


def set_index_environment_defaults() -> None:
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_COLLECTION", "default")
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
    if parsed.reset_code_intel:
        forwarded = ["--reset-code-intel", "--reset-only", *forwarded]
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
    return ingest_code_intel.cli_main(forwarded)


def mcp_smoke_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Call code_intel_status through the stdio MCP server.")
    _ = parser.parse_args(argv)
    arguments: dict[str, object] = {}
    params: dict[str, object] = {"name": "code_intel_status", "arguments": arguments}
    request: dict[str, object] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": params,
    }
    proc = process.run(
        [sys.executable, "-m", "project_code_intelligence.server"],
        process.RunOptions(
            input_text=json.dumps(request) + "\n",
            capture_output=True,
            timeout=30,
            check=False,
        ),
    )
    if proc.stderr:
        _ = sys.stderr.write(proc.stderr)
    if proc.stdout:
        _ = sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        return proc.returncode
    try:
        response_value = cast("object", json.loads(proc.stdout))
    except json.JSONDecodeError:
        return 1
    return 1 if isinstance(response_value, dict) and "error" in response_value else 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "mcp-smoke":
        return mcp_smoke_main(args[1:])
    return index_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
