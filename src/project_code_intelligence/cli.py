"""Public command-line entry points."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import cast

from project_code_intelligence import config, ingest_code_intel, process

DEFAULT_EMBED_RECORD_TYPES = (
    "code_chunk,package_definition,config_symbol,patch_hunk,dts_node,"
    "service_entrypoint,security_pattern,static_finding,doc_section"
)


def index_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index code intelligence in pgvector.")
    _ = parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested without writing.")
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
        "ingest_args",
        nargs=argparse.REMAINDER,
        help="Pass remaining arguments to pci-ingest-code. Prefix with -- when needed.",
    )
    return parser


class IndexNamespace(argparse.Namespace):
    dry_run: bool
    embed: bool | None
    ingest_args: list[str]


def normalized_passthrough(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def index_main(argv: list[str] | None = None) -> int:
    parsed = index_parser().parse_args(argv, namespace=IndexNamespace())
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_REPOS", ".")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_COLLECTION", "default")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_PROFILE", "generic")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_MODE", "incremental")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_EMBED", "1")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_EMBED_ONLY", "0")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_PREEMBED", "1")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_EMBEDDING_BATCH_SIZE", "32")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_EMBEDDING_MAX_CHARS", "3000")
    _ = os.environ.setdefault("PROJECT_CODE_INTELLIGENCE_EMBED_RECORD_TYPES", DEFAULT_EMBED_RECORD_TYPES)
    if parsed.embed is not None:
        os.environ["PROJECT_CODE_INTELLIGENCE_EMBED"] = "1" if parsed.embed else "0"
        if not parsed.embed:
            os.environ["PROJECT_CODE_INTELLIGENCE_EMBED_ONLY"] = "0"

    forwarded = normalized_passthrough(parsed.ingest_args)
    if parsed.dry_run:
        forwarded = [*forwarded, "--dry-run"]
    else:
        os.environ["PROJECT_CODE_INTELLIGENCE_ALLOW_WRITES"] = "1"

    embed = config.env_bool("PROJECT_CODE_INTELLIGENCE_EMBED")
    embed_only = config.env_bool("PROJECT_CODE_INTELLIGENCE_EMBED_ONLY")
    if embed or embed_only:
        embedding_args = [
            "--embedding-batch-size",
            os.environ["PROJECT_CODE_INTELLIGENCE_EMBEDDING_BATCH_SIZE"],
            "--embedding-max-chars",
            os.environ["PROJECT_CODE_INTELLIGENCE_EMBEDDING_MAX_CHARS"],
            "--embed-record-types",
            os.environ["PROJECT_CODE_INTELLIGENCE_EMBED_RECORD_TYPES"],
        ]
        endpoint = config.default_embedding_endpoint(local_default=True)
        embedding_args.extend(["--embedding-endpoint", endpoint] if endpoint else ["--llama-embed"])
        if embed_only:
            embedding_args.append("--embed-only")
        else:
            embedding_args.append("--embed")
            if os.environ["PROJECT_CODE_INTELLIGENCE_PREEMBED"] == "0":
                embedding_args.append("--no-preembed")
        forwarded = [*embedding_args, *forwarded]

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
