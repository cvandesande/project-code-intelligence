"""Public command-line entry points."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast

from project_code_intelligence import config, console_ui, ingest_code_intel, mcp_smoke_render, process, progress
from project_code_intelligence.common import database_scope_path_for_root_repos, default_collection, parse_repos
from project_code_intelligence.embeddings import (
    EmbeddingEndpointUnavailableError,
    preflight_embedding_endpoint,
    resolve_embedding_endpoint_model,
)
from project_code_intelligence.exceptions import ConfigError
from project_code_intelligence.hooks import runtime as hook_runtime

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
            "or the current working directory name for multiple paths."
        ),
    )
    _ = parser.add_argument(
        "--reset-code-intel",
        "--reset",
        action="store_true",
        help="Drop the inferred project database for the given repository path(s), then exit.",
    )
    _ = parser.add_argument(
        "--init-db",
        action="store_true",
        help="Create the inferred project database and schema for the given repository path(s), then exit.",
    )
    _ = parser.add_argument(
        "--mcp-config",
        choices=ingest_code_intel.MCP_CONFIG_FORMATS,
        help=(
            "Emit project-scoped read-only pci-mcp configuration and required environment exports "
            "after a successful run. "
            "Use with --init-db to initialize the DB and print config without indexing."
        ),
    )
    _ = parser.add_argument(
        "--mcp-server-name",
        help="Server key/name for generated MCP client config snippets. Defaults to project-code-intelligence.",
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
        "--prune-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After a successful index, delete old snapshots per repo, keeping only the N most recent "
            "(see --prune-keep). Enabled by default; pass --no-prune-snapshots to keep every snapshot."
        ),
    )
    _ = parser.add_argument(
        "--prune-keep",
        type=int,
        default=5,
        metavar="N",
        help="Number of recent snapshots to keep per repo when pruning (default: 5).",
    )
    _ = parser.add_argument(
        "--show-parser-failures",
        action="store_true",
        help=(
            "List failing source paths in the summary panel. Default off shows only the 'Parser fails' "
            "count; the JSON report carries the full path list under 'parser_failure_paths'."
        ),
    )
    _ = parser.add_argument(
        "--worktree",
        metavar="MAIN=WORKTREE",
        help=(
            "Index a linked git worktree's checkout under its MAIN repo's identity/collection "
            "instead of the worktree's own path. Mutually exclusive with repo_paths."
        ),
    )
    _ = parser.add_argument(
        "repo_paths",
        nargs="*",
        help=(
            "Repository path(s) to index. Use . for the current directory. "
            "For multiple paths, run from the workspace directory and pass repo subdirectories."
        ),
    )
    return parser


class IndexNamespace(argparse.Namespace):
    json: bool
    dry_run: bool
    collection: str | None
    reset_code_intel: bool
    init_db: bool
    mcp_config: str | None
    mcp_server_name: str | None
    i_know_this_deletes_code_intel_db: bool
    embed: bool | None
    prune_snapshots: bool
    prune_keep: int
    show_parser_failures: bool
    repo_paths: list[str]
    worktree: str | None


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
        root, workspace_repos = multi_repo_workspace_and_repos(repo_paths)
        repos = ",".join(workspace_repos)
    return ["--root", str(root), "--repos", repos]


def parse_worktree_spec(spec: str) -> tuple[Path, Path]:
    """Split a ``MAIN=WORKTREE`` spec into its (main, worktree) absolute paths."""
    main_str, _, worktree_str = spec.partition("=")
    return (
        Path(main_str).expanduser().resolve(strict=False),
        Path(worktree_str).expanduser().resolve(strict=False),
    )


def worktree_ingest_args(spec: str) -> list[str]:
    """--root/--repos/--repo-scan-root for a worktree ingest: identity/root come from the
    MAIN path (same single-repo naming as repo_paths_to_ingest_args), but the actual scan
    happens against the WORKTREE path via --repo-scan-root."""
    main_path, worktree_path = parse_worktree_spec(spec)
    root = main_path.parent
    repo = main_path.name or "."
    return ["--root", str(root), "--repos", repo, "--repo-scan-root", f"{repo}={worktree_path}"]


def inferred_collection_for_repo_paths(repo_paths: list[str]) -> str:
    absolute_paths = [Path(path).expanduser().resolve(strict=False) for path in repo_paths]
    if len(absolute_paths) == 1:
        return default_collection(absolute_paths[0])
    root, _ = multi_repo_workspace_and_repos(repo_paths)
    return default_collection(root)


def inferred_database_scope_path_for_repo_paths(repo_paths: list[str]) -> Path:
    absolute_paths = [Path(path).expanduser().resolve(strict=False) for path in repo_paths]
    if len(absolute_paths) == 1:
        return absolute_paths[0]
    if absolute_paths:
        root, _ = multi_repo_workspace_and_repos(repo_paths)
        return root
    return Path.cwd().resolve(strict=False)


def multi_repo_workspace_and_repos(repo_paths: list[str]) -> tuple[Path, list[str]]:
    workspace = Path.cwd().resolve(strict=False)
    repos: list[str] = []
    for path in repo_paths:
        resolved = Path(path).expanduser().resolve(strict=False)
        try:
            relative = resolved.relative_to(workspace)
        except ValueError as exc:
            message = (
                "multiple repository paths must be inside the current working directory; "
                "cd to the workspace directory and pass repo subdirectories"
            )
            raise ValueError(message) from exc
        repos.append(relative.as_posix() or ".")
    return workspace, repos


def parse_index_args(argv: list[str] | None = None) -> tuple[IndexNamespace, list[str]]:
    public_argv, passthrough = split_index_argv(argv)
    parser = index_parser()
    parsed = parser.parse_args(public_argv, namespace=IndexNamespace())
    if parsed.worktree and parsed.repo_paths:
        parser.error("--worktree cannot be combined with repo_paths")
    if not parsed.worktree and not parsed.repo_paths:
        parser.error("one or more repository paths are required; use . for the current directory")
    if len(parsed.repo_paths) > 1:
        try:
            _, _ = multi_repo_workspace_and_repos(parsed.repo_paths)
        except ValueError as exc:
            parser.error(str(exc))
    if parsed.reset_code_intel and parsed.init_db:
        parser.error("--init-db cannot be combined with --reset")
    if parsed.reset_code_intel and parsed.mcp_config:
        parser.error("--mcp-config cannot be combined with --reset")
    if parsed.json and parsed.mcp_config:
        parser.error("--mcp-config cannot be combined with --json")
    if parsed.dry_run and parsed.mcp_config:
        parser.error("--mcp-config cannot be combined with --dry-run")
    return parsed, normalized_passthrough(passthrough)


def set_index_environment_defaults() -> None:
    _ = os.environ.setdefault("PCI_PROFILE", "generic")
    _ = os.environ.setdefault("PCI_MODE", "incremental")
    _ = os.environ.setdefault("PCI_EMBED", "1")
    _ = os.environ.setdefault("PCI_EMBED_ONLY", "0")
    _ = os.environ.setdefault("PCI_PREEMBED", "1")
    _ = os.environ.setdefault("PCI_EMBEDDING_BATCH_SIZE", "32")
    _ = os.environ.setdefault("PCI_EMBEDDING_MAX_CHARS", "3000")
    _ = os.environ.setdefault("PCI_EMBED_RECORD_TYPES", DEFAULT_EMBED_RECORD_TYPES)


def set_index_database_scope_default(parsed: IndexNamespace) -> None:
    if parsed.worktree:
        main_path, _ = parse_worktree_spec(parsed.worktree)
        scope_path = main_path
    elif parsed.repo_paths:
        scope_path = inferred_database_scope_path_for_repo_paths(parsed.repo_paths)
    else:
        scope_path = database_scope_path_for_root_repos(Path.cwd(), parse_repos("."))
    os.environ[config.DATABASE_SCOPE_PATH_ENV] = str(scope_path)


def apply_index_embed_override(parsed: IndexNamespace) -> None:
    if parsed.embed is not None:
        os.environ["PCI_EMBED"] = "1" if parsed.embed else "0"
        if not parsed.embed:
            os.environ["PCI_EMBED_ONLY"] = "0"


def forwarded_mcp_config_args(parsed: IndexNamespace) -> list[str]:
    forwarded: list[str] = []
    if parsed.mcp_config:
        forwarded.extend(["--mcp-config", parsed.mcp_config])
    if parsed.mcp_server_name:
        forwarded.extend(["--mcp-server-name", parsed.mcp_server_name])
    return forwarded


def _identity_forwarded_args(parsed: IndexNamespace) -> list[str]:
    """--root/--repos(/--repo-scan-root) and, unless overridden, --collection for
    whichever of --worktree or repo_paths was given (parse_index_args already
    enforces they are mutually exclusive)."""
    if parsed.worktree:
        identity_args = worktree_ingest_args(parsed.worktree)
        main_path, _ = parse_worktree_spec(parsed.worktree)
        inferred_collection = default_collection(main_path)
    elif parsed.repo_paths:
        identity_args = repo_paths_to_ingest_args(parsed.repo_paths)
        inferred_collection = inferred_collection_for_repo_paths(parsed.repo_paths)
    else:
        return []
    if parsed.collection:
        return ["--collection", parsed.collection, *identity_args]
    if not config.collection_override_allowed():
        return ["--collection", inferred_collection, *identity_args]
    return identity_args


def forwarded_index_args(parsed: IndexNamespace, passthrough: list[str]) -> list[str]:
    forwarded = [*_identity_forwarded_args(parsed), *passthrough]
    if parsed.reset_code_intel:
        forwarded = ["--reset-code-intel", "--reset-only", *forwarded]
    if parsed.init_db:
        forwarded = ["--init-db-only", *forwarded]
    forwarded = [*forwarded_mcp_config_args(parsed), *forwarded]
    if parsed.i_know_this_deletes_code_intel_db:
        forwarded = ["--i-know-this-deletes-code-intel-db", *forwarded]
    if parsed.dry_run:
        forwarded = [*forwarded, "--dry-run"]
    else:
        os.environ["PCI_ALLOW_WRITES"] = "1"
    if parsed.prune_snapshots:
        forwarded = [*forwarded, "--prune-snapshots", "--prune-keep", str(parsed.prune_keep)]
    else:
        forwarded = [*forwarded, "--no-prune-snapshots"]
    if parsed.show_parser_failures:
        forwarded = [*forwarded, "--show-parser-failures"]
    return forwarded


def _resolve_index_embedding() -> tuple[str | None, str | None]:
    """Return (endpoint, model) for the current embedding config, or (None, None)."""
    endpoint = config.default_embedding_endpoint(local_default=True)
    if not endpoint:
        return None, None
    model = resolve_embedding_endpoint_model(
        endpoint,
        config.default_embedding_endpoint_model(endpoint=endpoint),
    )
    return endpoint, model


def index_embedding_args(*, embed_only: bool, endpoint: str | None, model: str | None) -> list[str]:
    embedding_args = [
        "--embedding-batch-size",
        os.environ["PCI_EMBEDDING_BATCH_SIZE"],
        "--embedding-max-chars",
        os.environ["PCI_EMBEDDING_MAX_CHARS"],
        "--embed-record-types",
        os.environ["PCI_EMBED_RECORD_TYPES"],
    ]
    if endpoint:
        embedding_args.extend(["--embedding-endpoint", endpoint, "--embedding-endpoint-model", model or ""])
    else:
        embedding_args.append("--llama-embed")
    if embed_only:
        embedding_args.append("--embed-only")
    else:
        embedding_args.append("--embed")
        if os.environ["PCI_PREEMBED"] == "0":
            embedding_args.append("--no-preembed")
    return embedding_args


def print_index_startup(
    parsed: IndexNamespace,
    *,
    embed: bool,
    endpoint: str | None,
    model: str | None,
) -> bool:
    """Run an embedding preflight check before indexing begins.

    Produces no output on success — embedding details appear in the final result
    panel. On failure, prints the error to stderr and returns False so the caller
    can exit before any work starts.
    """
    if not embed or not endpoint or parsed.dry_run:
        return True
    try:
        preflight_embedding_endpoint(endpoint, model or "")
    except EmbeddingEndpointUnavailableError as exc:
        use_color = console_ui.should_emit_pretty(sys.stderr)
        console_ui.build_console(file=sys.stderr, color=use_color).print(str(exc))
        return False
    return True


def load_index_user_config(parsed: IndexNamespace) -> bool:
    try:
        user_config = config.load_pci_index_user_config()
    except ConfigError as exc:
        _ = sys.stderr.write(f"pci-index: {exc}\n")
        return False
    if user_config is not None and user_config.loaded and not parsed.json and not parsed.mcp_config:
        _ = sys.stderr.write(f"pci-index: loaded config from {user_config.path}\n")
    return True


def index_main(argv: list[str] | None = None) -> int:
    parsed, passthrough = parse_index_args(argv)
    if not load_index_user_config(parsed):
        return 1
    set_index_environment_defaults()
    set_index_database_scope_default(parsed)
    apply_index_embed_override(parsed)
    forwarded = forwarded_index_args(parsed, passthrough)
    embed = config.env_bool("PCI_EMBED")
    embed_only = config.env_bool("PCI_EMBED_ONLY")

    embedding_endpoint: str | None = None
    embedding_model: str | None = None
    if (embed or embed_only) and not parsed.reset_code_intel and not parsed.init_db:
        embedding_endpoint, embedding_model = _resolve_index_embedding()
        forwarded = [
            *index_embedding_args(embed_only=embed_only, endpoint=embedding_endpoint, model=embedding_model),
            *forwarded,
        ]

    is_reset = parsed.reset_code_intel
    is_init_db = parsed.init_db
    if (
        not parsed.json
        and not is_reset
        and not is_init_db
        and not print_index_startup(
            parsed, embed=embed or embed_only, endpoint=embedding_endpoint, model=embedding_model
        )
    ):
        return 1

    if parsed.json:
        os.environ["PCI_OUTPUT"] = "json"
    _ = progress.set_emitter(progress.detect_progress_mode(requested="json" if parsed.json else None))
    rc = ingest_code_intel.cli_main(forwarded)
    if rc == 0 and parsed.repo_paths and not (parsed.dry_run or is_reset or is_init_db):
        # Pin the collection actually used (explicit flag or inferred), so a hook
        # replay cannot drift if its environment infers differently.
        collection = forwarded[forwarded.index("--collection") + 1] if "--collection" in forwarded else None
        hook_runtime.write_reindex_markers(
            [Path(path).expanduser().resolve(strict=False) for path in parsed.repo_paths],
            collection,
        )
    return rc


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


def _path_is_current_directory(path: str) -> bool:
    return Path(path).expanduser().resolve(strict=False) == Path.cwd().resolve()


def _mcp_response_payload(response: object) -> dict[str, object] | None:
    outer = console_ui.as_object(response)
    result = console_ui.as_object(outer.get("result"))
    content = console_ui.as_list(result.get("content")) or []
    first = console_ui.as_object(content[0] if content else None)
    text = first.get("text")
    payload: object | None = None
    if isinstance(text, str):
        try:
            payload = cast("object", json.loads(text))
        except json.JSONDecodeError:
            return None
    return console_ui.as_dict(payload)


def _mcp_response_has_error(response: object) -> bool:
    value = console_ui.as_dict(response)
    if value is not None and "error" in value:
        return True
    return _mcp_payload_error(response) is not None


def _mcp_payload_error(response: object) -> str | None:
    payload = _mcp_response_payload(response)
    if payload is None:
        return None
    error = payload.get("error")
    if isinstance(error, str):
        return error
    if error:
        return json.dumps(error, sort_keys=True, separators=(",", ":"))
    return None


def _status_repos(response: object) -> list[str]:
    payload = _mcp_response_payload(response)
    if payload is None:
        return []
    snapshots = console_ui.as_list(payload.get("snapshots")) or []
    repos: list[str] = []
    for snapshot in snapshots:
        item = console_ui.as_dict(snapshot)
        repo = item.get("repo") if item is not None else None
        if isinstance(repo, str) and repo not in repos:
            repos.append(repo)
    return repos


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


def _resolve_smoke_target_repos(repo_paths: list[str], status_response: object) -> list[str]:
    requested_repos = [_path_to_repo(path) for path in repo_paths]
    available_repos = _status_repos(status_response)
    if (
        len(repo_paths) == 1
        and _path_is_current_directory(repo_paths[0])
        and requested_repos[0] not in available_repos
        and available_repos
    ):
        return available_repos
    return requested_repos


# Tools the smoke run exercises, in order. Each entry is (tool name, kwargs).
# Read-only by design — we never trigger a write_tool here.
# `related_code_intel` requires a record_id or symbol; "main" is a reasonable
# generic probe — any project either has a `main` symbol or returns 0 edges.
_SMOKE_TOOLS: tuple[tuple[str, dict[str, object]], ...] = (
    ("list_code_intel_files", {"limit": 5}),
    ("search_code_intel_text", {"limit": 1}),
    ("search_code_intel_semantic", {"query": "main entry point", "limit": 1}),
    ("related_code_intel", {"symbol": "main", "limit": 1}),
)


def _run_smoke_probes(repos: list[str]) -> tuple[list[dict[str, object]], int]:
    """Run a short read-only sequence against the MCP server. Returns (probes, exit_code)."""
    probes: list[dict[str, object]] = []
    exit_code = 0
    request_id = 2
    for repo in repos:
        for tool_name, base_args in _SMOKE_TOOLS:
            arguments: dict[str, object] = {"repo": repo, **base_args}
            return_code, response, stderr_text = _run_mcp_call(tool_name, arguments, request_id=request_id)
            request_id += 1
            if stderr_text:
                _ = sys.stderr.write(stderr_text)
            probe: dict[str, object] = {"repo": repo, "tool": tool_name, "arguments": arguments}
            if return_code != 0:
                probe["status"] = "fail"
                probe["error"] = f"MCP server exited with code {return_code}"
                probes.append(probe)
                exit_code = max(exit_code, return_code)
                continue
            response_obj = cast("object", response)
            if isinstance(response_obj, dict) and "error" in cast("dict[object, object]", response_obj):
                probe["status"] = "fail"
                probe["response"] = response_obj
                probes.append(probe)
                exit_code = 1
                continue
            payload_error = _mcp_payload_error(cast("object", response_obj))
            if payload_error is not None:
                probe["status"] = "fail"
                probe["error"] = payload_error
                probe["response"] = response_obj
                probes.append(probe)
                exit_code = 1
                continue
            probe["status"] = "ok"
            probe["response"] = response_obj
            probes.append(probe)
    return probes, exit_code


def mcp_smoke_main(argv: list[str] | None = None) -> int:
    parser = mcp_smoke_parser()
    parsed = parser.parse_args(argv, namespace=McpSmokeNamespace())
    if not parsed.repo_paths:
        parser.error("one or more repository paths are required; use . for the current directory")

    return_code, response, stderr_text = _run_mcp_call("code_intel_status", {})
    status_response: object = response
    if stderr_text:
        _ = sys.stderr.write(stderr_text)
    if return_code != 0:
        return return_code

    use_pretty = console_ui.should_emit_pretty(sys.stdout, force=False if parsed.json else None)

    if _mcp_response_has_error(status_response):
        if use_pretty:
            mcp_smoke_render.render_error(status_response)
        else:
            _ = sys.stdout.write(json.dumps(status_response) + "\n")
        return 1

    target_repos = _resolve_smoke_target_repos(parsed.repo_paths, status_response)
    primary_repo = target_repos[0]

    probes, probe_exit = _run_smoke_probes(target_repos)

    if use_pretty:
        mcp_smoke_render.render_status(status_response, repo=primary_repo, probes=probes)
    else:
        payload: dict[str, object] = {"repos": target_repos, "status": status_response, "probes": probes}
        _ = sys.stdout.write(json.dumps(payload) + "\n")
    return probe_exit


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "mcp-smoke":
        return mcp_smoke_main(args[1:])
    return index_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
