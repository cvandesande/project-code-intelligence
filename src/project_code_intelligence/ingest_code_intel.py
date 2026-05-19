"""Orchestrate repository code-intelligence ingestion into Postgres."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from psycopg.errors import InsufficientPrivilege

from project_code_intelligence import config, db, profile_context, progress
from project_code_intelligence import runtime as runtime_state
from project_code_intelligence.code_profiles import load_profile
from project_code_intelligence.common import (
    database_scope_path_for_root_repos,
    default_collection,
    parse_repos,
    repo_for_source_path,
)
from project_code_intelligence.doctor.embeddings import check_embedding_options
from project_code_intelligence.doctor.hardware import check_npu_support, discover_gpus
from project_code_intelligence.embedding.framework import active_embedding_profile
from project_code_intelligence.embeddings import (
    EmbeddingBackend,
    EmbeddingContractMismatchError,
    EmbeddingEndpointUnavailableError,
    EmbeddingRunConfig,
    abandon_preembedding,
    code_preembedding_enabled,
    embed_db_records,
    insert_records_with_preembedding,
    preflight_embedding_endpoint,
    resolve_embedding_endpoint_framework,
    resolve_embedding_endpoint_model,
    start_record_preembedding,
)
from project_code_intelligence.git_utils import workspace_root
from project_code_intelligence.inventory import DiscoveryReuse, discover_files, make_snapshot
from project_code_intelligence.models import (
    DEFAULT_EMBED_RECORD_TYPES,
    IntelEdge,
    IntelFile,
    IntelRecord,
    JsonObject,
    PreviousFileState,
    RepoIngest,
    SarifIngest,
    Snapshot,
    StaticRun,
)
from project_code_intelligence.parsers import parse_file
from project_code_intelligence.profile_context import set_active_profile
from project_code_intelligence.progress import progress_event, runtime_heartbeat
from project_code_intelligence.reporting import report_ingests
from project_code_intelligence.runtime import (
    format_duration,
    runtime_heartbeat_seconds,
)
from project_code_intelligence.sarif import (
    SarifIngestContext,
    discover_sarif_files,
    explicit_sarif_patterns,
    ingest_sarif,
    relative_to_or_none,
    repo_for_sarif_file,
)
from project_code_intelligence.storage import (
    RecordInsertContext,
    copy_unchanged_parser_failures,
    copy_unchanged_records_and_edges,
    count_unresolved_edge_targets,
    ensure_schema,
    file_signature,
    insert_edges,
    insert_files,
    insert_parser_failures,
    insert_records,
    insert_snapshot,
    insert_static_runs,
    latest_snapshot_info,
    pre_resolvable_edge_count,
    pre_resolve_edge_targets,
    previous_file_state_signature,
    previous_file_states,
    prune_old_snapshots,
    replace_repos,
    resolve_edge_targets,
    snapshot_versions_compatible,
    stamp_embed_types,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from project_code_intelligence.code_profiles.base import CodeIntelProfile

_T = TypeVar("_T")

MIN_CHUNK_CHARS = 100
AUTO_SCAN_WORKERS = 0
MAX_AUTO_SCAN_WORKERS = 8
MIN_PARALLEL_PARSE_FILES = 64
PARSE_CHUNKS_PER_WORKER = 8
_DB_WRITE_BATCH_SIZE = 500
MCP_CONFIG_FORMATS = ("env", "codex", "claude", "opencode", "vscode", "copilot", "cline", "zed")
MCP_PROJECT_ENV_NAMES = (
    "PCI_MCP_DATABASE_URL",
    "PCI_MCP_DATABASE_USER",
    "PCI_MCP_DATABASE_PASSWORD",
)
MCP_STANDALONE_ENV_NAMES = (
    "PCI_MCP_DATABASE_URL",
    "PCI_MCP_DATABASE_USER",
    "PCI_MCP_DATABASE_PASSWORD",
    "PCI_COLLECTION",
    "PCI_DATABASE_SCOPE_PATH",
)


def _chunks(items: list[_T], size: int) -> list[list[_T]]:
    return [items[i : i + size] for i in range(0, max(len(items), 1), size)]


def write_stdout(message: str) -> None:
    _ = sys.stdout.write(message + "\n")


def write_stderr(message: str) -> None:
    _ = sys.stderr.write(message + "\n")


def _shell_export(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}"


@dataclass(frozen=True)
class CliArgs:
    root: Path
    collection: str | None
    profile: str
    repos: str | None
    max_file_bytes: int
    scan_workers: int
    chunk_chars: int
    overlap_lines: int
    limit_files: int | None
    progress_every: int
    dry_run: bool
    reset_code_intel: bool
    i_know_this_deletes_code_intel_db: bool
    reset_only: bool
    init_db_only: bool
    sarif: list[str]
    no_profile_sarif: bool
    sarif_max_bytes: int
    embed_only: bool
    mode: str
    full: bool
    no_replace: bool
    embed: bool
    embed_record_types: str
    embedding_batch_size: int
    embedding_max_chars: int
    embedding_endpoint: str | None
    embedding_endpoint_model: str
    llama_embed: bool
    no_preembed: bool
    prune_snapshots: bool
    prune_keep: int
    mcp_config: str | None
    mcp_server_name: str | None
    show_parser_failures: bool


@dataclass(frozen=True)
class IngestPlan:
    args: CliArgs
    profile: CodeIntelProfile
    root: Path
    collection: str
    repos: list[str]
    embed_types: set[str]
    sarif_files: list[Path]
    embedding_requested: bool
    preembedding_requested: bool
    mode: str


@dataclass(frozen=True)
class McpConfigContext:
    server_name: str
    command: str
    cwd: str
    database_url: str
    database_user: str
    database_password: str
    collection: str
    database_scope_path: str


@dataclass(frozen=True)
class RepoIngestConfig:
    root: Path
    repo: str
    collection: str
    profile_name: str
    max_file_bytes: int
    scan_workers: int
    max_chars: int
    overlap_lines: int
    limit_files: int | None
    progress_every: int
    previous_snapshot_id: int | None = None
    previous_files: dict[str, PreviousFileState] | None = None
    mode: str = "full"


@dataclass
class DbUploadSummary:
    snapshot_ids: list[int] = field(default_factory=list)
    inserted_files: int = 0
    inserted_records: int = 0
    inserted_edges: int = 0
    inserted_parser_failures: int = 0
    copied_records: int = 0
    copied_edges: int = 0
    copied_parser_failures: int = 0
    preembedded_records: int = 0
    preembedding_skipped: int = 0
    bootstrap: db.DatabaseBootstrapResult | None = None
    static_counts: dict[str, int] = field(
        default_factory=lambda: {
            "static_runs": 0,
            "static_rules": 0,
            "static_findings": 0,
            "static_locations": 0,
            "static_code_flow_steps": 0,
        }
    )
    # Paths of parser failures (newly inserted + copied), formatted as
    # "{repo}/{source_path}" so multi-repo workspaces stay unambiguous. Always
    # populated; the pci-index `--show-parser-failures` flag controls whether
    # they appear in the final summary report.
    parser_failure_paths: list[str] = field(default_factory=list)


@dataclass
class StaticSnapshotIndexes:
    snapshot_ids_by_repo: dict[str, int] = field(default_factory=dict)
    snapshot_by_repo: dict[str, Snapshot] = field(default_factory=dict)


@dataclass(frozen=True)
class RepoChangeState:
    previous_snapshot_id: int | None
    changed_paths: set[str]
    unchanged_paths: set[str]
    deleted_paths: set[str]


class CliNamespace(argparse.Namespace):
    root: Path
    collection: str | None
    profile: str
    repos: str | None
    max_file_bytes: int
    scan_workers: int
    chunk_chars: int
    overlap_lines: int
    limit_files: int | None
    progress_every: int
    dry_run: bool
    reset_code_intel: bool
    i_know_this_deletes_code_intel_db: bool
    reset_only: bool
    init_db_only: bool
    sarif: list[str]
    no_profile_sarif: bool
    sarif_max_bytes: int
    embed_only: bool
    mode: str
    full: bool
    no_replace: bool
    embed: bool
    embed_record_types: str
    embedding_batch_size: int
    embedding_max_chars: int
    embedding_endpoint: str | None
    embedding_endpoint_model: str
    llama_embed: bool
    no_preembed: bool
    prune_snapshots: bool
    prune_keep: int
    mcp_config: str | None
    mcp_server_name: str | None
    show_parser_failures: bool


def json_int(obj: JsonObject, key: str) -> int:
    value = obj.get(key)
    if isinstance(value, bool):
        raise TypeError(f"{key} is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise TypeError(f"{key} is not an integer")


@dataclass(frozen=True)
class ParseWorkerTask:
    intel_file: IntelFile
    max_chars: int
    overlap_lines: int


@dataclass(frozen=True)
class ParseWorkerResult:
    source_path: str
    records: list[IntelRecord]
    edges: list[IntelEdge]
    failures: list[JsonObject]


def initialize_parse_worker(profile_name: str) -> None:
    set_active_profile(load_profile(profile_name))


def parse_file_worker(task: ParseWorkerTask) -> ParseWorkerResult:
    records, edges, failures = parse_file(task.intel_file, task.max_chars, task.overlap_lines)
    return ParseWorkerResult(
        source_path=task.intel_file.source_path,
        records=records,
        edges=edges,
        failures=failures,
    )


def resolve_scan_workers(requested_workers: int, changed_files: int) -> int:
    if requested_workers > 0:
        return min(requested_workers, max(1, changed_files))
    if changed_files < MIN_PARALLEL_PARSE_FILES:
        return 1
    cpu_count = os.cpu_count() or 1
    return min(MAX_AUTO_SCAN_WORKERS, cpu_count, changed_files)


def parse_task_chunksize(task_count: int, workers: int) -> int:
    if task_count <= 0 or workers <= 1:
        return 1
    return max(1, task_count // (workers * PARSE_CHUNKS_PER_WORKER))


def serial_parse_results(tasks: list[ParseWorkerTask]) -> Iterator[ParseWorkerResult]:
    for task in tasks:
        yield parse_file_worker(task)


def parallel_parse_results(
    tasks: list[ParseWorkerTask],
    *,
    workers: int,
    profile_name: str,
) -> Iterator[ParseWorkerResult]:
    chunksize = parse_task_chunksize(len(tasks), workers)
    progress_event(
        "code_intel_scan_workers_started",
        workers=workers,
        files=len(tasks),
        chunksize=chunksize,
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_parse_worker,
        initargs=(profile_name,),
    ) as executor:
        yield from executor.map(parse_file_worker, tasks, chunksize=chunksize)


def collect_parse_results(
    files: list[IntelFile],
    changed_paths: set[str],
    results: Iterator[ParseWorkerResult],
    *,
    repo: str,
    progress_every: int,
) -> tuple[list[IntelRecord], list[IntelEdge], list[JsonObject]]:
    records: list[IntelRecord] = []
    edges: list[IntelEdge] = []
    failures: list[JsonObject] = []
    for idx, intel_file in enumerate(files, 1):
        runtime_state.active_metrics.add_phase_done(1)
        if intel_file.source_path in changed_paths:
            result = next(results)
            if result.source_path != intel_file.source_path:
                raise RuntimeError("parallel parser returned results out of order")
            records.extend(result.records)
            edges.extend(result.edges)
            failures.extend(result.failures)
        if progress_every and (idx % progress_every == 0 or idx == len(files)):
            progress_event(
                "code_intel_parsed",
                repo=repo,
                files=idx,
                total_files=len(files),
                changed_files=len(changed_paths),
                unchanged_files=len(files) - len(changed_paths),
                records=len(records),
                edges=len(edges),
                parser_failures=len(failures),
            )
    return records, edges, failures


def parse_changed_files(
    files: list[IntelFile],
    changed_paths: set[str],
    *,
    config: RepoIngestConfig,
) -> tuple[list[IntelRecord], list[IntelEdge], list[JsonObject]]:
    tasks = [
        ParseWorkerTask(
            intel_file=intel_file,
            max_chars=config.max_chars,
            overlap_lines=config.overlap_lines,
        )
        for intel_file in files
        if intel_file.source_path in changed_paths
    ]
    workers = resolve_scan_workers(config.scan_workers, len(tasks))
    runtime_state.active_metrics.set_scan_workers_max(workers)
    progress_event(
        "code_intel_parse_started",
        repo=config.repo,
        files=len(files),
        changed_files=len(changed_paths),
        unchanged_files=len(files) - len(changed_paths),
        workers=workers,
    )
    if workers > 1 and len(tasks) > 1:
        results = parallel_parse_results(tasks, workers=workers, profile_name=config.profile_name)
    else:
        results = serial_parse_results(tasks)
    return collect_parse_results(
        files,
        changed_paths,
        results,
        repo=config.repo,
        progress_every=config.progress_every,
    )


def previous_signatures_from_states(previous_files: dict[str, PreviousFileState]) -> dict[str, str]:
    return {source_path: previous_file_state_signature(state) for source_path, state in previous_files.items()}


def snapshot_dirty_paths(snapshot: Snapshot) -> set[str]:
    value = snapshot.metadata.get("dirty_paths")
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def repo_change_state(
    files: list[IntelFile],
    previous_signatures: dict[str, str],
    *,
    mode: str,
    previous_snapshot_id: int | None,
) -> RepoChangeState:
    current_signatures = {item.source_path: file_signature(item) for item in files}
    unchanged_paths: set[str] = {
        path for path, signature in current_signatures.items() if previous_signatures.get(path) == signature
    }
    deleted_paths = set(previous_signatures) - set(current_signatures)
    if mode != "incremental":
        return RepoChangeState(
            previous_snapshot_id=None,
            changed_paths=set(current_signatures),
            unchanged_paths=set(),
            deleted_paths=set(),
        )
    return RepoChangeState(
        previous_snapshot_id=previous_snapshot_id,
        changed_paths=set(current_signatures) - unchanged_paths,
        unchanged_paths=unchanged_paths,
        deleted_paths=deleted_paths,
    )


def count_reused_unchanged_files(files: list[IntelFile], previous_files: dict[str, PreviousFileState]) -> int:
    return sum(
        1
        for item in files
        if (previous := previous_files.get(item.source_path)) is not None
        and previous.git_blob_sha == item.git_blob_sha
        and previous_file_state_signature(previous) == file_signature(item)
    )


def ingest_repo(config: RepoIngestConfig) -> RepoIngest:
    progress_event("code_intel_repo_scan_started", repo=config.repo, mode=config.mode)
    started = time.monotonic()
    snapshot = make_snapshot(config.root, config.repo, config.collection)
    runtime_state.active_metrics.add("scan_git_seconds", time.monotonic() - started)
    previous_files = config.previous_files or {}
    previous_signatures = previous_signatures_from_states(previous_files)
    progress_event(
        "code_intel_repo_discovery_started",
        repo=config.repo,
        mode=config.mode,
        dirty=snapshot.dirty,
    )
    started = time.monotonic()
    files = discover_files(
        config.root,
        snapshot,
        config.max_file_bytes,
        reuse=DiscoveryReuse(
            previous_files=previous_files,
            reuse_unchanged_blobs=config.mode == "incremental" and bool(previous_files),
            dirty_paths=frozenset(snapshot_dirty_paths(snapshot)),
        ),
    )
    runtime_state.active_metrics.add("scan_discovery_seconds", time.monotonic() - started)
    if config.limit_files is not None:
        files = files[: config.limit_files]
    runtime_state.active_metrics.add_phase_total(len(files))
    changes = repo_change_state(
        files,
        previous_signatures,
        mode=config.mode,
        previous_snapshot_id=config.previous_snapshot_id,
    )
    reused_unchanged_files = count_reused_unchanged_files(files, previous_files)
    runtime_state.active_metrics.add("reused_unchanged_files", reused_unchanged_files)
    progress_event(
        "code_intel_discovered",
        repo=config.repo,
        files=len(files),
        changed_files=len(changes.changed_paths),
        unchanged_files=len(changes.unchanged_paths),
        deleted_files=len(changes.deleted_paths),
        mode=config.mode,
        commit_sha=snapshot.commit_sha,
        tree_sha=snapshot.tree_sha,
        reused_unchanged_files=reused_unchanged_files,
    )
    started = time.monotonic()
    records, edges, failures = parse_changed_files(files, changes.changed_paths, config=config)
    runtime_state.active_metrics.add("scan_parse_seconds", time.monotonic() - started)
    return RepoIngest(
        snapshot=snapshot,
        files=files,
        records=records,
        edges=edges,
        parser_failures=failures,
        mode=config.mode,
        previous_snapshot_id=changes.previous_snapshot_id,
        changed_paths=changes.changed_paths,
        unchanged_paths=changes.unchanged_paths,
        deleted_paths=changes.deleted_paths,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--root", type=Path, default=workspace_root())
    _ = parser.add_argument("--collection", default=config.env_text("PCI_COLLECTION"))
    _ = parser.add_argument("--profile", default=config.env_text("PCI_PROFILE", "generic") or "generic")
    _ = parser.add_argument("--repos", default=config.env_text("PCI_REPOS"))
    _ = parser.add_argument("--max-file-bytes", type=int, default=512 * 1024)
    _ = parser.add_argument(
        "--scan-workers",
        type=int,
        default=config.env_int("PCI_SCAN_WORKERS", AUTO_SCAN_WORKERS, minimum=0),
        help="Parser worker processes. 0 chooses a conservative auto value; 1 disables process-pool parsing.",
    )
    _ = parser.add_argument("--chunk-chars", type=int, default=2400)
    _ = parser.add_argument("--overlap-lines", type=int, default=6)
    _ = parser.add_argument("--limit-files", type=int)
    _ = parser.add_argument("--progress-every", type=int, default=250)
    _ = parser.add_argument("--dry-run", action="store_true")
    _ = parser.add_argument(
        "--reset-code-intel",
        action="store_true",
        help="Drop the inferred project database for selected repos. Prompts unless confirmation flag is set.",
    )
    _ = parser.add_argument(
        "--i-know-this-deletes-code-intel-db",
        action="store_true",
        help="Skip interactive confirmation for --reset-code-intel.",
    )
    _ = parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Reset code-intelligence tables and exit without scanning or indexing.",
    )
    _ = parser.add_argument(
        "--init-db-only",
        action="store_true",
        help="Create the inferred project database and schema, then exit without scanning or indexing.",
    )
    _ = parser.add_argument(
        "--sarif",
        action="append",
        default=[],
        help="SARIF file or glob to ingest. Can be repeated or comma-separated.",
    )
    _ = parser.add_argument(
        "--no-profile-sarif",
        action="store_true",
        help="Disable SARIF discovery globs supplied by the selected profile.",
    )
    _ = parser.add_argument(
        "--sarif-max-bytes",
        type=int,
        default=config.env_int("PCI_SARIF_MAX_BYTES", 50 * 1024 * 1024, minimum=0),
        help="Maximum SARIF file size to parse.",
    )
    _ = parser.add_argument(
        "--embed-only",
        action="store_true",
        help="Resume embeddings for the latest matching snapshots without reparsing or rewriting records.",
    )
    _ = parser.add_argument(
        "--mode",
        choices=("incremental", "full"),
        default=config.env_text("PCI_MODE", "incremental") or "incremental",
        help="incremental reuses unchanged records from the previous snapshot; full reparses all files.",
    )
    _ = parser.add_argument("--full", action="store_true", help="Alias for --mode full.")
    _ = parser.add_argument("--no-replace", action="store_true", help="Deprecated compatibility flag for full mode.")
    _ = parser.add_argument("--embed", action="store_true")
    _ = parser.add_argument("--embed-record-types", default=",".join(sorted(DEFAULT_EMBED_RECORD_TYPES)))
    _ = parser.add_argument("--embedding-batch-size", type=int, default=16)
    _ = parser.add_argument(
        "--embedding-max-chars",
        type=int,
        default=config.env_int("PCI_EMBEDDING_MAX_CHARS", 3000, minimum=1),
        help="Maximum characters sent to the embedding model per record; stored embedding_text is unchanged.",
    )
    embedding_endpoint_default = config.default_embedding_endpoint()
    _ = parser.add_argument("--embedding-endpoint", default=embedding_endpoint_default)
    _ = parser.add_argument(
        "--embedding-endpoint-model",
        default=config.default_embedding_endpoint_model(endpoint=embedding_endpoint_default),
    )
    _ = parser.add_argument(
        "--llama-embed", action="store_true", help="Use the slower llama-embedding CLI instead of an HTTP endpoint."
    )
    _ = parser.add_argument(
        "--no-preembed",
        action="store_true",
        help="Disable background pre-embedding while records are being inserted into Postgres.",
    )
    _ = parser.add_argument(
        "--prune-snapshots",
        action="store_true",
        help="Delete old snapshots, keeping only the N most recent per repo (see --prune-keep).",
    )
    _ = parser.add_argument(
        "--prune-keep",
        type=int,
        default=5,
        metavar="N",
        help="Number of recent snapshots to keep when --prune-snapshots is set (default: 5).",
    )
    _ = parser.add_argument(
        "--mcp-config",
        choices=MCP_CONFIG_FORMATS,
        help=(
            "Emit project-scoped read-only pci-mcp configuration and required environment exports "
            "after a successful run. "
            "Use --init-db-only with this option to initialize the DB and print config without indexing."
        ),
    )
    _ = parser.add_argument(
        "--mcp-server-name",
        help="Server key/name for generated MCP client config snippets. Defaults to project-code-intelligence.",
    )
    _ = parser.add_argument(
        "--show-parser-failures",
        action="store_true",
        help=(
            "List failing source paths in the summary panel. The 'Parser fails' count row is unchanged; "
            "with this flag, paths are appended below it (capped in the panel, full list in --json output)."
        ),
    )
    return parser


def parse_cli_args(argv: list[str] | None = None) -> CliArgs:
    parsed = build_parser().parse_args(argv, namespace=CliNamespace())
    embedding_endpoint_model = parsed.embedding_endpoint_model
    if (
        embedding_endpoint_model == config.DEFAULT_EMBEDDING_ENDPOINT_MODEL
        and config.env_text("PCI_EMBEDDING_ENDPOINT_MODEL") is None
    ):
        embedding_endpoint_model = config.default_embedding_endpoint_model(endpoint=parsed.embedding_endpoint)
    return CliArgs(
        root=parsed.root,
        collection=parsed.collection,
        profile=parsed.profile,
        repos=parsed.repos,
        max_file_bytes=parsed.max_file_bytes,
        scan_workers=parsed.scan_workers,
        chunk_chars=parsed.chunk_chars,
        overlap_lines=parsed.overlap_lines,
        limit_files=parsed.limit_files,
        progress_every=parsed.progress_every,
        dry_run=parsed.dry_run,
        reset_code_intel=parsed.reset_code_intel,
        i_know_this_deletes_code_intel_db=parsed.i_know_this_deletes_code_intel_db,
        reset_only=parsed.reset_only,
        init_db_only=parsed.init_db_only,
        sarif=parsed.sarif,
        no_profile_sarif=parsed.no_profile_sarif,
        sarif_max_bytes=parsed.sarif_max_bytes,
        embed_only=parsed.embed_only,
        mode=parsed.mode,
        full=parsed.full,
        no_replace=parsed.no_replace,
        embed=parsed.embed,
        embed_record_types=parsed.embed_record_types,
        embedding_batch_size=parsed.embedding_batch_size,
        embedding_max_chars=parsed.embedding_max_chars,
        embedding_endpoint=parsed.embedding_endpoint,
        embedding_endpoint_model=embedding_endpoint_model,
        llama_embed=parsed.llama_embed,
        no_preembed=parsed.no_preembed,
        prune_snapshots=parsed.prune_snapshots,
        prune_keep=parsed.prune_keep,
        mcp_config=parsed.mcp_config,
        mcp_server_name=parsed.mcp_server_name,
        show_parser_failures=parsed.show_parser_failures,
    )


def validate_non_negative_args(args: CliArgs) -> None:
    checks = {
        "--max-file-bytes": args.max_file_bytes,
        "--scan-workers": args.scan_workers,
        "--overlap-lines": args.overlap_lines,
        "--progress-every": args.progress_every,
        "--sarif-max-bytes": args.sarif_max_bytes,
    }
    if args.limit_files is not None:
        checks["--limit-files"] = args.limit_files
    for flag, value in checks.items():
        if value < 0:
            raise ValueError(f"{flag} must be non-negative")


def validate_args(args: CliArgs, *, embedding_requested: bool) -> None:
    validate_non_negative_args(args)
    if args.chunk_chars < MIN_CHUNK_CHARS:
        raise ValueError(f"--chunk-chars must be at least {MIN_CHUNK_CHARS}")
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be greater than 0")
    if embedding_requested and args.embedding_max_chars <= 0:
        raise ValueError("--embedding-max-chars must be greater than 0; omit --embed to disable embeddings")
    if args.reset_only and not args.reset_code_intel:
        raise ValueError("--reset-only requires --reset-code-intel")
    if args.init_db_only and args.reset_code_intel:
        raise ValueError("--init-db-only cannot be combined with --reset-code-intel")
    if args.init_db_only and args.embed_only:
        raise ValueError("--init-db-only cannot be combined with --embed-only")
    if args.reset_code_intel and args.embed_only:
        raise ValueError("--reset-code-intel cannot be combined with --embed-only")
    if args.reset_code_intel and args.mcp_config:
        raise ValueError("--mcp-config cannot be combined with --reset-code-intel")
    if args.dry_run and args.mcp_config:
        raise ValueError("--mcp-config cannot be combined with --dry-run")


def build_ingest_plan(args: CliArgs) -> IngestPlan:
    profile = load_profile(args.profile)
    set_active_profile(profile)
    root = args.root.resolve()
    collection = args.collection or default_collection(root)
    repos = parse_repos(args.repos or ",".join(profile.default_repos))
    set_ingest_database_scope_default(root, repos)
    embed_types = {item.strip() for item in args.embed_record_types.split(",") if item.strip()}
    embedding_requested = args.embed or args.embed_only
    preembedding_requested = args.embed and not args.no_preembed and code_preembedding_enabled()
    mode = "full" if args.full else args.mode
    if mode not in {"incremental", "full"}:
        raise ValueError("PCI_MODE must be 'incremental' or 'full'")
    validate_args(args, embedding_requested=embedding_requested)
    return IngestPlan(
        args=args,
        profile=profile,
        root=root,
        collection=collection,
        repos=repos,
        embed_types=embed_types,
        sarif_files=[],
        embedding_requested=embedding_requested,
        preembedding_requested=preembedding_requested,
        mode=mode,
    )


def embedding_run_config(args: CliArgs) -> EmbeddingRunConfig:
    return EmbeddingRunConfig(
        backend=EmbeddingBackend(
            endpoint=args.embedding_endpoint,
            endpoint_model=args.embedding_endpoint_model,
            use_llama_cli=args.llama_embed,
        ),
        max_chars=args.embedding_max_chars,
    )


def configure_ingest_progress(plan: IngestPlan) -> None:
    args = plan.args
    if args.embed_only:
        runtime_state.active_metrics.configure_progress({"embedding": 1.0})
    elif args.dry_run:
        runtime_state.active_metrics.configure_progress({"scan": 1.0})
    elif args.embed:
        runtime_state.active_metrics.configure_progress({"scan": 0.35, "db_upload": 0.2, "embedding": 0.45})
    else:
        runtime_state.active_metrics.configure_progress({"scan": 0.65, "db_upload": 0.35})


def discover_plan_sarif_files(plan: IngestPlan) -> list[Path]:
    if not plan.sarif_files:
        progress_event("code_intel_sarif_discovering", repos=plan.repos)
        sarif_patterns = explicit_sarif_patterns(plan.args.sarif)
        plan.sarif_files.extend(
            discover_sarif_files(
                plan.root,
                plan.repos,
                sarif_patterns,
                include_profile=not plan.args.no_profile_sarif,
            )
        )
    return plan.sarif_files


def emit_sarif_discovery(plan: IngestPlan, sarif_files: list[Path]) -> None:
    if sarif_files:
        progress_event(
            "code_intel_sarif_discovered",
            files=[relative_to_or_none(path, plan.root) or str(path) for path in sarif_files],
        )


def confirm_reset_code_intel(
    args: CliArgs, settings: config.DatabaseSettings, collection: str, repos: list[str]
) -> None:
    if not args.reset_code_intel:
        return
    repo_list = ", ".join(repos)
    write_stderr(f"About to drop PostgreSQL database: {settings.dbname or '<unset>'}")
    write_stderr(f"Repo(s): {repo_list}")
    write_stderr(f"Collection: {collection}")
    write_stderr("This permanently deletes all snapshots, records, edges, embeddings, findings, and schema in that DB.")
    write_stderr(f"Postgres admin connection: {db.maintenance_database_settings(settings).display_target()}")
    write_stderr("Other PCI-managed project databases are untouched.")
    if args.i_know_this_deletes_code_intel_db:
        write_stderr("Reset confirmed by --i-know-this-deletes-code-intel-db.")
        return
    if not sys.stdin.isatty():
        raise ValueError("--reset-code-intel requires --i-know-this-deletes-code-intel-db in non-interactive mode")
    _ = sys.stderr.write("Type yes to continue: ")
    _ = sys.stderr.flush()
    answer = sys.stdin.readline().strip().lower()
    if answer != "yes":
        raise ValueError("reset cancelled")


def apply_bootstrap_writer_credentials(
    settings: config.DatabaseSettings, bootstrap: db.DatabaseBootstrapResult
) -> config.DatabaseSettings:
    writer_settings = db.writable_settings_for_bootstrap(settings, bootstrap)
    rw_role = bootstrap.rw_role
    if rw_role is not None and rw_role.password:
        if settings.dsn:
            os.environ["PCI_DATABASE_USER"] = rw_role.name
            os.environ["PCI_DATABASE_PASSWORD"] = rw_role.password
        else:
            os.environ["PCI_PG_USER"] = rw_role.name
            os.environ["PCI_PG_PASS"] = rw_role.password
    return writer_settings


def database_bootstrap_report(bootstrap: db.DatabaseBootstrapResult | None) -> JsonObject:
    if bootstrap is None:
        return {}
    report: JsonObject = {
        "database_created": bootstrap.database_created,
        "database_dropped": bootstrap.database_dropped,
    }
    if bootstrap.rw_role is not None:
        report["rw_role"] = bootstrap.rw_role.name
    if bootstrap.ro_role is not None:
        report["ro_role"] = bootstrap.ro_role.name
    return report


def default_mcp_server_name(_collection: str) -> str:
    return "project-code-intelligence"


def mcp_command_path() -> str:
    return shutil.which("pci-mcp") or "pci-mcp"


def mcp_config_env(context: McpConfigContext) -> dict[str, str]:
    return {
        "PCI_MCP_DATABASE_URL": context.database_url,
        "PCI_MCP_DATABASE_USER": context.database_user,
        "PCI_MCP_DATABASE_PASSWORD": context.database_password,
        "PCI_COLLECTION": context.collection,
        "PCI_DATABASE_SCOPE_PATH": context.database_scope_path,
    }


def mcp_config_context(
    plan: IngestPlan, bootstrap: db.DatabaseBootstrapResult | None, *, command: str | None = None
) -> McpConfigContext | None:
    if bootstrap is None or bootstrap.ro_role is None:
        return None
    ro_role = bootstrap.ro_role
    password = ro_role.password
    if password is None:
        return None
    scope_path = str(config.configured_database_scope_path())
    database_url = config.database_url_without_credentials(ro_role.database_url)
    return McpConfigContext(
        server_name=plan.args.mcp_server_name or default_mcp_server_name(plan.collection),
        command=command or mcp_command_path(),
        cwd=scope_path,
        database_url=database_url,
        database_user=ro_role.name,
        database_password=password,
        collection=plan.collection,
        database_scope_path=scope_path,
    )


def _mcp_env_export_block(context: McpConfigContext, *, title: str, env_names: tuple[str, ...]) -> str:
    env = mcp_config_env(context)
    return "\n".join((
        title,
        *(_shell_export(name, env[name]) for name in env_names),
    ))


def mcp_ro_export_block(context: McpConfigContext | None) -> str | None:
    if context is None:
        return None
    return "\n" + _mcp_env_export_block(
        context,
        title="Export for pci-mcp (RO)",
        env_names=MCP_STANDALONE_ENV_NAMES,
    )


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_key(value: str) -> str:
    if value and all(char.isascii() and (char.isalnum() or char in {"_", "-"}) for char in value):
        return value
    return json.dumps(value)


def codex_mcp_config_block(context: McpConfigContext) -> str:
    key = _toml_key(context.server_name)
    lines = [
        f"[mcp_servers.{key}]",
        f"command = {_toml_string(context.command)}",
        f"cwd = {_toml_string(context.cwd)}",
        "startup_timeout_sec = 20",
        "tool_timeout_sec = 120",
        "env_vars = [",
    ]
    lines.extend(f"  {_toml_string(name)}," for name in MCP_PROJECT_ENV_NAMES)
    lines.append("]")
    return "\n".join(lines)


def _claude_env_references() -> dict[str, str]:
    return {name: "${" + name + "}" for name in MCP_PROJECT_ENV_NAMES}


def claude_mcp_config_block(context: McpConfigContext) -> str:
    payload: JsonObject = {
        "mcpServers": {
            context.server_name: {
                "type": "stdio",
                "command": context.command,
                "args": list[str](),
                "cwd": context.cwd,
                "env": _claude_env_references(),
            }
        }
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def _opencode_env_references() -> dict[str, str]:
    return {name: "{env:" + name + "}" for name in MCP_PROJECT_ENV_NAMES}


def _vscode_env_references() -> dict[str, str]:
    return {name: "${env:" + name + "}" for name in MCP_STANDALONE_ENV_NAMES}


def vscode_mcp_config_block(context: McpConfigContext) -> str:
    payload: JsonObject = {
        "servers": {
            context.server_name: {
                "type": "stdio",
                "command": context.command,
                "args": list[str](),
                "env": _vscode_env_references(),
            }
        }
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def zed_mcp_server_config(context: McpConfigContext) -> JsonObject:
    return {
        "command": context.command,
        "args": list[str](),
        "env": mcp_config_env(context),
    }


def zed_mcp_config_block(context: McpConfigContext) -> str:
    payload: JsonObject = {
        "context_servers": {
            context.server_name: zed_mcp_server_config(context),
        }
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def opencode_mcp_config_block(context: McpConfigContext) -> str:
    payload: JsonObject = {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {
            context.server_name: {
                "type": "local",
                "command": [context.command],
                "enabled": True,
                "cwd": context.cwd,
                "environment": _opencode_env_references(),
            }
        },
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def cline_mcp_config_block(context: McpConfigContext) -> str:
    payload: JsonObject = {
        "mcpServers": {
            context.server_name: {
                "command": context.command,
                "args": list[str](),
                "env": mcp_config_env(context),
                "autoApprove": list[str](),
                "disabled": False,
            }
        }
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def mcp_project_config_path(context: McpConfigContext, config_format: str) -> str:
    if config_format == "codex":
        return str(Path(context.cwd) / ".codex" / "config.toml")
    if config_format == "claude":
        return str(Path(context.cwd) / ".mcp.json")
    if config_format == "opencode":
        return str(Path(context.cwd) / "opencode.json")
    if config_format in {"vscode", "copilot"}:
        return str(Path(context.cwd) / ".vscode" / "mcp.json")
    if config_format == "cline":
        return "Cline MCP settings JSON"
    if config_format == "zed":
        return str(Path(context.cwd) / ".zed" / "settings.json")
    raise ValueError(f"unsupported MCP project config format: {config_format}")


def cline_mcp_config_guidance(context: McpConfigContext, body: str) -> str:
    return "\n".join((
        "",
        "Cline VS Code MCP config",
        f"Add or merge this snippet under mcpServers in {mcp_project_config_path(context, 'cline')}.",
        "Open it from Cline's MCP Servers icon, Configure tab, then Configure MCP Servers.",
        "Cline's VS Code MCP settings are user-scoped; use --mcp-server-name for per-project server keys.",
        "This JSON contains read-only database credentials because Cline does not document VS Code-style "
        "environment substitution here. Keep it local and do not commit it.",
        "",
        body,
    ))


def zed_mcp_config_guidance(context: McpConfigContext, body: str) -> str:
    return "\n".join((
        "",
        "Zed project-scoped MCP config",
        f"Write or merge this snippet into: {mcp_project_config_path(context, 'zed')}",
        "This Zed project settings snippet contains read-only database credentials because Zed does "
        "not document environment-variable interpolation for MCP env values.",
        "Keep .zed/settings.json local and do not commit it. Trust the worktree in Zed so project "
        "settings can start MCP servers.",
        "",
        body,
    ))


def mcp_project_config_guidance(
    context: McpConfigContext,
    config_format: str,
    body: str,
    *,
    env_names: tuple[str, ...] = MCP_PROJECT_ENV_NAMES,
) -> str:
    client = {
        "codex": "Codex",
        "claude": "Claude Code",
        "opencode": "OpenCode",
        "vscode": "VS Code Copilot",
        "copilot": "VS Code Copilot",
        "zed": "Zed",
    }[config_format]
    target = mcp_project_config_path(context, config_format)
    return "\n".join((
        "",
        f"{client} project-scoped MCP config",
        f"Write this snippet to: {target}",
        "This snippet is project-scoped and references environment variables for credentials.",
        "Load the required environment variables below before starting the MCP client.",
        "Do not paste this into a global MCP config; the server key is intentionally reused per project.",
        "",
        body,
        "",
        _mcp_env_export_block(
            context,
            title="Required environment variables for pci-mcp (RO)",
            env_names=env_names,
        ),
    ))


def mcp_config_block(context: McpConfigContext | None, config_format: str) -> str | None:
    if context is None:
        return None
    if config_format == "env":
        return mcp_ro_export_block(context)
    if config_format == "codex":
        body = codex_mcp_config_block(context)
        env_names = MCP_PROJECT_ENV_NAMES
    elif config_format == "claude":
        body = claude_mcp_config_block(context)
        env_names = MCP_PROJECT_ENV_NAMES
    elif config_format == "opencode":
        body = opencode_mcp_config_block(context)
        env_names = MCP_PROJECT_ENV_NAMES
    elif config_format in {"vscode", "copilot"}:
        body = vscode_mcp_config_block(context)
        env_names = MCP_STANDALONE_ENV_NAMES
    elif config_format == "cline":
        return cline_mcp_config_guidance(context, cline_mcp_config_block(context))
    elif config_format == "zed":
        return zed_mcp_config_guidance(context, zed_mcp_config_block(context))
    else:
        raise ValueError(f"unsupported MCP config format: {config_format}")
    return mcp_project_config_guidance(context, config_format, body, env_names=env_names)


def _bootstrap_used_fast_path(bootstrap: db.DatabaseBootstrapResult | None) -> bool:
    """Detect the rerun fast-path: the rw role exists but its password is not in-memory.

    bootstrap_inferred_database short-circuits to _connect_existing_inferred_database_with_scoped_role
    when the configured writer already matches the per-project rw role, leaving ro_role=None and
    rw_role.password=None. Re-emitting MCP creds isn't possible without re-deriving from an admin
    salt we don't have on this run, but the originally-emitted creds remain valid.
    """
    if bootstrap is None or bootstrap.rw_role is None:
        return False
    return bootstrap.rw_role.password is None and bootstrap.ro_role is None


def emit_mcp_config(plan: IngestPlan, bootstrap: db.DatabaseBootstrapResult | None) -> None:
    if progress.detect_summary_mode() == "json":
        return
    config_format = plan.args.mcp_config or "env"
    context = mcp_config_context(plan, bootstrap)
    block = mcp_config_block(context, config_format)
    if block is None:
        if plan.args.mcp_config:
            if _bootstrap_used_fast_path(bootstrap):
                write_stdout(
                    "MCP credentials were emitted when this database was first bootstrapped and "
                    "cannot be re-derived from the current writer credentials. Your existing MCP "
                    "configuration is still valid. To regenerate, run `pci-index --reset .` first."
                )
                return
            raise db.DatabaseConnectionError(
                "Could not emit MCP configuration because the project RO password is not available. "
                "Run pci-index --init-db with PCI_DATABASE_ADMIN_USER/"
                "PCI_DATABASE_ADMIN_PASSWORD set, then rerun the MCP config command."
            )
        return
    write_stdout(block)


def prepare_writable_database(args: CliArgs, *, embedding_requested: bool) -> db.DatabaseBootstrapResult | None:
    if args.dry_run:
        return None
    settings = config.DatabaseSettings.from_env()
    if not db.allow_writes(settings):
        raise PermissionError("set PCI_ALLOW_WRITES=1 to ingest")
    if embedding_requested and not args.embedding_endpoint and not args.llama_embed:
        raise ValueError("set --embedding-endpoint or --llama-embed when --embed is used")
    if embedding_requested and args.embedding_endpoint:
        preflight_embedding_endpoint(args.embedding_endpoint, args.embedding_endpoint_model)
    bootstrap = None
    if settings.database_inferred:
        bootstrap = db.bootstrap_inferred_database(settings)
        settings = apply_bootstrap_writer_credentials(settings, bootstrap)
    with db.connect(readonly=False, settings=settings) as conn:
        ensure_schema(conn)
        if bootstrap is not None and bootstrap.rw_role is not None and bootstrap.ro_role is not None:
            db.grant_project_database_object_privileges(
                conn,
                dbname=bootstrap.dbname,
                rw_role=bootstrap.rw_role.name,
                ro_role=bootstrap.ro_role.name,
            )
        conn.commit()
    return bootstrap


def set_ingest_database_scope_default(root: Path, repos: list[str]) -> None:
    scope_path = database_scope_path_for_root_repos(root, repos)
    _ = os.environ.setdefault(config.DATABASE_SCOPE_PATH_ENV, str(scope_path))


def resolve_reset_targets(args: CliArgs) -> tuple[str, list[str]]:
    profile = load_profile(args.profile)
    set_active_profile(profile)
    collection = args.collection or default_collection(args.root.resolve())
    repos = parse_repos(args.repos or ",".join(profile.default_repos))
    set_ingest_database_scope_default(args.root, repos)
    return collection, repos


def print_reset_only_report(
    args: CliArgs,
    settings: config.DatabaseSettings,
    collection: str,
    repos: list[str],
    bootstrap: db.DatabaseBootstrapResult | None,
) -> None:
    progress.emit_summary({
        "mode": "reset",
        "dry_run": args.dry_run,
        "reset": args.reset_code_intel and not args.dry_run,
        "database": settings.display_target(),
        "collection": collection,
        "repos": repos,
        **database_bootstrap_report(bootstrap),
    })


def run_reset_only(args: CliArgs) -> int:
    validate_args(args, embedding_requested=False)
    collection, repos = resolve_reset_targets(args)
    settings = config.DatabaseSettings.from_env()
    if not settings.database_inferred:
        raise db.DatabaseConnectionError(
            "Refusing to drop an explicit PostgreSQL database. "
            "Remove the database path from PCI_DATABASE_URL, or leave PCI_PG_DB unset, "
            "so pci-index --reset can target a PCI-managed inferred database."
        )
    confirm_reset_code_intel(args, settings, collection, repos)
    bootstrap: db.DatabaseBootstrapResult | None = None
    if not args.dry_run:
        progress_event("code_intel_reset_started", collection=collection, repos=repos)
        bootstrap = db.drop_inferred_database(settings)
        progress_event("code_intel_reset_completed", collection=collection, database_dropped=bootstrap.database_dropped)
    print_reset_only_report(args, settings, collection, repos, bootstrap)
    return 0


def latest_snapshot_ids(collection: str, repos: list[str]) -> list[int]:
    snapshot_ids: list[int] = []
    with db.connect() as conn:
        for repo in repos:
            snapshot = latest_snapshot_info(conn, collection, repo)
            if not snapshot:
                raise ValueError(f"no code-intel snapshot found for collection={collection!r} repo={repo!r}")
            snapshot_ids.append(json_int(snapshot, "id"))
    return snapshot_ids


@lru_cache(maxsize=1)
def index_embedding_ok_options() -> frozenset[str]:
    npu_results = check_npu_support(os.environ)
    options = check_embedding_options(env=os.environ, gpus=discover_gpus(), npu_results=npu_results)
    return frozenset(item.name for item in options if item.status == "ok")


def index_embedding_option_ok(name: str) -> bool:
    if name in {"option-cpu", "option-remote"}:
        return True
    return name in index_embedding_ok_options()


def resolve_index_embedding_framework(endpoint: str | None, response_model: str | None) -> str | None:
    if endpoint is None:
        return None
    profile = active_embedding_profile(
        endpoint=endpoint,
        response_model=response_model,
        endpoint_ok=True,
        option_ok=index_embedding_option_ok,
        advertised_framework=resolve_embedding_endpoint_framework(endpoint),
    )
    return profile.label


def print_embed_only_report(
    plan: IngestPlan,
    snapshot_ids: list[int],
    embedded_records: int | None,
    bootstrap: db.DatabaseBootstrapResult | None,
) -> None:
    args = plan.args
    report: JsonObject = {
        "repos": plan.repos,
        "collection": plan.collection,
        "database": config.DatabaseSettings.from_env().display_target(),
        "snapshot_ids": snapshot_ids,
        "mode": "embed-only",
        "profile": profile_context.active_profile.name,
        "embeddings": True,
        "embedding_model": args.embedding_endpoint_model if args.embedding_endpoint else None,
        "embedding_endpoint": args.embedding_endpoint,
        "embedding_framework": resolve_index_embedding_framework(
            args.embedding_endpoint, args.embedding_endpoint_model
        ),
        "embedding_max_chars": args.embedding_max_chars,
        "metrics": runtime_state.active_metrics.snapshot(),
        **database_bootstrap_report(bootstrap),
    }
    if embedded_records is not None:
        report.update({
            "embedded_records": embedded_records,
            "embedded_records_post_insert": embedded_records,
            "embedded_records_total": embedded_records,
        })
    progress.emit_summary(report)
    emit_mcp_config(plan, bootstrap)


def run_embed_only(plan: IngestPlan, bootstrap: db.DatabaseBootstrapResult | None) -> int:
    args = plan.args
    snapshot_ids = latest_snapshot_ids(plan.collection, plan.repos)
    if args.dry_run:
        print_embed_only_report(plan, snapshot_ids, None, bootstrap)
        return 0
    runtime_state.active_metrics.begin_phase("embedding")
    try:
        embedded_records = embed_db_records(
            snapshot_ids,
            record_types=plan.embed_types,
            batch_size=args.embedding_batch_size,
            run_config=embedding_run_config(args),
        )
    finally:
        runtime_state.active_metrics.complete_phase("embedding")
    with db.connect(readonly=False) as conn:
        stamp_embed_types(conn, snapshot_ids, plan.embed_types)
        conn.commit()
    print_embed_only_report(plan, snapshot_ids, embedded_records, bootstrap)
    return 0


def print_init_db_report(plan: IngestPlan, bootstrap: db.DatabaseBootstrapResult | None) -> None:
    progress.emit_summary({
        "mode": "init-db",
        "dry_run": plan.args.dry_run,
        "database": config.DatabaseSettings.from_env().display_target(),
        "collection": plan.collection,
        "repos": plan.repos,
        **database_bootstrap_report(bootstrap),
    })
    emit_mcp_config(plan, bootstrap)


def run_init_db_only(plan: IngestPlan, bootstrap: db.DatabaseBootstrapResult | None) -> int:
    print_init_db_report(plan, bootstrap)
    return 0


def previous_repo_states(plan: IngestPlan) -> dict[str, tuple[int | None, dict[str, PreviousFileState], str]]:
    if plan.mode != "incremental":
        return {repo: (None, {}, "full") for repo in plan.repos}
    try:
        with db.connect() as conn:
            return {repo: previous_repo_state(conn, plan.collection, repo) for repo in plan.repos}
    except (db.DatabaseConnectionError, db.OperationalError) as exc:
        progress_event("code_intel_incremental_unavailable", error=str(exc))
        return {repo: (None, {}, "full") for repo in plan.repos}


def previous_repo_state(
    conn: db.DbConnection,
    collection: str,
    repo: str,
) -> tuple[int | None, dict[str, PreviousFileState], str]:
    previous = latest_snapshot_info(conn, collection, repo)
    metadata = previous.get("metadata") if previous else None
    metadata_obj = metadata if isinstance(metadata, dict) else None
    if previous and snapshot_versions_compatible(metadata_obj):
        previous_id = json_int(previous, "id")
        return previous_id, previous_file_states(conn, previous_id), "incremental"
    return None, {}, "full"


def add_repo_ingest_metrics(ingest: RepoIngest) -> None:
    runtime_state.active_metrics.add("discovered_files", len(ingest.files))
    runtime_state.active_metrics.add("changed_files", len(ingest.changed_paths))
    runtime_state.active_metrics.add("unchanged_files", len(ingest.unchanged_paths))
    runtime_state.active_metrics.add(
        "parsed_files",
        sum(1 for file in ingest.files if file.source_path in ingest.changed_paths and not file.skipped_reason),
    )
    runtime_state.active_metrics.add("generated_records", len(ingest.records))
    runtime_state.active_metrics.add("generated_edges", len(ingest.edges))
    runtime_state.active_metrics.add("parser_failures", len(ingest.parser_failures))


def scan_repositories(
    plan: IngestPlan,
    previous_by_repo: dict[str, tuple[int | None, dict[str, PreviousFileState], str]],
) -> list[RepoIngest]:
    ingests: list[RepoIngest] = []
    for repo in plan.repos:
        previous_snapshot_id, previous_files, repo_mode = previous_by_repo.get(repo, (None, {}, "full"))
        ingest = ingest_repo(
            RepoIngestConfig(
                root=plan.root,
                repo=repo,
                collection=plan.collection,
                profile_name=plan.args.profile,
                max_file_bytes=plan.args.max_file_bytes,
                scan_workers=plan.args.scan_workers,
                max_chars=plan.args.chunk_chars,
                overlap_lines=plan.args.overlap_lines,
                limit_files=plan.args.limit_files,
                progress_every=plan.args.progress_every,
                previous_snapshot_id=previous_snapshot_id,
                previous_files=previous_files,
                mode=repo_mode,
            )
        )
        ingests.append(ingest)
        add_repo_ingest_metrics(ingest)
    return ingests


def merge_sarif_into_ingests(plan: IngestPlan, ingests: list[RepoIngest], sarif_ingest: SarifIngest) -> None:
    for ingest in ingests:
        static_records = sarif_ingest.records_by_repo.get(ingest.snapshot.repo, [])
        if static_records:
            ingest.records.extend(static_records)
            runtime_state.active_metrics.add("generated_records", len(static_records))
    ingest_by_repo = {ingest.snapshot.repo: ingest for ingest in ingests}
    for failure in sarif_ingest.failures:
        failure_repo = repo_for_source_path(
            str(failure.get("source_path") or ""),
            plan.repos,
            plan.repos[0] if plan.repos else None,
        )
        if failure_repo in ingest_by_repo:
            ingest_by_repo[failure_repo].parser_failures.append({"language": "sarif", **failure})
    runtime_state.active_metrics.add("parser_failures", len(sarif_ingest.failures))


def scan_sarif(plan: IngestPlan, ingests: list[RepoIngest]) -> SarifIngest:
    started = time.monotonic()
    sarif_ingest = SarifIngest(runs=[], records_by_repo={}, failures=[])
    try:
        sarif_files = discover_plan_sarif_files(plan)
        emit_sarif_discovery(plan, sarif_files)
        if not sarif_files:
            return sarif_ingest
        runtime_state.active_metrics.add_phase_total(len(sarif_files))
        file_by_source_path = {file.source_path: file for ingest in ingests for file in ingest.files}
        sarif_ingest = ingest_sarif(
            SarifIngestContext(
                root=plan.root,
                repos=plan.repos,
                collection=plan.collection,
                file_by_source_path=file_by_source_path,
                max_bytes=plan.args.sarif_max_bytes,
            ),
            sarif_files,
        )
        runtime_state.active_metrics.add_phase_done(len(sarif_files))
        merge_sarif_into_ingests(plan, ingests, sarif_ingest)
        sarif_ingest.warnings.extend(sarif_freshness_warnings(plan, ingests, sarif_ingest))
        attach_sarif_warnings_to_runs(sarif_ingest)
        progress_event(
            "code_intel_sarif_parsed",
            files=len(sarif_files),
            runs=len(sarif_ingest.runs),
            findings=sum(len(run.findings) for run in sarif_ingest.runs),
            records=sum(len(records) for records in sarif_ingest.records_by_repo.values()),
            parser_failures=len(sarif_ingest.failures),
            warnings=len(sarif_ingest.warnings),
        )
        return sarif_ingest
    finally:
        runtime_state.active_metrics.add("scan_sarif_seconds", time.monotonic() - started)


def parse_sarif_warning_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    with suppress(ValueError):
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def snapshot_commit_datetime(snapshot: Snapshot) -> datetime | None:
    return parse_sarif_warning_datetime(snapshot.metadata.get("commit_time"))


def sarif_warning_path(plan: IngestPlan, path: Path) -> str:
    return relative_to_or_none(path, plan.root) or str(path)


def warning_for_sarif_mtime(plan: IngestPlan, snapshot_by_repo: dict[str, Snapshot], path: Path) -> JsonObject | None:
    repo = repo_for_sarif_file(plan.root, plan.repos, path)
    if not repo:
        return None
    snapshot = snapshot_by_repo.get(repo)
    if snapshot is None:
        return None
    commit_time = snapshot_commit_datetime(snapshot)
    if commit_time is None:
        return None
    with suppress(OSError):
        sarif_mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if sarif_mtime < commit_time:
            sarif_path = sarif_warning_path(plan, path)
            return {
                "severity": "note",
                "reason": "sarif_older_than_snapshot_commit",
                "sarif_path": sarif_path,
                "repo": repo,
                "sarif_mtime": sarif_mtime.isoformat(),
                "snapshot_commit_time": commit_time.isoformat(),
                "snapshot_commit": snapshot.commit_sha,
                "message": (
                    f"{sarif_path} is older than the indexed {repo} commit; "
                    "findings may still be useful, but freshness could not be verified."
                ),
            }
    return None


def sarif_provenance_revision_ids(run_metadata: JsonObject) -> list[str]:
    provenance = run_metadata.get("versionControlProvenance")
    if not isinstance(provenance, list):
        return []
    revisions: list[str] = []
    for item in provenance:
        if isinstance(item, dict):
            revision = item.get("revisionId")
            if isinstance(revision, str) and revision:
                revisions.append(revision)
    return sorted(set(revisions))


def commit_matches_provenance(snapshot_commit: str, sarif_revision: str) -> bool:
    return (
        snapshot_commit == sarif_revision
        or snapshot_commit.startswith(sarif_revision)
        or sarif_revision.startswith(snapshot_commit)
    )


def warning_for_sarif_provenance(run: StaticRun, snapshot_by_repo: dict[str, Snapshot]) -> JsonObject | None:
    snapshot = snapshot_by_repo.get(run.repo)
    if snapshot is None:
        return None
    revisions = sarif_provenance_revision_ids(run.metadata)
    mismatches = [revision for revision in revisions if not commit_matches_provenance(snapshot.commit_sha, revision)]
    if not mismatches:
        return None
    return {
        "severity": "warn",
        "reason": "sarif_provenance_commit_mismatch",
        "sarif_path": run.sarif_path,
        "repo": run.repo,
        "sarif_revision_ids": mismatches,
        "snapshot_commit": snapshot.commit_sha,
        "message": f"{run.sarif_path} was generated for a different {run.repo} revision than the indexed snapshot.",
    }


def sarif_freshness_warnings(
    plan: IngestPlan,
    ingests: list[RepoIngest],
    sarif_ingest: SarifIngest,
) -> list[JsonObject]:
    snapshot_by_repo = {ingest.snapshot.repo: ingest.snapshot for ingest in ingests}
    warnings: list[JsonObject] = []
    for path in plan.sarif_files:
        warning = warning_for_sarif_mtime(plan, snapshot_by_repo, path)
        if warning is not None:
            warnings.append(warning)
    for run in sarif_ingest.runs:
        warning = warning_for_sarif_provenance(run, snapshot_by_repo)
        if warning is not None:
            warnings.append(warning)
    return warnings


def attach_sarif_warnings_to_runs(sarif_ingest: SarifIngest) -> None:
    warnings_by_path: dict[str, list[JsonObject]] = {}
    for warning in sarif_ingest.warnings:
        path = warning.get("sarif_path")
        if isinstance(path, str):
            warnings_by_path.setdefault(path, []).append(warning)
    for run in sarif_ingest.runs:
        warnings = warnings_by_path.get(run.sarif_path)
        if warnings:
            run.metadata["code_intel_warnings"] = warnings


def scan_plan(plan: IngestPlan) -> tuple[list[RepoIngest], SarifIngest]:
    runtime_state.active_metrics.begin_phase("scan")
    try:
        ingests = scan_repositories(plan, previous_repo_states(plan))
        sarif_ingest = scan_sarif(plan, ingests)
        return ingests, sarif_ingest
    finally:
        runtime_state.active_metrics.end_phase("scan", "scan_seconds")


def print_dry_run_report(plan: IngestPlan, ingests: list[RepoIngest], sarif_ingest: SarifIngest) -> None:
    report = report_ingests(ingests, embeddings=plan.args.embed)
    report["database"] = config.DatabaseSettings.from_env().display_target()
    report["sarif_files"] = [relative_to_or_none(path, plan.root) or str(path) for path in plan.sarif_files]
    report["sarif_file_count"] = len(plan.sarif_files)
    report["sarif_warnings"] = sarif_ingest.warnings
    report["static_runs"] = len(sarif_ingest.runs)
    report["static_findings"] = sum(len(run.findings) for run in sarif_ingest.runs)
    report["static_rules"] = sum(len(run.rules) for run in sarif_ingest.runs)
    report["static_locations"] = sum(len(finding.locations) for run in sarif_ingest.runs for finding in run.findings)
    report["static_code_flow_steps"] = sum(
        len(finding.code_flows) for run in sarif_ingest.runs for finding in run.findings
    )
    report["sarif_parser_failures"] = sarif_ingest.failures
    report["metrics"] = runtime_state.active_metrics.snapshot()
    report["dry_run"] = True
    progress.emit_summary(report, indent=2)


def db_upload_total(ingests: list[RepoIngest]) -> int:
    return sum(
        len(ingest.files) + len(ingest.records) + len(ingest.edges) + len(ingest.parser_failures) for ingest in ingests
    )


def copy_unchanged_data(conn: db.DbConnection, ingest: RepoIngest, snapshot_id: int, summary: DbUploadSummary) -> None:
    copied_record_count, copied_edge_count = copy_unchanged_records_and_edges(
        conn,
        previous_snapshot_id=ingest.previous_snapshot_id,
        snapshot=ingest.snapshot,
        snapshot_id=snapshot_id,
        unchanged_paths=ingest.unchanged_paths,
    )
    copied_parser_failure_count, copied_parser_failure_paths = copy_unchanged_parser_failures(
        conn,
        previous_snapshot_id=ingest.previous_snapshot_id,
        snapshot=ingest.snapshot,
        snapshot_id=snapshot_id,
        unchanged_paths=ingest.unchanged_paths,
    )
    summary.copied_records += copied_record_count
    summary.copied_edges += copied_edge_count
    summary.copied_parser_failures += copied_parser_failure_count
    for path in copied_parser_failure_paths:
        summary.parser_failure_paths.append(f"{ingest.snapshot.repo}/{path}")
    runtime_state.active_metrics.add_phase_total(copied_record_count + copied_edge_count + copied_parser_failure_count)
    runtime_state.active_metrics.add_phase_done(copied_record_count + copied_edge_count + copied_parser_failure_count)
    runtime_state.active_metrics.add("copied_records", copied_record_count)
    runtime_state.active_metrics.add("copied_edges", copied_edge_count)
    runtime_state.active_metrics.add("copied_parser_failures", copied_parser_failure_count)


def insert_repo_records(
    plan: IngestPlan,
    ingest: RepoIngest,
    insert_context: RecordInsertContext,
    summary: DbUploadSummary,
    *,
    progress_fn: Callable[[int], None] | None = None,
) -> tuple[int, int, int]:
    preembedding_state = (
        start_record_preembedding(
            ingest.records,
            record_types=plan.embed_types,
            batch_size=plan.args.embedding_batch_size,
            run_config=embedding_run_config(plan.args),
        )
        if plan.preembedding_requested
        else None
    )
    try:
        if preembedding_state is None:
            inserted_records = insert_records(insert_context, ingest.records, progress_fn=progress_fn)
            return inserted_records, 0, 0
        inserted_records, preembedded, skipped = insert_records_with_preembedding(
            insert_context,
            ingest.records,
            preembedding_state,
            progress_fn=progress_fn,
        )
        summary.preembedded_records += preembedded
        summary.preembedding_skipped += skipped
        return inserted_records, preembedded, skipped
    finally:
        abandon_preembedding(preembedding_state)


def record_inserted_failure_paths(summary: DbUploadSummary, ingest: RepoIngest) -> None:
    """Stash `{repo}/{source_path}` for each newly-parsed parser failure."""
    repo = ingest.snapshot.repo
    for failure in ingest.parser_failures:
        value = failure.get("source_path")
        if isinstance(value, str):
            summary.parser_failure_paths.append(f"{repo}/{value}")


def upload_repo_ingest(
    conn: db.DbConnection,
    plan: IngestPlan,
    ingest: RepoIngest,
    summary: DbUploadSummary,
    indexes: StaticSnapshotIndexes,
) -> None:
    add_progress = runtime_state.active_metrics.add_phase_done
    if plan.embedding_requested:
        ingest.snapshot.metadata["embed_record_types"] = sorted(plan.embed_types)
    snapshot_id = insert_snapshot(conn, ingest.snapshot)
    summary.snapshot_ids.append(snapshot_id)
    indexes.snapshot_ids_by_repo[ingest.snapshot.repo] = snapshot_id
    indexes.snapshot_by_repo[ingest.snapshot.repo] = ingest.snapshot
    runtime_state.active_metrics.set("db_write_op", "inserting files")
    file_ids: dict[str, int] = {}
    for file_batch in _chunks(ingest.files, _DB_WRITE_BATCH_SIZE):
        ids = insert_files(conn, snapshot_id, file_batch)
        file_ids.update(ids)
        add_progress(len(ids))
        runtime_state.active_metrics.add("inserted_files", len(ids))
    runtime_state.active_metrics.set("db_write_op", None)
    file_hashes = {item.source_path: item.file_sha256 for item in ingest.files}
    insert_context = RecordInsertContext(conn, ingest.snapshot, snapshot_id, file_ids, file_hashes)
    summary.inserted_files += len(file_ids)
    copy_unchanged_data(conn, ingest, snapshot_id, summary)
    runtime_state.active_metrics.set("db_write_op", "inserting records")

    def add_record_progress(count: int) -> None:
        add_progress(count)
        runtime_state.active_metrics.add("inserted_records", count)

    inserted_records, _preembedded, _skipped = insert_repo_records(
        plan, ingest, insert_context, summary, progress_fn=add_record_progress
    )
    summary.inserted_records += inserted_records
    runtime_state.active_metrics.set("db_write_op", "preparing edge targets")
    runtime_state.active_metrics.add_phase_total(pre_resolvable_edge_count(ingest.edges))
    pre_resolved_edges = pre_resolve_edge_targets(conn, snapshot_id, ingest.edges, progress_fn=add_progress)
    runtime_state.active_metrics.add("pre_resolved_edges", pre_resolved_edges)
    runtime_state.active_metrics.add("resolved_edges", pre_resolved_edges)
    runtime_state.active_metrics.set("db_write_op", "inserting edges")
    inserted_edges = insert_edges(conn, ingest.snapshot, snapshot_id, ingest.edges, progress_fn=add_progress)
    summary.inserted_edges += inserted_edges
    runtime_state.active_metrics.add("inserted_edges", inserted_edges)
    runtime_state.active_metrics.set("db_write_op", "resolving remaining edge targets")
    unresolved_edges = count_unresolved_edge_targets(conn, snapshot_id)
    runtime_state.active_metrics.add_phase_total(unresolved_edges)
    resolved_remaining_edges = resolve_edge_targets(conn, snapshot_id, progress_fn=add_progress)
    resolved_edges = pre_resolved_edges + resolved_remaining_edges
    runtime_state.active_metrics.add("resolved_edges", resolved_remaining_edges)
    runtime_state.active_metrics.set("db_write_op", "inserting failures")
    inserted_failures = insert_parser_failures(conn, ingest.snapshot, snapshot_id, ingest.parser_failures)
    summary.inserted_parser_failures += inserted_failures
    record_inserted_failure_paths(summary, ingest)
    add_progress(inserted_failures)
    runtime_state.active_metrics.set("db_write_op", None)
    runtime_state.active_metrics.add("inserted_parser_failures", inserted_failures)
    progress_event(
        "code_intel_inserted",
        repo=ingest.snapshot.repo,
        snapshot_id=snapshot_id,
        mode=ingest.mode,
        files=len(file_ids),
        changed_files=len(ingest.changed_paths),
        unchanged_files=len(ingest.unchanged_paths),
        records=len(ingest.records),
        edges=len(ingest.edges),
        resolved_edges=resolved_edges,
        copied_records=summary.copied_records,
        copied_edges=summary.copied_edges,
        copied_parser_failures=summary.copied_parser_failures,
        parser_failures=inserted_failures,
        preembedded_records=summary.preembedded_records,
        preembedding_skipped=summary.preembedding_skipped,
    )


def insert_static_analysis(
    conn: db.DbConnection,
    sarif_ingest: SarifIngest,
    snapshot_ids_by_repo: dict[str, int],
    snapshot_by_repo: dict[str, Snapshot],
    summary: DbUploadSummary,
) -> None:
    if not sarif_ingest.runs:
        return
    summary.static_counts = insert_static_runs(
        conn,
        snapshot_ids_by_repo=snapshot_ids_by_repo,
        snapshot_by_repo=snapshot_by_repo,
        runs=sarif_ingest.runs,
    )
    static_done = sum(summary.static_counts.values())
    runtime_state.active_metrics.add_phase_total(static_done)
    runtime_state.active_metrics.add_phase_done(static_done)
    for key, value in summary.static_counts.items():
        runtime_state.active_metrics.add(key, value)
    progress_event("code_intel_static_inserted", **summary.static_counts)


def upload_ingests(plan: IngestPlan, ingests: list[RepoIngest], sarif_ingest: SarifIngest) -> DbUploadSummary:
    summary = DbUploadSummary()
    runtime_state.active_metrics.begin_phase("db_upload", total=db_upload_total(ingests))
    try:
        with db.connect(readonly=False) as conn:
            ensure_schema(conn)
            replace_repos_for_full_ingests(conn, plan, ingests)
            indexes = StaticSnapshotIndexes()
            for ingest in ingests:
                upload_repo_ingest(conn, plan, ingest, summary, indexes)
            insert_static_analysis(conn, sarif_ingest, indexes.snapshot_ids_by_repo, indexes.snapshot_by_repo, summary)
            runtime_state.active_metrics.set("db_write_op", "committing")
            conn.commit()
            runtime_state.active_metrics.set("db_write_op", None)
    finally:
        runtime_state.active_metrics.end_phase("db_upload", "db_upload_seconds")
    return summary


def replace_repos_for_full_ingests(conn: db.DbConnection, plan: IngestPlan, ingests: list[RepoIngest]) -> None:
    if plan.args.no_replace:
        return
    repos = sorted({ingest.snapshot.repo for ingest in ingests if ingest.mode == "full"})
    if repos:
        replace_repos(conn, plan.collection, repos)


def embed_after_upload(plan: IngestPlan, snapshot_ids: list[int]) -> int:
    if not plan.args.embed:
        return 0
    runtime_state.active_metrics.begin_phase("embedding")
    try:
        return embed_db_records(
            snapshot_ids,
            record_types=plan.embed_types,
            batch_size=plan.args.embedding_batch_size,
            run_config=embedding_run_config(plan.args),
        )
    finally:
        runtime_state.active_metrics.complete_phase("embedding")


def effective_ingest_mode(ingests: list[RepoIngest]) -> str:
    modes = {ingest.mode for ingest in ingests}
    if not modes:
        return "full"
    return "full" if "full" in modes else next(iter(modes))


def print_ingest_result(
    plan: IngestPlan,
    ingests: list[RepoIngest],
    summary: DbUploadSummary,
    embedded_records: int,
    sarif_ingest: SarifIngest,
) -> None:
    report: JsonObject = {
        "repos": plan.repos,
        "collection": plan.collection,
        "database": config.DatabaseSettings.from_env().display_target(),
        "snapshot_ids": summary.snapshot_ids,
        "files": summary.inserted_files,
        "records": summary.inserted_records,
        "edges": summary.inserted_edges,
        "parser_failures": summary.inserted_parser_failures + summary.copied_parser_failures,
        "inserted_parser_failures": summary.inserted_parser_failures,
        **summary.static_counts,
        "copied_records": summary.copied_records,
        "copied_edges": summary.copied_edges,
        "copied_parser_failures": summary.copied_parser_failures,
        "mode": effective_ingest_mode(ingests),
        "profile": profile_context.active_profile.name,
        "embeddings": plan.args.embed,
        "embedding_model": (
            plan.args.embedding_endpoint_model if plan.args.embed and plan.args.embedding_endpoint else None
        ),
        "embedding_endpoint": plan.args.embedding_endpoint if plan.args.embed else None,
        "embedding_framework": (
            resolve_index_embedding_framework(plan.args.embedding_endpoint, plan.args.embedding_endpoint_model)
            if plan.args.embed
            else None
        ),
        "sarif_files": [relative_to_or_none(path, plan.root) or str(path) for path in plan.sarif_files],
        "sarif_file_count": len(plan.sarif_files),
        "sarif_warnings": sarif_ingest.warnings,
        "embedded_records": embedded_records,
        "embedded_records_post_insert": embedded_records,
        "preembedded_records": summary.preembedded_records,
        "embedded_records_total": summary.preembedded_records + embedded_records,
        "preembedding_skipped": summary.preembedding_skipped,
        "embedding_max_chars": plan.args.embedding_max_chars if plan.args.embed else None,
        "metrics": runtime_state.active_metrics.snapshot(),
        **database_bootstrap_report(summary.bootstrap),
    }
    if plan.args.show_parser_failures:
        report["parser_failure_paths"] = sorted(summary.parser_failure_paths)
    progress.emit_summary(report)
    emit_mcp_config(plan, summary.bootstrap)


def resolve_plan_embedding_model(plan: IngestPlan) -> IngestPlan:
    args = plan.args
    if not plan.embedding_requested or args.dry_run or not args.embedding_endpoint:
        return plan
    resolved_model = resolve_embedding_endpoint_model(args.embedding_endpoint, args.embedding_endpoint_model)
    if resolved_model == args.embedding_endpoint_model:
        return plan
    return replace(plan, args=replace(args, embedding_endpoint_model=resolved_model))


def run_ingest_plan(plan: IngestPlan) -> int:
    plan = resolve_plan_embedding_model(plan)
    embedding_endpoint = plan.args.embedding_endpoint if plan.embedding_requested else None
    embedding_framework = resolve_index_embedding_framework(embedding_endpoint, plan.args.embedding_endpoint_model)
    progress_event(
        "code_intel_plan",
        collection=plan.collection,
        repos=plan.repos,
        database=config.DatabaseSettings.from_env().display_target(),
        embedding_endpoint=embedding_endpoint,
        embedding_model=plan.args.embedding_endpoint_model if plan.embedding_requested else None,
        embedding_framework=embedding_framework,
    )
    configure_ingest_progress(plan)
    bootstrap = prepare_writable_database(plan.args, embedding_requested=plan.embedding_requested)
    if plan.args.init_db_only:
        return run_init_db_only(plan, bootstrap)
    if plan.args.embed_only:
        return run_embed_only(plan, bootstrap)
    ingests, sarif_ingest = scan_plan(plan)
    if plan.args.dry_run:
        print_dry_run_report(plan, ingests, sarif_ingest)
        return 0
    summary = upload_ingests(plan, ingests, sarif_ingest)
    summary.bootstrap = bootstrap
    embedded_records = embed_after_upload(plan, summary.snapshot_ids)
    print_ingest_result(plan, ingests, summary, embedded_records, sarif_ingest)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_cli_args(argv)
    if args.reset_only:
        return run_reset_only(args)
    try:
        result = run_ingest_plan(build_ingest_plan(args))
    except EmbeddingContractMismatchError as exc:
        raise EmbeddingContractMismatchError(
            f"{exc}.\n\n"
            "The index was built with a different embedding model than the current server.\n"
            "To reset and re-index with the current model:\n"
            f"  pci-index {args.root} --reset-code-intel --i-know-this-deletes-code-intel-db\n"
            f"  pci-index {args.root}"
        ) from None
    if args.prune_snapshots and result == 0:
        collection = args.collection or default_collection(args.root)
        repo_names = parse_repos(args.repos or "")
        with db.connect() as conn:
            for repo in repo_names:
                deleted = prune_old_snapshots(conn, collection, repo, keep=args.prune_keep)
                if deleted > 0:
                    progress_event("code_intel_prune", collection=collection, repo=repo, deleted=deleted)
    return result


def cli_main(argv: list[str] | None = None) -> int:
    _ = runtime_state.reset_active_metrics()
    started = time.monotonic()
    stop_heartbeat = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    heartbeat_interval = runtime_heartbeat_seconds()
    if heartbeat_interval:
        heartbeat_thread = threading.Thread(
            target=runtime_heartbeat,
            args=(started, stop_heartbeat, heartbeat_interval, runtime_state.active_metrics),
            daemon=True,
        )
        heartbeat_thread.start()
    exit_code: int | None = None
    interrupted = False
    deferred_stderr: str | None = None
    try:
        result = main(argv)
    except SystemExit as exc:
        code = exc.code
        exit_code = code if isinstance(code, int) else 1
        raise
    except KeyboardInterrupt:
        interrupted = True
        exit_code = 130
        return 130
    except (EmbeddingEndpointUnavailableError, db.DatabaseConnectionError, InsufficientPrivilege) as exc:
        exit_code = 1
        deferred_stderr = str(exc)
        return 1
    except (PermissionError, ValueError) as exc:
        exit_code = 1
        deferred_stderr = str(exc)
        return 1
    except BaseException:
        exit_code = 1
        raise
    else:
        exit_code = result
        return result
    finally:
        stop_heartbeat.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1)
        elapsed = time.monotonic() - started
        with suppress(BrokenPipeError):
            _ = sys.stdout.flush()
        progress_event(
            "code_intel_runtime",
            seconds=round(elapsed, 3),
            duration=format_duration(elapsed),
            interrupted=interrupted,
            exit_code=exit_code,
            metrics=runtime_state.active_metrics.snapshot(),
        )
        progress.close_emitter()
        if deferred_stderr is not None:
            write_stderr(deferred_stderr)


if __name__ == "__main__":
    raise SystemExit(cli_main())
