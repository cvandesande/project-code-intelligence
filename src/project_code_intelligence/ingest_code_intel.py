"""Orchestrate repository code-intelligence ingestion into Postgres."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from project_code_intelligence import config, db, profile_context, progress
from project_code_intelligence import runtime as runtime_state
from project_code_intelligence.code_profiles import load_profile
from project_code_intelligence.common import default_collection, parse_repos, repo_for_source_path
from project_code_intelligence.embeddings import (
    EmbeddingBackend,
    EmbeddingEndpointUnavailableError,
    EmbeddingRunConfig,
    abandon_preembedding,
    code_preembedding_enabled,
    embed_db_records,
    insert_records_with_preembedding,
    preflight_embedding_endpoint,
    resolve_embedding_endpoint_model,
    start_record_preembedding,
)
from project_code_intelligence.git_utils import workspace_root
from project_code_intelligence.inventory import discover_files, make_snapshot
from project_code_intelligence.models import (
    DEFAULT_EMBED_RECORD_TYPES,
    IntelEdge,
    IntelRecord,
    JsonObject,
    RepoIngest,
    SarifIngest,
    Snapshot,
)
from project_code_intelligence.parsers import parse_file
from project_code_intelligence.profile_context import set_active_profile
from project_code_intelligence.reporting import report_ingests
from project_code_intelligence.runtime import (
    format_duration,
    progress_event,
    runtime_heartbeat,
    runtime_heartbeat_seconds,
)
from project_code_intelligence.sarif import (
    SarifIngestContext,
    discover_sarif_files,
    explicit_sarif_patterns,
    ingest_sarif,
    relative_to_or_none,
)
from project_code_intelligence.storage import (
    RecordInsertContext,
    copy_unchanged_parser_failures,
    copy_unchanged_records_and_edges,
    delete_all_code_intel_data,
    delete_repo_data,
    ensure_schema,
    file_signature,
    insert_edges,
    insert_files,
    insert_parser_failures,
    insert_records,
    insert_snapshot,
    insert_static_runs,
    latest_snapshot_info,
    previous_file_signatures,
    replace_repos,
    snapshot_versions_compatible,
)

if TYPE_CHECKING:
    from project_code_intelligence.code_profiles.base import CodeIntelProfile

MIN_CHUNK_CHARS = 100


def write_stdout(message: str) -> None:
    _ = sys.stdout.write(message + "\n")


def write_stderr(message: str) -> None:
    _ = sys.stderr.write(message + "\n")


@dataclass(frozen=True)
class CliArgs:
    root: Path
    collection: str | None
    profile: str
    repos: str | None
    max_file_bytes: int
    chunk_chars: int
    overlap_lines: int
    limit_files: int | None
    progress_every: int
    dry_run: bool
    reset_code_intel: bool
    reset_all_code_intel: bool
    i_know_this_deletes_code_intel_db: bool
    reset_only: bool
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
class RepoIngestConfig:
    root: Path
    repo: str
    collection: str
    max_file_bytes: int
    max_chars: int
    overlap_lines: int
    limit_files: int | None
    progress_every: int
    previous_snapshot_id: int | None = None
    previous_signatures: dict[str, str] | None = None
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
    static_counts: dict[str, int] = field(
        default_factory=lambda: {
            "static_runs": 0,
            "static_rules": 0,
            "static_findings": 0,
            "static_locations": 0,
            "static_code_flow_steps": 0,
        }
    )


@dataclass
class StaticSnapshotIndexes:
    snapshot_ids_by_repo: dict[str, int] = field(default_factory=dict)
    snapshot_by_repo: dict[str, Snapshot] = field(default_factory=dict)


class CliNamespace(argparse.Namespace):
    root: Path
    collection: str | None
    profile: str
    repos: str | None
    max_file_bytes: int
    chunk_chars: int
    overlap_lines: int
    limit_files: int | None
    progress_every: int
    dry_run: bool
    reset_code_intel: bool
    reset_all_code_intel: bool
    i_know_this_deletes_code_intel_db: bool
    reset_only: bool
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


def json_int(obj: JsonObject, key: str) -> int:
    value = obj.get(key)
    if isinstance(value, bool):
        raise TypeError(f"{key} is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise TypeError(f"{key} is not an integer")


def ingest_repo(config: RepoIngestConfig) -> RepoIngest:
    snapshot = make_snapshot(config.root, config.repo, config.collection)
    files = discover_files(config.root, snapshot, config.max_file_bytes)
    if config.limit_files is not None:
        files = files[: config.limit_files]
    runtime_state.active_metrics.add_phase_total(len(files))
    previous_snapshot_id = config.previous_snapshot_id
    previous_signatures = config.previous_signatures or {}
    current_signatures = {item.source_path: file_signature(item) for item in files}
    unchanged_paths: set[str] = {
        path for path, signature in current_signatures.items() if previous_signatures.get(path) == signature
    }
    deleted_paths: set[str] = set(previous_signatures) - set(current_signatures)
    if config.mode != "incremental":
        previous_snapshot_id = None
        unchanged_paths = set[str]()
        deleted_paths = set[str]()
    changed_paths: set[str] = set(current_signatures) - unchanged_paths
    progress_event(
        "code_intel_discovered",
        repo=config.repo,
        files=len(files),
        changed_files=len(changed_paths),
        unchanged_files=len(unchanged_paths),
        deleted_files=len(deleted_paths),
        mode=config.mode,
        commit_sha=snapshot.commit_sha,
        tree_sha=snapshot.tree_sha,
    )
    records: list[IntelRecord] = []
    edges: list[IntelEdge] = []
    failures: list[JsonObject] = []
    for idx, intel_file in enumerate(files, 1):
        runtime_state.active_metrics.add_phase_done(1)
        if intel_file.source_path in unchanged_paths:
            if config.progress_every and (idx % config.progress_every == 0 or idx == len(files)):
                progress_event(
                    "code_intel_parsed",
                    repo=config.repo,
                    files=idx,
                    total_files=len(files),
                    changed_files=len(changed_paths),
                    unchanged_files=len(unchanged_paths),
                    records=len(records),
                    edges=len(edges),
                    parser_failures=len(failures),
                )
            continue
        file_records, file_edges, file_failures = parse_file(intel_file, config.max_chars, config.overlap_lines)
        records.extend(file_records)
        edges.extend(file_edges)
        failures.extend(file_failures)
        if config.progress_every and (idx % config.progress_every == 0 or idx == len(files)):
            progress_event(
                "code_intel_parsed",
                repo=config.repo,
                files=idx,
                total_files=len(files),
                changed_files=len(changed_paths),
                unchanged_files=len(unchanged_paths),
                records=len(records),
                edges=len(edges),
                parser_failures=len(failures),
            )
    return RepoIngest(
        snapshot=snapshot,
        files=files,
        records=records,
        edges=edges,
        parser_failures=failures,
        mode=config.mode,
        previous_snapshot_id=previous_snapshot_id,
        changed_paths=changed_paths,
        unchanged_paths=unchanged_paths,
        deleted_paths=deleted_paths,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--root", type=Path, default=workspace_root())
    _ = parser.add_argument("--collection", default=config.env_text("PROJECT_CODE_INTELLIGENCE_COLLECTION"))
    _ = parser.add_argument(
        "--profile", default=config.env_text("PROJECT_CODE_INTELLIGENCE_PROFILE", "generic") or "generic"
    )
    _ = parser.add_argument("--repos", default=config.env_text("PROJECT_CODE_INTELLIGENCE_REPOS"))
    _ = parser.add_argument("--max-file-bytes", type=int, default=512 * 1024)
    _ = parser.add_argument("--chunk-chars", type=int, default=2400)
    _ = parser.add_argument("--overlap-lines", type=int, default=6)
    _ = parser.add_argument("--limit-files", type=int)
    _ = parser.add_argument("--progress-every", type=int, default=250)
    _ = parser.add_argument("--dry-run", action="store_true")
    _ = parser.add_argument(
        "--reset-code-intel",
        action="store_true",
        help="Delete code-intelligence data for selected repos. Prompts unless confirmation flag is set.",
    )
    _ = parser.add_argument(
        "--reset-all-code-intel",
        action="store_true",
        help="Delete all code-intelligence data in the configured database. Prompts unless confirmation flag is set.",
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
        default=config.env_int("PROJECT_CODE_INTELLIGENCE_SARIF_MAX_BYTES", 50 * 1024 * 1024, minimum=0),
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
        default=config.env_text("PROJECT_CODE_INTELLIGENCE_MODE", "incremental") or "incremental",
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
        default=config.env_int("PROJECT_CODE_INTELLIGENCE_EMBEDDING_MAX_CHARS", 3000, minimum=1),
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
    return parser


def parse_cli_args(argv: list[str] | None = None) -> CliArgs:
    parsed = build_parser().parse_args(argv, namespace=CliNamespace())
    embedding_endpoint_model = parsed.embedding_endpoint_model
    if (
        embedding_endpoint_model == config.DEFAULT_EMBEDDING_ENDPOINT_MODEL
        and config.env_text("PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL") is None
    ):
        embedding_endpoint_model = config.default_embedding_endpoint_model(endpoint=parsed.embedding_endpoint)
    return CliArgs(
        root=parsed.root,
        collection=parsed.collection,
        profile=parsed.profile,
        repos=parsed.repos,
        max_file_bytes=parsed.max_file_bytes,
        chunk_chars=parsed.chunk_chars,
        overlap_lines=parsed.overlap_lines,
        limit_files=parsed.limit_files,
        progress_every=parsed.progress_every,
        dry_run=parsed.dry_run,
        reset_code_intel=parsed.reset_code_intel,
        reset_all_code_intel=parsed.reset_all_code_intel,
        i_know_this_deletes_code_intel_db=parsed.i_know_this_deletes_code_intel_db,
        reset_only=parsed.reset_only,
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
    )


def validate_non_negative_args(args: CliArgs) -> None:
    checks = {
        "--max-file-bytes": args.max_file_bytes,
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
    if args.reset_code_intel and args.embed_only:
        raise ValueError("--reset-code-intel cannot be combined with --embed-only")
    if args.reset_all_code_intel and not args.reset_code_intel:
        raise ValueError("--reset-all-code-intel requires --reset-code-intel")


def build_ingest_plan(args: CliArgs) -> IngestPlan:
    profile = load_profile(args.profile)
    set_active_profile(profile)
    root = args.root.resolve()
    collection = args.collection or default_collection(root)
    repos = parse_repos(args.repos or ",".join(profile.default_repos))
    embed_types = {item.strip() for item in args.embed_record_types.split(",") if item.strip()}
    sarif_patterns = explicit_sarif_patterns(args.sarif)
    sarif_files = discover_sarif_files(root, repos, sarif_patterns, include_profile=not args.no_profile_sarif)
    embedding_requested = args.embed or args.embed_only
    preembedding_requested = args.embed and not args.no_preembed and code_preembedding_enabled()
    mode = "full" if args.full else args.mode
    if mode not in {"incremental", "full"}:
        raise ValueError("PROJECT_CODE_INTELLIGENCE_MODE must be 'incremental' or 'full'")
    validate_args(args, embedding_requested=embedding_requested)
    return IngestPlan(
        args=args,
        profile=profile,
        root=root,
        collection=collection,
        repos=repos,
        embed_types=embed_types,
        sarif_files=sarif_files,
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


def emit_sarif_discovery(plan: IngestPlan) -> None:
    if plan.sarif_files:
        progress_event(
            "code_intel_sarif_discovered",
            files=[relative_to_or_none(path, plan.root) or str(path) for path in plan.sarif_files],
        )


def confirm_reset_code_intel(
    args: CliArgs, settings: config.DatabaseSettings, collection: str, repos: list[str]
) -> None:
    if not args.reset_code_intel:
        return
    if args.reset_all_code_intel:
        write_stderr("About to delete all project-code-intelligence data in the configured database.")
        write_stderr("Collections/repos: all")
        write_stderr("This permanently deletes all snapshots, records, edges, embeddings, and findings.")
    else:
        repo_list = ", ".join(repos)
        write_stderr(f"About to delete project-code-intelligence data for repo(s): {repo_list}")
        write_stderr(f"Collection: {collection}")
        write_stderr("This permanently deletes snapshots, records, edges, embeddings, and findings for those repos.")
    write_stderr(f"Database target: {settings.display_target()}")
    write_stderr("The schema is untouched.")
    if not args.reset_all_code_intel:
        write_stderr("Other repos are untouched.")
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


def prepare_writable_database(args: CliArgs, *, embedding_requested: bool) -> None:
    if args.dry_run:
        return
    settings = config.DatabaseSettings.from_env()
    if not db.allow_writes(settings):
        raise PermissionError("set PROJECT_CODE_INTELLIGENCE_ALLOW_WRITES=1 to ingest")
    if embedding_requested and not args.embedding_endpoint and not args.llama_embed:
        raise ValueError("set --embedding-endpoint or --llama-embed when --embed is used")
    if embedding_requested and args.embedding_endpoint:
        preflight_embedding_endpoint(args.embedding_endpoint, args.embedding_endpoint_model)
    with db.connect(readonly=False, settings=settings) as conn:
        ensure_schema(conn)
        conn.commit()


def resolve_reset_targets(args: CliArgs) -> tuple[str, list[str]]:
    profile = load_profile(args.profile)
    set_active_profile(profile)
    collection = args.collection or default_collection(args.root.resolve())
    repos = parse_repos(args.repos or ",".join(profile.default_repos))
    return collection, repos


def print_reset_only_report(
    args: CliArgs,
    settings: config.DatabaseSettings,
    collection: str,
    repos: list[str],
    deleted: dict[str, int],
) -> None:
    progress.emit_summary({
        "mode": "reset",
        "dry_run": args.dry_run,
        "reset": args.reset_code_intel and not args.dry_run,
        "database": settings.display_target(),
        "collection": collection,
        "repos": repos,
        "deleted_snapshots": deleted,
    })


def run_reset_only(args: CliArgs) -> int:
    validate_args(args, embedding_requested=False)
    settings = config.DatabaseSettings.from_env()
    collection, repos = resolve_reset_targets(args)
    confirm_reset_code_intel(args, settings, collection, repos)
    prepare_writable_database(args, embedding_requested=False)
    deleted: dict[str, int] = {"all": 0} if args.reset_all_code_intel else dict.fromkeys(repos, 0)
    if not args.dry_run:
        with db.connect(readonly=False, settings=settings) as conn:
            progress_event("code_intel_reset_started", collection=collection, repos=repos)
            if args.reset_all_code_intel:
                deleted = {"all": delete_all_code_intel_data(conn)}
            else:
                deleted = delete_repo_data(conn, collection, repos)
            conn.commit()
            progress_event("code_intel_reset_completed", collection=collection, deleted=deleted)
    print_reset_only_report(args, settings, collection, repos, deleted)
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


def print_embed_only_report(plan: IngestPlan, snapshot_ids: list[int], embedded_records: int | None) -> None:
    args = plan.args
    report: JsonObject = {
        "repos": plan.repos,
        "collection": plan.collection,
        "snapshot_ids": snapshot_ids,
        "mode": "embed-only",
        "profile": profile_context.active_profile.name,
        "embeddings": True,
        "embedding_max_chars": args.embedding_max_chars,
        "metrics": runtime_state.active_metrics.snapshot(),
    }
    if embedded_records is not None:
        report.update({
            "embedded_records": embedded_records,
            "embedded_records_post_insert": embedded_records,
            "embedded_records_total": embedded_records,
        })
    progress.emit_summary(report)


def run_embed_only(plan: IngestPlan) -> int:
    args = plan.args
    snapshot_ids = latest_snapshot_ids(plan.collection, plan.repos)
    if args.dry_run:
        print_embed_only_report(plan, snapshot_ids, None)
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
    print_embed_only_report(plan, snapshot_ids, embedded_records)
    return 0


def previous_repo_states(plan: IngestPlan) -> dict[str, tuple[int | None, dict[str, str], str]]:
    if plan.mode != "incremental":
        return {repo: (None, {}, "full") for repo in plan.repos}
    try:
        with db.connect() as conn:
            return {repo: previous_repo_state(conn, plan.collection, repo) for repo in plan.repos}
    except (db.DatabaseConnectionError, db.OperationalError) as exc:
        progress_event("code_intel_incremental_unavailable", error=str(exc))
        return {repo: (None, {}, "full") for repo in plan.repos}


def previous_repo_state(conn: db.DbConnection, collection: str, repo: str) -> tuple[int | None, dict[str, str], str]:
    previous = latest_snapshot_info(conn, collection, repo)
    metadata = previous.get("metadata") if previous else None
    metadata_obj = metadata if isinstance(metadata, dict) else None
    if previous and snapshot_versions_compatible(metadata_obj):
        previous_id = json_int(previous, "id")
        return previous_id, previous_file_signatures(conn, previous_id), "incremental"
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
    previous_by_repo: dict[str, tuple[int | None, dict[str, str], str]],
) -> list[RepoIngest]:
    ingests: list[RepoIngest] = []
    for repo in plan.repos:
        previous_snapshot_id, previous_signatures, repo_mode = previous_by_repo.get(repo, (None, {}, "full"))
        ingest = ingest_repo(
            RepoIngestConfig(
                root=plan.root,
                repo=repo,
                collection=plan.collection,
                max_file_bytes=plan.args.max_file_bytes,
                max_chars=plan.args.chunk_chars,
                overlap_lines=plan.args.overlap_lines,
                limit_files=plan.args.limit_files,
                progress_every=plan.args.progress_every,
                previous_snapshot_id=previous_snapshot_id,
                previous_signatures=previous_signatures,
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
    sarif_ingest = SarifIngest(runs=[], records_by_repo={}, failures=[])
    if not plan.sarif_files:
        return sarif_ingest
    runtime_state.active_metrics.add_phase_total(len(plan.sarif_files))
    file_by_source_path = {file.source_path: file for ingest in ingests for file in ingest.files}
    sarif_ingest = ingest_sarif(
        SarifIngestContext(
            root=plan.root,
            repos=plan.repos,
            collection=plan.collection,
            file_by_source_path=file_by_source_path,
            max_bytes=plan.args.sarif_max_bytes,
        ),
        plan.sarif_files,
    )
    runtime_state.active_metrics.add_phase_done(len(plan.sarif_files))
    merge_sarif_into_ingests(plan, ingests, sarif_ingest)
    progress_event(
        "code_intel_sarif_parsed",
        files=len(plan.sarif_files),
        runs=len(sarif_ingest.runs),
        findings=sum(len(run.findings) for run in sarif_ingest.runs),
        records=sum(len(records) for records in sarif_ingest.records_by_repo.values()),
        parser_failures=len(sarif_ingest.failures),
    )
    return sarif_ingest


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
    report["sarif_files"] = [relative_to_or_none(path, plan.root) or str(path) for path in plan.sarif_files]
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
    copied_parser_failure_count = copy_unchanged_parser_failures(
        conn,
        previous_snapshot_id=ingest.previous_snapshot_id,
        snapshot=ingest.snapshot,
        snapshot_id=snapshot_id,
        unchanged_paths=ingest.unchanged_paths,
    )
    summary.copied_records += copied_record_count
    summary.copied_edges += copied_edge_count
    summary.copied_parser_failures += copied_parser_failure_count
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
            inserted_records = insert_records(insert_context, ingest.records)
            return inserted_records, 0, 0
        inserted_records, preembedded, skipped = insert_records_with_preembedding(
            insert_context,
            ingest.records,
            preembedding_state,
        )
        summary.preembedded_records += preembedded
        summary.preembedding_skipped += skipped
        return inserted_records, preembedded, skipped
    finally:
        abandon_preembedding(preembedding_state)


def upload_repo_ingest(
    conn: db.DbConnection,
    plan: IngestPlan,
    ingest: RepoIngest,
    summary: DbUploadSummary,
    indexes: StaticSnapshotIndexes,
) -> None:
    snapshot_id = insert_snapshot(conn, ingest.snapshot)
    summary.snapshot_ids.append(snapshot_id)
    indexes.snapshot_ids_by_repo[ingest.snapshot.repo] = snapshot_id
    indexes.snapshot_by_repo[ingest.snapshot.repo] = ingest.snapshot
    file_ids = insert_files(conn, snapshot_id, ingest.files)
    file_hashes = {item.source_path: item.file_sha256 for item in ingest.files}
    insert_context = RecordInsertContext(conn, ingest.snapshot, snapshot_id, file_ids, file_hashes)
    summary.inserted_files += len(file_ids)
    runtime_state.active_metrics.add_phase_done(len(file_ids))
    runtime_state.active_metrics.add("inserted_files", len(file_ids))
    copy_unchanged_data(conn, ingest, snapshot_id, summary)
    inserted_records, _preembedded, _skipped = insert_repo_records(plan, ingest, insert_context, summary)
    summary.inserted_records += inserted_records
    runtime_state.active_metrics.add_phase_done(inserted_records)
    inserted_edges = insert_edges(conn, ingest.snapshot, snapshot_id, ingest.edges)
    summary.inserted_edges += inserted_edges
    runtime_state.active_metrics.add_phase_done(inserted_edges)
    inserted_failures = insert_parser_failures(conn, ingest.snapshot, snapshot_id, ingest.parser_failures)
    summary.inserted_parser_failures += inserted_failures
    runtime_state.active_metrics.add_phase_done(inserted_failures)
    runtime_state.active_metrics.add("inserted_records", inserted_records)
    runtime_state.active_metrics.add("inserted_edges", inserted_edges)
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
            if plan.mode == "full" and not plan.args.no_replace:
                replace_repos(conn, plan.collection, plan.repos)
            indexes = StaticSnapshotIndexes()
            for ingest in ingests:
                upload_repo_ingest(conn, plan, ingest, summary, indexes)
            insert_static_analysis(conn, sarif_ingest, indexes.snapshot_ids_by_repo, indexes.snapshot_by_repo, summary)
            conn.commit()
    finally:
        runtime_state.active_metrics.end_phase("db_upload", "db_upload_seconds")
    return summary


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


def print_ingest_result(plan: IngestPlan, summary: DbUploadSummary, embedded_records: int) -> None:
    progress.emit_summary({
        "repos": plan.repos,
        "collection": plan.collection,
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
        "mode": plan.mode,
        "profile": profile_context.active_profile.name,
        "embeddings": plan.args.embed,
        "embedded_records": embedded_records,
        "embedded_records_post_insert": embedded_records,
        "preembedded_records": summary.preembedded_records,
        "embedded_records_total": summary.preembedded_records + embedded_records,
        "preembedding_skipped": summary.preembedding_skipped,
        "embedding_max_chars": plan.args.embedding_max_chars if plan.args.embed else None,
        "metrics": runtime_state.active_metrics.snapshot(),
    })


def resolve_plan_embedding_model(plan: IngestPlan) -> IngestPlan:
    args = plan.args
    if not plan.embedding_requested or args.dry_run or not args.embedding_endpoint:
        return plan
    resolved_model = resolve_embedding_endpoint_model(args.embedding_endpoint, args.embedding_endpoint_model)
    if resolved_model == args.embedding_endpoint_model:
        return plan
    return replace(plan, args=replace(args, embedding_endpoint_model=resolved_model))


def run_ingest_plan(plan: IngestPlan) -> int:
    emit_sarif_discovery(plan)
    configure_ingest_progress(plan)
    plan = resolve_plan_embedding_model(plan)
    prepare_writable_database(plan.args, embedding_requested=plan.embedding_requested)
    if plan.args.embed_only:
        return run_embed_only(plan)
    ingests, sarif_ingest = scan_plan(plan)
    if plan.args.dry_run:
        print_dry_run_report(plan, ingests, sarif_ingest)
        return 0
    summary = upload_ingests(plan, ingests, sarif_ingest)
    embedded_records = embed_after_upload(plan, summary.snapshot_ids)
    print_ingest_result(plan, summary, embedded_records)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_cli_args(argv)
    if args.reset_only:
        return run_reset_only(args)
    return run_ingest_plan(build_ingest_plan(args))


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
    except EmbeddingEndpointUnavailableError as exc:
        exit_code = 1
        write_stderr(str(exc))
        return 1
    except db.DatabaseConnectionError as exc:
        exit_code = 1
        write_stderr(str(exc))
        return 1
    except (PermissionError, ValueError) as exc:
        exit_code = 1
        write_stderr(str(exc))
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


if __name__ == "__main__":
    raise SystemExit(cli_main())
