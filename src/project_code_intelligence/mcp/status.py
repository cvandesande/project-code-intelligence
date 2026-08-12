"""Implementation of the `code_intel_status` MCP tool body.

The handler in `mcp.tools` is a thin wrapper around `load_status_rows` and the
status formatters here. Everything below — snapshot annotation, queryability
rollups, runtime identity, dirty-tree detection — exists to compose that
response. Shared scope-warning helpers live in `mcp.scope`; this module focuses
on status-specific shape and freshness signals.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from project_code_intelligence import config, db, git_utils
from project_code_intelligence.mcp import db as mcp_db
from project_code_intelligence.mcp.filters import StatusFilters, query_with_where, snapshot_scope_response
from project_code_intelligence.mcp.protocol import (
    optional_bool,
    optional_int,
    optional_text,
    scoped_collection,
)
from project_code_intelligence.mcp.scope import empty_repo_scope_warning, empty_snapshot_scope_warning, make_warning
from project_code_intelligence.storage import schema_migration_versions

if TYPE_CHECKING:
    from project_code_intelligence.mcp.protocol import Json
    from project_code_intelligence.models import JsonValue


SERVER_STARTED_AT = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
PACKAGE_NAME = "project-code-intelligence"


_COMPACT_STATUS_SNAPSHOT_KEYS = (
    "id",
    "collection",
    "repo",
    "repo_role",
    "branch",
    "commit_sha",
    "dirty",
    "head_commit",
    "head_matches_snapshot",
    "head_status",
    "head_status_reason",
    "index_age_seconds",
)


def snapshot_scope_warning(conn: db.DbConnection, args: Json) -> Json | None:
    """Return `empty_snapshot_scope` warning when an explicit snapshot_id doesn't exist.

    Replaces the older "raise on unknown snapshot" pattern so unknown snapshots surface as a
    structured warning (parallel to `empty_repo_scope`) rather than a JSON-RPC error. This
    unifies "unknown scope" UX across repo, snapshot_id, and the enum-ish filter dimensions.
    Returns None when snapshot_id wasn't supplied, or when it was and the snapshot exists.
    """
    snapshot_id = optional_int(args, "snapshot_id")
    if snapshot_id is None:
        return None
    row = conn.execute(
        "SELECT 1 FROM project_code_intel_snapshots WHERE id = %s",
        [snapshot_id],
    ).fetchone()
    if row is not None:
        return None
    return empty_snapshot_scope_warning(snapshot_id)


def static_status_rows(conn: db.DbConnection, filters: StatusFilters) -> tuple[list[db.DbRow], list[db.DbRow]]:
    static_runs = []
    static_findings = []
    if mcp_db.table_regclass_exists(conn, "project_code_intel_static_runs"):
        static_runs = conn.execute(
            db.query_sql(
                query_with_where(
                    """
                SELECT r.collection, r.repo, r.tool_name, count(*) AS runs
                FROM project_code_intel_static_runs r
                """,
                    filters.static_runs.clauses,
                    """
                GROUP BY r.collection, r.repo, r.tool_name
                ORDER BY r.collection, r.repo, r.tool_name
                """,
                )
            ),
            filters.static_runs.params,
        ).fetchall()
    if mcp_db.table_regclass_exists(conn, "project_code_intel_static_findings"):
        static_findings = conn.execute(
            db.query_sql(
                query_with_where(
                    """
                SELECT f.collection, f.repo, f.rule_id, f.level, count(*) AS findings
                FROM project_code_intel_static_findings f
                """,
                    filters.static_findings.clauses,
                    """
                GROUP BY f.collection, f.repo, f.rule_id, f.level
                ORDER BY f.collection, f.repo, f.rule_id, f.level
                """,
                )
            ),
            filters.static_findings.params,
        ).fetchall()
    return static_runs, static_findings


def _path_has_repo_suffix(path: Path, repo: str) -> bool:
    repo_parts = tuple(part for part in Path(repo).parts if part not in {"", "."})
    return bool(repo_parts) and tuple(path.parts[-len(repo_parts) :]) == repo_parts


def _snapshot_repo_root_candidates(snapshot: Json) -> list[Path]:
    cwd = Path.cwd()
    repo_value = snapshot.get("repo")
    repo = repo_value if isinstance(repo_value, str) and repo_value else "."
    candidates: list[Path] = []
    metadata = snapshot.get("metadata")
    if isinstance(metadata, dict):
        repo_path = cast("dict[object, object]", metadata).get("repo_path")
        if isinstance(repo_path, str) and repo_path:
            candidates.append(Path(repo_path))
    if repo == "." or _path_has_repo_suffix(cwd, repo):
        candidates.append(cwd)
    if repo != ".":
        candidates.extend((cwd / repo, cwd.parent / repo))
    return list(dict.fromkeys(candidates))


def _snapshot_head_commit(snapshot: Json) -> str | None:
    for candidate in _snapshot_repo_root_candidates(snapshot):
        head = git_utils.run_git(candidate, ["rev-parse", "HEAD"])
        if head:
            return head.strip()
    return None


def _annotate_snapshot_head_status(snapshot: Json, head_commit: str | None) -> None:
    snapshot["head_commit"] = head_commit
    if head_commit is None:
        snapshot["head_matches_snapshot"] = None
        snapshot["head_status"] = "unknown"
        snapshot["head_status_reason"] = "local_repo_unavailable"
        return
    matches = snapshot.get("commit_sha") == head_commit
    snapshot["head_matches_snapshot"] = matches
    snapshot["head_status"] = "current" if matches else "stale"


def annotate_status_snapshots(snapshot_rows: list[db.DbRow]) -> list[Json]:
    now = datetime.datetime.now(datetime.timezone.utc)
    snapshots: list[Json] = []
    for snap in snapshot_rows:
        snap_dict: Json = cast("Json", dict(snap))
        created = snap_dict.get("created_at")
        if created is not None and isinstance(created, datetime.datetime):
            snap_dict["index_age_seconds"] = int((now - created).total_seconds())
        _annotate_snapshot_head_status(snap_dict, _snapshot_head_commit(snap_dict))
        snapshots.append(snap_dict)
    return snapshots


def compact_status_snapshots(snapshots: list[Json], *, omit_collection: bool, omit_repo: bool) -> list[Json]:
    compact: list[Json] = []
    for snapshot in snapshots:
        item: Json = {}
        for key in _COMPACT_STATUS_SNAPSHOT_KEYS:
            if omit_collection and key == "collection":
                continue
            if omit_repo and key == "repo":
                continue
            if key == "head_commit" and snapshot.get("head_commit") == snapshot.get("commit_sha"):
                continue
            if key in snapshot:
                item[key] = snapshot[key]
        compact.append(item)
    return compact


def status_rows_for_response(rows: list[db.DbRow], *, omit_collection: bool, omit_repo: bool) -> list[Json]:
    if not omit_collection and not omit_repo:
        return [{key: cast("JsonValue", value) for key, value in row.items()} for row in rows]
    result: list[Json] = []
    for row in rows:
        item: Json = {}
        for key, value in row.items():
            if omit_collection and key == "collection":
                continue
            if omit_repo and key == "repo":
                continue
            item[key] = cast("JsonValue", value)
        result.append(item)
    return result


def status_json_rows_for_response(rows: list[Json], *, omit_collection: bool, omit_repo: bool) -> list[Json]:
    if not omit_collection and not omit_repo:
        return rows
    result: list[Json] = []
    for row in rows:
        item = dict(row)
        if omit_collection:
            _ = item.pop("collection", None)
        if omit_repo:
            _ = item.pop("repo", None)
        result.append(item)
    return result


def _copy_snapshot_warning_fields(warning: Json, snapshot: Json, keys: tuple[str, ...]) -> None:
    for key in keys:
        value = snapshot.get(key)
        if value is not None:
            if key == "head_commit" and value == snapshot.get("commit_sha"):
                continue
            warning[key] = cast("JsonValue", value)


def _snapshot_dirty_paths_count(snapshot: Json) -> int | None:
    metadata = snapshot.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    dirty_paths = metadata.get("dirty_paths")
    if not isinstance(dirty_paths, list):
        return None
    return sum(1 for item in dirty_paths if isinstance(item, str))


def status_snapshot_warnings(snapshots: list[Json]) -> list[Json]:
    warnings: list[Json] = []
    for snapshot in snapshots:
        status = snapshot.get("head_status")
        warning: Json | None = None
        if status == "stale":
            warning = make_warning("snapshot_stale", message="snapshot is stale; verify with local source")
        elif status == "unknown":
            warning = make_warning(
                "snapshot_freshness_unknown",
                message="snapshot freshness could not be checked against local source",
            )
        if warning is not None:
            _copy_snapshot_warning_fields(
                warning,
                snapshot,
                ("id", "collection", "repo", "commit_sha", "head_commit", "head_status_reason"),
            )
            warnings.append(warning)
        if snapshot.get("dirty") is True:
            dirty_warning = make_warning(
                "snapshot_dirty",
                message="snapshot was indexed from a dirty working tree; verify dirty paths against local source",
                dirty=True,
                dirty_paths_count=_snapshot_dirty_paths_count(snapshot),
            )
            _copy_snapshot_warning_fields(
                dirty_warning,
                snapshot,
                ("id", "collection", "repo", "commit_sha", "head_commit", "head_status"),
            )
            warnings.append(dirty_warning)
    return warnings


def status_repo_not_found(args: Json, rows: StatusRows) -> bool:
    return bool(
        optional_text(args, "repo") and not rows.snapshots and not rows.records and not rows.files and not rows.edges
    )


def status_scope_warnings(args: Json, rows: StatusRows, *, missing_snapshot_warning: Json | None = None) -> list[Json]:
    warnings: list[Json] = []
    repo = optional_text(args, "repo")
    if repo and status_repo_not_found(args, rows):
        warnings.append(empty_repo_scope_warning(repo))
    if missing_snapshot_warning is not None:
        warnings.append(missing_snapshot_warning)
    return warnings


# Precomputed SELECT + IS-NOT-NULL fragments keyed by the file column we're enumerating.
# Stored as literals so we never interpolate the column name into a SQL string at runtime; this
# keeps Bandit / ruff S608 happy (the lints don't trust f-strings even when the input is bounded).
_FILE_DIMENSION_SELECTS: dict[str, str] = {
    "language": "SELECT DISTINCT f.language AS value FROM project_code_intel_files f",
    "file_role": "SELECT DISTINCT f.file_role AS value FROM project_code_intel_files f",
    "content_class": "SELECT DISTINCT f.content_class AS value FROM project_code_intel_files f",
}
_FILE_DIMENSION_NONEMPTY_CLAUSES: dict[str, tuple[str, str]] = {
    "language": ("f.language IS NOT NULL", "f.language <> ''"),
    "file_role": ("f.file_role IS NOT NULL", "f.file_role <> ''"),
    "content_class": ("f.content_class IS NOT NULL", "f.content_class <> ''"),
}


def _status_file_dimensions(conn: db.DbConnection, filters: StatusFilters) -> dict[str, list[str]]:
    """Return sorted distinct values for the enum-ish file columns the queryability section exposes.

    Used by the empty_<dim>_scope warnings to point callers at the authoritative valid-value list.
    """
    out: dict[str, list[str]] = {}
    for column_name, select in _FILE_DIMENSION_SELECTS.items():
        extra_clauses = [*filters.files.clauses, *_FILE_DIMENSION_NONEMPTY_CLAUSES[column_name]]
        rows = conn.execute(
            db.query_sql(query_with_where(select, extra_clauses, "ORDER BY value")),
            filters.files.params,
        ).fetchall()
        out[column_name] = [str(row["value"]) for row in rows]
    return out


def _status_file_breakdowns(
    conn: db.DbConnection, filters: StatusFilters, directory_depth: int
) -> dict[str, list[db.DbRow]]:
    language = conn.execute(
        db.query_sql(
            query_with_where(
                """
            SELECT f.language, count(*) AS files
            FROM project_code_intel_files f
            """,
                filters.files.clauses,
                """
            GROUP BY f.language
            ORDER BY files DESC, f.language
            """,
            )
        ),
        filters.files.params,
    ).fetchall()
    # Group by the first `directory_depth` path segments (excluding the filename).
    # At depth=1 a file at `crates/foo/lib.rs` rolls up to `crates`; at depth=2 to
    # `crates/foo`. Files in the repo root produce '.'.
    directory = conn.execute(
        db.query_sql(
            query_with_where(
                """
            SELECT
                CASE
                    WHEN array_length(string_to_array(f.source_path, '/'), 1) <= 1 THEN '.'
                    ELSE array_to_string(
                        (string_to_array(f.source_path, '/'))[
                            1:LEAST(%s, array_length(string_to_array(f.source_path, '/'), 1) - 1)
                        ],
                        '/'
                    )
                END AS directory,
                count(*) AS files
            FROM project_code_intel_files f
            """,
                filters.files.clauses,
                """
            GROUP BY directory
            ORDER BY files DESC, directory
            LIMIT 100
            """,
            )
        ),
        [directory_depth, *filters.files.params],
    ).fetchall()
    return {"language": language, "directory": directory}


def positive_int(value: object) -> int:
    match value:
        case bool():
            return 0
        case int():
            return max(0, value)
        case float():
            return max(0, int(value))
        case str() if value.isdecimal():
            return int(value)
        case _:
            return 0


def _snapshot_embed_record_types(snapshots: list[Json]) -> set[str]:
    record_types: set[str] = set()
    for snapshot in snapshots:
        match snapshot.get("metadata"):
            case {"embed_record_types": list() as values}:
                record_types.update(item for item in values if isinstance(item, str) and item)
            case _:
                pass
    return record_types


def status_queryability(
    snapshots: list[Json],
    records_by_type: list[db.DbRow],
    edges_by_type: list[db.DbRow],
    file_dimensions: dict[str, list[str]] | None,
    *,
    include_details: bool,
) -> Json:
    text_record_types = sorted({
        str(row.get("record_type"))
        for row in records_by_type
        if row.get("record_type") and positive_int(row.get("count"))
    })
    semantic_record_types = sorted({
        str(row.get("record_type"))
        for row in records_by_type
        if row.get("record_type") and positive_int(row.get("embedded_records"))
    })
    configured_embed_record_types = sorted(_snapshot_embed_record_types(snapshots))
    empty_embed_record_types = sorted(set(configured_embed_record_types) - set(semantic_record_types))
    edge_types = sorted({
        str(row.get("edge_type")) for row in edges_by_type if row.get("edge_type") and positive_int(row.get("edges"))
    })
    # Compact surface keeps only counts that suggest an action. `configured_embed_record_type_count`
    # is purely descriptive (it only describes what was configured at indexing time) and moves
    # behind `include_queryability`. `empty_embed_record_type_count` IS actionable — a non-zero
    # value flags a freshness/coverage gap — so we keep it in compact, but only when non-zero
    # (zero is silence, not signal). Both counts return unconditionally in the detailed surface
    # so callers that introspect queryability programmatically see a stable shape.
    empty_embed_record_type_count = len(empty_embed_record_types)
    queryability: dict[str, JsonValue] = {
        "text_record_type_count": len(text_record_types),
        "semantic_record_type_count": len(semantic_record_types),
        "text_only_record_type_count": len(set(text_record_types) - set(semantic_record_types)),
        "edge_type_count": len(edge_types),
        "has_text": bool(text_record_types),
        "has_semantic": bool(semantic_record_types),
        "has_edges": bool(edge_types),
    }
    if empty_embed_record_type_count > 0 or include_details:
        queryability["empty_embed_record_type_count"] = empty_embed_record_type_count
    if include_details:
        queryability["configured_embed_record_type_count"] = len(configured_embed_record_types)
    if file_dimensions is not None:
        languages = list(file_dimensions.get("language", []))
        file_roles = list(file_dimensions.get("file_role", []))
        content_classes = list(file_dimensions.get("content_class", []))
        queryability.update({
            "language_count": len(languages),
            "file_role_count": len(file_roles),
            "content_class_count": len(content_classes),
        })
    else:
        languages, file_roles, content_classes = [], [], []
    if include_details:
        queryability.update({
            "text_record_types": text_record_types,
            "semantic_record_types": semantic_record_types,
            "text_only_record_types": sorted(set(text_record_types) - set(semantic_record_types)),
            "configured_embed_record_types": configured_embed_record_types,
            "empty_embed_record_types": empty_embed_record_types,
            "edge_types": edge_types,
        })
        if file_dimensions is not None:
            queryability.update({
                "languages": languages,
                "file_roles": file_roles,
                "content_classes": content_classes,
            })
    return queryability


@dataclass(frozen=True)
class StatusIncludeFlags:
    verbose: bool
    snapshots: bool
    record_types: bool
    queryability: bool
    breakdowns: bool
    static_summary: bool
    runtime: bool


@dataclass(frozen=True)
class StatusRows:
    schema_versions: list[str]
    snapshots: list[Json]
    records: list[db.DbRow]
    records_by_type: list[db.DbRow]
    files: list[db.DbRow]
    edges: list[db.DbRow]
    edge_types: list[db.DbRow]
    breakdowns: dict[str, list[db.DbRow]] | None
    static_rows: tuple[list[db.DbRow], list[db.DbRow]] | None
    file_dimensions: dict[str, list[str]] | None = None


def status_include_flags(args: Json) -> StatusIncludeFlags:
    verbose = optional_bool(args, "verbose") or False
    return StatusIncludeFlags(
        verbose=verbose,
        snapshots=verbose or (optional_bool(args, "include_snapshots") or False),
        record_types=verbose or (optional_bool(args, "include_record_types") or False),
        queryability=verbose or (optional_bool(args, "include_queryability") or False),
        breakdowns=verbose or (optional_bool(args, "include_breakdowns") or False),
        static_summary=verbose or (optional_bool(args, "include_static_summary") or False),
        runtime=verbose or (optional_bool(args, "include_runtime") or False),
    )


def status_scope_response(args: Json) -> tuple[Json, bool]:
    result = snapshot_scope_response(args)
    collection = scoped_collection(args)
    if collection:
        result["collection"] = collection
    repo = optional_text(args, "repo")
    if repo:
        result["repo"] = repo
    return result, collection is not None


def _package_version() -> str | None:
    try:
        return importlib.metadata.version(PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_git_root(module_path: Path) -> Path | None:
    """Nearest ancestor of ``module_path`` containing .git (dir or worktree file). Shared
    with hooks/similar.py, which resolves an edited file to its enclosing repo."""
    for candidate in (module_path.parent, *module_path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _source_git_commit(module_path: Path) -> str | None:
    git_root = source_git_root(module_path)
    if git_root is None:
        return None
    commit = git_utils.run_git(git_root, ["rev-parse", "HEAD"])
    return commit.strip() if commit else None


def _database_runtime_identity() -> Json:
    settings = db.inferred_database_role_settings(config.DatabaseSettings.from_env(role="mcp"), "ro")
    target = settings.display_target()
    return {
        "target": target,
        "fingerprint": hashlib.sha256(target.encode("utf-8")).hexdigest()[:16],
        "dsn_source": settings.dsn_source,
        "user_source": settings.dsn_user_source if settings.dsn_user else "PCI_PG_USER",
        "password_source": settings.dsn_auth_source if settings.dsn_password else "PCI_PG_PASS",
        "password_set": bool(settings.dsn_password or settings.password),
        "database_inferred": settings.database_inferred,
        "scope_path": str(config.configured_database_scope_path()),
    }


def server_runtime_identity() -> Json:
    module_path = Path(__file__).resolve(strict=False)
    user_config_path = config.pci_index_user_config_path()
    user_config_exists = user_config_path.exists() if user_config_path is not None else False
    return {
        "package": {
            "name": PACKAGE_NAME,
            "version": _package_version(),
            "module_path": str(module_path),
            "source_git_commit": _source_git_commit(module_path),
        },
        "process": {
            "pid": os.getpid(),
            "executable": sys.executable,
            "started_at": SERVER_STARTED_AT,
            "cwd": str(Path.cwd()),
        },
        "database": _database_runtime_identity(),
        "config": {
            "user_config_path": str(user_config_path) if user_config_path is not None else None,
            "user_config_exists": user_config_exists,
        },
    }


def load_status_rows(
    conn: db.DbConnection,
    filters: StatusFilters,
    includes: StatusIncludeFlags,
    directory_depth: int,
) -> StatusRows:
    schema_versions = (
        schema_migration_versions(conn)
        if mcp_db.table_regclass_exists(conn, "project_code_intel_schema_migrations")
        else []
    )
    snapshot_rows = conn.execute(
        db.query_sql(
            query_with_where(
                """
            SELECT s.id, s.collection, s.repo, s.repo_role, s.branch, s.commit_sha,
                   s.tree_sha, s.dirty, s.metadata, s.created_at
            FROM project_code_intel_snapshots s
""",
                filters.snapshots.clauses,
                """
            ORDER BY s.created_at DESC, s.collection, s.repo
            LIMIT %s
            """,
            )
        ),
        [*filters.snapshots.params, mcp_db.mcp_max_status_rows()],
    ).fetchall()
    snapshots = annotate_status_snapshots(snapshot_rows)
    records = conn.execute(
        db.query_sql(
            query_with_where(
                """
            SELECT r.collection, r.repo, count(*) AS records, count(r.embedding) AS embedded_records
            FROM project_code_intel_records r
            """,
                filters.records.clauses,
                """
            GROUP BY r.collection, r.repo
            ORDER BY r.collection, r.repo
            """,
            )
        ),
        filters.records.params,
    ).fetchall()
    records_by_type = conn.execute(
        db.query_sql(
            query_with_where(
                """
            SELECT r.collection, r.repo, r.record_type,
                   count(*) AS count,
                   count(r.embedding) AS embedded_records
            FROM project_code_intel_records r
            """,
                filters.records.clauses,
                """
            GROUP BY r.collection, r.repo, r.record_type
            ORDER BY r.collection, r.repo, r.record_type
            """,
            )
        ),
        filters.records.params,
    ).fetchall()
    files = conn.execute(
        db.query_sql(
            query_with_where(
                """
            SELECT f.collection, f.repo, count(*) AS files,
                   count(*) FILTER (WHERE f.skipped_reason IS NOT NULL) AS skipped_files,
                   count(*) FILTER (WHERE f.is_untracked) AS untracked_files,
                   count(*) FILTER (WHERE f.indexed_dirty AND NOT f.is_untracked) AS dirty_files
            FROM project_code_intel_files f
            """,
                filters.files.clauses,
                """
            GROUP BY f.collection, f.repo
            ORDER BY f.collection, f.repo
            """,
            )
        ),
        filters.files.params,
    ).fetchall()
    edges = conn.execute(
        db.query_sql(
            query_with_where(
                """
            SELECT e.collection, e.repo, count(*) AS edges
            FROM project_code_intel_edges e
            """,
                filters.edges.clauses,
                """
            GROUP BY e.collection, e.repo
            ORDER BY e.collection, e.repo
            """,
            )
        ),
        filters.edges.params,
    ).fetchall()
    edge_types = conn.execute(
        db.query_sql(
            query_with_where(
                """
            SELECT e.edge_type, count(*) AS edges
            FROM project_code_intel_edges e
            """,
                filters.edges.clauses,
                """
            GROUP BY e.edge_type
            ORDER BY e.edge_type
            """,
            )
        ),
        filters.edges.params,
    ).fetchall()
    return StatusRows(
        schema_versions=schema_versions,
        snapshots=snapshots,
        records=records,
        records_by_type=records_by_type,
        files=files,
        edges=edges,
        edge_types=edge_types,
        breakdowns=_status_file_breakdowns(conn, filters, directory_depth) if includes.breakdowns else None,
        static_rows=static_status_rows(conn, filters) if includes.static_summary else None,
        file_dimensions=_status_file_dimensions(conn, filters) if includes.queryability else None,
    )
