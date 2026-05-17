"""SQL scope and filter helpers for MCP code-intelligence tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError
from project_code_intelligence.mcp.protocol import (
    Json,
    QueryParams,
    mcp_max_metadata_bytes,
    optional_bool,
    optional_int,
    optional_text,
    scoped_collection,
)


@dataclass(frozen=True)
class ClauseParams:
    clauses: list[str]
    params: QueryParams


@dataclass(frozen=True)
class StatusFilters:
    snapshots: ClauseParams
    records: ClauseParams
    files: ClauseParams
    edges: ClauseParams
    static_runs: ClauseParams
    static_findings: ClauseParams


def column(alias: str, name: str) -> str:
    return f"{alias}.{name}" if alias else name


def latest_record_snapshot_clause(alias: str) -> str:
    return " ".join([
        column(alias, "snapshot_id"),
        "=",
        "(",
        "SELECT latest_snapshot.id",
        "FROM project_code_intel_snapshots latest_snapshot",
        "WHERE latest_snapshot.collection =",
        column(alias, "collection"),
        "AND latest_snapshot.repo =",
        column(alias, "repo"),
        "ORDER BY latest_snapshot.created_at DESC, latest_snapshot.id DESC",
        "LIMIT 1",
        ")",
    ])


def latest_snapshot_table_clause(alias: str) -> str:
    return " ".join([
        column(alias, "id"),
        "=",
        "(",
        "SELECT latest_snapshot.id",
        "FROM project_code_intel_snapshots latest_snapshot",
        "WHERE latest_snapshot.collection =",
        column(alias, "collection"),
        "AND latest_snapshot.repo =",
        column(alias, "repo"),
        "ORDER BY latest_snapshot.created_at DESC, latest_snapshot.id DESC",
        "LIMIT 1",
        ")",
    ])


def query_with_where(prefix: str, clauses: list[str], suffix: str) -> str:
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    return prefix + "\n" + where + "\n" + suffix


def json_argument(value: object, name: str) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if len(text.encode("utf-8")) > mcp_max_metadata_bytes():
        raise McpProtocolError(f"{name} exceeds PROJECT_CODE_INTELLIGENCE_MCP_MAX_METADATA_BYTES")
    return text


def snapshot_scope(args: Json) -> tuple[str, int | None]:
    snapshot_id = optional_int(args, "snapshot_id")
    include_historical = optional_bool(args, "include_historical")
    if snapshot_id is not None and include_historical:
        raise McpProtocolError("snapshot_id and include_historical cannot both be set")
    if snapshot_id is not None:
        return "snapshot_id", snapshot_id
    if include_historical:
        return "historical", None
    return "latest", None


def snapshot_scope_response(args: Json) -> Json:
    scope, snapshot_id = snapshot_scope(args)
    result: Json = {}
    # "latest" is the default — only echo when the caller explicitly widened the
    # scope (snapshot_id pin or include_historical).
    if scope != "latest":
        result["snapshot_scope"] = scope
    if snapshot_id is not None:
        result["snapshot_id"] = snapshot_id
    return result


def scoped_snapshot_clauses(args: Json, alias: str) -> tuple[list[str], QueryParams]:
    scope, snapshot_id = snapshot_scope(args)
    if snapshot_id is not None:
        return [f"{column(alias, 'snapshot_id')} = %s"], [snapshot_id]
    if scope == "historical":
        return [], []
    return [latest_record_snapshot_clause(alias)], []


def scoped_snapshot_table_clauses(args: Json, alias: str) -> tuple[list[str], QueryParams]:
    scope, snapshot_id = snapshot_scope(args)
    if snapshot_id is not None:
        return [f"{column(alias, 'id')} = %s"], [snapshot_id]
    if scope == "historical":
        return [], []
    return [latest_snapshot_table_clause(alias)], []


def scoped_collection_repo_clauses(args: Json, alias: str) -> tuple[list[str], QueryParams]:
    clauses: list[str] = []
    params: QueryParams = []
    collection = scoped_collection(args)
    if collection:
        clauses.append(f"{column(alias, 'collection')} = %s")
        params.append(collection)
    repo = optional_text(args, "repo")
    if repo:
        clauses.append(f"{column(alias, 'repo')} = %s")
        params.append(repo)
    snapshot_clauses, snapshot_params = scoped_snapshot_clauses(args, alias)
    clauses.extend(snapshot_clauses)
    params.extend(snapshot_params)
    return clauses, params


def scoped_snapshot_table_collection_repo_clauses(args: Json, alias: str) -> tuple[list[str], QueryParams]:
    clauses: list[str] = []
    params: QueryParams = []
    collection = scoped_collection(args)
    if collection:
        clauses.append(f"{column(alias, 'collection')} = %s")
        params.append(collection)
    repo = optional_text(args, "repo")
    if repo:
        clauses.append(f"{column(alias, 'repo')} = %s")
        params.append(repo)
    snapshot_clauses, snapshot_params = scoped_snapshot_table_clauses(args, alias)
    clauses.extend(snapshot_clauses)
    params.extend(snapshot_params)
    return clauses, params


def source_path_prefix_pattern(prefix: str) -> str:
    # Strip a trailing slash so callers can pass either "cmd" or "cmd/" and get the
    # same subtree match. The pattern matches strict descendants only — to match a
    # file at the exact path, use source_path instead.
    normalized = prefix.rstrip("/")
    escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}/%"


def source_path_suffix_pattern(path: str) -> str:
    escaped = path.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%/{escaped}"


def source_path_prefix_suffix_pattern(prefix: str) -> str:
    normalized = prefix.rstrip("/")
    escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%/{escaped}/%"


_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:/")


def normalize_source_path_filter(path: str, arg_name: str = "source_path") -> str:
    normalized = path.strip().replace("\\", "/")
    if normalized.startswith("/") or _WINDOWS_ABSOLUTE_PATH_RE.match(normalized):
        raise McpProtocolError(
            f"{arg_name} must be repo-relative, not absolute; "
            "run code_intel_status to find repo keys and pass repo plus a repo-relative path"
        )
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def repo_relative_source_path_candidates(args: Json, path: str) -> list[str]:
    path = normalize_source_path_filter(path)
    repo = optional_text(args, "repo")
    if not repo or repo == ".":
        return [path]
    normalized_repo = normalize_source_path_filter(repo, "repo")
    if path == normalized_repo or path.startswith(f"{normalized_repo}/{normalized_repo}/"):
        return [path]
    if path.startswith(f"{normalized_repo}/"):
        repo_relative = path.removeprefix(f"{normalized_repo}/")
        candidates = [path]
        if repo_relative.startswith("src/"):
            candidates.append(f"{normalized_repo}/{path}")
        return list(dict.fromkeys(candidates))
    candidates = [path, f"{normalized_repo}/{path}"]
    if path.startswith("src/"):
        candidates.append(f"{normalized_repo}/{normalized_repo}/{path}")
    return list(dict.fromkeys(candidates))


def source_path_column_clauses(args: Json, alias: str, column_name: str) -> tuple[list[str], QueryParams]:
    """Build WHERE clauses for source_path (exact) and source_path_prefix (subtree).

    The two filters are mutually exclusive; passing both raises so callers see a
    clear error instead of silent ANDed narrowing.
    """
    source_path = optional_text(args, "source_path")
    source_path_prefix = optional_text(args, "source_path_prefix")
    if source_path and source_path_prefix:
        raise McpProtocolError("source_path and source_path_prefix are mutually exclusive")
    if source_path:
        candidates = repo_relative_source_path_candidates(args, source_path)
        path_column = column(alias, column_name)
        if len(candidates) == 1:
            if not optional_text(args, "repo"):
                return (
                    [f"({path_column} = %s OR {path_column} LIKE %s ESCAPE '\\')"],
                    [candidates[0], source_path_suffix_pattern(candidates[0])],
                )
            return [f"{path_column} = %s"], [candidates[0]]
        return [f"{path_column} = ANY(%s)"], [candidates]
    if source_path_prefix:
        patterns = [
            source_path_prefix_pattern(candidate)
            for candidate in repo_relative_source_path_candidates(args, source_path_prefix)
        ]
        if not optional_text(args, "repo"):
            patterns.append(source_path_prefix_suffix_pattern(source_path_prefix))
        if len(patterns) > 1:
            path_column = column(alias, column_name)
            pattern_params: QueryParams = list(patterns)
            return (
                ["(" + " OR ".join(f"{path_column} LIKE %s ESCAPE '\\'" for _ in patterns) + ")"],
                pattern_params,
            )
        return (
            [f"{column(alias, column_name)} LIKE %s ESCAPE '\\'"],
            [patterns[0]],
        )
    return [], []


def source_path_clauses(args: Json, alias: str) -> tuple[list[str], QueryParams]:
    return source_path_column_clauses(args, alias, "source_path")


def _metadata_clauses(args: Json, alias: str) -> tuple[list[str], QueryParams]:
    clauses: list[str] = []
    params: QueryParams = []
    metadata_key = optional_text(args, "metadata_key")
    metadata_value = optional_text(args, "metadata_value")
    if metadata_key and metadata_value:
        clauses.append(f"{column(alias, 'metadata')}->>%s = %s")
        params.extend([metadata_key, metadata_value])
    elif metadata_key:
        clauses.append(f"{column(alias, 'metadata')} ? %s")
        params.append(metadata_key)
    metadata_contains = args.get("metadata_contains")
    if metadata_contains is not None:
        if not isinstance(metadata_contains, dict):
            raise McpProtocolTypeError("metadata_contains must be an object")
        clauses.append(f"{column(alias, 'metadata')} @> %s::jsonb")
        params.append(json_argument(metadata_contains, "metadata_contains"))
    return clauses, params


def code_intel_clauses(args: Json, alias: str = "") -> tuple[list[str], QueryParams]:
    clauses = ["TRUE"]
    params: QueryParams = []
    collection = scoped_collection(args)
    if collection:
        clauses.append(f"{column(alias, 'collection')} = %s")
        params.append(collection)
    for name in ("repo", "record_type", "language", "file_role", "content_class", "confidence_kind"):
        value = optional_text(args, name)
        if value:
            clauses.append(f"{column(alias, name)} = %s")
            params.append(value)
    for name in ("symbol", "parent_record_id"):
        value = optional_text(args, name)
        if value:
            clauses.append(f"{column(alias, name)} = %s")
            params.append(value)
    path_clauses, path_params = source_path_clauses(args, alias)
    clauses.extend(path_clauses)
    params.extend(path_params)
    if "is_untracked" in args:
        # is_untracked lives on the joined files table (alias `f`) in record
        # SELECTs; null coalesces to false so the filter behaves on rows where
        # the join didn't match.
        clauses.append("coalesce(f.is_untracked, false) = %s")
        params.append(optional_bool(args, "is_untracked"))
    meta_clauses, meta_params = _metadata_clauses(args, alias)
    clauses.extend(meta_clauses)
    params.extend(meta_params)
    snapshot_clauses, snapshot_params = scoped_snapshot_clauses(args, alias)
    clauses.extend(snapshot_clauses)
    params.extend(snapshot_params)
    return clauses, params


def static_finding_clauses(args: Json) -> tuple[list[str], QueryParams]:
    clauses = ["TRUE"]
    params: QueryParams = []
    collection = scoped_collection(args)
    if collection:
        clauses.append("f.collection = %s")
        params.append(collection)
    for arg_name, db_column in (
        ("repo", "f.repo"),
        ("rule_id", "f.rule_id"),
        ("level", "f.level"),
        ("baseline_state", "f.baseline_state"),
        ("tool", "r.tool_name"),
    ):
        value = optional_text(args, arg_name)
        if value:
            clauses.append(f"{db_column} = %s")
            params.append(value)
    path_clauses, path_params = source_path_column_clauses(args, "f", "primary_source_path")
    clauses.extend(path_clauses)
    params.extend(path_params)
    snapshot_clauses, snapshot_params = scoped_snapshot_clauses(args, "f")
    clauses.extend(snapshot_clauses)
    params.extend(snapshot_params)
    return clauses, params


def status_filters(args: Json) -> StatusFilters:
    snapshot_clauses, snapshot_params = scoped_snapshot_table_collection_repo_clauses(args, "s")
    record_clauses, record_params = scoped_collection_repo_clauses(args, "r")
    file_clauses, file_params = scoped_collection_repo_clauses(args, "f")
    edge_clauses, edge_params = scoped_collection_repo_clauses(args, "e")
    static_run_clauses, static_run_params = scoped_collection_repo_clauses(args, "r")
    static_finding_clauses_for_status, static_finding_params = scoped_collection_repo_clauses(args, "f")
    return StatusFilters(
        snapshots=ClauseParams(snapshot_clauses, snapshot_params),
        records=ClauseParams(record_clauses, record_params),
        files=ClauseParams(file_clauses, file_params),
        edges=ClauseParams(edge_clauses, edge_params),
        static_runs=ClauseParams(static_run_clauses, static_run_params),
        static_findings=ClauseParams(static_finding_clauses_for_status, static_finding_params),
    )
