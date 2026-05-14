"""MCP tool handlers for the code-intelligence database."""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from project_code_intelligence import config, db, embeddings, git_utils
from project_code_intelligence.embedding import llama
from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError
from project_code_intelligence.mcp import db as mcp_db
from project_code_intelligence.mcp.filters import (
    StatusFilters,
    code_intel_clauses,
    query_with_where,
    scoped_collection_repo_clauses,
    scoped_snapshot_clauses,
    snapshot_scope_response,
    static_finding_clauses,
    status_filters,
)
from project_code_intelligence.mcp.protocol import (
    Json,
    QueryParams,
    mcp_max_record_content_chars,
    ok,
    optional_bool,
    optional_text,
    require_int,
    scoped_collection,
)
from project_code_intelligence.mcp.tool_catalog import TOOL_DEFINITIONS, ToolDefinition
from project_code_intelligence.storage import row_int, schema_migration_versions

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonValue


STATIC_FINDING_COMPACT_KEYS = (
    "id",
    "snapshot_id",
    "collection",
    "repo",
    "commit_sha",
    "finding_key",
    "rule_id",
    "level",
    "kind",
    "message",
    "baseline_state",
    "primary_source_path",
    "primary_uri",
    "line_start",
    "line_end",
    "column_start",
    "column_end",
    "tool_name",
    "tool_version",
    "automation_id",
    "sarif_path",
    "created_at",
)

STATIC_RULE_COMPACT_KEYS = (
    "id",
    "rule_id",
    "name",
    "short_description",
    "full_description",
    "default_level",
    "help_uri",
)

SearchQueryMode: TypeAlias = Literal["auto", "websearch", "all_terms", "any_terms"]
SearchQueryStrategy: TypeAlias = Literal[
    "list",
    "websearch",
    "all_terms",
    "all_terms_fallback",
    "any_terms",
    "any_terms_fallback",
]

SEARCH_TERM_RE = re.compile(r"[A-Za-z0-9_./:+@-]+")
SEARCH_OPERATOR_WORDS = frozenset({"and", "or", "not"})

CODE_INTEL_RECORD_SELECT_LIST = """
            SELECT r.id, r.snapshot_id, r.collection, r.repo, r.repo_role, r.branch,
                   r.commit_sha, r.tree_sha, r.source_path, r.language, r.file_role,
                   r.content_class, r.record_type, r.record_id, r.parent_record_id,
                   r.title, r.summary, r.line_start, r.line_end, r.symbol,
                   r.symbol_kind, r.confidence_kind, r.confidence, r.tool,
                   r.rule_id, r.severity, r.updated_at,
                   r.embedding IS NOT NULL AS has_embedding,
                   NULL::real AS rank,
                   coalesce(f.is_untracked, false) AS is_untracked,
                   coalesce(f.indexed_dirty, false) AS indexed_dirty,
                   left(r.display_content, 800) AS snippet_raw
            FROM project_code_intel_records r
            LEFT JOIN project_code_intel_files f ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
            """

CODE_INTEL_RECORD_SELECT_WEBSEARCH = """
            SELECT r.id, r.snapshot_id, r.collection, r.repo, r.repo_role, r.branch,
                   r.commit_sha, r.tree_sha, r.source_path, r.language, r.file_role,
                   r.content_class, r.record_type, r.record_id, r.parent_record_id,
                   r.title, r.summary, r.line_start, r.line_end, r.symbol,
                   r.symbol_kind, r.confidence_kind, r.confidence, r.tool,
                   r.rule_id, r.severity, r.updated_at,
                   r.embedding IS NOT NULL AS has_embedding,
                   ts_rank_cd(r.search_document, websearch_to_tsquery('english', %s)) AS rank,
                   coalesce(f.is_untracked, false) AS is_untracked,
                   coalesce(f.indexed_dirty, false) AS indexed_dirty,
                   left(r.display_content, 800) AS snippet_raw
            FROM project_code_intel_records r
            LEFT JOIN project_code_intel_files f ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
            """

CODE_INTEL_RECORD_SELECT_TERMS = """
            SELECT r.id, r.snapshot_id, r.collection, r.repo, r.repo_role, r.branch,
                   r.commit_sha, r.tree_sha, r.source_path, r.language, r.file_role,
                   r.content_class, r.record_type, r.record_id, r.parent_record_id,
                   r.title, r.summary, r.line_start, r.line_end, r.symbol,
                   r.symbol_kind, r.confidence_kind, r.confidence, r.tool,
                   r.rule_id, r.severity, r.updated_at,
                   r.embedding IS NOT NULL AS has_embedding,
                   (
                       SELECT count(*)::real
                       FROM unnest(%s::text[]) AS search_terms(pattern)
                       WHERE concat_ws(
                           ' ',
                           r.title,
                           r.summary,
                           r.symbol,
                           r.source_path,
                           r.record_id,
                           r.display_content
                       ) ILIKE search_terms.pattern ESCAPE '\\'
                   ) AS rank,
                   coalesce(f.is_untracked, false) AS is_untracked,
                   coalesce(f.indexed_dirty, false) AS indexed_dirty,
                   left(r.display_content, 800) AS snippet_raw
            FROM project_code_intel_records r
            LEFT JOIN project_code_intel_files f ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
            """

ALL_TERMS_SEARCH_CLAUSE = """
            NOT EXISTS (
                SELECT 1
                FROM unnest(%s::text[]) AS search_terms(pattern)
                WHERE NOT (
                    concat_ws(
                        ' ',
                        r.title,
                        r.summary,
                        r.symbol,
                        r.source_path,
                        r.record_id,
                        r.display_content
                    ) ILIKE search_terms.pattern ESCAPE '\\'
                )
            )
            """

ANY_TERMS_SEARCH_CLAUSE = """
            EXISTS (
                SELECT 1
                FROM unnest(%s::text[]) AS search_terms(pattern)
                WHERE concat_ws(
                    ' ',
                    r.title,
                    r.summary,
                    r.symbol,
                    r.source_path,
                    r.record_id,
                    r.display_content
                ) ILIKE search_terms.pattern ESCAPE '\\'
            )
            """


_SNIPPET_FENCE_RE = re.compile(r"`{3,}[^\n]*\n")

# Fields stripped in compact mode — per-result snapshot/git/repo metadata that is
# constant across all results in a single-snapshot query and redundant with the
# response envelope. Verbose mode (verbose=true) returns them.
_COMPACT_RECORD_STRIP = frozenset({
    "snapshot_id",
    "collection",
    "repo",
    "repo_role",
    "branch",
    "commit_sha",
    "tree_sha",
    "updated_at",
    "record_id",
    "confidence",
    "tool",
    "rule_id",
    "severity",
})
_COMPACT_EDGE_STRIP = frozenset({"snapshot_id", "collection", "repo", "commit_sha"})


def _extract_snippet(raw: str | None) -> str | None:
    """Return the first ~300 chars of code body from a display_content prefix."""
    if not raw:
        return None
    m = _SNIPPET_FENCE_RE.search(raw)
    if m:
        code = raw[m.end() :]
        return code[:300].rstrip() or None
    return None


def _compact_record(row: db.DbRow) -> dict[str, object]:
    snippet = _extract_snippet(row.get("snippet_raw"))  # type: ignore[arg-type]
    out = {k: v for k, v in row.items() if k not in _COMPACT_RECORD_STRIP and k != "snippet_raw"}
    if snippet:
        out["snippet"] = snippet
    return out


def _verbose_record(row: db.DbRow) -> dict[str, object]:
    snippet = _extract_snippet(row.get("snippet_raw"))  # type: ignore[arg-type]
    out = {k: v for k, v in row.items() if k != "snippet_raw"}
    if snippet:
        out["snippet"] = snippet
    return out


def _format_records(rows: list[db.DbRow], *, verbose: bool) -> list[dict[str, object]]:
    fmt = _verbose_record if verbose else _compact_record
    return [fmt(row) for row in rows]


def _format_edges(rows: list[db.DbRow], *, verbose: bool) -> list[dict[str, object]]:
    if verbose:
        return [dict(row) for row in rows]
    return [{k: v for k, v in row.items() if k not in _COMPACT_EDGE_STRIP} for row in rows]


@dataclass(frozen=True)
class TextSearchPlan:
    query: str | None
    strategy: SearchQueryStrategy
    terms: tuple[str, ...]
    limit: int


def search_query_mode(args: Json) -> SearchQueryMode:
    value = optional_text(args, "query_mode") or "auto"
    if value in {"auto", "websearch", "all_terms", "any_terms"}:
        return cast("SearchQueryMode", value)
    raise McpProtocolError("query_mode must be one of: auto, websearch, all_terms, any_terms")


def search_terms(query: str) -> list[str]:
    terms: list[str] = []
    for match in SEARCH_TERM_RE.finditer(query):
        term = match.group(0)
        if term.lower() in SEARCH_OPERATOR_WORDS:
            continue
        if term not in terms:
            terms.append(term)
    return terms[:16]


def like_pattern_for_term(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def run_text_search_query(
    conn: db.DbConnection,
    args: Json,
    plan: TextSearchPlan,
) -> list[db.DbRow]:
    clauses, filter_params = code_intel_clauses(args, "r")
    if plan.query and plan.strategy == "websearch":
        clauses.append("r.search_document @@ websearch_to_tsquery('english', %s)")
        query_sql = query_with_where(
            CODE_INTEL_RECORD_SELECT_WEBSEARCH,
            clauses,
            """
            ORDER BY rank DESC, r.updated_at DESC
            LIMIT %s
            """,
        )
        params: QueryParams = [plan.query, *filter_params, plan.query, plan.limit]
    elif plan.query and plan.strategy in {"all_terms", "all_terms_fallback", "any_terms", "any_terms_fallback"}:
        require_all = plan.strategy in {"all_terms", "all_terms_fallback"}
        patterns = [like_pattern_for_term(term) for term in plan.terms]
        clauses.append(ALL_TERMS_SEARCH_CLAUSE if require_all else ANY_TERMS_SEARCH_CLAUSE)
        query_sql = query_with_where(
            CODE_INTEL_RECORD_SELECT_TERMS,
            clauses,
            """
            ORDER BY rank DESC, r.updated_at DESC
            LIMIT %s
            """,
        )
        params = [patterns, *filter_params, patterns, plan.limit]
    else:
        query_sql = query_with_where(
            CODE_INTEL_RECORD_SELECT_LIST,
            clauses,
            """
            ORDER BY r.updated_at DESC
            LIMIT %s
            """,
        )
        params = [*filter_params, plan.limit]
    return conn.execute(db.query_sql(query_sql), params).fetchall()


def code_intel_tables_exist(conn: db.DbConnection) -> bool:
    return mcp_db.code_intel_tables_exist(conn)


def table_regclass_exists(conn: db.DbConnection, table: str) -> bool:
    return mcp_db.table_regclass_exists(conn, table)


def static_status_rows(conn: db.DbConnection, filters: StatusFilters) -> tuple[list[db.DbRow], list[db.DbRow]]:
    static_runs = []
    static_findings = []
    if table_regclass_exists(conn, "project_code_intel_static_runs"):
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
    if table_regclass_exists(conn, "project_code_intel_static_findings"):
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


def tool_code_intel_status(args: Json) -> Json:
    filters = status_filters(args)
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"schema_present": False})
        schema_versions = (
            schema_migration_versions(conn)
            if table_regclass_exists(conn, "project_code_intel_schema_migrations")
            else []
        )
        head_commit = git_utils.run_git(Path.cwd(), ["rev-parse", "HEAD"])
        now = datetime.datetime.now(datetime.timezone.utc)
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
        snapshots: list[Json] = []
        for snap in snapshot_rows:
            snap_dict: Json = cast("Json", dict(snap))
            created = snap_dict.get("created_at")
            if created is not None and isinstance(created, datetime.datetime):
                snap_dict["index_age_seconds"] = int((now - created).total_seconds())
            snap_dict["head_commit"] = head_commit
            snap_dict["head_matches_snapshot"] = (
                head_commit is not None and snap_dict.get("commit_sha") == head_commit.strip()
            )
            snap_meta = snap_dict.get("metadata")
            if isinstance(snap_meta, dict) and "embed_record_types" in snap_meta:
                snap_dict["embed_record_types"] = snap_meta["embed_record_types"]
            snapshots.append(snap_dict)
        counts = conn.execute(
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
        by_type = conn.execute(
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
        static_runs, static_findings = static_status_rows(conn, filters)
    return ok({
        "schema_present": True,
        "schema_versions": schema_versions,
        **snapshot_scope_response(args),
        "snapshots": snapshots,
        "files": files,
        "records": counts,
        "records_by_type": by_type,
        "edges": edges,
        "static_runs": static_runs,
        "static_findings": static_findings,
    })


def tool_search_code_intel_text(args: Json) -> Json:
    query = optional_text(args, "query")
    query_mode = search_query_mode(args)
    terms = tuple(search_terms(query)) if query else ()
    limit = require_int(args, "limit", 10, 1, 50)
    strategy: SearchQueryStrategy = "list"
    fallback_reason: str | None = None
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        if not query:
            rows = run_text_search_query(conn, args, TextSearchPlan(query, strategy, terms, limit))
        elif query_mode == "websearch":
            strategy = "websearch"
            rows = run_text_search_query(conn, args, TextSearchPlan(query, strategy, terms, limit))
        elif query_mode == "all_terms":
            strategy = "all_terms"
            rows = run_text_search_query(conn, args, TextSearchPlan(query, strategy, terms, limit))
        elif query_mode == "any_terms":
            strategy = "any_terms"
            rows = run_text_search_query(conn, args, TextSearchPlan(query, strategy, terms, limit))
        else:
            strategy = "websearch"
            rows = run_text_search_query(conn, args, TextSearchPlan(query, strategy, terms, limit))
            if not rows and len(terms) > 1:
                fallback_reason = "websearch returned no results for a multi-term query"
                strategy = "all_terms_fallback"
                rows = run_text_search_query(conn, args, TextSearchPlan(query, strategy, terms, limit))
            if not rows and len(terms) > 1:
                strategy = "any_terms_fallback"
                rows = run_text_search_query(conn, args, TextSearchPlan(query, strategy, terms, limit))
    verbose = optional_bool(args, "verbose") or False
    response: Json = {
        "query": query,
        "query_mode": query_mode,
        "query_strategy": strategy,
        **snapshot_scope_response(args),
        "results": cast("JsonValue", _format_records(rows, verbose=verbose)),
    }
    if terms:
        response["terms"] = list(terms)
    if fallback_reason:
        response["fallback_reason"] = fallback_reason
    return ok(response)


def vector_literal_dimensions(vector: str) -> int:
    inner = vector.strip().removeprefix("[").removesuffix("]").strip()
    return 0 if not inner else inner.count(",") + 1


def semantic_search_embedding_error(endpoint: str, exc: BaseException) -> McpProtocolError:
    return McpProtocolError(
        "semantic search requires an embedding endpoint because the MCP server "
        "must embed the query with a model compatible with the indexed record embeddings. "
        f"The configured endpoint is unavailable: {endpoint}. "
        "Start one of the local embedding profiles shown by pci-doctor, or set "
        "PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT to a trusted OpenAI-compatible "
        f"embedding provider. Detail: {exc}"
    )


def query_embedding(query: str) -> tuple[str, int]:
    endpoint = config.default_embedding_endpoint(local_default=True)
    if endpoint:
        try:
            model = config.default_embedding_endpoint_model(endpoint=endpoint)
            model = embeddings.resolve_embedding_endpoint_model(endpoint, model)
            vectors = embeddings.embed_with_endpoint(endpoint, [query], model, track_metrics=False)
        except embeddings.EmbeddingEndpointUnavailableError as exc:
            raise semantic_search_embedding_error(endpoint, exc) from exc
        if not vectors:
            raise McpProtocolError("embedding endpoint returned no query vector")
        vector = vectors[0]
        return vector, vector_literal_dimensions(vector)
    embedding_values = llama.embed_text(query)
    return db.vector_literal(embedding_values), len(embedding_values)


def tool_search_code_intel_semantic(args: Json) -> Json:
    query = optional_text(args, "query")
    if not query:
        raise McpProtocolError("query is required")
    limit = require_int(args, "limit", 10, 1, 50)
    clauses, params = code_intel_clauses(args, "r")
    clauses.append("r.embedding IS NOT NULL")
    embedding, embedding_dimensions = query_embedding(query)
    query_params = [embedding, *params, limit]
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        rows = conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT r.id, r.snapshot_id, r.collection, r.repo, r.repo_role,
                   r.branch, r.commit_sha, r.source_path, r.language, r.file_role,
                   r.content_class, r.record_type, r.record_id, r.parent_record_id,
                   r.title, r.summary, r.line_start, r.line_end, r.symbol,
                   r.symbol_kind, r.confidence_kind, r.confidence, r.tool, r.rule_id,
                   r.severity, r.embedding <=> %s::vector AS distance,
                   left(r.display_content, 800) AS snippet_raw
            FROM project_code_intel_records r
            """,
                    clauses,
                    """
            ORDER BY distance ASC, r.updated_at DESC
            LIMIT %s
            """,
                )
            ),
            query_params,
        ).fetchall()
    verbose = optional_bool(args, "verbose") or False
    return ok({
        "query": query,
        "embedding_dimensions": embedding_dimensions,
        **snapshot_scope_response(args),
        "results": cast("JsonValue", _format_records(rows, verbose=verbose)),
    })


def record_projection_query(*, include_content: bool, filter_collection: bool = False) -> str:
    if include_content:
        if filter_collection:
            return """
                SELECT r.id, r.collection, r.repo, r.repo_role, r.branch, r.commit_sha, r.tree_sha,
                       r.source_path, r.language, r.file_role, r.content_class,
                       r.record_type, r.record_id, r.parent_record_id, r.title, r.summary,
                       left(r.embedding_text, %s) AS embedding_text,
                       coalesce(length(r.embedding_text), 0) > %s AS embedding_text_truncated,
                       left(r.display_content, %s) AS display_content,
                       coalesce(length(r.display_content), 0) > %s AS display_content_truncated,
                       false AS content_omitted,
                       r.line_start, r.line_end, r.symbol, r.symbol_kind, r.confidence_kind,
                       r.confidence, r.tool, r.rule_id, r.severity, r.analyzer, r.analyzer_version,
                       r.parser, r.parser_version, r.chunker_version, r.metadata, r.created_at,
                       r.updated_at, r.embedding IS NOT NULL AS has_embedding,
                       coalesce(f.is_untracked, false) AS is_untracked,
                       coalesce(f.indexed_dirty, false) AS indexed_dirty
                FROM project_code_intel_records r
                LEFT JOIN project_code_intel_files f ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
                WHERE r.id = %s AND r.collection = %s
                """
        return """
            SELECT r.id, r.collection, r.repo, r.repo_role, r.branch, r.commit_sha, r.tree_sha,
                   r.source_path, r.language, r.file_role, r.content_class,
                   r.record_type, r.record_id, r.parent_record_id, r.title, r.summary,
                   left(r.embedding_text, %s) AS embedding_text,
                   coalesce(length(r.embedding_text), 0) > %s AS embedding_text_truncated,
                   left(r.display_content, %s) AS display_content,
                   coalesce(length(r.display_content), 0) > %s AS display_content_truncated,
                   false AS content_omitted,
                   r.line_start, r.line_end, r.symbol, r.symbol_kind, r.confidence_kind,
                   r.confidence, r.tool, r.rule_id, r.severity, r.analyzer, r.analyzer_version,
                   r.parser, r.parser_version, r.chunker_version, r.metadata, r.created_at,
                   r.updated_at, r.embedding IS NOT NULL AS has_embedding,
                   coalesce(f.is_untracked, false) AS is_untracked,
                   coalesce(f.indexed_dirty, false) AS indexed_dirty
            FROM project_code_intel_records r
            LEFT JOIN project_code_intel_files f ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
            WHERE r.id = %s
            """
    if filter_collection:
        return """
            SELECT r.id, r.collection, r.repo, r.repo_role, r.branch, r.commit_sha, r.tree_sha,
                   r.source_path, r.language, r.file_role, r.content_class,
                   r.record_type, r.record_id, r.parent_record_id, r.title, r.summary,
                   NULL::text AS embedding_text,
                   false AS embedding_text_truncated,
                   NULL::text AS display_content,
                   false AS display_content_truncated,
                   true AS content_omitted,
                   r.line_start, r.line_end, r.symbol, r.symbol_kind, r.confidence_kind,
                   r.confidence, r.tool, r.rule_id, r.severity, r.analyzer, r.analyzer_version,
                   r.parser, r.parser_version, r.chunker_version, r.metadata, r.created_at,
                   r.updated_at, r.embedding IS NOT NULL AS has_embedding,
                   coalesce(f.is_untracked, false) AS is_untracked,
                   coalesce(f.indexed_dirty, false) AS indexed_dirty
            FROM project_code_intel_records r
            LEFT JOIN project_code_intel_files f ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
            WHERE r.id = %s AND r.collection = %s
            """
    return """
            SELECT r.id, r.collection, r.repo, r.repo_role, r.branch, r.commit_sha, r.tree_sha,
                   r.source_path, r.language, r.file_role, r.content_class,
                   r.record_type, r.record_id, r.parent_record_id, r.title, r.summary,
                   NULL::text AS embedding_text,
                   false AS embedding_text_truncated,
                   NULL::text AS display_content,
                   false AS display_content_truncated,
                   true AS content_omitted,
                   r.line_start, r.line_end, r.symbol, r.symbol_kind, r.confidence_kind,
                   r.confidence, r.tool, r.rule_id, r.severity, r.analyzer, r.analyzer_version,
                   r.parser, r.parser_version, r.chunker_version, r.metadata, r.created_at,
                   r.updated_at, r.embedding IS NOT NULL AS has_embedding,
                   coalesce(f.is_untracked, false) AS is_untracked,
                   coalesce(f.indexed_dirty, false) AS indexed_dirty
            FROM project_code_intel_records r
            LEFT JOIN project_code_intel_files f ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
            WHERE r.id = %s
            """


def tool_get_code_intel_record(args: Json) -> Json:
    record_id = args.get("id")
    if not isinstance(record_id, int):
        raise McpProtocolTypeError("id must be an integer")
    include_content = optional_bool(args, "include_content")
    collection = scoped_collection({})
    params: QueryParams
    if include_content:
        content_limit = mcp_max_record_content_chars()
        params = [content_limit, content_limit, content_limit, content_limit, record_id]
    else:
        params = [record_id]
    if collection:
        params.append(collection)
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        row = conn.execute(
            record_projection_query(include_content=include_content, filter_collection=collection is not None),
            params,
        ).fetchone()
    if row is None:
        return ok({"found": False})
    return ok({"result": row})


def tool_related_code_intel(args: Json) -> Json:
    record_id = optional_text(args, "record_id")
    symbol = optional_text(args, "symbol")
    if not record_id and not symbol:
        raise McpProtocolError("record_id or symbol is required")
    limit = require_int(args, "limit", 20, 1, 100)
    collection = scoped_collection(args)
    repo = optional_text(args, "repo")
    edge_type = optional_text(args, "edge_type")
    confidence_kind = optional_text(args, "confidence_kind")

    clauses = ["TRUE"]
    params: QueryParams = []
    if record_id:
        clauses.append("(e.source_record_id = %s OR e.target_record_id = %s)")
        params.extend([record_id, record_id])
    if symbol:
        clauses.append("(e.source_symbol = %s OR e.target_symbol = %s)")
        params.extend([symbol, symbol])
    if repo:
        clauses.append("e.repo = %s")
        params.append(repo)
    if collection:
        clauses.append("e.collection = %s")
        params.append(collection)
    if edge_type:
        clauses.append("e.edge_type = %s")
        params.append(edge_type)
    if confidence_kind:
        clauses.append("e.confidence_kind = %s")
        params.append(confidence_kind)
    snapshot_clauses, snapshot_params = scoped_snapshot_clauses(args, "e")
    clauses.extend(snapshot_clauses)
    params.extend(snapshot_params)
    params.append(limit)

    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        edges = conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT e.id, e.snapshot_id, e.collection, e.repo, e.commit_sha,
                   e.source_record_id, e.target_record_id, e.edge_type,
                   e.source_symbol, e.target_symbol, e.source_path, e.target_path,
                   e.confidence_kind, e.metadata,
                   src.id AS source_record_db_id, src.title AS source_title,
                   src.summary AS source_summary, src.record_type AS source_record_type,
                   src.language AS source_language, src.line_start AS source_line_start,
                   src.line_end AS source_line_end,
                   tgt.id AS target_record_db_id, tgt.title AS target_title,
                   tgt.summary AS target_summary, tgt.record_type AS target_record_type,
                   tgt.language AS target_language, tgt.line_start AS target_line_start,
                   tgt.line_end AS target_line_end
            FROM project_code_intel_edges e
            LEFT JOIN project_code_intel_records src
                ON src.snapshot_id = e.snapshot_id AND src.record_id = e.source_record_id
            LEFT JOIN project_code_intel_records tgt
                ON tgt.snapshot_id = e.snapshot_id AND tgt.record_id = e.target_record_id
            """,
                    clauses,
                    """
            ORDER BY e.id DESC
            LIMIT %s
            """,
                )
            ),
            params,
        ).fetchall()
    verbose = optional_bool(args, "verbose") or False
    return ok({**snapshot_scope_response(args), "edges": cast("JsonValue", _format_edges(edges, verbose=verbose))})


def tool_search_static_findings(args: Json) -> Json:
    limit = require_int(args, "limit", 10, 1, 100)
    clauses, params = static_finding_clauses(args)
    params.append(limit)
    with mcp_db.connect() as conn:
        if not table_regclass_exists(conn, "project_code_intel_static_findings"):
            return ok({"error": "static-analysis schema is not initialized"})
        rows = conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT f.id, f.collection, f.repo, f.commit_sha, f.finding_key,
                   f.rule_id, f.level, f.kind, f.message, f.baseline_state,
                   f.primary_source_path, f.primary_uri, f.line_start, f.line_end,
                   f.column_start, f.column_end, f.fingerprints,
                   f.suppressions, f.properties, f.created_at,
                   r.id AS run_id, r.tool_name, r.tool_version, r.sarif_path,
                   r.automation_id
            FROM project_code_intel_static_findings f
            JOIN project_code_intel_static_runs r ON r.id = f.run_id
            """,
                    clauses,
                    """
            ORDER BY f.created_at DESC, f.id DESC
            LIMIT %s
            """,
                )
            ),
            params,
        ).fetchall()
    return ok({**snapshot_scope_response(args), "results": rows})


def row_to_json(row: db.DbRow | None) -> Json | None:
    if row is None:
        return None
    return {key: cast("JsonValue", value) for key, value in row.items()}


def compact_row(row: db.DbRow | None, keys: tuple[str, ...]) -> Json | None:
    if row is None:
        return None
    result: Json = {}
    for key in keys:
        if key in row:
            result[key] = cast("JsonValue", row[key])
    return result


def static_finding_warnings(finding: db.DbRow) -> list[object]:
    run_metadata = finding.get("run_metadata")
    if not isinstance(run_metadata, dict):
        return []
    run_metadata_obj = cast("dict[object, object]", run_metadata)
    raw_warnings = run_metadata_obj.get("code_intel_warnings")
    if not isinstance(raw_warnings, list):
        return []
    return list(cast("list[object]", raw_warnings))


def tool_get_static_finding(args: Json) -> Json:
    finding_id = args.get("id")
    if not isinstance(finding_id, int):
        raise McpProtocolTypeError("id must be an integer")
    include_raw = optional_bool(args, "include_raw")
    include_run_metadata = optional_bool(args, "include_run_metadata")
    include_code_flows = optional_bool(args, "include_code_flows")
    collection = scoped_collection({})
    finding_clauses = ["f.id = %s"]
    finding_params: QueryParams = [finding_id]
    if collection:
        finding_clauses.append("f.collection = %s")
        finding_params.append(collection)
    with mcp_db.connect() as conn:
        if not table_regclass_exists(conn, "project_code_intel_static_findings"):
            return ok({"error": "static-analysis schema is not initialized"})
        finding = conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT f.*, r.tool_name, r.tool_version, r.semantic_version,
                   r.information_uri, r.automation_id, r.sarif_path,
                   r.sarif_sha256, r.run_index, r.metadata AS run_metadata
            FROM project_code_intel_static_findings f
            JOIN project_code_intel_static_runs r ON r.id = f.run_id
            """,
                    finding_clauses,
                    "",
                )
            ),
            finding_params,
        ).fetchone()
        if not finding:
            return ok({"found": False})
        rule = conn.execute(
            """
            SELECT id, rule_id, name, short_description, full_description,
                   default_level, help_uri, properties, metadata
            FROM project_code_intel_static_rules
            WHERE run_id = %s AND rule_id = %s
            """,
            [finding["run_id"], finding["rule_id"]],
        ).fetchone()
        locations = conn.execute(
            """
            SELECT id, ordinal, location_kind, source_path, uri, message,
                   line_start, line_end, column_start, column_end, snippet,
                   properties
            FROM project_code_intel_static_locations
            WHERE finding_id = %s
            ORDER BY ordinal, id
            """,
            [finding_id],
        ).fetchall()
        if include_code_flows:
            code_flow_rows = conn.execute(
                """
                SELECT id, flow_index, thread_index, step_index, source_path, uri,
                       message, line_start, line_end, column_start, column_end,
                       importance, properties
                FROM project_code_intel_static_code_flows
                WHERE finding_id = %s
                ORDER BY flow_index, thread_index, step_index, id
                """,
                [finding_id],
            ).fetchall()
            code_flow_count = len(code_flow_rows)
        else:
            code_flow_rows = []
            code_flow_count_row = conn.execute(
                """
                SELECT count(*) AS code_flow_steps
                FROM project_code_intel_static_code_flows
                WHERE finding_id = %s
                """,
                [finding_id],
            ).fetchone()
            code_flow_count = 0 if code_flow_count_row is None else row_int(code_flow_count_row, "code_flow_steps")

    result: Json = {
        "finding": compact_row(finding, STATIC_FINDING_COMPACT_KEYS),
        "rule": compact_row(rule, STATIC_RULE_COMPACT_KEYS),
        "locations": cast("JsonValue", locations),
        "code_flow_steps": code_flow_count,
        "warnings": cast("JsonValue", static_finding_warnings(finding)),
    }
    if include_code_flows:
        result["code_flows"] = cast("JsonValue", code_flow_rows)
    if include_raw:
        result["raw"] = {
            "finding": cast("JsonValue", row_to_json(finding)),
            "rule": cast("JsonValue", row_to_json(rule)),
        }
    if include_run_metadata:
        result["run_metadata"] = cast("JsonValue", finding.get("run_metadata"))
    return ok(result)


def tool_get_static_code_flow(args: Json) -> Json:
    finding_id = args.get("finding_id")
    if not isinstance(finding_id, int):
        raise McpProtocolTypeError("finding_id must be an integer")
    flow_index = args.get("flow_index")
    collection = scoped_collection({})
    clauses = ["cf.finding_id = %s"]
    params: QueryParams = [finding_id]
    if collection:
        clauses.append("f.collection = %s")
        params.append(collection)
    if flow_index is not None:
        if not isinstance(flow_index, int):
            raise McpProtocolTypeError("flow_index must be an integer")
        clauses.append("cf.flow_index = %s")
        params.append(flow_index)
    with mcp_db.connect() as conn:
        if not table_regclass_exists(conn, "project_code_intel_static_code_flows"):
            return ok({"error": "static-analysis schema is not initialized"})
        finding_exists = (
            conn.execute(
                "SELECT 1 FROM project_code_intel_static_findings WHERE id = %s",
                [finding_id],
            ).fetchone()
            is not None
        )
        if not finding_exists:
            return ok({"found": False})
        rows = conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT cf.id, cf.finding_id, cf.flow_index, cf.thread_index, cf.step_index,
                   cf.source_path, cf.uri, cf.message, cf.line_start, cf.line_end,
                   cf.column_start, cf.column_end, cf.importance, cf.properties
            FROM project_code_intel_static_code_flows cf
            JOIN project_code_intel_static_findings f ON f.id = cf.finding_id
            """,
                    clauses,
                    """
            ORDER BY cf.flow_index, cf.thread_index, cf.step_index, cf.id
            """,
                )
            ),
            params,
        ).fetchall()
    return ok({"found": True, "finding_id": finding_id, "flow_index": flow_index, "steps": rows})


def tool_list_code_intel_files(args: Json) -> Json:
    limit = require_int(args, "limit", 50, 1, 500)
    clauses, params = scoped_collection_repo_clauses(args, "f")
    for arg_name in ("language", "file_role", "content_class"):
        value = optional_text(args, arg_name)
        if value:
            clauses.append(f"f.{arg_name} = %s")
            params.append(value)
    source_path = optional_text(args, "source_path")
    if source_path:
        clauses.append("f.source_path = %s")
        params.append(source_path)
    for arg_name in (
        "is_test",
        "is_doc",
        "is_generated",
        "is_vendor",
        "is_source",
        "is_build",
        "is_config",
    ):
        if arg_name in args:
            value = optional_bool(args, arg_name)
            clauses.append(f"f.{arg_name} = %s")
            params.append(value)
    if optional_bool(args, "only_skipped"):
        clauses.append("f.skipped_reason IS NOT NULL")
    include_metadata = optional_bool(args, "include_metadata")
    params.append(limit)

    files_select_slim = """
            SELECT f.id, f.snapshot_id, f.collection, f.repo, f.repo_role, f.branch,
                   f.commit_sha, f.tree_sha, f.source_path, f.git_blob_sha, f.file_sha256,
                   f.size_bytes, f.language, f.file_role, f.content_class,
                   f.is_generated, f.is_vendor, f.is_test, f.is_source, f.is_build,
                   f.is_config, f.is_doc, f.skipped_reason, f.created_at
            FROM project_code_intel_files f
            """
    files_select_full = """
            SELECT f.id, f.snapshot_id, f.collection, f.repo, f.repo_role, f.branch,
                   f.commit_sha, f.tree_sha, f.source_path, f.git_blob_sha, f.file_sha256,
                   f.size_bytes, f.language, f.file_role, f.content_class,
                   f.is_generated, f.is_vendor, f.is_test, f.is_source, f.is_build,
                   f.is_config, f.is_doc, f.skipped_reason, f.metadata, f.created_at
            FROM project_code_intel_files f
            """

    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        rows = conn.execute(
            db.query_sql(
                query_with_where(
                    files_select_full if include_metadata else files_select_slim,
                    clauses,
                    """
            ORDER BY f.source_path
            LIMIT %s
            """,
                )
            ),
            params,
        ).fetchall()
    return ok({**snapshot_scope_response(args), "files": rows})


def tool_list_code_intel_parser_failures(args: Json) -> Json:
    limit = require_int(args, "limit", 50, 1, 500)
    clauses, params = scoped_collection_repo_clauses(args, "pf")
    for arg_name in ("language", "parser"):
        value = optional_text(args, arg_name)
        if value:
            clauses.append(f"pf.{arg_name} = %s")
            params.append(value)
    source_path = optional_text(args, "source_path")
    if source_path:
        clauses.append("pf.source_path = %s")
        params.append(source_path)
    params.append(limit)

    with mcp_db.connect() as conn:
        if not table_regclass_exists(conn, "project_code_intel_parser_failures"):
            return ok({"error": "code intelligence schema is not initialized"})
        rows = conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT pf.id, pf.snapshot_id, pf.collection, pf.repo, pf.commit_sha,
                   pf.source_path, pf.language, pf.parser, pf.error, pf.metadata,
                   pf.created_at
            FROM project_code_intel_parser_failures pf
            """,
                    clauses,
                    """
            ORDER BY pf.source_path, pf.parser
            LIMIT %s
            """,
                )
            ),
            params,
        ).fetchall()
    return ok({**snapshot_scope_response(args), "parser_failures": rows})


ToolHandler = Callable[[Json], Json]
ToolRegistry = dict[str, tuple[ToolDefinition, ToolHandler]]


TOOLS: ToolRegistry = {
    "code_intel_status": (TOOL_DEFINITIONS["code_intel_status"], tool_code_intel_status),
    "search_code_intel_text": (TOOL_DEFINITIONS["search_code_intel_text"], tool_search_code_intel_text),
    "search_code_intel_semantic": (TOOL_DEFINITIONS["search_code_intel_semantic"], tool_search_code_intel_semantic),
    "get_code_intel_record": (TOOL_DEFINITIONS["get_code_intel_record"], tool_get_code_intel_record),
    "related_code_intel": (TOOL_DEFINITIONS["related_code_intel"], tool_related_code_intel),
    "list_code_intel_files": (TOOL_DEFINITIONS["list_code_intel_files"], tool_list_code_intel_files),
    "list_code_intel_parser_failures": (
        TOOL_DEFINITIONS["list_code_intel_parser_failures"],
        tool_list_code_intel_parser_failures,
    ),
    "search_static_findings": (TOOL_DEFINITIONS["search_static_findings"], tool_search_static_findings),
    "get_static_finding": (TOOL_DEFINITIONS["get_static_finding"], tool_get_static_finding),
    "get_static_code_flow": (TOOL_DEFINITIONS["get_static_code_flow"], tool_get_static_code_flow),
}


def advertised_tools() -> list[Json]:
    tools: list[Json] = []
    writes = db.allow_writes()
    for name, (definition, _handler) in TOOLS.items():
        if definition.write_tool and not writes:
            continue
        tools.append({
            "name": name,
            "description": definition.description,
            "inputSchema": definition.input_schema,
        })
    return tools
