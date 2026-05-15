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
    source_path_clauses,
    static_finding_clauses,
    status_filters,
)
from project_code_intelligence.mcp.protocol import (
    Json,
    QueryParams,
    mcp_max_record_content_chars,
    ok,
    optional_bool,
    optional_int,
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

SearchMode: TypeAlias = Literal["search", "enumerate"]
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
_SNIPPET_CLOSE_FENCE_RE = re.compile(r"\n`{3,}[^\n]*$")

# Fields stripped in compact mode — per-result snapshot/git/repo metadata that is
# constant across all results in a single-snapshot query and redundant with the
# response envelope. Verbose mode (verbose=true) returns them.
_COMPACT_RECORD_STRIP = frozenset({
    "id",
    "snapshot_id",
    "collection",
    "repo",
    "repo_role",
    "branch",
    "commit_sha",
    "tree_sha",
    "created_at",
    "updated_at",
    "confidence",
    "tool",
    "rule_id",
    "severity",
    # embedding_text duplicates display_content minus the markdown frame and is
    # truncated mid-body — useful for debugging embedding similarity, noise for
    # navigation. Verbose mode keeps it.
    "embedding_text",
    "embedding_text_truncated",
})
_COMPACT_EDGE_STRIP = frozenset({"snapshot_id", "collection", "repo", "commit_sha"})

# Per-record metadata fields that are valuable but heavy enough to balloon a
# compact response. doc_links in particular carries every URL in a README, which
# easily dwarfs the rest of the record. Verbose mode keeps these.
_HEAVY_METADATA_KEYS = frozenset({"doc_links"})

# Boolean fields where False is the uninteresting default. Stripped from compact
# responses so the absence of the key implies False. has_embedding stays — both
# True and False carry useful signal (False == not findable via semantic search).
_STRIP_WHEN_FALSE = frozenset({
    "is_test",
    "is_doc",
    "is_generated",
    "is_vendor",
    "is_source",
    "is_build",
    "is_config",
    "is_untracked",
    "indexed_dirty",
    "display_content_truncated",
    "embedding_text_truncated",
    "content_omitted",
})


def _is_compact_noise(key: str, value: object) -> bool:
    if value is None:
        return True
    if value in ([], {}):
        return True
    return value is False and key in _STRIP_WHEN_FALSE


_RECORD_TYPE_DEDUP_PRIORITY: dict[str, int] = {"code_chunk": 0, "symbol_definition": 1}


def _dedup_by_location(rows: list[db.DbRow]) -> list[db.DbRow]:
    """Keep one record per (source_path, line_start, line_end), preferring code_chunk.

    Records without line numbers are never deduplicated.
    Two passes: first find the winning record_type per location, then filter to keep
    only the first occurrence of the winner (preserving rank order).
    """
    best: dict[tuple[object, object, object], str] = {}
    for row in rows:
        line_start = row.get("line_start")
        if line_start is None:
            continue
        key = (row.get("source_path"), line_start, row.get("line_end"))
        rtype = str(row.get("record_type") or "")
        prev = best.get(key)
        if prev is None or _RECORD_TYPE_DEDUP_PRIORITY.get(rtype, 99) < _RECORD_TYPE_DEDUP_PRIORITY.get(prev, 99):
            best[key] = rtype
    seen: set[tuple[object, object, object]] = set()
    result: list[db.DbRow] = []
    for row in rows:
        line_start = row.get("line_start")
        if line_start is None:
            result.append(row)
            continue
        key = (row.get("source_path"), line_start, row.get("line_end"))
        rtype = str(row.get("record_type") or "")
        if key not in seen and rtype == best[key]:
            seen.add(key)
            result.append(row)
    return result


DEFAULT_SNIPPET_LENGTH = 300


def _extract_snippet(raw: str | None, length: int = DEFAULT_SNIPPET_LENGTH) -> str | None:
    """Return the first `length` chars of code body from a display_content prefix."""
    if not raw:
        return None
    m = _SNIPPET_FENCE_RE.search(raw)
    if m:
        code = raw[m.end() :]
        return _SNIPPET_CLOSE_FENCE_RE.sub("", code[:length]).rstrip() or None
    return None


def _compact_record(row: db.DbRow, snippet_length: int = DEFAULT_SNIPPET_LENGTH) -> dict[str, object]:
    snippet = _extract_snippet(row.get("snippet_raw"), snippet_length)  # type: ignore[arg-type]
    out: dict[str, object] = {
        k: v
        for k, v in row.items()
        if not _is_compact_noise(k, v) and k not in _COMPACT_RECORD_STRIP and k != "snippet_raw"
    }
    metadata = out.get("metadata")
    if isinstance(metadata, dict):
        metadata_dict = cast("dict[str, object]", metadata)
        trimmed = {
            k: v for k, v in metadata_dict.items() if k not in _HEAVY_METADATA_KEYS and not _is_compact_noise(k, v)
        }
        if trimmed != metadata_dict:
            if trimmed:
                out["metadata"] = trimmed
            else:
                del out["metadata"]
    if snippet:
        out["snippet"] = snippet
    return out


def _verbose_record(row: db.DbRow, snippet_length: int = DEFAULT_SNIPPET_LENGTH) -> dict[str, object]:
    snippet = _extract_snippet(row.get("snippet_raw"), snippet_length)  # type: ignore[arg-type]
    out = {k: v for k, v in row.items() if k != "snippet_raw"}
    if snippet:
        out["snippet"] = snippet
    return out


def _compact_file(row: db.DbRow) -> dict[str, object]:
    return {k: v for k, v in row.items() if not _is_compact_noise(k, v)}


def _format_records(
    rows: list[db.DbRow],
    *,
    verbose: bool,
    snippet_length: int = DEFAULT_SNIPPET_LENGTH,
) -> list[dict[str, object]]:
    fmt = _verbose_record if verbose else _compact_record
    return [fmt(row, snippet_length) for row in rows]


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


def search_mode(args: Json, query: str | None) -> SearchMode:
    value = optional_text(args, "mode")
    if value is None:
        return "enumerate" if not query else "search"
    if value not in {"search", "enumerate"}:
        raise McpProtocolError("mode must be one of: search, enumerate")
    if value == "search" and not query:
        raise McpProtocolError("mode=search requires a non-empty query")
    if value == "enumerate" and query:
        raise McpProtocolError("mode=enumerate must not be combined with a query")
    return cast("SearchMode", value)


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
        if plan.strategy == "all_terms_fallback":
            # Every result matches all terms so rank = count(terms) = constant — not a useful signal.
            # Use NULL rank and sort by recency instead.
            query_sql = query_with_where(
                CODE_INTEL_RECORD_SELECT_LIST,
                clauses,
                """
            ORDER BY r.updated_at DESC
            LIMIT %s
            """,
            )
            params = [*filter_params, patterns, plan.limit]
        else:
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


def validate_explicit_snapshot_id(conn: db.DbConnection, args: Json) -> None:
    """When snapshot_id is explicitly set, fail loudly if it doesn't exist."""
    snapshot_id = optional_int(args, "snapshot_id")
    if snapshot_id is None:
        return
    row = conn.execute(
        "SELECT 1 FROM project_code_intel_snapshots WHERE id = %s",
        [snapshot_id],
    ).fetchone()
    if row is None:
        raise McpProtocolError(f"snapshot_id {snapshot_id} does not exist")


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


def _annotate_status_snapshots(snapshot_rows: list[db.DbRow]) -> list[Json]:
    head_commit = git_utils.run_git(Path.cwd(), ["rev-parse", "HEAD"])
    now = datetime.datetime.now(datetime.timezone.utc)
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
        snapshots.append(snap_dict)
    return snapshots


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


def tool_code_intel_status(args: Json) -> Json:
    filters = status_filters(args)
    directory_depth = require_int(args, "directory_depth", 1, 1, 5)
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"schema_present": False})
        validate_explicit_snapshot_id(conn, args)
        schema_versions = (
            schema_migration_versions(conn)
            if table_regclass_exists(conn, "project_code_intel_schema_migrations")
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
        snapshots = _annotate_status_snapshots(snapshot_rows)
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
        breakdowns = _status_file_breakdowns(conn, filters, directory_depth)
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
        "language_breakdown": breakdowns["language"],
        "directory_breakdown": breakdowns["directory"],
        "static_runs": static_runs,
        "static_findings": static_findings,
    })


def _execute_text_search(
    conn: db.DbConnection,
    args: Json,
    terms: tuple[str, ...],
    limit: int,
) -> tuple[list[db.DbRow], SearchQueryStrategy, str | None]:
    query = optional_text(args, "query")
    query_mode = search_query_mode(args)
    strategy: SearchQueryStrategy = "list"
    fallback_reason: str | None = None
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
    return rows, strategy, fallback_reason


def tool_search_code_intel_text(args: Json) -> Json:
    query = optional_text(args, "query")
    mode = search_mode(args, query)
    query_mode = search_query_mode(args)
    terms = tuple(search_terms(query)) if query else ()
    limit = require_int(args, "limit", 10, 1, 50)
    snippet_length = require_int(args, "snippet_length", DEFAULT_SNIPPET_LENGTH, 1, 800)
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        validate_explicit_snapshot_id(conn, args)
        rows, strategy, fallback_reason = _execute_text_search(conn, args, terms, limit)
    if not optional_text(args, "record_type"):
        rows = _dedup_by_location(rows)
    verbose = optional_bool(args, "verbose") or False
    response: Json = {
        "query": query,
        "query_strategy": strategy,
        **snapshot_scope_response(args),
        "results": cast("JsonValue", _format_records(rows, verbose=verbose, snippet_length=snippet_length)),
    }
    # Echo mode/query_mode only when explicitly set — inferred defaults are noise.
    if optional_text(args, "mode") is not None:
        response["mode"] = mode
    if optional_text(args, "query_mode") is not None:
        response["query_mode"] = query_mode
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
    snippet_length = require_int(args, "snippet_length", DEFAULT_SNIPPET_LENGTH, 1, 800)
    clauses, params = code_intel_clauses(args, "r")
    clauses.append("r.embedding IS NOT NULL")
    embedding, embedding_dimensions = query_embedding(query)
    query_params = [embedding, *params, limit]
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        validate_explicit_snapshot_id(conn, args)
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
        "results": cast("JsonValue", _format_records(rows, verbose=verbose, snippet_length=snippet_length)),
    })


_RECORD_PROJECTION_WITH_CONTENT = """
            SELECT r.id, r.snapshot_id, r.collection, r.repo, r.repo_role, r.branch,
                   r.commit_sha, r.tree_sha,
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
            """

_RECORD_PROJECTION_WITHOUT_CONTENT = """
            SELECT r.id, r.snapshot_id, r.collection, r.repo, r.repo_role, r.branch,
                   r.commit_sha, r.tree_sha,
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
            """


def record_projection_select(*, include_content: bool) -> str:
    return _RECORD_PROJECTION_WITH_CONTENT if include_content else _RECORD_PROJECTION_WITHOUT_CONTENT


def _get_record_ids_arg(args: Json) -> tuple[list[str], bool]:
    record_id = args.get("record_id")
    record_ids = args.get("record_ids")
    if record_id is not None and record_ids is not None:
        raise McpProtocolError("provide exactly one of record_id or record_ids")
    if record_id is not None:
        if not isinstance(record_id, str) or not record_id:
            raise McpProtocolTypeError("record_id must be a non-empty string")
        return [record_id], False
    if record_ids is None:
        raise McpProtocolError("record_id or record_ids is required")
    if not isinstance(record_ids, list) or not record_ids:
        raise McpProtocolTypeError("record_ids must be a non-empty list of strings")
    out: list[str] = []
    for item in cast("list[object]", record_ids):
        if not isinstance(item, str) or not item:
            raise McpProtocolTypeError("record_ids entries must be non-empty strings")
        out.append(item)
    return out, True


def _build_record_lookup(args: Json, ids: list[str], *, batch: bool, include_content: bool) -> tuple[str, QueryParams]:
    where_clauses, where_params = scoped_collection_repo_clauses(args, "r")
    select_prefix = record_projection_select(include_content=include_content)
    if batch:
        where_clauses.append("r.record_id = ANY(%s::text[])")
        where_params.append(ids)
        suffix = "\n            ORDER BY r.record_id, r.updated_at DESC, r.id DESC\n            "
        select_prefix = select_prefix.replace("SELECT r.id,", "SELECT DISTINCT ON (r.record_id) r.id,", 1)
    else:
        where_clauses.append("r.record_id = %s")
        where_params.append(ids[0])
        suffix = "\n            ORDER BY r.updated_at DESC, r.id DESC\n            LIMIT 1\n            "
    query_sql = query_with_where(select_prefix, where_clauses, suffix)
    if include_content:
        limit = mcp_max_record_content_chars()
        params: QueryParams = [limit, limit, limit, limit, *where_params]
    else:
        params = list(where_params)
    return query_sql, params


def tool_get_code_intel_record(args: Json) -> Json:
    ids, batch = _get_record_ids_arg(args)
    include_content = optional_bool(args, "include_content")
    verbose = optional_bool(args, "verbose") or False
    query_sql, params = _build_record_lookup(args, ids, batch=batch, include_content=include_content)
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        validate_explicit_snapshot_id(conn, args)
        cursor = conn.execute(db.query_sql(query_sql), params)
        if not batch:
            row = cursor.fetchone()
            if row is None:
                return ok({"found": False})
            return ok({"result": dict(row) if verbose else _compact_record(row)})
        rows = cursor.fetchall()
    formatted = [dict(row) if verbose else _compact_record(row) for row in rows]
    missing = [rid for rid in ids if rid not in {str(row["record_id"]) for row in rows}]
    response: Json = {"results": cast("JsonValue", formatted)}
    if missing:
        response["missing"] = missing
    return ok(response)


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

    clauses = ["TRUE", "(e.target_record_id IS NULL OR e.source_record_id != e.target_record_id)"]
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
        validate_explicit_snapshot_id(conn, args)
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
        validate_explicit_snapshot_id(conn, args)
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
    path_clauses, path_params = source_path_clauses(args, "f")
    clauses.extend(path_clauses)
    params.extend(path_params)
    for arg_name in (
        "is_test",
        "is_doc",
        "is_generated",
        "is_vendor",
        "is_source",
        "is_build",
        "is_config",
        "is_untracked",
    ):
        if arg_name in args:
            value = optional_bool(args, arg_name)
            clauses.append(f"f.{arg_name} = %s")
            params.append(value)
    if optional_bool(args, "only_skipped"):
        clauses.append("f.skipped_reason IS NOT NULL")
    verbose = optional_bool(args, "verbose") or False
    params.append(limit)

    files_select_slim = """
            SELECT f.id, f.source_path, f.size_bytes, f.language, f.file_role, f.content_class,
                   f.is_generated, f.is_vendor, f.is_test, f.is_source, f.is_build,
                   f.is_config, f.is_doc, f.skipped_reason
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
        validate_explicit_snapshot_id(conn, args)
        rows = conn.execute(
            db.query_sql(
                query_with_where(
                    files_select_full if verbose else files_select_slim,
                    clauses,
                    """
            ORDER BY f.source_path
            LIMIT %s
            """,
                )
            ),
            params,
        ).fetchall()
    files: list[object] = list(rows) if verbose else [_compact_file(row) for row in rows]
    return ok({**snapshot_scope_response(args), "files": cast("JsonValue", files)})


def tool_list_code_intel_parser_failures(args: Json) -> Json:
    limit = require_int(args, "limit", 50, 1, 500)
    clauses, params = scoped_collection_repo_clauses(args, "pf")
    for arg_name in ("language", "parser"):
        value = optional_text(args, arg_name)
        if value:
            clauses.append(f"pf.{arg_name} = %s")
            params.append(value)
    path_clauses, path_params = source_path_clauses(args, "pf")
    clauses.extend(path_clauses)
    params.extend(path_params)
    params.append(limit)

    with mcp_db.connect() as conn:
        if not table_regclass_exists(conn, "project_code_intel_parser_failures"):
            return ok({"error": "code intelligence schema is not initialized"})
        validate_explicit_snapshot_id(conn, args)
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
