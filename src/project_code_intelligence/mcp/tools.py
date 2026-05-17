"""MCP tool handlers for the code-intelligence database."""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable, Mapping, Sequence
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
RelatedDirection: TypeAlias = Literal["any", "incoming", "outgoing"]
SearchQueryStrategy: TypeAlias = Literal[
    "list",
    "websearch",
    "all_terms",
    "all_terms_fallback",
    "any_terms",
    "any_terms_fallback",
]

SEARCH_TERM_RE = re.compile(r"[A-Za-z0-9_$./:+@-]+")
IDENTIFIER_QUERY_RE = re.compile(r"[$A-Za-z_][$A-Za-z0-9_$./:+@-]*\Z")
SEARCH_OPERATOR_WORDS = frozenset({"and", "or", "not"})
DEFAULT_MIXED_SEARCH_EXCLUDED_RECORD_TYPES = frozenset({"security_pattern"})
MIN_CENTERED_SNIPPET_TERM_CHARS = 3

CODE_INTEL_RECORD_SELECT_LIST = """
            SELECT r.id, r.snapshot_id, r.collection, r.repo, r.repo_role, r.branch,
                   r.commit_sha, r.tree_sha, r.source_path, r.language, r.file_role,
                   r.content_class, r.record_type, r.record_id, r.parent_record_id,
                   r.title, r.summary, r.line_start, r.line_end, r.symbol,
                   r.symbol_kind, r.confidence_kind, r.confidence, r.tool,
                   r.rule_id, r.severity, r.updated_at,
                   r.embedding IS NOT NULL AS has_embedding,
                   NULL::real AS rank,
                   NULL::real AS match_score,
                   coalesce(f.is_untracked, false) AS is_untracked,
                   coalesce(f.indexed_dirty, false) AS indexed_dirty,
                   left(r.display_content, 12000) AS snippet_raw
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
                   (
                       SELECT coalesce(sum(
                           CASE WHEN lower(coalesce(r.symbol, '')) = lower(search_terms.term) THEN 40 ELSE 0 END
                         + CASE WHEN lower(coalesce(r.title, '')) = lower(search_terms.term) THEN 32 ELSE 0 END
                         + CASE
                             WHEN r.record_type = 'config_symbol'
                              AND lower(regexp_replace(coalesce(r.symbol, ''), '^CONFIG_', '', 'i'))
                                  = lower(search_terms.term)
                             THEN 80
                             ELSE 0
                           END
                         + CASE
                             WHEN coalesce(r.symbol, '') ILIKE search_terms.prefix_pattern ESCAPE '\\' THEN 20
                             ELSE 0
                           END
                         + CASE WHEN coalesce(r.title, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 12 ELSE 0 END
                         + CASE
                             WHEN coalesce(r.source_path, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 8
                             ELSE 0
                           END
                         + CASE WHEN coalesce(r.record_id, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 6 ELSE 0 END
                         + CASE WHEN coalesce(r.summary, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 4 ELSE 0 END
                         + CASE
                             WHEN coalesce(r.display_content, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 1
                             ELSE 0
                           END
                       ), 0)::real
                       FROM unnest(%s::text[], %s::text[], %s::text[])
                            AS search_terms(term, prefix_pattern, pattern)
                   ) AS match_score,
                   coalesce(f.is_untracked, false) AS is_untracked,
                   coalesce(f.indexed_dirty, false) AS indexed_dirty,
                   left(r.display_content, 12000) AS snippet_raw
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
                   (
                       SELECT coalesce(sum(
                           CASE WHEN lower(coalesce(r.symbol, '')) = lower(search_terms.term) THEN 40 ELSE 0 END
                         + CASE WHEN lower(coalesce(r.title, '')) = lower(search_terms.term) THEN 32 ELSE 0 END
                         + CASE
                             WHEN r.record_type = 'config_symbol'
                              AND lower(regexp_replace(coalesce(r.symbol, ''), '^CONFIG_', '', 'i'))
                                  = lower(search_terms.term)
                             THEN 80
                             ELSE 0
                           END
                         + CASE
                             WHEN coalesce(r.symbol, '') ILIKE search_terms.prefix_pattern ESCAPE '\\' THEN 20
                             ELSE 0
                           END
                         + CASE WHEN coalesce(r.title, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 12 ELSE 0 END
                         + CASE
                             WHEN coalesce(r.source_path, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 8
                             ELSE 0
                           END
                         + CASE WHEN coalesce(r.record_id, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 6 ELSE 0 END
                         + CASE WHEN coalesce(r.summary, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 4 ELSE 0 END
                         + CASE
                             WHEN coalesce(r.display_content, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 1
                             ELSE 0
                           END
                       ), 0)::real
                       FROM unnest(%s::text[], %s::text[], %s::text[])
                            AS search_terms(term, prefix_pattern, pattern)
                   ) AS match_score,
                   coalesce(f.is_untracked, false) AS is_untracked,
                   coalesce(f.indexed_dirty, false) AS indexed_dirty,
                   left(r.display_content, 12000) AS snippet_raw
            FROM project_code_intel_records r
            LEFT JOIN project_code_intel_files f ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
            """

CODE_INTEL_RECORD_SELECT_MATCHED_LIST = """
            SELECT r.id, r.snapshot_id, r.collection, r.repo, r.repo_role, r.branch,
                   r.commit_sha, r.tree_sha, r.source_path, r.language, r.file_role,
                   r.content_class, r.record_type, r.record_id, r.parent_record_id,
                   r.title, r.summary, r.line_start, r.line_end, r.symbol,
                   r.symbol_kind, r.confidence_kind, r.confidence, r.tool,
                   r.rule_id, r.severity, r.updated_at,
                   r.embedding IS NOT NULL AS has_embedding,
                   NULL::real AS rank,
                   (
                       SELECT coalesce(sum(
                           CASE WHEN lower(coalesce(r.symbol, '')) = lower(search_terms.term) THEN 40 ELSE 0 END
                         + CASE WHEN lower(coalesce(r.title, '')) = lower(search_terms.term) THEN 32 ELSE 0 END
                         + CASE
                             WHEN r.record_type = 'config_symbol'
                              AND lower(regexp_replace(coalesce(r.symbol, ''), '^CONFIG_', '', 'i'))
                                  = lower(search_terms.term)
                             THEN 80
                             ELSE 0
                           END
                         + CASE
                             WHEN coalesce(r.symbol, '') ILIKE search_terms.prefix_pattern ESCAPE '\\' THEN 20
                             ELSE 0
                           END
                         + CASE WHEN coalesce(r.title, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 12 ELSE 0 END
                         + CASE
                             WHEN coalesce(r.source_path, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 8
                             ELSE 0
                           END
                         + CASE WHEN coalesce(r.record_id, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 6 ELSE 0 END
                         + CASE WHEN coalesce(r.summary, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 4 ELSE 0 END
                         + CASE
                             WHEN coalesce(r.display_content, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 1
                             ELSE 0
                           END
                       ), 0)::real
                       FROM unnest(%s::text[], %s::text[], %s::text[])
                            AS search_terms(term, prefix_pattern, pattern)
                   ) AS match_score,
                   coalesce(f.is_untracked, false) AS is_untracked,
                   coalesce(f.indexed_dirty, false) AS indexed_dirty,
                   left(r.display_content, 12000) AS snippet_raw
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
    "match_score",
    "tool",
    "rule_id",
    "severity",
    # embedding_text duplicates display_content minus the markdown frame and is
    # truncated mid-body — useful for debugging embedding similarity, noise for
    # navigation. Verbose mode keeps it.
    "embedding_text",
    "embedding_text_truncated",
})
_COMPACT_EDGE_KEYS = (
    "edge_type",
    "direction",
    "confidence_kind",
    "source_symbol",
    "target_symbol",
    "source_record_id",
    "target_record_id",
    "source_path",
    "target_path",
    "source_line_start",
    "source_line_end",
    "target_line_start",
    "target_line_end",
    "target_resolved",
    "target_kind",
    "edge_source",
)
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


def _display_content_body(raw: str | None) -> str | None:
    if not raw:
        return None
    m = _SNIPPET_FENCE_RE.search(raw)
    if m:
        return _SNIPPET_CLOSE_FENCE_RE.sub("", raw[m.end() :]).rstrip() or None
    return None


def _first_snippet_match(code: str, terms: tuple[str, ...]) -> int | None:
    if not terms:
        return None
    lower_code = code.casefold()
    preferred_terms = [term for term in terms if len(term) >= MIN_CENTERED_SNIPPET_TERM_CHARS] or list(terms)
    positions = [lower_code.find(term.casefold()) for term in preferred_terms if term]
    matches = [position for position in positions if position >= 0]
    return min(matches) if matches else None


def _centered_text_window(text: str, center: int | None, length: int) -> str | None:
    if center is None or len(text) <= length:
        return text[:length].rstrip()
    start = max(0, center - (length // 2))
    end = min(len(text), start + length)
    start = max(0, end - length)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    body_length = max(0, length - len(prefix) - len(suffix))
    if body_length != end - start:
        start = max(0, min(center - (body_length // 2), len(text) - body_length))
        end = min(len(text), start + body_length)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].rstrip()}{suffix}" or None


def _extract_snippet(
    raw: str | None,
    length: int = DEFAULT_SNIPPET_LENGTH,
    terms: tuple[str, ...] = (),
) -> str | None:
    """Return a bounded code-body snippet, centered on a matched search term when available."""
    code = _display_content_body(raw)
    if code is None:
        return None
    return _centered_text_window(code, _first_snippet_match(code, terms), length)


def _row_text(row: db.DbRow, key: str) -> str | None:
    value = row.get(key)
    return value if isinstance(value, str) else None


def _compact_record(
    row: db.DbRow,
    snippet_length: int = DEFAULT_SNIPPET_LENGTH,
    snippet_terms: tuple[str, ...] = (),
) -> dict[str, object]:
    snippet = _extract_snippet(_row_text(row, "snippet_raw"), snippet_length, snippet_terms)
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


def _verbose_record(
    row: db.DbRow,
    snippet_length: int = DEFAULT_SNIPPET_LENGTH,
    snippet_terms: tuple[str, ...] = (),
) -> dict[str, object]:
    snippet = _extract_snippet(_row_text(row, "snippet_raw"), snippet_length, snippet_terms)
    out = {k: v for k, v in row.items() if k not in {"snippet_raw", "match_score"}}
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
    snippet_terms: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    fmt = _verbose_record if verbose else _compact_record
    return [fmt(row, snippet_length, snippet_terms) for row in rows]


def _compact_edge(row: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key in _COMPACT_EDGE_KEYS:
        value = row.get(key)
        if value is not None and not _is_compact_noise(key, value):
            out[key] = value
    return out


def _format_edges(rows: Sequence[Mapping[str, object]], *, verbose: bool) -> list[dict[str, object]]:
    if verbose:
        return [dict(row) for row in rows]
    return [_compact_edge(row) for row in rows]


@dataclass(frozen=True)
class TextSearchPlan:
    query: str | None
    strategy: SearchQueryStrategy
    terms: tuple[str, ...]
    limit: int


@dataclass(frozen=True)
class RelatedQueryContext:
    record_id: str | None
    symbol: str | None
    direction: RelatedDirection
    scoped_record_ids: tuple[str, ...]
    parent_record_ids: frozenset[str]


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


def identifier_like_single_term(terms: tuple[str, ...]) -> bool:
    if len(terms) != 1:
        return False
    term = terms[0]
    if not IDENTIFIER_QUERY_RE.fullmatch(term):
        return False
    return any(char in term for char in "$_./:+@-") or any(char.isdigit() or char.isupper() for char in term)


def append_default_mixed_search_exclusions(args: Json, clauses: list[str], alias: str) -> None:
    if optional_text(args, "record_type") or optional_text(args, "content_class"):
        return
    clauses.extend(
        f"{alias}.record_type <> '{record_type}'" for record_type in sorted(DEFAULT_MIXED_SEARCH_EXCLUDED_RECORD_TYPES)
    )


def like_pattern_for_term(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def prefix_like_pattern_for_term(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"


def match_score_params(terms: tuple[str, ...]) -> QueryParams:
    return [
        list(terms),
        [prefix_like_pattern_for_term(term) for term in terms],
        [like_pattern_for_term(term) for term in terms],
    ]


def run_text_search_query(
    conn: db.DbConnection,
    args: Json,
    plan: TextSearchPlan,
) -> list[db.DbRow]:
    clauses, filter_params = code_intel_clauses(args, "r")
    if plan.query:
        append_default_mixed_search_exclusions(args, clauses, "r")
    if plan.query and plan.strategy == "websearch":
        clauses.append("r.search_document @@ websearch_to_tsquery('english', %s)")
        query_sql = query_with_where(
            CODE_INTEL_RECORD_SELECT_WEBSEARCH,
            clauses,
            """
            ORDER BY rank DESC, match_score DESC, r.updated_at DESC
            LIMIT %s
            """,
        )
        params: QueryParams = [plan.query, *match_score_params(plan.terms), *filter_params, plan.query, plan.limit]
    elif plan.query and plan.strategy in {"all_terms", "all_terms_fallback", "any_terms", "any_terms_fallback"}:
        require_all = plan.strategy in {"all_terms", "all_terms_fallback"}
        patterns = [like_pattern_for_term(term) for term in plan.terms]
        clauses.append(ALL_TERMS_SEARCH_CLAUSE if require_all else ANY_TERMS_SEARCH_CLAUSE)
        if require_all:
            # Every result matches all terms, so rank is constant and not useful.
            # Use NULL rank and sort by recency instead.
            query_sql = query_with_where(
                CODE_INTEL_RECORD_SELECT_MATCHED_LIST,
                clauses,
                """
            ORDER BY match_score DESC, r.updated_at DESC
            LIMIT %s
            """,
            )
            params = [*match_score_params(plan.terms), *filter_params, patterns, plan.limit]
        else:
            query_sql = query_with_where(
                CODE_INTEL_RECORD_SELECT_TERMS,
                clauses,
                """
            ORDER BY rank DESC, match_score DESC, r.updated_at DESC
            LIMIT %s
            """,
            )
            params = [patterns, *match_score_params(plan.terms), *filter_params, patterns, plan.limit]
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


def _annotate_status_snapshots(snapshot_rows: list[db.DbRow]) -> list[Json]:
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


def _compact_status_snapshots(snapshots: list[Json], *, omit_collection: bool) -> list[Json]:
    compact: list[Json] = []
    for snapshot in snapshots:
        item: Json = {}
        for key in _COMPACT_STATUS_SNAPSHOT_KEYS:
            if omit_collection and key == "collection":
                continue
            if key in snapshot:
                item[key] = snapshot[key]
        compact.append(item)
    return compact


def _status_rows_for_response(rows: list[db.DbRow], *, omit_collection: bool) -> list[Json]:
    if not omit_collection:
        return [{key: cast("JsonValue", value) for key, value in row.items()} for row in rows]
    result: list[Json] = []
    for row in rows:
        item: Json = {}
        for key, value in row.items():
            if key != "collection":
                item[key] = cast("JsonValue", value)
        result.append(item)
    return result


def _status_json_rows_for_response(rows: list[Json], *, omit_collection: bool) -> list[Json]:
    if not omit_collection:
        return rows
    result: list[Json] = []
    for row in rows:
        item = dict(row)
        _ = item.pop("collection", None)
        result.append(item)
    return result


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


def _positive_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return 0


def _snapshot_embed_record_types(snapshots: list[Json]) -> set[str]:
    record_types: set[str] = set()
    for snapshot in snapshots:
        metadata = snapshot.get("metadata")
        if not isinstance(metadata, dict):
            continue
        values = cast("dict[object, object]", metadata).get("embed_record_types")
        if not isinstance(values, list):
            continue
        record_types.update(item for item in cast("list[object]", values) if isinstance(item, str) and item)
    return record_types


def _status_queryability(
    snapshots: list[Json],
    records_by_type: list[db.DbRow],
    edges_by_type: list[db.DbRow],
    *,
    include_details: bool,
) -> Json:
    text_record_types = sorted({
        str(row.get("record_type"))
        for row in records_by_type
        if row.get("record_type") and _positive_int(row.get("count"))
    })
    semantic_record_types = sorted({
        str(row.get("record_type"))
        for row in records_by_type
        if row.get("record_type") and _positive_int(row.get("embedded_records"))
    })
    configured_embed_record_types = sorted(_snapshot_embed_record_types(snapshots))
    empty_embed_record_types = sorted(set(configured_embed_record_types) - set(semantic_record_types))
    edge_types = sorted({
        str(row.get("edge_type")) for row in edges_by_type if row.get("edge_type") and _positive_int(row.get("edges"))
    })
    queryability: dict[str, JsonValue] = {
        "text_record_type_count": len(text_record_types),
        "semantic_record_type_count": len(semantic_record_types),
        "text_only_record_type_count": len(set(text_record_types) - set(semantic_record_types)),
        "configured_embed_record_type_count": len(configured_embed_record_types),
        "empty_embed_record_type_count": len(empty_embed_record_types),
        "edge_type_count": len(edge_types),
        "has_text": bool(text_record_types),
        "has_semantic": bool(semantic_record_types),
        "has_edges": bool(edge_types),
    }
    if include_details:
        queryability.update({
            "text_record_types": text_record_types,
            "semantic_record_types": semantic_record_types,
            "text_only_record_types": sorted(set(text_record_types) - set(semantic_record_types)),
            "configured_embed_record_types": configured_embed_record_types,
            "empty_embed_record_types": empty_embed_record_types,
            "edge_types": edge_types,
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


def _status_include_flags(args: Json) -> StatusIncludeFlags:
    verbose = optional_bool(args, "verbose") or False
    return StatusIncludeFlags(
        verbose=verbose,
        snapshots=verbose or (optional_bool(args, "include_snapshots") or False),
        record_types=verbose or (optional_bool(args, "include_record_types") or False),
        queryability=verbose or (optional_bool(args, "include_queryability") or False),
        breakdowns=verbose or (optional_bool(args, "include_breakdowns") or False),
        static_summary=verbose or (optional_bool(args, "include_static_summary") or False),
    )


def _status_scope_response(args: Json) -> tuple[Json, bool]:
    result = snapshot_scope_response(args)
    collection = scoped_collection(args)
    if collection:
        result["collection"] = collection
    repo = optional_text(args, "repo")
    if repo:
        result["repo"] = repo
    return result, collection is not None


def _load_status_rows(
    conn: db.DbConnection,
    filters: StatusFilters,
    includes: StatusIncludeFlags,
    directory_depth: int,
) -> StatusRows:
    schema_versions = (
        schema_migration_versions(conn) if table_regclass_exists(conn, "project_code_intel_schema_migrations") else []
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
    )


def tool_code_intel_status(args: Json) -> Json:
    filters = status_filters(args)
    directory_depth = require_int(args, "directory_depth", 1, 1, 5)
    includes = _status_include_flags(args)
    scope_response, collection_scoped = _status_scope_response(args)
    omit_scoped_collection = collection_scoped and not includes.verbose
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"schema_present": False})
        validate_explicit_snapshot_id(conn, args)
        rows = _load_status_rows(conn, filters, includes, directory_depth)
    queryability = _status_queryability(
        rows.snapshots,
        rows.records_by_type,
        rows.edge_types,
        include_details=includes.queryability,
    )
    response: dict[str, object] = {
        "schema_present": True,
        "schema_versions": rows.schema_versions,
        **scope_response,
        "snapshots": (
            _status_json_rows_for_response(rows.snapshots, omit_collection=omit_scoped_collection)
            if includes.snapshots
            else _compact_status_snapshots(rows.snapshots, omit_collection=omit_scoped_collection)
        ),
        "files": _status_rows_for_response(rows.files, omit_collection=omit_scoped_collection),
        "records": _status_rows_for_response(rows.records, omit_collection=omit_scoped_collection),
        "edges": _status_rows_for_response(rows.edges, omit_collection=omit_scoped_collection),
        "queryability": queryability,
    }
    if includes.record_types:
        response["records_by_type"] = _status_rows_for_response(
            rows.records_by_type,
            omit_collection=omit_scoped_collection,
        )
    if rows.breakdowns is not None:
        response["language_breakdown"] = rows.breakdowns["language"]
        response["directory_breakdown"] = rows.breakdowns["directory"]
    if rows.static_rows is not None:
        static_runs, static_findings = rows.static_rows
        response["static_runs"] = static_runs
        response["static_findings"] = static_findings
    return ok(response)


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
    elif identifier_like_single_term(terms):
        strategy = "all_terms"
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


def related_direction(args: Json) -> RelatedDirection:
    value = optional_text(args, "direction") or "any"
    if value in {"any", "incoming", "outgoing"}:
        return cast("RelatedDirection", value)
    raise McpProtocolError("direction must be one of: any, incoming, outgoing")


def related_record_ids(conn: db.DbConnection, args: Json, record_id: str) -> tuple[list[str], set[str]]:
    lookup_args = {key: args[key] for key in ("collection", "repo", "snapshot_id", "include_historical") if key in args}
    clauses, params = code_intel_clauses(lookup_args, "r")
    clauses.append("r.record_id = %s")
    params.append(record_id)
    row = conn.execute(
        db.query_sql(
            query_with_where(
                """
            SELECT r.parent_record_id
            FROM project_code_intel_records r
            LEFT JOIN project_code_intel_files f ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
            """,
                clauses,
                """
            ORDER BY r.updated_at DESC, r.id DESC
            LIMIT 1
            """,
            )
        ),
        params,
    ).fetchone()
    parent_id = row.get("parent_record_id") if row is not None else None
    if isinstance(parent_id, str) and parent_id and parent_id != record_id:
        return [record_id, parent_id], {parent_id}
    return [record_id], set()


def related_record_clause(direction: RelatedDirection) -> str:
    if direction == "outgoing":
        return "e.source_record_id = ANY(%s)"
    if direction == "incoming":
        return "e.target_record_id = ANY(%s)"
    return "(e.source_record_id = ANY(%s) OR e.target_record_id = ANY(%s))"


def related_symbol_clause(direction: RelatedDirection) -> str:
    if direction == "outgoing":
        return "e.source_symbol = %s"
    if direction == "incoming":
        return "e.target_symbol = %s"
    return "(e.source_symbol = %s OR e.target_symbol = %s)"


def related_clause_params(direction: RelatedDirection, value: object) -> QueryParams:
    return [value] if direction != "any" else [value, value]


def annotate_related_edges(
    rows: list[db.DbRow],
    *,
    context: RelatedQueryContext,
) -> list[dict[str, object]]:
    edges = [dict(row) for row in rows]
    for edge in edges:
        source = edge.get("source_record_id")
        target = edge.get("target_record_id")
        edge_direction = related_edge_direction(edge, context=context)
        target_resolved = related_edge_target_resolved(edge)
        edge["direction"] = edge_direction
        edge["target_resolved"] = target_resolved
        edge["target_kind"] = "project_symbol" if target_resolved else unresolved_edge_target_kind(edge)
        if context.record_id and context.parent_record_ids:
            edge["edge_source"] = (
                "parent_record"
                if source in context.parent_record_ids or target in context.parent_record_ids
                else "record"
            )
    edges.sort(
        key=lambda edge: (
            related_edge_sort_priority(edge, context=context),
            -_positive_int(edge.get("id")),
        )
    )
    return edges


def unresolved_edge_target_kind(edge: Mapping[str, object]) -> str:
    metadata = edge.get("metadata")
    if isinstance(metadata, Mapping) and cast("Mapping[object, object]", metadata).get("call_kind") == "member_call":
        return "member_call"
    return "unresolved"


def related_edge_target_resolved(edge: Mapping[str, object]) -> bool:
    if isinstance(edge.get("target_resolved"), bool):
        return bool(edge["target_resolved"])
    return edge.get("target_record_db_id") is not None


def related_edge_direction(
    edge: Mapping[str, object],
    *,
    context: RelatedQueryContext,
) -> str:
    source = edge.get("source_record_id")
    target = edge.get("target_record_id")
    if context.scoped_record_ids:
        record_ids = set(context.scoped_record_ids)
        if source in record_ids:
            return "outgoing"
        if target in record_ids:
            return "incoming"
    if context.symbol:
        if edge.get("target_symbol") == context.symbol:
            return "incoming"
        if edge.get("source_symbol") == context.symbol:
            return "outgoing"
    return "outgoing"


def related_edge_sort_priority(
    edge: Mapping[str, object],
    *,
    context: RelatedQueryContext,
) -> int:
    resolved = related_edge_target_resolved(edge)
    edge_direction = str(edge.get("direction") or "outgoing")
    if context.direction != "any":
        return 0 if resolved else 1
    if context.symbol and not context.scoped_record_ids:
        base_priority = 0 if edge_direction == "incoming" else 1
    else:
        base_priority = 0 if edge_direction == "outgoing" else 1
    return base_priority if resolved else base_priority + 2


def related_order_clause(
    *,
    symbol: str | None,
    scoped_record_ids: list[str],
    direction: RelatedDirection,
) -> tuple[str, QueryParams]:
    if direction == "any" and symbol and not scoped_record_ids:
        return (
            """
            CASE
                WHEN e.target_symbol = %s AND tgt.id IS NOT NULL THEN 0
                WHEN e.source_symbol = %s AND tgt.id IS NOT NULL THEN 1
                WHEN e.target_symbol = %s THEN 2
                ELSE 3
            END,
            (tgt.id IS NOT NULL) DESC,
            e.id DESC
            """,
            [symbol, symbol, symbol],
        )
    if direction == "any" and scoped_record_ids:
        return (
            """
            CASE
                WHEN e.source_record_id = ANY(%s) AND tgt.id IS NOT NULL THEN 0
                WHEN e.target_record_id = ANY(%s) AND tgt.id IS NOT NULL THEN 1
                WHEN e.source_record_id = ANY(%s) THEN 2
                ELSE 3
            END,
            (tgt.id IS NOT NULL) DESC,
            e.id DESC
            """,
            [scoped_record_ids, scoped_record_ids, scoped_record_ids],
        )
    return "(tgt.id IS NOT NULL) DESC, e.id DESC", []


def related_base_edge_filters(args: Json, direction: RelatedDirection) -> tuple[list[str], QueryParams]:
    clauses = ["TRUE", "(e.target_record_id IS NULL OR e.source_record_id != e.target_record_id)"]
    params: QueryParams = []
    symbol = optional_text(args, "symbol")
    collection = scoped_collection(args)
    repo = optional_text(args, "repo")
    edge_type = optional_text(args, "edge_type")
    confidence_kind = optional_text(args, "confidence_kind")
    if symbol:
        clauses.append(related_symbol_clause(direction))
        params.extend(related_clause_params(direction, symbol))
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
    return clauses, params


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
        "results": cast(
            "JsonValue",
            _format_records(rows, verbose=verbose, snippet_length=snippet_length, snippet_terms=terms),
        ),
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
    append_default_mixed_search_exclusions(args, clauses, "r")
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
    direction = related_direction(args)
    clauses, params = related_base_edge_filters(args, direction)

    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        validate_explicit_snapshot_id(conn, args)
        parent_record_ids: set[str] = set()
        scoped_record_ids: list[str] = []
        if record_id:
            scoped_record_ids, parent_record_ids = related_record_ids(conn, args, record_id)
            clauses.append(related_record_clause(direction))
            params.extend(related_clause_params(direction, scoped_record_ids))
        order_clause, order_params = related_order_clause(
            symbol=symbol,
            scoped_record_ids=scoped_record_ids,
            direction=direction,
        )
        params.extend(order_params)
        params.append(limit)
        edges = conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT e.id, e.snapshot_id, e.collection, e.repo, e.commit_sha,
                   e.source_record_id, e.target_record_id, e.edge_type,
                   e.source_symbol, e.target_symbol, e.source_path, e.target_path,
                   e.confidence_kind, e.metadata,
                   tgt.id IS NOT NULL AS target_resolved,
                   CASE WHEN tgt.id IS NOT NULL THEN 'project_symbol' ELSE 'unresolved' END AS target_kind,
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
                    f"""
            ORDER BY {order_clause}
            LIMIT %s
            """,
                )
            ),
            params,
        ).fetchall()
    verbose = optional_bool(args, "verbose") or False
    related_context = RelatedQueryContext(
        record_id=record_id,
        symbol=symbol,
        direction=direction,
        scoped_record_ids=tuple(scoped_record_ids),
        parent_record_ids=frozenset(parent_record_ids),
    )
    annotated_edges = annotate_related_edges(
        edges,
        context=related_context,
    )
    return ok({
        **snapshot_scope_response(args),
        "edges": cast("JsonValue", _format_edges(annotated_edges, verbose=verbose)),
    })


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
