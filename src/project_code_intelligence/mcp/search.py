"""Text-search helpers for the code-intelligence MCP server.

Owns the SQL constants, query plan, and warning emitters that back
`tool_search_code_intel_text`. Kept separate from `tools.py` so the handler
file stays a thin glue layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from project_code_intelligence import db
from project_code_intelligence.exceptions import McpProtocolError
from project_code_intelligence.mcp.filters import (
    code_intel_clauses,
    query_with_where,
)
from project_code_intelligence.mcp.protocol import Json, QueryParams, optional_text
from project_code_intelligence.mcp.scope import make_warning

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

SEARCH_TERM_RE = re.compile(r"[A-Za-z0-9_$./:+@-]+")
IDENTIFIER_QUERY_RE = re.compile(r"[$A-Za-z_][$A-Za-z0-9_$./:+@-]*\Z")
REGEX_LIKE_QUERY_RE = re.compile(r"(\\[AbBdDsSwWZ]|\(\.\*\)|\.\*|\[[^\]]+\]|\{[0-9,]+\}|\^)")
SEARCH_OPERATOR_WORDS = frozenset({"and", "or", "not"})
DEFAULT_MIXED_SEARCH_EXCLUDED_RECORD_TYPES = frozenset({"security_pattern"})
SECURITY_PATTERN_QUERY_TERMS = frozenset({
    "cve",
    "cwe",
    "security",
    "security_pattern",
    "vulnerability",
    "vulnerabilities",
})

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
                           CASE WHEN coalesce(r.symbol, '') = search_terms.term THEN 120 ELSE 0 END
                         + CASE WHEN lower(coalesce(r.symbol, '')) = lower(search_terms.term) THEN 80 ELSE 0 END
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
                           CASE WHEN coalesce(r.symbol, '') = search_terms.term THEN 120 ELSE 0 END
                         + CASE WHEN lower(coalesce(r.symbol, '')) = lower(search_terms.term) THEN 80 ELSE 0 END
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
                           CASE WHEN coalesce(r.symbol, '') = search_terms.term THEN 120 ELSE 0 END
                         + CASE WHEN lower(coalesce(r.symbol, '')) = lower(search_terms.term) THEN 80 ELSE 0 END
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
        raise McpProtocolError(
            "mode=enumerate lists records by filters and must not be combined with query; "
            "omit mode for search, or use symbol/record_type/source_path filters"
        )
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


def query_implies_security_patterns(args: Json) -> bool:
    terms = {term.casefold() for term in search_terms(optional_text(args, "query") or "")}
    return bool(terms & SECURITY_PATTERN_QUERY_TERMS) or any(term.startswith(("cve-", "cwe-")) for term in terms)


def append_default_mixed_search_exclusions(args: Json, clauses: list[str], alias: str) -> None:
    if optional_text(args, "record_type") or query_implies_security_patterns(args):
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
            ORDER BY r.source_path ASC,
                     r.line_start ASC NULLS LAST,
                     r.line_end ASC NULLS LAST,
                     r.record_type ASC,
                     r.record_id ASC
            LIMIT %s
            """,
        )
        params = [*filter_params, plan.limit]
    return conn.execute(db.query_sql(query_sql), params).fetchall()


def execute_text_search(
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


def text_search_warnings(
    query: str | None,
    strategy: SearchQueryStrategy,
    fallback_reason: str | None,
    args: Json,
    mode: SearchMode,
) -> list[Json]:
    warnings: list[Json] = []
    if query and REGEX_LIKE_QUERY_RE.search(query):
        warnings.append(
            make_warning(
                "tokenized_text_search",
                message="text search is tokenized and ranked; regex syntax is treated as ordinary query text",
            )
        )
    if strategy.endswith("_fallback"):
        warnings.append(
            make_warning(
                "query_strategy_fallback",
                query_strategy=strategy,
                message="text search used a broader fallback strategy; ranking may be less precise",
                fallback_reason=fallback_reason,
            )
        )
    # Surface the silent search→enumerate switch: when mode is omitted and the query is empty,
    # search_mode() falls through to "enumerate" so the tool browses records by filter. That's
    # useful for ad hoc enumeration, but easy to hit by mistake when an LLM forgets to set query.
    if not query and mode == "enumerate" and optional_text(args, "mode") is None:
        warnings.append(
            make_warning(
                "mode_inferred_enumerate",
                message=(
                    "no query was supplied, so this call enumerated records matching the supplied filters "
                    "instead of searching. Set mode=search with a non-empty query for ranked text search."
                ),
            )
        )
    return warnings
