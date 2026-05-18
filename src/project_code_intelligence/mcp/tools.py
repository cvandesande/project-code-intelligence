"""MCP tool handlers for the code-intelligence database."""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import os
import re
import sys
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
    normalize_source_path_filter,
    query_with_where,
    scoped_collection_repo_clauses,
    scoped_snapshot_clauses,
    scoped_snapshot_table_collection_repo_clauses,
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
from project_code_intelligence.models import SOURCE_LANGUAGES
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
MIN_CENTERED_SNIPPET_TERM_CHARS = 3
SERVER_STARTED_AT = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
PACKAGE_NAME = "project-code-intelligence"

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
    "distance",
    "quality_penalty",
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
    *,
    include_metadata: bool = True,
) -> dict[str, object]:
    snippet = _extract_snippet(_row_text(row, "snippet_raw"), snippet_length, snippet_terms)
    out: dict[str, object] = {
        k: v
        for k, v in row.items()
        if not _is_compact_noise(k, v) and k not in _COMPACT_RECORD_STRIP and k != "snippet_raw"
    }
    if include_metadata:
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
    else:
        _ = out.pop("metadata", None)
    if snippet:
        out["snippet"] = snippet
    return out


def _verbose_record(
    row: db.DbRow,
    snippet_length: int = DEFAULT_SNIPPET_LENGTH,
    snippet_terms: tuple[str, ...] = (),
) -> dict[str, object]:
    snippet = _extract_snippet(_row_text(row, "snippet_raw"), snippet_length, snippet_terms)
    out = {k: v for k, v in row.items() if k not in {"snippet_raw", "match_score", "quality_penalty"}}
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


def _path_scope_matches_repo_root(args: Json) -> bool:
    raw_prefix = optional_text(args, "source_path_prefix")
    if not raw_prefix:
        return False
    prefix = normalize_source_path_filter(raw_prefix, "source_path_prefix")
    repo = optional_text(args, "repo")
    if repo and prefix == normalize_source_path_filter(repo, "repo"):
        return True
    collection = scoped_collection(args)
    return bool(collection and prefix == normalize_source_path_filter(collection, "collection"))


def repo_scope_exists(conn: db.DbConnection, args: Json) -> bool | None:
    if not optional_text(args, "repo"):
        return None
    clauses, params = scoped_snapshot_table_collection_repo_clauses(args, "s")
    return (
        conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT 1
            FROM project_code_intel_snapshots s
            """,
                    clauses,
                    """
            LIMIT 1
            """,
                )
            ),
            params,
        ).fetchone()
        is not None
    )


def _attach_warnings(response: dict[str, object], warnings: Sequence[Json]) -> None:
    """Set response['warnings'] when warnings is non-empty.

    Centralizes the conditional-assign pattern that several tools repeat, and keeps the
    `warnings` list out of the caller's local-variable count (matters because some handlers
    are right at PLR0914's threshold).
    """
    if warnings:
        response["warnings"] = cast("JsonValue", list(warnings))


def _empty_repo_scope_warning(repo: str) -> Json:
    return {
        "kind": "empty_repo_scope",
        "repo": repo,
        "message": "no results matched this repo filter; run code_intel_status without repo to see valid repo keys",
    }


# Enum-ish filter dimensions for which we emit `empty_<dim>_scope` warnings when a value is
# supplied but no rows match. Snapshot_id has its own warning shape (it's numeric, not enum).
_ENUM_FILTER_DIMENSIONS: tuple[str, ...] = ("language", "file_role", "record_type", "content_class")


def _empty_enum_scope_warning(dimension: str, value: str) -> Json:
    return {
        "kind": f"empty_{dimension}_scope",
        dimension: value,
        "message": (
            f"no results matched the {dimension}={value!r} filter; run code_intel_status with "
            f"include_queryability=true to see valid {dimension} values in this index"
        ),
    }


def _empty_snapshot_scope_warning(snapshot_id: int) -> Json:
    return {
        "kind": "empty_snapshot_scope",
        "snapshot_id": snapshot_id,
        "message": (
            f"snapshot_id={snapshot_id} does not exist in this index; "
            "run code_intel_status with include_snapshots=true to see valid snapshot ids"
        ),
    }


def _scope_filter_warnings(
    args: Json,
    rows: Sequence[object],
    *,
    repo_exists: bool | None = None,
    missing_snapshot_warning: Json | None = None,
) -> list[Json]:
    warnings: list[Json] = []
    source_path = optional_text(args, "source_path")
    source_path_prefix = optional_text(args, "source_path_prefix")
    repo = optional_text(args, "repo")
    if _path_scope_matches_repo_root(args):
        warnings.append({
            "kind": "repo_root_path_scope",
            "message": "source_path_prefix points at the repo root and is equivalent to a broad repo filter",
            "source_path_prefix": source_path_prefix,
        })
    if rows:
        return warnings
    if repo and repo_exists is False:
        warnings.append(_empty_repo_scope_warning(repo))
    if missing_snapshot_warning is not None:
        warnings.append(missing_snapshot_warning)
    for dimension in _ENUM_FILTER_DIMENSIONS:
        value = optional_text(args, dimension)
        if value:
            warnings.append(_empty_enum_scope_warning(dimension, value))
    if source_path or source_path_prefix:
        warning: Json = {
            "kind": "empty_path_scope",
            "message": (
                "no results matched this path scope; source_path and source_path_prefix are repo-relative, "
                "and directories should use source_path_prefix"
            ),
        }
        if source_path:
            warning["source_path"] = source_path
        if source_path_prefix:
            warning["source_path_prefix"] = source_path_prefix
        warnings.append(warning)
    return warnings


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


SEMANTIC_BOOST_STOP_WORDS = frozenset({
    "about",
    "after",
    "are",
    "before",
    "does",
    "happen",
    "happens",
    "how",
    "into",
    "that",
    "the",
    "then",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
})
MIN_SEMANTIC_BOOST_TERM_CHARS = 3
MAX_SEMANTIC_BOOST_TERMS = 8
SEMANTIC_EXECUTABLE_SYMBOL_DISTANCE_BOOST = 0.12
SEMANTIC_STRUCTURAL_SYMBOL_DISTANCE_PENALTY = 0.18
SEMANTIC_VALIDATION_DISTANCE_PENALTY = 0.16
SEMANTIC_SOURCE_ROLE_DISTANCE_BOOST = 0.16
SEMANTIC_NON_SOURCE_DISTANCE_PENALTY = 0.18
SEMANTIC_GENERATED_DISTANCE_PENALTY = 0.24
SEMANTIC_DIVERSITY_OVERFETCH_FACTOR = 4
SEMANTIC_DIVERSITY_MIN_EXTRA_ROWS = 20
SEMANTIC_DIVERSITY_MAX_SQL_LIMIT = 200
SEMANTIC_EXECUTABLE_QUERY_TERMS = frozenset({
    "add",
    "added",
    "adds",
    "build",
    "builds",
    "built",
    "config",
    "configuration",
    "configure",
    "configured",
    "configuring",
    "call",
    "called",
    "caller",
    "calls",
    "create",
    "creates",
    "creating",
    "emit",
    "emits",
    "emitted",
    "emitting",
    "execute",
    "executed",
    "executes",
    "flow",
    "generate",
    "generated",
    "generates",
    "generating",
    "generation",
    "handler",
    "handlers",
    "implement",
    "implementation",
    "implemented",
    "implements",
    "invoke",
    "invoked",
    "invokes",
    "logic",
    "render",
    "rendered",
    "rendering",
    "renders",
    "run",
    "runs",
    "translate",
    "translated",
    "translating",
    "translates",
    "workflow",
})
SEMANTIC_IMPLEMENTATION_SUPPLEMENTAL_TERMS = (
    "generate",
    "render",
    "build",
    "add",
    "config",
    "configuration",
    "template",
)
SEMANTIC_NON_SOURCE_QUERY_TERMS = frozenset({
    "doc",
    "docs",
    "documentation",
    "example",
    "examples",
    "fixture",
    "fixtures",
    "guide",
    "guides",
    "mock",
    "mocks",
    "readme",
    "spec",
    "specs",
    "test",
    "testing",
    "tests",
    "unit",
})
SEMANTIC_STRUCTURAL_QUERY_TERMS = frozenset({
    "api",
    "apis",
    "class",
    "classes",
    "crd",
    "crds",
    "definition",
    "definitions",
    "field",
    "fields",
    "interface",
    "interfaces",
    "model",
    "models",
    "schema",
    "schemas",
    "spec",
    "specs",
    "struct",
    "structs",
    "type",
    "types",
    "yaml",
})
SEMANTIC_VALIDATION_QUERY_TERMS = frozenset({
    "validate",
    "validated",
    "validates",
    "validating",
    "validation",
    "validator",
    "validators",
})


def semantic_query_terms(query: str) -> set[str]:
    return {term.casefold() for term in search_terms(query)}


def semantic_has_implementation_intent(args: Json, query: str) -> bool:
    if optional_text(args, "file_role"):
        return False
    content_class = optional_text(args, "content_class")
    if content_class and content_class != "source":
        return False
    query_terms = semantic_query_terms(query)
    if query_terms & (
        SEMANTIC_NON_SOURCE_QUERY_TERMS | SEMANTIC_STRUCTURAL_QUERY_TERMS | SEMANTIC_VALIDATION_QUERY_TERMS
    ):
        return False
    return bool(query_terms & SEMANTIC_EXECUTABLE_QUERY_TERMS)


def semantic_boost_terms(query: str) -> tuple[str, ...]:
    return tuple(
        term
        for term in search_terms(query)
        if len(term) >= MIN_SEMANTIC_BOOST_TERM_CHARS and term.casefold() not in SEMANTIC_BOOST_STOP_WORDS
    )[:MAX_SEMANTIC_BOOST_TERMS]


def semantic_match_terms(args: Json, query: str) -> tuple[str, ...]:
    terms = list(semantic_boost_terms(query))
    if semantic_has_implementation_intent(args, query):
        existing = {term.casefold() for term in terms}
        for term in SEMANTIC_IMPLEMENTATION_SUPPLEMENTAL_TERMS:
            if term not in existing:
                terms.append(term)
                existing.add(term)
    return tuple(terms)


def semantic_source_role_distance_boost(args: Json, query: str) -> float:
    if optional_text(args, "file_role"):
        return 0.0
    content_class = optional_text(args, "content_class")
    if content_class and content_class != "source":
        return 0.0
    query_terms = semantic_query_terms(query)
    if query_terms & SEMANTIC_NON_SOURCE_QUERY_TERMS:
        return 0.0
    return SEMANTIC_SOURCE_ROLE_DISTANCE_BOOST


def semantic_executable_symbol_distance_boost(args: Json, query: str) -> float:
    return SEMANTIC_EXECUTABLE_SYMBOL_DISTANCE_BOOST if semantic_has_implementation_intent(args, query) else 0.0


def semantic_structural_symbol_distance_penalty(args: Json, query: str) -> float:
    if any(optional_text(args, name) for name in ("source_path", "source_path_prefix", "file_role", "content_class")):
        return 0.0
    return SEMANTIC_STRUCTURAL_SYMBOL_DISTANCE_PENALTY if semantic_has_implementation_intent(args, query) else 0.0


def semantic_validation_distance_penalty(args: Json, query: str) -> float:
    if any(optional_text(args, name) for name in ("source_path", "source_path_prefix", "file_role", "content_class")):
        return 0.0
    return SEMANTIC_VALIDATION_DISTANCE_PENALTY if semantic_has_implementation_intent(args, query) else 0.0


def semantic_generated_distance_penalty(args: Json, query: str) -> float:
    if any(optional_text(args, name) for name in ("source_path", "source_path_prefix", "file_role", "content_class")):
        return 0.0
    query_terms = semantic_query_terms(query)
    if query_terms & SEMANTIC_NON_SOURCE_QUERY_TERMS:
        return 0.0
    return SEMANTIC_GENERATED_DISTANCE_PENALTY


def semantic_non_source_distance_penalty(args: Json, query: str) -> float:
    if any(optional_text(args, name) for name in ("source_path", "source_path_prefix", "file_role", "content_class")):
        return 0.0
    query_terms = semantic_query_terms(query)
    if query_terms & SEMANTIC_NON_SOURCE_QUERY_TERMS:
        return 0.0
    return SEMANTIC_NON_SOURCE_DISTANCE_PENALTY


def semantic_search_diversity_enabled(args: Json) -> bool:
    if "diversify" in args:
        return optional_bool(args, "diversify")
    return not (
        optional_text(args, "parent_record_id") or optional_text(args, "source_path") or optional_bool(args, "verbose")
    )


def semantic_search_sql_limit(limit: int, *, diversify: bool) -> int:
    if not diversify:
        return limit
    return min(
        max(limit * SEMANTIC_DIVERSITY_OVERFETCH_FACTOR, limit + SEMANTIC_DIVERSITY_MIN_EXTRA_ROWS),
        SEMANTIC_DIVERSITY_MAX_SQL_LIMIT,
    )


@dataclass(frozen=True)
class SemanticSearchLimitPlan:
    requested: int
    sql: int
    diversify: bool


def semantic_search_limit_plan(args: Json) -> SemanticSearchLimitPlan:
    requested = require_int(args, "limit", 10, 1, 50)
    diversify = semantic_search_diversity_enabled(args)
    return SemanticSearchLimitPlan(
        requested=requested,
        sql=semantic_search_sql_limit(requested, diversify=diversify),
        diversify=diversify,
    )


def semantic_diversity_key(row: Mapping[str, object]) -> str:
    return str(row.get("parent_record_id") or row.get("record_id") or "")


def diversify_semantic_rows(rows: list[db.DbRow], limit: int) -> list[db.DbRow]:
    seen: set[str] = set()
    primary: list[db.DbRow] = []
    siblings: list[db.DbRow] = []
    for row in rows:
        key = semantic_diversity_key(row)
        if key and key not in seen:
            seen.add(key)
            primary.append(row)
        else:
            siblings.append(row)
    return [*primary, *siblings][:limit]


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


def code_intel_tables_exist(conn: db.DbConnection) -> bool:
    return mcp_db.code_intel_tables_exist(conn)


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
    return _empty_snapshot_scope_warning(snapshot_id)


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


def _compact_status_snapshots(snapshots: list[Json], *, omit_collection: bool, omit_repo: bool) -> list[Json]:
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


def _status_rows_for_response(rows: list[db.DbRow], *, omit_collection: bool, omit_repo: bool) -> list[Json]:
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


def _status_json_rows_for_response(rows: list[Json], *, omit_collection: bool, omit_repo: bool) -> list[Json]:
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


def _status_snapshot_warnings(snapshots: list[Json]) -> list[Json]:
    warnings: list[Json] = []
    for snapshot in snapshots:
        status = snapshot.get("head_status")
        if status == "stale":
            warning: Json = {
                "kind": "snapshot_stale",
                "message": "snapshot is stale; verify with local source",
            }
        elif status == "unknown":
            warning = {
                "kind": "snapshot_freshness_unknown",
                "message": "snapshot freshness could not be checked against local source",
            }
        else:
            warning = {}
        if warning:
            _copy_snapshot_warning_fields(
                warning,
                snapshot,
                ("id", "collection", "repo", "commit_sha", "head_commit", "head_status_reason"),
            )
            warnings.append(warning)
        if snapshot.get("dirty") is True:
            dirty_warning: Json = {
                "kind": "snapshot_dirty",
                "message": "snapshot was indexed from a dirty working tree; verify dirty paths against local source",
                "dirty": True,
            }
            dirty_paths_count = _snapshot_dirty_paths_count(snapshot)
            if dirty_paths_count is not None:
                dirty_warning["dirty_paths_count"] = dirty_paths_count
            _copy_snapshot_warning_fields(
                dirty_warning,
                snapshot,
                ("id", "collection", "repo", "commit_sha", "head_commit", "head_status"),
            )
            warnings.append(dirty_warning)
    return warnings


def _status_repo_not_found(args: Json, rows: StatusRows) -> bool:
    return bool(
        optional_text(args, "repo") and not rows.snapshots and not rows.records and not rows.files and not rows.edges
    )


def _status_scope_warnings(args: Json, rows: StatusRows, *, missing_snapshot_warning: Json | None = None) -> list[Json]:
    warnings: list[Json] = []
    repo = optional_text(args, "repo")
    if repo and _status_repo_not_found(args, rows):
        warnings.append(_empty_repo_scope_warning(repo))
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
    file_dimensions: dict[str, list[str]] | None,
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


def _status_include_flags(args: Json) -> StatusIncludeFlags:
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


def _status_scope_response(args: Json) -> tuple[Json, bool]:
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


def _source_git_root(module_path: Path) -> Path | None:
    for candidate in (module_path.parent, *module_path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _source_git_commit(module_path: Path) -> str | None:
    git_root = _source_git_root(module_path)
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
        file_dimensions=_status_file_dimensions(conn, filters) if includes.queryability else None,
    )


def tool_code_intel_status(args: Json) -> Json:
    filters = status_filters(args)
    directory_depth = require_int(args, "directory_depth", 1, 1, 5)
    includes = _status_include_flags(args)
    scope_response, collection_scoped = _status_scope_response(args)
    omit_scoped_collection = collection_scoped and not includes.verbose
    omit_scoped_repo = optional_text(args, "repo") is not None and not includes.verbose
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            missing_schema_response: Json = {"schema_present": False}
            if includes.runtime:
                missing_schema_response["runtime"] = server_runtime_identity()
            return ok(missing_schema_response)
        missing_snapshot_warning = snapshot_scope_warning(conn, args)
        rows = _load_status_rows(conn, filters, includes, directory_depth)
    queryability = _status_queryability(
        rows.snapshots,
        rows.records_by_type,
        rows.edge_types,
        rows.file_dimensions,
        include_details=includes.queryability,
    )
    full_snapshots = includes.snapshots and includes.verbose
    response: dict[str, object] = {
        "schema_present": True,
        "schema_versions": rows.schema_versions,
        **scope_response,
        "snapshots": (
            _status_json_rows_for_response(
                rows.snapshots,
                omit_collection=omit_scoped_collection,
                omit_repo=omit_scoped_repo,
            )
            if full_snapshots
            else _compact_status_snapshots(
                rows.snapshots,
                omit_collection=omit_scoped_collection,
                omit_repo=omit_scoped_repo,
            )
        ),
        "files": _status_rows_for_response(
            rows.files,
            omit_collection=omit_scoped_collection,
            omit_repo=omit_scoped_repo,
        ),
        "records": _status_rows_for_response(
            rows.records,
            omit_collection=omit_scoped_collection,
            omit_repo=omit_scoped_repo,
        ),
        "edges": _status_rows_for_response(
            rows.edges,
            omit_collection=omit_scoped_collection,
            omit_repo=omit_scoped_repo,
        ),
        "queryability": queryability,
    }
    if includes.record_types:
        response["records_by_type"] = _status_rows_for_response(
            rows.records_by_type,
            omit_collection=omit_scoped_collection,
            omit_repo=omit_scoped_repo,
        )
    if rows.breakdowns is not None:
        response["language_breakdown"] = rows.breakdowns["language"]
        response["directory_breakdown"] = rows.breakdowns["directory"]
    if rows.static_rows is not None:
        static_runs, static_findings = rows.static_rows
        response["static_runs"] = static_runs
        response["static_findings"] = static_findings
    if includes.runtime:
        response["runtime"] = server_runtime_identity()
    if _status_repo_not_found(args, rows):
        response["found"] = False
    _attach_warnings(
        response,
        [
            *_status_snapshot_warnings(rows.snapshots),
            *_status_scope_warnings(args, rows, missing_snapshot_warning=missing_snapshot_warning),
        ],
    )
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


def _text_search_warnings(
    query: str | None,
    strategy: SearchQueryStrategy,
    fallback_reason: str | None,
    args: Json,
    mode: SearchMode,
) -> list[Json]:
    warnings: list[Json] = []
    if query and REGEX_LIKE_QUERY_RE.search(query):
        warnings.append({
            "kind": "tokenized_text_search",
            "message": "text search is tokenized and ranked; regex syntax is treated as ordinary query text",
        })
    if strategy.endswith("_fallback"):
        warning: Json = {
            "kind": "query_strategy_fallback",
            "query_strategy": strategy,
            "message": "text search used a broader fallback strategy; ranking may be less precise",
        }
        if fallback_reason:
            warning["fallback_reason"] = fallback_reason
        warnings.append(warning)
    # Surface the silent search→enumerate switch: when mode is omitted and the query is empty,
    # search_mode() falls through to "enumerate" so the tool browses records by filter. That's
    # useful for ad hoc enumeration, but easy to hit by mistake when an LLM forgets to set query.
    if not query and mode == "enumerate" and optional_text(args, "mode") is None:
        warnings.append({
            "kind": "mode_inferred_enumerate",
            "message": (
                "no query was supplied, so this call enumerated records matching the supplied filters "
                "instead of searching. Set mode=search with a non-empty query for ranked text search."
            ),
        })
    return warnings


def related_direction(args: Json) -> RelatedDirection:
    value = optional_text(args, "direction") or "any"
    if value in {"any", "incoming", "outgoing"}:
        return cast("RelatedDirection", value)
    raise McpProtocolError("direction must be one of: any, incoming, outgoing")


def related_record_ids(conn: db.DbConnection, args: Json, record_id: str) -> tuple[list[str], set[str], bool]:
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
    if row is None:
        return [record_id], set(), False
    parent_id = row.get("parent_record_id")
    if isinstance(parent_id, str) and parent_id and parent_id != record_id:
        return [record_id, parent_id], {parent_id}, True
    return [record_id], set(), True


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


def _related_edge_warnings(edges: Sequence[Mapping[str, object]]) -> list[Json]:
    if not any(edge.get("confidence_kind") == "heuristic_candidate" for edge in edges):
        return []
    return [
        {
            "kind": "heuristic_candidate_relationships",
            "confidence_kind": "heuristic_candidate",
            "message": "related_code_intel returns heuristic candidates; verify important relationships in source",
        }
    ]


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
    include_unresolved = optional_bool(args, "include_unresolved")
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
    if not include_unresolved:
        clauses.append("(e.confidence_kind <> 'heuristic_candidate' OR e.target_record_id IS NOT NULL)")
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
    repo_exists: bool | None = None
    missing_snapshot_warning: Json | None = None
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        missing_snapshot_warning = snapshot_scope_warning(conn, args)
        rows, strategy, fallback_reason = _execute_text_search(conn, args, terms, limit)
        if not rows:
            repo_exists = repo_scope_exists(conn, args)
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
    warnings = [
        *_text_search_warnings(query, strategy, fallback_reason, args, mode),
        *_scope_filter_warnings(args, rows, repo_exists=repo_exists, missing_snapshot_warning=missing_snapshot_warning),
    ]
    if warnings:
        response["warnings"] = warnings
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
        "PCI_EMBEDDING_ENDPOINT to a trusted OpenAI-compatible "
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


def semantic_filter_queryability_warning(conn: db.DbConnection, args: Json) -> Json | None:
    record_type = optional_text(args, "record_type")
    if not record_type:
        return None
    clauses, params = code_intel_clauses(args, "r")
    row = conn.execute(
        db.query_sql(
            query_with_where(
                """
            SELECT count(*) AS record_count,
                   count(r.embedding) AS embedded_records
            FROM project_code_intel_records r
            LEFT JOIN project_code_intel_files f ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
            """,
                clauses,
                "",
            )
        ),
        params,
    ).fetchone()
    if row is None:
        return None
    record_count = row_int(row, "record_count")
    embedded_records = row_int(row, "embedded_records")
    if record_count <= 0 or embedded_records > 0:
        return None
    return {
        "kind": "semantic_filter_has_no_embeddings",
        "record_type": record_type,
        "message": (
            "semantic search only searches embedded records; this filter matches records in the text index "
            "but none have embeddings. Use search_code_intel_text or remove the non-embedded filter."
        ),
    }


def semantic_filter_queryability_response(args: Json, query: str) -> Json | None:
    if not optional_text(args, "record_type"):
        return None
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return {"error": "code intelligence schema is not initialized"}
        missing_snapshot_warning = snapshot_scope_warning(conn, args)
        warning = semantic_filter_queryability_warning(conn, args)
    if not warning:
        return None
    warnings: list[Json] = [warning]
    if missing_snapshot_warning is not None:
        warnings.append(missing_snapshot_warning)
    return {
        "query": query,
        **snapshot_scope_response(args),
        "results": [],
        "warnings": warnings,
    }


def tool_search_code_intel_semantic(args: Json) -> Json:
    query = optional_text(args, "query")
    if not query:
        raise McpProtocolError("query is required")
    snippet_length = require_int(args, "snippet_length", DEFAULT_SNIPPET_LENGTH, 1, 800)
    queryability_response = semantic_filter_queryability_response(args, query)
    if queryability_response is not None:
        return ok(queryability_response)
    clauses, params = code_intel_clauses(args, "r")
    clauses.append("r.embedding IS NOT NULL")
    append_default_mixed_search_exclusions(args, clauses, "r")
    embedding, embedding_dimensions = query_embedding(query)
    lexical_terms = semantic_boost_terms(query)
    limit_plan = semantic_search_limit_plan(args)
    query_params = [
        embedding,
        *match_score_params(semantic_match_terms(args, query)),
        *params,
        semantic_executable_symbol_distance_boost(args, query),
        semantic_structural_symbol_distance_penalty(args, query),
        semantic_validation_distance_penalty(args, query),
        semantic_source_role_distance_boost(args, query),
        semantic_non_source_distance_penalty(args, query),
        semantic_generated_distance_penalty(args, query),
        limit_plan.sql,
    ]
    repo_exists: bool | None = None
    missing_snapshot_warning: Json | None = None
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        missing_snapshot_warning = snapshot_scope_warning(conn, args)
        rows = conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT *
            FROM (
            SELECT r.id, r.snapshot_id, r.collection, r.repo, r.repo_role,
                   r.branch, r.commit_sha, r.source_path, r.language, r.file_role,
                   r.content_class, r.record_type, r.record_id, r.parent_record_id,
                   r.title, r.summary, r.line_start, r.line_end, r.symbol,
                   r.symbol_kind, r.confidence_kind, r.confidence, r.tool, r.rule_id,
                   r.severity, r.updated_at, r.embedding <=> %s::vector AS distance,
                   coalesce(f.is_generated, false) AS is_generated,
                   (
                       SELECT coalesce(sum(
                           CASE WHEN coalesce(r.symbol, '') = search_terms.term THEN 120 ELSE 0 END
                         + CASE WHEN lower(coalesce(r.symbol, '')) = lower(search_terms.term) THEN 80 ELSE 0 END
                         + CASE WHEN lower(coalesce(r.title, '')) = lower(search_terms.term) THEN 32 ELSE 0 END
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
                             WHEN split_part(coalesce(r.embedding_text, ''), E'content:\\n', 2)
                                  ILIKE search_terms.pattern ESCAPE '\\' THEN 8
                             ELSE 0
                           END
                         + CASE
                             WHEN coalesce(r.display_content, '') ILIKE search_terms.pattern ESCAPE '\\' THEN 1
                             ELSE 0
                           END
                       ), 0)::real
                       FROM unnest(%s::text[], %s::text[], %s::text[])
                            AS search_terms(term, prefix_pattern, pattern)
                   ) AS match_score,
                   CASE
                     WHEN r.record_type = 'code_chunk'
                      AND r.metadata->>'fallback_reason' = 'coverage line window'
                      AND r.line_start IS NOT NULL
                      AND r.line_end IS NOT NULL
                      AND r.line_end - r.line_start <= 2
                      AND length(btrim(split_part(coalesce(r.embedding_text, ''), E'content:\\n', 2))) < 120
                     THEN 0.35::real
                     ELSE 0::real
                   END AS quality_penalty,
                   left(r.display_content, 800) AS snippet_raw
            FROM project_code_intel_records r
            LEFT JOIN project_code_intel_files f ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
            """,
                    clauses,
                    """
            ) ranked
            ORDER BY (
                         ranked.distance
                         + ranked.quality_penalty
                         - LEAST(ranked.match_score, 80) * 0.01
                         - CASE
                             WHEN ranked.symbol_kind IN ('function', 'method', 'shell_function') THEN %s::real
                             ELSE 0
                           END
                         + CASE
                             WHEN ranked.symbol_kind IN ('struct', 'interface', 'type') THEN %s::real
                             ELSE 0
                           END
                         + CASE
                             WHEN coalesce(ranked.symbol, '') ILIKE '%%validat%%'
                               OR coalesce(ranked.title, '') ILIKE '%%validat%%'
                               OR coalesce(ranked.source_path, '') ILIKE '%%validat%%'
                               OR coalesce(ranked.record_id, '') ILIKE '%%validat%%'
                             THEN %s::real
                             ELSE 0
                           END
                         - CASE WHEN ranked.file_role = 'source' THEN %s::real ELSE 0 END
                         + CASE WHEN ranked.content_class <> 'source' THEN %s::real ELSE 0 END
                         + CASE WHEN ranked.is_generated THEN %s::real ELSE 0 END
                     ) ASC,
                     ranked.distance ASC,
                     ranked.updated_at DESC
            LIMIT %s
            """,
                )
            ),
            query_params,
        ).fetchall()
        if not rows:
            repo_exists = repo_scope_exists(conn, args)
    rows = diversify_semantic_rows(rows, limit_plan.requested) if limit_plan.diversify else rows[: limit_plan.requested]
    verbose = optional_bool(args, "verbose") or False
    response: Json = {
        "query": query,
        "embedding_dimensions": embedding_dimensions,
        **snapshot_scope_response(args),
        "results": cast(
            "JsonValue",
            _format_records(rows, verbose=verbose, snippet_length=snippet_length, snippet_terms=lexical_terms),
        ),
    }
    _attach_warnings(
        response,
        _scope_filter_warnings(args, rows, repo_exists=repo_exists, missing_snapshot_warning=missing_snapshot_warning),
    )
    return ok(response)


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
        # Use .strip() in the non-empty check so whitespace-only strings (e.g. "   ") are rejected
        # here instead of silently falling through to a {found: false} miss. Pydantic's min_length
        # constraint passes whitespace because it measures raw length.
        if not isinstance(record_id, str) or not record_id.strip():
            raise McpProtocolTypeError("record_id must be a non-empty string")
        return [record_id], False
    if record_ids is None:
        raise McpProtocolError("record_id or record_ids is required")
    if not isinstance(record_ids, list) or not record_ids:
        raise McpProtocolTypeError("record_ids must be a non-empty list of strings")
    out: list[str] = []
    for item in cast("list[object]", record_ids):
        if not isinstance(item, str) or not item.strip():
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


def _format_record_batch_response(
    rows: Sequence[db.DbRow],
    ids: Sequence[str],
    *,
    verbose: bool,
    include_metadata: bool | None,
    missing_snapshot_warning: Json | None,
) -> Json:
    rows_by_record_id = {str(row["record_id"]): row for row in rows}
    ordered = (rows_by_record_id[rid] for rid in ids if rid in rows_by_record_id)
    formatted = [dict(row) if verbose else _compact_record(row, include_metadata=include_metadata) for row in ordered]
    response: dict[str, object] = {"results": cast("JsonValue", formatted)}
    missing = [rid for rid in ids if rid not in rows_by_record_id]
    if missing:
        response["missing"] = missing
    if missing_snapshot_warning is not None:
        response["warnings"] = [missing_snapshot_warning]
    return response


def tool_get_code_intel_record(args: Json) -> Json:
    ids, batch = _get_record_ids_arg(args)
    include_content = optional_bool(args, "include_content")
    include_metadata = optional_bool(args, "include_metadata")
    verbose = optional_bool(args, "verbose") or False
    query_sql, params = _build_record_lookup(args, ids, batch=batch, include_content=include_content)
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        missing_snapshot_warning = snapshot_scope_warning(conn, args)
        cursor = conn.execute(db.query_sql(query_sql), params)
        if not batch:
            row = cursor.fetchone()
            if row is None:
                response: Json = {"found": False}
                if missing_snapshot_warning is not None:
                    response["warnings"] = [missing_snapshot_warning]
                return ok(response)
            return ok({"result": dict(row) if verbose else _compact_record(row, include_metadata=include_metadata)})
        rows = cursor.fetchall()
    return ok(
        _format_record_batch_response(
            rows,
            ids,
            verbose=verbose,
            include_metadata=include_metadata,
            missing_snapshot_warning=missing_snapshot_warning,
        )
    )


def tool_related_code_intel(args: Json) -> Json:
    record_id = optional_text(args, "record_id")
    symbol = optional_text(args, "symbol")
    if not record_id and not symbol:
        raise McpProtocolError("record_id or symbol is required")
    if record_id and symbol:
        # Match the source_path / source_path_prefix mutex pattern: reject ambiguity rather than
        # silently picking record_id (the historical behavior, which made symbol look ignored).
        raise McpProtocolError("provide exactly one of record_id or symbol")
    direction = related_direction(args)
    clauses, params = related_base_edge_filters(args, direction)

    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        missing_snapshot_warning = snapshot_scope_warning(conn, args)
        parent_record_ids: set[str] = set()
        scoped_record_ids: list[str] = []
        if record_id:
            scoped_record_ids, parent_record_ids, record_found = related_record_ids(conn, args, record_id)
            if not record_found:
                return ok({
                    **snapshot_scope_response(args),
                    "found": False,
                    "edges": [],
                    "warnings": [
                        {
                            "kind": "record_not_found",
                            "record_id": record_id,
                            "message": "record_id was not found in the selected code intelligence scope",
                        },
                        *([missing_snapshot_warning] if missing_snapshot_warning is not None else []),
                    ],
                })
            clauses.append(related_record_clause(direction))
            params.extend(related_clause_params(direction, scoped_record_ids))
        order_clause, order_params = related_order_clause(
            symbol=symbol,
            scoped_record_ids=scoped_record_ids,
            direction=direction,
        )
        params.extend(order_params)
        params.append(require_int(args, "limit", 20, 1, 100))
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
    annotated_edges = annotate_related_edges(
        edges,
        context=RelatedQueryContext(
            record_id=record_id,
            symbol=symbol,
            direction=direction,
            scoped_record_ids=tuple(scoped_record_ids),
            parent_record_ids=frozenset(parent_record_ids),
        ),
    )
    response: Json = {
        **snapshot_scope_response(args),
        "edges": cast("JsonValue", _format_edges(annotated_edges, verbose=verbose)),
    }
    _attach_warnings(
        response,
        [
            *_related_edge_warnings(annotated_edges),
            *([missing_snapshot_warning] if missing_snapshot_warning is not None else []),
        ],
    )
    return ok(response)


def static_run_scope_exists(conn: db.DbConnection, args: Json) -> bool:
    clauses = ["TRUE"]
    params: QueryParams = []
    collection = scoped_collection(args)
    if collection:
        clauses.append("r.collection = %s")
        params.append(collection)
    repo = optional_text(args, "repo")
    if repo:
        clauses.append("r.repo = %s")
        params.append(repo)
    tool = optional_text(args, "tool")
    if tool:
        clauses.append("r.tool_name = %s")
        params.append(tool)
    snapshot_clauses, snapshot_params = scoped_snapshot_clauses(args, "r")
    clauses.extend(snapshot_clauses)
    params.extend(snapshot_params)
    return (
        conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT 1
            FROM project_code_intel_static_runs r
            """,
                    clauses,
                    """
            LIMIT 1
            """,
                )
            ),
            params,
        ).fetchone()
        is not None
    )


def tool_search_static_findings(args: Json) -> Json:
    limit = require_int(args, "limit", 10, 1, 100)
    clauses, params = static_finding_clauses(args)
    params.append(limit)
    repo_exists: bool | None = None
    static_runs_found: bool | None = None
    missing_snapshot_warning: Json | None = None
    with mcp_db.connect() as conn:
        if not table_regclass_exists(conn, "project_code_intel_static_findings"):
            return ok({"error": "static-analysis schema is not initialized"})
        missing_snapshot_warning = snapshot_scope_warning(conn, args)
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
        if not rows:
            repo_exists = repo_scope_exists(conn, args)
            static_runs_found = static_run_scope_exists(conn, args)
    response: Json = {**snapshot_scope_response(args), "results": cast("JsonValue", rows)}
    if static_runs_found is not None:
        response["static_runs_found"] = static_runs_found
    warnings = _scope_filter_warnings(
        args, rows, repo_exists=repo_exists, missing_snapshot_warning=missing_snapshot_warning
    )
    if static_runs_found is False:
        warnings.append({
            "kind": "static_analysis_not_run",
            "message": (
                "no static-analysis runs matched this scope; empty results do not mean a scanner found zero issues"
            ),
        })
    if warnings:
        response["warnings"] = warnings
    return ok(response)


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


LIST_CODE_INTEL_FILES_SELECT_SLIM = """
            WITH record_backed_files AS (
                SELECT
                    NULL::bigint AS id,
                    r.snapshot_id,
                    r.collection,
                    r.repo,
                    r.repo_role,
                    r.branch,
                    r.commit_sha,
                    r.tree_sha,
                    r.source_path,
                    NULL::text AS git_blob_sha,
                    max(r.file_sha256) AS file_sha256,
                    NULL::bigint AS size_bytes,
                    (array_agg(r.language ORDER BY r.id))[1] AS language,
                    (array_agg(r.file_role ORDER BY r.id))[1] AS file_role,
                    (array_agg(r.content_class ORDER BY r.id))[1] AS content_class,
                    bool_or(r.file_role = 'generated' OR r.content_class = 'generated') AS is_generated,
                    bool_or(r.file_role = 'vendor' OR r.content_class = 'vendor') AS is_vendor,
                    bool_or(r.file_role = 'test' OR r.content_class = 'test') AS is_test,
                    bool_or(
                        r.file_role IN ('source', 'source-include')
                        OR r.content_class = 'source'
                        OR r.language = ANY(%s::text[])
                    ) AS is_source,
                    bool_or(
                        r.file_role IN ('build', 'build-include', 'build-script', 'package', 'project-manifest')
                        OR r.content_class = 'build'
                    ) AS is_build,
                    bool_or(r.file_role = 'config' OR r.content_class = 'config') AS is_config,
                    bool_or(r.file_role = 'doc' OR r.content_class = 'doc' OR r.language = 'doc') AS is_doc,
                    NULL::text AS skipped_reason,
                    false AS is_untracked,
                    false AS indexed_dirty,
                    jsonb_build_object('inventory_source', 'records') AS metadata,
                    min(r.created_at) AS created_at
                FROM project_code_intel_records r
                LEFT JOIN project_code_intel_files existing
                  ON existing.snapshot_id = r.snapshot_id
                 AND existing.source_path = r.source_path
                WHERE existing.id IS NULL
                GROUP BY
                    r.snapshot_id,
                    r.collection,
                    r.repo,
                    r.repo_role,
                    r.branch,
                    r.commit_sha,
                    r.tree_sha,
                    r.source_path
            ),
            file_inventory AS (
                SELECT
                    f.id,
                    f.snapshot_id,
                    f.collection,
                    f.repo,
                    f.repo_role,
                    f.branch,
                    f.commit_sha,
                    f.tree_sha,
                    f.source_path,
                    f.git_blob_sha,
                    f.file_sha256,
                    f.size_bytes,
                    f.language,
                    f.file_role,
                    f.content_class,
                    f.is_generated,
                    f.is_vendor,
                    f.is_test,
                    f.is_source,
                    f.is_build,
                    f.is_config,
                    f.is_doc,
                    f.skipped_reason,
                    f.is_untracked,
                    f.indexed_dirty,
                    f.metadata,
                    f.created_at
                FROM project_code_intel_files f
                UNION ALL
                SELECT
                    id,
                    snapshot_id,
                    collection,
                    repo,
                    repo_role,
                    branch,
                    commit_sha,
                    tree_sha,
                    source_path,
                    git_blob_sha,
                    file_sha256,
                    size_bytes,
                    language,
                    file_role,
                    content_class,
                    is_generated,
                    is_vendor,
                    is_test,
                    is_source,
                    is_build,
                    is_config,
                    is_doc,
                    skipped_reason,
                    is_untracked,
                    indexed_dirty,
                    metadata,
                    created_at
                FROM record_backed_files
            )
            SELECT f.id, f.source_path, f.size_bytes, f.language, f.file_role, f.content_class,
                   f.is_generated, f.is_vendor, f.is_test, f.is_source, f.is_build,
                   f.is_config, f.is_doc, f.skipped_reason
            FROM file_inventory f
            """


LIST_CODE_INTEL_FILES_SELECT_FULL = """
            WITH record_backed_files AS (
                SELECT
                    NULL::bigint AS id,
                    r.snapshot_id,
                    r.collection,
                    r.repo,
                    r.repo_role,
                    r.branch,
                    r.commit_sha,
                    r.tree_sha,
                    r.source_path,
                    NULL::text AS git_blob_sha,
                    max(r.file_sha256) AS file_sha256,
                    NULL::bigint AS size_bytes,
                    (array_agg(r.language ORDER BY r.id))[1] AS language,
                    (array_agg(r.file_role ORDER BY r.id))[1] AS file_role,
                    (array_agg(r.content_class ORDER BY r.id))[1] AS content_class,
                    bool_or(r.file_role = 'generated' OR r.content_class = 'generated') AS is_generated,
                    bool_or(r.file_role = 'vendor' OR r.content_class = 'vendor') AS is_vendor,
                    bool_or(r.file_role = 'test' OR r.content_class = 'test') AS is_test,
                    bool_or(
                        r.file_role IN ('source', 'source-include')
                        OR r.content_class = 'source'
                        OR r.language = ANY(%s::text[])
                    ) AS is_source,
                    bool_or(
                        r.file_role IN ('build', 'build-include', 'build-script', 'package', 'project-manifest')
                        OR r.content_class = 'build'
                    ) AS is_build,
                    bool_or(r.file_role = 'config' OR r.content_class = 'config') AS is_config,
                    bool_or(r.file_role = 'doc' OR r.content_class = 'doc' OR r.language = 'doc') AS is_doc,
                    NULL::text AS skipped_reason,
                    false AS is_untracked,
                    false AS indexed_dirty,
                    jsonb_build_object('inventory_source', 'records') AS metadata,
                    min(r.created_at) AS created_at
                FROM project_code_intel_records r
                LEFT JOIN project_code_intel_files existing
                  ON existing.snapshot_id = r.snapshot_id
                 AND existing.source_path = r.source_path
                WHERE existing.id IS NULL
                GROUP BY
                    r.snapshot_id,
                    r.collection,
                    r.repo,
                    r.repo_role,
                    r.branch,
                    r.commit_sha,
                    r.tree_sha,
                    r.source_path
            ),
            file_inventory AS (
                SELECT
                    f.id,
                    f.snapshot_id,
                    f.collection,
                    f.repo,
                    f.repo_role,
                    f.branch,
                    f.commit_sha,
                    f.tree_sha,
                    f.source_path,
                    f.git_blob_sha,
                    f.file_sha256,
                    f.size_bytes,
                    f.language,
                    f.file_role,
                    f.content_class,
                    f.is_generated,
                    f.is_vendor,
                    f.is_test,
                    f.is_source,
                    f.is_build,
                    f.is_config,
                    f.is_doc,
                    f.skipped_reason,
                    f.is_untracked,
                    f.indexed_dirty,
                    f.metadata,
                    f.created_at
                FROM project_code_intel_files f
                UNION ALL
                SELECT
                    id,
                    snapshot_id,
                    collection,
                    repo,
                    repo_role,
                    branch,
                    commit_sha,
                    tree_sha,
                    source_path,
                    git_blob_sha,
                    file_sha256,
                    size_bytes,
                    language,
                    file_role,
                    content_class,
                    is_generated,
                    is_vendor,
                    is_test,
                    is_source,
                    is_build,
                    is_config,
                    is_doc,
                    skipped_reason,
                    is_untracked,
                    indexed_dirty,
                    metadata,
                    created_at
                FROM record_backed_files
            )
            SELECT f.id, f.snapshot_id, f.collection, f.repo, f.repo_role, f.branch,
                   f.commit_sha, f.tree_sha, f.source_path, f.git_blob_sha, f.file_sha256,
                   f.size_bytes, f.language, f.file_role, f.content_class,
                   f.is_generated, f.is_vendor, f.is_test, f.is_source, f.is_build,
                   f.is_config, f.is_doc, f.skipped_reason, f.is_untracked,
                   f.indexed_dirty, f.metadata, f.created_at
            FROM file_inventory f
            """


LIST_CODE_INTEL_FILES_BOOLEAN_FILTERS = (
    "is_test",
    "is_doc",
    "is_generated",
    "is_vendor",
    "is_source",
    "is_build",
    "is_config",
    "is_untracked",
)
OVERCONSTRAINED_FALSE_BOOLEAN_FILTER_MIN_COUNT = 5
OVERCONSTRAINED_FALSE_BOOLEAN_FILTER_WITH_PATH_MIN_COUNT = 2


def _explicit_false_file_boolean_filters(args: Json) -> list[str]:
    return [
        arg_name
        for arg_name in LIST_CODE_INTEL_FILES_BOOLEAN_FILTERS
        if arg_name in args and not optional_bool(args, arg_name)
    ]


def _looks_like_overconstrained_boolean_filter(args: Json, false_filter_count: int) -> bool:
    if false_filter_count >= OVERCONSTRAINED_FALSE_BOOLEAN_FILTER_MIN_COUNT:
        return True
    path_scoped = (
        optional_text(args, "source_path") is not None or optional_text(args, "source_path_prefix") is not None
    )
    return path_scoped and false_filter_count >= OVERCONSTRAINED_FALSE_BOOLEAN_FILTER_WITH_PATH_MIN_COUNT


def _overconstrained_boolean_filter_warning(args: Json, rows: Sequence[object]) -> Json | None:
    if rows:
        return None
    filters = _explicit_false_file_boolean_filters(args)
    if not _looks_like_overconstrained_boolean_filter(args, len(filters)):
        return None
    return {
        "kind": "overconstrained_boolean_filters",
        "message": (
            "Omit boolean filters unless you want to filter for that exact boolean value; "
            "false is an active filter, not a default."
        ),
        "filters": filters,
    }


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
    query_params = [sorted(SOURCE_LANGUAGES), *params, limit]

    repo_exists: bool | None = None
    missing_snapshot_warning: Json | None = None
    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        missing_snapshot_warning = snapshot_scope_warning(conn, args)
        rows = conn.execute(
            db.query_sql(
                query_with_where(
                    LIST_CODE_INTEL_FILES_SELECT_FULL if verbose else LIST_CODE_INTEL_FILES_SELECT_SLIM,
                    clauses,
                    """
            ORDER BY f.source_path
            LIMIT %s
            """,
                )
            ),
            query_params,
        ).fetchall()
        if not rows:
            repo_exists = repo_scope_exists(conn, args)
    files: list[object] = list(rows) if verbose else [_compact_file(row) for row in rows]
    response: Json = {**snapshot_scope_response(args), "files": cast("JsonValue", files)}
    warnings = _scope_filter_warnings(
        args, rows, repo_exists=repo_exists, missing_snapshot_warning=missing_snapshot_warning
    )
    false_filter_warning = _overconstrained_boolean_filter_warning(args, rows)
    if false_filter_warning:
        warnings.append(false_filter_warning)
    if warnings:
        response["warnings"] = warnings
    return ok(response)


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

    repo_exists: bool | None = None
    missing_snapshot_warning: Json | None = None
    with mcp_db.connect() as conn:
        if not table_regclass_exists(conn, "project_code_intel_parser_failures"):
            return ok({"error": "code intelligence schema is not initialized"})
        missing_snapshot_warning = snapshot_scope_warning(conn, args)
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
        if not rows:
            repo_exists = repo_scope_exists(conn, args)
    response: Json = {**snapshot_scope_response(args), "parser_failures": cast("JsonValue", rows)}
    warnings = _scope_filter_warnings(
        args, rows, repo_exists=repo_exists, missing_snapshot_warning=missing_snapshot_warning
    )
    if warnings:
        response["warnings"] = warnings
    return ok(response)


ToolHandler = Callable[[Json], Json]
ToolRegistry = dict[str, tuple[ToolDefinition, ToolHandler]]


TOOLS: ToolRegistry = {
    "code_intel_status": (TOOL_DEFINITIONS["code_intel_status"], tool_code_intel_status),
    "search_code_intel_text": (TOOL_DEFINITIONS["search_code_intel_text"], tool_search_code_intel_text),
    "search_code_intel_semantic": (TOOL_DEFINITIONS["search_code_intel_semantic"], tool_search_code_intel_semantic),
    "get_code_intel_record": (TOOL_DEFINITIONS["get_code_intel_record"], tool_get_code_intel_record),
    "get_code_intel_records": (TOOL_DEFINITIONS["get_code_intel_records"], tool_get_code_intel_record),
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
