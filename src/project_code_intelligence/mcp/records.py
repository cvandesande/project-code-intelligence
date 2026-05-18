"""Record-fetch helpers for the code-intelligence MCP server.

Owns the SELECT projections and argument-parsing helpers behind
`tool_get_code_intel_record` (and its `record_ids` batch form). The handler
in `tools.py` keeps the connection/cursor orchestration; this module
supplies the SQL shape and response packaging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError
from project_code_intelligence.mcp.filters import query_with_where, scoped_collection_repo_clauses
from project_code_intelligence.mcp.formatting import compact_record
from project_code_intelligence.mcp.protocol import (
    Json,
    QueryParams,
    mcp_max_record_content_chars,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from project_code_intelligence import db
    from project_code_intelligence.models import JsonValue


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


def get_record_ids_arg(args: Json) -> tuple[list[str], bool]:
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


def build_record_lookup(args: Json, ids: list[str], *, batch: bool, include_content: bool) -> tuple[str, QueryParams]:
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


def format_record_batch_response(
    rows: Sequence[db.DbRow],
    ids: Sequence[str],
    *,
    verbose: bool,
    include_metadata: bool | None,
    missing_snapshot_warning: Json | None,
) -> Json:
    rows_by_record_id = {str(row["record_id"]): row for row in rows}
    ordered = (rows_by_record_id[rid] for rid in ids if rid in rows_by_record_id)
    formatted = [dict(row) if verbose else compact_record(row, include_metadata=include_metadata) for row in ordered]
    response: Json = {"results": cast("JsonValue", formatted)}
    missing = [rid for rid in ids if rid not in rows_by_record_id]
    if missing:
        response["missing"] = missing
    if missing_snapshot_warning is not None:
        response["warnings"] = [missing_snapshot_warning]
    return response
