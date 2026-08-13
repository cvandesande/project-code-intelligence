"""MCP tool handlers for the code-intelligence database."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from project_code_intelligence import analyze, db, evidence
from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError
from project_code_intelligence.mcp import db as mcp_db
from project_code_intelligence.mcp.files import (
    LIST_CODE_INTEL_FILES_SELECT_FULL,
    LIST_CODE_INTEL_FILES_SELECT_SLIM,
    overconstrained_boolean_filter_warning,
)
from project_code_intelligence.mcp.filters import (
    code_intel_clauses,
    query_with_where,
    scoped_collection_repo_clauses,
    snapshot_scope_response,
    source_path_clauses,
    static_finding_clauses,
    status_filters,
)
from project_code_intelligence.mcp.formatting import (
    DEFAULT_SNIPPET_LENGTH,
    compact_file,
    compact_record,
    dedup_by_location,
    format_edges,
    format_records,
    verbose_file,
    verbose_record,
)
from project_code_intelligence.mcp.protocol import (
    Json,
    QueryParams,
    ok,
    optional_bool,
    optional_int,
    optional_text,
    require_int,
    scoped_collection,
)
from project_code_intelligence.mcp.records import (
    build_record_lookup,
    format_record_batch_response,
    get_record_ids_arg,
)
from project_code_intelligence.mcp.related import (
    RelatedDirection,
    RelatedQueryContext,
    annotate_related_edges,
    related_base_edge_filters,
    related_clause_params,
    related_direction,
    related_edge_warnings,
    related_order_clause,
    related_record_clause,
    related_record_ids,
)
from project_code_intelligence.mcp.scope import (
    attach_warnings,
    make_warning,
    repo_scope_exists,
    scope_filter_warnings,
)
from project_code_intelligence.mcp.search import (
    append_default_mixed_search_exclusions,
    execute_text_search,
    match_score_params,
    search_mode,
    search_query_mode,
    search_terms,
    text_search_warnings,
)
from project_code_intelligence.mcp.semantic import (
    diversify_semantic_rows,
    query_embedding,
    semantic_boost_terms,
    semantic_executable_symbol_distance_boost,
    semantic_filter_queryability_response,
    semantic_generated_distance_penalty,
    semantic_match_terms,
    semantic_non_source_distance_penalty,
    semantic_search_limit_plan,
    semantic_source_role_distance_boost,
    semantic_structural_symbol_distance_penalty,
    semantic_validation_distance_penalty,
)
from project_code_intelligence.mcp.static import (
    STATIC_FINDING_COMPACT_KEYS,
    STATIC_RULE_COMPACT_KEYS,
    compact_row,
    row_to_json,
    static_finding_warnings,
    static_run_scope_exists,
)
from project_code_intelligence.mcp.status import (
    active_index_run_rows,
    compact_status_snapshots,
    index_run_warnings,
    load_status_rows,
    server_runtime_identity,
    snapshot_scope_warning,
    status_include_flags,
    status_json_rows_for_response,
    status_queryability,
    status_repo_not_found,
    status_rows_for_response,
    status_scope_response,
    status_scope_warnings,
    status_snapshot_warnings,
)
from project_code_intelligence.mcp.tool_catalog import TOOL_DEFINITIONS, ToolDefinition
from project_code_intelligence.models import SOURCE_LANGUAGES
from project_code_intelligence.storage import row_int

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonValue


def tool_code_intel_status(args: Json) -> Json:
    filters = status_filters(args)
    directory_depth = require_int(args, "directory_depth", 1, 1, 5)
    includes = status_include_flags(args)
    scope_response, collection_scoped = status_scope_response(args)
    omit_scoped_collection = collection_scoped and not includes.verbose
    omit_scoped_repo = optional_text(args, "repo") is not None and not includes.verbose
    with mcp_db.connect() as conn:
        if not mcp_db.code_intel_tables_exist(conn):
            missing_schema_response: Json = {"schema_present": False}
            if includes.runtime:
                missing_schema_response["runtime"] = server_runtime_identity()
            return ok(missing_schema_response)
        missing_snapshot_warning = snapshot_scope_warning(conn, args)
        rows = load_status_rows(conn, filters, includes, directory_depth)
        active_runs = active_index_run_rows(conn, scoped_collection(args)) if includes.active_runs else None
    queryability = status_queryability(
        rows.snapshots,
        rows.records_by_type,
        rows.edge_types,
        rows.file_dimensions,
        include_details=includes.queryability,
    )
    response: Json = {
        "schema_present": True,
        "schema_versions": rows.schema_versions,
        **scope_response,
        "snapshots": cast(
            "JsonValue",
            status_json_rows_for_response(
                rows.snapshots,
                omit_collection=omit_scoped_collection,
                omit_repo=omit_scoped_repo,
            )
            if includes.snapshots and includes.verbose
            else compact_status_snapshots(
                rows.snapshots,
                omit_collection=omit_scoped_collection,
                omit_repo=omit_scoped_repo,
            ),
        ),
        "files": cast(
            "JsonValue",
            status_rows_for_response(
                rows.files,
                omit_collection=omit_scoped_collection,
                omit_repo=omit_scoped_repo,
            ),
        ),
        "records": cast(
            "JsonValue",
            status_rows_for_response(
                rows.records,
                omit_collection=omit_scoped_collection,
                omit_repo=omit_scoped_repo,
            ),
        ),
        "edges": cast(
            "JsonValue",
            status_rows_for_response(
                rows.edges,
                omit_collection=omit_scoped_collection,
                omit_repo=omit_scoped_repo,
            ),
        ),
        "queryability": queryability,
    }
    if includes.record_types:
        response["records_by_type"] = cast(
            "JsonValue",
            status_rows_for_response(
                rows.records_by_type,
                omit_collection=omit_scoped_collection,
                omit_repo=omit_scoped_repo,
            ),
        )
    if rows.breakdowns is not None:
        response["language_breakdown"] = cast("JsonValue", rows.breakdowns["language"])
        response["directory_breakdown"] = cast("JsonValue", rows.breakdowns["directory"])
    if rows.static_rows is not None:
        static_runs, static_findings = rows.static_rows
        response["static_runs"] = cast("JsonValue", static_runs)
        response["static_findings"] = cast("JsonValue", static_findings)
    if active_runs is not None:
        response["active_runs"] = cast("JsonValue", active_runs)
    if includes.runtime:
        response["runtime"] = server_runtime_identity()
    if status_repo_not_found(args, rows):
        response["found"] = False
    attach_warnings(
        response,
        [
            *status_snapshot_warnings(rows.snapshots),
            *status_scope_warnings(args, rows, missing_snapshot_warning=missing_snapshot_warning),
            *index_run_warnings(active_runs),
        ],
    )
    return ok(response)


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
        if not mcp_db.code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        missing_snapshot_warning = snapshot_scope_warning(conn, args)
        rows, strategy, fallback_reason = execute_text_search(conn, args, terms, limit)
        if not rows:
            repo_exists = repo_scope_exists(conn, args)
    if not optional_text(args, "record_type"):
        rows = dedup_by_location(rows)
    verbose = optional_bool(args, "verbose") or False
    response: Json = {
        "query": query,
        "query_strategy": strategy,
        **snapshot_scope_response(args),
        "results": cast(
            "JsonValue",
            format_records(rows, verbose=verbose, snippet_length=snippet_length, snippet_terms=terms),
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
        *text_search_warnings(query, strategy, fallback_reason, args, mode),
        *scope_filter_warnings(args, rows, repo_exists=repo_exists, missing_snapshot_warning=missing_snapshot_warning),
    ]
    if warnings:
        response["warnings"] = warnings
    return ok(response)


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
        if not mcp_db.code_intel_tables_exist(conn):
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
            format_records(rows, verbose=verbose, snippet_length=snippet_length, snippet_terms=lexical_terms),
        ),
    }
    attach_warnings(
        response,
        scope_filter_warnings(args, rows, repo_exists=repo_exists, missing_snapshot_warning=missing_snapshot_warning),
    )
    return ok(response)


def tool_get_code_intel_record(args: Json) -> Json:
    ids, batch = get_record_ids_arg(args)
    include_content = optional_bool(args, "include_content")
    include_metadata = optional_bool(args, "include_metadata")
    verbose = optional_bool(args, "verbose") or False
    query_sql, params = build_record_lookup(args, ids, batch=batch, include_content=include_content)
    with mcp_db.connect() as conn:
        if not mcp_db.code_intel_tables_exist(conn):
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
            result = verbose_record(row) if verbose else compact_record(row, include_metadata=include_metadata)
            return ok({"result": result})
        rows = cursor.fetchall()
    return ok(
        format_record_batch_response(
            rows,
            ids,
            verbose=verbose,
            include_metadata=include_metadata,
            missing_snapshot_warning=missing_snapshot_warning,
        )
    )


_RELATED_EDGES_SELECT = """
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
            """


@dataclass(frozen=True)
class _RelatedEdgesQuerySpec:
    """Per-direction inputs for `_run_related_edges_query`. Consolidated so the
    handler can pass shared scope (scoped_record_ids/symbol/limit) once instead
    of threading four parallel kwargs through every call.
    """

    direction: RelatedDirection
    scoped_record_ids: list[str]
    symbol: str | None
    limit: int


def _run_related_edges_query(
    conn: db.DbConnection,
    args: Json,
    spec: _RelatedEdgesQuerySpec,
) -> list[db.DbRow]:
    clauses, params = related_base_edge_filters(args, spec.direction)
    if spec.scoped_record_ids:
        clauses.append(related_record_clause(spec.direction))
        params.extend(related_clause_params(spec.direction, spec.scoped_record_ids))
    order_clause, order_params = related_order_clause(
        symbol=spec.symbol, scoped_record_ids=spec.scoped_record_ids, direction=spec.direction
    )
    params.extend(order_params)
    params.append(spec.limit)
    return conn.execute(
        db.query_sql(
            query_with_where(
                _RELATED_EDGES_SELECT,
                clauses,
                f"""
            ORDER BY {order_clause}
            LIMIT %s
            """,
            )
        ),
        params,
    ).fetchall()


def _interleave_related_edges(
    incoming: list[db.DbRow],
    outgoing: list[db.DbRow],
    *,
    limit: int,
) -> list[db.DbRow]:
    """Balance per-direction results so neither side starves the other within `limit`.

    Each input is pre-sorted resolved-first by the SQL ORDER BY. We alternate
    incoming/outgoing for fairness, dedup by edge id (self-edges are excluded
    by the base filter, but the same edge can surface in both queries when the
    record_id appears as both source and target across resolved pairs), and trim
    to the requested limit. When one side is exhausted, the other side fills the
    remaining slots.
    """
    seen: set[object] = set()
    merged: list[db.DbRow] = []
    i, j = 0, 0
    while len(merged) < limit and (i < len(incoming) or j < len(outgoing)):
        for stream, index_box in ((incoming, [i]), (outgoing, [j])):
            if len(merged) >= limit:
                break
            cursor = index_box[0]
            while cursor < len(stream):
                edge = stream[cursor]
                cursor += 1
                edge_id = edge.get("id")
                if edge_id in seen:
                    continue
                seen.add(edge_id)
                merged.append(edge)
                break
            if stream is incoming:
                i = cursor
            else:
                j = cursor
    return merged


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
    requested_limit = require_int(args, "limit", 20, 1, 100)

    with mcp_db.connect() as conn:
        if not mcp_db.code_intel_tables_exist(conn):
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
                        make_warning(
                            "record_not_found",
                            record_id=record_id,
                            message="record_id was not found in the selected code intelligence scope",
                        ),
                        *([missing_snapshot_warning] if missing_snapshot_warning is not None else []),
                    ],
                })
        if direction == "any":
            # Run each direction at the full limit, then interleave + trim. Fetching at the full
            # limit (rather than half) means a side with fewer matches doesn't artificially cap
            # the other; the merge balances actively-populated sides and falls back to single-side
            # output when only one direction has edges.
            incoming_edges = _run_related_edges_query(
                conn,
                args,
                _RelatedEdgesQuerySpec("incoming", scoped_record_ids, symbol, requested_limit),
            )
            outgoing_edges = _run_related_edges_query(
                conn,
                args,
                _RelatedEdgesQuerySpec("outgoing", scoped_record_ids, symbol, requested_limit),
            )
            edges = _interleave_related_edges(incoming_edges, outgoing_edges, limit=requested_limit)
        else:
            edges = _run_related_edges_query(
                conn,
                args,
                _RelatedEdgesQuerySpec(direction, scoped_record_ids, symbol, requested_limit),
            )
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
        "edges": cast("JsonValue", format_edges(annotated_edges, verbose=verbose)),
    }
    attach_warnings(
        response,
        [
            *related_edge_warnings(annotated_edges),
            *([missing_snapshot_warning] if missing_snapshot_warning is not None else []),
        ],
    )
    return ok(response)


def tool_blast_radius(args: Json) -> Json:
    symbol = optional_text(args, "symbol")
    source_path = optional_text(args, "source_path")
    if not symbol and not source_path:
        raise McpProtocolError("symbol or source_path is required")
    query = evidence.EvidenceQuery(
        symbol=symbol,
        source_path=source_path,
        line=optional_int(args, "line"),
        neighbors=require_int(args, "neighbors", 3, 0, 20),
        collection=optional_text(args, "collection"),
        repo=optional_text(args, "repo"),
    )
    with mcp_db.connect() as conn:
        if not mcp_db.code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        bundles = evidence.collect_evidence(conn, query)
    symbols = [evidence.bundle_to_json(bundle) for bundle in bundles]
    response: Json = {"found": bool(symbols), "count": len(symbols), "symbols": cast("JsonValue", symbols)}
    if not symbols:
        attach_warnings(
            response,
            [
                make_warning(
                    "symbol_not_found",
                    message="no matching definition in the selected code intelligence scope",
                    symbol=symbol or source_path,
                )
            ],
        )
    return ok(response)


def tool_search_static_findings(args: Json) -> Json:
    limit = require_int(args, "limit", 10, 1, 100)
    clauses, params = static_finding_clauses(args)
    params.append(limit)
    repo_exists: bool | None = None
    static_runs_found: bool | None = None
    missing_snapshot_warning: Json | None = None
    with mcp_db.connect() as conn:
        if not mcp_db.table_regclass_exists(conn, "project_code_intel_static_findings"):
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
    warnings = scope_filter_warnings(
        args, rows, repo_exists=repo_exists, missing_snapshot_warning=missing_snapshot_warning
    )
    if static_runs_found is False:
        warnings.append(
            make_warning(
                "static_analysis_not_run",
                message=(
                    "no static-analysis runs matched this scope; empty results do not mean a scanner found zero issues"
                ),
            )
        )
    if warnings:
        response["warnings"] = warnings
    return ok(response)


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
        if not mcp_db.table_regclass_exists(conn, "project_code_intel_static_findings"):
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


def tool_find_redundancy(args: Json) -> Json:
    options = analyze.AnalysisOptions(
        limit=require_int(args, "limit", 10, 1, 50),
        path_prefix=optional_text(args, "source_path_prefix"),
    )
    collection = optional_text(args, "collection")
    repo = optional_text(args, "repo")
    branch = optional_text(args, "branch")
    with mcp_db.connect() as conn:
        if not mcp_db.code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        all_snapshots = analyze.latest_snapshots(conn)
        snapshots = analyze.select_snapshots(all_snapshots, collection=collection, repo=repo, branch=branch)
        if branch is None:
            # No branch pinned: collapse multi-branch history to one snapshot per repo,
            # same as before branch-keyed selection existed.
            snapshots = analyze.newest_per_repo(snapshots)
        results = [analyze.analyze_snapshot(conn, snapshot, options) for snapshot in snapshots]
    groups: list[Json] = []
    functions_analyzed = 0
    for result in results:
        functions_analyzed += result.functions_analyzed
        groups.extend(
            cast("Json", {"snapshot": result.label, **analyze.group_to_json(group)}) for group in result.groups
        )
    response: Json = {
        "found": bool(groups),
        "count": len(groups),
        "functions_analyzed": functions_analyzed,
        "groups": cast("JsonValue", groups),
    }
    if not snapshots:
        attach_warnings(response, [make_warning("empty_repo_scope", message="no snapshot matched the given scope")])
    return ok(response)


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
        if not mcp_db.code_intel_tables_exist(conn):
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
    files: list[object] = [verbose_file(row) for row in rows] if verbose else [compact_file(row) for row in rows]
    response: Json = {**snapshot_scope_response(args), "files": cast("JsonValue", files)}
    warnings = scope_filter_warnings(
        args, rows, repo_exists=repo_exists, missing_snapshot_warning=missing_snapshot_warning
    )
    false_filter_warning = overconstrained_boolean_filter_warning(args, rows)
    if false_filter_warning:
        warnings.append(false_filter_warning)
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
    "related_code_intel": (TOOL_DEFINITIONS["related_code_intel"], tool_related_code_intel),
    "blast_radius": (TOOL_DEFINITIONS["blast_radius"], tool_blast_radius),
    "find_redundancy": (TOOL_DEFINITIONS["find_redundancy"], tool_find_redundancy),
    "list_code_intel_files": (TOOL_DEFINITIONS["list_code_intel_files"], tool_list_code_intel_files),
    "search_static_findings": (TOOL_DEFINITIONS["search_static_findings"], tool_search_static_findings),
    "get_static_finding": (TOOL_DEFINITIONS["get_static_finding"], tool_get_static_finding),
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
