"""MCP tool handlers for the code-intelligence database."""

from __future__ import annotations

from collections.abc import Callable

from project_code_intelligence import config, db, embeddings
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
from project_code_intelligence.storage import schema_migration_versions


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
        snapshots = conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT s.id, s.collection, s.repo, s.repo_role, s.branch, s.commit_sha,
                   s.tree_sha, s.dirty, s.created_at
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
            SELECT r.collection, r.repo, r.record_type, count(*) AS count
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
                   count(*) FILTER (WHERE f.skipped_reason IS NOT NULL) AS skipped_files
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
    limit = require_int(args, "limit", 10, 1, 50)
    clauses, filter_params = code_intel_clauses(args, "r")
    if query:
        clauses.append("r.search_document @@ websearch_to_tsquery('english', %s)")
        query_sql = query_with_where(
            """
            SELECT r.id, r.snapshot_id, r.collection, r.repo, r.repo_role, r.branch,
                   r.commit_sha, r.tree_sha, r.source_path, r.language, r.file_role,
                   r.content_class, r.record_type, r.record_id, r.parent_record_id,
                   r.title, r.summary, r.line_start, r.line_end, r.symbol,
                   r.symbol_kind, r.confidence_kind, r.confidence, r.tool,
                   r.rule_id, r.severity, r.metadata, r.updated_at,
                   r.embedding IS NOT NULL AS has_embedding,
                   ts_rank_cd(r.search_document, websearch_to_tsquery('english', %s)) AS rank
            FROM project_code_intel_records r
            """,
            clauses,
            """
            ORDER BY rank DESC, r.updated_at DESC
            LIMIT %s
            """,
        )
        params = [query, *filter_params, query, limit]
    else:
        query_sql = query_with_where(
            """
            SELECT r.id, r.snapshot_id, r.collection, r.repo, r.repo_role, r.branch,
                   r.commit_sha, r.tree_sha, r.source_path, r.language, r.file_role,
                   r.content_class, r.record_type, r.record_id, r.parent_record_id,
                   r.title, r.summary, r.line_start, r.line_end, r.symbol,
                   r.symbol_kind, r.confidence_kind, r.confidence, r.tool,
                   r.rule_id, r.severity, r.metadata, r.updated_at,
                   r.embedding IS NOT NULL AS has_embedding,
                   NULL::real AS rank
            FROM project_code_intel_records r
            """,
            clauses,
            """
            ORDER BY r.updated_at DESC
            LIMIT %s
            """,
        )
        params = [*filter_params, limit]

    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        rows = conn.execute(
            db.query_sql(query_sql),
            params,
        ).fetchall()
    return ok({"query": query, **snapshot_scope_response(args), "results": rows})


def vector_literal_dimensions(vector: str) -> int:
    inner = vector.strip().removeprefix("[").removesuffix("]").strip()
    return 0 if not inner else inner.count(",") + 1


def query_embedding(query: str) -> tuple[str, int]:
    endpoint = config.default_embedding_endpoint(local_default=True)
    if endpoint:
        model = config.default_embedding_endpoint_model(endpoint=endpoint)
        model = embeddings.resolve_embedding_endpoint_model(endpoint, model)
        vectors = embeddings.embed_with_endpoint(endpoint, [query], model, track_metrics=False)
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
                   r.severity, r.metadata, r.embedding <=> %s::vector AS distance
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
    return ok({
        "query": query,
        "embedding_dimensions": embedding_dimensions,
        **snapshot_scope_response(args),
        "results": rows,
    })


def record_projection_query(*, include_content: bool, filter_collection: bool = False) -> str:
    if include_content:
        if filter_collection:
            return """
                SELECT id, collection, repo, repo_role, branch, commit_sha, tree_sha,
                       source_path, language, file_role, content_class,
                       record_type, record_id, parent_record_id, title, summary,
                       left(embedding_text, %s) AS embedding_text,
                       coalesce(length(embedding_text), 0) > %s AS embedding_text_truncated,
                       left(display_content, %s) AS display_content,
                       coalesce(length(display_content), 0) > %s AS display_content_truncated,
                       false AS content_omitted,
                       line_start, line_end, symbol, symbol_kind, confidence_kind,
                       confidence, tool, rule_id, severity, analyzer, analyzer_version,
                       parser, parser_version, chunker_version, metadata, created_at,
                       updated_at, embedding IS NOT NULL AS has_embedding
                FROM project_code_intel_records
                WHERE id = %s AND collection = %s
                """
        return """
            SELECT id, collection, repo, repo_role, branch, commit_sha, tree_sha,
                   source_path, language, file_role, content_class,
                   record_type, record_id, parent_record_id, title, summary,
                   left(embedding_text, %s) AS embedding_text,
                   coalesce(length(embedding_text), 0) > %s AS embedding_text_truncated,
                   left(display_content, %s) AS display_content,
                   coalesce(length(display_content), 0) > %s AS display_content_truncated,
                   false AS content_omitted,
                   line_start, line_end, symbol, symbol_kind, confidence_kind,
                   confidence, tool, rule_id, severity, analyzer, analyzer_version,
                   parser, parser_version, chunker_version, metadata, created_at,
                   updated_at, embedding IS NOT NULL AS has_embedding
            FROM project_code_intel_records
            WHERE id = %s
            """
    if filter_collection:
        return """
            SELECT id, collection, repo, repo_role, branch, commit_sha, tree_sha,
                   source_path, language, file_role, content_class,
                   record_type, record_id, parent_record_id, title, summary,
                   NULL::text AS embedding_text,
                   false AS embedding_text_truncated,
                   NULL::text AS display_content,
                   false AS display_content_truncated,
                   true AS content_omitted,
                   line_start, line_end, symbol, symbol_kind, confidence_kind,
                   confidence, tool, rule_id, severity, analyzer, analyzer_version,
                   parser, parser_version, chunker_version, metadata, created_at,
                   updated_at, embedding IS NOT NULL AS has_embedding
            FROM project_code_intel_records
            WHERE id = %s AND collection = %s
            """
    return """
            SELECT id, collection, repo, repo_role, branch, commit_sha, tree_sha,
                   source_path, language, file_role, content_class,
                   record_type, record_id, parent_record_id, title, summary,
                   NULL::text AS embedding_text,
                   false AS embedding_text_truncated,
                   NULL::text AS display_content,
                   false AS display_content_truncated,
                   true AS content_omitted,
                   line_start, line_end, symbol, symbol_kind, confidence_kind,
                   confidence, tool, rule_id, severity, analyzer, analyzer_version,
                   parser, parser_version, chunker_version, metadata, created_at,
                   updated_at, embedding IS NOT NULL AS has_embedding
            FROM project_code_intel_records
            WHERE id = %s
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
    return ok({**snapshot_scope_response(args), "edges": edges})


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


def tool_get_static_finding(args: Json) -> Json:
    finding_id = args.get("id")
    if not isinstance(finding_id, int):
        raise McpProtocolTypeError("id must be an integer")
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
            return ok({"result": None})
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
        code_flows = conn.execute(
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
    return ok({"finding": finding, "rule": rule, "locations": locations, "code_flows": code_flows})


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
    return ok({"finding_id": finding_id, "flow_index": flow_index, "steps": rows})


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
    params.append(limit)

    with mcp_db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        rows = conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT f.id, f.snapshot_id, f.collection, f.repo, f.repo_role, f.branch,
                   f.commit_sha, f.tree_sha, f.source_path, f.git_blob_sha, f.file_sha256,
                   f.size_bytes, f.language, f.file_role, f.content_class,
                   f.is_generated, f.is_vendor, f.is_test, f.is_source, f.is_build,
                   f.is_config, f.is_doc, f.skipped_reason, f.metadata, f.created_at
            FROM project_code_intel_files f
            """,
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
