#!/usr/bin/env python3
"""Minimal stdio MCP server for a code-intelligence database."""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, cast

from project_code_intelligence import config, db, embeddings, llama_embed
from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError, McpWritePermissionError
from project_code_intelligence.mcp_filters import (
    StatusFilters,
    code_intel_clauses,
    query_with_where,
    scoped_snapshot_clauses,
    snapshot_scope_response,
    static_finding_clauses,
    status_filters,
)
from project_code_intelligence.mcp_protocol import (
    Json,
    QueryParams,
    log,
    mcp_debug_errors,
    mcp_max_batch_items,
    mcp_max_request_bytes,
    ok,
    optional_text,
    require_int,
    scoped_collection,
)
from project_code_intelligence.mcp_tool_catalog import TOOL_DEFINITIONS, ToolDefinition
from project_code_intelligence.storage import schema_migration_versions

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject, JsonValue

PROTOCOL_VERSION = "2024-11-05"


def code_intel_tables_exist(conn: db.DbConnection) -> bool:
    row = conn.execute(
        """
        SELECT to_regclass('public.project_code_intel_records') IS NOT NULL AS exists
        """
    ).fetchone()
    return bool(db.require_row(row, "code-intel table existence")["exists"])


def table_regclass_exists(conn: db.DbConnection, table: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s) IS NOT NULL AS exists", [f"public.{table}"]).fetchone()
    return bool(db.require_row(row, "table existence")["exists"])


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
    with db.connect() as conn:
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
            """,
                )
            ),
            filters.snapshots.params,
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

    with db.connect() as conn:
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
        vectors = embeddings.embed_with_endpoint(endpoint, [query], model, track_metrics=False)
        if not vectors:
            raise McpProtocolError("embedding endpoint returned no query vector")
        vector = vectors[0]
        return vector, vector_literal_dimensions(vector)
    embedding_values = llama_embed.embed_text(query)
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
    with db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        rows = conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT r.id, r.snapshot_id, r.collection, r.repo, r.repo_role,
                   r.branch, r.commit_sha, r.source_path, r.language, r.file_role,
                   r.content_class, r.record_type, r.record_id, r.title, r.summary,
                   r.line_start, r.line_end, r.symbol, r.symbol_kind,
                   r.confidence_kind, r.metadata, r.embedding <=> %s::vector AS distance
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


def tool_get_code_intel_record(args: Json) -> Json:
    record_id = args.get("id")
    if not isinstance(record_id, int):
        raise McpProtocolTypeError("id must be an integer")
    with db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        row = conn.execute(
            """
            SELECT id, collection, repo, repo_role, branch, commit_sha, tree_sha,
                   source_path, language, file_role, content_class,
                   record_type, record_id, parent_record_id, title, summary,
                   embedding_text, display_content, line_start, line_end,
                   symbol, symbol_kind, confidence_kind, confidence, tool,
                   rule_id, severity, analyzer, analyzer_version, parser,
                   parser_version, chunker_version, metadata, created_at,
                   updated_at, embedding IS NOT NULL AS has_embedding
            FROM project_code_intel_records
            WHERE id = %s
            """,
            [record_id],
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
    snapshot_clauses, snapshot_params = scoped_snapshot_clauses(args, "e")
    clauses.extend(snapshot_clauses)
    params.extend(snapshot_params)
    params.append(limit)

    with db.connect() as conn:
        if not code_intel_tables_exist(conn):
            return ok({"error": "code intelligence schema is not initialized"})
        edges = conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT e.id, e.snapshot_id, e.collection, e.repo, e.commit_sha,
                   e.source_record_id, e.target_record_id, e.edge_type,
                   e.source_symbol, e.target_symbol, e.source_path, e.target_path,
                   e.confidence_kind, e.metadata
            FROM project_code_intel_edges e
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
    with db.connect() as conn:
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
    with db.connect() as conn:
        if not table_regclass_exists(conn, "project_code_intel_static_findings"):
            return ok({"error": "static-analysis schema is not initialized"})
        finding = conn.execute(
            """
            SELECT f.*, r.tool_name, r.tool_version, r.semantic_version,
                   r.information_uri, r.automation_id, r.sarif_path,
                   r.sarif_sha256, r.run_index, r.metadata AS run_metadata
            FROM project_code_intel_static_findings f
            JOIN project_code_intel_static_runs r ON r.id = f.run_id
            WHERE f.id = %s
            """,
            [finding_id],
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
    clauses = ["finding_id = %s"]
    params: QueryParams = [finding_id]
    if flow_index is not None:
        if not isinstance(flow_index, int):
            raise McpProtocolTypeError("flow_index must be an integer")
        clauses.append("flow_index = %s")
        params.append(flow_index)
    with db.connect() as conn:
        if not table_regclass_exists(conn, "project_code_intel_static_code_flows"):
            return ok({"error": "static-analysis schema is not initialized"})
        rows = conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT id, finding_id, flow_index, thread_index, step_index,
                   source_path, uri, message, line_start, line_end,
                   column_start, column_end, importance, properties
            FROM project_code_intel_static_code_flows
            """,
                    clauses,
                    """
            ORDER BY flow_index, thread_index, step_index, id
            """,
                )
            ),
            params,
        ).fetchall()
    return ok({"finding_id": finding_id, "flow_index": flow_index, "steps": rows})


ToolHandler = Callable[[Json], Json]
ToolRegistry = dict[str, tuple[ToolDefinition, ToolHandler]]


TOOLS: ToolRegistry = {
    "code_intel_status": (TOOL_DEFINITIONS["code_intel_status"], tool_code_intel_status),
    "search_code_intel_text": (TOOL_DEFINITIONS["search_code_intel_text"], tool_search_code_intel_text),
    "search_code_intel_semantic": (TOOL_DEFINITIONS["search_code_intel_semantic"], tool_search_code_intel_semantic),
    "get_code_intel_record": (TOOL_DEFINITIONS["get_code_intel_record"], tool_get_code_intel_record),
    "related_code_intel": (TOOL_DEFINITIONS["related_code_intel"], tool_related_code_intel),
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


def result_response(request_id: JsonValue, result: Json) -> Json:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def control_response(method: object, request_id: JsonValue) -> Json | None:
    if method == "initialize":
        return result_response(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "project-code-intelligence",
                    "version": "0.1.0",
                },
            },
        )
    if method == "ping":
        return result_response(request_id, {})
    if method == "tools/list":
        return result_response(request_id, {"tools": advertised_tools()})
    if method == "resources/list":
        return result_response(request_id, {"resources": []})
    if method == "prompts/list":
        return result_response(request_id, {"prompts": []})
    return None


def handle_tool_call(request: Json, request_id: JsonValue) -> Json:
    params_value = request.get("params") or {}
    if not isinstance(params_value, dict):
        raise McpProtocolTypeError("params must be an object")
    params = params_value
    name = params.get("name")
    arguments_value = params.get("arguments") or {}
    if not isinstance(name, str):
        raise McpProtocolTypeError("tool name must be a string")
    if name not in TOOLS:
        raise McpProtocolError(f"unknown tool: {name}")
    if not isinstance(arguments_value, dict):
        raise McpProtocolTypeError("arguments must be an object")
    definition, handler = TOOLS[name]
    if definition.write_tool and not db.allow_writes():
        raise McpWritePermissionError("writes are disabled")
    return result_response(request_id, handler(arguments_value))


def handle_request(request: Json) -> Json | None:
    method = request.get("method")
    request_id = request.get("id")

    if isinstance(method, str) and method.startswith("notifications/"):
        return None

    response = control_response(method, request_id)
    if response is not None:
        return response

    if method == "tools/call":
        return handle_tool_call(request, request_id)

    if not isinstance(method, str):
        raise McpProtocolTypeError("method must be a string")
    raise McpProtocolError(f"unsupported method: {method}")


def error_response(request_id: JsonValue, code: int, message: str) -> Json:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def write_response(response: Json | list[Json]) -> None:
    _ = sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
    _ = sys.stdout.flush()


def jsonrpc_input_lines() -> Iterator[str | None]:
    max_bytes = mcp_max_request_bytes()
    while True:
        raw = sys.stdin.buffer.readline(max_bytes + 1)
        if not raw:
            return
        if len(raw) > max_bytes:
            while raw and not raw.endswith(b"\n"):
                raw = sys.stdin.buffer.readline(8192)
            yield None
            continue
        yield raw.decode("utf-8", errors="strict")


def handle_batch_request(batch: list[object]) -> list[Json] | None:
    if len(batch) > mcp_max_batch_items():
        raise McpProtocolError("batch exceeds PROJECT_CODE_INTELLIGENCE_MCP_MAX_BATCH_ITEMS")
    responses: list[Json] = []
    for item in batch:
        if not isinstance(item, dict):
            raise McpProtocolTypeError("batch items must be objects")
        response = handle_request(cast("JsonObject", item))
        if response is not None:
            responses.append(response)
    return responses or None


def handle_jsonrpc_value(request_value: object) -> tuple[JsonValue, Json | list[Json] | None]:
    if isinstance(request_value, list):
        return None, handle_batch_request(cast("list[object]", request_value))
    if not isinstance(request_value, dict):
        raise McpProtocolTypeError("request must be an object")
    request = cast("JsonObject", request_value)
    request_id = request.get("id")
    return request_id, handle_request(request)


def error_message(exc: BaseException) -> str:
    if isinstance(exc, (TypeError, ValueError, PermissionError, json.JSONDecodeError, UnicodeDecodeError)):
        return str(exc)
    if mcp_debug_errors():
        return str(exc)
    return "internal server error"


def main() -> int:
    for line_value in jsonrpc_input_lines():
        if line_value is None:
            write_response(
                error_response(None, -32000, "request exceeds PROJECT_CODE_INTELLIGENCE_MCP_MAX_REQUEST_BYTES")
            )
            continue
        line = line_value
        line = line.strip()
        if not line:
            continue
        request_id: JsonValue = None
        try:
            request_value = cast("object", json.loads(line))
            request_id, response = handle_jsonrpc_value(request_value)
            if response is not None:
                write_response(response)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive server boundary
            if mcp_debug_errors():
                log(traceback.format_exc())
            else:
                log(f"{type(exc).__name__}: {exc}")
            write_response(error_response(request_id, -32000, error_message(exc)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
