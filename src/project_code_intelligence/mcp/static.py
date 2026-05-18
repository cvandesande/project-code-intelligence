"""Static-analysis (SARIF) helpers for the code-intelligence MCP server.

Owns the compact-key projections, scope existence probes, and finding /
run-metadata transformations that back the `*_static_*` MCP tools. The
handlers in `tools.py` keep the connection orchestration; this module
supplies the row-shape rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from project_code_intelligence import db
from project_code_intelligence.mcp.filters import (
    query_with_where,
    scoped_snapshot_clauses,
)
from project_code_intelligence.mcp.protocol import (
    Json,
    QueryParams,
    optional_text,
    scoped_collection,
)

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonValue

STATIC_FINDING_COMPACT_KEYS: tuple[str, ...] = (
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

STATIC_RULE_COMPACT_KEYS: tuple[str, ...] = (
    "id",
    "rule_id",
    "name",
    "short_description",
    "full_description",
    "default_level",
    "help_uri",
)


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
