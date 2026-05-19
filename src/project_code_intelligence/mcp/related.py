"""Related-edge helpers for the code-intelligence MCP server.

Owns the SQL clause builders, direction classifier, sort priority logic,
and warning emitter that back `tool_related_code_intel`. The handler in
`tools.py` orchestrates query parameter assembly and result formatting;
this module supplies the per-edge rules.

`RelatedDirection` and `RelatedQueryContext` live here so callers
(handlers in `tools.py`, tests) have one canonical import site.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from project_code_intelligence import db
from project_code_intelligence.exceptions import McpProtocolError
from project_code_intelligence.mcp.filters import (
    code_intel_clauses,
    query_with_where,
    scoped_snapshot_clauses,
)
from project_code_intelligence.mcp.protocol import (
    Json,
    QueryParams,
    optional_bool,
    optional_text,
    scoped_collection,
)
from project_code_intelligence.mcp.scope import make_warning
from project_code_intelligence.mcp.status import positive_int

if TYPE_CHECKING:
    from collections.abc import Sequence


RelatedDirection: TypeAlias = Literal["any", "incoming", "outgoing"]


@dataclass(frozen=True)
class RelatedQueryContext:
    record_id: str | None
    symbol: str | None
    direction: RelatedDirection
    scoped_record_ids: tuple[str, ...]
    parent_record_ids: frozenset[str]


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
    # Called only for `direction in {"incoming", "outgoing"}` — the `any` case
    # preserves the interleave order produced by `tool_related_code_intel`'s
    # merge layer and skips this sort entirely.
    _ = context
    return 0 if related_edge_target_resolved(edge) else 1


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
    if context.direction != "any":
        # `any` is already balanced + resolved-first by the merge in `tool_related_code_intel`;
        # re-sorting here would defeat the interleave.
        edges.sort(
            key=lambda edge: (
                related_edge_sort_priority(edge, context=context),
                -positive_int(edge.get("id")),
            )
        )
    return edges


def related_edge_warnings(edges: Sequence[Mapping[str, object]]) -> list[Json]:
    if not any(edge.get("confidence_kind") == "heuristic_candidate" for edge in edges):
        return []
    return [
        make_warning(
            "heuristic_candidate_relationships",
            confidence_kind="heuristic_candidate",
            message="related_code_intel returns heuristic candidates; verify important relationships in source",
        )
    ]


def related_order_clause(
    *,
    symbol: str | None,
    scoped_record_ids: list[str],
    direction: RelatedDirection,
) -> tuple[str, QueryParams]:
    # All callers pass a concrete direction ("incoming" or "outgoing").
    # `tool_related_code_intel` runs each direction independently for
    # `direction=any` and balances them in Python; the per-direction order is
    # resolved-first, newest-first.
    _ = symbol
    _ = scoped_record_ids
    _ = direction
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
