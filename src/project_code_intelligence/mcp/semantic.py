"""Semantic-search helpers for the code-intelligence MCP server.

Owns the lexical-boost / distance-penalty heuristics, diversity logic,
embedding bridge, and queryability checks that back
`tool_search_code_intel_semantic`. The handler in `tools.py` orchestrates;
this module supplies the math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from project_code_intelligence import config, db, embeddings
from project_code_intelligence.embedding import llama
from project_code_intelligence.exceptions import McpProtocolError
from project_code_intelligence.mcp import db as mcp_db
from project_code_intelligence.mcp.filters import (
    code_intel_clauses,
    query_with_where,
    snapshot_scope_response,
)
from project_code_intelligence.mcp.protocol import (
    Json,
    optional_bool,
    optional_text,
    require_int,
)
from project_code_intelligence.mcp.scope import make_warning
from project_code_intelligence.mcp.search import search_terms
from project_code_intelligence.mcp.status import snapshot_scope_warning
from project_code_intelligence.storage import row_int

if TYPE_CHECKING:
    from collections.abc import Mapping

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


def semantic_implementation_distance_penalty(args: Json, query: str, penalty: float) -> float:
    if any(optional_text(args, name) for name in ("source_path", "source_path_prefix", "file_role", "content_class")):
        return 0.0
    return penalty if semantic_has_implementation_intent(args, query) else 0.0


def semantic_source_distance_penalty(args: Json, query: str, penalty: float) -> float:
    if any(optional_text(args, name) for name in ("source_path", "source_path_prefix", "file_role", "content_class")):
        return 0.0
    if semantic_query_terms(query) & SEMANTIC_NON_SOURCE_QUERY_TERMS:
        return 0.0
    return penalty


def semantic_structural_symbol_distance_penalty(args: Json, query: str) -> float:
    return semantic_implementation_distance_penalty(args, query, SEMANTIC_STRUCTURAL_SYMBOL_DISTANCE_PENALTY)


def semantic_validation_distance_penalty(args: Json, query: str) -> float:
    return semantic_implementation_distance_penalty(args, query, SEMANTIC_VALIDATION_DISTANCE_PENALTY)


def semantic_generated_distance_penalty(args: Json, query: str) -> float:
    return semantic_source_distance_penalty(args, query, SEMANTIC_GENERATED_DISTANCE_PENALTY)


def semantic_non_source_distance_penalty(args: Json, query: str) -> float:
    return semantic_source_distance_penalty(args, query, SEMANTIC_NON_SOURCE_DISTANCE_PENALTY)


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
    return make_warning(
        "semantic_filter_has_no_embeddings",
        record_type=record_type,
        message=(
            "semantic search only searches embedded records; this filter matches records in the text index "
            "but none have embeddings. Use search_code_intel_text or remove the non-embedded filter."
        ),
    )


def semantic_filter_queryability_response(args: Json, query: str) -> Json | None:
    if not optional_text(args, "record_type"):
        return None
    with mcp_db.connect() as conn:
        if not mcp_db.code_intel_tables_exist(conn):
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
