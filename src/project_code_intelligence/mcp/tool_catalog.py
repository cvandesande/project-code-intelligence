"""Public MCP tool descriptions and input schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError
from project_code_intelligence.mcp.tool_inputs import TOOL_INPUT_MODELS

if TYPE_CHECKING:
    from project_code_intelligence.mcp.protocol import Json


@dataclass(frozen=True)
class ToolDefinition:
    description: str
    input_schema: Json
    write_tool: bool = False


TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "code_intel_status": ToolDefinition(
        "Check code intelligence snapshot, file, record, edge, and embedding state. Snapshots include "
        "index_age_seconds (freshness), head_commit (current HEAD), head_matches_snapshot (whether "
        "the index is up-to-date), and embed_record_types (the record types that were configured for "
        "embedding when this snapshot was indexed — absent if the index ran without --embed). "
        "File counts include untracked_files and dirty_files. "
        "records_by_type includes embedded_records per type so you can compare against embed_record_types "
        "to understand which types are expected to have embeddings and which are not.",
        {
            "type": "object",
            "properties": {
                "collection": {"type": "string"},
                "repo": {"type": "string"},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    ),
    "search_code_intel_text": ToolDefinition(
        "Search or list code intelligence records. By default, plain multi-term searches use PostgreSQL "
        "full-text search first, then automatically fall back to exact all-term and any-term matching "
        "when full-text search returns no results. Omit query to enumerate records by filter "
        "(e.g. parent_record_id, symbol) ordered by most recently updated. The query argument, when "
        "supplied, must be non-empty. "
        "source_path is an exact full-path filter, not a directory prefix; pass the complete relative "
        "file path (e.g. 'agent/internal/grpc/grpc.go') to match. "
        "Results are deduplicated by source location: when a code_chunk and a symbol_definition "
        "cover the same lines, only the code_chunk is returned. Pass record_type='symbol_definition' "
        "to see symbol definitions instead. "
        "Results are compact by default: per-result snapshot/git/repo metadata is omitted and a short "
        "code snippet is included inline so most navigational tasks need no follow-up call. "
        "Pass verbose=true to restore all fields (snapshot_id, collection, repo, branch, commit_sha, "
        "tree_sha, record_id, updated_at, confidence, tool, rule_id, severity). "
        "When the auto fallback activates (query_strategy=all_terms_fallback), rank is null because "
        "every result matched all terms equally — use fallback_reason to detect this case.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "query_mode": {
                    "type": "string",
                    "enum": ["auto", "websearch", "all_terms", "any_terms"],
                    "description": "Defaults to auto. Use websearch for PostgreSQL full-text only, "
                    "all_terms when every extracted term must match, or any_terms for broad fallback search.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "collection": {"type": "string"},
                "repo": {"type": "string"},
                "record_type": {"type": "string"},
                "language": {"type": "string"},
                "file_role": {"type": "string"},
                "content_class": {"type": "string"},
                "confidence_kind": {"type": "string"},
                "source_path": {"type": "string"},
                "symbol": {"type": "string"},
                "parent_record_id": {"type": "string"},
                "metadata_key": {"type": "string"},
                "metadata_value": {"type": "string"},
                "metadata_contains": {"type": "object"},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
                "verbose": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    ),
    "search_code_intel_semantic": ToolDefinition(
        "Embed a query with the configured embedding backend and search embedded code intelligence records. "
        "Only record types listed in embed_record_types (visible in code_intel_status) have embeddings — "
        "typically code_chunk and doc_section. symbol_definition records are not embedded by default, so "
        "semantic search cannot find function/type definitions; use search_code_intel_text with the symbol "
        "filter or source_path enumeration for those. "
        "Results are compact by default: per-result snapshot/git/repo metadata is omitted and a short "
        "code snippet is included inline. Pass verbose=true to restore all fields.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "collection": {"type": "string"},
                "repo": {"type": "string"},
                "record_type": {"type": "string"},
                "language": {"type": "string"},
                "file_role": {"type": "string"},
                "content_class": {"type": "string"},
                "confidence_kind": {"type": "string"},
                "source_path": {"type": "string"},
                "symbol": {"type": "string"},
                "parent_record_id": {"type": "string"},
                "metadata_key": {"type": "string"},
                "metadata_value": {"type": "string"},
                "metadata_contains": {"type": "object"},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
                "verbose": {"type": "boolean"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    "get_code_intel_record": ToolDefinition(
        "Fetch one code intelligence record by numeric ID, including display content.",
        {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "minimum": 1},
                "include_content": {"type": "boolean"},
            },
            "required": ["id"],
            "additionalProperties": False,
        },
    ),
    "related_code_intel": ToolDefinition(
        "Return code intelligence graph edges related to a record id or symbol, "
        "joined with source and target record details so a single call resolves both "
        "ends of every edge. At least one of record_id or symbol must be provided. "
        "For call_candidate edges: source = caller, target = callee. Querying by "
        "record_id returns all edges where that record is source or target. Querying "
        "by symbol matches on source_symbol or target_symbol and typically surfaces "
        "callers of functions with that name across the whole codebase. "
        "Filter by edge_type ('call_candidate' for function calls, "
        "'include' for C/C++ #include relationships) when you only want one kind of "
        "relationship. For short, common names (Close, Read, Write, etc.) expect "
        "many heuristic_candidate edges — use a more specific record_id instead of symbol "
        "to reduce noise. confidence_kind='confirmed' edges are only produced by "
        "type-aware parsers; heuristic parsers (stdlib-heuristic-*) produce "
        "heuristic_candidate edges exclusively.",
        {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "symbol": {"type": "string"},
                "edge_type": {"type": "string"},
                "confidence_kind": {"type": "string"},
                "collection": {"type": "string"},
                "repo": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
                "verbose": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    ),
    "list_code_intel_files": ToolDefinition(
        "List indexed source files filtered by language, role, content class, or skip status. "
        "Use this to discover the shape of the codebase (e.g. all test files, only Python sources, "
        "files that were skipped during ingestion). File-level language metadata (go_functions, "
        "go_imports, etc.) is omitted by default; pass include_metadata=true to include it. "
        "source_path is an exact full-path filter (e.g. 'agent/internal/grpc/grpc.go'), not a "
        "directory prefix — passing a directory path returns no results.",
        {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                "collection": {"type": "string"},
                "repo": {"type": "string"},
                "language": {"type": "string"},
                "file_role": {"type": "string"},
                "content_class": {"type": "string"},
                "source_path": {"type": "string"},
                "is_test": {"type": "boolean"},
                "is_doc": {"type": "boolean"},
                "is_generated": {"type": "boolean"},
                "is_vendor": {"type": "boolean"},
                "is_source": {"type": "boolean"},
                "is_build": {"type": "boolean"},
                "is_config": {"type": "boolean"},
                "only_skipped": {"type": "boolean"},
                "include_metadata": {"type": "boolean"},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    ),
    "list_code_intel_parser_failures": ToolDefinition(
        "List files that failed to parse during ingestion, so an agent can report honestly which "
        "parts of the codebase are missing from the index.",
        {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                "collection": {"type": "string"},
                "repo": {"type": "string"},
                "language": {"type": "string"},
                "parser": {"type": "string"},
                "source_path": {"type": "string"},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    ),
    "search_static_findings": ToolDefinition(
        "Search SARIF/static-analysis findings with exact filters. "
        "This tool covers SARIF-based findings only (populated by pci-index --sarif). "
        "Heuristic security patterns detected during indexing (e.g. shell backtick usage, "
        "insecure API calls) are stored as security_pattern records and are accessible via "
        "search_code_intel_text with record_type='security_pattern', not through this tool.",
        {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "collection": {"type": "string"},
                "repo": {"type": "string"},
                "tool": {"type": "string"},
                "rule_id": {"type": "string"},
                "level": {"type": "string"},
                "baseline_state": {"type": "string"},
                "source_path": {"type": "string"},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    ),
    "get_static_finding": ToolDefinition(
        "Fetch one SARIF/static-analysis finding with compact rule and location details. "
        "Set include_code_flows, include_raw, or include_run_metadata for larger diagnostic payloads.",
        {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "include_raw": {"type": "boolean"},
                "include_run_metadata": {"type": "boolean"},
                "include_code_flows": {"type": "boolean"},
            },
            "required": ["id"],
            "additionalProperties": False,
        },
    ),
    "get_static_code_flow": ToolDefinition(
        "Fetch ordered SARIF/CodeQL code-flow steps for one static-analysis finding.",
        {
            "type": "object",
            "properties": {
                "finding_id": {"type": "integer"},
                "flow_index": {"type": "integer"},
            },
            "required": ["finding_id"],
            "additionalProperties": False,
        },
    ),
}


_PRETTY_TYPES: dict[str, str] = {
    "int_type": "an integer",
    "string_type": "a string",
    "bool_type": "a boolean",
    "dict_type": "an object",
    "list_type": "an array",
}


def _format_loc(loc: tuple[int | str, ...]) -> str:
    return ".".join(str(p) for p in loc) if loc else "<root>"


def _translate_validation_error(exc: ValidationError) -> Exception:
    # pydantic_core.ErrorDetails is a TypedDict with type, loc, msg, input fields.
    first = exc.errors()[0]
    err_type = first["type"]
    loc = _format_loc(first["loc"])
    msg = first["msg"]
    if err_type == "extra_forbidden":
        return McpProtocolError(f"unknown argument: {loc}")
    if err_type == "missing":
        return McpProtocolError(f"missing required argument: {loc}")
    if err_type in _PRETTY_TYPES:
        return McpProtocolTypeError(f"{loc} must be {_PRETTY_TYPES[err_type]}")
    return McpProtocolError(f"{loc}: {msg}")


# ToolDefinition holds a dict (input_schema), so it isn't hashable. Use id() as
# the reverse-lookup key — definitions are module-level singletons that live for
# the process lifetime.
_DEFINITION_NAMES: dict[int, str] = {id(definition): name for name, definition in TOOL_DEFINITIONS.items()}


def validate_tool_arguments(definition: ToolDefinition, arguments: Json) -> None:
    name = _DEFINITION_NAMES.get(id(definition))
    if name is None:
        raise McpProtocolError("unknown tool definition")
    model = TOOL_INPUT_MODELS.get(name)
    if model is None:
        raise McpProtocolError(f"no validator registered for tool {name!r}")
    try:
        _ = model.model_validate(arguments)
    except ValidationError as exc:
        raise _translate_validation_error(exc) from exc
