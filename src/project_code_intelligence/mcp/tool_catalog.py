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


# Shared property-description text. Lives once here so adding a new tool that
# exposes the same filter doesn't re-litigate the gotchas in prose.
_SOURCE_PATH_DESC = "Exact path match (not a prefix). For subtrees, use source_path_prefix where supported."
_SOURCE_PATH_PREFIX_DESC = (
    "Subtree filter (strict descendants, trailing slash optional). Must match stored paths "
    "verbatim — include any repo segment. Mutex with source_path."
)
_CONFIDENCE_KIND_DESC = "'confirmed' = type-aware parser; 'heuristic_candidate' = heuristic parser."
_SNIPPET_LENGTH_DESC = "Inline snippet size in chars. Lower for many results, raise for few."


TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "code_intel_status": ToolDefinition(
        "Snapshot, file, record, edge, and embedding state for the indexed codebase. Each snapshot "
        "carries index_age_seconds, head_commit, head_matches_snapshot, and embed_record_types "
        "(the types configured for embedding at index time — absent if --embed wasn't passed). "
        "records_by_type includes embedded_records per type for comparison against "
        "embed_record_types. language_breakdown and directory_breakdown summarize the file "
        "inventory in one call — prefer over paginating list_code_intel_files for project shape.",
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
        "Search or list code intelligence records. Multi-term queries try PostgreSQL FTS first, "
        "then fall back to all-term then any-term matching; query_strategy reports the path taken "
        "(rank is null on all_terms_fallback — check fallback_reason). With query set, runs a "
        "search (mode=search); without it, enumerates by filter (mode=enumerate) — pass mode "
        "explicitly to fail on mismatch. Source-location dedup: code_chunk wins over "
        "symbol_definition at the same lines; pass record_type='symbol_definition' to override. "
        "Compact by default; verbose=true restores stripped fields.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "mode": {"type": "string", "enum": ["search", "enumerate"]},
                "query_mode": {
                    "type": "string",
                    "enum": ["auto", "websearch", "all_terms", "any_terms"],
                    "description": "auto (default) tries FTS then falls back; websearch = FTS only; "
                    "all_terms = every term must match; any_terms = broad.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "collection": {"type": "string"},
                "repo": {"type": "string"},
                "record_type": {"type": "string"},
                "language": {"type": "string"},
                "file_role": {"type": "string"},
                "content_class": {"type": "string"},
                "confidence_kind": {"type": "string", "description": _CONFIDENCE_KIND_DESC},
                "source_path": {"type": "string", "description": _SOURCE_PATH_DESC},
                "symbol": {"type": "string"},
                "parent_record_id": {"type": "string"},
                "metadata_key": {"type": "string"},
                "metadata_value": {"type": "string"},
                "metadata_contains": {"type": "object"},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
                "verbose": {"type": "boolean"},
                "snippet_length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 800,
                    "description": _SNIPPET_LENGTH_DESC,
                },
            },
            "additionalProperties": False,
        },
    ),
    "search_code_intel_semantic": ToolDefinition(
        "Vector search of indexed records via the configured embedding backend. Only record types "
        "in embed_record_types (see code_intel_status) are searchable — symbol_definition is not "
        "embedded by default; use search_code_intel_text for function/type lookups. Compact by "
        "default; verbose=true restores all stripped fields.",
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
                "confidence_kind": {"type": "string", "description": _CONFIDENCE_KIND_DESC},
                "source_path": {"type": "string", "description": _SOURCE_PATH_DESC},
                "symbol": {"type": "string"},
                "parent_record_id": {"type": "string"},
                "metadata_key": {"type": "string"},
                "metadata_value": {"type": "string"},
                "metadata_contains": {"type": "object"},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
                "verbose": {"type": "boolean"},
                "snippet_length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 800,
                    "description": _SNIPPET_LENGTH_DESC,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    "get_code_intel_record": ToolDefinition(
        "Fetch records by stable record_id (returned by search and related-edge results, e.g. "
        "'README.md::doc::000001'). Pass record_id (singular) for one record → {result: {...}} "
        "or {found: false}; record_ids (array, max 100) for a batch → {results: [...], "
        "missing: [...]}. Exactly one is required. Compact by default; verbose=true keeps the "
        "internal int id and heavy metadata like metadata.doc_links.",
        {
            "type": "object",
            "properties": {
                "record_id": {"type": "string", "minLength": 1},
                "record_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "maxItems": 100,
                },
                "collection": {"type": "string"},
                "repo": {"type": "string"},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
                "include_content": {"type": "boolean"},
                "verbose": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    ),
    "related_code_intel": ToolDefinition(
        "Return graph edges related to a record_id or symbol, joined with source and target "
        "record details. At least one of record_id or symbol is required. For call_candidate "
        "edges, source = caller, target = callee. Symbol queries match both ends — prefer "
        "record_id for common names (Close/Read/Write) to avoid noise. edge_type filters to a "
        "kind: 'call_candidate' for calls, 'include' for C/C++ #include.",
        {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "symbol": {"type": "string"},
                "edge_type": {"type": "string"},
                "confidence_kind": {"type": "string", "description": _CONFIDENCE_KIND_DESC},
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
        "List indexed source files filtered by language, role, content class, or skip status. Use "
        "this to discover the shape of the codebase (test files, sources by language, files skipped "
        "at ingestion). Compact by default; verbose=true restores every column including metadata.",
        {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                "collection": {"type": "string"},
                "repo": {"type": "string"},
                "language": {"type": "string"},
                "file_role": {"type": "string"},
                "content_class": {"type": "string"},
                "source_path": {"type": "string", "description": _SOURCE_PATH_DESC},
                "source_path_prefix": {"type": "string", "description": _SOURCE_PATH_PREFIX_DESC},
                "is_test": {"type": "boolean"},
                "is_doc": {"type": "boolean"},
                "is_generated": {"type": "boolean"},
                "is_vendor": {"type": "boolean"},
                "is_source": {"type": "boolean"},
                "is_build": {"type": "boolean"},
                "is_config": {"type": "boolean"},
                "only_skipped": {"type": "boolean"},
                "verbose": {"type": "boolean"},
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
                "source_path": {"type": "string", "description": _SOURCE_PATH_DESC},
                "source_path_prefix": {"type": "string", "description": _SOURCE_PATH_PREFIX_DESC},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    ),
    "search_static_findings": ToolDefinition(
        "Search SARIF/static-analysis findings (populated by pci-index --sarif) with exact "
        "filters. Heuristic security patterns detected during indexing are stored as "
        "security_pattern records — fetch those via search_code_intel_text with "
        "record_type='security_pattern', not this tool.",
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
                "source_path": {"type": "string", "description": _SOURCE_PATH_DESC},
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
