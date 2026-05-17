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


# Shared property-description text.
_SOURCE_PATH_DESC = "Exact path; repo-relative accepted."
_SOURCE_PATH_PREFIX_DESC = "Subtree prefix; repo-relative accepted. Mutex with source_path."
_CONFIDENCE_KIND_DESC = "confirmed or heuristic_candidate."
_COLLECTION_DESC = "Collection scope; omit when repo alone is enough."
_LIMIT_50_DESC = "Max results, 1-50."
_LIMIT_100_DESC = "Max results, 1-100."
_LIMIT_500_DESC = "Max results, 1-500."
_SNIPPET_LENGTH_DESC = "Snippet chars, default 300."


TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "code_intel_status": ToolDefinition(
        "First call for non-trivial code discovery; checks index freshness, coverage, and queryability.",
        {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": _COLLECTION_DESC},
                "repo": {"type": "string"},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
                "directory_depth": {"type": "integer", "minimum": 1, "maximum": 5},
                "verbose": {"type": "boolean"},
                "include_snapshots": {"type": "boolean"},
                "include_record_types": {"type": "boolean"},
                "include_queryability": {
                    "type": "boolean",
                    "description": "Full queryability record-type lists.",
                },
                "include_breakdowns": {"type": "boolean"},
                "include_static_summary": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    ),
    "search_code_intel_text": ToolDefinition(
        "Exact indexed search for identifiers, symbols, filenames, config keys, and known strings.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Empty string is treated as omitted."},
                "mode": {
                    "type": "string",
                    "enum": ["search", "enumerate"],
                    "description": "search requires query; enumerate lists records by filters and forbids query.",
                },
                "query_mode": {
                    "type": "string",
                    "enum": ["auto", "websearch", "all_terms", "any_terms"],
                    "description": "auto (default), websearch (FTS only), all_terms, any_terms.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": _LIMIT_50_DESC},
                "collection": {"type": "string", "description": _COLLECTION_DESC},
                "repo": {"type": "string"},
                "record_type": {"type": "string"},
                "language": {"type": "string"},
                "file_role": {"type": "string"},
                "content_class": {"type": "string"},
                "confidence_kind": {"type": "string", "description": _CONFIDENCE_KIND_DESC},
                "source_path": {"type": "string", "description": _SOURCE_PATH_DESC},
                "source_path_prefix": {"type": "string", "description": _SOURCE_PATH_PREFIX_DESC},
                "symbol": {"type": "string"},
                "parent_record_id": {"type": "string"},
                "metadata_key": {"type": "string"},
                "metadata_value": {"type": "string"},
                "metadata_contains": {"type": "object"},
                "is_untracked": {"type": "boolean"},
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
        "Concept search for behavior when identifiers are unknown. Use text search for symbols.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": _LIMIT_50_DESC},
                "collection": {"type": "string", "description": _COLLECTION_DESC},
                "repo": {"type": "string"},
                "record_type": {"type": "string"},
                "language": {"type": "string"},
                "file_role": {"type": "string"},
                "content_class": {"type": "string"},
                "confidence_kind": {"type": "string", "description": _CONFIDENCE_KIND_DESC},
                "source_path": {"type": "string", "description": _SOURCE_PATH_DESC},
                "source_path_prefix": {"type": "string", "description": _SOURCE_PATH_PREFIX_DESC},
                "symbol": {"type": "string"},
                "parent_record_id": {"type": "string"},
                "metadata_key": {"type": "string"},
                "metadata_value": {"type": "string"},
                "metadata_contains": {"type": "object"},
                "is_untracked": {"type": "boolean"},
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
        "Fetch one record by record_id. Compact by default; include_content/include_metadata add body/metadata.",
        {
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Stable record ID.",
                },
                "collection": {"type": "string", "description": _COLLECTION_DESC},
                "repo": {"type": "string"},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
                "include_content": {"type": "boolean"},
                "include_metadata": {"type": "boolean"},
                "verbose": {"type": "boolean"},
            },
            "required": ["record_id"],
            "additionalProperties": False,
        },
    ),
    "get_code_intel_records": ToolDefinition(
        "Fetch records by record_ids. Compact by default; include_content/include_metadata add body/metadata.",
        {
            "type": "object",
            "properties": {
                "record_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "maxItems": 100,
                    "description": "Stable record IDs.",
                },
                "collection": {"type": "string", "description": _COLLECTION_DESC},
                "repo": {"type": "string"},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
                "include_content": {"type": "boolean"},
                "include_metadata": {"type": "boolean"},
                "verbose": {"type": "boolean"},
            },
            "required": ["record_ids"],
            "additionalProperties": False,
        },
    ),
    "related_code_intel": ToolDefinition(
        "Heuristic caller/callee and related-symbol candidates. Verify important edges in source.",
        {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "symbol": {"type": "string"},
                "direction": {
                    "type": "string",
                    "enum": ["any", "incoming", "outgoing"],
                    "description": "Edge direction; default any.",
                },
                "edge_type": {"type": "string"},
                "confidence_kind": {"type": "string", "description": _CONFIDENCE_KIND_DESC},
                "include_unresolved": {
                    "type": "boolean",
                    "description": "Include unresolved heuristic targets; default false.",
                },
                "collection": {"type": "string", "description": _COLLECTION_DESC},
                "repo": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": _LIMIT_100_DESC},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
                "verbose": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    ),
    "list_code_intel_files": ToolDefinition(
        "List indexed files. Compact by default.",
        {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 500, "description": _LIMIT_500_DESC},
                "collection": {"type": "string", "description": _COLLECTION_DESC},
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
                "is_untracked": {"type": "boolean"},
                "only_skipped": {"type": "boolean"},
                "verbose": {"type": "boolean"},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    ),
    "list_code_intel_parser_failures": ToolDefinition(
        "List files that failed to parse.",
        {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 500, "description": _LIMIT_500_DESC},
                "collection": {"type": "string", "description": _COLLECTION_DESC},
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
        "Search SARIF/static-analysis findings.",
        {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": _LIMIT_100_DESC},
                "collection": {"type": "string", "description": _COLLECTION_DESC},
                "repo": {"type": "string"},
                "tool": {"type": "string"},
                "rule_id": {"type": "string"},
                "level": {"type": "string"},
                "baseline_state": {"type": "string"},
                "source_path": {"type": "string", "description": _SOURCE_PATH_DESC},
                "source_path_prefix": {"type": "string", "description": _SOURCE_PATH_PREFIX_DESC},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    ),
    "get_static_finding": ToolDefinition(
        "Fetch one static finding; flags add larger diagnostics.",
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
        "Fetch code-flow steps for a static finding.",
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
    if err_type == "value_error" and msg.startswith("Value error, "):
        return McpProtocolError(msg.removeprefix("Value error, "))
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
