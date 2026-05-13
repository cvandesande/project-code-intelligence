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
        "Check code intelligence snapshot, file, record, edge, and embedding state.",
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
        "Search or list code intelligence records with optional PostgreSQL full-text search and exact filters.",
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
            },
            "additionalProperties": False,
        },
    ),
    "search_code_intel_semantic": ToolDefinition(
        "Embed a query with the configured embedding backend and search embedded code intelligence records.",
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
        "ends of every edge. Filter by edge_type (e.g. 'imports', 'calls', 'inherits') "
        "when you only want one kind of relationship.",
        {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "symbol": {"type": "string"},
                "edge_type": {"type": "string"},
                "collection": {"type": "string"},
                "repo": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "snapshot_id": {"type": "integer", "minimum": 1},
                "include_historical": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    ),
    "list_code_intel_files": ToolDefinition(
        "List indexed source files filtered by language, role, content class, or skip status. "
        "Use this to discover the shape of the codebase (e.g. all test files, only Python sources, "
        "files that were skipped during ingestion).",
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
        "Search SARIF/static-analysis findings with exact filters.",
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
        "Fetch one SARIF/static-analysis finding with rule, locations, and code-flow steps.",
        {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
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
