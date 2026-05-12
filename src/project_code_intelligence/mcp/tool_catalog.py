"""Public MCP tool descriptions and input schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError

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
        "Return code intelligence graph edges related to a record id or symbol.",
        {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "symbol": {"type": "string"},
                "collection": {"type": "string"},
                "repo": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
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


def schema_properties(definition: ToolDefinition) -> dict[str, object]:
    properties_value = definition.input_schema.get("properties", {})
    if not isinstance(properties_value, dict):
        return {}
    properties = cast("dict[object, object]", properties_value)
    return {str(name): schema for name, schema in properties.items()}


def schema_required(definition: ToolDefinition) -> set[str]:
    required_value = definition.input_schema.get("required", [])
    if not isinstance(required_value, list):
        return set()
    required = cast("list[object]", required_value)
    return {name for name in required if isinstance(name, str)}


def property_schema_type(schema: object) -> str | None:
    if not isinstance(schema, dict):
        return None
    schema_object = cast("dict[object, object]", schema)
    value = schema_object.get("type")
    return value if isinstance(value, str) else None


def schema_int_bound(schema: object, name: str) -> int | None:
    if not isinstance(schema, dict):
        return None
    schema_object = cast("dict[object, object]", schema)
    value = schema_object.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def validate_integer_bounds(name: str, value: int, schema: object) -> None:
    minimum = schema_int_bound(schema, "minimum")
    maximum = schema_int_bound(schema, "maximum")
    if minimum is not None and value < minimum:
        raise McpProtocolError(f"{name} must be greater than or equal to {minimum}")
    if maximum is not None and value > maximum:
        raise McpProtocolError(f"{name} must be less than or equal to {maximum}")


def validate_schema_type(name: str, value: object, schema: object) -> None:
    expected = property_schema_type(schema)
    if expected is None:
        return
    if expected == "string":
        if not isinstance(value, str):
            raise McpProtocolTypeError(f"{name} must be a string")
        return
    if expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise McpProtocolTypeError(f"{name} must be an integer")
        validate_integer_bounds(name, value, schema)
        return
    if expected == "boolean":
        if not isinstance(value, bool):
            raise McpProtocolTypeError(f"{name} must be a boolean")
        return
    if expected == "object" and not isinstance(value, dict):
        raise McpProtocolTypeError(f"{name} must be an object")


def validate_tool_arguments(definition: ToolDefinition, arguments: Json) -> None:
    properties = schema_properties(definition)
    if definition.input_schema.get("additionalProperties") is False:
        unknown = sorted(name for name in arguments if name not in properties)
        if unknown:
            raise McpProtocolError("unknown argument(s): " + ", ".join(unknown))
    missing = sorted(name for name in schema_required(definition) if name not in arguments)
    if missing:
        raise McpProtocolError("missing required argument(s): " + ", ".join(missing))
    for name, value in arguments.items():
        validate_schema_type(name, value, properties.get(name))
