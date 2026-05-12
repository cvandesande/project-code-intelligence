"""Public MCP tool descriptions and input schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from project_code_intelligence.mcp_protocol import Json


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
            "properties": {"id": {"type": "integer"}},
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
