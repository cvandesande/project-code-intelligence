"""Pydantic models describing the validated shape of each MCP tool's arguments.

These models are the authoritative boundary between untrusted JSON-RPC input and
the rest of the codebase. The dispatcher (`mcp/transport.py`) routes through
`validate_tool_arguments` in `tool_catalog.py`, which uses these models to
reject malformed input before it reaches a tool implementation.

The JSON Schemas in `tool_catalog.TOOL_DEFINITIONS` remain the wire-level
contract exposed via `tools/list`; the models below mirror them.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictArgs(BaseModel):
    """Base for all tool-input models: forbid extras, no implicit coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)


class CodeIntelStatusArgs(StrictArgs):
    collection: str | None = None
    repo: str | None = None
    snapshot_id: int | None = Field(default=None, ge=1)
    include_historical: bool | None = None


class _SearchFilterArgs(StrictArgs):
    """Common filter fields shared between text-search and semantic-search tools."""

    collection: str | None = None
    repo: str | None = None
    record_type: str | None = None
    language: str | None = None
    file_role: str | None = None
    content_class: str | None = None
    confidence_kind: str | None = None
    source_path: str | None = None
    symbol: str | None = None
    metadata_key: str | None = None
    metadata_value: str | None = None
    metadata_contains: dict[str, object] | None = None
    snapshot_id: int | None = Field(default=None, ge=1)
    include_historical: bool | None = None


class SearchCodeIntelTextArgs(_SearchFilterArgs):
    query: str | None = None
    limit: int | None = Field(default=None, ge=1, le=50)


class SearchCodeIntelSemanticArgs(_SearchFilterArgs):
    query: str
    limit: int | None = Field(default=None, ge=1, le=50)


class GetCodeIntelRecordArgs(StrictArgs):
    id: int = Field(ge=1)
    include_content: bool | None = None


class RelatedCodeIntelArgs(StrictArgs):
    record_id: str | None = None
    symbol: str | None = None
    collection: str | None = None
    repo: str | None = None
    limit: int | None = Field(default=None, ge=1, le=100)
    snapshot_id: int | None = Field(default=None, ge=1)
    include_historical: bool | None = None


class SearchStaticFindingsArgs(StrictArgs):
    limit: int | None = Field(default=None, ge=1, le=100)
    collection: str | None = None
    repo: str | None = None
    tool: str | None = None
    rule_id: str | None = None
    level: str | None = None
    baseline_state: str | None = None
    source_path: str | None = None
    snapshot_id: int | None = Field(default=None, ge=1)
    include_historical: bool | None = None


class GetStaticFindingArgs(StrictArgs):
    id: int


class GetStaticCodeFlowArgs(StrictArgs):
    finding_id: int
    flow_index: int | None = None


TOOL_INPUT_MODELS: dict[str, type[StrictArgs]] = {
    "code_intel_status": CodeIntelStatusArgs,
    "search_code_intel_text": SearchCodeIntelTextArgs,
    "search_code_intel_semantic": SearchCodeIntelSemanticArgs,
    "get_code_intel_record": GetCodeIntelRecordArgs,
    "related_code_intel": RelatedCodeIntelArgs,
    "search_static_findings": SearchStaticFindingsArgs,
    "get_static_finding": GetStaticFindingArgs,
    "get_static_code_flow": GetStaticCodeFlowArgs,
}
