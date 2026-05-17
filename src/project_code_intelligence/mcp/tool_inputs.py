"""Pydantic models describing the validated shape of each MCP tool's arguments.

These models are the authoritative boundary between untrusted JSON-RPC input and
the rest of the codebase. The dispatcher (`mcp/transport.py`) routes through
`validate_tool_arguments` in `tool_catalog.py`, which uses these models to
reject malformed input before it reaches a tool implementation.

The JSON Schemas in `tool_catalog.TOOL_DEFINITIONS` remain the wire-level
contract exposed via `tools/list`; the models below mirror them.
"""

from __future__ import annotations

from types import UnionType
from typing import Literal, Union, cast, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _is_optional_string_like_annotation(annotation: object) -> bool:
    origin = get_origin(annotation)
    if origin is not Union and origin is not UnionType:
        return False
    args = cast("tuple[object, ...]", get_args(annotation))
    if type(None) not in args:
        return False
    string_like_args = 0
    for arg in args:
        if arg is type(None):
            continue
        arg_origin = get_origin(arg)
        if arg is str:
            string_like_args += 1
            continue
        literal_choices = cast("tuple[object, ...]", get_args(arg))
        if arg_origin is Literal and all(isinstance(choice, str) for choice in literal_choices):
            string_like_args += 1
            continue
        return False
    return string_like_args > 0


class StrictArgs(BaseModel):
    """Base for all tool-input models: forbid extras, no implicit coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)

    @model_validator(mode="before")
    @classmethod
    def empty_optional_strings_are_omitted(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        values = cast("dict[object, object]", data)
        normalized: dict[object, object] | None = None
        for key, value in values.items():
            if not isinstance(key, str) or not isinstance(value, str) or value:
                continue
            field = cls.model_fields.get(key)
            if field is None or not _is_optional_string_like_annotation(field.annotation):
                continue
            if normalized is None:
                normalized = dict(values)
            normalized[key] = None
        return values if normalized is None else normalized


class CodeIntelStatusArgs(StrictArgs):
    collection: str | None = None
    repo: str | None = None
    snapshot_id: int | None = Field(default=None, ge=1)
    include_historical: bool | None = None
    directory_depth: int | None = Field(default=None, ge=1, le=5)
    verbose: bool | None = None
    include_snapshots: bool | None = None
    include_record_types: bool | None = None
    include_queryability: bool | None = None
    include_breakdowns: bool | None = None
    include_static_summary: bool | None = None


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
    source_path_prefix: str | None = None
    symbol: str | None = None
    parent_record_id: str | None = None
    metadata_key: str | None = None
    metadata_value: str | None = None
    metadata_contains: dict[str, object] | None = None
    is_untracked: bool | None = None
    snapshot_id: int | None = Field(default=None, ge=1)
    include_historical: bool | None = None
    verbose: bool | None = None


class SearchCodeIntelTextArgs(_SearchFilterArgs):
    query: str | None = Field(default=None, min_length=1)
    mode: Literal["search", "enumerate"] | None = None
    query_mode: Literal["auto", "websearch", "all_terms", "any_terms"] | None = None
    limit: int | None = Field(default=None, ge=1, le=50)
    snippet_length: int | None = Field(default=None, ge=1, le=800)


class SearchCodeIntelSemanticArgs(_SearchFilterArgs):
    query: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=1, le=50)
    snippet_length: int | None = Field(default=None, ge=1, le=800)


class GetCodeIntelRecordArgs(StrictArgs):
    record_id: str = Field(min_length=1)
    collection: str | None = None
    repo: str | None = None
    snapshot_id: int | None = Field(default=None, ge=1)
    include_historical: bool | None = None
    include_content: bool | None = None
    include_metadata: bool | None = None
    verbose: bool | None = None


class GetCodeIntelRecordsArgs(StrictArgs):
    record_ids: list[str] = Field(min_length=1, max_length=100)
    collection: str | None = None
    repo: str | None = None
    snapshot_id: int | None = Field(default=None, ge=1)
    include_historical: bool | None = None
    include_content: bool | None = None
    include_metadata: bool | None = None
    verbose: bool | None = None


class RelatedCodeIntelArgs(StrictArgs):
    record_id: str | None = None
    symbol: str | None = None
    direction: Literal["any", "incoming", "outgoing"] | None = None
    edge_type: str | None = None
    confidence_kind: str | None = None
    include_unresolved: bool | None = None
    collection: str | None = None
    repo: str | None = None
    limit: int | None = Field(default=None, ge=1, le=100)
    snapshot_id: int | None = Field(default=None, ge=1)
    include_historical: bool | None = None
    verbose: bool | None = None


class ListCodeIntelFilesArgs(StrictArgs):
    limit: int | None = Field(default=None, ge=1, le=500)
    collection: str | None = None
    repo: str | None = None
    language: str | None = None
    file_role: str | None = None
    content_class: str | None = None
    source_path: str | None = None
    source_path_prefix: str | None = None
    is_test: bool | None = None
    is_doc: bool | None = None
    is_generated: bool | None = None
    is_vendor: bool | None = None
    is_source: bool | None = None
    is_build: bool | None = None
    is_config: bool | None = None
    is_untracked: bool | None = None
    only_skipped: bool | None = None
    verbose: bool | None = None
    snapshot_id: int | None = Field(default=None, ge=1)
    include_historical: bool | None = None


class ListCodeIntelParserFailuresArgs(StrictArgs):
    limit: int | None = Field(default=None, ge=1, le=500)
    collection: str | None = None
    repo: str | None = None
    language: str | None = None
    parser: str | None = None
    source_path: str | None = None
    source_path_prefix: str | None = None
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
    source_path_prefix: str | None = None
    snapshot_id: int | None = Field(default=None, ge=1)
    include_historical: bool | None = None


class GetStaticFindingArgs(StrictArgs):
    id: int
    include_raw: bool | None = None
    include_run_metadata: bool | None = None
    include_code_flows: bool | None = None


class GetStaticCodeFlowArgs(StrictArgs):
    finding_id: int
    flow_index: int | None = None


TOOL_INPUT_MODELS: dict[str, type[StrictArgs]] = {
    "code_intel_status": CodeIntelStatusArgs,
    "search_code_intel_text": SearchCodeIntelTextArgs,
    "search_code_intel_semantic": SearchCodeIntelSemanticArgs,
    "get_code_intel_record": GetCodeIntelRecordArgs,
    "get_code_intel_records": GetCodeIntelRecordsArgs,
    "related_code_intel": RelatedCodeIntelArgs,
    "list_code_intel_files": ListCodeIntelFilesArgs,
    "list_code_intel_parser_failures": ListCodeIntelParserFailuresArgs,
    "search_static_findings": SearchStaticFindingsArgs,
    "get_static_finding": GetStaticFindingArgs,
    "get_static_code_flow": GetStaticCodeFlowArgs,
}
