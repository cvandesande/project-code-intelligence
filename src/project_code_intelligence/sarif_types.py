"""Shared SARIF ingest data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

    from project_code_intelligence.models import (
        IntelFile,
        IntelRecord,
        JsonObject,
        StaticFinding,
        StaticRule,
        StaticRun,
    )
    from project_code_intelligence.sarif_paths import SarifPathContext


class SarifToolMetadata(TypedDict):
    tool_name: str
    tool_version: str | None
    semantic_version: str | None
    information_uri: str | None


@dataclass(frozen=True)
class SarifResultContext:
    path_context: SarifPathContext
    rules: list[StaticRule]
    run_index: int


@dataclass(frozen=True)
class SarifIngestContext:
    root: Path
    repos: list[str]
    collection: str
    file_by_source_path: dict[str, IntelFile]
    max_bytes: int


@dataclass(frozen=True)
class LoadedSarifFile:
    sarif_path: Path
    source_path: str
    sarif_hash: str
    default_repo: str | None
    sarif: JsonObject


@dataclass
class SarifIngestState:
    runs_by_key: dict[tuple[str, int, str], StaticRun]
    records_by_repo: dict[str, list[IntelRecord]]
    failures: list[JsonObject]
    known_source_paths: set[str]


@dataclass(frozen=True)
class SarifRunContext:
    loaded: LoadedSarifFile
    run_index: int
    run: JsonObject
    tool_meta: SarifToolMetadata
    rules: list[StaticRule]
    result_context: SarifResultContext


@dataclass(frozen=True)
class SarifFileRecordContext:
    source_path: str
    language: str
    file_role: str
    content_class: str
    location_text: str


@dataclass(frozen=True)
class SarifRuleRecordContext:
    rule: StaticRule | None
    rule_metadata: JsonObject
    severity: object
    security_severity: object
    tags: list[str]
    cwe_values: list[str]


@dataclass(frozen=True)
class SarifFlowRecordContext:
    code_flow_summary: list[str]
    code_flow_source: str | None
    code_flow_sink: str | None
    location_summary: list[str]
    path_mappings: list[str]
    primary_path_mapping: object
    suppressed: bool


@dataclass(frozen=True)
class SarifRecordRenderContext:
    title: str
    summary: str
    repo: str
    run: StaticRun
    finding: StaticFinding
    file: SarifFileRecordContext
    rule: SarifRuleRecordContext
    flow: SarifFlowRecordContext
