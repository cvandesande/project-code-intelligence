"""SARIF discovery, normalization, and record conversion."""

from __future__ import annotations

from project_code_intelligence.sarif.discovery import discover_sarif_files, explicit_sarif_patterns, repo_for_sarif_file
from project_code_intelligence.sarif.ingest import (
    append_sarif_load_failure,
    ensure_sarif_static_run,
    ingest_sarif,
    ingest_sarif_result,
    ingest_sarif_run,
    load_sarif_file,
    sarif_file_bytes,
    sarif_static_run,
)
from project_code_intelligence.sarif.parse import (
    json_array,
    parse_sarif_result,
    sarif_automation_id,
    sarif_message_text,
    sarif_original_uri_base_ids,
    sarif_rule_id,
    sarif_rule_items,
    sarif_tool_metadata,
)
from project_code_intelligence.sarif.paths import (
    SarifPathContext,
    combine_sarif_base_uri,
    normalize_sarif_uri,
    relative_to_or_none,
    resolve_sarif_source_path,
    source_path_from_sarif_uri,
)
from project_code_intelligence.sarif.render import sarif_record_for_finding
from project_code_intelligence.sarif.types import (
    LoadedSarifFile,
    SarifFileRecordContext,
    SarifFlowRecordContext,
    SarifIngestContext,
    SarifIngestState,
    SarifRecordRenderContext,
    SarifResultContext,
    SarifRuleRecordContext,
    SarifRunContext,
    SarifToolMetadata,
)

__all__ = [
    "LoadedSarifFile",
    "SarifFileRecordContext",
    "SarifFlowRecordContext",
    "SarifIngestContext",
    "SarifIngestState",
    "SarifPathContext",
    "SarifRecordRenderContext",
    "SarifResultContext",
    "SarifRuleRecordContext",
    "SarifRunContext",
    "SarifToolMetadata",
    "append_sarif_load_failure",
    "combine_sarif_base_uri",
    "discover_sarif_files",
    "ensure_sarif_static_run",
    "explicit_sarif_patterns",
    "ingest_sarif",
    "ingest_sarif_result",
    "ingest_sarif_run",
    "json_array",
    "load_sarif_file",
    "normalize_sarif_uri",
    "parse_sarif_result",
    "relative_to_or_none",
    "repo_for_sarif_file",
    "resolve_sarif_source_path",
    "sarif_automation_id",
    "sarif_file_bytes",
    "sarif_message_text",
    "sarif_original_uri_base_ids",
    "sarif_record_for_finding",
    "sarif_rule_id",
    "sarif_rule_items",
    "sarif_static_run",
    "sarif_tool_metadata",
    "source_path_from_sarif_uri",
]
