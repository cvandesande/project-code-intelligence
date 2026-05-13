"""Database persistence for code-intelligence snapshots and records."""

from __future__ import annotations

from project_code_intelligence.storage.copy import copy_unchanged_parser_failures, copy_unchanged_records_and_edges
from project_code_intelligence.storage.core import (
    RecordInsertContext,
    delete_all_code_intel_data,
    delete_repo_data,
    insert_edges,
    insert_files,
    insert_parser_failures,
    insert_records,
    insert_snapshot,
    parser_failure_metadata,
    replace_repos,
)
from project_code_intelligence.storage.schema import (
    ensure_schema,
    file_signature,
    latest_snapshot_info,
    previous_file_signatures,
    row_int,
    schema_migration_versions,
    snapshot_versions_compatible,
)
from project_code_intelligence.storage.static import insert_static_runs

__all__ = [
    "RecordInsertContext",
    "copy_unchanged_parser_failures",
    "copy_unchanged_records_and_edges",
    "delete_all_code_intel_data",
    "delete_repo_data",
    "ensure_schema",
    "file_signature",
    "insert_edges",
    "insert_files",
    "insert_parser_failures",
    "insert_records",
    "insert_snapshot",
    "insert_static_runs",
    "latest_snapshot_info",
    "parser_failure_metadata",
    "previous_file_signatures",
    "replace_repos",
    "row_int",
    "schema_migration_versions",
    "snapshot_versions_compatible",
]
