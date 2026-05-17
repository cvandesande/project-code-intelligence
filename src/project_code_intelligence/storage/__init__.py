"""Database persistence for code-intelligence snapshots and records."""

from __future__ import annotations

from project_code_intelligence.storage.copy import copy_unchanged_parser_failures, copy_unchanged_records_and_edges
from project_code_intelligence.storage.core import (
    RecordInsertContext,
    count_unresolved_edge_targets,
    insert_edges,
    insert_files,
    insert_parser_failures,
    insert_records,
    insert_snapshot,
    parser_failure_metadata,
    pre_resolvable_edge_count,
    pre_resolve_edge_targets,
    prune_old_snapshots,
    replace_repos,
    resolve_edge_targets,
    stamp_embed_types,
)
from project_code_intelligence.storage.schema import (
    ensure_schema,
    file_signature,
    latest_snapshot_info,
    previous_file_signatures,
    previous_file_state_signature,
    previous_file_states,
    row_int,
    schema_migration_versions,
    snapshot_versions_compatible,
)
from project_code_intelligence.storage.static import insert_static_runs

__all__ = [
    "RecordInsertContext",
    "copy_unchanged_parser_failures",
    "copy_unchanged_records_and_edges",
    "count_unresolved_edge_targets",
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
    "pre_resolvable_edge_count",
    "pre_resolve_edge_targets",
    "previous_file_signatures",
    "previous_file_state_signature",
    "previous_file_states",
    "prune_old_snapshots",
    "replace_repos",
    "resolve_edge_targets",
    "row_int",
    "schema_migration_versions",
    "snapshot_versions_compatible",
    "stamp_embed_types",
]
