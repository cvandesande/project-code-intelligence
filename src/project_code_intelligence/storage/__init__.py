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
    prune_dead_branch_snapshots,
    prune_old_snapshots,
    replace_repos,
    resolve_edge_targets,
    stamp_embed_types,
)
from project_code_intelligence.storage.runs import (
    finish_index_run,
    heartbeat_index_run,
    load_index_runs,
    set_index_run_modes,
    start_index_run,
)
from project_code_intelligence.storage.schema import (
    ensure_schema,
    file_signature,
    latest_snapshot_info,
    load_latest_snapshots,
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
    "finish_index_run",
    "heartbeat_index_run",
    "insert_edges",
    "insert_files",
    "insert_parser_failures",
    "insert_records",
    "insert_snapshot",
    "insert_static_runs",
    "latest_snapshot_info",
    "load_index_runs",
    "load_latest_snapshots",
    "parser_failure_metadata",
    "pre_resolvable_edge_count",
    "pre_resolve_edge_targets",
    "previous_file_signatures",
    "previous_file_state_signature",
    "previous_file_states",
    "prune_dead_branch_snapshots",
    "prune_old_snapshots",
    "replace_repos",
    "resolve_edge_targets",
    "row_int",
    "schema_migration_versions",
    "set_index_run_modes",
    "snapshot_versions_compatible",
    "stamp_embed_types",
    "start_index_run",
]
