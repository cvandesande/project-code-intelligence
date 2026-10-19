"""File-inventory helpers for the code-intelligence MCP server.

Owns the SQL `SELECT` constants and overconstrained-boolean-filter warning
emitter that back `tool_list_code_intel_files`. Kept separate from `tools.py`
so the handler file stays a thin glue layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from project_code_intelligence.mcp.protocol import optional_bool, optional_text
from project_code_intelligence.mcp.scope import make_warning

if TYPE_CHECKING:
    from collections.abc import Sequence

    from project_code_intelligence.mcp.protocol import Json


# NOTE (content-addressed records migration, 2026-08-19): the old `record_backed_files`
# fallback synthesized a file row from orphan records (a record with no matching files
# row for its snapshot) by reading `r.snapshot_id`/`.branch`/`.commit_sha`/`.tree_sha`
# directly off the record. Those columns no longer exist on records post-migration --
# a record's only surviving link to a snapshot is the files-membership join, so a
# record with no matching files row has no snapshot to attribute a synthetic row to.
# The write path now always inserts a files membership row alongside a snapshot's
# records (see HANDOFF.md "Write path"), so this fallback's scenario should no longer
# arise in normal operation. Dropped rather than guessed at; if it turns out to matter,
# re-derive it via a join to `project_code_intel_snapshots` scoped the same way the
# outer query is scoped.
LIST_CODE_INTEL_FILES_SELECT_SLIM = """
            SELECT f.id, f.repo, f.source_path, f.size_bytes, f.language, f.file_role, f.content_class,
                   f.is_generated, f.is_vendor, f.is_test, f.is_source, f.is_build,
                   f.is_config, f.is_doc, f.skipped_reason
            FROM project_code_intel_files f
            """


# See the note above LIST_CODE_INTEL_FILES_SELECT_SLIM: the record-backed fallback
# was dropped, not translated, because records no longer carry the snapshot-varying
# columns it read.
LIST_CODE_INTEL_FILES_SELECT_FULL = """
            SELECT f.id, f.snapshot_id, f.collection, f.repo, f.repo_role, f.branch,
                   f.commit_sha, f.tree_sha, f.source_path, f.git_blob_sha, f.file_sha256,
                   f.size_bytes, f.language, f.file_role, f.content_class,
                   f.is_generated, f.is_vendor, f.is_test, f.is_source, f.is_build,
                   f.is_config, f.is_doc, f.skipped_reason, f.is_untracked,
                   f.indexed_dirty, f.metadata, f.created_at
            FROM project_code_intel_files f
            """


LIST_CODE_INTEL_FILES_BOOLEAN_FILTERS = (
    "is_test",
    "is_doc",
    "is_generated",
    "is_vendor",
    "is_source",
    "is_build",
    "is_config",
    "is_untracked",
)
OVERCONSTRAINED_FALSE_BOOLEAN_FILTER_MIN_COUNT = 5
OVERCONSTRAINED_FALSE_BOOLEAN_FILTER_WITH_PATH_MIN_COUNT = 2


def explicit_false_file_boolean_filters(args: Json) -> list[str]:
    return [
        arg_name
        for arg_name in LIST_CODE_INTEL_FILES_BOOLEAN_FILTERS
        if arg_name in args and not optional_bool(args, arg_name)
    ]


def looks_like_overconstrained_boolean_filter(args: Json, false_filter_count: int) -> bool:
    if false_filter_count >= OVERCONSTRAINED_FALSE_BOOLEAN_FILTER_MIN_COUNT:
        return True
    path_scoped = (
        optional_text(args, "source_path") is not None or optional_text(args, "source_path_prefix") is not None
    )
    return path_scoped and false_filter_count >= OVERCONSTRAINED_FALSE_BOOLEAN_FILTER_WITH_PATH_MIN_COUNT


def overconstrained_boolean_filter_warning(args: Json, rows: Sequence[object]) -> Json | None:
    if rows:
        return None
    filters = explicit_false_file_boolean_filters(args)
    if not looks_like_overconstrained_boolean_filter(args, len(filters)):
        return None
    return {
        **make_warning(
            "overconstrained_boolean_filters",
            message=(
                "Omit boolean filters unless you want to filter for that exact boolean value; "
                "false is an active filter, not a default."
            ),
        ),
        "filters": filters,
    }
