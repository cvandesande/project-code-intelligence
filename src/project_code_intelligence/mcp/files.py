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


LIST_CODE_INTEL_FILES_SELECT_SLIM = """
            WITH record_backed_files AS (
                SELECT
                    NULL::bigint AS id,
                    r.snapshot_id,
                    r.collection,
                    r.repo,
                    r.repo_role,
                    r.branch,
                    r.commit_sha,
                    r.tree_sha,
                    r.source_path,
                    NULL::text AS git_blob_sha,
                    max(r.file_sha256) AS file_sha256,
                    NULL::bigint AS size_bytes,
                    (array_agg(r.language ORDER BY r.id))[1] AS language,
                    (array_agg(r.file_role ORDER BY r.id))[1] AS file_role,
                    (array_agg(r.content_class ORDER BY r.id))[1] AS content_class,
                    bool_or(r.file_role = 'generated' OR r.content_class = 'generated') AS is_generated,
                    bool_or(r.file_role = 'vendor' OR r.content_class = 'vendor') AS is_vendor,
                    bool_or(r.file_role = 'test' OR r.content_class = 'test') AS is_test,
                    bool_or(
                        r.file_role IN ('source', 'source-include')
                        OR r.content_class = 'source'
                        OR r.language = ANY(%s::text[])
                    ) AS is_source,
                    bool_or(
                        r.file_role IN ('build', 'build-include', 'build-script', 'package', 'project-manifest')
                        OR r.content_class = 'build'
                    ) AS is_build,
                    bool_or(r.file_role = 'config' OR r.content_class = 'config') AS is_config,
                    bool_or(r.file_role = 'doc' OR r.content_class = 'doc' OR r.language = 'doc') AS is_doc,
                    NULL::text AS skipped_reason,
                    false AS is_untracked,
                    false AS indexed_dirty,
                    jsonb_build_object('inventory_source', 'records') AS metadata,
                    min(r.created_at) AS created_at
                FROM project_code_intel_records r
                LEFT JOIN project_code_intel_files existing
                  ON existing.snapshot_id = r.snapshot_id
                 AND existing.source_path = r.source_path
                WHERE existing.id IS NULL
                GROUP BY
                    r.snapshot_id,
                    r.collection,
                    r.repo,
                    r.repo_role,
                    r.branch,
                    r.commit_sha,
                    r.tree_sha,
                    r.source_path
            ),
            file_inventory AS (
                SELECT
                    f.id,
                    f.snapshot_id,
                    f.collection,
                    f.repo,
                    f.repo_role,
                    f.branch,
                    f.commit_sha,
                    f.tree_sha,
                    f.source_path,
                    f.git_blob_sha,
                    f.file_sha256,
                    f.size_bytes,
                    f.language,
                    f.file_role,
                    f.content_class,
                    f.is_generated,
                    f.is_vendor,
                    f.is_test,
                    f.is_source,
                    f.is_build,
                    f.is_config,
                    f.is_doc,
                    f.skipped_reason,
                    f.is_untracked,
                    f.indexed_dirty,
                    f.metadata,
                    f.created_at
                FROM project_code_intel_files f
                UNION ALL
                SELECT
                    id,
                    snapshot_id,
                    collection,
                    repo,
                    repo_role,
                    branch,
                    commit_sha,
                    tree_sha,
                    source_path,
                    git_blob_sha,
                    file_sha256,
                    size_bytes,
                    language,
                    file_role,
                    content_class,
                    is_generated,
                    is_vendor,
                    is_test,
                    is_source,
                    is_build,
                    is_config,
                    is_doc,
                    skipped_reason,
                    is_untracked,
                    indexed_dirty,
                    metadata,
                    created_at
                FROM record_backed_files
            )
            SELECT f.id, f.source_path, f.size_bytes, f.language, f.file_role, f.content_class,
                   f.is_generated, f.is_vendor, f.is_test, f.is_source, f.is_build,
                   f.is_config, f.is_doc, f.skipped_reason
            FROM file_inventory f
            """


LIST_CODE_INTEL_FILES_SELECT_FULL = """
            WITH record_backed_files AS (
                SELECT
                    NULL::bigint AS id,
                    r.snapshot_id,
                    r.collection,
                    r.repo,
                    r.repo_role,
                    r.branch,
                    r.commit_sha,
                    r.tree_sha,
                    r.source_path,
                    NULL::text AS git_blob_sha,
                    max(r.file_sha256) AS file_sha256,
                    NULL::bigint AS size_bytes,
                    (array_agg(r.language ORDER BY r.id))[1] AS language,
                    (array_agg(r.file_role ORDER BY r.id))[1] AS file_role,
                    (array_agg(r.content_class ORDER BY r.id))[1] AS content_class,
                    bool_or(r.file_role = 'generated' OR r.content_class = 'generated') AS is_generated,
                    bool_or(r.file_role = 'vendor' OR r.content_class = 'vendor') AS is_vendor,
                    bool_or(r.file_role = 'test' OR r.content_class = 'test') AS is_test,
                    bool_or(
                        r.file_role IN ('source', 'source-include')
                        OR r.content_class = 'source'
                        OR r.language = ANY(%s::text[])
                    ) AS is_source,
                    bool_or(
                        r.file_role IN ('build', 'build-include', 'build-script', 'package', 'project-manifest')
                        OR r.content_class = 'build'
                    ) AS is_build,
                    bool_or(r.file_role = 'config' OR r.content_class = 'config') AS is_config,
                    bool_or(r.file_role = 'doc' OR r.content_class = 'doc' OR r.language = 'doc') AS is_doc,
                    NULL::text AS skipped_reason,
                    false AS is_untracked,
                    false AS indexed_dirty,
                    jsonb_build_object('inventory_source', 'records') AS metadata,
                    min(r.created_at) AS created_at
                FROM project_code_intel_records r
                LEFT JOIN project_code_intel_files existing
                  ON existing.snapshot_id = r.snapshot_id
                 AND existing.source_path = r.source_path
                WHERE existing.id IS NULL
                GROUP BY
                    r.snapshot_id,
                    r.collection,
                    r.repo,
                    r.repo_role,
                    r.branch,
                    r.commit_sha,
                    r.tree_sha,
                    r.source_path
            ),
            file_inventory AS (
                SELECT
                    f.id,
                    f.snapshot_id,
                    f.collection,
                    f.repo,
                    f.repo_role,
                    f.branch,
                    f.commit_sha,
                    f.tree_sha,
                    f.source_path,
                    f.git_blob_sha,
                    f.file_sha256,
                    f.size_bytes,
                    f.language,
                    f.file_role,
                    f.content_class,
                    f.is_generated,
                    f.is_vendor,
                    f.is_test,
                    f.is_source,
                    f.is_build,
                    f.is_config,
                    f.is_doc,
                    f.skipped_reason,
                    f.is_untracked,
                    f.indexed_dirty,
                    f.metadata,
                    f.created_at
                FROM project_code_intel_files f
                UNION ALL
                SELECT
                    id,
                    snapshot_id,
                    collection,
                    repo,
                    repo_role,
                    branch,
                    commit_sha,
                    tree_sha,
                    source_path,
                    git_blob_sha,
                    file_sha256,
                    size_bytes,
                    language,
                    file_role,
                    content_class,
                    is_generated,
                    is_vendor,
                    is_test,
                    is_source,
                    is_build,
                    is_config,
                    is_doc,
                    skipped_reason,
                    is_untracked,
                    indexed_dirty,
                    metadata,
                    created_at
                FROM record_backed_files
            )
            SELECT f.id, f.snapshot_id, f.collection, f.repo, f.repo_role, f.branch,
                   f.commit_sha, f.tree_sha, f.source_path, f.git_blob_sha, f.file_sha256,
                   f.size_bytes, f.language, f.file_role, f.content_class,
                   f.is_generated, f.is_vendor, f.is_test, f.is_source, f.is_build,
                   f.is_config, f.is_doc, f.skipped_reason, f.is_untracked,
                   f.indexed_dirty, f.metadata, f.created_at
            FROM file_inventory f
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
