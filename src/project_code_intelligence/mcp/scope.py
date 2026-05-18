"""Shared scope-warning helpers for MCP tools.

When a filter (repo, snapshot_id, language, source_path, ...) is supplied but
yields no rows, tools surface a structured warning so the client can
distinguish "filter spelled wrong" from "no matching data". These helpers
build those warnings consistently across every tool that does scoped lookups.

The `kind` field of every emitted warning is typed via
`project_code_intelligence.mcp.taxonomies.WarningKind` so typos at the
producer or consumer side fail type-checking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from project_code_intelligence import db
from project_code_intelligence.mcp.filters import (
    normalize_source_path_filter,
    query_with_where,
    scoped_snapshot_table_collection_repo_clauses,
)
from project_code_intelligence.mcp.protocol import optional_text, scoped_collection

if TYPE_CHECKING:
    from collections.abc import Sequence

    from project_code_intelligence.mcp.protocol import Json
    from project_code_intelligence.mcp.taxonomies import WarningKind
    from project_code_intelligence.models import JsonValue


# Enum-ish filter dimensions for which we emit `empty_<dim>_scope` warnings when a value is
# supplied but no rows match. Snapshot_id has its own warning shape (it's numeric, not enum).
_ENUM_FILTER_DIMENSIONS: tuple[str, ...] = ("language", "file_role", "record_type", "content_class")


def make_warning(kind: WarningKind, **fields: object) -> Json:
    """Construct a warning dict with a type-checked `kind` field.

    Centralizes the `{"kind": "X", ...}` pattern so the WarningKind alias is
    enforced at every warning-emission site without each call needing its own
    typed local. Caller passes kind as a positional argument; basedpyright
    rejects strings outside the WarningKind literal set.
    """
    warning: Json = {"kind": kind}
    for key, value in fields.items():
        if value is not None:
            warning[key] = cast("JsonValue", value)
    return warning


def attach_warnings(response: Json, warnings: Sequence[Json]) -> None:
    """Set response['warnings'] when warnings is non-empty.

    Centralizes the conditional-assign pattern that several tools repeat, and keeps the
    `warnings` list out of the caller's local-variable count (matters because some handlers
    are right at PLR0914's threshold).
    """
    if warnings:
        response["warnings"] = cast("JsonValue", list(warnings))


def repo_scope_exists(conn: db.DbConnection, args: Json) -> bool | None:
    if not optional_text(args, "repo"):
        return None
    clauses, params = scoped_snapshot_table_collection_repo_clauses(args, "s")
    return (
        conn.execute(
            db.query_sql(
                query_with_where(
                    """
            SELECT 1
            FROM project_code_intel_snapshots s
            """,
                    clauses,
                    """
            LIMIT 1
            """,
                )
            ),
            params,
        ).fetchone()
        is not None
    )


def empty_repo_scope_warning(repo: str) -> Json:
    return make_warning(
        "empty_repo_scope",
        repo=repo,
        message="no results matched this repo filter; run code_intel_status without repo to see valid repo keys",
    )


# Maps the enum-ish filter dimension name to its typed WarningKind so the
# f-string above can't drift away from the WarningKind alias.
_EMPTY_ENUM_WARNING_KINDS: dict[str, WarningKind] = {
    "language": "empty_language_scope",
    "file_role": "empty_file_role_scope",
    "record_type": "empty_record_type_scope",
    "content_class": "empty_content_class_scope",
}


def empty_enum_scope_warning(dimension: str, value: str) -> Json:
    return make_warning(
        _EMPTY_ENUM_WARNING_KINDS[dimension],
        **{
            dimension: value,
            "message": (
                f"no results matched the {dimension}={value!r} filter; run code_intel_status with "
                f"include_queryability=true to see valid {dimension} values in this index"
            ),
        },
    )


def empty_snapshot_scope_warning(snapshot_id: int) -> Json:
    return make_warning(
        "empty_snapshot_scope",
        snapshot_id=snapshot_id,
        message=(
            f"snapshot_id={snapshot_id} does not exist in this index; "
            "run code_intel_status with include_snapshots=true to see valid snapshot ids"
        ),
    )


def path_scope_matches_repo_root(args: Json) -> bool:
    raw_prefix = optional_text(args, "source_path_prefix")
    if not raw_prefix:
        return False
    prefix = normalize_source_path_filter(raw_prefix, "source_path_prefix")
    repo = optional_text(args, "repo")
    if repo and prefix == normalize_source_path_filter(repo, "repo"):
        return True
    collection = scoped_collection(args)
    return bool(collection and prefix == normalize_source_path_filter(collection, "collection"))


def scope_filter_warnings(
    args: Json,
    rows: Sequence[object],
    *,
    repo_exists: bool | None = None,
    missing_snapshot_warning: Json | None = None,
) -> list[Json]:
    warnings: list[Json] = []
    source_path = optional_text(args, "source_path")
    source_path_prefix = optional_text(args, "source_path_prefix")
    repo = optional_text(args, "repo")
    if path_scope_matches_repo_root(args):
        warnings.append(
            make_warning(
                "repo_root_path_scope",
                message="source_path_prefix points at the repo root and is equivalent to a broad repo filter",
                source_path_prefix=source_path_prefix,
            )
        )
    if rows:
        return warnings
    if repo and repo_exists is False:
        warnings.append(empty_repo_scope_warning(repo))
    if missing_snapshot_warning is not None:
        warnings.append(missing_snapshot_warning)
    for dimension in _ENUM_FILTER_DIMENSIONS:
        value = optional_text(args, dimension)
        if value:
            warnings.append(empty_enum_scope_warning(dimension, value))
    if source_path or source_path_prefix:
        warnings.append(
            make_warning(
                "empty_path_scope",
                message=(
                    "no results matched this path scope; source_path and source_path_prefix are repo-relative, "
                    "and directories should use source_path_prefix"
                ),
                source_path=source_path,
                source_path_prefix=source_path_prefix,
            )
        )
    return warnings
