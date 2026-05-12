"""Schema, migration, and snapshot compatibility helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from project_code_intelligence import db, profile_context
from project_code_intelligence.models import CHUNKER_VERSION, PARSER_VERSION, SCHEMA_VERSION

if TYPE_CHECKING:
    from project_code_intelligence.models import IntelFile, JsonObject

CODE_INTEL_TABLES = [
    "project_code_intel_schema_migrations",
    "project_code_intel_static_code_flows",
    "project_code_intel_static_locations",
    "project_code_intel_static_findings",
    "project_code_intel_static_rules",
    "project_code_intel_static_runs",
    "project_code_intel_parser_failures",
    "project_code_intel_edges",
    "project_code_intel_records",
    "project_code_intel_files",
    "project_code_intel_snapshots",
]


def file_signature(item: IntelFile) -> str:
    if item.file_sha256:
        return f"sha256:{item.file_sha256}"
    if item.git_blob_sha:
        return f"blob:{item.git_blob_sha}"
    return f"meta:{item.size_bytes}:{item.skipped_reason or ''}"


def row_int(row: db.DbRow, key: str) -> int:
    value = row[key]
    if isinstance(value, bool):
        raise TypeError(f"{key} is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise TypeError(f"{key} is not an integer")


def snapshot_versions_compatible(metadata: JsonObject | None) -> bool:
    if not metadata:
        return False
    return (
        metadata.get("schema_version") == SCHEMA_VERSION
        and metadata.get("chunker_version") == CHUNKER_VERSION
        and metadata.get("parser_version") == PARSER_VERSION
        and metadata.get("profile_name") == profile_context.active_profile.name
        and metadata.get("profile_version") == profile_context.active_profile.version
    )


def latest_snapshot_info(conn: db.DbConnection, collection: str, repo: str) -> JsonObject | None:
    row = conn.execute(
        """
        SELECT id, collection, repo, commit_sha, tree_sha, metadata
        FROM project_code_intel_snapshots
        WHERE collection = %s AND repo = %s
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        [collection, repo],
    ).fetchone()
    return cast("JsonObject", dict(row)) if row else None


def previous_file_signatures(conn: db.DbConnection, snapshot_id: int) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT source_path, file_sha256, git_blob_sha, size_bytes, skipped_reason
        FROM project_code_intel_files
        WHERE snapshot_id = %s
        """,
        [snapshot_id],
    ).fetchall()
    signatures: dict[str, str] = {}
    for row in rows:
        source_path = str(row["source_path"])
        if row["file_sha256"]:
            signatures[source_path] = f"sha256:{row['file_sha256']}"
        elif row["git_blob_sha"]:
            signatures[source_path] = f"blob:{row['git_blob_sha']}"
        else:
            signatures[source_path] = f"meta:{row['size_bytes']}:{row['skipped_reason'] or ''}"
    return signatures


def ensure_schema(conn: db.DbConnection) -> None:
    _ = conn.execute(db.schema_sql())
    record_schema_migration(conn)


def record_schema_migration(conn: db.DbConnection) -> None:
    _ = conn.execute(
        """
        INSERT INTO project_code_intel_schema_migrations (version)
        VALUES (%s)
        ON CONFLICT (version) DO NOTHING
        """,
        [SCHEMA_VERSION],
    )


def schema_migration_versions(conn: db.DbConnection) -> list[str]:
    rows = conn.execute(
        """
        SELECT version
        FROM project_code_intel_schema_migrations
        ORDER BY applied_at, version
        """
    ).fetchall()
    return [str(row["version"]) for row in rows]


def reset_code_intel_schema(conn: db.DbConnection) -> None:
    for table in CODE_INTEL_TABLES:
        _ = conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
