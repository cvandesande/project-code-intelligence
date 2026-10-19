"""Copy-forward helpers for unchanged snapshot rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from project_code_intelligence import db
from project_code_intelligence.storage.schema import row_int

if TYPE_CHECKING:
    from project_code_intelligence.models import Snapshot


def copy_unchanged_parser_failures(
    conn: db.DbConnection,
    *,
    previous_snapshot_id: int | None,
    snapshot: Snapshot,
    snapshot_id: int,
    unchanged_paths: set[str],
) -> tuple[int, list[str]]:
    """Copy parser-failure rows forward; return (count, source_paths).

    The returned paths are the rows actually inserted (after ON CONFLICT
    filtering), in SQL output order. Callers that only need the count can
    discard the list.
    """
    if not previous_snapshot_id or previous_snapshot_id == snapshot_id or not unchanged_paths:
        return 0, []
    rows = conn.execute(
        """
        WITH copied AS (
            INSERT INTO project_code_intel_parser_failures (
                snapshot_id, collection, repo, commit_sha, source_path,
                language, parser, error, metadata
            )
            SELECT
                %s, %s, %s, %s, source_path, language, parser, error, metadata
            FROM project_code_intel_parser_failures
            WHERE snapshot_id = %s
              AND source_path = ANY(%s)
            ON CONFLICT DO NOTHING
            RETURNING source_path
        )
        SELECT source_path FROM copied
        """,
        [
            snapshot_id,
            snapshot.collection,
            snapshot.repo,
            snapshot.commit_sha,
            previous_snapshot_id,
            sorted(unchanged_paths),
        ],
    ).fetchall()
    paths: list[str] = []
    for row in rows:
        value = row["source_path"]
        if isinstance(value, str):
            paths.append(value)
    return len(paths), paths


def copy_unchanged_edges(
    conn: db.DbConnection,
    *,
    previous_snapshot_id: int | None,
    snapshot: Snapshot,
    snapshot_id: int,
    unchanged_paths: set[str],
) -> int:
    """Copy edge rows forward for files whose content did not change.

    Records no longer need this (schema v2 made them content-addressed:
    `insert_records` skips re-inserting unchanged content via `ON CONFLICT
    DO NOTHING`). Edges stay per-snapshot -- rows are small, carry no
    vector or tsvector -- so they are still copied forward here.
    """
    if not previous_snapshot_id or previous_snapshot_id == snapshot_id or not unchanged_paths:
        return 0
    paths = sorted(unchanged_paths)
    edge_row = conn.execute(
        """
        WITH copied AS (
            INSERT INTO project_code_intel_edges (
                snapshot_id, collection, repo, commit_sha, source_record_id, target_record_id,
                edge_type, source_symbol, target_symbol, source_path, target_path,
                confidence_kind, metadata
            )
            SELECT
                %s, %s, %s, %s, source_record_id, target_record_id,
                edge_type, source_symbol, target_symbol, source_path, target_path,
                confidence_kind, metadata
            FROM project_code_intel_edges
            WHERE snapshot_id = %s
              AND source_path = ANY(%s)
            ON CONFLICT DO NOTHING
            RETURNING 1
        )
        SELECT count(*) AS count FROM copied
        """,
        [snapshot_id, snapshot.collection, snapshot.repo, snapshot.commit_sha, previous_snapshot_id, paths],
    ).fetchone()
    return row_int(db.require_row(edge_row, "copy unchanged edges"), "count")
