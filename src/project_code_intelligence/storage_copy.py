"""Copy-forward helpers for unchanged snapshot rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from project_code_intelligence import db
from project_code_intelligence.storage_schema import row_int

if TYPE_CHECKING:
    from project_code_intelligence.models import Snapshot


def copy_unchanged_parser_failures(
    conn: db.DbConnection,
    *,
    previous_snapshot_id: int | None,
    snapshot: Snapshot,
    snapshot_id: int,
    unchanged_paths: set[str],
) -> int:
    if not previous_snapshot_id or previous_snapshot_id == snapshot_id or not unchanged_paths:
        return 0
    row = conn.execute(
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
            RETURNING 1
        )
        SELECT count(*) AS count FROM copied
        """,
        [
            snapshot_id,
            snapshot.collection,
            snapshot.repo,
            snapshot.commit_sha,
            previous_snapshot_id,
            sorted(unchanged_paths),
        ],
    ).fetchone()
    return row_int(db.require_row(row, "copy unchanged parser failures"), "count")


def copy_unchanged_records_and_edges(
    conn: db.DbConnection,
    *,
    previous_snapshot_id: int | None,
    snapshot: Snapshot,
    snapshot_id: int,
    unchanged_paths: set[str],
) -> tuple[int, int]:
    if not previous_snapshot_id or previous_snapshot_id == snapshot_id or not unchanged_paths:
        return 0, 0
    paths = sorted(unchanged_paths)
    record_row = conn.execute(
        """
        WITH copied AS (
            INSERT INTO project_code_intel_records (
                snapshot_id, file_id, collection, repo, repo_role, branch, commit_sha,
                tree_sha, source_path, file_sha256, language, file_role,
                content_class, record_type, record_id, parent_record_id,
                title, summary, embedding_text, display_content,
                embedding_text_hash, display_hash, line_start, line_end,
                byte_start, byte_end, symbol, symbol_kind, confidence_kind,
                confidence, tool, rule_id, severity, analyzer, analyzer_version,
                parser, parser_version, chunker_version, metadata, embedding
            )
            SELECT
                %s, nf.id, %s, %s, %s, %s, %s,
                %s, r.source_path, nf.file_sha256, nf.language, nf.file_role,
                nf.content_class, r.record_type, r.record_id, r.parent_record_id,
                r.title, r.summary, r.embedding_text, r.display_content,
                r.embedding_text_hash, r.display_hash, r.line_start, r.line_end,
                r.byte_start, r.byte_end, r.symbol, r.symbol_kind, r.confidence_kind,
                r.confidence, r.tool, r.rule_id, r.severity, r.analyzer, r.analyzer_version,
                r.parser, r.parser_version, r.chunker_version, r.metadata, r.embedding
            FROM project_code_intel_records r
            JOIN project_code_intel_files nf
              ON nf.snapshot_id = %s
             AND nf.source_path = r.source_path
            WHERE r.snapshot_id = %s
              AND r.source_path = ANY(%s)
            ON CONFLICT (snapshot_id, record_type, record_id, embedding_text_hash)
            DO UPDATE SET file_id = EXCLUDED.file_id,
                          repo_role = EXCLUDED.repo_role,
                          branch = EXCLUDED.branch,
                          commit_sha = EXCLUDED.commit_sha,
                          tree_sha = EXCLUDED.tree_sha,
                          file_sha256 = EXCLUDED.file_sha256,
                          language = EXCLUDED.language,
                          file_role = EXCLUDED.file_role,
                          content_class = EXCLUDED.content_class,
                          metadata = EXCLUDED.metadata,
                          embedding = coalesce(EXCLUDED.embedding, project_code_intel_records.embedding)
            RETURNING 1
        )
        SELECT count(*) AS count FROM copied
        """,
        [
            snapshot_id,
            snapshot.collection,
            snapshot.repo,
            snapshot.repo_role,
            snapshot.branch,
            snapshot.commit_sha,
            snapshot.tree_sha,
            snapshot_id,
            previous_snapshot_id,
            paths,
        ],
    ).fetchone()
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
    return (
        row_int(db.require_row(record_row, "copy unchanged records"), "count"),
        row_int(db.require_row(edge_row, "copy unchanged edges"), "count"),
    )
