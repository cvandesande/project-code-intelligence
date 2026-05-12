"""Database persistence for code-intelligence snapshots and records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from project_code_intelligence import db
from project_code_intelligence.common import sha256_text
from project_code_intelligence.storage.copy import copy_unchanged_parser_failures, copy_unchanged_records_and_edges
from project_code_intelligence.storage.schema import (
    ensure_schema,
    file_signature,
    latest_snapshot_info,
    previous_file_signatures,
    reset_code_intel_schema,
    row_int,
    schema_migration_versions,
    snapshot_versions_compatible,
)
from project_code_intelligence.storage.static import insert_static_runs

if TYPE_CHECKING:
    from project_code_intelligence.models import IntelEdge, IntelFile, IntelRecord, JsonObject, Snapshot

__all__ = [
    "RecordInsertContext",
    "copy_unchanged_parser_failures",
    "copy_unchanged_records_and_edges",
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
    "reset_code_intel_schema",
    "row_int",
    "schema_migration_versions",
    "snapshot_versions_compatible",
]


@dataclass(frozen=True)
class RecordInsertContext:
    conn: db.DbConnection
    snapshot: Snapshot
    snapshot_id: int
    file_ids: dict[str, int]
    file_hashes: dict[str, str | None]


def replace_repos(conn: db.DbConnection, collection: str, repos: list[str]) -> None:
    for repo in repos:
        _ = conn.execute(
            "DELETE FROM project_code_intel_snapshots WHERE collection = %s AND repo = %s",
            [collection, repo],
        )


def insert_snapshot(conn: db.DbConnection, snapshot: Snapshot) -> int:
    row = conn.execute(
        """
        INSERT INTO project_code_intel_snapshots (
            collection, repo, repo_role, branch, commit_sha, tree_sha, dirty, metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (collection, repo, commit_sha, tree_sha)
        DO UPDATE SET branch = EXCLUDED.branch,
                      repo_role = EXCLUDED.repo_role,
                      dirty = EXCLUDED.dirty,
                      metadata = EXCLUDED.metadata ||
                        CASE
                          WHEN project_code_intel_snapshots.metadata ? 'embedding_contract'
                          THEN jsonb_build_object(
                            'embedding_contract',
                            project_code_intel_snapshots.metadata->'embedding_contract'
                          )
                          ELSE '{}'::jsonb
                        END
        RETURNING id
        """,
        [
            snapshot.collection,
            snapshot.repo,
            snapshot.repo_role,
            snapshot.branch,
            snapshot.commit_sha,
            snapshot.tree_sha,
            snapshot.dirty,
            json.dumps(snapshot.metadata, sort_keys=True, separators=(",", ":")),
        ],
    ).fetchone()
    return row_int(db.require_row(row, "insert snapshot"), "id")


def insert_files(conn: db.DbConnection, snapshot_id: int, files: list[IntelFile]) -> dict[str, int]:
    if not files:
        return {}
    payload = [
        {
            "snapshot_id": snapshot_id,
            "collection": item.collection,
            "repo": item.repo,
            "repo_role": item.repo_role,
            "branch": item.branch,
            "commit_sha": item.commit_sha,
            "tree_sha": item.tree_sha,
            "source_path": item.source_path,
            "git_blob_sha": item.git_blob_sha,
            "file_sha256": item.file_sha256,
            "size_bytes": item.size_bytes,
            "language": item.language,
            "file_role": item.file_role,
            "content_class": item.content_class,
            "is_generated": item.is_generated,
            "is_vendor": item.is_vendor,
            "is_test": item.is_test,
            "is_source": item.is_source,
            "is_build": item.is_build,
            "is_config": item.is_config,
            "is_doc": item.is_doc,
            "skipped_reason": item.skipped_reason,
            "metadata": item.metadata,
        }
        for item in files
    ]
    rows = conn.execute(
        """
        WITH input_rows AS (
            SELECT *
            FROM jsonb_to_recordset(%s::jsonb) AS r(
                snapshot_id bigint,
                collection text,
                repo text,
                repo_role text,
                branch text,
                commit_sha text,
                tree_sha text,
                source_path text,
                git_blob_sha text,
                file_sha256 text,
                size_bytes bigint,
                language text,
                file_role text,
                content_class text,
                is_generated boolean,
                is_vendor boolean,
                is_test boolean,
                is_source boolean,
                is_build boolean,
                is_config boolean,
                is_doc boolean,
                skipped_reason text,
                metadata jsonb
            )
        ),
        upserted AS (
            INSERT INTO project_code_intel_files (
                snapshot_id, collection, repo, repo_role, branch, commit_sha, tree_sha,
                source_path, git_blob_sha, file_sha256, size_bytes, language,
                file_role, content_class, is_generated, is_vendor, is_test,
                is_source, is_build, is_config, is_doc, skipped_reason, metadata
            )
            SELECT
                snapshot_id, collection, repo, repo_role, branch, commit_sha, tree_sha,
                source_path, git_blob_sha, file_sha256, size_bytes, language,
                file_role, content_class, is_generated, is_vendor, is_test,
                is_source, is_build, is_config, is_doc, skipped_reason, metadata
            FROM input_rows
            WHERE true
            ON CONFLICT (snapshot_id, source_path)
            DO UPDATE SET file_sha256 = EXCLUDED.file_sha256,
                          git_blob_sha = EXCLUDED.git_blob_sha,
                          size_bytes = EXCLUDED.size_bytes,
                          language = EXCLUDED.language,
                          file_role = EXCLUDED.file_role,
                          content_class = EXCLUDED.content_class,
                          is_generated = EXCLUDED.is_generated,
                          is_vendor = EXCLUDED.is_vendor,
                          is_test = EXCLUDED.is_test,
                          is_source = EXCLUDED.is_source,
                          is_build = EXCLUDED.is_build,
                          is_config = EXCLUDED.is_config,
                          is_doc = EXCLUDED.is_doc,
                          skipped_reason = EXCLUDED.skipped_reason,
                          metadata = EXCLUDED.metadata
            RETURNING source_path, id
        )
        SELECT source_path, id
        FROM upserted
        """,
        [json.dumps(payload, sort_keys=True, separators=(",", ":"))],
    ).fetchall()
    return {str(row["source_path"]): row_int(row, "id") for row in rows}


def insert_records(
    context: RecordInsertContext,
    records: list[IntelRecord],
) -> int:
    if not records:
        return 0
    params: list[list[object]] = []
    for record in records:
        embedding_hash = sha256_text(record.embedding_text)
        display_hash = sha256_text(record.display_content)
        params.append([
            context.snapshot_id,
            context.file_ids.get(record.source_path),
            record.collection,
            context.snapshot.repo,
            context.snapshot.repo_role,
            context.snapshot.branch,
            context.snapshot.commit_sha,
            context.snapshot.tree_sha,
            record.source_path,
            context.file_hashes.get(record.source_path),
            record.language,
            record.file_role,
            record.content_class,
            record.record_type,
            record.record_id,
            record.parent_record_id,
            record.title,
            record.summary,
            record.embedding_text,
            record.display_content,
            embedding_hash,
            display_hash,
            record.line_start,
            record.line_end,
            record.symbol,
            record.symbol_kind,
            record.confidence_kind,
            record.confidence,
            record.tool,
            record.rule_id,
            record.severity,
            record.analyzer,
            record.analyzer_version,
            record.parser,
            record.parser_version,
            record.chunker_version,
            json.dumps(record.metadata, sort_keys=True, separators=(",", ":")),
            record.embedding,
        ])
    # psycopg3 exposes executemany() on cursors, while this module otherwise
    # uses connection.execute() for single-statement operations.
    with context.conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO project_code_intel_records (
                snapshot_id, file_id, collection, repo, repo_role, branch, commit_sha,
                tree_sha, source_path, file_sha256, language, file_role,
                content_class, record_type, record_id, parent_record_id,
                title, summary, embedding_text, display_content,
                embedding_text_hash, display_hash, line_start, line_end,
                symbol, symbol_kind, confidence_kind, confidence, tool,
                rule_id, severity, analyzer, analyzer_version, parser,
                parser_version, chunker_version, metadata, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
            ON CONFLICT (snapshot_id, record_type, record_id, embedding_text_hash)
            DO UPDATE SET file_id = EXCLUDED.file_id,
                          file_sha256 = EXCLUDED.file_sha256,
                          summary = EXCLUDED.summary,
                          display_content = EXCLUDED.display_content,
                          display_hash = EXCLUDED.display_hash,
                          metadata = EXCLUDED.metadata,
                          embedding = coalesce(EXCLUDED.embedding, project_code_intel_records.embedding)
            """,
            params,
        )
    return len(records)


def insert_edges(conn: db.DbConnection, snapshot: Snapshot, snapshot_id: int, edges: list[IntelEdge]) -> int:
    if not edges:
        return 0
    params = [
        [
            snapshot_id,
            snapshot.collection,
            snapshot.repo,
            snapshot.commit_sha,
            edge.source_record_id,
            edge.target_record_id,
            edge.edge_type,
            edge.source_symbol,
            edge.target_symbol,
            edge.source_path,
            edge.target_path,
            edge.confidence_kind,
            json.dumps(edge.metadata, sort_keys=True, separators=(",", ":")),
        ]
        for edge in edges
    ]
    # psycopg3 exposes executemany() on cursors, while this module otherwise
    # uses connection.execute() for single-statement operations.
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO project_code_intel_edges (
                snapshot_id, collection, repo, commit_sha, source_record_id, target_record_id,
                edge_type, source_symbol, target_symbol, source_path, target_path,
                confidence_kind, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT DO NOTHING
            """,
            params,
        )
    return len(edges)


def parser_failure_metadata(failure: JsonObject) -> JsonObject:
    return {key: value for key, value in failure.items() if key not in {"source_path", "language", "parser", "error"}}


def insert_parser_failures(
    conn: db.DbConnection,
    snapshot: Snapshot,
    snapshot_id: int,
    failures: list[JsonObject],
) -> int:
    if not failures:
        return 0
    params = [
        [
            snapshot_id,
            snapshot.collection,
            snapshot.repo,
            snapshot.commit_sha,
            str(failure.get("source_path") or ""),
            failure.get("language"),
            str(failure.get("parser") or "unknown"),
            str(failure.get("error") or "")[:2000],
            json.dumps(parser_failure_metadata(failure), sort_keys=True, separators=(",", ":")),
        ]
        for failure in failures
    ]
    # psycopg3 exposes executemany() on cursors, while this module otherwise
    # uses connection.execute() for single-statement operations.
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO project_code_intel_parser_failures (
                snapshot_id, collection, repo, commit_sha, source_path,
                language, parser, error, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT DO NOTHING
            """,
            params,
        )
    return len(failures)
