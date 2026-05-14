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
    previous_file_state_signature,
    previous_file_states,
    row_int,
    schema_migration_versions,
    snapshot_versions_compatible,
)
from project_code_intelligence.storage.static import insert_static_runs

if TYPE_CHECKING:
    from collections.abc import Callable

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


def delete_repo_data(conn: db.DbConnection, collection: str, repos: list[str]) -> dict[str, int]:
    """Delete all snapshots and cascading data for the given repos. Returns deleted counts per repo."""
    deleted: dict[str, int] = {}
    for repo in repos:
        rows = conn.execute(
            "DELETE FROM project_code_intel_snapshots WHERE collection = %s AND repo = %s RETURNING id",
            [collection, repo],
        ).fetchall()
        deleted[repo] = len(rows)
    return deleted


def delete_all_code_intel_data(conn: db.DbConnection) -> int:
    """Delete all code-intelligence snapshots and cascading data. Returns deleted snapshot count."""
    row = conn.execute("SELECT count(*) AS count FROM project_code_intel_snapshots").fetchone()
    snapshot_count = row_int(db.require_row(row, "snapshot count"), "count")
    _ = conn.execute(
        """
        TRUNCATE
            project_code_intel_snapshots,
            project_code_intel_files,
            project_code_intel_records,
            project_code_intel_edges,
            project_code_intel_parser_failures,
            project_code_intel_static_runs,
            project_code_intel_static_rules,
            project_code_intel_static_findings,
            project_code_intel_static_locations,
            project_code_intel_static_code_flows
        RESTART IDENTITY CASCADE
        """
    )
    return snapshot_count


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
            "is_untracked": item.is_untracked,
            "indexed_dirty": item.indexed_dirty,
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
                is_untracked boolean,
                indexed_dirty boolean,
                metadata jsonb
            )
        ),
        upserted AS (
            INSERT INTO project_code_intel_files (
                snapshot_id, collection, repo, repo_role, branch, commit_sha, tree_sha,
                source_path, git_blob_sha, file_sha256, size_bytes, language,
                file_role, content_class, is_generated, is_vendor, is_test,
                is_source, is_build, is_config, is_doc, skipped_reason, is_untracked, indexed_dirty, metadata
            )
            SELECT
                snapshot_id, collection, repo, repo_role, branch, commit_sha, tree_sha,
                source_path, git_blob_sha, file_sha256, size_bytes, language,
                file_role, content_class, is_generated, is_vendor, is_test,
                is_source, is_build, is_config, is_doc, skipped_reason, is_untracked, indexed_dirty, metadata
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
                          is_untracked = EXCLUDED.is_untracked,
                          indexed_dirty = EXCLUDED.indexed_dirty,
                          metadata = EXCLUDED.metadata
            RETURNING source_path, id
        )
        SELECT source_path, id
        FROM upserted
        """,
        [json.dumps(payload, sort_keys=True, separators=(",", ":"))],
    ).fetchall()
    return {str(row["source_path"]): row_int(row, "id") for row in rows}


_INSERT_BATCH_SIZE = 5_000


def insert_records(
    context: RecordInsertContext,
    records: list[IntelRecord],
    *,
    progress_fn: Callable[[int], None] | None = None,
) -> int:
    if not records:
        return 0
    payload: list[dict[str, object]] = []
    for record in records:
        embedding_hash = sha256_text(record.embedding_text)
        display_hash = sha256_text(record.display_content)
        payload.append({
            "snapshot_id": context.snapshot_id,
            "file_id": context.file_ids.get(record.source_path),
            "collection": record.collection,
            "repo": context.snapshot.repo,
            "repo_role": context.snapshot.repo_role,
            "branch": context.snapshot.branch,
            "commit_sha": context.snapshot.commit_sha,
            "tree_sha": context.snapshot.tree_sha,
            "source_path": record.source_path,
            "file_sha256": context.file_hashes.get(record.source_path),
            "language": record.language,
            "file_role": record.file_role,
            "content_class": record.content_class,
            "record_type": record.record_type,
            "record_id": record.record_id,
            "parent_record_id": record.parent_record_id,
            "title": record.title,
            "summary": record.summary,
            "embedding_text": record.embedding_text,
            "display_content": record.display_content,
            "embedding_text_hash": embedding_hash,
            "display_hash": display_hash,
            "line_start": record.line_start,
            "line_end": record.line_end,
            "symbol": record.symbol,
            "symbol_kind": record.symbol_kind,
            "confidence_kind": record.confidence_kind,
            "confidence": record.confidence,
            "tool": record.tool,
            "rule_id": record.rule_id,
            "severity": record.severity,
            "analyzer": record.analyzer,
            "analyzer_version": record.analyzer_version,
            "parser": record.parser,
            "parser_version": record.parser_version,
            "chunker_version": record.chunker_version,
            "metadata": record.metadata,
            "embedding": record.embedding,
        })
    for i in range(0, len(payload), _INSERT_BATCH_SIZE):
        batch = payload[i : i + _INSERT_BATCH_SIZE]
        _ = context.conn.execute(
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
            SELECT
                snapshot_id, file_id, collection, repo, repo_role, branch, commit_sha,
                tree_sha, source_path, file_sha256, language, file_role,
                content_class, record_type, record_id, parent_record_id,
                title, summary, embedding_text, display_content,
                embedding_text_hash, display_hash, line_start, line_end,
                symbol, symbol_kind, confidence_kind, confidence, tool,
                rule_id, severity, analyzer, analyzer_version, parser,
                parser_version, chunker_version, metadata, embedding::vector
            FROM jsonb_to_recordset(%s::jsonb) AS r(
                snapshot_id bigint,
                file_id bigint,
                collection text,
                repo text,
                repo_role text,
                branch text,
                commit_sha text,
                tree_sha text,
                source_path text,
                file_sha256 text,
                language text,
                file_role text,
                content_class text,
                record_type text,
                record_id text,
                parent_record_id text,
                title text,
                summary text,
                embedding_text text,
                display_content text,
                embedding_text_hash text,
                display_hash text,
                line_start integer,
                line_end integer,
                symbol text,
                symbol_kind text,
                confidence_kind text,
                confidence real,
                tool text,
                rule_id text,
                severity text,
                analyzer text,
                analyzer_version text,
                parser text,
                parser_version text,
                chunker_version text,
                metadata jsonb,
                embedding text
            )
            ON CONFLICT (snapshot_id, record_type, record_id, embedding_text_hash)
            DO UPDATE SET file_id = EXCLUDED.file_id,
                          file_sha256 = EXCLUDED.file_sha256,
                          summary = EXCLUDED.summary,
                          display_content = EXCLUDED.display_content,
                          display_hash = EXCLUDED.display_hash,
                          metadata = EXCLUDED.metadata,
                          embedding = coalesce(EXCLUDED.embedding, project_code_intel_records.embedding)
            """,
            [json.dumps(batch, sort_keys=True, separators=(",", ":"))],
        )
        if progress_fn is not None:
            progress_fn(len(batch))
    return len(records)


def insert_edges(
    conn: db.DbConnection,
    snapshot: Snapshot,
    snapshot_id: int,
    edges: list[IntelEdge],
    *,
    progress_fn: Callable[[int], None] | None = None,
) -> int:
    if not edges:
        return 0
    payload = [
        {
            "snapshot_id": snapshot_id,
            "collection": snapshot.collection,
            "repo": snapshot.repo,
            "commit_sha": snapshot.commit_sha,
            "source_record_id": edge.source_record_id,
            "target_record_id": edge.target_record_id,
            "edge_type": edge.edge_type,
            "source_symbol": edge.source_symbol,
            "target_symbol": edge.target_symbol,
            "source_path": edge.source_path,
            "target_path": edge.target_path,
            "confidence_kind": edge.confidence_kind,
            "metadata": edge.metadata,
        }
        for edge in edges
    ]
    for i in range(0, len(payload), _INSERT_BATCH_SIZE):
        batch = payload[i : i + _INSERT_BATCH_SIZE]
        _ = conn.execute(
            """
            INSERT INTO project_code_intel_edges (
                snapshot_id, collection, repo, commit_sha, source_record_id, target_record_id,
                edge_type, source_symbol, target_symbol, source_path, target_path,
                confidence_kind, metadata
            )
            SELECT
                snapshot_id, collection, repo, commit_sha, source_record_id, target_record_id,
                edge_type, source_symbol, target_symbol, source_path, target_path,
                confidence_kind, metadata
            FROM jsonb_to_recordset(%s::jsonb) AS r(
                snapshot_id bigint,
                collection text,
                repo text,
                commit_sha text,
                source_record_id text,
                target_record_id text,
                edge_type text,
                source_symbol text,
                target_symbol text,
                source_path text,
                target_path text,
                confidence_kind text,
                metadata jsonb
            )
            ON CONFLICT DO NOTHING
            """,
            [json.dumps(batch, sort_keys=True, separators=(",", ":"))],
        )
        if progress_fn is not None:
            progress_fn(len(batch))
    return len(edges)


def resolve_edge_targets(conn: db.DbConnection, snapshot_id: int) -> int:
    """Fill target_record_id / target_path on edges that have only target_symbol.

    call_candidate edges are created with target_symbol (the callee name) but no
    target_record_id, because at parse time we only know the callee by name.  This
    pass joins against symbol_definition records in the same snapshot and fills in
    the stable record_id string.  When multiple records share the same symbol name
    the lexicographically first source_path wins — an acknowledged heuristic.
    """
    rows = conn.execute(
        """
        UPDATE project_code_intel_edges e
        SET target_record_id = matches.record_id,
            target_path       = matches.source_path
        FROM (
            SELECT e2.id AS edge_id,
                   r.record_id,
                   r.source_path
            FROM   project_code_intel_edges e2
            CROSS JOIN LATERAL (
                SELECT record_id, source_path
                FROM   project_code_intel_records
                WHERE  snapshot_id = e2.snapshot_id
                  AND  symbol      = e2.target_symbol
                  AND  record_type = 'symbol_definition'
                ORDER  BY source_path
                LIMIT  1
            ) r
            WHERE  e2.snapshot_id      = %s
              AND  e2.target_record_id IS NULL
              AND  e2.target_symbol    IS NOT NULL
        ) matches
        WHERE e.id          = matches.edge_id
          AND e.snapshot_id = %s
        RETURNING e.id
        """,
        [snapshot_id, snapshot_id],
    ).fetchall()
    return len(rows)


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


def stamp_embed_types(conn: db.DbConnection, snapshot_ids: list[int], embed_types: set[str]) -> None:
    """Record which record types were embedded for a set of snapshots.

    Written as a jsonb merge so other metadata keys (e.g. embedding_contract)
    are preserved.
    """
    _ = conn.execute(
        """
        UPDATE project_code_intel_snapshots
        SET metadata = metadata || jsonb_build_object('embed_record_types', %s::jsonb)
        WHERE id = ANY(%s)
        """,
        [json.dumps(sorted(embed_types)), snapshot_ids],
    )


def prune_old_snapshots(conn: db.DbConnection, collection: str, repo: str, keep: int = 5) -> int:
    rows = conn.execute(
        """
        DELETE FROM project_code_intel_snapshots
        WHERE collection = %s AND repo = %s
          AND id NOT IN (
              SELECT id FROM project_code_intel_snapshots
              WHERE collection = %s AND repo = %s
              ORDER BY created_at DESC
              LIMIT %s
          )
        RETURNING id
        """,
        [collection, repo, collection, repo, keep],
    ).fetchall()
    return len(rows)
