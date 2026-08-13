"""Database persistence for code-intelligence snapshots and records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

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


def _strip_nul_str(value: str) -> str:
    """Remove U+0000 from a string."""
    return value.replace("\x00", "") if "\x00" in value else value


def _strip_nul(value: object) -> object:
    """Remove U+0000 from strings recursively; pass non-strings through.

    PostgreSQL `text` and `jsonb` cannot store NUL bytes. They arrive when a
    file with mixed text and binary content slips past binary detection (e.g.
    a compiled artifact with an ASCII preamble). Stripping at the storage
    boundary keeps the insert from failing without losing the readable parts.
    """
    if isinstance(value, str):
        return _strip_nul_str(value)
    if isinstance(value, dict):
        items = cast("dict[object, object]", value).items()
        return {k: _strip_nul(v) for k, v in items}
    if isinstance(value, list):
        items_list = cast("list[object]", value)
        return [_strip_nul(item) for item in items_list]
    return value


@dataclass(frozen=True)
class RecordInsertContext:
    conn: db.DbConnection
    snapshot: Snapshot
    snapshot_id: int
    file_ids: dict[str, int]
    file_hashes: dict[str, str | None]


@dataclass(frozen=True)
class _SymbolDefinitionTarget:
    record_id: str
    source_path: str
    symbol_kind: str | None = None


@dataclass(frozen=True)
class _SymbolDefinitionChoices:
    by_source_path: dict[str, _SymbolDefinitionTarget]
    by_source_dir: dict[str, _SymbolDefinitionTarget]
    fallback: _SymbolDefinitionTarget

    def choose(self, source_path: str | None) -> _SymbolDefinitionTarget:
        if source_path is not None:
            same_file = self.by_source_path.get(source_path)
            if same_file is not None:
                return same_file
            same_dir = self.by_source_dir.get(_source_dir(source_path))
            if same_dir is not None:
                return same_dir
        return self.fallback


def _source_dir(source_path: str) -> str:
    return source_path.rsplit("/", maxsplit=1)[0] if "/" in source_path else source_path


def _symbol_kind_resolve_rank(symbol_kind: str | None) -> int:
    if symbol_kind in {"function", "method", "constant", "class", "enum", "shell_function"}:
        return 0
    if symbol_kind in {"interface", "type"}:
        return 2
    return 1


def _symbol_definition_choices(definitions: list[_SymbolDefinitionTarget]) -> _SymbolDefinitionChoices:
    ordered = sorted(
        definitions,
        key=lambda item: (_symbol_kind_resolve_rank(item.symbol_kind), item.source_path, item.record_id),
    )
    by_source_path: dict[str, _SymbolDefinitionTarget] = {}
    by_source_dir: dict[str, _SymbolDefinitionTarget] = {}
    for definition in sorted(
        definitions,
        key=lambda item: (item.source_path, _symbol_kind_resolve_rank(item.symbol_kind), item.record_id),
    ):
        _ = by_source_path.setdefault(definition.source_path, definition)
        _ = by_source_dir.setdefault(_source_dir(definition.source_path), definition)
    return _SymbolDefinitionChoices(by_source_path=by_source_path, by_source_dir=by_source_dir, fallback=ordered[0])


def edge_target_resolvable(edge: IntelEdge) -> bool:
    return edge.metadata.get("target_resolvable") is not False


def pre_resolvable_edge_count(edges: list[IntelEdge]) -> int:
    return sum(
        1 for edge in edges if edge.target_record_id is None and edge.target_symbol and edge_target_resolvable(edge)
    )


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
            db.compact_json(snapshot.metadata),
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
            "metadata": _strip_nul(item.metadata),
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
        [db.compact_json(payload)],
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
        input_order = len(payload)
        title = _strip_nul_str(record.title)
        summary = _strip_nul_str(record.summary)
        embedding_text = _strip_nul_str(record.embedding_text)
        display_content = _strip_nul_str(record.display_content)
        metadata = _strip_nul(record.metadata)
        embedding_hash = sha256_text(embedding_text)
        display_hash = sha256_text(display_content)
        payload.append({
            "input_order": input_order,
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
            "title": title,
            "summary": summary,
            "embedding_text": embedding_text,
            "display_content": display_content,
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
            "metadata": metadata,
            "embedding": record.embedding,
        })
    for i in range(0, len(payload), _INSERT_BATCH_SIZE):
        batch = payload[i : i + _INSERT_BATCH_SIZE]
        _ = context.conn.execute(
            """
            WITH input_rows AS (
                SELECT *
                FROM jsonb_to_recordset(%s::jsonb) AS r(
                    input_order bigint,
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
            ),
            deduped AS (
                SELECT DISTINCT ON (snapshot_id, record_type, record_id, embedding_text_hash) *
                FROM input_rows
                ORDER BY snapshot_id, record_type, record_id, embedding_text_hash, input_order DESC
            )
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
            FROM deduped
            ON CONFLICT (snapshot_id, record_type, record_id, embedding_text_hash)
            DO UPDATE SET file_id = EXCLUDED.file_id,
                          file_sha256 = EXCLUDED.file_sha256,
                          summary = EXCLUDED.summary,
                          display_content = EXCLUDED.display_content,
                          display_hash = EXCLUDED.display_hash,
                          metadata = EXCLUDED.metadata,
                          embedding = coalesce(EXCLUDED.embedding, project_code_intel_records.embedding)
            """,
            [db.compact_json(batch)],
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
            "metadata": _strip_nul(edge.metadata),
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
            [db.compact_json(batch)],
        )
        if progress_fn is not None:
            progress_fn(len(batch))
    return len(edges)


def _load_symbol_definition_index(
    conn: db.DbConnection,
    snapshot_id: int,
    target_symbols: set[str],
) -> dict[str, _SymbolDefinitionChoices]:
    if not target_symbols:
        return {}
    rows = conn.execute(
        """
        SELECT symbol, record_id, source_path, symbol_kind
        FROM project_code_intel_records
        WHERE snapshot_id = %s
          AND record_type = 'symbol_definition'
          AND symbol IS NOT NULL
          AND symbol = ANY(%s::text[])
        ORDER BY symbol, source_path, record_id
        """,
        [snapshot_id, sorted(target_symbols)],
    ).fetchall()
    definitions_by_symbol: dict[str, list[_SymbolDefinitionTarget]] = {}
    for row in rows:
        symbol = str(row["symbol"])
        definitions_by_symbol.setdefault(symbol, []).append(
            _SymbolDefinitionTarget(
                record_id=str(row["record_id"]),
                source_path=str(row["source_path"]),
                symbol_kind=str(row["symbol_kind"]) if row.get("symbol_kind") is not None else None,
            )
        )
    return {
        symbol: _symbol_definition_choices(definitions)
        for symbol, definitions in definitions_by_symbol.items()
        if definitions
    }


_EDGE_TARGET_PRE_RESOLVE_BATCH_SIZE = 5_000


def pre_resolve_edge_targets(
    conn: db.DbConnection,
    snapshot_id: int,
    edges: list[IntelEdge],
    *,
    batch_size: int = _EDGE_TARGET_PRE_RESOLVE_BATCH_SIZE,
    progress_fn: Callable[[int], None] | None = None,
) -> int:
    """Resolve newly generated edge targets in memory before inserting edge rows."""
    unresolved_edges = [
        edge for edge in edges if edge.target_record_id is None and edge.target_symbol and edge_target_resolvable(edge)
    ]
    if not unresolved_edges:
        return 0
    batch_size = max(1, batch_size)
    symbols: dict[str, _SymbolDefinitionChoices | None] = {}
    resolved = 0
    for offset in range(0, len(unresolved_edges), batch_size):
        batch = unresolved_edges[offset : offset + batch_size]
        missing_symbols = {str(edge.target_symbol) for edge in batch if str(edge.target_symbol) not in symbols}
        loaded_symbols = _load_symbol_definition_index(conn, snapshot_id, missing_symbols)
        for symbol in missing_symbols:
            symbols[symbol] = loaded_symbols.get(symbol)
        for edge in batch:
            choices = symbols.get(str(edge.target_symbol))
            if choices is None:
                continue
            target = choices.choose(edge.source_path)
            edge.target_record_id = target.record_id
            edge.target_path = target.source_path
            resolved += 1
        if progress_fn is not None:
            progress_fn(len(batch))
    return resolved


def count_unresolved_edge_targets(conn: db.DbConnection, snapshot_id: int) -> int:
    row = conn.execute(
        """
        SELECT count(*) AS count
        FROM project_code_intel_edges
        WHERE snapshot_id = %s
          AND target_record_id IS NULL
          AND target_symbol IS NOT NULL
          AND COALESCE((metadata->>'target_resolvable')::boolean, true)
        """,
        [snapshot_id],
    ).fetchone()
    return row_int(db.require_row(row, "count unresolved edge targets"), "count")


_EDGE_TARGET_RESOLVE_BATCH_SIZE = 5_000


def resolve_edge_targets(
    conn: db.DbConnection,
    snapshot_id: int,
    *,
    batch_size: int = _EDGE_TARGET_RESOLVE_BATCH_SIZE,
    progress_fn: Callable[[int], None] | None = None,
) -> int:
    """Fill target_record_id / target_path on edges that have only target_symbol.

    call_candidate edges are created with target_symbol (the callee name) but no
    target_record_id, because at parse time we only know the callee by name. This
    pass joins against symbol_definition records in the same snapshot and fills
    in the stable record_id string. When multiple records share the same symbol
    name, prefer same-file matches, then same-directory matches, then anything
    else (lexicographic tiebreak) — collisions across unrelated subtrees were
    otherwise resolving by alphabetical accident. The resolver walks unresolved
    edges in bounded batches so large snapshots do not spend minutes in one
    opaque UPDATE statement.
    """
    resolved = 0
    last_target_symbol: str | None = None
    last_edge_id = 0
    while True:
        row = conn.execute(
            """
            WITH candidate_edges AS MATERIALIZED (
                SELECT id, source_path, target_symbol
                FROM project_code_intel_edges
                WHERE snapshot_id = %s
                  AND target_record_id IS NULL
                  AND target_symbol IS NOT NULL
                  AND COALESCE((metadata->>'target_resolvable')::boolean, true)
                  AND (
                      %s::text IS NULL
                      OR target_symbol > %s::text
                      OR (target_symbol = %s::text AND id > %s)
                  )
                ORDER BY target_symbol, id
                LIMIT %s
            ),
            candidate_cursor AS (
                SELECT
                    count(*) AS candidate_count,
                    (array_agg(target_symbol ORDER BY target_symbol DESC, id DESC))[1] AS last_target_symbol,
                    (array_agg(id ORDER BY target_symbol DESC, id DESC))[1] AS last_edge_id
                FROM candidate_edges
            ),
            resolved_pairs AS MATERIALIZED (
                SELECT DISTINCT ON (c.source_path, c.target_symbol)
                       c.source_path AS edge_source_path,
                       c.target_symbol,
                       r.record_id,
                       r.source_path AS target_path
                FROM candidate_edges c
                CROSS JOIN LATERAL (
                    SELECT r.record_id, r.source_path
                    FROM project_code_intel_records r
                    WHERE r.snapshot_id = %s
                      AND r.symbol = c.target_symbol
                      AND r.record_type = 'symbol_definition'
                    ORDER BY
                        CASE
                            WHEN r.source_path = c.source_path THEN 0
                            WHEN regexp_replace(r.source_path, '/[^/]+$', '')
                               = regexp_replace(c.source_path, '/[^/]+$', '') THEN 1
                            ELSE 2
                        END,
                        CASE
                            WHEN r.symbol_kind IN ('function', 'method', 'constant', 'class', 'enum', 'shell_function')
                                THEN 0
                            WHEN r.symbol_kind IN ('interface', 'type') THEN 2
                            ELSE 1
                        END,
                        r.source_path,
                        r.record_id
                    LIMIT 1
                ) r
                ORDER BY c.source_path, c.target_symbol
            ),
            updated AS (
                UPDATE project_code_intel_edges e
                SET target_record_id = p.record_id,
                    target_path = p.target_path
                FROM candidate_edges c
                JOIN resolved_pairs p
                  ON p.target_symbol = c.target_symbol
                 AND p.edge_source_path IS NOT DISTINCT FROM c.source_path
                WHERE e.id = c.id
                RETURNING e.id
            )
            SELECT
                candidate_count,
                last_target_symbol,
                last_edge_id,
                (SELECT count(*) FROM updated) AS updated_count
            FROM candidate_cursor
            """,
            [
                snapshot_id,
                last_target_symbol,
                last_target_symbol,
                last_target_symbol,
                last_edge_id,
                max(1, batch_size),
                snapshot_id,
            ],
        ).fetchone()
        row = db.require_row(row, "resolve edge targets")
        candidate_count = row_int(row, "candidate_count")
        if candidate_count == 0:
            return resolved
        updated_count = row_int(row, "updated_count")
        resolved += updated_count
        if progress_fn is not None:
            progress_fn(candidate_count)
        last_symbol = row["last_target_symbol"]
        if not isinstance(last_symbol, str):
            return resolved
        last_target_symbol = last_symbol
        last_edge_id = row_int(row, "last_edge_id")


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
            db.compact_json(parser_failure_metadata(failure)),
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
    """Delete old snapshots for (collection, repo), keeping the newest ``keep``.

    Branch-aware: the newest snapshot of each distinct branch is never deleted
    (null branch counts as one shared group), even if that pushes the total
    kept above ``keep``. The keep-N cut applies only to the remainder -- the
    rows that are not each branch's newest.
    """
    rows = conn.execute(
        """
        WITH ranked AS (
            SELECT id, created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(branch, '')
                    ORDER BY created_at DESC, id DESC
                ) AS branch_rank
            FROM project_code_intel_snapshots
            WHERE collection = %s AND repo = %s
        ),
        remainder AS (
            SELECT id,
                ROW_NUMBER() OVER (ORDER BY created_at DESC, id DESC) AS remainder_rank
            FROM ranked
            WHERE branch_rank > 1
        )
        DELETE FROM project_code_intel_snapshots
        WHERE id IN (SELECT id FROM remainder WHERE remainder_rank > %s)
        RETURNING id
        """,
        [collection, repo, keep],
    ).fetchall()
    return len(rows)


def prune_dead_branch_snapshots(conn: db.DbConnection, collection: str, repo: str, live_branches: set[str]) -> int:
    """Delete this repo's snapshots stamped with a branch no longer in ``live_branches``.

    Never touches null-branch (legacy) snapshots, and never deletes the newest snapshot
    overall for the repo even if its branch is not in ``live_branches`` (e.g. a git
    for-each-ref race, or a detached-HEAD indexing run) -- losing the newest snapshot would
    make the repo look unindexed rather than just missing one stale branch.
    """
    rows = conn.execute(
        """
        DELETE FROM project_code_intel_snapshots
        WHERE collection = %s AND repo = %s
          AND branch IS NOT NULL
          AND branch <> ALL(%s)
          AND id <> (
              SELECT id FROM project_code_intel_snapshots
              WHERE collection = %s AND repo = %s
              ORDER BY created_at DESC, id DESC
              LIMIT 1
          )
        RETURNING id
        """,
        [collection, repo, list(live_branches), collection, repo],
    ).fetchall()
    return len(rows)
