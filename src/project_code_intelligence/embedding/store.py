"""Database-backed embedding resume and compatibility logic."""

from __future__ import annotations

import json
import time
from operator import itemgetter
from typing import TYPE_CHECKING, TypeVar, cast

from project_code_intelligence import config, db
from project_code_intelligence import runtime as runtime_state
from project_code_intelligence.embedding.core import embed_items_with_retry
from project_code_intelligence.embedding.endpoint import retry_sleep_seconds
from project_code_intelligence.embedding.types import EmbeddingRow, EmbeddingRunConfig, SkippedEmbeddingRow
from project_code_intelligence.progress import progress_event

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

T = TypeVar("T")
EMBEDDING_CONTRACT_METADATA_KEY = "embedding_contract"


def database_retries() -> int:
    return config.env_int("PCI_DB_RETRIES", 3, minimum=0)


def vector_literal_dimensions(value: str) -> int:
    text = value.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError("embedding vector literal must use pgvector list syntax")
    body = text[1:-1].strip()
    if not body:
        raise ValueError("embedding vector literal must not be empty")
    return body.count(",") + 1


def embedding_metadata(run_config: EmbeddingRunConfig, embedding: str) -> dict[str, object]:
    backend_name = "endpoint" if run_config.backend.endpoint else "llama-cli"
    return {
        "embedding_backend": backend_name,
        "embedding_model": run_config.backend.endpoint_model,
        "embedding_dimensions": vector_literal_dimensions(embedding),
    }


def embedding_contract(run_config: EmbeddingRunConfig, embedding: str) -> dict[str, object]:
    backend_name = "endpoint" if run_config.backend.endpoint else "llama-cli"
    return {
        "version": 1,
        "backend": backend_name,
        "model": run_config.backend.endpoint_model,
        "dimensions": vector_literal_dimensions(embedding),
    }


def object_int_value(value: object, key: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"{key} must be an integer")


def row_int_value(row: db.DbRow, key: str) -> int:
    return object_int_value(row[key], key)


def embedding_contract_from_metadata(metadata: object) -> dict[str, object] | None:
    if not isinstance(metadata, dict):
        return None
    metadata_obj = cast("dict[object, object]", metadata)
    contract = metadata_obj.get(EMBEDDING_CONTRACT_METADATA_KEY)
    if contract is None:
        return None
    if isinstance(contract, dict):
        contract_obj = cast("dict[object, object]", contract)
        return {str(key): value for key, value in contract_obj.items()}
    raise ValueError("snapshot embedding contract metadata is malformed")


class EmbeddingContractMismatchError(ValueError):
    """Raised when the DB contains embeddings from a different model than the current server."""


def require_compatible_embedding_contract(existing: Mapping[str, object], current: Mapping[str, object]) -> None:
    existing_model = existing.get("model")
    current_model = current.get("model")
    if not isinstance(existing_model, str) or not isinstance(current_model, str):
        raise TypeError("snapshot embedding contract is missing a model")
    existing_dimensions = object_int_value(existing.get("dimensions"), "dimensions")
    current_dimensions = object_int_value(current.get("dimensions"), "dimensions")
    if existing_model != current_model or existing_dimensions != current_dimensions:
        raise EmbeddingContractMismatchError(
            "existing embeddings use model "
            f"{existing_model} with {existing_dimensions} dimensions; "
            f"current embedding model is {current_model} with {current_dimensions} dimensions"
        )


def run_database_operation(description: str, operation: Callable[[], T]) -> T:
    attempts = database_retries() + 1
    return run_database_operation_attempt(description, operation, attempt=1, attempts=attempts)


def run_database_operation_attempt(
    description: str,
    operation: Callable[[], T],
    *,
    attempt: int,
    attempts: int,
) -> T:
    try:
        return operation()
    except (db.DatabaseConnectionError, db.OperationalError) as exc:
        if attempt >= attempts:
            raise
        runtime_state.active_metrics.add("db_retries", 1)
        progress_event(
            "code_intel_db_retry",
            operation=description,
            attempt=attempt,
            attempts=attempts,
            reason=str(exc)[:240],
        )
        time.sleep(retry_sleep_seconds(attempt))
        return run_database_operation_attempt(description, operation, attempt=attempt + 1, attempts=attempts)


def embedding_snapshot_counts(snapshot_id: int, record_types: set[str]) -> db.DbRow:
    def operation() -> db.DbRow:
        with db.connect() as conn:
            row = conn.execute(
                """
                SELECT
                  count(*) FILTER (WHERE record_type = ANY(%s)) AS selected_total,
                  count(*) FILTER (WHERE record_type = ANY(%s) AND embedding IS NOT NULL) AS selected_embedded,
                  count(*) FILTER (
                    WHERE record_type = ANY(%s)
                      AND embedding IS NULL
                      AND metadata->>'embedding_skipped' IS DISTINCT FROM 'true'
                  ) AS selected_pending,
                  count(*) FILTER (
                    WHERE record_type = ANY(%s)
                      AND metadata->>'embedding_skipped' = 'true'
                  ) AS selected_skipped
                FROM project_code_intel_records
                WHERE snapshot_id = %s
                """,
                [sorted(record_types), sorted(record_types), sorted(record_types), sorted(record_types), snapshot_id],
            ).fetchone()
            return db.require_row(row, "embedding snapshot counts")

    return run_database_operation("embedding snapshot counts", operation)


def fetch_embedding_batch(snapshot_id: int, record_types: set[str], batch_size: int) -> list[EmbeddingRow]:
    def operation() -> list[EmbeddingRow]:
        with db.connect() as conn:
            return cast(
                "list[EmbeddingRow]",
                conn.execute(
                    """
                    SELECT id, source_path, record_id, embedding_text
                    FROM project_code_intel_records
                    WHERE snapshot_id = %s
                      AND record_type = ANY(%s)
                      AND embedding IS NULL
                      AND metadata->>'embedding_skipped' IS DISTINCT FROM 'true'
                    ORDER BY id
                    LIMIT %s
                    """,
                    [snapshot_id, sorted(record_types), batch_size],
                ).fetchall(),
            )

    return run_database_operation("fetch embedding batch", operation)


def snapshot_embedding_contract(conn: db.DbConnection, snapshot_id: int) -> dict[str, object] | None:
    row = conn.execute(
        """
        SELECT metadata
        FROM project_code_intel_snapshots
        WHERE id = %s
        """,
        [snapshot_id],
    ).fetchone()
    metadata = db.require_row(row, "snapshot embedding contract")["metadata"]
    return embedding_contract_from_metadata(metadata)


def existing_embedding_contract_from_rows(conn: db.DbConnection, snapshot_id: int) -> dict[str, object] | None:
    summary = db.require_row(
        conn.execute(
            """
            SELECT count(*) AS embedded_total,
                   count(*) FILTER (
                     WHERE metadata ? 'embedding_model'
                       AND metadata ? 'embedding_dimensions'
                   ) AS embedded_with_metadata
            FROM project_code_intel_records
            WHERE snapshot_id = %s AND embedding IS NOT NULL
            """,
            [snapshot_id],
        ).fetchone(),
        "existing embedding metadata summary",
    )
    embedded_total = row_int_value(summary, "embedded_total")
    embedded_with_metadata = row_int_value(summary, "embedded_with_metadata")
    if embedded_total == 0:
        return None
    if embedded_with_metadata != embedded_total:
        raise ValueError(
            "existing embeddings do not have a snapshot embedding contract and cannot be verified safely. "
            "Resume with the original version of the tool, or reset/rebuild embeddings for this snapshot."
        )
    rows = conn.execute(
        """
        SELECT DISTINCT
               metadata->>'embedding_model' AS model,
               metadata->>'embedding_dimensions' AS dimensions
        FROM project_code_intel_records
        WHERE snapshot_id = %s
          AND embedding IS NOT NULL
          AND metadata ? 'embedding_model'
          AND metadata ? 'embedding_dimensions'
        """,
        [snapshot_id],
    ).fetchall()
    contracts = [contract for row in rows if (contract := row_embedding_contract(row)) is not None]
    if len(contracts) != 1:
        raise ValueError("existing embeddings contain multiple embedding models or dimensions")
    return contracts[0]


def row_embedding_contract(row: db.DbRow) -> dict[str, object] | None:
    if row["model"] is None or row["dimensions"] is None:
        return None
    return {"model": str(row["model"]), "dimensions": row_int_value(row, "dimensions")}


def set_snapshot_embedding_contract(
    conn: db.DbConnection,
    snapshot_id: int,
    contract: dict[str, object],
) -> None:
    _ = conn.execute(
        """
        UPDATE project_code_intel_snapshots
        SET metadata = coalesce(metadata, '{}'::jsonb) || %s::jsonb
        WHERE id = %s
        """,
        [
            json.dumps({EMBEDDING_CONTRACT_METADATA_KEY: contract}, sort_keys=True, separators=(",", ":")),
            snapshot_id,
        ],
    )


def validate_embedding_compatibility(
    conn: db.DbConnection,
    snapshot_id: int,
    *,
    run_config: EmbeddingRunConfig,
    embeddings: list[str],
) -> None:
    if not embeddings:
        return
    new_dimensions = {vector_literal_dimensions(embedding) for embedding in embeddings}
    if len(new_dimensions) != 1:
        raise ValueError("embedding endpoint returned mixed vector dimensions in one batch")
    current_contract = embedding_contract(run_config, embeddings[0])
    snapshot_contract = snapshot_embedding_contract(conn, snapshot_id)
    if snapshot_contract is not None:
        require_compatible_embedding_contract(snapshot_contract, current_contract)
    else:
        row_contract = existing_embedding_contract_from_rows(conn, snapshot_id)
        if row_contract is not None:
            require_compatible_embedding_contract(row_contract, current_contract)
        set_snapshot_embedding_contract(conn, snapshot_id, current_contract)

    dimension = next(iter(new_dimensions))
    rows = conn.execute(
        """
        SELECT DISTINCT vector_dims(embedding) AS dimensions
        FROM project_code_intel_records
        WHERE snapshot_id = %s AND embedding IS NOT NULL
        """,
        [snapshot_id],
    ).fetchall()
    existing_dimensions = {row_int_value(row, "dimensions") for row in rows if row["dimensions"] is not None}
    if existing_dimensions and existing_dimensions != {dimension}:
        raise ValueError(
            "existing embeddings use dimensions "
            + ", ".join(str(item) for item in sorted(existing_dimensions))
            + f"; current embedding endpoint returned {dimension}"
        )

    model_rows = conn.execute(
        """
        SELECT DISTINCT metadata->>'embedding_model' AS model
        FROM project_code_intel_records
        WHERE snapshot_id = %s
          AND embedding IS NOT NULL
          AND metadata ? 'embedding_model'
        """,
        [snapshot_id],
    ).fetchall()
    existing_models = {str(row["model"]) for row in model_rows if row["model"]}
    current_model = run_config.backend.endpoint_model
    if existing_models and current_model not in existing_models:
        raise ValueError(
            "existing embeddings use model "
            + ", ".join(sorted(existing_models))
            + f"; current embedding model is {current_model}"
        )


def update_embedding_batch(
    snapshot_id: int,
    *,
    embedded: list[tuple[EmbeddingRow, str]],
    skipped: list[SkippedEmbeddingRow],
    run_config: EmbeddingRunConfig,
) -> None:
    if not embedded and not skipped:
        return

    def operation() -> None:
        started = time.monotonic()
        try:
            with db.connect(readonly=False) as conn:
                validate_embedding_compatibility(
                    conn,
                    snapshot_id,
                    run_config=run_config,
                    embeddings=[embedding for _row, embedding in embedded],
                )
                for row, embedding in embedded:
                    _ = conn.execute(
                        """
                        UPDATE project_code_intel_records
                        SET embedding = %s::vector,
                            metadata = coalesce(metadata, '{}'::jsonb) || %s::jsonb
                        WHERE id = %s AND snapshot_id = %s
                        """,
                        [
                            embedding,
                            json.dumps(
                                embedding_metadata(run_config, embedding),
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            row["id"],
                            snapshot_id,
                        ],
                    )
                for item in skipped:
                    metadata = {
                        "embedding_skipped": True,
                        "embedding_skip_reason": item.reason[:500],
                        "embedding_skip_max_chars": item.max_chars,
                    }
                    _ = conn.execute(
                        """
                        UPDATE project_code_intel_records
                        SET metadata = coalesce(metadata, '{}'::jsonb) || %s::jsonb
                        WHERE id = %s AND snapshot_id = %s
                        """,
                        [json.dumps(metadata, sort_keys=True, separators=(",", ":")), item.row["id"], snapshot_id],
                    )
                conn.commit()
        finally:
            runtime_state.active_metrics.add("embedding_db_update_seconds", time.monotonic() - started)

    run_database_operation("update embedding batch", operation)


def embed_db_records(
    snapshot_ids: list[int],
    *,
    record_types: set[str],
    batch_size: int,
    run_config: EmbeddingRunConfig,
) -> int:
    if not record_types:
        return 0

    def embed_batch(
        batch: list[EmbeddingRow], batch_max_chars: int
    ) -> tuple[list[tuple[EmbeddingRow, str]], list[SkippedEmbeddingRow]]:
        skipped_rows: list[SkippedEmbeddingRow] = []

        def mark_skipped(row: EmbeddingRow, reason: BaseException, skipped_max_chars: int) -> None:
            skipped_rows.append(SkippedEmbeddingRow(row=row, reason=str(reason), max_chars=skipped_max_chars))
            runtime_state.active_metrics.add("embedding_skipped_records", 1)
            progress_event(
                "code_intel_embedding_skipped",
                record_id=row["id"],
                source_path=row.get("source_path"),
                source_record_id=row.get("record_id"),
                reason=str(reason)[:240],
            )

        embedded, skipped_count = embed_items_with_retry(
            batch,
            run_config=EmbeddingRunConfig(backend=run_config.backend, max_chars=batch_max_chars),
            text_for=itemgetter("embedding_text"),
            skip_item=mark_skipped,
            retry_event_values=lambda row: {
                "record_id": row["id"],
                "source_path": row.get("source_path"),
                "source_record_id": row.get("record_id"),
            },
        )
        if skipped_count != len(skipped_rows):
            raise RuntimeError("embedding skip accounting mismatch")
        return embedded, skipped_rows

    embedded = 0
    skipped = 0
    batch_size = max(1, batch_size)
    for snapshot_id in snapshot_ids:
        counts = embedding_snapshot_counts(snapshot_id, record_types)
        selected_total = row_int_value(counts, "selected_total")
        already_embedded = row_int_value(counts, "selected_embedded")
        pending_total = row_int_value(counts, "selected_pending")
        already_skipped = row_int_value(counts, "selected_skipped")
        runtime_state.active_metrics.add_phase_total(pending_total)
        runtime_state.active_metrics.add("embedding_records_selected", pending_total)
        progress_event(
            "code_intel_embedding_selected",
            snapshot_id=snapshot_id,
            records=pending_total,
            total_records=selected_total,
            already_embedded=already_embedded,
            already_skipped=already_skipped,
        )
        processed = 0
        while True:
            batch = fetch_embedding_batch(snapshot_id, record_types, batch_size)
            if not batch:
                break
            batch_embedded, batch_skipped = embed_batch(batch, run_config.max_chars)
            update_embedding_batch(
                snapshot_id,
                embedded=batch_embedded,
                skipped=batch_skipped,
                run_config=run_config,
            )
            batch_embedded_count = len(batch_embedded)
            batch_skipped_count = len(batch_skipped)
            embedded += batch_embedded_count
            skipped += batch_skipped_count
            processed += len(batch)
            runtime_state.active_metrics.add_phase_done(len(batch))
            runtime_state.active_metrics.add("embedded_records", batch_embedded_count)
            progress_event(
                "code_intel_embedded",
                snapshot_id=snapshot_id,
                records=processed,
                total_records=pending_total,
                embedded_total=embedded,
                skipped_total=skipped,
            )
    return embedded
