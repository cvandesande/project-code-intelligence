"""Pre-insert embedding pipeline helpers."""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING, cast

from project_code_intelligence import config
from project_code_intelligence import runtime as runtime_state
from project_code_intelligence.embedding.core import embed_items_with_retry
from project_code_intelligence.embedding.store import embedding_metadata
from project_code_intelligence.runtime import PreEmbeddingResult, PreEmbeddingState, progress_event
from project_code_intelligence.storage import RecordInsertContext, insert_records

if TYPE_CHECKING:
    from collections.abc import Callable

    from project_code_intelligence.embedding.types import EmbeddingRunConfig
    from project_code_intelligence.models import IntelRecord, JsonObject


def code_preembedding_enabled() -> bool:
    return config.env_bool("PROJECT_CODE_INTELLIGENCE_PREEMBED", default=True)


def preembedding_ahead_batches() -> int:
    return config.env_int("PROJECT_CODE_INTELLIGENCE_PREEMBED_AHEAD_BATCHES", 16, minimum=1)


def mark_record_embedding_skipped(record: IntelRecord, reason: BaseException, max_chars: int) -> None:
    record.metadata = {
        **record.metadata,
        "embedding_skipped": True,
        "embedding_skip_reason": str(reason)[:500],
        "embedding_skip_max_chars": max_chars,
    }
    runtime_state.active_metrics.add("embedding_skipped_records", 1)
    progress_event(
        "code_intel_preembedding_skipped",
        record_id=record.record_id,
        reason=str(reason)[:240],
    )


def mark_record_embedded(record: IntelRecord, embedding: str, run_config: EmbeddingRunConfig) -> None:
    record.embedding = embedding
    record.metadata = {
        **record.metadata,
        **cast("JsonObject", embedding_metadata(run_config, embedding)),
    }


def embed_record_batch(
    batch: list[IntelRecord],
    *,
    run_config: EmbeddingRunConfig,
) -> tuple[int, int]:
    embedded, skipped = embed_items_with_retry(
        batch,
        run_config=run_config,
        text_for=lambda record: record.embedding_text,
        skip_item=mark_record_embedding_skipped,
        retry_event_values=lambda record: {"record_id": record.record_id},
    )
    for record, embedding in embedded:
        mark_record_embedded(record, embedding, run_config)
    return len(embedded), skipped


def preembedding_worker(
    state: PreEmbeddingState,
    *,
    run_config: EmbeddingRunConfig,
) -> None:
    for batch in state.batches:
        if state.cancel_event.is_set():
            return
        try:
            embedded, skipped = embed_record_batch(
                batch,
                run_config=run_config,
            )
            result = PreEmbeddingResult(batch=batch, embedded=embedded, skipped=skipped)
        except Exception as exc:  # noqa: BLE001 - post-insert embedding remains authoritative.
            result = PreEmbeddingResult(batch=batch, error=exc)
        while not state.cancel_event.is_set():
            try:
                state.results.put(result, timeout=0.5)
                break
            except queue.Full:
                continue
        if state.cancel_event.is_set():
            return
        if result.error is not None:
            return


def start_record_preembedding(
    records: list[IntelRecord],
    *,
    record_types: set[str],
    batch_size: int,
    run_config: EmbeddingRunConfig,
) -> PreEmbeddingState | None:
    selected = [
        record
        for record in records
        if record.record_type in record_types
        and record.embedding is None
        and record.metadata.get("embedding_skipped") is not True
    ]
    if not selected:
        return None
    batch_size = max(1, batch_size)
    batches = [selected[offset : offset + batch_size] for offset in range(0, len(selected), batch_size)]
    state = PreEmbeddingState(
        batches=batches,
        selected_ids={id(record) for record in selected},
        results=queue.Queue(maxsize=preembedding_ahead_batches()),
        total_records=len(selected),
    )
    thread = threading.Thread(
        target=preembedding_worker,
        kwargs={
            "state": state,
            "run_config": run_config,
        },
        daemon=True,
    )
    state.thread = thread
    thread.start()
    progress_event(
        "code_intel_preembedding_selected",
        records=len(selected),
        batches=len(batches),
        batch_size=batch_size,
        ahead_batches=state.results.maxsize,
    )
    runtime_state.active_metrics.add("preembedding_records_selected", len(selected))
    return state


def next_preembedding_result(state: PreEmbeddingState, *, block: bool) -> PreEmbeddingResult | None:
    if state.consumed_batches >= len(state.batches):
        return None
    if not block:
        try:
            return state.results.get_nowait()
        except queue.Empty:
            return None
    while True:
        if not state.results.empty():
            return state.results.get_nowait()
        if state.thread is not None and not state.thread.is_alive():
            return None
        time.sleep(0.5)


def consume_preembedding_results(
    insert_context: RecordInsertContext,
    state: PreEmbeddingState,
    *,
    block: bool,
) -> int:
    inserted = 0
    while True:
        result = next_preembedding_result(state, block=block)
        if result is None:
            return inserted
        state.consumed_batches += 1
        if result.error is not None:
            state.cancel_event.set()
            remaining_batches = [result.batch, *state.batches[state.consumed_batches :]]
            progress_event(
                "code_intel_preembedding_disabled",
                snapshot_id=insert_context.snapshot_id,
                error=str(result.error)[:240],
                remaining_batches=len(remaining_batches),
            )
            for batch in remaining_batches:
                inserted += insert_records(insert_context, batch)
            state.consumed_batches = len(state.batches)
            return inserted
        state.processed_records += len(result.batch)
        state.embedded += result.embedded
        state.skipped += result.skipped
        runtime_state.active_metrics.add("preembedded_records", result.embedded)
        runtime_state.active_metrics.add("embedded_records", result.embedded)
        inserted += insert_records(insert_context, result.batch)
        progress_event(
            "code_intel_preembedded",
            snapshot_id=insert_context.snapshot_id,
            records=state.processed_records,
            total_records=state.total_records,
            embedded_total=state.embedded,
            skipped_total=state.skipped,
        )
        if not block:
            continue


def insert_records_with_preembedding(
    insert_context: RecordInsertContext,
    records: list[IntelRecord],
    state: PreEmbeddingState,
    *,
    progress_fn: Callable[[int], None] | None = None,
) -> tuple[int, int, int]:
    inserted = 0
    non_embedding_records = [record for record in records if id(record) not in state.selected_ids]
    for offset in range(0, len(non_embedding_records), 1000):
        inserted += insert_records(
            insert_context, non_embedding_records[offset : offset + 1000], progress_fn=progress_fn
        )
        inserted += consume_preembedding_results(insert_context, state, block=False)
    inserted += consume_preembedding_results(insert_context, state, block=False)
    if state.consumed_batches < len(state.batches):
        state.cancel_event.set()
        remaining_batches = state.batches[state.consumed_batches :]
        remaining_records = sum(len(batch) for batch in remaining_batches)
        progress_event(
            "code_intel_preembedding_deferred",
            snapshot_id=insert_context.snapshot_id,
            records=remaining_records,
            batches=len(remaining_batches),
        )
        for batch in remaining_batches:
            inserted += insert_records(insert_context, batch, progress_fn=progress_fn)
        state.consumed_batches = len(state.batches)
    return inserted, state.embedded, state.skipped


def abandon_preembedding(state: PreEmbeddingState | None) -> None:
    if state is not None:
        state.cancel_event.set()
