"""Embedding endpoint, retry, and pre-embedding helpers."""

from __future__ import annotations

import ipaddress
import json
import queue
import threading
import time
import urllib.error
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, TypedDict, TypeVar, cast
from urllib.parse import urlsplit

from project_code_intelligence import config, db, http_client
from project_code_intelligence import runtime as runtime_state
from project_code_intelligence.embedding_utils import llama_batch_embeddings
from project_code_intelligence.runtime import PreEmbeddingResult, PreEmbeddingState, progress_event
from project_code_intelligence.storage import RecordInsertContext, insert_records

if TYPE_CHECKING:
    from collections.abc import Callable

    from project_code_intelligence.models import IntelRecord, JsonObject

T = TypeVar("T")


class EmbeddingRow(TypedDict):
    id: int
    source_path: str
    record_id: str
    embedding_text: str


class EmbeddingEndpointUnavailableError(RuntimeError):
    """Raised when the configured embedding HTTP endpoint cannot be reached."""

    def __init__(self, message: str, *, recoverable_batch: bool = False) -> None:
        super().__init__(message)
        self.recoverable_batch: bool = recoverable_batch


EmbeddingEndpointUnavailable = EmbeddingEndpointUnavailableError


@dataclass(frozen=True)
class EmbeddingBackend:
    endpoint: str | None
    endpoint_model: str
    use_llama_cli: bool


@dataclass(frozen=True)
class EmbeddingRunConfig:
    backend: EmbeddingBackend
    max_chars: int


def embedding_input_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        raise ValueError("embedding max chars must be positive")
    if len(text) <= max_chars:
        return text
    marker = f"\n[embedding input truncated from {len(text)} chars to fit model context]"
    if len(marker) >= max_chars:
        return text[:max_chars]
    keep = max_chars - len(marker)
    return text[:keep].rstrip() + marker


def embedding_retry_min_chars() -> int:
    return config.env_int("PROJECT_CODE_INTELLIGENCE_EMBEDDING_MIN_CHARS", 800, minimum=200)


def smaller_embedding_max_chars(max_chars: int) -> int | None:
    if max_chars <= 0:
        return None
    minimum = embedding_retry_min_chars()
    if max_chars <= minimum:
        return None
    return max(minimum, max_chars // 2)


def is_context_size_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "context size" in message
        or "context has been exceeded" in message
        or ("context" in message and "exceeded" in message)
        or "n_ctx" in message
    )


def http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except (OSError, UnicodeError, ValueError):
        body = ""
    if body:
        return f"{exc}; response body: {body[:1200]}"
    return str(exc)


def endpoint_host_is_loopback(hostname: str) -> bool:
    normalized = hostname.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_embedding_endpoint(endpoint: str, *, env: config.Env | None = None) -> None:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("embedding endpoint must use http or https")
    if not parsed.netloc or not parsed.hostname:
        raise ValueError("embedding endpoint must include a host")
    if endpoint_host_is_loopback(parsed.hostname):
        return
    if config.env_bool("PROJECT_CODE_INTELLIGENCE_ALLOW_REMOTE_EMBEDDING", default=False, env=env):
        return
    raise ValueError(
        "remote embedding endpoints are disabled by default because code-derived "
        "text is sent to the endpoint; set PROJECT_CODE_INTELLIGENCE_ALLOW_REMOTE_EMBEDDING=1 "
        "to allow a trusted remote endpoint"
    )


def embedding_headers(endpoint: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = config.embedding_api_key(endpoint)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def read_embedding_response(endpoint: str, payload: bytes, headers: dict[str, str], *, track_metrics: bool) -> str:
    request = http_client.request(
        endpoint,
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        return http_client.read_text(request, timeout=3600)
    except urllib.error.HTTPError as exc:
        detail = http_error_detail(exc)
        if track_metrics:
            runtime_state.active_metrics.add("embedding_batch_errors", 1)
            if is_context_size_error(ValueError(detail)):
                runtime_state.active_metrics.add("embedding_context_errors", 1)
        raise EmbeddingEndpointUnavailableError(
            embedding_endpoint_hint(endpoint, ValueError(detail)),
            recoverable_batch=exc.code in {400, 413, 500},
        ) from exc
    except (ConnectionError, TimeoutError, urllib.error.URLError) as exc:
        raise EmbeddingEndpointUnavailableError(embedding_endpoint_hint(endpoint, exc)) from exc


def parse_embedding_items(endpoint: str, raw_response: str, expected_count: int) -> tuple[list[JsonObject], JsonObject]:
    try:
        data_value = cast("object", json.loads(raw_response))
    except json.JSONDecodeError as exc:
        raise EmbeddingEndpointUnavailableError(embedding_endpoint_hint(endpoint, exc)) from exc
    if not isinstance(data_value, dict):
        raise EmbeddingEndpointUnavailableError(
            embedding_endpoint_hint(endpoint, ValueError("embedding API response must be an object"))
        )
    data = cast("JsonObject", data_value)
    items_value = data.get("data")
    if not isinstance(items_value, list) or len(items_value) != expected_count:
        raise EmbeddingEndpointUnavailableError(
            embedding_endpoint_hint(endpoint, ValueError("unexpected embedding API response"))
        )
    if not all(isinstance(item, dict) for item in items_value):
        raise EmbeddingEndpointUnavailableError(
            embedding_endpoint_hint(endpoint, ValueError("embedding API response items must be objects"))
        )
    items = [cast("JsonObject", item) for item in items_value]
    return items, data


def embedding_index(item: JsonObject) -> int:
    value = item.get("index")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def vector_literals_from_items(endpoint: str, items: list[JsonObject]) -> list[str]:
    vectors: list[str] = []
    for item in sorted(items, key=embedding_index):
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise EmbeddingEndpointUnavailableError(
                embedding_endpoint_hint(endpoint, ValueError("embedding API response item missing embedding list"))
            )
        vectors.append(db.vector_literal(embedding))
    return vectors


def embed_with_endpoint(endpoint: str, texts: list[str], model: str, *, track_metrics: bool = True) -> list[str]:
    validate_embedding_endpoint(endpoint)
    if track_metrics:
        runtime_state.active_metrics.add_embedding_inputs(texts)
    started = time.monotonic()
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    try:
        raw_response = read_embedding_response(
            endpoint,
            payload,
            embedding_headers(endpoint),
            track_metrics=track_metrics,
        )
    finally:
        if track_metrics:
            runtime_state.active_metrics.add("embedding_seconds", time.monotonic() - started)
    items, data = parse_embedding_items(endpoint, raw_response, len(texts))
    if track_metrics:
        runtime_state.active_metrics.add_embedding_usage(data.get("usage"))
    return vector_literals_from_items(endpoint, items)


def embedding_endpoint_hint(endpoint: str, exc: BaseException) -> str:
    return (
        f"Embedding endpoint is not reachable or is not serving embeddings: {endpoint}\n"
        "\n"
        "For the portable local embedding demo, start FastEmbed from the project-code-intelligence checkout:\n"
        "  docker compose --profile cpu up -d --build fastembed\n"
        "\n"
        "The FastEmbed service listens on:\n"
        "  http://127.0.0.1:18081/v1/embeddings\n"
        "\n"
        "Or point PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT at another trusted OpenAI-compatible "
        "embeddings provider.\n"
        f"Connection detail: {exc}"
    )


def preflight_embedding_endpoint(endpoint: str, model: str) -> None:
    started = time.monotonic()
    try:
        _ = embed_with_endpoint(endpoint, ["code intelligence embedding preflight"], model, track_metrics=False)
    finally:
        runtime_state.active_metrics.add("embedding_preflight_seconds", time.monotonic() - started)


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


def embed_texts_once(
    texts: list[str],
    *,
    backend: EmbeddingBackend,
) -> list[str]:
    if backend.endpoint:
        return embed_with_endpoint(backend.endpoint, texts, backend.endpoint_model)
    if backend.use_llama_cli:
        runtime_state.active_metrics.add_embedding_inputs(texts)
        started = time.monotonic()
        try:
            return llama_batch_embeddings(texts, len(texts))
        finally:
            runtime_state.active_metrics.add("embedding_seconds", time.monotonic() - started)
    raise ValueError("embedding endpoint or llama CLI is required")


def embed_items_with_retry(
    items: list[T],
    *,
    run_config: EmbeddingRunConfig,
    text_for: Callable[[T], str],
    skip_item: Callable[[T, BaseException, int], None],
    retry_event_values: Callable[[T], JsonObject],
) -> tuple[list[tuple[T, str]], int]:
    texts = [embedding_input_text(text_for(item), run_config.max_chars) for item in items]
    try:
        embeddings = embed_texts_once(
            texts,
            backend=run_config.backend,
        )
    except EmbeddingEndpointUnavailableError as exc:
        if not exc.recoverable_batch:
            raise
        if len(items) > 1:
            midpoint = len(items) // 2
            left_embedded, left_skipped = embed_items_with_retry(
                items[:midpoint],
                run_config=run_config,
                text_for=text_for,
                skip_item=skip_item,
                retry_event_values=retry_event_values,
            )
            right_embedded, right_skipped = embed_items_with_retry(
                items[midpoint:],
                run_config=run_config,
                text_for=text_for,
                skip_item=skip_item,
                retry_event_values=retry_event_values,
            )
            return left_embedded + right_embedded, left_skipped + right_skipped
        retry_max_chars = smaller_embedding_max_chars(run_config.max_chars)
        if retry_max_chars is not None:
            runtime_state.active_metrics.add("embedding_retried_smaller", 1)
            progress_event(
                "code_intel_embedding_retry_smaller",
                **retry_event_values(items[0]),
                max_chars=run_config.max_chars,
                retry_max_chars=retry_max_chars,
            )
            return embed_items_with_retry(
                items,
                run_config=EmbeddingRunConfig(backend=run_config.backend, max_chars=retry_max_chars),
                text_for=text_for,
                skip_item=skip_item,
                retry_event_values=retry_event_values,
            )
        skip_item(items[0], exc, run_config.max_chars)
        return [], 1

    return list(zip(items, embeddings, strict=True)), 0


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
        record.embedding = embedding
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
) -> tuple[int, int, int]:
    inserted = 0
    non_embedding_records = [record for record in records if id(record) not in state.selected_ids]
    for offset in range(0, len(non_embedding_records), 1000):
        inserted += insert_records(insert_context, non_embedding_records[offset : offset + 1000])
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
            inserted += insert_records(insert_context, batch)
        state.consumed_batches = len(state.batches)
    return inserted, state.embedded, state.skipped


def abandon_preembedding(state: PreEmbeddingState | None) -> None:
    if state is not None:
        state.cancel_event.set()


def embed_db_records(
    snapshot_ids: list[int],
    *,
    record_types: set[str],
    batch_size: int,
    run_config: EmbeddingRunConfig,
) -> int:
    if not record_types:
        return 0

    def mark_skipped(conn: db.DbConnection, row: EmbeddingRow, reason: BaseException, skipped_max_chars: int) -> None:
        metadata = {
            "embedding_skipped": True,
            "embedding_skip_reason": str(reason)[:500],
            "embedding_skip_max_chars": skipped_max_chars,
        }
        started = time.monotonic()
        try:
            _ = conn.execute(
                """
                UPDATE project_code_intel_records
                SET metadata = coalesce(metadata, '{}'::jsonb) || %s::jsonb
                WHERE id = %s
                """,
                [json.dumps(metadata, sort_keys=True, separators=(",", ":")), row["id"]],
            )
        finally:
            runtime_state.active_metrics.add("embedding_db_update_seconds", time.monotonic() - started)
        runtime_state.active_metrics.add("embedding_skipped_records", 1)
        progress_event(
            "code_intel_embedding_skipped",
            record_id=row["id"],
            source_path=row.get("source_path"),
            source_record_id=row.get("record_id"),
            reason=str(reason)[:240],
        )

    def embed_batch(conn: db.DbConnection, batch: list[EmbeddingRow], batch_max_chars: int) -> tuple[int, int]:
        embedded, skipped_count = embed_items_with_retry(
            batch,
            run_config=EmbeddingRunConfig(backend=run_config.backend, max_chars=batch_max_chars),
            text_for=itemgetter("embedding_text"),
            skip_item=lambda row, reason, skipped_max_chars: mark_skipped(conn, row, reason, skipped_max_chars),
            retry_event_values=lambda row: {
                "record_id": row["id"],
                "source_path": row.get("source_path"),
                "source_record_id": row.get("record_id"),
            },
        )

        update_started = time.monotonic()
        try:
            for row, embedding in embedded:
                _ = conn.execute(
                    "UPDATE project_code_intel_records SET embedding = %s::vector WHERE id = %s",
                    [embedding, row["id"]],
                )
        finally:
            runtime_state.active_metrics.add("embedding_db_update_seconds", time.monotonic() - update_started)
        return len(embedded), skipped_count

    embedded = 0
    skipped = 0
    with db.connect(readonly=False) as conn:
        for snapshot_id in snapshot_ids:
            rows = cast(
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
                    """,
                    [snapshot_id, sorted(record_types)],
                ).fetchall(),
            )
            total = len(rows)
            runtime_state.active_metrics.add_phase_total(total)
            runtime_state.active_metrics.add("embedding_records_selected", total)
            progress_event("code_intel_embedding_selected", snapshot_id=snapshot_id, records=total)
            for offset in range(0, total, batch_size):
                batch = rows[offset : offset + batch_size]
                batch_embedded, batch_skipped = embed_batch(conn, batch, run_config.max_chars)
                commit_started = time.monotonic()
                try:
                    conn.commit()
                finally:
                    runtime_state.active_metrics.add("embedding_db_update_seconds", time.monotonic() - commit_started)
                embedded += batch_embedded
                skipped += batch_skipped
                runtime_state.active_metrics.add_phase_done(len(batch))
                runtime_state.active_metrics.add("embedded_records", batch_embedded)
                progress_event(
                    "code_intel_embedded",
                    snapshot_id=snapshot_id,
                    records=offset + len(batch),
                    total_records=total,
                    embedded_total=embedded,
                    skipped_total=skipped,
                )
    return embedded
