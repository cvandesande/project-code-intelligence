"""Core embedding retry and truncation behavior."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, TypeVar

from project_code_intelligence import config
from project_code_intelligence import runtime as runtime_state
from project_code_intelligence.embedding.endpoint import embed_with_endpoint
from project_code_intelligence.embedding.types import (
    EmbeddingBackend,
    EmbeddingEndpointUnavailableError,
    EmbeddingRunConfig,
)
from project_code_intelligence.embedding.utils import llama_batch_embeddings
from project_code_intelligence.progress import progress_event

if TYPE_CHECKING:
    from collections.abc import Callable

    from project_code_intelligence.models import JsonObject

T = TypeVar("T")


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
    return config.env_int("PCI_EMBEDDING_MIN_CHARS", 800, minimum=200)


def smaller_embedding_max_chars(max_chars: int) -> int | None:
    if max_chars <= 0:
        return None
    minimum = embedding_retry_min_chars()
    if max_chars <= minimum:
        return None
    return max(minimum, max_chars // 2)


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
