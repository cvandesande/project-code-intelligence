"""OpenAI-compatible embedding endpoint helpers."""

from __future__ import annotations

import ipaddress
import json
import time
import urllib.error
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from project_code_intelligence import config, db, http_client
from project_code_intelligence import runtime as runtime_state
from project_code_intelligence.embedding.types import EmbeddingEndpointUnavailableError
from project_code_intelligence.runtime import progress_event

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject


def embedding_request_timeout_seconds() -> float:
    return config.env_float("PROJECT_CODE_INTELLIGENCE_EMBEDDING_REQUEST_TIMEOUT_SECONDS", 300.0, minimum=1.0)


def embedding_request_retries() -> int:
    return config.env_int("PROJECT_CODE_INTELLIGENCE_EMBEDDING_REQUEST_RETRIES", 3, minimum=0)


def retry_sleep_seconds(attempt: int) -> float:
    return min(30.0, 2.0 ** max(0, attempt - 1))


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
        return http_client.read_text(request, timeout=embedding_request_timeout_seconds())
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
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    attempts = embedding_request_retries() + 1
    raw_response = ""
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            raw_response = read_embedding_response(
                endpoint,
                payload,
                embedding_headers(endpoint),
                track_metrics=track_metrics,
            )
            break
        except EmbeddingEndpointUnavailableError as exc:
            if exc.recoverable_batch or attempt >= attempts:
                raise
            runtime_state.active_metrics.add("embedding_endpoint_retries", 1)
            progress_event(
                "code_intel_embedding_endpoint_retry",
                attempt=attempt,
                attempts=attempts,
                reason=str(exc)[:240],
            )
            time.sleep(retry_sleep_seconds(attempt))
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
