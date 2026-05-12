"""Shared embedding data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


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


@dataclass(frozen=True)
class SkippedEmbeddingRow:
    row: EmbeddingRow
    reason: str
    max_chars: int
