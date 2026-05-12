"""Shared data types for native environment diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from project_code_intelligence import config
    from project_code_intelligence.embedding.bench import EmbeddingRequestResult

Status = Literal["ok", "warn", "fail", "skip"]
EmbeddingMode = Literal["auto", "required", "skip"]
ColorMode = Literal["auto", "always", "never"]
EmbeddingRequester: TypeAlias = Callable[[str, str, list[str], float], "EmbeddingRequestResult"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    detail: str | None = None


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vendor: str
    vendor_id: str | None = None
    device_id: str | None = None
    driver: str | None = None
    card: str | None = None
    pci_slot: str | None = None
    vram_bytes: int | None = None
    visible_vram_bytes: int | None = None
    shared_bytes: int | None = None
    source: str = "sysfs"


@dataclass(frozen=True)
class EmbeddingEndpointCheck:
    env: config.Env
    endpoint: str
    model: str
    required: bool
    timeout: float
    requester: EmbeddingRequester
