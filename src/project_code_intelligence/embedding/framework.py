"""Shared user-facing embedding runtime labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from project_code_intelligence.embedding.endpoint import endpoint_host_is_loopback

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class EmbeddingRuntimeProfile:
    profile: str
    label: str


OPTION_LABELS: dict[str, str] = {
    "option-cpu": "CPU",
    "option-npu": "NPU",
    "option-gpu-amd": "AMD ROCm",
    "option-gpu-nvidia": "NVIDIA CUDA",
    "option-gpu-apple": "Apple MLX",
    "option-gpu": "GPU",
    "option-gpu-large-model": "Large GPU model",
    "option-remote": "Remote",
}


def option_label(name: str) -> str:
    return OPTION_LABELS.get(name, name)


def endpoint_is_remote(endpoint: str | None) -> bool:
    if not endpoint:
        return False
    hostname = urlsplit(endpoint).hostname
    return bool(hostname and not endpoint_host_is_loopback(hostname))


def _advertised_profile(framework: str | None) -> EmbeddingRuntimeProfile | None:
    if not framework:
        return None
    normalized = framework.lower()
    profiles = (
        ("amdgpu", ("rocm",)),
        ("nvidia", ("cuda", "nvidia")),
        ("apple", ("mlx", "apple")),
        ("cpu", ("fastembed", "cpu")),
        ("npu", ("lemonade", "npu")),
    )
    profile = next(
        (profile_name for profile_name, markers in profiles if any(marker in normalized for marker in markers)),
        "local",
    )
    return EmbeddingRuntimeProfile(profile, framework)


def active_embedding_profile(
    *,
    endpoint: str | None,
    response_model: str | None,
    option_ok: Callable[[str], bool],
    endpoint_ok: bool,
    advertised_framework: str | None = None,
) -> EmbeddingRuntimeProfile:
    if endpoint_ok and endpoint_is_remote(endpoint):
        return EmbeddingRuntimeProfile("remote", "Remote")

    model = (response_model or "").lower()
    candidates = [
        (
            "npu",
            "NPU",
            ("embed-gemma" in model or model.endswith("-flm")) and option_ok("option-npu"),
        ),
        (
            "cpu",
            "CPU",
            ("jina" in model or "bge" in model or "fastembed" in model) and option_ok("option-cpu"),
        ),
        (
            "amdgpu",
            "AMD ROCm",
            ("qwen" in model or ".gguf" in model) and option_ok("option-gpu-amd"),
        ),
        (
            "nvidia",
            "NVIDIA CUDA",
            ("qwen" in model or ".gguf" in model) and option_ok("option-gpu-nvidia"),
        ),
        (
            "apple",
            "Apple MLX",
            (".gguf" in model or "nomic" in model or "qwen" in model) and option_ok("option-gpu-apple"),
        ),
        ("gpu", "GPU", "qwen" in model or ".gguf" in model),
    ]
    for profile, label, matched in candidates:
        if matched:
            return EmbeddingRuntimeProfile(profile, label)
    if advertised := _advertised_profile(advertised_framework):
        return advertised
    return EmbeddingRuntimeProfile("local", "Local endpoint")
