"""Embedding option and endpoint diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from project_code_intelligence import config
from project_code_intelligence.doctor_common import bool_from_env, result
from project_code_intelligence.doctor_hardware import (
    GIB,
    GPU_QWEN3_DEFAULT_MODEL,
    GPU_QWEN3_LARGE_MODEL,
    has_gpu_vendor,
    has_ready_npu,
    max_gpu_memory_bytes,
)
from project_code_intelligence.doctor_types import (
    CheckResult,
    EmbeddingEndpointCheck,
    EmbeddingMode,
    EmbeddingRequester,
    GpuInfo,
    Status,
)
from project_code_intelligence.embedding_bench import request_embeddings
from project_code_intelligence.embeddings import endpoint_host_is_loopback, validate_embedding_endpoint

if TYPE_CHECKING:
    from collections.abc import Sequence

OPENAI_EMBEDDING_ENDPOINT = "https://api.openai.com/v1/embeddings"
VOYAGE_EMBEDDING_ENDPOINT = "https://api.voyageai.com/v1/embeddings"


def default_gpu_model_detail(env: config.Env) -> str:
    model_file = config.env_text("PROJECT_CODE_INTELLIGENCE_HF_MODEL_FILE", GPU_QWEN3_DEFAULT_MODEL, env=env)
    model_path = Path("models") / (model_file or GPU_QWEN3_DEFAULT_MODEL)
    if model_path.is_file():
        return f"default GGUF present at {model_path}"
    return (
        f"default GGUF is {model_file}; Linux GPU profiles download it into ./models, "
        "Apple users should download it for host-native llama.cpp."
    )


def check_embedding_options(
    *,
    env: config.Env,
    gpus: Sequence[GpuInfo],
    npu_results: Sequence[CheckResult],
) -> list[CheckResult]:
    results = [
        result(
            "option-cpu",
            "ok",
            f"CPU embeddings: FastEmbed default {config.DEFAULT_FASTEMBED_MODEL}; fast option BAAI/bge-small-en-v1.5.",
        )
    ]

    if has_ready_npu(npu_results):
        results.append(
            result(
                "option-npu",
                "ok",
                f"AMD NPU embeddings: Lemonade FLM default {config.DEFAULT_LEMONADE_EMBEDDING_MODEL}.",
                "Use docker compose --profile npu up -d lemonade-npu. Set "
                "PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL only to override the default.",
            )
        )
    else:
        results.append(
            result(
                "option-npu",
                "skip",
                "No ready AMD XDNA 2 NPU embedding path was detected.",
                f"NPU model when supported: {config.DEFAULT_LEMONADE_EMBEDDING_MODEL}.",
            )
        )

    gpu_model_detail = default_gpu_model_detail(env)
    if has_gpu_vendor(gpus, "AMD"):
        results.append(
            result(
                "option-gpu-amd",
                "ok",
                f"AMD GPU embeddings: llama.cpp ROCm default {GPU_QWEN3_DEFAULT_MODEL}.",
                f"{gpu_model_detail} Use docker compose --profile amdgpu up -d --build llama-rocm.",
            )
        )
    if has_gpu_vendor(gpus, "NVIDIA"):
        results.append(
            result(
                "option-gpu-nvidia",
                "ok",
                f"NVIDIA GPU embeddings: llama.cpp CUDA default {GPU_QWEN3_DEFAULT_MODEL}.",
                f"{gpu_model_detail} Use docker compose --profile nvidia up -d --build llama-cuda.",
            )
        )
    if has_gpu_vendor(gpus, "Apple"):
        results.append(
            result(
                "option-gpu-apple",
                "ok",
                f"Apple GPU embeddings: host-native llama.cpp Metal default {GPU_QWEN3_DEFAULT_MODEL}.",
                f"{gpu_model_detail} Docker is not used for Apple GPU embeddings.",
            )
        )
    if not any(gpu.vendor in {"AMD", "NVIDIA", "Apple"} for gpu in gpus):
        results.append(
            result(
                "option-gpu",
                "skip",
                "No supported local GPU embedding profile was detected.",
                f"GPU model when supported: {GPU_QWEN3_DEFAULT_MODEL}.",
            )
        )

    gpu_memory = max_gpu_memory_bytes(gpus)
    if (
        gpu_memory is not None
        and gpu_memory >= 8 * GIB
        and any(gpu.vendor in {"AMD", "NVIDIA", "Apple"} for gpu in gpus)
    ):
        results.append(
            result(
                "option-gpu-large-model",
                "ok",
                f"Experimental GPU model candidate: {GPU_QWEN3_LARGE_MODEL}.",
                "This repository's small retrieval harness did not show a meaningful quality gain over "
                "the 0.6B default.",
            )
        )

    results.append(
        result(
            "option-remote",
            "ok",
            "Remote OpenAI-compatible embeddings: OpenAI text-embedding-3-small or Voyage voyage-3.5.",
            "Requires PROJECT_CODE_INTELLIGENCE_ALLOW_REMOTE_EMBEDDING=1 because source-derived text leaves "
            "the machine.",
        )
    )
    return results


def endpoint_is_remote(endpoint: str) -> bool:
    hostname = urlsplit(endpoint).hostname
    return bool(hostname and not endpoint_host_is_loopback(hostname))


def endpoint_host(endpoint: str) -> str:
    return (urlsplit(endpoint).hostname or "").lower()


def configured_embedding_requested(env: config.Env) -> tuple[bool, CheckResult | None]:
    embed, error = bool_from_env("PROJECT_CODE_INTELLIGENCE_EMBED", default=False, env=env)
    if error:
        return False, error
    return (
        embed
        or bool(config.env_text("PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT", env=env))
        or bool(config.env_text("PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL", env=env))
        or bool(config.env_text("PROJECT_CODE_INTELLIGENCE_LLAMA_MODEL", env=env)),
        None,
    )


def remote_provider_precheck(endpoint: str, model: str, *, required: bool, env: config.Env) -> CheckResult | None:
    host = endpoint_host(endpoint)
    status: Status = "fail" if required else "warn"
    if host == "api.anthropic.com":
        return result(
            "embedding-provider",
            status,
            "Anthropic's Claude API is not a first-party embeddings endpoint.",
            (
                "Use an embeddings endpoint such as Voyage AI "
                f"({VOYAGE_EMBEDDING_ENDPOINT}) or an OpenAI-compatible proxy."
            ),
        )
    if endpoint_is_remote(endpoint) and model == config.DEFAULT_EMBEDDING_ENDPOINT_MODEL:
        return result(
            "embedding-model",
            status,
            "Remote embedding endpoints need an explicit provider model.",
            (
                f"For OpenAI, try {config.DEFAULT_OPENAI_EMBEDDING_MODEL}. "
                f"For Anthropic-recommended Voyage, try {config.DEFAULT_VOYAGE_EMBEDDING_MODEL}."
            ),
        )
    if host == "api.openai.com" and not config.embedding_api_key(endpoint, env=env):
        return result(
            "embedding-auth",
            status,
            "OpenAI embeddings require an API key.",
            "Set PROJECT_CODE_INTELLIGENCE_EMBEDDING_API_KEY or OPENAI_API_KEY.",
        )
    if host == "api.voyageai.com" and not config.embedding_api_key(endpoint, env=env):
        return result(
            "embedding-auth",
            status,
            "Voyage embeddings require an API key.",
            "Set PROJECT_CODE_INTELLIGENCE_EMBEDDING_API_KEY or VOYAGE_API_KEY.",
        )
    return None


def check_embedding_endpoint_health(check: EmbeddingEndpointCheck, results: list[CheckResult]) -> list[CheckResult]:
    failure_status: Status = "fail" if check.required else "warn"
    provider_result = remote_provider_precheck(
        check.endpoint,
        check.model,
        required=check.required,
        env=check.env,
    )
    if provider_result:
        results.append(provider_result)
        return results

    try:
        validate_embedding_endpoint(check.endpoint, env=check.env)
    except ValueError as exc:
        results.append(result("embedding-policy", failure_status, str(exc)))
        return results

    try:
        response = check.requester(
            check.endpoint,
            check.model,
            ["project-code-intelligence doctor embedding check"],
            check.timeout,
        )
    except RuntimeError as exc:
        results.append(
            result(
                "embedding-endpoint",
                failure_status,
                "Embedding endpoint is not reachable or did not return OpenAI-compatible embeddings.",
                str(exc),
            )
        )
        return results

    results.append(
        result(
            "embedding-endpoint",
            "ok",
            f"response model={response.response_model}; "
            f"dimensions={response.dimensions}; latency={response.seconds:.3f}s",
        )
    )
    return results


def check_embedding_endpoint(
    *,
    env: config.Env,
    mode: EmbeddingMode,
    timeout: float,
    requester: EmbeddingRequester = request_embeddings,
) -> list[CheckResult]:
    if mode == "skip":
        return [result("embedding", "skip", "embedding check skipped")]

    requested, request_error = configured_embedding_requested(env)
    if request_error:
        return [request_error]

    endpoint = config.default_embedding_endpoint(env=env, local_default=True)
    if endpoint is None:
        return [result("embedding", "skip", "no embedding endpoint configured")]
    model = config.default_embedding_endpoint_model(env=env, endpoint=endpoint)
    results = [result("embedding-config", "ok", f"endpoint={endpoint} model={model}")]
    return check_embedding_endpoint_health(
        EmbeddingEndpointCheck(
            env=env,
            endpoint=endpoint,
            model=model,
            required=mode == "required" or requested,
            timeout=timeout,
            requester=requester,
        ),
        results,
    )
