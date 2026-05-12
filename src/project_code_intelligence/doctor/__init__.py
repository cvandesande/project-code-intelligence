"""Native environment diagnostics for project-code-intelligence."""

from __future__ import annotations

from project_code_intelligence.doctor.cli import DoctorArgs, check_results, main, parser, write_stdout
from project_code_intelligence.doctor.common import (
    human_bytes,
    result,
    status_for_requirement,
    version_at_least,
    version_tuple,
)
from project_code_intelligence.doctor.embeddings import (
    check_embedding_endpoint,
    check_embedding_options,
    remote_provider_precheck,
)
from project_code_intelligence.doctor.hardware import (
    check_gpu_support,
    check_npu_support,
    check_platform,
    cpu_suggests_supported_amd_npu,
    discover_gpus,
    gpu_memory_summary,
    parse_nvidia_smi_csv,
)
from project_code_intelligence.doctor.output import (
    color_text,
    exit_code,
    format_result,
    format_summary,
    should_use_color,
    status_rank,
    summary_status,
)
from project_code_intelligence.doctor.types import CheckResult, ColorMode, EmbeddingMode, GpuInfo

__all__ = [
    "CheckResult",
    "ColorMode",
    "DoctorArgs",
    "EmbeddingMode",
    "GpuInfo",
    "check_embedding_endpoint",
    "check_embedding_options",
    "check_gpu_support",
    "check_npu_support",
    "check_platform",
    "check_results",
    "color_text",
    "cpu_suggests_supported_amd_npu",
    "discover_gpus",
    "exit_code",
    "format_result",
    "format_summary",
    "gpu_memory_summary",
    "human_bytes",
    "main",
    "parse_nvidia_smi_csv",
    "parser",
    "remote_provider_precheck",
    "result",
    "should_use_color",
    "status_for_requirement",
    "status_rank",
    "summary_status",
    "version_at_least",
    "version_tuple",
    "write_stdout",
]
