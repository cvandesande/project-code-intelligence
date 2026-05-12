"""Native environment diagnostics for project-code-intelligence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING

from project_code_intelligence.doctor_common import (
    human_bytes,
    result,
    status_for_requirement,
    version_at_least,
    version_tuple,
)
from project_code_intelligence.doctor_database import check_database
from project_code_intelligence.doctor_embeddings import (
    check_embedding_endpoint,
    check_embedding_options,
    remote_provider_precheck,
)
from project_code_intelligence.doctor_hardware import (
    check_gpu_support,
    check_npu_support,
    check_platform,
    cpu_suggests_supported_amd_npu,
    discover_gpus,
    gpu_memory_summary,
    parse_nvidia_smi_csv,
)
from project_code_intelligence.doctor_output import (
    color_text,
    exit_code,
    format_result,
    format_summary,
    should_use_color,
    status_rank,
    summary_status,
)
from project_code_intelligence.doctor_types import (
    CheckResult,
    ColorMode,
    EmbeddingMode,
    GpuInfo,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from project_code_intelligence import config

__all__ = [
    "CheckResult",
    "GpuInfo",
    "check_embedding_endpoint",
    "check_embedding_options",
    "color_text",
    "cpu_suggests_supported_amd_npu",
    "format_result",
    "format_summary",
    "gpu_memory_summary",
    "human_bytes",
    "parse_nvidia_smi_csv",
    "remote_provider_precheck",
    "should_use_color",
    "status_for_requirement",
    "summary_status",
    "version_at_least",
    "version_tuple",
]


def write_stdout(message: str) -> None:
    _ = sys.stdout.write(message + "\n")


class DoctorArgs(argparse.Namespace):
    json: bool
    verbose: bool
    timeout: float
    embedding: EmbeddingMode
    skip_db: bool
    color: ColorMode


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description="Check local database and embedding configuration.")
    _ = argument_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON results.")
    _ = argument_parser.add_argument("--verbose", action="store_true", help="Print every diagnostic check.")
    _ = argument_parser.add_argument(
        "--color",
        "--colour",
        choices=("auto", "always", "never"),
        default="auto",
        help="Colorize text output. Defaults to auto.",
    )
    _ = argument_parser.add_argument(
        "--no-color",
        "--no-colour",
        action="store_const",
        const="never",
        dest="color",
        help="Disable ANSI color in text output.",
    )
    _ = argument_parser.add_argument(
        "--embedding",
        choices=("auto", "required", "skip"),
        default="auto",
        help=(
            "Embedding check mode. auto checks configured embeddings and treats the "
            "default local endpoint as optional; required fails when embeddings are unavailable."
        ),
    )
    _ = argument_parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Embedding preflight timeout in seconds.",
    )
    _ = argument_parser.add_argument("--skip-db", action="store_true", help="Skip PostgreSQL/pgvector checks.")
    return argument_parser


def check_results(args: DoctorArgs, env: config.Env | None = None) -> list[CheckResult]:
    env = os.environ if env is None else env
    gpus = discover_gpus()
    results = check_platform(env)
    results.extend(check_gpu_support(gpus))
    npu_results = check_npu_support(env)
    results.extend(npu_results)
    results.extend(check_embedding_options(env=env, gpus=gpus, npu_results=npu_results))
    if args.skip_db:
        results.append(result("database", "skip", "database check skipped"))
    else:
        results.extend(check_database())
    results.extend(check_embedding_endpoint(env=env, mode=args.embedding, timeout=args.timeout))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parsed = parser().parse_args(argv, namespace=DoctorArgs())
    results = check_results(parsed)
    if parsed.json:
        payload: Mapping[str, object] = {
            "ok": exit_code(results) == 0,
            "results": [asdict(item) for item in results],
        }
        write_stdout(json.dumps(payload, indent=2, sort_keys=True))
    else:
        use_color = should_use_color(parsed.color)
        if parsed.verbose:
            write_stdout("project-code-intelligence doctor")
            for item in sorted(results, key=lambda check: (-status_rank(check.status), check.name)):
                write_stdout(format_result(item, color=use_color))
        else:
            write_stdout(format_summary(results, color=use_color))
    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
