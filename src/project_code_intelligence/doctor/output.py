"""Human-readable doctor output formatting."""

from __future__ import annotations

import os
import re
import sys
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from project_code_intelligence.doctor.types import CheckResult, ColorMode, Status

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_BOLD_CYAN = "\033[1;36m"
ANSI_DIM = "\033[2m"
STATUS_COLORS: dict[Status, str] = {
    "ok": "\033[32m",
    "warn": "\033[33m",
    "fail": "\033[31m",
    "skip": "\033[36m",
}
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def status_rank(status: Status) -> int:
    return {"ok": 0, "skip": 0, "warn": 1, "fail": 2}[status]


def exit_code(results: Sequence[CheckResult]) -> int:
    return 1 if any(item.status == "fail" for item in results) else 0


def should_use_color(
    mode: ColorMode, *, stdout_isatty: bool | None = None, env: Mapping[str, str] | None = None
) -> bool:
    env = os.environ if env is None else env
    if mode == "always":
        return True
    if mode == "never" or "NO_COLOR" in env:
        return False
    if env.get("FORCE_COLOR"):
        return True
    is_tty = sys.stdout.isatty() if stdout_isatty is None else stdout_isatty
    return is_tty and env.get("TERM") != "dumb"


def color_text(text: str, color: str, *, enabled: bool) -> str:
    return f"{color}{text}{ANSI_RESET}" if enabled else text


def heading_text(text: str, *, color: bool = False) -> str:
    return color_text(text, ANSI_BOLD, enabled=color)


def label_text(text: str, *, color: bool = False) -> str:
    return color_text(text, ANSI_BOLD, enabled=color)


def profile_text(text: str, *, color: bool = False) -> str:
    return color_text(text, ANSI_BOLD_CYAN, enabled=color)


def status_text(status: Status, *, color: bool = False) -> str:
    return color_text(status, STATUS_COLORS[status], enabled=color)


def format_result(item: CheckResult, *, color: bool = False) -> str:
    status = status_text(item.status, color=color)
    line = f"[{status}] {item.name}: {item.message}"
    if item.detail:
        line += f"\n    {color_text(item.detail, ANSI_DIM, enabled=color)}"
    return line


def result_map(results: Sequence[CheckResult]) -> dict[str, CheckResult]:
    return {item.name: item for item in results}


def matching_results(results: Sequence[CheckResult], pattern: str) -> list[CheckResult]:
    compiled = re.compile(pattern)
    return [item for item in results if compiled.fullmatch(item.name)]


def concise_line(label: str, item: CheckResult | None, *, color: bool = False) -> str:
    formatted_label = label_text(label, color=color)
    if item is None:
        return f"  {formatted_label}: {status_text('skip', color=color)} not checked"
    return f"  {formatted_label}: {status_text(item.status, color=color)} {item.message}"


def summary_status(results: Sequence[CheckResult]) -> tuple[Status, str]:
    if any(item.status == "fail" for item in results):
        return "fail", "needs attention"
    if any(item.status == "warn" for item in results):
        return "warn", "usable with notes"
    return "ok", "ready"


def summary_issue_items(results: Sequence[CheckResult]) -> list[CheckResult]:
    return [item for item in results if item.status in {"fail", "warn"}]


def option_label(name: str) -> str:
    return {
        "option-cpu": "CPU",
        "option-npu": "NPU",
        "option-gpu-amd": "AMD GPU",
        "option-gpu-nvidia": "NVIDIA GPU",
        "option-gpu-apple": "Apple GPU",
        "option-gpu": "GPU",
        "option-gpu-large-model": "Large GPU model",
        "option-remote": "Remote",
    }.get(name, name)


def check_label(name: str) -> str:
    if name.startswith("option-"):
        return option_label(name)
    gpu_match = re.fullmatch(r"gpu-(\d+)", name)
    if gpu_match:
        return f"GPU {gpu_match.group(1)}"
    return {
        "apple-coreml": "Apple Core ML",
        "apple-coreml-model": "Apple Core ML model",
        "apple-metal": "Apple Metal",
        "apple-metal-model": "Apple embedding model",
        "configuration": "Configuration",
        "database": "Database",
        "database-config": "Database configuration",
        "embedding": "Embedding",
        "embedding-auth": "Embedding authentication",
        "embedding-config": "Embedding configuration",
        "embedding-endpoint": "Embedding endpoint",
        "embedding-model": "Embedding model",
        "embedding-policy": "Embedding policy",
        "embedding-provider": "Embedding provider",
        "gpu": "GPU",
        "gpu-runtime-amd": "AMD GPU runtime",
        "gpu-runtime-intel": "Intel GPU runtime",
        "gpu-runtime-nvidia": "NVIDIA GPU runtime",
        "npu": "NPU",
        "npu-driver": "NPU driver",
        "npu-firmware": "NPU firmware",
        "npu-kernel": "NPU kernel",
        "pgvector": "pgvector",
        "platform": "Platform",
        "schema": "Schema",
        "schema-version": "Schema version",
    }.get(name, name)


def format_option(item: CheckResult, *, color: bool = False) -> str:
    label = label_text(option_label(item.name), color=color)
    return f"  {label}: {status_text(item.status, color=color)} {item.message}"


def format_startup_command(profile: str, command: str, *, color: bool = False) -> str:
    return f"  {profile_text(profile, color=color)}: {command}"


def ok_result(by_name: Mapping[str, CheckResult], name: str) -> bool:
    item = by_name.get(name)
    return item is not None and item.status == "ok"


def embedding_config_endpoint(by_name: Mapping[str, CheckResult]) -> str | None:
    config_item = by_name.get("embedding-config")
    if config_item is None:
        return None
    match = re.search(r"\bendpoint=(\S+)", config_item.message)
    return match.group(1) if match else None


def remote_embedding_validated(by_name: Mapping[str, CheckResult]) -> bool:
    endpoint = embedding_config_endpoint(by_name)
    if not endpoint or not ok_result(by_name, "embedding-endpoint"):
        return False
    hostname = (urlsplit(endpoint).hostname or "").lower()
    return bool(hostname and hostname not in LOOPBACK_HOSTS)


def embedding_response_model(by_name: Mapping[str, CheckResult]) -> str | None:
    endpoint = by_name.get("embedding-endpoint")
    if endpoint is None:
        return None
    match = re.search(r"\bresponse model=([^;]+)", endpoint.message)
    return match.group(1).strip() if match else None


def active_embedding_profile(by_name: Mapping[str, CheckResult]) -> tuple[str, str]:
    if remote_embedding_validated(by_name):
        return "remote", "Remote"

    model = (embedding_response_model(by_name) or "").lower()
    candidates = [
        ("npu", "NPU", ("embed-gemma" in model or model.endswith("-flm")) and ok_result(by_name, "option-npu")),
        (
            "cpu",
            "CPU",
            ("jina" in model or "bge" in model or "fastembed" in model) and ok_result(by_name, "option-cpu"),
        ),
        ("amdgpu", "AMD GPU", ("qwen" in model or ".gguf" in model) and ok_result(by_name, "option-gpu-amd")),
        (
            "nvidia",
            "NVIDIA GPU",
            ("qwen" in model or ".gguf" in model) and ok_result(by_name, "option-gpu-nvidia"),
        ),
        (
            "apple",
            "Apple",
            ok_result(by_name, "option-gpu-apple")
            and (ok_result(by_name, "apple-coreml") or "qwen" in model or ".gguf" in model),
        ),
        ("gpu", "GPU", "qwen" in model or ".gguf" in model),
    ]
    for profile, label, matched in candidates:
        if matched:
            return profile, label
    return "local", "Local endpoint"


def active_embedding_lines(by_name: Mapping[str, CheckResult], *, color: bool = False) -> list[str]:
    endpoint = by_name.get("embedding-endpoint")
    if endpoint is None or endpoint.status != "ok":
        return []
    profile, label = active_embedding_profile(by_name)
    message = endpoint.message
    if config_item := by_name.get("embedding-config"):
        message = f"{message}; {config_item.message}"
    lines = [
        "",
        heading_text("Active embedding path", color=color),
        f"  {label_text(label, color=color)}: {status_text('ok', color=color)} {message}",
    ]
    if profile == "apple" and ok_result(by_name, "apple-coreml"):
        hint = "Run pci-coreml-server --diagnose to inspect ANE/GPU/CPU scheduling."
        lines.append(f"  {color_text(hint, ANSI_DIM, enabled=color)}")
    return lines


def summary_option_names(by_name: Mapping[str, CheckResult]) -> list[str]:
    has_apple = ok_result(by_name, "option-gpu-apple")
    option_names: list[str] = []
    # Skip CPU option when a better accelerated path is available.
    if not has_apple:
        option_names.append("option-cpu")
    option_names.extend([
        "option-npu",
        "option-gpu-amd",
        "option-gpu-nvidia",
        "option-gpu-apple",
        "option-gpu-large-model",
    ])
    if remote_embedding_validated(by_name):
        option_names.append("option-remote")
    return option_names


def local_embedding_startup_commands(by_name: Mapping[str, CheckResult]) -> list[tuple[str, str]]:
    commands: list[tuple[str, str]] = []
    # Skip CPU option when a better accelerated path is available.
    if ok_result(by_name, "option-cpu") and not ok_result(by_name, "option-gpu-apple"):
        commands.append(("cpu", "docker compose --profile cpu up -d --build fastembed"))
    if ok_result(by_name, "option-npu"):
        commands.append(("npu", "docker compose --profile npu up -d lemonade-npu"))
    if ok_result(by_name, "option-gpu-amd") and ok_result(by_name, "gpu-runtime-amd"):
        commands.append(("amdgpu", "docker compose --profile amdgpu up -d --build llama-rocm"))
    if ok_result(by_name, "option-gpu-nvidia") and ok_result(by_name, "gpu-runtime-nvidia"):
        commands.append(("nvidia", "docker compose --profile nvidia up -d --build llama-cuda"))
    if ok_result(by_name, "option-gpu-apple"):
        if ok_result(by_name, "apple-coreml"):
            commands.append(("apple", "pci-coreml-server"))
        else:
            commands.append(("apple", "pci-embedding-server"))
    return commands


def startup_command_lines(by_name: Mapping[str, CheckResult], *, color: bool = False) -> list[str]:
    lines = ["", heading_text("Startup commands", color=color)]
    if not ok_result(by_name, "database"):
        lines.append(format_startup_command("database", "docker compose up -d pgvector", color=color))
    for profile, command in local_embedding_startup_commands(by_name):
        lines.append(format_startup_command(profile, command, color=color))
    if remote_embedding_validated(by_name):
        lines.append(
            format_startup_command(
                "remote",
                "no embedding container needed; use the configured remote embedding endpoint",
                color=color,
            )
        )
    return lines


DOCKER_COMPOSE_SERVICES = ("fastembed", "lemonade-npu", "llama-rocm", "llama-cuda")
HOST_EMBEDDING_PROCESSES = ("pci-coreml-server", "pci-embedding-server")


def switch_embedding_lines(by_name: Mapping[str, CheckResult], *, color: bool = False) -> list[str]:
    active_profile, _ = active_embedding_profile(by_name)
    commands = [
        (profile, command)
        for profile, command in local_embedding_startup_commands(by_name)
        if profile != active_profile
    ]
    if not commands:
        return []
    lines = ["", heading_text("Switch embedding runtime", color=color)]
    lines.append("  Stop current local embedding service first: pci-doctor --stop-embedding")
    lines.extend(format_startup_command(profile, command, color=color) for profile, command in commands)
    return lines


def embedding_summary_lines(by_name: Mapping[str, CheckResult], *, color: bool = False) -> list[str]:
    option_names = summary_option_names(by_name)
    options = [by_name[name] for name in option_names if name in by_name and by_name[name].status == "ok"]
    if ok_result(by_name, "embedding-endpoint"):
        return [*active_embedding_lines(by_name, color=color), *switch_embedding_lines(by_name, color=color)]
    if not options:
        return []
    return [
        "",
        heading_text("Available embedding paths", color=color),
        *(format_option(item, color=color) for item in options),
    ]


def _stop_hints(by_name: Mapping[str, CheckResult], *, color: bool = False) -> list[str]:
    """Return contextual stop command hints based on running services."""
    db_running = ok_result(by_name, "database")
    embedding_running = ok_result(by_name, "embedding-endpoint")
    if not db_running and not embedding_running:
        return []
    lines: list[str] = []
    if db_running and embedding_running:
        lines.append(color_text("Stop all services: pci-doctor --stop", ANSI_DIM, enabled=color))
    if embedding_running:
        lines.append(color_text("Stop embedding server: pci-doctor --stop-embedding", ANSI_DIM, enabled=color))
    if db_running:
        lines.append(color_text("Stop database: pci-doctor --stop-database", ANSI_DIM, enabled=color))
    return lines


def _embedding_profile_description(profile: str) -> str:
    return {
        "cpu": "a local CPU embedding server",
        "npu": "a local AMD NPU embedding server",
        "amdgpu": "a local AMD GPU embedding server",
        "nvidia": "a local NVIDIA GPU embedding server",
        "apple": "a local Apple native embedding server",
    }.get(profile, f"a local {profile} embedding server")


def _suggestion_line(text: str, command: str, *, color: bool = False) -> str:
    return f"{text} {color_text(command, ANSI_BOLD_CYAN, enabled=color)}"


def _issue_suggestions(
    by_name: Mapping[str, CheckResult],
    issues: Sequence[CheckResult],
    *,
    color: bool = False,
) -> dict[str, list[str]]:
    """Map the first matching warn/fail check to friendly remediation hints."""
    suggestions: dict[str, list[str]] = {}
    db_names = {"database", "database-config", "pgvector", "schema", "schema-version"}
    embedding_names = {"embedding-endpoint", "embedding-config", "embedding"}
    db_done = False
    embedding_done = False
    for item in issues:
        if item.name in db_names and not db_done:
            suggestions[item.name] = [
                _suggestion_line(
                    "To start a local database, run:",
                    "docker compose up -d pgvector",
                    color=color,
                ),
                _suggestion_line(
                    "Or set",
                    "PROJECT_CODE_INTELLIGENCE_DATABASE_URL",
                    color=color,
                )
                + " to use an existing Postgres instance.",
            ]
            db_done = True
        elif item.name in embedding_names and not embedding_done:
            hints: list[str] = []
            commands = local_embedding_startup_commands(by_name)
            if commands:
                for profile, cmd in commands:
                    desc = _embedding_profile_description(profile)
                    hints.append(_suggestion_line(f"To start {desc}, run:", cmd, color=color))
            hints.append(
                _suggestion_line(
                    "Or set",
                    "PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT",
                    color=color,
                )
                + " to use a remote provider."
            )
            suggestions[item.name] = hints
            embedding_done = True
    return suggestions


def _needs_attention_lines(
    results: Sequence[CheckResult],
    by_name: Mapping[str, CheckResult],
    *,
    color: bool = False,
) -> list[str]:
    issues = summary_issue_items(results)
    if not issues:
        return []
    suggestions = _issue_suggestions(by_name, issues, color=color)
    lines = ["", heading_text("Needs attention", color=color)]
    for item in issues:
        label = check_label(item.name)
        lines.append(f"  {label_text(label, color=color)}: {status_text(item.status, color=color)} {item.message}")
        if item.detail:
            lines.append(f"    {color_text(item.detail, ANSI_DIM, enabled=color)}")
        lines.extend(f"    {s}" for s in suggestions.get(item.name, []))
    return lines


def format_summary(results: Sequence[CheckResult], *, color: bool = False) -> str:
    by_name = result_map(results)
    status, status_message = summary_status(results)
    lines = [
        heading_text("project-code-intelligence doctor", color=color),
        f"Status: {status_text(status, color=color)} {status_message}",
        "",
        heading_text("Detected", color=color),
        concise_line("Platform", by_name.get("platform"), color=color),
    ]

    gpus = matching_results(results, r"gpu-\d+")
    if gpus:
        lines.extend(concise_line("GPU", gpu, color=color) for gpu in gpus)
    else:
        lines.append(concise_line("GPU", by_name.get("gpu"), color=color))

    npu = by_name.get("npu")
    if npu and npu.status != "skip":
        lines.append(concise_line("NPU", npu, color=color))
    elif npu:
        lines.append(f"  {label_text('NPU', color=color)}: {status_text('skip', color=color)} not available")

    lines.extend([
        "",
        heading_text("Services", color=color),
        concise_line("Database", by_name.get("database"), color=color),
    ])
    if (pgvector := by_name.get("pgvector")) and pgvector.status != "ok":
        lines.append(concise_line("pgvector", pgvector, color=color))
    if (schema := by_name.get("schema")) and schema.status != "ok":
        lines.append(concise_line("Schema", schema, color=color))
    lines.append(
        concise_line("Embedding endpoint", by_name.get("embedding-endpoint") or by_name.get("embedding"), color=color)
    )

    lines.extend(embedding_summary_lines(by_name, color=color))
    lines.extend(_needs_attention_lines(results, by_name, color=color))

    stop_hints = _stop_hints(by_name, color=color)
    if stop_hints:
        lines.extend(["", *stop_hints])

    footer = color_text("Use --verbose for all checks, or --json for machine-readable output.", ANSI_DIM, enabled=color)
    lines.extend(["", footer])
    return "\n".join(lines)
