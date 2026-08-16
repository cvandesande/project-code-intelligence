"""Human-readable doctor output formatting."""

from __future__ import annotations

import io
import re
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from rich.box import ROUNDED
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from project_code_intelligence import config, console_ui
from project_code_intelligence.embedding.framework import (
    active_embedding_profile as select_active_embedding_profile,
)
from project_code_intelligence.embedding.framework import (
    option_label,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rich.console import RenderableType

    from project_code_intelligence.db import PostgresBootstrapResult
    from project_code_intelligence.doctor.types import CheckResult, ColorMode, Status

ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[2m"
STATUS_COLORS: dict[Status, str] = {
    "ok": "\033[32m",
    "warn": "\033[33m",
    "fail": "\033[31m",
    "skip": "\033[36m",
}
STATUS_GLYPHS: dict[Status, str] = {"ok": "✓", "warn": "⚠", "fail": "✗", "skip": "○"}
STATUS_STYLES: dict[Status, str] = {"ok": "green", "warn": "yellow", "fail": "red", "skip": "cyan"}
STATUS_PILL_TEXT: dict[Status, str] = {
    "ok": "READY",
    "warn": "USABLE WITH NOTES",
    "fail": "NEEDS ATTENTION",
    "skip": "READY",
}
SUMMARY_WIDTH = 78
SECONDS_AS_MS_THRESHOLD = 10


def status_rank(status: Status) -> int:
    return {"ok": 0, "skip": 0, "warn": 1, "fail": 2}[status]


def exit_code(results: Sequence[CheckResult]) -> int:
    return 1 if any(item.status == "fail" for item in results) else 0


def should_use_color(
    mode: ColorMode, *, stdout_isatty: bool | None = None, env: Mapping[str, str] | None = None
) -> bool:
    force = {"always": True, "never": False, "auto": None}[mode]
    return console_ui.should_emit_pretty(sys.stdout, force=force, env=env, isatty=stdout_isatty)


def color_text(text: str, color: str, *, enabled: bool) -> str:
    return f"{color}{text}{ANSI_RESET}" if enabled else text


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


def summary_status(results: Sequence[CheckResult]) -> tuple[Status, str]:
    if any(item.status == "fail" for item in results):
        return "fail", "needs attention"
    if any(item.status == "warn" for item in results):
        return "warn", "usable with notes"
    return "ok", "ready"


def summary_issue_items(results: Sequence[CheckResult]) -> list[CheckResult]:
    return [item for item in results if item.status in {"fail", "warn"}]


def check_label(name: str) -> str:
    if name.startswith("option-"):
        return option_label(name)
    gpu_match = re.fullmatch(r"gpu-(\d+)", name)
    if gpu_match:
        return f"GPU {gpu_match.group(1)}"
    return {
        "apple-metal": "Apple MLX",
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
        "project": "Project",
        "project-hooks": "Agent hooks",
        "project-mcp": "MCP configuration",
        "schema": "Schema",
        "schema-version": "Schema version",
    }.get(name, name)


def ok_result(by_name: Mapping[str, CheckResult], name: str) -> bool:
    item = by_name.get(name)
    return item is not None and item.status == "ok"


def embedding_config_endpoint(by_name: Mapping[str, CheckResult]) -> str | None:
    config_item = by_name.get("embedding-config")
    if config_item is None:
        return None
    match = re.search(r"\bendpoint=(\S+)", config_item.message)
    return match.group(1) if match else None


def embedding_response_model(by_name: Mapping[str, CheckResult]) -> str | None:
    endpoint = by_name.get("embedding-endpoint")
    if endpoint is None:
        return None
    match = re.search(r"\bresponse model=([^;]+)", endpoint.message)
    return match.group(1).strip() if match else None


def active_embedding_profile(by_name: Mapping[str, CheckResult]) -> tuple[str, str]:
    profile = select_active_embedding_profile(
        endpoint=embedding_config_endpoint(by_name),
        response_model=embedding_response_model(by_name),
        endpoint_ok=ok_result(by_name, "embedding-endpoint"),
        option_ok=lambda name: ok_result(by_name, name),
    )
    return profile.profile, profile.label


def local_embedding_startup_commands(by_name: Mapping[str, CheckResult]) -> list[tuple[str, str]]:
    commands: list[tuple[str, str]] = []
    if ok_result(by_name, "option-cpu") and not ok_result(by_name, "option-gpu-apple"):
        commands.append(("cpu", "systemctl --user start pci-fastembed.service"))
    if ok_result(by_name, "option-npu"):
        commands.append(("npu", "systemctl --user start pci-lemonade-npu.service"))
    if ok_result(by_name, "option-gpu-amd") and ok_result(by_name, "gpu-runtime-amd"):
        commands.append(("amdgpu", "systemctl --user start pci-llama-rocm.service"))
    if ok_result(by_name, "option-gpu-nvidia") and ok_result(by_name, "gpu-runtime-nvidia"):
        commands.append(("nvidia", "systemctl --user start pci-llama-cuda.service"))
    if ok_result(by_name, "option-gpu-apple"):
        commands.append(("apple", "pci embed apple"))
    return commands


def _shorten_platform(msg: str) -> str:
    match = re.match(r"Python (\S+) on (\S+) \S+ \(([^)]+)\)", msg)
    if match:
        return f"Python {match.group(1)} · {match.group(2)} {match.group(3)}"
    return msg


def _detail_value(detail: str | None, key: str) -> str | None:
    if not detail:
        return None
    match = re.search(rf"(?:^|;\s*){re.escape(key)}=([^;]+)", detail)
    return match.group(1).strip() if match else None


def _shorten_pci_ids(text: str) -> str:
    return re.sub(
        r"0x([0-9a-fA-F]+):0x([0-9a-fA-F]+)",
        lambda match: f"{match.group(1).upper()}:{match.group(2).upper()}",
        text,
    )


def _shorten_gpu_result(item: CheckResult) -> str:
    match = re.match(r"(.+):\s*(.*)", item.message)
    if not match:
        return item.message
    name, rest = match.group(1), match.group(2)
    name = _shorten_pci_ids(re.sub(r"\bGPU\b\s*", "", name).strip())
    parts = [name]
    if driver := _detail_value(item.detail, "driver"):
        parts.append(driver)
    rest = rest.replace("shared/unified=", "shared ")
    rest = rest.replace("visible VRAM=", "visible ")
    rest = rest.replace("VRAM=", "VRAM ")
    rest = rest.replace("; ", " · ")
    rest = re.sub(r"\b(\d+)\.0 (MiB|GiB)\b", r"\1 \2", rest)
    if rest:
        parts.append(rest)
    return " · ".join(parts)


def _shorten_npu(item: CheckResult) -> str:
    msg = _shorten_pci_ids(item.message)
    cores = re.search(r"(\d+) cores", msg)
    if "Apple Neural Engine" in msg and cores:
        return f"Apple Neural Engine · {cores.group(1)} cores"
    if msg.startswith("AMD NPU device detected:"):
        path = msg.split(":", 1)[1].strip()
        msg = f"AMD · {Path(path).name}"
    elif msg.startswith("AMD NPU "):
        msg = "AMD " + msg.removeprefix("AMD NPU ")
    parts = [msg]
    if driver := _detail_value(item.detail, "driver"):
        parts.append(driver)
    if device := _detail_value(item.detail, "device"):
        parts.append(device)
    if len(parts) == 1:
        return parts[0]
    return " · ".join(parts)


def _shorten_db(msg: str) -> str:
    match = re.search(r"\bat\s+(postgresql://\S+)", msg)
    if match:
        dsn = match.group(1)
        parts = urlsplit(dsn)
        host = parts.hostname
        if host:
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            port = f":{parts.port}" if parts.port is not None else ""
            return f"{host}{port}"
    return msg


def _shorten_model(model: str) -> str:
    model = model.strip()
    if "/" in model:
        model = model.rsplit("/", 1)[-1]
    return re.sub(r"\.(mlpackage|gguf|safetensors)$", "", model)


def _beautify_embedding(msg: str) -> str:
    model_match = re.search(r"response model=([^;]+)", msg)
    dim_match = re.search(r"dimensions=(\d+)", msg)
    lat_match = re.search(r"latency=([\d.]+)s", msg)
    parts: list[str] = []
    if model_match:
        parts.append(_shorten_model(model_match.group(1)))
    if lat_match:
        seconds = float(lat_match.group(1))
        parts.append(f"{round(seconds * 1000)} ms" if seconds < SECONDS_AS_MS_THRESHOLD else f"{seconds:.1f} s")
    if dim_match:
        parts.append(f"{dim_match.group(1)}d")
    return " · ".join(parts) if parts else msg


def _glyph(status: Status) -> Text:
    return Text(STATUS_GLYPHS[status], style=STATUS_STYLES[status])


def _section_table() -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(width=1, no_wrap=True)
    table.add_column(min_width=10, no_wrap=True)
    table.add_column(overflow="fold")
    return table


def _row(table: Table, status: Status, label: str, detail: str) -> None:
    table.add_row(_glyph(status), Text(label, style="bold"), Text(detail))


def _system_section(by_name: Mapping[str, CheckResult], results: Sequence[CheckResult]) -> Table:
    table = _section_table()
    if platform := by_name.get("platform"):
        _row(table, platform.status, "Platform", _shorten_platform(platform.message))
    gpus = matching_results(results, r"gpu-\d+")
    if gpus:
        for gpu in gpus:
            _row(table, gpu.status, "GPU", _shorten_gpu_result(gpu))
    elif (gpu := by_name.get("gpu")) is not None:
        if gpu.status == "skip":
            _row(table, "skip", "GPU", "not detected")
        else:
            _row(table, gpu.status, "GPU", _shorten_gpu_result(gpu))
    if (npu := by_name.get("npu")) is not None:
        if npu.status == "skip":
            _row(table, "skip", "NPU", "not available")
        else:
            _row(table, npu.status, "NPU", _shorten_npu(npu))
    return table


def _services_section(by_name: Mapping[str, CheckResult]) -> Table:
    table = _section_table()
    if database := by_name.get("database"):
        detail = _shorten_db(database.message) if database.status == "ok" else database.message
        _row(table, database.status, "Postgres", detail)
    if (pgvector := by_name.get("pgvector")) and pgvector.status != "ok":
        _row(table, pgvector.status, "pgvector", pgvector.message)
    if (schema := by_name.get("schema")) and schema.status != "ok":
        _row(table, schema.status, "Schema", schema.message)
    embedding = by_name.get("embedding-endpoint") or by_name.get("embedding")
    if embedding is not None:
        detail = _beautify_embedding(embedding.message) if embedding.status == "ok" else embedding.message
        _row(table, embedding.status, "Embedding", detail)
    return table


def _project_section(by_name: Mapping[str, CheckResult]) -> Table | None:
    project = by_name.get("project")
    if project is None:
        return None
    table = _section_table()
    _row(table, project.status, "Directory", project.message)
    if mcp := by_name.get("project-mcp"):
        _row(table, mcp.status, "MCP", mcp.message)
    if hooks := by_name.get("project-hooks"):
        _row(table, hooks.status, "Hooks", hooks.message)
    if freshness := by_name.get("index-freshness"):
        _row(table, freshness.status, "Index", freshness.message)
    return table


def _active_path_section(by_name: Mapping[str, CheckResult]) -> Group | None:
    endpoint = by_name.get("embedding-endpoint")
    if endpoint is None or endpoint.status != "ok":
        return None
    _profile, label = active_embedding_profile(by_name)
    url = embedding_config_endpoint(by_name)
    header = Table.grid(expand=True)
    header.add_column()
    header.add_column(justify="right")
    header.add_row(Text("Active path", style="bold"), Text(label, style="bold cyan"))
    pieces: list[RenderableType] = [header]
    if url:
        pieces.append(Text(f"  {url}", overflow="fold"))
    return Group(*pieces)


def _status_pill(status: Status) -> Text:
    return Text(f" {STATUS_GLYPHS[status]} {STATUS_PILL_TEXT[status]} ", style=f"bold reverse {STATUS_STYLES[status]}")


def _build_main_panel(by_name: Mapping[str, CheckResult], results: Sequence[CheckResult]) -> Panel:
    overall, _ = summary_status(results)

    header = Table.grid(expand=True)
    header.add_column()
    header.add_column(justify="right")
    header.add_row(
        Text("project-code-intelligence doctor", style="bold"),
        _status_pill(overall),
    )

    parts: list[RenderableType] = [
        header,
        Text(),
        Text("System", style="bold"),
        _system_section(by_name, results),
        Text(),
        Text("Services", style="bold"),
        _services_section(by_name),
    ]
    project_section = _project_section(by_name)
    if project_section is not None:
        parts.extend((Text(), Text("Current project", style="bold"), project_section))
    issues = summary_issue_items(results)
    service_names = {
        "database",
        "database-config",
        "pgvector",
        "embedding-endpoint",
        "embedding-config",
        "embedding",
        "project-mcp",
        "index-freshness",
    }
    non_service_issues = [item for item in issues if item.name not in service_names]
    if non_service_issues:
        parts.extend((Text(), Text("Needs attention", style="bold"), _issues_table(non_service_issues)))
    if issues:
        steps = _next_steps(by_name, issues)
        if steps:
            parts.extend((Text(), _next_steps_section(steps)))
    else:
        active = _active_path_section(by_name)
        if active is not None:
            parts.extend((Text(), active))
    return Panel(Group(*parts), box=ROUNDED, padding=(1, 2), border_style="dim", expand=True, width=SUMMARY_WIDTH)


def _issues_table(issues: Sequence[CheckResult]) -> Table:
    table = _section_table()
    for item in issues:
        _row(table, item.status, check_label(item.name), item.message)
    return table


_PROFILE_FRIENDLY_DESCRIPTIONS = {
    "cpu": "Start CPU embeddings",
    "npu": "Start AMD NPU embeddings",
    "amdgpu": "Start AMD ROCm embeddings",
    "nvidia": "Start NVIDIA CUDA embeddings",
    "apple": "Start Apple native embeddings",
}


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _target_is_local_postgres() -> bool | None:
    """Return True if Postgres is configured for the bundled loopback container,
    False if explicitly pointed at a remote host, and None if config can't be parsed.

    The bundled `docker compose pgvector` container ships with the pci-index role
    and pgvector pre-seeded, so --init-postgres is unnecessary in that case.
    """
    try:
        settings = config.DatabaseSettings.from_env()
    except ValueError:
        return None
    host = urlsplit(settings.dsn).hostname if settings.dsn else settings.host
    if host is None:
        return None
    return host.lower() in _LOOPBACK_HOSTS


def _next_steps(
    by_name: Mapping[str, CheckResult],
    issues: Sequence[CheckResult],
) -> list[tuple[str, str]]:
    server_db_names = {"database", "database-config"}
    project_db_names = {"pgvector", "schema", "schema-version"}
    embedding_names = {"embedding-endpoint", "embedding-config", "embedding", "apple-metal", "apple-metal-model"}
    issue_names = {item.name for item in issues}
    steps: list[tuple[str, str]] = []
    if issue_names & server_db_names:
        is_local = _target_is_local_postgres()
        if is_local is True:
            steps.extend((
                ("Start the bundled local database", "pci doctor --start-db"),
                ("Use an existing Postgres instead", "set PCI_DATABASE_URL"),
            ))
        elif is_local is False:
            steps.append(("Bootstrap a remote Postgres", "pci doctor --init-postgres"))
        else:
            steps.extend((
                ("Start the bundled local database", "pci doctor --start-db"),
                ("Bootstrap a remote Postgres", "pci doctor --init-postgres"),
                ("Use an existing Postgres instead", "set PCI_DATABASE_URL"),
            ))
    elif issue_names & project_db_names:
        steps.append(("Index a repo and bootstrap its inferred database", "pci-index ."))
    if issue_names & embedding_names:
        steps.extend(
            (_PROFILE_FRIENDLY_DESCRIPTIONS.get(profile, f"Start {profile} embeddings"), cmd)
            for profile, cmd in local_embedding_startup_commands(by_name)
        )
        steps.append((
            "Configure OpenAI embeddings",
            "export PCI_ALLOW_REMOTE_EMBEDDING=1 "
            "PCI_EMBEDDING_ENDPOINT=https://api.openai.com/v1/embeddings "
            f"PCI_EMBEDDING_ENDPOINT_MODEL={config.DEFAULT_OPENAI_EMBEDDING_MODEL} OPENAI_API_KEY=...",
        ))
    if "project-mcp" in issue_names:
        steps.extend((
            ("Install MCP for this project", "pci mcp install --target <client>"),
            ("Optionally install agent hooks", "pci hook install --target <client>"),
        ))
    return steps


def _next_steps_section(steps: Sequence[tuple[str, str]]) -> Group:
    table = Table.grid(padding=(0, 1))
    table.add_column(width=1, no_wrap=True)
    table.add_column(overflow="fold")
    for description, command in steps:
        table.add_row(Text("→", style="dim"), Text(description))
        table.add_row("", Text(command, style="bold cyan"))
    return Group(Text("Next steps", style="bold"), table)


def _build_console(buffer: io.StringIO, *, color: bool) -> Console:
    return Console(
        file=buffer,
        force_terminal=color,
        color_system="truecolor" if color else None,
        width=SUMMARY_WIDTH,
        record=False,
        highlight=False,
        legacy_windows=False,
        soft_wrap=False,
    )


def format_summary(results: Sequence[CheckResult], *, color: bool = False) -> str:
    by_name = result_map(results)
    buffer = io.StringIO()
    console = _build_console(buffer, color=color)

    console.print(_build_main_panel(by_name, results))

    return buffer.getvalue().rstrip("\n")


def _shell_export(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}"


def format_postgres_bootstrap_result(result: PostgresBootstrapResult, *, color: bool = False) -> str:
    buffer = io.StringIO()
    console = _build_console(buffer, color=color)
    role_state = "created" if result.index_role.created else "ready"

    header = Table.grid(expand=True)
    header.add_column()
    header.add_column(justify="right")
    header.add_row(
        Text("project-code-intelligence postgres roles", style="bold"),
        _status_pill("ok"),
    )

    accounts = _section_table()
    _row(accounts, "ok", "pci-index", f"{result.index_role.name} {role_state} · CREATEDB · CREATEROLE")
    if result.vector_template_ready:
        vector_state = (
            f"created in {result.template_database}"
            if result.vector_template_created
            else f"ready in {result.template_database}"
        )
        _row(accounts, "ok", "pgvector", vector_state)

    panel = Panel(
        Group(
            header,
            Text(),
            Text("Postgres", style="bold"),
            Text(f"  {result.postgres_url}", overflow="fold"),
            Text(),
            Text("Accounts", style="bold"),
            accounts,
            Text(),
            Text("Use this role as PCI_DATABASE_ADMIN_* for pci-index.", overflow="fold"),
            Text(
                "pci-index creates inferred project databases, schema, and scoped RO/RW roles. "
                f"New project databases inherit pgvector from {result.template_database}.",
                overflow="fold",
            ),
        ),
        box=ROUNDED,
        padding=(1, 2),
        border_style="dim",
        expand=True,
        width=SUMMARY_WIDTH,
    )
    console.print(panel)

    password = result.index_role.password or ""
    exports = "\n".join((
        "",
        "Export for pci-index",
        _shell_export("PCI_DATABASE_URL", result.postgres_url),
        _shell_export("PCI_DATABASE_ADMIN_USER", result.index_role.name),
        _shell_export("PCI_DATABASE_ADMIN_PASSWORD", password),
    ))
    return buffer.getvalue().rstrip("\n") + exports
