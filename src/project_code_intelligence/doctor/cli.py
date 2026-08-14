"""Native environment diagnostics for project-code-intelligence."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING

from project_code_intelligence import config, db, process
from project_code_intelligence.doctor.common import (
    human_bytes,
    result,
    row_text,
    status_for_requirement,
    table_exists,
    version_at_least,
    version_tuple,
)
from project_code_intelligence.doctor.database import check_database
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
    format_postgres_bootstrap_result,
    format_result,
    format_summary,
    local_embedding_startup_commands,
    should_use_color,
    status_rank,
    summary_status,
)
from project_code_intelligence.doctor.types import (
    CheckResult,
    ColorMode,
    EmbeddingMode,
    GpuInfo,
)
from project_code_intelligence.embedding import apple_embed_server
from project_code_intelligence.embedding.apple_embed_server import APPLE_EMBED_SERVER_PID_FILE
from project_code_intelligence.exceptions import ConfigError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

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
    stop: bool
    stop_embedding: bool
    stop_database: bool
    clean: bool
    start: bool
    start_db: bool
    start_embedding: bool
    init_postgres: bool
    write_config: bool


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Check local database and embedding configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  PCI_DATABASE_URL='postgresql://codeintel:codeintel@127.0.0.1:5433/codeintel?sslmode=prefer' "
            "pci doctor\n"
            "  PCI_ALLOW_REMOTE_EMBEDDING=1 "
            "PCI_EMBEDDING_ENDPOINT='https://api.openai.com/v1/embeddings' "
            "PCI_EMBEDDING_ENDPOINT_MODEL='text-embedding-3-small' pci doctor\n"
            "  pci doctor --start-db\n"
        ),
    )
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
    _ = argument_parser.add_argument(
        "--stop", action="store_true", help="Stop all local services (database and embedding)."
    )
    _ = argument_parser.add_argument(
        "--stop-embedding", action="store_true", help="Stop local embedding services (Quadlet and host-native)."
    )
    _ = argument_parser.add_argument(
        "--stop-database", action="store_true", help="Stop the local pgvector database container."
    )
    _ = argument_parser.add_argument(
        "--clean",
        action="store_true",
        help="Stop all services and remove containers, volumes, and PID files. Prompts before destructive actions.",
    )
    _ = argument_parser.add_argument(
        "--start",
        action="store_true",
        help="Start the local pgvector database and the best available embedding service for this hardware.",
    )
    _ = argument_parser.add_argument(
        "--start-db",
        action="store_true",
        help="Start the local pgvector database container.",
    )
    _ = argument_parser.add_argument(
        "--start-embedding",
        action="store_true",
        help="Start the best available local embedding service for this hardware.",
    )
    _ = argument_parser.add_argument(
        "--init-postgres",
        action="store_true",
        help=(
            "Bootstrap a remote PostgreSQL: create/update the role pci-index uses to initialize "
            "project databases and install pgvector into template1. Not needed for the bundled local "
            "container at 127.0.0.1. Requires PCI_POSTGRES_ADMIN_* credentials."
        ),
    )
    _ = argument_parser.add_argument(
        "--no-write-config",
        action="store_false",
        dest="write_config",
        default=True,
        help="Do not write pci-index credentials to the user config directory after --init-postgres.",
    )
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


_COMPOSE_DB_PROFILE = ("--profile", "db")
EMBEDDING_CONTAINER_NAMES = ("fastembed", "lemonade-npu", "llama-rocm", "llama-cuda")
_EMBEDDING_SERVICE_UNITS: dict[str, str] = {
    "cpu": "pci-fastembed.service",
    "npu": "pci-lemonade-npu.service",
    "amdgpu": "pci-llama-rocm.service",
    "nvidia": "pci-llama-cuda.service",
}
EMBEDDING_VOLUME_NAMES = (
    "pci-fastembed-models",
    "pci-lemonade-huggingface",
    "pci-lemonade-llama",
    "pci-lemonade-cache",
    "pci-llamacpp-rocm",
)
_HOST_PROCESSES = (
    "pci-embedding-server",
    "project_code_intelligence.embedding.apple_embed_server",
    "project_code_intelligence.embedding.coreml_server",
    "project_code_intelligence.embedding.fastembed_server",
)


def _stop_pid_file(pid_file: Path) -> None:
    """Send SIGTERM to the process recorded in pid_file, then remove it."""
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        with contextlib.suppress(OSError):
            pid_file.unlink()
    except OSError:
        return
    else:
        with contextlib.suppress(OSError):
            pid_file.unlink()


def _stop_pid_file_process() -> None:
    """Send SIGTERM to all host-native server processes recorded in PID files."""
    _stop_pid_file(APPLE_EMBED_SERVER_PID_FILE)


def stop_embedding_services() -> int:
    """Stop all local embedding services and return 0 on success."""
    # Stop Quadlet-managed embedding services (ignore errors if not running).
    try:
        for unit in _EMBEDDING_SERVICE_UNITS.values():
            _ = process.run_systemctl_user(["stop", unit], process.RunOptions(capture_output=True))
        write_stdout("Stopped Quadlet embedding services.")
    except FileNotFoundError:
        write_stdout("systemctl not found; skipping Quadlet embedding services.")

    _stop_pid_file_process()

    with contextlib.suppress(FileNotFoundError):
        for proc_name in _HOST_PROCESSES:
            _ = process.run(
                ["pkill", "-f", proc_name],
                process.RunOptions(capture_output=True),
            )
    write_stdout("Sent stop signals to host-native embedding processes.")

    return 0


def stop_database() -> int:
    """Stop the local pgvector database container and return 0 on success."""
    try:
        _ = process.run_docker(
            ["compose", *process.compose_file_args(), *_COMPOSE_DB_PROFILE, "stop", "pgvector"],
            process.RunOptions(capture_output=True),
        )
        write_stdout("Stopped pgvector database container.")
    except FileNotFoundError:
        write_stdout("docker not found; cannot stop pgvector container.")
    return 0


def stop_all_services() -> int:
    """Stop all local services (database + embedding) and return 0 on success."""
    _ = stop_embedding_services()
    _ = stop_database()
    return 0


def _confirm(prompt: str) -> bool:
    """Ask the user a yes/no question on stderr and return True for 'y'."""
    try:
        answer = input(prompt + " [y/N] ")
    except (EOFError, KeyboardInterrupt):
        write_stdout("")
        return False
    return answer.strip().lower() == "y"


def _database_content_summary(conn: db.DbConnection) -> str:
    if not table_exists(conn, "project_code_intel_records"):
        return "schema not initialized (no data)"
    snapshot_row = conn.execute("SELECT count(*) AS cnt FROM project_code_intel_snapshots").fetchone()
    record_row = conn.execute("SELECT count(*) AS cnt FROM project_code_intel_records").fetchone()
    snapshots = int(row_text(snapshot_row, "cnt")) if snapshot_row else 0
    records = int(row_text(record_row, "cnt")) if record_row else 0
    if snapshots == 0 and records == 0:
        return "database is empty"
    return f"database contains {snapshots} snapshot(s) and {records} record(s)"


def _database_summary() -> str | None:
    """Return a short summary of database content, or None if unavailable."""
    try:
        settings = db.inferred_database_role_settings(config.DatabaseSettings.from_env(), "rw")
    except ValueError:
        return None
    try:
        with db.connect(settings=settings) as conn:
            return _database_content_summary(conn)
    except db.DatabaseConnectionError:
        return None


def _remove_quadlet_embedding_state(quadlet_dir: Path) -> None:
    """Remove Quadlet-managed embedding volumes and generated unit files.

    Containers are already gone by this point: Quadlet's generated units
    remove their own container on `systemctl stop` (see stop_embedding_services).
    """
    try:
        for volume in EMBEDDING_VOLUME_NAMES:
            _ = process.run_podman(
                ["volume", "rm", "-f", f"systemd-{volume}"],
                process.RunOptions(capture_output=True),
            )
        write_stdout("Removed Quadlet embedding volumes.")
    except FileNotFoundError:
        write_stdout("podman not found; skipping Quadlet volume removal.")

    removed_units = False
    for filename in process.QUADLET_UNIT_FILES:
        target = quadlet_dir / filename
        if target.exists():
            target.unlink()
            removed_units = True
    if removed_units:
        write_stdout(f"Removed generated Quadlet unit files from {quadlet_dir}.")
        with contextlib.suppress(FileNotFoundError):
            _ = process.run_systemctl_user(["daemon-reload"])


def clean_all() -> int:
    """Stop all services and remove containers, volumes, PID files, and generated caches after confirmation."""
    # Gather what will be affected.
    summary = _database_summary()
    compose_cache = process.compose_cache_dir()
    quadlet_dir = process.quadlet_unit_dir()
    user_config = config.pci_index_user_config_path()
    lines = [
        "This will:",
        "  - Stop all embedding services (Quadlet and host-native)",
        "  - Stop and remove the pgvector database container and its volume",
        f"  - Remove generated Compose cache at {compose_cache}",
        f"  - Remove generated Quadlet unit files from {quadlet_dir}",
    ]
    if user_config is not None:
        lines.append(f"  - Remove generated pci-index user config at {user_config}")
    if summary:
        lines.append(f"  - {summary.upper() if 'contains' in summary else summary}")
    write_stdout("\n".join(lines))

    if summary and "contains" in summary:
        write_stdout("")
        write_stdout(f"WARNING: {summary}. All data will be permanently deleted.")
        if not _confirm("Are you sure you want to delete all data and remove services?"):
            write_stdout("Aborted.")
            return 1
    else:
        write_stdout("")
        if not _confirm("Proceed?"):
            write_stdout("Aborted.")
            return 1

    # 1. Stop embedding services and remove their volumes and unit files.
    _ = stop_embedding_services()
    _remove_quadlet_embedding_state(quadlet_dir)

    # 2. Stop and remove database container + volume.
    try:
        _ = process.run_docker(
            ["compose", *process.compose_file_args(), *_COMPOSE_DB_PROFILE, "down", "-v", "--remove-orphans"],
            process.RunOptions(capture_output=True),
        )
        write_stdout("Removed Docker Compose containers and volumes.")
    except FileNotFoundError:
        write_stdout("docker not found; skipping Docker Compose removal.")

    if compose_cache.exists():
        shutil.rmtree(compose_cache)
        write_stdout(f"Removed generated Compose cache at {compose_cache}.")

    if user_config is not None and user_config.exists():
        user_config.unlink()
        with contextlib.suppress(OSError):
            user_config.parent.rmdir()
        write_stdout(f"Removed generated pci-index user config at {user_config}.")

    write_stdout("Clean complete.")
    return 0


def _dispatch_stop(parsed: DoctorArgs) -> int | None:
    """Handle stop/clean flags and return exit code, or None to continue."""
    if parsed.clean:
        return clean_all()
    if parsed.stop:
        return stop_all_services()
    if parsed.stop_embedding:
        return stop_embedding_services()
    if parsed.stop_database:
        return stop_database()
    return None


def _detect_embedding_options() -> dict[str, CheckResult]:
    """Run hardware and options checks only — no DB or endpoint probing."""
    gpus = discover_gpus()
    results = check_platform(os.environ)
    results.extend(check_gpu_support(gpus))
    npu_results = check_npu_support(os.environ)
    results.extend(npu_results)
    results.extend(check_embedding_options(env=os.environ, gpus=gpus, npu_results=npu_results))
    return {r.name: r for r in results}


def start_database() -> int:
    """Start the local pgvector database container and return 0 on success."""
    engine = process.container_engine_name()
    write_stdout(f"Starting database: {engine} compose up -d pgvector")
    try:
        _ = process.run_docker(
            ["compose", *process.compose_file_args(), "up", "-d", "pgvector"],
            process.RunOptions(),
        )
    except FileNotFoundError:
        write_stdout("docker/podman not found; cannot start pgvector container.")
        return 1
    return 0


def start_embedding_services() -> int:
    """Detect available hardware and start the best local embedding service."""
    by_name = _detect_embedding_options()
    commands = local_embedding_startup_commands(by_name)
    if not commands:
        write_stdout("No local embedding service is available for this hardware.")
        return 1
    # Take the last option: Apple is always last when present; otherwise the most
    # capable detected hardware (NVIDIA/AMD > NPU > CPU) ends up last.
    profile, display_command = commands[-1]
    write_stdout(f"Starting {profile} embedding: {display_command}")
    if profile == "apple":
        # In-process call: the launcher daemonizes itself with sys.executable,
        # so no separate executable needs to be on PATH.
        apple_embed_server.main()
        return 0
    unit = _EMBEDDING_SERVICE_UNITS.get(profile)
    if unit is None:
        write_stdout(f"Unknown embedding profile: {profile!r}")
        return 1
    try:
        changed = process.materialize_quadlet_units()
        if changed:
            _ = process.run_systemctl_user(["daemon-reload"])
        _ = process.run_systemctl_user(["start", unit])
    except FileNotFoundError:
        write_stdout("podman/systemctl not found; cannot start embedding container.")
        return 1
    return 0


def start_all_services() -> int:
    """Start the database and best available embedding service."""
    db_code = start_database()
    emb_code = start_embedding_services()
    return db_code or emb_code


def _dispatch_start(parsed: DoctorArgs) -> int | None:
    """Handle start flags and return exit code, or None to continue."""
    if parsed.start:
        return start_all_services()
    if parsed.start_db:
        return start_database()
    if parsed.start_embedding:
        return start_embedding_services()
    return None


def init_postgres_roles(parsed: DoctorArgs) -> int:
    use_color = should_use_color(parsed.color)
    try:
        bootstrap = db.bootstrap_postgres_roles(config.DatabaseSettings.from_env(admin_scope="postgres"))
    except db.DatabaseConnectionError as exc:
        write_stdout(str(exc))
        return 1
    output = format_postgres_bootstrap_result(bootstrap, color=use_color)
    status = 0
    if getattr(parsed, "write_config", True):
        password = bootstrap.index_role.password
        if password is None:
            output += "\n\nCould not write pci-index config because the generated password is unavailable."
            status = 1
        else:
            try:
                config_path = config.write_pci_index_user_config(
                    database_url=bootstrap.postgres_url,
                    database_admin_user=bootstrap.index_role.name,
                    database_admin_password=password,
                )
            except (ConfigError, OSError) as exc:
                output += f"\n\nCould not write pci-index config: {exc}"
                status = 1
            else:
                output += f"\n\nSaved pci-index config to {config_path}\nPermissions: 0600"
    write_stdout(output)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parsed = parser().parse_args(argv, namespace=DoctorArgs())
    stop_result = _dispatch_stop(parsed)
    if stop_result is not None:
        return stop_result
    start_result = _dispatch_start(parsed)
    if start_result is not None:
        return start_result
    if parsed.init_postgres:
        return init_postgres_roles(parsed)
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
