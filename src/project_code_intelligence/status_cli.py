"""``pci status``: show in-flight and recent index runs plus snapshot freshness.

Reads the index-run ledger (``project_code_intel_index_runs``) that the
indexer's ledger heartbeat maintains, so the background post-commit reindex is
observable: whether a run is active, its phase, embed progress, and a rough
time remaining.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

from rich.console import Group
from rich.text import Text

from project_code_intelligence import console_ui, db, process
from project_code_intelligence import runtime as runtime_state
from project_code_intelligence.doctor.cli import DOCKER_EMBEDDING_SERVICES
from project_code_intelligence.embedding.apple_embed_server import (
    APPLE_EMBED_SERVER_LOG_FILE,
    apple_embed_server_is_running,
)
from project_code_intelligence.mcp import db as mcp_db
from project_code_intelligence.mcp.status import annotate_status_snapshots
from project_code_intelligence.progress import estimate_remaining_seconds
from project_code_intelligence.storage import load_index_runs

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

# A run counts as live while its heartbeat is younger than this many ledger
# intervals; older with no finished_at means the process died or stalled.
_STALE_INTERVALS = 3
_MIN_STALE_SECONDS = 60


class StatusNamespace(argparse.Namespace):
    collection: str | None = None
    limit: int = 10
    json: bool = False


def _parse_args(argv: list[str] | None) -> StatusNamespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--collection", default=None, help="Only show this collection.")
    _ = parser.add_argument("--limit", type=int, default=10, help="Maximum run rows to show (default: 10).")
    _ = parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv, namespace=StatusNamespace())


def _stale_after_seconds() -> float:
    interval = runtime_state.run_ledger_seconds() or _MIN_STALE_SECONDS
    return max(interval * _STALE_INTERVALS, _MIN_STALE_SECONDS)


def _age_seconds(value: object, now: datetime) -> float | None:
    if not isinstance(value, datetime):
        return None
    stamped = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return (now - stamped).total_seconds()


def run_state(run: JsonObject, now: datetime) -> str:
    """One of: running, stalled, ok, failed, interrupted."""
    if run.get("finished_at") is None:
        age = _age_seconds(run.get("heartbeat_at"), now)
        if age is not None and age > _stale_after_seconds():
            return "stalled"
        return "running"
    if run.get("interrupted"):
        return "interrupted"
    exit_code = run.get("exit_code")
    return "ok" if exit_code == 0 else "failed"


def _progress_detail(run: JsonObject) -> str:
    progress = console_ui.as_object(run.get("progress"))
    prog = console_ui.as_object(progress.get("progress"))
    counts = console_ui.as_object(progress.get("counts"))
    timing = console_ui.as_object(progress.get("timing"))
    parts: list[str] = []
    phase = run.get("phase")
    if isinstance(phase, str) and phase:
        parts.append(phase.upper())
    done = console_ui.coerce_int(prog.get("phase_done"))
    total = console_ui.coerce_int(prog.get("phase_total"))
    if total:
        parts.append(f"{console_ui.format_count(done)}/{console_ui.format_count(total)}")
    remaining = estimate_remaining_seconds(
        cast("JsonObject", prog), cast("JsonObject", counts), cast("JsonObject", timing)
    )
    if remaining is not None and remaining > 0:
        parts.append(f"~{runtime_state.format_duration(remaining)} left")
    return " ".join(parts)


def _run_detail(run: JsonObject, now: datetime) -> str:
    state = run_state(run, now)
    parts = [state]
    progress_part = _progress_detail(run)
    if progress_part and state in {"running", "stalled"}:
        parts.append(progress_part)
    repos = run.get("repos")
    if isinstance(repos, list) and repos:
        parts.append(",".join(str(r) for r in repos))
    modes = console_ui.as_object(run.get("repo_modes"))
    mode_values = {str(v) for v in modes.values()}
    if mode_values:
        parts.append("mode " + (",".join(sorted(mode_values))))
    started_age = _age_seconds(run.get("started_at"), now)
    if started_age is not None and started_age >= 0:
        parts.append(f"started {runtime_state.format_duration(started_age)} ago")
    error = run.get("error")
    if isinstance(error, str) and error:
        parts.append(error.splitlines()[0][:120])
    return " · ".join(parts)


def _snapshot_detail(snapshot: JsonObject) -> str:
    metadata = console_ui.as_object(snapshot.get("metadata"))
    commit_sha = snapshot.get("commit_sha")
    parts = [
        console_ui.short_sha(commit_sha if isinstance(commit_sha, str) else None),
        str(snapshot.get("head_status") or "unknown"),
    ]
    mode = metadata.get("ingest_mode")
    if isinstance(mode, str):
        changed = console_ui.coerce_int(metadata.get("changed_files"))
        unchanged = console_ui.coerce_int(metadata.get("unchanged_files"))
        parts.append(f"{mode}, {changed} changed / {unchanged} unchanged")
    return " · ".join(parts)


_LATEST_SNAPSHOTS_SELECT = """
    SELECT DISTINCT ON (collection, repo)
           id, collection, repo, repo_role, branch, commit_sha,
           tree_sha, dirty, metadata, created_at
    FROM project_code_intel_snapshots
"""
_LATEST_SNAPSHOTS_SUFFIX = "ORDER BY collection, repo, created_at DESC, id DESC"


def load_status(collection: str | None, limit: int) -> tuple[list[JsonObject], list[JsonObject]]:
    with db.connect() as conn:
        runs: list[JsonObject] = []
        if mcp_db.table_regclass_exists(conn, "project_code_intel_index_runs"):
            runs = [cast("JsonObject", dict(row)) for row in load_index_runs(conn, collection=collection, limit=limit)]
        snapshots: list[JsonObject] = []
        if mcp_db.table_regclass_exists(conn, "project_code_intel_snapshots"):
            clause = "WHERE collection = %s" if collection is not None else ""
            params: list[object] = [collection] if collection is not None else []
            rows = conn.execute(
                "\n".join([_LATEST_SNAPSHOTS_SELECT, clause, _LATEST_SNAPSHOTS_SUFFIX]), params
            ).fetchall()
            snapshots = annotate_status_snapshots(rows)
    return runs, snapshots


def _running_embedding_container() -> str | None:
    """Name of a running compose embedding container, or None. Errors are swallowed."""
    try:
        result = process.run_docker(
            ["ps", "--format", "{{.Names}}"],
            process.RunOptions(capture_output=True, timeout=5),
        )
    except (OSError, process.SubprocessError):
        return None
    for name in (result.stdout or "").split():
        if any(service in name for service in DOCKER_EMBEDDING_SERVICES):
            return name
    return None


def embedding_log_hint() -> str | None:
    """Where the active embedding backend writes its log, or None when none runs."""
    if apple_embed_server_is_running():
        return str(APPLE_EMBED_SERVER_LOG_FILE)
    container = _running_embedding_container()
    if container is not None:
        return f"{process.container_engine_name()} logs {container}"
    return None


def _write_line(line: str) -> None:
    _ = sys.stdout.write(f"{line}\n")


def render_plain(runs: list[JsonObject], snapshots: list[JsonObject], now: datetime, log_hint: str | None) -> None:
    _write_line("pci status")
    if not runs:
        _write_line("runs: none recorded")
    for run in runs:
        _write_line(f"run {run.get('collection')}: {_run_detail(run, now)}")
    for snapshot in snapshots:
        _write_line(f"snapshot {snapshot.get('collection')}/{snapshot.get('repo')}: {_snapshot_detail(snapshot)}")
    if log_hint is not None:
        _write_line(f"embedding log: {log_hint}")


def render_pretty(runs: list[JsonObject], snapshots: list[JsonObject], now: datetime, log_hint: str | None) -> None:
    states = {run_state(run, now) for run in runs}
    kind: console_ui.PillKind
    if "running" in states:
        kind, label = "running", "INDEXING"
    elif "stalled" in states or "failed" in states:
        kind, label = "warn", "ATTENTION"
    else:
        kind, label = "ok", "IDLE"
    grid = console_ui.section_grid()
    if not runs:
        console_ui.add_row(grid, "runs", "none recorded")
    for run in runs:
        console_ui.add_row(grid, str(run.get("collection")), _run_detail(run, now))
    if snapshots:
        console_ui.add_row(grid, "", "")
    for snapshot in snapshots:
        console_ui.add_row(grid, str(snapshot.get("repo")), _snapshot_detail(snapshot))
    if log_hint is not None:
        console_ui.add_row(grid, "", "")
        console_ui.add_row(grid, "embedding log", log_hint)
    header = console_ui.header_row("pci status · index runs", kind, label)
    console_ui.build_console().print(console_ui.main_panel(Group(header, Text(), grid)))


def main(argv: list[str] | None = None) -> int:
    parsed = _parse_args(argv)
    try:
        runs, snapshots = load_status(parsed.collection, parsed.limit)
    except db.DatabaseConnectionError as exc:
        if parsed.json:
            _write_line(json.dumps({"ok": False, "error": str(exc)}))
        else:
            _ = sys.stderr.write(f"database unreachable: {exc}\n")
        return 1
    now = datetime.now(timezone.utc)
    for run in runs:
        run["state"] = run_state(run, now)
    log_hint = embedding_log_hint()
    if parsed.json:
        payload = {"ok": True, "runs": runs, "snapshots": snapshots, "embedding_log": log_hint}
        _write_line(json.dumps(payload, default=str, indent=2))
        return 0
    if console_ui.should_emit_pretty(sys.stdout):
        render_pretty(runs, snapshots, now, log_hint)
    else:
        render_plain(runs, snapshots, now, log_hint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
