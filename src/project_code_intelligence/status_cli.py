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
from typing import TYPE_CHECKING, NamedTuple, cast

from rich.console import Group
from rich.table import Table
from rich.text import Text

from project_code_intelligence import console_ui, db, process
from project_code_intelligence import runtime as runtime_state
from project_code_intelligence.doctor.cli import EMBEDDING_CONTAINER_NAMES
from project_code_intelligence.embedding.apple_embed_server import (
    APPLE_EMBED_SERVER_LOG_FILE,
    apple_embed_server_is_running,
)
from project_code_intelligence.mcp import db as mcp_db
from project_code_intelligence.mcp.status import annotate_status_snapshots
from project_code_intelligence.progress import PHASE_LABELS, estimate_remaining_seconds
from project_code_intelligence.storage import load_index_runs, load_latest_snapshots

if TYPE_CHECKING:
    from rich.console import RenderableType

    from project_code_intelligence.models import JsonObject

# A run counts as live while its heartbeat is younger than this many ledger
# intervals; older with no finished_at means the process died or stalled.
_STALE_INTERVALS = 3
_MIN_STALE_SECONDS = 60


class StatusNamespace(argparse.Namespace):
    collection: str | None = None
    limit: int = 10
    json: bool = False
    all: bool = False


def _parse_args(argv: list[str] | None) -> StatusNamespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--collection", default=None, help="Only show this collection.")
    _ = parser.add_argument("--limit", type=int, default=10, help="Maximum run rows to load (default: 10).")
    _ = parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    _ = parser.add_argument(
        "--all", action="store_true", help="Show full run history and every snapshot instead of the summary view."
    )
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


def _run_repos(run: JsonObject) -> str:
    repos = run.get("repos")
    if not isinstance(repos, list):
        return ""
    # "." is the workspace root repo; the literal dot reads as noise.
    return ", ".join("root" if str(r) == "." else str(r) for r in repos)


def _run_mode(run: JsonObject) -> str:
    modes = console_ui.as_object(run.get("repo_modes"))
    return ",".join(sorted({str(v) for v in modes.values()}))


def _run_timing(run: JsonObject, now: datetime) -> str:
    started_age = _age_seconds(run.get("started_at"), now)
    finished_age = _age_seconds(run.get("finished_at"), now)
    if finished_age is not None and started_age is not None and started_age >= finished_age >= 0:
        # A finished run shows how long it took; "started X ago" alone reads
        # as a run that is still going after X.
        took = runtime_state.format_duration(started_age - finished_age)
        return f"took {took} · finished {runtime_state.format_duration(finished_age)} ago"
    if started_age is not None and started_age >= 0:
        return f"started {runtime_state.format_duration(started_age)} ago"
    return ""


def _run_activity(run: JsonObject, now: datetime) -> str:
    """Progress (while live), timing, and first error line, dot-joined."""
    state = run_state(run, now)
    parts: list[str] = []
    progress_part = _progress_detail(run)
    if progress_part and state in {"running", "stalled"}:
        parts.append(progress_part)
    timing = _run_timing(run, now)
    if timing:
        parts.append(timing)
    error = run.get("error")
    if isinstance(error, str) and error:
        parts.append(error.splitlines()[0][:120])
    return " · ".join(parts)


def run_detail(run: JsonObject, now: datetime, *, include_state: bool = True) -> str:
    parts = [run_state(run, now)] if include_state else []
    parts.extend(part for part in (_run_repos(run), _run_mode(run), _run_activity(run, now)) if part)
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


def load_status(collection: str | None, limit: int) -> tuple[list[JsonObject], list[JsonObject]]:
    with db.connect() as conn:
        runs: list[JsonObject] = []
        if mcp_db.table_regclass_exists(conn, "project_code_intel_index_runs"):
            runs = [cast("JsonObject", dict(row)) for row in load_index_runs(conn, collection=collection, limit=limit)]
        snapshots: list[JsonObject] = []
        if mcp_db.table_regclass_exists(conn, "project_code_intel_snapshots"):
            snapshots = annotate_status_snapshots(load_latest_snapshots(conn, collection=collection))
    return runs, snapshots


def _running_embedding_container() -> str | None:
    """Name of a running Quadlet-managed embedding container, or None. Errors are swallowed."""
    try:
        result = process.run_podman(
            ["ps", "--format", "{{.Names}}"],
            process.RunOptions(capture_output=True, timeout=5),
        )
    except (OSError, process.SubprocessError):
        return None
    for name in (result.stdout or "").split():
        if any(service in name for service in EMBEDDING_CONTAINER_NAMES):
            return name
    return None


def other_host_index_runs() -> list[str]:
    """Collections other `pci index` processes on this host are indexing now.

    The status view reads one collection's database, so an index run for a
    different workspace is invisible there while it still loads the shared
    embedding backend. A process scan closes that gap. Errors are swallowed.
    """
    try:
        result = process.run(
            ["ps", "-eo", "args"],
            process.RunOptions(capture_output=True, timeout=5),
        )
    except (OSError, process.SubprocessError):
        return []
    collections: list[str] = []
    for line in (result.stdout or "").splitlines():
        if " index " not in line or "pci" not in line:
            continue
        parts = line.split()
        if "--collection" in parts:
            value = parts[parts.index("--collection") + 1 :][:1]
            collections.extend(value)
        else:
            collections.append("(unnamed)")
    return sorted(set(collections))


def embedding_log_hint() -> str | None:
    """Where the active embedding backend writes its log, or None when none runs."""
    if apple_embed_server_is_running():
        return str(APPLE_EMBED_SERVER_LOG_FILE)
    container = _running_embedding_container()
    if container is not None:
        return f"podman logs {container}"
    return None


def _write_line(line: str) -> None:
    _ = sys.stdout.write(f"{line}\n")


def classify_other_runs(ps_collections: list[str], runs: list[JsonObject]) -> list[tuple[str, str]]:
    """Label ps-detected `pci index` processes that have no active ledger row.

    A collection this database's ledger knows is not "other workspace"; the
    process just has not written (or refreshed) its row yet. Only collections
    the ledger has never seen live in another workspace's database.
    """
    active = {str(run.get("collection")) for run in runs if run.get("state") in {"running", "stalled"}}
    known = {str(run.get("collection")) for run in runs}
    labelled: list[tuple[str, str]] = []
    for coll in ps_collections:
        if coll in active:
            continue
        note = "starting (no ledger row yet)" if coll in known else "other workspace"
        labelled.append((coll, note))
    return labelled


def render_plain(
    runs: list[JsonObject],
    snapshots: list[JsonObject],
    now: datetime,
    log_hint: str | None,
    other_runs: list[tuple[str, str]] | None = None,
) -> None:
    _write_line("pci status")
    for coll, note in other_runs or []:
        _write_line(f"run {coll}: running · {note}")
    if not runs:
        _write_line("runs: none recorded")
    for run in runs:
        _write_line(f"run {run.get('collection')}: {run_detail(run, now)}")
    for snapshot in snapshots:
        branch = snapshot.get("branch")
        label = (
            f"{snapshot.get('collection')}/{snapshot.get('repo')}@{branch}"
            if branch
            else (f"{snapshot.get('collection')}/{snapshot.get('repo')}")
        )
        _write_line(f"snapshot {label}: {_snapshot_detail(snapshot)}")
    if log_hint is not None:
        _write_line(f"embedding log: {log_hint}")


def status_pill(
    runs: list[JsonObject], other_runs: list[tuple[str, str]], now: datetime
) -> tuple[console_ui.PillKind, str]:
    """Header pill from the actual run state, like the live `pci index` pill.

    A running run shows its ledger phase through the same PHASE_LABELS map the
    indexer's own header uses. An other-workspace `pci index` process has its
    ledger in that workspace's database, so only its running state is known.
    """
    states = {run_state(run, now) for run in runs}
    if "running" in states:
        phase = next(
            (run.get("phase") for run in runs if run_state(run, now) == "running"),
            None,
        )
        if isinstance(phase, str) and phase:
            return "running", PHASE_LABELS.get(phase, phase.upper())
        return "running", "INDEXING"
    if "stalled" in states or "failed" in states:
        return "warn", "ATTENTION"
    if other_runs:
        return "running", "INDEXING"
    return "ok", "IDLE"


# Row-label colour per run/snapshot state, so the eye can scan the first column.
_STATE_STYLES: dict[str, str] = {
    "running": "bold cyan",
    "ok": "green",
    "failed": "bold red",
    "stalled": "bold yellow",
    "interrupted": "yellow",
    "current": "green",
    "stale": "yellow",
}


def select_runs(runs: list[JsonObject], now: datetime) -> tuple[list[JsonObject], int]:
    """Runs worth showing by default, plus the count of hidden ones.

    Keeps every run that needs attention (running, stalled, failed,
    interrupted) and the newest successful run as proof the background
    reindex works; older successes are history, not status.
    """
    shown: list[JsonObject] = []
    hidden = 0
    ok_shown = False
    for run in runs:  # newest first (load_index_runs orders by started_at DESC)
        if run_state(run, now) != "ok":
            shown.append(run)
        elif not ok_shown:
            shown.append(run)
            ok_shown = True
        else:
            hidden += 1
    return shown, hidden


def select_snapshots(snapshots: list[JsonObject]) -> tuple[list[JsonObject], int]:
    """Snapshots worth showing by default, plus the count of hidden ones.

    A repo with a current snapshot is fresh; its stale rows are old branches
    nobody reindexed since and hide behind a count. A repo with no current
    snapshot shows all its rows, because one of them needs a reindex.
    """
    fresh_repos = {str(snap.get("repo")) for snap in snapshots if snap.get("head_status") == "current"}
    shown = [s for s in snapshots if s.get("head_status") == "current" or str(s.get("repo")) not in fresh_repos]
    return shown, len(snapshots) - len(shown)


def _section_table(*columns: str) -> Table:
    table = Table(box=None, pad_edge=False, show_header=True, header_style="dim", padding=(0, 2, 0, 0))
    for name in columns:
        table.add_column(name, overflow="fold")
    return table


def _snapshots_table(snapshots: list[JsonObject]) -> Table:
    table = _section_table("indexed", "commit", "freshness", "last ingest")
    for snapshot in snapshots:
        branch = snapshot.get("branch")
        repo = str(snapshot.get("repo"))
        if repo == ".":
            repo = "root"
        repo_label = f"{repo}@{branch}" if branch else repo
        metadata = console_ui.as_object(snapshot.get("metadata"))
        commit_sha = snapshot.get("commit_sha")
        freshness = str(snapshot.get("head_status") or "unknown")
        ingest = ""
        mode = metadata.get("ingest_mode")
        if isinstance(mode, str):
            changed = console_ui.coerce_int(metadata.get("changed_files"))
            unchanged = console_ui.coerce_int(metadata.get("unchanged_files"))
            ingest = f"{mode} · {changed} changed / {unchanged} unchanged"
        table.add_row(
            Text(repo_label, style="bold"),
            Text(console_ui.short_sha(commit_sha if isinstance(commit_sha, str) else None)),
            Text(freshness, style=_STATE_STYLES.get(freshness, "")),
            Text(ingest),
        )
    return table


class StatusView(NamedTuple):
    """What the pretty renderer shows: pre-selected rows plus hidden counts."""

    runs: list[JsonObject]
    hidden_runs: int
    snapshots: list[JsonObject]
    hidden_snapshots: int
    other_runs: list[tuple[str, str]]

    @classmethod
    def build(
        cls,
        runs: list[JsonObject],
        snapshots: list[JsonObject],
        other_runs: list[tuple[str, str]],
        now: datetime,
        *,
        show_all: bool,
    ) -> StatusView:
        shown_runs, hidden_runs = (runs, 0) if show_all else select_runs(runs, now)
        shown_snapshots, hidden_snapshots = (snapshots, 0) if show_all else select_snapshots(snapshots)
        return cls(shown_runs, hidden_runs, shown_snapshots, hidden_snapshots, other_runs)


def render_pretty(view: StatusView, all_runs: list[JsonObject], now: datetime, log_hint: str | None) -> None:
    # The pill reflects every loaded run, not just the ones shown.
    other_runs = view.other_runs
    kind, label = status_pill(all_runs, other_runs, now)
    collections = sorted({str(run.get("collection")) for run in all_runs})
    title = "pci status" + (f" · {collections[0]}" if len(collections) == 1 else "")
    body: list[RenderableType] = []

    if view.snapshots:
        body.append(_snapshots_table(view.snapshots))
        if view.hidden_snapshots:
            noun = "branch" if view.hidden_snapshots == 1 else "branches"
            body.append(
                Text(f"+ {view.hidden_snapshots} stale {noun} not checked out here · pci status --all", style="dim")
            )
        body.append(Text())

    shown_runs, hidden_runs = view.runs, view.hidden_runs
    runs_table = _section_table("index runs", "repos", "mode", "activity")
    for coll, note in other_runs or []:
        runs_table.add_row(Text("running", style=_STATE_STYLES["running"]), Text(coll), Text(""), Text(note))
    for run in shown_runs:
        state = run_state(run, now)
        repos = _run_repos(run)
        if len(collections) > 1:
            repos = f"{run.get('collection')} · {repos}"
        runs_table.add_row(
            Text(state, style=_STATE_STYLES.get(state, "bold")),
            Text(repos),
            Text(_run_mode(run)),
            Text(_run_activity(run, now)),
        )
    body.append(runs_table if shown_runs or other_runs else Text("no index runs recorded", style="dim"))
    if hidden_runs:
        body.append(Text(f"+ {hidden_runs} older successful runs · pci status --all", style="dim"))

    if log_hint is not None:
        body.extend((Text(), Text.assemble(("embedding log  ", "dim"), log_hint)))

    header = console_ui.header_row(title, kind, label)
    console_ui.build_console().print(console_ui.main_panel(Group(header, Text(), *body)))


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
    other_runs = classify_other_runs(other_host_index_runs(), runs)
    if parsed.json:
        payload = {
            "ok": True,
            "runs": runs,
            "snapshots": snapshots,
            "embedding_log": log_hint,
            "other_host_index_runs": [coll for coll, _ in other_runs],
        }
        _write_line(json.dumps(payload, default=str, indent=2))
        return 0
    if console_ui.should_emit_pretty(sys.stdout):
        view = StatusView.build(runs, snapshots, other_runs, now, show_all=parsed.all)
        render_pretty(view, runs, now, log_hint)
    else:
        render_plain(runs, snapshots, now, log_hint, other_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
