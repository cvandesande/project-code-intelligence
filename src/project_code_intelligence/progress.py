"""Console output for pci-index: streaming progress and summary panels."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import TYPE_CHECKING, Literal, Protocol, cast
from urllib.parse import urlsplit

from rich.console import Group
from rich.live import Live
from rich.text import Text

from project_code_intelligence import console_ui
from project_code_intelligence import runtime as runtime_state
from project_code_intelligence.console_ui import (
    add_row as _add_row,
)
from project_code_intelligence.console_ui import (
    coerce_int as _coerce_int,
)
from project_code_intelligence.console_ui import (
    format_count as _format_count,
)
from project_code_intelligence.console_ui import (
    section_grid as _section_grid,
)
from project_code_intelligence.console_ui import (
    short_sha as _short_sha,
)


def _as_object(value: object) -> JsonObject:
    return cast("JsonObject", console_ui.as_object(value))


if TYPE_CHECKING:
    from rich.console import Console, RenderableType
    from rich.panel import Panel
    from rich.table import Table

    from project_code_intelligence.models import JsonObject

OutputMode = Literal["pretty", "json"]
LIVE_REFRESH_PER_SECOND = 8
SECONDS_AS_MS_THRESHOLD = 1
MINUTES_BOUNDARY_SECONDS = 60
HOURS_BOUNDARY_MINUTES = 60
PHASE_LABELS: dict[str, str] = {
    "scan": "PARSING",
    "db_upload": "WRITING",
    "embedding": "EMBEDDING",
}


def _resolve_mode(stream: object, *, requested: OutputMode | None, env_var: str) -> OutputMode:
    env_value = os.environ.get(env_var, "").lower()
    if requested == "json" or env_value == "json":
        force: bool | None = False
    elif requested == "pretty" or env_value == "pretty":
        force = True
    else:
        force = None
    return "pretty" if console_ui.should_emit_pretty(stream, force=force) else "json"


def detect_progress_mode(*, requested: OutputMode | None = None) -> OutputMode:
    return _resolve_mode(sys.stderr, requested=requested, env_var="PROJECT_CODE_INTELLIGENCE_OUTPUT")


def detect_summary_mode(*, requested: OutputMode | None = None) -> OutputMode:
    return _resolve_mode(sys.stdout, requested=requested, env_var="PROJECT_CODE_INTELLIGENCE_OUTPUT")


def _format_seconds(value: float) -> str:
    if value < SECONDS_AS_MS_THRESHOLD:
        return f"{round(value * 1000)} ms"
    if value < MINUTES_BOUNDARY_SECONDS:
        return f"{value:.1f} s"
    minutes, seconds = divmod(int(value), MINUTES_BOUNDARY_SECONDS)
    if minutes < HOURS_BOUNDARY_MINUTES:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, HOURS_BOUNDARY_MINUTES)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


class ProgressEmitter(Protocol):
    def emit(self, event: str, values: JsonObject) -> None: ...

    def close(self) -> None: ...


class JsonEmitter:
    """Emit one JSON line per event to stderr (legacy behavior)."""

    @staticmethod
    def emit(event: str, values: JsonObject) -> None:
        _ = sys.stderr.write(json.dumps({"event": event, **values}, sort_keys=True) + "\n")
        _ = sys.stderr.flush()

    @staticmethod
    def close() -> None:
        return


class NullEmitter:
    """Swallow events after the run has wrapped up (used in pretty mode)."""

    @staticmethod
    def emit(event: str, values: JsonObject) -> None:
        _ = event
        _ = values

    @staticmethod
    def close() -> None:
        return


SHORT_EVENT_LABELS: dict[str, str] = {
    "code_intel_reset_started": "Deleting code-intelligence data for the given repo(s)…",
    "code_intel_reset_completed": "Reset complete.",
    "code_intel_incremental_unavailable": "Database unreachable; falling back to full ingest.",
    "code_intel_preembedding_disabled": "Pre-embedding disabled.",
}
INDEXING_EVENTS: frozenset[str] = frozenset({
    "code_intel_plan",
    "code_intel_discovered",
    "code_intel_parsed",
    "code_intel_inserted",
    "code_intel_preembedded",
    "code_intel_embedded",
    "code_intel_runtime_heartbeat",
    "code_intel_static_inserted",
    "code_intel_preembedding_selected",
    "code_intel_embedding_selected",
    "code_intel_sarif_discovering",
})


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast("list[object]", value)
    return [item for item in items if isinstance(item, str) and item]


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def compact_database_target(target: str) -> str:
    """Shorten a masked Postgres URL for progress panels."""
    dsn = target.split(maxsplit=1)[0]
    parts = urlsplit(dsn)
    if not parts.scheme or not parts.hostname:
        return target
    database = parts.path.lstrip("/") or "<unset>"
    port = f":{parts.port}" if parts.port is not None else ""
    return f"{database} @ {parts.hostname}{port}"


class RichEmitter:
    """Render progress as a Rich Live panel on stderr, with one summary panel per terminal."""

    def __init__(self) -> None:
        self.console = console_ui.build_console(file=sys.stderr)
        self.started_at: float = time.monotonic()
        self.repos: list[str] = []
        self.repo: str | None = None
        self.branch: str | None = None
        self.commit_sha: str | None = None
        self.database: str | None = None
        self.dirty: bool = False
        self.mode: str | None = None
        self.last_event: str = "starting"
        self.last_message: str | None = None
        self.live: Live | None = None

    def emit(self, event: str, values: JsonObject) -> None:
        if event == "code_intel_runtime":
            self.close()
            return
        self.last_event = event
        self._capture_identity(event, values)
        self._capture_message(event, values)
        label = SHORT_EVENT_LABELS.get(event)
        if event not in INDEXING_EVENTS and label is not None:
            self.console.print(Text(label, style="dim"))
            return
        if self.live is None:
            self.live = Live(
                self._render(),
                console=self.console,
                refresh_per_second=LIVE_REFRESH_PER_SECOND,
                transient=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self.live.start()
        else:
            self.live.update(self._render())

    def close(self) -> None:
        if self.live is None:
            return
        self.live.stop()
        self.live = None

    def _capture_identity(self, event: str, values: JsonObject) -> None:
        if event == "code_intel_plan":
            repos = _string_list(values.get("repos"))
            if repos:
                self.repos = repos
            database = values.get("database")
            if isinstance(database, str) and database:
                self.database = database
            return
        if event == "code_intel_discovered":
            repo_value = values.get("repo")
            if isinstance(repo_value, str):
                self.repo = repo_value
                _append_unique(self.repos, repo_value)
            commit = values.get("commit_sha")
            if isinstance(commit, str):
                self.commit_sha = commit
            tree = values.get("tree_sha")
            if isinstance(tree, str) and ":dirty:" in tree:
                self.dirty = True
            mode = values.get("mode")
            if isinstance(mode, str):
                self.mode = mode

    def _capture_message(self, event: str, values: JsonObject) -> None:
        if event == "code_intel_sarif_discovering":
            self.last_message = "Looking for SARIF reports…"
        elif event in {"code_intel_db_retry", "code_intel_embedding_endpoint_retry"}:
            reason = values.get("reason") or values.get("error") or "transient failure"
            attempt = values.get("attempt")
            attempts = values.get("attempts")
            attempt_text = f" ({attempt}/{attempts})" if attempt and attempts else ""
            self.last_message = f"{event.replace('code_intel_', '').replace('_', ' ')}{attempt_text}: {reason}"

    @staticmethod
    def _phase_label() -> tuple[console_ui.PillKind, str]:
        metrics = runtime_state.active_metrics.snapshot()
        progress = metrics.get("progress", {})
        if isinstance(progress, dict):
            phase = progress.get("phase")
            if isinstance(phase, str):
                return "running", PHASE_LABELS.get(phase, phase.upper())
        return "running", "STARTING"

    def live_title_text(self) -> str:
        if len(self.repos) > 1:
            return f"pci-index {len(self.repos)} repos"
        if len(self.repos) == 1:
            return f"pci-index {self.repos[0]}"
        return f"pci-index {self.repo or '...'}"

    def _live_header(self) -> Table:
        status, label = self._phase_label()
        return console_ui.header_row(self.live_title_text(), status, label)

    def repos_row_text(self) -> str:
        if not self.repos:
            return "discovering…"
        return ", ".join(self.repos)

    def current_repo_row_text(self) -> str:
        if not self.repo:
            return "discovering…"
        commit = _short_sha(self.commit_sha)
        dirty_suffix = " · dirty" if self.dirty else ""
        mode_suffix = f" · {self.mode}" if self.mode else ""
        return f"{self.repo} ({commit}{dirty_suffix}){mode_suffix}"

    def database_row_text(self) -> str:
        if not self.database:
            return "resolving…"
        return compact_database_target(self.database)

    def _live_rows(self, counts: JsonObject, progress: JsonObject, timing: JsonObject) -> Table:
        rows = _section_grid()
        if len(self.repos) > 1:
            _add_row(rows, "Repositories", self.repos_row_text())
            if progress.get("phase") == "scan" and self.repo:
                _add_row(rows, "Current", self.current_repo_row_text())
        else:
            _add_row(rows, "Repository", self.current_repo_row_text())
        _add_row(rows, "Database", self.database_row_text())
        _add_live_progress_row(rows, progress)
        _add_live_files_row(rows, counts)
        _add_live_records_row(rows, counts)
        _add_live_edges_row(rows, counts)
        _add_live_embeddings_row(rows, counts)
        _add_row(rows, "Elapsed", _format_seconds(time.monotonic() - self.started_at))
        _add_live_eta_row(rows, progress, counts, timing)
        return rows

    def _render(self) -> Panel:
        metrics = runtime_state.active_metrics.snapshot()
        counts = _as_object(metrics.get("counts"))
        progress = _as_object(metrics.get("progress"))
        timing = _as_object(metrics.get("timing"))
        body: list[RenderableType] = [self._live_header(), Text(), self._live_rows(counts, progress, timing)]
        if self.last_message:
            body.extend((Text(), Text(self.last_message, style="dim")))
        return console_ui.main_panel(Group(*body))


def _estimate_remaining_seconds(progress: JsonObject, counts: JsonObject, timing: JsonObject) -> float | None:
    phase = progress.get("phase")
    if phase == "embedding":
        rate = counts.get("embedded_records_per_second")
        selected = _coerce_int(counts.get("embedding_records_selected"))
        done = _coerce_int(counts.get("embedded_records"))
        if isinstance(rate, (int, float)) and rate > 0 and selected > done:
            return (selected - done) / float(rate)
    if phase == "scan":
        discovered = _coerce_int(counts.get("discovered_files"))
        parsed = _coerce_int(counts.get("parsed_files"))
        scan_seconds = timing.get("scan_seconds")
        if isinstance(scan_seconds, (int, float)) and scan_seconds > 0 and parsed > 0 and discovered > parsed:
            rate = parsed / float(scan_seconds)
            return (discovered - parsed) / rate if rate > 0 else None
    return None


def _add_live_eta_row(rows: Table, progress: JsonObject, counts: JsonObject, timing: JsonObject) -> None:
    remaining = _estimate_remaining_seconds(progress, counts, timing)
    if remaining is None or remaining <= 0:
        return
    _add_row(rows, "ETA", f"~ {_format_seconds(remaining)} remaining")


def _add_live_progress_row(rows: Table, progress: JsonObject) -> None:
    phase_done = _coerce_int(progress.get("phase_done"))
    phase_total = _coerce_int(progress.get("phase_total"))
    if phase_total:
        bar = _bar(phase_done, phase_total)
        _add_row(rows, "Progress", f"{bar} {phase_done:,}/{phase_total:,}")
        return
    overall = progress.get("overall_percent_estimated")
    if isinstance(overall, (int, float)):
        _add_row(rows, "Progress", f"~{overall:.0f}%")


def _add_live_files_row(rows: Table, counts: JsonObject) -> None:
    discovered = _coerce_int(counts.get("discovered_files"))
    if not discovered:
        return
    parsed = _coerce_int(counts.get("parsed_files")) or discovered
    _add_row(rows, "Files", f"{_format_count(parsed)} parsed of {_format_count(discovered)}")


def _add_live_records_row(rows: Table, counts: JsonObject) -> None:
    records = _coerce_int(counts.get("generated_records"))
    if not records:
        return
    inserted = _coerce_int(counts.get("inserted_records"))
    detail = f"{_format_count(records)} generated"
    if inserted:
        detail += f" · {_format_count(inserted)} inserted"
    _add_row(rows, "Records", detail)


def _add_live_edges_row(rows: Table, counts: JsonObject) -> None:
    edges = _coerce_int(counts.get("generated_edges"))
    if not edges:
        return
    _add_row(rows, "Edges", f"{_format_count(edges)} generated")


def _add_live_embeddings_row(rows: Table, counts: JsonObject) -> None:
    embedded = _coerce_int(counts.get("embedded_records")) + _coerce_int(counts.get("preembedded_records"))
    if not embedded:
        return
    rate = counts.get("embedded_records_per_second")
    rate_text = f" · {rate:.0f}/s" if isinstance(rate, (int, float)) and rate else ""
    _add_row(rows, "Embeddings", f"{_format_count(embedded)}{rate_text}")


def _resolve_duration(report: JsonObject, timing: JsonObject) -> float | None:
    direct = report.get("seconds")
    if isinstance(direct, (int, float)) and direct:
        return float(direct)
    total = 0.0
    for key in ("scan_seconds", "db_upload_seconds", "embedding_seconds"):
        value = timing.get(key)
        if isinstance(value, (int, float)):
            total += float(value)
    return total if total > 0 else None


def _bar(done: int, total: int, *, width: int = 24) -> str:
    if total <= 0:
        return " " * width
    filled = max(0, min(width, round(width * done / total)))
    return "█" * filled + "░" * (width - filled)


# === Module-level emitter wiring ===========================================

_emitter: ProgressEmitter | None = None


def set_emitter(mode: OutputMode) -> ProgressEmitter:
    global _emitter  # noqa: PLW0603 - module-level singleton for the process.
    _emitter = RichEmitter() if mode == "pretty" else JsonEmitter()
    return _emitter


def get_emitter() -> ProgressEmitter:
    global _emitter  # noqa: PLW0603 - module-level singleton for the process.
    if _emitter is None:
        _emitter = JsonEmitter() if detect_progress_mode() == "json" else RichEmitter()
    return _emitter


def close_emitter() -> None:
    global _emitter  # noqa: PLW0603 - module-level singleton for the process.
    if _emitter is not None:
        _emitter.close()
    _emitter = NullEmitter()


# === Final summary panel for stdout ========================================


def _summary_status(report: JsonObject) -> tuple[console_ui.PillKind, str]:
    interrupted = report.get("interrupted")
    exit_code = report.get("exit_code")
    if interrupted:
        return "fail", "INTERRUPTED"
    if isinstance(exit_code, int) and exit_code != 0:
        return "fail", "FAILED"
    counts_obj = _as_object(_as_object(report.get("metrics")).get("counts"))
    parser_failures = _coerce_int(counts_obj.get("parser_failures")) or _coerce_int(report.get("parser_failures"))
    if parser_failures:
        return "warn", "DONE WITH WARNINGS"
    if report.get("dry_run"):
        return "ok", "DRY RUN"
    if report.get("mode") == "reset":
        return "ok", "RESET"
    return "ok", "DONE"


def _summary_header(report: JsonObject) -> Table:
    status, label = _summary_status(report)
    repos_value = report.get("repos")
    if isinstance(repos_value, list) and repos_value:
        repos_text = ", ".join(str(item) for item in repos_value)
        title = f"pci-index {repos_text}"
    elif isinstance(repos_value, str):
        title = f"pci-index {repos_value}"
    else:
        title = "pci-index"
    return console_ui.header_row(title, status, label)


def _add_identity_rows(rows: Table, report: JsonObject) -> None:
    mode = report.get("mode")
    if isinstance(mode, str):
        _add_row(rows, "Mode", mode)
    profile = report.get("profile")
    if isinstance(profile, str):
        _add_row(rows, "Profile", profile)
    database = report.get("database")
    if isinstance(database, str):
        _add_row(rows, "Database", compact_database_target(database))
    collection = report.get("collection")
    if isinstance(collection, str) and mode == "reset":
        _add_row(rows, "Collection", collection)
    snapshot_ids = report.get("snapshot_ids")
    if isinstance(snapshot_ids, list) and snapshot_ids:
        snapshot_text = ", ".join(str(item) for item in snapshot_ids)
        _add_row(rows, "Snapshot ids", snapshot_text)
    deleted = report.get("deleted_snapshots")
    if isinstance(deleted, dict) and deleted:
        total = sum(int(v) for v in deleted.values() if isinstance(v, (int, float)))
        if total:
            _add_row(rows, "Deleted", f"{_format_count(total)} snapshot(s)")
        else:
            _add_row(rows, "Deleted", "no snapshots matched")


def _files_row_text(counts: JsonObject) -> str | None:
    parsed = _coerce_int(counts.get("parsed_files"))
    discovered = _coerce_int(counts.get("discovered_files"))
    if not (discovered or parsed):
        return None
    parts = [f"{_format_count(parsed or discovered)} parsed"]
    changed = _coerce_int(counts.get("changed_files"))
    unchanged = _coerce_int(counts.get("unchanged_files"))
    if changed and changed != parsed:
        parts.append(f"{_format_count(changed)} changed")
    if unchanged:
        parts.append(f"{_format_count(unchanged)} unchanged")
    return " · ".join(parts)


def _kind_row_text(*, generated: int, inserted: int, copied: int, dry_run: bool) -> str | None:
    if not (generated or inserted or copied):
        return None
    parts: list[str] = []
    if generated:
        parts.append(f"{_format_count(generated)} generated")
    if not dry_run and inserted:
        parts.append(f"{_format_count(inserted)} inserted")
    if not dry_run and copied:
        parts.append(f"{_format_count(copied)} reused")
    return " · ".join(parts)


def _embedding_row_text(report: JsonObject, counts: JsonObject, timing: JsonObject) -> str | None:
    embedded = _coerce_int(report.get("embedded_records_total")) or _coerce_int(counts.get("embedded_records"))
    if not embedded:
        return None
    seconds = timing.get("embedding_seconds")
    suffix = f" · {_format_seconds(float(seconds))}" if isinstance(seconds, (int, float)) and seconds else ""
    return f"{_format_count(embedded)} records{suffix}"


def _sarif_warning_counts(report: JsonObject) -> tuple[int, int]:
    warnings = report.get("sarif_warnings")
    if not isinstance(warnings, list):
        return 0, 0
    warn_count = 0
    note_count = 0
    for item in warnings:
        if isinstance(item, dict) and item.get("severity") == "warn":
            warn_count += 1
        else:
            note_count += 1
    return warn_count, note_count


def _sarif_row_text(report: JsonObject) -> str | None:
    files = _coerce_int(report.get("sarif_file_count"))
    findings = _coerce_int(report.get("static_findings"))
    warnings, notes = _sarif_warning_counts(report)
    if not (files or findings or warnings or notes):
        return None
    parts: list[str] = []
    if files:
        parts.append(f"{_format_count(files)} files")
    if findings:
        parts.append(f"{_format_count(findings)} findings")
    if warnings:
        parts.append(f"{_format_count(warnings)} warnings")
    if notes:
        parts.append(f"{_format_count(notes)} notes")
    return " · ".join(parts)


def _add_count_rows(rows: Table, report: JsonObject, *, counts: JsonObject, timing: JsonObject) -> None:
    dry_run = bool(report.get("dry_run"))

    files_text = _files_row_text(counts)
    if files_text:
        _add_row(rows, "Files", files_text)

    records_text = _kind_row_text(
        generated=_coerce_int(counts.get("generated_records")),
        inserted=_coerce_int(counts.get("inserted_records")),
        copied=_coerce_int(report.get("copied_records")) or _coerce_int(counts.get("copied_records")),
        dry_run=dry_run,
    )
    if records_text:
        _add_row(rows, "Records", records_text)

    edges_text = _kind_row_text(
        generated=_coerce_int(counts.get("generated_edges")),
        inserted=_coerce_int(counts.get("inserted_edges")),
        copied=_coerce_int(report.get("copied_edges")) or _coerce_int(counts.get("copied_edges")),
        dry_run=dry_run,
    )
    if edges_text:
        _add_row(rows, "Edges", edges_text)

    embedding_text = _embedding_row_text(report, counts, timing)
    if embedding_text:
        _add_row(rows, "Embeddings", embedding_text)


def _add_outcome_rows(rows: Table, report: JsonObject, *, counts: JsonObject, timing: JsonObject) -> None:
    parser_failures = _coerce_int(counts.get("parser_failures"))
    if parser_failures:
        _add_row(rows, "Parser fails", _format_count(parser_failures))
    sarif_text = _sarif_row_text(report)
    if sarif_text:
        _add_row(rows, "SARIF", sarif_text)
    static_findings = _coerce_int(report.get("static_findings"))
    if static_findings:
        runs = _coerce_int(report.get("static_runs"))
        _add_row(rows, "Static findings", f"{_format_count(static_findings)} in {_format_count(runs)} run(s)")
    duration = _resolve_duration(report, timing)
    if duration is not None:
        _add_row(rows, "Duration", _format_seconds(duration))


def render_summary_panel(report: JsonObject, *, console: Console | None = None) -> None:
    console = console or console_ui.build_console()
    metrics = _as_object(report.get("metrics"))
    counts = _as_object(metrics.get("counts"))
    timing = _as_object(metrics.get("timing"))

    rows = _section_grid()
    _add_identity_rows(rows, report)
    _add_count_rows(rows, report, counts=counts, timing=timing)
    _add_outcome_rows(rows, report, counts=counts, timing=timing)

    console.print(console_ui.main_panel(Group(_summary_header(report), Text(), rows)))


def emit_summary(report: JsonObject, *, mode: OutputMode | None = None, indent: int | None = None) -> None:
    resolved = detect_summary_mode(requested=mode)
    if resolved == "json":
        _ = sys.stdout.write(json.dumps(report, indent=indent, sort_keys=True) + "\n")
        return
    close_emitter()
    render_summary_panel(report)
