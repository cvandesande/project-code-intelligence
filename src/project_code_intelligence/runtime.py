"""Runtime progress and worker-state helpers for ingestion."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from project_code_intelligence import config

if TYPE_CHECKING:
    import queue

    from project_code_intelligence.models import IntelRecord, JsonObject, JsonValue


def embedding_token_estimate_chars_per_token() -> float:
    return config.env_float("PROJECT_CODE_INTELLIGENCE_TOKEN_CHARS_PER_TOKEN", 4.0, minimum=1.0)


def estimate_embedding_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / embedding_token_estimate_chars_per_token()))


@dataclass
class PreEmbeddingResult:
    batch: list[IntelRecord]
    embedded: int = 0
    skipped: int = 0
    error: Exception | None = None


@dataclass
class PreEmbeddingState:
    batches: list[list[IntelRecord]]
    selected_ids: set[int]
    results: queue.Queue[PreEmbeddingResult]
    total_records: int
    thread: threading.Thread | None = None
    processed_records: int = 0
    embedded: int = 0
    skipped: int = 0
    consumed_batches: int = 0
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class RuntimeMetrics:
    lock: threading.Lock = field(default_factory=threading.Lock)
    active_phase: str | None = None
    active_phase_started: float | None = None
    scan_seconds: float = 0.0
    scan_git_seconds: float = 0.0
    scan_discovery_seconds: float = 0.0
    scan_parse_seconds: float = 0.0
    scan_sarif_seconds: float = 0.0
    db_upload_seconds: float = 0.0
    db_retries: int = 0
    embedding_seconds: float = 0.0
    embedding_db_update_seconds: float = 0.0
    embedding_preflight_seconds: float = 0.0
    discovered_files: int = 0
    changed_files: int = 0
    unchanged_files: int = 0
    reused_unchanged_files: int = 0
    parsed_files: int = 0
    generated_records: int = 0
    generated_edges: int = 0
    parser_failures: int = 0
    scan_workers: int = 1
    inserted_files: int = 0
    inserted_records: int = 0
    inserted_edges: int = 0
    inserted_parser_failures: int = 0
    static_runs: int = 0
    static_rules: int = 0
    static_findings: int = 0
    static_locations: int = 0
    static_code_flow_steps: int = 0
    copied_records: int = 0
    copied_edges: int = 0
    copied_parser_failures: int = 0
    preembedding_records_selected: int = 0
    preembedded_records: int = 0
    embedding_records_selected: int = 0
    embedding_records_attempted: int = 0
    embedded_records: int = 0
    embedding_skipped_records: int = 0
    embedding_batches: int = 0
    embedding_batch_errors: int = 0
    embedding_context_errors: int = 0
    embedding_endpoint_retries: int = 0
    embedding_retried_smaller: int = 0
    embedding_input_chars: int = 0
    embedding_input_tokens_estimated: int = 0
    embedding_input_tokens_reported: int = 0
    phase_done: int = 0
    phase_total: int = 0
    progress_weights: dict[str, float] = field(
        default_factory=lambda: {"scan": 0.4, "db_upload": 0.2, "embedding": 0.4}
    )
    completed_phases: set[str] = field(default_factory=set)

    def add(self, field_name: str, value: float | int) -> None:
        with self.lock:
            setattr(self, field_name, getattr(self, field_name) + value)

    def set(self, field_name: str, value: float | int | str | None) -> None:
        with self.lock:
            setattr(self, field_name, value)

    def set_scan_workers_max(self, value: int) -> None:
        with self.lock:
            self.scan_workers = max(self.scan_workers, value)

    def configure_progress(self, weights: dict[str, float]) -> None:
        total = sum(value for value in weights.values() if value > 0)
        if total <= 0:
            weights = {"scan": 1.0}
            total = 1.0
        with self.lock:
            self.progress_weights = {key: value / total for key, value in weights.items() if value > 0}
            self.completed_phases.clear()
            self.phase_done = 0
            self.phase_total = 0

    def begin_phase(self, name: str, total: int = 0) -> None:
        with self.lock:
            self.active_phase = name
            self.active_phase_started = time.monotonic()
            self.completed_phases.discard(name)
            self.phase_done = 0
            self.phase_total = max(0, total)

    def add_phase_total(self, value: int) -> None:
        if value <= 0:
            return
        with self.lock:
            self.phase_total += value

    def add_phase_done(self, value: int) -> None:
        if value <= 0:
            return
        with self.lock:
            self.phase_done += value

    def set_phase_progress(self, done: int, total: int | None = None) -> None:
        with self.lock:
            self.phase_done = max(0, done)
            if total is not None:
                self.phase_total = max(0, total)

    def end_phase(self, name: str, metric_field: str) -> None:
        with self.lock:
            if self.active_phase == name and self.active_phase_started is not None:
                setattr(self, metric_field, getattr(self, metric_field) + time.monotonic() - self.active_phase_started)
                if self.phase_total:
                    self.phase_done = max(self.phase_done, self.phase_total)
                self.completed_phases.add(name)
                self.active_phase = None
                self.active_phase_started = None

    def complete_phase(self, name: str) -> None:
        with self.lock:
            if self.active_phase == name:
                if self.phase_total:
                    self.phase_done = max(self.phase_done, self.phase_total)
                self.completed_phases.add(name)
                self.active_phase = None
                self.active_phase_started = None

    def add_embedding_inputs(self, texts: list[str]) -> None:
        chars = sum(len(text) for text in texts)
        estimated_tokens = sum(estimate_embedding_tokens(text) for text in texts)
        with self.lock:
            self.embedding_records_attempted += len(texts)
            self.embedding_batches += 1
            self.embedding_input_chars += chars
            self.embedding_input_tokens_estimated += estimated_tokens

    def add_embedding_usage(self, usage: object) -> None:
        if not isinstance(usage, dict):
            return
        usage_obj = cast("dict[str, object]", usage)
        value = usage_obj.get("prompt_tokens")
        if value is None:
            value = usage_obj.get("total_tokens")
        if isinstance(value, int):
            self.add("embedding_input_tokens_reported", value)

    def snapshot(self) -> JsonObject:
        with self.lock:
            active_phase = self.active_phase
            active_seconds = (
                time.monotonic() - self.active_phase_started
                if self.active_phase and self.active_phase_started is not None
                else 0.0
            )
            timing: JsonObject = {
                "scan_seconds": round(self.scan_seconds + (active_seconds if active_phase == "scan" else 0.0), 3),
                "scan_git_seconds": round(self.scan_git_seconds, 3),
                "scan_discovery_seconds": round(self.scan_discovery_seconds, 3),
                "scan_parse_seconds": round(self.scan_parse_seconds, 3),
                "scan_sarif_seconds": round(self.scan_sarif_seconds, 3),
                "db_upload_seconds": round(
                    self.db_upload_seconds + (active_seconds if active_phase == "db_upload" else 0.0), 3
                ),
                "embedding_seconds": round(self.embedding_seconds, 3),
                "embedding_db_update_seconds": round(self.embedding_db_update_seconds, 3),
                "embedding_preflight_seconds": round(self.embedding_preflight_seconds, 3),
            }
            counts: JsonObject = {
                "discovered_files": self.discovered_files,
                "changed_files": self.changed_files,
                "unchanged_files": self.unchanged_files,
                "reused_unchanged_files": self.reused_unchanged_files,
                "parsed_files": self.parsed_files,
                "generated_records": self.generated_records,
                "generated_edges": self.generated_edges,
                "parser_failures": self.parser_failures,
                "scan_workers": self.scan_workers,
                "inserted_files": self.inserted_files,
                "inserted_records": self.inserted_records,
                "inserted_edges": self.inserted_edges,
                "inserted_parser_failures": self.inserted_parser_failures,
                "static_runs": self.static_runs,
                "static_rules": self.static_rules,
                "static_findings": self.static_findings,
                "static_locations": self.static_locations,
                "static_code_flow_steps": self.static_code_flow_steps,
                "copied_records": self.copied_records,
                "copied_edges": self.copied_edges,
                "copied_parser_failures": self.copied_parser_failures,
                "db_retries": self.db_retries,
                "preembedding_records_selected": self.preembedding_records_selected,
                "preembedded_records": self.preembedded_records,
                "embedding_records_selected": self.embedding_records_selected,
                "embedding_records_attempted": self.embedding_records_attempted,
                "embedded_records": self.embedded_records,
                "embedding_skipped_records": self.embedding_skipped_records,
                "embedding_batches": self.embedding_batches,
                "embedding_batch_errors": self.embedding_batch_errors,
                "embedding_context_errors": self.embedding_context_errors,
                "embedding_endpoint_retries": self.embedding_endpoint_retries,
                "embedding_retried_smaller": self.embedding_retried_smaller,
            }
            token_use: JsonObject = {
                "embedding_input_chars": self.embedding_input_chars,
                "embedding_input_tokens_estimated": self.embedding_input_tokens_estimated,
                "embedding_input_tokens_reported": self.embedding_input_tokens_reported or None,
                "embedding_reported_tokens_available": bool(self.embedding_input_tokens_reported),
                "embedding_token_estimate_basis": f"ceil(chars/{embedding_token_estimate_chars_per_token():g})",
            }
            if active_phase:
                timing["active_phase"] = active_phase
                timing["active_phase_seconds"] = round(active_seconds, 3)
            if self.embedding_seconds > 0:
                counts["embedded_records_per_second"] = round(self.embedded_records / self.embedding_seconds, 3)
            phase_total = self.phase_total
            phase_done = self.phase_done
            phase_fraction = min(1.0, phase_done / phase_total) if phase_total > 0 else None
            completed_fraction = sum(self.progress_weights.get(phase, 0.0) for phase in self.completed_phases)
            active_fraction = (
                self.progress_weights.get(active_phase, 0.0) * phase_fraction
                if active_phase and phase_fraction is not None
                else 0.0
            )
            overall_fraction = min(1.0, completed_fraction + active_fraction)
            progress: JsonObject = {
                "phase": active_phase,
                "phase_done": phase_done,
                "phase_total": phase_total or None,
                "phase_percent": round(phase_fraction * 100, 2) if phase_fraction is not None else None,
                "overall_percent_estimated": round(overall_fraction * 100, 2),
                "overall_is_estimated": True,
            }
        return {"timing": timing, "counts": counts, "token_use": token_use, "progress": progress}


active_metrics = RuntimeMetrics()


def reset_active_metrics() -> RuntimeMetrics:
    global active_metrics  # noqa: PLW0603 - this resets per-run ingestion metrics.
    active_metrics = RuntimeMetrics()
    return active_metrics


def progress_event(event: str, **values: JsonValue) -> None:
    from project_code_intelligence import progress  # noqa: PLC0415 - lazy import avoids a cycle.

    progress.get_emitter().emit(event, dict(values))


def format_duration(seconds: float) -> str:
    whole = int(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def runtime_heartbeat_seconds() -> int:
    return config.env_int("PROJECT_CODE_INTELLIGENCE_RUNTIME_HEARTBEAT_SECONDS", 300, minimum=0)


def runtime_heartbeat(started: float, stop_event: threading.Event, interval: int, metrics: RuntimeMetrics) -> None:
    while not stop_event.wait(interval):
        elapsed = time.monotonic() - started
        progress_event(
            "code_intel_runtime_heartbeat",
            seconds=round(elapsed, 3),
            duration=format_duration(elapsed),
            metrics=metrics.snapshot(),
        )
