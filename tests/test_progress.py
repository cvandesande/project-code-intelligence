from __future__ import annotations

import io
import threading
import unittest
from typing import TYPE_CHECKING
from unittest.mock import patch

from project_code_intelligence import progress
from project_code_intelligence.runtime import RuntimeMetrics

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject, JsonValue


class FakeLive:
    def __init__(self, renderable: object, **kwargs: object) -> None:
        self.renderable = renderable
        self.kwargs = kwargs
        self.started = False

    def start(self) -> None:
        self.started = True

    def update(self, renderable: object) -> None:
        self.renderable = renderable

    def stop(self) -> None:
        self.started = False


def emit_without_live(emitter: progress.RichEmitter, event: str, values: JsonObject) -> None:
    with patch.object(progress, "Live", FakeLive):
        emitter.emit(event, values)


class ProgressRenderingTests(unittest.TestCase):
    def test_multi_repo_live_title_uses_aggregate_scope(self) -> None:
        emitter = progress.RichEmitter()

        emit_without_live(
            emitter,
            "code_intel_plan",
            {"collection": "product-workspace", "repos": ["service-api", "web-ui", "shared-lib", "cli-tool"]},
        )
        emit_without_live(
            emitter,
            "code_intel_discovered",
            {
                "repo": "cli-tool",
                "commit_sha": "e7673fe0123456789",
                "mode": "full",
            },
        )

        self.assertEqual(emitter.live_title_text(), "pci-index 4 repos")
        self.assertEqual(
            emitter.repos_row_text(),
            "service-api, web-ui, shared-lib, cli-tool",
        )
        self.assertEqual(emitter.current_repo_row_text(), "cli-tool (e7673fe) · full")

    def test_single_repo_live_title_keeps_repo_name(self) -> None:
        emitter = progress.RichEmitter()

        emit_without_live(
            emitter,
            "code_intel_plan",
            {"collection": "project-code-intelligence", "repos": ["project-code-intelligence"]},
        )

        self.assertEqual(emitter.live_title_text(), "pci-index project-code-intelligence")

    def test_live_progress_captures_compact_database_target(self) -> None:
        emitter = progress.RichEmitter()

        emit_without_live(
            emitter,
            "code_intel_plan",
            {
                "collection": "project-code-intelligence",
                "repos": ["project-code-intelligence"],
                "database": "postgresql://app@db.example.invalid:30432/code-intel?sslmode=prefer",
            },
        )

        self.assertEqual(emitter.database_row_text(), "code-intel @ db.example.invalid:30432")
        self.assertEqual(
            progress.compact_database_target("postgresql://codeintel@127.0.0.1:5433/codeintel sslmode=prefer"),
            "codeintel @ 127.0.0.1:5433",
        )

    def test_plan_event_captures_authoritative_framework(self) -> None:
        emitter = progress.RichEmitter()

        emit_without_live(
            emitter,
            "code_intel_plan",
            {
                "collection": "project-code-intelligence",
                "repos": ["project-code-intelligence"],
                "embedding_endpoint": "http://127.0.0.1:18081/v1/embeddings",
                "embedding_model": "mlx-community/Qwen3-Embedding-0.6B-8bit",
                "embedding_framework": "Apple MLX",
            },
        )

        self.assertEqual(emitter.embedding_framework, "Apple MLX")
        self.assertEqual(emitter.embedding_model, "mlx-community/Qwen3-Embedding-0.6B-8bit")
        self.assertEqual(emitter.embedding_endpoint, "http://127.0.0.1:18081/v1/embeddings")
        self.assertEqual(emitter.endpoint_row_text(), "127.0.0.1:18081 · Apple MLX")

    def test_plan_event_without_framework_keeps_endpoint_for_fallback(self) -> None:
        emitter = progress.RichEmitter()

        emit_without_live(
            emitter,
            "code_intel_plan",
            {
                "collection": "project-code-intelligence",
                "repos": ["project-code-intelligence"],
                "embedding_endpoint": "http://127.0.0.1:18081/v1/embeddings",
                "embedding_model": "Qwen3-Embedding-0.6B-Q8_0.gguf",
            },
        )

        self.assertIsNone(emitter.embedding_framework)
        self.assertEqual(emitter.embedding_endpoint, "http://127.0.0.1:18081/v1/embeddings")
        self.assertEqual(
            progress.compact_endpoint_target(emitter.embedding_endpoint),
            "127.0.0.1:18081",
        )
        self.assertEqual(emitter.endpoint_row_text(), "127.0.0.1:18081")

    def test_endpoint_row_text_combines_endpoint_and_framework(self) -> None:
        self.assertEqual(
            progress.embedding_endpoint_row_text("http://127.0.0.1:18081/v1/embeddings", "AMD ROCm"),
            "127.0.0.1:18081 · AMD ROCm",
        )

    def test_endpoint_row_text_labels_embedding_rate_before_framework(self) -> None:
        self.assertEqual(
            progress.embedding_endpoint_row_text("http://127.0.0.1:18081/v1/embeddings", "AMD ROCm", 42.4),
            "127.0.0.1:18081 · 42 embeddings/s · AMD ROCm",
        )
        self.assertEqual(progress.embedding_rate_text(1.25), "1.2 embeddings/s")

    def test_compact_endpoint_target_handles_remote_host(self) -> None:
        self.assertEqual(
            progress.compact_endpoint_target("https://f5ai.pd.f5net.com/v1/embeddings"),
            "f5ai.pd.f5net.com",
        )

    def test_live_progress_row_shows_percent_without_count_detail(self) -> None:
        detail = progress.live_progress_row_text({"phase": "embedding", "phase_done": 12, "phase_total": 40})

        if detail is None:
            raise AssertionError("expected progress row detail")
        self.assertIn("30%", detail)
        self.assertNotIn("12/40 embedding records", detail)

    def test_live_embeddings_row_labels_preembedding_during_upload(self) -> None:
        counts: JsonObject = {"embedded_records": 12, "preembedded_records": 5, "embedded_records_per_second": 4.8}

        self.assertIsNone(progress.live_embeddings_row_text(counts, {"phase": "embedding"}))
        self.assertEqual(progress.live_embeddings_row_text(counts, {"phase": "db_upload"}), "5 ready")
        self.assertIsNone(progress.live_endpoint_embedding_rate(counts, {"phase": "db_upload"}))
        self.assertEqual(progress.live_endpoint_embedding_rate(counts, {"phase": "embedding"}), 4.8)

    def test_live_progress_shows_current_repo_before_discovery_finishes(self) -> None:
        emitter = progress.RichEmitter()

        emit_without_live(
            emitter,
            "code_intel_plan",
            {"collection": "product-workspace", "repos": ["service-api", "web-ui"]},
        )
        emit_without_live(
            emitter,
            "code_intel_repo_scan_started",
            {"repo": "web-ui", "mode": "incremental"},
        )

        self.assertEqual(emitter.current_repo_row_text(), "web-ui (—) · incremental")
        self.assertEqual(emitter.last_message, "Checking web-ui…")
        self.assertEqual(emitter.last_event, "code_intel_repo_scan_started")


class JsonEventLoggingTests(unittest.TestCase):
    """Cover the structured JSON-event path that callers see in CI / non-TTY runs."""

    def test_json_emitter_writes_one_event_per_line_sorted_keys(self) -> None:
        buffer = io.StringIO()
        with patch.object(progress.sys, "stderr", buffer):
            progress.JsonEmitter.emit("code_intel_parsed", {"records": 3, "files": 1})
            progress.JsonEmitter.emit("code_intel_inserted", {"records": 3})

        lines = [line for line in buffer.getvalue().splitlines() if line]
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith('{"event": "code_intel_parsed"'))
        self.assertIn('"files": 1', lines[0])
        self.assertIn('"records": 3', lines[0])
        self.assertEqual(lines[1], '{"event": "code_intel_inserted", "records": 3}')

    def test_null_emitter_swallows_events_and_close(self) -> None:
        buffer = io.StringIO()
        with patch.object(progress.sys, "stderr", buffer):
            progress.NullEmitter.emit("anything", {"k": "v"})
            progress.NullEmitter.close()
        self.assertEqual(buffer.getvalue(), "")

    def test_progress_event_routes_through_set_emitter_to_stderr(self) -> None:
        buffer = io.StringIO()
        with patch.object(progress.sys, "stderr", buffer):
            _ = progress.set_emitter("json")
            try:
                progress.progress_event("code_intel_inserted", records=5)
            finally:
                progress.close_emitter()

        line = buffer.getvalue().strip()
        self.assertEqual(line, '{"event": "code_intel_inserted", "records": 5}')

    def test_close_emitter_swaps_in_null_emitter(self) -> None:
        _ = progress.set_emitter("json")
        progress.close_emitter()
        # After close, the active emitter should be a NullEmitter that drops events silently.
        buffer = io.StringIO()
        with patch.object(progress.sys, "stderr", buffer):
            progress.progress_event("code_intel_inserted", records=1)
        self.assertEqual(buffer.getvalue(), "")
        self.assertIsInstance(progress.get_emitter(), progress.NullEmitter)

    def test_runtime_heartbeat_fires_event_until_stopped(self) -> None:
        # Stop after the first wait so the heartbeat loop emits exactly once.
        stop_event = threading.Event()
        captured: list[tuple[str, JsonObject]] = []

        def capture(event: str, **values: JsonValue) -> None:
            captured.append((event, dict(values)))

        with (
            patch.object(stop_event, "wait", side_effect=[False, True]),
            patch.object(progress, "progress_event", capture),
        ):
            progress.runtime_heartbeat(0.0, stop_event, 60, RuntimeMetrics())

        self.assertEqual(len(captured), 1)
        event, values = captured[0]
        self.assertEqual(event, "code_intel_runtime_heartbeat")
        self.assertIn("seconds", values)
        self.assertIn("duration", values)
        self.assertIn("metrics", values)


class SummaryPanelTests(unittest.TestCase):
    """End-to-end coverage of the final-summary panel through emit_summary."""

    @staticmethod
    def _summary_text(report: JsonObject, *, indent: int | None = None) -> str:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with (
            patch.object(progress.sys, "stdout", stdout_buffer),
            patch.object(progress.sys, "stderr", stderr_buffer),
            patch.object(progress, "detect_summary_mode", return_value="pretty"),
        ):
            progress.emit_summary(report, indent=indent)
        return stdout_buffer.getvalue()

    def test_pretty_summary_renders_done_for_default_report(self) -> None:
        output = self._summary_text({
            "mode": "incremental",
            "repos": ["service-api"],
            "database": "postgresql://app@db.example.invalid:5432/code-intel",
            "metrics": {
                "counts": {
                    "parsed_files": 12,
                    "changed_files": 3,
                    "unchanged_files": 9,
                    "inserted_records": 30,
                    "generated_records": 30,
                },
                "timing": {"scan_seconds": 1.2, "db_upload_seconds": 0.3},
            },
        })
        self.assertIn("pci-index", output)
        self.assertIn("DONE", output)
        self.assertIn("service-api", output)
        # Files row composed via _files_row_text
        self.assertIn("12 parsed", output)
        self.assertIn("3 changed", output)
        self.assertIn("9 unchanged", output)
        # Records row composed via _kind_row_text
        self.assertIn("30 generated", output)
        self.assertIn("30 inserted", output)

    def test_pretty_summary_marks_failed_when_exit_code_nonzero(self) -> None:
        output = self._summary_text({"exit_code": 2, "mode": "incremental"})
        self.assertIn("FAILED", output)

    def test_pretty_summary_marks_interrupted(self) -> None:
        output = self._summary_text({"interrupted": True, "mode": "incremental"})
        self.assertIn("INTERRUPTED", output)

    def test_pretty_summary_marks_dry_run(self) -> None:
        output = self._summary_text({"dry_run": True, "mode": "incremental"})
        self.assertIn("DRY RUN", output)
        # Dry run suppresses "inserted"/"reused" annotations in the records row.
        self.assertNotIn("inserted", output)

    def test_pretty_summary_marks_reset(self) -> None:
        output = self._summary_text({
            "mode": "reset",
            "collection": "demo",
            "database_dropped": True,
        })
        self.assertIn("RESET", output)
        self.assertIn("Collection", output)

    def test_pretty_summary_warns_when_parser_failures_present(self) -> None:
        output = self._summary_text({
            "mode": "incremental",
            "metrics": {"counts": {"parser_failures": 4}},
        })
        self.assertIn("DONE WITH WARNINGS", output)
        self.assertIn("Parser fails", output)

    def test_pretty_summary_renders_sarif_and_static_findings_row(self) -> None:
        output = self._summary_text({
            "mode": "incremental",
            "sarif_file_count": 2,
            "static_findings": 5,
            "static_runs": 1,
            "sarif_warnings": [{"severity": "warn"}, {"severity": "note"}, {"severity": "note"}],
            "metrics": {"counts": {}, "timing": {}},
        })
        self.assertIn("SARIF", output)
        self.assertIn("2 files", output)
        self.assertIn("5 findings", output)
        self.assertIn("1 warnings", output)
        self.assertIn("2 notes", output)
        self.assertIn("Static findings", output)

    def test_pretty_summary_renders_embedding_row_with_duration(self) -> None:
        output = self._summary_text({
            "mode": "incremental",
            "embedded_records_total": 100,
            "embedding_model": "mlx-community/Qwen3-Embedding-0.6B-8bit",
            "embedding_endpoint": "http://127.0.0.1:18081/v1/embeddings",
            "embedding_framework": "Apple MLX",
            "metrics": {
                "counts": {"embedded_records_per_second": 42.4},
                "timing": {"embedding_seconds": 2.5},
            },
        })
        self.assertIn("100 records", output)
        # Embedding duration is formatted as seconds (2.5 s) by _format_seconds
        self.assertIn("2.5 s", output)
        # Model name is shortened (strips the `mlx-community/` prefix)
        self.assertIn("Qwen3-Embedding-0.6B-8bit", output)
        self.assertNotIn("mlx-community/", output)
        # Endpoint compacted to host:port
        self.assertIn("127.0.0.1:18081", output)
        self.assertIn("Apple MLX", output)

    def test_pretty_summary_formats_duration_across_units(self) -> None:
        # Exercise the s/m/h branches of _format_seconds via the duration row.
        for duration, expected in ((0.4, "400 ms"), (5.0, "5.0 s"), (75.0, "1m 15s"), (3725.0, "1h 02m 05s")):
            output = self._summary_text({
                "mode": "incremental",
                "metrics": {"counts": {}, "timing": {"scan_seconds": duration}},
            })
            self.assertIn(expected, output)

    def test_pretty_summary_lists_deleted_snapshots_when_present(self) -> None:
        output = self._summary_text({
            "mode": "incremental",
            "snapshot_ids": [7, 9],
            "deleted_snapshots": {"snapshots": 2},
        })
        self.assertIn("Snapshot ids", output)
        self.assertIn("7, 9", output)
        self.assertIn("Deleted", output)
        self.assertIn("2 snapshot", output)

    def test_emit_summary_json_mode_writes_indented_sorted_report(self) -> None:
        buffer = io.StringIO()
        with (
            patch.object(progress.sys, "stdout", buffer),
            patch.object(progress, "detect_summary_mode", return_value="json"),
        ):
            progress.emit_summary({"mode": "incremental", "exit_code": 0}, indent=2)

        # Sorted keys means `exit_code` precedes `mode` lexicographically.
        output = buffer.getvalue()
        self.assertIn('"exit_code": 0', output)
        self.assertIn('"mode": "incremental"', output)
        self.assertLess(output.index('"exit_code"'), output.index('"mode"'))


if __name__ == "__main__":
    _ = unittest.main()
