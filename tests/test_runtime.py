"""Unit tests for `project_code_intelligence.runtime`.

Covers RuntimeMetrics arithmetic, phase tracking, the module-level
`active_metrics` `__getattr__` indirection, format_duration boundaries,
and estimate_embedding_tokens edge cases.
"""

from __future__ import annotations

import os
import unittest
from typing import cast
from unittest.mock import patch

from project_code_intelligence import runtime
from project_code_intelligence.runtime import (
    RuntimeMetrics,
    estimate_embedding_tokens,
    format_duration,
    reset_active_metrics,
    runtime_heartbeat_seconds,
)
from project_code_intelligence.runtime import __getattr__ as runtime_module_getattr


class EstimateEmbeddingTokensTests(unittest.TestCase):
    def test_empty_string_returns_zero(self) -> None:
        self.assertEqual(estimate_embedding_tokens(""), 0)

    def test_short_string_rounds_up_to_one_token(self) -> None:
        # 1 char / 4 chars-per-token = 0.25 → ceil → 1
        self.assertEqual(estimate_embedding_tokens("a"), 1)

    def test_boundary_is_ceil_not_floor(self) -> None:
        # 5 chars / 4 = 1.25 → ceil → 2
        self.assertEqual(estimate_embedding_tokens("abcde"), 2)

    def test_exact_multiple_of_chars_per_token(self) -> None:
        # 8 chars / 4 = exactly 2
        self.assertEqual(estimate_embedding_tokens("abcdabcd"), 2)

    def test_chars_per_token_is_configurable(self) -> None:
        with patch.dict(os.environ, {"PCI_TOKEN_CHARS_PER_TOKEN": str(2)}, clear=False):
            # 4 chars / 2 chars-per-token = 2
            self.assertEqual(estimate_embedding_tokens("abcd"), 2)


class FormatDurationTests(unittest.TestCase):
    def test_zero_seconds_renders_as_zero(self) -> None:
        self.assertEqual(format_duration(0), "0s")

    def test_sub_minute_renders_as_seconds_only(self) -> None:
        self.assertEqual(format_duration(42), "42s")
        # Fractional seconds are truncated (int conversion).
        self.assertEqual(format_duration(42.9), "42s")

    def test_exactly_one_minute(self) -> None:
        self.assertEqual(format_duration(60), "1m00s")

    def test_minutes_with_zero_pad(self) -> None:
        self.assertEqual(format_duration(75), "1m15s")
        self.assertEqual(format_duration(605), "10m05s")

    def test_exactly_one_hour(self) -> None:
        self.assertEqual(format_duration(3600), "1h00m00s")

    def test_multi_hour_uses_zero_padded_minutes_and_seconds(self) -> None:
        # 2h 03m 04s
        self.assertEqual(format_duration(2 * 3600 + 3 * 60 + 4), "2h03m04s")


class RuntimeMetricsTests(unittest.TestCase):
    def test_add_increments_int_field(self) -> None:
        metrics = RuntimeMetrics()
        metrics.add("inserted_records", 5)
        metrics.add("inserted_records", 3)
        self.assertEqual(metrics.inserted_records, 8)

    def test_add_accepts_float_for_seconds_fields(self) -> None:
        metrics = RuntimeMetrics()
        metrics.add("scan_seconds", 1.25)
        metrics.add("scan_seconds", 0.75)
        self.assertEqual(metrics.scan_seconds, 2.0)

    def test_set_overrides_value(self) -> None:
        metrics = RuntimeMetrics()
        metrics.set("scan_workers", 4)
        self.assertEqual(metrics.scan_workers, 4)
        metrics.set("scan_workers", 1)
        self.assertEqual(metrics.scan_workers, 1)

    def test_set_db_write_op_supports_none_and_string(self) -> None:
        metrics = RuntimeMetrics()
        metrics.set("db_write_op", "COPY records")
        self.assertEqual(metrics.db_write_op, "COPY records")
        metrics.set("db_write_op", None)
        self.assertIsNone(metrics.db_write_op)

    def test_set_scan_workers_max_only_grows(self) -> None:
        metrics = RuntimeMetrics()
        metrics.set_scan_workers_max(8)
        self.assertEqual(metrics.scan_workers, 8)
        metrics.set_scan_workers_max(4)
        # Lower value is ignored.
        self.assertEqual(metrics.scan_workers, 8)
        metrics.set_scan_workers_max(12)
        self.assertEqual(metrics.scan_workers, 12)


class RuntimeMetricsConfigureProgressTests(unittest.TestCase):
    def test_configure_progress_normalizes_weights(self) -> None:
        metrics = RuntimeMetrics()
        metrics.configure_progress({"scan": 4, "embedding": 6})
        self.assertAlmostEqual(metrics.progress_weights["scan"], 0.4)
        self.assertAlmostEqual(metrics.progress_weights["embedding"], 0.6)

    def test_configure_progress_drops_non_positive_weights(self) -> None:
        metrics = RuntimeMetrics()
        metrics.configure_progress({"scan": 1.0, "ignored": 0.0, "embedding": 1.0})
        self.assertNotIn("ignored", metrics.progress_weights)
        self.assertAlmostEqual(metrics.progress_weights["scan"], 0.5)
        self.assertAlmostEqual(metrics.progress_weights["embedding"], 0.5)

    def test_configure_progress_falls_back_to_scan_when_all_zero(self) -> None:
        metrics = RuntimeMetrics()
        metrics.configure_progress({"scan": 0, "embedding": -1})
        self.assertEqual(metrics.progress_weights, {"scan": 1.0})

    def test_configure_progress_clears_completed_phases(self) -> None:
        metrics = RuntimeMetrics()
        metrics.completed_phases.add("scan")
        metrics.phase_done = 100
        metrics.phase_total = 100
        metrics.configure_progress({"scan": 1.0})
        self.assertEqual(metrics.completed_phases, set())
        self.assertEqual(metrics.phase_done, 0)
        self.assertEqual(metrics.phase_total, 0)


class RuntimeMetricsPhaseTrackingTests(unittest.TestCase):
    def test_begin_phase_sets_active_phase_and_resets_progress(self) -> None:
        metrics = RuntimeMetrics()
        metrics.begin_phase("scan", total=50)
        self.assertEqual(metrics.active_phase, "scan")
        self.assertEqual(metrics.phase_total, 50)
        self.assertEqual(metrics.phase_done, 0)
        self.assertNotIn("scan", metrics.completed_phases)

    def test_begin_phase_clamps_negative_total_to_zero(self) -> None:
        metrics = RuntimeMetrics()
        metrics.begin_phase("scan", total=-5)
        self.assertEqual(metrics.phase_total, 0)

    def test_add_phase_total_grows_only_for_positive_values(self) -> None:
        metrics = RuntimeMetrics()
        metrics.begin_phase("scan", total=10)
        metrics.add_phase_total(5)
        self.assertEqual(metrics.phase_total, 15)
        metrics.add_phase_total(0)
        metrics.add_phase_total(-3)
        self.assertEqual(metrics.phase_total, 15)

    def test_add_phase_done_grows_only_for_positive_values(self) -> None:
        metrics = RuntimeMetrics()
        metrics.begin_phase("scan", total=10)
        metrics.add_phase_done(3)
        metrics.add_phase_done(2)
        self.assertEqual(metrics.phase_done, 5)
        metrics.add_phase_done(0)
        metrics.add_phase_done(-1)
        self.assertEqual(metrics.phase_done, 5)

    def test_end_phase_accumulates_into_metric_field_and_marks_completed(self) -> None:
        metrics = RuntimeMetrics()
        metrics.begin_phase("scan", total=10)
        metrics.phase_done = 4
        # Force a known elapsed window by rewinding the start time directly.
        started = metrics.active_phase_started
        if started is None:
            self.fail("begin_phase did not set active_phase_started")
        metrics.active_phase_started = started - 1.5
        metrics.end_phase("scan", "scan_seconds")
        self.assertIsNone(metrics.active_phase)
        self.assertIn("scan", metrics.completed_phases)
        # phase_done is clamped to phase_total at end_phase
        self.assertEqual(metrics.phase_done, 10)
        # Elapsed ≥ ~1.5s
        self.assertGreaterEqual(metrics.scan_seconds, 1.4)

    def test_end_phase_for_inactive_phase_is_a_noop(self) -> None:
        metrics = RuntimeMetrics()
        metrics.begin_phase("scan", total=10)
        # Try to end a different phase
        before = metrics.scan_seconds
        metrics.end_phase("embedding", "embedding_seconds")
        self.assertEqual(metrics.scan_seconds, before)
        self.assertEqual(metrics.active_phase, "scan")
        self.assertNotIn("embedding", metrics.completed_phases)

    def test_complete_phase_marks_completed_without_accumulating_time(self) -> None:
        metrics = RuntimeMetrics()
        metrics.begin_phase("scan", total=10)
        metrics.phase_done = 4
        before_scan_seconds = metrics.scan_seconds
        metrics.complete_phase("scan")
        self.assertIsNone(metrics.active_phase)
        self.assertIn("scan", metrics.completed_phases)
        # complete_phase does not roll active_seconds into scan_seconds
        self.assertEqual(metrics.scan_seconds, before_scan_seconds)
        # phase_done is clamped to phase_total
        self.assertEqual(metrics.phase_done, 10)


class RuntimeMetricsEmbeddingInputsTests(unittest.TestCase):
    def test_add_embedding_inputs_tracks_batch_chars_and_tokens(self) -> None:
        metrics = RuntimeMetrics()
        metrics.add_embedding_inputs(["abcd", "abcdefgh"])
        # 2 texts → 1 batch, attempted=2
        self.assertEqual(metrics.embedding_batches, 1)
        self.assertEqual(metrics.embedding_records_attempted, 2)
        # chars summed
        self.assertEqual(metrics.embedding_input_chars, 4 + 8)
        # tokens estimated: ceil(4/4) + ceil(8/4) = 1 + 2 = 3
        self.assertEqual(metrics.embedding_input_tokens_estimated, 3)

    def test_add_embedding_inputs_empty_list_still_increments_batch_count(self) -> None:
        metrics = RuntimeMetrics()
        metrics.add_embedding_inputs([])
        self.assertEqual(metrics.embedding_batches, 1)
        self.assertEqual(metrics.embedding_records_attempted, 0)
        self.assertEqual(metrics.embedding_input_chars, 0)
        self.assertEqual(metrics.embedding_input_tokens_estimated, 0)


class RuntimeMetricsSnapshotTests(unittest.TestCase):
    def test_snapshot_returns_timing_counts_token_use_and_progress(self) -> None:
        metrics = RuntimeMetrics()
        metrics.set("inserted_records", 7)
        snapshot = metrics.snapshot()
        self.assertIn("timing", snapshot)
        self.assertIn("counts", snapshot)
        self.assertIn("token_use", snapshot)
        self.assertIn("progress", snapshot)

    def test_snapshot_counts_reflect_recent_writes(self) -> None:
        metrics = RuntimeMetrics()
        metrics.set("inserted_records", 7)
        metrics.set("inserted_edges", 3)
        counts = cast("dict[str, object]", metrics.snapshot()["counts"])
        self.assertEqual(counts["inserted_records"], 7)
        self.assertEqual(counts["inserted_edges"], 3)

    def test_snapshot_progress_includes_active_phase_seconds_when_phase_running(self) -> None:
        metrics = RuntimeMetrics()
        metrics.begin_phase("scan", total=20)
        metrics.add_phase_done(5)
        snapshot = metrics.snapshot()
        timing = cast("dict[str, object]", snapshot["timing"])
        self.assertEqual(timing.get("active_phase"), "scan")
        self.assertIn("active_phase_seconds", timing)
        progress = cast("dict[str, object]", snapshot["progress"])
        self.assertEqual(progress["phase"], "scan")
        self.assertEqual(progress["phase_done"], 5)
        self.assertEqual(progress["phase_total"], 20)
        self.assertEqual(progress["phase_percent"], 25.0)

    def test_snapshot_progress_falls_back_when_no_active_phase(self) -> None:
        metrics = RuntimeMetrics()
        progress = cast("dict[str, object]", metrics.snapshot()["progress"])
        self.assertIsNone(progress["phase"])
        self.assertIsNone(progress["phase_total"])
        self.assertIsNone(progress["phase_percent"])

    def test_snapshot_emits_embedded_records_per_second_when_embedding_time_observed(self) -> None:
        metrics = RuntimeMetrics()
        metrics.set("embedded_records", 100)
        metrics.set("embedding_seconds", 4.0)
        counts = cast("dict[str, object]", metrics.snapshot()["counts"])
        self.assertEqual(counts["embedded_records_per_second"], 25.0)

    def test_snapshot_token_use_basis_string_matches_env(self) -> None:
        metrics = RuntimeMetrics()
        with patch.dict(os.environ, {"PCI_TOKEN_CHARS_PER_TOKEN": str(5)}, clear=False):
            token_use = cast("dict[str, object]", metrics.snapshot()["token_use"])
        self.assertEqual(token_use["embedding_token_estimate_basis"], "ceil(chars/5)")


class ResetActiveMetricsTests(unittest.TestCase):
    def test_reset_active_metrics_replaces_singleton(self) -> None:
        original = runtime.active_metrics
        original.set("inserted_records", 42)
        fresh = reset_active_metrics()
        # The returned instance is the new singleton, distinct from the original.
        self.assertIsNot(fresh, original)
        self.assertEqual(fresh.inserted_records, 0)

    def test_module_getattr_returns_current_instance_after_reset(self) -> None:
        first = runtime.active_metrics
        first.set("inserted_records", 9)
        _ = reset_active_metrics()
        # After reset, module-level attribute access returns the new instance.
        self.assertEqual(runtime.active_metrics.inserted_records, 0)
        self.assertIsNot(runtime.active_metrics, first)

    def test_module_getattr_raises_for_unknown_attribute(self) -> None:
        # Call the module __getattr__ hook directly: it's a typed function in
        # runtime.py, so we don't need an attribute-access suppression to
        # verify the error path.
        with self.assertRaises(AttributeError):
            _ = runtime_module_getattr("does_not_exist")


class RuntimeHeartbeatSecondsTests(unittest.TestCase):
    def test_default_heartbeat_is_five_minutes(self) -> None:
        env_without_var = {k: v for k, v in os.environ.items() if k != "PCI_RUNTIME_HEARTBEAT_SECONDS"}
        with patch.dict(os.environ, env_without_var, clear=True):
            self.assertEqual(runtime_heartbeat_seconds(), 300)

    def test_heartbeat_reads_env_var_when_set(self) -> None:
        with patch.dict(os.environ, {"PCI_RUNTIME_HEARTBEAT_SECONDS": "30"}, clear=False):
            self.assertEqual(runtime_heartbeat_seconds(), 30)


if __name__ == "__main__":
    _ = unittest.main()
