from __future__ import annotations

import unittest
from typing import TYPE_CHECKING
from unittest.mock import patch

from project_code_intelligence import progress

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject


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


if __name__ == "__main__":
    _ = unittest.main()
