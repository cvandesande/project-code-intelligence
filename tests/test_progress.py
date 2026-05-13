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
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def update(self, renderable: object) -> None:
        self.renderable = renderable

    def stop(self) -> None:
        self.stopped = True


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


if __name__ == "__main__":
    _ = unittest.main()
