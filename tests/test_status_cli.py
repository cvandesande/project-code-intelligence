"""Unit tests for `pci status` (`project_code_intelligence.status_cli`)."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from project_code_intelligence import db as pci_db
from project_code_intelligence import status_cli

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject


def _run(*, finished_at: object = None, heartbeat_age: float = 5.0, **extra: object) -> JsonObject:
    now = datetime.now(timezone.utc)
    run: dict[str, object] = {
        "id": 7,
        "collection": "ws",
        "repos": ["repo-a"],
        "repo_modes": {"repo-a": "incremental"},
        "phase": "embedding",
        "progress": {},
        "started_at": now - timedelta(seconds=90),
        "heartbeat_at": now - timedelta(seconds=heartbeat_age),
        "finished_at": finished_at,
        "exit_code": None,
        "interrupted": False,
        "error": None,
    }
    run.update(extra)
    return cast("JsonObject", run)


class RunStateTests(unittest.TestCase):
    now = datetime.now(timezone.utc)

    def test_fresh_heartbeat_without_finish_is_running(self) -> None:
        self.assertEqual(status_cli.run_state(_run(), self.now), "running")

    def test_stale_heartbeat_without_finish_is_stalled(self) -> None:
        self.assertEqual(status_cli.run_state(_run(heartbeat_age=3600.0), self.now), "stalled")

    def test_finished_zero_exit_is_ok(self) -> None:
        run = _run(finished_at=self.now, exit_code=0)
        self.assertEqual(status_cli.run_state(run, self.now), "ok")

    def test_finished_nonzero_exit_is_failed(self) -> None:
        run = _run(finished_at=self.now, exit_code=1)
        self.assertEqual(status_cli.run_state(run, self.now), "failed")

    def test_interrupted_wins_over_exit_code(self) -> None:
        run = _run(finished_at=self.now, exit_code=130, interrupted=True)
        self.assertEqual(status_cli.run_state(run, self.now), "interrupted")


class EmbeddingLogHintTests(unittest.TestCase):
    def test_apple_server_running_points_at_log_file(self) -> None:
        with patch.object(status_cli, "apple_embed_server_is_running", return_value=True):
            hint = status_cli.embedding_log_hint()
        self.assertEqual(hint, str(status_cli.APPLE_EMBED_SERVER_LOG_FILE))

    def test_docker_container_running_points_at_docker_logs(self) -> None:
        completed = status_cli.process.CompletedProcess(
            args=["docker"], returncode=0, stdout="pgvector-1\nproject-code-intelligence-fastembed-1\n"
        )
        with (
            patch.object(status_cli, "apple_embed_server_is_running", return_value=False),
            patch.object(status_cli.process, "run_docker", return_value=completed),
            patch.object(status_cli.process, "container_engine_name", return_value="docker"),
        ):
            hint = status_cli.embedding_log_hint()
        self.assertEqual(hint, "docker logs project-code-intelligence-fastembed-1")

    def test_no_backend_running_returns_none(self) -> None:
        with (
            patch.object(status_cli, "apple_embed_server_is_running", return_value=False),
            patch.object(status_cli.process, "run_docker", side_effect=FileNotFoundError),
        ):
            self.assertIsNone(status_cli.embedding_log_hint())


class MainRenderTests(unittest.TestCase):
    def test_json_output_carries_runs_and_state(self) -> None:
        runs = [_run()]
        out = io.StringIO()
        with (
            patch.object(status_cli, "load_status", return_value=(runs, [])),
            patch.object(status_cli, "embedding_log_hint", return_value="/var/log/embed.log"),
            redirect_stdout(out),
        ):
            code = status_cli.main(["--json"])
        self.assertEqual(code, 0)
        payload = cast("dict[str, object]", json.loads(out.getvalue()))
        self.assertTrue(payload["ok"])
        runs_out = cast("list[dict[str, object]]", payload["runs"])
        self.assertEqual(runs_out[0]["state"], "running")
        self.assertEqual(payload["embedding_log"], "/var/log/embed.log")

    def test_plain_output_renders_run_and_snapshot_lines(self) -> None:
        snapshots: list[dict[str, object]] = [
            {
                "collection": "ws",
                "repo": "repo-a",
                "commit_sha": "abc1234def",
                "head_status": "current",
                "metadata": {"ingest_mode": "incremental", "changed_files": 2, "unchanged_files": 40},
            }
        ]
        out = io.StringIO()
        with (
            patch.object(status_cli, "load_status", return_value=([_run()], snapshots)),
            patch.object(status_cli.console_ui, "should_emit_pretty", return_value=False),
            patch.object(status_cli, "embedding_log_hint", return_value="docker logs pci-fastembed-1"),
            redirect_stdout(out),
        ):
            code = status_cli.main([])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("run ws: running", text)
        self.assertIn("EMBEDDING", text)
        self.assertIn("snapshot ws/repo-a: abc1234 · current · incremental, 2 changed / 40 unchanged", text)
        self.assertIn("embedding log: docker logs pci-fastembed-1", text)

    def test_database_unreachable_exits_nonzero(self) -> None:
        err = io.StringIO()
        with (
            patch.object(status_cli, "load_status", side_effect=pci_db.DatabaseConnectionError("db down")),
            patch.object(status_cli.sys, "stderr", err),
        ):
            code = status_cli.main([])
        self.assertEqual(code, 1)
        self.assertIn("database unreachable", err.getvalue())
