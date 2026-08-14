"""Unit tests for `pci status` (`project_code_intelligence.status_cli`)."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from typing_extensions import override

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


class RunDetailTests(unittest.TestCase):
    now = datetime.now(timezone.utc)

    def test_finished_run_shows_duration_not_started_ago(self) -> None:
        run = _run(
            finished_at=self.now - timedelta(seconds=30),
            exit_code=0,
            started_at=self.now - timedelta(seconds=90),
        )
        detail = status_cli.run_detail(run, self.now)
        self.assertIn("took 1m00s", detail)
        self.assertIn("finished 30s ago", detail)
        self.assertNotIn("started", detail)

    def test_running_run_still_shows_started_ago(self) -> None:
        run = _run(started_at=self.now - timedelta(seconds=90))
        detail = status_cli.run_detail(run, self.now)
        self.assertIn("started 1m30s ago", detail)
        self.assertNotIn("took", detail)


class StatusPillTests(unittest.TestCase):
    now = datetime.now(timezone.utc)

    def test_running_run_shows_its_ledger_phase(self) -> None:
        # Same label the live `pci index` header shows for this phase.
        self.assertEqual(status_cli.status_pill([_run()], [], self.now), ("running", "EMBEDDING"))

    def test_running_run_without_phase_shows_running(self) -> None:
        self.assertEqual(status_cli.status_pill([_run(phase=None)], [], self.now), ("running", "RUNNING"))

    def test_stalled_run_shows_attention(self) -> None:
        self.assertEqual(
            status_cli.status_pill([_run(heartbeat_age=3600.0)], [], self.now),
            ("warn", "ATTENTION"),
        )

    def test_other_workspace_run_shows_running(self) -> None:
        self.assertEqual(status_cli.status_pill([], ["project-ngf-ew"], self.now), ("running", "RUNNING"))

    def test_no_runs_is_idle(self) -> None:
        self.assertEqual(status_cli.status_pill([], [], self.now), ("ok", "IDLE"))


class OtherHostIndexRunsTests(unittest.TestCase):
    def test_reports_collections_of_running_index_processes(self) -> None:
        ps_out = "\n".join([
            "ARGS",
            "/usr/bin/python /Users/x/.local/bin/pci index --collection project-ngf-ew --worktree a=b",
            "grep pci index",
            "/usr/bin/vim notes.md",
        ])
        completed = status_cli.process.CompletedProcess(args=["ps"], returncode=0, stdout=ps_out)
        with patch.object(status_cli.process, "run", return_value=completed):
            self.assertEqual(status_cli.other_host_index_runs(), ["project-ngf-ew"])

    def test_ps_failure_is_swallowed(self) -> None:
        with patch.object(status_cli.process, "run", side_effect=OSError):
            self.assertEqual(status_cli.other_host_index_runs(), [])


class EmbeddingLogHintTests(unittest.TestCase):
    def test_apple_server_running_points_at_log_file(self) -> None:
        with patch.object(status_cli, "apple_embed_server_is_running", return_value=True):
            hint = status_cli.embedding_log_hint()
        self.assertEqual(hint, str(status_cli.APPLE_EMBED_SERVER_LOG_FILE))

    def test_podman_container_running_points_at_podman_logs(self) -> None:
        completed = status_cli.process.CompletedProcess(args=["podman"], returncode=0, stdout="fastembed\n")
        with (
            patch.object(status_cli, "apple_embed_server_is_running", return_value=False),
            patch.object(status_cli.process, "run_podman", return_value=completed),
        ):
            hint = status_cli.embedding_log_hint()
        self.assertEqual(hint, "podman logs fastembed")

    def test_no_backend_running_returns_none(self) -> None:
        with (
            patch.object(status_cli, "apple_embed_server_is_running", return_value=False),
            patch.object(status_cli.process, "run_podman", side_effect=FileNotFoundError),
        ):
            self.assertIsNone(status_cli.embedding_log_hint())


class MainRenderTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        # Keep main() hermetic: never scan the host process table in tests.
        patcher = patch.object(status_cli, "other_host_index_runs", return_value=[])
        _ = patcher.start()
        self.addCleanup(patcher.stop)

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

    def test_plain_output_renders_one_row_per_repo_branch(self) -> None:
        # Two branches of the same collection/repo are two distinct snapshots;
        # each must get its own row, suffixed with its own branch.
        snapshots: list[dict[str, object]] = [
            {
                "collection": "ws",
                "repo": "repo-a",
                "branch": "main",
                "commit_sha": "abc1234def",
                "head_status": "current",
                "metadata": {},
            },
            {
                "collection": "ws",
                "repo": "repo-a",
                "branch": "feature",
                "commit_sha": "def5678abc",
                "head_status": "current",
                "metadata": {},
            },
        ]
        out = io.StringIO()
        with (
            patch.object(status_cli, "load_status", return_value=([], snapshots)),
            patch.object(status_cli.console_ui, "should_emit_pretty", return_value=False),
            patch.object(status_cli, "embedding_log_hint", return_value=None),
            redirect_stdout(out),
        ):
            code = status_cli.main([])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("snapshot ws/repo-a@main: ", text)
        self.assertIn("snapshot ws/repo-a@feature: ", text)

    def test_database_unreachable_exits_nonzero(self) -> None:
        err = io.StringIO()
        with (
            patch.object(status_cli, "load_status", side_effect=pci_db.DatabaseConnectionError("db down")),
            patch.object(status_cli.sys, "stderr", err),
        ):
            code = status_cli.main([])
        self.assertEqual(code, 1)
        self.assertIn("database unreachable", err.getvalue())
