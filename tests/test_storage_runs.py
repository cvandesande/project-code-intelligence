"""Unit tests for the index-run ledger (`project_code_intelligence.storage.runs`).

Uses the QueuedConnection / FakeCursor pattern from tests/test_storage_schema.py.
Writers open their own connection via db.connect, so tests patch that and wrap
the fake connection in a context manager.
"""

from __future__ import annotations

import unittest
from typing import cast
from unittest.mock import patch

from project_code_intelligence import db as pci_db
from project_code_intelligence.storage import runs as storage_runs


class FakeCursor:
    def __init__(self, *, one: object | None = None, many: list[object] | None = None) -> None:
        self.one = one
        self.many = list(many or [])

    def fetchone(self) -> object | None:
        return self.one

    def fetchall(self) -> list[object]:
        return self.many


class QueuedConnection:
    def __init__(self, cursors: list[FakeCursor]) -> None:
        self.cursors = cursors
        self.calls: list[tuple[str, list[object]]] = []

    def execute(self, query: object, params: object | None = None) -> FakeCursor:
        if params is None:
            query_params: list[object] = []
        elif isinstance(params, list):
            query_params = cast("list[object]", params)
        else:
            raise TypeError("fake connection expects list query params")
        self.calls.append((str(query), query_params))
        return self.cursors.pop(0)


class FakeConnect:
    def __init__(self, conn: object) -> None:
        self.conn = conn

    def __enter__(self) -> object:
        return self.conn

    def __exit__(self, _exc_type: object, exc: object, traceback: object) -> None:
        return None


class SchemaContainsIndexRunsTests(unittest.TestCase):
    def test_schema_sql_creates_the_index_runs_table_and_index(self) -> None:
        schema = str(pci_db.schema_sql())
        self.assertIn("CREATE TABLE IF NOT EXISTS project_code_intel_index_runs", schema)
        self.assertIn("project_code_intel_index_runs_collection_idx", schema)


class StartIndexRunTests(unittest.TestCase):
    def test_inserts_a_row_and_returns_its_id(self) -> None:
        conn = QueuedConnection([FakeCursor(one={"id": 7})])
        with patch.object(pci_db, "connect", return_value=FakeConnect(conn)):
            run_id = storage_runs.start_index_run("ws", ["repo-a", "repo-b"], pid=123, host="mac")
        self.assertEqual(run_id, 7)
        query, params = conn.calls[0]
        self.assertIn("INSERT INTO project_code_intel_index_runs", query)
        self.assertEqual(params[0], "ws")
        self.assertEqual(params[1], '["repo-a","repo-b"]')
        self.assertEqual(params[2:], [123, "mac"])

    def test_returns_none_when_the_database_is_unreachable(self) -> None:
        with patch.object(pci_db, "connect", side_effect=pci_db.DatabaseConnectionError("db down")):
            run_id = storage_runs.start_index_run("ws", ["repo-a"], pid=1, host="mac")
        self.assertIsNone(run_id)


class HeartbeatIndexRunTests(unittest.TestCase):
    def test_updates_heartbeat_phase_and_progress(self) -> None:
        conn = QueuedConnection([FakeCursor()])
        with patch.object(pci_db, "connect", return_value=FakeConnect(conn)):
            storage_runs.heartbeat_index_run(7, phase="embedding", progress={"counts": {"embedded_records": 5}})
        query, params = conn.calls[0]
        self.assertIn("SET heartbeat_at = now()", query)
        self.assertEqual(params, ["embedding", '{"counts":{"embedded_records":5}}', 7])

    def test_swallows_connection_errors(self) -> None:
        with patch.object(pci_db, "connect", side_effect=pci_db.DatabaseConnectionError("db down")):
            self.assertIsNone(storage_runs.heartbeat_index_run(7, phase="scan", progress={}))


class SetIndexRunModesTests(unittest.TestCase):
    def test_stamps_repo_modes(self) -> None:
        conn = QueuedConnection([FakeCursor()])
        with patch.object(pci_db, "connect", return_value=FakeConnect(conn)):
            storage_runs.set_index_run_modes(7, {"repo-a": "incremental", "repo-b": "full:version_mismatch"})
        query, params = conn.calls[0]
        self.assertIn("SET repo_modes", query)
        self.assertEqual(params, ['{"repo-a":"incremental","repo-b":"full:version_mismatch"}', 7])


class FinishIndexRunTests(unittest.TestCase):
    def test_stamps_terminal_state_then_prunes(self) -> None:
        conn = QueuedConnection([FakeCursor(), FakeCursor()])
        with patch.object(pci_db, "connect", return_value=FakeConnect(conn)):
            storage_runs.finish_index_run(7, exit_code=0, interrupted=False, error=None, progress={})
        finish_query, finish_params = conn.calls[0]
        self.assertIn("SET finished_at = now()", finish_query)
        self.assertEqual(finish_params, [0, False, None, "{}", 7])
        prune_query, prune_params = conn.calls[1]
        self.assertIn("DELETE FROM project_code_intel_index_runs", prune_query)
        self.assertIn("finished_at IS NOT NULL", prune_query)
        self.assertEqual(prune_params, [storage_runs.INDEX_RUNS_KEEP])

    def test_swallows_connection_errors(self) -> None:
        with patch.object(pci_db, "connect", side_effect=pci_db.DatabaseConnectionError("db down")):
            self.assertIsNone(
                storage_runs.finish_index_run(7, exit_code=1, interrupted=True, error="boom", progress={})
            )


class LoadIndexRunsTests(unittest.TestCase):
    def test_scopes_by_collection_when_given(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[{"id": 1}])])
        rows = storage_runs.load_index_runs(cast("pci_db.DbConnection", conn), collection="ws", limit=5)
        self.assertEqual(rows, [{"id": 1}])
        query, params = conn.calls[0]
        self.assertIn("WHERE collection = %s", query)
        self.assertEqual(params, ["ws", 5])

    def test_no_collection_lists_all(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])
        _ = storage_runs.load_index_runs(cast("pci_db.DbConnection", conn))
        query, params = conn.calls[0]
        self.assertNotIn("WHERE collection", query)
        self.assertEqual(params, [20])


class ActiveRunHolderTests(unittest.TestCase):
    def test_roundtrip_and_reset(self) -> None:
        storage_runs.set_active_index_run(9)
        self.assertEqual(storage_runs.active_index_run_id(), 9)
        storage_runs.set_active_index_run(None)
        self.assertIsNone(storage_runs.active_index_run_id())
