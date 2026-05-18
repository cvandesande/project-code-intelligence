from __future__ import annotations

import os
import unittest
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from project_code_intelligence.mcp import db as mcp_db

if TYPE_CHECKING:
    from project_code_intelligence import db


class FakeCursor:
    def __init__(self, row: db.DbRow | None = None) -> None:
        self.row = row

    def fetchone(self) -> db.DbRow | None:
        return self.row


class FakeConnection:
    def __init__(self, rows: list[db.DbRow] | None = None) -> None:
        self.rows = list(rows or [])
        self.calls: list[tuple[str, list[object]]] = []

    def execute(self, query: str, params: list[object] | None = None) -> FakeCursor:
        self.calls.append((query, params or []))
        return FakeCursor(self.rows.pop(0) if self.rows else None)


class McpDatabaseSessionTests(unittest.TestCase):
    def test_session_timeout_settings_use_mcp_environment(self) -> None:
        env = {
            "PCI_MCP_STATEMENT_TIMEOUT_MS": "250",
            "PCI_MCP_LOCK_TIMEOUT_MS": "50",
            "PCI_MCP_IDLE_IN_TRANSACTION_TIMEOUT_MS": "500",
            "PCI_MCP_MAX_STATUS_ROWS": "25",
        }

        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(mcp_db.mcp_statement_timeout_ms(), 250)
            self.assertEqual(mcp_db.mcp_lock_timeout_ms(), 50)
            self.assertEqual(mcp_db.mcp_idle_in_transaction_timeout_ms(), 500)
            self.assertEqual(mcp_db.mcp_max_status_rows(), 25)

            fake = FakeConnection()
            mcp_db.configure_session(cast("db.DbConnection", fake))

        self.assertEqual(len(fake.calls), 1)
        query, params = fake.calls[0]
        self.assertIn("set_config('statement_timeout'", query)
        self.assertEqual(params, ["250ms", "50ms", "500ms"])

    def test_table_existence_checks_use_regclass(self) -> None:
        fake = FakeConnection([{"exists": True}, {"exists": False}])

        records_exist = mcp_db.code_intel_tables_exist(cast("db.DbConnection", fake))
        static_exists = mcp_db.table_regclass_exists(
            cast("db.DbConnection", fake),
            "project_code_intel_static_runs",
        )

        self.assertTrue(records_exist)
        self.assertFalse(static_exists)
        self.assertIn("project_code_intel_records", fake.calls[0][0])
        self.assertEqual(fake.calls[0][1], [])
        self.assertEqual(fake.calls[1][1], ["public.project_code_intel_static_runs"])


if __name__ == "__main__":
    _ = unittest.main()
