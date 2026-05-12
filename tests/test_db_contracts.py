from __future__ import annotations

import json
import unittest

from project_code_intelligence.config import DatabaseSettings
from project_code_intelligence.db import DatabaseConnectionError, conninfo, json_metadata, require_row, vector_literal


class DatabaseContractTests(unittest.TestCase):
    def test_conninfo_uses_dsn_or_complete_parts(self) -> None:
        self.assertEqual(
            conninfo(DatabaseSettings(dsn="postgresql://example.invalid/db")), "postgresql://example.invalid/db"
        )
        credential = "p"

        text = conninfo(DatabaseSettings(host="db", port="5432", dbname="codeintel", user="u", password=credential))

        self.assertIn("host=db", text)
        self.assertIn("dbname=codeintel", text)
        self.assertIn("user=u", text)

    def test_conninfo_reports_missing_connection_parts(self) -> None:
        credential = "p"
        with self.assertRaises(DatabaseConnectionError):
            _ = conninfo(DatabaseSettings(dbname=None, user="u", password=credential))

    def test_require_row_rejects_empty_database_results(self) -> None:
        row = {"id": 1}

        self.assertEqual(require_row(row, "demo"), row)
        with self.assertRaises(RuntimeError):
            _ = require_row(None, "demo")

    def test_vector_literal_accepts_only_non_empty_numeric_lists(self) -> None:
        self.assertEqual(vector_literal([1, 2.5, -3]), "[1,2.5,-3]")

        with self.assertRaises(ValueError):
            _ = vector_literal([])

        for value in ((1, 2), [True], ["1"]):
            with self.subTest(value=value), self.assertRaises(TypeError):
                _ = vector_literal(value)

    def test_json_metadata_uses_stable_object_encoding(self) -> None:
        self.assertEqual(json_metadata(None), "{}")
        self.assertEqual(json.loads(json_metadata({"b": 2, "a": 1})), {"a": 1, "b": 2})

        with self.assertRaises(TypeError):
            _ = json_metadata(["not", "an", "object"])


if __name__ == "__main__":
    _ = unittest.main()
