from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError
from project_code_intelligence.mcp_filters import (
    code_intel_clauses,
    json_argument,
    status_filters,
)
from project_code_intelligence.mcp_protocol import (
    optional_bool,
    optional_int,
    optional_text,
    require_int,
    result_text,
)
from project_code_intelligence.server import vector_literal_dimensions


class McpContractTests(unittest.TestCase):
    def test_result_text_wraps_json_as_mcp_text_content(self) -> None:
        result = result_text({"ok": True})
        content_value = result.get("content")

        if not isinstance(content_value, list):
            self.fail("MCP result content should be a list")
        first_value = content_value[0]
        if not isinstance(first_value, dict):
            self.fail("MCP result content item should be an object")
        first = first_value
        self.assertEqual(first["type"], "text")
        self.assertIn('"ok": true', str(first["text"]))

    def test_integer_arguments_clamp_and_reject_bool_values(self) -> None:
        self.assertEqual(require_int({"limit": 500}, "limit", 10, 1, 50), 50)
        self.assertEqual(require_int({"limit": -1}, "limit", 10, 1, 50), 1)
        self.assertEqual(optional_int({"snapshot_id": 10}, "snapshot_id"), 10)

        with self.assertRaises(McpProtocolTypeError):
            _ = require_int({"limit": True}, "limit", 10, 1, 50)
        with self.assertRaises(McpProtocolTypeError):
            _ = optional_int({"snapshot_id": False}, "snapshot_id")
        with self.assertRaises(McpProtocolError):
            _ = optional_int({"snapshot_id": 0}, "snapshot_id")

    def test_text_and_bool_arguments_validate_mcp_boundaries(self) -> None:
        self.assertIsNone(optional_text({"query": ""}, "query"))
        self.assertEqual(optional_text({"query": "hello"}, "query"), "hello")
        self.assertTrue(optional_bool({"include_historical": True}, "include_historical"))

        with self.assertRaises(McpProtocolTypeError):
            _ = optional_text({"query": 123}, "query")
        with self.assertRaises(McpProtocolTypeError):
            _ = optional_bool({"include_historical": "yes"}, "include_historical")
        with (
            patch.dict(os.environ, {"PROJECT_CODE_INTELLIGENCE_MCP_MAX_TEXT_CHARS": "5"}, clear=True),
            self.assertRaises(McpProtocolError),
        ):
            _ = optional_text({"query": "too long"}, "query")

    def test_metadata_arguments_are_size_limited_and_parameterized(self) -> None:
        clauses, params = code_intel_clauses({"metadata_contains": {"symbol": "main"}}, "r")

        self.assertIn("r.metadata @> %s::jsonb", clauses)
        self.assertEqual(params[-1], '{"symbol":"main"}')
        self.assertEqual(json_argument({"a": 1}, "metadata"), '{"a":1}')
        with (
            patch.dict(os.environ, {"PROJECT_CODE_INTELLIGENCE_MCP_MAX_METADATA_BYTES": "1024"}, clear=True),
            self.assertRaises(McpProtocolError),
        ):
            _ = json_argument({"large": "x" * 2000}, "metadata")

    def test_status_filters_apply_same_scope_to_status_tables(self) -> None:
        filters = status_filters({"collection": "test", "repo": "repo-a"})

        self.assertEqual(filters.snapshots.params, ["test", "repo-a"])
        self.assertEqual(filters.records.params, ["test", "repo-a"])
        self.assertEqual(filters.files.params, ["test", "repo-a"])
        self.assertEqual(filters.edges.params, ["test", "repo-a"])
        self.assertEqual(filters.static_runs.params, ["test", "repo-a"])
        self.assertEqual(filters.static_findings.params, ["test", "repo-a"])

    def test_vector_literal_dimensions_counts_pgvector_literal_dimensions(self) -> None:
        self.assertEqual(vector_literal_dimensions("[]"), 0)
        self.assertEqual(vector_literal_dimensions("[0.1]"), 1)
        self.assertEqual(vector_literal_dimensions("[0.1,0.2,0.3]"), 3)


if __name__ == "__main__":
    _ = unittest.main()
