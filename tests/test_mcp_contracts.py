from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from project_code_intelligence.embedding.types import EmbeddingEndpointUnavailableError
from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError
from project_code_intelligence.mcp import tools as mcp_tools
from project_code_intelligence.mcp.filters import (
    code_intel_clauses,
    json_argument,
    status_filters,
)
from project_code_intelligence.mcp.protocol import (
    optional_bool,
    optional_int,
    optional_text,
    require_int,
    result_text,
)
from project_code_intelligence.mcp.tool_catalog import TOOL_DEFINITIONS, validate_tool_arguments
from project_code_intelligence.mcp.transport import error_message, handle_tool_call, set_mcp_environment_defaults
from project_code_intelligence.server import vector_literal_dimensions


class FakeCursor:
    @staticmethod
    def fetchone() -> None:
        return None

    @staticmethod
    def fetchall() -> list[object]:
        return []


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[object]]] = []

    def execute(self, query: object, params: object | None = None) -> FakeCursor:
        if params is None:
            query_params: list[object] = []
        elif isinstance(params, list):
            query_params = cast("list[object]", params)
        else:
            raise TypeError("fake connection expects list query params")
        self.calls.append((str(query), query_params))
        return FakeCursor()


class FakeConnect:
    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn

    def __enter__(self) -> FakeConnection:
        return self.conn

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


class McpContractTests(unittest.TestCase):
    def test_mcp_defaults_collection_from_process_cwd(self) -> None:
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "project-code-intelligence"
            workspace.mkdir()
            try:
                os.chdir(workspace)
                with patch.dict(os.environ, {}, clear=True):
                    set_mcp_environment_defaults()
                    self.assertEqual(
                        os.environ["PROJECT_CODE_INTELLIGENCE_COLLECTION"],
                        "project-code-intelligence",
                    )
            finally:
                os.chdir(old_cwd)

    def test_mcp_keeps_explicit_collection_override(self) -> None:
        with patch.dict(os.environ, {"PROJECT_CODE_INTELLIGENCE_COLLECTION": "configured"}, clear=True):
            set_mcp_environment_defaults()
            self.assertEqual(os.environ["PROJECT_CODE_INTELLIGENCE_COLLECTION"], "configured")

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
        with patch.dict(os.environ, {}, clear=True):
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
        with patch.dict(os.environ, {}, clear=True):
            filters = status_filters({"collection": "test", "repo": "repo-a"})

        self.assertEqual(filters.snapshots.params, ["test", "repo-a"])
        self.assertEqual(filters.records.params, ["test", "repo-a"])
        self.assertEqual(filters.files.params, ["test", "repo-a"])
        self.assertEqual(filters.edges.params, ["test", "repo-a"])
        self.assertEqual(filters.static_runs.params, ["test", "repo-a"])
        self.assertEqual(filters.static_findings.params, ["test", "repo-a"])

    def test_id_record_fetch_is_scoped_to_configured_collection(self) -> None:
        conn = FakeConnection()

        with (
            patch.dict(
                os.environ,
                {"PROJECT_CODE_INTELLIGENCE_COLLECTION": "project-code-intelligence"},
                clear=True,
            ),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_get_code_intel_record({"id": 1})

        query, params = conn.calls[0]
        self.assertIn("WHERE id = %s", query)
        self.assertIn("AND collection = %s", query)
        self.assertEqual(params, [1, "project-code-intelligence"])

    def test_id_static_finding_fetch_is_scoped_to_configured_collection(self) -> None:
        conn = FakeConnection()

        with (
            patch.dict(
                os.environ,
                {"PROJECT_CODE_INTELLIGENCE_COLLECTION": "project-code-intelligence"},
                clear=True,
            ),
            patch.object(mcp_tools, "table_regclass_exists", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_get_static_finding({"id": 1})

        query, params = conn.calls[0]
        self.assertIn("f.id = %s", query)
        self.assertIn("f.collection = %s", query)
        self.assertEqual(params, [1, "project-code-intelligence"])

    def test_static_code_flow_fetch_is_scoped_to_configured_collection(self) -> None:
        conn = FakeConnection()

        with (
            patch.dict(
                os.environ,
                {"PROJECT_CODE_INTELLIGENCE_COLLECTION": "project-code-intelligence"},
                clear=True,
            ),
            patch.object(mcp_tools, "table_regclass_exists", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_get_static_code_flow({"finding_id": 1})

        query, params = conn.calls[0]
        self.assertIn("JOIN project_code_intel_static_findings f", query)
        self.assertIn("cf.finding_id = %s", query)
        self.assertIn("f.collection = %s", query)
        self.assertEqual(params, [1, "project-code-intelligence"])

    def test_vector_literal_dimensions_counts_pgvector_literal_dimensions(self) -> None:
        self.assertEqual(vector_literal_dimensions("[]"), 0)
        self.assertEqual(vector_literal_dimensions("[0.1]"), 1)
        self.assertEqual(vector_literal_dimensions("[0.1,0.2,0.3]"), 3)

    def test_semantic_search_reports_missing_embedding_endpoint_clearly(self) -> None:
        endpoint = "http://127.0.0.1:18081/v1/embeddings"

        with (
            patch.object(mcp_tools.config, "default_embedding_endpoint", return_value=endpoint),
            patch.object(mcp_tools.embeddings, "resolve_embedding_endpoint_model", return_value="local"),
            patch.object(
                mcp_tools.embeddings,
                "embed_with_endpoint",
                side_effect=EmbeddingEndpointUnavailableError("connection refused"),
            ),
            self.assertRaises(McpProtocolError) as raised,
        ):
            _ = mcp_tools.query_embedding("find request handler")

        message = str(raised.exception)
        self.assertIn("semantic search requires an embedding endpoint", message)
        self.assertIn(endpoint, message)
        self.assertIn("PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT", message)

    def test_semantic_search_endpoint_failure_is_user_visible_mcp_error(self) -> None:
        endpoint = "http://127.0.0.1:18081/v1/embeddings"

        with (
            patch.object(mcp_tools.config, "default_embedding_endpoint", return_value=endpoint),
            patch.object(mcp_tools.embeddings, "resolve_embedding_endpoint_model", return_value="local"),
            patch.object(
                mcp_tools.embeddings,
                "embed_with_endpoint",
                side_effect=EmbeddingEndpointUnavailableError("connection refused"),
            ),
            self.assertRaises(McpProtocolError) as raised,
        ):
            _ = handle_tool_call(
                {
                    "params": {
                        "name": "search_code_intel_semantic",
                        "arguments": {"query": "find request handler"},
                    }
                },
                1,
            )

        message = error_message(raised.exception)
        self.assertIn("semantic search requires an embedding endpoint", message)
        self.assertNotEqual(message, "internal server error")

    def test_tool_schema_validation_rejects_unknown_and_bad_arguments(self) -> None:
        text_search = TOOL_DEFINITIONS["search_code_intel_text"]

        validate_tool_arguments(text_search, {"query": "hello", "limit": 10})

        with self.assertRaises(McpProtocolError):
            validate_tool_arguments(text_search, {"query": "hello", "surprise": True})
        with self.assertRaises(McpProtocolTypeError):
            validate_tool_arguments(text_search, {"query": "hello", "limit": "10"})
        with self.assertRaises(McpProtocolError):
            validate_tool_arguments(text_search, {"query": "hello", "limit": 500})

    def test_required_tool_arguments_are_enforced_before_handlers(self) -> None:
        record_fetch = TOOL_DEFINITIONS["get_code_intel_record"]

        with self.assertRaises(McpProtocolError):
            validate_tool_arguments(record_fetch, {})
        with self.assertRaises(McpProtocolTypeError):
            validate_tool_arguments(record_fetch, {"id": True})
        with self.assertRaises(McpProtocolError):
            validate_tool_arguments(record_fetch, {"id": 0})

    def test_tool_call_rejects_non_object_params_and_arguments(self) -> None:
        with self.assertRaises(McpProtocolTypeError):
            _ = handle_tool_call({"params": []}, None)
        with self.assertRaises(McpProtocolTypeError):
            _ = handle_tool_call({"params": {"name": "code_intel_status", "arguments": []}}, None)


if __name__ == "__main__":
    _ = unittest.main()
