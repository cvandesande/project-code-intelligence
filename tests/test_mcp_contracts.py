from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from project_code_intelligence.embedding.types import EmbeddingEndpointUnavailableError
from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError, McpWritePermissionError
from project_code_intelligence.mcp import tools as mcp_tools
from project_code_intelligence.mcp.filters import (
    code_intel_clauses,
    json_argument,
    source_path_prefix_pattern,
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
from project_code_intelligence.mcp.transport import (
    error_message,
    handle_jsonrpc_value,
    handle_tool_call,
    request_id_from_jsonrpc_value,
    set_mcp_environment_defaults,
)
from project_code_intelligence.server import vector_literal_dimensions


class FakeCursor:
    def __init__(self, *, one: object | None = None, many: list[object] | None = None) -> None:
        self.one = one
        self.many = list(many or [])

    def fetchone(self) -> object | None:
        return self.one

    def fetchall(self) -> list[object]:
        return self.many


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
    def __init__(self, conn: object) -> None:
        self.conn = conn

    def __enter__(self) -> object:
        return self.conn

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


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


def mcp_text_payload(response: object) -> dict[str, object]:
    if not isinstance(response, dict):
        raise TypeError("response should be an object")
    response_dict = cast("dict[object, object]", response)
    content = response_dict["content"]
    if not isinstance(content, list):
        raise TypeError("content should be a list")
    content_items = cast("list[object]", content)
    first = content_items[0]
    if not isinstance(first, dict):
        raise TypeError("content item should be an object")
    first_dict = cast("dict[object, object]", first)
    text = first_dict["text"]
    if not isinstance(text, str):
        raise TypeError("content text should be a string")
    payload = cast("object", json.loads(text))
    if not isinstance(payload, dict):
        raise TypeError("payload should be an object")
    return cast("dict[str, object]", payload)


class McpTextSearchTests(unittest.TestCase):
    def test_text_search_terms_extract_code_identifiers(self) -> None:
        self.assertEqual(
            mcp_tools.search_terms("CONFIG_SELINUX procd-selinux busybox-selinux setfiles"),
            ["CONFIG_SELINUX", "procd-selinux", "busybox-selinux", "setfiles"],
        )
        self.assertEqual(mcp_tools.search_terms("alpha AND beta OR gamma"), ["alpha", "beta", "gamma"])
        self.assertEqual(mcp_tools.like_pattern_for_term("a_b%"), "%a\\_b\\%%")

    def test_text_search_auto_falls_back_to_all_terms_for_multi_term_misses(self) -> None:
        conn = QueuedConnection([
            FakeCursor(many=[]),
            FakeCursor(many=[{"id": 7, "source_path": "config/Config-build.in"}]),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({
                "query": "CONFIG_SELINUX procd-selinux busybox-selinux setfiles",
            })

        payload = mcp_text_payload(response)
        # query_mode is no longer echoed when left at its default ("auto").
        self.assertNotIn("query_mode", payload)
        self.assertEqual(payload["query_strategy"], "all_terms_fallback")
        self.assertEqual(
            payload["terms"],
            ["CONFIG_SELINUX", "procd-selinux", "busybox-selinux", "setfiles"],
        )
        self.assertEqual(payload["fallback_reason"], "websearch returned no results for a multi-term query")
        self.assertEqual(payload["results"], [{"source_path": "config/Config-build.in"}])

        websearch_query, websearch_params = conn.calls[0]
        fallback_query, fallback_params = conn.calls[1]
        self.assertIn("websearch_to_tsquery", websearch_query)
        self.assertIn("ILIKE", fallback_query)
        self.assertEqual(websearch_params[0], "CONFIG_SELINUX procd-selinux busybox-selinux setfiles")
        first_fallback_param = fallback_params[0]
        self.assertIsInstance(first_fallback_param, list)
        self.assertIn("%CONFIG\\_SELINUX%", cast("list[object]", first_fallback_param))

    def test_text_search_auto_falls_back_to_any_terms_when_all_terms_misses(self) -> None:
        conn = QueuedConnection([
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[{"id": 11, "symbol": "setfiles"}]),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({"query": "missing setfiles"})

        payload = mcp_text_payload(response)
        self.assertEqual(payload["query_strategy"], "any_terms_fallback")
        self.assertEqual(payload["results"], [{"symbol": "setfiles"}])
        self.assertEqual(len(conn.calls), 3)
        self.assertIn("EXISTS", conn.calls[2][0])

    def test_text_search_explicit_websearch_mode_does_not_fallback(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({
                "query": "CONFIG_SELINUX procd-selinux",
                "query_mode": "websearch",
            })

        payload = mcp_text_payload(response)
        self.assertEqual(payload["query_strategy"], "websearch")
        self.assertEqual(payload["results"], [])
        self.assertEqual(len(conn.calls), 1)

    def test_text_search_rejects_empty_query_string(self) -> None:
        definition = TOOL_DEFINITIONS["search_code_intel_text"]
        with self.assertRaises(McpProtocolError) as ctx:
            validate_tool_arguments(definition, {"query": ""})
        self.assertIn("query", str(ctx.exception))

    def test_text_search_mode_search_requires_query(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            self.assertRaises(McpProtocolError) as ctx,
        ):
            _ = mcp_tools.tool_search_code_intel_text({"mode": "search"})
        self.assertIn("mode=search requires a non-empty query", str(ctx.exception))

    def test_text_search_mode_enumerate_rejects_query(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            self.assertRaises(McpProtocolError) as ctx,
        ):
            _ = mcp_tools.tool_search_code_intel_text({"mode": "enumerate", "query": "hello"})
        self.assertIn("mode=enumerate", str(ctx.exception))

    def test_explicit_missing_snapshot_id_raises(self) -> None:
        # First cursor responds to the existence probe with no row.
        conn = QueuedConnection([FakeCursor(one=None)])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
            self.assertRaises(McpProtocolError) as ctx,
        ):
            _ = mcp_tools.tool_list_code_intel_files({"snapshot_id": 9999})
        self.assertIn("9999", str(ctx.exception))

    def test_text_search_is_untracked_filter(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_search_code_intel_text({"is_untracked": False})
        query, params = conn.calls[0]
        self.assertIn("coalesce(f.is_untracked, false) = %s", query)
        self.assertTrue(any(p is False for p in params))

    def test_list_files_is_untracked_filter(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_list_code_intel_files({"is_untracked": False})
        query, params = conn.calls[0]
        self.assertIn("f.is_untracked = %s", query)
        self.assertTrue(any(p is False for p in params))

    def test_text_search_accepts_source_path_prefix(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_search_code_intel_text({"source_path_prefix": "cmd/"})
        query, params = conn.calls[0]
        self.assertIn("r.source_path LIKE %s ESCAPE", query)
        self.assertIn("cmd/%", params)

    def test_text_search_rejects_source_path_with_prefix(self) -> None:
        conn = QueuedConnection([FakeCursor(one=None)])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
            self.assertRaises(McpProtocolError),
        ):
            _ = mcp_tools.tool_search_code_intel_text({
                "source_path": "cmd/main.go",
                "source_path_prefix": "cmd",
            })

    def test_text_search_snippet_length_truncates_inline_snippet(self) -> None:
        long_body = "x" * 600
        snippet_raw = f"```go\n{long_body}\n```"
        conn = QueuedConnection([FakeCursor(many=[{"id": 1, "snippet_raw": snippet_raw}])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({"query": "x", "snippet_length": 50})
        payload = mcp_text_payload(response)
        results = cast("list[dict[str, object]]", payload["results"])
        self.assertEqual(len(cast("str", results[0]["snippet"])), 50)

    def test_text_search_snippet_length_rejects_out_of_range(self) -> None:
        definition = TOOL_DEFINITIONS["search_code_intel_text"]
        with self.assertRaises(McpProtocolError):
            validate_tool_arguments(definition, {"query": "x", "snippet_length": 0})
        with self.assertRaises(McpProtocolError):
            validate_tool_arguments(definition, {"query": "x", "snippet_length": 1000})

    def test_text_search_mode_echoed_only_when_explicitly_set(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({})
        payload = mcp_text_payload(response)
        self.assertNotIn("mode", payload)

        conn = QueuedConnection([FakeCursor(many=[])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({"mode": "enumerate"})
        payload = mcp_text_payload(response)
        self.assertEqual(payload["mode"], "enumerate")


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

    def test_jsonrpc_error_paths_preserve_request_id(self) -> None:
        request = {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {
                "name": "code_intel_status",
                "arguments": {"collection": "other"},
            },
        }

        with (
            patch.dict(os.environ, {"PROJECT_CODE_INTELLIGENCE_COLLECTION": "configured"}, clear=True),
            self.assertRaises(McpWritePermissionError),
        ):
            _ = handle_jsonrpc_value(request)

        self.assertEqual(request_id_from_jsonrpc_value(request), 42)

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

    def test_record_fetch_by_record_id_is_scoped_to_configured_collection(self) -> None:
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
            _ = mcp_tools.tool_get_code_intel_record({"record_id": "README.md::doc::000001"})

        query, params = conn.calls[0]
        self.assertIn("r.collection = %s", query)
        self.assertIn("r.record_id = %s", query)
        self.assertIn("project-code-intelligence", params)
        self.assertIn("README.md::doc::000001", params)

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
        # First query: existence check (returns a row so the code-flows query runs).
        # Second query: code-flows query with collection scoping.
        conn = QueuedConnection([FakeCursor(one={"id": 1}), FakeCursor(many=[])])

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

        existence_query, existence_params = conn.calls[0]
        self.assertIn("project_code_intel_static_findings", existence_query)
        self.assertEqual(existence_params, [1])

        query, params = conn.calls[1]
        self.assertIn("JOIN project_code_intel_static_findings f", query)
        self.assertIn("cf.finding_id = %s", query)
        self.assertIn("f.collection = %s", query)
        self.assertEqual(params, [1, "project-code-intelligence"])

    def test_static_finding_fetch_is_compact_by_default(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                one={
                    "id": 7,
                    "snapshot_id": 3,
                    "run_id": 11,
                    "rule_id": "cpp/example",
                    "message": "Finding message",
                    "raw_result": {"large": True},
                    "run_metadata": {"code_intel_warnings": [{"message": "SARIF may be stale"}]},
                }
            ),
            FakeCursor(one={"id": 13, "rule_id": "cpp/example", "properties": {"precision": "high"}}),
            FakeCursor(many=[]),
            FakeCursor(one={"code_flow_steps": 4}),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "table_regclass_exists", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_get_static_finding({"id": 7})

        payload = mcp_text_payload(response)
        self.assertEqual(payload["code_flow_steps"], 4)
        self.assertNotIn("code_flows", payload)
        self.assertNotIn("raw", payload)
        self.assertNotIn("run_metadata", payload)
        self.assertEqual(payload["warnings"], [{"message": "SARIF may be stale"}])
        finding = payload["finding"]
        self.assertIsInstance(finding, dict)
        self.assertNotIn("raw_result", cast("dict[str, object]", finding))

    def test_static_finding_fetch_can_include_large_diagnostics(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                one={
                    "id": 7,
                    "snapshot_id": 3,
                    "run_id": 11,
                    "rule_id": "cpp/example",
                    "message": "Finding message",
                    "raw_result": {"large": True},
                    "run_metadata": {"toolExecutionNotifications": [{"message": "debug"}]},
                }
            ),
            FakeCursor(one={"id": 13, "rule_id": "cpp/example", "properties": {"precision": "high"}}),
            FakeCursor(many=[]),
            FakeCursor(many=[{"flow_index": 0, "step_index": 0}]),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "table_regclass_exists", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_get_static_finding({
                "id": 7,
                "include_raw": True,
                "include_run_metadata": True,
                "include_code_flows": True,
            })

        payload = mcp_text_payload(response)
        self.assertIn("raw", payload)
        self.assertIn("run_metadata", payload)
        self.assertEqual(payload["code_flows"], [{"flow_index": 0, "step_index": 0}])

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
        validate_tool_arguments(text_search, {"query": "hello world", "query_mode": "all_terms"})

        with self.assertRaises(McpProtocolError):
            validate_tool_arguments(text_search, {"query": "hello", "surprise": True})
        with self.assertRaises(McpProtocolTypeError):
            validate_tool_arguments(text_search, {"query": "hello", "limit": "10"})
        with self.assertRaises(McpProtocolError):
            validate_tool_arguments(text_search, {"query": "hello", "limit": 500})
        with self.assertRaises(McpProtocolError):
            validate_tool_arguments(text_search, {"query": "hello", "query_mode": "broad"})

    def test_required_tool_arguments_are_enforced_before_handlers(self) -> None:
        record_fetch = TOOL_DEFINITIONS["get_code_intel_record"]

        # record_id and record_ids are both optional in the schema; the handler
        # enforces "exactly one." The schema still rejects malformed values.
        with self.assertRaises(McpProtocolTypeError):
            validate_tool_arguments(record_fetch, {"record_id": 1})
        with self.assertRaises(McpProtocolError):
            validate_tool_arguments(record_fetch, {"record_id": ""})
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(FakeConnection())),
            self.assertRaises(McpProtocolError),
        ):
            _ = mcp_tools.tool_get_code_intel_record({})
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(FakeConnection())),
            self.assertRaises(McpProtocolError),
        ):
            _ = mcp_tools.tool_get_code_intel_record({"record_id": "a", "record_ids": ["b"]})

    def test_tool_call_rejects_non_object_params_and_arguments(self) -> None:
        with self.assertRaises(McpProtocolTypeError):
            _ = handle_tool_call({"params": []}, None)
        with self.assertRaises(McpProtocolTypeError):
            _ = handle_tool_call({"params": {"name": "code_intel_status", "arguments": []}}, None)


class McpToolShapeTests(unittest.TestCase):
    def test_list_files_default_selects_slim_columns(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_list_code_intel_files({})

        query, _ = conn.calls[0]
        select_clause = query.split("FROM project_code_intel_files")[0]
        self.assertNotIn("f.snapshot_id", select_clause)
        self.assertNotIn("f.commit_sha", select_clause)
        self.assertNotIn("f.metadata", select_clause)
        self.assertNotIn("f.created_at", select_clause)
        self.assertIn("f.source_path", select_clause)
        self.assertIn("f.file_role", select_clause)

    def test_list_files_verbose_selects_all_columns(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_list_code_intel_files({"verbose": True})

        query, _ = conn.calls[0]
        select_clause = query.split("FROM project_code_intel_files")[0]
        self.assertIn("f.snapshot_id", select_clause)
        self.assertIn("f.commit_sha", select_clause)
        self.assertIn("f.metadata", select_clause)
        self.assertIn("f.created_at", select_clause)

    def test_list_files_rejects_legacy_include_metadata(self) -> None:
        definition = TOOL_DEFINITIONS["list_code_intel_files"]
        with self.assertRaises(McpProtocolError):
            validate_tool_arguments(definition, {"include_metadata": True})

    def test_compact_record_drops_null_envelope_fields(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                one={
                    "id": 1,
                    "record_id": "README.md::doc::000001",
                    "source_path": "README.md",
                    "title": "Overview",
                    "summary": "",
                    "symbol": None,
                    "symbol_kind": None,
                    "parent_record_id": None,
                    "analyzer": None,
                    "analyzer_version": None,
                    "rule_id": None,
                    "severity": None,
                    "tool": None,
                    "embedding_text": None,
                    "display_content": None,
                    "metadata": {"doc_headings": ["Overview"]},
                }
            ),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_get_code_intel_record({"record_id": "README.md::doc::000001"})

        payload = mcp_text_payload(response)
        result = cast("dict[str, object]", payload["result"])
        # Null-valued fields are dropped entirely; the agent infers their absence.
        for null_field in (
            "symbol",
            "symbol_kind",
            "parent_record_id",
            "analyzer",
            "analyzer_version",
            "embedding_text",
            "display_content",
        ):
            self.assertNotIn(null_field, result)
        self.assertEqual(result["record_id"], "README.md::doc::000001")
        self.assertEqual(result["title"], "Overview")

    def test_get_record_strips_doc_links_by_default(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                one={
                    "id": 1,
                    "record_id": "README.md::doc::000001",
                    "source_path": "README.md",
                    "metadata": {
                        "doc_headings": ["Overview"],
                        "doc_links": ["https://a", "https://b", "https://c"],
                        "doc_fenced_languages": ["bash"],
                    },
                }
            ),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_get_code_intel_record({"record_id": "README.md::doc::000001"})

        payload = mcp_text_payload(response)
        result = cast("dict[str, object]", payload["result"])
        self.assertNotIn("id", result)
        self.assertEqual(result["record_id"], "README.md::doc::000001")
        metadata = cast("dict[str, object]", result["metadata"])
        self.assertNotIn("doc_links", metadata)
        self.assertIn("doc_headings", metadata)
        self.assertIn("doc_fenced_languages", metadata)

    def test_list_files_source_path_prefix_matches_subtree(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_list_code_intel_files({"source_path_prefix": "cmd/"})

        query, params = conn.calls[0]
        self.assertIn("f.source_path LIKE %s ESCAPE", query)
        self.assertIn("cmd/%", params)

    def test_list_files_source_path_prefix_escapes_like_metacharacters(self) -> None:
        self.assertEqual(source_path_prefix_pattern("a%b_c\\d"), "a\\%b\\_c\\\\d/%")
        self.assertEqual(source_path_prefix_pattern("foo/"), "foo/%")
        self.assertEqual(source_path_prefix_pattern("foo"), "foo/%")

    def test_list_files_rejects_both_source_path_and_prefix(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            self.assertRaises(McpProtocolError),
        ):
            _ = mcp_tools.tool_list_code_intel_files({
                "source_path": "cmd/main.go",
                "source_path_prefix": "cmd",
            })

    def test_status_includes_language_and_directory_breakdowns(self) -> None:
        # code_intel_status executes several aggregation queries. We only care
        # about the new ones here; feed every cursor an empty result and check
        # the SQL.
        conn = QueuedConnection([FakeCursor(many=[]) for _ in range(10)])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "table_regclass_exists", return_value=False),
            patch.object(mcp_tools, "schema_migration_versions", return_value=[]),
            patch.object(mcp_tools.git_utils, "run_git", return_value="abc123"),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({})

        payload = mcp_text_payload(response)
        self.assertIn("language_breakdown", payload)
        self.assertIn("directory_breakdown", payload)
        queries = [call[0] for call in conn.calls]
        self.assertTrue(
            any("GROUP BY f.language" in q for q in queries),
            msg=f"no language breakdown SQL in {queries!r}",
        )
        self.assertTrue(
            any("string_to_array(f.source_path, '/')" in q for q in queries),
            msg=f"no directory breakdown SQL in {queries!r}",
        )

    def test_get_record_strips_embedding_text_by_default(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                one={
                    "id": 1,
                    "record_id": "src/lib.rs::function::handler::000001",
                    "source_path": "src/lib.rs",
                    "embedding_text": "fn handler() { ... }",
                    "embedding_text_truncated": True,
                    "display_content": "# handler\n\n```rust\nfn handler() { ... }\n```",
                }
            ),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_get_code_intel_record({
                "record_id": "src/lib.rs::function::handler::000001",
                "include_content": True,
            })

        payload = mcp_text_payload(response)
        result = cast("dict[str, object]", payload["result"])
        self.assertNotIn("embedding_text", result)
        self.assertNotIn("embedding_text_truncated", result)
        # display_content remains — it's the canonical rendering.
        self.assertIn("display_content", result)

    def test_get_record_verbose_keeps_embedding_text(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                one={
                    "id": 1,
                    "record_id": "src/lib.rs::function::handler::000001",
                    "source_path": "src/lib.rs",
                    "embedding_text": "fn handler() { ... }",
                    "embedding_text_truncated": True,
                    "display_content": "# handler\n\n```rust\nfn handler() { ... }\n```",
                }
            ),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_get_code_intel_record({
                "record_id": "src/lib.rs::function::handler::000001",
                "include_content": True,
                "verbose": True,
            })

        payload = mcp_text_payload(response)
        result = cast("dict[str, object]", payload["result"])
        self.assertEqual(result["embedding_text"], "fn handler() { ... }")
        self.assertEqual(result["embedding_text_truncated"], True)

    def test_get_record_batch_returns_results_and_missing(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {"id": 1, "record_id": "a::doc::000001", "source_path": "a.md"},
                    {"id": 2, "record_id": "b::doc::000001", "source_path": "b.md"},
                ]
            ),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_get_code_intel_record({
                "record_ids": ["a::doc::000001", "b::doc::000001", "missing::doc::000001"],
            })

        payload = mcp_text_payload(response)
        results = cast("list[dict[str, object]]", payload["results"])
        self.assertEqual({r["record_id"] for r in results}, {"a::doc::000001", "b::doc::000001"})
        self.assertEqual(payload["missing"], ["missing::doc::000001"])

        query, _ = conn.calls[0]
        self.assertIn("DISTINCT ON (r.record_id)", query)
        self.assertIn("r.record_id = ANY(%s::text[])", query)

    def test_get_record_verbose_keeps_doc_links_and_int_id(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                one={
                    "id": 1,
                    "record_id": "README.md::doc::000001",
                    "source_path": "README.md",
                    "metadata": {
                        "doc_headings": ["Overview"],
                        "doc_links": ["https://a", "https://b"],
                    },
                }
            ),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_get_code_intel_record({
                "record_id": "README.md::doc::000001",
                "verbose": True,
            })

        payload = mcp_text_payload(response)
        result = cast("dict[str, object]", payload["result"])
        self.assertEqual(result["id"], 1)
        metadata = cast("dict[str, object]", result["metadata"])
        self.assertEqual(metadata["doc_links"], ["https://a", "https://b"])


if __name__ == "__main__":
    _ = unittest.main()
