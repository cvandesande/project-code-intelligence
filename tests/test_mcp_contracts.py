from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from unittest.mock import patch

from project_code_intelligence import config
from project_code_intelligence.embedding.types import EmbeddingEndpointUnavailableError
from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError, McpWritePermissionError
from project_code_intelligence.mcp import tools as mcp_tools
from project_code_intelligence.mcp.filters import (
    code_intel_clauses,
    json_argument,
    source_path_clauses,
    source_path_prefix_pattern,
    static_finding_clauses,
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

    def __exit__(self, _exc_type: object, exc: object, traceback: object) -> None:
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
        self.assertEqual(mcp_tools.search_terms("$ZodLazy defineLazy"), ["$ZodLazy", "defineLazy"])
        self.assertEqual(mcp_tools.search_terms("alpha AND beta OR gamma"), ["alpha", "beta", "gamma"])
        self.assertEqual(mcp_tools.like_pattern_for_term("a_b%"), "%a\\_b\\%%")

    def test_text_search_auto_uses_term_matching_for_identifier_like_single_terms(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[{"id": 5, "symbol": "defineLazy"}])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({"query": "defineLazy"})

        payload = mcp_text_payload(response)
        self.assertEqual(payload["query_strategy"], "all_terms")
        self.assertEqual(payload["results"], [{"symbol": "defineLazy"}])
        self.assertEqual(len(conn.calls), 1)
        query, params = conn.calls[0]
        self.assertIn("ILIKE", query)
        self.assertNotIn("websearch_to_tsquery", query)
        self.assertEqual(params[3], ["%defineLazy%"])

    def test_text_search_auto_uses_term_matching_for_path_like_single_terms(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({"query": "procd-selinux"})

        payload = mcp_text_payload(response)
        self.assertEqual(payload["query_strategy"], "all_terms")
        query, params = conn.calls[0]
        self.assertIn("ILIKE", query)
        self.assertNotIn("websearch_to_tsquery", query)
        self.assertEqual(params[3], ["%procd-selinux%"])


class McpSearchRankingTests(unittest.TestCase):
    def test_identifier_search_boosts_config_symbol_aliases_when_present(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({"query": "KERNEL_SECURITY_SELINUX"})

        payload = mcp_text_payload(response)
        self.assertEqual(payload["query_strategy"], "all_terms")
        query, _params = conn.calls[0]
        score_sql = query[query.index("SELECT coalesce(sum(") : query.index(") AS match_score")]
        self.assertIn("r.record_type = 'config_symbol'", score_sql)
        self.assertIn("regexp_replace(coalesce(r.symbol, ''), '^CONFIG_', '', 'i')", score_sql)


class McpStaticFindingFilterTests(unittest.TestCase):
    def test_static_finding_source_path_accepts_repo_relative_paths(self) -> None:
        clauses, params = static_finding_clauses({
            "repo": "openwrt",
            "source_path": "build_dir/target-aarch64/ask-cmm-17.03.1/src/pppoe.c",
        })

        self.assertIn("f.primary_source_path = ANY(%s)", clauses)
        self.assertEqual(
            params[1],
            [
                "build_dir/target-aarch64/ask-cmm-17.03.1/src/pppoe.c",
                "openwrt/build_dir/target-aarch64/ask-cmm-17.03.1/src/pppoe.c",
            ],
        )

    def test_static_finding_source_path_prefix_matches_repo_relative_subtree(self) -> None:
        clauses, params = static_finding_clauses({
            "repo": "openwrt",
            "source_path_prefix": "build_dir/target-aarch64/ask-cmm-17.03.1",
        })

        self.assertIn("f.primary_source_path LIKE %s ESCAPE '\\'", clauses[2])
        self.assertEqual(
            params[1:3],
            [
                "build\\_dir/target-aarch64/ask-cmm-17.03.1/%",
                "openwrt/build\\_dir/target-aarch64/ask-cmm-17.03.1/%",
            ],
        )


class McpTextSearchExecutionTests(unittest.TestCase):
    def test_text_search_centers_snippet_around_matched_term(self) -> None:
        snippet_raw = (
            "```ts\n" + ("a" * 180) + "\nthrow new Error('Duplicate schema id found')\n" + ("b" * 180) + "\n```"
        )
        conn = QueuedConnection([FakeCursor(many=[{"id": 5, "snippet_raw": snippet_raw, "match_score": 99}])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({
                "query": "Duplicate schema id",
                "snippet_length": 80,
            })

        payload = mcp_text_payload(response)
        results = cast("list[dict[str, object]]", payload["results"])
        snippet_obj = results[0]["snippet"]
        self.assertIsInstance(snippet_obj, str)
        snippet = cast("str", snippet_obj)
        self.assertIn("Duplicate schema id", snippet)
        self.assertLessEqual(len(snippet), 80)
        self.assertNotIn("match_score", results[0])

    def test_websearch_ranking_uses_language_neutral_match_score(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({"query": "parse", "query_mode": "websearch"})

        payload = mcp_text_payload(response)
        self.assertEqual(payload["query_strategy"], "websearch")
        query, params = conn.calls[0]
        self.assertIn("AS match_score", query)
        self.assertIn("ORDER BY rank DESC, match_score DESC, r.updated_at DESC", query)
        score_sql = query[query.index("SELECT coalesce(sum(") : query.index(") AS match_score")]
        self.assertIn("r.symbol", score_sql)
        self.assertIn("r.title", score_sql)
        self.assertIn("r.source_path", score_sql)
        self.assertNotIn("r.language", score_sql)
        self.assertNotIn(".ts", score_sql)
        self.assertNotIn("typescript", score_sql.lower())
        self.assertNotIn("javascript", score_sql.lower())
        self.assertEqual(params[1], ["parse"])
        self.assertEqual(params[2], ["parse%"])
        self.assertEqual(params[3], ["%parse%"])

    def test_text_search_excludes_security_patterns_from_broad_results(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({"query": "schema"})

        payload = mcp_text_payload(response)
        self.assertEqual(payload["results"], [])
        query, _params = conn.calls[0]
        self.assertIn("r.record_type <> 'security_pattern'", query)

    def test_text_search_record_type_filter_can_request_security_patterns(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({
                "query": "shell_backtick_execution",
                "record_type": "security_pattern",
            })

        payload = mcp_text_payload(response)
        self.assertEqual(payload["results"], [])
        query, _params = conn.calls[0]
        self.assertNotIn("r.record_type <> 'security_pattern'", query)

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
        first_fallback_param = fallback_params[3]
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

    def test_source_path_filters_accept_repo_relative_paths_when_repo_is_known(self) -> None:
        clauses, params = source_path_clauses({"repo": "zod", "source_path": "packages/zod/src/index.ts"}, "r")

        self.assertEqual(clauses, ["r.source_path = ANY(%s)"])
        self.assertEqual(params, [["packages/zod/src/index.ts", "zod/packages/zod/src/index.ts"]])

        clauses, params = source_path_clauses({"repo": "zod", "source_path_prefix": "packages/zod/src"}, "r")

        self.assertEqual(
            clauses,
            ["(r.source_path LIKE %s ESCAPE '\\' OR r.source_path LIKE %s ESCAPE '\\')"],
        )
        self.assertEqual(params, ["packages/zod/src/%", "zod/packages/zod/src/%"])

    def test_source_path_filters_accept_repo_relative_paths_without_repo_filter(self) -> None:
        clauses, params = source_path_clauses({"source_path": "packages/zod/src/index.ts"}, "r")

        self.assertEqual(clauses, ["(r.source_path = %s OR r.source_path LIKE %s ESCAPE '\\')"])
        self.assertEqual(params, ["packages/zod/src/index.ts", "%/packages/zod/src/index.ts"])

        clauses, params = source_path_clauses({"source_path_prefix": "packages/zod/src"}, "r")

        self.assertEqual(
            clauses,
            ["(r.source_path LIKE %s ESCAPE '\\' OR r.source_path LIKE %s ESCAPE '\\')"],
        )
        self.assertEqual(params, ["packages/zod/src/%", "%/packages/zod/src/%"])

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


class McpSemanticSearchTests(unittest.TestCase):
    def test_semantic_search_excludes_security_patterns_from_broad_results(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_semantic({"query": "break circular import"})

        payload = mcp_text_payload(response)
        self.assertEqual(payload["results"], [])
        query, _params = conn.calls[0]
        self.assertIn("r.record_type <> 'security_pattern'", query)

    def test_semantic_search_record_type_filter_can_request_security_patterns(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_semantic({
                "query": "shell command execution",
                "record_type": "security_pattern",
            })

        payload = mcp_text_payload(response)
        self.assertEqual(payload["results"], [])
        query, _params = conn.calls[0]
        self.assertNotIn("r.record_type <> 'security_pattern'", query)


class McpRelatedCodeIntelTests(unittest.TestCase):
    def test_related_accepts_direction_argument(self) -> None:
        definition = TOOL_DEFINITIONS["related_code_intel"]

        validate_tool_arguments(definition, {"symbol": "parse", "direction": "outgoing"})

        with self.assertRaises(McpProtocolError):
            validate_tool_arguments(definition, {"symbol": "parse", "direction": "sideways"})

    def test_related_direction_filters_symbol_edges(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_related_code_intel({"symbol": "parse", "direction": "outgoing"})

        query, params = conn.calls[0]
        self.assertIn("e.source_symbol = %s", query)
        self.assertNotIn("e.target_symbol = %s", query)
        self.assertEqual(params[0], "parse")

    def test_related_record_id_uses_parent_edges_for_chunks(self) -> None:
        chunk_id = "src/app.ts::function_chunk::build::000001"
        parent_id = "src/app.ts::function::build::000001"
        conn = QueuedConnection([
            FakeCursor(one={"parent_record_id": parent_id}),
            FakeCursor(
                many=[
                    {
                        "id": 1,
                        "source_record_id": parent_id,
                        "target_record_id": "src/app.ts::function::helper::000010",
                        "edge_type": "call_candidate",
                        "source_symbol": "build",
                        "target_symbol": "helper",
                        "target_record_db_id": 9,
                    }
                ]
            ),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"record_id": chunk_id})

        payload = mcp_text_payload(response)
        edges = cast("list[dict[str, object]]", payload["edges"])
        self.assertEqual(edges[0]["edge_source"], "parent_record")
        self.assertEqual(edges[0]["direction"], "outgoing")
        self.assertTrue(edges[0]["target_resolved"])
        _lookup_query, lookup_params = conn.calls[0]
        edge_query, edge_params = conn.calls[1]
        self.assertEqual(lookup_params[-1], chunk_id)
        self.assertIn("e.source_record_id = ANY(%s)", edge_query)
        self.assertIn("e.target_record_id = ANY(%s)", edge_query)
        self.assertIn([chunk_id, parent_id], edge_params)

    def test_related_symbol_defaults_to_incoming_resolved_callers_first(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "id": 1,
                        "source_record_id": "src/util.ts::function::defineLazy::000001",
                        "target_record_id": None,
                        "edge_type": "call_candidate",
                        "source_symbol": "defineLazy",
                        "target_symbol": "getter",
                    },
                    {
                        "id": 2,
                        "source_record_id": "src/schema.ts::constant::ObjectSchema::000010",
                        "target_record_id": "src/util.ts::function::defineLazy::000001",
                        "target_record_db_id": 7,
                        "edge_type": "call_candidate",
                        "source_symbol": "ObjectSchema",
                        "target_symbol": "defineLazy",
                    },
                ]
            )
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"symbol": "defineLazy"})

        payload = mcp_text_payload(response)
        edges = cast("list[dict[str, object]]", payload["edges"])
        self.assertEqual(edges[0]["source_symbol"], "ObjectSchema")
        self.assertEqual(edges[0]["direction"], "incoming")
        self.assertTrue(edges[0]["target_resolved"])
        self.assertEqual(edges[0]["target_kind"], "project_symbol")
        self.assertEqual(edges[1]["target_symbol"], "getter")
        self.assertEqual(edges[1]["direction"], "outgoing")
        self.assertFalse(edges[1]["target_resolved"])
        query, _params = conn.calls[0]
        self.assertIn("WHEN e.target_symbol = %s AND tgt.id IS NOT NULL THEN 0", query)

    def test_related_edges_rank_resolved_targets_above_unresolved(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "id": 2,
                        "source_record_id": "src/app.ts::function::bootstrap::000001",
                        "target_record_id": None,
                        "edge_type": "call_candidate",
                        "source_symbol": "bootstrap",
                        "target_symbol": "getter",
                    },
                    {
                        "id": 1,
                        "source_record_id": "src/app.ts::function::bootstrap::000001",
                        "target_record_id": "src/app.ts::function::parse::000020",
                        "target_record_db_id": 8,
                        "edge_type": "call_candidate",
                        "source_symbol": "bootstrap",
                        "target_symbol": "parse",
                    },
                ]
            )
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"symbol": "bootstrap", "direction": "outgoing"})

        payload = mcp_text_payload(response)
        edges = cast("list[dict[str, object]]", payload["edges"])
        self.assertEqual(edges[0]["target_symbol"], "parse")
        self.assertTrue(edges[0]["target_resolved"])
        self.assertEqual(edges[1]["target_symbol"], "getter")
        self.assertFalse(edges[1]["target_resolved"])

    def test_related_labels_unresolved_member_calls(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "id": 1,
                        "source_record_id": "src/schema.ts::function::parse::000001",
                        "target_record_id": None,
                        "edge_type": "call_candidate",
                        "source_symbol": "parse",
                        "target_symbol": "run",
                        "metadata": {"call_kind": "member_call", "target_resolvable": False},
                    },
                ]
            )
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"symbol": "parse", "direction": "outgoing"})

        payload = mcp_text_payload(response)
        edge = cast("list[dict[str, object]]", payload["edges"])[0]
        self.assertFalse(edge["target_resolved"])
        self.assertEqual(edge["target_kind"], "member_call")

    def test_related_default_edge_shape_is_compact_but_navigable(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "id": 2,
                        "snapshot_id": 1,
                        "collection": "zod",
                        "repo": "zod",
                        "commit_sha": "abc123",
                        "source_record_id": "src/app.ts::function::bootstrap::000001",
                        "target_record_id": "src/app.ts::function::parse::000020",
                        "edge_type": "call_candidate",
                        "source_symbol": "bootstrap",
                        "target_symbol": "parse",
                        "source_path": "src/app.ts",
                        "target_path": "src/app.ts",
                        "confidence_kind": "heuristic_candidate",
                        "metadata": {"heavy": "debug"},
                        "target_resolved": True,
                        "target_kind": "project_symbol",
                        "source_record_db_id": 7,
                        "source_summary": "source summary",
                        "source_line_start": 10,
                        "source_line_end": 20,
                        "target_record_db_id": 8,
                        "target_summary": "target summary",
                        "target_line_start": 30,
                        "target_line_end": 40,
                    },
                ]
            )
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"symbol": "bootstrap", "direction": "outgoing"})

        payload = mcp_text_payload(response)
        edge = cast("list[dict[str, object]]", payload["edges"])[0]
        self.assertEqual(edge["source_line_start"], 10)
        self.assertEqual(edge["target_line_end"], 40)
        self.assertEqual(edge["target_kind"], "project_symbol")
        self.assertNotIn("source_summary", edge)
        self.assertNotIn("target_summary", edge)
        self.assertNotIn("source_record_db_id", edge)
        self.assertNotIn("metadata", edge)

    def test_related_symbol_handles_dollar_prefixed_project_symbols(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "id": 10,
                        "source_record_id": "src/mini.ts::constant::MiniLazy::000001",
                        "target_record_id": "src/core.ts::constant::$BaseLazy::000001",
                        "target_record_db_id": 12,
                        "edge_type": "call_candidate",
                        "source_symbol": "MiniLazy",
                        "target_symbol": "$BaseLazy",
                    },
                    {
                        "id": 9,
                        "source_record_id": "src/classic.ts::constant::ClassicLazy::000001",
                        "target_record_id": "src/core.ts::constant::$BaseLazy::000001",
                        "target_record_db_id": 12,
                        "edge_type": "call_candidate",
                        "source_symbol": "ClassicLazy",
                        "target_symbol": "$BaseLazy",
                    },
                ]
            )
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"symbol": "$BaseLazy"})

        payload = mcp_text_payload(response)
        sources = [edge["source_symbol"] for edge in cast("list[dict[str, object]]", payload["edges"])]
        self.assertEqual(sources, ["MiniLazy", "ClassicLazy"])


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
                    self.assertEqual(os.environ[config.DATABASE_SCOPE_PATH_ENV], str(workspace.resolve()))
                    self.assertEqual(os.environ["PROJECT_CODE_INTELLIGENCE_COLLECTION_DEFAULTED"], "1")
            finally:
                os.chdir(old_cwd)

    def test_mcp_keeps_explicit_collection_override(self) -> None:
        with patch.dict(os.environ, {"PROJECT_CODE_INTELLIGENCE_COLLECTION": "configured"}, clear=True):
            set_mcp_environment_defaults()
            self.assertEqual(os.environ["PROJECT_CODE_INTELLIGENCE_COLLECTION"], "configured")
            self.assertEqual(os.environ[config.DATABASE_SCOPE_PATH_ENV], str(Path.cwd().resolve()))
            self.assertNotIn("PROJECT_CODE_INTELLIGENCE_COLLECTION_DEFAULTED", os.environ)

        with patch.dict(os.environ, {config.DATABASE_SCOPE_PATH_ENV: "configured-scope"}, clear=True):
            set_mcp_environment_defaults()
            self.assertEqual(os.environ[config.DATABASE_SCOPE_PATH_ENV], "configured-scope")

    def test_repo_filter_skips_defaulted_collection_scope(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PROJECT_CODE_INTELLIGENCE_COLLECTION": "project-code-intelligence",
                "PROJECT_CODE_INTELLIGENCE_COLLECTION_DEFAULTED": "1",
            },
            clear=True,
        ):
            clauses, params = code_intel_clauses({"repo": "zod"}, "r")

        self.assertNotIn("r.collection = %s", clauses)
        self.assertIn("r.repo = %s", clauses)
        self.assertIn("zod", params)

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

    def test_status_default_uses_compact_freshness_snapshots(self) -> None:
        created_at = datetime.now(timezone.utc) - timedelta(seconds=42)
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "id": 1,
                        "collection": "zod",
                        "repo": "zod",
                        "repo_role": "project",
                        "branch": "main",
                        "commit_sha": "abc123",
                        "tree_sha": "tree123",
                        "dirty": False,
                        "metadata": {"embed_record_types": ["code_chunk"], "large": "omitted"},
                        "created_at": created_at,
                    }
                ]
            ),
            FakeCursor(many=[]),
            FakeCursor(many=[{"record_type": "code_chunk", "count": 1, "embedded_records": 1}]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
        ])

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
        snapshots = cast("list[dict[str, object]]", payload["snapshots"])
        self.assertEqual(snapshots[0]["repo"], "zod")
        self.assertEqual(snapshots[0]["branch"], "main")
        self.assertEqual(snapshots[0]["commit_sha"], "abc123")
        self.assertFalse(cast("bool", snapshots[0]["dirty"]))
        self.assertEqual(snapshots[0]["head_status"], "current")
        self.assertIsInstance(snapshots[0]["index_age_seconds"], int)
        self.assertNotIn("metadata", snapshots[0])
        self.assertNotIn("tree_sha", snapshots[0])
        self.assertNotIn("records_by_type", payload)
        self.assertNotIn("language_breakdown", payload)
        self.assertNotIn("static_findings", payload)

    def test_status_omits_redundant_collection_when_scoped(self) -> None:
        conn = QueuedConnection([
            FakeCursor(many=[{"id": 1, "collection": "zod", "repo": "zod", "metadata": {}}]),
            FakeCursor(many=[{"collection": "zod", "repo": "zod", "records": 7, "embedded_records": 5}]),
            FakeCursor(many=[{"collection": "zod", "repo": "zod", "record_type": "code_chunk", "count": 7}]),
            FakeCursor(many=[{"collection": "zod", "repo": "zod", "files": 3, "skipped_files": 0}]),
            FakeCursor(many=[{"collection": "zod", "repo": "zod", "edges": 2}]),
            FakeCursor(many=[]),
        ])

        with (
            patch.dict(os.environ, {"PROJECT_CODE_INTELLIGENCE_COLLECTION": "zod"}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "table_regclass_exists", return_value=False),
            patch.object(mcp_tools, "schema_migration_versions", return_value=[]),
            patch.object(mcp_tools.git_utils, "run_git", return_value="abc123"),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({})

        payload = mcp_text_payload(response)
        self.assertEqual(payload["collection"], "zod")
        for row_set in ("snapshots", "files", "records", "edges"):
            rows = cast("list[dict[str, object]]", payload[row_set])
            self.assertNotIn("collection", rows[0])
            self.assertEqual(rows[0]["repo"], "zod")

    def test_status_verbose_preserves_full_sections(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[{"id": 1, "collection": "zod", "commit_sha": "abc123", "metadata": {"embed_record_types": []}}]
            ),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
        ])

        with (
            patch.dict(os.environ, {"PROJECT_CODE_INTELLIGENCE_COLLECTION": "zod"}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "table_regclass_exists", return_value=False),
            patch.object(mcp_tools, "schema_migration_versions", return_value=[]),
            patch.object(mcp_tools.git_utils, "run_git", return_value="abc123"),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({"verbose": True})

        payload = mcp_text_payload(response)
        snapshots = cast("list[dict[str, object]]", payload["snapshots"])
        self.assertEqual(snapshots[0]["collection"], "zod")
        self.assertIn("metadata", snapshots[0])
        self.assertIn("records_by_type", payload)
        self.assertIn("language_breakdown", payload)
        self.assertIn("directory_breakdown", payload)
        self.assertIn("static_runs", payload)
        self.assertIn("static_findings", payload)

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
            response = mcp_tools.tool_code_intel_status({"include_breakdowns": True})

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

    def test_status_reports_compact_queryability_surface(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "id": 1,
                        "commit_sha": "abc123",
                        "metadata": {
                            "embed_record_types": ["code_chunk", "resource_object", "security_pattern"],
                        },
                    }
                ]
            ),
            FakeCursor(many=[]),
            FakeCursor(
                many=[
                    {"record_type": "code_chunk", "count": 10, "embedded_records": 10},
                    {"record_type": "resource_object", "count": 2, "embedded_records": 0},
                    {"record_type": "security_pattern", "count": 4, "embedded_records": 4},
                ]
            ),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[{"edge_type": "call_candidate", "edges": 3}]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
        ])

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
        self.assertEqual(
            payload["queryability"],
            {
                "text_record_type_count": 3,
                "semantic_record_type_count": 2,
                "text_only_record_type_count": 1,
                "configured_embed_record_type_count": 3,
                "empty_embed_record_type_count": 1,
                "edge_type_count": 1,
                "has_text": True,
                "has_semantic": True,
                "has_edges": True,
            },
        )

    def test_status_can_include_queryability_record_type_lists(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "id": 1,
                        "commit_sha": "abc123",
                        "metadata": {
                            "embed_record_types": ["code_chunk", "resource_object", "security_pattern"],
                        },
                    }
                ]
            ),
            FakeCursor(many=[]),
            FakeCursor(
                many=[
                    {"record_type": "code_chunk", "count": 10, "embedded_records": 10},
                    {"record_type": "resource_object", "count": 2, "embedded_records": 0},
                    {"record_type": "security_pattern", "count": 4, "embedded_records": 4},
                ]
            ),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[{"edge_type": "call_candidate", "edges": 3}]),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "table_regclass_exists", return_value=False),
            patch.object(mcp_tools, "schema_migration_versions", return_value=[]),
            patch.object(mcp_tools.git_utils, "run_git", return_value="abc123"),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({"include_queryability": True})

        payload = mcp_text_payload(response)
        self.assertEqual(
            payload["queryability"],
            {
                "text_record_type_count": 3,
                "semantic_record_type_count": 2,
                "text_only_record_type_count": 1,
                "configured_embed_record_type_count": 3,
                "empty_embed_record_type_count": 1,
                "edge_type_count": 1,
                "has_text": True,
                "has_semantic": True,
                "has_edges": True,
                "text_record_types": ["code_chunk", "resource_object", "security_pattern"],
                "semantic_record_types": ["code_chunk", "security_pattern"],
                "text_only_record_types": ["resource_object"],
                "configured_embed_record_types": ["code_chunk", "resource_object", "security_pattern"],
                "empty_embed_record_types": ["resource_object"],
                "edge_types": ["call_candidate"],
            },
        )

    def test_status_head_match_is_unknown_when_local_repo_is_unavailable(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "id": 1,
                        "collection": "zod",
                        "repo": "zod",
                        "commit_sha": "b6071fc0",
                        "metadata": {},
                    }
                ]
            ),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "table_regclass_exists", return_value=False),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
            patch.object(mcp_tools.Path, "cwd", return_value=Path("/work/project-code-intelligence")),
            patch.object(mcp_tools.git_utils, "run_git", return_value=None) as run_git,
        ):
            response = mcp_tools.tool_code_intel_status({"collection": "zod", "repo": "zod"})

        payload = mcp_text_payload(response)
        snapshots = cast("list[dict[str, object]]", payload["snapshots"])
        self.assertIsNone(snapshots[0]["head_commit"])
        self.assertIsNone(snapshots[0]["head_matches_snapshot"])
        self.assertEqual(snapshots[0]["head_status"], "unknown")
        self.assertEqual(snapshots[0]["head_status_reason"], "local_repo_unavailable")
        run_git.assert_any_call(Path("/work/project-code-intelligence/zod"), ["rev-parse", "HEAD"])
        run_git.assert_any_call(Path("/work/zod"), ["rev-parse", "HEAD"])

    def test_status_head_match_checks_sibling_repo_checkout(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "id": 1,
                        "collection": "zod",
                        "repo": "zod",
                        "commit_sha": "b6071fc0",
                        "metadata": {},
                    }
                ]
            ),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_tools, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "table_regclass_exists", return_value=False),
            patch.object(mcp_tools.mcp_db, "connect", return_value=FakeConnect(conn)),
            patch.object(mcp_tools.Path, "cwd", return_value=Path("/work/project-code-intelligence")),
            patch.object(mcp_tools.git_utils, "run_git", side_effect=[None, "b6071fc0"]) as run_git,
        ):
            response = mcp_tools.tool_code_intel_status({"collection": "zod", "repo": "zod"})

        payload = mcp_text_payload(response)
        snapshots = cast("list[dict[str, object]]", payload["snapshots"])
        self.assertEqual(snapshots[0]["head_commit"], "b6071fc0")
        self.assertTrue(snapshots[0]["head_matches_snapshot"])
        self.assertEqual(snapshots[0]["head_status"], "current")
        run_git.assert_any_call(Path("/work/zod"), ["rev-parse", "HEAD"])

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
