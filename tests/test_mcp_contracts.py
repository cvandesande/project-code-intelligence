from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from typing_extensions import override

from project_code_intelligence import analyze, config, git_utils
from project_code_intelligence import db as pci_db
from project_code_intelligence import server as mcp_server
from project_code_intelligence.embedding.types import EmbeddingEndpointUnavailableError
from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError, McpWritePermissionError
from project_code_intelligence.mcp import db as mcp_db
from project_code_intelligence.mcp import search as mcp_search
from project_code_intelligence.mcp import semantic as mcp_semantic
from project_code_intelligence.mcp import status as mcp_status
from project_code_intelligence.mcp import tools as mcp_tools
from project_code_intelligence.mcp import transport as mcp_transport
from project_code_intelligence.mcp.filters import (
    code_intel_clauses,
    json_argument,
    normalize_source_path_filter,
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
from project_code_intelligence.mcp.tool_inputs import TOOL_INPUT_MODELS
from project_code_intelligence.mcp.transport import (
    error_message,
    handle_jsonrpc_value,
    handle_tool_call,
    request_id_from_jsonrpc_value,
    set_mcp_environment_defaults,
)
from project_code_intelligence.server import vector_literal_dimensions

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject


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


def _git_show_current_branch_returns_main(_root: object, args: list[str]) -> str:
    """Fake ``run_git``: fixture snapshots below are stamped branch "main", so the
    live checkout must answer the same for `branch --show-current` calls, else the
    branch_mismatch check in mcp/status.py fires spuriously. Other git calls
    (rev-parse HEAD, rev-list --count) keep answering "abc123" like the old
    ``return_value="abc123"`` stub did.
    """
    return "main" if args[:2] == ["branch", "--show-current"] else "abc123"


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


def _mcp_defaults_for_empty_env() -> tuple[str, str, str]:
    with patch.dict(os.environ, {}, clear=True):
        set_mcp_environment_defaults()
        return (
            os.environ["PCI_COLLECTION"],
            os.environ[config.DATABASE_SCOPE_PATH_ENV],
            os.environ["PCI_COLLECTION_DEFAULTED"],
        )


class McpTextSearchTests(unittest.TestCase):
    def test_text_search_terms_extract_code_identifiers(self) -> None:
        self.assertEqual(
            mcp_search.search_terms("CONFIG_SELINUX procd-selinux busybox-selinux setfiles"),
            ["CONFIG_SELINUX", "procd-selinux", "busybox-selinux", "setfiles"],
        )
        self.assertEqual(mcp_search.search_terms("$ZodLazy defineLazy"), ["$ZodLazy", "defineLazy"])
        self.assertEqual(mcp_search.search_terms("alpha AND beta OR gamma"), ["alpha", "beta", "gamma"])
        self.assertEqual(mcp_search.like_pattern_for_term("a_b%"), "%a\\_b\\%%")

    def test_text_search_auto_uses_term_matching_for_identifier_like_single_terms(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[{"id": 5, "symbol": "defineLazy"}])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
        with patch.dict(os.environ, {}, clear=True):
            clauses, params = static_finding_clauses({
                "repo": "firmware",
                "source_path": "build_dir/target-aarch64/ask-cmm-17.03.1/src/pppoe.c",
            })

        self.assertIn("f.primary_source_path = ANY(%s)", clauses)
        self.assertEqual(
            params[1],
            [
                "build_dir/target-aarch64/ask-cmm-17.03.1/src/pppoe.c",
                "firmware/build_dir/target-aarch64/ask-cmm-17.03.1/src/pppoe.c",
            ],
        )

    def test_static_finding_source_path_prefix_matches_repo_relative_subtree(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            clauses, params = static_finding_clauses({
                "repo": "firmware",
                "source_path_prefix": "build_dir/target-aarch64/ask-cmm-17.03.1",
            })

        self.assertIn("f.primary_source_path LIKE %s ESCAPE '\\'", clauses[2])
        self.assertEqual(
            params[1:3],
            [
                "build\\_dir/target-aarch64/ask-cmm-17.03.1/%",
                "firmware/build\\_dir/target-aarch64/ask-cmm-17.03.1/%",
            ],
        )

    def test_static_finding_search_reports_when_no_static_run_exists(self) -> None:
        conn = QueuedConnection([
            FakeCursor(many=[]),
            FakeCursor(one={"repo": "kubernetes-ingress"}),
            FakeCursor(one=None),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_static_findings({"repo": "kubernetes-ingress"})

        payload = mcp_text_payload(response)
        self.assertEqual(payload["results"], [])
        self.assertFalse(payload["static_runs_found"])
        warnings = cast("list[dict[str, object]]", payload["warnings"])
        self.assertEqual(warnings[0]["kind"], "static_analysis_not_run")
        run_query, run_params = conn.calls[2]
        self.assertIn("FROM project_code_intel_static_runs r", run_query)
        self.assertEqual(run_params, ["kubernetes-ingress"])


class McpTextSearchRankingTests(unittest.TestCase):
    def test_text_search_exact_symbol_ranking_prefers_same_case_symbols(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({
                "query": "NewConfigurator",
                "record_type": "symbol_definition",
            })

        payload = mcp_text_payload(response)
        self.assertEqual(payload["query_strategy"], "all_terms")
        query, _params = conn.calls[0]
        score_sql = query[query.index("SELECT coalesce(sum(") : query.index(") AS match_score")]
        self.assertIn("coalesce(r.symbol, '') = search_terms.term THEN 120", score_sql)
        self.assertIn("lower(coalesce(r.symbol, '')) = lower(search_terms.term) THEN 80", score_sql)
        self.assertIn("ORDER BY match_score DESC", query)

    def test_text_search_warns_when_query_looks_like_regex(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({
                "query": r"func \(.*\) AddOrUpdateVirtualServer",
                "query_mode": "all_terms",
            })

        payload = mcp_text_payload(response)
        warnings = cast("list[dict[str, object]]", payload["warnings"])
        self.assertEqual(warnings[0]["kind"], "tokenized_text_search")


class McpTextSearchExecutionTests(unittest.TestCase):
    def test_text_search_centers_snippet_around_matched_term(self) -> None:
        snippet_raw = (
            "```ts\n" + ("a" * 180) + "\nthrow new Error('Duplicate schema id found')\n" + ("b" * 180) + "\n```"
        )
        conn = QueuedConnection([FakeCursor(many=[{"id": 5, "snippet_raw": snippet_raw, "match_score": 99}])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({
                "query": "shell_backtick_execution",
                "record_type": "security_pattern",
            })

        payload = mcp_text_payload(response)
        self.assertEqual(payload["results"], [])
        query, _params = conn.calls[0]
        self.assertNotIn("r.record_type <> 'security_pattern'", query)

    def test_text_search_content_class_filter_still_excludes_security_patterns(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({
                "query": "cooperative scheduling budget",
                "query_mode": "websearch",
                "content_class": "source",
            })

        payload = mcp_text_payload(response)
        self.assertEqual(payload["results"], [])
        query, _params = conn.calls[0]
        self.assertIn("r.record_type <> 'security_pattern'", query)

    def test_text_search_security_query_can_request_security_patterns_by_intent(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({
                "query": "security vulnerability",
                "query_mode": "websearch",
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
        self.assertEqual(
            payload["warnings"],
            [
                {
                    "kind": "query_strategy_fallback",
                    "query_strategy": "all_terms_fallback",
                    "message": "text search used a broader fallback strategy; ranking may be less precise",
                    "fallback_reason": "websearch returned no results for a multi-term query",
                }
            ],
        )
        self.assertEqual(
            payload["results"],
            [{"source_path": "config/Config-build.in", "repo_path": "config/Config-build.in"}],
        )

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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({"query": "missing setfiles"})

        payload = mcp_text_payload(response)
        self.assertEqual(payload["query_strategy"], "any_terms_fallback")
        warnings = cast("list[dict[str, object]]", payload["warnings"])
        self.assertEqual(warnings[0]["kind"], "query_strategy_fallback")
        self.assertEqual(warnings[0]["query_strategy"], "any_terms_fallback")
        self.assertEqual(payload["results"], [{"symbol": "setfiles"}])
        self.assertEqual(len(conn.calls), 3)
        self.assertIn("EXISTS", conn.calls[2][0])

    def test_text_search_explicit_websearch_mode_does_not_fallback(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({
                "query": "CONFIG_SELINUX procd-selinux",
                "query_mode": "websearch",
            })

        payload = mcp_text_payload(response)
        self.assertEqual(payload["query_strategy"], "websearch")
        self.assertEqual(payload["results"], [])
        self.assertEqual(len(conn.calls), 1)

    def test_text_search_treats_empty_optional_query_string_as_omitted(self) -> None:
        definition = TOOL_DEFINITIONS["search_code_intel_text"]
        _ = validate_tool_arguments(definition, {"query": ""})
        _ = validate_tool_arguments(definition, {"mode": "enumerate", "query": ""})

        conn = QueuedConnection([FakeCursor(many=[])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({"mode": "enumerate", "query": ""})

        payload = mcp_text_payload(response)
        self.assertIsNone(payload["query"])
        self.assertEqual(payload["mode"], "enumerate")

    def test_text_search_mode_search_requires_query(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            self.assertRaises(McpProtocolError) as ctx,
        ):
            _ = mcp_tools.tool_search_code_intel_text({"mode": "search"})
        self.assertIn("mode=search requires a non-empty query", str(ctx.exception))

    def test_text_search_mode_enumerate_rejects_query(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            self.assertRaises(McpProtocolError) as ctx,
        ):
            _ = mcp_tools.tool_search_code_intel_text({"mode": "enumerate", "query": "hello"})
        self.assertIn("lists records by filters", str(ctx.exception))

    def test_explicit_missing_snapshot_id_emits_warning_not_error(self) -> None:
        # First cursor responds to the snapshot existence probe with no row; the second is the
        # main list query that returns no rows because the snapshot doesn't exist.
        conn = QueuedConnection([FakeCursor(one=None), FakeCursor(many=[])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_list_code_intel_files({"snapshot_id": 9999})
        payload = mcp_text_payload(response)
        self.assertEqual(payload.get("files"), [])
        warnings = cast("list[dict[str, object]]", payload.get("warnings", []))
        snapshot_warnings = [w for w in warnings if w.get("kind") == "empty_snapshot_scope"]
        self.assertEqual(len(snapshot_warnings), 1)
        self.assertEqual(snapshot_warnings[0].get("snapshot_id"), 9999)

    def test_text_search_is_untracked_filter(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_search_code_intel_text({"is_untracked": False, "is_generated": False})
        query, params = conn.calls[0]
        self.assertIn("coalesce(f.is_untracked, false) = %s", query)
        self.assertIn("coalesce(f.is_generated, false) = %s", query)
        self.assertTrue(any(p is False for p in params))

    def test_semantic_search_is_untracked_filter_has_files_join(self) -> None:
        conn = QueuedConnection([
            FakeCursor(one={"exists": 1}),
            FakeCursor(many=[]),
            FakeCursor(one={"exists": 1}),
        ])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_search_code_intel_semantic({
                "query": "MCP transport",
                "repo": "project-code-intelligence",
                "snapshot_id": 1,
                "is_untracked": False,
                "is_generated": False,
            })

        query, params = conn.calls[1]
        self.assertIn("LEFT JOIN project_code_intel_files f", query)
        self.assertIn("coalesce(f.is_untracked, false) = %s", query)
        self.assertIn("coalesce(f.is_generated, false) = %s", query)
        self.assertTrue(any(p is False for p in params))

    def test_list_files_is_untracked_filter(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_list_code_intel_files({"is_untracked": False})
        query, params = conn.calls[0]
        self.assertIn("f.is_untracked = %s", query)
        self.assertTrue(any(p is False for p in params))

    def test_text_search_snippet_length_truncates_inline_snippet(self) -> None:
        long_body = "x" * 600
        snippet_raw = f"```go\n{long_body}\n```"
        conn = QueuedConnection([FakeCursor(many=[{"id": 1, "snippet_raw": snippet_raw}])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({"query": "x", "snippet_length": 50})
        payload = mcp_text_payload(response)
        results = cast("list[dict[str, object]]", payload["results"])
        self.assertEqual(len(cast("str", results[0]["snippet"])), 50)

    def test_text_search_snippet_length_rejects_out_of_range(self) -> None:
        definition = TOOL_DEFINITIONS["search_code_intel_text"]
        with self.assertRaises(McpProtocolError):
            _ = validate_tool_arguments(definition, {"query": "x", "snippet_length": 0})
        with self.assertRaises(McpProtocolError):
            _ = validate_tool_arguments(definition, {"query": "x", "snippet_length": 1000})

    def test_text_search_mode_echoed_only_when_explicitly_set(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({})
        payload = mcp_text_payload(response)
        self.assertNotIn("mode", payload)

        conn = QueuedConnection([FakeCursor(many=[])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({"mode": "enumerate"})
        payload = mcp_text_payload(response)
        self.assertEqual(payload["mode"], "enumerate")

    def test_text_search_enumerate_uses_prefix_stable_ordering(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_search_code_intel_text({"mode": "enumerate", "limit": 2})

        query, _params = conn.calls[0]
        self.assertIn("ORDER BY r.source_path ASC", query)
        self.assertIn("r.line_start ASC NULLS LAST", query)
        self.assertIn("r.record_id ASC", query)
        self.assertNotIn("ORDER BY r.updated_at DESC", query)


class McpPathFilterTests(unittest.TestCase):
    def test_text_search_accepts_source_path_prefix(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_search_code_intel_text({"source_path_prefix": "cmd/"})
        query, params = conn.calls[0]
        self.assertIn("r.source_path LIKE %s ESCAPE", query)
        self.assertIn("cmd/%", params)

    def test_source_path_filters_accept_repo_relative_paths_when_repo_is_known(self) -> None:
        clauses, params = source_path_clauses({"repo": "zod", "source_path": "packages/zod/src/index.ts"}, "r")

        self.assertEqual(clauses, ["r.source_path = ANY(%s)"])
        self.assertEqual(params, [["packages/zod/src/index.ts", "zod/packages/zod/src/index.ts"]])

        clauses, params = source_path_clauses({"repo": "zod", "source_path": "zod/packages/zod/src/index.ts"}, "r")

        self.assertEqual(clauses, ["r.source_path = %s"])
        self.assertEqual(params, ["zod/packages/zod/src/index.ts"])

        clauses, params = source_path_clauses({"repo": "zod", "source_path_prefix": "packages/zod/src"}, "r")

        self.assertEqual(
            clauses,
            ["(r.source_path LIKE %s ESCAPE '\\' OR r.source_path LIKE %s ESCAPE '\\')"],
        )
        self.assertEqual(params, ["packages/zod/src/%", "zod/packages/zod/src/%"])

        clauses, params = source_path_clauses({"repo": "zod", "source_path_prefix": "zod/packages/zod/src"}, "r")

        self.assertEqual(clauses, ["r.source_path LIKE %s ESCAPE '\\'"])
        self.assertEqual(params, ["zod/packages/zod/src/%"])

    def test_source_path_filters_expand_nested_crate_src_paths_without_broadening_manifests(self) -> None:
        clauses, params = source_path_clauses({"repo": "tokio", "source_path": "src/lib.rs"}, "r")

        self.assertEqual(clauses, ["r.source_path = ANY(%s)"])
        self.assertEqual(params, [["src/lib.rs", "tokio/src/lib.rs", "tokio/tokio/src/lib.rs"]])

        clauses, params = source_path_clauses({"repo": "tokio", "source_path": "tokio/src/lib.rs"}, "r")

        self.assertEqual(clauses, ["r.source_path = ANY(%s)"])
        self.assertEqual(params, [["tokio/src/lib.rs", "tokio/tokio/src/lib.rs"]])

        clauses, params = source_path_clauses({"repo": "tokio", "source_path": "tokio/Cargo.toml"}, "r")

        self.assertEqual(clauses, ["r.source_path = %s"])
        self.assertEqual(params, ["tokio/Cargo.toml"])

        clauses, params = source_path_clauses({"repo": "tokio", "source_path": "tokio/tokio/Cargo.toml"}, "r")

        self.assertEqual(clauses, ["r.source_path = %s"])
        self.assertEqual(params, ["tokio/tokio/Cargo.toml"])

        clauses, params = source_path_clauses({"repo": "tokio", "source_path_prefix": "tokio/tokio"}, "r")

        self.assertEqual(clauses, ["r.source_path LIKE %s ESCAPE '\\'"])
        self.assertEqual(params, ["tokio/tokio/%"])

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

    def test_source_path_filters_reject_absolute_paths(self) -> None:
        with self.assertRaises(McpProtocolError) as ctx:
            _ = source_path_clauses({"source_path": "/home/me/tokio/src/lib.rs"}, "r")
        self.assertIn("repo-relative", str(ctx.exception))

        with self.assertRaises(McpProtocolError):
            _ = normalize_source_path_filter("C:\\Users\\me\\tokio\\src\\lib.rs")

    def test_text_search_rejects_source_path_with_prefix(self) -> None:
        conn = QueuedConnection([FakeCursor(one=None)])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
            self.assertRaises(McpProtocolError),
        ):
            _ = mcp_tools.tool_search_code_intel_text({
                "source_path": "cmd/main.go",
                "source_path_prefix": "cmd",
            })


class McpSemanticSearchTests(unittest.TestCase):
    def test_semantic_search_excludes_security_patterns_from_broad_results(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_semantic({"query": "break circular import"})

        payload = mcp_text_payload(response)
        self.assertEqual(payload["results"], [])
        query, _params = conn.calls[0]
        self.assertIn("r.record_type <> 'security_pattern'", query)

    def test_semantic_search_uses_lexical_reranking_with_similarity_only(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "record_id": "tokio/src/sync/oneshot.rs::chunk::000001-000020",
                        "source_path": "tokio/src/sync/oneshot.rs",
                        "title": "oneshot receiver dropped",
                        "summary": "oneshot sender observes receiver drop",
                        "record_type": "code_chunk",
                        "distance": 0.42,
                        "match_score": 12.0,
                        "quality_penalty": 0.0,
                        "snippet_raw": "```rust\nreceiver.close();\n```",
                    }
                ]
            )
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_semantic({
                "query": "what happens when a oneshot receiver is dropped",
            })

        query, params = conn.calls[0]
        self.assertIn("match_score", query)
        self.assertIn("split_part(coalesce(r.embedding_text", query)
        self.assertIn("quality_penalty", query)
        self.assertIn("r.updated_at", query)
        self.assertIn("LEAST(ranked.match_score, 80)", query)
        self.assertIn("+ ranked.quality_penalty", query)
        self.assertIn("ranked.symbol_kind IN ('function', 'method', 'shell_function')", query)
        self.assertIn("ranked.symbol_kind IN ('struct', 'interface', 'type')", query)
        self.assertIn("coalesce(ranked.symbol, '') ILIKE '%%validat%%'", query)
        self.assertIn("ranked.file_role = 'source'", query)
        self.assertIn("ranked.content_class <> 'source'", query)
        self.assertIn("ranked.is_generated", query)
        self.assertIn("oneshot", cast("list[str]", params[1]))
        self.assertEqual(params[-7], 0.0)
        self.assertEqual(params[-6], 0.0)
        self.assertEqual(params[-5], 0.0)
        self.assertEqual(params[-4], mcp_semantic.SEMANTIC_SOURCE_ROLE_DISTANCE_BOOST)
        self.assertEqual(params[-3], mcp_semantic.SEMANTIC_NON_SOURCE_DISTANCE_PENALTY)
        self.assertEqual(params[-2], mcp_semantic.SEMANTIC_GENERATED_DISTANCE_PENALTY)

        payload = mcp_text_payload(response)
        result = cast("list[dict[str, object]]", payload["results"])[0]
        self.assertNotIn("distance", result)
        self.assertNotIn("match_score", result)
        self.assertNotIn("quality_penalty", result)
        # similarity = 1 - distance (cosine similarity); higher = closer match, parallel to text
        # search's `rank`. Lets consumers self-judge confidence without a follow-up call.
        self.assertIn("similarity", result)
        self.assertAlmostEqual(cast("float", result["similarity"]), 1.0 - 0.42, places=6)

    def test_semantic_search_omits_similarity_when_distance_missing(self) -> None:
        # Defensive: rows without a numeric `distance` (NULL, missing, non-vector record path)
        # must not emit `similarity` at all. Otherwise we'd silently surface `similarity: 1.0`
        # as a meaningless strong signal.
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "record_id": "tokio/src/sync/oneshot.rs::chunk::000001-000020",
                        "source_path": "tokio/src/sync/oneshot.rs",
                        "title": "oneshot receiver dropped",
                        "summary": "oneshot sender observes receiver drop",
                        "record_type": "code_chunk",
                        "match_score": 12.0,
                        "quality_penalty": 0.0,
                        "snippet_raw": "```rust\nreceiver.close();\n```",
                    }
                ]
            )
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_semantic({
                "query": "what happens when a oneshot receiver is dropped",
            })

        payload = mcp_text_payload(response)
        result = cast("list[dict[str, object]]", payload["results"])[0]
        self.assertNotIn("similarity", result)

    def test_semantic_search_verbose_includes_distance_and_similarity(self) -> None:
        # Verbose mode keeps `distance` (debugging signal) and gains `similarity` (consumer-
        # facing confidence score). The two must be related by `similarity = 1 - distance`.
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "record_id": "tokio/src/sync/oneshot.rs::chunk::000001-000020",
                        "source_path": "tokio/src/sync/oneshot.rs",
                        "title": "oneshot receiver dropped",
                        "summary": "oneshot sender observes receiver drop",
                        "record_type": "code_chunk",
                        "distance": 0.25,
                        "match_score": 5.0,
                        "quality_penalty": 0.0,
                        "snippet_raw": "```rust\nreceiver.close();\n```",
                    }
                ]
            )
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_semantic({
                "query": "what happens when a oneshot receiver is dropped",
                "verbose": True,
            })

        payload = mcp_text_payload(response)
        result = cast("list[dict[str, object]]", payload["results"])[0]
        self.assertIn("distance", result)
        self.assertIn("similarity", result)
        self.assertAlmostEqual(cast("float", result["similarity"]), 1.0 - 0.25, places=6)

    def test_semantic_search_diversifies_results_by_parent_by_default(self) -> None:
        def row(record_id: str, parent_record_id: str) -> dict[str, object]:
            return {
                "record_id": record_id,
                "parent_record_id": parent_record_id,
                "source_path": "tokio/src/net/unix/datagram/socket.rs",
                "title": record_id,
                "summary": record_id,
                "record_type": "code_chunk",
                "distance": 0.2,
                "match_score": 1.0,
                "quality_penalty": 0.0,
                "snippet_raw": "```rust\nready();\n```",
            }

        conn = QueuedConnection([
            FakeCursor(
                many=[
                    row("chunk-a1", "parent-a"),
                    row("chunk-a2", "parent-a"),
                    row("chunk-b1", "parent-b"),
                    row("chunk-a3", "parent-a"),
                ]
            )
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_semantic({
                "query": "socket readiness false positive clear readiness loop",
                "limit": 3,
            })

        payload = mcp_text_payload(response)
        results = cast("list[dict[str, object]]", payload["results"])
        self.assertEqual([result["record_id"] for result in results], ["chunk-a1", "chunk-b1", "chunk-a2"])
        _query, params = conn.calls[0]
        self.assertGreater(cast("int", params[-1]), 3)

    def test_semantic_search_disables_source_role_boost_for_test_queries(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_search_code_intel_semantic({"query": "tests for oneshot receiver drop"})

        _query, params = conn.calls[0]
        self.assertEqual(params[-7], 0.0)
        self.assertEqual(params[-6], 0.0)
        self.assertEqual(params[-5], 0.0)
        self.assertEqual(params[-4], 0.0)
        self.assertEqual(params[-3], 0.0)
        self.assertEqual(params[-2], 0.0)

    def test_semantic_search_boosts_executable_symbols_for_implementation_queries(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_search_code_intel_semantic({
                "query": "where does VirtualServer policy generation call generatePolicies",
            })

        _query, params = conn.calls[0]
        self.assertEqual(params[-7], mcp_semantic.SEMANTIC_EXECUTABLE_SYMBOL_DISTANCE_BOOST)
        self.assertEqual(params[-6], mcp_semantic.SEMANTIC_STRUCTURAL_SYMBOL_DISTANCE_PENALTY)
        self.assertEqual(params[-5], mcp_semantic.SEMANTIC_VALIDATION_DISTANCE_PENALTY)

    def test_semantic_search_expands_translation_queries_toward_implementation_terms(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_search_code_intel_semantic({
                "query": "how are VirtualServer policies translated into nginx configuration",
            })

        _query, params = conn.calls[0]
        rank_terms = cast("list[str]", params[1])
        self.assertIn("VirtualServer", rank_terms)
        self.assertIn("policies", rank_terms)
        self.assertIn("translated", rank_terms)
        self.assertIn("nginx", rank_terms)
        self.assertIn("configuration", rank_terms)
        self.assertNotIn("how", rank_terms)
        self.assertNotIn("are", rank_terms)
        for supplemental_term in ("generate", "render", "build", "add", "config", "template"):
            self.assertIn(supplemental_term, rank_terms)
        self.assertEqual(params[-7], mcp_semantic.SEMANTIC_EXECUTABLE_SYMBOL_DISTANCE_BOOST)
        self.assertEqual(params[-6], mcp_semantic.SEMANTIC_STRUCTURAL_SYMBOL_DISTANCE_PENALTY)
        self.assertEqual(params[-5], mcp_semantic.SEMANTIC_VALIDATION_DISTANCE_PENALTY)

    def test_semantic_search_treats_emitted_generated_config_queries_as_implementation_intent(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_search_code_intel_semantic({
                "query": "how are VirtualServer policies emitted in generated nginx configuration",
            })

        _query, params = conn.calls[0]
        rank_terms = cast("list[str]", params[1])
        self.assertIn("emitted", rank_terms)
        self.assertIn("generated", rank_terms)
        self.assertIn("configuration", rank_terms)
        self.assertIn("generate", rank_terms)
        self.assertIn("template", rank_terms)
        self.assertEqual(params[-7], mcp_semantic.SEMANTIC_EXECUTABLE_SYMBOL_DISTANCE_BOOST)
        self.assertEqual(params[-6], mcp_semantic.SEMANTIC_STRUCTURAL_SYMBOL_DISTANCE_PENALTY)
        self.assertEqual(params[-5], mcp_semantic.SEMANTIC_VALIDATION_DISTANCE_PENALTY)

    def test_semantic_search_disables_executable_boost_for_structural_queries(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_search_code_intel_semantic({"query": "generate policy struct fields"})

        _query, params = conn.calls[0]
        self.assertEqual(params[-7], 0.0)
        self.assertEqual(params[-6], 0.0)
        self.assertEqual(params[-5], 0.0)

    def test_semantic_search_disables_implementation_bias_for_validation_schema_queries(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_search_code_intel_semantic({
                "query": "VirtualServer policy validation schema fields",
            })

        _query, params = conn.calls[0]
        rank_terms = cast("list[str]", params[1])
        self.assertIn("validation", rank_terms)
        self.assertIn("schema", rank_terms)
        self.assertIn("fields", rank_terms)
        self.assertNotIn("generate", rank_terms)
        self.assertNotIn("template", rank_terms)
        self.assertEqual(params[-7], 0.0)
        self.assertEqual(params[-6], 0.0)
        self.assertEqual(params[-5], 0.0)

    def test_semantic_search_record_type_filter_can_request_security_patterns(self) -> None:
        conn = QueuedConnection([
            FakeCursor(one={"record_count": 2, "embedded_records": 2}),
            FakeCursor(many=[]),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_semantic({
                "query": "shell command execution",
                "record_type": "security_pattern",
            })

        payload = mcp_text_payload(response)
        self.assertEqual(payload["results"], [])
        query, _params = conn.calls[1]
        self.assertNotIn("r.record_type <> 'security_pattern'", query)

    def test_semantic_search_warns_for_text_only_record_type_filter(self) -> None:
        conn = QueuedConnection([FakeCursor(one={"record_count": 3, "embedded_records": 0})])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)) as query_embedding,
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_semantic({
                "query": "defineLazy",
                "record_type": "symbol_definition",
            })

        query_embedding.assert_not_called()
        payload = mcp_text_payload(response)
        self.assertEqual(payload["results"], [])
        warnings = cast("list[dict[str, object]]", payload["warnings"])
        self.assertEqual(warnings[0]["kind"], "semantic_filter_has_no_embeddings")
        self.assertEqual(warnings[0]["record_type"], "symbol_definition")
        self.assertIn("search_code_intel_text", cast("str", warnings[0]["message"]))
        query, _params = conn.calls[0]
        self.assertIn("count(r.embedding) AS embedded_records", query)

    def test_semantic_search_content_class_filter_still_excludes_security_patterns(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_semantic({
                "query": "cooperative scheduling budget",
                "content_class": "source",
            })

        payload = mcp_text_payload(response)
        self.assertEqual(payload["results"], [])
        query, _params = conn.calls[0]
        self.assertIn("r.record_type <> 'security_pattern'", query)

    def test_empty_repo_and_path_scopes_return_actionable_warnings(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[]), FakeCursor(one=None)])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({
                "query": "budget",
                "repo": "missing-repo",
                "source_path_prefix": "src/runtime",
            })

        payload = mcp_text_payload(response)
        warnings = cast("list[dict[str, object]]", payload["warnings"])
        self.assertEqual([warning["kind"] for warning in warnings], ["empty_repo_scope", "empty_path_scope"])
        self.assertIn("code_intel_status", cast("str", warnings[0]["message"]))
        self.assertIn("repo-relative", cast("str", warnings[1]["message"]))

    def test_valid_repo_empty_path_scope_does_not_warn_repo_is_empty(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[]), FakeCursor(one={"exists": 1})])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({
                "query": "budget",
                "repo": "tokio",
                "source_path_prefix": "missing/path",
            })

        payload = mcp_text_payload(response)
        warnings = cast("list[dict[str, object]]", payload["warnings"])
        self.assertEqual([warning["kind"] for warning in warnings], ["empty_path_scope"])

    def test_repo_root_path_scope_returns_warning(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[{"source_path": "tokio/src/lib.rs"}])])

        with (
            patch.dict(os.environ, {"PCI_COLLECTION": "tokio"}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({
                "query": "runtime",
                "source_path_prefix": "tokio/",
            })

        payload = mcp_text_payload(response)
        warnings = cast("list[dict[str, object]]", payload["warnings"])
        self.assertEqual(warnings[0]["kind"], "repo_root_path_scope")
        self.assertIn("broad repo filter", cast("str", warnings[0]["message"]))


class McpRelatedCodeIntelTests(unittest.TestCase):
    def test_related_accepts_direction_argument(self) -> None:
        definition = TOOL_DEFINITIONS["related_code_intel"]

        _ = validate_tool_arguments(definition, {"symbol": "parse", "direction": "outgoing"})

        with self.assertRaises(McpProtocolError):
            _ = validate_tool_arguments(definition, {"symbol": "parse", "direction": "sideways"})

    def test_related_direction_filters_symbol_edges(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_related_code_intel({"symbol": "parse", "direction": "outgoing"})

        query, params = conn.calls[0]
        self.assertIn("e.source_symbol = %s", query)
        self.assertNotIn("e.target_symbol = %s", query)
        self.assertIn("e.target_record_id IS NOT NULL", query)
        self.assertEqual(params[0], "parse")

    def test_related_can_include_unresolved_heuristic_edges(self) -> None:
        # Default direction is "any", which now runs incoming + outgoing in parallel.
        conn = QueuedConnection([FakeCursor(many=[]), FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_related_code_intel({"symbol": "parse", "include_unresolved": True})

        for query, _params in conn.calls:
            self.assertNotIn("e.target_record_id IS NOT NULL", query)

    def test_related_record_id_uses_parent_edges_for_chunks(self) -> None:
        chunk_id = "src/app.ts::function_chunk::build::000001"
        parent_id = "src/app.ts::function::build::000001"
        outgoing_edge = {
            "id": 1,
            "source_record_id": parent_id,
            "target_record_id": "src/app.ts::function::helper::000010",
            "edge_type": "call_candidate",
            "source_symbol": "build",
            "target_symbol": "helper",
            "target_record_db_id": 9,
        }
        # Default direction "any" runs incoming + outgoing; the parent_id row only
        # appears on the outgoing side because its source matches the scoped record IDs.
        conn = QueuedConnection([
            FakeCursor(one={"parent_record_id": parent_id}),
            FakeCursor(many=[]),
            FakeCursor(many=[outgoing_edge]),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"record_id": chunk_id})

        payload = mcp_text_payload(response)
        edges = cast("list[dict[str, object]]", payload["edges"])
        self.assertEqual(edges[0]["edge_source"], "parent_record")
        self.assertEqual(edges[0]["direction"], "outgoing")
        self.assertTrue(edges[0]["target_resolved"])
        _lookup_query, lookup_params = conn.calls[0]
        incoming_query, incoming_params = conn.calls[1]
        outgoing_query, outgoing_params = conn.calls[2]
        self.assertEqual(lookup_params[-1], chunk_id)
        self.assertIn("e.target_record_id = ANY(%s)", incoming_query)
        self.assertIn([chunk_id, parent_id], incoming_params)
        self.assertIn("e.source_record_id = ANY(%s)", outgoing_query)
        self.assertIn([chunk_id, parent_id], outgoing_params)

    def test_related_missing_record_id_reports_found_false(self) -> None:
        record_id = "src/app.ts::function::missing::000001"
        conn = QueuedConnection([FakeCursor(one=None)])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"record_id": record_id})

        payload = mcp_text_payload(response)
        self.assertFalse(payload["found"])
        self.assertEqual(payload["edges"], [])
        warnings = cast("list[dict[str, object]]", payload["warnings"])
        self.assertEqual(warnings[0]["kind"], "record_not_found")
        self.assertEqual(warnings[0]["record_id"], record_id)
        self.assertEqual(len(conn.calls), 1)

    def test_related_symbol_default_direction_returns_both_sides_balanced(self) -> None:
        # `direction=any` (the default) now runs incoming + outgoing in parallel and
        # interleaves them so neither side starves the other within the limit. The
        # caller still gets resolved-first within each side.
        incoming_edge = {
            "id": 2,
            "source_record_id": "src/schema.ts::constant::ObjectSchema::000010",
            "target_record_id": "src/util.ts::function::defineLazy::000001",
            "target_record_db_id": 7,
            "edge_type": "call_candidate",
            "source_symbol": "ObjectSchema",
            "target_symbol": "defineLazy",
        }
        outgoing_edge = {
            "id": 1,
            "source_record_id": "src/util.ts::function::defineLazy::000001",
            "target_record_id": None,
            "edge_type": "call_candidate",
            "source_symbol": "defineLazy",
            "target_symbol": "getter",
        }
        conn = QueuedConnection([
            FakeCursor(many=[incoming_edge]),
            FakeCursor(many=[outgoing_edge]),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"symbol": "defineLazy"})

        payload = mcp_text_payload(response)
        edges = cast("list[dict[str, object]]", payload["edges"])
        directions = [edge["direction"] for edge in edges]
        self.assertIn("incoming", directions)
        self.assertIn("outgoing", directions)
        self.assertEqual(len(edges), 2)
        incoming_query, incoming_params = conn.calls[0]
        outgoing_query, outgoing_params = conn.calls[1]
        # Each per-direction query filters by symbol on its own side only — no OR.
        self.assertIn("e.target_symbol = %s", incoming_query)
        self.assertEqual(incoming_params[0], "defineLazy")
        self.assertIn("e.source_symbol = %s", outgoing_query)
        self.assertEqual(outgoing_params[0], "defineLazy")

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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"symbol": "bootstrap", "direction": "outgoing"})

        payload = mcp_text_payload(response)
        edge = cast("list[dict[str, object]]", payload["edges"])[0]
        self.assertEqual(edge["source_line_start"], 10)
        self.assertEqual(edge["target_line_end"], 40)
        self.assertEqual(edge["target_kind"], "project_symbol")
        self.assertEqual(edge["confidence_kind"], "heuristic_candidate")
        self.assertEqual(
            payload["warnings"],
            [
                {
                    "kind": "heuristic_candidate_relationships",
                    "confidence_kind": "heuristic_candidate",
                    "message": (
                        "related_code_intel returns heuristic candidates; verify important relationships in source"
                    ),
                }
            ],
        )
        self.assertNotIn("source_summary", edge)
        self.assertNotIn("target_summary", edge)
        self.assertNotIn("source_record_db_id", edge)
        self.assertNotIn("metadata", edge)

    def test_related_symbol_handles_dollar_prefixed_project_symbols(self) -> None:
        # Both edges have target_symbol == "$BaseLazy", so they're returned on the
        # incoming side; outgoing is empty.
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
            ),
            FakeCursor(many=[]),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"symbol": "$BaseLazy"})

        payload = mcp_text_payload(response)
        sources = [edge["source_symbol"] for edge in cast("list[dict[str, object]]", payload["edges"])]
        self.assertEqual(sources, ["MiniLazy", "ClassicLazy"])


class McpRelatedDirectionBalancingTests(unittest.TestCase):
    """`direction=any` (the default) runs incoming + outgoing in parallel and
    interleaves them so neither side starves the other within the limit. When one
    side is empty, the other side fills the limit.
    """

    @staticmethod
    def _incoming_edge(edge_id: int, source: str) -> dict[str, object]:
        return {
            "id": edge_id,
            "source_record_id": f"src/{source}.ts::function::{source}::000001",
            "target_record_id": "src/util.ts::function::target::000001",
            "target_record_db_id": 99,
            "edge_type": "call_candidate",
            "source_symbol": source,
            "target_symbol": "target",
        }

    @staticmethod
    def _outgoing_edge(edge_id: int, target: str) -> dict[str, object]:
        return {
            "id": edge_id,
            "source_record_id": "src/util.ts::function::target::000001",
            "target_record_id": f"src/{target}.ts::function::{target}::000001",
            "target_record_db_id": 99 + edge_id,
            "edge_type": "call_candidate",
            "source_symbol": "target",
            "target_symbol": target,
        }

    def test_any_returns_both_sides_when_both_have_edges(self) -> None:
        # 3 incoming + 3 outgoing at limit=6 → 3 of each.
        incoming = cast("list[object]", [self._incoming_edge(100 + i, f"in{i}") for i in range(3)])
        outgoing = cast("list[object]", [self._outgoing_edge(200 + i, f"out{i}") for i in range(3)])
        conn = QueuedConnection([FakeCursor(many=incoming), FakeCursor(many=outgoing)])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"symbol": "target", "limit": 6})

        edges = cast("list[dict[str, object]]", mcp_text_payload(response)["edges"])
        directions = [edge["direction"] for edge in edges]
        self.assertEqual(directions.count("incoming"), 3)
        self.assertEqual(directions.count("outgoing"), 3)
        self.assertEqual(len(edges), 6)

    def test_any_falls_back_to_one_side_when_other_is_empty(self) -> None:
        # 0 outgoing + 5 incoming at limit=5 → the outgoing slack is reallocated.
        incoming = cast("list[object]", [self._incoming_edge(100 + i, f"in{i}") for i in range(5)])
        conn = QueuedConnection([FakeCursor(many=incoming), FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"symbol": "target", "limit": 5})

        edges = cast("list[dict[str, object]]", mcp_text_payload(response)["edges"])
        self.assertEqual(len(edges), 5)
        self.assertTrue(all(edge["direction"] == "incoming" for edge in edges))

    def test_any_preserves_one_minority_edge_when_other_side_is_large(self) -> None:
        # 1 outgoing + 19 incoming at limit=20 → outgoing must not be starved.
        # This reproduces the original bug report (14 incoming + 1 outgoing fit
        # in 20 slots, but the old ordering put outgoing last and dropped it).
        incoming = cast("list[object]", [self._incoming_edge(100 + i, f"in{i}") for i in range(19)])
        outgoing = cast("list[object]", [self._outgoing_edge(200, "lone")])
        conn = QueuedConnection([FakeCursor(many=incoming), FakeCursor(many=outgoing)])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"symbol": "target", "limit": 20})

        edges = cast("list[dict[str, object]]", mcp_text_payload(response)["edges"])
        directions = [edge["direction"] for edge in edges]
        self.assertEqual(directions.count("outgoing"), 1)
        self.assertEqual(directions.count("incoming"), 19)

    def test_explicit_direction_outgoing_runs_one_query(self) -> None:
        # The single-direction path is unchanged: one SQL call, ordering preserved.
        outgoing = cast("list[object]", [self._outgoing_edge(200 + i, f"out{i}") for i in range(3)])
        conn = QueuedConnection([FakeCursor(many=outgoing)])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"symbol": "target", "direction": "outgoing"})

        self.assertEqual(len(conn.calls), 1)
        edges = cast("list[dict[str, object]]", mcp_text_payload(response)["edges"])
        self.assertTrue(all(edge["direction"] == "outgoing" for edge in edges))


class McpContractTests(unittest.TestCase):
    def test_mcp_defaults_collection_from_process_cwd(self) -> None:
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "project-code-intelligence"
            workspace.mkdir()
            try:
                os.chdir(workspace)
                collection, scope_path, defaulted = _mcp_defaults_for_empty_env()
            finally:
                os.chdir(old_cwd)
        self.assertEqual(collection, "project-code-intelligence")
        self.assertEqual(scope_path, str(workspace.resolve()))
        self.assertEqual(defaulted, "1")

    def test_mcp_keeps_explicit_collection_override(self) -> None:
        with patch.dict(os.environ, {"PCI_COLLECTION": "configured"}, clear=True):
            set_mcp_environment_defaults()
            self.assertEqual(os.environ["PCI_COLLECTION"], "configured")
            self.assertEqual(os.environ[config.DATABASE_SCOPE_PATH_ENV], str(Path.cwd().resolve()))
            self.assertNotIn("PCI_COLLECTION_DEFAULTED", os.environ)

        with patch.dict(os.environ, {config.DATABASE_SCOPE_PATH_ENV: "configured-scope"}, clear=True):
            set_mcp_environment_defaults()
            self.assertEqual(os.environ[config.DATABASE_SCOPE_PATH_ENV], "configured-scope")

    def test_repo_filter_skips_defaulted_collection_scope(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PCI_COLLECTION": "project-code-intelligence",
                "PCI_COLLECTION_DEFAULTED": "1",
            },
            clear=True,
        ):
            clauses, params = code_intel_clauses({"repo": "zod"}, "r")

        self.assertNotIn("r.collection = %s", clauses)
        self.assertIn("r.repo = %s", clauses)
        self.assertIn("zod", params)

        with patch.dict(
            os.environ,
            {
                "PCI_COLLECTION": "project-code-intelligence",
                "PCI_COLLECTION_DEFAULTED": "1",
            },
            clear=True,
        ):
            clauses, params = code_intel_clauses({"collection": ""}, "r")

        self.assertNotIn("r.collection = %s", clauses)
        self.assertNotIn("project-code-intelligence", params)

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
            patch.dict(os.environ, {"PCI_COLLECTION": "configured"}, clear=True),
            self.assertRaises(McpWritePermissionError) as ctx,
        ):
            _ = handle_jsonrpc_value(request)

        self.assertEqual(request_id_from_jsonrpc_value(request), 42)
        message = str(ctx.exception)
        self.assertIn("does not match PCI_COLLECTION", message)
        self.assertIn("omit collection", message)

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
            patch.dict(os.environ, {"PCI_MCP_MAX_TEXT_CHARS": "5"}, clear=True),
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
            patch.dict(os.environ, {"PCI_MCP_MAX_METADATA_BYTES": "1024"}, clear=True),
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
                {"PCI_COLLECTION": "project-code-intelligence"},
                clear=True,
            ),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
                {"PCI_COLLECTION": "project-code-intelligence"},
                clear=True,
            ),
            patch.object(mcp_db, "table_regclass_exists", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_get_static_finding({"id": 1})

        query, params = conn.calls[0]
        self.assertIn("f.id = %s", query)
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
            patch.object(mcp_db, "table_regclass_exists", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
            patch.object(mcp_db, "table_regclass_exists", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
            patch.object(mcp_semantic.config, "default_embedding_endpoint", return_value=endpoint),
            patch.object(mcp_semantic.embeddings, "resolve_embedding_endpoint_model", return_value="local"),
            patch.object(
                mcp_semantic.embeddings,
                "embed_with_endpoint",
                side_effect=EmbeddingEndpointUnavailableError("connection refused"),
            ),
            self.assertRaises(McpProtocolError) as raised,
        ):
            _ = mcp_semantic.query_embedding("find request handler")

        message = str(raised.exception)
        self.assertIn("semantic search requires an embedding endpoint", message)
        self.assertIn(endpoint, message)
        self.assertIn("PCI_EMBEDDING_ENDPOINT", message)

    def test_semantic_search_endpoint_failure_is_user_visible_mcp_error(self) -> None:
        endpoint = "http://127.0.0.1:18081/v1/embeddings"

        with (
            patch.object(mcp_semantic.config, "default_embedding_endpoint", return_value=endpoint),
            patch.object(mcp_semantic.embeddings, "resolve_embedding_endpoint_model", return_value="local"),
            patch.object(
                mcp_semantic.embeddings,
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
        schema = cast("dict[str, object]", text_search.input_schema)
        properties = cast("dict[str, dict[str, object]]", schema["properties"])
        self.assertEqual(properties["limit"]["description"], "Max results, 1-50.")
        self.assertIn("Exact indexed search", text_search.description)
        self.assertNotIn("diversify", properties)

        semantic_search = TOOL_DEFINITIONS["search_code_intel_semantic"]
        semantic_schema = cast("dict[str, object]", semantic_search.input_schema)
        semantic_properties = cast("dict[str, dict[str, object]]", semantic_schema["properties"])
        self.assertIn("diversify", semantic_properties)

        _ = validate_tool_arguments(text_search, {"query": "hello", "limit": 10})
        _ = validate_tool_arguments(text_search, {"query": "hello world", "query_mode": "all_terms"})

        list_files = TOOL_DEFINITIONS["list_code_intel_files"]
        normalized = validate_tool_arguments(
            list_files,
            {"language": "", "source_path_prefix": "", "is_generated": None, "is_source": False},
        )
        self.assertNotIn("language", normalized)
        self.assertNotIn("source_path_prefix", normalized)
        self.assertNotIn("is_generated", normalized)
        self.assertFalse(cast("bool", normalized["is_source"]))

        with self.assertRaises(McpProtocolError):
            _ = validate_tool_arguments(text_search, {"query": "hello", "surprise": True})
        with self.assertRaises(McpProtocolTypeError):
            _ = validate_tool_arguments(text_search, {"query": "hello", "limit": "10"})
        with self.assertRaises(McpProtocolError):
            _ = validate_tool_arguments(text_search, {"query": "hello", "limit": 500})
        with self.assertRaises(McpProtocolError):
            _ = validate_tool_arguments(text_search, {"query": "hello", "query_mode": "broad"})

    def test_required_tool_arguments_are_enforced_before_handlers(self) -> None:
        record_fetch = TOOL_DEFINITIONS["get_code_intel_record"]
        record_schema = cast("dict[str, object]", record_fetch.input_schema)
        record_properties = cast("dict[str, object]", record_schema["properties"])
        self.assertNotIn("oneOf", record_schema)
        self.assertEqual(record_schema["type"], "object")
        self.assertNotIn("required", record_schema)
        self.assertIn("record_id", record_properties)
        self.assertIn("record_ids", record_properties)

        _ = validate_tool_arguments(record_fetch, {"record_id": "a", "include_metadata": True})
        _ = validate_tool_arguments(record_fetch, {"record_ids": ["a", "b"], "include_metadata": True})
        with self.assertRaises(McpProtocolTypeError):
            _ = validate_tool_arguments(record_fetch, {"record_id": 1})
        with self.assertRaises(McpProtocolError):
            _ = validate_tool_arguments(record_fetch, {"record_ids": []})

    def test_tool_call_rejects_non_object_params_and_arguments(self) -> None:
        with self.assertRaises(McpProtocolTypeError):
            _ = handle_tool_call({"params": []}, None)
        with self.assertRaises(McpProtocolTypeError):
            _ = handle_tool_call({"params": {"name": "code_intel_status", "arguments": []}}, None)


class McpErrorVisibilityTests(unittest.TestCase):
    def test_database_connection_failure_is_user_visible_mcp_error(self) -> None:
        exc = pci_db.DatabaseConnectionError(
            "Could not connect to PostgreSQL/pgvector using PCI_MCP_DATABASE_URL=<hidden>"
        )

        message = error_message(exc)

        self.assertIn("PCI_MCP_DATABASE_URL", message)
        self.assertNotEqual(message, "internal server error")

    def test_mcp_database_connection_failure_mentions_project_env_exports(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "PCI_MCP_DATABASE_URL": "postgresql://db.example.invalid/pci_demo?sslmode=prefer",
                    config.DATABASE_SCOPE_PATH_ENV: "/work/demo",
                },
                clear=True,
            ),
            patch(
                "project_code_intelligence.mcp.db.db.connect",
                side_effect=pci_db.DatabaseConnectionError("connection failed"),
            ),
            self.assertRaises(pci_db.DatabaseConnectionError) as raised,
            mcp_db.connect(),
        ):
            pass

        message = str(raised.exception)
        self.assertIn("PCI_MCP_DATABASE_URL", message)
        self.assertIn("PCI_MCP_DATABASE_USER", message)
        self.assertIn("PCI_MCP_DATABASE_PASSWORD", message)
        self.assertIn("PCI_DATABASE_SCOPE_PATH", message)
        self.assertIn("restart the client", message)


class McpArgumentNormalizationTests(unittest.TestCase):
    def test_tool_call_passes_normalized_arguments_to_handlers(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = handle_tool_call(
                {
                    "params": {
                        "name": "list_code_intel_files",
                        "arguments": {
                            "source_path": "pkg/client/file.go",
                            "language": "",
                            "is_generated": None,
                            "is_source": False,
                        },
                    }
                },
                1,
            )

        query, _ = conn.calls[0]
        self.assertNotIn("f.language = %s", query)
        self.assertNotIn("f.is_generated = %s", query)
        self.assertIn("f.is_source = %s", query)


class McpToolSchemaCompatibilityTests(unittest.TestCase):
    def test_tool_schemas_match_argument_models(self) -> None:
        self.assertEqual(set(TOOL_DEFINITIONS), set(TOOL_INPUT_MODELS))
        for name, definition in TOOL_DEFINITIONS.items():
            with self.subTest(tool=name):
                schema = cast("dict[str, object]", definition.input_schema)
                properties = cast("dict[str, object]", schema["properties"])
                advertised_args = set(properties)
                accepted_args = set(TOOL_INPUT_MODELS[name].model_fields)
                self.assertEqual(advertised_args, accepted_args)

    def test_tool_schemas_are_client_compatible_at_top_level(self) -> None:
        forbidden = {"oneOf", "anyOf", "allOf", "enum", "not"}
        for name, definition in TOOL_DEFINITIONS.items():
            with self.subTest(name=name):
                schema = cast("dict[str, object]", definition.input_schema)
                self.assertEqual(schema.get("type"), "object")
                self.assertFalse(forbidden.intersection(schema), schema)


class McpStatusWarningTests(unittest.TestCase):
    def test_status_warns_when_snapshot_is_stale(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "id": 1,
                        "collection": "zod",
                        "repo": "zod",
                        "commit_sha": "indexed123",
                        "metadata": {},
                    }
                ]
            ),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(mcp_status, "schema_migration_versions", return_value=[]),
            patch.object(git_utils, "run_git", return_value="head456"),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({})

        payload = mcp_text_payload(response)
        snapshots = cast("list[dict[str, object]]", payload["snapshots"])
        self.assertEqual(snapshots[0]["head_status"], "stale")
        self.assertEqual(snapshots[0]["head_commit"], "head456")
        self.assertEqual(
            payload["warnings"],
            [
                {
                    "kind": "snapshot_stale",
                    "message": "snapshot is stale; verify with local source",
                    "id": 1,
                    "collection": "zod",
                    "repo": "zod",
                    "commit_sha": "indexed123",
                    "head_commit": "head456",
                }
            ],
        )

    def test_status_warns_when_snapshot_was_indexed_dirty(self) -> None:
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
                        "dirty": True,
                        "metadata": {"dirty_paths": ["src/lib.rs", "tests/test_lib.rs", 7]},
                    }
                ]
            ),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[{"collection": "zod", "repo": "zod", "files": 2, "dirty_files": 2}]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(mcp_status, "schema_migration_versions", return_value=[]),
            patch.object(git_utils, "run_git", side_effect=_git_show_current_branch_returns_main),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({})

        payload = mcp_text_payload(response)
        self.assertEqual(
            payload["warnings"],
            [
                {
                    "kind": "snapshot_dirty",
                    "message": (
                        "snapshot was indexed from a dirty working tree; verify dirty paths against local source"
                    ),
                    "dirty": True,
                    "dirty_paths_count": 2,
                    "id": 1,
                    "collection": "zod",
                    "repo": "zod",
                    "commit_sha": "abc123",
                    "head_status": "current",
                }
            ],
        )

    def test_status_include_active_runs_reports_ledger_rows_and_warning(self) -> None:
        heartbeat = datetime.now(timezone.utc) - timedelta(seconds=5)
        conn = QueuedConnection([
            *[FakeCursor(many=[]) for _ in range(6)],
            FakeCursor(
                many=[
                    {
                        "id": 7,
                        "collection": "ws",
                        "repos": ["repo-a"],
                        "repo_modes": {"repo-a": "full:version_mismatch"},
                        "pid": 123,
                        "host": "mac",
                        "phase": "embedding",
                        "progress": {"progress": {"phase_done": 5, "phase_total": 10}},
                        "started_at": heartbeat,
                        "heartbeat_at": heartbeat,
                        "finished_at": None,
                        "exit_code": None,
                        "interrupted": False,
                        "error": None,
                        "running": True,
                    }
                ]
            ),
        ])

        def regclass(_conn: object, table: str) -> bool:
            return table == "project_code_intel_index_runs"

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", side_effect=regclass),
            patch.object(mcp_status, "schema_migration_versions", return_value=[]),
            patch.object(git_utils, "run_git", return_value="abc123"),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({"include_active_runs": True})

        payload = mcp_text_payload(response)
        active_runs = cast("list[dict[str, object]]", payload["active_runs"])
        self.assertEqual(len(active_runs), 1)
        run = active_runs[0]
        self.assertEqual(run["phase"], "embedding")
        self.assertEqual(run["repo_modes"], {"repo-a": "full:version_mismatch"})
        self.assertTrue(run["running"])
        self.assertIsInstance(run["heartbeat_age_seconds"], int)
        self.assertIsInstance(run["heartbeat_at"], str)
        warnings = cast("list[dict[str, object]]", payload["warnings"])
        self.assertEqual(warnings[-1]["kind"], "index_run_active")
        self.assertEqual(warnings[-1]["collections"], ["ws"])

    def test_status_without_the_flag_omits_active_runs(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[]) for _ in range(6)])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(mcp_status, "schema_migration_versions", return_value=[]),
            patch.object(git_utils, "run_git", side_effect=_git_show_current_branch_returns_main),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({})
        payload = mcp_text_payload(response)
        self.assertNotIn("active_runs", payload)

    def test_status_active_runs_empty_when_table_is_absent(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[]) for _ in range(6)])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(mcp_status, "schema_migration_versions", return_value=[]),
            patch.object(git_utils, "run_git", return_value="abc123"),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({"include_active_runs": True})
        payload = mcp_text_payload(response)
        self.assertEqual(payload["active_runs"], [])
        warnings = cast("list[dict[str, object]]", payload.get("warnings", []))
        self.assertNotIn("index_run_active", [w["kind"] for w in warnings])

    def test_status_active_runs_excludes_completed_history(self) -> None:
        conn = QueuedConnection([
            FakeCursor(many=[{"id": 7, "running": False, "finished_at": datetime.now(timezone.utc)}])
        ])
        with patch.object(mcp_db, "table_regclass_exists", return_value=True):
            self.assertEqual(mcp_status.active_index_run_rows(cast("pci_db.DbConnection", conn), None), [])

    def test_status_unknown_repo_reports_found_false_warning(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[]) for _ in range(6)])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(mcp_status, "schema_migration_versions", return_value=[]),
            patch.object(git_utils, "run_git", return_value="abc123"),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({"repo": "missing-repo"})

        payload = mcp_text_payload(response)
        self.assertFalse(payload["found"])
        warnings = cast("list[dict[str, object]]", payload["warnings"])
        self.assertEqual(warnings[0]["kind"], "empty_repo_scope")
        self.assertEqual(warnings[0]["repo"], "missing-repo")


class McpEnumScopeWarningTests(unittest.TestCase):
    def test_text_search_unknown_language_emits_empty_language_scope(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[]), FakeCursor(one={"exists": 1})])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_text({"query": "needle", "language": "cobol"})
        warnings = cast("list[dict[str, object]]", mcp_text_payload(response).get("warnings", []))
        kinds = {w["kind"] for w in warnings}
        self.assertIn("empty_language_scope", kinds)
        language_warning = next(w for w in warnings if w["kind"] == "empty_language_scope")
        self.assertEqual(language_warning["language"], "cobol")

    def test_list_files_unknown_file_role_emits_empty_file_role_scope(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[]), FakeCursor(one={"exists": 1})])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_list_code_intel_files({"file_role": "no_such_role"})
        warnings = cast("list[dict[str, object]]", mcp_text_payload(response).get("warnings", []))
        self.assertTrue(any(w["kind"] == "empty_file_role_scope" for w in warnings))

    def test_text_search_inferred_enumerate_emits_mode_warning(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[]), FakeCursor(one={"exists": 1})])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            # No query, no explicit mode → silently falls through to enumerate; warning surfaces it.
            response = mcp_tools.tool_search_code_intel_text({})
        warnings = cast("list[dict[str, object]]", mcp_text_payload(response).get("warnings", []))
        self.assertTrue(any(w["kind"] == "mode_inferred_enumerate" for w in warnings))

    def test_related_code_intel_rejects_record_id_plus_symbol(self) -> None:
        with self.assertRaises(McpProtocolError) as ctx:
            _ = mcp_tools.tool_related_code_intel({"record_id": "foo::bar", "symbol": "baz"})
        self.assertIn("exactly one of record_id or symbol", str(ctx.exception))

    def test_get_code_intel_record_rejects_whitespace_record_id(self) -> None:
        with self.assertRaises(McpProtocolTypeError):
            _ = mcp_tools.tool_get_code_intel_record({"record_id": "   "})

    def test_get_code_intel_record_rejects_both_record_id_and_record_ids(self) -> None:
        with self.assertRaises(McpProtocolError) as ctx:
            _ = mcp_tools.tool_get_code_intel_record({"record_id": "a", "record_ids": ["b"]})
        self.assertIn("exactly one of record_id or record_ids", str(ctx.exception))

    def test_get_code_intel_record_rejects_neither_record_id_nor_record_ids(self) -> None:
        with self.assertRaises(McpProtocolError) as ctx:
            _ = mcp_tools.tool_get_code_intel_record({})
        self.assertIn("record_id or record_ids is required", str(ctx.exception))


class McpListFilesRecordBackedTests(unittest.TestCase):
    def test_list_files_includes_record_backed_paths_when_file_row_is_missing(self) -> None:
        generated_path = "kubernetes-ingress/pkg/client/applyconfiguration/configuration/v1/accesscontrol.go"
        conn = QueuedConnection([
            FakeCursor(one={"exists": 1}),
            FakeCursor(
                many=[
                    {
                        "id": None,
                        "snapshot_id": 1,
                        "collection": "ingress",
                        "repo": "kubernetes-ingress",
                        "repo_role": "project",
                        "branch": "main",
                        "commit_sha": "abc123",
                        "tree_sha": "def456",
                        "source_path": generated_path,
                        "git_blob_sha": None,
                        "file_sha256": None,
                        "size_bytes": None,
                        "language": "go",
                        "file_role": "generated",
                        "content_class": "generated",
                        "is_generated": True,
                        "is_vendor": False,
                        "is_test": False,
                        "is_source": True,
                        "is_build": False,
                        "is_config": False,
                        "is_doc": False,
                        "skipped_reason": None,
                        "is_untracked": False,
                        "indexed_dirty": False,
                        "metadata": {"inventory_source": "records"},
                        "created_at": "2026-05-17T22:09:39+00:00",
                    }
                ]
            ),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_list_code_intel_files({
                "repo": "kubernetes-ingress",
                "snapshot_id": 1,
                "source_path": generated_path,
                "verbose": True,
            })

        payload = mcp_text_payload(response)
        files = cast("list[dict[str, object]]", payload["files"])
        self.assertEqual(files[0]["source_path"], generated_path)
        self.assertEqual(files[0]["file_role"], "generated")
        self.assertNotIn("warnings", payload)
        query, params = conn.calls[1]
        self.assertIn("record_backed_files AS", query)
        self.assertIn("FROM project_code_intel_records r", query)
        self.assertIn("existing.id IS NULL", query)
        self.assertIn("FROM file_inventory f", query)
        self.assertIn(generated_path, params)

    def test_list_files_generated_filter_applies_to_record_backed_inventory(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[]), FakeCursor(one={"exists": 1})])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_list_code_intel_files({
                "repo": "kubernetes-ingress",
                "language": "go",
                "file_role": "generated",
                "is_generated": True,
            })

        query, params = conn.calls[0]
        self.assertIn("bool_or(r.file_role = 'generated' OR r.content_class = 'generated') AS is_generated", query)
        self.assertIn("f.file_role = %s", query)
        self.assertIn("f.is_generated = %s", query)
        self.assertIn("generated", params)
        self.assertTrue(any(param is True for param in params))


class McpStatusRuntimeIdentityTests(unittest.TestCase):
    def test_source_commit_falls_back_to_installed_direct_url_metadata(self) -> None:
        class DirectUrlDistribution(importlib.metadata.Distribution):
            @override
            def read_text(self, filename: str) -> str | None:
                if filename == "direct_url.json":
                    return json.dumps({"vcs_info": {"commit_id": "installed123"}})
                return None

        distribution = DirectUrlDistribution()
        with (
            patch.object(mcp_status, "source_git_root", return_value=None),
            patch.object(mcp_status.importlib.metadata, "distribution", return_value=distribution),
        ):
            identity = mcp_status.server_runtime_identity()
        package = cast("dict[str, object]", identity["package"])
        self.assertEqual(package["source_git_commit"], "installed123")

    def test_status_can_include_runtime_identity_without_secret_values(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[]) for _ in range(6)])
        credential_value = "".join(("super", "-", "secret"))

        with (
            patch.dict(
                os.environ,
                {
                    "HOME": "/home/tester",
                    "PCI_MCP_DATABASE_URL": "postgresql://db.example/pci?sslmode=prefer",
                    "PCI_MCP_DATABASE_USER": "pci_ro",
                    "PCI_MCP_DATABASE_PASSWORD": credential_value,
                    "PCI_DATABASE_SCOPE_PATH": "/work/kubernetes-ingress",
                    "PCI_COLLECTION": "kubernetes-ingress",
                },
                clear=True,
            ),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(git_utils, "run_git", return_value="abc123"),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({"include_runtime": True})

        payload = mcp_text_payload(response)
        runtime = cast("dict[str, object]", payload["runtime"])
        package = cast("dict[str, object]", runtime["package"])
        process = cast("dict[str, object]", runtime["process"])
        database = cast("dict[str, object]", runtime["database"])
        config_section = cast("dict[str, object]", runtime["config"])

        self.assertEqual(package["name"], "project-code-intelligence")
        # Checks that server_runtime_identity reports a real package file, not a
        # specific implementation file (status helpers were extracted to a
        # submodule, so the path may resolve to _status.py instead of tools.py).
        self.assertIn("project_code_intelligence/mcp", cast("str", package["module_path"]))
        self.assertEqual(package["source_git_commit"], "abc123")
        self.assertIsInstance(process["pid"], int)
        self.assertIn("python", cast("str", process["executable"]))
        self.assertEqual(database["dsn_source"], "PCI_MCP_DATABASE_URL")
        self.assertEqual(database["user_source"], "PCI_MCP_DATABASE_USER")
        self.assertEqual(database["password_source"], "PCI_MCP_DATABASE_PASSWORD")
        self.assertTrue(database["password_set"])
        self.assertEqual(database["scope_path"], "/work/kubernetes-ingress")
        self.assertIn("pci_ro", cast("str", database["target"]))
        self.assertEqual(len(cast("str", database["fingerprint"])), 16)
        self.assertEqual(
            config_section["user_config_path"], "/home/tester/.config/project-code-intelligence/pci-index.env"
        )
        self.assertNotIn(credential_value, json.dumps(runtime))


class McpListFilesBooleanFilterTests(unittest.TestCase):
    def test_list_files_omitted_boolean_is_not_a_filter(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_list_code_intel_files({"source_path": "pkg/client/file.go"})

        query, _ = conn.calls[0]
        self.assertNotIn("f.is_generated = %s", query)
        self.assertNotIn("f.is_source = %s", query)

    def test_list_files_warns_on_empty_results_with_overconstrained_false_boolean_filters(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_list_code_intel_files({
                "source_path": "pkg/client/applyconfiguration/configuration/v1/accesscontrol.go",
                "is_generated": False,
                "is_source": False,
                "only_skipped": False,
            })

        payload = mcp_text_payload(response)
        warnings = cast("list[dict[str, object]]", payload["warnings"])
        overconstrained_warning = next(
            warning for warning in warnings if warning["kind"] == "overconstrained_boolean_filters"
        )
        self.assertEqual(overconstrained_warning["filters"], ["is_generated", "is_source"])
        self.assertIn("false is an active filter", cast("str", overconstrained_warning["message"]))

    def test_list_files_does_not_warn_for_single_false_boolean_filter(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_list_code_intel_files({"is_generated": False})

        payload = mcp_text_payload(response)
        warnings = cast("list[dict[str, object]]", payload.get("warnings", []))
        self.assertFalse(any(warning["kind"] == "overconstrained_boolean_filters" for warning in warnings))


class McpToolShapeTests(unittest.TestCase):
    def test_list_files_default_selects_slim_columns(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_list_code_intel_files({})

        query, _ = conn.calls[0]
        select_clause = query.rsplit("FROM file_inventory f", 1)[0].rsplit("SELECT", 1)[1]
        self.assertNotIn("f.snapshot_id", select_clause)
        self.assertNotIn("f.commit_sha", select_clause)
        self.assertNotIn("f.metadata", select_clause)
        self.assertNotIn("f.created_at", select_clause)
        self.assertIn("f.source_path", select_clause)
        self.assertIn("f.file_role", select_clause)
        # `f.repo` is needed so the formatter can strip the prefix from
        # `repo_path`; it's selected then dropped from the compact output.
        self.assertIn("f.repo", select_clause)

    def test_list_files_verbose_selects_all_columns(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            _ = mcp_tools.tool_list_code_intel_files({"verbose": True})

        query, _ = conn.calls[0]
        select_clause = query.rsplit("FROM file_inventory f", 1)[0].rsplit("SELECT", 1)[1]
        self.assertIn("f.snapshot_id", select_clause)
        self.assertIn("f.commit_sha", select_clause)
        self.assertIn("f.metadata", select_clause)
        self.assertIn("f.created_at", select_clause)

    def test_list_files_rejects_legacy_include_metadata(self) -> None:
        definition = TOOL_DEFINITIONS["list_code_intel_files"]
        with self.assertRaises(McpProtocolError):
            _ = validate_tool_arguments(definition, {"include_metadata": True})

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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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

    def test_get_record_omits_metadata_by_default_and_includes_compact_metadata_on_request(self) -> None:
        default_conn = QueuedConnection([
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(default_conn)),
        ):
            response = mcp_tools.tool_get_code_intel_record({"record_id": "README.md::doc::000001"})

        payload = mcp_text_payload(response)
        result = cast("dict[str, object]", payload["result"])
        self.assertNotIn("id", result)
        self.assertEqual(result["record_id"], "README.md::doc::000001")
        self.assertNotIn("metadata", result)

        metadata_conn = QueuedConnection([
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(metadata_conn)),
        ):
            response = mcp_tools.tool_get_code_intel_record({
                "record_id": "README.md::doc::000001",
                "include_metadata": True,
            })

        payload = mcp_text_payload(response)
        result = cast("dict[str, object]", payload["result"])
        metadata = cast("dict[str, object]", result["metadata"])
        self.assertNotIn("doc_links", metadata)
        self.assertIn("doc_headings", metadata)
        self.assertIn("doc_fenced_languages", metadata)

    def test_list_files_source_path_prefix_matches_subtree(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            self.assertRaises(McpProtocolError),
        ):
            _ = mcp_tools.tool_list_code_intel_files({
                "source_path": "cmd/main.go",
                "source_path_prefix": "cmd",
            })

    def test_status_default_uses_compact_freshness_snapshots(self) -> None:
        status_args: tuple[JsonObject, ...] = ({}, {"include_snapshots": True})
        for args in status_args:
            with self.subTest(args=args):
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
                    patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
                    patch.object(mcp_db, "table_regclass_exists", return_value=False),
                    patch.object(mcp_status, "schema_migration_versions", return_value=[]),
                    patch.object(git_utils, "run_git", side_effect=_git_show_current_branch_returns_main),
                    patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
                ):
                    response = mcp_tools.tool_code_intel_status(args)

                payload = mcp_text_payload(response)
                snapshots = cast("list[dict[str, object]]", payload["snapshots"])
                self.assertEqual(snapshots[0]["repo"], "zod")
                self.assertEqual(snapshots[0]["branch"], "main")
                self.assertEqual(snapshots[0]["commit_sha"], "abc123")
                self.assertNotIn("head_commit", snapshots[0])
                self.assertFalse(cast("bool", snapshots[0]["dirty"]))
                self.assertEqual(snapshots[0]["head_status"], "current")
                self.assertIsInstance(snapshots[0]["index_age_seconds"], int)
                self.assertNotIn("metadata", snapshots[0])
                self.assertNotIn("tree_sha", snapshots[0])
                self.assertNotIn("warnings", payload)
                self.assertNotIn("records_by_type", payload)
                self.assertNotIn("language_breakdown", payload)
                self.assertNotIn("static_findings", payload)

    def test_status_flags_branch_mismatch_when_commit_matches_but_branch_differs(self) -> None:
        # Same commit as live HEAD, but the snapshot is stamped "main" while the
        # live checkout has since moved to "feature": head_status must go
        # "stale" with reason "branch_mismatch", not "current".
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

        def fake_run_git(_root: object, args: list[str]) -> str:
            return "feature" if args[:2] == ["branch", "--show-current"] else "abc123"

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(mcp_status, "schema_migration_versions", return_value=[]),
            patch.object(git_utils, "run_git", side_effect=fake_run_git),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({"include_snapshots": True})

        payload = mcp_text_payload(response)
        snapshots = cast("list[dict[str, object]]", payload["snapshots"])
        self.assertEqual(snapshots[0]["head_status"], "stale")
        self.assertEqual(snapshots[0]["head_status_reason"], "branch_mismatch")

    def test_status_omits_redundant_scope_when_scoped(self) -> None:
        conn = QueuedConnection([
            FakeCursor(many=[{"id": 1, "collection": "zod", "repo": "zod", "metadata": {}}]),
            FakeCursor(many=[{"collection": "zod", "repo": "zod", "records": 7, "embedded_records": 5}]),
            FakeCursor(many=[{"collection": "zod", "repo": "zod", "record_type": "code_chunk", "count": 7}]),
            FakeCursor(many=[{"collection": "zod", "repo": "zod", "files": 3, "skipped_files": 0}]),
            FakeCursor(many=[{"collection": "zod", "repo": "zod", "edges": 2}]),
            FakeCursor(many=[]),
        ])

        with (
            patch.dict(os.environ, {"PCI_COLLECTION": "zod"}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(mcp_status, "schema_migration_versions", return_value=[]),
            patch.object(git_utils, "run_git", return_value="abc123"),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({"repo": "zod"})

        payload = mcp_text_payload(response)
        self.assertEqual(payload["collection"], "zod")
        self.assertEqual(payload["repo"], "zod")
        for row_set in ("snapshots", "files", "records", "edges"):
            rows = cast("list[dict[str, object]]", payload[row_set])
            self.assertNotIn("collection", rows[0])
            self.assertNotIn("repo", rows[0])

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
            # verbose=True also pulls file dimensions (language / file_role / content_class) for
            # the queryability section so empty_<dim>_scope warnings can point at valid values.
            FakeCursor(many=[]),
            FakeCursor(many=[]),
            FakeCursor(many=[]),
        ])

        with (
            patch.dict(os.environ, {"PCI_COLLECTION": "zod"}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(mcp_status, "schema_migration_versions", return_value=[]),
            patch.object(git_utils, "run_git", return_value="abc123"),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(mcp_status, "schema_migration_versions", return_value=[]),
            patch.object(git_utils, "run_git", return_value="abc123"),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
            patch.object(Path, "cwd", return_value=Path("/work/project-code-intelligence")),
            patch.object(git_utils, "run_git", return_value=None) as run_git,
        ):
            response = mcp_tools.tool_code_intel_status({"collection": "zod", "repo": "zod"})

        payload = mcp_text_payload(response)
        snapshots = cast("list[dict[str, object]]", payload["snapshots"])
        self.assertIsNone(snapshots[0]["head_commit"])
        self.assertIsNone(snapshots[0]["head_matches_snapshot"])
        self.assertEqual(snapshots[0]["head_status"], "unknown")
        self.assertEqual(snapshots[0]["head_status_reason"], "local_repo_unavailable")
        self.assertEqual(
            payload["warnings"],
            [
                {
                    "kind": "snapshot_freshness_unknown",
                    "message": "snapshot freshness could not be checked against local source",
                    "id": 1,
                    "collection": "zod",
                    "repo": "zod",
                    "commit_sha": "b6071fc0",
                    "head_status_reason": "local_repo_unavailable",
                }
            ],
        )
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
            patch.object(Path, "cwd", return_value=Path("/work/project-code-intelligence")),
            patch.object(git_utils, "run_git", side_effect=[None, "b6071fc0", None, None]) as run_git,
        ):
            response = mcp_tools.tool_code_intel_status({"collection": "zod", "repo": "zod"})

        payload = mcp_text_payload(response)
        snapshots = cast("list[dict[str, object]]", payload["snapshots"])
        self.assertNotIn("head_commit", snapshots[0])
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
                    {"id": 2, "record_id": "b::doc::000001", "source_path": "b.md"},
                    {"id": 1, "record_id": "a::doc::000001", "source_path": "a.md"},
                ]
            ),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_get_code_intel_record({
                "record_ids": ["a::doc::000001", "b::doc::000001", "missing::doc::000001"],
            })

        payload = mcp_text_payload(response)
        results = cast("list[dict[str, object]]", payload["results"])
        self.assertEqual([r["record_id"] for r in results], ["a::doc::000001", "b::doc::000001"])
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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


class McpStatusQueryabilityTests(unittest.TestCase):
    """Compact `code_intel_status` queryability surface vs. the detailed surface gated by
    `include_queryability=true`. Compact keeps only counts that suggest an action;
    `configured_embed_record_type_count` is descriptive (what's planned at indexing time)
    and lives behind `include_queryability`. `empty_embed_record_type_count` IS actionable —
    a non-zero value flags a freshness/coverage gap — so it stays in compact, but only when
    non-zero. The detailed surface still emits both counts unconditionally.
    """

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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(mcp_status, "schema_migration_versions", return_value=[]),
            patch.object(git_utils, "run_git", side_effect=_git_show_current_branch_returns_main),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({})

        payload = mcp_text_payload(response)
        # `configured_embed_record_type_count` is descriptive and now lives behind
        # `include_queryability`; only the actionable `empty_embed_record_type_count` stays
        # in compact (and only because it's non-zero here — see the companion test below).
        self.assertEqual(
            payload["queryability"],
            {
                "text_record_type_count": 3,
                "semantic_record_type_count": 2,
                "text_only_record_type_count": 1,
                "empty_embed_record_type_count": 1,
                "edge_type_count": 1,
                "has_text": True,
                "has_semantic": True,
                "has_edges": True,
            },
        )

    def test_status_compact_queryability_omits_empty_embed_count_when_zero(self) -> None:
        # When every configured embed type has at least one embedded record, the empty count
        # carries no signal and is dropped from the compact surface. include_queryability=True
        # would still surface it (covered by `test_status_can_include_queryability_record_type_lists`).
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "id": 1,
                        "commit_sha": "abc123",
                        "metadata": {
                            "embed_record_types": ["code_chunk", "security_pattern"],
                        },
                    }
                ]
            ),
            FakeCursor(many=[]),
            FakeCursor(
                many=[
                    {"record_type": "code_chunk", "count": 10, "embedded_records": 10},
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
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(mcp_status, "schema_migration_versions", return_value=[]),
            patch.object(git_utils, "run_git", side_effect=_git_show_current_branch_returns_main),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_code_intel_status({})

        payload = mcp_text_payload(response)
        queryability = cast("dict[str, object]", payload["queryability"])
        self.assertNotIn("empty_embed_record_type_count", queryability)
        self.assertNotIn("configured_embed_record_type_count", queryability)

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
            # include_queryability=True triggers distinct-value lookups for the three file
            # dimensions so empty_<dim>_scope warnings can point at concrete valid values.
            FakeCursor(many=[{"value": "go"}, {"value": "python"}]),
            FakeCursor(many=[{"value": "source"}, {"value": "test"}]),
            FakeCursor(many=[{"value": "code"}]),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "table_regclass_exists", return_value=False),
            patch.object(mcp_status, "schema_migration_versions", return_value=[]),
            patch.object(git_utils, "run_git", return_value="abc123"),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
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
                "language_count": 2,
                "file_role_count": 2,
                "content_class_count": 1,
                "has_text": True,
                "has_semantic": True,
                "has_edges": True,
                "text_record_types": ["code_chunk", "resource_object", "security_pattern"],
                "semantic_record_types": ["code_chunk", "security_pattern"],
                "text_only_record_types": ["resource_object"],
                "configured_embed_record_types": ["code_chunk", "resource_object", "security_pattern"],
                "empty_embed_record_types": ["resource_object"],
                "edge_types": ["call_candidate"],
                "languages": ["go", "python"],
                "file_roles": ["source", "test"],
                "content_classes": ["code"],
            },
        )


class McpRepoPathFieldTests(unittest.TestCase):
    """`repo_path` (and `source_repo_path` / `target_repo_path` on edges) is the
    repo-relative form of `source_path`, suitable for `Read`/`open` with cwd at the
    repo root. It must appear on every record/edge/file shape in both compact and
    verbose modes, and must equal `source_path` when no repo prefix is present.
    """

    def test_get_record_compact_adds_repo_path_stripped_of_repo_prefix(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                one={
                    "id": 1,
                    "repo": "zod",
                    "record_id": "zod/src/util.ts::function::defineLazy::000001",
                    "source_path": "zod/src/util.ts",
                    "title": "defineLazy",
                }
            ),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_get_code_intel_record({
                "record_id": "zod/src/util.ts::function::defineLazy::000001",
            })

        result = cast("dict[str, object]", mcp_text_payload(response)["result"])
        self.assertEqual(result["source_path"], "zod/src/util.ts")
        self.assertEqual(result["repo_path"], "src/util.ts")

    def test_get_record_verbose_adds_repo_path(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                one={
                    "id": 1,
                    "repo": "zod",
                    "record_id": "zod/src/util.ts::function::defineLazy::000001",
                    "source_path": "zod/src/util.ts",
                }
            ),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_get_code_intel_record({
                "record_id": "zod/src/util.ts::function::defineLazy::000001",
                "verbose": True,
            })

        result = cast("dict[str, object]", mcp_text_payload(response)["result"])
        self.assertEqual(result["source_path"], "zod/src/util.ts")
        self.assertEqual(result["repo_path"], "src/util.ts")
        # Verbose keeps repo intact.
        self.assertEqual(result["repo"], "zod")

    def test_repo_path_equals_source_path_when_repo_is_dot(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                one={
                    "id": 1,
                    "repo": ".",
                    "record_id": "src/util.ts::function::defineLazy::000001",
                    "source_path": "src/util.ts",
                }
            ),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_get_code_intel_record({
                "record_id": "src/util.ts::function::defineLazy::000001",
            })

        result = cast("dict[str, object]", mcp_text_payload(response)["result"])
        self.assertEqual(result["source_path"], "src/util.ts")
        self.assertEqual(result["repo_path"], "src/util.ts")

    def test_get_record_batch_adds_repo_path_to_each_result(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "id": 1,
                        "repo": "zod",
                        "record_id": "zod/a.ts::doc::000001",
                        "source_path": "zod/a.ts",
                    },
                    {
                        "id": 2,
                        "repo": "zod",
                        "record_id": "zod/b.ts::doc::000001",
                        "source_path": "zod/b.ts",
                    },
                ]
            ),
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_get_code_intel_record({
                "record_ids": ["zod/a.ts::doc::000001", "zod/b.ts::doc::000001"],
            })

        results = cast("list[dict[str, object]]", mcp_text_payload(response)["results"])
        self.assertEqual([r["repo_path"] for r in results], ["a.ts", "b.ts"])

    def test_related_edges_add_source_and_target_repo_path(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "id": 2,
                        "snapshot_id": 1,
                        "collection": "zod",
                        "repo": "zod",
                        "commit_sha": "abc123",
                        "source_record_id": "zod/src/app.ts::function::bootstrap::000001",
                        "target_record_id": "zod/src/parse.ts::function::parse::000020",
                        "edge_type": "call_candidate",
                        "source_symbol": "bootstrap",
                        "target_symbol": "parse",
                        "source_path": "zod/src/app.ts",
                        "target_path": "zod/src/parse.ts",
                        "confidence_kind": "high_confidence_fact",
                        "target_resolved": True,
                        "target_kind": "project_symbol",
                        "source_record_db_id": 7,
                        "target_record_db_id": 8,
                    },
                ]
            )
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_related_code_intel({"symbol": "bootstrap", "direction": "outgoing"})

        edge = cast("list[dict[str, object]]", mcp_text_payload(response)["edges"])[0]
        self.assertEqual(edge["source_path"], "zod/src/app.ts")
        self.assertEqual(edge["source_repo_path"], "src/app.ts")
        self.assertEqual(edge["target_path"], "zod/src/parse.ts")
        self.assertEqual(edge["target_repo_path"], "src/parse.ts")

    def test_list_files_compact_adds_repo_path(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "repo": "zod",
                        "source_path": "zod/src/index.ts",
                        "language": "typescript",
                        "file_role": "source",
                    }
                ]
            )
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_list_code_intel_files({})

        files = cast("list[dict[str, object]]", mcp_text_payload(response)["files"])
        self.assertEqual(files[0]["source_path"], "zod/src/index.ts")
        self.assertEqual(files[0]["repo_path"], "src/index.ts")
        # `repo` is selected so the formatter can compute `repo_path`, but it's
        # redundant with the response envelope in compact mode.
        self.assertNotIn("repo", files[0])

    def test_semantic_search_results_carry_repo_path(self) -> None:
        conn = QueuedConnection([
            FakeCursor(
                many=[
                    {
                        "record_id": "zod/src/util.ts::chunk::000001-000020",
                        "repo": "zod",
                        "source_path": "zod/src/util.ts",
                        "title": "util chunk",
                        "summary": "util chunk",
                        "record_type": "code_chunk",
                        "distance": 0.2,
                        "match_score": 1.0,
                        "quality_penalty": 0.0,
                        "snippet_raw": "```ts\nfoo();\n```",
                    }
                ]
            )
        ])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_tools, "query_embedding", return_value=("[0.1,0.2]", 2)),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_search_code_intel_semantic({"query": "utility helper"})

        result = cast("list[dict[str, object]]", mcp_text_payload(response)["results"])[0]
        self.assertEqual(result["source_path"], "zod/src/util.ts")
        self.assertEqual(result["repo_path"], "src/util.ts")


class McpServerEntryPointTests(unittest.TestCase):
    def test_initialize_response_advertises_dynamic_server_version(self) -> None:
        with patch("project_code_intelligence.mcp.transport.server_version", return_value="9.9.9+abcdef0"):
            response = mcp_transport.control_response("initialize", "id-1")

        if response is None:
            raise AssertionError("expected initialize response")
        result = cast("dict[str, object]", response["result"])
        server_info = cast("dict[str, object]", result["serverInfo"])
        self.assertEqual(server_info["version"], "9.9.9+abcdef0")
        self.assertEqual(server_info["name"], "project-code-intelligence")

    def test_server_version_appends_short_commit_when_available(self) -> None:
        with (
            patch.object(mcp_transport, "_git_short_commit", return_value="abc1234"),
            patch.object(mcp_transport.importlib_metadata, "version", return_value="0.1.0"),
        ):
            self.assertEqual(mcp_transport.server_version(), "0.1.0+abc1234")

    def test_server_version_omits_commit_when_unavailable(self) -> None:
        with (
            patch.object(mcp_transport, "_git_short_commit", return_value=None),
            patch.object(mcp_transport.importlib_metadata, "version", return_value="0.1.0"),
        ):
            self.assertEqual(mcp_transport.server_version(), "0.1.0")

    def test_pci_mcp_help_flag_exits_cleanly(self) -> None:
        captured = io.StringIO()
        with (
            contextlib.redirect_stdout(captured),
            self.assertRaises(SystemExit) as raised,
        ):
            _ = mcp_server.main(["--help"])

        # argparse exits 0 for --help / --version
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("pci mcp", captured.getvalue())

    def test_pci_mcp_version_flag_exits_cleanly(self) -> None:
        captured = io.StringIO()
        with (
            patch("project_code_intelligence.server.server_version", return_value="0.1.0+abc1234"),
            contextlib.redirect_stdout(captured),
            self.assertRaises(SystemExit) as raised,
        ):
            _ = mcp_server.main(["--version"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("0.1.0+abc1234", captured.getvalue())

    def test_pci_mcp_rejects_unknown_arguments(self) -> None:
        captured_err = io.StringIO()
        with (
            contextlib.redirect_stderr(captured_err),
            self.assertRaises(SystemExit) as raised,
        ):
            _ = mcp_server.main(["--bogus"])

        # argparse exits 2 for usage errors
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--bogus", captured_err.getvalue())

    def test_pci_mcp_without_args_runs_stdio_loop(self) -> None:
        with patch("project_code_intelligence.server.stdio_main", return_value=0) as stdio_main:
            exit_code = mcp_server.main([])

        self.assertEqual(exit_code, 0)
        stdio_main.assert_called_once_with()

    def test_pci_mcp_scope_loads_private_credentials_before_stdio(self) -> None:
        with (
            patch("project_code_intelligence.server.mcp_credentials.load") as load,
            patch("project_code_intelligence.server.stdio_main", return_value=0) as stdio_main,
        ):
            exit_code = mcp_server.main(["--scope", "/work/demo"])

        self.assertEqual(exit_code, 0)
        load.assert_called_once_with(Path("/work/demo"))
        stdio_main.assert_called_once_with()


class ToolRegistryConsistencyTests(unittest.TestCase):
    """Catch drift between the three name-keyed tool registries.

    Tools live in three places keyed by name: TOOL_DEFINITIONS (JSON Schema),
    TOOL_INPUT_MODELS (Pydantic validation), and TOOLS (runtime dispatch).
    Adding a tool requires editing all three; missing one is a silent failure.
    """

    def test_tool_definitions_match_input_models(self) -> None:
        self.assertEqual(TOOL_DEFINITIONS.keys(), TOOL_INPUT_MODELS.keys())

    def test_tools_registry_matches_definitions(self) -> None:
        self.assertEqual(mcp_tools.TOOLS.keys(), TOOL_DEFINITIONS.keys())
        for name, (definition, handler) in mcp_tools.TOOLS.items():
            self.assertIs(
                definition,
                TOOL_DEFINITIONS[name],
                msg=f"{name}: TOOLS definition is not the same object as TOOL_DEFINITIONS[{name!r}]",
            )
            self.assertTrue(callable(handler), msg=f"{name}: handler is not callable")


class BlastRadiusToolContractTests(unittest.TestCase):
    def test_advertised_as_read_tool(self) -> None:
        self.assertIn("blast_radius", TOOL_DEFINITIONS)
        self.assertIn("blast_radius", TOOL_INPUT_MODELS)
        self.assertFalse(TOOL_DEFINITIONS["blast_radius"].write_tool)
        names = {cast("dict[str, object]", tool)["name"] for tool in mcp_tools.advertised_tools()}
        self.assertIn("blast_radius", names)

    def test_input_rejects_unknown_property(self) -> None:
        with self.assertRaises((McpProtocolError, McpProtocolTypeError)):
            _ = validate_tool_arguments(TOOL_DEFINITIONS["blast_radius"], {"symbol": "x", "bogus": 1})

    def test_input_accepts_symbol_and_neighbors(self) -> None:
        validated = validate_tool_arguments(TOOL_DEFINITIONS["blast_radius"], {"symbol": "render_text", "neighbors": 0})
        self.assertEqual(cast("dict[str, object]", validated)["symbol"], "render_text")

    def test_requires_symbol_or_source_path(self) -> None:
        with self.assertRaises(McpProtocolError):
            _ = mcp_tools.tool_blast_radius({})

    def test_reports_when_schema_not_initialized(self) -> None:
        with (
            patch.object(mcp_db, "code_intel_tables_exist", return_value=False),
            patch.object(mcp_db, "connect", return_value=FakeConnect(FakeConnection())),
        ):
            response = mcp_tools.tool_blast_radius({"symbol": "x"})
        self.assertIn("error", mcp_text_payload(response))

    def test_missing_symbol_returns_not_found_warning(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])  # latest_snapshots -> no snapshots
        with (
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_blast_radius({"symbol": "does_not_exist"})
        payload = mcp_text_payload(response)
        self.assertEqual(payload["found"], False)
        self.assertEqual(payload["count"], 0)
        warnings = cast("list[object]", payload["warnings"])
        kinds = {cast("dict[str, object]", warning)["kind"] for warning in warnings}
        self.assertIn("symbol_not_found", kinds)


class FindRedundancyToolContractTests(unittest.TestCase):
    def test_advertised_as_read_tool(self) -> None:
        self.assertIn("find_redundancy", TOOL_DEFINITIONS)
        self.assertIn("find_redundancy", TOOL_INPUT_MODELS)
        self.assertFalse(TOOL_DEFINITIONS["find_redundancy"].write_tool)
        names = {cast("dict[str, object]", tool)["name"] for tool in mcp_tools.advertised_tools()}
        self.assertIn("find_redundancy", names)

    def test_input_rejects_unknown_property(self) -> None:
        with self.assertRaises((McpProtocolError, McpProtocolTypeError)):
            _ = validate_tool_arguments(TOOL_DEFINITIONS["find_redundancy"], {"limit": 5, "bogus": 1})

    def test_input_accepts_prefix_and_limit(self) -> None:
        validated = validate_tool_arguments(
            TOOL_DEFINITIONS["find_redundancy"], {"source_path_prefix": "src/pkg", "limit": 3}
        )
        self.assertEqual(cast("dict[str, object]", validated)["source_path_prefix"], "src/pkg")

    def test_input_rejects_limit_over_maximum(self) -> None:
        with self.assertRaises((McpProtocolError, McpProtocolTypeError)):
            _ = validate_tool_arguments(TOOL_DEFINITIONS["find_redundancy"], {"limit": 51})

    def test_reports_when_schema_not_initialized(self) -> None:
        with (
            patch.object(mcp_db, "code_intel_tables_exist", return_value=False),
            patch.object(mcp_db, "connect", return_value=FakeConnect(FakeConnection())),
        ):
            response = mcp_tools.tool_find_redundancy({})
        self.assertIn("error", mcp_text_payload(response))

    def test_empty_scope_warns_and_finds_nothing(self) -> None:
        conn = QueuedConnection([FakeCursor(many=[])])  # latest_snapshots -> no snapshots
        with (
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_find_redundancy({"repo": "absent"})
        payload = mcp_text_payload(response)
        self.assertEqual(payload["found"], False)
        self.assertEqual(payload["count"], 0)
        warnings = cast("list[object]", payload["warnings"])
        kinds = {cast("dict[str, object]", warning)["kind"] for warning in warnings}
        self.assertIn("empty_repo_scope", kinds)

    def test_group_surfaces_text_similarity(self) -> None:
        # avg_text is evidence-only: it must reach the wire response alongside
        # the existing graph/semantic similarity fields.
        members = [
            analyze.FunctionNode(
                record_id="a.py::function::create_user::000010",
                symbol="create_user",
                source_path="svc/user.py",
                line_start=10,
                line_end=20,
                callee_roles=analyze.role_set(["validate_user", "convert_user", "repo.insert", "map_error"]),
            ),
            analyze.FunctionNode(
                record_id="a.py::function::create_team::000010",
                symbol="create_team",
                source_path="svc/team.py",
                line_start=10,
                line_end=20,
                callee_roles=analyze.role_set(["validate_team", "convert_team", "repo.insert", "map_error"]),
            ),
        ]
        group = analyze.build_group(members, avg_semantic=0.9, avg_text=0.95, max_text=1.0)
        snapshot_result = analyze.SnapshotResult(
            label="default/demo", groups=(group,), functions_analyzed=2, clones_folded=0
        )
        conn = QueuedConnection([FakeCursor(many=[{"id": 1, "collection": "default", "repo": "demo"}])])
        with (
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
            patch.object(analyze, "analyze_snapshot", return_value=snapshot_result),
        ):
            response = mcp_tools.tool_find_redundancy({})
        payload = mcp_text_payload(response)
        groups = cast("list[dict[str, object]]", payload["groups"])
        self.assertEqual(groups[0]["text_similarity"], 0.95)
        self.assertEqual(groups[0]["max_text_similarity"], 1.0)
        self.assertEqual(groups[0]["semantic_similarity"], 0.9)
        self.assertEqual(groups[0]["coherence"], 0.95)

    def test_group_surfaces_typed_variants(self) -> None:
        # typed_variants is evidence-only: it must reach the wire response, and
        # a group flagged this way must not be recommended as worth-collapsing.
        members = [
            analyze.FunctionNode(
                record_id="a.py::function::get_int::000010",
                symbol="get_int",
                source_path="svc/opt.py",
                line_start=10,
                line_end=20,
                callee_roles=analyze.role_set(["validate_x", "convert_x", "repo.get", "map_error"]),
            ),
            analyze.FunctionNode(
                record_id="a.py::function::get_str::000010",
                symbol="get_str",
                source_path="svc/opt.py",
                line_start=40,
                line_end=50,
                callee_roles=analyze.role_set(["validate_x", "convert_x", "repo.get", "map_error"]),
            ),
        ]
        group = analyze.build_group(members, avg_semantic=0.9, avg_text=0.95, typed_variants=True)
        snapshot_result = analyze.SnapshotResult(
            label="default/demo", groups=(group,), functions_analyzed=2, clones_folded=0
        )
        conn = QueuedConnection([FakeCursor(many=[{"id": 1, "collection": "default", "repo": "demo"}])])
        with (
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
            patch.object(analyze, "analyze_snapshot", return_value=snapshot_result),
        ):
            response = mcp_tools.tool_find_redundancy({})
        payload = mcp_text_payload(response)
        groups = cast("list[dict[str, object]]", payload["groups"])
        self.assertEqual(groups[0]["typed_variants"], True)
        self.assertEqual(groups[0]["recommendation"], "leave-as-is")

    @staticmethod
    def _two_branch_rows() -> list[object]:
        return [
            {"id": 1, "collection": "default", "repo": "demo", "branch": "main"},
            {"id": 2, "collection": "default", "repo": "demo", "branch": "feature"},
        ]

    def test_branch_arg_filters_snapshots_to_that_branch(self) -> None:
        # A branch given as an argument keeps only the same-branch snapshot.
        snapshot_result = analyze.SnapshotResult(label="default/demo", groups=(), functions_analyzed=0, clones_folded=0)
        conn = QueuedConnection([FakeCursor(many=self._two_branch_rows())])
        with (
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
            patch.object(analyze, "analyze_snapshot", return_value=snapshot_result) as analyze_snapshot,
        ):
            _ = mcp_tools.tool_find_redundancy({"branch": "feature"})
        analyzed_snapshots = [cast("analyze.SnapshotRef", call.args[1]) for call in analyze_snapshot.call_args_list]
        self.assertEqual([s.snapshot_id for s in analyzed_snapshots], [2])

    def test_branch_arg_with_no_match_warns_empty_scope(self) -> None:
        # A branch given as an argument that matches nothing gets no silent
        # fallback for MCP callers; it warns instead (unlike the CLI heuristic).
        conn = QueuedConnection([FakeCursor(many=self._two_branch_rows())])
        with (
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
        ):
            response = mcp_tools.tool_find_redundancy({"branch": "nope"})
        payload = mcp_text_payload(response)
        self.assertEqual(payload["found"], False)
        warnings = cast("list[object]", payload["warnings"])
        kinds = {cast("dict[str, object]", warning)["kind"] for warning in warnings}
        self.assertIn("empty_repo_scope", kinds)

    def test_no_branch_arg_collapses_to_newest_per_repo(self) -> None:
        # No branch given: process one snapshot per repo (the newest), not one per branch.
        snapshot_result = analyze.SnapshotResult(label="default/demo", groups=(), functions_analyzed=0, clones_folded=0)
        conn = QueuedConnection([FakeCursor(many=self._two_branch_rows())])
        with (
            patch.object(mcp_db, "code_intel_tables_exist", return_value=True),
            patch.object(mcp_db, "connect", return_value=FakeConnect(conn)),
            patch.object(analyze, "analyze_snapshot", return_value=snapshot_result) as analyze_snapshot,
        ):
            _ = mcp_tools.tool_find_redundancy({})
        analyzed_snapshots = [cast("analyze.SnapshotRef", call.args[1]) for call in analyze_snapshot.call_args_list]
        self.assertEqual([s.snapshot_id for s in analyzed_snapshots], [2])


if __name__ == "__main__":
    _ = unittest.main()
