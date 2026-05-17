from __future__ import annotations

import io
import json
import unittest
from typing import TYPE_CHECKING
from unittest.mock import patch

from project_code_intelligence import console_ui, mcp_smoke_render

if TYPE_CHECKING:
    from collections.abc import Callable

    from project_code_intelligence.models import JsonObject


def mcp_response(payload: object) -> JsonObject:
    return {"result": {"content": [{"type": "text", "text": json.dumps(payload)}]}}


def render_output(render: Callable[[], None]) -> str:
    output = io.StringIO()
    console = console_ui.build_console(file=output, color=False)
    with patch.object(console_ui, "build_console", return_value=console):
        render()
    return output.getvalue()


class McpSmokeRenderTests(unittest.TestCase):
    def test_render_status_outputs_health_counts_and_probe_details(self) -> None:
        payload: JsonObject = {
            "schema_present": True,
            "schema_versions": ["schema-v2", "parser-v15"],
            "snapshots": [
                {
                    "repo": "demo",
                    "id": 7,
                    "commit_sha": "abcdef1234567890",
                    "branch": "main",
                    "dirty": True,
                }
            ],
            "files": [{"repo": "demo", "files": 1200, "skipped_files": 3}],
            "records": [{"repo": "demo", "records": 10, "embedded_records": 7}],
            "edges": [{"repo": "demo", "edges": 4}],
            "static_findings": [{"repo": "demo"}, {"repo": "demo"}, {"repo": "other"}],
            "records_by_type": [
                {"repo": "demo", "record_type": "static_finding", "count": 2},
                {"repo": "demo", "record_type": "code_chunk", "count": 8},
                {"repo": "other", "record_type": "code_chunk", "count": 99},
            ],
        }
        probes: list[dict[str, object]] = [
            {
                "repo": "demo",
                "tool": "search_code_intel_text",
                "status": "ok",
                "response": mcp_response({"results": [{"id": 1}, {"id": 2}]}),
            },
            {"repo": "demo", "tool": "get_code_intel_status", "status": "ok", "response": mcp_response({})},
            {"repo": "demo", "tool": "related_code_intel", "status": "error", "error": "probe failed"},
        ]

        output = render_output(
            lambda: mcp_smoke_render.render_status(mcp_response(payload), repo="demo", probes=probes)
        )

        self.assertIn("pci-mcp demo", output)
        self.assertIn("HEALTHY", output)
        self.assertIn("schema-v2, parser-v15", output)
        self.assertIn("#7", output)
        self.assertIn("abcdef1", output)
        self.assertIn("main", output)
        self.assertIn("dirty", output)
        self.assertIn("1,200", output)
        self.assertIn("3 skipped", output)
        self.assertIn("10", output)
        self.assertIn("7 embedded (70%)", output)
        self.assertIn("Static findings", output)
        self.assertIn("Records by type", output)
        self.assertLess(output.index("code_chunk"), output.index("static_finding"))
        self.assertIn("MCP tool probes", output)
        self.assertIn("demo: search_code_intel_text", output)
        self.assertIn("2 item(s)", output)
        self.assertIn("get_code_intel_status", output)
        self.assertIn("ok", output)
        self.assertIn("probe failed", output)

    def test_render_status_reports_missing_schema_and_unparseable_payload(self) -> None:
        missing_schema = render_output(
            lambda: mcp_smoke_render.render_status(mcp_response({"schema_present": False}), repo="demo")
        )
        bad_payload = render_output(
            lambda: mcp_smoke_render.render_status({"result": {"content": [{"text": "not json"}]}}, repo="demo")
        )

        self.assertIn("SCHEMA MISSING", missing_schema)
        self.assertIn("Schema", missing_schema)
        self.assertIn("missing", missing_schema)
        self.assertIn("Could not parse MCP response payload.", bad_payload)

    def test_render_error_prefers_payload_error_over_outer_error(self) -> None:
        payload_error = {
            "error": {"message": "outer failure"},
            "result": {"content": [{"type": "text", "text": json.dumps({"error": "inner failure"})}]},
        }
        outer_error = {"error": {"message": "outer failure"}}

        output = render_output(lambda: mcp_smoke_render.render_error(payload_error))
        fallback = render_output(lambda: mcp_smoke_render.render_error(outer_error))

        self.assertIn("inner failure", output)
        self.assertNotIn("outer failure", output)
        self.assertIn("outer failure", fallback)


if __name__ == "__main__":
    _ = unittest.main()
