from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from project_code_intelligence import cli


def mcp_response(payload: dict[str, object]) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload),
                }
            ]
        },
    }


class CliWrapperTests(unittest.TestCase):
    def test_pci_index_enables_embeddings_by_default(self) -> None:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main([".", "--dry-run"])

        self.assertEqual(status, 0)
        self.assertIn("--embed", forwarded)
        self.assertIn("--embedding-endpoint", forwarded)
        self.assertIn("--root", forwarded)
        self.assertIn("--repos", forwarded)
        self.assertIn("--collection", forwarded)
        self.assertEqual(forwarded[forwarded.index("--repos") + 1], Path.cwd().name)
        self.assertEqual(forwarded[forwarded.index("--collection") + 1], Path.cwd().name)
        self.assertIn("--dry-run", forwarded)

    def test_pci_index_requires_repo_path_for_indexing(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main") as ingest_main,
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            _ = cli.index_main(["--dry-run"])

        self.assertEqual(raised.exception.code, 2)
        ingest_main.assert_not_called()

    def test_repo_paths_map_to_common_root_and_repo_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_a = root / "repo-a"
            repo_b = root / "repo-b"
            repo_a.mkdir()
            repo_b.mkdir()

            args = cli.repo_paths_to_ingest_args([str(repo_a), str(repo_b)])

        self.assertEqual(args, ["--root", str(root.resolve()), "--repos", "repo-a,repo-b"])

    def test_single_repo_path_maps_to_parent_root_and_repo_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "openwrt"
            repo.mkdir()

            args = cli.repo_paths_to_ingest_args([str(repo)])

        self.assertEqual(args, ["--root", str(root.resolve()), "--repos", "openwrt"])

    def test_collection_is_inferred_from_repo_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_a = root / "repo-a"
            repo_b = root / "repo-b"
            repo_a.mkdir()
            repo_b.mkdir()

            single_collection = cli.inferred_collection_for_repo_paths([str(repo_a)])
            workspace_collection = cli.inferred_collection_for_repo_paths([str(repo_a), str(repo_b)])

        self.assertEqual(single_collection, "repo-a")
        self.assertEqual(workspace_collection, root.name)

    def test_pci_index_collection_override_forwards_to_ingest(self) -> None:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main(["--collection", "custom-workspace", ".", "--dry-run"])

        self.assertEqual(status, 0)
        self.assertIn("--collection", forwarded)
        self.assertEqual(forwarded[forwarded.index("--collection") + 1], "custom-workspace")

    def test_pci_index_collection_env_remains_an_override(self) -> None:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {"PROJECT_CODE_INTELLIGENCE_COLLECTION": "env-workspace"}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main([".", "--dry-run"])
            self.assertEqual(os.environ["PROJECT_CODE_INTELLIGENCE_COLLECTION"], "env-workspace")

        self.assertEqual(status, 0)
        self.assertNotIn("--collection", forwarded)

    def test_pci_index_no_embed_is_explicit_text_only_mode(self) -> None:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main([".", "--no-embed", "--dry-run"])
            self.assertEqual(os.environ["PROJECT_CODE_INTELLIGENCE_EMBED"], "0")

        self.assertEqual(status, 0)
        self.assertNotIn("--embed", forwarded)
        self.assertNotIn("--embedding-endpoint", forwarded)
        self.assertIn("--dry-run", forwarded)

    def test_pci_index_exposes_and_forwards_reset_flags(self) -> None:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main([
                "--reset-code-intel",
                "--i-know-this-deletes-code-intel-db",
                ".",
            ])

        self.assertEqual(status, 0)
        self.assertIn("--reset-code-intel", cli.index_parser().format_help())
        self.assertIn("--reset", cli.index_parser().format_help())
        self.assertIn("--reset-code-intel", forwarded)
        self.assertIn("--reset-only", forwarded)
        self.assertIn("--i-know-this-deletes-code-intel-db", forwarded)
        self.assertIn("--root", forwarded)
        self.assertIn("--repos", forwarded)
        self.assertNotIn("--embed", forwarded)
        self.assertNotIn("--embedding-endpoint", forwarded)

    def test_pci_index_reset_alias_forwards_reset_flags(self) -> None:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main(["--reset", "--i-know-this-deletes-code-intel-db", "."])

        self.assertEqual(status, 0)
        self.assertIn("--reset-code-intel", forwarded)
        self.assertIn("--reset-only", forwarded)

    def test_pci_index_reset_all_forwards_explicit_all_reset(self) -> None:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main(["--reset-all", "--i-know-this-deletes-code-intel-db"])

        self.assertEqual(status, 0)
        self.assertIn("--reset-all", cli.index_parser().format_help())
        self.assertIn("--reset-all-code-intel", forwarded)
        self.assertIn("--reset-code-intel", forwarded)
        self.assertIn("--reset-only", forwarded)
        self.assertNotIn("--root", forwarded)
        self.assertNotIn("--repos", forwarded)

    def test_pci_index_reset_all_rejects_repo_paths(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main") as ingest_main,
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            _ = cli.index_main(["--reset-all", "."])

        self.assertEqual(raised.exception.code, 2)
        ingest_main.assert_not_called()

    def test_pci_index_reset_all_rejects_collection(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main") as ingest_main,
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            _ = cli.index_main(["--reset-all", "--collection", "workspace"])

        self.assertEqual(raised.exception.code, 2)
        ingest_main.assert_not_called()

    def test_mcp_smoke_current_workspace_probes_available_repos(self) -> None:
        calls: list[tuple[str, dict[str, object], int]] = []

        def fake_mcp_call(
            tool_name: str, arguments: dict[str, object], request_id: int = 1
        ) -> tuple[int, object | None, str]:
            calls.append((tool_name, arguments, request_id))
            if tool_name == "code_intel_status":
                return (
                    0,
                    mcp_response({
                        "schema_present": True,
                        "snapshots": [
                            {"repo": "openwrt", "id": 1},
                            {"repo": "ask-cmm", "id": 2},
                        ],
                    }),
                    "",
                )
            return 0, mcp_response({"results": [], "edges": [], "files": [], "parser_failures": []}), ""

        with tempfile.TemporaryDirectory() as directory:
            old_cwd = Path.cwd()
            try:
                os.chdir(directory)
                stdout = io.StringIO()
                with (
                    patch("project_code_intelligence.cli._run_mcp_call", side_effect=fake_mcp_call),
                    patch("sys.stdout", stdout),
                ):
                    status = cli.mcp_smoke_main(["--json", "."])
            finally:
                os.chdir(old_cwd)

        self.assertEqual(status, 0)
        self.assertEqual(calls[0], ("code_intel_status", {}, 1))
        probe_repos = [args["repo"] for tool, args, _request_id in calls[1:] if tool == "search_code_intel_text"]
        self.assertEqual(probe_repos, ["openwrt", "ask-cmm"])

        payload = cast("dict[str, object]", json.loads(stdout.getvalue()))
        self.assertEqual(payload["repos"], ["openwrt", "ask-cmm"])

    def test_mcp_smoke_fails_on_status_payload_error(self) -> None:
        def fake_mcp_call(
            tool_name: str, arguments: dict[str, object], request_id: int = 1
        ) -> tuple[int, object | None, str]:
            del tool_name, arguments, request_id
            return 0, mcp_response({"error": "code intelligence schema is not initialized"}), ""

        stdout = io.StringIO()
        with (
            patch("project_code_intelligence.cli._run_mcp_call", side_effect=fake_mcp_call),
            patch("sys.stdout", stdout),
        ):
            status = cli.mcp_smoke_main(["--json", "."])

        self.assertEqual(status, 1)
        payload = cast("dict[str, object]", json.loads(stdout.getvalue()))
        result = cast("dict[str, object]", payload["result"])
        content = cast("list[dict[str, object]]", result["content"])
        self.assertIn("code intelligence schema is not initialized", str(content[0]["text"]))

    def test_mcp_smoke_fails_on_probe_payload_error(self) -> None:
        def fake_mcp_call(
            tool_name: str, arguments: dict[str, object], request_id: int = 1
        ) -> tuple[int, object | None, str]:
            del arguments, request_id
            if tool_name == "code_intel_status":
                return 0, mcp_response({"schema_present": True, "snapshots": [{"repo": "repo-a", "id": 1}]}), ""
            return 0, mcp_response({"error": "semantic search requires an embedding endpoint"}), ""

        stdout = io.StringIO()
        with (
            patch("project_code_intelligence.cli._run_mcp_call", side_effect=fake_mcp_call),
            patch("sys.stdout", stdout),
        ):
            status = cli.mcp_smoke_main(["--json", "repo-a"])

        self.assertEqual(status, 1)
        payload = cast("dict[str, object]", json.loads(stdout.getvalue()))
        probes = cast("list[dict[str, object]]", payload["probes"])
        self.assertEqual(probes[0]["status"], "fail")
        self.assertEqual(probes[0]["error"], "semantic search requires an embedding endpoint")
