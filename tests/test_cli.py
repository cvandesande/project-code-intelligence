from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from project_code_intelligence import cli, config
from project_code_intelligence.embeddings import EmbeddingEndpointUnavailableError


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
        captured_scope_path: str | None = None

        def fake_ingest_main(args: list[str]) -> int:
            nonlocal captured_scope_path
            captured_scope_path = os.environ.get(config.DATABASE_SCOPE_PATH_ENV)
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
        self.assertEqual(captured_scope_path, str(Path.cwd().resolve()))
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

    def test_multiple_repo_paths_map_to_cwd_root_and_repo_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_a = root / "repo-a"
            repo_b = root / "repo-b"
            repo_a.mkdir()
            repo_b.mkdir()

            original_cwd = Path.cwd()
            os.chdir(root)
            try:
                args = cli.repo_paths_to_ingest_args([str(repo_a), str(repo_b)])
            finally:
                os.chdir(original_cwd)

        self.assertEqual(args, ["--root", str(root.resolve()), "--repos", "repo-a,repo-b"])

    def test_multiple_repo_paths_must_be_under_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            inside_repo = workspace / "repo-a"
            outside_repo = root / "repo-b"
            inside_repo.mkdir(parents=True)
            outside_repo.mkdir()

            original_cwd = Path.cwd()
            os.chdir(workspace)
            try:
                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch("project_code_intelligence.cli.ingest_code_intel.cli_main") as ingest_main,
                    patch("sys.stderr", io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    _ = cli.index_main(["repo-a", str(outside_repo), "--dry-run"])
            finally:
                os.chdir(original_cwd)

        self.assertEqual(raised.exception.code, 2)
        ingest_main.assert_not_called()

    def test_single_repo_path_maps_to_parent_root_and_repo_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "service-api"
            repo.mkdir()

            args = cli.repo_paths_to_ingest_args([str(repo)])

        self.assertEqual(args, ["--root", str(root.resolve()), "--repos", "service-api"])

    def test_collection_is_inferred_from_repo_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_a = root / "repo-a"
            repo_b = root / "repo-b"
            repo_a.mkdir()
            repo_b.mkdir()

            original_cwd = Path.cwd()
            os.chdir(root)
            try:
                single_collection = cli.inferred_collection_for_repo_paths([str(repo_a)])
                workspace_collection = cli.inferred_collection_for_repo_paths(["repo-a", "repo-b"])
            finally:
                os.chdir(original_cwd)

        self.assertEqual(single_collection, "repo-a")
        self.assertEqual(workspace_collection, root.name)

    def test_database_scope_is_inferred_from_repo_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_a = root / "repo-a"
            repo_b = root / "repo-b"
            repo_a.mkdir()
            repo_b.mkdir()

            original_cwd = Path.cwd()
            os.chdir(root)
            try:
                single_scope = cli.inferred_database_scope_path_for_repo_paths([str(repo_a)])
                workspace_scope = cli.inferred_database_scope_path_for_repo_paths(["repo-a", "repo-b"])
            finally:
                os.chdir(original_cwd)

        self.assertEqual(single_scope, repo_a.resolve())
        self.assertEqual(workspace_scope, root.resolve())

    def test_pci_index_multiple_repos_use_one_workspace_database_scope(self) -> None:
        forwarded: list[str] = []
        captured_scope_path: str | None = None

        def fake_ingest_main(args: list[str]) -> int:
            nonlocal captured_scope_path
            captured_scope_path = os.environ.get(config.DATABASE_SCOPE_PATH_ENV)
            forwarded.extend(args)
            return 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_a = root / "repo-a"
            repo_b = root / "repo-b"
            repo_a.mkdir()
            repo_b.mkdir()

            original_cwd = Path.cwd()
            os.chdir(root)
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
            ):
                try:
                    status = cli.index_main(["repo-a", "repo-b", "--no-embed", "--dry-run"])
                finally:
                    os.chdir(original_cwd)

            expected_root = str(root.resolve())

        self.assertEqual(status, 0)
        self.assertEqual(captured_scope_path, expected_root)
        self.assertEqual(forwarded[forwarded.index("--root") + 1], expected_root)
        self.assertEqual(forwarded[forwarded.index("--repos") + 1], "repo-a,repo-b")
        self.assertEqual(forwarded[forwarded.index("--collection") + 1], root.name)

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

    def test_pci_index_repo_paths_override_stale_project_environment(self) -> None:
        forwarded: list[str] = []
        captured_scope_path: str | None = None

        def fake_ingest_main(args: list[str]) -> int:
            nonlocal captured_scope_path
            captured_scope_path = os.environ.get(config.DATABASE_SCOPE_PATH_ENV)
            forwarded.extend(args)
            return 0

        stale_env = {
            "PCI_COLLECTION": "tokio",
            "PCI_DATABASE_SCOPE_PATH": "/home/cvandesande/github/tokio",
            "PCI_MCP_DATABASE_URL": "postgresql://example.invalid/pci_tokio?sslmode=prefer",
            "PCI_MCP_DATABASE_USER": "pci_tokio_ro",
            "PCI_MCP_DATABASE_PASSWORD": "-".join(("tokio", "ro", "credential")),
        }
        with (
            patch.dict(os.environ, stale_env, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main([".", "--no-embed", "--dry-run"])

        self.assertEqual(status, 0)
        self.assertEqual(captured_scope_path, str(Path.cwd().resolve()))
        self.assertIn("--collection", forwarded)
        self.assertEqual(forwarded[forwarded.index("--collection") + 1], Path.cwd().name)

    def test_pci_index_collection_env_override_requires_opt_in(self) -> None:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(
                os.environ,
                {
                    "PCI_COLLECTION": "env-workspace",
                    "PCI_ALLOW_COLLECTION_OVERRIDE": "1",
                },
                clear=True,
            ),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main([".", "--dry-run"])
            self.assertEqual(os.environ["PCI_COLLECTION"], "env-workspace")

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
            self.assertEqual(os.environ["PCI_EMBED"], "0")

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

    def test_pci_index_exposes_and_forwards_init_db(self) -> None:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main(["--init-db", "."])

        self.assertEqual(status, 0)
        self.assertIn("--init-db", cli.index_parser().format_help())
        self.assertIn("--init-db-only", forwarded)
        self.assertIn("--root", forwarded)
        self.assertIn("--repos", forwarded)
        self.assertNotIn("--embed", forwarded)
        self.assertNotIn("--embedding-endpoint", forwarded)

    def test_pci_index_forwards_mcp_config_options(self) -> None:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main(["--init-db", "--mcp-config", "codex", "--mcp-server-name", "pci-demo", "."])

        self.assertEqual(status, 0)
        help_text = cli.index_parser().format_help()
        self.assertIn("--mcp-config", help_text)
        self.assertIn("vscode", help_text)
        self.assertIn("cline", help_text)
        self.assertIn("zed", help_text)
        self.assertIn("--mcp-server-name", cli.index_parser().format_help())
        self.assertIn("--mcp-config", forwarded)
        self.assertEqual(forwarded[forwarded.index("--mcp-config") + 1], "codex")
        self.assertIn("--mcp-server-name", forwarded)
        self.assertEqual(forwarded[forwarded.index("--mcp-server-name") + 1], "pci-demo")
        self.assertIn("--init-db-only", forwarded)

    def test_pci_index_init_db_and_reset_conflict(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main") as ingest_main,
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            _ = cli.index_main(["--init-db", "--reset", "."])

        self.assertEqual(raised.exception.code, 2)
        ingest_main.assert_not_called()

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

    def test_pci_index_reset_all_is_removed(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main") as ingest_main,
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            _ = cli.index_main(["--reset-all"])

        self.assertEqual(raised.exception.code, 2)
        ingest_main.assert_not_called()
        self.assertNotIn("--reset-all", cli.index_parser().format_help())

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
                            {"repo": "service-api", "id": 1},
                            {"repo": "ask-cmm", "id": 2},
                        ],
                    }),
                    "",
                )
            return 0, mcp_response({"results": [], "edges": [], "files": []}), ""

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
        self.assertEqual(probe_repos, ["service-api", "ask-cmm"])

        payload = cast("dict[str, object]", json.loads(stdout.getvalue()))
        self.assertEqual(payload["repos"], ["service-api", "ask-cmm"])

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


class WorktreeIngestArgsTests(unittest.TestCase):
    def test_worktree_spec_maps_to_main_repo_identity_and_scan_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main_repo = root / "main-repo"
            main_repo.mkdir()
            worktree = root / "worktrees" / "feature-wt"
            worktree.mkdir(parents=True)

            args = cli.worktree_ingest_args(f"{main_repo}={worktree}")

        self.assertEqual(
            args,
            [
                "--root",
                str(root.resolve()),
                "--repos",
                "main-repo",
                "--repo-scan-root",
                f"main-repo={worktree.resolve()}",
            ],
        )


class PciIndexShowParserFailuresFlagTests(unittest.TestCase):
    """Forwarding contract for `pci-index --show-parser-failures`."""

    @staticmethod
    def _run_with_capture(argv: list[str]) -> tuple[int, list[str]]:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main(argv)
        return status, forwarded

    def test_flag_is_forwarded_when_set(self) -> None:
        status, forwarded = self._run_with_capture(["--show-parser-failures", "--dry-run", "."])
        self.assertEqual(status, 0)
        self.assertIn("--show-parser-failures", cli.index_parser().format_help())
        self.assertIn("--show-parser-failures", forwarded)

    def test_flag_is_omitted_by_default(self) -> None:
        status, forwarded = self._run_with_capture(["--dry-run", "."])
        self.assertEqual(status, 0)
        self.assertNotIn("--show-parser-failures", forwarded)


class PciIndexPruneSnapshotsFlagTests(unittest.TestCase):
    """Forwarding contract for `pci-index --prune-snapshots` (default on)."""

    @staticmethod
    def _run_with_capture(argv: list[str]) -> tuple[int, list[str]]:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main(argv)
        return status, forwarded

    def test_pruning_is_forwarded_by_default(self) -> None:
        status, forwarded = self._run_with_capture(["--dry-run", "."])
        self.assertEqual(status, 0)
        self.assertIn("--prune-snapshots", forwarded)
        self.assertNotIn("--no-prune-snapshots", forwarded)
        self.assertEqual(forwarded[forwarded.index("--prune-keep") + 1], "5")

    def test_no_prune_snapshots_opts_out(self) -> None:
        status, forwarded = self._run_with_capture(["--no-prune-snapshots", "--dry-run", "."])
        self.assertEqual(status, 0)
        self.assertIn("--no-prune-snapshots", forwarded)
        self.assertNotIn("--prune-snapshots", forwarded)


class IndexUserConfigTests(unittest.TestCase):
    def test_pci_index_loads_pci_doctor_user_config(self) -> None:
        captured_env: dict[str, str | None] = {}
        credential = " ".join(("secret", "value"))

        def fake_ingest_main(_args: list[str]) -> int:
            for name in config.PCI_INDEX_USER_CONFIG_ENV_NAMES:
                captured_env[name] = os.environ.get(name)
            return 0

        with tempfile.TemporaryDirectory() as directory:
            path = config.write_pci_index_user_config(
                database_url="postgresql://db.example.invalid:5432?sslmode=prefer",
                database_admin_user="pci_index_admin",
                database_admin_password=credential,
                env={"XDG_CONFIG_HOME": directory},
            )
            stderr = io.StringIO()
            with (
                patch.dict(os.environ, {"XDG_CONFIG_HOME": directory}, clear=True),
                patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
                patch("sys.stderr", stderr),
            ):
                status = cli.index_main([".", "--dry-run"])

        self.assertEqual(status, 0)
        self.assertEqual(
            captured_env["PCI_DATABASE_URL"],
            "postgresql://db.example.invalid:5432?sslmode=prefer",
        )
        self.assertEqual(captured_env["PCI_DATABASE_ADMIN_USER"], "pci_index_admin")
        self.assertEqual(captured_env["PCI_DATABASE_ADMIN_PASSWORD"], credential)
        self.assertIn(f"pci-index: loaded config from {path}", stderr.getvalue())

    def test_pci_index_mcp_config_loads_user_config_without_notice(self) -> None:
        captured_env: dict[str, str | None] = {}
        credential = " ".join(("secret", "value"))

        def fake_ingest_main(_args: list[str]) -> int:
            for name in config.PCI_INDEX_USER_CONFIG_ENV_NAMES:
                captured_env[name] = os.environ.get(name)
            return 0

        with tempfile.TemporaryDirectory() as directory:
            _ = config.write_pci_index_user_config(
                database_url="postgresql://db.example.invalid:5432?sslmode=prefer",
                database_admin_user="pci_index_admin",
                database_admin_password=credential,
                env={"XDG_CONFIG_HOME": directory},
            )
            stderr = io.StringIO()
            with (
                patch.dict(os.environ, {"XDG_CONFIG_HOME": directory}, clear=True),
                patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
                patch("sys.stderr", stderr),
            ):
                status = cli.index_main(["--init-db", "--mcp-config", "codex", "."])

        self.assertEqual(status, 0)
        self.assertEqual(captured_env["PCI_DATABASE_ADMIN_USER"], "pci_index_admin")
        self.assertNotIn("loaded config from", stderr.getvalue())


class IndexStartupTests(unittest.TestCase):
    def test_pci_index_startup_header_shown_before_ingest(self) -> None:
        call_order: list[str] = []

        def fake_startup(_parsed: object, *, embed: object, endpoint: object, model: object) -> bool:
            del embed, endpoint, model
            call_order.append("startup")
            return True

        def fake_ingest_main(_args: list[str]) -> int:
            call_order.append("ingest")
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
            patch("project_code_intelligence.cli._resolve_index_embedding", return_value=(None, None)),
            patch("project_code_intelligence.cli.print_index_startup", side_effect=fake_startup),
        ):
            status = cli.index_main([".", "--no-embed", "--dry-run"])

        self.assertEqual(status, 0)
        self.assertEqual(call_order, ["startup", "ingest"])

    def test_pci_index_exits_early_when_embedding_preflight_fails(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main") as ingest_main,
            patch(
                "project_code_intelligence.cli._resolve_index_embedding",
                return_value=("http://127.0.0.1:18081/v1/embeddings", "test-model"),
            ),
            patch(
                "project_code_intelligence.cli.preflight_embedding_endpoint",
                side_effect=EmbeddingEndpointUnavailableError("endpoint down"),
            ),
            patch("sys.stderr", io.StringIO()),
        ):
            status = cli.index_main(["."])

        self.assertEqual(status, 1)
        ingest_main.assert_not_called()

    def test_pci_index_reset_skips_startup_header(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", return_value=0),
            patch("project_code_intelligence.cli.print_index_startup") as mock_startup,
        ):
            status = cli.index_main(["--reset", "--i-know-this-deletes-code-intel-db", "."])

        self.assertEqual(status, 0)
        mock_startup.assert_not_called()

    def test_pci_index_json_mode_skips_startup_header(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", return_value=0),
            patch("project_code_intelligence.cli._resolve_index_embedding", return_value=(None, None)),
            patch("project_code_intelligence.cli.print_index_startup") as mock_startup,
        ):
            status = cli.index_main([".", "--json", "--dry-run"])

        self.assertEqual(status, 0)
        mock_startup.assert_not_called()

    def test_pci_index_dry_run_skips_preflight(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", return_value=0),
            patch(
                "project_code_intelligence.cli._resolve_index_embedding",
                return_value=("http://127.0.0.1:18081/v1/embeddings", "test-model"),
            ),
            patch("project_code_intelligence.cli.preflight_embedding_endpoint") as mock_preflight,
            patch("sys.stderr", io.StringIO()),
        ):
            status = cli.index_main([".", "--dry-run"])

        self.assertEqual(status, 0)
        mock_preflight.assert_not_called()

    def test_pci_index_no_embed_skips_preflight(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", return_value=0),
            patch("project_code_intelligence.cli.preflight_embedding_endpoint") as mock_preflight,
            patch("sys.stderr", io.StringIO()),
        ):
            status = cli.index_main([".", "--no-embed", "--dry-run"])

        self.assertEqual(status, 0)
        mock_preflight.assert_not_called()

    def test_pci_index_startup_header_passes_embedding_info_from_resolver(self) -> None:
        captured: dict[str, object] = {}

        def fake_startup(_parsed: object, *, embed: object, endpoint: object, model: object) -> bool:
            captured["embed"] = embed
            captured["endpoint"] = endpoint
            captured["model"] = model
            return True

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", return_value=0),
            patch(
                "project_code_intelligence.cli._resolve_index_embedding",
                return_value=("http://127.0.0.1:18081/v1/embeddings", "resolved-model"),
            ),
            patch("project_code_intelligence.cli.print_index_startup", side_effect=fake_startup),
        ):
            status = cli.index_main([".", "--dry-run"])

        self.assertEqual(status, 0)
        self.assertEqual(captured["endpoint"], "http://127.0.0.1:18081/v1/embeddings")
        self.assertEqual(captured["model"], "resolved-model")
        self.assertTrue(captured["embed"])

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
