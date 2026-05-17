from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from unittest.mock import patch

from typing_extensions import override

from project_code_intelligence import db, profile_context
from project_code_intelligence.code_profiles import load_profile
from project_code_intelligence.common import default_database_name
from project_code_intelligence.config import (
    DEFAULT_EMBEDDING_ENDPOINT_MODEL,
    DEFAULT_LEMONADE_EMBEDDING_ENDPOINT,
    DatabaseSettings,
    IngestSettings,
    default_embedding_endpoint,
    default_embedding_endpoint_model,
    env_bool,
)
from project_code_intelligence.embedding.fastembed_server import embedding_response, normalize_input
from project_code_intelligence.embeddings import validate_embedding_endpoint
from project_code_intelligence.exceptions import ConfigError, ProfileLoadError
from project_code_intelligence.ingest_code_intel import (
    CliArgs,
    IngestPlan,
    build_ingest_plan,
    claude_mcp_config_block,
    cline_mcp_config_block,
    codex_mcp_config_block,
    confirm_reset_code_intel,
    database_bootstrap_report,
    default_mcp_server_name,
    discover_plan_sarif_files,
    mcp_config_block,
    mcp_config_context,
    mcp_project_config_path,
    mcp_ro_export_block,
    opencode_mcp_config_block,
    replace_repos_for_full_ingests,
    resolve_scan_workers,
    run_ingest_plan,
    run_reset_only,
    validate_args,
    vscode_mcp_config_block,
    warning_for_sarif_mtime,
    zed_mcp_config_block,
)
from project_code_intelligence.language_profiles.go import go_file_metadata
from project_code_intelligence.mcp.filters import (
    code_intel_clauses,
    scoped_snapshot_clauses,
    snapshot_scope_response,
    static_finding_clauses,
)
from project_code_intelligence.models import IntelFile, JsonValue, RepoIngest, Snapshot
from project_code_intelligence.parsers import (
    go_records,
    javascript_records,
    make_records,
    python_records,
    rust_records,
    security_records,
)
from project_code_intelligence.records import line_for_offset_with_index, line_offsets, line_window_records
from project_code_intelligence.runtime import RuntimeMetrics
from project_code_intelligence.sarif import (
    SarifIngestContext,
    SarifPathContext,
    discover_sarif_files,
    ingest_sarif,
    resolve_sarif_source_path,
    source_path_from_sarif_uri,
)
from project_code_intelligence.server import query_embedding


class FakeFastEmbedModel:
    def embed(self, documents: list[str]) -> list[list[float]]:
        return [[float(index), float(len(text))] for index, text in enumerate(documents)]


class TtyStringIO(io.StringIO):
    @override
    def isatty(self) -> bool:
        return True


def fixture_file(path: str, language: str) -> IntelFile:
    return IntelFile(
        collection="test",
        repo=".",
        repo_role="source",
        branch="main",
        commit_sha="commit",
        tree_sha="tree",
        source_path=path,
        repo_rel_path=path,
        abs_path=Path.cwd() / "__fixture__" / path,
        git_blob_sha=None,
        file_sha256="sha",
        size_bytes=0,
        language=language,
        file_role="source",
        content_class="source",
        is_generated=False,
        is_vendor=False,
        is_test=False,
        is_source=True,
        is_build=False,
        is_config=False,
        is_doc=False,
        skipped_reason=None,
        metadata={},
    )


def require_list(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise TypeError(f"expected list metadata value, got {type(value).__name__}")
    return value


def cli_args(**overrides: object) -> CliArgs:
    values: dict[str, object] = {
        "root": Path(),
        "collection": None,
        "profile": "generic",
        "repos": None,
        "max_file_bytes": 0,
        "scan_workers": 1,
        "chunk_chars": 2400,
        "overlap_lines": 0,
        "limit_files": None,
        "progress_every": 0,
        "dry_run": False,
        "reset_code_intel": False,
        "i_know_this_deletes_code_intel_db": False,
        "reset_only": False,
        "init_db_only": False,
        "sarif": [],
        "no_profile_sarif": False,
        "sarif_max_bytes": 1024 * 1024,
        "embed_only": False,
        "mode": "incremental",
        "full": False,
        "no_replace": False,
        "embed": False,
        "embed_record_types": "code_chunk",
        "embedding_batch_size": 1,
        "embedding_max_chars": 3000,
        "embedding_endpoint": None,
        "embedding_endpoint_model": "local",
        "llama_embed": False,
        "no_preembed": False,
        "prune_snapshots": False,
        "prune_keep": 5,
        "mcp_config": None,
        "mcp_server_name": None,
    }
    values.update(overrides)
    return CliArgs(**values)  # type: ignore[arg-type]


def snapshot_for_repo(repo: str) -> Snapshot:
    return Snapshot(
        collection="test",
        repo=repo,
        repo_role="source",
        branch="main",
        commit_sha="commit",
        tree_sha="tree",
        dirty=False,
        metadata={},
    )


class DatabaseSettingsTests(unittest.TestCase):
    def test_default_to_inferred_local_project_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = DatabaseSettings.from_env({"PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": directory})
            expected_dbname = default_database_name(Path(directory))

        self.assertEqual(settings.missing_connection_names(), [])
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, "5433")
        self.assertEqual(settings.dbname, expected_dbname)
        self.assertEqual(settings.user, "codeintel")
        self.assertEqual(settings.password, "codeintel")
        self.assertTrue(settings.database_inferred)
        self.assertIn(f"PGVECTOR_DB={expected_dbname} (inferred)", settings.connection_hint())
        self.assertIn("PGVECTOR_PASS=<set>", settings.connection_hint())

    def test_database_url_without_database_uses_inferred_dbname(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = DatabaseSettings.from_env({
                "PROJECT_CODE_INTELLIGENCE_DATABASE_URL": "postgresql://example.invalid:5432?sslmode=prefer",
                "PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": directory,
            })
            expected_dbname = default_database_name(Path(directory))

        self.assertEqual(settings.dbname, expected_dbname)
        self.assertTrue(settings.database_inferred)
        self.assertEqual(settings.connection_hint(), "PROJECT_CODE_INTELLIGENCE_DATABASE_URL=<hidden>")
        self.assertEqual(
            settings.display_target(), f"postgresql://example.invalid:5432/{expected_dbname}?sslmode=prefer"
        )

    def test_database_url_with_database_disables_inference(self) -> None:
        settings = DatabaseSettings.from_env({
            "PROJECT_CODE_INTELLIGENCE_DATABASE_URL": "postgresql://example.invalid/db",
            "PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": "ignored-scope",
        })

        self.assertEqual(settings.dbname, "db")
        self.assertFalse(settings.database_inferred)
        self.assertEqual(settings.display_target(), "postgresql://example.invalid/db")

    def test_explicit_pgvector_db_disables_inference_when_url_has_no_database(self) -> None:
        settings = DatabaseSettings.from_env({
            "PROJECT_CODE_INTELLIGENCE_DATABASE_URL": "postgresql://example.invalid",
            "PGVECTOR_DB": "explicit_db",
            "PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": "ignored-scope",
        })

        self.assertEqual(settings.dbname, "explicit_db")
        self.assertFalse(settings.database_inferred)
        self.assertEqual(settings.display_target(), "postgresql://example.invalid/explicit_db")

    def test_report_missing_connection_parts(self) -> None:
        settings = DatabaseSettings(dbname="codeintel", user="reader", password=None)

        self.assertEqual(settings.missing_connection_names(), ["PGVECTOR_PASS"])

    def test_accept_database_url_without_individual_parts(self) -> None:
        settings = DatabaseSettings.from_env({
            "PROJECT_CODE_INTELLIGENCE_DATABASE_URL": "postgresql://example.invalid/db"
        })

        self.assertEqual(settings.missing_connection_names(), [])
        self.assertEqual(settings.connection_hint(), "PROJECT_CODE_INTELLIGENCE_DATABASE_URL=<hidden>")
        self.assertEqual(settings.display_target(), "postgresql://example.invalid/db")

    def test_accept_database_url_with_separate_credentials(self) -> None:
        credential = "test-credential"
        settings = DatabaseSettings.from_env({
            "PROJECT_CODE_INTELLIGENCE_DATABASE_URL": "postgresql://example.invalid/db",
            "PROJECT_CODE_INTELLIGENCE_DATABASE_USER": "app",
            "PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD": credential,
        })

        self.assertEqual(settings.dsn_user, "app")
        self.assertEqual(settings.dsn_password, credential)
        self.assertEqual(
            settings.connection_hint(),
            "PROJECT_CODE_INTELLIGENCE_DATABASE_URL=<hidden> "
            "PROJECT_CODE_INTELLIGENCE_DATABASE_USER=<set> "
            "PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD=<set>",
        )

    def test_mcp_database_url_reports_mcp_credential_sources(self) -> None:
        settings = DatabaseSettings.from_env(
            {
                "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL": "postgresql://example.invalid/db",
                "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER": "project_ro",
                "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD": "-".join(("ro", "credential")),
            },
            role="mcp",
        )

        self.assertEqual(
            settings.connection_hint(),
            "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL=<hidden> "
            "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER=<set> "
            "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD=<set>",
        )

    def test_accept_legacy_pgvector_dsn_without_individual_parts(self) -> None:
        settings = DatabaseSettings.from_env({"PGVECTOR_DSN": "postgresql://example.invalid/db"})

        self.assertEqual(settings.missing_connection_names(), [])
        self.assertEqual(settings.connection_hint(), "PGVECTOR_DSN=<hidden>")
        self.assertEqual(settings.display_target(), "postgresql://example.invalid/db")

    def test_database_url_takes_precedence_over_legacy_dsn(self) -> None:
        settings = DatabaseSettings.from_env({
            "PROJECT_CODE_INTELLIGENCE_DATABASE_URL": "postgresql://primary.example.invalid/db",
            "PGVECTOR_DSN": "postgresql://legacy.example.invalid/db",
        })

        self.assertEqual(settings.connection_hint(), "PROJECT_CODE_INTELLIGENCE_DATABASE_URL=<hidden>")
        self.assertEqual(settings.display_target(), "postgresql://primary.example.invalid/db")

    def test_display_target_hides_passwords(self) -> None:
        settings = DatabaseSettings.from_env({
            "PROJECT_CODE_INTELLIGENCE_DATABASE_URL": (
                "postgresql://user:secret@example.invalid:5432/db?sslmode=require&password=secret"
            )
        })

        self.assertEqual(
            settings.display_target(), "postgresql://user@example.invalid:5432/db?sslmode=require&password=<hidden>"
        )


class ResetConfirmationTests(unittest.TestCase):
    def test_reset_confirmation_prints_target_and_accepts_yes(self) -> None:
        credential = "secret"
        stderr = io.StringIO()

        with patch("sys.stdin", TtyStringIO("yes\n")), patch("sys.stderr", stderr):
            confirm_reset_code_intel(
                cli_args(reset_code_intel=True),
                DatabaseSettings(host="db", port="5432", dbname="codeintel", user="app", password=credential),
                "default",
                ["acme-repo"],
            )

        output = stderr.getvalue()
        self.assertIn("Postgres admin connection: postgresql://app@db:5432/postgres sslmode=prefer", output)
        self.assertIn("About to drop PostgreSQL database: codeintel", output)
        self.assertIn("acme-repo", output)
        self.assertIn("schema in that DB", output)
        self.assertIn("Other PCI-managed project databases are untouched", output)
        self.assertNotIn("secret", output)

    def test_reset_confirmation_shows_admin_connection_not_runtime_role(self) -> None:
        stderr = io.StringIO()
        runtime_credential = "-".join(("runtime", "credential"))
        admin_credential = "-".join(("admin", "credential"))
        settings = DatabaseSettings(
            dsn="postgresql://db.example.invalid:5432/pci_zod_cfa53486?sslmode=prefer",
            dsn_user="pci_project_code_intelligence_38fc61c9_rw",
            dsn_password=runtime_credential,
            dbname="pci_zod_cfa53486",
            admin_user="pci_index_admin",
            admin_password=admin_credential,
            database_inferred=True,
        )

        with patch("sys.stdin", TtyStringIO("yes\n")), patch("sys.stderr", stderr):
            confirm_reset_code_intel(cli_args(reset_code_intel=True), settings, "zod", ["zod"])

        output = stderr.getvalue()
        self.assertIn("About to drop PostgreSQL database: pci_zod_cfa53486", output)
        self.assertIn(
            "Postgres admin connection: postgresql://pci_index_admin@db.example.invalid:5432/postgres?sslmode=prefer",
            output,
        )
        self.assertNotIn("pci_project_code_intelligence_38fc61c9_rw", output)
        self.assertNotIn(runtime_credential, output)
        self.assertNotIn(admin_credential, output)

    def test_reset_confirmation_does_not_offer_global_scope(self) -> None:
        stderr = io.StringIO()

        with patch("sys.stdin", TtyStringIO("yes\n")), patch("sys.stderr", stderr):
            confirm_reset_code_intel(
                cli_args(reset_code_intel=True),
                DatabaseSettings(),
                "default",
                ["acme-repo"],
            )

        output = stderr.getvalue()
        self.assertNotIn("Collections/repos: all", output)
        self.assertIn("Other PCI-managed project databases are untouched", output)

    def test_reset_confirmation_requires_flag_in_noninteractive_mode(self) -> None:
        with (
            patch("sys.stdin", io.StringIO("yes\n")),
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(ValueError),
        ):
            confirm_reset_code_intel(cli_args(reset_code_intel=True), DatabaseSettings(), "default", ["acme-repo"])

    def test_reset_refuses_explicit_database_before_prompt(self) -> None:
        with (
            patch(
                "project_code_intelligence.ingest_code_intel.config.DatabaseSettings.from_env",
                return_value=DatabaseSettings(dbname="shared", database_inferred=False),
            ),
            patch("project_code_intelligence.ingest_code_intel.confirm_reset_code_intel") as confirm,
            self.assertRaises(db.DatabaseConnectionError),
        ):
            _ = run_reset_only(cli_args(reset_code_intel=True, i_know_this_deletes_code_intel_db=True))

        confirm.assert_not_called()


class DatabaseBootstrapReportTests(unittest.TestCase):
    def test_database_bootstrap_report_exposes_roles_without_credentials(self) -> None:
        rw_value = "rw-fixture"
        ro_value = "ro-fixture"
        bootstrap = db.DatabaseBootstrapResult(
            dbname="pci_demo",
            database_created=True,
            rw_role=db.DatabaseRole(
                name="pci_demo_rw",
                password=rw_value,
                created=True,
                database_url=f"postgresql://pci_demo_rw:{rw_value}@db.example.invalid/pci_demo",
            ),
            ro_role=db.DatabaseRole(
                name="pci_demo_ro",
                password=ro_value,
                created=True,
                database_url=f"postgresql://pci_demo_ro:{ro_value}@db.example.invalid/pci_demo",
            ),
        )

        report = database_bootstrap_report(bootstrap)

        self.assertEqual(report["rw_role"], "pci_demo_rw")
        self.assertEqual(report["ro_role"], "pci_demo_ro")
        self.assertNotIn("rw_database_url", report)
        self.assertNotIn("ro_database_url", report)

    def test_mcp_ro_export_block_prints_split_read_only_credentials(self) -> None:
        ro_value = " ".join(("ro", "fixture"))
        plan = build_ingest_plan(
            cli_args(collection="demo-workspace", root=Path.cwd(), repos=".", mcp_server_name="pci-demo")
        )
        bootstrap = db.DatabaseBootstrapResult(
            dbname="pci_demo",
            ro_role=db.DatabaseRole(
                name="pci_demo_ro",
                password=ro_value,
                created=True,
                database_url="postgresql://pci_demo_ro:ro%20fixture@db.example.invalid:5432/pci_demo?sslmode=prefer",
            ),
        )

        with patch.dict(
            os.environ,
            {"PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": "/work/demo-workspace"},
            clear=False,
        ):
            output = mcp_ro_export_block(mcp_config_context(plan, bootstrap, command="/usr/bin/pci-mcp"))

        if output is None:
            raise AssertionError("expected MCP export block")
        self.assertIn("Export for pci-mcp (RO)", output)
        self.assertIn(
            "export PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL='postgresql://db.example.invalid:5432/pci_demo?sslmode=prefer'",
            output,
        )
        self.assertIn("export PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER=pci_demo_ro", output)
        self.assertIn("export PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD='ro fixture'", output)
        self.assertIn("export PROJECT_CODE_INTELLIGENCE_COLLECTION=demo-workspace", output)
        self.assertIn("export PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH=/work/demo-workspace", output)
        self.assertNotIn("pci_demo_ro:ro%20fixture@", output)

    def test_default_mcp_server_name_is_generic(self) -> None:
        self.assertEqual(default_mcp_server_name("demo-workspace"), "project-code-intelligence")
        self.assertEqual(default_mcp_server_name("hexyl"), "project-code-intelligence")

    def test_codex_mcp_config_block_references_inherited_environment(self) -> None:
        ro_value = " ".join(("ro", "fixture"))
        plan = build_ingest_plan(
            cli_args(collection="demo-workspace", root=Path.cwd(), repos=".", mcp_server_name="pci-demo")
        )
        bootstrap = db.DatabaseBootstrapResult(
            dbname="pci_demo",
            ro_role=db.DatabaseRole(
                name="pci_demo_ro",
                password=ro_value,
                created=True,
                database_url="postgresql://pci_demo_ro:ro%20fixture@db.example.invalid:5432/pci_demo?sslmode=prefer",
            ),
        )

        with patch.dict(
            os.environ,
            {"PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": "/work/demo-workspace"},
            clear=False,
        ):
            context = mcp_config_context(plan, bootstrap, command="/usr/bin/pci-mcp")

        if context is None:
            raise AssertionError("expected MCP config context")
        output = codex_mcp_config_block(context)

        self.assertIn("[mcp_servers.pci-demo]", output)
        self.assertIn('command = "/usr/bin/pci-mcp"', output)
        self.assertIn('cwd = "/work/demo-workspace"', output)
        self.assertIn("env_vars = [", output)
        self.assertIn('"PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL"', output)
        self.assertIn('"PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER"', output)
        self.assertIn('"PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD"', output)
        self.assertNotIn('"PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH"', output)
        self.assertNotIn('"PROJECT_CODE_INTELLIGENCE_COLLECTION"', output)
        self.assertNotIn("postgresql://db.example.invalid", output)
        self.assertNotIn("pci_demo_ro", output)
        self.assertNotIn("ro fixture", output)
        self.assertNotIn("pci_demo_ro:ro%20fixture@", output)

    def test_json_mcp_config_blocks_reference_environment(self) -> None:
        ro_value = "-".join(("ro", "fixture"))
        plan = build_ingest_plan(cli_args(collection="demo-workspace", root=Path.cwd(), repos="."))
        bootstrap = db.DatabaseBootstrapResult(
            dbname="pci_demo",
            ro_role=db.DatabaseRole(
                name="pci_demo_ro",
                password=ro_value,
                created=True,
                database_url="postgresql://pci_demo_ro:ro-fixture@db.example.invalid:5432/pci_demo?sslmode=prefer",
            ),
        )
        context = mcp_config_context(plan, bootstrap, command="/usr/bin/pci-mcp")

        if context is None:
            raise AssertionError("expected MCP config context")

        claude_output = claude_mcp_config_block(context)
        opencode_output = opencode_mcp_config_block(context)
        vscode_output = vscode_mcp_config_block(context)
        zed_output = zed_mcp_config_block(context)
        credential_key = "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_" + "PASSWORD"

        self.assertIn(f'"{credential_key}": "${{{credential_key}}}"', claude_output)
        self.assertIn(f'"{credential_key}": "{{env:{credential_key}}}"', opencode_output)
        self.assertIn(f'"{credential_key}": "${{env:{credential_key}}}"', vscode_output)
        self.assertIn('"type": "stdio"', vscode_output)
        self.assertIn('"servers": {', vscode_output)
        self.assertIn(
            '"PROJECT_CODE_INTELLIGENCE_COLLECTION": "${env:PROJECT_CODE_INTELLIGENCE_COLLECTION}"',
            vscode_output,
        )
        self.assertIn('"context_servers": {', zed_output)
        self.assertIn('"project-code-intelligence": {', zed_output)
        self.assertIn('"command": "/usr/bin/pci-mcp"', zed_output)
        self.assertIn('"PROJECT_CODE_INTELLIGENCE_COLLECTION": "demo-workspace"', zed_output)
        self.assertIn('"PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH":', zed_output)
        self.assertIn('"PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD": "ro-fixture"', zed_output)
        self.assertIn(
            '"PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL": "postgresql://db.example.invalid:5432/pci_demo?sslmode=prefer"',
            zed_output,
        )
        self.assertNotIn("PROJECT_CODE_INTELLIGENCE_COLLECTION", claude_output)
        self.assertNotIn("PROJECT_CODE_INTELLIGENCE_COLLECTION", opencode_output)
        self.assertNotIn("PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH", claude_output)
        self.assertNotIn("PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH", opencode_output)
        self.assertNotIn(ro_value, claude_output)
        self.assertNotIn(ro_value, opencode_output)
        self.assertNotIn(ro_value, vscode_output)
        self.assertNotIn("postgresql://db.example.invalid", claude_output)
        self.assertNotIn("postgresql://db.example.invalid", opencode_output)
        self.assertNotIn("postgresql://db.example.invalid", vscode_output)

    def test_cline_mcp_config_block_embeds_user_scoped_environment(self) -> None:
        ro_value = "-".join(("ro", "fixture"))
        plan = build_ingest_plan(cli_args(collection="demo-workspace", root=Path.cwd(), repos="."))
        bootstrap = db.DatabaseBootstrapResult(
            dbname="pci_demo",
            ro_role=db.DatabaseRole(
                name="pci_demo_ro",
                password=ro_value,
                created=True,
                database_url="postgresql://pci_demo_ro:ro-fixture@db.example.invalid:5432/pci_demo?sslmode=prefer",
            ),
        )
        context = mcp_config_context(plan, bootstrap, command="/usr/bin/pci-mcp")

        if context is None:
            raise AssertionError("expected MCP config context")

        output = cline_mcp_config_block(context)

        self.assertIn('"mcpServers": {', output)
        self.assertIn('"command": "/usr/bin/pci-mcp"', output)
        self.assertIn('"autoApprove": []', output)
        self.assertIn('"disabled": false', output)
        self.assertIn('"PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD": "ro-fixture"', output)
        self.assertIn('"PROJECT_CODE_INTELLIGENCE_COLLECTION": "demo-workspace"', output)
        self.assertIn(
            '"PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL": "postgresql://db.example.invalid:5432/pci_demo?sslmode=prefer"',
            output,
        )

    def test_mcp_config_block_wraps_client_config_with_project_scoped_guidance(self) -> None:
        ro_value = " ".join(("ro", "fixture"))
        plan = build_ingest_plan(cli_args(collection="demo-workspace", root=Path.cwd(), repos="."))
        bootstrap = db.DatabaseBootstrapResult(
            dbname="pci_demo",
            ro_role=db.DatabaseRole(
                name="pci_demo_ro",
                password=ro_value,
                created=True,
                database_url="postgresql://pci_demo_ro:ro%20fixture@db.example.invalid:5432/pci_demo?sslmode=prefer",
            ),
        )

        with patch.dict(
            os.environ,
            {"PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": "/work/demo-workspace"},
            clear=False,
        ):
            context = mcp_config_context(plan, bootstrap, command="/usr/bin/pci-mcp")

        if context is None:
            raise AssertionError("expected MCP config context")

        self.assertEqual(mcp_project_config_path(context, "codex"), "/work/demo-workspace/.codex/config.toml")
        self.assertEqual(mcp_project_config_path(context, "claude"), "/work/demo-workspace/.mcp.json")
        self.assertEqual(mcp_project_config_path(context, "opencode"), "/work/demo-workspace/opencode.json")
        self.assertEqual(mcp_project_config_path(context, "vscode"), "/work/demo-workspace/.vscode/mcp.json")
        self.assertEqual(mcp_project_config_path(context, "copilot"), "/work/demo-workspace/.vscode/mcp.json")
        self.assertEqual(mcp_project_config_path(context, "cline"), "Cline MCP settings JSON")

        output = mcp_config_block(context, "codex")

        if output is None:
            raise AssertionError("expected MCP config block")
        self.assertIn("Codex project-scoped MCP config", output)
        self.assertIn("Write this snippet to: /work/demo-workspace/.codex/config.toml", output)
        self.assertIn("references environment variables for credentials", output)
        self.assertIn("Do not paste this into a global MCP config", output)
        self.assertIn("[mcp_servers.project-code-intelligence]", output)
        self.assertIn("Required environment variables for pci-mcp (RO)", output)
        self.assertIn(
            "export PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL='postgresql://db.example.invalid:5432/pci_demo?sslmode=prefer'",
            output,
        )
        self.assertIn("export PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD='ro fixture'", output)
        self.assertNotIn("export PROJECT_CODE_INTELLIGENCE_COLLECTION=", output)
        self.assertNotIn("export PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH=", output)

        project_config = output.split("Required environment variables for pci-mcp (RO)", maxsplit=1)[0]
        self.assertNotIn("ro fixture", project_config)
        self.assertNotIn("postgresql://db.example.invalid", project_config)
        self.assertNotIn("PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH", project_config)

        vscode_output = mcp_config_block(context, "vscode")

        if vscode_output is None:
            raise AssertionError("expected VS Code MCP config block")
        self.assertIn("VS Code Copilot project-scoped MCP config", vscode_output)
        self.assertIn("Write this snippet to: /work/demo-workspace/.vscode/mcp.json", vscode_output)
        self.assertIn('"servers": {', vscode_output)
        self.assertIn("export PROJECT_CODE_INTELLIGENCE_COLLECTION=demo-workspace", vscode_output)
        self.assertIn("export PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH=/work/demo-workspace", vscode_output)

        cline_output = mcp_config_block(context, "cline")

        if cline_output is None:
            raise AssertionError("expected Cline MCP config block")
        self.assertIn("Cline VS Code MCP config", cline_output)
        self.assertIn("Add or merge this snippet under mcpServers", cline_output)
        self.assertIn("Cline's VS Code MCP settings are user-scoped", cline_output)
        self.assertIn("This JSON contains read-only database credentials", cline_output)
        self.assertIn('"PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD": "ro fixture"', cline_output)
        self.assertNotIn("Required environment variables for pci-mcp (RO)", cline_output)

    def test_zed_mcp_config_block_embeds_project_settings_environment(self) -> None:
        ro_value = " ".join(("ro", "fixture"))
        plan = build_ingest_plan(cli_args(collection="demo-workspace", root=Path.cwd(), repos="."))
        bootstrap = db.DatabaseBootstrapResult(
            dbname="pci_demo",
            ro_role=db.DatabaseRole(
                name="pci_demo_ro",
                password=ro_value,
                created=True,
                database_url="postgresql://pci_demo_ro:ro%20fixture@db.example.invalid:5432/pci_demo?sslmode=prefer",
            ),
        )

        with patch.dict(
            os.environ,
            {"PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": "/work/demo-workspace"},
            clear=False,
        ):
            context = mcp_config_context(plan, bootstrap, command="/usr/bin/pci-mcp")

        if context is None:
            raise AssertionError("expected MCP config context")

        self.assertEqual(mcp_project_config_path(context, "zed"), "/work/demo-workspace/.zed/settings.json")
        zed_output = mcp_config_block(context, "zed")

        if zed_output is None:
            raise AssertionError("expected Zed MCP config block")
        self.assertIn("Zed project-scoped MCP config", zed_output)
        self.assertIn("Write or merge this snippet into: /work/demo-workspace/.zed/settings.json", zed_output)
        self.assertIn('{\n  "context_servers": {\n    "project-code-intelligence": {', zed_output)
        self.assertIn('"command": "/usr/bin/pci-mcp"', zed_output)
        self.assertIn('"PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL"', zed_output)
        self.assertIn('"PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER": "pci_demo_ro"', zed_output)
        self.assertIn('"PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD": "ro fixture"', zed_output)
        self.assertIn(
            '"PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL": "postgresql://db.example.invalid:5432/pci_demo?sslmode=prefer"',
            zed_output,
        )
        self.assertIn('"PROJECT_CODE_INTELLIGENCE_COLLECTION": "demo-workspace"', zed_output)
        self.assertIn('"PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": "/work/demo-workspace"', zed_output)
        self.assertIn("contains read-only database credentials", zed_output)
        self.assertIn("Trust the worktree in Zed", zed_output)
        self.assertIn("do not commit it", zed_output)
        self.assertNotIn("Add MCP Server", zed_output)
        self.assertNotIn("Required environment variables for pci-mcp (RO)", zed_output)
        self.assertNotIn("export PROJECT_CODE_INTELLIGENCE_COLLECTION=", zed_output)

    def test_codex_mcp_config_block_quotes_non_bare_toml_server_names(self) -> None:
        ro_value = "-".join(("ro", "fixture"))
        plan = build_ingest_plan(
            cli_args(collection="demo-workspace", root=Path.cwd(), repos=".", mcp_server_name="PCI Demo")
        )
        bootstrap = db.DatabaseBootstrapResult(
            dbname="pci_demo",
            ro_role=db.DatabaseRole(
                name="pci_demo_ro",
                password=ro_value,
                created=True,
                database_url="postgresql://pci_demo_ro:ro-fixture@db.example.invalid:5432/pci_demo?sslmode=prefer",
            ),
        )

        context = mcp_config_context(plan, bootstrap, command="/usr/bin/pci-mcp")

        if context is None:
            raise AssertionError("expected MCP config context")
        self.assertIn('[mcp_servers."PCI Demo"]', codex_mcp_config_block(context))

    def test_mcp_ro_export_block_skips_when_password_is_unavailable(self) -> None:
        plan = build_ingest_plan(cli_args(collection="demo-workspace", root=Path.cwd(), repos="."))
        bootstrap = db.DatabaseBootstrapResult(
            dbname="pci_demo",
            ro_role=db.DatabaseRole(
                name="pci_demo_ro",
                password=None,
                created=False,
                database_url="postgresql://pci_demo_ro@db.example.invalid:5432/pci_demo?sslmode=prefer",
            ),
        )

        self.assertIsNone(mcp_config_context(plan, bootstrap, command="/usr/bin/pci-mcp"))

    def test_init_db_only_bootstraps_without_scanning(self) -> None:
        bootstrap = db.DatabaseBootstrapResult(dbname="pci_demo", database_created=True)
        summaries: list[dict[str, object]] = []
        plan = build_ingest_plan(cli_args(init_db_only=True, root=Path.cwd(), repos="."))

        with (
            patch("project_code_intelligence.ingest_code_intel.prepare_writable_database", return_value=bootstrap),
            patch("project_code_intelligence.ingest_code_intel.scan_plan") as scan_plan,
            patch("project_code_intelligence.ingest_code_intel.progress.emit_summary", side_effect=summaries.append),
        ):
            status = run_ingest_plan(plan)

        self.assertEqual(status, 0)
        scan_plan.assert_not_called()
        self.assertEqual(summaries[0]["mode"], "init-db")
        self.assertEqual(summaries[0]["database_created"], True)


class CodeIntelParserTests(unittest.TestCase):
    def test_env_bool_rejects_ambiguous_values(self) -> None:
        with self.assertRaises(ConfigError):
            _ = env_bool("PROJECT_CODE_INTELLIGENCE_PREEMBED", env={"PROJECT_CODE_INTELLIGENCE_PREEMBED": "maybe"})

    def test_profile_loader_reports_configuration_errors(self) -> None:
        with self.assertRaises(ProfileLoadError):
            _ = load_profile("missing-profile")

    def test_default_embedding_endpoint_infers_lemonade_for_flm_model(self) -> None:
        self.assertEqual(
            default_embedding_endpoint({"PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL": "embed-gemma-300m-FLM"}),
            DEFAULT_LEMONADE_EMBEDDING_ENDPOINT,
        )

    def test_default_embedding_endpoint_keeps_non_flm_unset(self) -> None:
        self.assertIsNone(default_embedding_endpoint({"PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL": "local"}))

    def test_default_embedding_endpoint_can_use_local_fastembed_default(self) -> None:
        self.assertEqual(
            default_embedding_endpoint(
                {"PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL": "local"}, local_default=True
            ),
            "http://127.0.0.1:18081/v1/embeddings",
        )

    def test_default_embedding_endpoint_prefers_configured_endpoint(self) -> None:
        self.assertEqual(
            default_embedding_endpoint({
                "PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT": "http://127.0.0.1:18081/v1/embeddings",
                "PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL": "embed-gemma-300m-FLM",
            }),
            "http://127.0.0.1:18081/v1/embeddings",
        )

    def test_default_embedding_endpoint_model_uses_strict_local_runtime_default(self) -> None:
        self.assertEqual(
            default_embedding_endpoint_model(endpoint=DEFAULT_LEMONADE_EMBEDDING_ENDPOINT),
            DEFAULT_EMBEDDING_ENDPOINT_MODEL,
        )

    def test_default_embedding_endpoint_model_prefers_configured_model(self) -> None:
        self.assertEqual(
            default_embedding_endpoint_model(
                {"PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL": "custom-model"},
                endpoint=DEFAULT_LEMONADE_EMBEDDING_ENDPOINT,
            ),
            "custom-model",
        )

    def test_ingest_settings_infer_lemonade_endpoint_for_flm_model(self) -> None:
        settings = IngestSettings.from_env({
            "PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL": "embed-gemma-300m-FLM"
        })

        self.assertEqual(settings.embedding_endpoint, DEFAULT_LEMONADE_EMBEDDING_ENDPOINT)

    def test_ingest_settings_uses_strict_local_runtime_model_for_shared_endpoint(self) -> None:
        settings = IngestSettings.from_env({
            "PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT": DEFAULT_LEMONADE_EMBEDDING_ENDPOINT
        })

        self.assertEqual(settings.embedding_endpoint_model, DEFAULT_EMBEDDING_ENDPOINT_MODEL)

    def test_embedding_endpoint_policy_defaults_to_loopback(self) -> None:
        validate_embedding_endpoint("http://127.0.0.1:18081/v1/embeddings", env={})
        validate_embedding_endpoint("http://localhost:18081/v1/embeddings", env={})

        with self.assertRaises(ValueError):
            validate_embedding_endpoint("file:///etc/passwd", env={})
        with self.assertRaises(ValueError):
            validate_embedding_endpoint("https://embedding.example.invalid/v1/embeddings", env={})

        validate_embedding_endpoint(
            "https://embedding.example.invalid/v1/embeddings",
            env={"PROJECT_CODE_INTELLIGENCE_ALLOW_REMOTE_EMBEDDING": "1"},
        )

    def test_semantic_query_uses_configured_embedding_endpoint(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT": "http://127.0.0.1:18081/v1/embeddings",
                    "PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL": "demo-model",
                },
            ),
            patch(
                "project_code_intelligence.mcp.tools.embeddings.embed_with_endpoint",
                return_value=["[0.1,0.2,0.3]"],
            ) as embed_with_endpoint,
            patch("project_code_intelligence.mcp.tools.llama.embed_text") as llama_embed_text,
        ):
            vector, dimensions = query_embedding("hello")

        self.assertEqual(vector, "[0.1,0.2,0.3]")
        self.assertEqual(dimensions, 3)
        embed_with_endpoint.assert_called_once_with(
            "http://127.0.0.1:18081/v1/embeddings",
            ["hello"],
            "demo-model",
            track_metrics=False,
        )
        llama_embed_text.assert_not_called()

    def test_fastembed_server_builds_openai_embedding_response(self) -> None:
        response = embedding_response(FakeFastEmbedModel(), "demo-model", {"input": ["alpha", "beta"]})

        self.assertEqual(response["object"], "list")
        self.assertEqual(response["model"], "demo-model")
        self.assertEqual(
            response["data"],
            [
                {"object": "embedding", "index": 0, "embedding": [0.0, 5.0]},
                {"object": "embedding", "index": 1, "embedding": [1.0, 4.0]},
            ],
        )
        self.assertEqual(response["usage"], {"prompt_tokens": 3, "total_tokens": 3})

    def test_fastembed_server_rejects_non_string_input_items(self) -> None:
        with self.assertRaises(TypeError):
            _ = normalize_input(["ok", 123])

    def test_ingest_settings_parse_ci_friendly_environment(self) -> None:
        settings = IngestSettings.from_env({
            "PROJECT_CODE_INTELLIGENCE_COLLECTION": "nightly",
            "PROJECT_CODE_INTELLIGENCE_REPOS": "repo-a,repo-b",
            "PROJECT_CODE_INTELLIGENCE_MODE": "full",
            "PROJECT_CODE_INTELLIGENCE_PREEMBED": "0",
            "PROJECT_CODE_INTELLIGENCE_RUNTIME_HEARTBEAT_SECONDS": "60",
        })

        self.assertEqual(settings.collection, "nightly")
        self.assertEqual(settings.repos, "repo-a,repo-b")
        self.assertEqual(settings.mode, "full")
        self.assertFalse(settings.preembed)
        self.assertEqual(settings.runtime_heartbeat_seconds, 60)


class TypescriptParserRegressionTests(unittest.TestCase):
    def test_typescript_function_body_ignores_default_parameter_object_literal(self) -> None:
        text = "\n".join([
            "export function process(",
            "  schema: Schema,",
            "  _params = { path: [], schemaPath: [] }",
            "): void {",
            "  const inner = normalize(schema);",
            "  return inner._zod.run({ value: schema, issues: [] }, ctx);",
            "}",
        ])

        records, edges = javascript_records(fixture_file("core/to-json-schema.ts", "typescript"), text, 2400, 0)
        process_chunk = next(
            record for record in records if record.record_type == "code_chunk" and record.symbol == "process"
        )
        run_edge = next(edge for edge in edges if edge.target_symbol == "run")

        self.assertEqual(process_chunk.line_start, 1)
        self.assertEqual(process_chunk.line_end, 7)
        self.assertIn("inner._zod.run", process_chunk.display_content)
        self.assertEqual(run_edge.metadata["call_kind"], "member_call")
        self.assertEqual(run_edge.metadata["target_resolvable"], False)
        self.assertEqual(run_edge.metadata["full_symbol"], "inner._zod.run")

    def test_typescript_tiny_coverage_chunks_skip_embedding(self) -> None:
        text = "\n".join([
            "export function build() { return value(); }",
            "// Never",
            "export function parse() { return build(); }",
            "const trailing = build();",
        ])

        records, _edges = javascript_records(fixture_file("src/app.ts", "typescript"), text, 1200, 0)
        coverage_chunks = [
            record
            for record in records
            if record.record_type == "code_chunk" and record.metadata.get("fallback_reason") == "coverage line window"
        ]
        tiny = next(record for record in coverage_chunks if "// Never" in record.display_content)
        trailing = next(record for record in coverage_chunks if "trailing" in record.display_content)

        self.assertTrue(tiny.metadata.get("embedding_skipped"))
        self.assertEqual(
            tiny.metadata.get("embedding_skip_reason"),
            "tiny coverage chunk omitted from semantic embedding",
        )
        self.assertNotIn("embedding_skipped", trailing.metadata)

    def test_typescript_annotated_export_function_is_symbol_definition(self) -> None:
        text = "\n".join([
            "export interface $constructor<T> {",
            "  new (def: unknown): T;",
            "}",
            "export /*@__NO_SIDE_EFFECTS__*/ function $constructor<T>(name: string) {",
            "  return name as T;",
            "}",
        ])

        records, _edges = javascript_records(fixture_file("core/core.ts", "typescript"), text, 2400, 0)
        kinds_by_symbol = {
            (record.symbol, record.symbol_kind) for record in records if record.record_type == "symbol_definition"
        }

        self.assertIn(("$constructor", "interface"), kinds_by_symbol)
        self.assertIn(("$constructor", "function"), kinds_by_symbol)


class GoParserRegressionTests(unittest.TestCase):
    def test_go_records_emit_method_receiver_metadata(self) -> None:
        text = "\n".join([
            "package configs",
            "",
            "func (cnf *Configurator) AddOrUpdateVirtualServer(name string) {",
            "    cnf.generatePolicies(name)",
            "}",
            "",
            "func (c Configuration) AddOrUpdateVirtualServer(name string) {",
            "    c.Apply(name)",
            "}",
        ])

        records, _edges = go_records(fixture_file("internal/configs/virtualserver.go", "go"), text, 2400, 0)
        methods = [
            record
            for record in records
            if record.record_type == "symbol_definition" and record.symbol == "AddOrUpdateVirtualServer"
        ]

        self.assertEqual(len(methods), 2)
        by_receiver = {str(record.metadata["go_receiver_type"]): record for record in methods}
        self.assertEqual(set(by_receiver), {"Configurator", "Configuration"})
        self.assertEqual(by_receiver["Configurator"].symbol_kind, "method")
        self.assertEqual(by_receiver["Configurator"].metadata["go_receiver_name"], "cnf")
        self.assertTrue(by_receiver["Configurator"].metadata["go_receiver_pointer"])
        self.assertEqual(by_receiver["Configurator"].metadata["go_package"], "configs")
        self.assertEqual(
            by_receiver["Configurator"].metadata["qualified_symbol"],
            "Configurator.AddOrUpdateVirtualServer",
        )
        self.assertFalse(by_receiver["Configuration"].metadata["go_receiver_pointer"])
        self.assertEqual(
            by_receiver["Configuration"].metadata["qualified_symbol"],
            "Configuration.AddOrUpdateVirtualServer",
        )

    def test_go_records_strip_comments_from_call_edges(self) -> None:
        text = "\n".join([
            "package configs",
            "",
            "func generatePolicies() {",
            "    // syncPolicy() is handled by the caller.",
            "    applyPolicy()",
            "}",
        ])

        _records, edges = go_records(fixture_file("internal/configs/policy.go", "go"), text, 2400, 0)
        targets = {edge.target_symbol for edge in edges}

        self.assertIn("applyPolicy", targets)
        self.assertNotIn("syncPolicy", targets)

    def test_security_records_ignore_comment_only_backticks(self) -> None:
        text = "\n".join([
            "# Use `pci-index` to refresh the local index.",
            "run_command() {",
            "    echo ok",
            "}",
        ])

        records = security_records(fixture_file("scripts/README.sh", "shell"), text)

        self.assertFalse([record for record in records if record.rule_id == "shell_backtick_execution"])

    def test_security_records_ignore_python_docstring_backticks(self) -> None:
        text = "\n".join([
            '"""',
            "Run `pci-index` before checking examples.",
            '"""',
            "",
            "def run():",
            "    return True",
        ])

        records = security_records(fixture_file("pkg/example.py", "python"), text)

        self.assertFalse([record for record in records if record.rule_id == "shell_backtick_execution"])

    def test_security_records_ignore_python_string_and_inline_comment_backticks(self) -> None:
        text = "\n".join([
            "def validate_listener(name):",
            "    expected = 'listener.http must use `http` mode'",
            "    return name == expected  # reject unknown `listener.http` values",
        ])

        records = security_records(fixture_file("pkg/listeners.py", "python"), text)

        self.assertFalse([record for record in records if record.rule_id == "shell_backtick_execution"])


class RustParserRegressionTests(unittest.TestCase):
    def test_rust_records_collapse_doc_examples_and_skip_doc_edges(self) -> None:
        text = "\n".join([
            "impl Thing {",
            "    /// Returns readiness.",
            "    ///",
            "    /// ```no_run",
            "    /// fn f() {",
            "    ///     let guard = ready();",
            "    ///     guard.get_ref();",
            "    ///     guard.clear_ready_matching();",
            "    /// }",
            "    /// ```",
            "    pub fn poll(&self) {",
            "        self.actual();",
            "    }",
            "    pub fn actual(&self) {}",
            "}",
        ])

        records, edges = rust_records(fixture_file("src/io/async_fd.rs", "rust"), text, 2400, 0)
        impl_record = next(
            record for record in records if record.record_type == "symbol_definition" and record.symbol == "Thing"
        )
        targets = {edge.target_symbol for edge in edges}

        self.assertIn("[rustdoc example collapsed]", impl_record.display_content)
        for doc_example_text in ("fn f()", "ready();", "guard.get_ref();", "guard.clear_ready_matching();"):
            self.assertNotIn(doc_example_text, impl_record.display_content)
        for doc_symbol in ("f", "ready", "get_ref", "clear_ready_matching"):
            self.assertNotIn(doc_symbol, targets, msg=f"doc example emitted edge to {doc_symbol!r}")
        self.assertIn("Thing::actual", targets)

    def test_rust_records_qualify_same_impl_unqualified_method_edges(self) -> None:
        text = "\n".join([
            "pub struct Budget;",
            "impl Budget {",
            "    pub fn poll(&self) -> bool {",
            "        initial();",
            "        has_remaining()",
            "    }",
            "    pub fn initial() -> Self {",
            "        Budget",
            "    }",
            "    pub fn has_remaining() -> bool {",
            "        true",
            "    }",
            "}",
        ])

        records, edges = rust_records(fixture_file("task/coop/mod.rs", "rust"), text, 2400, 0)
        symbols = {record.symbol for record in records if record.record_type == "symbol_definition"}
        targets = {edge.target_symbol for edge in edges}

        self.assertIn("Budget::initial", symbols)
        self.assertIn("Budget::has_remaining", symbols)
        self.assertIn("Budget::initial", targets)
        self.assertIn("Budget::has_remaining", targets)
        self.assertNotIn("initial", targets)
        self.assertNotIn("has_remaining", targets)

    def test_rust_records_skip_callable_function_parameter_edges_with_lifetimes(self) -> None:
        text = "\n".join([
            "impl<T> AsyncFdReadyGuard<'_, T> {",
            "    pub(crate) fn try_io<'a, R>(",
            "        &self,",
            "        f: impl FnOnce(&'a AsyncFd<Inner>) -> io::Result<R>,",
            "    ) -> io::Result<R> {",
            "        let result = f(self.async_fd);",
            "        self.finish();",
            "        result",
            "    }",
            "    fn finish(&self) {}",
            "}",
        ])

        _records, edges = rust_records(fixture_file("src/io/async_fd.rs", "rust"), text, 2400, 0)
        targets = {edge.target_symbol for edge in edges}

        self.assertIn("AsyncFdReadyGuard::finish", targets)
        self.assertNotIn("f", targets)

    def test_rust_records_skip_local_closure_callable_edges(self) -> None:
        text = "\n".join([
            "pub fn drive() {",
            "    let wrapper = |f| {",
            "        f();",
            "        helper();",
            "    };",
            "    wrapper();",
            "    helper();",
            "}",
            "pub fn helper() {}",
        ])

        _records, edges = rust_records(fixture_file("src/lib.rs", "rust"), text, 2400, 0)
        targets = {edge.target_symbol for edge in edges}

        self.assertIn("helper", targets)
        self.assertNotIn("f", targets)
        self.assertNotIn("wrapper", targets)


class ParserAndRuntimeTests(unittest.TestCase):
    def test_full_repo_ingest_replaces_rows_even_when_plan_started_incremental(self) -> None:
        plan = IngestPlan(
            args=cli_args(),
            profile=load_profile("generic"),
            root=Path(),
            collection="test",
            repos=["repo-a", "repo-b"],
            embed_types=set(),
            sarif_files=[],
            embedding_requested=False,
            preembedding_requested=False,
            mode="incremental",
        )
        ingests = [
            RepoIngest(
                snapshot=snapshot_for_repo("repo-a"), files=[], records=[], edges=[], parser_failures=[], mode="full"
            ),
            RepoIngest(
                snapshot=snapshot_for_repo("repo-b"),
                files=[],
                records=[],
                edges=[],
                parser_failures=[],
                mode="incremental",
            ),
        ]
        conn = cast("db.DbConnection", object())

        with patch("project_code_intelligence.ingest_code_intel.replace_repos") as mocked_replace:
            replace_repos_for_full_ingests(conn, plan, ingests)

        mocked_replace.assert_called_once_with(conn, "test", ["repo-a"])

    def test_line_offsets_are_one_based(self) -> None:
        offsets = line_offsets("one\ntwo\nthree")

        self.assertEqual(line_for_offset_with_index(offsets, 0), 1)
        self.assertEqual(line_for_offset_with_index(offsets, 4), 2)
        self.assertEqual(line_for_offset_with_index(offsets, 8), 3)

    def test_python_records_use_qualified_symbols(self) -> None:
        text = "\n".join([
            "class Foo:",
            "    def a(self):",
            "        def inner():",
            "            return 1",
            "        return inner()",
            "",
            "def top():",
            "    return 2",
        ])

        records, _edges = python_records(fixture_file("pkg/example.py", "python"), text, 2400, 0)
        symbols = {record.symbol for record in records if record.record_type == "symbol_definition"}

        self.assertIn("Foo", symbols)
        self.assertIn("Foo.a", symbols)
        self.assertIn("Foo.a.inner", symbols)
        self.assertIn("top", symbols)

    def test_go_records_detect_generic_function(self) -> None:
        text = "\n".join([
            "package main",
            "",
            "func Map[T any](items []T) []T {",
            "    return items",
            "}",
        ])

        records, _edges = go_records(fixture_file("main.go", "go"), text, 2400, 0)
        symbols = {record.symbol for record in records if record.record_type == "symbol_definition"}

        self.assertIn("Map", symbols)

    def test_go_records_skip_call_edges_to_builtins(self) -> None:
        # Without the blocklist, `append` here would emit a heuristic edge whose
        # target_symbol="append", which the SQL resolver would later bind to any
        # user-defined symbol named "append" in the same snapshot.
        text = "\n".join([
            "package main",
            "",
            "func Acc(xs []int, x int) []int {",
            "    helper(xs)",
            "    return append(xs, x)",
            "}",
        ])

        _records, edges = go_records(fixture_file("main.go", "go"), text, 2400, 0)
        targets = {edge.target_symbol for edge in edges}

        self.assertIn("helper", targets)
        self.assertNotIn("append", targets)

    def test_rust_records_filter_keyword_and_method_edges(self) -> None:
        # The Rust parser is heuristic: extract_referenced_symbols pulls every
        # identifier-shaped token from the body. Without the non_resolvable_targets
        # filter, keywords (let/pub/ref) and ubiquitous Option/Result methods
        # (unwrap/ok_or_else/map_err) would emit call_candidate edges and the SQL
        # resolver would bind them to any same-named user symbol.
        text = "\n".join([
            "pub fn validate(&self) -> Result<()> {",
            "    let parsed = self.parse_config_file()?;",
            "    let merged = build_merged_ast_if_needed(&parsed)",
            "        .map_err(|e| Error::new(e.to_string()))",
            "        .ok_or_else(|| Error::missing())?;",
            "    Ok(merged)",
            "}",
        ])
        _records, edges = rust_records(fixture_file("lib.rs", "rust"), text, 2400, 0)
        targets = {edge.target_symbol for edge in edges}

        self.assertIn("parse_config_file", targets)
        self.assertIn("build_merged_ast_if_needed", targets)
        for noise in ("let", "pub", "ref", "unwrap", "ok_or_else", "map_err", "to_string"):
            self.assertNotIn(noise, targets, msg=f"unexpected edge to {noise!r}")

        budget_text = "\n".join([
            "pub struct Budget(Option<usize>);",
            "impl Budget {",
            "    pub fn poll(&self) -> Option<bool> {",
            "        Some(self.has_remaining())",
            "    }",
            "    pub fn has_remaining(&self) -> bool {",
            "        self.0.map_or(false, |value| value > 0)",
            "    }",
            "}",
        ])

        _budget_records, budget_edges = rust_records(fixture_file("task/coop/mod.rs", "rust"), budget_text, 2400, 0)
        budget_targets = {edge.target_symbol for edge in budget_edges}

        self.assertIn("Budget::has_remaining", budget_targets)
        for noise in ("Some", "map_or", "Option"):
            self.assertNotIn(noise, budget_targets, msg=f"unexpected edge to {noise!r}")

    def test_rust_records_strip_noise_from_symbols_referenced(self) -> None:
        text = "\n".join([
            "pub fn handler(&self) {",
            "    let result = self.fetch().unwrap_or_default();",
            "    result.to_string()",
            "}",
        ])
        records, _edges = rust_records(fixture_file("lib.rs", "rust"), text, 2400, 0)
        chunk = next(r for r in records if r.record_type == "code_chunk")
        symbols_referenced = chunk.metadata.get("symbols_referenced", [])
        if not isinstance(symbols_referenced, list):
            self.fail("symbols_referenced should be a list")
        names = {str(s) for s in symbols_referenced}

        # Real symbol still surfaces.
        self.assertIn("fetch", names)
        # Noise is gone from the metadata view too — same set drives both.
        for noise in ("let", "pub", "unwrap_or_default", "to_string"):
            self.assertNotIn(noise, names, msg=f"noise {noise!r} leaked into symbols_referenced")

        long_body = [f"    let value_{idx} = {idx};" for idx in range(205)]
        long_body.extend([
            "    let printer = PrinterBuilder::new();",
            "    printer.print_all();",
        ])
        text = "\n".join([
            "pub struct PrinterBuilder;",
            "impl PrinterBuilder {",
            "    pub fn new() -> Self {",
            "        Self",
            "    }",
            "    pub fn print_all(&self) {}",
            "}",
            "pub struct GroupSize;",
            "impl From<GroupSize> for usize {",
            "    fn from(value: GroupSize) -> Self {",
            "        1",
            "    }",
            "}",
            "pub fn run() {",
            *long_body,
            "}",
            "#[cfg(test)]",
            "mod tests {",
            "    #[test]",
            "    fn empty_file_passes() {}",
            "}",
        ])

        records, edges = rust_records(fixture_file("src/main.rs", "rust"), text, 2400, 0)
        symbols = {record.symbol for record in records if record.record_type == "symbol_definition"}

        self.assertIn("PrinterBuilder::new", symbols)
        self.assertIn("PrinterBuilder::print_all", symbols)
        self.assertIn("From<GroupSize>::from", symbols)
        self.assertNotIn("new", symbols)

        run_record = next(
            record for record in records if record.record_type == "symbol_definition" and record.symbol == "run"
        )
        self.assertEqual(run_record.line_end, 222)
        self.assertIn("PrinterBuilder::new", run_record.display_content)

        subchunks = [record for record in records if record.metadata.get("rust_symbol_subchunk")]
        self.assertTrue(any("PrinterBuilder::new" in record.display_content for record in subchunks))

        method = next(record for record in records if record.symbol == "PrinterBuilder::new")
        self.assertEqual(method.metadata["impl_owner"], "PrinterBuilder")
        trait_method = next(record for record in records if record.symbol == "From<GroupSize>::from")
        self.assertEqual(trait_method.metadata["impl_owner"], "usize")
        self.assertEqual(trait_method.metadata["impl_trait"], "From<GroupSize>")
        self.assertIn("PrinterBuilder::new", {edge.target_symbol for edge in edges})

        inline_test = next(record for record in records if record.symbol == "empty_file_passes")
        self.assertEqual(inline_test.file_role, "test")
        self.assertTrue(inline_test.metadata["rust_test"])

    def test_typescript_template_literals_do_not_emit_shell_backtick_security_records(self) -> None:
        text = "\n".join([
            "export function failOnConsole(message: string) {",
            "  throw new Error(`Invalid console output: ${message}`);",
            "}",
        ])

        records = security_records(fixture_file("scripts/check-semver.ts", "typescript"), text)

        self.assertFalse([record for record in records if record.rule_id == "shell_backtick_execution"])

        records, _edges = javascript_records(fixture_file("scripts/fail-on-console.ts", "typescript"), text, 2400, 0)
        records_with_security_metadata = [record for record in records if "security_sensitive_apis" in record.metadata]

        self.assertTrue(records_with_security_metadata)
        for record in records_with_security_metadata:
            sensitive_apis = record.metadata.get("security_sensitive_apis", [])
            if not isinstance(sensitive_apis, list):
                self.fail("security_sensitive_apis should be a list")
            self.assertNotIn("shell_backtick_execution", sensitive_apis)

    def test_typescript_records_emit_symbols_and_conservative_call_edges(self) -> None:
        text = "\n".join([
            "export interface LazyOptions {",
            "  readonly path: string;",
            "}",
            "",
            "export type LazyFactory = () => unknown;",
            "",
            "export const $ZodLazy = createLazyFactory();",
            "",
            "export function defineLazy<T>(getter: () => T) {",
            "  const schema = buildSchema(getter());",
            "  return parse(schema);",
            "}",
            "",
            "const helper = (value: unknown) => {",
            "  describe('noise', () => expect(value).toBeDefined());",
            "  return normalize(value);",
            "};",
        ])

        records, edges = javascript_records(fixture_file("core/schemas.ts", "typescript"), text, 2400, 0)
        symbols = {record.symbol for record in records if record.record_type == "symbol_definition"}
        targets = {edge.target_symbol for edge in edges}

        self.assertIn("LazyOptions", symbols)
        self.assertIn("LazyFactory", symbols)
        self.assertIn("$ZodLazy", symbols)
        self.assertIn("defineLazy", symbols)
        self.assertIn("helper", symbols)
        self.assertIn("buildSchema", targets)
        self.assertIn("parse", targets)
        self.assertIn("normalize", targets)
        for noise in ("describe", "expect"):
            self.assertNotIn(noise, targets)

    def test_typescript_multiline_function_body_extends_past_signature(self) -> None:
        text = "\n".join([
            "export function extractDefs<T extends Schema>(",
            "  ctx: Context,",
            "  schema: T",
            "  // params: EmitParams",
            "): void {",
            "  const idToSchema = new Map<string, Schema>();",
            "  for (const entry of ctx.seen.entries()) {",
            "    const id = ctx.metadataRegistry.get(entry[0])?.id;",
            "    if (id) {",
            '      throw new Error(`Duplicate schema id "${id}" detected during JSON Schema conversion.`);',
            "    }",
            "  }",
            "}",
        ])

        records, _edges = javascript_records(fixture_file("core/to-json-schema.ts", "typescript"), text, 2400, 0)
        extract_defs = next(
            record for record in records if record.record_type == "code_chunk" and record.symbol == "extractDefs"
        )

        self.assertEqual(extract_defs.line_start, 1)
        self.assertEqual(extract_defs.line_end, 13)
        self.assertIn("Duplicate schema id", extract_defs.display_content)
        self.assertIn("idToSchema", extract_defs.display_content)

    def test_typescript_records_keep_nonblank_lines_searchable(self) -> None:
        text = "\n".join([
            'import { z } from "zod";',
            "",
            "export interface Options {",
            "  readonly name: string;",
            "}",
            "",
            "export function build(options: Options) {",
            "  return z.object({ name: z.string().default(options.name) });",
            "}",
            "",
            "const trailing = build({ name: 'demo' });",
        ])

        records, _edges = javascript_records(fixture_file("src/app.ts", "typescript"), text, 1200, 0)
        code_chunks = [record for record in records if record.record_type == "code_chunk"]
        covered_lines = {
            line
            for record in code_chunks
            if record.line_start is not None and record.line_end is not None
            for line in range(record.line_start, record.line_end + 1)
        }
        nonblank_lines = {idx for idx, line in enumerate(text.splitlines(), 1) if line.strip()}

        self.assertLessEqual(nonblank_lines, covered_lines)
        coverage_chunks = [
            record for record in code_chunks if record.metadata.get("fallback_reason") == "coverage line window"
        ]
        self.assertTrue(any("trailing" in record.display_content for record in coverage_chunks))

    def test_typescript_records_filter_keyword_builtin_and_member_noise_edges(self) -> None:
        text = "\n".join([
            "export const $ZodLazy = makeLazy();",
            "",
            "export function bootstrap(object: Record<string, unknown>) {",
            "  if (object.get) object.get();",
            "  object.set('x', 1);",
            "  util.defineLazy(object);",
            "  core.$ZodLazy.init(object);",
            "  inner._zod.run(object);",
            "}",
        ])

        _records, edges = javascript_records(fixture_file("core/schemas.ts", "typescript"), text, 2400, 0)
        targets = {edge.target_symbol for edge in edges}
        run_edge = next(edge for edge in edges if edge.target_symbol == "run")
        define_lazy_edge = next(edge for edge in edges if edge.target_symbol == "defineLazy")

        self.assertIn("defineLazy", targets)
        self.assertIn("$ZodLazy", targets)
        self.assertEqual(run_edge.metadata["target_resolvable"], False)
        self.assertNotIn("target_resolvable", define_lazy_edge.metadata)
        for noise in ("if", "get", "set", "init"):
            self.assertNotIn(noise, targets)

    def test_typescript_exported_const_initializers_emit_member_call_edges(self) -> None:
        text = "\n".join([
            "export interface FancyLazy {}",
            "export const FancyLazy: Constructor<FancyLazy> = constructor(",
            '  "FancyLazy",',
            "  (inst, def) => {",
            "    core.$BaseLazy.init(inst, def);",
            "    WidgetType.init(inst, def);",
            "  }",
            ");",
        ])

        records, edges = javascript_records(fixture_file("src/lazy.ts", "typescript"), text, 2400, 0)
        symbols = {record.symbol for record in records if record.record_type == "symbol_definition"}
        targets = {edge.target_symbol for edge in edges}

        self.assertIn("FancyLazy", symbols)
        self.assertIn("$BaseLazy", targets)
        self.assertIn("WidgetType", targets)
        self.assertNotIn("init", targets)

    def test_go_records_keep_file_only_metadata_off_records(self) -> None:
        # Simulate what inventory.py does: populate file metadata via the Go
        # language profile, then build records. Sibling-list keys belong on the
        # file row, not duplicated onto every per-function record.
        text = "\n".join([
            "package main",
            "",
            'import "context"',
            "",
            "func Alpha() {}",
            "func Beta() {}",
        ])
        intel_file = fixture_file("main.go", "go")
        intel_file = replace(intel_file, metadata=dict(go_file_metadata("main.go", text)))
        records, _edges = go_records(intel_file, text, 2400, 0)
        function_records = [r for r in records if r.symbol_kind == "function"]
        self.assertTrue(function_records)
        for record in function_records:
            self.assertNotIn("go_functions", record.metadata)
            self.assertNotIn("go_imports", record.metadata)
            self.assertNotIn("go_methods", record.metadata)
            # go_package is single-value and explicitly propagated.
            self.assertEqual(record.metadata.get("go_package"), "main")

    def test_make_records_include_top_level_package_pins_when_blocks_exist(self) -> None:
        text = "\n".join([
            "include $(TOPDIR)/rules.mk",
            "",
            "PKG_SOURCE_VERSION:=c7e364d5fbdfc44e19ec86c80142d2fdf498c702",
            "PKG_MIRROR_HASH:=3a9e5f3489d1d1abcf514332946ef55fda0c4bb9c51e8f4f9d51ef50f0fd8b56",
            "",
            "define KernelPackage/ask-cdx",
            "  TITLE:=ASK CDX driver",
            "endef",
        ])
        previous_profile = profile_context.active_profile
        try:
            profile_context.set_active_profile(load_profile("generic"))
            records, _edges = make_records(fixture_file("package/kernel/ask-cdx/Makefile", "make"), text, 2400, 0)
        finally:
            profile_context.set_active_profile(previous_profile)

        source_version_records = [
            record
            for record in records
            if record.record_type == "make_assignment" and record.symbol == "PKG_SOURCE_VERSION"
        ]
        self.assertEqual(len(source_version_records), 1)
        rendered_records = "\n".join(record.display_content for record in records)
        self.assertIn("c7e364d5fbdfc44e19ec86c80142d2fdf498c702", rendered_records)
        self.assertIn("3a9e5f3489d1d1abcf514332946ef55fda0c4bb9c51e8f4f9d51ef50f0fd8b56", rendered_records)

    def test_small_line_window_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _ = line_window_records(fixture_file("small.txt", "text"), "hello", 20, 0)

    def test_cli_bounds_are_validated(self) -> None:
        args = cli_args(chunk_chars=99)

        with self.assertRaises(ValueError):
            validate_args(args, embedding_requested=False)

    def test_scan_workers_auto_stays_serial_for_small_scans(self) -> None:
        self.assertEqual(resolve_scan_workers(0, 1), 1)
        self.assertEqual(resolve_scan_workers(0, 63), 1)
        self.assertGreaterEqual(resolve_scan_workers(0, 64), 1)
        self.assertEqual(resolve_scan_workers(8, 3), 3)
        self.assertEqual(resolve_scan_workers(0, 10_000), 8)

    def test_reset_only_requires_reset_flag(self) -> None:
        with self.assertRaises(ValueError):
            validate_args(cli_args(reset_only=True), embedding_requested=False)

    def test_runtime_progress_percent(self) -> None:
        metrics = RuntimeMetrics()
        metrics.configure_progress({"scan": 0.5, "db_upload": 0.5})
        metrics.begin_phase("scan", total=10)
        metrics.add_phase_done(5)

        snapshot = metrics.snapshot()
        progress_value = snapshot.get("progress")
        if not isinstance(progress_value, dict):
            self.fail("runtime snapshot progress should be an object")
        progress = progress_value

        self.assertEqual(progress["phase"], "scan")
        self.assertEqual(progress["phase_percent"], 50.0)
        self.assertEqual(progress["overall_percent_estimated"], 25.0)


class McpQueryTests(unittest.TestCase):
    def test_mcp_record_queries_default_to_latest_snapshot(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            clauses, params = code_intel_clauses({"collection": "test", "repo": "sample-repo"}, "r")
        sql = " AND ".join(clauses)

        self.assertIn("r.collection = %s", sql)
        self.assertIn("r.repo = %s", sql)
        self.assertIn("r.snapshot_id", sql)
        self.assertIn("project_code_intel_snapshots latest_snapshot", sql)
        self.assertEqual(params, ["test", "sample-repo"])
        # "latest" is the implicit default — only non-default scopes are echoed.
        self.assertEqual(snapshot_scope_response({}), {})

    def test_mcp_record_queries_can_include_historical_snapshots(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            clauses, params = code_intel_clauses(
                {"collection": "test", "repo": "sample-repo", "include_historical": True},
                "r",
            )
        sql = " AND ".join(clauses)

        self.assertNotIn("latest_snapshot", sql)
        self.assertNotIn("snapshot_id", sql)
        self.assertEqual(params, ["test", "sample-repo"])
        self.assertEqual(
            snapshot_scope_response({"include_historical": True}),
            {"snapshot_scope": "historical"},
        )

    def test_mcp_collection_env_is_hard_scope_by_default(self) -> None:
        with patch.dict(os.environ, {"PROJECT_CODE_INTELLIGENCE_COLLECTION": "public"}, clear=True):
            clauses, params = code_intel_clauses({"repo": "sample-repo"}, "r")
            self.assertIn("r.collection = %s", " AND ".join(clauses))
            self.assertEqual(params[0], "public")

            with self.assertRaises(PermissionError):
                _ = code_intel_clauses({"collection": "private"}, "r")

    def test_mcp_collection_override_requires_explicit_opt_in(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PROJECT_CODE_INTELLIGENCE_COLLECTION": "public",
                "PROJECT_CODE_INTELLIGENCE_ALLOW_COLLECTION_OVERRIDE": "1",
            },
            clear=True,
        ):
            _clauses, params = code_intel_clauses({"collection": "private"}, "r")

        self.assertEqual(params[0], "private")

    def test_mcp_record_queries_can_target_snapshot_id(self) -> None:
        clauses, params = scoped_snapshot_clauses({"snapshot_id": 42}, "r")

        self.assertEqual(clauses, ["r.snapshot_id = %s"])
        self.assertEqual(params, [42])
        self.assertEqual(
            snapshot_scope_response({"snapshot_id": 42}), {"snapshot_scope": "snapshot_id", "snapshot_id": 42}
        )

    def test_mcp_snapshot_scope_rejects_ambiguous_args(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            _ = code_intel_clauses({"snapshot_id": 42, "include_historical": True}, "r")

    def test_mcp_static_finding_queries_default_to_latest_snapshot(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            clauses, params = static_finding_clauses({"collection": "test", "repo": "sample-repo"})
        sql = " AND ".join(clauses)

        self.assertIn("f.collection = %s", sql)
        self.assertIn("f.repo = %s", sql)
        self.assertIn("f.snapshot_id", sql)
        self.assertIn("project_code_intel_snapshots latest_snapshot", sql)
        self.assertEqual(params, ["test", "sample-repo"])


class SarifTests(unittest.TestCase):
    def test_generic_profile_discovers_sarif_under_selected_repos(self) -> None:
        previous_profile = profile_context.active_profile
        try:
            profile_context.set_active_profile(load_profile("generic"))
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                selected = root / "repo-a" / "codeql-results"
                fixture = root / "repo-a" / "build_dir" / "host" / "cmake" / "Tests" / "RunCMake"
                ignored = root / "unselected-repo" / "codeql-results"
                selected.mkdir(parents=True)
                fixture.mkdir(parents=True)
                ignored.mkdir(parents=True)
                selected_sarif = selected / "results.sarif"
                fixture_sarif = fixture / "example-expected.sarif"
                ignored_sarif = ignored / "results.sarif"
                _ = selected_sarif.write_text('{"version":"2.1.0","runs":[]}', encoding="utf-8")
                _ = fixture_sarif.write_text('{"version":"2.1.0","runs":[]}', encoding="utf-8")
                _ = ignored_sarif.write_text('{"version":"2.1.0","runs":[]}', encoding="utf-8")

                discovered = discover_sarif_files(root, ["repo-a"], [], include_profile=True)
                explicit_discovered = discover_sarif_files(
                    root,
                    ["repo-a"],
                    [str(fixture_sarif)],
                    include_profile=True,
                )

            self.assertEqual(discovered, [selected_sarif.resolve()])
            self.assertEqual(set(explicit_discovered), {selected_sarif.resolve(), fixture_sarif.resolve()})
        finally:
            profile_context.set_active_profile(previous_profile)

    def test_build_plan_defers_sarif_discovery_until_scan_phase(self) -> None:
        args = cli_args(root=Path(), repos="repo-a")

        with (
            patch(
                "project_code_intelligence.ingest_code_intel.discover_sarif_files",
                return_value=[Path("repo-a/results.sarif")],
            ) as mocked_discover,
            patch("project_code_intelligence.ingest_code_intel.progress_event") as mocked_progress,
        ):
            plan = build_ingest_plan(args)
            self.assertEqual(plan.sarif_files, [])
            mocked_discover.assert_not_called()

            discovered = discover_plan_sarif_files(plan)

        self.assertEqual(discovered, [Path("repo-a/results.sarif")])
        mocked_discover.assert_called_once()
        mocked_progress.assert_called_once_with("code_intel_sarif_discovering", repos=["repo-a"])

    def test_sarif_mtime_warning_is_soft_when_report_is_older_than_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sarif_dir = root / "repo-a" / "codeql-results"
            sarif_dir.mkdir(parents=True)
            sarif_path = sarif_dir / "results.sarif"
            _ = sarif_path.write_text('{"version":"2.1.0","runs":[]}', encoding="utf-8")
            old_timestamp = datetime(2026, 5, 9, tzinfo=timezone.utc).timestamp()
            os.utime(sarif_path, (old_timestamp, old_timestamp))
            plan = IngestPlan(
                args=cli_args(),
                profile=load_profile("generic"),
                root=root,
                collection="test",
                repos=["repo-a"],
                embed_types=set(),
                sarif_files=[sarif_path],
                embedding_requested=False,
                preembedding_requested=False,
                mode="full",
            )
            snapshot = Snapshot(
                collection="test",
                repo="repo-a",
                repo_role="source",
                branch="main",
                commit_sha="abc123",
                tree_sha="tree",
                dirty=False,
                metadata={"commit_time": "2026-05-10T00:00:00+00:00"},
            )

            warning = warning_for_sarif_mtime(plan, {"repo-a": snapshot}, sarif_path)

        self.assertIsNotNone(warning)
        if warning is None:
            self.fail("expected SARIF freshness warning")
        self.assertEqual(warning["severity"], "note")
        self.assertEqual(warning["reason"], "sarif_older_than_snapshot_commit")
        self.assertEqual(warning["sarif_path"], "repo-a/codeql-results/results.sarif")

    def test_sarif_ingest_creates_static_finding_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sarif_dir = root / "sample-repo" / "results"
            sarif_dir.mkdir(parents=True)
            sarif_path = sarif_dir / "semgrep.sarif"
            sarif = {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {
                            "driver": {
                                "name": "Semgrep",
                                "version": "1.0",
                                "rules": [
                                    {
                                        "id": "c.system",
                                        "name": "system-call",
                                        "shortDescription": {"text": "system call"},
                                        "fullDescription": {"text": "Avoid shelling out with unsanitized input."},
                                        "defaultConfiguration": {"level": "warning"},
                                        "helpUri": "https://example.invalid/rules/c.system",
                                        "properties": {
                                            "tags": ["security", "cwe-078"],
                                            "precision": "high",
                                            "security-severity": "8.1",
                                        },
                                    }
                                ],
                            }
                        },
                        "results": [
                            {
                                "ruleId": "c.system",
                                "level": "warning",
                                "message": {"text": "Avoid system()"},
                                "partialFingerprints": {"primaryLocationLineHash": "abc"},
                                "suppressions": [{"kind": "external"}],
                                "locations": [
                                    {
                                        "physicalLocation": {
                                            "artifactLocation": {"uri": "src/foo.c"},
                                            "region": {"startLine": 7, "snippet": {"text": "system(cmd);"}},
                                        }
                                    }
                                ],
                                "codeFlows": [
                                    {
                                        "threadFlows": [
                                            {
                                                "locations": [
                                                    {
                                                        "location": {
                                                            "physicalLocation": {
                                                                "artifactLocation": {"uri": "src/foo.c"},
                                                                "region": {"startLine": idx + 1},
                                                            },
                                                            "message": {"text": f"flow step {idx}"},
                                                        },
                                                        "importance": "important",
                                                    }
                                                    for idx in range(10)
                                                ]
                                            }
                                        ]
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
            _ = sarif_path.write_text(json.dumps(sarif), encoding="utf-8")
            intel_file = fixture_file("sample-repo/src/foo.c", "c")

            ingested = ingest_sarif(
                SarifIngestContext(
                    root=root,
                    repos=["sample-repo"],
                    collection="test",
                    file_by_source_path={intel_file.source_path: intel_file},
                    max_bytes=1024 * 1024,
                ),
                [sarif_path],
            )

            self.assertEqual(len(ingested.runs), 1)
            self.assertEqual(len(ingested.runs[0].findings), 1)
            records = ingested.records_by_repo["sample-repo"]
            self.assertEqual(records[0].record_type, "static_finding")
            self.assertEqual(records[0].source_path, "sample-repo/src/foo.c")
            self.assertEqual(records[0].tool, "Semgrep")
            self.assertEqual(records[0].metadata["rule_name"], "system-call")
            self.assertEqual(records[0].metadata["rule_precision"], "high")
            self.assertEqual(records[0].metadata["rule_security_severity"], "8.1")
            self.assertIn("cwe-078", require_list(records[0].metadata["rule_cwe"]))
            self.assertTrue(records[0].metadata["suppressed"])
            self.assertEqual(records[0].metadata["primary_path_mapping"], "indexed_source")
            self.assertIn("indexed_source", require_list(records[0].metadata["path_mappings"]))
            self.assertIn("flow step 0", records[0].embedding_text)
            self.assertIn("flow step 9", records[0].embedding_text)
            self.assertIn("flow step 0", str(records[0].metadata["code_flow_source"]))
            self.assertIn("flow step 9", str(records[0].metadata["code_flow_sink"]))
            self.assertIn("code_flow_source:", records[0].embedding_text)
            self.assertIn("code_flow_sink:", records[0].embedding_text)
            self.assertIn("...", require_list(records[0].metadata["code_flow_summary"]))
            self.assertIn("Rule short description", records[0].display_content)
            self.assertIn("```", records[0].display_content)

    def test_sarif_uri_base_id_maps_to_workspace_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample-repo" / "src" / "foo.c"
            source.parent.mkdir(parents=True)
            _ = source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
            source_path, repo = source_path_from_sarif_uri(
                SarifPathContext(
                    root=root,
                    repos=["sample-repo"],
                    default_repo="sample-repo",
                    uri_base_ids={"%SRCROOT%": source.parent.parent.as_uri() + "/"},
                    known_source_paths=None,
                ),
                "src/foo.c",
                uri_base_id="%SRCROOT%",
            )

            self.assertEqual(source_path, "sample-repo/src/foo.c")
            self.assertEqual(repo, "sample-repo")

    def test_sarif_uri_base_variants_map_to_workspace_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample-repo" / "src" / "foo.c"
            source.parent.mkdir(parents=True)
            _ = source.write_text("int main(void) { return 0; }\n", encoding="utf-8")

            cases = [
                (source.parent.parent.as_uri() + "/", "src/foo.c"),
                (source.parent.parent.as_uri(), "src/foo.c"),
                (source.parent.as_uri() + "/", "foo.c"),
            ]
            for base_uri, uri in cases:
                with self.subTest(base_uri=base_uri, uri=uri):
                    resolution = resolve_sarif_source_path(
                        SarifPathContext(
                            root=root,
                            repos=["sample-repo"],
                            default_repo="sample-repo",
                            uri_base_ids={"%SRCROOT%": base_uri},
                            known_source_paths={"sample-repo/src/foo.c"},
                        ),
                        uri,
                        uri_base_id="%SRCROOT%",
                    )

                    self.assertEqual(resolution.source_path, "sample-repo/src/foo.c")
                    self.assertEqual(resolution.repo, "sample-repo")
                    self.assertEqual(resolution.path_mapping, "indexed_source")

    def test_sarif_rule_index_without_rule_id_uses_rule_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sarif_dir = root / "sample-repo" / "results"
            sarif_dir.mkdir(parents=True)
            sarif_path = sarif_dir / "codeql.sarif"
            sarif = {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {
                            "driver": {
                                "name": "CodeQL",
                                "rules": [{"id": "cpp/unsafe-allocation", "name": "unsafe allocation"}],
                            }
                        },
                        "results": [
                            {
                                "ruleIndex": 0,
                                "message": {"text": "allocation uses unchecked input"},
                                "locations": [
                                    {
                                        "physicalLocation": {
                                            "artifactLocation": {"uri": "src/foo.c"},
                                            "region": {"startLine": 3},
                                        }
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
            _ = sarif_path.write_text(json.dumps(sarif), encoding="utf-8")
            intel_file = fixture_file("sample-repo/src/foo.c", "c")

            ingested = ingest_sarif(
                SarifIngestContext(
                    root=root,
                    repos=["sample-repo"],
                    collection="test",
                    file_by_source_path={intel_file.source_path: intel_file},
                    max_bytes=1024 * 1024,
                ),
                [sarif_path],
            )

            self.assertEqual(ingested.runs[0].findings[0].rule_id, "cpp/unsafe-allocation")
            self.assertEqual(ingested.records_by_repo["sample-repo"][0].symbol, "cpp/unsafe-allocation")

    def test_external_absolute_sarif_path_tracks_mapping_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample-repo").mkdir()
            resolution = resolve_sarif_source_path(
                SarifPathContext(
                    root=root,
                    repos=["sample-repo"],
                    default_repo="sample-repo",
                    uri_base_ids={},
                    known_source_paths=set(),
                ),
                "/workspace/build/foo.c",
            )

            self.assertEqual(resolution.source_path, "/workspace/build/foo.c")
            self.assertEqual(resolution.repo, "sample-repo")
            self.assertEqual(resolution.path_mapping, "external_absolute")

    def test_unmatched_sarif_relative_path_is_not_force_prefixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample-repo").mkdir()
            resolution = resolve_sarif_source_path(
                SarifPathContext(
                    root=root,
                    repos=["sample-repo"],
                    default_repo="sample-repo",
                    uri_base_ids={},
                    known_source_paths=set(),
                ),
                "../build/generated.c",
            )

            self.assertEqual(resolution.source_path, "../build/generated.c")
            self.assertEqual(resolution.repo, "sample-repo")
            self.assertEqual(resolution.path_mapping, "unresolved_relative")


if __name__ == "__main__":
    _ = unittest.main()
