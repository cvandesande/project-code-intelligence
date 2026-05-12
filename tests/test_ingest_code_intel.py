from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typing_extensions import override

from project_code_intelligence.code_profiles import load_profile
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
from project_code_intelligence.ingest_code_intel import CliArgs, confirm_reset_code_intel, validate_args
from project_code_intelligence.mcp.filters import (
    code_intel_clauses,
    scoped_snapshot_clauses,
    snapshot_scope_response,
    static_finding_clauses,
)
from project_code_intelligence.models import IntelFile, JsonValue
from project_code_intelligence.parsers import go_records, python_records
from project_code_intelligence.records import line_for_offset_with_index, line_offsets, line_window_records
from project_code_intelligence.runtime import RuntimeMetrics
from project_code_intelligence.sarif import (
    SarifIngestContext,
    SarifPathContext,
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
        "chunk_chars": 2400,
        "overlap_lines": 0,
        "limit_files": None,
        "progress_every": 0,
        "dry_run": False,
        "reset_code_intel": False,
        "i_know_this_deletes_code_intel_db": False,
        "reset_only": False,
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
    }
    values.update(overrides)
    return CliArgs(**values)  # type: ignore[arg-type]


class DatabaseSettingsTests(unittest.TestCase):
    def test_default_to_local_compose_pgvector(self) -> None:
        settings = DatabaseSettings.from_env({})

        self.assertEqual(settings.missing_connection_names(), [])
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, "5433")
        self.assertEqual(settings.dbname, "codeintel")
        self.assertEqual(settings.user, "codeintel")
        self.assertEqual(settings.password, "codeintel")
        self.assertIn("PGVECTOR_DB=codeintel", settings.connection_hint())
        self.assertIn("PGVECTOR_PASS=<set>", settings.connection_hint())

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
            )

        output = stderr.getvalue()
        self.assertIn("Database target: postgresql://app@db:5432/codeintel sslmode=prefer", output)
        self.assertIn("Tables: project_code_intel_*", output)
        self.assertNotIn("secret", output)

    def test_reset_confirmation_requires_flag_in_noninteractive_mode(self) -> None:
        with (
            patch("sys.stdin", io.StringIO("yes\n")),
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(ValueError),
        ):
            confirm_reset_code_intel(cli_args(reset_code_intel=True), DatabaseSettings())


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


class ParserAndRuntimeTests(unittest.TestCase):
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

    def test_small_line_window_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _ = line_window_records(fixture_file("small.txt", "text"), "hello", 20, 0)

    def test_cli_bounds_are_validated(self) -> None:
        args = cli_args(chunk_chars=99)

        with self.assertRaises(ValueError):
            validate_args(args, embedding_requested=False)

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
        clauses, params = code_intel_clauses({"collection": "test", "repo": "sample-repo"}, "r")
        sql = " AND ".join(clauses)

        self.assertIn("r.collection = %s", sql)
        self.assertIn("r.repo = %s", sql)
        self.assertIn("r.snapshot_id", sql)
        self.assertIn("project_code_intel_snapshots latest_snapshot", sql)
        self.assertEqual(params, ["test", "sample-repo"])
        self.assertEqual(snapshot_scope_response({}), {"snapshot_scope": "latest"})

    def test_mcp_record_queries_can_include_historical_snapshots(self) -> None:
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
        with self.assertRaises(ValueError):
            _ = code_intel_clauses({"snapshot_id": 42, "include_historical": True}, "r")

    def test_mcp_static_finding_queries_default_to_latest_snapshot(self) -> None:
        clauses, params = static_finding_clauses({"collection": "test", "repo": "sample-repo"})
        sql = " AND ".join(clauses)

        self.assertIn("f.collection = %s", sql)
        self.assertIn("f.repo = %s", sql)
        self.assertIn("f.snapshot_id", sql)
        self.assertIn("project_code_intel_snapshots latest_snapshot", sql)
        self.assertEqual(params, ["test", "sample-repo"])


class SarifTests(unittest.TestCase):
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
