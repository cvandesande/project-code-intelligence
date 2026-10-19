from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from typing import cast

from project_code_intelligence import db, profile_context
from project_code_intelligence.code_profiles.base import GenericProfile
from project_code_intelligence.models import (
    PARSER_VERSION,
    SCHEMA_VERSION,
    IntelEdge,
    IntelFile,
    IntelRecord,
    JsonObject,
    Snapshot,
    StaticCodeFlowStep,
    StaticFinding,
    StaticLocation,
    StaticRule,
    StaticRun,
)
from project_code_intelligence.storage import (
    RecordInsertContext,
    SnapshotScope,
    copy_unchanged_edges,
    copy_unchanged_parser_failures,
    delete_orphaned_records,
    file_signature,
    insert_records,
    insert_static_runs,
    lock_ingest_branch,
    parser_failure_metadata,
    pre_resolvable_edge_count,
    pre_resolve_edge_targets,
    replace_repos,
    resolve_edge_targets,
    row_int,
    snapshot_versions_compatible,
)


def snapshot_fixture() -> Snapshot:
    return Snapshot(
        collection="test",
        repo=".",
        repo_role="project",
        branch="main",
        commit_sha="commit",
        tree_sha="tree",
        dirty=False,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "chunker_version": "code-intel-v1",
            "parser_version": PARSER_VERSION,
            "profile_name": "generic",
            "profile_version": "v1",
        },
    )


def snapshot_compatibility_with_generic_profile() -> tuple[bool, bool, bool]:
    previous_profile = profile_context.active_profile
    try:
        profile_context.set_active_profile(GenericProfile())
        return _snapshot_compatibility_results()
    finally:
        profile_context.set_active_profile(previous_profile)


def _snapshot_compatibility_results() -> tuple[bool, bool, bool]:
    compatible = snapshot_versions_compatible(snapshot_fixture().metadata)
    incompatible: JsonObject = dict(snapshot_fixture().metadata)
    incompatible["profile_version"] = "old"
    old_profile_compatible = snapshot_versions_compatible(incompatible)
    missing_metadata_compatible = snapshot_versions_compatible(None)
    return compatible, old_profile_compatible, missing_metadata_compatible


def file_fixture(
    *,
    file_sha256: str | None = "filesha",
    git_blob_sha: str | None = "blobsha",
    skipped_reason: str | None = None,
) -> IntelFile:
    return IntelFile(
        collection="test",
        repo=".",
        repo_role="project",
        branch="main",
        commit_sha="commit",
        tree_sha="tree",
        source_path="src/main.py",
        repo_rel_path="src/main.py",
        abs_path=Path.cwd() / "src" / "main.py",
        git_blob_sha=git_blob_sha,
        file_sha256=file_sha256,
        size_bytes=123,
        language="python",
        file_role="source",
        content_class="source",
        is_generated=False,
        is_vendor=False,
        is_test=False,
        is_source=True,
        is_build=False,
        is_config=False,
        is_doc=False,
        skipped_reason=skipped_reason,
        metadata={},
    )


def record_fixture() -> IntelRecord:
    return IntelRecord(
        collection="test",
        source_path="src/main.py",
        language="python",
        file_role="source",
        content_class="source",
        record_type="code_chunk",
        record_id="src/main.py::chunk::000001-000002",
        parent_record_id=None,
        title="src/main.py:1-2",
        summary="python chunk",
        embedding_text="type: code_chunk\ncontent:\ndef main(): pass",
        display_content="# src/main.py:1-2\n\n```python\ndef main(): pass\n```",
        line_start=1,
        line_end=2,
        symbol="main",
        symbol_kind="function",
        metadata={"b": 2, "a": 1},
        embedding="[0.1,0.2]",
    )


class FakeConnection:
    def __init__(self) -> None:
        self.sql: str | None = None
        self.params: list[object] = []

    def execute(self, sql: str, params: list[object] | None = None) -> FakeConnection:
        self.sql = sql
        self.params = params or []
        return self


class FakeRows:
    def __init__(self, rows: list[db.DbRow]) -> None:
        self.rows = rows

    def fetchall(self) -> list[db.DbRow]:
        return self.rows

    def fetchone(self) -> db.DbRow | None:
        return self.rows[0] if self.rows else None


class FakeRowsConnection:
    def __init__(self, row_sets: list[list[db.DbRow]]) -> None:
        self.row_sets = row_sets
        self.sql: list[str] = []
        self.params: list[list[object]] = []

    def execute(self, sql: str, params: list[object] | None = None) -> FakeRows:
        self.sql.append(sql)
        self.params.append(params or [])
        return FakeRows(self.row_sets.pop(0))


class StorageContractTests(unittest.TestCase):
    def test_file_signature_prefers_content_hash_then_blob_then_metadata(self) -> None:
        self.assertEqual(file_signature(file_fixture(file_sha256="sha", git_blob_sha="blob")), "sha256:sha")
        self.assertEqual(file_signature(file_fixture(file_sha256=None, git_blob_sha="blob")), "blob:blob")
        self.assertEqual(
            file_signature(file_fixture(file_sha256=None, git_blob_sha=None, skipped_reason="binary_suffix")),
            "meta:123:binary_suffix",
        )

    def test_row_int_accepts_database_integer_strings_but_not_bools(self) -> None:
        self.assertEqual(row_int(cast("db.DbRow", {"count": 12}), "count"), 12)
        self.assertEqual(row_int(cast("db.DbRow", {"count": "12"}), "count"), 12)

        with self.assertRaises(TypeError):
            _ = row_int(cast("db.DbRow", {"count": True}), "count")
        with self.assertRaises(TypeError):
            _ = row_int(cast("db.DbRow", {"count": "12.5"}), "count")

    def test_snapshot_versions_compatible_matches_schema_parser_and_profile(self) -> None:
        compatible, old_profile_compatible, missing_metadata_compatible = snapshot_compatibility_with_generic_profile()

        self.assertTrue(compatible)
        self.assertFalse(old_profile_compatible)
        self.assertFalse(missing_metadata_compatible)

    def test_parser_failure_metadata_removes_promoted_columns(self) -> None:
        metadata = parser_failure_metadata({
            "source_path": "src/main.py",
            "language": "python",
            "parser": "python",
            "error": "bad syntax",
            "detail": "line 1",
        })

        self.assertEqual(metadata, {"detail": "line 1"})

    def test_insert_records_strips_nul_bytes_from_text_and_metadata(self) -> None:
        # NUL bytes leak in when a file with mixed text/binary slips past the
        # binary detector. PG rejects U+0000 in both text and jsonb columns.
        # Storage must scrub before insert so the rest of the batch still lands.
        fake = FakeConnection()
        context = RecordInsertContext(
            conn=cast("db.DbConnection", fake),
            snapshot=snapshot_fixture(),
            snapshot_id=7,
            file_ids={"src/main.py": 9},
            file_hashes={"src/main.py": "filesha"},
        )
        tainted = IntelRecord(
            collection="test",
            source_path="src/main.py",
            language="python",
            file_role="source",
            content_class="source",
            record_type="code_chunk",
            record_id="src/main.py::chunk::000001-000002",
            parent_record_id=None,
            title="ok\x00title",
            summary="ok summary",
            embedding_text="prefix\x00suffix",
            display_content="display\x00body",
            line_start=1,
            line_end=2,
            symbol="main",
            symbol_kind="function",
            metadata={"snippet": "code\x00here", "tags": ["a\x00b", "clean"], "n": 1},
            embedding=None,
        )

        _ = insert_records(context, [tainted])

        batch = cast("list[dict[str, object]]", json.loads(cast("str", fake.params[0])))
        row = batch[0]
        for key in ("title", "summary", "embedding_text", "display_content"):
            self.assertNotIn("\x00", cast("str", row[key]), msg=f"{key} kept NUL byte")
        metadata = cast("dict[str, object]", row["metadata"])
        self.assertEqual(metadata["snippet"], "codehere")
        self.assertEqual(metadata["tags"], ["ab", "clean"])
        self.assertEqual(metadata["n"], 1)

    def test_insert_records_serializes_database_write_contract(self) -> None:
        fake = FakeConnection()
        context = RecordInsertContext(
            conn=cast("db.DbConnection", fake),
            snapshot=snapshot_fixture(),
            snapshot_id=7,
            file_ids={"src/main.py": 9},
            file_hashes={"src/main.py": "filesha"},
        )

        inserted = insert_records(context, [record_fixture()])

        self.assertEqual(inserted, 1)
        self.assertEqual(len(fake.params), 1)
        batch = cast("list[dict[str, object]]", json.loads(cast("str", fake.params[0])))
        self.assertIsInstance(batch, list)
        self.assertEqual(len(batch), 1)
        row = batch[0]
        self.assertIsInstance(row, dict)
        self.assertNotIn("snapshot_id", row)
        self.assertNotIn("file_id", row)
        self.assertEqual(row["repo"], ".")
        self.assertEqual(row["file_sha256"], "filesha")
        self.assertEqual(row["source_path"], "src/main.py")
        self.assertEqual(row["metadata"], {"a": 1, "b": 2})
        self.assertEqual(row["embedding"], "[0.1,0.2]")
        self.assertIn("ON CONFLICT", fake.sql or "")
        self.assertIn("DO UPDATE SET", fake.sql or "")
        self.assertIn("updated_at = now()", fake.sql or "")
        self.assertIn("embedding = coalesce(EXCLUDED.embedding, project_code_intel_records.embedding)", fake.sql or "")

    def test_insert_records_conflict_update_preserves_embedding_contract_keys(self) -> None:
        # A re-parsed record carries no embedding contract keys. When the
        # upsert keeps the stored embedding, it must also keep those keys,
        # or the snapshot contract check fails on the next embedding run.
        fake = FakeConnection()
        context = RecordInsertContext(
            conn=cast("db.DbConnection", fake),
            snapshot=snapshot_fixture(),
            snapshot_id=7,
            file_ids={"src/main.py": 9},
            file_hashes={"src/main.py": "filesha"},
        )

        _ = insert_records(context, [record_fixture()])

        sql = fake.sql or ""
        self.assertIn("metadata = EXCLUDED.metadata ||", sql)
        self.assertIn("WHEN EXCLUDED.embedding IS NULL", sql)
        self.assertIn("project_code_intel_records.metadata->'embedding_backend'", sql)
        self.assertIn("project_code_intel_records.metadata->'embedding_model'", sql)
        self.assertIn("project_code_intel_records.metadata->'embedding_dimensions'", sql)

    def test_insert_records_maps_missing_file_hash_to_empty_string_sentinel(self) -> None:
        # Postgres unique constraints treat NULL as distinct from itself, so a
        # bare NULL file_sha256 (inventory read failures) would never dedupe
        # across ingests. '' is the NOT NULL sentinel instead.
        fake = FakeConnection()
        context = RecordInsertContext(
            conn=cast("db.DbConnection", fake),
            snapshot=snapshot_fixture(),
            snapshot_id=7,
            file_ids={"src/main.py": 9},
            file_hashes={"src/main.py": None},
        )

        _ = insert_records(context, [record_fixture()])

        batch = cast("list[dict[str, object]]", json.loads(cast("str", fake.params[0])))
        self.assertEqual(batch[0]["file_sha256"], "")

    def test_insert_records_deduplicates_same_batch_upsert_keys_in_sql(self) -> None:
        fake = FakeConnection()
        context = RecordInsertContext(
            conn=cast("db.DbConnection", fake),
            snapshot=snapshot_fixture(),
            snapshot_id=7,
            file_ids={"src/main.py": 9},
            file_hashes={"src/main.py": "filesha"},
        )
        first = record_fixture()
        duplicate = replace(
            first, summary="updated summary", display_content="updated display", metadata={"dupe": True}
        )

        inserted = insert_records(context, [first, duplicate])

        self.assertEqual(inserted, 2)
        self.assertIn("DISTINCT ON (", fake.sql or "")
        self.assertIn(
            "collection, repo, source_path, file_sha256, record_type, record_id, embedding_text_hash",
            fake.sql or "",
        )
        self.assertIn("input_order DESC", fake.sql or "")
        batch = cast("list[dict[str, object]]", json.loads(cast("str", fake.params[0])))
        self.assertEqual([row["input_order"] for row in batch], [0, 1])
        self.assertEqual(
            [
                (row["collection"], row["repo"], row["source_path"], row["record_type"], row["record_id"])
                for row in batch
            ],
            [
                ("test", ".", "src/main.py", "code_chunk", "src/main.py::chunk::000001-000002"),
                ("test", ".", "src/main.py", "code_chunk", "src/main.py::chunk::000001-000002"),
            ],
        )

    def test_insert_static_runs_persists_normalized_sarif_children(self) -> None:
        fake = FakeRowsConnection([
            [{"id": 101}],
            [],
            [{"id": 201}],
            [],
            [],
            [],
            [],
        ])
        snapshot = snapshot_fixture()
        skipped = StaticRun(
            repo="missing",
            sarif_path="missing.sarif",
            sarif_sha256="missing-sha",
            run_index=0,
            tool_name="missing-tool",
        )
        run = StaticRun(
            repo=".",
            sarif_path="report.sarif",
            sarif_sha256="report-sha",
            run_index=2,
            tool_name="ruff",
            tool_version="1.2.3",
            semantic_version="1.2.3",
            information_uri="https://example.invalid/ruff",
            automation_id="ci",
            metadata={"b": 2, "a": 1},
            rules=[
                StaticRule(
                    rule_id="F401",
                    name="unused-import",
                    short_description="unused import",
                    full_description="import is unused",
                    default_level="warning",
                    help_uri="https://example.invalid/F401",
                    properties={"z": 1, "a": 2},
                    metadata={"source": "sarif"},
                )
            ],
            findings=[
                StaticFinding(
                    finding_key="finding-1",
                    rule_id="F401",
                    rule_index=0,
                    level="warning",
                    kind="fail",
                    message="unused import",
                    baseline_state="new",
                    primary_source_path="src/main.py",
                    primary_uri="file:///repo/src/main.py",
                    line_start=3,
                    line_end=3,
                    column_start=1,
                    column_end=5,
                    fingerprints={"stable": "fingerprint"},
                    suppressions=[{"kind": "external"}],
                    properties={"precision": "high"},
                    raw_result={"ruleId": "F401"},
                    locations=[
                        StaticLocation(
                            ordinal=0,
                            location_kind="primary",
                            source_path="src/main.py",
                            uri="file:///repo/src/main.py",
                            message="unused import",
                            line_start=3,
                            line_end=3,
                            column_start=1,
                            column_end=5,
                            snippet="import os",
                            properties={"region": "primary"},
                        )
                    ],
                    code_flows=[
                        StaticCodeFlowStep(
                            flow_index=0,
                            thread_index=0,
                            step_index=0,
                            source_path="src/main.py",
                            uri="file:///repo/src/main.py",
                            message="import introduced",
                            line_start=3,
                            line_end=3,
                            column_start=1,
                            column_end=5,
                            importance="essential",
                            properties={"flow": "main"},
                        )
                    ],
                )
            ],
        )

        counts = insert_static_runs(
            cast("db.DbConnection", fake),
            snapshot_ids_by_repo={".": 7},
            snapshot_by_repo={".": snapshot},
            runs=[skipped, run],
        )

        self.assertEqual(
            counts,
            {
                "static_runs": 1,
                "static_rules": 1,
                "static_findings": 1,
                "static_locations": 1,
                "static_code_flow_steps": 1,
            },
        )
        self.assertEqual(len(fake.sql), 7)
        self.assertIn("INSERT INTO project_code_intel_static_runs", fake.sql[0])
        self.assertEqual(fake.params[0][0:5], [7, "test", ".", "commit", "report.sarif"])
        self.assertEqual(fake.params[0][12], '{"a":1,"b":2}')
        self.assertIn("INSERT INTO project_code_intel_static_rules", fake.sql[1])
        self.assertEqual(fake.params[1][0:4], [101, "test", ".", "F401"])
        self.assertEqual(fake.params[1][9], '{"a":2,"z":1}')
        self.assertIn("INSERT INTO project_code_intel_static_findings", fake.sql[2])
        self.assertEqual(fake.params[2][0:6], [101, 7, "test", ".", "commit", "finding-1"])
        self.assertEqual(fake.params[2][18], '{"stable":"fingerprint"}')
        self.assertEqual(fake.params[2][19], '[{"kind":"external"}]')
        self.assertEqual(fake.params[2][20], '{"precision":"high"}')
        self.assertEqual(fake.params[2][21], '{"ruleId":"F401"}')
        self.assertIn("DELETE FROM project_code_intel_static_locations", fake.sql[3])
        self.assertEqual(fake.params[3], [201])
        self.assertIn("DELETE FROM project_code_intel_static_code_flows", fake.sql[4])
        self.assertEqual(fake.params[4], [201])
        self.assertIn("INSERT INTO project_code_intel_static_locations", fake.sql[5])
        self.assertEqual(
            fake.params[5][0:6],
            [201, 0, "primary", "src/main.py", "file:///repo/src/main.py", "unused import"],
        )
        self.assertIn("INSERT INTO project_code_intel_static_code_flows", fake.sql[6])
        self.assertEqual(
            fake.params[6][0:7],
            [201, 0, 0, 0, "src/main.py", "file:///repo/src/main.py", "import introduced"],
        )

    def test_copy_unchanged_rows_returns_zero_for_empty_or_self_copy_inputs(self) -> None:
        fake = FakeRowsConnection([])
        snapshot = snapshot_fixture()

        self.assertEqual(
            copy_unchanged_parser_failures(
                cast("db.DbConnection", fake),
                previous_snapshot_id=None,
                snapshot=snapshot,
                snapshot_id=7,
                unchanged_paths={"src/main.py"},
            ),
            (0, []),
        )
        self.assertEqual(
            copy_unchanged_parser_failures(
                cast("db.DbConnection", fake),
                previous_snapshot_id=7,
                snapshot=snapshot,
                snapshot_id=7,
                unchanged_paths={"src/main.py"},
            ),
            (0, []),
        )
        self.assertEqual(
            copy_unchanged_edges(
                cast("db.DbConnection", fake),
                previous_snapshot_id=6,
                snapshot=snapshot,
                snapshot_id=7,
                unchanged_paths=set(),
            ),
            0,
        )
        self.assertEqual(fake.sql, [])

    def test_copy_unchanged_rows_uses_sorted_paths_and_snapshot_metadata(self) -> None:
        fake = FakeRowsConnection([
            [{"source_path": "src/a.py"}, {"source_path": "src/b.py"}],
            [{"count": 4}],
        ])
        snapshot = snapshot_fixture()
        paths = {"src/b.py", "src/a.py"}

        parser_failure_count, parser_failure_paths = copy_unchanged_parser_failures(
            cast("db.DbConnection", fake),
            previous_snapshot_id=6,
            snapshot=snapshot,
            snapshot_id=7,
            unchanged_paths=paths,
        )
        edges = copy_unchanged_edges(
            cast("db.DbConnection", fake),
            previous_snapshot_id=6,
            snapshot=snapshot,
            snapshot_id=7,
            unchanged_paths=paths,
        )

        self.assertEqual(parser_failure_count, 2)
        self.assertEqual(parser_failure_paths, ["src/a.py", "src/b.py"])
        self.assertEqual(edges, 4)
        self.assertIn("project_code_intel_parser_failures", fake.sql[0])
        self.assertEqual(fake.params[0], [7, "test", ".", "commit", 6, ["src/a.py", "src/b.py"]])
        self.assertIn("project_code_intel_edges", fake.sql[1])
        self.assertEqual(fake.params[1], [7, "test", ".", "commit", 6, ["src/a.py", "src/b.py"]])

    def test_pre_resolve_edge_targets_uses_same_file_then_same_directory_priority(self) -> None:
        fake = FakeRowsConnection([
            cast(
                "list[db.DbRow]",
                [
                    {
                        "symbol": "target",
                        "record_id": "src/other.py::function::target::000001",
                        "source_path": "src/other.py",
                    },
                    {
                        "symbol": "target",
                        "record_id": "src/main.py::function::target::000001",
                        "source_path": "src/main.py",
                    },
                ],
            )
        ])
        same_file = IntelEdge(
            source_record_id="src/main.py::function::caller::000010",
            edge_type="call_candidate",
            target_symbol="target",
            source_path="src/main.py",
        )
        same_dir = IntelEdge(
            source_record_id="src/caller.py::function::caller::000010",
            edge_type="call_candidate",
            target_symbol="target",
            source_path="src/caller.py",
        )
        missing = IntelEdge(
            source_record_id="src/missing.py::function::caller::000010",
            edge_type="call_candidate",
            target_symbol="missing",
            source_path="src/missing.py",
        )

        resolved = pre_resolve_edge_targets(
            cast("db.DbConnection", fake), SnapshotScope("test", ".", 7), [same_file, same_dir, missing]
        )

        self.assertEqual(resolved, 2)
        self.assertEqual(same_file.target_record_id, "src/main.py::function::target::000001")
        self.assertEqual(same_file.target_path, "src/main.py")
        self.assertEqual(same_dir.target_record_id, "src/main.py::function::target::000001")
        self.assertEqual(same_dir.target_path, "src/main.py")
        self.assertIsNone(missing.target_record_id)

    def test_pre_resolve_edge_targets_binds_bare_name_to_unambiguous_qualified_definition(self) -> None:
        """A call site records `start`; the definition is stored `Configurator.start`."""
        fake = FakeRowsConnection([
            cast(
                "list[db.DbRow]",
                [
                    {
                        "symbol": "Configurator.start",
                        "record_id": "src/configurator.py::function::Configurator.start::000001",
                        "source_path": "src/configurator.py",
                    },
                ],
            )
        ])
        edge = IntelEdge(
            source_record_id="src/main.py::function::caller::000010",
            edge_type="call_candidate",
            target_symbol="start",
            source_path="src/main.py",
        )

        resolved = pre_resolve_edge_targets(cast("db.DbConnection", fake), SnapshotScope("test", ".", 7), [edge])

        self.assertEqual(resolved, 1)
        self.assertEqual(edge.target_record_id, "src/configurator.py::function::Configurator.start::000001")
        self.assertEqual(edge.target_path, "src/configurator.py")

    def test_pre_resolve_edge_targets_leaves_ambiguous_qualified_suffix_unresolved(self) -> None:
        """Two qualified names end in the bare callee name: binding either one is a guess."""
        fake = FakeRowsConnection([
            cast(
                "list[db.DbRow]",
                [
                    {
                        "symbol": "Bar.start",
                        "record_id": "src/bar.py::function::Bar.start::000001",
                        "source_path": "src/bar.py",
                    },
                    {
                        "symbol": "Foo.start",
                        "record_id": "src/foo.py::function::Foo.start::000001",
                        "source_path": "src/foo.py",
                    },
                ],
            )
        ])
        edge = IntelEdge(
            source_record_id="src/main.py::function::caller::000010",
            edge_type="call_candidate",
            target_symbol="start",
            source_path="src/main.py",
        )

        resolved = pre_resolve_edge_targets(cast("db.DbConnection", fake), SnapshotScope("test", ".", 7), [edge])

        self.assertEqual(resolved, 0)
        self.assertIsNone(edge.target_record_id)

    def test_pre_resolve_edge_targets_batches_and_reports_progress(self) -> None:
        fake = FakeRowsConnection([
            cast(
                "list[db.DbRow]",
                [
                    {"symbol": "alpha", "record_id": "src/a.py::function::alpha::000001", "source_path": "src/a.py"},
                    {"symbol": "beta", "record_id": "src/b.py::function::beta::000001", "source_path": "src/b.py"},
                ],
            ),
            cast(
                "list[db.DbRow]",
                [{"symbol": "gamma", "record_id": "src/c.py::function::gamma::000001", "source_path": "src/c.py"}],
            ),
        ])
        edges = [
            IntelEdge(
                source_record_id="src/main.py::function::caller::000010",
                edge_type="call_candidate",
                target_symbol="alpha",
            ),
            IntelEdge(
                source_record_id="src/main.py::function::caller::000011",
                edge_type="call_candidate",
                target_symbol="beta",
            ),
            IntelEdge(
                source_record_id="src/main.py::function::caller::000012",
                edge_type="call_candidate",
                target_symbol="gamma",
            ),
        ]
        progress: list[int] = []

        self.assertEqual(pre_resolvable_edge_count(edges), 3)
        resolved = pre_resolve_edge_targets(
            cast("db.DbConnection", fake),
            SnapshotScope("test", ".", 7),
            edges,
            batch_size=2,
            progress_fn=progress.append,
        )

        self.assertEqual(resolved, 3)
        self.assertEqual(pre_resolvable_edge_count(edges), 0)
        self.assertEqual(progress, [2, 1])
        self.assertEqual(fake.params[0], [7, "test", ".", ["alpha", "beta"], ["alpha", "beta"]])
        self.assertEqual(fake.params[1], [7, "test", ".", ["gamma"], ["gamma"]])

    def test_pre_resolve_edge_targets_skips_non_resolvable_member_calls(self) -> None:
        fake = FakeRowsConnection([
            cast(
                "list[db.DbRow]",
                [
                    {
                        "symbol": "run",
                        "record_id": "src/bench.ts::function::run::000001",
                        "source_path": "src/bench.ts",
                    },
                ],
            )
        ])
        member_call = IntelEdge(
            source_record_id="src/schema.ts::function::parse::000010",
            edge_type="call_candidate",
            target_symbol="run",
            source_path="src/schema.ts",
            metadata={"call_kind": "member_call", "target_resolvable": False},
        )

        resolved = pre_resolve_edge_targets(cast("db.DbConnection", fake), SnapshotScope("test", ".", 7), [member_call])

        self.assertEqual(resolved, 0)
        self.assertIsNone(member_call.target_record_id)
        self.assertEqual(fake.sql, [])

    def test_pre_resolve_edge_targets_prefers_value_symbols_over_type_declarations(self) -> None:
        fake = FakeRowsConnection([
            cast(
                "list[db.DbRow]",
                [
                    {
                        "symbol": "$constructor",
                        "record_id": "src/core.ts::interface::$constructor::000001",
                        "source_path": "src/core.ts",
                        "symbol_kind": "interface",
                    },
                    {
                        "symbol": "$constructor",
                        "record_id": "src/core.ts::function::$constructor::000020",
                        "source_path": "src/core.ts",
                        "symbol_kind": "function",
                    },
                ],
            )
        ])
        call = IntelEdge(
            source_record_id="src/schema.ts::constant::$ZodString::000010",
            edge_type="call_candidate",
            target_symbol="$constructor",
            source_path="src/schema.ts",
            metadata={"call_kind": "member_call", "member": "$constructor", "qualifier": "core"},
        )

        resolved = pre_resolve_edge_targets(cast("db.DbConnection", fake), SnapshotScope("test", ".", 7), [call])

        self.assertEqual(resolved, 1)
        self.assertEqual(call.target_record_id, "src/core.ts::function::$constructor::000020")

    def test_resolve_edge_targets_batches_candidates_and_reports_progress(self) -> None:
        fake = FakeRowsConnection([
            cast(
                "list[db.DbRow]",
                [{"candidate_count": 3, "last_target_symbol": "target", "last_edge_id": 42, "updated_count": 2}],
            ),
            cast(
                "list[db.DbRow]",
                [{"candidate_count": 0, "last_target_symbol": None, "last_edge_id": None, "updated_count": 0}],
            ),
        ])
        progress: list[int] = []

        resolved = resolve_edge_targets(
            cast("db.DbConnection", fake), SnapshotScope("test", ".", 7), batch_size=3, progress_fn=progress.append
        )

        self.assertEqual(resolved, 2)
        self.assertEqual(progress, [3])
        self.assertEqual(fake.params[0], [7, None, None, None, 0, 3, 7, "test", ".", 7, "test", "."])
        self.assertEqual(fake.params[1], [7, "target", "target", "target", 42, 3, 7, "test", ".", 7, "test", "."])
        self.assertIn("candidate_edges AS MATERIALIZED", fake.sql[0])
        self.assertIn("IS NOT DISTINCT FROM", fake.sql[0])
        self.assertIn("metadata->>'target_resolvable'", fake.sql[0])
        self.assertIn("suffix_choices AS MATERIALIZED", fake.sql[0])
        self.assertIn("HAVING count(DISTINCT r.symbol) = 1", fake.sql[0])
        self.assertIn(
            "r.symbol_kind IN ('function', 'method', 'constant', 'class', 'enum', 'shell_function')", fake.sql[0]
        )


class QualifiedCallSiteResolutionTests(unittest.TestCase):
    """Call sites that name their receiver bind exactly; a module-qualified one falls back."""

    def test_binds_qualified_call_site_despite_ambiguous_member(self) -> None:
        """`DatabaseSettings.from_env` names its receiver, so a second `from_env` is not a guess."""
        fake = FakeRowsConnection([
            cast(
                "list[db.DbRow]",
                [
                    {
                        "symbol": "DatabaseSettings.from_env",
                        "record_id": "src/db.py::function::DatabaseSettings.from_env::000001",
                        "source_path": "src/db.py",
                    },
                    {
                        "symbol": "OtherSettings.from_env",
                        "record_id": "src/other.py::function::OtherSettings.from_env::000001",
                        "source_path": "src/other.py",
                    },
                ],
            )
        ])
        edge = IntelEdge(
            source_record_id="src/main.py::function::build::000010",
            edge_type="call_candidate",
            target_symbol="DatabaseSettings.from_env",
            source_path="src/main.py",
        )

        resolved = pre_resolve_edge_targets(cast("db.DbConnection", fake), SnapshotScope("test", ".", 7), [edge])

        self.assertEqual(resolved, 1)
        self.assertEqual(edge.target_record_id, "src/db.py::function::DatabaseSettings.from_env::000001")

    def test_falls_back_to_member_when_receiver_is_a_module(self) -> None:
        """`evidence.collect_evidence` is a module call: no such qualified definition, so the member binds."""
        fake = FakeRowsConnection([
            cast(
                "list[db.DbRow]",
                [
                    {
                        "symbol": "collect_evidence",
                        "record_id": "src/evidence.py::function::collect_evidence::000001",
                        "source_path": "src/evidence.py",
                    },
                ],
            )
        ])
        edge = IntelEdge(
            source_record_id="src/mcp/tools.py::function::search::000010",
            edge_type="call_candidate",
            target_symbol="evidence.collect_evidence",
            source_path="src/mcp/tools.py",
        )

        resolved = pre_resolve_edge_targets(cast("db.DbConnection", fake), SnapshotScope("test", ".", 7), [edge])

        self.assertEqual(resolved, 1)
        self.assertEqual(edge.target_record_id, "src/evidence.py::function::collect_evidence::000001")


class LockIngestBranchTests(unittest.TestCase):
    def test_takes_a_transaction_scoped_advisory_lock_keyed_on_collection_repo_branch(self) -> None:
        fake = FakeConnection()

        lock_ingest_branch(cast("db.DbConnection", fake), "test", ".", "main")

        self.assertIn("pg_advisory_xact_lock", fake.sql or "")
        self.assertIn("hashtextextended", fake.sql or "")
        self.assertEqual(fake.params, ["test\x1f.\x1fmain"])

    def test_null_branch_folds_to_a_stable_key_instead_of_the_string_none(self) -> None:
        fake = FakeConnection()

        lock_ingest_branch(cast("db.DbConnection", fake), "test", ".", None)

        self.assertEqual(fake.params, ["test\x1f.\x1f"])


class DeleteOrphanedRecordsTests(unittest.TestCase):
    def test_deletes_records_with_no_matching_files_membership_row(self) -> None:
        fake = FakeRowsConnection([[{"id": 1}, {"id": 2}]])

        deleted = delete_orphaned_records(cast("db.DbConnection", fake), "test", ".")

        self.assertEqual(deleted, 2)
        self.assertIn("DELETE FROM project_code_intel_records", fake.sql[0])
        self.assertIn("NOT EXISTS", fake.sql[0])
        self.assertIn("project_code_intel_files", fake.sql[0])
        self.assertEqual(fake.params[0], ["test", "."])


class ReplaceReposTests(unittest.TestCase):
    def test_deletes_snapshots_then_sweeps_orphaned_records_per_repo(self) -> None:
        fake = FakeRowsConnection([[], [{"id": 1}], [], []])

        replace_repos(cast("db.DbConnection", fake), "test", ["repo-a", "repo-b"])

        # One DELETE-snapshots + one orphan-sweep call per repo, interleaved
        # per repo rather than batched, so a partial failure only leaves one
        # repo's records momentarily orphaned.
        self.assertEqual(len(fake.sql), 4)
        self.assertIn("DELETE FROM project_code_intel_snapshots", fake.sql[0])
        self.assertEqual(fake.params[0], ["test", "repo-a"])
        self.assertIn("DELETE FROM project_code_intel_records", fake.sql[1])
        self.assertEqual(fake.params[1], ["test", "repo-a"])
        self.assertIn("DELETE FROM project_code_intel_snapshots", fake.sql[2])
        self.assertEqual(fake.params[2], ["test", "repo-b"])
        self.assertIn("DELETE FROM project_code_intel_records", fake.sql[3])
        self.assertEqual(fake.params[3], ["test", "repo-b"])


if __name__ == "__main__":
    _ = unittest.main()
