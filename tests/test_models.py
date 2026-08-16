"""Unit tests for `project_code_intelligence.models`.

Most types in `models.py` are dataclasses. These tests pin the defaults and
mutable-default isolation contracts the rest of the pipeline relies on.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path

from project_code_intelligence.models import (
    CHUNKER_VERSION,
    DEFAULT_EMBED_RECORD_TYPES,
    PARSER_VERSION,
    SCHEMA_VERSION,
    SOURCE_LANGUAGES,
    IntelEdge,
    IntelFile,
    IntelRecord,
    PreviousFileState,
    SarifPathResolution,
    Snapshot,
    StaticFinding,
    StaticLocation,
    StaticRule,
    StaticRun,
)


class SchemaConstantsTests(unittest.TestCase):
    def test_schema_version_string_pinned(self) -> None:
        self.assertEqual(SCHEMA_VERSION, "code-intel-schema-v2")

    def test_chunker_version_string_pinned(self) -> None:
        self.assertEqual(CHUNKER_VERSION, "code-intel-v1")

    def test_parser_version_is_non_empty(self) -> None:
        # PARSER_VERSION is bumped on parser changes; the exact value is volatile,
        # but it must always be a non-empty string so DB upserts have something to compare.
        self.assertIsInstance(PARSER_VERSION, str)
        self.assertNotEqual(PARSER_VERSION, "")

    def test_default_embed_record_types_includes_core_types(self) -> None:
        for record_type in ("code_chunk", "config_symbol", "doc_section", "static_finding"):
            self.assertIn(record_type, DEFAULT_EMBED_RECORD_TYPES)

    def test_source_languages_is_a_set_of_strings(self) -> None:
        self.assertIsInstance(SOURCE_LANGUAGES, set)
        for language in SOURCE_LANGUAGES:
            self.assertIsInstance(language, str)


def _intel_file() -> IntelFile:
    return IntelFile(
        collection="c",
        repo="r",
        repo_role="project",
        branch="main",
        commit_sha="0" * 40,
        tree_sha="1" * 40,
        source_path="src/foo.py",
        repo_rel_path="src/foo.py",
        abs_path=Path(tempfile.gettempdir()) / "foo.py",
        git_blob_sha=None,
        file_sha256=None,
        size_bytes=0,
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
        skipped_reason=None,
    )


class IntelFileTests(unittest.TestCase):
    def test_intel_file_has_per_instance_metadata_default(self) -> None:
        # Mutable defaults via field(default_factory=dict) must produce a fresh
        # dict per instance; otherwise mutating one file's metadata would leak
        # into other files. This is the most common dataclass footgun.
        file_a = _intel_file()
        file_b = _intel_file()
        file_a.metadata["touched"] = True
        self.assertNotIn("touched", file_b.metadata)

    def test_intel_file_is_untracked_and_indexed_dirty_default_false(self) -> None:
        file = _intel_file()
        self.assertFalse(file.is_untracked)
        self.assertFalse(file.indexed_dirty)

    def test_intel_file_is_a_dataclass_with_expected_identity_fields(self) -> None:
        self.assertTrue(is_dataclass(IntelFile))
        field_names = {f.name for f in fields(IntelFile)}
        # Identity fields callers depend on for joins must be present.
        for identity_field in ("collection", "repo", "commit_sha", "source_path", "language"):
            self.assertIn(identity_field, field_names)


class PreviousFileStateTests(unittest.TestCase):
    def test_previous_file_state_is_frozen_to_protect_change_detection(self) -> None:
        # PreviousFileState backs change-detection comparisons; mutability would
        # let buggy callers retroactively shift the "previous" baseline.
        previous = PreviousFileState(
            source_path="a",
            git_blob_sha=None,
            file_sha256=None,
            size_bytes=0,
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
            skipped_reason=None,
        )
        field_name = "size_bytes"
        with self.assertRaises(FrozenInstanceError):
            setattr(previous, field_name, 1)


class IntelRecordTests(unittest.TestCase):
    def test_intel_record_defaults_use_pinned_parser_and_chunker_versions(self) -> None:
        record = IntelRecord(
            collection="c",
            source_path="a.py",
            language="python",
            file_role="source",
            content_class="source",
            record_type="code_chunk",
            record_id="rid",
            title="t",
            summary="s",
            embedding_text="e",
            display_content="d",
        )
        # Pinned defaults flow into every record so downstream stores don't
        # have to retrieve a version string explicitly.
        self.assertEqual(record.parser_version, PARSER_VERSION)
        self.assertEqual(record.chunker_version, CHUNKER_VERSION)
        # Confidence defaults to the high bucket; callers downgrade explicitly.
        self.assertEqual(record.confidence_kind, "high_confidence_fact")
        # Metadata default is a fresh dict, not a shared module-level singleton.
        self.assertEqual(record.metadata, {})

    def test_intel_record_metadata_default_is_per_instance(self) -> None:
        record_a = IntelRecord(
            collection="c",
            source_path="a.py",
            language="python",
            file_role="source",
            content_class="source",
            record_type="code_chunk",
            record_id="a",
            title="t",
            summary="s",
            embedding_text="e",
            display_content="d",
        )
        record_b = IntelRecord(
            collection="c",
            source_path="a.py",
            language="python",
            file_role="source",
            content_class="source",
            record_type="code_chunk",
            record_id="b",
            title="t",
            summary="s",
            embedding_text="e",
            display_content="d",
        )
        record_a.metadata["a_key"] = "a_value"
        self.assertNotIn("a_key", record_b.metadata)

    def test_intel_record_optional_diagnostic_fields_default_none(self) -> None:
        record = IntelRecord(
            collection="c",
            source_path="a.py",
            language="python",
            file_role="source",
            content_class="source",
            record_type="code_chunk",
            record_id="r",
            title="t",
            summary="s",
            embedding_text="e",
            display_content="d",
        )
        # Optional diagnostic columns must round-trip as NULL in SQL when the
        # parser didn't supply them. Defaulting these to None enforces that.
        # Explicit attribute access keeps each Optional checked under its real
        # type rather than the `Any` getattr() would produce.
        self.assertIsNone(record.line_start)
        self.assertIsNone(record.line_end)
        self.assertIsNone(record.symbol)
        self.assertIsNone(record.symbol_kind)
        self.assertIsNone(record.parent_record_id)
        self.assertIsNone(record.confidence)
        self.assertIsNone(record.tool)
        self.assertIsNone(record.rule_id)
        self.assertIsNone(record.severity)
        self.assertIsNone(record.analyzer)
        self.assertIsNone(record.analyzer_version)
        self.assertIsNone(record.parser)
        self.assertIsNone(record.embedding)


class IntelEdgeTests(unittest.TestCase):
    def test_intel_edge_defaults_to_approximate_fact_confidence(self) -> None:
        edge = IntelEdge(source_record_id="src", edge_type="calls")
        # Edges originate from heuristics by default; explicit upgrades happen
        # when a parser knows the relationship is definitive.
        self.assertEqual(edge.confidence_kind, "approximate_fact")
        self.assertIsNone(edge.target_record_id)
        self.assertIsNone(edge.source_symbol)
        self.assertEqual(edge.metadata, {})

    def test_intel_edge_metadata_default_is_per_instance(self) -> None:
        edge_a = IntelEdge(source_record_id="a", edge_type="calls")
        edge_b = IntelEdge(source_record_id="b", edge_type="calls")
        edge_a.metadata["call_kind"] = "direct"
        self.assertNotIn("call_kind", edge_b.metadata)


class SnapshotTests(unittest.TestCase):
    def test_snapshot_metadata_default_is_per_instance(self) -> None:
        snap_a = Snapshot(
            collection="c",
            repo="r",
            repo_role="project",
            branch=None,
            commit_sha="0" * 40,
            tree_sha="1" * 40,
            dirty=False,
        )
        snap_b = Snapshot(
            collection="c",
            repo="r",
            repo_role="project",
            branch=None,
            commit_sha="0" * 40,
            tree_sha="1" * 40,
            dirty=False,
        )
        snap_a.metadata["ingest_tool"] = "pci"
        self.assertNotIn("ingest_tool", snap_b.metadata)


class StaticFindingTests(unittest.TestCase):
    def test_static_finding_optional_collections_default_to_empty(self) -> None:
        finding = StaticFinding(finding_key="k", rule_id="R1", message="msg")
        self.assertEqual(finding.locations, [])
        self.assertEqual(finding.code_flows, [])
        self.assertEqual(finding.fingerprints, {})
        self.assertEqual(finding.suppressions, [])
        self.assertEqual(finding.properties, {})
        self.assertEqual(finding.raw_result, {})

    def test_static_finding_default_lists_are_per_instance(self) -> None:
        a = StaticFinding(finding_key="a", rule_id="R", message="m")
        b = StaticFinding(finding_key="b", rule_id="R", message="m")
        a.locations.append(StaticLocation(ordinal=1, location_kind="primary", source_path="x", uri=None))
        self.assertEqual(b.locations, [])

    def test_static_run_default_collections_are_per_instance(self) -> None:
        a = StaticRun(repo="r", sarif_path="a.sarif", sarif_sha256="aa", run_index=0, tool_name="t")
        b = StaticRun(repo="r", sarif_path="b.sarif", sarif_sha256="bb", run_index=0, tool_name="t")
        a.rules.append(StaticRule(rule_id="X"))
        self.assertEqual(b.rules, [])


class SarifPathResolutionTests(unittest.TestCase):
    def test_sarif_path_resolution_is_frozen(self) -> None:
        resolution = SarifPathResolution(source_path="src/a.py", repo="r", path_mapping="absolute")
        field_name = "path_mapping"
        with self.assertRaises(FrozenInstanceError):
            setattr(resolution, field_name, "relative")


if __name__ == "__main__":
    _ = unittest.main()
