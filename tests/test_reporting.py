from __future__ import annotations

import unittest
from pathlib import Path
from typing import cast

from typing_extensions import override

from project_code_intelligence import profile_context
from project_code_intelligence.code_profiles.base import GenericProfile
from project_code_intelligence.models import IntelEdge, IntelFile, IntelRecord, RepoIngest, Snapshot
from project_code_intelligence.reporting import report_ingests


class ReportProfile(GenericProfile):
    name = "report-test"
    version = "v2"

    @override
    def report_metadata_keys(self) -> list[str]:
        return ["target", "missing"]


def snapshot_fixture() -> Snapshot:
    return Snapshot(
        collection="demo",
        repo="repo-a",
        repo_role="project",
        branch="main",
        commit_sha="commit",
        tree_sha="tree",
        dirty=False,
    )


def file_fixture(source_path: str, *, skipped_reason: str | None = None) -> IntelFile:
    return IntelFile(
        collection="demo",
        repo="repo-a",
        repo_role="project",
        branch="main",
        commit_sha="commit",
        tree_sha="tree",
        source_path=source_path,
        repo_rel_path=source_path,
        abs_path=Path.cwd() / source_path,
        git_blob_sha="blob",
        file_sha256="file-sha",
        size_bytes=12,
        language="python" if source_path.endswith(".py") else "text",
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
    )


def record_fixture(record_id: str, *, confidence_kind: str = "high_confidence_fact") -> IntelRecord:
    return IntelRecord(
        collection="demo",
        source_path="src/main.py",
        language="python",
        file_role="source",
        content_class="source",
        record_type="code_chunk",
        record_id=record_id,
        title="main",
        summary="python chunk",
        embedding_text="def main(): pass",
        display_content="def main(): pass",
        confidence_kind=confidence_kind,
        metadata={"target": "main", "missing": ""},
    )


class ReportingTests(unittest.TestCase):
    def test_report_ingests_summarizes_counts_and_examples(self) -> None:
        previous_profile = profile_context.active_profile
        try:
            profile_context.set_active_profile(ReportProfile())
            ingest = RepoIngest(
                snapshot=snapshot_fixture(),
                files=[
                    file_fixture("src/main.py"),
                    file_fixture("README.txt", skipped_reason="binary_suffix"),
                ],
                records=[
                    record_fixture("src/main.py::chunk::000001-000001"),
                    record_fixture("src/main.py::chunk::000002-000002", confidence_kind="heuristic_candidate"),
                ],
                edges=[
                    IntelEdge(
                        source_record_id="src/main.py::chunk::000001-000001",
                        edge_type="references",
                        target_record_id="src/main.py::chunk::000002-000002",
                    )
                ],
                parser_failures=[{"source_path": "broken.py"}],
                mode="incremental",
                previous_snapshot_id=41,
                changed_paths={"src/main.py"},
                unchanged_paths={"README.txt"},
                deleted_paths={"old.py"},
            )

            report = report_ingests([ingest], embeddings=True)
        finally:
            profile_context.set_active_profile(previous_profile)

        self.assertEqual(report["profile"], "report-test")
        self.assertEqual(report["profile_version"], "v2")
        self.assertEqual(report["collections"], ["demo"])
        self.assertEqual(report["modes"], {"incremental": 1})
        self.assertEqual(report["repos"], ["repo-a"])
        self.assertEqual(report["files"], 2)
        self.assertEqual(report["changed_files"], 1)
        self.assertEqual(report["unchanged_files"], 1)
        self.assertEqual(report["deleted_files"], 1)
        self.assertEqual(report["parseable_files"], 1)
        self.assertEqual(report["parsed_files"], 1)
        self.assertEqual(report["skipped_files"], 1)
        self.assertEqual(report["skipped_reasons"], {"binary_suffix": 1})
        self.assertEqual(report["records"], 2)
        self.assertEqual(report["edges"], 1)
        self.assertEqual(report["parser_failures"], 1)
        self.assertEqual(report["records_by_type"], {"code_chunk": 2})
        self.assertEqual(report["files_by_language"], {"python": 1, "text": 1})
        self.assertEqual(report["confidence_by_kind"], {"heuristic_candidate": 1, "high_confidence_fact": 1})
        self.assertTrue(report["embeddings"])

        snapshots = cast("list[dict[str, object]]", report["snapshots"])
        self.assertEqual(snapshots[0]["previous_snapshot_id"], 41)
        examples = cast("list[dict[str, object]]", report["examples"])
        self.assertEqual(examples[0]["metadata"], {"target": "main"})


if __name__ == "__main__":
    _ = unittest.main()
