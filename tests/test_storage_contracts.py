from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, cast

from project_code_intelligence import db, profile_context
from project_code_intelligence.code_profiles.base import GenericProfile
from project_code_intelligence.models import IntelFile, IntelRecord, JsonObject, Snapshot
from project_code_intelligence.storage import (
    RecordInsertContext,
    file_signature,
    insert_records,
    parser_failure_metadata,
    row_int,
    snapshot_versions_compatible,
)

if TYPE_CHECKING:
    from types import TracebackType


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
            "schema_version": "code-intel-schema-v2",
            "chunker_version": "code-intel-v1",
            "parser_version": "stdlib-heuristic-v3",
            "profile_name": "generic",
            "profile_version": "v1",
        },
    )


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


class FakeCursor:
    def __init__(self) -> None:
        self.sql: str | None = None
        self.params: list[list[object]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback

    def executemany(self, sql: str, params: list[list[object]]) -> None:
        self.sql = sql
        self.params = params


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_obj = FakeCursor()

    def cursor(self) -> FakeCursor:
        return self.cursor_obj


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
        previous_profile = profile_context.active_profile
        try:
            profile_context.set_active_profile(GenericProfile())
            self.assertTrue(snapshot_versions_compatible(snapshot_fixture().metadata))
            incompatible: JsonObject = dict(snapshot_fixture().metadata)
            incompatible["profile_version"] = "old"
            self.assertFalse(snapshot_versions_compatible(incompatible))
            self.assertFalse(snapshot_versions_compatible(None))
        finally:
            profile_context.set_active_profile(previous_profile)

    def test_parser_failure_metadata_removes_promoted_columns(self) -> None:
        metadata = parser_failure_metadata({
            "source_path": "src/main.py",
            "language": "python",
            "parser": "python",
            "error": "bad syntax",
            "detail": "line 1",
        })

        self.assertEqual(metadata, {"detail": "line 1"})

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
        self.assertEqual(len(fake.cursor_obj.params), 1)
        payload = fake.cursor_obj.params[0]
        metadata_value = cast("object", json.loads(cast("str", payload[36])))
        if not isinstance(metadata_value, dict):
            self.fail("record metadata payload should decode to an object")
        self.assertEqual(payload[0], 7)
        self.assertEqual(payload[1], 9)
        self.assertEqual(payload[8], "src/main.py")
        self.assertEqual(metadata_value, {"a": 1, "b": 2})
        self.assertEqual(payload[37], "[0.1,0.2]")
        self.assertIn("ON CONFLICT", fake.cursor_obj.sql or "")


if __name__ == "__main__":
    _ = unittest.main()
