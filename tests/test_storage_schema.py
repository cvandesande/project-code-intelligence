"""Unit tests for `project_code_intelligence.storage.schema`.

Covers the DB-bound helpers (snapshot lookup, signature reconstruction,
state hydration, migration recording) using the same QueuedConnection /
FakeCursor pattern as tests/test_mcp_contracts.py, plus the pure
`previous_file_state_signature` mirror of `file_signature`.
"""

from __future__ import annotations

import unittest
from typing import cast
from unittest.mock import patch

from project_code_intelligence import db as pci_db
from project_code_intelligence.models import SCHEMA_VERSION, PreviousFileState
from project_code_intelligence.storage import schema as storage_schema


class FakeCursor:
    """Mirror of the FakeCursor used in tests/test_mcp_contracts.py."""

    def __init__(self, *, one: object | None = None, many: list[object] | None = None) -> None:
        self.one = one
        self.many = list(many or [])

    def fetchone(self) -> object | None:
        return self.one

    def fetchall(self) -> list[object]:
        return self.many


class QueuedConnection:
    """Mirror of the QueuedConnection used in tests/test_mcp_contracts.py.

    Pops cursors in submission order and records the SQL/params it received,
    which lets each test assert the right query ran with the right
    parameters.
    """

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


def _make_previous_state(
    *,
    git_blob_sha: str | None,
    file_sha256: str | None,
    size_bytes: int = 0,
    skipped_reason: str | None = None,
) -> PreviousFileState:
    """Build a PreviousFileState with only the fields the signature cares about."""

    return PreviousFileState(
        source_path="src/example.py",
        git_blob_sha=git_blob_sha,
        file_sha256=file_sha256,
        size_bytes=size_bytes,
        language="python",
        file_role="source",
        content_class="code",
        is_generated=False,
        is_vendor=False,
        is_test=False,
        is_source=True,
        is_build=False,
        is_config=False,
        is_doc=False,
        skipped_reason=skipped_reason,
    )


class PreviousFileStateSignatureTests(unittest.TestCase):
    """Pure helper, mirrors `file_signature` over `PreviousFileState`."""

    def test_sha256_takes_precedence_over_blob_and_meta(self) -> None:
        # When file_sha256 is set, blob and meta should be ignored.
        state = _make_previous_state(
            git_blob_sha="blob-abc",
            file_sha256="abc123",
            size_bytes=42,
            skipped_reason="too_large",
        )
        self.assertEqual(storage_schema.previous_file_state_signature(state), "sha256:abc123")

    def test_blob_sha_used_when_sha256_is_absent(self) -> None:
        state = _make_previous_state(git_blob_sha="blob-deadbeef", file_sha256=None, size_bytes=7)
        self.assertEqual(storage_schema.previous_file_state_signature(state), "blob:blob-deadbeef")

    def test_falls_back_to_size_and_skipped_reason_meta(self) -> None:
        # Neither hash present → composite meta key.
        with_reason = _make_previous_state(
            git_blob_sha=None,
            file_sha256=None,
            size_bytes=99,
            skipped_reason="binary",
        )
        self.assertEqual(
            storage_schema.previous_file_state_signature(with_reason),
            "meta:99:binary",
        )

        # Missing skipped_reason should fold to the empty-string tail.
        no_reason = _make_previous_state(
            git_blob_sha=None,
            file_sha256=None,
            size_bytes=99,
            skipped_reason=None,
        )
        self.assertEqual(
            storage_schema.previous_file_state_signature(no_reason),
            "meta:99:",
        )


class LatestSnapshotInfoTests(unittest.TestCase):
    """One SQL query, dict-wraps the row or returns None."""

    def test_returns_dict_view_of_row_when_present(self) -> None:
        row = {
            "id": 7,
            "collection": "demo",
            "repo": "demo-repo",
            "commit_sha": "deadbeef",
            "tree_sha": "cafef00d",
            "metadata": {"schema_version": SCHEMA_VERSION},
        }
        conn = QueuedConnection([FakeCursor(one=row)])
        connection = cast("pci_db.DbConnection", conn)

        result = storage_schema.latest_snapshot_info(connection, "demo", "demo-repo")

        self.assertEqual(result, row)
        # Snapshot lookup must be scoped to collection AND repo.
        query, params = conn.calls[0]
        self.assertIn("FROM project_code_intel_snapshots", query)
        self.assertEqual(params, ["demo", "demo-repo"])

    def test_returns_none_when_no_snapshot_recorded(self) -> None:
        conn = QueuedConnection([FakeCursor(one=None)])
        connection = cast("pci_db.DbConnection", conn)

        self.assertIsNone(
            storage_schema.latest_snapshot_info(connection, "demo", "demo-repo"),
        )


class PreviousFileSignaturesTests(unittest.TestCase):
    """Reshapes file rows into a {source_path: signature} dict."""

    def test_signature_dict_covers_sha256_blob_and_meta_branches_in_one_query(self) -> None:
        rows: list[object] = [
            {
                "source_path": "a.py",
                "file_sha256": "hash-a",
                "git_blob_sha": "blob-a",  # ignored when sha256 present
                "size_bytes": 1,
                "skipped_reason": None,
            },
            {
                "source_path": "b.py",
                "file_sha256": None,
                "git_blob_sha": "blob-b",
                "size_bytes": 2,
                "skipped_reason": None,
            },
            {
                "source_path": "c.bin",
                "file_sha256": None,
                "git_blob_sha": None,
                "size_bytes": 5_000_000,
                "skipped_reason": "too_large",
            },
        ]
        conn = QueuedConnection([FakeCursor(many=rows)])
        connection = cast("pci_db.DbConnection", conn)

        signatures = storage_schema.previous_file_signatures(connection, 9)

        self.assertEqual(
            signatures,
            {
                "a.py": "sha256:hash-a",
                "b.py": "blob:blob-b",
                "c.bin": "meta:5000000:too_large",
            },
        )
        # The query must be parameterised on the snapshot_id, not interpolated.
        query, params = conn.calls[0]
        self.assertIn("FROM project_code_intel_files", query)
        self.assertEqual(params, [9])


class PreviousFileStatesTests(unittest.TestCase):
    """Hydrates rows into PreviousFileState records, covering the
    conditional `None`-vs-string branches for the hash and metadata
    fields in one shot."""

    def test_rows_hydrate_with_optional_hashes_and_dict_metadata_normalized(self) -> None:
        rows: list[object] = [
            {
                "source_path": "a.py",
                "file_sha256": "h1",
                "git_blob_sha": "b1",
                "size_bytes": 12,
                "language": "python",
                "file_role": "source",
                "content_class": "code",
                "is_generated": False,
                "is_vendor": False,
                "is_test": False,
                "is_source": True,
                "is_build": False,
                "is_config": False,
                "is_doc": False,
                "skipped_reason": "trimmed",
                "metadata": {"doc_links": ["https://example"]},
            },
            {
                "source_path": "b.py",
                "file_sha256": None,  # exercises the None branch
                "git_blob_sha": None,  # exercises the None branch
                "size_bytes": 0,
                "language": "python",
                "file_role": "source",
                "content_class": "code",
                "is_generated": False,
                "is_vendor": False,
                "is_test": False,
                "is_source": True,
                "is_build": False,
                "is_config": False,
                "is_doc": False,
                "skipped_reason": None,  # exercises the None branch
                # Non-dict metadata is normalised away by the cast/isinstance check.
                "metadata": "not a dict",
            },
        ]
        conn = QueuedConnection([FakeCursor(many=rows)])
        connection = cast("pci_db.DbConnection", conn)

        states = storage_schema.previous_file_states(connection, 3)

        self.assertEqual(set(states.keys()), {"a.py", "b.py"})

        state_a = states["a.py"]
        self.assertEqual(state_a.file_sha256, "h1")
        self.assertEqual(state_a.git_blob_sha, "b1")
        self.assertEqual(state_a.skipped_reason, "trimmed")
        self.assertEqual(state_a.size_bytes, 12)
        self.assertEqual(state_a.metadata, {"doc_links": ["https://example"]})

        state_b = states["b.py"]
        # None hashes must hydrate to None, not the string "None".
        self.assertIsNone(state_b.file_sha256)
        self.assertIsNone(state_b.git_blob_sha)
        self.assertIsNone(state_b.skipped_reason)
        # Non-dict metadata must collapse to an empty dict, not crash hydration.
        self.assertEqual(state_b.metadata, {})

        query, params = conn.calls[0]
        self.assertIn("FROM project_code_intel_files", query)
        self.assertEqual(params, [3])


class EnsureSchemaTests(unittest.TestCase):
    """`ensure_schema` runs `schema_sql()` then records the migration row."""

    def test_executes_schema_sql_then_records_migration_version(self) -> None:
        conn = QueuedConnection([FakeCursor(), FakeCursor()])
        connection = cast("pci_db.DbConnection", conn)

        with patch.object(pci_db, "schema_sql", return_value="-- fake schema sql"):
            storage_schema.ensure_schema(connection)

        # The schema text must reach the connection first, then the
        # migration insert. Anything else risks a half-bootstrapped DB.
        self.assertEqual(len(conn.calls), 2)
        first_query, first_params = conn.calls[0]
        self.assertEqual(str(first_query), "-- fake schema sql")
        self.assertEqual(first_params, [])

        second_query, second_params = conn.calls[1]
        self.assertIn("project_code_intel_schema_migrations", second_query)
        self.assertEqual(second_params, [SCHEMA_VERSION])


class RecordSchemaMigrationTests(unittest.TestCase):
    """Insert is idempotent (ON CONFLICT DO NOTHING) and parameterised."""

    def test_insert_uses_on_conflict_do_nothing_and_current_schema_version(self) -> None:
        conn = QueuedConnection([FakeCursor()])
        connection = cast("pci_db.DbConnection", conn)

        storage_schema.record_schema_migration(connection)

        query, params = conn.calls[0]
        self.assertIn("INSERT INTO project_code_intel_schema_migrations", query)
        # ON CONFLICT DO NOTHING is what makes repeated bootstrap safe.
        self.assertIn("ON CONFLICT", query)
        self.assertIn("DO NOTHING", query)
        self.assertEqual(params, [SCHEMA_VERSION])


class SchemaMigrationVersionsTests(unittest.TestCase):
    """Returns the version column values, coerced to str, in apply order."""

    def test_returns_versions_in_apply_order_coerced_to_str(self) -> None:
        rows: list[object] = [
            {"version": "code-intel-schema-v1"},
            {"version": "code-intel-schema-v2"},
        ]
        conn = QueuedConnection([FakeCursor(many=rows)])
        connection = cast("pci_db.DbConnection", conn)

        versions = storage_schema.schema_migration_versions(connection)

        self.assertEqual(versions, ["code-intel-schema-v1", "code-intel-schema-v2"])
        query, params = conn.calls[0]
        self.assertIn("project_code_intel_schema_migrations", query)
        # No parameters; the query is a straight ORDER BY apply order.
        self.assertEqual(params, [])


if __name__ == "__main__":
    _ = unittest.main()
