"""Unit tests for snapshot pruning (`project_code_intelligence.storage.core`).

Uses a fake cursor/connection that captures the executed SQL and params, since
these are DELETE statements whose safety depends entirely on the WHERE clause
shape (branch-awareness, null-branch protection, newest-row protection).
"""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, cast

from project_code_intelligence.storage.core import prune_dead_branch_snapshots, prune_old_snapshots

if TYPE_CHECKING:
    from project_code_intelligence import db as pci_db


class FakeCursor:
    def __init__(self, rows: list[object] | None = None) -> None:
        self.rows = list(rows or [])

    def fetchall(self) -> list[object]:
        return self.rows


class RecordingConnection:
    def __init__(self, rows: list[object] | None = None) -> None:
        self.rows = rows or []
        self.calls: list[tuple[str, list[object]]] = []

    def execute(self, query: object, params: object | None = None) -> FakeCursor:
        query_params = cast("list[object]", params) if params is not None else []
        self.calls.append((str(query), query_params))
        return FakeCursor(self.rows)


class PruneOldSnapshotsTests(unittest.TestCase):
    def test_scopes_by_collection_and_repo(self) -> None:
        conn = RecordingConnection(rows=[{"id": 1}])
        deleted = prune_old_snapshots(cast("pci_db.DbConnection", conn), "ws", "repo-a", keep=5)
        self.assertEqual(deleted, 1)
        query, params = conn.calls[0]
        self.assertIn("WHERE collection = %s AND repo = %s", query)
        self.assertEqual(params, ["ws", "repo-a", 5])

    def test_protects_newest_snapshot_per_branch_via_ranking(self) -> None:
        conn = RecordingConnection(rows=[])
        _ = prune_old_snapshots(cast("pci_db.DbConnection", conn), "ws", "repo-a", keep=5)
        query, _params = conn.calls[0]
        # Newest-per-branch protection: a branch_rank window partitioned on branch
        # (null folded into one shared group), with only rank > 1 eligible for the
        # keep-N cut.
        self.assertIn("PARTITION BY COALESCE(branch, '')", query)
        self.assertIn("branch_rank > 1", query)
        self.assertIn("remainder_rank > %s", query)

    def test_newest_ordering_has_deterministic_id_tiebreak(self) -> None:
        conn = RecordingConnection(rows=[])
        _ = prune_old_snapshots(cast("pci_db.DbConnection", conn), "ws", "repo-a", keep=5)
        query, _params = conn.calls[0]
        self.assertIn("ORDER BY created_at DESC, id DESC", query)


class PruneDeadBranchSnapshotsTests(unittest.TestCase):
    def test_deletes_only_dead_non_null_branches(self) -> None:
        conn = RecordingConnection(rows=[{"id": 3}])
        deleted = prune_dead_branch_snapshots(cast("pci_db.DbConnection", conn), "ws", "repo-a", {"main", "feature"})
        self.assertEqual(deleted, 1)
        query, params = conn.calls[0]
        self.assertIn("branch IS NOT NULL", query)
        self.assertIn("branch <> ALL(%s)", query)
        self.assertEqual(params[0], "ws")
        self.assertEqual(params[1], "repo-a")
        self.assertEqual(set(cast("list[str]", params[2])), {"main", "feature"})

    def test_never_deletes_the_newest_row_even_if_its_branch_is_dead(self) -> None:
        conn = RecordingConnection(rows=[])
        _ = prune_dead_branch_snapshots(cast("pci_db.DbConnection", conn), "ws", "repo-a", set())
        query, _params = conn.calls[0]
        self.assertIn("id <> (", query)
        self.assertIn("ORDER BY created_at DESC, id DESC", query)
        self.assertIn("LIMIT 1", query)

    def test_scopes_by_collection_and_repo(self) -> None:
        conn = RecordingConnection(rows=[])
        _ = prune_dead_branch_snapshots(cast("pci_db.DbConnection", conn), "ws", "repo-a", {"main"})
        query, params = conn.calls[0]
        self.assertIn("collection = %s AND repo = %s", query)
        # collection/repo appear twice: once for the DELETE scope, once for the
        # "newest row overall" subquery scope.
        self.assertEqual(params[0], "ws")
        self.assertEqual(params[1], "repo-a")
        self.assertEqual(params[3], "ws")
        self.assertEqual(params[4], "repo-a")


if __name__ == "__main__":
    _ = unittest.main()
