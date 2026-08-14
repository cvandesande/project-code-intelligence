"""Persistence for `pci check` regression-ratchet baselines.

One baseline per (collection, repo, branch) -- the same identity scheme
snapshots use. Freezing a baseline replaces the prior finding set for that
key; there is no history of baselines, only the current one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from project_code_intelligence import db
from project_code_intelligence.check_core import BaselineEntry
from project_code_intelligence.storage.schema import row_int

if TYPE_CHECKING:
    from collections.abc import Sequence

    from project_code_intelligence.check_core import CheckFinding


def _find_baseline_id(conn: db.DbConnection, *, collection: str, repo: str, branch: str | None) -> int | None:
    """Baseline id for (collection, repo, branch), NULL-safe on `branch` (detached HEAD).

    Plain `branch = %s` is never true for a NULL parameter (SQL's NULL != NULL),
    and `ON CONFLICT` never matches an existing NULL column either -- both would
    silently miss every detached-HEAD baseline. `IS NOT DISTINCT FROM` is the
    NULL-safe equality that treats two NULLs as equal.
    """
    row = conn.execute(
        """
        SELECT id FROM project_code_intel_check_baselines
        WHERE collection = %s AND repo = %s AND branch IS NOT DISTINCT FROM %s
        """,
        [collection, repo, branch],
    ).fetchone()
    return row_int(row, "id") if row is not None else None


def freeze_baseline(
    conn: db.DbConnection,
    *,
    collection: str,
    repo: str,
    branch: str | None,
    findings: Sequence[CheckFinding],
) -> int:
    """Replace the (collection, repo, branch) baseline with `findings`. Returns the count stored."""
    baseline_id = _find_baseline_id(conn, collection=collection, repo=repo, branch=branch)
    if baseline_id is None:
        row = conn.execute(
            """
            INSERT INTO project_code_intel_check_baselines (collection, repo, branch)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            [collection, repo, branch],
        ).fetchone()
        baseline_id = row_int(db.require_row(row, "insert check baseline"), "id")
    else:
        _ = conn.execute(
            "UPDATE project_code_intel_check_baselines SET created_at = now() WHERE id = %s",
            [baseline_id],
        )
    _ = conn.execute(
        "DELETE FROM project_code_intel_check_baseline_findings WHERE baseline_id = %s",
        [baseline_id],
    )
    for finding in findings:
        _ = conn.execute(
            """
            INSERT INTO project_code_intel_check_baseline_findings (
                baseline_id, fingerprint, rule_id, level, tool_name, message,
                primary_source_path, line_start, line_end
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                baseline_id,
                finding.fingerprint,
                finding.rule_id,
                finding.level,
                finding.tool_name,
                finding.message,
                finding.primary_source_path,
                finding.line_start,
                finding.line_end,
            ],
        )
    return len(findings)


def check_tables_exist(conn: db.DbConnection) -> bool:
    """False on a database that has never run `pci check --baseline` (read-only callers can't
    create the tables themselves, so they need to distinguish "no baseline yet" from an error)."""
    row = conn.execute(
        "SELECT to_regclass('public.project_code_intel_check_baselines') IS NOT NULL AS exists"
    ).fetchone()
    return bool(db.require_row(row, "check-baseline table existence")["exists"])


def load_baseline(
    conn: db.DbConnection,
    *,
    collection: str,
    repo: str,
    branch: str | None,
) -> list[BaselineEntry] | None:
    """The frozen baseline for (collection, repo, branch), or None if never frozen."""
    if not check_tables_exist(conn):
        return None
    baseline_id = _find_baseline_id(conn, collection=collection, repo=repo, branch=branch)
    if baseline_id is None:
        return None
    rows = conn.execute(
        """
        SELECT fingerprint, rule_id, level
        FROM project_code_intel_check_baseline_findings
        WHERE baseline_id = %s
        """,
        [baseline_id],
    ).fetchall()
    return [
        BaselineEntry(
            fingerprint=str(r["fingerprint"]),
            rule_id=str(r["rule_id"]),
            level=str(r["level"]) if r["level"] is not None else None,
        )
        for r in rows
    ]
