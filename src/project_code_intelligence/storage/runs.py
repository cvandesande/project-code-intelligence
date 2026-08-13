"""Index-run ledger: live status rows for in-flight and recent ingest runs.

Writers are best-effort by design: an unavailable database must never block
or fail an index run, so every write helper swallows connection errors and
returns a null result instead. Each write opens its own short-lived
connection -- the ingest's shared connection holds a long transaction, and
the ledger heartbeat runs on a separate thread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from project_code_intelligence import db
from project_code_intelligence.storage.schema import row_int

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

INDEX_RUNS_KEEP = 20


class _ActiveRunState:
    # Class-attribute holder mirroring runtime._MetricsState: rebindable without
    # `global`, readable from the ledger-heartbeat thread.
    run_id: int | None = None


def set_active_index_run(run_id: int | None) -> None:
    _ActiveRunState.run_id = run_id


def active_index_run_id() -> int | None:
    return _ActiveRunState.run_id


def start_index_run(collection: str, repos: list[str], *, pid: int, host: str) -> int | None:
    """Insert a run row and return its id, or None when the database is unreachable."""
    try:
        with db.connect(readonly=False) as conn:
            row = conn.execute(
                """
                INSERT INTO project_code_intel_index_runs (collection, repos, pid, host, phase)
                VALUES (%s, %s::jsonb, %s, %s, 'starting')
                RETURNING id
                """,
                [collection, db.compact_json(repos), pid, host],
            ).fetchone()
    except (db.DatabaseConnectionError, db.PsycopgError):
        return None
    return row_int(row, "id") if row else None


def heartbeat_index_run(run_id: int, *, phase: str | None, progress: JsonObject) -> None:
    """Refresh heartbeat_at, phase, and the metrics snapshot. Errors are swallowed."""
    try:
        with db.connect(readonly=False) as conn:
            _ = conn.execute(
                """
                UPDATE project_code_intel_index_runs
                SET heartbeat_at = now(), phase = %s, progress = %s::jsonb
                WHERE id = %s
                """,
                [phase, db.compact_json(progress, default=str), run_id],
            )
    except (db.DatabaseConnectionError, db.PsycopgError):
        return


def set_index_run_modes(run_id: int, repo_modes: JsonObject) -> None:
    """Stamp per-repo mode ("incremental" / "full:<reason>"). Errors are swallowed."""
    try:
        with db.connect(readonly=False) as conn:
            _ = conn.execute(
                "UPDATE project_code_intel_index_runs SET repo_modes = %s::jsonb WHERE id = %s",
                [db.compact_json(repo_modes), run_id],
            )
    except (db.DatabaseConnectionError, db.PsycopgError):
        return


def finish_index_run(
    run_id: int,
    *,
    exit_code: int,
    interrupted: bool,
    error: str | None,
    progress: JsonObject,
) -> None:
    """Stamp the terminal state, then prune old finished rows. Errors are swallowed."""
    try:
        with db.connect(readonly=False) as conn:
            _ = conn.execute(
                """
                UPDATE project_code_intel_index_runs
                SET finished_at = now(), heartbeat_at = now(),
                    exit_code = %s, interrupted = %s, error = %s,
                    phase = 'done', progress = %s::jsonb
                WHERE id = %s
                """,
                [exit_code, interrupted, error, db.compact_json(progress, default=str), run_id],
            )
            _ = conn.execute(
                """
                DELETE FROM project_code_intel_index_runs
                WHERE finished_at IS NOT NULL
                  AND id IN (
                    SELECT id FROM (
                        SELECT id, row_number() OVER (
                            PARTITION BY collection ORDER BY started_at DESC, id DESC
                        ) AS rn
                        FROM project_code_intel_index_runs
                        WHERE finished_at IS NOT NULL
                    ) ranked
                    WHERE ranked.rn > %s
                  )
                """,
                [INDEX_RUNS_KEEP],
            )
    except (db.DatabaseConnectionError, db.PsycopgError):
        return


_LOAD_RUNS_SELECT = """
    SELECT id, collection, repos, repo_modes, pid, host, phase, progress,
           started_at, heartbeat_at, finished_at, exit_code, interrupted, error,
           finished_at IS NULL AS running
    FROM project_code_intel_index_runs
"""
_LOAD_RUNS_SUFFIX = "ORDER BY started_at DESC, id DESC LIMIT %s"


def load_index_runs(conn: db.DbConnection, *, collection: str | None = None, limit: int = 20) -> list[db.DbRow]:
    """Newest run rows, optionally scoped to one collection. Read side; raises on DB errors."""
    clause = "WHERE collection = %s" if collection is not None else ""
    params: list[object] = [collection] if collection is not None else []
    params.append(limit)
    return conn.execute("\n".join([_LOAD_RUNS_SELECT, clause, _LOAD_RUNS_SUFFIX]), params).fetchall()
