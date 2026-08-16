"""Database diagnostics for project-code-intelligence."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from project_code_intelligence import config, console_ui, db
from project_code_intelligence.doctor.common import result, row_text
from project_code_intelligence.mcp.status import annotate_status_snapshots
from project_code_intelligence.storage import load_latest_snapshots

if TYPE_CHECKING:
    from project_code_intelligence.doctor.types import CheckResult
    from project_code_intelligence.models import JsonObject


def _dsn_has_credentials(settings: config.DatabaseSettings) -> bool:
    if not settings.dsn:
        return False
    parts = urlsplit(settings.dsn)
    return bool(settings.dsn_user or settings.dsn_password or parts.username or parts.password)


def _database_check_settings(settings: config.DatabaseSettings) -> config.DatabaseSettings:
    check_settings = db.maintenance_database_settings(settings) if settings.database_inferred else settings
    if check_settings.dsn and not _dsn_has_credentials(check_settings):
        admin_settings = db.postgres_admin_settings()
        if admin_settings is not None:
            return db.settings_for_database(admin_settings, db.DEFAULT_MAINTENANCE_DB)
    return check_settings


def check_database() -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        settings = config.DatabaseSettings.from_env()
    except ValueError as exc:
        return [result("database-config", "fail", str(exc))]

    missing = settings.missing_connection_names()
    if missing:
        return [
            result(
                "database-config",
                "fail",
                "Missing PostgreSQL connection settings: " + ", ".join(missing),
                "Set PCI_DATABASE_URL, or set PCI_PG_HOST/PCI_PG_PORT/PCI_PG_DB/PCI_PG_USER/PCI_PG_PASS.",
            )
        ]

    try:
        check_settings = _database_check_settings(settings)
    except ValueError as exc:
        return [result("database-config", "fail", str(exc))]
    database_target = check_settings.display_target()
    results.append(result("database-config", "ok", database_target, check_settings.connection_hint()))
    try:
        with db.connect(settings=check_settings) as conn:
            info_row = conn.execute(
                "SELECT current_database() AS database_name, current_user AS user_name, version() AS version"
            ).fetchone()
            if info_row is None:
                results.append(result("database", "fail", "PostgreSQL version query returned no row."))
                return results
            results.append(
                result(
                    "database",
                    "ok",
                    f"connected to {row_text(info_row, 'database_name')} as "
                    f"{row_text(info_row, 'user_name')} at {database_target}",
                    row_text(info_row, "version"),
                )
            )
    except db.DatabaseConnectionError as exc:
        results.append(result("database", "fail", "Could not connect to PostgreSQL/pgvector.", str(exc)))
    return results


def _sha_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _snapshot_label(snapshot: JsonObject) -> str:
    repo = snapshot.get("repo")
    collection = snapshot.get("collection")
    base = f"{collection}/{repo}" if collection else str(repo)
    branch = snapshot.get("branch")
    return f"{base}@{branch}" if isinstance(branch, str) and branch else base


def _relevant_snapshots() -> list[JsonObject]:
    """Latest snapshots with a resolvable local git HEAD, or [] if none apply.

    Empty covers: no usable database settings, no connection, no snapshots
    table yet, no snapshot rows, and no snapshot whose repo maps to a local
    git checkout -- ``database`` and ``project`` already report those cases.
    """
    try:
        settings = db.inferred_database_role_settings(config.DatabaseSettings.from_env(), "rw")
    except ValueError:
        return []
    try:
        with db.connect(settings=settings) as conn:
            if not db.table_exists(conn, "project_code_intel_snapshots"):
                return []
            rows = load_latest_snapshots(conn)
    except db.DatabaseConnectionError:
        return []
    return [snap for snap in annotate_status_snapshots(rows) if snap.get("head_status") != "unknown"]


def check_index_freshness() -> list[CheckResult]:
    """Compare the newest indexed snapshot's commit against the local git HEAD.

    Ground truth from the database, distinct from ``project-hooks`` (which only
    reports whether the git post-commit hook is installed): a hook that fires
    but whose reindex fails partway through still shows up as stale here.
    """
    snapshots = _relevant_snapshots()
    if not snapshots:
        return []
    stale = [snap for snap in snapshots if snap.get("head_status") != "current"]
    if stale:
        detail = "; ".join(
            f"{_snapshot_label(snap)}: indexed {console_ui.short_sha(_sha_text(snap.get('commit_sha')))}, "
            f"HEAD is {console_ui.short_sha(_sha_text(snap.get('head_commit')))}"
            for snap in stale
        )
        return [
            result(
                "index-freshness",
                "warn",
                f"{len(stale)} of {len(snapshots)} indexed repo(s) behind HEAD",
                f"{detail} -- check `pci hook status` and `pci status` for the reindex hook/run state.",
            )
        ]
    return [result("index-freshness", "ok", f"index matches HEAD for {len(snapshots)} repo(s)")]
