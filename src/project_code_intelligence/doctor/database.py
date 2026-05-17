"""Database diagnostics for project-code-intelligence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from project_code_intelligence import config, db
from project_code_intelligence.doctor.common import result, row_text

if TYPE_CHECKING:
    from project_code_intelligence.doctor.types import CheckResult


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
                "Set PROJECT_CODE_INTELLIGENCE_DATABASE_URL, or set "
                "PGVECTOR_HOST/PGVECTOR_PORT/PGVECTOR_DB/PGVECTOR_USER/PGVECTOR_PASS.",
            )
        ]

    check_settings = db.maintenance_database_settings(settings) if settings.database_inferred else settings
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
