"""Database diagnostics for project-code-intelligence."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from project_code_intelligence import config, db
from project_code_intelligence.doctor.common import result, row_text

if TYPE_CHECKING:
    from project_code_intelligence.doctor.types import CheckResult


def _dsn_has_credentials(settings: config.DatabaseSettings) -> bool:
    if not settings.dsn:
        return False
    parts = urlsplit(settings.dsn)
    return bool(settings.dsn_user or settings.dsn_password or parts.username or parts.password)


def _postgres_admin_check_settings() -> config.DatabaseSettings | None:
    settings = config.DatabaseSettings.from_env(admin_scope="postgres")
    if bool(settings.admin_user) != bool(settings.admin_password):
        raise ValueError("Set both PCI_POSTGRES_ADMIN_USER and PCI_POSTGRES_ADMIN_PASSWORD, or set neither.")
    if not settings.admin_user or not settings.admin_password:
        return None
    return db.settings_for_database(
        db.settings_with_credentials(settings, settings.admin_user, settings.admin_password),
        db.DEFAULT_MAINTENANCE_DB,
    )


def _database_check_settings(settings: config.DatabaseSettings) -> config.DatabaseSettings:
    check_settings = db.maintenance_database_settings(settings) if settings.database_inferred else settings
    if check_settings.dsn and not _dsn_has_credentials(check_settings):
        admin_settings = _postgres_admin_check_settings()
        if admin_settings is not None:
            return admin_settings
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
