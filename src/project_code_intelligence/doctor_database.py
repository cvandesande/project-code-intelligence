"""Database diagnostics for project-code-intelligence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from project_code_intelligence import config, db
from project_code_intelligence.doctor_common import result, row_bool, row_text, table_exists
from project_code_intelligence.models import SCHEMA_VERSION
from project_code_intelligence.storage import schema_migration_versions

if TYPE_CHECKING:
    from project_code_intelligence.doctor_types import CheckResult, Status


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
                "Set PGVECTOR_DSN, or set PGVECTOR_HOST/PGVECTOR_PORT/PGVECTOR_DB/PGVECTOR_USER/PGVECTOR_PASS.",
            )
        ]

    results.append(result("database-config", "ok", settings.connection_hint()))
    try:
        with db.connect(settings=settings) as conn:
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
                    f"connected to {row_text(info_row, 'database_name')} as {row_text(info_row, 'user_name')}",
                    row_text(info_row, "version"),
                )
            )

            extension_row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
            if extension_row is not None:
                results.append(
                    result("pgvector", "ok", f"pgvector extension installed: {row_text(extension_row, 'extversion')}")
                )
            else:
                privilege_row = conn.execute(
                    "SELECT has_database_privilege(current_database(), 'CREATE') AS can_create"
                ).fetchone()
                can_create = bool(privilege_row and row_bool(privilege_row, "can_create"))
                results.append(
                    result(
                        "pgvector",
                        "warn" if can_create else "fail",
                        "pgvector extension is not installed in the configured database.",
                        (
                            "pci-index will attempt CREATE EXTENSION vector during schema setup."
                            if can_create
                            else "Install pgvector or grant CREATE on the database before running pci-index."
                        ),
                    )
                )

            if table_exists(conn, "project_code_intel_records"):
                snapshot_row = conn.execute("SELECT count(*) AS count FROM project_code_intel_snapshots").fetchone()
                record_row = conn.execute(
                    "SELECT count(*) AS count, count(embedding) AS embedded FROM project_code_intel_records"
                ).fetchone()
                snapshot_count = row_text(snapshot_row, "count") if snapshot_row else "0"
                record_count = row_text(record_row, "count") if record_row else "0"
                embedded_count = row_text(record_row, "embedded") if record_row else "0"
                results.append(
                    result(
                        "schema",
                        "ok",
                        "schema initialized; "
                        f"snapshots={snapshot_count}, records={record_count}, embedded={embedded_count}",
                    )
                )
                if table_exists(conn, "project_code_intel_schema_migrations"):
                    versions = schema_migration_versions(conn)
                    status: Status = "ok" if SCHEMA_VERSION in versions else "warn"
                    results.append(
                        result(
                            "schema-version",
                            status,
                            f"schema versions: {', '.join(versions) if versions else '<none>'}",
                            None if status == "ok" else f"current package expects {SCHEMA_VERSION}",
                        )
                    )
            else:
                results.append(
                    result(
                        "schema",
                        "warn",
                        "code-intelligence schema is not initialized.",
                        "Run pci-index once to create the schema and ingest a repository.",
                    )
                )
    except db.DatabaseConnectionError as exc:
        results.append(result("database", "fail", "Could not connect to PostgreSQL/pgvector.", str(exc)))
    return results
