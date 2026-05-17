"""MCP database session helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from project_code_intelligence import config, db

if TYPE_CHECKING:
    from collections.abc import Generator

DEFAULT_MCP_STATEMENT_TIMEOUT_MS = 15_000
DEFAULT_MCP_LOCK_TIMEOUT_MS = 5_000
DEFAULT_MCP_IDLE_IN_TRANSACTION_TIMEOUT_MS = 30_000
DEFAULT_MCP_MAX_STATUS_ROWS = 1_000
MCP_DATABASE_ENV_NAMES = (
    "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL",
    "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER",
    "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD",
    "PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH",
)


def mcp_statement_timeout_ms() -> int:
    return config.env_int(
        "PROJECT_CODE_INTELLIGENCE_MCP_STATEMENT_TIMEOUT_MS",
        DEFAULT_MCP_STATEMENT_TIMEOUT_MS,
        minimum=1,
    )


def mcp_lock_timeout_ms() -> int:
    return config.env_int(
        "PROJECT_CODE_INTELLIGENCE_MCP_LOCK_TIMEOUT_MS",
        DEFAULT_MCP_LOCK_TIMEOUT_MS,
        minimum=1,
    )


def mcp_idle_in_transaction_timeout_ms() -> int:
    return config.env_int(
        "PROJECT_CODE_INTELLIGENCE_MCP_IDLE_IN_TRANSACTION_TIMEOUT_MS",
        DEFAULT_MCP_IDLE_IN_TRANSACTION_TIMEOUT_MS,
        minimum=1,
    )


def mcp_max_status_rows() -> int:
    return config.env_int(
        "PROJECT_CODE_INTELLIGENCE_MCP_MAX_STATUS_ROWS",
        DEFAULT_MCP_MAX_STATUS_ROWS,
        minimum=1,
    )


def configure_session(conn: db.DbConnection) -> None:
    _ = conn.execute(
        """
        SELECT
          set_config('statement_timeout', %s, true),
          set_config('lock_timeout', %s, true),
          set_config('idle_in_transaction_session_timeout', %s, true)
        """,
        [
            f"{mcp_statement_timeout_ms()}ms",
            f"{mcp_lock_timeout_ms()}ms",
            f"{mcp_idle_in_transaction_timeout_ms()}ms",
        ],
    )


@contextmanager
def connect() -> Generator[db.DbConnection]:
    settings = db.inferred_database_role_settings(config.DatabaseSettings.from_env(role="mcp"), "ro")
    try:
        with db.connect(settings=settings, readonly=True) as conn:
            configure_session(conn)
            yield conn
    except db.DatabaseConnectionError as exc:
        env_list = ", ".join(MCP_DATABASE_ENV_NAMES)
        raise db.DatabaseConnectionError(
            f"{exc}\n"
            "pci-mcp could not open its read-only project database. "
            f"If you generated project-scoped MCP config, export {env_list} in the MCP client's "
            "environment and restart the client."
        ) from exc


def code_intel_tables_exist(conn: db.DbConnection) -> bool:
    row = conn.execute(
        """
        SELECT to_regclass('public.project_code_intel_records') IS NOT NULL AS exists
        """
    ).fetchone()
    return bool(db.require_row(row, "code-intel table existence")["exists"])


def table_regclass_exists(conn: db.DbConnection, table: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s) IS NOT NULL AS exists", [f"public.{table}"]).fetchone()
    return bool(db.require_row(row, "table existence")["exists"])
