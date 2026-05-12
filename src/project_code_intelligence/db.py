#!/usr/bin/env python3
"""Database access helpers for the project code-intelligence MCP server."""

from __future__ import annotations

import json
from importlib import resources
from typing import TYPE_CHECKING, cast

from psycopg import Connection, OperationalError
from psycopg.conninfo import make_conninfo
from psycopg.rows import DictRow, RowFactory, dict_row
from psycopg.sql import SQL

from project_code_intelligence.config import DatabaseSettings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import LiteralString

DICT_ROW_FACTORY: RowFactory[DictRow] = dict_row
DbConnection = Connection[DictRow]
DbRow = DictRow


class DatabaseConnectionError(RuntimeError):
    """Raised when the configured PostgreSQL connection is unavailable."""


def require_row(row: DbRow | None, description: str) -> DbRow:
    if row is None:
        raise RuntimeError(f"{description} query returned no rows")
    return row


def allow_writes(settings: DatabaseSettings | None = None) -> bool:
    settings = settings or DatabaseSettings.from_env()
    return settings.allow_writes


def connection_hint(settings: DatabaseSettings | None = None) -> str:
    settings = settings or DatabaseSettings.from_env()
    return settings.connection_hint()


def conninfo(settings: DatabaseSettings | None = None) -> str:
    settings = settings or DatabaseSettings.from_env()
    if settings.dsn:
        return settings.dsn

    missing = settings.missing_connection_names()
    if missing:
        raise DatabaseConnectionError(
            "Missing PostgreSQL connection settings for pgvector: "
            + ", ".join(missing)
            + ". Set PGVECTOR_DSN, or set PGVECTOR_HOST/PGVECTOR_PORT/"
            "PGVECTOR_DB/PGVECTOR_USER/PGVECTOR_PASS."
        )

    return make_conninfo(
        "",
        host=settings.host,
        port=settings.port,
        dbname=settings.dbname,
        user=settings.user,
        password=settings.password,
        sslmode=settings.sslmode,
    )


def connect(*, readonly: bool | None = None, settings: DatabaseSettings | None = None) -> DbConnection:
    settings = settings or DatabaseSettings.from_env()
    if readonly is None:
        readonly = not allow_writes(settings)

    try:
        conn = Connection[DictRow].connect(conninfo(settings), row_factory=DICT_ROW_FACTORY)
    except OperationalError as exc:
        raise DatabaseConnectionError(
            "Could not connect to PostgreSQL/pgvector using "
            + connection_hint(settings)
            + ". Set PGVECTOR_DSN, or set PGVECTOR_HOST/PGVECTOR_PORT/"
            "PGVECTOR_DB/PGVECTOR_USER/PGVECTOR_PASS for your database.\n" + str(exc)
        ) from exc
    if readonly:
        conn.read_only = True
    return conn


def schema_sql() -> SQL:
    text = resources.files("project_code_intelligence").joinpath("schema.sql").read_text(encoding="utf-8")
    return SQL(cast("LiteralString", text))


def query_sql(text: str) -> SQL:
    return SQL(cast("LiteralString", text))


def vector_literal(values: Sequence[object]) -> str:
    if not isinstance(values, list):
        raise TypeError("embedding must be a list of numbers")
    if not values:
        raise ValueError("embedding must be a non-empty list of numbers")

    out: list[str] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("embedding values must be numbers")
        out.append(format(float(value), ".9g"))
    return "[" + ",".join(out) + "]"


def json_metadata(value: object) -> str:
    if value is None:
        return "{}"
    if not isinstance(value, dict):
        raise TypeError("metadata must be an object")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
