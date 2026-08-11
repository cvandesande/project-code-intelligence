#!/usr/bin/env python3
"""Database access helpers for the project code-intelligence MCP server."""

from __future__ import annotations

import hmac
import json
import secrets
from base64 import urlsafe_b64encode
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib import resources
from typing import TYPE_CHECKING, Protocol, cast
from urllib.parse import urlsplit

from psycopg import Connection
from psycopg.conninfo import make_conninfo
from psycopg.errors import Error as PsycopgError
from psycopg.errors import OperationalError
from psycopg.rows import DictRow, RowFactory, dict_row
from psycopg.sql import SQL

from project_code_intelligence.common import sha256_text
from project_code_intelligence.config import DatabaseSettings, database_url_with_dbname
from project_code_intelligence.exceptions import DatabaseConnectionError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from typing_extensions import LiteralString

DICT_ROW_FACTORY: RowFactory[DictRow] = dict_row
DbConnection = Connection[DictRow]
DbRow = DictRow
MAX_POSTGRES_IDENTIFIER_CHARS = 63
_PG_IDENTITY_ARG_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_, [].")
DEFAULT_MAINTENANCE_DB = "postgres"
DEFAULT_TEMPLATE_DB = "template1"
DEFAULT_POSTGRES_INDEX_ADMIN_ROLE = "pci_index_admin"
VECTOR_EXTENSION = "vector"


class AutocommitConnect(Protocol):
    def __call__(self, conninfo: str, *, autocommit: bool, row_factory: RowFactory[DictRow]) -> DbConnection: ...


@dataclass(frozen=True)
class DatabaseRole:
    name: str
    password: str | None
    created: bool
    database_url: str


@dataclass(frozen=True)
class InferredDatabaseCreateContext:
    dbname: str
    rw_role_name: str
    ro_role_name: str
    create_project_roles: bool


@dataclass(frozen=True)
class DatabaseBootstrapResult:
    dbname: str
    database_created: bool = False
    database_dropped: bool = False
    rw_role: DatabaseRole | None = None
    ro_role: DatabaseRole | None = None


@dataclass(frozen=True)
class PostgresBootstrapResult:
    postgres_url: str
    index_role: DatabaseRole
    template_database: str = DEFAULT_TEMPLATE_DB
    vector_template_ready: bool = False
    vector_template_created: bool = False


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


def postgres_bootstrap_connection_hint(settings: DatabaseSettings) -> str:
    extras: list[str] = []
    if settings.admin_user:
        extras.append("PCI_POSTGRES_ADMIN_USER=<set>")
    if settings.admin_password:
        extras.append("PCI_POSTGRES_ADMIN_PASSWORD=<set>")
    suffix = " " + " ".join(extras) if extras else ""
    if settings.dsn:
        return f"{settings.dsn_source}=<hidden>{suffix}"
    return f"PCI_PG_HOST={settings.host} PCI_PG_PORT={settings.port}{suffix}"


def conninfo(settings: DatabaseSettings | None = None) -> str:
    settings = settings or DatabaseSettings.from_env()
    connection_options: dict[str, str] = {
        "connect_timeout": str(settings.connect_timeout_seconds),
        "keepalives": "1",
        "keepalives_idle": str(settings.keepalives_idle_seconds),
        "keepalives_interval": str(settings.keepalives_interval_seconds),
        "keepalives_count": str(settings.keepalives_count),
    }
    if settings.dsn:
        if settings.dsn_user:
            connection_options["user"] = settings.dsn_user
        if settings.dsn_password:
            connection_options["password"] = settings.dsn_password
        return make_conninfo(settings.dsn, **connection_options)

    missing = settings.missing_connection_names()
    if missing:
        raise DatabaseConnectionError(
            "Missing PostgreSQL connection settings for pgvector: "
            + ", ".join(missing)
            + ". Set PCI_DATABASE_URL, or set PCI_PG_HOST/PCI_PG_PORT/"
            "PCI_PG_DB/PCI_PG_USER/PCI_PG_PASS."
        )

    return make_conninfo(
        "",
        host=settings.host,
        port=settings.port,
        dbname=settings.dbname,
        user=settings.user,
        password=settings.password,
        sslmode=settings.sslmode,
        **connection_options,
    )


def settings_for_database(settings: DatabaseSettings, dbname: str) -> DatabaseSettings:
    dsn = database_url_with_dbname(settings.dsn, dbname) if settings.dsn else None
    return replace(settings, dsn=dsn, dbname=dbname, database_inferred=False)


def settings_with_credentials(settings: DatabaseSettings, user: str, password: str | None) -> DatabaseSettings:
    if settings.dsn:
        return replace(settings, dsn_user=user, dsn_password=password)
    return replace(settings, user=user, password=password)


def configured_database_user(settings: DatabaseSettings) -> str | None:
    if settings.dsn_user:
        return settings.dsn_user
    if settings.dsn:
        return urlsplit(settings.dsn).username
    return settings.user


def inferred_database_role_settings(settings: DatabaseSettings, access: str) -> DatabaseSettings:
    if access not in {"rw", "ro"}:
        raise ValueError("database role access must be 'rw' or 'ro'")
    if not settings.database_inferred or not settings.admin_password or not settings.dbname:
        return settings
    if bool(settings.admin_user) != bool(settings.admin_password):
        raise DatabaseConnectionError(
            "Set both PCI_DATABASE_ADMIN_USER and PCI_DATABASE_ADMIN_PASSWORD, or set neither."
        )
    if settings.dsn_user or settings.dsn_password:
        return settings
    role_name = project_database_role_name(settings.dbname, access)
    password = project_database_role_password(settings.dbname, role_name, settings.admin_password)
    return settings_with_credentials(settings, role_name, password)


def _validate_identifier(identifier: str, kind: str) -> None:
    if (
        not identifier
        or not identifier.isascii()
        or not identifier.replace("_", "").isalnum()
        or len(identifier) > MAX_POSTGRES_IDENTIFIER_CHARS
    ):
        raise DatabaseConnectionError(f"refusing to use unsafe PostgreSQL {kind}: {identifier!r}")


def create_database_sql(dbname: str, *, owner: str | None = None) -> SQL:
    _validate_identifier(dbname, "database name")
    if owner is not None:
        _validate_identifier(owner, "role name")
        return SQL(cast("LiteralString", f"CREATE DATABASE {dbname} OWNER {owner}"))
    return SQL(cast("LiteralString", f"CREATE DATABASE {dbname}"))


def drop_database_sql(dbname: str) -> SQL:
    _validate_identifier(dbname, "database name")
    return SQL(cast("LiteralString", f"DROP DATABASE IF EXISTS {dbname} WITH (FORCE)"))


def postgres_string_literal(value: str) -> str:
    if "\x00" in value:
        raise DatabaseConnectionError("refusing to use PostgreSQL string literal containing NUL")
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"E'{escaped}'"


def create_role_sql(role_name: str, password: str) -> SQL:
    _validate_identifier(role_name, "role name")
    password_literal = postgres_string_literal(password)
    return SQL(cast("LiteralString", f"CREATE ROLE {role_name} LOGIN PASSWORD {password_literal}"))


def alter_role_password_sql(role_name: str, password: str) -> SQL:
    _validate_identifier(role_name, "role name")
    password_literal = postgres_string_literal(password)
    return SQL(cast("LiteralString", f"ALTER ROLE {role_name} PASSWORD {password_literal}"))


def create_index_admin_role_sql(role_name: str, password: str) -> SQL:
    _validate_identifier(role_name, "role name")
    password_literal = postgres_string_literal(password)
    return SQL(cast("LiteralString", f"CREATE ROLE {role_name} LOGIN CREATEDB CREATEROLE PASSWORD {password_literal}"))


def alter_index_admin_role_sql(role_name: str, password: str) -> SQL:
    _validate_identifier(role_name, "role name")
    password_literal = postgres_string_literal(password)
    return SQL(
        cast("LiteralString", f"ALTER ROLE {role_name} WITH LOGIN CREATEDB CREATEROLE PASSWORD {password_literal}")
    )


def project_database_role_name(dbname: str, access: str) -> str:
    if access not in {"rw", "ro"}:
        raise ValueError("database role access must be 'rw' or 'ro'")
    _validate_identifier(dbname, "database name")
    suffix = f"_{access}"
    if len(dbname) + len(suffix) <= MAX_POSTGRES_IDENTIFIER_CHARS:
        return dbname + suffix
    digest = sha256_text(dbname)[:8]
    max_prefix = MAX_POSTGRES_IDENTIFIER_CHARS - len(suffix) - len("_") - len(digest)
    prefix = dbname[:max_prefix].rstrip("_") or "pci"
    return f"{prefix}_{digest}{suffix}"


def _generated_role_password() -> str:
    return secrets.token_urlsafe(32)


def project_database_role_password(dbname: str, role_name: str, admin_password: str | None) -> str:
    if admin_password is None:
        return _generated_role_password()
    _validate_identifier(dbname, "database name")
    _validate_identifier(role_name, "role name")
    message = f"project-code-intelligence:v1:{dbname}:{role_name}".encode()
    digest = hmac.new(admin_password.encode(), message, sha256).digest()
    return "pci_" + urlsafe_b64encode(digest).decode().rstrip("=")


def postgres_bootstrap_role_password(role_name: str, admin_password: str) -> str:
    _validate_identifier(role_name, "role name")
    message = f"project-code-intelligence:postgres-role:v1:{role_name}".encode()
    digest = hmac.new(admin_password.encode(), message, sha256).digest()
    return "pci_" + urlsafe_b64encode(digest).decode().rstrip("=")


def _bootstrap_connection_settings(settings: DatabaseSettings) -> DatabaseSettings:
    if bool(settings.admin_user) != bool(settings.admin_password):
        raise DatabaseConnectionError(
            "Set both PCI_DATABASE_ADMIN_USER and PCI_DATABASE_ADMIN_PASSWORD, or set neither."
        )
    if settings.admin_user and settings.admin_password:
        return settings_with_credentials(settings, settings.admin_user, settings.admin_password)
    return settings


def writer_admin_fallback(settings: DatabaseSettings) -> DatabaseSettings:
    """Promote writer credentials to the effective admin when no admin is configured.

    The bundled local pgvector container ships with `codeintel:codeintel` as a superuser
    and nothing else, so users who don't set PCI_DATABASE_ADMIN_*
    still expect pci-index to create per-project rw/ro roles. Falling back to the writer
    creds here makes role passwords deterministic across runs (HMAC keyed on the writer
    password) and unblocks MCP config emission for the default local DB.
    """
    if settings.admin_user or settings.admin_password:
        return settings
    user = settings.dsn_user if settings.dsn else settings.user
    password = settings.dsn_password if settings.dsn else settings.password
    if not user or not password:
        return settings
    return replace(settings, admin_user=user, admin_password=password)


def maintenance_database_settings(settings: DatabaseSettings) -> DatabaseSettings:
    """Return settings for checking the PostgreSQL server without creating a project DB."""
    return settings_for_database(_bootstrap_connection_settings(settings), DEFAULT_MAINTENANCE_DB)


def _ensure_inferred_database_target(settings: DatabaseSettings, *, operation: str) -> str:
    if not settings.database_inferred or not settings.dbname:
        raise DatabaseConnectionError(
            f"Refusing to {operation} an explicit PostgreSQL database. "
            "Omit the database path from PCI_DATABASE_URL, or leave PCI_PG_DB unset, "
            "so project-code-intelligence can infer and manage a PCI-owned database."
        )
    _validate_identifier(settings.dbname, "database name")
    return settings.dbname


def _role_exists(conn: DbConnection, role_name: str) -> bool:
    row = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", [role_name]).fetchone()
    return row is not None


def _ensure_login_role(conn: DbConnection, settings: DatabaseSettings, role_name: str) -> DatabaseRole:
    dbname = settings.dbname
    if dbname is None:
        raise DatabaseConnectionError("cannot derive scoped PostgreSQL role password without a database name")
    password = project_database_role_password(dbname, role_name, settings.admin_password)
    if _role_exists(conn, role_name):
        # The generated pci_index_admin role intentionally is not a superuser.
        # PostgreSQL can refuse ALTER ROLE for pre-existing project roles unless
        # the current role has ADMIN OPTION on each target role. The scoped
        # passwords are deterministic, so existing roles can be reused here.
        if settings.admin_password is not None and settings.admin_user != DEFAULT_POSTGRES_INDEX_ADMIN_ROLE:
            _ = conn.execute(alter_role_password_sql(role_name, password))
        return DatabaseRole(
            name=role_name,
            password=password if settings.admin_password is not None else None,
            created=False,
            database_url=settings.role_database_url(
                role_name, password if settings.admin_password is not None else None
            ),
        )
    _ = conn.execute(create_role_sql(role_name, password))
    return DatabaseRole(
        name=role_name,
        password=password,
        created=True,
        database_url=settings.role_database_url(role_name, password),
    )


def _ensure_index_admin_role(conn: DbConnection, settings: DatabaseSettings, role_name: str) -> DatabaseRole:
    if settings.admin_password is None:
        raise DatabaseConnectionError("cannot derive PostgreSQL bootstrap role password without an admin password")
    password = postgres_bootstrap_role_password(role_name, settings.admin_password)
    postgres_url = settings.postgres_url()
    if _role_exists(conn, role_name):
        _ = conn.execute(alter_index_admin_role_sql(role_name, password))
        return DatabaseRole(
            name=role_name,
            password=password,
            created=False,
            database_url=postgres_url,
        )
    _ = conn.execute(create_index_admin_role_sql(role_name, password))
    return DatabaseRole(
        name=role_name,
        password=password,
        created=True,
        database_url=postgres_url,
    )


def _database_exists(conn: DbConnection, dbname: str) -> bool:
    row = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", [dbname]).fetchone()
    return row is not None


def _extension_exists(conn: DbConnection, extension_name: str) -> bool:
    row = conn.execute("SELECT 1 FROM pg_extension WHERE extname = %s", [extension_name]).fetchone()
    return row is not None


def ensure_vector_extension(conn: DbConnection) -> bool:
    if _extension_exists(conn, VECTOR_EXTENSION):
        return False
    _ = conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    return True


def _reject_other_project_scoped_runtime_role(settings: DatabaseSettings, expected_rw_role: str) -> None:
    runtime_user = configured_database_user(settings)
    if (
        runtime_user is not None
        and runtime_user.startswith("pci_")
        and runtime_user.endswith("_rw")
        and runtime_user != expected_rw_role
    ):
        raise DatabaseConnectionError(
            f"Configured database user {runtime_user!r} looks scoped to a different inferred project database. "
            f"For {settings.dbname!r}, use {expected_rw_role!r}'s credentials after initialization, or unset "
            "PCI_DATABASE_USER/PCI_DATABASE_PASSWORD and provide "
            "PCI_DATABASE_ADMIN_USER/PCI_DATABASE_ADMIN_PASSWORD "
            "so pci-index can initialize this project database."
        )


def _connect_existing_inferred_database_with_scoped_role(
    settings: DatabaseSettings, *, dbname: str, rw_role_name: str
) -> DatabaseBootstrapResult:
    try:
        with connect(readonly=False, settings=settings):
            pass
    except DatabaseConnectionError as exc:
        raise DatabaseConnectionError(
            f"Configured database user {rw_role_name!r} matches the inferred project RW role for {dbname!r}, "
            "but it could not connect to that project database. Run pci-index --init-db with "
            "PCI_DATABASE_ADMIN_USER/PCI_DATABASE_ADMIN_PASSWORD, "
            "or use credentials with CREATEDB/CREATEROLE to initialize the database first.\n" + str(exc)
        ) from exc
    return DatabaseBootstrapResult(
        dbname=dbname,
        rw_role=DatabaseRole(
            name=rw_role_name,
            password=None,
            created=False,
            database_url=settings.role_database_url(rw_role_name),
        ),
    )


def database_exists(conn: DbConnection, dbname: str) -> bool:
    _validate_identifier(dbname, "database name")
    return _database_exists(conn, dbname)


def _same_database_connection_settings(left: DatabaseSettings, right: DatabaseSettings) -> bool:
    return (
        left.dsn == right.dsn
        and left.dsn_user == right.dsn_user
        and left.dsn_password == right.dsn_password
        and left.host == right.host
        and left.port == right.port
        and left.dbname == right.dbname
        and left.user == right.user
        and left.password == right.password
        and left.sslmode == right.sslmode
    )


def _append_unique_database_settings(candidates: list[DatabaseSettings], candidate: DatabaseSettings) -> None:
    if not any(_same_database_connection_settings(existing, candidate) for existing in candidates):
        candidates.append(candidate)


def _settings_have_runtime_authentication(settings: DatabaseSettings) -> bool:
    if settings.dsn:
        parts = urlsplit(settings.dsn)
        return bool(settings.dsn_user or settings.dsn_password or parts.username or parts.password)
    return bool(settings.user and settings.password)


def _terminate_database_connections(conn: DbConnection, dbname: str) -> None:
    _ = conn.execute(
        """
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = %s
          AND pid <> pg_backend_pid()
        """,
        [dbname],
    )


def _initialize_inferred_database_privileges(
    target_settings: DatabaseSettings,
    *,
    dbname: str,
    rw_role: DatabaseRole | None,
    ro_role: DatabaseRole | None,
) -> None:
    with connect(readonly=False, settings=target_settings) as conn:
        _ = ensure_vector_extension(conn)
        if rw_role is not None and ro_role is not None:
            grant_project_database_access_privileges(conn, dbname=dbname, rw_role=rw_role.name, ro_role=ro_role.name)
            reassign_project_database_objects_to_rw_role(conn, rw_role.name)
        conn.commit()


def _target_database_init_candidates(
    settings: DatabaseSettings,
    bootstrap_settings: DatabaseSettings,
    *,
    dbname: str,
    target_fallback_settings: Sequence[DatabaseSettings],
) -> list[DatabaseSettings]:
    candidates = [settings_for_database(bootstrap_settings, dbname)]
    if _settings_have_runtime_authentication(settings):
        _append_unique_database_settings(candidates, settings_for_database(settings, dbname))
    for fallback_settings in target_fallback_settings:
        _append_unique_database_settings(candidates, settings_for_database(fallback_settings, dbname))
    return candidates


def _initialize_inferred_database_privileges_with_fallbacks(
    target_settings_candidates: Sequence[DatabaseSettings],
    *,
    dbname: str,
    rw_role: DatabaseRole | None,
    ro_role: DatabaseRole | None,
) -> None:
    target_errors: list[DatabaseConnectionError | PsycopgError] = []
    for target_settings in target_settings_candidates:
        try:
            _initialize_inferred_database_privileges(
                target_settings,
                dbname=dbname,
                rw_role=rw_role,
                ro_role=ro_role,
            )
        except (DatabaseConnectionError, PsycopgError) as exc:
            target_errors.append(exc)
            continue
        return
    final_error = target_errors[-1]
    first_error = f"\nFirst attempt: {target_errors[0]}" if len(target_errors) > 1 else ""
    config_guidance = (
        " Saved pci-index config stores the generated non-superuser PCI_DATABASE_ADMIN_* role. "
        "Legacy tables owned by another PostgreSQL role require that table owner or a PostgreSQL admin "
        "credential for one-time repair; add PCI_POSTGRES_ADMIN_USER/PCI_POSTGRES_ADMIN_PASSWORD to the "
        "private pci-index config or run with those variables set."
        if len(target_settings_candidates) == 1
        else ""
    )
    raise DatabaseConnectionError(
        "Could not initialize privileges for inferred PostgreSQL database "
        + repr(dbname)
        + " using "
        + connection_hint(target_settings_candidates[0])
        + ". The admin credentials must be able to use pgvector in new project databases. Run "
        "pci-doctor --init-postgres with real PostgreSQL admin credentials to install pgvector in template1, "
        "or use a database admin role that can run CREATE EXTENSION vector. If this inferred database was "
        "already created before template1 had pgvector, drop and recreate it with pci-index --reset ."
        + config_guidance
        + first_error
        + "\n"
        + str(final_error)
    ) from final_error


def bootstrap_inferred_database(
    settings: DatabaseSettings, *, target_fallback_settings: Sequence[DatabaseSettings] = ()
) -> DatabaseBootstrapResult:
    dbname = _ensure_inferred_database_target(settings, operation="bootstrap")
    rw_role_name = project_database_role_name(dbname, "rw")
    ro_role_name = project_database_role_name(dbname, "ro")
    has_explicit_admin = bool(settings.admin_user and settings.admin_password)
    if not has_explicit_admin:
        _reject_other_project_scoped_runtime_role(settings, rw_role_name)
        if configured_database_user(settings) == rw_role_name:
            return _connect_existing_inferred_database_with_scoped_role(
                settings, dbname=dbname, rw_role_name=rw_role_name
            )
        settings = writer_admin_fallback(settings)
    bootstrap_settings = _bootstrap_connection_settings(settings)
    maintenance_settings = maintenance_database_settings(settings)
    create_project_roles = bool(settings.admin_user and settings.admin_password)
    try:
        create_context = InferredDatabaseCreateContext(
            dbname=dbname,
            rw_role_name=rw_role_name,
            ro_role_name=ro_role_name,
            create_project_roles=create_project_roles,
        )
        database_created, rw_role, ro_role = _create_inferred_database(settings, maintenance_settings, create_context)
    except (DatabaseConnectionError, PsycopgError) as exc:
        raise DatabaseConnectionError(
            "Could not bootstrap inferred PostgreSQL database "
            + repr(dbname)
            + " using "
            + connection_hint(maintenance_settings)
            + ". Set PCI_DATABASE_ADMIN_USER/"
            "PCI_DATABASE_ADMIN_PASSWORD, use credentials with CREATEDB/CREATEROLE and "
            "permission to create the vector extension, or set PCI_DATABASE_URL/PCI_PG_DB "
            "to an existing database.\n" + str(exc)
        ) from exc

    _initialize_inferred_database_privileges_with_fallbacks(
        _target_database_init_candidates(
            settings,
            bootstrap_settings,
            dbname=dbname,
            target_fallback_settings=target_fallback_settings,
        ),
        dbname=dbname,
        rw_role=rw_role,
        ro_role=ro_role,
    )

    return DatabaseBootstrapResult(
        dbname=dbname,
        database_created=database_created,
        rw_role=rw_role,
        ro_role=ro_role,
    )


def _create_inferred_database(
    settings: DatabaseSettings,
    maintenance_settings: DatabaseSettings,
    create_context: InferredDatabaseCreateContext,
) -> tuple[bool, DatabaseRole | None, DatabaseRole | None]:
    connect_autocommit = cast("AutocommitConnect", Connection[DictRow].connect)
    with connect_autocommit(conninfo(maintenance_settings), autocommit=True, row_factory=DICT_ROW_FACTORY) as conn:
        rw_role = (
            _ensure_login_role(conn, settings, create_context.rw_role_name)
            if create_context.create_project_roles
            else None
        )
        ro_role = (
            _ensure_login_role(conn, settings, create_context.ro_role_name)
            if create_context.create_project_roles
            else None
        )
        database_created = False
        if not _database_exists(conn, create_context.dbname):
            _ = conn.execute(create_database_sql(create_context.dbname))
            database_created = True
    return database_created, rw_role, ro_role


def bootstrap_postgres_roles(
    settings: DatabaseSettings, *, role_name: str = DEFAULT_POSTGRES_INDEX_ADMIN_ROLE
) -> PostgresBootstrapResult:
    if not settings.admin_user or not settings.admin_password:
        raise DatabaseConnectionError(
            "Set PCI_POSTGRES_ADMIN_USER and PCI_POSTGRES_ADMIN_PASSWORD to bootstrap PostgreSQL roles."
        )
    if settings.admin_user == role_name:
        raise DatabaseConnectionError(
            f"PCI_POSTGRES_ADMIN_USER must be a PostgreSQL admin role, not {role_name!r}. "
            "Use a real PostgreSQL admin role such as 'postgres' so pci-doctor can create or reset "
            f"{role_name!r}, set CREATEDB/CREATEROLE, and install pgvector into template1."
        )
    maintenance_settings = maintenance_database_settings(settings)
    template_settings = settings_for_database(_bootstrap_connection_settings(settings), DEFAULT_TEMPLATE_DB)
    connect_autocommit = cast("AutocommitConnect", Connection[DictRow].connect)
    try:
        with connect_autocommit(conninfo(maintenance_settings), autocommit=True, row_factory=DICT_ROW_FACTORY) as conn:
            index_role = _ensure_index_admin_role(conn, maintenance_settings, role_name)
        with connect_autocommit(conninfo(template_settings), autocommit=True, row_factory=DICT_ROW_FACTORY) as conn:
            vector_template_created = ensure_vector_extension(conn)
    except (DatabaseConnectionError, PsycopgError) as exc:
        raise DatabaseConnectionError(
            "Could not bootstrap PostgreSQL roles using "
            + postgres_bootstrap_connection_hint(settings)
            + ". Run pci-doctor --init-postgres with PCI_POSTGRES_ADMIN_USER/"
            "PCI_POSTGRES_ADMIN_PASSWORD credentials that have CREATEROLE and can run "
            "CREATE EXTENSION vector in template1.\n" + str(exc)
        ) from exc
    return PostgresBootstrapResult(
        postgres_url=settings.postgres_url(),
        index_role=index_role,
        vector_template_ready=True,
        vector_template_created=vector_template_created,
    )


def drop_inferred_database(settings: DatabaseSettings) -> DatabaseBootstrapResult:
    dbname = _ensure_inferred_database_target(settings, operation="drop")
    maintenance_settings = maintenance_database_settings(settings)
    connect_autocommit = cast("AutocommitConnect", Connection[DictRow].connect)
    try:
        with connect_autocommit(conninfo(maintenance_settings), autocommit=True, row_factory=DICT_ROW_FACTORY) as conn:
            database_dropped = _database_exists(conn, dbname)
            if database_dropped:
                _terminate_database_connections(conn, dbname)
                _ = conn.execute(drop_database_sql(dbname))
    except (DatabaseConnectionError, PsycopgError) as exc:
        raise DatabaseConnectionError(
            "Could not drop inferred PostgreSQL database "
            + repr(dbname)
            + " using "
            + connection_hint(maintenance_settings)
            + ". Set PCI_DATABASE_ADMIN_USER/"
            "PCI_DATABASE_ADMIN_PASSWORD or use credentials with DROP DATABASE privilege.\n" + str(exc)
        ) from exc
    return DatabaseBootstrapResult(dbname=dbname, database_dropped=database_dropped)


def writable_settings_for_bootstrap(settings: DatabaseSettings, bootstrap: DatabaseBootstrapResult) -> DatabaseSettings:
    rw_role = bootstrap.rw_role
    if rw_role is None:
        return settings
    if rw_role.password:
        return settings_with_credentials(settings, rw_role.name, rw_role.password)
    if configured_database_user(settings) == rw_role.name:
        return settings
    raise DatabaseConnectionError(
        "The inferred PostgreSQL RW role already exists, but its password is not available to this process. "
        "Set PCI_DATABASE_USER/PCI_DATABASE_PASSWORD to "
        f"{rw_role.name!r}'s credentials, or drop the PCI-managed database roles and run pci-index again."
    )


def connection_configuration_guidance(settings: DatabaseSettings) -> str:
    if settings.dsn_source == "PCI_MCP_DATABASE_URL":
        return (
            "Set PCI_MCP_DATABASE_URL, "
            "PCI_MCP_DATABASE_USER, "
            "PCI_MCP_DATABASE_PASSWORD, and "
            "PCI_DATABASE_SCOPE_PATH for pci-mcp, or configure generic database credentials."
        )
    return "Set PCI_DATABASE_URL, or set PCI_PG_HOST/PCI_PG_PORT/PCI_PG_DB/PCI_PG_USER/PCI_PG_PASS for your database."


def _validate_project_database_privilege_inputs(*, dbname: str, rw_role: str, ro_role: str) -> None:
    for identifier, kind in ((dbname, "database name"), (rw_role, "role name"), (ro_role, "role name")):
        _validate_identifier(identifier, kind)


def grant_project_database_access_privileges(conn: DbConnection, *, dbname: str, rw_role: str, ro_role: str) -> None:
    _validate_project_database_privilege_inputs(dbname=dbname, rw_role=rw_role, ro_role=ro_role)
    _ = conn.execute(SQL(cast("LiteralString", f"GRANT CONNECT ON DATABASE {dbname} TO {rw_role}, {ro_role}")))
    _ = conn.execute(SQL(cast("LiteralString", f"GRANT USAGE, CREATE ON SCHEMA public TO {rw_role}")))
    _ = conn.execute(SQL(cast("LiteralString", f"GRANT USAGE ON SCHEMA public TO {ro_role}")))


def reassign_project_database_objects_to_rw_role(conn: DbConnection, rw_role: str) -> None:
    """Transfer ownership of legacy PCI tables, sequences, and functions to the per-project rw role.

    Existing project databases created before per-project roles existed have objects owned by
    whichever writer ran first (typically the bundled `codeintel` superuser). When we later
    introduce a scoped rw role and switch the runtime writer to it, ensure_schema's
    `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` and `CREATE OR REPLACE FUNCTION` fail with
    "must be owner" even when the object is already up to date. Reassigning ownership here is
    idempotent and one-shot: on freshly created DBs the rw role already owns everything, so
    each ALTER is a no-op.
    """
    _validate_identifier(rw_role, "role name")
    table_rows = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'project_code_intel_%'"
    ).fetchall()
    for row in table_rows:
        name = str(row["tablename"])
        _validate_identifier(name, "table name")
        _ = conn.execute(SQL(cast("LiteralString", f"ALTER TABLE public.{name} OWNER TO {rw_role}")))
    sequence_rows = conn.execute(
        "SELECT sequencename FROM pg_sequences WHERE schemaname = 'public' AND sequencename LIKE 'project_code_intel_%'"
    ).fetchall()
    for row in sequence_rows:
        name = str(row["sequencename"])
        _validate_identifier(name, "sequence name")
        _ = conn.execute(SQL(cast("LiteralString", f"ALTER SEQUENCE public.{name} OWNER TO {rw_role}")))
    function_rows = conn.execute(
        """
        SELECT p.proname AS proname, pg_get_function_identity_arguments(p.oid) AS args
        FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public' AND p.proname LIKE 'project_code_intel_%'
        """
    ).fetchall()
    for row in function_rows:
        name = str(row["proname"])
        args = str(row["args"] or "")
        _validate_identifier(name, "function name")
        # pg_get_function_identity_arguments returns type names from a system catalog, but we
        # still gate interpolation to a conservative charset (identifier chars, commas, spaces,
        # array brackets, schema-qualifier dot). Empty string is valid for no-arg functions.
        if any(ch not in _PG_IDENTITY_ARG_CHARS for ch in args):
            raise DatabaseConnectionError(
                f"refusing to reassign function with unsafe identity arguments: public.{name}({args!r})"
            )
        _ = conn.execute(SQL(cast("LiteralString", f"ALTER FUNCTION public.{name}({args}) OWNER TO {rw_role}")))


def grant_project_database_object_privileges(conn: DbConnection, *, dbname: str, rw_role: str, ro_role: str) -> None:
    _validate_project_database_privilege_inputs(dbname=dbname, rw_role=rw_role, ro_role=ro_role)
    _ = conn.execute(
        SQL(cast("LiteralString", f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {rw_role}"))
    )
    _ = conn.execute(
        SQL(cast("LiteralString", f"GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO {rw_role}"))
    )
    _ = conn.execute(SQL(cast("LiteralString", f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {ro_role}")))
    _ = conn.execute(SQL(cast("LiteralString", f"GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO {ro_role}")))
    _ = conn.execute(
        SQL(cast("LiteralString", f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {ro_role}"))
    )
    _ = conn.execute(
        SQL(cast("LiteralString", f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON SEQUENCES TO {ro_role}"))
    )


def connect(*, readonly: bool | None = None, settings: DatabaseSettings | None = None) -> DbConnection:
    settings = settings or DatabaseSettings.from_env()
    if readonly is None:
        readonly = not allow_writes(settings)

    try:
        conn = Connection[DictRow].connect(conninfo(settings), row_factory=DICT_ROW_FACTORY)
    except OperationalError as exc:
        inferred_note = (
            f" The database name {settings.dbname!r} was inferred from the working directory; "
            "run from the project root or set PCI_DATABASE_SCOPE_PATH."
            if settings.database_inferred and settings.dbname
            else ""
        )
        raise DatabaseConnectionError(
            "Could not connect to PostgreSQL/pgvector using "
            + connection_hint(settings)
            + ". "
            + connection_configuration_guidance(settings)
            + inferred_note
            + "\n"
            + str(exc)
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


def compact_json(value: object, *, default: Callable[[object], object] | None = None) -> str:
    """Deterministic compact JSON for jsonb columns: sorted keys, no whitespace.

    ``default`` forwards to ``json.dumps`` (pass ``str`` to coerce values that
    are not natively JSON-serializable). Unlike ``json_metadata`` this accepts
    any JSON value, not only objects.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=default)
