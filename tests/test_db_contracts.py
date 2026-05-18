from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

from project_code_intelligence import db
from project_code_intelligence.common import default_database_name
from project_code_intelligence.config import DatabaseSettings
from project_code_intelligence.db import (
    DEFAULT_POSTGRES_INDEX_ADMIN_ROLE,
    DatabaseBootstrapResult,
    DatabaseConnectionError,
    DatabaseRole,
    bootstrap_postgres_roles,
    conninfo,
    create_role_sql,
    drop_inferred_database,
    inferred_database_role_settings,
    json_metadata,
    postgres_bootstrap_role_password,
    postgres_string_literal,
    project_database_role_name,
    project_database_role_password,
    require_row,
    vector_literal,
    writable_settings_for_bootstrap,
)


class _FakeCursor:
    def __init__(self, row: dict[str, object] | None = None, rows: list[dict[str, object]] | None = None) -> None:
        self._row = row
        self._rows = rows or ([] if row is None else [row])

    def fetchone(self) -> dict[str, object] | None:
        return self._row

    def fetchall(self) -> list[dict[str, object]]:
        return list(self._rows)


@dataclass(frozen=True)
class _FakePgCatalog:
    """Bundle of pg_catalog rows returned by `_FakeRoleBootstrapConnection.execute`."""

    tables: list[str] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)
    functions: list[tuple[str, str]] = field(default_factory=list)


_DEFAULT_FAKE_ROLE_ATTRIBUTES: dict[str, object] = {
    "rolcanlogin": True,
    "rolcreatedb": True,
    "rolcreaterole": True,
}


class _FakeRoleBootstrapConnection:
    def __init__(
        self,
        *,
        database_exists: bool = True,
        extension_exists: bool = False,
        role_exists: bool = False,
        catalog: _FakePgCatalog | None = None,
    ) -> None:
        self.statements: list[str] = []
        self.database_exists = database_exists
        self.extension_exists = extension_exists
        self.role_exists = role_exists
        self.catalog = catalog or _FakePgCatalog()

    def __enter__(self) -> _FakeRoleBootstrapConnection:
        return self

    def __exit__(self, _exc_type: object, exc: object, traceback: object) -> None:
        return None

    def commit(self) -> None:
        self.statements.append("COMMIT")

    def execute(self, query: object, params: object | None = None) -> _FakeCursor:
        _ = params
        text = str(query)
        self.statements.append(text)
        return self._cursor_for_query(text)

    def _cursor_for_query(self, text: str) -> _FakeCursor:
        # rolcanlogin probes carry both "rolcanlogin" and "FROM pg_roles"; check first so the
        # generic pg_roles branch below doesn't claim them.
        if "rolcanlogin" in text and "FROM pg_roles" in text:
            return _FakeCursor(_DEFAULT_FAKE_ROLE_ATTRIBUTES if self.role_exists else None)
        existence_probes = (
            ("FROM pg_roles", self.role_exists),
            ("FROM pg_database", self.database_exists),
            ("FROM pg_extension", self.extension_exists),
        )
        for needle, present in existence_probes:
            if needle in text:
                return _FakeCursor({"ok": 1} if present else None)
        catalog_probes: tuple[tuple[str, list[dict[str, object]]], ...] = (
            ("FROM pg_tables", [{"tablename": name} for name in self.catalog.tables]),
            ("FROM pg_sequences", [{"sequencename": name} for name in self.catalog.sequences]),
            ("FROM pg_proc", [{"proname": name, "args": args} for name, args in self.catalog.functions]),
        )
        for needle, rows in catalog_probes:
            if needle in text:
                return _FakeCursor(rows=rows)
        return _FakeCursor({"ok": 1})


class DatabaseContractTests(unittest.TestCase):
    TEST_CREDENTIAL = "test-db-credential"

    def test_default_database_name_is_postgres_safe_and_path_stable(self) -> None:
        path = Path("one/Project Code Intelligence!")
        sibling = Path("two/Project Code Intelligence!")

        name = default_database_name(path)

        self.assertRegex(name, r"^pci_project_code_intelligence_[0-9a-f]{8}$")
        self.assertLessEqual(len(name), 63)
        self.assertEqual(name, default_database_name(path))
        self.assertNotEqual(name, default_database_name(sibling))

    def test_conninfo_uses_dsn_or_complete_parts(self) -> None:
        dsn_text = conninfo(DatabaseSettings(dsn="postgresql://example.invalid/db"))
        self.assertIn("host=example.invalid", dsn_text)
        self.assertIn("dbname=db", dsn_text)
        self.assertIn("connect_timeout=10", dsn_text)
        credential = "p"

        text = conninfo(DatabaseSettings(host="db", port="5432", dbname="codeintel", user="u", password=credential))

        self.assertIn("host=db", text)
        self.assertIn("dbname=codeintel", text)
        self.assertIn("user=u", text)
        self.assertIn("connect_timeout=10", text)
        self.assertIn("keepalives=1", text)

    def test_conninfo_can_add_credentials_to_database_url(self) -> None:
        credential = "secret"
        text = conninfo(
            DatabaseSettings(
                dsn="postgresql://db.example.invalid/codeintel?sslmode=prefer",
                dsn_user="app",
                dsn_password=credential,
            )
        )

        self.assertIn("host=db.example.invalid", text)
        self.assertIn("dbname=codeintel", text)
        self.assertIn("user=app", text)
        self.assertIn(f"password={credential}", text)
        self.assertIn("sslmode=prefer", text)

    def test_database_admin_credentials_are_parsed_without_affecting_normal_conninfo(self) -> None:
        admin_credential = "-".join(("admin", "credential"))
        settings = DatabaseSettings.from_env({
            "PCI_DATABASE_URL": "postgresql://db.example.invalid/codeintel",
            "PCI_DATABASE_ADMIN_USER": "postgres",
            "PCI_DATABASE_ADMIN_PASSWORD": admin_credential,
        })

        self.assertEqual(settings.admin_user, "postgres")
        self.assertEqual(settings.admin_password, admin_credential)
        self.assertNotIn(admin_credential, conninfo(settings))

    def test_role_database_url_renders_scope_in_userinfo(self) -> None:
        settings = DatabaseSettings.from_env({
            "PCI_DATABASE_URL": "postgresql://db.example.invalid/codeintel?sslmode=prefer"
        })

        url = settings.role_database_url("pci_demo_rw", "secret")

        self.assertEqual(url, "postgresql://pci_demo_rw:secret@db.example.invalid/codeintel?sslmode=prefer")

    def test_project_database_role_names_fit_postgres_identifier_limit(self) -> None:
        dbname = "pci_" + ("x" * 59)

        rw_role = project_database_role_name(dbname, "rw")
        ro_role = project_database_role_name(dbname, "ro")

        self.assertLessEqual(len(rw_role), 63)
        self.assertLessEqual(len(ro_role), 63)
        self.assertTrue(rw_role.endswith("_rw"))
        self.assertTrue(ro_role.endswith("_ro"))
        self.assertNotEqual(rw_role, ro_role)

    def test_create_role_sql_renders_password_literal(self) -> None:
        literal = postgres_string_literal("password 'with' \\ spaces")

        self.assertEqual(literal, r"E'password \'with\' \\ spaces'")
        self.assertIsNotNone(create_role_sql("pci_demo_rw", "password with spaces"))

    def test_project_database_role_password_is_stable_for_admin_secret(self) -> None:
        first = project_database_role_password("pci_demo", "pci_demo_rw", self.TEST_CREDENTIAL)
        second = project_database_role_password("pci_demo", "pci_demo_rw", self.TEST_CREDENTIAL)
        other_role = project_database_role_password("pci_demo", "pci_demo_ro", self.TEST_CREDENTIAL)
        other_db = project_database_role_password("pci_other", "pci_other_rw", self.TEST_CREDENTIAL)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("pci_"))
        self.assertNotEqual(first, other_role)
        self.assertNotEqual(first, other_db)

    def test_postgres_bootstrap_role_password_is_stable_for_admin_secret(self) -> None:
        first = postgres_bootstrap_role_password(DEFAULT_POSTGRES_INDEX_ADMIN_ROLE, self.TEST_CREDENTIAL)
        second = postgres_bootstrap_role_password(DEFAULT_POSTGRES_INDEX_ADMIN_ROLE, self.TEST_CREDENTIAL)
        other_role = postgres_bootstrap_role_password("pci_other_admin", self.TEST_CREDENTIAL)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("pci_"))
        self.assertNotEqual(first, other_role)

    def test_inferred_database_role_settings_derives_scoped_role_from_admin_secret(self) -> None:
        settings = DatabaseSettings(
            dsn="postgresql://db.example.invalid/pci_demo?sslmode=prefer",
            dbname="pci_demo",
            admin_user="postgres",
            admin_password=self.TEST_CREDENTIAL,
            database_inferred=True,
        )

        writer = inferred_database_role_settings(settings, "rw")
        reader = inferred_database_role_settings(settings, "ro")

        self.assertEqual(writer.dsn_user, "pci_demo_rw")
        self.assertEqual(reader.dsn_user, "pci_demo_ro")
        self.assertEqual(
            writer.dsn_password,
            project_database_role_password("pci_demo", "pci_demo_rw", self.TEST_CREDENTIAL),
        )

    def test_inferred_database_role_settings_replaces_url_credentials_with_scoped_role(self) -> None:
        settings = DatabaseSettings(
            dsn="postgresql://app:credential@db.example.invalid/pci_demo?sslmode=prefer",
            dbname="pci_demo",
            admin_user="postgres",
            admin_password=self.TEST_CREDENTIAL,
            database_inferred=True,
        )

        writer = inferred_database_role_settings(settings, "rw")

        self.assertIsNot(writer, settings)
        self.assertEqual(writer.dsn_user, "pci_demo_rw")
        self.assertEqual(
            writer.dsn_password,
            project_database_role_password("pci_demo", "pci_demo_rw", self.TEST_CREDENTIAL),
        )

    def test_inferred_database_role_settings_preserves_separate_runtime_credentials(self) -> None:
        settings = DatabaseSettings(
            dsn="postgresql://db.example.invalid/pci_demo?sslmode=prefer",
            dsn_user="runtime_user",
            dsn_password=self.TEST_CREDENTIAL,
            dbname="pci_demo",
            admin_user="postgres",
            admin_password=self.TEST_CREDENTIAL,
            database_inferred=True,
        )

        writer = inferred_database_role_settings(settings, "rw")

        self.assertIs(writer, settings)

    def test_explicit_database_is_not_dropped_as_pci_managed(self) -> None:
        with self.assertRaises(DatabaseConnectionError):
            _ = drop_inferred_database(DatabaseSettings(dbname="shared", database_inferred=False))

    def test_bootstrap_rejects_runtime_role_from_another_inferred_database(self) -> None:
        settings = DatabaseSettings(
            dsn="postgresql://db.example.invalid/pci_zod?sslmode=prefer",
            dsn_user="pci_project_code_intelligence_38fc61c9_rw",
            dsn_password=self.TEST_CREDENTIAL,
            dbname="pci_zod",
            database_inferred=True,
        )

        with self.assertRaises(DatabaseConnectionError) as raised:
            _ = db.bootstrap_inferred_database(settings)

        self.assertIn("looks scoped to a different inferred project database", str(raised.exception))

    def test_writable_settings_use_generated_rw_role_credentials(self) -> None:
        settings = DatabaseSettings(dsn="postgresql://db.example.invalid/pci_demo")
        bootstrap = DatabaseBootstrapResult(
            dbname="pci_demo",
            rw_role=DatabaseRole(
                name="pci_demo_rw",
                password=self.TEST_CREDENTIAL,
                created=True,
                database_url=f"postgresql://pci_demo_rw:{self.TEST_CREDENTIAL}@db.example.invalid/pci_demo",
            ),
        )

        writer = writable_settings_for_bootstrap(settings, bootstrap)

        self.assertEqual(writer.dsn_user, "pci_demo_rw")
        self.assertEqual(writer.dsn_password, self.TEST_CREDENTIAL)

    def test_postgres_role_bootstrap_does_not_create_project_database(self) -> None:
        settings = DatabaseSettings(
            dsn="postgresql://db.example.invalid?sslmode=prefer",
            admin_user="postgres",
            admin_password=self.TEST_CREDENTIAL,
            database_inferred=True,
        )
        maintenance_connection = _FakeRoleBootstrapConnection()
        template_connection = _FakeRoleBootstrapConnection()

        with (
            patch(
                "project_code_intelligence.db.Connection.connect",
                side_effect=[maintenance_connection, template_connection],
            ),
            patch(
                "project_code_intelligence.db.create_database_sql",
                side_effect=AssertionError("role bootstrap must not create databases"),
            ),
        ):
            bootstrap = bootstrap_postgres_roles(settings)

        self.assertEqual(bootstrap.postgres_url, "postgresql://db.example.invalid?sslmode=prefer")
        self.assertEqual(bootstrap.index_role.name, DEFAULT_POSTGRES_INDEX_ADMIN_ROLE)
        self.assertTrue(bootstrap.vector_template_ready)
        self.assertTrue(bootstrap.vector_template_created)
        maintenance_statements = "\n".join(maintenance_connection.statements)
        template_statements = "\n".join(template_connection.statements)
        self.assertIn("CREATE ROLE", maintenance_statements)
        self.assertIn("CREATEDB", maintenance_statements)
        self.assertIn("CREATEROLE", maintenance_statements)
        self.assertIn("CREATE EXTENSION IF NOT EXISTS vector", template_statements)
        self.assertNotIn("CREATE DATABASE", maintenance_statements + "\n" + template_statements)

    def test_conninfo_reports_missing_connection_parts(self) -> None:
        credential = "p"
        with self.assertRaises(DatabaseConnectionError):
            _ = conninfo(DatabaseSettings(dbname=None, user="u", password=credential))

    def test_require_row_rejects_empty_database_results(self) -> None:
        row = {"id": 1}

        self.assertEqual(require_row(row, "demo"), row)
        with self.assertRaises(RuntimeError):
            _ = require_row(None, "demo")

    def test_vector_literal_accepts_only_non_empty_numeric_lists(self) -> None:
        self.assertEqual(vector_literal([1, 2.5, -3]), "[1,2.5,-3]")

        with self.assertRaises(ValueError):
            _ = vector_literal([])

        for value in ((1, 2), [True], ["1"]):
            with self.subTest(value=value), self.assertRaises(TypeError):
                _ = vector_literal(value)

    def test_json_metadata_uses_stable_object_encoding(self) -> None:
        self.assertEqual(json_metadata(None), "{}")
        self.assertEqual(json.loads(json_metadata({"b": 2, "a": 1})), {"a": 1, "b": 2})

        with self.assertRaises(TypeError):
            _ = json_metadata(["not", "an", "object"])


class DatabaseGuidanceTests(unittest.TestCase):
    def test_mcp_connection_guidance_names_mcp_environment(self) -> None:
        settings = DatabaseSettings(
            dsn="postgresql://db.example.invalid/pci_demo?sslmode=prefer",
            dsn_source="PCI_MCP_DATABASE_URL",
        )

        message = db.connection_configuration_guidance(settings)

        self.assertIn("PCI_MCP_DATABASE_URL", message)
        self.assertIn("PCI_MCP_DATABASE_USER", message)
        self.assertIn("PCI_MCP_DATABASE_PASSWORD", message)
        self.assertIn("PCI_DATABASE_SCOPE_PATH", message)


class DatabaseBootstrapTests(unittest.TestCase):
    TEST_CREDENTIAL = "test-db-credential"

    def test_postgres_admin_credentials_are_parsed_separately_for_doctor_bootstrap(self) -> None:
        postgres_credential = "-".join(("postgres", "credential"))
        index_credential = "-".join(("index", "credential"))
        env = {
            "PCI_DATABASE_URL": "postgresql://db.example.invalid/codeintel",
            "PCI_POSTGRES_ADMIN_USER": "postgres",
            "PCI_POSTGRES_ADMIN_PASSWORD": postgres_credential,
            "PCI_DATABASE_ADMIN_USER": DEFAULT_POSTGRES_INDEX_ADMIN_ROLE,
            "PCI_DATABASE_ADMIN_PASSWORD": index_credential,
        }

        doctor_settings = DatabaseSettings.from_env(env, admin_scope="postgres")
        index_settings = DatabaseSettings.from_env(env)

        self.assertEqual(doctor_settings.admin_user, "postgres")
        self.assertEqual(doctor_settings.admin_password, postgres_credential)
        self.assertEqual(index_settings.admin_user, DEFAULT_POSTGRES_INDEX_ADMIN_ROLE)
        self.assertEqual(index_settings.admin_password, index_credential)
        self.assertNotIn(postgres_credential, conninfo(doctor_settings))

    def test_postgres_role_bootstrap_rejects_generated_index_admin_as_bootstrap_admin(self) -> None:
        settings = DatabaseSettings(
            dsn="postgresql://db.example.invalid?sslmode=prefer",
            admin_user=DEFAULT_POSTGRES_INDEX_ADMIN_ROLE,
            admin_password=self.TEST_CREDENTIAL,
            database_inferred=True,
        )

        with (
            patch(
                "project_code_intelligence.db.Connection.connect",
                side_effect=AssertionError("generated index admin should fail before connecting"),
            ),
            self.assertRaises(DatabaseConnectionError) as raised,
        ):
            _ = bootstrap_postgres_roles(settings)

        self.assertIn("PCI_POSTGRES_ADMIN_USER must be a PostgreSQL admin role", str(raised.exception))

    def test_postgres_role_bootstrap_resets_existing_index_admin_with_real_admin(self) -> None:
        settings = DatabaseSettings(
            dsn="postgresql://db.example.invalid?sslmode=prefer",
            admin_user="postgres",
            admin_password=self.TEST_CREDENTIAL,
            database_inferred=True,
        )
        maintenance_connection = _FakeRoleBootstrapConnection(role_exists=True)
        template_connection = _FakeRoleBootstrapConnection(extension_exists=True)

        with patch(
            "project_code_intelligence.db.Connection.connect",
            side_effect=[maintenance_connection, template_connection],
        ):
            bootstrap = bootstrap_postgres_roles(settings)

        self.assertEqual(bootstrap.index_role.name, DEFAULT_POSTGRES_INDEX_ADMIN_ROLE)
        self.assertEqual(
            bootstrap.index_role.password,
            db.postgres_bootstrap_role_password(DEFAULT_POSTGRES_INDEX_ADMIN_ROLE, self.TEST_CREDENTIAL),
        )
        self.assertFalse(bootstrap.index_role.created)
        maintenance_statements = "\n".join(maintenance_connection.statements)
        template_statements = "\n".join(template_connection.statements)
        self.assertIn("ALTER ROLE", maintenance_statements)
        self.assertIn("CREATEDB", maintenance_statements)
        self.assertIn("CREATEROLE", maintenance_statements)
        self.assertNotIn("CREATE ROLE", maintenance_statements)
        self.assertNotIn("CREATE EXTENSION", template_statements)

    def test_bootstrap_accepts_matching_runtime_role_for_existing_inferred_database(self) -> None:
        settings = DatabaseSettings(
            dsn="postgresql://db.example.invalid/pci_demo?sslmode=prefer",
            dsn_user="pci_demo_rw",
            dsn_password=self.TEST_CREDENTIAL,
            dbname="pci_demo",
            database_inferred=True,
        )
        connection = _FakeRoleBootstrapConnection()

        with (
            patch("project_code_intelligence.db.connect", return_value=connection),
            patch(
                "project_code_intelligence.db.Connection.connect",
                side_effect=AssertionError("matching scoped role should not need maintenance database bootstrap"),
            ),
        ):
            bootstrap = db.bootstrap_inferred_database(settings)

        self.assertFalse(bootstrap.database_created)
        self.assertIsNotNone(bootstrap.rw_role)
        self.assertEqual(bootstrap.rw_role.name if bootstrap.rw_role else None, "pci_demo_rw")

    def test_bootstrap_does_not_create_vector_extension_when_template_provided_it(self) -> None:
        settings = DatabaseSettings(
            dsn="postgresql://db.example.invalid/pci_demo?sslmode=prefer",
            dbname="pci_demo",
            admin_user="postgres",
            admin_password=self.TEST_CREDENTIAL,
            database_inferred=True,
        )
        maintenance_connection = _FakeRoleBootstrapConnection()
        target_connection = _FakeRoleBootstrapConnection(extension_exists=True)

        with (
            patch("project_code_intelligence.db.Connection.connect", return_value=maintenance_connection),
            patch("project_code_intelligence.db.connect", return_value=target_connection),
        ):
            bootstrap = db.bootstrap_inferred_database(settings)

        self.assertFalse(bootstrap.database_created)
        target_statements = "\n".join(target_connection.statements)
        self.assertIn("FROM pg_extension", target_statements)
        self.assertNotIn("CREATE EXTENSION", target_statements)

    def test_index_admin_reuses_existing_project_roles_without_altering_passwords(self) -> None:
        settings = DatabaseSettings(
            dsn="postgresql://db.example.invalid/pci_demo?sslmode=prefer",
            dbname="pci_demo",
            admin_user=DEFAULT_POSTGRES_INDEX_ADMIN_ROLE,
            admin_password=self.TEST_CREDENTIAL,
            database_inferred=True,
        )
        maintenance_connection = _FakeRoleBootstrapConnection(role_exists=True)
        target_connection = _FakeRoleBootstrapConnection(extension_exists=True)

        with (
            patch("project_code_intelligence.db.Connection.connect", return_value=maintenance_connection),
            patch("project_code_intelligence.db.connect", return_value=target_connection),
        ):
            bootstrap = db.bootstrap_inferred_database(settings)

        self.assertIsNotNone(bootstrap.rw_role)
        self.assertIsNotNone(bootstrap.ro_role)
        self.assertEqual(
            bootstrap.rw_role.password if bootstrap.rw_role else None,
            project_database_role_password("pci_demo", "pci_demo_rw", self.TEST_CREDENTIAL),
        )
        maintenance_statements = "\n".join(maintenance_connection.statements)
        self.assertNotIn("ALTER ROLE", maintenance_statements)
        self.assertNotIn("CREATE ROLE", maintenance_statements)

    def test_bootstrap_creates_project_roles_from_writer_credentials_when_admin_missing(self) -> None:
        # Bundled-local scenario: PCI_PG_USER/PASS supply codeintel:codeintel and the user has
        # not set PCI_DATABASE_ADMIN_*. The writer is the container's superuser
        # so we should still create per-project rw/ro roles for pci-mcp.
        writer_credential = "-".join(("codeintel", "writer"))
        settings = DatabaseSettings(
            host="127.0.0.1",
            port="5433",
            dbname="pci_demo",
            user="codeintel",
            password=writer_credential,
            database_inferred=True,
        )
        maintenance_connection = _FakeRoleBootstrapConnection()
        target_connection = _FakeRoleBootstrapConnection(extension_exists=True)

        with (
            patch("project_code_intelligence.db.Connection.connect", return_value=maintenance_connection),
            patch("project_code_intelligence.db.connect", return_value=target_connection),
        ):
            bootstrap = db.bootstrap_inferred_database(settings)

        self.assertIsNotNone(bootstrap.rw_role)
        self.assertIsNotNone(bootstrap.ro_role)
        self.assertEqual(
            bootstrap.rw_role.password if bootstrap.rw_role else None,
            project_database_role_password("pci_demo", "pci_demo_rw", writer_credential),
        )
        self.assertEqual(
            bootstrap.ro_role.password if bootstrap.ro_role else None,
            project_database_role_password("pci_demo", "pci_demo_ro", writer_credential),
        )
        maintenance_statements = "\n".join(maintenance_connection.statements)
        self.assertIn("CREATE ROLE pci_demo_rw", maintenance_statements)
        self.assertIn("CREATE ROLE pci_demo_ro", maintenance_statements)

    def test_writer_admin_fallback_is_noop_when_admin_already_set(self) -> None:
        settings = DatabaseSettings(
            host="127.0.0.1",
            dbname="pci_demo",
            user="codeintel",
            password=self.TEST_CREDENTIAL,
            admin_user="postgres",
            admin_password=self.TEST_CREDENTIAL,
            database_inferred=True,
        )

        result = db._writer_admin_fallback(settings)

        self.assertIs(result, settings)

    def test_bootstrap_reassigns_legacy_table_ownership_to_rw_role(self) -> None:
        # Pre-existing DB whose tables were populated by an old writer (e.g. codeintel) before
        # per-project roles existed. After we create the rw role and switch the runtime writer to
        # it, ensure_schema's ALTER TABLE statements would otherwise fail with 'must be owner'.
        writer_credential = "-".join(("codeintel", "writer"))
        settings = DatabaseSettings(
            host="127.0.0.1",
            port="5433",
            dbname="pci_demo",
            user="codeintel",
            password=writer_credential,
            database_inferred=True,
        )
        maintenance_connection = _FakeRoleBootstrapConnection()
        target_connection = _FakeRoleBootstrapConnection(
            extension_exists=True,
            catalog=_FakePgCatalog(
                tables=["project_code_intel_snapshots", "project_code_intel_files"],
                sequences=["project_code_intel_snapshots_id_seq"],
                functions=[("project_code_intel_touch_updated_at", "")],
            ),
        )

        with (
            patch("project_code_intelligence.db.Connection.connect", return_value=maintenance_connection),
            patch("project_code_intelligence.db.connect", return_value=target_connection),
        ):
            _ = db.bootstrap_inferred_database(settings)

        target_statements = "\n".join(target_connection.statements)
        self.assertIn("ALTER TABLE public.project_code_intel_snapshots OWNER TO pci_demo_rw", target_statements)
        self.assertIn("ALTER TABLE public.project_code_intel_files OWNER TO pci_demo_rw", target_statements)
        self.assertIn(
            "ALTER SEQUENCE public.project_code_intel_snapshots_id_seq OWNER TO pci_demo_rw",
            target_statements,
        )
        self.assertIn(
            "ALTER FUNCTION public.project_code_intel_touch_updated_at() OWNER TO pci_demo_rw",
            target_statements,
        )

    def test_writer_admin_fallback_is_noop_when_writer_credentials_missing(self) -> None:
        settings = DatabaseSettings(
            host="db.example.invalid",
            dbname="pci_demo",
            user=None,
            password=None,
            database_inferred=True,
        )

        result = db._writer_admin_fallback(settings)

        self.assertIs(result, settings)


if __name__ == "__main__":
    _ = unittest.main()
