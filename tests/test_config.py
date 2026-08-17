from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from project_code_intelligence import common, config
from project_code_intelligence.exceptions import ConfigError


class UserConfigTests(unittest.TestCase):
    def test_write_pci_index_user_config_uses_xdg_config_home_and_private_permissions(self) -> None:
        credential = " ".join(("secret", "value"))
        with tempfile.TemporaryDirectory() as directory:
            path = config.write_pci_index_user_config(
                database_url="postgresql://db.example.invalid:5432?sslmode=prefer",
                database_admin_user="pci_index_admin",
                database_admin_password=credential,
                env={"XDG_CONFIG_HOME": directory},
            )

            self.assertEqual(path, Path(directory) / "project-code-intelligence" / "pci-index.env")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
            text = path.read_text(encoding="utf-8")
            self.assertIn("PCI_DATABASE_URL=", text)
            self.assertIn("PCI_DATABASE_ADMIN_USER=pci_index_admin", text)
            self.assertIn("PCI_DATABASE_ADMIN_PASSWORD='secret value'", text)

    def test_load_pci_index_user_config_sets_missing_values_without_overriding_environment(self) -> None:
        credential = " ".join(("secret", "value"))
        with tempfile.TemporaryDirectory() as directory:
            _ = config.write_pci_index_user_config(
                database_url="postgresql://db.example.invalid:5432?sslmode=prefer",
                database_admin_user="pci_index_admin",
                database_admin_password=credential,
                env={"XDG_CONFIG_HOME": directory},
            )
            env = {
                "XDG_CONFIG_HOME": directory,
                "PCI_DATABASE_ADMIN_USER": "explicit_admin",
            }

            result = config.load_pci_index_user_config(env)

            self.assertIsNotNone(result)
            self.assertEqual(env["PCI_DATABASE_URL"], "postgresql://db.example.invalid:5432?sslmode=prefer")
            self.assertEqual(env["PCI_DATABASE_ADMIN_USER"], "explicit_admin")
            self.assertEqual(env["PCI_DATABASE_ADMIN_PASSWORD"], "secret value")
            self.assertIn("PCI_DATABASE_ADMIN_USER", result.skipped if result else ())

    def test_load_pci_index_user_config_accepts_postgres_admin_repair_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project-code-intelligence" / "pci-index.env"
            path.parent.mkdir()
            _ = path.write_text(
                "\n".join((
                    "PCI_DATABASE_URL=postgresql://db.example.invalid:5432?sslmode=prefer",
                    "PCI_DATABASE_ADMIN_USER=pci_index_admin",
                    "PCI_DATABASE_ADMIN_PASSWORD=index-secret",
                    "PCI_POSTGRES_ADMIN_USER=postgres",
                    "PCI_POSTGRES_ADMIN_PASSWORD=postgres-secret",
                    "",
                )),
                encoding="utf-8",
            )
            path.chmod(0o600)
            env = {"XDG_CONFIG_HOME": directory}

            result = config.load_pci_index_user_config(env)

            self.assertIsNotNone(result)
            self.assertEqual(env["PCI_POSTGRES_ADMIN_USER"], "postgres")
            self.assertEqual(env["PCI_POSTGRES_ADMIN_PASSWORD"], "postgres-secret")

    def test_load_pci_index_user_config_refuses_group_or_world_readable_file(self) -> None:
        credential = " ".join(("secret", "value"))
        with tempfile.TemporaryDirectory() as directory:
            path = config.write_pci_index_user_config(
                database_url="postgresql://db.example.invalid:5432?sslmode=prefer",
                database_admin_user="pci_index_admin",
                database_admin_password=credential,
                env={"XDG_CONFIG_HOME": directory},
            )
            path.chmod(0o644)

            with self.assertRaises(ConfigError):
                _ = config.load_pci_index_user_config({"XDG_CONFIG_HOME": directory})


class DatabaseScopePathTests(unittest.TestCase):
    @staticmethod
    def _write_workspace(directory: str) -> tuple[Path, Path]:
        """A multi-repo workspace with a nested child repo, both carrying the
        reindex marker a real `pci index` run writes into every indexed repo."""
        workspace = Path(directory) / "big-workspace"
        child = workspace / "child-repo"
        for repo in (workspace, child):
            (repo / ".git").mkdir(parents=True)
        marker = json.dumps({
            "cwd": str(workspace),
            "repo_paths": [str(workspace), str(child)],
            "collection": "big-workspace",
        })
        for repo in (workspace, child):
            _ = (repo / ".git" / common.REINDEX_MARKER_NAME).write_text(marker, encoding="utf-8")
        return workspace, child

    def test_scope_path_in_nested_repo_resolves_to_the_workspace_invocation(self) -> None:
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            workspace, child = self._write_workspace(directory)
            deep = child / "src" / "inner"
            deep.mkdir(parents=True)
            try:
                os.chdir(deep)
                scope = config.configured_database_scope_path(env={})
            finally:
                os.chdir(old_cwd)
            self.assertEqual(scope, workspace.resolve())
            self.assertEqual(common.default_database_name(scope), common.default_database_name(workspace))

    def test_explicit_scope_env_overrides_the_marker(self) -> None:
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            _, child = self._write_workspace(directory)
            try:
                os.chdir(child)
                scope = config.configured_database_scope_path(env={config.DATABASE_SCOPE_PATH_ENV: str(child)})
            finally:
                os.chdir(old_cwd)
            self.assertEqual(scope, child.resolve())

    def test_scope_path_without_marker_stays_the_working_directory(self) -> None:
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / "plain-repo"
            (plain / ".git").mkdir(parents=True)
            try:
                os.chdir(plain)
                scope = config.configured_database_scope_path(env={})
            finally:
                os.chdir(old_cwd)
            self.assertEqual(scope, plain.resolve())

    def test_single_repo_marker_scopes_to_that_repo_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            repo = root / "only-repo"
            repo.mkdir(parents=True)
            scope = common.database_scope_path_for_marker(root, [str(repo)])
            self.assertEqual(scope, repo.resolve())


if __name__ == "__main__":
    _ = unittest.main()
