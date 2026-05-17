from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path

from project_code_intelligence import config
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
            self.assertIn("PROJECT_CODE_INTELLIGENCE_DATABASE_URL=", text)
            self.assertIn("PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_USER=pci_index_admin", text)
            self.assertIn("PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_PASSWORD='secret value'", text)

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
                "PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_USER": "explicit_admin",
            }

            result = config.load_pci_index_user_config(env)

            self.assertIsNotNone(result)
            self.assertEqual(
                env["PROJECT_CODE_INTELLIGENCE_DATABASE_URL"], "postgresql://db.example.invalid:5432?sslmode=prefer"
            )
            self.assertEqual(env["PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_USER"], "explicit_admin")
            self.assertEqual(env["PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_PASSWORD"], "secret value")
            self.assertIn("PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_USER", result.skipped if result else ())

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


if __name__ == "__main__":
    _ = unittest.main()
