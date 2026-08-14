import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from project_code_intelligence import config, mcp_credentials


class McpCredentialTests(unittest.TestCase):
    def test_write_and_load_private_project_credentials(self) -> None:
        credential_value = " ".join(("secret", "fixture"))
        values = {
            "PCI_MCP_DATABASE_URL": "postgresql://localhost/pci_demo",
            "PCI_MCP_DATABASE_USER": "pci_demo_ro",
            "PCI_MCP_DATABASE_PASSWORD": credential_value,
            "PCI_COLLECTION": "demo",
            "PCI_DATABASE_SCOPE_PATH": "/work/demo",
        }
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"XDG_CONFIG_HOME": tmp}, clear=True):
            scope = Path(tmp) / "project"
            path = mcp_credentials.write(scope, values)

            self.assertFalse(path.is_relative_to(scope))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
            for name in values:
                _ = os.environ.pop(name, None)

            self.assertEqual(mcp_credentials.load(scope), path)
            self.assertEqual({name: os.environ[name] for name in values}, values)

    def test_load_missing_credentials_explains_how_to_create_them(self) -> None:
        with (
            TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"XDG_CONFIG_HOME": tmp}, clear=True),
            self.assertRaisesRegex(config.ConfigError, r"pci index \. --mcp-config <client>"),
        ):
            _ = mcp_credentials.load(Path(tmp) / "project")
