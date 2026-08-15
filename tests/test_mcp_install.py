import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from project_code_intelligence import mcp_install


class CodexMcpInstallTests(unittest.TestCase):
    def test_install_update_and_uninstall_preserve_foreign_toml(self) -> None:
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            path = project / ".codex" / "config.toml"
            path.parent.mkdir()
            _ = path.write_text('model = "example"\n', encoding="utf-8")
            first, _ = mcp_install.install_codex(project, uninstall=False, dry_run=False)
            second, _ = mcp_install.install_codex(project, uninstall=False, dry_run=False)
            self.assertEqual((first, second), ("installed", "updated"))
            installed = path.read_text(encoding="utf-8")
            self.assertEqual(installed.count(">>> pci mcp"), 1)
            self.assertIn('"--scope"', installed)
            self.assertNotIn("PCI_MCP_DATABASE", installed)
            removed, _ = mcp_install.install_codex(project, uninstall=True, dry_run=False)
            self.assertEqual(removed, "removed")
            self.assertEqual(path.read_text(encoding="utf-8"), 'model = "example"\n')

    def test_dry_run_does_not_write(self) -> None:
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            action, path = mcp_install.install_codex(project, uninstall=False, dry_run=True)
            self.assertEqual(action, "installed")
            self.assertFalse(path.exists())

    def test_pi_installs_project_extension(self) -> None:
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            parsed = mcp_install.InstallNamespace(target="pi", project=tmp)
            action, path = mcp_install._run_install(parsed, project)  # pyright: ignore[reportPrivateUsage]
            self.assertEqual(action, "installed")
            text = path.read_text(encoding="utf-8")
            self.assertIn('spawn(PCI, ["mcp", "--scope", cwd]', text)
            self.assertNotIn("__PCI_COMMAND__", text)


class JsonMcpInstallTests(unittest.TestCase):
    def test_every_project_scoped_json_target_installs_and_uninstalls(self) -> None:
        for target in ("claude", "opencode", "vscode", "copilot", "zed"):
            with self.subTest(target=target), TemporaryDirectory() as tmp:
                project = Path(tmp)
                action, path = mcp_install.install_json_target(
                    target, project, config_path=None, uninstall=False, dry_run=False
                )
                self.assertEqual(action, "installed")
                data = path.read_text(encoding="utf-8")
                self.assertIn("project-code-intelligence", data)
                self.assertIn("--scope", data)
                self.assertNotIn("PCI_MCP_DATABASE", data)
                removed, _ = mcp_install.install_json_target(
                    target, project, config_path=None, uninstall=True, dry_run=False
                )
                self.assertEqual(removed, "removed")
                remaining = path.read_text(encoding="utf-8") if path.exists() else ""
                self.assertNotIn("project-code-intelligence", remaining)

    def test_merge_preserves_foreign_server(self) -> None:
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            path = project / ".mcp.json"
            _ = path.write_text(json.dumps({"mcpServers": {"foreign": {"command": "other"}}}), encoding="utf-8")
            _ = mcp_install.install_json_target("claude", project, config_path=None, uninstall=False, dry_run=False)
            data = path.read_text(encoding="utf-8")
            self.assertIn("foreign", data)

    def test_cline_requires_explicit_path(self) -> None:
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "requires --config-path"):
            _ = mcp_install.install_json_target("cline", Path(tmp), config_path=None, uninstall=False, dry_run=True)

    def test_cline_uses_explicit_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "cline-mcp.json"
            action, actual = mcp_install.install_json_target(
                "cline", root, config_path=path, uninstall=False, dry_run=False
            )
            self.assertEqual((action, actual), ("installed", path))
            self.assertIn("project-code-intelligence", path.read_text(encoding="utf-8"))
