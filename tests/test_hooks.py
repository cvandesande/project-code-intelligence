from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast
from unittest import mock

if TYPE_CHECKING:
    from collections.abc import Mapping

from project_code_intelligence.hooks import detect, install, runtime
from project_code_intelligence.hooks.opencode_assets import OPENCODE_FILES


class DetectTests(unittest.TestCase):
    def test_removed_definition_on_delete(self) -> None:
        self.assertEqual(detect.removed_definitions("def render_text(x):\n    return 1\n", ""), ["render_text"])

    def test_in_place_edit_removes_nothing(self) -> None:
        old = "def render_text(x):\n    return 1\n"
        new = "def render_text(x):\n    return 2\n"
        self.assertEqual(detect.removed_definitions(old, new), [])

    def test_rename_reports_old_name(self) -> None:
        self.assertEqual(detect.removed_definitions("def foo():\n pass\n", "def bar():\n pass\n"), ["foo"])

    def test_detects_class_func_type_and_shell(self) -> None:
        blob = "class A:\n    pass\nfunc Go() {}\ntype T struct{}\nsh_fn() {\n  :\n}\n"
        self.assertEqual(detect.defined_names(blob), {"A", "Go", "T", "sh_fn"})

    def test_source_path_gate(self) -> None:
        self.assertTrue(detect.is_source_path("a/b.py"))
        self.assertFalse(detect.is_source_path("notes.md"))


class _StubReports:
    """Stand-in for evidence.render_symbol_reports that avoids the database."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def __call__(self, symbol: str, **_: object) -> list[str]:
        self.calls.append(symbol)
        return self.mapping.get(symbol, [])


class EvidenceRuntimeTests(unittest.TestCase):
    def _run(self, agent: str, event: Mapping[str, object], reports: _StubReports) -> str:
        out = io.StringIO()
        with mock.patch.object(runtime.evidence, "render_symbol_reports", reports):
            code = runtime.run_evidence(agent, stdin=io.StringIO(json.dumps(event)), stdout=out)
        self.assertEqual(code, 0)
        return out.getvalue()

    def test_opencode_delete_injects_raw_block(self) -> None:
        reports = _StubReports({"gone": ["gone  function  a.py:1-3\ncallers (0)"]})
        event = {"filePath": "a.py", "oldString": "def gone():\n pass\n", "newString": ""}
        text = self._run("opencode", event, reports)
        self.assertIn("[pci blast-radius", text)
        self.assertIn("gone  function", text)
        self.assertEqual(reports.calls, ["gone"])

    def test_claude_delete_wraps_in_additional_context(self) -> None:
        reports = _StubReports({"gone": ["gone  function  a.py:1-3\ncallers (0)"]})
        event = {
            "tool_name": "Edit",
            "tool_input": {"file_path": "a.py", "old_string": "def gone():\n x\n", "new_string": ""},
        }
        payload = cast("dict[str, object]", json.loads(self._run("claude", event, reports)))
        hook_out = cast("dict[str, object]", payload["hookSpecificOutput"])
        self.assertEqual(hook_out["hookEventName"], "PostToolUse")
        self.assertIn("[pci blast-radius", cast("str", hook_out["additionalContext"]))

    def test_in_place_edit_is_silent(self) -> None:
        reports = _StubReports({"gone": ["should not appear"]})
        event = {"filePath": "a.py", "oldString": "def keep():\n return 1\n", "newString": "def keep():\n return 2\n"}
        self.assertEqual(self._run("opencode", event, reports), "")
        self.assertEqual(reports.calls, [])

    def test_non_source_file_is_silent(self) -> None:
        reports = _StubReports({"gone": ["x"]})
        event = {"filePath": "notes.md", "oldString": "def gone(): pass", "newString": ""}
        self.assertEqual(self._run("opencode", event, reports), "")

    def test_empty_reports_stay_silent(self) -> None:
        reports = _StubReports({})  # symbol resolves to nothing in the index
        event = {"filePath": "a.py", "oldString": "def gone():\n x\n", "newString": ""}
        self.assertEqual(self._run("opencode", event, reports), "")


class InstallOpencodeTests(unittest.TestCase):
    def test_install_then_uninstall(self) -> None:
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            outcome = install.install_opencode(project, uninstall=False, dry_run=False)
            self.assertEqual(outcome.action, "installed")
            for rel in OPENCODE_FILES:
                self.assertTrue((project / ".opencode" / rel).is_file())

            again = install.install_opencode(project, uninstall=False, dry_run=False)
            self.assertEqual(again.action, "updated")

            removed = install.install_opencode(project, uninstall=True, dry_run=False)
            self.assertEqual(removed.action, "removed")
            for rel in OPENCODE_FILES:
                self.assertFalse((project / ".opencode" / rel).exists())

    def test_dry_run_writes_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            outcome = install.install_opencode(project, uninstall=False, dry_run=True)
            self.assertEqual(outcome.action, "installed")
            self.assertFalse((project / ".opencode").exists())


def _read_json_file(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


class InstallClaudeTests(unittest.TestCase):
    def test_install_merges_and_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = Path(tmp) / ".claude" / "settings.json"
            first = install.install_claude(settings, uninstall=False, dry_run=False)
            self.assertEqual(first.action, "installed")
            second = install.install_claude(settings, uninstall=False, dry_run=False)
            self.assertEqual(second.action, "updated")

            data = _read_json_file(settings)
            hooks = cast("dict[str, object]", data["hooks"])
            self.assertEqual(len(cast("list[object]", hooks["PostToolUse"])), 1)
            self.assertEqual(len(cast("list[object]", hooks["Stop"])), 1)

    def test_install_preserves_foreign_hooks(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = Path(tmp) / ".claude" / "settings.json"
            settings.parent.mkdir(parents=True)
            foreign = {
                "hooks": {"PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "lint"}]}]}
            }
            _ = settings.write_text(json.dumps(foreign), encoding="utf-8")

            _ = install.install_claude(settings, uninstall=False, dry_run=False)
            groups = cast("list[object]", cast("dict[str, object]", _read_json_file(settings)["hooks"])["PostToolUse"])
            commands = {
                cast("str", cast("dict[str, object]", handler).get("command"))
                for group in groups
                for handler in cast("list[object]", cast("dict[str, object]", group).get("hooks"))
            }
            self.assertIn("lint", commands)

            removed = install.install_claude(settings, uninstall=True, dry_run=False)
            self.assertEqual(removed.action, "removed")
            # Foreign hook survives uninstall; our handlers are gone.
            hooks_after = cast("dict[str, object]", _read_json_file(settings)["hooks"])
            groups_after = cast("list[object]", hooks_after["PostToolUse"])
            self.assertEqual(len(groups_after), 1)

    def test_uninstall_on_clean_config_is_unchanged(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = Path(tmp) / ".claude" / "settings.json"
            outcome = install.install_claude(settings, uninstall=True, dry_run=False)
            self.assertEqual(outcome.action, "unchanged")


if __name__ == "__main__":
    _ = unittest.main()
