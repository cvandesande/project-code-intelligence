from __future__ import annotations

import io
import json
import os
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast
from unittest import mock

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from contextlib import AbstractContextManager

from project_code_intelligence import analyze
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

    def test_added_definition_on_insert(self) -> None:
        self.assertEqual(detect.added_definitions("", "def fresh():\n    return 1\n"), ["fresh"])

    def test_rename_is_both_a_removal_and_an_addition(self) -> None:
        old, new = "def foo():\n pass\n", "def bar():\n pass\n"
        self.assertEqual(detect.removed_definitions(old, new), ["foo"])
        self.assertEqual(detect.added_definitions(old, new), ["bar"])


def _write_event(path: Path, content: str) -> dict[str, object]:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": "Write",
        "tool_input": {"file_path": str(path), "content": content},
    }


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
            "hook_event_name": "PreToolUse",
            "tool_name": "Edit",
            "tool_input": {"file_path": "a.py", "old_string": "def gone():\n x\n", "new_string": ""},
        }
        payload = cast("dict[str, object]", json.loads(self._run("claude", event, reports)))
        hook_out = cast("dict[str, object]", payload["hookSpecificOutput"])
        # Output event name echoes the firing event (preventive PreToolUse).
        self.assertEqual(hook_out["hookEventName"], "PreToolUse")
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

    def test_claude_write_over_file_reports_removals(self) -> None:
        """A whole-file Write carries no old_string; the old side comes from disk."""
        reports = _StubReports({"gone": ["gone  function  a.py:1-3\ncallers (2)"]})
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.py"
            _ = target.write_text("def gone():\n    return 1\n", encoding="utf-8")
            event = _write_event(target, "# rewritten\n")
            payload = cast("dict[str, object]", json.loads(self._run("claude", event, reports)))
        hook_out = cast("dict[str, object]", payload["hookSpecificOutput"])
        self.assertIn("[pci blast-radius", cast("str", hook_out["additionalContext"]))
        self.assertEqual(reports.calls, ["gone"])

    def test_claude_write_keeping_definition_is_silent(self) -> None:
        reports = _StubReports({"gone": ["should not appear"]})
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.py"
            _ = target.write_text("def gone():\n    return 1\n", encoding="utf-8")
            event = _write_event(target, "def gone():\n    return 2\n")
            self.assertEqual(self._run("claude", event, reports), "")
        self.assertEqual(reports.calls, [])

    def test_claude_write_new_file_is_silent(self) -> None:
        reports = _StubReports({"gone": ["should not appear"]})
        with TemporaryDirectory() as tmp:
            event = _write_event(Path(tmp) / "new.py", "def fresh():\n    return 1\n")
            self.assertEqual(self._run("claude", event, reports), "")
        self.assertEqual(reports.calls, [])


_ADDED_BODY = "def report_for(symbol):\n    query = build(symbol)\n    rows = collect(query)\n    return render(rows)\n"


def _fake_node(symbol: str) -> analyze.FunctionNode:
    return analyze.FunctionNode(
        record_id=f"repo/a.py::function::{symbol}",
        symbol=symbol,
        source_path="repo/src/a.py",
        line_start=10,
        line_end=20,
        callee_roles=frozenset({"build", "collect", "render"}),
        callee_symbols=frozenset({"build", "collect", "render"}),
    )


@contextmanager
def _shape_matches(matches: list[tuple[analyze.FunctionNode, float]]) -> Generator[None]:
    """Run the shape check against a fixed match list instead of the database."""
    snapshot = analyze.SnapshotRef(snapshot_id=1, collection="c", repo="r")

    def connect() -> AbstractContextManager[object]:
        return nullcontext(object())

    def tables_exist(_conn: object) -> bool:
        return True

    def latest_snapshots(_conn: object) -> list[analyze.SnapshotRef]:
        return [snapshot]

    def shape_matches(*_args: object, **_kwargs: object) -> list[tuple[analyze.FunctionNode, float]]:
        return matches

    with mock.patch.multiple(
        runtime,
        db=mock.MagicMock(connect=connect, code_intel_tables_exist=tables_exist),
        analyze=mock.MagicMock(
            role_set=analyze.role_set,
            latest_snapshots=latest_snapshots,
            shape_matches=shape_matches,
        ),
    ):
        yield


class AntiSlopShapeTests(unittest.TestCase):
    """The add-side check is structural: call-shape overlap, never embedding distance."""

    def test_reports_matching_shape(self) -> None:
        with _shape_matches([(_fake_node("render_symbol_reports"), 0.64)]):
            block = runtime.shape_report(["report_for"], _ADDED_BODY)
        self.assertIsNotNone(block)
        text = cast("str", block)
        self.assertIn("[pci anti-slop", text)
        self.assertIn("render_symbol_reports", text)
        self.assertIn("0.64", text)

    def test_no_match_is_silent(self) -> None:
        with _shape_matches([]):
            self.assertIsNone(runtime.shape_report(["report_for"], _ADDED_BODY))

    def test_two_additions_at_once_are_skipped(self) -> None:
        """Shape is read from the whole new text, so two additions would blend into one."""
        with _shape_matches([(_fake_node("x"), 0.9)]):
            self.assertIsNone(runtime.shape_report(["one", "two"], _ADDED_BODY))

    def test_thin_function_is_below_the_role_floor(self) -> None:
        with _shape_matches([(_fake_node("x"), 0.9)]):
            self.assertIsNone(runtime.shape_report(["add"], "def add(a, b):\n    return a + b\n"))

    def test_database_failure_stays_silent(self) -> None:
        broken = mock.MagicMock(connect=mock.MagicMock(side_effect=OSError("no socket")))
        with mock.patch.object(runtime, "db", broken):
            self.assertIsNone(runtime.shape_report(["report_for"], _ADDED_BODY))

    def test_removal_wins_over_addition_on_rename(self) -> None:
        """A rename is both; the removal is the costlier mistake, so it takes the channel."""
        reports = _StubReports({"old_name": ["old_name  function  a.py:1-3\ncallers (1)"]})
        event = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Edit",
            "tool_input": {
                "file_path": "a.py",
                "old_string": "def old_name():\n    return build()\n",
                "new_string": "def new_name():\n    return build()\n",
            },
        }
        out = io.StringIO()
        with mock.patch.object(runtime.evidence, "render_symbol_reports", reports):
            code = runtime.run_evidence("claude", stdin=io.StringIO(json.dumps(event)), stdout=out)
        self.assertEqual(code, 0)
        self.assertIn("[pci blast-radius", out.getvalue())
        self.assertNotIn("[pci anti-slop", out.getvalue())


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
            self.assertEqual(len(cast("list[object]", hooks["PreToolUse"])), 1)
            self.assertNotIn("PostToolUse", hooks)  # evidence is preventive now
            self.assertNotIn("Stop", hooks)  # reindex moved to the git post-commit hook

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


def _post_commit_path(repo: Path) -> Path:
    return repo / ".git" / "hooks" / "post-commit"


class InstallGitTests(unittest.TestCase):
    def test_install_writes_executable_managed_block(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            outcome = install.install_git(repo, uninstall=False, dry_run=False)
            self.assertEqual(outcome.action, "installed")
            hook = _post_commit_path(repo)
            text = hook.read_text(encoding="utf-8")
            self.assertIn("pci-hook reindex (managed)", text)
            self.assertIn("--behavior reindex", text)
            self.assertTrue(os.access(hook, os.X_OK))

    def test_install_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _ = install.install_git(repo, uninstall=False, dry_run=False)
            again = install.install_git(repo, uninstall=False, dry_run=False)
            self.assertEqual(again.action, "updated")
            text = _post_commit_path(repo).read_text(encoding="utf-8")
            self.assertEqual(text.count(">>> pci-hook reindex (managed) >>>"), 1)

    def test_uninstall_keeps_user_script(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            hook = _post_commit_path(repo)
            hook.parent.mkdir(parents=True)
            _ = hook.write_text("#!/bin/sh\necho mine\n", encoding="utf-8")
            _ = install.install_git(repo, uninstall=False, dry_run=False)
            self.assertIn("echo mine", hook.read_text(encoding="utf-8"))
            removed = install.install_git(repo, uninstall=True, dry_run=False)
            self.assertEqual(removed.action, "removed")
            surviving = hook.read_text(encoding="utf-8")
            self.assertIn("echo mine", surviving)
            self.assertNotIn("managed", surviving)

    def test_uninstall_deletes_when_only_ours(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _ = install.install_git(repo, uninstall=False, dry_run=False)
            self.assertTrue(_post_commit_path(repo).exists())
            _ = install.install_git(repo, uninstall=True, dry_run=False)
            self.assertFalse(_post_commit_path(repo).exists())


if __name__ == "__main__":
    _ = unittest.main()
