"""Unit tests for `project_code_intelligence.git_utils`.

`git_utils` is a thin wrapper over the constrained subprocess boundary in
`process.py`. These tests mock `process.run` the same way `tests/test_process.py`
does, plus the `shutil.which` seam that `git_binary` consults, so every branch
is exercised without invoking real `git`.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from project_code_intelligence import git_utils, process


def _completed(stdout: str) -> process.CompletedProcess[str]:
    """Build a minimal CompletedProcess mock with the requested stdout."""

    return process.CompletedProcess(args=["git"], returncode=0, stdout=stdout, stderr="")


class GitBinaryTests(unittest.TestCase):
    """`git_binary` is just `shutil.which("git")` — pin the contract here."""

    def test_returns_resolved_path_when_git_is_on_path(self) -> None:
        with patch.object(git_utils.shutil, "which", return_value="/usr/bin/git") as which_mock:
            self.assertEqual(git_utils.git_binary(), "/usr/bin/git")
        which_mock.assert_called_once_with("git")

    def test_returns_none_when_git_missing_from_path(self) -> None:
        with patch.object(git_utils.shutil, "which", return_value=None):
            self.assertIsNone(git_utils.git_binary())


class WorkspaceRootTests(unittest.TestCase):
    """Covers the three workspace_root branches: no git, git error, git ok."""

    def test_falls_back_to_cwd_when_git_binary_is_absent(self) -> None:
        # No git on PATH → workspace_root must return the resolved cwd and
        # never reach process.run.
        with (
            patch.object(git_utils, "git_binary", return_value=None),
            patch.object(git_utils.process, "run") as run_mock,
        ):
            result = git_utils.workspace_root()

        self.assertEqual(result, Path.cwd().resolve())
        run_mock.assert_not_called()

    def test_falls_back_to_cwd_when_rev_parse_raises_called_process_error(self) -> None:
        # Outside a checkout → `git rev-parse --show-toplevel` exits non-zero
        # with check=True → CalledProcessError. workspace_root swallows that
        # and falls back to cwd.
        def raise_called_process_error(*_args: object, **_kwargs: object) -> object:
            raise process.CalledProcessError(returncode=128, cmd=["git", "rev-parse"])

        with (
            patch.object(git_utils, "git_binary", return_value="/usr/bin/git"),
            patch.object(git_utils.process, "run", side_effect=raise_called_process_error),
        ):
            result = git_utils.workspace_root()

        self.assertEqual(result, Path.cwd().resolve())

    def test_falls_back_to_cwd_when_rev_parse_raises_os_error(self) -> None:
        # The except clause also catches OSError (e.g. ENOENT if the binary
        # disappeared between `which` and `run`). Exercise that path too.
        with (
            patch.object(git_utils, "git_binary", return_value="/usr/bin/git"),
            patch.object(git_utils.process, "run", side_effect=OSError("missing")),
        ):
            result = git_utils.workspace_root()

        self.assertEqual(result, Path.cwd().resolve())

    def test_returns_resolved_git_toplevel_on_success(self) -> None:
        # Happy path: git returns a path on stdout (with trailing whitespace
        # to verify the .strip()), and workspace_root resolves it.
        tmp_root = Path(tempfile.gettempdir()).resolve()
        captured: dict[str, object] = {}

        def fake_run(command: object, options: object | None = None) -> process.CompletedProcess[str]:
            captured["command"] = command
            captured["options"] = options
            return _completed(f"{tmp_root}\n")

        with (
            patch.object(git_utils, "git_binary", return_value="/usr/bin/git"),
            patch.object(git_utils.process, "run", side_effect=fake_run),
        ):
            result = git_utils.workspace_root()

        self.assertEqual(result, tmp_root)
        # The argv must be the exact rev-parse invocation — guards against
        # someone accidentally changing the contract.
        self.assertEqual(
            captured["command"],
            ["/usr/bin/git", "rev-parse", "--show-toplevel"],
        )
        options = cast("process.RunOptions", captured["options"])
        # check=True is what causes a non-checkout cwd to raise rather than
        # return a stale path silently.
        self.assertTrue(options.check)
        self.assertEqual(options.timeout, git_utils.GIT_TIMEOUT_SECONDS)


class RunGitTests(unittest.TestCase):
    """Covers run_git: no-binary, OSError, CalledProcessError, empty stdout,
    success."""

    def test_returns_none_when_git_binary_is_absent(self) -> None:
        with (
            patch.object(git_utils, "git_binary", return_value=None),
            patch.object(git_utils.process, "run") as run_mock,
        ):
            self.assertIsNone(git_utils.run_git(Path(tempfile.gettempdir()), ["rev-parse", "HEAD"]))
        run_mock.assert_not_called()

    def test_returns_none_when_run_raises_called_process_error(self) -> None:
        def raise_called_process_error(*_args: object, **_kwargs: object) -> object:
            raise process.CalledProcessError(returncode=128, cmd=["git", "rev-parse"])

        with (
            patch.object(git_utils, "git_binary", return_value="/usr/bin/git"),
            patch.object(git_utils.process, "run", side_effect=raise_called_process_error),
        ):
            self.assertIsNone(git_utils.run_git(Path(tempfile.gettempdir()), ["rev-parse", "HEAD"]))

    def test_returns_none_when_run_raises_os_error(self) -> None:
        with (
            patch.object(git_utils, "git_binary", return_value="/usr/bin/git"),
            patch.object(git_utils.process, "run", side_effect=OSError("missing")),
        ):
            self.assertIsNone(git_utils.run_git(Path(tempfile.gettempdir()), ["rev-parse", "HEAD"]))

    def test_returns_none_when_stdout_is_empty_or_whitespace_only(self) -> None:
        # `proc.stdout.strip() or None` — empty stdout collapses to None so
        # callers don't get a misleading "" success value.
        with (
            patch.object(git_utils, "git_binary", return_value="/usr/bin/git"),
            patch.object(git_utils.process, "run", return_value=_completed("   \n")),
        ):
            self.assertIsNone(git_utils.run_git(Path(tempfile.gettempdir()), ["rev-parse", "HEAD"]))

    def test_returns_stripped_stdout_on_success(self) -> None:
        captured: dict[str, object] = {}

        def fake_run(command: object, options: object | None = None) -> process.CompletedProcess[str]:
            captured["command"] = command
            captured["options"] = options
            return _completed("deadbeef\n")

        with (
            patch.object(git_utils, "git_binary", return_value="/usr/bin/git"),
            patch.object(git_utils.process, "run", side_effect=fake_run),
        ):
            result = git_utils.run_git(Path(tempfile.gettempdir()), ["rev-parse", "HEAD"])

        self.assertEqual(result, "deadbeef")
        # The argv must lead with the resolved git binary and forward args
        # verbatim — guards against any future quoting/wrapping change.
        self.assertEqual(captured["command"], ["/usr/bin/git", "rev-parse", "HEAD"])
        options = cast("process.RunOptions", captured["options"])
        self.assertEqual(options.cwd, Path(tempfile.gettempdir()))
        self.assertTrue(options.check)
        self.assertEqual(options.timeout, git_utils.GIT_TIMEOUT_SECONDS)


if __name__ == "__main__":
    _ = unittest.main()
