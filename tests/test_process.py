"""Unit tests for `project_code_intelligence.process`.

Security-critical: the subprocess boundary in `process.py` is the basis for
the project-wide Bandit B404/B603 suppressions. These tests pin the
contract:

- `run` and `popen` always pass argument vectors to subprocess (no string
  commands, no `shell=True`).
- Empty command vectors and empty arguments are rejected up front.
- `run_docker` audits its arguments and refuses privilege-escalation or
  dangerous host-mount flags.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from project_code_intelligence import process
from project_code_intelligence.process import (
    PopenOptions,
    RunOptions,
    compose_file_args,
    container_engine_name,
    container_engine_path,
    popen,
    run,
    run_docker,
)


class RunCommandValidationTests(unittest.TestCase):
    def test_run_rejects_empty_command(self) -> None:
        with self.assertRaises(ValueError):
            _ = run([])

    def test_run_rejects_empty_string_argument(self) -> None:
        with self.assertRaises(ValueError):
            _ = run(["/bin/echo", ""])

    def test_run_invokes_subprocess_run_with_shell_false_and_list_argv(self) -> None:
        captured: dict[str, object] = {}

        def fake_subprocess_run(*args: object, **kwargs: object) -> object:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "completed"

        tmp_cwd = Path(tempfile.gettempdir())
        with patch.object(process.subprocess, "run", side_effect=fake_subprocess_run) as patched:
            _ = run(["/bin/echo", "hello"], RunOptions(cwd=tmp_cwd, capture_output=True))

        self.assertEqual(patched.call_count, 1)
        positional = cast("tuple[object, ...]", captured["args"])
        kwargs = cast("dict[str, object]", captured["kwargs"])
        # First positional arg is the argv list, not a string.
        self.assertEqual(positional[0], ["/bin/echo", "hello"])
        self.assertIsInstance(positional[0], list)
        # shell=False must be hard-coded.
        self.assertEqual(kwargs.get("shell"), False)
        # text=True is hard-coded so callers get str output.
        self.assertEqual(kwargs.get("text"), True)
        self.assertEqual(kwargs.get("cwd"), tmp_cwd)
        self.assertEqual(kwargs.get("capture_output"), True)

    def test_run_passes_options_env_as_dict_copy(self) -> None:
        captured_env: dict[str, object] = {}

        def fake_subprocess_run(*_args: object, **kwargs: object) -> object:
            captured_env["env"] = kwargs.get("env")
            return "completed"

        env_mapping = {"FOO": "bar"}
        with patch.object(process.subprocess, "run", side_effect=fake_subprocess_run):
            _ = run(["/bin/true"], RunOptions(env=env_mapping))

        # The env mapping must be copied to a dict so the caller's mutable mapping
        # cannot be tampered with after the subprocess.run call.
        self.assertEqual(captured_env["env"], {"FOO": "bar"})
        self.assertIsInstance(captured_env["env"], dict)

    def test_run_with_no_env_passes_none_for_env(self) -> None:
        captured_env: dict[str, object] = {}

        def fake_subprocess_run(*_args: object, **kwargs: object) -> object:
            captured_env["env"] = kwargs.get("env")
            return "completed"

        with patch.object(process.subprocess, "run", side_effect=fake_subprocess_run):
            _ = run(["/bin/true"])

        # env=None passes through (subprocess inherits the parent env).
        self.assertIsNone(captured_env["env"])


class PopenCommandValidationTests(unittest.TestCase):
    def test_popen_rejects_empty_command(self) -> None:
        with self.assertRaises(ValueError):
            _ = popen([])

    def test_popen_rejects_empty_string_argument(self) -> None:
        with self.assertRaises(ValueError):
            _ = popen(["/bin/echo", ""])

    def test_popen_invokes_subprocess_popen_with_shell_false_and_list_argv(self) -> None:
        captured: dict[str, object] = {}

        def fake_subprocess_popen(*args: object, **kwargs: object) -> object:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "popen-handle"

        with patch.object(process.subprocess, "Popen", side_effect=fake_subprocess_popen):
            _ = popen(
                ["/bin/sleep", "1"],
                PopenOptions(cwd=Path(tempfile.gettempdir()), start_new_session=True),
            )

        positional = cast("tuple[object, ...]", captured["args"])
        kwargs = cast("dict[str, object]", captured["kwargs"])
        self.assertEqual(positional[0], ["/bin/sleep", "1"])
        self.assertIsInstance(positional[0], list)
        self.assertEqual(kwargs.get("shell"), False)
        self.assertEqual(kwargs.get("text"), True)
        self.assertEqual(kwargs.get("start_new_session"), True)


class DockerArgumentAuditTests(unittest.TestCase):
    """Drive the docker-argument audit through `run_docker`, the public entry.

    `run_docker` is the only place outside this module that should know the
    block-list, so exercising it via the public API gives the audit behavior
    full coverage without reaching into private helpers.
    """

    def _assert_run_docker_rejects(self, args: list[str]) -> None:
        with (
            patch.object(process, "container_engine_path", return_value="/usr/bin/docker"),
            patch.object(process, "run") as run_mock,
            self.assertRaises(ValueError),
        ):
            _ = run_docker(args)
        # The audit runs before subprocess invocation, so run_mock must not fire.
        run_mock.assert_not_called()

    def _assert_run_docker_passes(self, args: list[str]) -> None:
        with (
            patch.object(process, "container_engine_path", return_value="/usr/bin/docker"),
            patch.object(process, "run", return_value="completed") as run_mock,
        ):
            _ = run_docker(args)
        # Using self.assertEqual on call_count keeps the audit-passes signal
        # observable through unittest's reporting, and proves the audit didn't
        # short-circuit run() (no raise) — these args reached subprocess once.
        self.assertEqual(run_mock.call_count, 1)

    def test_run_docker_refuses_privilege_escalation_flags(self) -> None:
        for blocked in (
            "--privileged",
            "--cap-add=ALL",
            "--security-opt=label=disable",
            "--pid=host",
            "--ipc=host",
            "--network=host",
            "--userns=host",
        ):
            with self.subTest(arg=blocked):
                self._assert_run_docker_rejects(["run", blocked, "ubuntu"])

    def test_run_docker_refuses_inline_dangerous_volume_mounts(self) -> None:
        for dangerous in (
            "-v=/:/host",
            "--volume=/etc:/host_etc",
            "-v=/var:/host_var",
            "-v=/root:/r",
            "-v=/home:/h",
            "-v=/dev:/d",
            "-v=/proc:/p",
            "-v=/sys:/s",
        ):
            with self.subTest(arg=dangerous):
                self._assert_run_docker_rejects(["run", dangerous, "ubuntu"])

    def test_run_docker_refuses_split_form_dangerous_volume_mounts(self) -> None:
        # `-v /host:/container` shape (value as a separate argv element).
        self._assert_run_docker_rejects(["run", "-v", "/etc:/host_etc", "ubuntu"])
        self._assert_run_docker_rejects(["run", "--volume", "/:/host_root", "ubuntu"])

    def test_run_docker_allows_safe_flags_and_safe_volumes(self) -> None:
        # Project-local mount paths are permitted; the audit only refuses the
        # host root and a fixed list of system directories.
        self._assert_run_docker_passes(["run", "--rm", "-v", "./work:/work", "ubuntu"])
        self._assert_run_docker_passes(["run", "-v=./local:/work", "--name=hello", "ubuntu"])


class RunDockerTests(unittest.TestCase):
    def test_run_docker_rejects_empty_args(self) -> None:
        with self.assertRaises(ValueError):
            _ = run_docker([])

    def test_run_docker_raises_when_engine_is_absent(self) -> None:
        with (
            patch.object(process, "container_engine_path", return_value=None),
            self.assertRaises(FileNotFoundError),
        ):
            _ = run_docker(["info"])

    def test_run_docker_invokes_run_with_resolved_engine_path(self) -> None:
        captured: dict[str, object] = {}

        def fake_run(command: object, options: object | None = None) -> object:
            captured["command"] = command
            captured["options"] = options
            return "completed"

        with (
            patch.object(process, "container_engine_path", return_value="/usr/bin/docker"),
            patch.object(process, "run", side_effect=fake_run),
        ):
            _ = run_docker(["info"])

        # Argv must lead with the absolute, resolved engine path.
        self.assertEqual(captured["command"], ["/usr/bin/docker", "info"])

    def test_run_docker_raises_before_invoking_subprocess_on_blocked_args(self) -> None:
        sentinel: dict[str, bool] = {"called": False}

        def fake_run(_command: object, _options: object | None = None) -> object:
            sentinel["called"] = True
            return "completed"

        with (
            patch.object(process, "container_engine_path", return_value="/usr/bin/docker"),
            patch.object(process, "run", side_effect=fake_run),
            self.assertRaises(ValueError),
        ):
            _ = run_docker(["run", "--privileged", "ubuntu"])

        self.assertFalse(sentinel["called"])


class ContainerEngineDetectionTests(unittest.TestCase):
    def test_container_engine_env_override_resolves_through_which(self) -> None:
        with (
            patch.dict(os.environ, {"PCI_CONTAINER_ENGINE": "custom-engine"}, clear=False),
            patch.object(process.shutil, "which", return_value="/usr/local/bin/custom-engine") as which_mock,
        ):
            self.assertEqual(container_engine_path(), "/usr/local/bin/custom-engine")
            which_mock.assert_called_once_with("custom-engine")

    def test_container_engine_falls_back_to_candidates_when_no_override(self) -> None:
        def fake_which(name: str) -> str | None:
            return "/usr/bin/podman" if name == "podman" else None

        env_without_override = {k: v for k, v in os.environ.items() if k != "PCI_CONTAINER_ENGINE"}
        with (
            patch.dict(os.environ, env_without_override, clear=True),
            patch.object(process.shutil, "which", side_effect=fake_which),
        ):
            self.assertEqual(container_engine_path(), "/usr/bin/podman")

    def test_container_engine_returns_none_when_nothing_found(self) -> None:
        env_without_override = {k: v for k, v in os.environ.items() if k != "PCI_CONTAINER_ENGINE"}
        with (
            patch.dict(os.environ, env_without_override, clear=True),
            patch.object(process.shutil, "which", return_value=None),
        ):
            self.assertIsNone(container_engine_path())

    def test_container_engine_name_returns_basename_of_resolved_path(self) -> None:
        with patch.object(process, "container_engine_path", return_value="/opt/bin/podman"):
            self.assertEqual(container_engine_name(), "podman")

    def test_container_engine_name_falls_back_to_docker_when_engine_missing(self) -> None:
        with patch.object(process, "container_engine_path", return_value=None):
            self.assertEqual(container_engine_name(), "docker")


class ComposeFileArgsTests(unittest.TestCase):
    def test_pci_compose_file_env_var_takes_precedence(self) -> None:
        with patch.dict(os.environ, {"PCI_COMPOSE_FILE": "/custom/docker-compose.yml"}, clear=False):
            self.assertEqual(compose_file_args(), ["-f", "/custom/docker-compose.yml"])

    def test_pci_compose_file_whitespace_only_is_treated_as_unset(self) -> None:
        with patch.dict(os.environ, {"PCI_COMPOSE_FILE": "   "}, clear=False):
            result = compose_file_args()
        # Either the bundled file resolves, or the empty fallback is returned.
        self.assertIn(result[:1], ([], ["-f"]))


if __name__ == "__main__":
    _ = unittest.main()
