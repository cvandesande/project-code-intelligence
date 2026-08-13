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

import contextlib
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import tomli

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

TomlTable = dict[str, object]


def load_toml(path: Path) -> TomlTable:
    return cast("TomlTable", tomli.loads(path.read_text(encoding="utf-8")))


def toml_table(value: object) -> TomlTable:
    return cast("TomlTable", value)


def toml_string_list(value: object) -> list[str]:
    return cast("list[str]", value)


class PackageDataTests(unittest.TestCase):
    def test_docker_build_context_pyproject_matches_runtime_package_metadata(self) -> None:
        root = Path(__file__).resolve().parents[1]
        project = load_toml(root / "pyproject.toml")
        context = load_toml(root / "docker" / "build-context" / "pyproject.toml")
        project_metadata = toml_table(project["project"])
        context_metadata = toml_table(context["project"])
        project_scripts = toml_table(project_metadata["scripts"])
        context_optional_dependencies = toml_table(context_metadata["optional-dependencies"])
        project_optional_dependencies = toml_table(project_metadata["optional-dependencies"])

        self.assertEqual(context["build-system"], project["build-system"])
        self.assertEqual(context_metadata["name"], project_metadata["name"])
        self.assertEqual(context_metadata["version"], project_metadata["version"])
        self.assertEqual(context_metadata["requires-python"], project_metadata["requires-python"])
        self.assertEqual(context_metadata["dependencies"], [])
        # The host package installs the single `pci` executable; the container keeps its own
        # pci-fastembed-server entry point, targeting the same module as `pci embed fastembed`.
        self.assertEqual(project_scripts, {"pci": "project_code_intelligence.pci:main"})
        self.assertEqual(
            context_metadata["scripts"],
            {"pci-fastembed-server": "project_code_intelligence.embedding.fastembed_server:main"},
        )
        self.assertEqual(
            context_optional_dependencies["local-embeddings"],
            project_optional_dependencies["local-embeddings"],
        )

    def test_declared_project_code_intelligence_package_data_exists(self) -> None:
        root = Path(__file__).resolve().parents[1]
        pyproject = load_toml(root / "pyproject.toml")
        tool = toml_table(pyproject["tool"])
        setuptools = toml_table(tool["setuptools"])
        package_data = toml_table(setuptools["package-data"])
        project_code_intelligence_data = toml_string_list(package_data["project_code_intelligence"])
        missing = [
            relative_path
            for relative_path in project_code_intelligence_data
            if not (root / "src" / "project_code_intelligence" / relative_path).is_file()
        ]

        self.assertEqual(missing, [])

    def test_compose_context_package_source_whitelist_is_import_sufficient(self) -> None:
        """A tree containing only the whitelisted files must satisfy the container's imports.

        The materialized Compose context copies only
        `_COMPOSE_CONTEXT_PACKAGE_SOURCE_FILES`; if `fastembed_server`'s
        transitive runtime imports outgrow that whitelist, the container
        fails with ModuleNotFoundError at runtime. Catch the drift here.
        """
        package_dir = Path(process.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "src" / "project_code_intelligence"
            for relative_path in process._COMPOSE_CONTEXT_PACKAGE_SOURCE_FILES:  # pyright: ignore[reportPrivateUsage]
                destination = target_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                _ = shutil.copyfile(package_dir / relative_path, destination)
            result = run(
                [sys.executable, "-c", "import project_code_intelligence.embedding.fastembed_server"],
                RunOptions(
                    capture_output=True,
                    cwd=root,
                    env={**os.environ, "PYTHONPATH": str(root / "src")},
                ),
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_flake_python_dependencies_match_pyproject(self) -> None:
        """flake.nix hand-lists runtime deps; fail when pyproject.toml drifts."""
        root = Path(__file__).resolve().parents[1]
        flake_text = (root / "flake.nix").read_text(encoding="utf-8")
        project = load_toml(root / "pyproject.toml")
        project_metadata = toml_table(project["project"])
        for dependency in toml_string_list(project_metadata["dependencies"]):
            if ";" in dependency:
                continue  # platform-marked deps (e.g. mlx-lm on darwin) are not in the flake
            name = re.split(r"[\s<>=!\[]", dependency, maxsplit=1)[0]
            self.assertIn(
                f"pythonPackages.{name}",
                flake_text,
                f"runtime dependency {name!r} from pyproject.toml is missing from flake.nix",
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

    def test_bundled_compose_is_materialized_with_source_checkout_project_dir(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            env = {
                "PCI_COMPOSE_FILE": "   ",
                "PCI_COMPOSE_CACHE_DIR": cache_dir,
                "PCI_COMPOSE_PROJECT_DIR": "",
            }
            with patch.dict(os.environ, env, clear=False):
                result = compose_file_args()

            self.assertEqual(result[:2], ["--project-directory", str(Path.cwd())])
            self.assertEqual(result[2], "-f")
            compose_path = Path(result[3])
            self.assertEqual(compose_path.parent, Path(cache_dir))
            compose_text = compose_path.read_text(encoding="utf-8")
            self.assertIn(str(Path.cwd() / "docker" / "pgvector" / "init-extensions.sql"), compose_text)

    def test_bundled_compose_can_materialize_installed_package_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / "site-packages" / "project_code_intelligence"
            package_dir.mkdir(parents=True)
            package_source = {
                "__init__.py": "",
                "common.py": "def default_database_name(path):\n    return 'codeintel'\n",
                "config.py": "",
                "exceptions.py": "",
                "http_client.py": "",
                "process.py": "",
                "rocm_bundles.py": "",
                "runtime.py": "",
                "embedding/__init__.py": "",
                "embedding/fastembed_server.py": "",
                "embedding/http_common.py": "",
            }
            for relative_path, content in package_source.items():
                source_path = package_dir / relative_path
                source_path.parent.mkdir(parents=True, exist_ok=True)
                _ = source_path.write_text(content, encoding="utf-8")
            _ = (package_dir / "docker-compose.yml").write_text(
                "services:\n"
                "  pgvector:\n"
                "    volumes:\n"
                "      - ./docker/pgvector/init-extensions.sql:/docker-entrypoint-initdb.d/init-extensions.sql:ro\n",
                encoding="utf-8",
            )
            init_sql = package_dir / "docker" / "pgvector" / "init-extensions.sql"
            init_sql.parent.mkdir(parents=True)
            _ = init_sql.write_text("CREATE EXTENSION IF NOT EXISTS vector;\n", encoding="utf-8")
            package_assets = {
                "docker/build-context/pyproject.toml": '[project]\nname = "project-code-intelligence"\n',
                "docker/build-context/README.md": "# packaged context\n",
                "docker/build-context/LICENSE": "test license\n",
                "docker/fastembed/Dockerfile": "FROM python:3.13-slim\n",
                "docker/llamacpp-rocm/Dockerfile": "FROM debian:stable-slim\n",
                "docker/llamacpp-rocm/entrypoint.sh": "#!/bin/sh\n",
                "docker/llamacpp-cuda/Dockerfile": "FROM ghcr.io/ggml-org/llama.cpp:server-cuda\n",
                "docker/llamacpp-cuda/entrypoint.sh": "#!/bin/sh\n",
                "scripts/select_llamacpp_rocm_bundle.py": "#!/usr/bin/env python3\n",
                "bin/pci-embedding-server": "#!/bin/sh\n",
            }
            for relative_path, content in package_assets.items():
                asset_path = package_dir / relative_path
                asset_path.parent.mkdir(parents=True, exist_ok=True)
                _ = asset_path.write_text(content, encoding="utf-8")
            cache_dir = root / "cache"
            env = {
                "PCI_COMPOSE_FILE": "",
                "PCI_COMPOSE_CACHE_DIR": str(cache_dir),
                "PCI_COMPOSE_PROJECT_DIR": "",
            }
            with (
                patch.dict(os.environ, env, clear=False),
                patch.object(process, "_package_dir", return_value=package_dir),
            ):
                result = compose_file_args()

            project_dir = cache_dir / "compose-context"
            self.assertEqual(
                result,
                ["--project-directory", str(project_dir), "-f", str(cache_dir / "docker-compose.yml")],
            )
            self.assertEqual(
                (project_dir / "pyproject.toml").read_text(encoding="utf-8"),
                '[project]\nname = "project-code-intelligence"\n',
            )
            self.assertEqual((project_dir / "README.md").read_text(encoding="utf-8"), "# packaged context\n")
            self.assertEqual((project_dir / "LICENSE").read_text(encoding="utf-8"), "test license\n")
            self.assertTrue((project_dir / "src" / "project_code_intelligence" / "__init__.py").is_file())
            self.assertTrue((project_dir / "src" / "project_code_intelligence" / "config.py").is_file())
            self.assertTrue(
                (project_dir / "src" / "project_code_intelligence" / "embedding" / "fastembed_server.py").is_file()
            )
            self.assertFalse((project_dir / "src" / "project_code_intelligence" / "doctor").exists())
            self.assertTrue((project_dir / "docker" / "fastembed" / "Dockerfile").is_file())
            self.assertTrue((project_dir / "docker" / "llamacpp-rocm" / "Dockerfile").is_file())
            self.assertTrue((project_dir / "docker" / "llamacpp-rocm" / "entrypoint.sh").is_file())
            self.assertTrue((project_dir / "docker" / "llamacpp-cuda" / "Dockerfile").is_file())
            self.assertTrue((project_dir / "docker" / "llamacpp-cuda" / "entrypoint.sh").is_file())
            self.assertTrue((project_dir / "scripts" / "select_llamacpp_rocm_bundle.py").is_file())
            self.assertTrue((project_dir / "pci-embedding-server").is_file())
            self.assertTrue((project_dir / "models").is_dir())
            self.assertIn(
                str(project_dir / "docker" / "pgvector" / "init-extensions.sql"),
                (cache_dir / "docker-compose.yml").read_text(encoding="utf-8"),
            )

    def test_installed_package_context_replaces_stale_read_only_source_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / "site-packages" / "project_code_intelligence"
            package_dir.mkdir(parents=True)
            _ = (package_dir / "__init__.py").write_text("", encoding="utf-8")
            _ = (package_dir / "config.py").write_text("", encoding="utf-8")
            _ = (package_dir / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
            cache_dir = root / "cache"
            stale_package_dir = cache_dir / "compose-context" / "src" / "project_code_intelligence"
            stale_doctor_dir = stale_package_dir / "doctor"
            stale_doctor_dir.mkdir(parents=True)
            stale_file = stale_doctor_dir / "cli.py"
            _ = stale_file.write_text("stale\n", encoding="utf-8")
            stale_file.chmod(0o400)
            stale_doctor_dir.chmod(0o500)
            stale_package_dir.chmod(0o500)
            env = {
                "PCI_COMPOSE_FILE": "",
                "PCI_COMPOSE_CACHE_DIR": str(cache_dir),
                "PCI_COMPOSE_PROJECT_DIR": "",
            }
            try:
                with (
                    patch.dict(os.environ, env, clear=False),
                    patch.object(process, "_package_dir", return_value=package_dir),
                ):
                    result = compose_file_args()
            finally:
                with contextlib.suppress(OSError):
                    stale_package_dir.chmod(0o700)

            project_dir = cache_dir / "compose-context"
            self.assertEqual(
                result,
                ["--project-directory", str(project_dir), "-f", str(cache_dir / "docker-compose.yml")],
            )
            self.assertFalse(stale_doctor_dir.exists())
            self.assertTrue((stale_package_dir / "config.py").is_file())

    def test_pci_compose_file_whitespace_only_is_treated_as_unset(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            env = {
                "PCI_COMPOSE_FILE": "   ",
                "PCI_COMPOSE_CACHE_DIR": cache_dir,
                "PCI_COMPOSE_PROJECT_DIR": "",
            }
            with patch.dict(os.environ, env, clear=False):
                result = compose_file_args()
        # Either the bundled file resolves, or the empty fallback is returned.
        self.assertIn(result[:1], ([], ["--project-directory"]))


if __name__ == "__main__":
    _ = unittest.main()
