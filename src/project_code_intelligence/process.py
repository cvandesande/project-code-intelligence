"""Constrained subprocess helpers.

All project subprocess calls go through this module so command execution rules
are easy to audit: no shell, explicit timeouts at call sites, text mode, and
argument-vector commands only.
"""

from __future__ import annotations

import contextlib
import os
import shutil

# Centralized, shell-free subprocess boundary.
import subprocess  # nosec B404
import sys
import tempfile
from dataclasses import dataclass
from importlib import resources as importlib_resources
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import TextIO

# Names of container engines we know how to drive. Order matters: docker is
# tried first because that's still the most common config, then podman as the
# drop-in replacement. Operators can pin via the env var below to skip the
# search entirely.
_CONTAINER_ENGINE_CANDIDATES = ("docker", "podman")
_CONTAINER_ENGINE_ENV_VAR = "PCI_CONTAINER_ENGINE"

PIPE = subprocess.PIPE
STDOUT = subprocess.STDOUT
DEVNULL = subprocess.DEVNULL
CalledProcessError = subprocess.CalledProcessError
CompletedProcess = subprocess.CompletedProcess
SubprocessError = subprocess.SubprocessError
TimeoutExpired = subprocess.TimeoutExpired


@dataclass(frozen=True)
class RunOptions:
    cwd: Path | None = None
    env: Mapping[str, str] | None = None
    input_text: str | None = None
    capture_output: bool = False
    stdout: int | TextIO | None = None
    stderr: int | TextIO | None = None
    timeout: float | None = None
    check: bool = False


@dataclass(frozen=True)
class PopenOptions:
    cwd: Path | None = None
    env: Mapping[str, str] | None = None
    stdout: int | TextIO | None = None
    stderr: int | TextIO | None = None
    stdin: int | TextIO | None = None
    start_new_session: bool = False


def run(command: Sequence[str], options: RunOptions | None = None) -> subprocess.CompletedProcess[str]:
    options = options or RunOptions()
    if not command:
        msg = "subprocess command must not be empty"
        raise ValueError(msg)
    if any(not part for part in command):
        msg = "subprocess command arguments must be non-empty strings"
        raise ValueError(msg)
    # shell=False and validated argv; callers supply fixed commands or trusted config.
    return subprocess.run(  # nosec B603
        list(command),
        cwd=options.cwd,
        env=dict(options.env) if options.env is not None else None,
        input=options.input_text,
        text=True,
        capture_output=options.capture_output,
        stdout=options.stdout,
        stderr=options.stderr,
        timeout=options.timeout,
        check=options.check,
        shell=False,
    )


def popen(command: Sequence[str], options: PopenOptions | None = None) -> subprocess.Popen[str]:
    opts = options or PopenOptions()
    if not command:
        msg = "subprocess command must not be empty"
        raise ValueError(msg)
    if any(not part for part in command):
        msg = "subprocess command arguments must be non-empty strings"
        raise ValueError(msg)
    return subprocess.Popen(  # nosec B603
        list(command),
        cwd=opts.cwd,
        env=dict(opts.env) if opts.env is not None else None,
        stdout=opts.stdout,
        stderr=opts.stderr,
        stdin=opts.stdin,
        text=True,
        start_new_session=opts.start_new_session,
        shell=False,
    )


# === Container engine detection ===========================================
#
# We accept docker or podman as the container engine — they share enough CLI
# surface (`<engine> compose`, `<engine> info`, etc.) for the things we drive.
# A single detector keeps every callsite consistent and lets operators pin
# via the env var.


def container_engine_path() -> str | None:
    """Return the absolute path of the configured container engine, or None if absent.

    Not cached: shutil.which is cheap and callers may legitimately re-probe (tests,
    or environments where PATH changes mid-run). If you need a stable handle within
    a single high-frequency loop, capture the result in a local.
    """
    override = os.environ.get(_CONTAINER_ENGINE_ENV_VAR, "").strip()
    if override:
        return shutil.which(override)
    for candidate in _CONTAINER_ENGINE_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
    return None


def container_engine_name() -> str:
    """Return the basename of the resolved engine ('docker' or 'podman'), or 'docker' as a label fallback."""
    path = container_engine_path()
    if path is None:
        # Fall back to 'docker' purely for display in error messages.
        return "docker"
    return Path(path).name


# === Docker / Podman sandbox helper ========================================
#
# All container-engine invocations go through `run_docker` so we have one
# place that (a) restricts the executable to a known engine, (b) resolves it
# to an absolute path so PATH manipulation can't swap the binary, and (c)
# rejects argument shapes known to escalate privileges or expose the host.

_DOCKER_ARG_BLOCKLIST = {
    "--privileged",
    "--cap-add=ALL",
    "--security-opt=label=disable",
    "--pid=host",
    "--ipc=host",
    "--network=host",
    "--userns=host",
}
_DOCKER_DANGEROUS_VOLUME_PREFIXES = (
    "/:",
    "/etc:",
    "/var:",
    "/root:",
    "/home:",
    "/dev:",
    "/proc:",
    "/sys:",
)


def _argument_is_blocked(arg: str) -> bool:
    if arg in _DOCKER_ARG_BLOCKLIST:
        return True
    if arg.startswith(("-v=", "--volume=")):
        value = arg.split("=", 1)[1]
        if value.startswith(_DOCKER_DANGEROUS_VOLUME_PREFIXES):
            return True
    return False


def _audit_docker_args(args: Sequence[str]) -> None:
    saw_volume_flag = False
    for arg in args:
        if _argument_is_blocked(arg):
            raise ValueError(f"docker argument refused by run_docker: {arg!r}")
        if saw_volume_flag and arg.startswith(_DOCKER_DANGEROUS_VOLUME_PREFIXES):
            raise ValueError(f"docker volume mount refused by run_docker: {arg!r}")
        saw_volume_flag = arg in {"-v", "--volume"}


_COMPOSE_FILE_ENV_VAR = "PCI_COMPOSE_FILE"
_PACKAGE_NAME = "project_code_intelligence"
_COMPOSE_RESOURCE = "docker-compose.yml"
_COMPOSE_PROJECT_DIR_ENV_VAR = "PCI_COMPOSE_PROJECT_DIR"
_COMPOSE_CACHE_DIR_ENV_VAR = "PCI_COMPOSE_CACHE_DIR"
_APP_CACHE_DIR_NAME = "project-code-intelligence"
_INIT_EXTENSIONS_RESOURCE = Path("docker/pgvector/init-extensions.sql")
_COMPOSE_CONTEXT_PACKAGE_FILES = (
    (Path("docker/build-context/pyproject.toml"), Path("pyproject.toml")),
    (Path("docker/build-context/README.md"), Path("README.md")),
    (Path("docker/build-context/LICENSE"), Path("LICENSE")),
    (Path("docker/fastembed/Dockerfile"), Path("docker/fastembed/Dockerfile")),
    (Path("docker/llamacpp-rocm/Dockerfile"), Path("docker/llamacpp-rocm/Dockerfile")),
    (Path("docker/llamacpp-rocm/entrypoint.sh"), Path("docker/llamacpp-rocm/entrypoint.sh")),
    (Path("docker/llamacpp-cuda/Dockerfile"), Path("docker/llamacpp-cuda/Dockerfile")),
    (Path("docker/llamacpp-cuda/entrypoint.sh"), Path("docker/llamacpp-cuda/entrypoint.sh")),
    (Path("scripts/select_llamacpp_rocm_bundle.py"), Path("scripts/select_llamacpp_rocm_bundle.py")),
    (Path("bin/pci-embedding-server"), Path("pci-embedding-server")),
)
_COMPOSE_CONTEXT_PACKAGE_SOURCE_FILES = (
    Path("__init__.py"),
    Path("common.py"),
    Path("config.py"),
    Path("exceptions.py"),
    Path("http_client.py"),
    Path("process.py"),
    Path("rocm_bundles.py"),
    Path("runtime.py"),
    Path("embedding/__init__.py"),
    Path("embedding/fastembed_server.py"),
    Path("embedding/http_common.py"),
)


def _cache_root() -> Path:
    override = os.environ.get(_COMPOSE_CACHE_DIR_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / _APP_CACHE_DIR_NAME
    home = os.environ.get("HOME", "").strip()
    if home:
        if sys.platform == "darwin":
            return Path(home).expanduser() / "Library" / "Caches" / _APP_CACHE_DIR_NAME
        return Path(home).expanduser() / ".cache" / _APP_CACHE_DIR_NAME
    return Path(tempfile.gettempdir()) / _APP_CACHE_DIR_NAME


def compose_cache_dir() -> Path:
    """Return the directory used for materialized bundled Compose assets."""
    return _cache_root()


def _write_text_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.parent.chmod(0o700)
    try:
        if path.read_text(encoding="utf-8") == content:
            return
    except OSError:
        pass
    if path.exists():
        with contextlib.suppress(OSError):
            path.chmod(0o600)
    _ = path.write_text(content, encoding="utf-8")
    with contextlib.suppress(OSError):
        path.chmod(0o600)


def _copy_file_if_changed(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        target.parent.chmod(0o700)
    try:
        if target.exists() and target.read_bytes() == source.read_bytes():
            return
    except OSError:
        pass
    if target.exists():
        with contextlib.suppress(OSError):
            target.chmod(0o600)
    _ = shutil.copyfile(source, target)
    with contextlib.suppress(OSError):
        target.chmod(0o600)


def _remove_generated_tree(path: Path) -> None:
    if not path.exists():
        return
    for current_dir, dir_names, file_names in os.walk(path):
        current_path = Path(current_dir)
        with contextlib.suppress(OSError):
            current_path.chmod(0o700)
        for name in dir_names:
            with contextlib.suppress(OSError):
                (current_path / name).chmod(0o700)
        for name in file_names:
            with contextlib.suppress(OSError):
                (current_path / name).chmod(0o600)
    shutil.rmtree(path)


def _copy_minimal_package_source(package_dir: Path, target_dir: Path) -> None:
    _remove_generated_tree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    for relative_path in _COMPOSE_CONTEXT_PACKAGE_SOURCE_FILES:
        source = package_dir / relative_path
        if source.is_file():
            _copy_file_if_changed(source, target_dir / relative_path)


def _copy_package_context_assets(package_dir: Path, context_dir: Path) -> None:
    for package_relative, context_relative in _COMPOSE_CONTEXT_PACKAGE_FILES:
        source = package_dir / package_relative
        if source.is_file():
            _copy_file_if_changed(source, context_dir / context_relative)


def _package_dir() -> Path | None:
    try:
        ref = importlib_resources.files(_PACKAGE_NAME)
    except (ModuleNotFoundError, TypeError, AttributeError):
        return None
    path = Path(str(ref))
    return path if path.exists() else None


def _source_checkout_root(package_dir: Path) -> Path | None:
    candidate = package_dir.parent.parent
    if (candidate / "pyproject.toml").is_file() and (candidate / "docker-compose.yml").is_file():
        return candidate
    return None


def _materialize_installed_project_context(package_dir: Path, context_dir: Path) -> Path:
    _copy_minimal_package_source(package_dir, context_dir / "src" / _PACKAGE_NAME)
    _copy_package_context_assets(package_dir, context_dir)
    (context_dir / "models").mkdir(parents=True, exist_ok=True)
    return context_dir


def _materialize_compose_file(package_dir: Path) -> tuple[Path, Path]:
    override_project_dir = os.environ.get(_COMPOSE_PROJECT_DIR_ENV_VAR, "").strip()
    if override_project_dir:
        project_dir = Path(override_project_dir).expanduser()
    elif source_root := _source_checkout_root(package_dir):
        project_dir = source_root
    else:
        project_dir = _materialize_installed_project_context(package_dir, _cache_root() / "compose-context")

    source_compose = package_dir / _COMPOSE_RESOURCE
    compose_text = source_compose.read_text(encoding="utf-8")
    init_extensions = project_dir / _INIT_EXTENSIONS_RESOURCE
    if not init_extensions.is_file():
        package_init_extensions = package_dir / _INIT_EXTENSIONS_RESOURCE
        if package_init_extensions.is_file():
            _copy_file_if_changed(package_init_extensions, init_extensions)
    if init_extensions.is_file():
        compose_text = compose_text.replace(f"./{_INIT_EXTENSIONS_RESOURCE.as_posix()}", str(init_extensions))

    materialized = _cache_root() / "docker-compose.yml"
    _write_text_if_changed(materialized, compose_text)
    return materialized, project_dir


def compose_file_args() -> list[str]:
    """Return ["-f", "/path/to/docker-compose.yml"] for use in compose subcommands.

    Resolution order:
    1. PCI_COMPOSE_FILE environment variable — use this to
       point at a customised compose file without modifying the installed package.
    2. Materialized bundled docker-compose.yml from installed package data.
    3. Empty list — callers degrade to CWD-based discovery.
    """
    override = os.environ.get(_COMPOSE_FILE_ENV_VAR, "").strip()
    if override:
        return ["-f", override]
    package_dir = _package_dir()
    if package_dir is not None and (package_dir / _COMPOSE_RESOURCE).is_file():
        compose_path, project_dir = _materialize_compose_file(package_dir)
        return ["--project-directory", str(project_dir), "-f", str(compose_path)]
    return []


def run_docker(args: Sequence[str], options: RunOptions | None = None) -> subprocess.CompletedProcess[str]:
    """Run a docker- or podman-compatible subprocess with arg sandbox and absolute-path resolution."""
    if not args:
        msg = "container engine invocation must include at least one subcommand"
        raise ValueError(msg)
    resolved = container_engine_path()
    if resolved is None:
        msg = "no container engine (docker or podman) found on PATH"
        raise FileNotFoundError(msg)
    _audit_docker_args(args)
    return run([resolved, *args], options)
