"""Constrained subprocess helpers.

All project subprocess calls go through this module so command execution rules
are easy to audit: no shell, explicit timeouts at call sites, text mode, and
argument-vector commands only.
"""

from __future__ import annotations

import os
import shutil

# Centralized, shell-free subprocess boundary.
import subprocess  # nosec B404
from dataclasses import dataclass
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
_CONTAINER_ENGINE_ENV_VAR = "PROJECT_CODE_INTELLIGENCE_CONTAINER_ENGINE"

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
