"""Constrained subprocess helpers.

All project subprocess calls go through this module so command execution rules
are easy to audit: no shell, explicit timeouts at call sites, text mode, and
argument-vector commands only.
"""

from __future__ import annotations

# Centralized, shell-free subprocess boundary.
import subprocess  # nosec B404
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from typing import TextIO

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
