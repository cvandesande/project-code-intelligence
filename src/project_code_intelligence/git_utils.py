"""Shared Git and workspace helpers for code-intelligence tooling."""

from __future__ import annotations

import shutil
from pathlib import Path

from project_code_intelligence import process

GIT_TIMEOUT_SECONDS = 30


def git_binary() -> str | None:
    return shutil.which("git")


def workspace_root() -> Path:
    """Return the workspace root for CLI defaults.

    Prefer the Git top-level for the current directory, but fall back to the
    current directory so standalone checkouts and non-Git smoke tests do not
    inherit an unrelated parent checkout.
    """
    cwd = Path.cwd().resolve()
    binary = git_binary()
    if binary is None:
        return cwd
    try:
        proc = process.run(
            [binary, "rev-parse", "--show-toplevel"],
            process.RunOptions(
                cwd=cwd,
                check=True,
                stdout=process.PIPE,
                stderr=process.DEVNULL,
                timeout=GIT_TIMEOUT_SECONDS,
            ),
        )
    except (OSError, process.CalledProcessError):
        return cwd
    return Path(proc.stdout.strip()).resolve()


def run_git(root: Path, args: list[str]) -> str | None:
    binary = git_binary()
    if binary is None:
        return None
    try:
        proc = process.run(
            [binary, *args],
            process.RunOptions(
                cwd=root,
                check=True,
                stdout=process.PIPE,
                stderr=process.DEVNULL,
                timeout=GIT_TIMEOUT_SECONDS,
            ),
        )
    except (OSError, process.CalledProcessError):
        return None
    return proc.stdout.strip() or None
