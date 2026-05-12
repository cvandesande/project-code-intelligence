"""PID file management for the Core ML embedding server."""

from __future__ import annotations

import os
import signal
from pathlib import Path

DEFAULT_PID_DIR = Path.home() / ".cache" / "project-code-intelligence"
PID_FILE_NAME = "pci-coreml-server.pid"


def pid_file_path() -> Path:
    """Return the path to the PID file for the Core ML server."""
    return DEFAULT_PID_DIR / PID_FILE_NAME


def write_pid_file() -> None:
    """Write the current process PID to the PID file."""
    pid_file = pid_file_path()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    _ = pid_file.write_text(str(os.getpid()) + "\n")


def remove_pid_file() -> None:
    """Remove the PID file if it exists."""
    pid_file_path().unlink(missing_ok=True)


def read_pid_file() -> int | None:
    """Read the PID from the PID file, or None if absent/invalid."""
    try:
        text = pid_file_path().read_text().strip()
        return int(text) if text else None
    except (FileNotFoundError, ValueError):
        return None


def is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but we can't signal it
    return True


def stop_server() -> bool:
    """Stop a running Core ML server via PID file. Returns True if a signal was sent."""
    pid = read_pid_file()
    if pid is None:
        return False
    if not is_pid_alive(pid):
        remove_pid_file()
        return False
    os.kill(pid, signal.SIGTERM)
    remove_pid_file()
    return True
