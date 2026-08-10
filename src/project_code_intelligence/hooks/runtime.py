"""The ``pci-hook run`` runtime: turn an agent hook event into an injection.

Two behaviours, shared across agents:

* ``evidence`` -- on an edit that removed a definition, emit that symbol's
  blast radius so the agent can judge "safe to cut?" while the change is fresh.
* ``reindex`` -- refresh the code index in the background (debounced/coalesced
  by the caller), serialised by a lock so runs never overlap.

Input arrives as JSON on stdin in the agent's native shape; output is written
to stdout in the agent's native shape. On no-match or any failure the runtime
stays silent and exits 0, so it never breaks the tool call.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
from typing import TYPE_CHECKING, cast

from project_code_intelligence import evidence, process
from project_code_intelligence.hooks import detect

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO

# Budget: ~2 x 5 lines + header keeps the injection near the ~15-line cap.
_MAX_SYMBOLS = 2
_NEIGHBORS = 0
_LOCK_NAME = "pci-reindex.lock"

Agent = str  # "opencode" | "claude"


# --- input coercion -------------------------------------------------------------


def _as_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    typed = cast("dict[object, object]", value)
    return {str(k): v for k, v in typed.items()}


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _read_json(stream: IO[str]) -> dict[str, object]:
    raw = stream.read()
    if not raw.strip():
        return {}
    try:
        loaded = cast("object", json.loads(raw))
    except json.JSONDecodeError:
        return {}
    return _as_object(loaded)


# --- event -> (file_path, removed names) ----------------------------------------


def _edit_fields(agent: Agent, event: dict[str, object]) -> tuple[str, str, str]:
    """Return (file_path, old_string, new_string) for the agent's edit event."""
    if agent == "claude":
        tool_input = _as_object(event.get("tool_input"))
        return (
            _as_str(tool_input.get("file_path")),
            _as_str(tool_input.get("old_string")),
            _as_str(tool_input.get("new_string")),
        )
    # opencode: the JS shim forwards the edit tool args verbatim.
    return (
        _as_str(event.get("filePath")),
        _as_str(event.get("oldString")),
        _as_str(event.get("newString")),
    )


def _removed_symbols(agent: Agent, event: dict[str, object]) -> list[str]:
    file_path, old_string, new_string = _edit_fields(agent, event)
    if not detect.is_source_path(file_path):
        return []
    return detect.removed_definitions(old_string, new_string)


# --- evidence block -------------------------------------------------------------


def _build_block(removed: list[str], max_symbols: int) -> str | None:
    reports: list[str] = []
    for name in removed[:max_symbols]:
        reports.extend(text for text in evidence.render_symbol_reports(name, neighbors=_NEIGHBORS) if text.strip())
    if not reports:
        return None
    hidden = len(removed) - max_symbols
    overflow = f" (+{hidden} more removed, not shown)" if hidden > 0 else ""
    header = (
        f"[pci blast-radius{overflow} -- you removed the symbol(s) below. "
        "The index likely predates this edit, so treat as approximate; "
        "confirm no live caller before finalizing.]"
    )
    return header + "\n" + "\n---\n".join(report.rstrip("\n") for report in reports)


def _emit_evidence(agent: Agent, block: str, out: IO[str], *, event_name: str) -> None:
    if agent == "claude":
        # Echo the firing event (PreToolUse/PostToolUse) so hookSpecificOutput matches it.
        payload = {"hookSpecificOutput": {"hookEventName": event_name, "additionalContext": block}}
        _ = out.write(json.dumps(payload))
        return
    # opencode: the JS shim appends raw stdout to the tool result.
    _ = out.write(block)


def run_evidence(agent: Agent, *, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> int:
    in_stream: IO[str] = stdin if stdin is not None else cast("IO[str]", sys.stdin)
    out_stream: IO[str] = stdout if stdout is not None else cast("IO[str]", sys.stdout)
    event = _read_json(in_stream)
    removed = _removed_symbols(agent, event)
    if not removed:
        return 0
    block = _build_block(removed, _MAX_SYMBOLS)
    if block is None:
        return 0
    event_name = _as_str(event.get("hook_event_name")) or "PreToolUse"
    _emit_evidence(agent, block, out_stream, event_name=event_name)
    return 0


# --- reindex --------------------------------------------------------------------


def _index_bin(repo: Path) -> str:
    configured = os.environ.get("PCI_INDEX_BIN")
    if configured:
        return configured
    local = repo / ".venv" / "bin" / "pci-index"
    return str(local) if local.exists() else "pci-index"


def _lock_path(repo: Path) -> Path:
    git_dir = repo / ".git"
    return (git_dir if git_dir.is_dir() else repo) / _LOCK_NAME


def run_reindex(repo: Path) -> int:
    """Run pci-index for ``repo`` unless another run holds the lock.

    Blocking by design: callers run this in the background (an async agent hook
    or a debounced detached spawn), and the lock coalesces overlapping calls.
    """
    lock_path = _lock_path(repo)
    try:
        lock_file = lock_path.open("w")
    except OSError:
        return 0
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0  # another reindex is in flight; it will pick up the latest state
        try:
            _ = process.run(
                [_index_bin(repo), str(repo)],
                process.RunOptions(cwd=repo, stdout=process.DEVNULL, stderr=process.DEVNULL),
            )
        except (process.SubprocessError, OSError):
            return 0  # binary missing / not runnable -> stay silent, index just not refreshed
    finally:
        lock_file.close()
    return 0
