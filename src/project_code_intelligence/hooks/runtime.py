"""The ``pci-hook run`` runtime: turn an agent hook event into an injection.

Two behaviours, shared across agents:

* ``evidence`` -- on an edit that removed a definition, emit that symbol's
  blast radius so the agent can judge "safe to cut?" while the change is fresh;
  on an edit that added one, emit existing functions sharing its call-shape so
  the agent can judge "does this already exist?" before the duplicate lands.
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
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, cast

from project_code_intelligence import analyze, evidence, process, records
from project_code_intelligence.hooks import detect
from project_code_intelligence.mcp import db

if TYPE_CHECKING:
    from typing import IO

# Budget: ~2 x 5 lines + header keeps the injection near the ~15-line cap.
_MAX_SYMBOLS = 2
_NEIGHBORS = 0
_LOCK_NAME = "pci-reindex.lock"

# Anti-slop on add. Structural, not semantic: embedding similarity does not separate a
# duplicate from code that merely resembles its neighbours (a real duplicate measured
# 0.57 against 0.55 for unrelated new code), so a cosine hook would fire on every
# addition. IDF-weighted call-shape overlap is the signal the compression pass already
# clusters on, and it does separate.
#
# Do not trust this hook to catch a duplicate. Measured twice, 30 blind
# reimplementations each (a model handed only a signature and docstring, never the body
# -- the input distribution that matters), it fires on 11-13% of them at this threshold:
#   * rank carries real signal: the duplicated function is the top hit of 1385 about 60%
#     of the time, top 3 in 64-70%, top 10 in 87%;
#   * absolute score does not: a real reimplementation scores 0.36-0.39 median, inside
#     the range unrelated code produces, so no threshold separates them. At 0.40 recall
#     reaches 0.5 but 18-27% of genuinely-novel functions also fire, and this channel is
#     shared with the removal hook -- noise there costs more than the recall is worth.
# The second run gave the model the target module's import block, to rule out that the
# first had starved it of real helper names. It changed almost nothing: only 26% of the
# callees a fresh implementation writes name anything already in the repo (23% before).
# That is not a harness artifact, it is what new code looks like.
# The threshold is therefore set for quiet, not for recall, and equals the compression
# pass's own so there is no second magic number to drift.
#
# An earlier calibration over the index put recall near 1.00. It scored each function
# against its OWN text, where extraction agrees almost exactly -- a tautology, not a
# measurement. Hand-written "plausible duplicate" probes are just as optimistic: one
# scored 0.66 where the blind rewrite of the same function scored 0.13. Re-measure with
# blind rewrites, or not at all.
#
# The minimum role count drops functions too thin to have a shape: 25% of functions
# here (353 of 1385) never reach it and the hook is silent on them.
_SHAPE_THRESHOLD = analyze.DEFAULT_THRESHOLD
_SHAPE_MIN_ROLES = analyze.DEFAULT_MIN_ROLES
_SHAPE_MATCHES = 3

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


def _disk_text(path: str) -> str:
    """The file as it stands before the write ("" if unreadable, so we stay silent)."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _edit_fields(agent: Agent, event: dict[str, object]) -> tuple[str, str, str]:
    """Return (file_path, old_string, new_string) for the agent's edit event."""
    if agent == "claude":
        tool_input = _as_object(event.get("tool_input"))
        file_path = _as_str(tool_input.get("file_path"))
        content = tool_input.get("content")
        if isinstance(content, str):
            # Write replaces the file wholesale, so the old side has to come from disk.
            # PreToolUse fires before the write; on PostToolUse disk already holds the new
            # text, which yields an empty diff -- silent, never a false removal.
            return file_path, _disk_text(file_path), content
        return (
            file_path,
            _as_str(tool_input.get("old_string")),
            _as_str(tool_input.get("new_string")),
        )
    # opencode: the JS shim forwards the edit tool args verbatim.
    return (
        _as_str(event.get("filePath")),
        _as_str(event.get("oldString")),
        _as_str(event.get("newString")),
    )


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


def shape_report(added: list[str], new_string: str) -> str | None:
    """Existing functions whose call-shape matches the single definition being added.

    Only fires for a one-definition edit: the shape is read from the whole new text, so
    two additions at once would blend into one meaningless shape. Silent on any failure.
    """
    if len(added) != 1:
        return None
    roles = analyze.role_set(records.extract_referenced_symbols(new_string))
    if len(roles) < _SHAPE_MIN_ROLES:
        return None
    try:
        with db.connect() as conn:
            if not db.code_intel_tables_exist(conn):
                return None
            matches = [
                (node, score)
                for snapshot in analyze.latest_snapshots(conn)
                for node, score in analyze.shape_matches(
                    conn, snapshot, roles, threshold=_SHAPE_THRESHOLD, limit=_SHAPE_MATCHES
                )
            ]
    except Exception:  # noqa: BLE001 -- a hook never breaks the tool call it decorates
        return None
    if not matches:
        return None
    ranked = sorted(matches, key=itemgetter(1), reverse=True)[:_SHAPE_MATCHES]
    header = (
        f"[pci anti-slop -- '{added[0]}' has the call-shape of existing code. "
        "Evidence, not a verdict: read these before duplicating, then reuse, extend, or proceed.]"
    )
    lines = [f"  {score:.2f}  {node.symbol}  {node.source_path}:{node.line_start}" for node, score in ranked]
    return header + "\n" + "\n".join(lines)


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
    file_path, old_string, new_string = _edit_fields(agent, event)
    if not detect.is_source_path(file_path):
        return 0
    removed = detect.removed_definitions(old_string, new_string)
    # A rename reads as both a removal and an addition; the removal is the costlier
    # mistake, so it wins and the shape check never competes with it.
    block = (
        _build_block(removed, _MAX_SYMBOLS)
        if removed
        else shape_report(detect.added_definitions(old_string, new_string), new_string)
    )
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
