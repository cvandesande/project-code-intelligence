"""The ``pci-hook run`` runtime: turn an agent hook event into an injection.

Two behaviours, shared across agents:

* ``evidence`` -- on an edit that removed a definition, emit that symbol's
  blast radius so the agent can judge "safe to cut?" while the change is fresh.
* ``reindex`` -- refresh the code index in the background (debounced/coalesced
  by the caller), serialised by a lock so runs never overlap.

Input arrives as JSON on stdin in the agent's native shape; output is written
to stdout in the agent's native shape. On no-match the runtime stays silent
and exits 0. When a removal IS detected but the index database is unreachable
(typically a wrong working directory -- the database name is cwd-inferred),
it emits a one-line warning instead of silence: a silently skipped check reads
as "no callers found", which is the dangerous direction.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from project_code_intelligence import evidence, inventory, process
from project_code_intelligence.exceptions import DatabaseConnectionError, McpProtocolError
from project_code_intelligence.hooks import detect, similar

if TYPE_CHECKING:
    from typing import IO

# Budget: ~2 x 5 lines + header keeps the injection near the ~15-line cap.
_MAX_SYMBOLS = 2
_NEIGHBORS = 0
_LOCK_NAME = "pci-reindex.lock"

# The add side runs a SEMANTIC query, not a call-shape one. Call-shape overlap was the
# original attempt and was deleted in 052a303: it fired on 11-13% of real duplicates and no
# threshold separated them from novel code. Embedding distance measured far better on the
# same ground truth -- see hooks/similar.py for the numbers and for why its gate is a
# per-language constant rather than a universal one.
#
# _REMINDER is the fallback, and it is what the hook emits whenever the query cannot run or
# returns nothing above the gate. It makes no claim that the new definition duplicates
# anything; it restates the AGENTS.md rule at the moment it applies, since AGENTS.md loads
# once per session and then sinks under the transcript. Silence is never an option on this
# path: a skipped check reads as "checked, nothing found", which is the dangerous direction.
_REMINDER = (
    "[pci add-side -- this edit defines {names}. Per AGENTS.md, call "
    "search_code_intel_semantic for what it does before writing, and read the closest hit "
    "in source: reuse or extend it rather than duplicating. No duplicate check was run; "
    "this is a reminder, not a finding.]"
)
# Distinct from _REMINDER on purpose: when the query ran, saying "no duplicate check was
# run" is false, and the honest version has to state its own weakness. The copy stays
# generic: recall is calibration- and language-specific (69% on this repo's Python gate,
# unmeasured elsewhere -- see similar.py), so no number belongs in text shipped everywhere.
_NO_HITS = (
    "[pci add-side -- this edit defines {names}. A similarity check ran against the index "
    "and found nothing close enough to show. The check misses some real duplicates, so it "
    "is weak evidence of absence, not proof: if this is a well-trodden area, search "
    "yourself with search_code_intel_semantic before finalizing.]"
)
_PRIOR_ART = (
    "[pci add-side -- this edit defines {names}. The index holds these nearby definitions "
    "(closest first, embedding distance; the index may predate this edit):\n{hits}\n"
    "Read the closest in source and reuse or extend it rather than duplicating. Ranked by "
    "similarity, not verified -- evidence, not a finding.]"
)
_QUERY_FAILED = (
    "[pci add-side -- this edit defines {names}, but the duplicate check could not run: "
    "{reason} (cwd {cwd}). Search for prior art yourself with search_code_intel_semantic "
    "before finalizing; do not read this as 'nothing found'.]"
)
_MAX_ADDED = 3
# Test code is exempt: a new test case has no prior art to reuse, so the reminder is pure
# noise there. Path gate covers tests/ dirs and *_test / test_* files; the name gate covers
# test helpers living in a source file. Deletions are NOT exempt -- a removed test is a
# coverage loss the blast-radius report should still surface.
_TEST_PATH = re.compile(r"(?:^|/)tests?/|(?:^|/)(?:test_[^/]*|[^/]*_test)\.[A-Za-z]+$")
_TEST_NAME = re.compile(r"^(?:Test|test_)")

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
    try:
        for name in removed[:max_symbols]:
            reports.extend(text for text in evidence.render_symbol_reports(name, neighbors=_NEIGHBORS) if text.strip())
    except DatabaseConnectionError as exc:
        first_line = str(exc).splitlines()[0]
        return (
            f"[pci blast-radius unavailable -- you removed {', '.join(removed)} but the index could not be "
            f"checked: {first_line} (cwd {Path.cwd()}). Verify callers manually before finalizing.]"
        )
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


def _add_side_block(file_path: str, new_string: str, added: list[str], names: str) -> str:
    """Prior-art hits for what this edit adds, or the reminder if none can be shown.

    Only the added definitions are embedded, never the edit payload: on a Write the payload
    is the whole file, and the gate in ``similar`` was calibrated on single-definition
    queries, so a blended vector would sit at distances it does not describe.

    Every failure degrades to text, never to silence. A missing index or a dead embedding
    endpoint reads as "no duplicates here" if the hook stays quiet, which is exactly the
    wrong inference to hand an agent mid-edit.
    """
    slices = detect.definition_slices(new_string, added)
    if not slices:
        return _REMINDER.format(names=names)
    try:
        # The gate is language-dependent, and inventory.language_for is the same mapping the
        # indexer used, so the query is gated the way the corpus it searches was measured.
        hits = similar.nearest(slices, language=inventory.language_for(file_path))
    except (DatabaseConnectionError, McpProtocolError, OSError) as exc:
        reason = str(exc).splitlines()[0] if str(exc).strip() else type(exc).__name__
        return _QUERY_FAILED.format(names=names, reason=reason, cwd=Path.cwd())
    if not hits:
        return _NO_HITS.format(names=names)
    repo = Path.cwd().name
    return _PRIOR_ART.format(names=names, hits="\n".join(hit.render(repo) for hit in hits))


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
    event_name = _as_str(event.get("hook_event_name")) or "PreToolUse"
    removed = detect.removed_definitions(old_string, new_string)
    if not removed:
        # Added names = the removal set with the two sides swapped. A rename shows up on
        # both sides; removal wins, since blast radius is evidence and this is only a nudge.
        added = (
            []
            if _TEST_PATH.search(file_path)
            else [n for n in detect.removed_definitions(new_string, old_string) if not _TEST_NAME.match(n)]
        )
        if added:
            names = ", ".join(added[:_MAX_ADDED])
            overflow = f" (+{len(added) - _MAX_ADDED} more)" if len(added) > _MAX_ADDED else ""
            _emit_evidence(
                agent,
                _add_side_block(file_path, new_string, added[:_MAX_ADDED], names + overflow),
                out_stream,
                event_name=event_name,
            )
        return 0
    block = _build_block(removed, _MAX_SYMBOLS)
    if block is None:
        return 0
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
