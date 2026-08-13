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

import contextlib
import fcntl
import json
import os
import re
import shutil
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
# No-candidate paths (query ran and found nothing above the gate, or no definition slice
# could be extracted) stay SILENT: a message with nothing actionable in it interrupts the
# agent for no decision. The known cost, accepted deliberately (2026-08-13): recall is
# imperfect (69% on this repo's Python gate -- see similar.py), so silence now also covers
# some real duplicates. The banner carries the standing "search before you write" rule;
# only actionable output (hits, or a check that could not run) interrupts an edit.
_PRIOR_ART = (
    "[pci add-side -- this edit defines {names}. The index holds these nearby definitions "
    "(closest first, embedding distance; the index may predate this edit):\n{hits}\n"
    "Read the closest in source and reuse or extend it rather than duplicating. Ranked by "
    "similarity, not verified -- evidence, not a finding.]"
)
# Not silent like the no-hit path on purpose: a similarity query against an index that holds none
# of the file's repo returns a no-hit that carries near-zero information, and rendering it
# as the calibrated no-hit above would launder that emptiness into weak evidence.
_OUTSIDE_INDEX = (
    "[pci add-side -- this edit defines {names}, but {root} is outside the indexed repos, "
    "so no duplicate check ran against its code. Index it (pci index) or search prior art "
    "yourself; do not read this as 'nothing found'.]"
)
# Static practice text, appended to every add-side message. It restates the minimal-change
# ladder at the moment new code is written, the same delivery ponytail's benchmark showed
# cuts LOC/tokens (prompt-only, -54% LOC on feature tasks). PCI_HOOK_PRACTICE=0 turns it
# off so an A/B run can measure the text alone against the evidence-only baseline.
_PRACTICE = (
    "[pci practice -- write the smallest change that works: reuse an existing definition, "
    "then the standard library, then a platform feature, before new code. Do not add "
    "abstractions, options, or scaffolding for needs that are not in the task.]"
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


def _add_side_block(file_path: str, new_string: str, added: list[str], names: str) -> str | None:
    """Prior-art hits for what this edit adds; None (silence) when there are none to show.

    Only the added definitions are embedded, never the edit payload: on a Write the payload
    is the whole file, and the gate in ``similar`` was calibrated on single-definition
    queries, so a blended vector would sit at distances it does not describe.

    Failures still degrade to text, never to silence: a missing index or a dead embedding
    endpoint would otherwise read the same as a clean check that never happened.
    """
    slices = detect.definition_slices(new_string, added)
    if not slices:
        return None
    try:
        # The gate is language-dependent, and inventory.language_for is the same mapping the
        # indexer used, so the query is gated the way the corpus it searches was measured.
        hits = similar.nearest(slices, language=inventory.language_for(file_path), file_path=file_path)
    except similar.UnindexedRepoError as exc:
        return _OUTSIDE_INDEX.format(names=names, root=exc)
    except (DatabaseConnectionError, McpProtocolError, OSError) as exc:
        reason = str(exc).splitlines()[0] if str(exc).strip() else type(exc).__name__
        return _QUERY_FAILED.format(names=names, reason=reason, cwd=Path.cwd())
    if not hits:
        return None
    repo = Path.cwd().name
    return _PRIOR_ART.format(names=names, hits="\n".join(hit.render(repo) for hit in hits))


def _with_practice(block: str) -> str:
    if os.environ.get("PCI_HOOK_PRACTICE", "1").lower() in {"0", "false", "off"}:
        return block
    return block + "\n" + _PRACTICE


def _emit_evidence(agent: Agent, block: str, out: IO[str], *, event_name: str) -> None:
    if agent == "claude":
        # Echo the firing event (PreToolUse/PostToolUse) so hookSpecificOutput matches it.
        payload = {"hookSpecificOutput": {"hookEventName": event_name, "additionalContext": block}}
        _ = out.write(json.dumps(payload))
        return
    # opencode: the JS shim appends raw stdout to the tool result.
    _ = out.write(block)


def run_evidence(agent: Agent, *, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> int:
    # Full off-switch: lets a benchmark or a user silence the evidence hook for one
    # session without touching the installed agent config.
    if os.environ.get("PCI_HOOK_DISABLE", "").lower() in {"1", "true", "on"}:
        return 0
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
            block = _add_side_block(file_path, new_string, added[:_MAX_ADDED], names + overflow)
            if block is not None:
                _emit_evidence(agent, _with_practice(block), out_stream, event_name=event_name)
        return 0
    block = _build_block(removed, _MAX_SYMBOLS)
    if block is None:
        return 0
    _emit_evidence(agent, block, out_stream, event_name=event_name)
    return 0


# --- banner ---------------------------------------------------------------------

# Session-start banner: a named regime + persona, the delivery frame ponytail uses.
# Measurement note (2026-08-13, 4 tasks x 6 conditions x 2 reps, Haiku): no reproducible
# LOC or test-pass effect vs baseline at that sample size -- shipped as requested UX, not
# as a measured win. Details in the harness results
# (~/pci-measurement-harness/hook-practice-ab/), not in this repo.
_BANNER = (
    "PCI RADAR MODE ACTIVE. You edit with radar: the index has swept every definition in "
    "this repo, and the evidence hook pings you with nearby prior art on every edit. Do "
    "not fly blind -- sweep before you build (search_code_intel_semantic), reuse or "
    "extend what the radar shows, and check blast_radius before you remove. Write the "
    "smallest change that works: an existing definition, then the standard library, then "
    "a platform feature, before new code; no abstractions or scaffolding for needs that "
    "are not in the task. Never cut input validation, error handling, or anything "
    "explicitly requested. RADAR stays on for the whole session."
)


def run_banner(agent: Agent, *, stdout: IO[str] | None = None) -> int:
    out: IO[str] = stdout if stdout is not None else cast("IO[str]", sys.stdout)
    if agent == "claude":
        payload = {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": _BANNER}}
        _ = out.write(json.dumps(payload))
        return 0
    _ = out.write(_BANNER)
    return 0


# --- reindex --------------------------------------------------------------------


def _index_command(repo: Path) -> list[str]:
    """Command prefix that runs the indexer: `pci index` where available, with the legacy
    pci-index shim as the fallback for systems installed before the single-binary change."""
    configured = os.environ.get("PCI_INDEX_BIN")
    if configured:
        return [configured]
    for name, extra in (("pci", ["index"]), ("pci-index", [])):
        local = repo / ".venv" / "bin" / name
        if local.exists():
            return [str(local), *extra]
        found = shutil.which(name)
        if found:
            return [found, *extra]
    return ["pci-index"]


def _lock_path(repo: Path) -> Path:
    git_dir = repo / ".git"
    return (git_dir if git_dir.is_dir() else repo) / _LOCK_NAME


_MARKER_NAME = "pci-reindex.json"


def write_reindex_markers(repo_paths: list[Path], collection: str | None) -> None:
    """Record the index invocation in each indexed repo so the post-commit hook
    can replay it. Without this, a hook in a repo indexed as part of a multi-repo
    workspace would reindex it standalone -- into a different database/collection."""
    payload = json.dumps({"cwd": str(Path.cwd()), "repo_paths": [str(p) for p in repo_paths], "collection": collection})
    for repo in repo_paths:
        git_dir = repo / ".git"
        if git_dir.is_dir():
            with contextlib.suppress(OSError):
                _ = (git_dir / _MARKER_NAME).write_text(payload, encoding="utf-8")


def reindex_target(repo: Path) -> tuple[Path, list[str]] | None:
    """Resolve (cwd, index args) for a reindex from the workspace invocation
    recorded at index time, or None when no valid marker exists.

    A marker-less repo was never indexed deliberately from that path (every
    `pci index` run writes one), so a reindex there would full-index into a
    fresh collection. Git worktrees hit this: they inherit the parent repo's
    post-commit hook, but `.git` is a file, so no marker is ever present."""
    try:
        data = cast("object", json.loads((repo / ".git" / _MARKER_NAME).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
    marker = _as_object(data)
    cwd = marker.get("cwd")
    raw = marker.get("repo_paths")
    if not (isinstance(cwd, str) and Path(cwd).is_dir() and isinstance(raw, list)):
        return None
    raw_paths = cast("list[object]", raw)
    paths = [p for p in raw_paths if isinstance(p, str) and Path(p).is_dir()]
    if not paths or len(paths) != len(raw_paths):
        return None
    collection = marker.get("collection")
    args = ["--collection", collection, *paths] if isinstance(collection, str) else paths
    return Path(cwd), args


def run_reindex(repo: Path) -> int:
    """Run pci-index for ``repo`` unless another run holds the lock.

    Blocking by design: callers run this in the background (an async agent hook
    or a debounced detached spawn), and the lock coalesces overlapping calls.
    Skips silently when no valid reindex marker exists (see reindex_target).
    """
    target = reindex_target(repo)
    if target is None:
        return 0
    workspace, index_args = target
    lock_path = _lock_path(workspace)
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
                [*_index_command(repo), *index_args],
                process.RunOptions(cwd=workspace, stdout=process.DEVNULL, stderr=process.DEVNULL),
            )
        except (process.SubprocessError, OSError):
            return 0  # binary missing / not runnable -> stay silent, index just not refreshed
    finally:
        lock_file.close()
    return 0
