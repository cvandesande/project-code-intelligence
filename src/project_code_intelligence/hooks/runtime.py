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
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

from project_code_intelligence import evidence, git_utils, inventory, process
from project_code_intelligence.console_ui import as_object
from project_code_intelligence.exceptions import DatabaseConnectionError, McpProtocolError
from project_code_intelligence.hooks import detect, similar

if TYPE_CHECKING:
    from typing import IO

# Budget: ~2 x 5 lines + header keeps the injection near the ~15-line cap.
_MAX_SYMBOLS = 2
_NEIGHBORS = 0
_LOCK_NAME = "pci-reindex.lock"
_STATE_NAME = "pci-reindex-state.json"

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

Agent = str  # "opencode" | "claude" | "codex"


# --- input coercion -------------------------------------------------------------


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
    return as_object(loaded)


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
        tool_input = as_object(event.get("tool_input"))
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


def _codex_edits(event: dict[str, object]) -> list[tuple[str, str, str]]:
    """Extract per-file before/after fragments from a Codex apply_patch envelope."""
    command = _as_str(as_object(event.get("tool_input")).get("command"))
    edits: list[tuple[str, str, str]] = []
    path = ""
    mode = ""
    before: list[str] = []
    after: list[str] = []

    def finish() -> None:
        nonlocal path, mode, before, after
        if not path:
            return
        if mode == "delete":
            edits.append((path, _disk_text(path), ""))
        else:
            edits.append((path, "\n".join(before), "\n".join(after)))
        path, mode, before, after = "", "", [], []

    markers = (
        ("*** Update File: ", "update"),
        ("*** Add File: ", "add"),
        ("*** Delete File: ", "delete"),
    )
    for line in command.splitlines():
        marker = next(((prefix, kind) for prefix, kind in markers if line.startswith(prefix)), None)
        if marker is not None:
            finish()
            prefix, mode = marker
            path = line.removeprefix(prefix).strip()
        elif line.startswith("*** Move to: "):
            path = line.removeprefix("*** Move to: ").strip()
        elif path and mode != "delete" and line.startswith("+"):
            after.append(line[1:])
        elif path and mode == "update" and line.startswith("-"):
            before.append(line[1:])
    finish()
    return edits


def _event_edits(agent: Agent, event: dict[str, object]) -> list[tuple[str, str, str]]:
    if agent == "codex":
        return _codex_edits(event)
    return [_edit_fields(agent, event)]


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
    if agent in {"claude", "codex"}:
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
    event_name = _as_str(event.get("hook_event_name")) or "PreToolUse"
    blocks: list[str] = []
    for file_path, old_string, new_string in _event_edits(agent, event):
        if not detect.is_source_path(file_path):
            continue
        removed = detect.removed_definitions(old_string, new_string)
        if removed:
            block = _build_block(removed, _MAX_SYMBOLS)
            if block is not None:
                blocks.append(block)
            continue
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
                blocks.append(_with_practice(block))
    if blocks:
        _emit_evidence(agent, "\n\n".join(blocks), out_stream, event_name=event_name)
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
    if agent in {"claude", "codex"}:
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


def _state_path(repo: Path) -> Path:
    git_dir = repo / ".git"
    return (git_dir if git_dir.is_dir() else repo) / _STATE_NAME


def _read_reindex_state(repo: Path) -> dict[str, object]:
    try:
        value = cast("object", json.loads(_state_path(repo).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}
    return dict(cast("dict[str, object]", value)) if isinstance(value, dict) else {}


def _write_reindex_state(repo: Path, **changes: object) -> dict[str, object]:
    state = _read_reindex_state(repo)
    state.update(changes)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = _state_path(repo)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        _ = temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        _ = temporary.replace(path)
    except OSError:
        with contextlib.suppress(OSError):
            temporary.unlink()
    return state


def reindex_status(repo: Path) -> dict[str, object]:
    state = _read_reindex_state(repo)
    state.update({
        "repo": str(repo),
        "marker_valid": reindex_target(repo) is not None,
        "systemd_run": shutil.which("systemd-run"),
        "head": (git_utils.run_git(repo, ["rev-parse", "HEAD"]) or "").strip() or None,
    })
    return state


def _runtime_command() -> list[str]:
    invoked = Path(sys.argv[0]).resolve(strict=False)
    if invoked.name == "pci":
        return [str(invoked), "hook", "run"]
    if invoked.name.startswith("pci-hook"):
        return [str(invoked), "run"]
    return [sys.executable, "-m", "project_code_intelligence.hooks.cli", "run"]


def submit_reindex(repo: Path) -> int:
    head = (git_utils.run_git(repo, ["rev-parse", "HEAD"]) or "").strip() or None
    unit = f"pci-reindex-{hashlib.sha256(str(repo).encode()).hexdigest()[:12]}-{os.getpid()}-{time.time_ns()}"
    _ = _write_reindex_state(repo, outcome="submitting", requested_head=head, unit=unit, error=None)
    systemd_run = shutil.which("systemd-run")
    if systemd_run is not None:
        worker = [*_runtime_command(), "--target", "git", "--behavior", "reindex", "--repo", str(repo)]
        result = process.run(
            [
                systemd_run,
                "--user",
                "--quiet",
                "--collect",
                f"--unit={unit}",
                "--property=Type=exec",
                "--property=SyslogIdentifier=pci-reindex",
                "--",
                *worker,
            ],
            process.RunOptions(capture_output=True),
        )
        if result.returncode == 0:
            _ = _write_reindex_state(repo, outcome="submitted", requested_head=head, unit=unit)
            return 0
        reason = result.stderr.strip() or f"systemd-run exited {result.returncode}"
        _ = _write_reindex_state(repo, outcome="submission_failed", error=reason)
    else:
        _ = _write_reindex_state(repo, outcome="systemd_unavailable", error="systemd-run not found")
    return run_reindex(repo)


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


def _read_marker(git_dir: Path) -> tuple[Path, list[str], str | None] | None:
    """Validated (cwd, repo_paths, collection) from a marker file in ``git_dir``, or
    None on any I/O, parse, or shape problem. Shared by the direct and worktree paths
    through ``reindex_target`` so the validation rules live in exactly one place."""
    try:
        data = cast("object", json.loads((git_dir / _MARKER_NAME).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
    marker = as_object(data)
    cwd = marker.get("cwd")
    raw = marker.get("repo_paths")
    if not (isinstance(cwd, str) and Path(cwd).is_dir() and isinstance(raw, list)):
        return None
    raw_paths = cast("list[object]", raw)
    paths = [p for p in raw_paths if isinstance(p, str) and Path(p).is_dir()]
    if not paths or len(paths) != len(raw_paths):
        return None
    collection = marker.get("collection")
    return Path(cwd), paths, collection if isinstance(collection, str) else None


def _worktree_reindex_target(repo: Path) -> tuple[Path, list[str]] | None:
    """(cwd, index args) for a commit made inside a linked worktree, reindexing that
    worktree's checkout under its MAIN repo's collection/repo identity -- never a new
    collection keyed on the worktree's own directory name.

    The worktree's own `.git` is a file, so its marker lives with the main repo instead
    (worktrees never get their own marker -- write_reindex_markers only writes into
    directory `.git`s). Only replay when the main repo's resolved root is itself one of
    that marker's `repo_paths`: this is the same "was this path indexed on purpose"
    guard `reindex_target` applies to the non-worktree case, so an unrelated worktree
    whose main repo happens to have some other marker still safely no-ops.
    """
    main_root = git_utils.worktree_main_root(repo)
    if main_root is None:
        return None
    marker = _read_marker(main_root / ".git")
    if marker is None:
        return None
    cwd, paths, collection = marker
    resolved_main = main_root.resolve()
    if resolved_main not in {Path(p).resolve() for p in paths}:
        return None
    collection_args = ["--collection", collection] if collection is not None else []
    return cwd, [*collection_args, "--worktree", f"{resolved_main}={repo.resolve()}"]


def reindex_target(repo: Path) -> tuple[Path, list[str]] | None:
    """Resolve (cwd, index args) for a reindex from the workspace invocation
    recorded at index time, or None when no valid marker exists.

    A marker-less repo was never indexed deliberately from that path (every
    `pci index` run writes one), so a reindex there would full-index into a
    fresh collection. Git worktrees hit this differently: `.git` is a file there, so
    no marker is ever written for the worktree itself -- see _worktree_reindex_target,
    which replays against the MAIN repo's marker instead."""
    git_entry = repo / ".git"
    if not git_entry.is_dir():
        return _worktree_reindex_target(repo)
    marker = _read_marker(git_entry)
    if marker is None:
        return None
    cwd, paths, collection = marker
    args = ["--collection", collection, *paths] if collection is not None else paths
    return cwd, args


def _run_locked_reindex(repo: Path, workspace: Path, index_args: list[str]) -> int:
    state = _read_reindex_state(repo)
    requested_head = state.get("requested_head")
    if requested_head and state.get("completed_head") == requested_head:
        _ = _write_reindex_state(repo, outcome="already_current", error=None)
        return 0
    _ = _write_reindex_state(repo, outcome="running", started_at=datetime.now(timezone.utc).isoformat(), error=None)
    try:
        result = process.run(
            [*_index_command(repo), *index_args],
            process.RunOptions(cwd=workspace, stdout=process.DEVNULL, stderr=process.DEVNULL),
        )
    except (process.SubprocessError, OSError) as exc:
        _ = _write_reindex_state(
            repo,
            outcome="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
        return 0
    if result.returncode != 0:
        _ = _write_reindex_state(
            repo,
            outcome="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            exit_code=result.returncode,
            error=f"pci-index exited {result.returncode}",
        )
        return 0
    completed_head = (git_utils.run_git(repo, ["rev-parse", "HEAD"]) or "").strip() or requested_head
    _ = _write_reindex_state(
        repo,
        outcome="completed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        completed_head=completed_head,
        exit_code=0,
        error=None,
    )
    return 0


def run_reindex(repo: Path) -> int:
    """Serialize reindex workers and record their outcome for ``pci hook status``."""
    target = reindex_target(repo)
    if target is None:
        _ = _write_reindex_state(repo, outcome="skipped_no_marker", error="no valid reindex marker")
        return 0
    workspace, index_args = target
    try:
        lock_file = _lock_path(workspace).open("w")
    except OSError as exc:
        _ = _write_reindex_state(repo, outcome="lock_failed", error=str(exc))
        return 0
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            _ = _write_reindex_state(repo, outcome="lock_failed", error=str(exc))
            return 0
        return _run_locked_reindex(repo, workspace, index_args)
    finally:
        lock_file.close()
