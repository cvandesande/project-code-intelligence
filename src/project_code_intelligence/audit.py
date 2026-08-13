"""Whole-tree cleanup audit (`pci-analyze audit`).

One user-invoked sweep over the latest indexed snapshot(s), reporting only
checks with a measured precision number (see HANDOFF/git history for the
measurement record):

* index staleness -- is the snapshot behind the local HEAD, was the tree dirty;
* duplicate bare names -- function/method names defined in more than one file
  (a counted fact, not a verdict: some names are conventional);
* redundancy candidates -- the ``find_redundancy`` motif groups, presented as
  an explicitly UNRANKED candidate list (measured: ~42% of groups are real;
  only near-identical body text is near-certain);
* static findings -- passthrough counts of ingested SARIF results.

Everything here is read-only and advisory. The audit prints evidence for a
human (or agent) to judge; it never asserts a refactor is correct.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text

from project_code_intelligence import console_ui
from project_code_intelligence.analyze import (
    DEFAULT_LIMIT,
    AnalysisOptions,
    SnapshotResult,
    analyze_snapshot,
    coerce_int,
    coerce_str,
    latest_snapshots,
    resolve_and_select_snapshots,
    resolve_repo_branch,
)
from project_code_intelligence.exceptions import DatabaseConnectionError
from project_code_intelligence.mcp import db as mcp_db
from project_code_intelligence.mcp.status import annotate_status_snapshots

if TYPE_CHECKING:
    from collections.abc import Sequence

    from project_code_intelligence import db
    from project_code_intelligence.analyze import MotifGroup, SnapshotRef
    from project_code_intelligence.mcp.protocol import Json

# Blind-labeled measurement (n=40 groups, 26 under blind protocol, this repo):
# ~42% of motif groups are real duplicates, and no computed score separates
# real from junk EXCEPT near-identical body text -- exact-text copies
# (pairwise text similarity ~1.0) were reliably real. The gate uses the MAX
# pairwise similarity: the group average dilutes a byte-identical pair inside
# a larger group. The blind set's highest NON-duplicate pair scored 0.9876
# (a designed records-table/snapshot-table parallel), so the gate sits above
# it. Groups at or above are near-certain; everything else is unranked.
# Replicated on a Rust repo (n=20 blind, 2026-08-11): max_text did not separate
# real from junk in 0.25-0.94 (a real group at 0.25, junk at 0.74); overall
# precision there 12/20 (60%).
NEAR_CERTAIN_TEXT = 0.99


@dataclass(frozen=True)
class DuplicateName:
    """One bare (unqualified) name defined in more than one source file."""

    name: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class StaticFindingCount:
    tool: str
    level: str
    count: int


@dataclass(frozen=True)
class AuditResult:
    """Everything the audit found for one snapshot."""

    label: str
    repo: str
    snapshot_id: int
    staleness: Json | None
    names_total: int
    duplicate_names: tuple[DuplicateName, ...]
    redundancy: SnapshotResult
    static_commit: str | None
    static_counts: tuple[StaticFindingCount, ...]


# --- pure logic -------------------------------------------------------------


def bare_name(symbol: str) -> str:
    return symbol.rsplit(".", 1)[-1]


def duplicate_names(definitions: Sequence[tuple[str, str]]) -> tuple[int, list[DuplicateName]]:
    """(total bare names, names defined in >1 file) from (symbol, source_path) pairs.

    Measured on this repo: precise and trivial -- every reported name really is
    defined in multiple files. Whether that is a problem is the reader's call
    (``main`` in every entry point is convention, a drifting helper is not).
    """
    by_name: dict[str, set[str]] = {}
    for symbol, path in definitions:
        by_name.setdefault(bare_name(symbol), set()).add(path)
    dups = [DuplicateName(name=name, paths=tuple(sorted(paths))) for name, paths in by_name.items() if len(paths) > 1]
    dups.sort(key=lambda dup: (-len(dup.paths), dup.name))
    return len(by_name), dups


def _first_member_key(group: MotifGroup) -> tuple[str, str]:
    return (group.members[0].source_path, group.members[0].symbol)


def split_groups(groups: Sequence[MotifGroup]) -> tuple[list[MotifGroup], list[MotifGroup]]:
    """(near-certain, unranked candidates) by the measured text-similarity gate.

    Both lists are ordered by first member path purely for stable output; the
    order carries no confidence information (measured: coherence does not
    separate real duplicates from junk below the exact-text gate).
    """
    near: list[MotifGroup] = []
    rest: list[MotifGroup] = []
    for group in groups:
        is_near = group.max_text is not None and group.max_text >= NEAR_CERTAIN_TEXT
        (near if is_near else rest).append(group)
    return sorted(near, key=_first_member_key), sorted(rest, key=_first_member_key)


# --- database loading ---------------------------------------------------------


def snapshot_staleness(conn: db.DbConnection, snapshot_id: int) -> Json | None:
    """The snapshot row annotated with head_status/index_age (reuses MCP status logic)."""
    row = conn.execute(
        """
        SELECT id, collection, repo, branch, commit_sha, tree_sha, dirty, metadata, created_at
        FROM project_code_intel_snapshots
        WHERE id = %s
        """,
        [snapshot_id],
    ).fetchone()
    if row is None:
        return None
    return annotate_status_snapshots([row])[0]


def load_definitions(conn: db.DbConnection, snapshot_id: int) -> list[tuple[str, str]]:
    """(symbol, source_path) for every non-test source function/method definition.

    Trait-impl methods (records carrying ``impl_trait`` metadata) are excluded:
    every type implementing a trait defines the same qualified name
    (``Default::default`` in N files), which is language mechanics, not
    duplication. Measured on a Rust repo 2026-08-11: they were ~100% of the
    duplicate-name section's noise.
    """
    rows = conn.execute(
        """
        SELECT r.symbol, r.source_path
        FROM project_code_intel_records r
        JOIN project_code_intel_files f
          ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
        WHERE r.snapshot_id = %s
          AND r.record_type = 'symbol_definition'
          AND r.symbol IS NOT NULL
          AND r.symbol_kind IN ('function', 'method', 'shell_function')
          AND r.file_role != 'test'
          AND r.metadata ->> 'impl_trait' IS NULL
          AND f.is_source = true
          AND f.is_test = false
        """,
        [snapshot_id],
    ).fetchall()
    out: list[tuple[str, str]] = []
    for row in rows:
        symbol = coerce_str(row["symbol"])
        path = coerce_str(row["source_path"])
        if symbol is not None and path is not None:
            out.append((symbol, path))
    return out


def static_finding_counts(conn: db.DbConnection, snapshot: SnapshotRef) -> tuple[str | None, list[StaticFindingCount]]:
    """(commit of the latest SARIF-bearing snapshot, counts by tool/level) for one repo.

    SARIF ingest is manual while reindex is post-commit, so findings often hang
    off an older snapshot than the one being audited; the newest snapshot that
    has findings is reported, with its commit so the reader can judge drift.
    """
    rows = conn.execute(
        """
        SELECT f.commit_sha, r.tool_name, coalesce(f.level, 'none') AS level, count(*) AS findings
        FROM project_code_intel_static_findings f
        JOIN project_code_intel_static_runs r ON r.id = f.run_id
        WHERE f.collection = %s
          AND f.repo = %s
          AND f.snapshot_id = (
              SELECT max(snapshot_id) FROM project_code_intel_static_findings
              WHERE collection = %s AND repo = %s
          )
        GROUP BY f.commit_sha, r.tool_name, level
        ORDER BY r.tool_name, level
        """,
        [snapshot.collection, snapshot.repo, snapshot.collection, snapshot.repo],
    ).fetchall()
    commit: str | None = None
    counts: list[StaticFindingCount] = []
    for row in rows:
        commit = coerce_str(row["commit_sha"]) or commit
        tool = coerce_str(row["tool_name"])
        level = coerce_str(row["level"])
        count = coerce_int(row["findings"])
        if tool is not None and level is not None and count is not None:
            counts.append(StaticFindingCount(tool=tool, level=level, count=count))
    return commit, counts


def audit_snapshot(conn: db.DbConnection, snapshot: SnapshotRef, options: AnalysisOptions) -> AuditResult:
    names_total, dups = duplicate_names(load_definitions(conn, snapshot.snapshot_id))
    static_commit, static_counts = static_finding_counts(conn, snapshot)
    return AuditResult(
        label=f"{snapshot.collection}/{snapshot.repo}",
        repo=snapshot.repo,
        snapshot_id=snapshot.snapshot_id,
        staleness=snapshot_staleness(conn, snapshot.snapshot_id),
        names_total=names_total,
        duplicate_names=tuple(dups),
        redundancy=analyze_snapshot(conn, snapshot, options),
        static_commit=static_commit,
        static_counts=tuple(static_counts),
    )


# --- rendering ------------------------------------------------------------------
#
# Both renderers strip the leading "<repo>/" from paths (the section header
# names the repo) and print only max pairwise text similarity per group: the
# measurement showed no other computed score separates real duplicates from
# junk, so printing them would only spend reader attention / agent tokens.


def _rel(path: str, repo: str) -> str:
    return path.removeprefix(repo + "/")


def _staleness_lines(staleness: Json | None) -> list[str]:
    if staleness is None:
        return ["snapshot row missing -- reindex (pci-index) before trusting anything below"]
    commit = staleness.get("commit_sha")
    commit_short = commit[:10] if isinstance(commit, str) else "?"
    status = staleness.get("head_status")
    age = staleness.get("index_age_seconds")
    age_text = f", indexed {int(age) // 3600}h{(int(age) % 3600) // 60:02d}m ago" if isinstance(age, int) else ""
    line = f"commit {commit_short} -- {status}{age_text}"
    out = [line]
    if staleness.get("dirty") is True:
        out.append("working tree was dirty at index time; line numbers are approximate")
    if status == "stale":
        head = staleness.get("head_commit")
        head_short = head[:10] if isinstance(head, str) else "?"
        out.append(f"local HEAD is {head_short}; commit or reindex before acting on line numbers")
    lag = staleness.get("upstream_commits_behind")
    # A checkout can match its own index (head_status "current") while sitting on a stale
    # branch far behind its upstream -- "current" answers "is the index stale", not "is the
    # branch stale". Both matter: a redundancy hit against 134-commits-old code is not evidence.
    if isinstance(lag, int) and lag > 0:
        out.append(f"local branch is {lag} commit(s) behind its upstream -- checkout may be stale, not just the index")
    return out


def _group_lines(group: MotifGroup, repo: str) -> list[str]:
    max_text = "n/a" if group.max_text is None else f"{group.max_text:.2f}"
    return [
        f"  {len(group.members)} members (max pairwise text {max_text}):",
        *(f"    {m.symbol}  {_rel(m.source_path, repo)}:{m.line_start}" for m in group.members),
    ]


def _group_section(groups: list[MotifGroup], repo: str) -> list[str]:
    if not groups:
        return ["  none"]
    return [line for group in groups for line in _group_lines(group, repo)]


def _static_lines(result: AuditResult) -> list[str]:
    if not result.static_counts:
        return ["no SARIF ingested for this repo (make scan, or pci-index --sarif)"]
    commit_short = result.static_commit[:10] if result.static_commit else "?"
    return [
        f"latest ingested scan is for commit {commit_short}:",
        *(f"  {c.tool}  {c.level}: {c.count}" for c in result.static_counts),
        "detail: search_static_findings / get_static_finding MCP tools",
    ]


# Path lists longer than this wrap onto indented lines in the text report.
_PATHS_PER_LINE = 3


def _duplicate_name_lines(result: AuditResult) -> list[str]:
    out: list[str] = []
    for dup in result.duplicate_names:
        paths = [_rel(path, result.repo) for path in dup.paths]
        if len(paths) > _PATHS_PER_LINE:
            out.append(f"  {dup.name}  x{len(paths)}:")
            out.extend(f"    {path}" for path in paths)
        else:
            out.append(f"  {dup.name}  x{len(paths)}: {', '.join(paths)}")
    return out


def render_result(result: AuditResult) -> list[str]:
    near, rest = split_groups(result.redundancy.groups)
    return [
        f"## {result.label}",
        "",
        "### Index staleness",
        *_staleness_lines(result.staleness),
        "",
        "### Duplicate names (defined in more than one file)",
        f"{len(result.duplicate_names)} of {result.names_total} function/method names are defined in >1 file.",
        "Counted fact, not a verdict: entry-point names such as `main` are conventional.",
        *_duplicate_name_lines(result),
        "",
        "### Redundancy candidates",
        f"_{result.redundancy.functions_analyzed} functions analyzed, "
        f"{result.redundancy.clones_folded} exact clones folded_",
        "Measured precision: ~42% of groups are real duplicates on a Python repo "
        "(n=40 labeled, 26 blind); 60% on a Rust repo (n=20, blind). "
        "Only near-identical body text is near-certain; every other group needs a source read. "
        "The candidate list is UNRANKED -- order carries no confidence.",
        "",
        f"Near-certain (contains an exact-text pair, max pairwise text >= {NEAR_CERTAIN_TEXT}):",
        *_group_section(near, result.repo),
        "",
        "Candidates (unranked; verify in source):",
        *_group_section(rest, result.repo),
        "",
        "### Static findings (SARIF passthrough)",
        *_static_lines(result),
        "",
    ]


def render_lines(results: Sequence[AuditResult]) -> list[str]:
    lines = [
        "# PCI audit -- whole-tree cleanup candidates",
        "",
        "Advisory only; every check below carries its measured precision. Verify in",
        "source before acting.",
        "",
    ]
    for result in results:
        lines.extend(render_result(result))
    return lines


def render_text(results: Sequence[AuditResult]) -> str:
    return "\n".join(render_lines(results)) + "\n"


# First match wins; the palette follows the console_ui status pills
# (green ok / yellow warn / red fail). Purely cosmetic: a rule that stops
# matching just loses its color.
_PREFIX_STYLES: tuple[tuple[str, str], ...] = (
    ("### ", "bold cyan"),
    ("#", "bold"),
    ("  none", "dim"),
    ("Near-certain", "bold"),
    ("Candidates (unranked", "bold"),
    ("Advisory only", "dim"),
    ("source before acting", "dim"),
    ("Counted fact", "dim"),
    ("Measured on this repo", "dim"),
    ("working tree was dirty", "yellow"),
    ("local HEAD is", "yellow"),
    ("no SARIF ingested", "yellow"),
    ("snapshot row missing", "red"),
)
_CONTAINS_STYLES: tuple[tuple[str, str], ...] = (
    ("-- stale", "red"),
    ("-- current", "green"),
)


def line_style(line: str) -> str | None:
    for prefix, style in _PREFIX_STYLES:
        if line.startswith(prefix):
            return style
    for marker, style in _CONTAINS_STYLES:
        if marker in line:
            return style
    return None


# Keys the machine output keeps from the annotated snapshot row.
_STALENESS_KEYS = ("commit_sha", "head_status", "head_commit", "dirty", "index_age_seconds", "upstream_commits_behind")


def _group_to_compact(group: MotifGroup, repo: str) -> dict[str, object]:
    """Token-minimal group: max pairwise text plus "symbol path:start-end" members.

    Everything else group_to_json carries (coherence, evidence, recommendation,
    averages, record ids) measured as non-separating for this decision, so the
    machine output omits it; the full record stays available via the
    find_redundancy MCP tool and ``pci-analyze compression``.
    """
    return {
        "max_text": group.max_text,
        "members": [f"{m.symbol} {_rel(m.source_path, repo)}:{m.line_start}-{m.line_end}" for m in group.members],
    }


def _result_to_json(result: AuditResult) -> dict[str, object]:
    near, rest = split_groups(result.redundancy.groups)
    staleness: object = None
    if result.staleness is not None:
        staleness = {key: result.staleness.get(key) for key in _STALENESS_KEYS if result.staleness.get(key) is not None}
    static_counts: dict[str, dict[str, int]] = {}
    for count in result.static_counts:
        static_counts.setdefault(count.tool, {})[count.level] = count.count
    return {
        "snapshot_id": result.snapshot_id,
        "staleness": staleness,
        "duplicate_names": {
            "names_total": result.names_total,
            "items": {dup.name: [_rel(path, result.repo) for path in dup.paths] for dup in result.duplicate_names},
        },
        "redundancy": {
            "functions_analyzed": result.redundancy.functions_analyzed,
            "clones_folded": result.redundancy.clones_folded,
            "near_certain_text_threshold": NEAR_CERTAIN_TEXT,
            "near_certain": [_group_to_compact(group, result.repo) for group in near],
            "candidates_unranked": [_group_to_compact(group, result.repo) for group in rest],
        },
        "static_findings": {
            "commit_sha": result.static_commit,
            "counts": static_counts,
        },
    }


def render_json(results: Sequence[AuditResult]) -> str:
    payload = {result.label: _result_to_json(result) for result in results}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str) + "\n"


# --- CLI ------------------------------------------------------------------------


@dataclass
class AuditNamespace(argparse.Namespace):
    collection: str | None = None
    repo: str | None = None
    limit: int = DEFAULT_LIMIT
    json: bool = False
    extra: list[str] = field(default_factory=list)


def _audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pci-analyze audit",
        description="Whole-tree cleanup audit: staleness, duplicate names, redundancy candidates, static findings.",
    )
    _ = parser.add_argument("--collection", help="Restrict to one collection/workspace.")
    _ = parser.add_argument("--repo", help="Restrict to one repo within the collection(s).")
    _ = parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum redundancy groups to report per snapshot (default {DEFAULT_LIMIT}).",
    )
    _ = parser.add_argument("--json", action="store_true", help="Emit JSON instead of the text report.")
    return parser


def audit_main(argv: list[str] | None = None) -> int:
    parsed = _audit_parser().parse_args(argv, namespace=AuditNamespace())
    options = AnalysisOptions(limit=parsed.limit)
    try:
        with mcp_db.connect() as conn:
            if not mcp_db.code_intel_tables_exist(conn):
                _ = sys.stderr.write("pci-analyze: no code-intelligence tables; run pci-index first\n")
                return 1
            all_snapshots = latest_snapshots(conn)
            branch = resolve_repo_branch(Path.cwd())
            snapshots, branch_miss = resolve_and_select_snapshots(
                all_snapshots, collection=parsed.collection, repo=parsed.repo, branch=branch
            )
            if branch_miss:
                _ = sys.stderr.write(f"pci-analyze: no snapshot on branch {branch!r}; using newest per repo\n")
            if not snapshots:
                _ = sys.stderr.write("pci-analyze: no matching snapshots found\n")
                return 1
            results = [audit_snapshot(conn, snapshot, options) for snapshot in snapshots]
    except DatabaseConnectionError as exc:
        _ = sys.stderr.write(f"pci-analyze: {exc}\n")
        return 1
    if parsed.json:
        _ = sys.stdout.write(render_json(results))
    elif console_ui.should_emit_pretty(sys.stdout):
        console = console_ui.build_console()
        for line in render_lines(results):
            console.print(Text(line, style=line_style(line) or ""))
    else:
        _ = sys.stdout.write(render_text(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(audit_main())
