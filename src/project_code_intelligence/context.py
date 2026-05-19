"""Generate a markdown map of the indexed project.

Reads the latest snapshot per (collection, repo) from the project
code-intelligence index and emits a markdown sketch listing each source module
and its top-level symbols, preceded by a self-describing provenance header so
stale or under-covering maps fail loudly rather than silently.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from project_code_intelligence.common import repo_relative_path
from project_code_intelligence.exceptions import DatabaseConnectionError
from project_code_intelligence.mcp import db as mcp_db
from project_code_intelligence.mcp.status import annotate_status_snapshots

if TYPE_CHECKING:
    from collections.abc import Iterable

    from project_code_intelligence import db
    from project_code_intelligence.mcp.protocol import Json


OMITTED_LINES: tuple[str, ...] = (
    "- Private names (`_*`).",
    "- Methods and nested functions (recorded as `Class.method` / `outer.inner` — query directly to retrieve).",
    "- Test files (`is_test=true`).",
    "- `scripts/` (tooling, not project logic).",
    "- Non-source files (config, docs, generated, vendor) — run `list_code_intel_files` to enumerate.",
)

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 60 * 60
_SECONDS_PER_DAY = 24 * 60 * 60

DRILL_IN_HINT = (
    "Drill into anything below via the `pci-mcp` tools: `search_code_intel_text`, "
    "`search_code_intel_semantic`, `get_code_intel_record`, `related_code_intel`, `list_code_intel_files`."
)


@dataclass(frozen=True)
class _Snapshot:
    snapshot_id: int
    collection: str
    repo: str
    branch: str | None
    commit_sha: str | None
    dirty: bool
    head_status: str
    head_commit: str | None
    index_age_seconds: int | None
    dirty_paths_count: int | None


@dataclass(frozen=True)
class _Symbol:
    source_path: str
    symbol: str
    symbol_kind: str


def _short_sha(value: str | None) -> str:
    if value is None:
        return "?"
    return value[:12]


def _format_age(seconds: int | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds}s ago"
    if seconds < _SECONDS_PER_HOUR:
        return f"{seconds // _SECONDS_PER_MINUTE}m ago"
    if seconds < _SECONDS_PER_DAY:
        return f"{seconds // _SECONDS_PER_HOUR}h ago"
    return f"{seconds // _SECONDS_PER_DAY}d ago"


def _dirty_paths_count(metadata: object) -> int | None:
    if not isinstance(metadata, dict):
        return None
    dirty_paths = cast("dict[object, object]", metadata).get("dirty_paths")
    if not isinstance(dirty_paths, list):
        return None
    items = cast("list[object]", dirty_paths)
    return sum(1 for item in items if isinstance(item, str))


def _coerce_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _build_snapshot(annotated: Json) -> _Snapshot | None:
    snapshot_id_value = annotated.get("id")
    if not isinstance(snapshot_id_value, int) or isinstance(snapshot_id_value, bool):
        return None
    collection = _coerce_str(annotated.get("collection")) or ""
    repo = _coerce_str(annotated.get("repo")) or ""
    head_status_value = annotated.get("head_status")
    head_status = head_status_value if isinstance(head_status_value, str) else "unknown"
    return _Snapshot(
        snapshot_id=snapshot_id_value,
        collection=collection,
        repo=repo,
        branch=_coerce_str(annotated.get("branch")),
        commit_sha=_coerce_str(annotated.get("commit_sha")),
        dirty=bool(annotated.get("dirty")),
        head_status=head_status,
        head_commit=_coerce_str(annotated.get("head_commit")),
        index_age_seconds=_coerce_int(annotated.get("index_age_seconds")),
        dirty_paths_count=_dirty_paths_count(annotated.get("metadata")),
    )


def _latest_snapshots(conn: db.DbConnection) -> list[_Snapshot]:
    rows = conn.execute(
        """
        SELECT DISTINCT ON (collection, repo)
               id, collection, repo, repo_role, branch, commit_sha, tree_sha,
               dirty, metadata, created_at
        FROM project_code_intel_snapshots
        ORDER BY collection, repo, created_at DESC, id DESC
        """
    ).fetchall()
    annotated = annotate_status_snapshots(rows)
    out: list[_Snapshot] = []
    for snapshot_json in annotated:
        item = _build_snapshot(snapshot_json)
        if item is not None:
            out.append(item)
    return out


def _public_source_symbols(conn: db.DbConnection, snapshot_id: int) -> list[_Symbol]:
    rows = conn.execute(
        """
        SELECT r.source_path, r.symbol, r.symbol_kind
        FROM project_code_intel_records r
        JOIN project_code_intel_files f
          ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
        WHERE r.snapshot_id = %s
          AND r.record_type = 'symbol_definition'
          AND r.symbol IS NOT NULL
          AND left(r.symbol, 1) <> '_'
          AND position('.' in r.symbol) = 0
          AND f.is_source = true
          AND f.is_test = false
        ORDER BY r.source_path, r.line_start, r.symbol
        """,
        [snapshot_id],
    ).fetchall()
    out: list[_Symbol] = []
    for row in rows:
        symbol_value = row["symbol"]
        if not isinstance(symbol_value, str) or not symbol_value:
            continue
        kind_value = row["symbol_kind"]
        out.append(
            _Symbol(
                source_path=str(row["source_path"]),
                symbol=symbol_value,
                symbol_kind=str(kind_value) if isinstance(kind_value, str) else "",
            )
        )
    return out


def _is_orientation_path(path: str) -> bool:
    return not path.startswith("scripts/")


def _group_by_path(symbols: Iterable[_Symbol], repo: str) -> dict[str, dict[str, list[str]]]:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"classes": [], "functions": []})
    for item in symbols:
        rel = repo_relative_path(item.source_path, repo) or item.source_path
        if not _is_orientation_path(rel):
            continue
        bucket = "classes" if item.symbol_kind == "class" else "functions"
        grouped[rel][bucket].append(item.symbol)
    return grouped


def _format_snapshot_line(snapshot: _Snapshot) -> str:
    repo_label = f"{snapshot.collection}/{snapshot.repo}"
    branch = snapshot.branch or "?"
    commit = _short_sha(snapshot.commit_sha)
    age = _format_age(snapshot.index_age_seconds)
    parts = [
        f"**{repo_label}** — snapshot {snapshot.snapshot_id}",
        f"branch `{branch}`",
        f"commit `{commit}`",
        f"indexed {age}",
        f"HEAD status: **{snapshot.head_status}**",
    ]
    if snapshot.head_status == "stale" and snapshot.head_commit:
        parts.append(f"local HEAD `{_short_sha(snapshot.head_commit)}`")
    if snapshot.dirty:
        if snapshot.dirty_paths_count is not None:
            parts.append(f"working tree was **dirty** ({snapshot.dirty_paths_count} paths)")
        else:
            parts.append("working tree was **dirty**")
    return "- " + ", ".join(parts) + "."


def _render_header(snapshots: list[_Snapshot]) -> list[str]:
    lines: list[str] = [
        "# Project code-intelligence map",
        "",
        "Source: `project-code-intelligence` MCP index (Postgres/pgvector). Generated by `pci-context`.",
        DRILL_IN_HINT,
        "",
        "## Snapshots",
        "",
    ]
    lines.extend(_format_snapshot_line(snapshot) for snapshot in snapshots)
    lines.extend(("", "## Omitted from this map", ""))
    lines.extend(OMITTED_LINES)
    return lines


def _render_repo(snapshot: _Snapshot, symbols: list[_Symbol]) -> list[str]:
    grouped = _group_by_path(symbols, snapshot.repo)
    lines: list[str] = [f"# {snapshot.repo}"]
    if not grouped:
        lines.extend(("", "_no source-file symbols indexed_"))
        return lines
    for path in sorted(grouped):
        groups = grouped[path]
        lines.extend(("", f"## {path}"))
        classes = groups["classes"]
        functions = groups["functions"]
        if classes:
            lines.append(f"- classes: {', '.join(classes)}")
        if functions:
            lines.append(f"- functions: {', '.join(functions)}")
    return lines


def _build_map(conn: db.DbConnection) -> str:
    snapshots = _latest_snapshots(conn)
    if not snapshots:
        return ""
    sections: list[str] = ["\n".join(_render_header(snapshots))]
    for snapshot in snapshots:
        symbols = _public_source_symbols(conn, snapshot.snapshot_id)
        sections.append("\n".join(_render_repo(snapshot, symbols)))
    return "\n\n---\n\n".join(sections) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a markdown map of the indexed project.")
    _ = parser.parse_args(argv)
    try:
        with mcp_db.connect() as conn:
            if not mcp_db.code_intel_tables_exist(conn):
                _ = sys.stderr.write("pci-context: no code-intelligence tables; run pci-index first\n")
                return 1
            output = _build_map(conn)
    except DatabaseConnectionError as exc:
        _ = sys.stderr.write(f"pci-context: {exc}\n")
        return 1
    if not output:
        _ = sys.stderr.write("pci-context: no snapshots found; run pci-index first\n")
        return 1
    _ = sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
