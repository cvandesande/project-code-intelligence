"""Blast-radius evidence bundles for a symbol (phase-1 core).

Assembles the facts a judge (an LLM agent or a human) needs to decide "is this
slop / is it safe to cut", drawn only from the existing index: inbound callers,
outbound callees, test coverage, entry-point and module-level wiring, the top
semantic neighbours, and an index-staleness guard. It emits *evidence, never a
verdict* -- the model judges. Only facts that are expensive to derive per-turn
(cross-file reachability, semantic neighbours, staleness) are included; a local
quality read is left to the model that already has the code in front of it.

The same bundle backs three surfaces: a CLI (`pci-evidence`), a future edit-path
hook, and a future MCP tool. Reachability rests on heuristic ``call_candidate``
edges, so caller lists for dynamically dispatched methods can under-count; the
``name_reference_count`` field is the coarse text-level backstop for that case.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from project_code_intelligence.analyze import SnapshotRef, latest_snapshots
from project_code_intelligence.exceptions import DatabaseConnectionError
from project_code_intelligence.mcp import db as mcp_db
from project_code_intelligence.mcp.status import annotate_status_snapshots

if TYPE_CHECKING:
    from collections.abc import Sequence

    from project_code_intelligence import db
    from project_code_intelligence.mcp.protocol import Json

DEFAULT_NEIGHBORS = 3
DEFAULT_NEIGHBOR_THRESHOLD = 0.80
# Over-fetch factor so parent/child and prefix filtering still leaves N results.
_NEIGHBOR_OVERFETCH = 5
_ENTRYPOINT_NAMES = frozenset({"main"})


def _coerce_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _last_component(symbol: str) -> str:
    return symbol.rsplit(".", 1)[-1]


# --- data model -----------------------------------------------------------------


@dataclass(frozen=True)
class Caller:
    """One inbound reference resolved to a specific definition."""

    symbol: str | None
    source_path: str
    line: int | None
    is_test: bool
    at_module_level: bool


@dataclass(frozen=True)
class Callee:
    target_symbol: str | None
    target_path: str | None
    resolved: bool


@dataclass(frozen=True)
class Neighbor:
    """A semantically close definition (raw evidence; not a duplication verdict)."""

    symbol: str
    source_path: str
    line: int | None
    similarity: float


@dataclass(frozen=True)
class Staleness:
    head_status: str
    dirty: bool
    index_age_seconds: int | None
    target_file_dirty: bool

    @property
    def is_stale(self) -> bool:
        return self.head_status == "stale" or self.dirty or self.target_file_dirty


@dataclass(frozen=True)
class Evidence:
    label: str
    record_id: str
    symbol: str
    symbol_kind: str
    source_path: str
    line_start: int | None
    line_end: int | None
    callers: tuple[Caller, ...]
    callees: tuple[Callee, ...]
    name_reference_count: int
    is_service_entrypoint: bool
    neighbors: tuple[Neighbor, ...]
    staleness: Staleness

    @property
    def inbound_count(self) -> int:
        return len(self.callers)

    @property
    def covered_by_tests(self) -> tuple[str, ...]:
        return tuple(sorted({caller.source_path for caller in self.callers if caller.is_test}))

    @property
    def wired_at_module_level(self) -> bool:
        return any(caller.at_module_level for caller in self.callers)

    @property
    def is_conventional_entrypoint(self) -> bool:
        return _last_component(self.symbol) in _ENTRYPOINT_NAMES

    @property
    def looks_orphaned(self) -> bool:
        """No resolved caller, no name mention, not an entry point -- dead on arrival.

        Advisory: heuristic edges can miss dynamic dispatch, so this is a prompt
        to verify, not a proof of deadness.
        """
        if self.inbound_count > 0 or self.name_reference_count > 0:
            return False
        return not (self.is_conventional_entrypoint or self.is_service_entrypoint)


# --- pure logic -----------------------------------------------------------------


Span = tuple[int | None, int | None]
# (symbol, source_path, line span) -- the minimum needed to test structural nesting.
Located = tuple[str, str, Span]


def is_parent_child(target: Located, neighbor: Located) -> bool:
    """True when two definitions are structurally nested, not independent twins.

    The semantic-neighbour probe showed the dominant false positive is a class
    matched against its own method (``A`` vs ``A.method``) or an outer function
    against a nested closure -- their bodies overlap, so high similarity is an
    artefact. Excludes symbol prefix relations and, within one file, overlapping
    line spans.
    """
    (target_symbol, target_path, target_span) = target
    (neighbor_symbol, neighbor_path, neighbor_span) = neighbor
    if target_symbol == neighbor_symbol:
        return True
    if neighbor_symbol.startswith(target_symbol + ".") or target_symbol.startswith(neighbor_symbol + "."):
        return True
    if target_path != neighbor_path:
        return False
    t_start, t_end = target_span
    n_start, n_end = neighbor_span
    if t_start is None or t_end is None or n_start is None or n_end is None:
        return False
    return t_start <= n_end and n_start <= t_end


# --- database loading -----------------------------------------------------------


@dataclass(frozen=True)
class TargetDef:
    snapshot: SnapshotRef
    record_id: str
    symbol: str
    symbol_kind: str
    source_path: str
    line_start: int | None
    line_end: int | None


def resolve_targets(
    conn: db.DbConnection,
    snapshot: SnapshotRef,
    *,
    symbol: str | None,
    source_path: str | None,
    line: int | None,
) -> list[TargetDef]:
    """Find symbol_definition records matching a symbol or a path+line location.

    Every optional filter is guarded by a ``%s IS NULL OR ...`` clause so the SQL
    text is static (no string building); an unset argument passes NULL and the
    guard drops that predicate.
    """
    symbol_like = f"%.{symbol}" if symbol is not None else None
    path_like = f"%{source_path}" if source_path is not None else None
    rows = conn.execute(
        """
        SELECT record_id, symbol, symbol_kind, source_path, line_start, line_end
        FROM project_code_intel_records
        WHERE snapshot_id = %s
          AND record_type = 'symbol_definition'
          AND symbol IS NOT NULL
          AND (%s::text IS NULL OR symbol = %s OR symbol LIKE %s)
          AND (%s::text IS NULL OR source_path LIKE %s)
          AND (%s::int IS NULL OR (line_start <= %s AND line_end >= %s))
        ORDER BY source_path, line_start
        LIMIT 25
        """,
        [snapshot.snapshot_id, symbol, symbol, symbol_like, source_path, path_like, line, line, line],
    ).fetchall()
    out: list[TargetDef] = []
    for row in rows:
        record_id = _coerce_str(row["record_id"])
        sym = _coerce_str(row["symbol"])
        path = _coerce_str(row["source_path"])
        if record_id is None or sym is None or path is None:
            continue
        out.append(
            TargetDef(
                snapshot=snapshot,
                record_id=record_id,
                symbol=sym,
                symbol_kind=_coerce_str(row["symbol_kind"]) or "",
                source_path=path,
                line_start=_coerce_int(row["line_start"]),
                line_end=_coerce_int(row["line_end"]),
            )
        )
    return out


def load_callers(conn: db.DbConnection, target: TargetDef) -> list[Caller]:
    rows = conn.execute(
        """
        SELECT e.source_symbol, e.source_path, r.line_start, r.record_type AS source_record_type,
               coalesce(f.is_test, false) AS is_test
        FROM project_code_intel_edges e
        LEFT JOIN project_code_intel_records r
          ON r.snapshot_id = e.snapshot_id AND r.record_id = e.source_record_id
        LEFT JOIN project_code_intel_files f
          ON f.snapshot_id = e.snapshot_id AND f.source_path = e.source_path
        WHERE e.snapshot_id = %s
          AND e.edge_type = 'call_candidate'
          AND e.target_record_id = %s
          AND e.source_record_id <> %s
        ORDER BY e.source_path, r.line_start
        """,
        [target.snapshot.snapshot_id, target.record_id, target.record_id],
    ).fetchall()
    out: list[Caller] = []
    for row in rows:
        path = _coerce_str(row["source_path"])
        if path is None:
            continue
        out.append(
            Caller(
                symbol=_coerce_str(row["source_symbol"]),
                source_path=path,
                line=_coerce_int(row["line_start"]),
                is_test=bool(row["is_test"]),
                at_module_level=_coerce_str(row["source_record_type"]) == "module_chunk",
            )
        )
    return out


def load_callees(conn: db.DbConnection, target: TargetDef) -> list[Callee]:
    rows = conn.execute(
        """
        SELECT DISTINCT e.target_symbol, e.target_path, (e.target_record_id IS NOT NULL) AS resolved
        FROM project_code_intel_edges e
        WHERE e.snapshot_id = %s
          AND e.edge_type = 'call_candidate'
          AND e.source_record_id = %s
          AND e.target_record_id IS DISTINCT FROM e.source_record_id
        ORDER BY e.target_symbol
        """,
        [target.snapshot.snapshot_id, target.record_id],
    ).fetchall()
    out: list[Callee] = []
    for row in rows:
        target_symbol = _coerce_str(row["target_symbol"])
        if target_symbol is None:
            continue
        out.append(
            Callee(
                target_symbol=target_symbol,
                target_path=_coerce_str(row["target_path"]),
                resolved=bool(row["resolved"]),
            )
        )
    return out


def name_reference_count(conn: db.DbConnection, target: TargetDef) -> int:
    """Edges whose callee name matches this symbol's bare name, from elsewhere.

    A text-level backstop for the heuristic call graph: when precise resolution
    misses (dynamically dispatched methods resolve by bare name), a positive
    count still shows the name is referenced somewhere.
    """
    row = conn.execute(
        """
        SELECT count(*) AS n
        FROM project_code_intel_edges e
        WHERE e.snapshot_id = %s
          AND e.edge_type = 'call_candidate'
          AND e.target_symbol = %s
          AND e.source_record_id <> %s
        """,
        [target.snapshot.snapshot_id, _last_component(target.symbol), target.record_id],
    ).fetchone()
    return _coerce_int(row["n"]) or 0 if row is not None else 0


def is_service_entrypoint(conn: db.DbConnection, target: TargetDef) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM project_code_intel_records
        WHERE snapshot_id = %s AND record_type = 'service_entrypoint' AND symbol = %s
        LIMIT 1
        """,
        [target.snapshot.snapshot_id, _last_component(target.symbol)],
    ).fetchone()
    return row is not None


def load_neighbors(conn: db.DbConnection, target: TargetDef, *, limit: int, threshold: float) -> list[Neighbor]:
    rows = conn.execute(
        """
        WITH target AS (
            SELECT embedding
            FROM project_code_intel_records
            WHERE snapshot_id = %s AND record_type = 'code_chunk'
              AND parent_record_id = %s AND embedding IS NOT NULL
            LIMIT 1
        )
        SELECT c.symbol, c.source_path, c.line_start, 1 - (c.embedding <=> t.embedding) AS similarity
        FROM project_code_intel_records c
        CROSS JOIN target t
        JOIN project_code_intel_files f
          ON f.snapshot_id = c.snapshot_id AND f.source_path = c.source_path
        WHERE c.snapshot_id = %s AND c.record_type = 'code_chunk' AND c.embedding IS NOT NULL
          AND c.symbol IS NOT NULL AND c.parent_record_id <> %s
          AND f.is_source AND NOT f.is_test
        ORDER BY c.embedding <=> t.embedding
        LIMIT %s
        """,
        [
            target.snapshot.snapshot_id,
            target.record_id,
            target.snapshot.snapshot_id,
            target.record_id,
            limit * _NEIGHBOR_OVERFETCH,
        ],
    ).fetchall()
    out: list[Neighbor] = []
    target_span = (target.line_start, target.line_end)
    for row in rows:
        symbol = _coerce_str(row["symbol"])
        path = _coerce_str(row["source_path"])
        similarity = row["similarity"]
        if symbol is None or path is None or not isinstance(similarity, (int, float)) or isinstance(similarity, bool):
            continue
        if float(similarity) < threshold:
            break
        line = _coerce_int(row["line_start"])
        if is_parent_child((target.symbol, target.source_path, target_span), (symbol, path, (line, line))):
            continue
        out.append(Neighbor(symbol=symbol, source_path=path, line=line, similarity=float(similarity)))
        if len(out) >= limit:
            break
    return out


def load_staleness(conn: db.DbConnection, target: TargetDef) -> Staleness:
    row = conn.execute(
        """
        SELECT id, collection, repo, repo_role, branch, commit_sha, tree_sha, dirty, metadata, created_at
        FROM project_code_intel_snapshots WHERE id = %s
        """,
        [target.snapshot.snapshot_id],
    ).fetchone()
    if row is None:
        return Staleness(head_status="unknown", dirty=False, index_age_seconds=None, target_file_dirty=False)
    annotated: Json = annotate_status_snapshots([row])[0]
    head_status = annotated.get("head_status")
    metadata = annotated.get("metadata")
    dirty_paths = metadata.get("dirty_paths") if isinstance(metadata, dict) else None
    target_file_dirty = isinstance(dirty_paths, list) and any(
        isinstance(item, str) and target.source_path.endswith(item) for item in dirty_paths
    )
    return Staleness(
        head_status=head_status if isinstance(head_status, str) else "unknown",
        dirty=bool(annotated.get("dirty")),
        index_age_seconds=_coerce_int(annotated.get("index_age_seconds")),
        target_file_dirty=target_file_dirty,
    )


def build_evidence(conn: db.DbConnection, target: TargetDef, *, neighbors: int, threshold: float) -> Evidence:
    return Evidence(
        label=f"{target.snapshot.collection}/{target.snapshot.repo}",
        record_id=target.record_id,
        symbol=target.symbol,
        symbol_kind=target.symbol_kind,
        source_path=target.source_path,
        line_start=target.line_start,
        line_end=target.line_end,
        callers=tuple(load_callers(conn, target)),
        callees=tuple(load_callees(conn, target)),
        name_reference_count=name_reference_count(conn, target),
        is_service_entrypoint=is_service_entrypoint(conn, target),
        neighbors=tuple(load_neighbors(conn, target, limit=neighbors, threshold=threshold)),
        staleness=load_staleness(conn, target),
    )


# --- rendering ------------------------------------------------------------------


def _strip_repo(path: str, repo: str) -> str:
    """Drop the leading ``{repo}/`` so text-report paths read repo-relative."""
    prefix = f"{repo}/"
    return path[len(prefix) :] if repo and path.startswith(prefix) else path


def _location(path: str, line: int | None) -> str:
    return f"{path}:{line}" if line is not None else path


def _staleness_banner(staleness: Staleness) -> str | None:
    if not staleness.is_stale:
        return None
    reasons: list[str] = []
    if staleness.head_status == "stale":
        reasons.append("HEAD moved since index")
    if staleness.dirty:
        reasons.append("working tree was dirty at index time")
    if staleness.target_file_dirty:
        reasons.append("this file had unstaged edits")
    return "index stale: " + "; ".join(reasons) + " -- treat as approximate"


def render_text(evidence: Evidence) -> str:
    # Paths are stored repo-prefixed; strip it so the compact report stays readable.
    repo = evidence.label.rsplit("/", 1)[-1]
    lines: list[str] = []
    banner = _staleness_banner(evidence.staleness)
    if banner is not None:
        lines.append(f"! {banner}")
    span = f"{evidence.line_start}-{evidence.line_end}" if evidence.line_start is not None else "?"
    lines.append(f"{evidence.symbol}  {evidence.symbol_kind}  {_strip_repo(evidence.source_path, repo)}:{span}")
    if evidence.callers:
        shown = evidence.callers[:5]
        rendered = "  ".join(
            f"{_location(_strip_repo(caller.source_path, repo), caller.line)}"
            + ("[test]" if caller.is_test else "")
            + ("[module]" if caller.at_module_level else "")
            for caller in shown
        )
        extra = f" (+{len(evidence.callers) - len(shown)} more)" if len(evidence.callers) > len(shown) else ""
        lines.append(f"callers ({evidence.inbound_count}): {rendered}{extra}")
    else:
        hint = f"; name referenced {evidence.name_reference_count}x elsewhere" if evidence.name_reference_count else ""
        lines.append(f"callers (0){hint}")
    tests = (
        "yes (" + ", ".join(_strip_repo(path, repo) for path in evidence.covered_by_tests) + ")"
        if evidence.covered_by_tests
        else "no"
    )
    flags: list[str] = [f"tests: {tests}"]
    if evidence.is_conventional_entrypoint or evidence.is_service_entrypoint:
        flags.append("entrypoint")
    if evidence.wired_at_module_level:
        flags.append("wired at module level")
    if evidence.looks_orphaned:
        flags.append("looks orphaned (verify: heuristic edges miss dynamic dispatch)")
    lines.append("; ".join(flags))
    # Callees omitted here (noisy, leak builtins); callers answer delete-safety, --json keeps them.
    if evidence.neighbors:
        lines.append("semantic neighbours (same meaning candidates -- verify in source):")
        for neighbor in evidence.neighbors:
            loc = _location(_strip_repo(neighbor.source_path, repo), neighbor.line)
            lines.append(f"  {neighbor.similarity:.2f}  {neighbor.symbol}  {loc}")
    return "\n".join(lines) + "\n"


def _evidence_to_json(evidence: Evidence) -> dict[str, object]:
    return {
        "symbol": evidence.symbol,
        "symbol_kind": evidence.symbol_kind,
        "source_path": evidence.source_path,
        "line_start": evidence.line_start,
        "line_end": evidence.line_end,
        "record_id": evidence.record_id,
        "callers": [
            {
                "symbol": caller.symbol,
                "source_path": caller.source_path,
                "line": caller.line,
                "is_test": caller.is_test,
                "at_module_level": caller.at_module_level,
            }
            for caller in evidence.callers
        ],
        "inbound_count": evidence.inbound_count,
        "name_reference_count": evidence.name_reference_count,
        "callees": [
            {"target_symbol": callee.target_symbol, "target_path": callee.target_path, "resolved": callee.resolved}
            for callee in evidence.callees
        ],
        "covered_by_tests": list(evidence.covered_by_tests),
        "is_entrypoint": evidence.is_conventional_entrypoint or evidence.is_service_entrypoint,
        "wired_at_module_level": evidence.wired_at_module_level,
        "looks_orphaned": evidence.looks_orphaned,
        "neighbors": [
            {
                "symbol": neighbor.symbol,
                "source_path": neighbor.source_path,
                "line": neighbor.line,
                "similarity": round(neighbor.similarity, 4),
            }
            for neighbor in evidence.neighbors
        ],
        "staleness": {
            "head_status": evidence.staleness.head_status,
            "dirty": evidence.staleness.dirty,
            "index_age_seconds": evidence.staleness.index_age_seconds,
            "target_file_dirty": evidence.staleness.target_file_dirty,
            "is_stale": evidence.staleness.is_stale,
        },
    }


def render_json(bundles: Sequence[Evidence]) -> str:
    return json.dumps([_evidence_to_json(bundle) for bundle in bundles], indent=2, sort_keys=True) + "\n"


# --- CLI ------------------------------------------------------------------------


@dataclass
class EvidenceNamespace(argparse.Namespace):
    symbol: str | None = None
    path: str | None = None
    line: int | None = None
    collection: str | None = None
    repo: str | None = None
    neighbors: int = DEFAULT_NEIGHBORS
    threshold: float = DEFAULT_NEIGHBOR_THRESHOLD
    json: bool = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pci-evidence",
        description="Assemble blast-radius evidence for a symbol (callers, tests, neighbours, staleness).",
    )
    _ = parser.add_argument("--symbol", help="Symbol name (bare or qualified) to look up.")
    _ = parser.add_argument("--path", help="Restrict to a source path (suffix match); with --line, locate by position.")
    _ = parser.add_argument("--line", type=int, help="Line number inside the target definition (use with --path).")
    _ = parser.add_argument("--collection", help="Restrict to one collection/workspace.")
    _ = parser.add_argument("--repo", help="Restrict to one repo.")
    _ = parser.add_argument(
        "--neighbors",
        type=int,
        default=DEFAULT_NEIGHBORS,
        help=f"Semantic neighbours to show (default {DEFAULT_NEIGHBORS}).",
    )
    _ = parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_NEIGHBOR_THRESHOLD,
        help=f"Minimum cosine similarity for a neighbour (default {DEFAULT_NEIGHBOR_THRESHOLD}).",
    )
    _ = parser.add_argument("--json", action="store_true", help="Emit JSON instead of the text report.")
    return parser


def _select_snapshots(snapshots: Sequence[SnapshotRef], parsed: EvidenceNamespace) -> list[SnapshotRef]:
    selected: list[SnapshotRef] = []
    for snapshot in snapshots:
        if parsed.collection is not None and snapshot.collection != parsed.collection:
            continue
        if parsed.repo is not None and snapshot.repo != parsed.repo:
            continue
        selected.append(snapshot)
    return selected


def main(argv: list[str] | None = None) -> int:
    parsed = _parser().parse_args(argv, namespace=EvidenceNamespace())
    if parsed.symbol is None and parsed.path is None:
        _ = sys.stderr.write("pci-evidence: provide --symbol or --path\n")
        return 2
    try:
        with mcp_db.connect() as conn:
            if not mcp_db.code_intel_tables_exist(conn):
                _ = sys.stderr.write("pci-evidence: no code-intelligence tables; run pci-index first\n")
                return 1
            snapshots = _select_snapshots(latest_snapshots(conn), parsed)
            bundles: list[Evidence] = []
            for snapshot in snapshots:
                targets = resolve_targets(
                    conn, snapshot, symbol=parsed.symbol, source_path=parsed.path, line=parsed.line
                )
                bundles.extend(
                    build_evidence(conn, target, neighbors=parsed.neighbors, threshold=parsed.threshold)
                    for target in targets
                )
    except DatabaseConnectionError as exc:
        _ = sys.stderr.write(f"pci-evidence: {exc}\n")
        return 1
    if not bundles:
        _ = sys.stderr.write("pci-evidence: no matching symbol found\n")
        return 1
    if parsed.json:
        _ = sys.stdout.write(render_json(bundles))
    else:
        _ = sys.stdout.write("\n".join(render_text(bundle) for bundle in bundles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
