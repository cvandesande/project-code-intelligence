"""Structural compression analysis (Gate A prototype).

Read-only analysis over the existing code-intelligence index. It finds groups
of functions/methods whose *call shape* is structurally similar — repeated
motifs such as ``validate -> convert -> repository write -> error mapping`` —
using only facts already in the index: heuristic ``call_candidate`` edges plus
the embeddings on their code chunks.

This is the Gate A prototype from ``docs/feasibility-structural-compression.md``.
It uses no Rust extractor and no new schema. Results are advisory: they explain
why PCI considers two regions similar; they do not assert a refactor is correct.

Scope note: analysis runs within a single snapshot (one repo) at a time, since
call-shape motifs are most interpretable inside one module tree.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from project_code_intelligence.exceptions import DatabaseConnectionError
from project_code_intelligence.mcp import db as mcp_db

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from project_code_intelligence import db

# --- tuning constants (all overridable from the CLI) ---------------------------
DEFAULT_THRESHOLD = 0.6
DEFAULT_MIN_ROLES = 3
DEFAULT_MIN_MEMBERS = 2
DEFAULT_LIMIT = 20
_MAJORITY = 0.5

# --- net-value tuning constants (advisory; MDL-flavored, all in role-weight units) ---
# Cost to name and wire any new abstraction, so tiny groups do not get an
# infinite value ratio. One "unit" is roughly one distinctive shared role.
_ABSTRACTION_BASE = 1.0
# Extra cost per additional module a shared abstraction must reach across
# (a new import/dependency edge). In-module groups pay nothing.
_SPREAD_UNIT = 0.5
# When members already share an internal helper, positive net value is damped
# toward zero so "already abstracted" groups rank low, not high.
_EXISTING_HELPER_FACTOR = 0.1
# Residual (parameterization) heavier than the shared core => a leaky abstraction.
_LEAKY_RATIO = 1.0
# Below this mean pairwise structural agreement, or this semantic cosine, a
# cluster is likely chained/incoherent — flagged so a loose group is not trusted
# on size alone. Single-linkage clustering can admit members below the join
# threshold by transitivity, which is exactly what this catches.
_LOW_COHERENCE_STRUCTURAL = 0.6
_LOW_COHERENCE_SEMANTIC = 0.6
# Recommendations worth acting on now; ranked above already-abstracted and
# leave-as-is so actionable candidates surface first.
_ACTIONABLE_RECOMMENDATIONS = frozenset({"worth-collapsing", "parameterize-carefully"})

_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-z]+")
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


@dataclass(frozen=True)
class FunctionNode:
    """One function/method plus the normalized shape of what it calls."""

    record_id: str
    symbol: str
    source_path: str
    line_start: int | None
    line_end: int | None
    callee_roles: frozenset[str]
    callee_symbols: frozenset[str] = frozenset()

    @property
    def loc(self) -> int | None:
        if self.line_start is None or self.line_end is None:
            return None
        return max(self.line_end - self.line_start + 1, 0)


@dataclass(frozen=True)
class MotifGroup:
    """A cluster of structurally similar functions and its net-value evidence.

    ``net_value`` is the MDL-flavored rank key: structural complexity removed by
    a shared abstraction minus the complexity that abstraction introduces. It is
    advisory. ``recommendation`` turns the number into a verdict — one of
    ``worth-collapsing``, ``parameterize-carefully``, ``already-abstracted`` or
    ``leave-as-is`` — and the cost breakdown fields explain why.
    """

    members: tuple[FunctionNode, ...]
    common_roles: tuple[str, ...]
    avg_structural: float
    avg_semantic: float | None
    net_value: float
    value_ratio: float
    redundancy_removed: float
    abstraction_cost: float
    residual_cost: float
    spread_penalty: float
    shared_helper: tuple[str, ...]
    recommendation: str

    @property
    def residual_roles(self) -> tuple[str, ...]:
        """Roles that vary across members (evidence for the residual cost)."""
        return tuple(sorted(residual_role_union(self.members, self.common_roles)))

    @property
    def low_coherence(self) -> bool:
        """The cluster is loose — weak mean agreement or weak semantic overlap.

        A true value warns that the group may be a single-linkage chain rather
        than one shared shape; verify the members in source before trusting it.
        """
        if self.avg_structural < _LOW_COHERENCE_STRUCTURAL:
            return True
        return self.avg_semantic is not None and self.avg_semantic < _LOW_COHERENCE_SEMANTIC

    @property
    def estimated_loc_removed(self) -> int | None:
        """Advisory "keep one, fold the rest" proxy: total LOC minus the largest member.

        This is a crude upper bound on lines a shared abstraction might remove.
        It ignores abstraction cost and does not imply the members should merge;
        LOC is deliberately not the optimization target (see the feasibility doc).
        Returns None when fewer than two members have a known line range.
        """
        locs = [member.loc for member in self.members if member.loc is not None]
        if len(locs) < DEFAULT_MIN_MEMBERS:
            return None
        return sum(locs) - max(locs)


@dataclass
class AnalysisOptions:
    threshold: float = DEFAULT_THRESHOLD
    min_roles: int = DEFAULT_MIN_ROLES
    min_members: int = DEFAULT_MIN_MEMBERS
    limit: int = DEFAULT_LIMIT


# --- pure analysis logic (no database) -----------------------------------------


def normalize_role(callee: str) -> str:
    """Abstract a callee name into a structural role.

    Keeps the leading verb token and drops the domain noun, so that
    ``validate_user``, ``validate_team`` and ``self.validateApp`` all collapse
    to ``validate``. This is a deliberately simple heuristic for the prototype;
    it assumes verb-first naming and is documented as such.
    """
    tail = callee.rsplit(".", 1)[-1].strip()
    tail = tail.lstrip("_")
    if not tail:
        return callee.strip().lower()
    parts: list[str] = []
    for chunk in _TOKEN_SPLIT.split(tail):
        parts.extend(piece for piece in _CAMEL_SPLIT.split(chunk) if piece)
    if not parts:
        return tail.lower()
    return parts[0].lower()


def role_set(callees: Sequence[str]) -> frozenset[str]:
    return frozenset(normalize_role(callee) for callee in callees if callee)


def role_weights(nodes: Sequence[FunctionNode]) -> dict[str, float]:
    """Inverse-document-frequency weight per role over the given functions.

    ``idf = log(N / df)``. A role present in every function has weight 0 (pure
    boilerplate, ignored by similarity); a rare role scores high. This is the
    generic, language-agnostic way to stop shared boilerplate such as ``str`` or
    ``isinstance`` from manufacturing false motifs.
    """
    total = len(nodes)
    if total == 0:
        return {}
    document_frequency: dict[str, int] = {}
    for node in nodes:
        for role in node.callee_roles:
            document_frequency[role] = document_frequency.get(role, 0) + 1
    return {role: math.log(total / count) for role, count in document_frequency.items()}


def _weight(weights: Mapping[str, float] | None, role: str) -> float:
    if weights is None:
        return 1.0
    return weights.get(role, 1.0)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Unweighted Jaccard (kept as a primitive; equals ``weighted_jaccard(..., None)``)."""
    return weighted_jaccard(left, right, None)


def weighted_jaccard(left: frozenset[str], right: frozenset[str], weights: Mapping[str, float] | None) -> float:
    """Weighted Jaccard over roles. ``weights=None`` gives the plain set Jaccard."""
    union = left | right
    if not union:
        return 0.0
    union_weight = sum(_weight(weights, role) for role in union)
    if union_weight <= 0.0:
        return 0.0
    intersection_weight = sum(_weight(weights, role) for role in (left & right))
    return intersection_weight / union_weight


def _union_find_root(parent: dict[int, int], node: int) -> int:
    root = node
    while parent[root] != root:
        root = parent[root]
    while parent[node] != root:
        parent[node], node = root, parent[node]
    return root


def dedupe_clones(nodes: Sequence[FunctionNode]) -> tuple[list[FunctionNode], int]:
    """Fold byte-identical clones into one representative.

    Two functions with the same symbol name, the same call shape, and the same
    line count are treated as one node. This removes the double-counting caused
    by a file that exists at two paths (e.g. a vendored/package-data copy) so a
    literal duplicate cannot masquerade as a structural motif. Returns the
    deduplicated nodes and the number folded away.
    """
    representatives: dict[tuple[str, frozenset[str], int | None], FunctionNode] = {}
    folded = 0
    for node in nodes:
        key = (node.symbol, node.callee_roles, node.loc)
        existing = representatives.get(key)
        if existing is None:
            representatives[key] = node
        else:
            folded += 1
            if node.source_path < existing.source_path:
                representatives[key] = node
    return list(representatives.values()), folded


def cluster_functions(
    nodes: Sequence[FunctionNode],
    options: AnalysisOptions,
    weights: Mapping[str, float] | None = None,
) -> list[list[FunctionNode]]:
    """Group functions by approximate call-shape similarity (union-find)."""
    eligible = [node for node in nodes if len(node.callee_roles) >= options.min_roles]
    parent = {index: index for index in range(len(eligible))}
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            if weighted_jaccard(eligible[i].callee_roles, eligible[j].callee_roles, weights) >= options.threshold:
                parent[_union_find_root(parent, i)] = _union_find_root(parent, j)
    clusters: dict[int, list[FunctionNode]] = {}
    for index, node in enumerate(eligible):
        clusters.setdefault(_union_find_root(parent, index), []).append(node)
    return [members for members in clusters.values() if len(members) >= options.min_members]


def common_roles(members: Sequence[FunctionNode], weights: Mapping[str, float] | None = None) -> tuple[str, ...]:
    """Roles shared by a majority of members — the motif's common shape.

    Pure-boilerplate roles (weight 0) are dropped, and the rest are ordered by
    weight so the distinctive roles that define the motif appear first.
    """
    counts: dict[str, int] = {}
    for member in members:
        for role in member.callee_roles:
            counts[role] = counts.get(role, 0) + 1
    threshold = max(len(members) * _MAJORITY, 1.0)
    shared = [role for role, count in counts.items() if count >= threshold]
    if weights is not None:
        shared = [role for role in shared if _weight(weights, role) > 0.0]
    shared.sort(key=lambda role: (-_weight(weights, role), role))
    return tuple(shared)


def average_structural_similarity(members: Sequence[FunctionNode], weights: Mapping[str, float] | None = None) -> float:
    pairs = [
        weighted_jaccard(members[i].callee_roles, members[j].callee_roles, weights)
        for i in range(len(members))
        for j in range(i + 1, len(members))
    ]
    if not pairs:
        return 0.0
    return sum(pairs) / len(pairs)


# --- net-value (MDL-flavored) scoring -------------------------------------------


def _last_component(name: str) -> str:
    """Final dotted component of a callee/symbol name (``repo.insert`` -> ``insert``)."""
    return name.rsplit(".", 1)[-1].strip()


def core_shared_weight(core_roles: Sequence[str], weights: Mapping[str, float] | None) -> float:
    """Information content of the repeated skeleton, in IDF role-weight units."""
    return sum(_weight(weights, role) for role in core_roles)


def residual_role_union(members: Sequence[FunctionNode], core_roles: Sequence[str]) -> frozenset[str]:
    """Roles any member carries beyond the shared core — the part that varies."""
    core = frozenset(core_roles)
    union: set[str] = set()
    for member in members:
        union |= member.callee_roles - core
    return frozenset(union)


def residual_cost(
    members: Sequence[FunctionNode], core_roles: Sequence[str], weights: Mapping[str, float] | None
) -> float:
    """Parameterization the abstraction must carry: weight of the residual role union.

    This is the "type-parameter variability" term. Identical call shapes give an
    empty residual (cheap abstraction); members that each do different extra work
    give a large residual (a leaky, heavily parameterized abstraction).
    """
    return sum(_weight(weights, role) for role in residual_role_union(members, core_roles))


def spread_penalty(members: Sequence[FunctionNode]) -> float:
    """Cost of reaching across modules: each extra directory is a new dependency edge."""
    directories = {member.source_path.rsplit("/", 1)[0] for member in members}
    return max(len(directories) - 1, 0) * _SPREAD_UNIT


def abstraction_cost(
    members: Sequence[FunctionNode], core_roles: Sequence[str], weights: Mapping[str, float] | None
) -> float:
    """Complexity introduced by the abstraction: base + residual + cross-module spread."""
    return _ABSTRACTION_BASE + residual_cost(members, core_roles, weights) + spread_penalty(members)


def redundancy_removed(
    members: Sequence[FunctionNode],
    core_roles: Sequence[str],
    avg_structural: float,
    weights: Mapping[str, float] | None,
) -> float:
    """Structural complexity removed: (K-1) redundant copies of the shared core,
    discounted by how tightly the members actually agree.

    Agreement enters squared. Single-linkage clustering chains transitively
    related functions into large, loose groups whose mean pairwise agreement can
    fall below the join threshold; a linear discount let such a chain out-rank a
    small, tight motif on member count alone. Squaring penalizes looseness
    harder so a coherent group beats a sprawling one of similar raw mass.
    """
    return (len(members) - 1) * core_shared_weight(core_roles, weights) * avg_structural * avg_structural


def existing_helper(members: Sequence[FunctionNode], function_symbols: frozenset[str]) -> tuple[str, ...]:
    """Internal helper(s) every member already calls — evidence the motif is
    already abstracted. Members' own symbols are excluded so mutual recursion is
    not mistaken for a shared helper. Returns names sorted for stable output."""
    if not members:
        return ()
    member_names = {_last_component(member.symbol) for member in members}
    shared: set[str] | None = None
    for member in members:
        concrete = {_last_component(callee) for callee in member.callee_symbols}
        shared = concrete if shared is None else (shared & concrete)
    if not shared:
        return ()
    internal = {name for name in shared if name in function_symbols and name not in member_names}
    return tuple(sorted(internal))


def net_value(redundancy: float, cost: float, *, has_helper: bool) -> float:
    """Rank key: bits saved (removed - introduced). Positive value is damped when
    an existing helper already realizes the motif, so such groups rank low."""
    base = redundancy - cost
    if has_helper and base > 0.0:
        return base * _EXISTING_HELPER_FACTOR
    return base


def value_ratio(redundancy: float, cost: float) -> float:
    """Secondary evidence: the issue's literal "removed / introduced" reading."""
    if cost <= 0.0:
        return 0.0
    return redundancy / cost


def recommendation(value: float, shared: float, residual: float, *, has_helper: bool) -> str:
    """Advisory verdict — explains, does not assert. See module docstring."""
    if has_helper:
        return "already-abstracted"
    if value <= 0.0:
        return "leave-as-is"
    if residual > shared * _LEAKY_RATIO:
        return "parameterize-carefully"
    return "worth-collapsing"


def build_group(
    members: Sequence[FunctionNode],
    avg_semantic: float | None,
    weights: Mapping[str, float] | None = None,
    function_symbols: frozenset[str] = frozenset(),
) -> MotifGroup:
    ordered = tuple(sorted(members, key=lambda node: (node.source_path, node.line_start or 0, node.symbol)))
    avg_structural = average_structural_similarity(ordered, weights)
    core = common_roles(ordered, weights)
    shared = core_shared_weight(core, weights)
    residual = residual_cost(ordered, core, weights)
    spread = spread_penalty(ordered)
    cost = _ABSTRACTION_BASE + residual + spread
    removed = redundancy_removed(ordered, core, avg_structural, weights)
    helper = existing_helper(ordered, function_symbols)
    value = net_value(removed, cost, has_helper=bool(helper))
    return MotifGroup(
        members=ordered,
        common_roles=core,
        avg_structural=avg_structural,
        avg_semantic=avg_semantic,
        net_value=value,
        value_ratio=value_ratio(removed, cost),
        redundancy_removed=removed,
        abstraction_cost=cost,
        residual_cost=residual,
        spread_penalty=spread,
        shared_helper=helper,
        recommendation=recommendation(value, shared, residual, has_helper=bool(helper)),
    )


# --- database loading -----------------------------------------------------------


@dataclass(frozen=True)
class SnapshotRef:
    snapshot_id: int
    collection: str
    repo: str


def _coerce_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def latest_snapshots(conn: db.DbConnection) -> list[SnapshotRef]:
    rows = conn.execute(
        """
        SELECT DISTINCT ON (collection, repo)
               id, collection, repo
        FROM project_code_intel_snapshots
        ORDER BY collection, repo, created_at DESC, id DESC
        """
    ).fetchall()
    out: list[SnapshotRef] = []
    for row in rows:
        snapshot_id = _coerce_int(row["id"])
        collection = _coerce_str(row["collection"])
        repo = _coerce_str(row["repo"])
        if snapshot_id is None or collection is None or repo is None:
            continue
        out.append(SnapshotRef(snapshot_id=snapshot_id, collection=collection, repo=repo))
    return out


def _callees_by_source(conn: db.DbConnection, snapshot_id: int) -> dict[str, list[str]]:
    rows = conn.execute(
        """
        SELECT source_record_id, target_symbol
        FROM project_code_intel_edges
        WHERE snapshot_id = %s
          AND edge_type = 'call_candidate'
          AND target_symbol IS NOT NULL
        """,
        [snapshot_id],
    ).fetchall()
    out: dict[str, list[str]] = {}
    for row in rows:
        source = _coerce_str(row["source_record_id"])
        target = _coerce_str(row["target_symbol"])
        if source is None or target is None:
            continue
        out.setdefault(source, []).append(target)
    return out


def load_function_nodes(conn: db.DbConnection, snapshot_id: int) -> list[FunctionNode]:
    callees = _callees_by_source(conn, snapshot_id)
    rows = conn.execute(
        """
        SELECT r.record_id, r.symbol, r.source_path, r.line_start, r.line_end
        FROM project_code_intel_records r
        JOIN project_code_intel_files f
          ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
        WHERE r.snapshot_id = %s
          AND r.record_type = 'symbol_definition'
          AND r.symbol IS NOT NULL
          AND r.symbol_kind IN ('function', 'method', 'shell_function')
          AND f.is_source = true
          AND f.is_test = false
        """,
        [snapshot_id],
    ).fetchall()
    out: list[FunctionNode] = []
    for row in rows:
        record_id = _coerce_str(row["record_id"])
        symbol = _coerce_str(row["symbol"])
        source_path = _coerce_str(row["source_path"])
        if record_id is None or symbol is None or source_path is None:
            continue
        raw_callees = callees.get(record_id, [])
        out.append(
            FunctionNode(
                record_id=record_id,
                symbol=symbol,
                source_path=source_path,
                line_start=_coerce_int(row["line_start"]),
                line_end=_coerce_int(row["line_end"]),
                callee_roles=role_set(raw_callees),
                callee_symbols=frozenset(callee for callee in raw_callees if callee),
            )
        )
    return out


def group_semantic_similarity(conn: db.DbConnection, snapshot_id: int, member_ids: Sequence[str]) -> float | None:
    """Average pairwise embedding cosine over members' code-chunk children.

    ``symbol_definition`` records carry no embedding; the embedded body lives on
    the child ``code_chunk`` record (``parent_record_id`` = the definition id).
    Returns ``None`` when fewer than two members have an embedded chunk.
    """
    row = conn.execute(
        """
        WITH chunks AS (
            SELECT c.parent_record_id AS def_id, c.embedding
            FROM project_code_intel_records c
            WHERE c.snapshot_id = %s
              AND c.record_type = 'code_chunk'
              AND c.parent_record_id = ANY(%s)
              AND c.embedding IS NOT NULL
        )
        SELECT avg(1 - (a.embedding <=> b.embedding)) AS sim
        FROM chunks a
        JOIN chunks b ON a.def_id < b.def_id
        """,
        [snapshot_id, list(member_ids)],
    ).fetchone()
    if row is None:
        return None
    sim = row["sim"]
    if isinstance(sim, (int, float)) and not isinstance(sim, bool):
        return float(sim)
    return None


def _group_sort_key(group: MotifGroup) -> tuple[bool, float, float, int, int]:
    """Rank key: actionable groups first, then net value, then coherence.

    Semantic similarity breaks net-value ties so a tighter group wins; groups
    with no embedded body (``avg_semantic is None``) sort last within a tie.
    """
    return (
        group.recommendation in _ACTIONABLE_RECOMMENDATIONS,
        group.net_value,
        group.avg_semantic if group.avg_semantic is not None else -1.0,
        len(group.members),
        group.estimated_loc_removed or 0,
    )


def rank_groups(groups: Sequence[MotifGroup]) -> list[MotifGroup]:
    """Order groups so actionable, high-value, coherent candidates come first."""
    return sorted(groups, key=_group_sort_key, reverse=True)


@dataclass(frozen=True)
class SnapshotResult:
    """Per-snapshot analysis output plus the counts that explain what was scanned."""

    label: str
    groups: tuple[MotifGroup, ...]
    functions_analyzed: int
    clones_folded: int


def analyze_snapshot(conn: db.DbConnection, snapshot: SnapshotRef, options: AnalysisOptions) -> SnapshotResult:
    nodes = load_function_nodes(conn, snapshot.snapshot_id)
    unique, folded = dedupe_clones(nodes)
    weights = role_weights(unique)
    function_symbols = frozenset(_last_component(node.symbol) for node in unique)
    clusters = cluster_functions(unique, options, weights)
    groups = [
        build_group(
            members,
            group_semantic_similarity(conn, snapshot.snapshot_id, [member.record_id for member in members]),
            weights,
            function_symbols,
        )
        for members in clusters
    ]
    # Actionable groups first, then net value; LOC is only a tiebreak, never the target.
    ranked = rank_groups(groups)
    return SnapshotResult(
        label=f"{snapshot.collection}/{snapshot.repo}",
        groups=tuple(ranked[: options.limit]),
        functions_analyzed=len(unique),
        clones_folded=folded,
    )


# --- rendering ------------------------------------------------------------------


def _group_to_json(group: MotifGroup) -> dict[str, object]:
    return {
        "members": [
            {
                "symbol": member.symbol,
                "source_path": member.source_path,
                "line_start": member.line_start,
                "line_end": member.line_end,
                "loc": member.loc,
                "record_id": member.record_id,
            }
            for member in group.members
        ],
        "common_shape": list(group.common_roles),
        "graph_similarity": round(group.avg_structural, 4),
        "semantic_similarity": None if group.avg_semantic is None else round(group.avg_semantic, 4),
        "recommendation": group.recommendation,
        "net_value": round(group.net_value, 4),
        "evidence": {
            "redundancy_removed": round(group.redundancy_removed, 4),
            "abstraction_cost": round(group.abstraction_cost, 4),
            "residual_cost": round(group.residual_cost, 4),
            "spread_penalty": round(group.spread_penalty, 4),
            "value_ratio": round(group.value_ratio, 4),
            "residual_roles": list(group.residual_roles),
            "shared_helper": list(group.shared_helper),
            "estimated_loc_removed": group.estimated_loc_removed,
            "low_coherence": group.low_coherence,
        },
    }


def render_json(results: Sequence[SnapshotResult]) -> str:
    payload = {
        result.label: {
            "functions_analyzed": result.functions_analyzed,
            "clones_folded": result.clones_folded,
            "groups": [_group_to_json(group) for group in result.groups],
        }
        for result in results
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _render_member_line(member: FunctionNode) -> str:
    loc = f"{member.loc} LOC" if member.loc is not None else "LOC ?"
    return f"    {member.symbol:<28} {member.source_path}:{member.line_start} ({loc})"


def _why(group: MotifGroup) -> str:
    """One-line rationale for the recommendation — explain, do not assert."""
    if group.recommendation == "already-abstracted":
        return f"members already share internal helper(s): {', '.join(group.shared_helper)}"
    if group.recommendation == "leave-as-is":
        return "abstraction cost meets or exceeds the redundancy it would remove"
    if group.recommendation == "parameterize-carefully":
        return (
            f"varying roles ({', '.join(group.residual_roles) or 'none'}) rival the shared core; abstraction would leak"
        )
    return "shared shape outweighs abstraction cost with little variation"


def _render_group(group: MotifGroup, ordinal: int) -> list[str]:
    semantic = "n/a" if group.avg_semantic is None else f"{group.avg_semantic:.2f}"
    loc_removed = "n/a" if group.estimated_loc_removed is None else str(group.estimated_loc_removed)
    cost_breakdown = f"base + residual {group.residual_cost:.2f} + spread {group.spread_penalty:.2f}"
    coherence = (
        ["  Caveat: low coherence — loose cluster, may be chained; verify members share one shape"]
        if group.low_coherence
        else []
    )
    return [
        f"Motif {ordinal}: {len(group.members)} instances — {group.recommendation} (net value {group.net_value:.2f})",
        f"  Why: {_why(group)}",
        *coherence,
        "  Common shape:",
        *(f"    {role}" for role in group.common_roles),
        "  Instances:",
        *(_render_member_line(member) for member in group.members),
        f"  Graph similarity:    {group.avg_structural:.2f}",
        f"  Semantic similarity: {semantic}",
        f"  Redundancy removed:  {group.redundancy_removed:.2f}  (weight units)",
        f"  Abstraction cost:    {group.abstraction_cost:.2f}  ({cost_breakdown})",
        f"  Est. LOC removed:    {loc_removed} (advisory; LOC is evidence, not the target)",
        "",
    ]


def render_text(results: Sequence[SnapshotResult]) -> str:
    lines: list[str] = [
        "# Structural compression candidates (Gate A prototype)",
        "",
        "Advisory only. Shape is inferred from heuristic call_candidate edges;",
        "verify in source before proposing any refactor. Roles are IDF-weighted so",
        "shared boilerplate counts less than shared distinctive calls.",
        "",
    ]
    for result in results:
        lines.extend((
            f"## {result.label}",
            f"_{result.functions_analyzed} functions analyzed, {result.clones_folded} exact clones folded_",
            "",
        ))
        if not result.groups:
            lines.extend(("_no repeated call-shape motifs above thresholds_", ""))
            continue
        for ordinal, group in enumerate(result.groups, start=1):
            lines.extend(_render_group(group, ordinal))
    return "\n".join(lines) + "\n"


# --- CLI ------------------------------------------------------------------------


@dataclass
class AnalyzeNamespace(argparse.Namespace):
    collection: str | None = None
    repo: str | None = None
    threshold: float = DEFAULT_THRESHOLD
    min_roles: int = DEFAULT_MIN_ROLES
    min_members: int = DEFAULT_MIN_MEMBERS
    limit: int = DEFAULT_LIMIT
    json: bool = False
    extra: list[str] = field(default_factory=list)


def _analyze_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pci-analyze compression",
        description="Find groups of functions with repeated call-shape motifs (advisory).",
    )
    _ = parser.add_argument("--collection", help="Restrict to one collection/workspace.")
    _ = parser.add_argument("--repo", help="Restrict to one repo within the collection(s).")
    _ = parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Call-shape Jaccard similarity to cluster two functions (default {DEFAULT_THRESHOLD}).",
    )
    _ = parser.add_argument(
        "--min-roles",
        type=int,
        default=DEFAULT_MIN_ROLES,
        help=f"Minimum distinct call roles a function must have to be considered (default {DEFAULT_MIN_ROLES}).",
    )
    _ = parser.add_argument(
        "--min-members",
        type=int,
        default=DEFAULT_MIN_MEMBERS,
        help=f"Minimum functions in a reported motif group (default {DEFAULT_MIN_MEMBERS}).",
    )
    _ = parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum motif groups to report per snapshot (default {DEFAULT_LIMIT}).",
    )
    _ = parser.add_argument("--json", action="store_true", help="Emit JSON instead of the text report.")
    return parser


def _select_snapshots(snapshots: Sequence[SnapshotRef], parsed: AnalyzeNamespace) -> list[SnapshotRef]:
    selected: list[SnapshotRef] = []
    for snapshot in snapshots:
        if parsed.collection is not None and snapshot.collection != parsed.collection:
            continue
        if parsed.repo is not None and snapshot.repo != parsed.repo:
            continue
        selected.append(snapshot)
    return selected


def compression_main(argv: list[str] | None = None) -> int:
    parser = _analyze_parser()
    parsed = parser.parse_args(argv, namespace=AnalyzeNamespace())
    options = AnalysisOptions(
        threshold=parsed.threshold,
        min_roles=parsed.min_roles,
        min_members=parsed.min_members,
        limit=parsed.limit,
    )
    try:
        with mcp_db.connect() as conn:
            if not mcp_db.code_intel_tables_exist(conn):
                _ = sys.stderr.write("pci-analyze: no code-intelligence tables; run pci-index first\n")
                return 1
            snapshots = _select_snapshots(latest_snapshots(conn), parsed)
            if not snapshots:
                _ = sys.stderr.write("pci-analyze: no matching snapshots found\n")
                return 1
            results = [analyze_snapshot(conn, snapshot, options) for snapshot in snapshots]
    except DatabaseConnectionError as exc:
        _ = sys.stderr.write(f"pci-analyze: {exc}\n")
        return 1
    output = render_json(results) if parsed.json else render_text(results)
    _ = sys.stdout.write(output)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "compression":
        return compression_main(args[1:])
    parser = argparse.ArgumentParser(prog="pci-analyze", description="Structural analysis passes over the index.")
    _ = parser.add_argument("subcommand", choices=["compression"], help="Analysis pass to run.")
    _ = parser.parse_args(args[:1])
    # parse_args above exits on an invalid/missing subcommand; unreachable otherwise.
    return compression_main(args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
