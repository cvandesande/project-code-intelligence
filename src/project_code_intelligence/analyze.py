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
from difflib import SequenceMatcher
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
# Residual (parameterization) heavier than the shared core => a leaky abstraction.
_LEAKY_RATIO = 1.0
# Below this mean pairwise structural agreement, or this semantic cosine, a
# cluster is likely chained/incoherent — flagged so a loose group is not trusted
# on size alone. Single-linkage clustering can admit members below the join
# threshold by transitivity, which is exactly what this catches.
_LOW_COHERENCE_STRUCTURAL = 0.6
_LOW_COHERENCE_SEMANTIC = 0.6
# At or above this avg body-text similarity, members are near-identical text —
# the "typed variants" downgrade (see is_typed_variant_group) treats a
# same-body-different-return-type group as one shape with N type parameters,
# not N shapes worth collapsing.
_TYPED_VARIANT_TEXT = 0.85
# Two or more distinct annotated return-type cores => the family varies by type.
_MIN_DISTINCT_RETURN_CORES = 2
# Recommendations worth acting on now; ranked above leave-as-is so actionable
# candidates surface first.
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

    ``coherence`` is the rank key (see its docstring). ``net_value`` is the
    MDL-flavored evidence field: structural complexity removed by a shared
    abstraction minus the complexity that abstraction introduces. Both are
    advisory. ``recommendation`` turns the numbers into a verdict — one of
    ``worth-collapsing``, ``parameterize-carefully`` or ``leave-as-is`` — and
    the cost breakdown fields explain why.
    """

    members: tuple[FunctionNode, ...]
    common_roles: tuple[str, ...]
    avg_structural: float
    avg_semantic: float | None
    avg_text: float | None
    # Max pairwise body-text similarity. The average dilutes a byte-identical
    # pair inside a larger group, so "contains an exact-text copy" needs the max.
    max_text: float | None
    net_value: float
    value_ratio: float
    redundancy_removed: float
    abstraction_cost: float
    residual_cost: float
    spread_penalty: float
    shared_helper: tuple[str, ...]
    recommendation: str
    typed_variants: bool = False

    @property
    def residual_roles(self) -> tuple[str, ...]:
        """Roles that vary across members (evidence for the residual cost)."""
        return tuple(sorted(residual_role_union(self.members, self.common_roles)))

    @property
    def coherence(self) -> float:
        """Rank key: ``max(avg_semantic, avg_text)``, ``None`` ignored, ``0.0`` if both ``None``.

        Provisionally calibrated on a 10-group labeled sample, this repo.
        """
        values = [value for value in (self.avg_semantic, self.avg_text) if value is not None]
        return max(values) if values else 0.0

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
    path_prefix: str | None = None


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
            if _is_call_chain(eligible[i], eligible[j]):
                continue
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


_DEF_LINE_RE = re.compile(r"^\s*(?:async\s+)?def\s")
_DOCSTRING_OPEN_RE = re.compile(r'^[a-zA-Z]{0,2}("""|\'\'\')')


def normalize_body_text(text: str) -> str:
    """Crude textual normalization for body-similarity comparison.

    Drops the ``def`` line (including a multi-line signature) so names and
    signatures do not inflate the score, strips a leading docstring (a
    triple-quoted string as the first statement), and drops blank lines and
    trailing whitespace so formatting noise does not either.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _DEF_LINE_RE.match(line):
            end = index
            while end < len(lines) and not lines[end].rstrip().endswith(":"):
                end += 1
            lines = lines[end + 1 :]
            break
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start < len(lines):
        match = _DOCSTRING_OPEN_RE.match(lines[start].strip())
        if match:
            quote = match.group(1)
            rest = lines[start].strip()[match.end() :]
            start += 1
            if quote not in rest:
                while start < len(lines) and quote not in lines[start]:
                    start += 1
                start += 1
    lines = lines[start:]
    return "\n".join(line.rstrip() for line in lines if line.strip())


def body_text_similarity(a: str, b: str) -> float:
    """``SequenceMatcher`` ratio between two normalized function bodies."""
    return SequenceMatcher(None, normalize_body_text(a), normalize_body_text(b)).ratio()


_RETURN_ANNOTATION_RE = re.compile(r"->(.*):\s*$", re.DOTALL)
_OPTIONAL_WRAP_RE = re.compile(r"^Optional\[(.*)\]$")


def return_annotation(body_text: str) -> str | None:
    """Extract and normalize a function's return annotation from its ``def`` line(s).

    Reads the signature text between ``->`` and the signature's closing ``:``
    (which may span multiple lines), then strips whitespace/quotes and drops
    optionality (``| None`` / a wrapping ``Optional[...]``) so ``int | None``
    and ``int`` normalize to the same core. Returns ``None`` when there is no
    ``def`` line or no ``->`` annotation.
    """
    lines = body_text.splitlines()
    signature: str | None = None
    for index, line in enumerate(lines):
        if _DEF_LINE_RE.match(line):
            end = index
            while end < len(lines) and not lines[end].rstrip().endswith(":"):
                end += 1
            signature = "\n".join(lines[index : end + 1])
            break
    if signature is None:
        return None
    match = _RETURN_ANNOTATION_RE.search(signature)
    if match is None:
        return None
    core = _normalize_annotation(match.group(1).strip())
    return core or None


def _normalize_annotation(annotation: str) -> str:
    """Strip quotes and optionality from one extracted return annotation."""
    text = annotation.strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    parts = [part.strip() for part in text.split("|")]
    non_none = [part for part in parts if part != "None"]
    text = " | ".join(non_none) if non_none else "None"
    match = _OPTIONAL_WRAP_RE.match(text)
    if match:
        text = match.group(1).strip()
    return text


def has_typed_variants(annotations: Sequence[str | None]) -> bool:
    """True when the annotated (non-``None``) members disagree on return-type core.

    Members without an extractable annotation are ignored rather than counted
    as a distinct "unknown" core, since a missing annotation is not evidence
    the type differs.
    """
    cores = {annotation for annotation in annotations if annotation is not None}
    return len(cores) >= _MIN_DISTINCT_RETURN_CORES


def is_typed_variant_group(avg_text: float | None, annotations: Sequence[str | None]) -> bool:
    """Both required: near-identical bodies, and return types differing beyond optionality.

    Cheap deterministic proxy for families like ``optional_int``/``optional_bool``
    or ``classification_text``/``classification_bool`` — same body, only the
    extracted/returned type varies. Collapsing those loses type-checker
    precision, so they must not be recommended as worth-collapsing.
    """
    if avg_text is None or avg_text < _TYPED_VARIANT_TEXT:
        return False
    return has_typed_variants(annotations)


# --- net-value (MDL-flavored) scoring -------------------------------------------


def _last_component(name: str) -> str:
    """Final dotted component of a callee/symbol name (``repo.insert`` -> ``insert``)."""
    return name.rsplit(".", 1)[-1].strip()


def _is_call_chain(a: FunctionNode, b: FunctionNode) -> bool:
    """True if one node calls the other — composition, not duplication.

    A caller's call shape overlaps its callee's by construction (it calls the
    callee plus whatever the callee itself calls), so such a pair must never
    join one motif group. Deterministic rule, not a score.
    """
    a_callees = {_last_component(name) for name in a.callee_symbols}
    b_callees = {_last_component(name) for name in b.callee_symbols}
    return _last_component(b.symbol) in a_callees or _last_component(a.symbol) in b_callees


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
    """Internal helper(s) every member already calls — surfaced as evidence only.

    A shared callee is NOT proof the motif is abstracted: blind labeling (n=26,
    this repo) found every real duplicate pair shared some low-level callee they
    each duplicated code around, so this signal classifies at base rate. It no
    longer feeds ``net_value`` or ``recommendation``; a human reads it next to
    the members. Members' own symbols are excluded so mutual recursion is not
    mistaken for a shared helper. Returns names sorted for stable output."""
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


def net_value(redundancy: float, cost: float) -> float:
    """Tiebreak evidence: bits saved (removed - introduced)."""
    return redundancy - cost


def value_ratio(redundancy: float, cost: float) -> float:
    """Secondary evidence: the issue's literal "removed / introduced" reading."""
    if cost <= 0.0:
        return 0.0
    return redundancy / cost


def recommendation(value: float, shared: float, residual: float, *, typed_variants: bool = False) -> str:
    """Advisory verdict — explains, does not assert. See module docstring.

    ``typed_variants`` downgrades to ``leave-as-is`` regardless of ``value``:
    a same-body-different-return-type family (see ``is_typed_variant_group``)
    would lose type-checker precision if collapsed, so it is never recommended
    as worth collapsing. There is deliberately no already-abstracted verdict:
    the shared-callee signal behind it classified at base rate on 40 labeled
    groups and buried real duplicates (``shared_helper`` stays as evidence).
    """
    if typed_variants:
        return "leave-as-is"
    if value <= 0.0:
        return "leave-as-is"
    if residual > shared * _LEAKY_RATIO:
        return "parameterize-carefully"
    return "worth-collapsing"


def build_group(  # noqa: PLR0913 -- one more keyword-only evidence field; see AGENTS.md on keyword-with-default additions.
    members: Sequence[FunctionNode],
    avg_semantic: float | None,
    weights: Mapping[str, float] | None = None,
    function_symbols: frozenset[str] = frozenset(),
    avg_text: float | None = None,
    *,
    max_text: float | None = None,
    typed_variants: bool = False,
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
    value = net_value(removed, cost)
    return MotifGroup(
        members=ordered,
        common_roles=core,
        avg_structural=avg_structural,
        avg_semantic=avg_semantic,
        avg_text=avg_text,
        max_text=max_text,
        net_value=value,
        value_ratio=value_ratio(removed, cost),
        redundancy_removed=removed,
        abstraction_cost=cost,
        residual_cost=residual,
        spread_penalty=spread,
        shared_helper=helper,
        recommendation=recommendation(value, shared, residual, typed_variants=typed_variants),
        typed_variants=typed_variants,
    )


# --- database loading -----------------------------------------------------------


@dataclass(frozen=True)
class SnapshotRef:
    snapshot_id: int
    collection: str
    repo: str


def coerce_str(value: object) -> str | None:
    """Non-empty string from a DB row value, else None (shared: analyze, evidence, audit)."""
    return value if isinstance(value, str) and value else None


def coerce_int(value: object) -> int | None:
    """Int (not bool) from a DB row value, else None (shared: analyze, evidence, audit)."""
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
        snapshot_id = coerce_int(row["id"])
        collection = coerce_str(row["collection"])
        repo = coerce_str(row["repo"])
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
        source = coerce_str(row["source_record_id"])
        target = coerce_str(row["target_symbol"])
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
        record_id = coerce_str(row["record_id"])
        symbol = coerce_str(row["symbol"])
        source_path = coerce_str(row["source_path"])
        if record_id is None or symbol is None or source_path is None:
            continue
        raw_callees = callees.get(record_id, [])
        out.append(
            FunctionNode(
                record_id=record_id,
                symbol=symbol,
                source_path=source_path,
                line_start=coerce_int(row["line_start"]),
                line_end=coerce_int(row["line_end"]),
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


def _group_body_texts(conn: db.DbConnection, snapshot_id: int, member_ids: Sequence[str]) -> dict[str, str]:
    """Each member's full chunk text (still including the ``def`` line), by def id.

    Shared by ``group_text_similarity`` (which normalizes away the def line) and
    ``group_return_annotations`` (which reads only the def line), so both work
    from one query.
    """
    rows = conn.execute(
        """
        SELECT c.parent_record_id AS def_id, c.display_content AS content
        FROM project_code_intel_records c
        WHERE c.snapshot_id = %s
          AND c.record_type = 'code_chunk'
          AND c.parent_record_id = ANY(%s)
        ORDER BY c.parent_record_id, c.line_start
        """,
        [snapshot_id, list(member_ids)],
    ).fetchall()
    bodies: dict[str, list[str]] = {}
    for row in rows:
        def_id = coerce_str(row["def_id"])
        content = coerce_str(row["content"])
        if def_id is None or content is None:
            continue
        bodies.setdefault(def_id, []).append(content)
    return {def_id: "\n".join(chunks) for def_id, chunks in bodies.items()}


def group_text_similarity(
    conn: db.DbConnection, snapshot_id: int, member_ids: Sequence[str]
) -> tuple[float, float] | None:
    """(average, max) pairwise textual similarity over members' code-chunk children.

    Mirrors ``group_semantic_similarity``: the body text lives on the child
    ``code_chunk`` record (``parent_record_id`` = the definition id); a member
    can have several chunks, concatenated here in line order. Textual
    similarity is the strongest true-duplicate signal (byte-identical bodies,
    copied helpers) that graph shape and embedding cosine can miss. The max is
    reported alongside the average because the average dilutes a byte-identical
    pair inside a larger group. Returns ``None`` when fewer than two members
    have a chunk.
    """
    texts = _group_body_texts(conn, snapshot_id, member_ids)
    ids = sorted(texts)
    if len(ids) < DEFAULT_MIN_MEMBERS:
        return None
    ratios = [
        body_text_similarity(texts[ids[i]], texts[ids[j]]) for i in range(len(ids)) for j in range(i + 1, len(ids))
    ]
    if not ratios:
        return None
    return sum(ratios) / len(ratios), max(ratios)


def group_return_annotations(conn: db.DbConnection, snapshot_id: int, member_ids: Sequence[str]) -> list[str | None]:
    """Each member's normalized return annotation (see ``return_annotation``), or ``None``."""
    texts = _group_body_texts(conn, snapshot_id, member_ids)
    return [return_annotation(text) for text in texts.values()]


def _group_sort_key(group: MotifGroup) -> tuple[bool, float, float, int, int]:
    """Rank key: actionable groups first, then coherence, then net value.

    A 10-group labeled sample on this repo showed ``net_value`` order does not
    separate real duplicates from noise, while ``coherence`` does (every real
    group landed in the top 6). ``net_value`` remains the tiebreak.
    """
    return (
        group.recommendation in _ACTIONABLE_RECOMMENDATIONS,
        group.coherence,
        group.net_value,
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


def path_matches_prefix(source_path: str, prefix: str) -> bool:
    """Match a stored (repo-prefixed) path against a repo-relative or repo-prefixed prefix."""
    if source_path.startswith(prefix):
        return True
    _, _, repo_relative = source_path.partition("/")
    return repo_relative.startswith(prefix)


def analyze_snapshot(conn: db.DbConnection, snapshot: SnapshotRef, options: AnalysisOptions) -> SnapshotResult:
    nodes = load_function_nodes(conn, snapshot.snapshot_id)
    unique, folded = dedupe_clones(nodes)
    weights = role_weights(unique)
    function_symbols = frozenset(_last_component(node.symbol) for node in unique)
    clusters = cluster_functions(unique, options, weights)
    groups: list[MotifGroup] = []
    for members in clusters:
        member_ids = [member.record_id for member in members]
        text_similarity = group_text_similarity(conn, snapshot.snapshot_id, member_ids)
        avg_text = text_similarity[0] if text_similarity is not None else None
        max_text = text_similarity[1] if text_similarity is not None else None
        # Only worth a second query once the cheap text-similarity gate has passed.
        annotations = (
            group_return_annotations(conn, snapshot.snapshot_id, member_ids)
            if avg_text is not None and avg_text >= _TYPED_VARIANT_TEXT
            else []
        )
        groups.append(
            build_group(
                members,
                group_semantic_similarity(conn, snapshot.snapshot_id, member_ids),
                weights,
                function_symbols,
                avg_text,
                max_text=max_text,
                typed_variants=is_typed_variant_group(avg_text, annotations),
            )
        )
    if options.path_prefix:
        # Filter before ranking so the limit is spent on groups that touch the caller's area.
        prefix = options.path_prefix
        groups = [group for group in groups if any(path_matches_prefix(m.source_path, prefix) for m in group.members)]
    # Actionable groups first, then coherence, then net value; LOC is only a tiebreak, never the target.
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
        "text_similarity": None if group.avg_text is None else round(group.avg_text, 4),
        "max_text_similarity": None if group.max_text is None else round(group.max_text, 4),
        "coherence": round(group.coherence, 4),
        "recommendation": group.recommendation,
        "typed_variants": group.typed_variants,
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


def group_to_json(group: MotifGroup) -> dict[str, object]:
    """Public structured form of one motif group (for the MCP tool and JSON CLI)."""
    return _group_to_json(group)


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
    if group.typed_variants:
        return "same body, return types differ — collapsing loses type precision"
    if group.recommendation == "leave-as-is":
        return "abstraction cost meets or exceeds the redundancy it would remove"
    if group.recommendation == "parameterize-carefully":
        return (
            f"varying roles ({', '.join(group.residual_roles) or 'none'}) rival the shared core; abstraction would leak"
        )
    return "shared shape outweighs abstraction cost with little variation"


def _render_group(group: MotifGroup, ordinal: int) -> list[str]:
    semantic = "n/a" if group.avg_semantic is None else f"{group.avg_semantic:.2f}"
    text = "n/a" if group.avg_text is None else f"{group.avg_text:.2f}"
    loc_removed = "n/a" if group.estimated_loc_removed is None else str(group.estimated_loc_removed)
    cost_breakdown = f"base + residual {group.residual_cost:.2f} + spread {group.spread_penalty:.2f}"
    low_coherence_caveat = (
        ["  Caveat: low coherence — loose cluster, may be chained; verify members share one shape"]
        if group.low_coherence
        else []
    )
    return [
        f"Motif {ordinal}: {len(group.members)} instances — {group.recommendation} (net value {group.net_value:.2f})",
        f"  Why: {_why(group)}",
        *low_coherence_caveat,
        "  Common shape:",
        *(f"    {role}" for role in group.common_roles),
        "  Instances:",
        *(_render_member_line(member) for member in group.members),
        f"  Graph similarity:    {group.avg_structural:.2f}",
        f"  Semantic similarity: {semantic}",
        f"  Text similarity:     {text}",
        f"  Coherence (rank key): {group.coherence:.2f}",
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
    path_prefix: str | None = None
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
        "--path-prefix",
        dest="path_prefix",
        help="Only report groups with a member under this path prefix (repo-relative or repo-prefixed).",
    )
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


def select_snapshots(
    snapshots: Sequence[SnapshotRef], *, collection: str | None, repo: str | None
) -> list[SnapshotRef]:
    """Snapshots matching an optional collection/repo scope (public: CLI and MCP)."""
    selected: list[SnapshotRef] = []
    for snapshot in snapshots:
        if collection is not None and snapshot.collection != collection:
            continue
        if repo is not None and snapshot.repo != repo:
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
        path_prefix=parsed.path_prefix,
    )
    try:
        with mcp_db.connect() as conn:
            if not mcp_db.code_intel_tables_exist(conn):
                _ = sys.stderr.write("pci-analyze: no code-intelligence tables; run pci-index first\n")
                return 1
            snapshots = select_snapshots(latest_snapshots(conn), collection=parsed.collection, repo=parsed.repo)
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
    if args and args[0] == "audit":
        # audit imports this module, so a top-level import would be a cycle.
        from project_code_intelligence.audit import audit_main  # noqa: PLC0415

        return audit_main(args[1:])
    parser = argparse.ArgumentParser(prog="pci-analyze", description="Structural analysis passes over the index.")
    _ = parser.add_argument("subcommand", choices=["compression", "audit"], help="Analysis pass to run.")
    _ = parser.parse_args(args[:1])
    # parse_args above exits on an invalid/missing subcommand; unreachable otherwise.
    return compression_main(args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
