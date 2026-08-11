"""Nearest indexed definitions for code about to be written -- the add side of the hook.

``evidence`` answers "safe to remove this?" with a call graph. This answers "does this
already exist?" with embedding distance, and it is a weaker kind of answer: a ranked
neighbour list, never a verdict.

Measured on this repo 2026-08-11, blind-labelled (labeller saw the new function and three
candidates with no distances, ranks, or knowledge that a ranking existed):

* 53% of functions newly added over 50 commits had duplicate-or-reusable prior art already
  in the index, 43% an outright duplicate -- the check is worth running, not hypothetical.
* the true prior art was the nearest neighbour in 12 of 16 cases, within the top 2 in 15.
* prior art sat at median distance 0.215; edits with nothing reusable sat at 0.398 and
  never came closer than 0.236. ``GATE`` is set inside that gap.

Two earlier designs were measured and rejected, and both failure modes are worth keeping
in mind before widening anything here:

* call-shape overlap (the original add-side detector) fired on 11-13% of real duplicates
  and no threshold separated them from novel code -- deleted in 052a303.
* the gate does NOT transfer across languages. The same corpus shape on a Rust repo put
  known duplicates at 0.331 and arbitrary neighbours at 0.242, both ~0.09 above these
  numbers, and expressing the gate as a percentile of each repo's own nearest-neighbour
  distribution did not fix it (62nd percentile here, 86th there). ``GATE`` is therefore a
  Python-on-this-embedding-model constant, and ``PCI_ADD_SIDE_GATE`` exists so another
  repo can retune it without a code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from project_code_intelligence import analyze
from project_code_intelligence.mcp import db as mcp_db
from project_code_intelligence.mcp import semantic

if TYPE_CHECKING:
    from collections.abc import Mapping

# Distance below which a neighbour is worth showing. 0.25 keeps 69% of labelled prior art
# and suppresses 93% of the edits that had none; 0.20 suppresses all of it but keeps only
# 44%. Showing a wrong hit costs more than missing one here, since the injection competes
# with the agent's own reading, so the tighter half of the usable range wins.
GATE = 0.25
GATE_ENV = "PCI_ADD_SIDE_GATE"
# Shown lines, across all added definitions together. The injection shares a ~15-line
# budget with the reminder text and fires on every add, so it stays small.
MAX_HITS = 3
PER_DEFINITION = 2

_SQL = """
    SELECT r.symbol, r.source_path, r.line_start, r.embedding <=> %s::vector AS distance
    FROM project_code_intel_records r
    JOIN project_code_intel_files f
      ON f.snapshot_id = r.snapshot_id AND f.source_path = r.source_path
    WHERE r.snapshot_id = %s
      AND r.record_type = 'code_chunk'
      AND r.symbol IS NOT NULL
      AND r.symbol_kind IN ('function', 'method')
      AND r.file_role != 'test'
      AND r.metadata ->> 'impl_trait' IS NULL
      AND f.is_source = true
      AND f.is_test = false
      AND r.embedding IS NOT NULL
    ORDER BY distance
    LIMIT %s
"""


@dataclass(frozen=True)
class Hit:
    """One indexed definition close to something the edit is adding."""

    added_name: str
    symbol: str
    source_path: str
    line_start: int | None
    distance: float

    def render(self, repo: str) -> str:
        _, _, rel = (
            self.source_path.partition("/") if self.source_path.startswith(repo + "/") else ("", "", self.source_path)
        )
        location = f"{rel or self.source_path}:{self.line_start}" if self.line_start else (rel or self.source_path)
        return f"  {self.distance:.2f}  {self.symbol}  {location}  (vs your {self.added_name})"


def _coerce_distance(value: object) -> float | None:
    """Float from a pgvector distance column, else None. analyze.coerce_int rejects bool
    before int for the same reason: a bool is an int and would silently pass as 0.0/1.0."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def gate() -> float:
    """The distance cut-off, overridable per repo -- see the module docstring on why one
    constant cannot serve two languages."""
    configured = os.environ.get(GATE_ENV, "").strip()
    if not configured:
        return GATE
    try:
        return float(configured)
    except ValueError:
        return GATE


def nearest(slices: Mapping[str, str]) -> list[Hit]:
    """Indexed definitions closest to each added definition, best first, gate applied.

    Raises whatever the index or embedding endpoint raises -- the caller decides whether a
    failure means silence or a warning, and for this hook it must never mean silence.
    """
    limit = gate()
    hits: list[Hit] = []
    with mcp_db.connect() as conn:
        snapshots = analyze.latest_snapshots(conn)
        if not snapshots:
            return []
        snapshot = snapshots[0]
        for added_name, code in slices.items():
            vector, _dimensions = semantic.query_embedding(code)
            rows = conn.execute(_SQL, [vector, snapshot.snapshot_id, PER_DEFINITION]).fetchall()
            for row in rows:
                distance = _coerce_distance(row["distance"])
                symbol = analyze.coerce_str(row["symbol"])
                source_path = analyze.coerce_str(row["source_path"])
                if distance is None or distance > limit or symbol is None or source_path is None:
                    continue
                # The added definition is not in the index yet on a PreToolUse, but an
                # in-place rewrite of an existing function would match itself. Drop it:
                # "this is similar to itself" is noise, not prior art.
                if symbol.rpartition(".")[2] == added_name:
                    continue
                hits.append(
                    Hit(
                        added_name=added_name,
                        symbol=symbol,
                        source_path=source_path,
                        line_start=analyze.coerce_int(row["line_start"]),
                        distance=distance,
                    )
                )
    hits.sort(key=lambda hit: hit.distance)
    return hits[:MAX_HITS]
