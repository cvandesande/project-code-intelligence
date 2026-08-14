"""Drift-tolerant fingerprints for the `pci check` regression ratchet.

Matching prefers a producer's own `partialFingerprints` (SARIF's own identity
mechanism). When a producer omits them, this module computes a fallback:
rule ID plus a normalized window of source lines around the finding location,
whitespace-stripped and line-number-agnostic. Code shifting up or down in the
file does not change the fingerprint as long as the finding's neighborhood is
unchanged -- the approach GitHub code scanning uses for baseline matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from project_code_intelligence.common import sha256_text

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject, StaticFinding

# Lines of source read above and below the finding's start line for the
# fallback fingerprint's code-context window.
CONTEXT_LINES = 3

# SARIF result levels, lowest to highest severity. "none" covers findings with
# no level set. Order matters: it is the ratchet's escalation ordering.
_LEVEL_ORDER = ("none", "note", "warning", "error")
_LEVEL_RANK = {level: rank for rank, level in enumerate(_LEVEL_ORDER)}


def level_rank(level: str | None) -> int:
    """Ordinal severity of a SARIF level string, unknown/missing treated as 'none'."""
    return _LEVEL_RANK.get(level or "none", 0)


def is_worsened(old_level: str | None, new_level: str | None) -> bool:
    """True when `new_level` is a strict escalation over `old_level` (note->warning->error)."""
    return level_rank(new_level) > level_rank(old_level)


def normalized_code_context(repo_root: Path, source_path: str | None, line_start: int | None) -> str:
    """Whitespace-stripped, blank-line-free window of source around a finding.

    Returns "" when the file or line cannot be read, so callers can fall back
    to a coarser fingerprint instead of matching on an empty context. An
    absolute `source_path` is also treated as unreadable: `repo_root / path`
    silently discards `repo_root` for an absolute `path` (`Path` semantics),
    which would read whatever happens to sit at that absolute path outside
    the repo rather than failing loudly.
    """
    if not source_path or line_start is None or line_start < 1:
        return ""
    relative = Path(source_path)
    if relative.is_absolute():
        return ""
    try:
        lines = (repo_root / relative).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    start = max(0, line_start - 1 - CONTEXT_LINES)
    end = min(len(lines), line_start - 1 + CONTEXT_LINES + 1)
    window = [line.strip() for line in lines[start:end]]
    return "\n".join(line for line in window if line)


@dataclass(frozen=True)
class FindingLocation:
    """The location fields `own_fingerprint` needs, bundled to keep its arity down."""

    source_path: str | None
    line_start: int | None
    column_start: int | None


def own_fingerprint(repo_root: Path, rule_id: str, location: FindingLocation, message: str) -> str:
    """Fallback fingerprint: rule ID + normalized code context + column, when available.

    The context window alone is not enough to separate two distinct findings
    of the same rule that both land on the same line (or the same ±CONTEXT_LINES
    neighborhood) at different columns, so `column_start` is folded in whenever
    the producer supplied one; it is drift-stable in the common case (unlike
    the line number, which the context window already tolerates shifting).
    When no code context could be read at all (missing file, no location),
    `message` disambiguates findings of the same rule that otherwise share no
    identity; residual duplicates (same rule, same message, same/no location)
    are handled by the caller's occurrence-index disambiguation, not here.
    """
    context = normalized_code_context(repo_root, location.source_path, location.line_start)
    column_part = f"\ncol:{location.column_start}" if location.column_start is not None else ""
    if context:
        return sha256_text(f"{rule_id}\n{context}{column_part}")
    return sha256_text(f"{rule_id}\n{location.source_path or ''}{column_part}\n{message}")


def partial_fingerprint(fingerprints: JsonObject) -> str | None:
    """Stable hash of a SARIF `partialFingerprints` object, or None when empty.

    Uses every key/value (sorted) rather than picking one, so a producer that
    ships more than one fingerprint key still gets one stable identity.
    """
    if not fingerprints:
        return None
    items = sorted((str(key), fingerprints[key]) for key in fingerprints)
    return sha256_text(repr(items))


def finding_fingerprint(repo_root: Path, finding: StaticFinding) -> str:
    """The base fingerprint used to match a finding across runs: producer's own, else our own.

    "Base" because two distinct findings can still legitimately collide here
    (e.g. two identical no-location findings from the same rule) -- callers
    disambiguate repeated occurrences of the same base fingerprint themselves
    (see `check_core.disambiguate_occurrences`).
    """
    fingerprint = partial_fingerprint(finding.fingerprints)
    if fingerprint is not None:
        return fingerprint
    location = FindingLocation(
        source_path=finding.primary_source_path,
        line_start=finding.line_start,
        column_start=finding.column_start,
    )
    return own_fingerprint(repo_root, finding.rule_id, location, finding.message)
