"""Shared types and diff logic for the `pci check` regression ratchet.

Producer-agnostic by construction: everything here works from `CheckFinding`,
a flat shape any SARIF producer's results are reduced to. Nothing branches on
tool name.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from project_code_intelligence.common import sha256_text
from project_code_intelligence.sarif.fingerprint import is_worsened

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class CheckFinding:
    """One normalized finding, reduced to what the ratchet needs to match and report it."""

    fingerprint: str
    rule_id: str
    level: str | None
    tool_name: str | None
    message: str
    primary_source_path: str | None
    line_start: int | None
    line_end: int | None


@dataclass(frozen=True)
class BaselineEntry:
    """One finding as frozen into a baseline: identity plus the level it had then."""

    fingerprint: str
    rule_id: str
    level: str | None


@dataclass(frozen=True)
class Regression:
    """A finding that fails the gate: new, or the same finding at a worse level."""

    status: str  # "new" | "worsened"
    finding: CheckFinding
    baseline_level: str | None = None


def _occurrence_sort_key(finding: CheckFinding) -> tuple[object, ...]:
    """Deterministic ordering for findings that share a base fingerprint.

    Sorts on the finding's own content, never on SARIF result order (which is
    not stable across producer runs) -- two occurrence lists built from the
    same underlying findings sort identically regardless of scan order.
    """
    return (
        finding.rule_id,
        finding.message,
        finding.primary_source_path or "",
        finding.line_start if finding.line_start is not None else -1,
        finding.line_end if finding.line_end is not None else -1,
    )


def disambiguate_occurrences(findings: Sequence[CheckFinding]) -> list[CheckFinding]:
    """Give every finding a unique, occurrence-aware fingerprint.

    Findings that share a base fingerprint (same rule + drift-tolerant location,
    or the producer's own partialFingerprints) are still distinct occurrences:
    N baselined occurrences of a finding vs N+1 fresh ones is itself a
    regression, so collapsing them to one entry (the old `dedupe_findings`
    behavior) would hide it. Occurrences are sorted by content and given a
    0-based index folded into the final fingerprint, so the same set of
    findings gets the same fingerprints regardless of SARIF result order.
    """
    grouped: dict[str, list[CheckFinding]] = {}
    for finding in findings:
        grouped.setdefault(finding.fingerprint, []).append(finding)
    out: list[CheckFinding] = []
    for base_fingerprint, group in grouped.items():
        for index, finding in enumerate(sorted(group, key=_occurrence_sort_key)):
            out.append(replace(finding, fingerprint=sha256_text(f"{base_fingerprint}:{index}")))
    return out


def branch_label(branch: str | None) -> str:
    """Display text for a baseline/snapshot branch, matching the None-means-detached scheme."""
    return branch or "(detached HEAD)"


def regression_line(regression: Regression) -> str:
    """One-line human-readable rendering of a regression, shared by `pci check` and `pci audit --gate`."""
    finding = regression.finding
    location = finding.primary_source_path or "(no location)"
    if finding.line_start is not None:
        location = f"{location}:{finding.line_start}"
    level = finding.level or "none"
    if regression.status == "worsened":
        return f"WORSENED  {finding.rule_id}  {level} (was {regression.baseline_level or 'none'})  {location}"
    return f"NEW       {finding.rule_id}  {level}  {location}"


def diff_against_baseline(baseline: Sequence[BaselineEntry], current: Sequence[CheckFinding]) -> list[Regression]:
    """(new findings, worsened findings) against `baseline`, as one ordered list.

    A finding not in `current` (fixed since the baseline) is not reported --
    the ratchet only fails builds forward, it never asks for cleanup credit.
    """
    baseline_by_fingerprint = {entry.fingerprint: entry for entry in baseline}
    regressions: list[Regression] = []
    for finding in current:
        base = baseline_by_fingerprint.get(finding.fingerprint)
        if base is None:
            regressions.append(Regression(status="new", finding=finding))
        elif is_worsened(base.level, finding.level):
            regressions.append(Regression(status="worsened", finding=finding, baseline_level=base.level))
    return regressions
