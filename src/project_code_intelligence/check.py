"""`pci check`: a SARIF regression ratchet.

External static-analysis producers emit SARIF; `pci check` ingests it (reusing
the `sarif` ingest module), freezes or diffs a per-branch baseline, and fails
only on findings that are new or have escalated in level since the baseline.
Producer-agnostic: every finding, from every tool, goes through the same pool.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from project_code_intelligence import config, db
from project_code_intelligence.analyze import resolve_repo_branch
from project_code_intelligence.check_core import (
    CheckFinding,
    Regression,
    branch_label,
    diff_against_baseline,
    disambiguate_occurrences,
    regression_line,
)
from project_code_intelligence.common import default_collection
from project_code_intelligence.exceptions import DatabaseConnectionError, SarifLoadError
from project_code_intelligence.rulepacks import RulepackRuleInfo, discover_rulepacks, rule_lookup
from project_code_intelligence.sarif.fingerprint import finding_fingerprint
from project_code_intelligence.sarif.ingest import ingest_sarif
from project_code_intelligence.sarif.types import SarifIngestContext
from project_code_intelligence.storage.check import freeze_baseline, load_baseline
from project_code_intelligence.storage.schema import ensure_schema

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from project_code_intelligence.models import StaticFinding, StaticRun


@dataclass
class CheckIdentity:
    """(collection, repo, branch) for the repo `pci check` runs from -- same scheme as snapshots.

    `branch` is None for a detached HEAD, matching `resolve_repo_branch`/snapshot
    identity: there is no real branch name to collide with, and None is a
    distinct baseline bucket rather than a literal branch called "HEAD".
    """

    root: Path
    repo: str
    collection: str
    branch: str | None


def resolve_check_identity(cwd: Path, *, collection: str | None, repo: str | None) -> CheckIdentity:
    """Infer identity from the current checkout, the way single-path `pci index` does."""
    resolved = cwd.expanduser().resolve(strict=False)
    inferred_repo = resolved.name or "."
    return CheckIdentity(
        root=resolved,
        repo=repo or inferred_repo,
        collection=collection or default_collection(resolved),
        branch=resolve_repo_branch(resolved),
    )


def static_run_to_check_findings(repo_root: Path, run: StaticRun) -> list[CheckFinding]:
    return [static_finding_to_check_finding(repo_root, run, finding) for finding in run.findings]


def static_finding_to_check_finding(repo_root: Path, run: StaticRun, finding: StaticFinding) -> CheckFinding:
    return CheckFinding(
        fingerprint=finding_fingerprint(repo_root, finding),
        rule_id=finding.rule_id,
        level=finding.level,
        tool_name=run.tool_name,
        message=finding.message,
        primary_source_path=finding.primary_source_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
    )


def load_current_findings(identity: CheckIdentity, sarif_paths: list[Path]) -> tuple[list[CheckFinding], list[object]]:
    """Ingest `sarif_paths` and reduce every result to a deduplicated `CheckFinding` list."""
    context = SarifIngestContext(
        root=identity.root.parent,
        repos=[identity.repo],
        collection=identity.collection,
        file_by_source_path={},
        max_bytes=config.env_int("PCI_SARIF_MAX_BYTES", 50 * 1024 * 1024, minimum=0),
    )
    ingested = ingest_sarif(context, sarif_paths)
    findings = [
        check_finding for run in ingested.runs for check_finding in static_run_to_check_findings(identity.root, run)
    ]
    return disambiguate_occurrences(findings), list(ingested.failures)


def render_regressions(regressions: Sequence[Regression], rules: Mapping[str, RulepackRuleInfo] | None = None) -> str:
    lines = [regression_line(r, rules) for r in regressions]
    return "\n".join(lines) + "\n" if lines else ""


@dataclass
class CheckNamespace(argparse.Namespace):
    baseline: bool = False
    collection: str | None = None
    repo: str | None = None
    sarif_files: list[str] = field(default_factory=list)


def check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pci check",
        description=(
            "SARIF regression ratchet: ingest fresh SARIF and fail only on findings that are "
            "new or worsened since the branch's frozen baseline."
        ),
    )
    _ = parser.add_argument(
        "--baseline",
        action="store_true",
        help="Freeze the ingested finding set as the baseline for the current branch, instead of diffing.",
    )
    _ = parser.add_argument("--collection", help="Collection/workspace name. Defaults to the repo-inferred name.")
    _ = parser.add_argument("--repo", help="Repo name for baseline identity. Defaults to the current directory name.")
    _ = parser.add_argument("sarif_files", nargs="+", help="SARIF file(s) to ingest.")
    return parser


def check_main(argv: list[str] | None = None) -> int:
    parsed = check_parser().parse_args(argv, namespace=CheckNamespace())
    identity = resolve_check_identity(Path.cwd(), collection=parsed.collection, repo=parsed.repo)
    sarif_paths = [Path(p).expanduser().resolve(strict=False) for p in parsed.sarif_files]

    try:
        current, failures = load_current_findings(identity, sarif_paths)
    except SarifLoadError as exc:
        _ = sys.stderr.write(f"pci check: {exc}\n")
        return 1
    for failure in failures:
        _ = sys.stderr.write(f"pci check: {failure}\n")

    label = f"{identity.collection}/{identity.repo}@{branch_label(identity.branch)}"
    try:
        if parsed.baseline:
            os.environ["PCI_ALLOW_WRITES"] = "1"
            with db.connect(readonly=False) as conn:
                ensure_schema(conn)
                count = freeze_baseline(
                    conn,
                    collection=identity.collection,
                    repo=identity.repo,
                    branch=identity.branch,
                    findings=current,
                )
                conn.commit()
            _ = sys.stdout.write(f"pci check: froze baseline for {label}: {count} finding(s)\n")
            return 0

        with db.connect(readonly=True) as conn:
            baseline = load_baseline(conn, collection=identity.collection, repo=identity.repo, branch=identity.branch)
    except DatabaseConnectionError as exc:
        _ = sys.stderr.write(f"pci check: {exc}\n")
        return 1

    if baseline is None:
        _ = sys.stderr.write(f"pci check: no baseline for {label}; run `pci check --baseline` first\n")
        return 1

    regressions = diff_against_baseline(baseline, current)
    if not regressions:
        _ = sys.stdout.write(f"pci check: {label}: no new or worsened findings ({len(current)} total)\n")
        return 0
    rules = rule_lookup(discover_rulepacks(identity.root).packs)
    _ = sys.stdout.write(f"pci check: {label}: {len(regressions)} new/worsened finding(s)\n")
    _ = sys.stdout.write(render_regressions(regressions, rules))
    return 1


if __name__ == "__main__":
    raise SystemExit(check_main())
