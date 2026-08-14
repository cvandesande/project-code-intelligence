"""``pci rulepack``: discover and validate `.pci/rulepacks/<name>/` in a repo.

Read-only and database-free: rulepacks are files in the target repo, not
indexed state. `list` reports what is there; `validate` is the CI-facing
check, exiting 1 on any actionable error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from project_code_intelligence.rulepacks import DiscoveryResult, RulePack, discover_rulepacks, validate_rulepacks

_TIERS = (1, 2, 3)


class RulepackNamespace(argparse.Namespace):
    subcommand: str = ""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pci rulepack",
        description="Discover and validate `.pci/rulepacks/<name>/` under the current directory.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    _ = subparsers.add_parser("list", help="List discovered rulepacks and their rule counts per tier.")
    _ = subparsers.add_parser("validate", help="Validate discovered rulepacks; exit 1 on any error.")
    return parser


def _tier_counts(pack: RulePack) -> dict[int, int]:
    counts: dict[int, int] = dict.fromkeys(_TIERS, 0)
    for rule in pack.rules:
        if rule.tier in counts:
            counts[rule.tier] += 1
    return counts


def _render_list(discovery: DiscoveryResult) -> str:
    if not discovery.packs and not discovery.errors:
        return "pci rulepack: no rulepacks found under .pci/rulepacks/\n"
    lines: list[str] = []
    for pack in discovery.packs:
        counts = _tier_counts(pack)
        tier_text = ", ".join(f"tier {tier}: {counts[tier]}" for tier in _TIERS)
        lines.append(f"{pack.name} {pack.version}  ({len(pack.rules)} rule(s) -- {tier_text})")
    lines.extend(f"{error.path}: UNREADABLE -- {error.reason}" for error in discovery.errors)
    return "\n".join(lines) + "\n"


def _render_validate(discovery: DiscoveryResult) -> tuple[str, bool]:
    """(report text, has_error). Load errors and rule-level errors both fail the gate."""
    has_error = bool(discovery.errors)
    lines = [f"error: {error.path}: {MANIFEST_LOAD_FIELD}: {error.reason}" for error in discovery.errors]
    issues = validate_rulepacks(discovery.packs)
    for issue in issues:
        lines.append(issue.render())
        has_error = has_error or issue.severity == "error"
    if not discovery.packs and not discovery.errors:
        lines.append("no rulepacks found under .pci/rulepacks/")
    elif not has_error:
        lines.append(f"pci rulepack: {len(discovery.packs)} pack(s) valid")
    return "\n".join(lines) + "\n", has_error


MANIFEST_LOAD_FIELD = "manifest"


def rulepack_main(argv: list[str] | None = None) -> int:
    parsed = _parser().parse_args(argv, namespace=RulepackNamespace())
    discovery = discover_rulepacks(Path.cwd())
    if parsed.subcommand == "list":
        _ = sys.stdout.write(_render_list(discovery))
        return 0
    text, has_error = _render_validate(discovery)
    stream = sys.stderr if has_error else sys.stdout
    _ = stream.write(text)
    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(rulepack_main())
