"""Rulepack loading: `.pci/rulepacks/<name>/` in a target repo, as data.

A rulepack declares an enforcement ruleset without running or bundling any
producer. Tier 1 (mechanical) and Tier 2 (metric-gateable) rules point at an
external producer's config (an ast-grep YAML path, a metric name + threshold);
PCI stores/points at that config, it never executes it. Tier 3 (LLM-judge)
rules carry no producer -- their rationale lives in the pack's rubric file,
consumed by a host agent (Phase 4), not by PCI itself.

Manifests are JSON: this repo has no TOML/YAML parser as a runtime dependency
(`tomli` is dev-only, and stdlib `tomllib` is 3.11+ while this project
supports 3.10), and JSON needs neither -- consistent with the SARIF ingest
path, which is JSON throughout.

This module only discovers and validates packs. Nothing here enforces
anything; `pci check`/`pci audit --gate` do the enforcing, this just tells
them which rule IDs are known and what tier/rationale to attach.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from project_code_intelligence.sarif.parse import json_array, json_object

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

MANIFEST_NAME = "rulepack.json"
RUBRIC_NAME = "rubric.json"
_TIER_JUDGE = 3
_VALID_TIERS = (1, 2, _TIER_JUDGE)


@dataclass(frozen=True)
class Producer:
    """Where a Tier 1/2 rule's enforcement config lives. PCI never runs it."""

    kind: str  # e.g. "ast_grep", "metric"
    path: str | None  # producer config path, relative to the pack directory
    tool: str | None  # e.g. "bca", for metric producers
    threshold: float | None  # metric producers only


@dataclass(frozen=True)
class Rule:
    id: str
    tier: int
    description: str
    rationale: str
    producer: Producer | None


@dataclass(frozen=True)
class RulePack:
    name: str
    version: str
    path: Path  # the pack directory
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class RulePackLoadError:
    """A pack directory that could not be parsed at all (bad JSON, wrong shape)."""

    path: Path
    reason: str


@dataclass(frozen=True)
class DiscoveryResult:
    packs: tuple[RulePack, ...]
    errors: tuple[RulePackLoadError, ...]


def _parse_producer(raw_value: object, rule_id: str) -> Producer:
    if not isinstance(raw_value, dict):
        raise TypeError(f"rule {rule_id!r}: producer must be an object")
    raw = json_object(cast("object", raw_value))
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind:
        raise TypeError(f"rule {rule_id!r}: producer.kind must be a non-empty string")
    path = raw.get("path")
    if path is not None and not isinstance(path, str):
        raise TypeError(f"rule {rule_id!r}: producer.path must be a string")
    tool = raw.get("tool")
    if tool is not None and not isinstance(tool, str):
        raise TypeError(f"rule {rule_id!r}: producer.tool must be a string")
    threshold = raw.get("threshold")
    if threshold is not None and not isinstance(threshold, int | float):
        raise TypeError(f"rule {rule_id!r}: producer.threshold must be a number")
    return Producer(
        kind=kind,
        path=path,
        tool=tool,
        threshold=float(threshold) if threshold is not None else None,
    )


def _parse_rule(raw_value: object) -> Rule:
    if not isinstance(raw_value, dict):
        raise TypeError("each rule entry must be an object")
    raw = json_object(cast("object", raw_value))
    rule_id = raw.get("id")
    if not isinstance(rule_id, str) or not rule_id:
        raise ValueError("rule entry missing a non-empty 'id'")
    tier = raw.get("tier")
    if not isinstance(tier, int) or isinstance(tier, bool):
        raise TypeError(f"rule {rule_id!r}: 'tier' must be an integer")
    description = raw.get("description")
    if not isinstance(description, str) or not description:
        raise ValueError(f"rule {rule_id!r}: missing non-empty 'description'")
    rationale = raw.get("rationale")
    if not isinstance(rationale, str) or not rationale:
        raise ValueError(f"rule {rule_id!r}: missing non-empty 'rationale'")
    producer_raw = raw.get("producer")
    producer = _parse_producer(producer_raw, rule_id) if producer_raw is not None else None
    return Rule(id=rule_id, tier=tier, description=description, rationale=rationale, producer=producer)


def _parse_manifest(pack_dir: Path) -> RulePack:
    manifest_path = pack_dir / MANIFEST_NAME
    raw_data = cast("object", json.loads(manifest_path.read_text(encoding="utf-8")))
    if not isinstance(raw_data, dict):
        raise TypeError("manifest must be a JSON object")
    data = json_object(cast("object", raw_data))
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("manifest missing non-empty 'name'")
    version = data.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("manifest missing non-empty 'version'")
    raw_rules = data.get("rules")
    if raw_rules is not None and not isinstance(raw_rules, list):
        raise TypeError("manifest 'rules' must be an array of rule objects")
    rules = tuple(_parse_rule(raw) for raw in json_array(raw_rules))
    return RulePack(name=name, version=version, path=pack_dir, rules=rules)


def discover_rulepacks(repo_root: Path) -> DiscoveryResult:
    """Every `.pci/rulepacks/<name>/` directory under `repo_root`, parsed.

    A directory without a `rulepack.json`, or with one that fails to parse,
    becomes a `RulePackLoadError` rather than aborting the whole discovery --
    one broken pack should not hide the others.
    """
    rulepacks_dir = repo_root / ".pci" / "rulepacks"
    if not rulepacks_dir.is_dir():
        return DiscoveryResult(packs=(), errors=())
    packs: list[RulePack] = []
    errors: list[RulePackLoadError] = []
    for pack_dir in sorted(p for p in rulepacks_dir.iterdir() if p.is_dir()):
        manifest_path = pack_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            errors.append(RulePackLoadError(path=pack_dir, reason=f"missing {MANIFEST_NAME}"))
            continue
        try:
            packs.append(_parse_manifest(pack_dir))
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
            errors.append(RulePackLoadError(path=pack_dir, reason=str(exc)))
    return DiscoveryResult(packs=tuple(packs), errors=tuple(errors))


@dataclass(frozen=True)
class ValidationIssue:
    """One actionable validation problem: which file, which field, why."""

    file: str
    field: str
    reason: str
    severity: str = "error"  # "error" | "warning"

    def render(self) -> str:
        return f"{self.severity}: {self.file}: {self.field}: {self.reason}"


def _rubric_ids(pack: RulePack) -> set[str] | None:
    """Rule IDs declared in the pack's rubric file, or None if the file is absent/unreadable."""
    rubric_path = pack.path / RUBRIC_NAME
    if not rubric_path.is_file():
        return None
    try:
        data = json_object(cast("object", json.loads(rubric_path.read_text(encoding="utf-8"))))
    except (json.JSONDecodeError, OSError):
        return None
    ids: set[str] = set()
    for entry_value in json_array(data.get("rubric")):
        entry = json_object(entry_value)
        entry_id = entry.get("id")
        if isinstance(entry_id, str):
            ids.add(entry_id)
    return ids


def _duplicate_id_issues(pack: RulePack, manifest_file: str) -> list[ValidationIssue]:
    seen_ids: dict[str, int] = {}
    for rule in pack.rules:
        seen_ids[rule.id] = seen_ids.get(rule.id, 0) + 1
    return [
        ValidationIssue(
            file=manifest_file,
            field=f"rule[id={rule_id}]",
            reason=f"duplicate rule id ({count} entries) within pack {pack.name!r}",
        )
        for rule_id, count in seen_ids.items()
        if count > 1
    ]


def _tier3_rubric_issue(pack: RulePack, rule: Rule, rubric_ids: set[str] | None) -> ValidationIssue | None:
    if rubric_ids is None:
        return ValidationIssue(
            file=str(pack.path / RUBRIC_NAME),
            field=f"rule[id={rule.id}]",
            reason="tier-3 rule has no readable rubric.json in the pack directory",
        )
    if rule.id not in rubric_ids:
        return ValidationIssue(
            file=str(pack.path / RUBRIC_NAME),
            field=f"rubric[id={rule.id}]",
            reason="tier-3 rule has no matching rubric entry",
        )
    return None


def _missing_producer_issue(rule: Rule, manifest_file: str) -> ValidationIssue | None:
    if rule.producer is not None:
        return None
    return ValidationIssue(
        file=manifest_file,
        field=f"rule[id={rule.id}].producer",
        reason=f"tier-{rule.tier} rule has no producer config; it would be enforced nowhere",
    )


def _producer_path_issue(pack: RulePack, rule: Rule, manifest_file: str) -> ValidationIssue | None:
    if rule.producer is None or rule.producer.path is None:
        return None
    producer_path = pack.path / rule.producer.path
    if producer_path.is_file():
        return None
    return ValidationIssue(
        file=manifest_file,
        field=f"rule[id={rule.id}].producer.path",
        reason=f"producer config {rule.producer.path!r} does not exist under {pack.path}",
    )


def _rule_issue(pack: RulePack, rule: Rule, manifest_file: str, rubric_ids: set[str] | None) -> ValidationIssue | None:
    if rule.tier not in _VALID_TIERS:
        return ValidationIssue(
            file=manifest_file,
            field=f"rule[id={rule.id}].tier",
            reason=f"unknown tier {rule.tier!r} (must be 1, 2, or {_TIER_JUDGE})",
        )
    if rule.tier == _TIER_JUDGE:
        return _tier3_rubric_issue(pack, rule, rubric_ids)
    return _missing_producer_issue(rule, manifest_file) or _producer_path_issue(pack, rule, manifest_file)


def validate_rulepack(pack: RulePack) -> list[ValidationIssue]:
    """Actionable validation errors/warnings for one already-parsed pack.

    Checks: unknown tier, duplicate rule IDs within the pack, a missing rubric
    file (or a missing rubric entry) for any Tier 3 rule, a missing producer
    block for any Tier 1/2 rule (Tier 3 rules legitimately have none -- they
    use the rubric instead), and a dangling producer-config path for any
    Tier 1/2 rule that does have one.
    """
    manifest_file = str(pack.path / MANIFEST_NAME)
    issues = _duplicate_id_issues(pack, manifest_file)
    has_tier3 = any(rule.tier == _TIER_JUDGE for rule in pack.rules)
    rubric_ids = _rubric_ids(pack) if has_tier3 else None
    for rule in pack.rules:
        issue = _rule_issue(pack, rule, manifest_file, rubric_ids)
        if issue is not None:
            issues.append(issue)
    return issues


@dataclass(frozen=True)
class RulepackRuleInfo:
    """The bit of a rulepack rule worth attaching to a `pci check` finding: tier + why."""

    tier: int
    rationale: str


def rule_lookup(packs: Sequence[RulePack]) -> dict[str, RulepackRuleInfo]:
    """rule ID -> (tier, rationale), for enriching `pci check`/`pci audit --gate` output.

    Last pack wins on a rule ID collision across packs (an ordering choice, not
    a correctness one -- `validate_rulepacks` already warns on the collision).
    """
    lookup: dict[str, RulepackRuleInfo] = {}
    for pack in packs:
        for rule in pack.rules:
            lookup[rule.id] = RulepackRuleInfo(tier=rule.tier, rationale=rule.rationale)
    return lookup


def validate_rulepacks(packs: Sequence[RulePack]) -> list[ValidationIssue]:
    """Per-pack validation plus a cross-pack duplicate-ID warning (non-fatal).

    A rule ID reused across packs is not itself invalid -- packs are
    independent, standalone units -- but it is worth flagging since a caller
    that loads multiple packs together may resolve it ambiguously.
    """
    issues: list[ValidationIssue] = []
    for pack in packs:
        issues.extend(validate_rulepack(pack))
    id_to_packs: dict[str, list[str]] = {}
    for pack in packs:
        for rule in pack.rules:
            id_to_packs.setdefault(rule.id, []).append(pack.name)
    for rule_id, pack_names in id_to_packs.items():
        if len(pack_names) > 1:
            issues.append(
                ValidationIssue(
                    file="<multiple packs>",
                    field=f"rule[id={rule_id}]",
                    reason=f"rule id reused across packs: {', '.join(sorted(pack_names))}",
                    severity="warning",
                )
            )
    return issues
