import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from project_code_intelligence import rulepacks as rp


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(content, encoding="utf-8")


def _manifest(*, extra_rules: list[dict[str, object]] | None = None) -> str:
    rules: list[dict[str, object]] = [
        {
            "id": "T1",
            "tier": 1,
            "description": "desc",
            "rationale": "why",
            "producer": {"kind": "ast_grep", "path": "grep.yml"},
        },
        *(extra_rules or []),
    ]
    return json.dumps({"name": "sample", "version": "1.0.0", "rules": rules})


class DiscoverRulepacksTests(unittest.TestCase):
    def test_no_pci_dir_returns_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            result = rp.discover_rulepacks(Path(tmp))
        self.assertEqual(result.packs, ())
        self.assertEqual(result.errors, ())

    def test_empty_rulepacks_dir_returns_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".pci" / "rulepacks").mkdir(parents=True)
            result = rp.discover_rulepacks(root)
        self.assertEqual(result.packs, ())
        self.assertEqual(result.errors, ())

    def test_discovers_multiple_packs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / ".pci/rulepacks/a/rulepack.json", _manifest())
            _write(root / ".pci/rulepacks/a/grep.yml", "id: x\n")
            _write(root / ".pci/rulepacks/b/rulepack.json", _manifest())
            _write(root / ".pci/rulepacks/b/grep.yml", "id: x\n")
            result = rp.discover_rulepacks(root)
        self.assertEqual({p.name for p in result.packs}, {"sample"})
        self.assertEqual(len(result.packs), 2)
        self.assertEqual(result.errors, ())

    def test_missing_manifest_is_a_load_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".pci/rulepacks/broken").mkdir(parents=True)
            result = rp.discover_rulepacks(root)
        self.assertEqual(result.packs, ())
        self.assertEqual(len(result.errors), 1)
        self.assertIn("rulepack.json", result.errors[0].reason)

    def test_malformed_json_is_a_load_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / ".pci/rulepacks/broken/rulepack.json", "{not valid json")
            result = rp.discover_rulepacks(root)
        self.assertEqual(result.packs, ())
        self.assertEqual(len(result.errors), 1)

    def test_non_object_manifest_reports_generic_shape_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / ".pci/rulepacks/broken/rulepack.json", json.dumps(["not", "a", "dict"]))
            result = rp.discover_rulepacks(root)
        self.assertEqual(result.packs, ())
        self.assertEqual(len(result.errors), 1)
        self.assertIn("must be a JSON object", result.errors[0].reason)

    def test_empty_object_manifest_names_missing_field(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / ".pci/rulepacks/broken/rulepack.json", "{}")
            result = rp.discover_rulepacks(root)
        self.assertEqual(result.packs, ())
        self.assertEqual(len(result.errors), 1)
        self.assertIn("'name'", result.errors[0].reason)
        self.assertNotIn("must be a JSON object", result.errors[0].reason)


class ValidateRulepackTests(unittest.TestCase):
    def test_valid_pack_has_no_issues(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / ".pci/rulepacks/a/rulepack.json", _manifest())
            _write(root / ".pci/rulepacks/a/grep.yml", "id: x\n")
            result = rp.discover_rulepacks(root)
            issues = rp.validate_rulepacks(result.packs)
        self.assertEqual(issues, [])

    def test_unknown_tier_is_an_error(self) -> None:
        pack = rp.RulePack(
            name="p",
            version="1",
            path=Path("/nonexistent"),
            rules=(rp.Rule(id="X1", tier=9, description="d", rationale="r", producer=None),),
        )
        issues = rp.validate_rulepack(pack)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "error")
        self.assertIn("unknown tier", issues[0].reason)

    def test_duplicate_rule_id_within_pack_is_an_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _manifest(
                extra_rules=[
                    {
                        "id": "T1",
                        "tier": 1,
                        "description": "d2",
                        "rationale": "r2",
                        "producer": {"kind": "ast_grep", "path": "grep.yml"},
                    }
                ]
            )
            _write(root / ".pci/rulepacks/a/rulepack.json", manifest)
            _write(root / ".pci/rulepacks/a/grep.yml", "id: x\n")
            result = rp.discover_rulepacks(root)
            issues = rp.validate_rulepack(result.packs[0])
        self.assertEqual(len(issues), 1)
        self.assertIn("duplicate rule id", issues[0].reason)

    def test_missing_rubric_file_for_tier3_rule_is_an_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = json.dumps({
                "name": "p",
                "version": "1",
                "rules": [{"id": "N1", "tier": 3, "description": "d", "rationale": "r"}],
            })
            _write(root / ".pci/rulepacks/a/rulepack.json", manifest)
            result = rp.discover_rulepacks(root)
            issues = rp.validate_rulepack(result.packs[0])
        self.assertEqual(len(issues), 1)
        self.assertIn("rubric.json", issues[0].reason)

    def test_tier3_rule_with_no_matching_rubric_entry_is_an_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = json.dumps({
                "name": "p",
                "version": "1",
                "rules": [{"id": "N1", "tier": 3, "description": "d", "rationale": "r"}],
            })
            _write(root / ".pci/rulepacks/a/rulepack.json", manifest)
            _write(root / ".pci/rulepacks/a/rubric.json", json.dumps({"rubric": [{"id": "N2", "text": "unrelated"}]}))
            result = rp.discover_rulepacks(root)
            issues = rp.validate_rulepack(result.packs[0])
        self.assertEqual(len(issues), 1)
        self.assertIn("no matching rubric entry", issues[0].reason)

    def test_tier3_rule_with_matching_rubric_entry_is_valid(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = json.dumps({
                "name": "p",
                "version": "1",
                "rules": [{"id": "N1", "tier": 3, "description": "d", "rationale": "r"}],
            })
            _write(root / ".pci/rulepacks/a/rulepack.json", manifest)
            _write(root / ".pci/rulepacks/a/rubric.json", json.dumps({"rubric": [{"id": "N1", "text": "judge this"}]}))
            result = rp.discover_rulepacks(root)
            issues = rp.validate_rulepack(result.packs[0])
        self.assertEqual(issues, [])

    def test_dangling_producer_path_is_an_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / ".pci/rulepacks/a/rulepack.json", _manifest())
            # Deliberately do not write grep.yml.
            result = rp.discover_rulepacks(root)
            issues = rp.validate_rulepack(result.packs[0])
        self.assertEqual(len(issues), 1)
        self.assertIn("does not exist", issues[0].reason)

    def test_tier1_rule_with_no_producer_is_an_error(self) -> None:
        pack = rp.RulePack(
            name="p",
            version="1",
            path=Path("/nonexistent"),
            rules=(rp.Rule(id="T1", tier=1, description="d", rationale="r", producer=None),),
        )
        issues = rp.validate_rulepack(pack)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "error")
        self.assertIn("T1", issues[0].field)
        self.assertIn("no producer config", issues[0].reason)

    def test_tier2_rule_with_no_producer_is_an_error(self) -> None:
        pack = rp.RulePack(
            name="p",
            version="1",
            path=Path("/nonexistent"),
            rules=(rp.Rule(id="T2", tier=2, description="d", rationale="r", producer=None),),
        )
        issues = rp.validate_rulepack(pack)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "error")
        self.assertIn("T2", issues[0].field)
        self.assertIn("no producer config", issues[0].reason)

    def test_cross_pack_duplicate_id_is_a_warning_not_an_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / ".pci/rulepacks/a/rulepack.json", _manifest())
            _write(root / ".pci/rulepacks/a/grep.yml", "id: x\n")
            manifest_b = _manifest().replace('"sample"', '"other"')
            _write(root / ".pci/rulepacks/b/rulepack.json", manifest_b)
            _write(root / ".pci/rulepacks/b/grep.yml", "id: x\n")
            result = rp.discover_rulepacks(root)
            issues = rp.validate_rulepacks(result.packs)
        warnings = [i for i in issues if i.severity == "warning"]
        errors = [i for i in issues if i.severity == "error"]
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("reused across packs", warnings[0].reason)


class RuleLookupTests(unittest.TestCase):
    def test_builds_id_to_tier_rationale_map(self) -> None:
        pack = rp.RulePack(
            name="p",
            version="1",
            path=Path("/nonexistent"),
            rules=(rp.Rule(id="X1", tier=2, description="d", rationale="because", producer=None),),
        )
        lookup = rp.rule_lookup([pack])
        self.assertEqual(lookup["X1"], rp.RulepackRuleInfo(tier=2, rationale="because"))
        self.assertNotIn("X2", lookup)


if __name__ == "__main__":
    _ = unittest.main()
