import unittest

from project_code_intelligence import check_core as cc


def _finding(fingerprint: str, level: str | None = "warning", message: str = "bad thing") -> cc.CheckFinding:
    return cc.CheckFinding(
        fingerprint=fingerprint,
        rule_id="RULE1",
        level=level,
        tool_name="tool",
        message=message,
        primary_source_path="a.py",
        line_start=3,
        line_end=3,
    )


class DisambiguateOccurrencesTests(unittest.TestCase):
    def test_distinct_base_fingerprints_pass_through_unchanged_count(self) -> None:
        findings = [_finding("fp1"), _finding("fp2")]
        out = cc.disambiguate_occurrences(findings)
        self.assertEqual(len(out), 2)
        self.assertEqual(len({f.fingerprint for f in out}), 2)

    def test_same_base_fingerprint_gets_distinct_final_fingerprints(self) -> None:
        """Two occurrences that share a base fingerprint (e.g. an unresolved location
        collision) must not collapse to one entry -- occurrence count itself matters."""
        findings = [
            _finding("fp1", message="first"),
            _finding("fp1", message="second"),
        ]
        out = cc.disambiguate_occurrences(findings)
        self.assertEqual(len(out), 2)
        self.assertEqual(len({f.fingerprint for f in out}), 2)

    def test_deterministic_regardless_of_input_order(self) -> None:
        a = _finding("fp1", message="first")
        b = _finding("fp1", message="second")
        out_ab = cc.disambiguate_occurrences([a, b])
        out_ba = cc.disambiguate_occurrences([b, a])
        self.assertEqual({f.fingerprint for f in out_ab}, {f.fingerprint for f in out_ba})

    def test_occurrence_count_increase_is_a_regression(self) -> None:
        """N baselined occurrences of the same base fingerprint vs N+1 fresh ones must
        register as a new finding, not disappear into a fingerprint collision."""
        baseline_findings = cc.disambiguate_occurrences([
            _finding("fp1", message="dup"),
            _finding("fp1", message="dup"),
        ])
        baseline = [
            cc.BaselineEntry(fingerprint=f.fingerprint, rule_id=f.rule_id, level=f.level) for f in baseline_findings
        ]

        current_findings = cc.disambiguate_occurrences([
            _finding("fp1", message="dup"),
            _finding("fp1", message="dup"),
            _finding("fp1", message="dup"),
        ])
        regressions = cc.diff_against_baseline(baseline, current_findings)
        self.assertEqual(len(regressions), 1)
        self.assertEqual(regressions[0].status, "new")

    def test_occurrence_count_decrease_is_not_reported(self) -> None:
        baseline_findings = cc.disambiguate_occurrences([
            _finding("fp1", message="dup"),
            _finding("fp1", message="dup"),
            _finding("fp1", message="dup"),
        ])
        baseline = [
            cc.BaselineEntry(fingerprint=f.fingerprint, rule_id=f.rule_id, level=f.level) for f in baseline_findings
        ]

        current_findings = cc.disambiguate_occurrences([_finding("fp1", message="dup"), _finding("fp1", message="dup")])
        self.assertEqual(cc.diff_against_baseline(baseline, current_findings), [])


class DiffAgainstBaselineTests(unittest.TestCase):
    def test_finding_not_in_baseline_is_new(self) -> None:
        baseline: list[cc.BaselineEntry] = []
        current = [_finding("fp1")]
        regressions = cc.diff_against_baseline(baseline, current)
        self.assertEqual(len(regressions), 1)
        self.assertEqual(regressions[0].status, "new")

    def test_same_level_is_not_a_regression(self) -> None:
        baseline = [cc.BaselineEntry(fingerprint="fp1", rule_id="RULE1", level="warning")]
        current = [_finding("fp1", level="warning")]
        self.assertEqual(cc.diff_against_baseline(baseline, current), [])

    def test_escalated_level_is_worsened(self) -> None:
        baseline = [cc.BaselineEntry(fingerprint="fp1", rule_id="RULE1", level="note")]
        current = [_finding("fp1", level="error")]
        regressions = cc.diff_against_baseline(baseline, current)
        self.assertEqual(len(regressions), 1)
        self.assertEqual(regressions[0].status, "worsened")
        self.assertEqual(regressions[0].baseline_level, "note")

    def test_deescalated_level_is_not_a_regression(self) -> None:
        baseline = [cc.BaselineEntry(fingerprint="fp1", rule_id="RULE1", level="error")]
        current = [_finding("fp1", level="note")]
        self.assertEqual(cc.diff_against_baseline(baseline, current), [])

    def test_fixed_finding_absent_from_current_is_not_reported(self) -> None:
        baseline = [cc.BaselineEntry(fingerprint="fp1", rule_id="RULE1", level="error")]
        current: list[cc.CheckFinding] = []
        self.assertEqual(cc.diff_against_baseline(baseline, current), [])


class BranchLabelTests(unittest.TestCase):
    def test_named_branch_passes_through(self) -> None:
        self.assertEqual(cc.branch_label("main"), "main")

    def test_none_is_labeled_detached(self) -> None:
        self.assertEqual(cc.branch_label(None), "(detached HEAD)")


class RegressionLineTests(unittest.TestCase):
    def test_new_finding_line(self) -> None:
        regression = cc.Regression(status="new", finding=_finding("fp1", level="error"))
        line = cc.regression_line(regression)
        self.assertIn("NEW", line)
        self.assertIn("RULE1", line)
        self.assertIn("a.py:3", line)

    def test_worsened_finding_line_shows_old_level(self) -> None:
        regression = cc.Regression(status="worsened", finding=_finding("fp1", level="error"), baseline_level="note")
        line = cc.regression_line(regression)
        self.assertIn("WORSENED", line)
        self.assertIn("was note", line)


if __name__ == "__main__":
    _ = unittest.main()
