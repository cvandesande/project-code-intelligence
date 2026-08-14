import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from project_code_intelligence.models import JsonObject, StaticFinding
from project_code_intelligence.sarif import fingerprint as fp


class LevelRankTests(unittest.TestCase):
    def test_order_is_none_note_warning_error(self) -> None:
        self.assertLess(fp.level_rank("none"), fp.level_rank("note"))
        self.assertLess(fp.level_rank("note"), fp.level_rank("warning"))
        self.assertLess(fp.level_rank("warning"), fp.level_rank("error"))

    def test_missing_or_unknown_level_is_none_rank(self) -> None:
        self.assertEqual(fp.level_rank(None), fp.level_rank("none"))
        self.assertEqual(fp.level_rank("bogus"), fp.level_rank("none"))


class IsWorsenedTests(unittest.TestCase):
    def test_escalation_is_worsened(self) -> None:
        self.assertTrue(fp.is_worsened("note", "warning"))
        self.assertTrue(fp.is_worsened("warning", "error"))
        self.assertTrue(fp.is_worsened(None, "error"))

    def test_same_or_lower_level_is_not_worsened(self) -> None:
        self.assertFalse(fp.is_worsened("warning", "warning"))
        self.assertFalse(fp.is_worsened("error", "warning"))
        self.assertFalse(fp.is_worsened("error", None))


class PartialFingerprintTests(unittest.TestCase):
    def test_empty_returns_none(self) -> None:
        self.assertIsNone(fp.partial_fingerprint({}))

    def test_stable_across_key_order(self) -> None:
        first = fp.partial_fingerprint({"a": "1", "b": "2"})
        second = fp.partial_fingerprint({"b": "2", "a": "1"})
        self.assertEqual(first, second)

    def test_differs_on_value_change(self) -> None:
        first = fp.partial_fingerprint({"primaryLocationLineHash": "abc"})
        second = fp.partial_fingerprint({"primaryLocationLineHash": "def"})
        self.assertNotEqual(first, second)


class NormalizedCodeContextTests(unittest.TestCase):
    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(fp.normalized_code_context(Path("/nonexistent"), "missing.py", 3), "")

    def test_missing_inputs_return_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(fp.normalized_code_context(root, None, 3), "")
            self.assertEqual(fp.normalized_code_context(root, "a.py", None), "")

    def test_line_drift_within_window_keeps_same_context(self) -> None:
        """Inserting a blank line above the finding shifts its line number but the
        stripped +/- CONTEXT_LINES window around it is unchanged (drift tolerance)."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _ = (root / "a.py").write_text("def f():\n    x = 1\n    bad_call()\n    y = 2\n", encoding="utf-8")
            before = fp.normalized_code_context(root, "a.py", 3)

            _ = (root / "a.py").write_text(
                "\ndef f():\n    x = 1\n    bad_call()\n    y = 2\n",
                encoding="utf-8",
            )
            after = fp.normalized_code_context(root, "a.py", 4)
            self.assertEqual(before, after)

    def test_different_neighborhood_changes_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _ = (root / "a.py").write_text("bad_call()\n", encoding="utf-8")
            _ = (root / "b.py").write_text("something_else()\n", encoding="utf-8")
            self.assertNotEqual(
                fp.normalized_code_context(root, "a.py", 1),
                fp.normalized_code_context(root, "b.py", 1),
            )

    def test_absolute_source_path_is_treated_as_unreadable(self) -> None:
        """Path("/root") / "/abs" discards "/root" entirely (Path semantics), which would
        read whatever sits at the absolute path outside the repo rather than failing."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = Path(tmp) / "outside.py"
            _ = outside.write_text("secret_stuff()\n", encoding="utf-8")
            self.assertEqual(fp.normalized_code_context(root, str(outside), 1), "")


class OwnFingerprintTests(unittest.TestCase):
    def test_falls_back_to_rule_and_path_when_unreadable(self) -> None:
        first = fp.own_fingerprint(
            Path("/nonexistent"),
            "RULE1",
            fp.FindingLocation(source_path="missing.py", line_start=1, column_start=None),
            "bad thing",
        )
        second = fp.own_fingerprint(
            Path("/nonexistent"),
            "RULE1",
            fp.FindingLocation(source_path="missing.py", line_start=99, column_start=None),
            "bad thing",
        )
        self.assertEqual(first, second)

    def test_different_rule_id_differs(self) -> None:
        first = fp.own_fingerprint(
            Path("/nonexistent"),
            "RULE1",
            fp.FindingLocation(source_path="missing.py", line_start=1, column_start=None),
            "bad thing",
        )
        second = fp.own_fingerprint(
            Path("/nonexistent"),
            "RULE2",
            fp.FindingLocation(source_path="missing.py", line_start=1, column_start=None),
            "bad thing",
        )
        self.assertNotEqual(first, second)

    def test_different_column_on_same_line_differs(self) -> None:
        """Two distinct findings of the same rule on the same line (identical context
        window) must not collapse to one fingerprint -- column disambiguates them."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _ = (root / "a.py").write_text("bad_call(); bad_call()\n", encoding="utf-8")
            first = fp.own_fingerprint(
                root, "RULE1", fp.FindingLocation(source_path="a.py", line_start=1, column_start=1), "bad thing"
            )
            second = fp.own_fingerprint(
                root, "RULE1", fp.FindingLocation(source_path="a.py", line_start=1, column_start=14), "bad thing"
            )
            self.assertNotEqual(first, second)

    def test_missing_column_still_produces_a_fingerprint(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _ = (root / "a.py").write_text("bad_call()\n", encoding="utf-8")
            self.assertTrue(
                fp.own_fingerprint(
                    root, "RULE1", fp.FindingLocation(source_path="a.py", line_start=1, column_start=None), "bad thing"
                )
            )

    def test_no_readable_location_uses_message_to_disambiguate(self) -> None:
        """No file, no context: two findings of the same rule with different messages
        must not collapse to one identity."""
        first = fp.own_fingerprint(
            Path("/nonexistent"),
            "RULE1",
            fp.FindingLocation(source_path=None, line_start=None, column_start=None),
            "first problem",
        )
        second = fp.own_fingerprint(
            Path("/nonexistent"),
            "RULE1",
            fp.FindingLocation(source_path=None, line_start=None, column_start=None),
            "second problem",
        )
        self.assertNotEqual(first, second)


def _finding(
    fingerprints: JsonObject,
    line_start: int | None = 3,
    column_start: int | None = None,
) -> StaticFinding:
    return StaticFinding(
        finding_key="k",
        rule_id="RULE1",
        message="bad thing",
        level="warning",
        primary_source_path="a.py",
        line_start=line_start,
        column_start=column_start,
        fingerprints=fingerprints,
    )


class FindingFingerprintTests(unittest.TestCase):
    def test_prefers_partial_fingerprints_when_present(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _ = (root / "a.py").write_text("bad_call()\n", encoding="utf-8")
            finding = _finding(fingerprints={"primaryLocationLineHash": "abc123"})
            self.assertEqual(fp.finding_fingerprint(root, finding), fp.partial_fingerprint(finding.fingerprints))

    def test_falls_back_to_own_fingerprint_when_absent(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _ = (root / "a.py").write_text("x = 1\nbad_call()\ny = 2\n", encoding="utf-8")
            finding = _finding(fingerprints={}, line_start=2)
            expected = fp.own_fingerprint(
                root,
                "RULE1",
                fp.FindingLocation(source_path="a.py", line_start=2, column_start=finding.column_start),
                finding.message,
            )
            self.assertEqual(fp.finding_fingerprint(root, finding), expected)

    def test_two_findings_same_rule_same_line_different_column_get_distinct_fingerprints(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _ = (root / "a.py").write_text("bad_call(); bad_call()\n", encoding="utf-8")
            first = _finding(fingerprints={}, line_start=1, column_start=1)
            second = _finding(fingerprints={}, line_start=1, column_start=14)
            self.assertNotEqual(fp.finding_fingerprint(root, first), fp.finding_fingerprint(root, second))


if __name__ == "__main__":
    _ = unittest.main()
