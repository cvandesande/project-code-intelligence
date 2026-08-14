from __future__ import annotations

import json
import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast
from unittest import mock

from typing_extensions import override

from project_code_intelligence import analyze, audit, audit_triage, db
from project_code_intelligence.check_core import BaselineEntry

if TYPE_CHECKING:
    from collections.abc import Generator


def _group(symbols: list[str], avg_text: float | None, max_text: float | None) -> analyze.MotifGroup:
    members = tuple(
        analyze.FunctionNode(
            record_id=f"r-{symbol}",
            symbol=symbol,
            source_path=f"{symbol}.py",
            line_start=1,
            line_end=5,
            callee_roles=frozenset({"a", "b", "c"}),
        )
        for symbol in symbols
    )
    return analyze.build_group(members, avg_semantic=None, avg_text=avg_text, max_text=max_text)


class DuplicateNamesTests(unittest.TestCase):
    def test_bare_name_collision_across_files(self) -> None:
        total, dups = audit.duplicate_names([
            ("main", "a.py"),
            ("cli.main", "b.py"),
            ("unique", "a.py"),
            ("Klass.method", "a.py"),
            ("method", "a.py"),  # same bare name, same file: not a multi-file dup
        ])
        self.assertEqual(total, 3)  # main, unique, method
        self.assertEqual([(d.name, d.paths) for d in dups], [("main", ("a.py", "b.py"))])

    def test_sorted_by_spread_then_name(self) -> None:
        _, dups = audit.duplicate_names([
            ("zz", "a.py"),
            ("zz", "b.py"),
            ("aa", "a.py"),
            ("aa", "b.py"),
            ("wide", "a.py"),
            ("wide", "b.py"),
            ("wide", "c.py"),
        ])
        self.assertEqual([d.name for d in dups], ["wide", "aa", "zz"])


class SplitGroupsTests(unittest.TestCase):
    def test_gate_uses_max_not_average(self) -> None:
        # A byte-identical pair inside a larger group: average diluted, max 1.0.
        diluted = _group(["a", "b", "c"], avg_text=0.97, max_text=1.0)
        # The blind set's highest false positive scored 0.9876 -- must stay a candidate.
        false_positive = _group(["d", "e"], avg_text=0.9876, max_text=0.9876)
        unknown = _group(["f", "g"], avg_text=None, max_text=None)
        near, rest = audit.split_groups([false_positive, diluted, unknown])
        self.assertEqual([g.members[0].symbol for g in near], ["a"])
        self.assertEqual([g.members[0].symbol for g in rest], ["d", "f"])


def _result() -> audit.AuditResult:
    return audit.AuditResult(
        label="c/r",
        collection="c",
        repo="r",
        snapshot_id=1,
        staleness=None,
        names_total=10,
        duplicate_names=(audit.DuplicateName(name="main", paths=("r/a.py", "r/b.py")),),
        redundancy=analyze.SnapshotResult(
            label="c/r",
            groups=(_group(["x", "y"], avg_text=1.0, max_text=1.0),),
            functions_analyzed=10,
            clones_folded=0,
        ),
        static_commit=None,
        static_counts=(),
    )


class RenderTests(unittest.TestCase):
    def test_render_text_smoke(self) -> None:
        text = audit.render_text([_result()])
        self.assertIn("UNRANKED", text)
        # "r/" repo prefix stripped: the header names the repo.
        self.assertIn("main  x2: a.py, b.py", text)
        self.assertIn("x  x.py:1", text)
        self.assertIn("no SARIF ingested", text)
        self.assertIn("reindex (pci-index)", text)

    def test_render_json_compact(self) -> None:
        rendered = audit.render_json([_result()])
        payload = cast("dict[str, dict[str, object]]", json.loads(rendered))["c/r"]
        dup_names = cast("dict[str, object]", payload["duplicate_names"])
        self.assertEqual(dup_names["items"], {"main": ["a.py", "b.py"]})
        redundancy = cast("dict[str, object]", payload["redundancy"])
        group = cast("list[object]", redundancy["near_certain"])[0]
        # Members collapse to one string each; scores with no measured signal are omitted.
        self.assertEqual(
            group,
            {
                "id": audit_triage.candidate_id(_result().redundancy.groups[0], "c", "r"),
                "max_text": 1.0,
                "members": ["x x.py:1-5", "y y.py:1-5"],
            },
        )
        self.assertNotIn("evidence", rendered)

    def test_line_style_smoke(self) -> None:
        self.assertEqual(audit.line_style("### Index staleness"), "bold cyan")
        self.assertEqual(audit.line_style("commit abc -- current, indexed 0h01m ago"), "green")
        self.assertEqual(audit.line_style("commit abc -- stale"), "red")
        self.assertIsNone(audit.line_style("    x  x.py:1"))


class ArgumentValidationTests(unittest.TestCase):
    def test_invalid_triage_flags_fail_before_database_access(self) -> None:
        with mock.patch.object(audit, "_load_audit_results") as load:
            self.assertEqual(audit.audit_main(["--candidate", "redundancy-example"]), 2)
        load.assert_not_called()

    def test_init_requires_explicit_full_triage(self) -> None:
        with mock.patch.object(audit, "_load_audit_results") as load:
            self.assertEqual(audit.audit_main(["--init-triage"]), 2)
        load.assert_not_called()

    def test_full_triage_selects_exhaustive_limit(self) -> None:
        def assert_limit(parsed: audit.AuditNamespace) -> int:
            self.assertEqual(parsed.limit, audit_triage.FULL_TRIAGE_LIMIT)
            return 1

        with mock.patch.object(audit, "_load_audit_results", side_effect=assert_limit):
            self.assertEqual(audit.audit_main(["--full-triage"]), 1)

    def test_default_audit_does_not_create_triage_directory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                _chdir(root),
                mock.patch.object(audit, "_load_audit_results", return_value=([_result()], [])),
                mock.patch.object(audit, "_render_audit_output"),
            ):
                self.assertEqual(audit.audit_main([]), 0)
            self.assertFalse((root / ".pci").exists())


@contextmanager
def _chdir(path: Path) -> Generator[None]:
    """Basedpyright's configured `pythonVersion = "3.10"` predates `contextlib.chdir` (3.11)."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _FakeCursor:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[dict[str, object]]:
        return self._rows


class _FakeGateConnection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def execute(self, _query: str, _params: list[object]) -> _FakeCursor:
        return _FakeCursor(self._rows)


def _finding_row(*, rule_id: str, level: str, line: int, source_path: str = "a.py") -> dict[str, object]:
    return {
        "tool_name": "demo-tool",
        "rule_id": rule_id,
        "level": level,
        "message": "bad thing",
        "primary_source_path": source_path,
        "line_start": line,
        "line_end": line,
        "fingerprints": {},
    }


class GateSnapshotTests(unittest.TestCase):
    """`gate_snapshot`'s exit-code-relevant fields (`regressions`) must not depend on
    whether `.pci/rulepacks/` exists -- rulepack enrichment only annotates lines."""

    @override
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name) / "repo"
        self.repo.mkdir()
        _ = (self.repo / "a.py").write_text("x = 1\nbad_call()\ny = 2\n", encoding="utf-8")

    @staticmethod
    def _snapshot() -> analyze.SnapshotRef:
        return analyze.SnapshotRef(snapshot_id=1, collection="c", repo=".", branch="main")

    def _write_rulepack(self, *, rule_id: str, tier: int, rationale: str) -> None:
        pack_dir = self.repo / ".pci" / "rulepacks" / "a"
        pack_dir.mkdir(parents=True)
        _ = (pack_dir / "rulepack.json").write_text(
            json.dumps({
                "name": "sample",
                "version": "1.0.0",
                "rules": [
                    {
                        "id": rule_id,
                        "tier": tier,
                        "description": "desc",
                        "rationale": rationale,
                        "producer": {"kind": "ast_grep", "path": "grep.yml"},
                    }
                ],
            }),
            encoding="utf-8",
        )
        _ = (pack_dir / "grep.yml").write_text("id: x\n", encoding="utf-8")

    def test_no_rulepacks_dir_is_a_no_op_for_regression_lines(self) -> None:
        baseline = [BaselineEntry(fingerprint="does-not-match", rule_id="RULE1", level="warning")]
        rows = [_finding_row(rule_id="RULE1", level="warning", line=2)]
        conn = _FakeGateConnection(rows)
        with (
            _chdir(self.repo),
            mock.patch.object(audit, "load_baseline", return_value=baseline),
        ):
            result = audit.gate_snapshot(cast("db.DbConnection", conn), self._snapshot())
        self.assertFalse(result.baseline_missing)
        self.assertEqual(len(result.regressions), 1)
        self.assertIn("NEW       RULE1", result.regressions[0])
        self.assertNotIn("tier", result.regressions[0])

    def test_matching_rulepack_rule_id_annotates_regression_line(self) -> None:
        self._write_rulepack(rule_id="RULE1", tier=1, rationale="why-this-matters")
        baseline = [BaselineEntry(fingerprint="does-not-match", rule_id="RULE1", level="warning")]
        rows = [_finding_row(rule_id="RULE1", level="warning", line=2)]
        conn = _FakeGateConnection(rows)
        with (
            _chdir(self.repo),
            mock.patch.object(audit, "load_baseline", return_value=baseline),
        ):
            result = audit.gate_snapshot(cast("db.DbConnection", conn), self._snapshot())
        self.assertEqual(len(result.regressions), 1)
        self.assertIn("[tier 1: why-this-matters]", result.regressions[0])

    def test_exit_relevant_regression_count_unaffected_by_rulepacks_presence(self) -> None:
        """Exit codes derive from `bool(regressions)`; that must hold with or without a
        rulepacks dir, and regardless of whether a rule ID in it matches a finding."""
        baseline: list[BaselineEntry] = []
        rows = [_finding_row(rule_id="RULE1", level="warning", line=2)]
        conn = _FakeGateConnection(rows)
        with (
            _chdir(self.repo),
            mock.patch.object(audit, "load_baseline", return_value=baseline),
        ):
            result_without = audit.gate_snapshot(cast("db.DbConnection", conn), self._snapshot())
        self._write_rulepack(rule_id="RULE1", tier=1, rationale="r")
        with (
            _chdir(self.repo),
            mock.patch.object(audit, "load_baseline", return_value=baseline),
        ):
            result_with = audit.gate_snapshot(cast("db.DbConnection", conn), self._snapshot())
        self.assertEqual(len(result_without.regressions), len(result_with.regressions))
        self.assertTrue(bool(result_without.regressions))
        self.assertTrue(bool(result_with.regressions))


if __name__ == "__main__":
    _ = unittest.main()
