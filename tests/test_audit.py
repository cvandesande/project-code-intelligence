from __future__ import annotations

import unittest

from project_code_intelligence import analyze, audit


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


class RenderTests(unittest.TestCase):
    def test_render_text_smoke(self) -> None:
        result = audit.AuditResult(
            label="c/r",
            snapshot_id=1,
            staleness=None,
            names_total=10,
            duplicate_names=(audit.DuplicateName(name="main", paths=("a.py", "b.py")),),
            redundancy=analyze.SnapshotResult(
                label="c/r",
                groups=(_group(["x", "y"], avg_text=1.0, max_text=1.0),),
                functions_analyzed=10,
                clones_folded=0,
            ),
            static_commit=None,
            static_counts=(),
        )
        text = audit.render_text([result])
        self.assertIn("UNRANKED", text)
        self.assertIn("main  x2: a.py, b.py", text)
        self.assertIn("x  x.py:1", text)
        self.assertIn("no SARIF ingested", text)
        self.assertIn("reindex (pci-index)", text)


if __name__ == "__main__":
    _ = unittest.main()
