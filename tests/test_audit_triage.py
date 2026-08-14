from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from project_code_intelligence import analyze, audit_triage


def _group(*members: tuple[str, str, int]) -> analyze.MotifGroup:
    return analyze.MotifGroup(
        members=tuple(
            analyze.FunctionNode(
                record_id=f"id-{index}",
                symbol=symbol,
                source_path=path,
                line_start=line,
                line_end=line + 2,
                callee_roles=frozenset({"call"}),
            )
            for index, (symbol, path, line) in enumerate(members)
        ),
        common_roles=("call",),
        avg_structural=1.0,
        avg_semantic=None,
        avg_text=1.0,
        max_text=1.0,
        net_value=0,
        value_ratio=0,
        redundancy_removed=1,
        abstraction_cost=1,
        residual_cost=0,
        spread_penalty=0,
        shared_helper=(),
        recommendation="review",
    )


class CandidateIdentityTests(unittest.TestCase):
    def test_id_ignores_member_order_lines_and_repo_prefix(self) -> None:
        first = _group(("a", "repo/a.py", 1), ("b", "repo/b.py", 4))
        moved = _group(("b", "repo/b.py", 40), ("a", "repo/a.py", 10))
        self.assertEqual(
            audit_triage.candidate_id(first, "collection", "repo"),
            audit_triage.candidate_id(moved, "collection", "repo"),
        )

    def test_id_is_language_agnostic(self) -> None:
        python = _group(("parse", "repo/parser.py", 1), ("read", "repo/input.py", 4))
        rust = _group(("Parser::parse", "repo/src/parser.rs", 1), ("read", "repo/src/input.rs", 4))
        self.assertTrue(audit_triage.candidate_id(python, "collection", "repo").startswith("redundancy-"))
        self.assertTrue(audit_triage.candidate_id(rust, "collection", "repo").startswith("redundancy-"))

    def test_collection_participates_in_identity(self) -> None:
        group = _group(("a", "repo/a.py", 1), ("b", "repo/b.py", 4))
        self.assertNotEqual(
            audit_triage.candidate_id(group, "first", "repo"),
            audit_triage.candidate_id(group, "second", "repo"),
        )

    def test_summary_infers_fixed_when_saved_candidate_disappears(self) -> None:
        saved = audit_triage.TriageEntry(status="open", reason=None, members=("a a.py", "b b.py"))
        summary = audit_triage.summary({}, {"redundancy-abc": saved}, infer_fixed=True)
        self.assertEqual(summary["fixed"], [("redundancy-abc", saved)])

    def test_bounded_audit_does_not_infer_fixed(self) -> None:
        saved = audit_triage.TriageEntry(status="open", reason=None, members=("a a.py", "b b.py"))
        summary = audit_triage.summary({}, {"redundancy-abc": saved})
        self.assertEqual(summary["fixed"], [])

    def test_unique_two_member_overlap_preserves_disposition(self) -> None:
        current = {
            "redundancy-new": audit_triage.Candidate(scope="collection/repo", members=("a a.py", "b b.py", "c c.py"))
        }
        saved = audit_triage.TriageEntry(
            status="dismissed",
            reason="intentional",
            members=("a a.py", "b b.py"),
            scope="collection/repo",
        )
        reconciled = audit_triage.reconcile(current, {"redundancy-old": saved})
        self.assertNotIn("redundancy-old", reconciled)
        self.assertEqual(reconciled["redundancy-new"].status, "dismissed")
        self.assertEqual(reconciled["redundancy-new"].members, current["redundancy-new"].members)

    def test_ambiguous_overlap_does_not_guess(self) -> None:
        current = {
            "redundancy-one": audit_triage.Candidate(scope="c/r", members=("a a.py", "b b.py", "c c.py")),
            "redundancy-two": audit_triage.Candidate(scope="c/r", members=("a a.py", "b b.py", "d d.py")),
        }
        saved = audit_triage.TriageEntry(
            status="dismissed", reason="intentional", members=("a a.py", "b b.py"), scope="c/r"
        )
        self.assertEqual(audit_triage.reconcile(current, {"redundancy-old": saved}), {"redundancy-old": saved})

    def test_overlap_does_not_cross_collection_scope(self) -> None:
        current = {"new": audit_triage.Candidate(scope="second/repo", members=("a a.py", "b b.py"))}
        saved = audit_triage.TriageEntry(
            status="dismissed", reason="intentional", members=("a a.py", "b b.py"), scope="first/repo"
        )
        self.assertEqual(audit_triage.reconcile(current, {"old": saved}), {"old": saved})


class TriageFileTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "triage.json"
            expected = {
                "redundancy-abc": audit_triage.TriageEntry(
                    status="dismissed", reason="intentional boundary", members=("a a.py", "b b.py")
                )
            }
            audit_triage.write_triage(path, expected)
            self.assertEqual(audit_triage.load_triage(path), expected)
            payload = cast("dict[str, object]", json.loads(path.read_text()))
            self.assertEqual(payload["version"], audit_triage.TRIAGE_FILE_VERSION)
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            self.assertIn("generator", payload)
            self.assertIn("updated_at", payload)

    def test_loads_version_one_for_migration(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "triage.json"
            _ = path.write_text(
                '{"version":1,"candidates":{"old":{"status":"open","reason":null,"members":["a a.py"]}}}'
            )
            self.assertEqual(audit_triage.load_triage(path)["old"].scope, None)

    def test_rejects_unknown_version(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "triage.json"
            _ = path.write_text('{"version":999,"candidates":{}}')
            with self.assertRaisesRegex(ValueError, "unsupported"):
                _ = audit_triage.load_triage(path)


if __name__ == "__main__":
    _ = unittest.main()
