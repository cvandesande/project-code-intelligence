from __future__ import annotations

import json
import unittest
from dataclasses import replace
from typing import cast

from project_code_intelligence import evidence
from project_code_intelligence.evidence import Caller, Evidence, Neighbor, Staleness


def _staleness(*, stale: bool = False) -> Staleness:
    return Staleness(
        head_status="stale" if stale else "current",
        dirty=False,
        index_age_seconds=42,
        target_file_dirty=False,
    )


_BASE = Evidence(
    label="default/demo",
    record_id="svc/user.py::function::create_user::000010",
    symbol="create_user",
    symbol_kind="function",
    source_path="svc/user.py",
    line_start=10,
    line_end=20,
    callers=(),
    callees=(),
    name_reference_count=0,
    is_service_entrypoint=False,
    neighbors=(),
    staleness=_staleness(),
)


def _evidence(**overrides: object) -> Evidence:
    return replace(_BASE, **overrides)


class IsParentChildTests(unittest.TestCase):
    def test_same_symbol_is_nested(self) -> None:
        self.assertTrue(evidence.is_parent_child(("A", "m.py", (1, 9)), ("A", "m.py", (1, 9))))

    def test_symbol_prefix_relation_is_nested(self) -> None:
        # A class matched against its own method — the dominant probe false positive.
        self.assertTrue(evidence.is_parent_child(("A", "m.py", (1, 20)), ("A.method", "m.py", (5, 8))))
        self.assertTrue(evidence.is_parent_child(("outer.inner", "m.py", (5, 8)), ("outer", "m.py", (1, 20))))

    def test_overlapping_spans_same_file_are_nested(self) -> None:
        # A nested closure inside the target — bodies overlap.
        self.assertTrue(evidence.is_parent_child(("outer", "m.py", (1, 30)), ("helper", "m.py", (10, 15))))

    def test_distinct_definitions_are_not_nested(self) -> None:
        self.assertFalse(
            evidence.is_parent_child(("create_user", "user.py", (1, 9)), ("create_team", "team.py", (1, 9)))
        )
        # Same file, disjoint spans — independent siblings, not nested.
        self.assertFalse(evidence.is_parent_child(("a", "m.py", (1, 9)), ("b", "m.py", (20, 30))))

    def test_missing_spans_same_file_are_not_nested(self) -> None:
        self.assertFalse(evidence.is_parent_child(("a", "m.py", (None, None)), ("b", "m.py", (1, 9))))


class EvidencePropertyTests(unittest.TestCase):
    def test_covered_by_tests_lists_distinct_test_files(self) -> None:
        callers = (
            Caller(symbol="t1", source_path="tests/test_user.py", line=5, is_test=True, at_module_level=False),
            Caller(symbol="t2", source_path="tests/test_user.py", line=9, is_test=True, at_module_level=False),
            Caller(symbol="api", source_path="api/routes.py", line=88, is_test=False, at_module_level=False),
        )
        item = _evidence(callers=callers)
        self.assertEqual(item.inbound_count, 3)
        self.assertEqual(item.covered_by_tests, ("tests/test_user.py",))

    def test_wired_at_module_level_from_module_caller(self) -> None:
        callers = (Caller(symbol=None, source_path="reg.py", line=1, is_test=False, at_module_level=True),)
        self.assertTrue(_evidence(callers=callers).wired_at_module_level)
        self.assertFalse(_evidence().wired_at_module_level)

    def test_conventional_entrypoint_detected_by_bare_name(self) -> None:
        self.assertTrue(_evidence(symbol="main").is_conventional_entrypoint)
        self.assertTrue(_evidence(symbol="cli.main").is_conventional_entrypoint)
        self.assertFalse(_evidence(symbol="create_user").is_conventional_entrypoint)

    def test_looks_orphaned_only_when_no_reference_and_not_entrypoint(self) -> None:
        # No callers, no name mentions, not an entry point.
        self.assertTrue(_evidence().looks_orphaned)
        # A bare-name mention elsewhere clears it (dynamic dispatch backstop).
        self.assertFalse(_evidence(name_reference_count=2).looks_orphaned)
        # An entry point is never orphaned.
        self.assertFalse(_evidence(symbol="main").looks_orphaned)
        self.assertFalse(_evidence(is_service_entrypoint=True).looks_orphaned)
        # A resolved caller clears it.
        caller = (Caller(symbol="x", source_path="a.py", line=1, is_test=False, at_module_level=False),)
        self.assertFalse(_evidence(callers=caller).looks_orphaned)

    def test_staleness_is_stale_covers_each_trigger(self) -> None:
        self.assertFalse(_staleness().is_stale)
        self.assertTrue(replace(_staleness(), head_status="stale").is_stale)
        self.assertTrue(replace(_staleness(), dirty=True).is_stale)
        self.assertTrue(replace(_staleness(), target_file_dirty=True).is_stale)


class RenderTextTests(unittest.TestCase):
    def test_orphan_render_shows_banner_and_flag(self) -> None:
        text = evidence.render_text(_evidence(staleness=_staleness(stale=True)))
        self.assertIn("! index stale", text)
        self.assertIn("callers (0)", text)
        self.assertIn("looks orphaned", text)

    def test_zero_callers_hints_name_reference_count(self) -> None:
        text = evidence.render_text(_evidence(name_reference_count=3))
        self.assertIn("name referenced 3x elsewhere", text)
        # A name-referenced symbol is not orphaned, so no orphan flag.
        self.assertNotIn("looks orphaned", text)

    def test_render_marks_test_and_module_callers_and_neighbours(self) -> None:
        callers = (
            Caller(symbol="t", source_path="tests/test_user.py", line=5, is_test=True, at_module_level=False),
            Caller(symbol=None, source_path="reg.py", line=1, is_test=False, at_module_level=True),
        )
        neighbors = (Neighbor(symbol="create_team", source_path="svc/team.py", line=10, similarity=0.94),)
        text = evidence.render_text(_evidence(callers=callers, neighbors=neighbors))
        self.assertIn("[test]", text)
        self.assertIn("[module]", text)
        self.assertIn("wired at module level", text)
        self.assertIn("semantic neighbours", text)
        self.assertIn("create_team", text)
        self.assertIn("0.94", text)

    def test_render_omits_banner_when_fresh(self) -> None:
        self.assertNotIn("index stale", evidence.render_text(_evidence()))


class RenderJsonTests(unittest.TestCase):
    def test_json_shape_is_parseable_and_complete(self) -> None:
        callers = (Caller(symbol="api", source_path="api/routes.py", line=88, is_test=False, at_module_level=False),)
        neighbors = (Neighbor(symbol="create_team", source_path="svc/team.py", line=10, similarity=0.9412),)
        payload = cast(
            "list[dict[str, object]]",
            json.loads(evidence.render_json([_evidence(callers=callers, neighbors=neighbors)])),
        )
        self.assertEqual(len(payload), 1)
        item = payload[0]
        self.assertEqual(item["symbol"], "create_user")
        self.assertEqual(item["inbound_count"], 1)
        self.assertEqual(item["is_entrypoint"], False)
        neighbor_rows = cast("list[dict[str, object]]", item["neighbors"])
        self.assertEqual(neighbor_rows[0]["similarity"], 0.9412)
        staleness = cast("dict[str, object]", item["staleness"])
        self.assertIn("is_stale", staleness)


if __name__ == "__main__":
    _ = unittest.main()
