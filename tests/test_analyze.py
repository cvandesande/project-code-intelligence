from __future__ import annotations

import json
import unittest
from typing import cast

from project_code_intelligence import analyze
from project_code_intelligence.analyze import AnalysisOptions, FunctionNode


def _node(symbol: str, path: str, line: int, callees: list[str]) -> FunctionNode:
    return FunctionNode(
        record_id=f"{path}::function::{symbol}::{line:06d}",
        symbol=symbol,
        source_path=path,
        line_start=line,
        line_end=line + 20,
        callee_roles=analyze.role_set(callees),
    )


class NormalizeRoleTests(unittest.TestCase):
    def test_snake_case_keeps_leading_verb(self) -> None:
        self.assertEqual(analyze.normalize_role("validate_user"), "validate")

    def test_camel_case_keeps_leading_verb(self) -> None:
        self.assertEqual(analyze.normalize_role("validateApp"), "validate")

    def test_dotted_name_uses_final_component(self) -> None:
        self.assertEqual(analyze.normalize_role("repo.insert"), "insert")

    def test_leading_underscores_stripped(self) -> None:
        self.assertEqual(analyze.normalize_role("__map_error"), "map")

    def test_single_token_preserved(self) -> None:
        self.assertEqual(analyze.normalize_role("insert"), "insert")


class RoleSetAndJaccardTests(unittest.TestCase):
    def test_domain_variants_collapse_to_same_roles(self) -> None:
        user = analyze.role_set(["validate_user", "convert_user", "repo.insert", "map_error"])
        team = analyze.role_set(["validate_team", "convert_team", "repo.insert", "map_error"])
        self.assertEqual(user, team)
        self.assertEqual(user, frozenset({"validate", "convert", "insert", "map"}))

    def test_jaccard_bounds(self) -> None:
        self.assertEqual(analyze.jaccard(frozenset(), frozenset()), 0.0)
        self.assertEqual(analyze.jaccard(frozenset({"a"}), frozenset({"a"})), 1.0)
        self.assertEqual(analyze.jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})), 1 / 3)


class WeightingTests(unittest.TestCase):
    def test_role_weights_zero_for_ubiquitous_positive_for_rare(self) -> None:
        nodes = [
            _node("a", "m.py", 1, ["get_x", "validate_x", "store_x"]),
            _node("b", "m.py", 30, ["get_y", "convert_y", "emit_y"]),
        ]
        weights = analyze.role_weights(nodes)
        # "get" appears in every function -> idf log(2/2) == 0.
        self.assertEqual(weights["get"], 0.0)
        # distinctive roles appear once -> idf log(2/1) > 0.
        self.assertGreater(weights["validate"], 0.0)

    def test_weighted_jaccard_ignores_shared_boilerplate(self) -> None:
        weights = {"get": 0.0, "validate": 1.0, "convert": 1.0}
        # Two functions sharing only the zero-weight role are not similar.
        self.assertEqual(
            analyze.weighted_jaccard(frozenset({"get", "validate"}), frozenset({"get", "convert"}), weights),
            0.0,
        )
        # None weights reduce to plain Jaccard.
        self.assertEqual(
            analyze.weighted_jaccard(frozenset({"a", "b"}), frozenset({"b", "c"}), None),
            1 / 3,
        )


class DedupeClonesTests(unittest.TestCase):
    def test_identical_function_across_two_paths_is_folded(self) -> None:
        roles = ["read_cfg", "parse_cfg", "emit_cfg"]
        nodes = [
            _node("load", "src/pkg/scripts/tool.py", 5, roles),
            _node("load", "scripts/tool.py", 5, roles),
            _node("other", "svc/other.py", 5, ["open_x", "read_x", "close_x"]),
        ]
        unique, folded = analyze.dedupe_clones(nodes)
        self.assertEqual(folded, 1)
        self.assertEqual(len(unique), 2)
        # The lexically-smaller path is kept as the representative.
        kept = next(node for node in unique if node.symbol == "load")
        self.assertEqual(kept.source_path, "scripts/tool.py")

    def test_distinct_functions_are_not_folded(self) -> None:
        nodes = [
            _node("load", "a.py", 5, ["read_a", "parse_a"]),
            _node("load", "b.py", 40, ["read_b", "parse_b", "emit_b"]),
        ]
        unique, folded = analyze.dedupe_clones(nodes)
        self.assertEqual(folded, 0)
        self.assertEqual(len(unique), 2)


class ClusteringTests(unittest.TestCase):
    def test_issue_motif_clusters_together(self) -> None:
        nodes = [
            _node("create_user", "svc/user.py", 10, ["validate_user", "convert_user", "repo.insert", "map_error"]),
            _node("create_team", "svc/team.py", 10, ["validate_team", "convert_team", "repo.insert", "map_error"]),
            _node("create_app", "svc/app.py", 10, ["validate_app", "convert_app", "repo.insert", "map_error"]),
            _node("unrelated", "svc/other.py", 10, ["open_file", "read_line", "close_file"]),
        ]
        clusters = analyze.cluster_functions(nodes, AnalysisOptions())
        self.assertEqual(len(clusters), 1)
        self.assertEqual({node.symbol for node in clusters[0]}, {"create_user", "create_team", "create_app"})

    def test_min_roles_filters_thin_functions(self) -> None:
        nodes = [
            _node("a", "m.py", 1, ["validate_x", "convert_x"]),
            _node("b", "m.py", 30, ["validate_y", "convert_y"]),
        ]
        self.assertEqual(analyze.cluster_functions(nodes, AnalysisOptions(min_roles=3)), [])
        clusters = analyze.cluster_functions(nodes, AnalysisOptions(min_roles=2))
        self.assertEqual(len(clusters), 1)

    def test_min_members_drops_singletons(self) -> None:
        nodes = [
            _node("solo", "m.py", 1, ["validate_x", "convert_x", "store_x"]),
            _node("pair_a", "m.py", 40, ["read_a", "parse_a", "emit_a"]),
            _node("pair_b", "m.py", 80, ["read_b", "parse_b", "emit_b"]),
        ]
        clusters = analyze.cluster_functions(nodes, AnalysisOptions(min_members=2))
        self.assertEqual(len(clusters), 1)
        self.assertEqual({node.symbol for node in clusters[0]}, {"pair_a", "pair_b"})


class GroupBuildingTests(unittest.TestCase):
    def test_build_group_computes_common_shape_and_score(self) -> None:
        members = [
            _node("create_user", "svc/user.py", 10, ["validate_user", "convert_user", "repo.insert", "map_error"]),
            _node("create_team", "svc/team.py", 10, ["validate_team", "convert_team", "repo.insert", "map_error"]),
        ]
        group = analyze.build_group(members, avg_semantic=0.9)
        self.assertEqual(group.common_roles, ("convert", "insert", "map", "validate"))
        self.assertEqual(group.avg_structural, 1.0)
        self.assertEqual(group.avg_semantic, 0.9)
        self.assertEqual(group.score, 1.0)
        self.assertEqual(group.label, "medium")
        # _node() gives every member 21 LOC; keep one, fold the rest.
        self.assertEqual(group.estimated_loc_removed, 21)

    def test_estimated_loc_removed_needs_two_known_ranges(self) -> None:
        known = _node("create_user", "u.py", 10, ["validate_user", "convert_user", "repo.insert", "map_error"])
        unknown = analyze.FunctionNode(
            record_id="u.py::function::create_team::000010",
            symbol="create_team",
            source_path="t.py",
            line_start=None,
            line_end=None,
            callee_roles=analyze.role_set(["validate_team", "convert_team", "repo.insert", "map_error"]),
        )
        group = analyze.build_group([known, unknown], avg_semantic=None)
        self.assertIsNone(group.estimated_loc_removed)

    def test_high_label_needs_three_strong_members(self) -> None:
        members = [
            _node("create_user", "u.py", 10, ["validate_user", "convert_user", "repo.insert", "map_error"]),
            _node("create_team", "t.py", 10, ["validate_team", "convert_team", "repo.insert", "map_error"]),
            _node("create_app", "a.py", 10, ["validate_app", "convert_app", "repo.insert", "map_error"]),
        ]
        group = analyze.build_group(members, avg_semantic=None)
        self.assertEqual(group.label, "high")


def _sample_results() -> list[analyze.SnapshotResult]:
    members = [
        _node("create_user", "svc/user.py", 10, ["validate_user", "convert_user", "repo.insert", "map_error"]),
        _node("create_team", "svc/team.py", 10, ["validate_team", "convert_team", "repo.insert", "map_error"]),
    ]
    group = analyze.build_group(members, avg_semantic=0.87)
    return [analyze.SnapshotResult(label="default/demo", groups=(group,), functions_analyzed=2, clones_folded=0)]


class RenderingTests(unittest.TestCase):
    def test_json_render_is_parseable_and_shaped(self) -> None:
        payload = cast(
            "dict[str, dict[str, object]]",
            json.loads(analyze.render_json(_sample_results())),
        )
        snapshot = payload["default/demo"]
        self.assertEqual(snapshot["functions_analyzed"], 2)
        self.assertEqual(snapshot["clones_folded"], 0)
        groups = cast("list[dict[str, object]]", snapshot["groups"])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["common_shape"], ["convert", "insert", "map", "validate"])
        self.assertEqual(groups[0]["graph_similarity"], 1.0)
        self.assertEqual(groups[0]["semantic_similarity"], 0.87)
        self.assertEqual(groups[0]["estimated_compression"], "medium")
        self.assertEqual(groups[0]["estimated_loc_removed"], 21)

    def test_text_render_mentions_members_and_shape(self) -> None:
        text = analyze.render_text(_sample_results())
        self.assertIn("Motif 1", text)
        self.assertIn("create_user", text)
        self.assertIn("validate", text)
        self.assertIn("2 functions analyzed, 0 exact clones folded", text)

    def test_empty_results_render_without_error(self) -> None:
        empty = [analyze.SnapshotResult(label="default/demo", groups=(), functions_analyzed=0, clones_folded=0)]
        text = analyze.render_text(empty)
        self.assertIn("no repeated call-shape motifs", text)


if __name__ == "__main__":
    _ = unittest.main()
