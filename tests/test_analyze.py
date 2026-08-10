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
        callee_symbols=frozenset(callees),
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
    def test_build_group_computes_common_shape_and_net_value(self) -> None:
        members = [
            _node("create_user", "svc/user.py", 10, ["validate_user", "convert_user", "repo.insert", "map_error"]),
            _node("create_team", "svc/team.py", 10, ["validate_team", "convert_team", "repo.insert", "map_error"]),
        ]
        group = analyze.build_group(members, avg_semantic=0.9)
        self.assertEqual(group.common_roles, ("convert", "insert", "map", "validate"))
        self.assertEqual(group.avg_structural, 1.0)
        self.assertEqual(group.avg_semantic, 0.9)
        # Unweighted: shared core = 4 roles, no residual, one module -> cost = base 1.0.
        # removed = (2-1) * 4 * 1.0 = 4.0; net = 4.0 - 1.0 = 3.0.
        self.assertEqual(group.redundancy_removed, 4.0)
        self.assertEqual(group.abstraction_cost, 1.0)
        self.assertEqual(group.net_value, 3.0)
        self.assertEqual(group.recommendation, "worth-collapsing")
        self.assertEqual(group.residual_roles, ())
        self.assertFalse(group.low_coherence)
        # _node() gives every member 21 LOC; keep one, fold the rest.
        self.assertEqual(group.estimated_loc_removed, 21)

    def test_redundancy_squares_structural_agreement(self) -> None:
        members = [_node("a", "m.py", 1, ["x_a", "y_a"]), _node("b", "m.py", 40, ["x_b", "y_b"])]
        # (K-1) * weight(core=2 roles, unweighted=2.0) * 0.5**2 = 1 * 2.0 * 0.25 = 0.5.
        self.assertEqual(analyze.redundancy_removed(members, ["x", "y"], 0.5, None), 0.5)
        # Perfect agreement is unchanged by squaring.
        self.assertEqual(analyze.redundancy_removed(members, ["x", "y"], 1.0, None), 2.0)

    def test_loose_cluster_flags_low_coherence(self) -> None:
        # Pairwise Jaccard ~0.33 across all pairs (2 shared roles, 4 unique) ->
        # mean below the 0.6 floor. Distinct leading verbs keep roles distinct.
        members = [
            _node("a", "m.py", 1, ["alpha_x", "beta_x", "gamma_x", "delta_x"]),
            _node("b", "m.py", 40, ["alpha_x", "beta_x", "epsilon_x", "zeta_x"]),
            _node("c", "m.py", 80, ["alpha_x", "beta_x", "eta_x", "theta_x"]),
        ]
        group = analyze.build_group(members, avg_semantic=0.9)
        self.assertLess(group.avg_structural, 0.6)
        self.assertTrue(group.low_coherence)

    def test_weak_semantic_flags_low_coherence(self) -> None:
        members = [
            _node("create_user", "svc/user.py", 10, ["validate_user", "convert_user", "repo.insert", "map_error"]),
            _node("create_team", "svc/team.py", 10, ["validate_team", "convert_team", "repo.insert", "map_error"]),
        ]
        # Structurally identical (1.0) but semantically far apart -> still flagged.
        group = analyze.build_group(members, avg_semantic=0.4)
        self.assertEqual(group.avg_structural, 1.0)
        self.assertTrue(group.low_coherence)

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

    def test_cross_module_spread_raises_cost(self) -> None:
        roles = ["validate_x", "convert_x", "repo.insert", "map_error"]
        same = analyze.build_group(
            [_node("a", "svc/one.py", 10, roles), _node("b", "svc/two.py", 10, roles)],
            avg_semantic=None,
        )
        spread = analyze.build_group(
            [_node("a", "svc/one.py", 10, roles), _node("b", "api/two.py", 10, roles)],
            avg_semantic=None,
        )
        self.assertEqual(same.spread_penalty, 0.0)
        self.assertGreater(spread.spread_penalty, 0.0)
        self.assertLess(spread.net_value, same.net_value)


class NetValueTests(unittest.TestCase):
    def test_recommendation_branches(self) -> None:
        # existing helper wins regardless of value.
        self.assertEqual(analyze.recommendation(5.0, 4.0, 0.0, has_helper=True), "already-abstracted")
        # non-positive value -> leave it alone.
        self.assertEqual(analyze.recommendation(0.0, 4.0, 0.0, has_helper=False), "leave-as-is")
        self.assertEqual(analyze.recommendation(-2.0, 4.0, 0.0, has_helper=False), "leave-as-is")
        # positive value but residual heavier than the shared core -> leaky.
        self.assertEqual(analyze.recommendation(1.0, 3.0, 4.0, has_helper=False), "parameterize-carefully")
        # positive value, low residual -> collapse.
        self.assertEqual(analyze.recommendation(3.0, 4.0, 0.0, has_helper=False), "worth-collapsing")

    def test_net_value_damped_when_helper_present(self) -> None:
        undamped = analyze.net_value(4.0, 1.0, has_helper=False)
        self.assertEqual(undamped, 3.0)
        # Positive value is damped toward zero so already-abstracted groups rank low.
        damped = analyze.net_value(4.0, 1.0, has_helper=True)
        self.assertGreater(damped, 0.0)
        self.assertLess(damped, undamped)
        # A negative value is not "improved" by the damping.
        self.assertEqual(analyze.net_value(1.0, 5.0, has_helper=True), -4.0)

    def test_existing_helper_detects_shared_internal_callee(self) -> None:
        members = [
            _node("create_user", "svc/user.py", 10, ["validate_user", "compact_json", "repo.insert", "map_error"]),
            _node("create_team", "svc/team.py", 10, ["validate_team", "compact_json", "repo.insert", "map_error"]),
        ]
        helper = analyze.existing_helper(members, frozenset({"compact_json"}))
        self.assertEqual(helper, ("compact_json",))
        # Nothing internal known -> no helper claimed.
        self.assertEqual(analyze.existing_helper(members, frozenset()), ())

    def test_existing_helper_excludes_member_symbols(self) -> None:
        # Mutual calls between members must not read as a shared external helper.
        members = [
            _node("alpha", "m.py", 10, ["beta", "validate_x", "convert_x"]),
            _node("beta", "m.py", 40, ["beta", "validate_y", "convert_y"]),
        ]
        self.assertEqual(analyze.existing_helper(members, frozenset({"alpha", "beta"})), ())

    def test_build_group_flags_already_abstracted(self) -> None:
        members = [
            _node("create_user", "svc/user.py", 10, ["validate_user", "compact_json", "repo.insert", "map_error"]),
            _node("create_team", "svc/team.py", 10, ["validate_team", "compact_json", "repo.insert", "map_error"]),
        ]
        group = analyze.build_group(members, avg_semantic=None, function_symbols=frozenset({"compact_json"}))
        self.assertEqual(group.shared_helper, ("compact_json",))
        self.assertEqual(group.recommendation, "already-abstracted")
        # net value damped below the un-damped 3.0 so it ranks low.
        self.assertLess(group.net_value, 3.0)


def _motif(recommendation: str, net_value: float, avg_semantic: float | None = None) -> analyze.MotifGroup:
    return analyze.MotifGroup(
        members=(_node("m", "m.py", 1, ["x_m", "y_m", "z_m"]),),
        common_roles=("x", "y", "z"),
        avg_structural=1.0,
        avg_semantic=avg_semantic,
        net_value=net_value,
        value_ratio=0.0,
        redundancy_removed=0.0,
        abstraction_cost=1.0,
        residual_cost=0.0,
        spread_penalty=0.0,
        shared_helper=(),
        recommendation=recommendation,
    )


class RankingTests(unittest.TestCase):
    def test_actionable_outranks_non_actionable_regardless_of_net_value(self) -> None:
        # An already-abstracted group with far higher net value must still sort
        # below any actionable candidate.
        helper = _motif("already-abstracted", 50.0)
        action = _motif("worth-collapsing", 1.0)
        ranked = analyze.rank_groups([helper, action])
        self.assertEqual([group.recommendation for group in ranked], ["worth-collapsing", "already-abstracted"])

    def test_semantic_breaks_net_value_ties(self) -> None:
        weak = _motif("worth-collapsing", 5.0, avg_semantic=0.5)
        strong = _motif("worth-collapsing", 5.0, avg_semantic=0.9)
        ranked = analyze.rank_groups([weak, strong])
        self.assertEqual([group.avg_semantic for group in ranked], [0.9, 0.5])


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
        self.assertEqual(groups[0]["recommendation"], "worth-collapsing")
        self.assertEqual(groups[0]["net_value"], 3.0)
        evidence = cast("dict[str, object]", groups[0]["evidence"])
        self.assertEqual(evidence["redundancy_removed"], 4.0)
        self.assertEqual(evidence["abstraction_cost"], 1.0)
        self.assertEqual(evidence["estimated_loc_removed"], 21)

    def test_text_render_mentions_members_and_shape(self) -> None:
        text = analyze.render_text(_sample_results())
        self.assertIn("Motif 1", text)
        self.assertIn("create_user", text)
        self.assertIn("validate", text)
        self.assertIn("worth-collapsing", text)
        self.assertIn("Why:", text)
        self.assertIn("2 functions analyzed, 0 exact clones folded", text)

    def test_empty_results_render_without_error(self) -> None:
        empty = [analyze.SnapshotResult(label="default/demo", groups=(), functions_analyzed=0, clones_folded=0)]
        text = analyze.render_text(empty)
        self.assertIn("no repeated call-shape motifs", text)


if __name__ == "__main__":
    _ = unittest.main()
