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

    def test_caller_callee_pair_never_grouped(self) -> None:
        # normalize_rocm_bundle calls bundle_for_gfx_target; their shapes overlap
        # by composition, not duplication, so they must not cluster together.
        nodes = [
            _node(
                "normalize_rocm_bundle",
                "svc/rocm.py",
                10,
                ["validate_bundle", "convert_bundle", "bundle_for_gfx_target", "map_error"],
            ),
            _node(
                "bundle_for_gfx_target",
                "svc/rocm.py",
                40,
                ["validate_bundle", "convert_bundle", "resolve_target", "map_error"],
            ),
        ]
        self.assertEqual(analyze.cluster_functions(nodes, AnalysisOptions(min_members=2)), [])


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

    def test_coherence_is_max_of_semantic_and_text_ignoring_none(self) -> None:
        members = [
            _node("create_user", "svc/user.py", 10, ["validate_user", "convert_user", "repo.insert", "map_error"]),
            _node("create_team", "svc/team.py", 10, ["validate_team", "convert_team", "repo.insert", "map_error"]),
        ]
        self.assertEqual(analyze.build_group(members, avg_semantic=0.4, avg_text=0.9).coherence, 0.9)
        self.assertEqual(analyze.build_group(members, avg_semantic=0.9, avg_text=0.4).coherence, 0.9)
        self.assertEqual(analyze.build_group(members, avg_semantic=None, avg_text=None).coherence, 0.0)

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
        # non-positive value -> leave it alone.
        self.assertEqual(analyze.recommendation(0.0, 4.0, 0.0), "leave-as-is")
        self.assertEqual(analyze.recommendation(-2.0, 4.0, 0.0), "leave-as-is")
        # positive value but residual heavier than the shared core -> leaky.
        self.assertEqual(analyze.recommendation(1.0, 3.0, 4.0), "parameterize-carefully")
        # positive value, low residual -> collapse.
        self.assertEqual(analyze.recommendation(3.0, 4.0, 0.0), "worth-collapsing")

    def test_net_value_is_removed_minus_introduced(self) -> None:
        self.assertEqual(analyze.net_value(4.0, 1.0), 3.0)
        self.assertEqual(analyze.net_value(1.0, 5.0), -4.0)

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

    def test_shared_helper_is_evidence_only(self) -> None:
        # A shared internal callee is surfaced but never changes the verdict or
        # the rank key: parallel wrappers around one helper are what real
        # duplication looks like (measured at base rate on 40 labeled groups).
        members = [
            _node("create_user", "svc/user.py", 10, ["validate_user", "compact_json", "repo.insert", "map_error"]),
            _node("create_team", "svc/team.py", 10, ["validate_team", "compact_json", "repo.insert", "map_error"]),
        ]
        group = analyze.build_group(members, avg_semantic=None, function_symbols=frozenset({"compact_json"}))
        self.assertEqual(group.shared_helper, ("compact_json",))
        bare = analyze.build_group(members, avg_semantic=None, function_symbols=frozenset())
        self.assertEqual(group.recommendation, bare.recommendation)
        self.assertEqual(group.net_value, bare.net_value)


def _motif(recommendation: str, net_value: float, avg_semantic: float | None = None) -> analyze.MotifGroup:
    return analyze.MotifGroup(
        members=(_node("m", "m.py", 1, ["x_m", "y_m", "z_m"]),),
        common_roles=("x", "y", "z"),
        avg_structural=1.0,
        avg_semantic=avg_semantic,
        avg_text=None,
        max_text=None,
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
        # A leave-as-is group with far higher net value must still sort below
        # any actionable candidate.
        idle = _motif("leave-as-is", 50.0)
        action = _motif("worth-collapsing", 1.0)
        ranked = analyze.rank_groups([idle, action])
        self.assertEqual([group.recommendation for group in ranked], ["worth-collapsing", "leave-as-is"])

    def test_semantic_breaks_net_value_ties(self) -> None:
        weak = _motif("worth-collapsing", 5.0, avg_semantic=0.5)
        strong = _motif("worth-collapsing", 5.0, avg_semantic=0.9)
        ranked = analyze.rank_groups([weak, strong])
        self.assertEqual([group.avg_semantic for group in ranked], [0.9, 0.5])

    def test_coherence_outranks_net_value_within_actionable_tier(self) -> None:
        # coherence (max of semantic/text similarity) is the primary rank key
        # within a tier; a lower-net_value group with higher coherence must
        # still sort first.
        high_value_low_coherence = _motif("worth-collapsing", net_value=50.0, avg_semantic=0.2)
        low_value_high_coherence = _motif("parameterize-carefully", net_value=1.0, avg_semantic=0.9)
        ranked = analyze.rank_groups([high_value_low_coherence, low_value_high_coherence])
        self.assertEqual([group.net_value for group in ranked], [1.0, 50.0])


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
        self.assertIsNone(groups[0]["text_similarity"])
        self.assertEqual(groups[0]["recommendation"], "worth-collapsing")
        self.assertEqual(groups[0]["net_value"], 3.0)
        self.assertEqual(groups[0]["coherence"], 0.87)
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
        self.assertIn("Coherence (rank key): 0.87", text)
        self.assertIn("2 functions analyzed, 0 exact clones folded", text)

    def test_empty_results_render_without_error(self) -> None:
        empty = [analyze.SnapshotResult(label="default/demo", groups=(), functions_analyzed=0, clones_folded=0)]
        text = analyze.render_text(empty)
        self.assertIn("no repeated call-shape motifs", text)


class TextSimilarityTests(unittest.TestCase):
    def test_identical_bodies_score_near_one(self) -> None:
        body = '''def create_user(payload):
    """Create a user."""
    validated = validate_user(payload)
    return repo.insert(validated)
'''
        self.assertGreater(analyze.body_text_similarity(body, body), 0.99)

    def test_unrelated_bodies_score_low(self) -> None:
        left = '''def create_user(payload):
    """Create a user."""
    validated = validate_user(payload)
    return repo.insert(validated)
'''
        right = '''def render_report(rows):
    """Render a text report."""
    lines = [format_row(row) for row in rows]
    return "\\n".join(lines)
'''
        self.assertLess(analyze.body_text_similarity(left, right), 0.5)

    def test_normalize_drops_def_line_and_docstring(self) -> None:
        text = '''def create_user(
    payload,
):
    """Create a user.

    Longer description.
    """
    validated = validate_user(payload)
    return repo.insert(validated)
'''
        self.assertEqual(
            analyze.normalize_body_text(text),
            "    validated = validate_user(payload)\n    return repo.insert(validated)",
        )


def _core_for(annotation: str) -> str | None:
    """Extracted, normalized return-type core for one raw annotation text."""
    return analyze.return_annotation(f"def f() -> {annotation}:\n    pass\n")


class TypedVariantTests(unittest.TestCase):
    """The four annotation cases from the task, plus the recommendation-level check."""

    def test_int_bool_str_optionals_are_distinct_cores(self) -> None:
        # Must fire: cores int/bool/str differ.
        cores = [_core_for("int | None"), _core_for("bool"), _core_for("str | None")]
        self.assertEqual(cores, ["int", "bool", "str"])
        self.assertTrue(analyze.has_typed_variants(cores))

    def test_str_and_bool_are_distinct_cores(self) -> None:
        # Must fire: str vs bool differ outright.
        cores = [_core_for("str"), _core_for("bool")]
        self.assertTrue(analyze.has_typed_variants(cores))

    def test_optional_dict_matches_plain_dict_core(self) -> None:
        # Must NOT fire: optionality-only difference normalizes to the same core.
        cores = [_core_for("dict[str, object] | None"), _core_for("dict[str, object]")]
        self.assertEqual(cores[0], cores[1])
        self.assertFalse(analyze.has_typed_variants(cores))

    def test_identical_str_annotations_do_not_fire(self) -> None:
        # Must NOT fire: no difference at all.
        cores = [_core_for("str"), _core_for("str")]
        self.assertFalse(analyze.has_typed_variants(cores))

    def test_high_text_similarity_and_differing_cores_downgrade_recommendation(self) -> None:
        cores = [_core_for("int | None"), _core_for("str | None")]
        typed_variants = analyze.is_typed_variant_group(0.9, cores)
        self.assertTrue(typed_variants)
        # Value/shared/residual that would otherwise say "worth-collapsing" must
        # still downgrade — collapsing would lose type-checker precision.
        self.assertEqual(
            analyze.recommendation(3.0, 4.0, 0.0, typed_variants=typed_variants),
            "leave-as-is",
        )

    def test_build_group_wires_typed_variants_without_touching_net_value(self) -> None:
        members = [
            _node("get_int", "svc/opt.py", 10, ["validate_x", "convert_x", "repo.get", "map_error"]),
            _node("get_str", "svc/opt.py", 40, ["validate_x", "convert_x", "repo.get", "map_error"]),
        ]
        collapsible = analyze.build_group(members, avg_semantic=0.9)
        typed = analyze.build_group(members, avg_semantic=0.9, typed_variants=True)
        self.assertEqual(typed.net_value, collapsible.net_value)
        self.assertEqual(collapsible.recommendation, "worth-collapsing")
        self.assertEqual(typed.recommendation, "leave-as-is")
        self.assertTrue(typed.typed_variants)


class PathPrefixMatchTests(unittest.TestCase):
    """Stored paths are repo-prefixed; a caller may pass either form."""

    def test_repo_prefixed_prefix_matches(self) -> None:
        self.assertTrue(analyze.path_matches_prefix("pci/src/pkg/mcp/tools.py", "pci/src/pkg/mcp"))

    def test_repo_relative_prefix_matches(self) -> None:
        self.assertTrue(analyze.path_matches_prefix("pci/src/pkg/mcp/tools.py", "src/pkg/mcp"))

    def test_unrelated_prefix_does_not_match(self) -> None:
        self.assertFalse(analyze.path_matches_prefix("pci/src/pkg/mcp/tools.py", "src/pkg/sarif"))


if __name__ == "__main__":
    _ = unittest.main()
