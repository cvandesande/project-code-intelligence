"""Typed Literal aliases for MCP response taxonomies.

Closed sets of strings that flow across the MCP wire. Pinning them as Literal
aliases gives basedpyright the typo-protection it can't infer from
`{"kind": "X"}` shape literals. The producer side (warning emitters in tools
and helpers) and the consumer side (contract tests that assert on these
strings) both reference the same alias so a typo at either side fails type
checking.

Literal aliases are preferred over Enums because:
- They serialize to JSON as plain strings (Enums don't without custom encoders).
- pydantic Literal[] in `mcp.tool_inputs` composes with them directly.
- Tests that read `response["warnings"][0]["kind"]` get a `str` value, so no
  unwrap-by-`.value` ceremony is needed.
"""

from __future__ import annotations

from typing import Literal, TypeAlias, cast, get_args

WarningKind: TypeAlias = Literal[
    "empty_repo_scope",
    "empty_snapshot_scope",
    "empty_path_scope",
    "empty_language_scope",
    "empty_file_role_scope",
    "empty_record_type_scope",
    "empty_content_class_scope",
    "repo_root_path_scope",
    "snapshot_stale",
    "snapshot_freshness_unknown",
    "snapshot_dirty",
    "tokenized_text_search",
    "query_strategy_fallback",
    "mode_inferred_enumerate",
    "heuristic_candidate_relationships",
    "semantic_filter_has_no_embeddings",
    "record_not_found",
    "symbol_not_found",
    "index_run_active",
    "static_analysis_not_run",
    "overconstrained_boolean_filters",
]


ConfidenceKind: TypeAlias = Literal[
    "high_confidence_fact",
    "approximate_fact",
    "heuristic_candidate",
    "tool_finding",
]


# Display string for the confidence_kind property in tool schemas. Derived
# from the ConfidenceKind alias so the JSON Schema description and the
# runtime pydantic validator can never drift apart — adding a value above
# automatically propagates to the schema description.
CONFIDENCE_KIND_VALUES: tuple[str, ...] = cast("tuple[str, ...]", get_args(ConfidenceKind))
