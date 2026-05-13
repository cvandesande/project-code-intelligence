"""Portable GraphQL metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

GRAPHQL_OPERATION_RE = re.compile(r"(?m)^\s*(query|mutation|subscription)\s+([A-Za-z_][A-Za-z0-9_]*)?")
GRAPHQL_FRAGMENT_RE = re.compile(r"(?m)^\s*fragment\s+([A-Za-z_][A-Za-z0-9_]*)\s+on\s+([A-Za-z_][A-Za-z0-9_]*)")
GRAPHQL_TYPE_RE = re.compile(r"(?m)^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)")
GRAPHQL_INPUT_RE = re.compile(r"(?m)^\s*input\s+([A-Za-z_][A-Za-z0-9_]*)")
GRAPHQL_INTERFACE_RE = re.compile(r"(?m)^\s*interface\s+([A-Za-z_][A-Za-z0-9_]*)")
GRAPHQL_ENUM_RE = re.compile(r"(?m)^\s*enum\s+([A-Za-z_][A-Za-z0-9_]*)")
GRAPHQL_UNION_RE = re.compile(r"(?m)^\s*union\s+([A-Za-z_][A-Za-z0-9_]*)")
GRAPHQL_SCALAR_RE = re.compile(r"(?m)^\s*scalar\s+([A-Za-z_][A-Za-z0-9_]*)")

GRAPHQL_METADATA_KEYS = (
    "graphql_operations",
    "graphql_operation_kinds",
    "graphql_fragments",
    "graphql_fragment_types",
    "graphql_types",
    "graphql_inputs",
    "graphql_interfaces",
    "graphql_enums",
    "graphql_unions",
    "graphql_scalars",
)


def graphql_file_metadata(_path: str, text: str) -> JsonObject:
    operations: list[str] = []
    operation_kinds: list[str] = []
    for match in GRAPHQL_OPERATION_RE.finditer(text):
        operation_kinds.append(match.group(1))
        if match.group(2):
            operations.append(match.group(2))
    return compact_metadata({
        "graphql_operations": unique_limited(operations),
        "graphql_operation_kinds": unique_limited(operation_kinds),
        "graphql_fragments": unique_limited(match.group(1) for match in GRAPHQL_FRAGMENT_RE.finditer(text)),
        "graphql_fragment_types": unique_limited(match.group(2) for match in GRAPHQL_FRAGMENT_RE.finditer(text)),
        "graphql_types": unique_limited(match.group(1) for match in GRAPHQL_TYPE_RE.finditer(text)),
        "graphql_inputs": unique_limited(match.group(1) for match in GRAPHQL_INPUT_RE.finditer(text)),
        "graphql_interfaces": unique_limited(match.group(1) for match in GRAPHQL_INTERFACE_RE.finditer(text)),
        "graphql_enums": unique_limited(match.group(1) for match in GRAPHQL_ENUM_RE.finditer(text)),
        "graphql_unions": unique_limited(match.group(1) for match in GRAPHQL_UNION_RE.finditer(text)),
        "graphql_scalars": unique_limited(match.group(1) for match in GRAPHQL_SCALAR_RE.finditer(text)),
    })


GRAPHQL_PROFILE = LanguageProfile(
    name="graphql",
    languages=frozenset({"graphql"}),
    metadata_keys=GRAPHQL_METADATA_KEYS,
    file_metadata=graphql_file_metadata,
)
