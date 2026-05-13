"""Metadata extraction for Starlark and Bazel files."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

STARLARK_LOAD_RE = re.compile(r"""(?m)^\s*load\(\s*["']([^"']+)["']([^)]*)\)""")
STARLARK_STRING_RE = re.compile(r"""["']([^"']+)["']""")
STARLARK_FUNCTION_RE = re.compile(r"(?m)^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
BAZEL_TARGET_RE = re.compile(r"""(?ms)^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\(\s*name\s*=\s*["']([^"']+)["']""")
BAZEL_PACKAGE_RE = re.compile(r"""(?m)^\s*package\s*\(""")

BAZEL_METADATA_KEYS = (
    "starlark_loads",
    "starlark_loaded_symbols",
    "starlark_functions",
    "bazel_rules",
    "bazel_targets",
    "bazel_has_package_declaration",
)


def loaded_symbols(value: str) -> list[str]:
    return [match.group(1) for match in STARLARK_STRING_RE.finditer(value)]


def bazel_file_metadata(_path: str, text: str) -> JsonObject:
    loaded: list[str] = []
    symbols: list[str] = []
    for match in STARLARK_LOAD_RE.finditer(text):
        loaded.append(match.group(1))
        symbols.extend(loaded_symbols(match.group(2)))
    rules: list[str] = []
    targets: list[str] = []
    for match in BAZEL_TARGET_RE.finditer(text):
        rules.append(match.group(1))
        targets.append(match.group(2))
    return compact_metadata({
        "starlark_loads": unique_limited(loaded),
        "starlark_loaded_symbols": unique_limited(symbols),
        "starlark_functions": unique_limited(match.group(1) for match in STARLARK_FUNCTION_RE.finditer(text)),
        "bazel_rules": unique_limited(rules),
        "bazel_targets": unique_limited(targets),
        "bazel_has_package_declaration": BAZEL_PACKAGE_RE.search(text) is not None,
    })


BAZEL_PROFILE = LanguageProfile(
    name="bazel",
    languages=frozenset({"bazel", "starlark"}),
    metadata_keys=BAZEL_METADATA_KEYS,
    file_metadata=bazel_file_metadata,
)
