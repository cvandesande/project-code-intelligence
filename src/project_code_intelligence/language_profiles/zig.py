"""Portable Zig metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

ZIG_IMPORT_RE = re.compile(r"""@import\(\s*["']([^"']+)["']\s*\)""")
ZIG_FUNCTION_RE = re.compile(r"(?m)^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
ZIG_CONST_TYPE_RE = re.compile(r"(?m)^\s*(?:pub\s+)?const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(struct|enum|union)\b")
ZIG_TEST_RE = re.compile(r"""(?m)^\s*test\s+["']([^"']+)["']""")

ZIG_METADATA_KEYS = (
    "zig_imports",
    "zig_functions",
    "zig_structs",
    "zig_enums",
    "zig_unions",
    "zig_tests",
)


def zig_file_metadata(_path: str, text: str) -> JsonObject:
    structs: list[str] = []
    enums: list[str] = []
    unions: list[str] = []
    for match in ZIG_CONST_TYPE_RE.finditer(text):
        name = match.group(1)
        kind = match.group(2)
        if kind == "struct":
            structs.append(name)
        elif kind == "enum":
            enums.append(name)
        else:
            unions.append(name)
    return compact_metadata({
        "zig_imports": unique_limited(match.group(1) for match in ZIG_IMPORT_RE.finditer(text)),
        "zig_functions": unique_limited(match.group(1) for match in ZIG_FUNCTION_RE.finditer(text)),
        "zig_structs": unique_limited(structs),
        "zig_enums": unique_limited(enums),
        "zig_unions": unique_limited(unions),
        "zig_tests": unique_limited(match.group(1) for match in ZIG_TEST_RE.finditer(text)),
    })


ZIG_PROFILE = LanguageProfile(
    name="zig",
    languages=frozenset({"zig"}),
    metadata_keys=ZIG_METADATA_KEYS,
    file_metadata=zig_file_metadata,
)
