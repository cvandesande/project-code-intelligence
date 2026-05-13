"""Portable C and C++ family metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

C_INCLUDE_RE = re.compile(r'(?m)^\s*#\s*include\s+([<"])([^>"]+)[>"]')
C_DEFINE_RE = re.compile(r"(?m)^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)")
C_TYPE_RE = re.compile(r"(?m)^\s*(?:typedef\s+)?(?:struct|enum|union)\s+([A-Za-z_][A-Za-z0-9_]*)")
C_FUNCTION_RE = re.compile(r"(?m)^\s*(?:[A-Za-z_][A-Za-z0-9_]*[\w\s*]*\s+)+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{")
C_CONTROL_KEYWORDS = frozenset({"if", "for", "while", "switch", "return", "sizeof"})

C_FAMILY_METADATA_KEYS = (
    "c_family_local_includes",
    "c_family_system_includes",
    "c_family_defines",
    "c_family_declared_functions",
    "c_family_types",
    "c_family_has_inline_asm",
)


def c_family_file_metadata(_path: str, text: str) -> JsonObject:
    local_includes: list[str] = []
    system_includes: list[str] = []
    for match in C_INCLUDE_RE.finditer(text):
        include = match.group(2)
        if match.group(1) == "<":
            system_includes.append(include)
        else:
            local_includes.append(include)
    functions = [match.group(1) for match in C_FUNCTION_RE.finditer(text) if match.group(1) not in C_CONTROL_KEYWORDS]
    return compact_metadata({
        "c_family_local_includes": unique_limited(local_includes),
        "c_family_system_includes": unique_limited(system_includes),
        "c_family_defines": unique_limited(match.group(1) for match in C_DEFINE_RE.finditer(text)),
        "c_family_declared_functions": unique_limited(functions),
        "c_family_types": unique_limited(match.group(1) for match in C_TYPE_RE.finditer(text)),
        "c_family_has_inline_asm": "__asm__" in text or " asm(" in text or "\tasm(" in text,
    })


C_FAMILY_PROFILE = LanguageProfile(
    name="c-family",
    languages=frozenset({"c", "objective_c", "objective_cpp"}),
    metadata_keys=C_FAMILY_METADATA_KEYS,
    file_metadata=c_family_file_metadata,
)
