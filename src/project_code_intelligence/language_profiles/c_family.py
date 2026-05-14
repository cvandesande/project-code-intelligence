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
C_FUNCTION_LINE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]{0,240}\)\s*\{\s*$")
C_CONTROL_KEYWORDS = frozenset({"if", "for", "while", "switch", "return", "sizeof"})
MAX_C_FUNCTION_CANDIDATE_CHARS = 512

C_FAMILY_METADATA_KEYS = (
    "c_family_local_includes",
    "c_family_system_includes",
    "c_family_defines",
    "c_family_declared_functions",
    "c_family_types",
    "c_family_has_inline_asm",
)


def c_function_names(text: str) -> list[str]:
    names: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if (
            len(line) > MAX_C_FUNCTION_CANDIDATE_CHARS
            or not line.endswith("{")
            or "(" not in line
            or ")" not in line
            or line.startswith("#")
        ):
            continue
        match = C_FUNCTION_LINE_RE.search(line)
        if match is None:
            continue
        name = match.group(1)
        if name not in C_CONTROL_KEYWORDS:
            names.append(name)
    return unique_limited(names)


def c_family_file_metadata(_path: str, text: str) -> JsonObject:
    local_includes: list[str] = []
    system_includes: list[str] = []
    for match in C_INCLUDE_RE.finditer(text):
        include = match.group(2)
        if match.group(1) == "<":
            system_includes.append(include)
        else:
            local_includes.append(include)
    return compact_metadata({
        "c_family_local_includes": unique_limited(local_includes),
        "c_family_system_includes": unique_limited(system_includes),
        "c_family_defines": unique_limited(match.group(1) for match in C_DEFINE_RE.finditer(text)),
        "c_family_declared_functions": c_function_names(text),
        "c_family_types": unique_limited(match.group(1) for match in C_TYPE_RE.finditer(text)),
        "c_family_has_inline_asm": "__asm__" in text or " asm(" in text or "\tasm(" in text,
    })


C_FAMILY_PROFILE = LanguageProfile(
    name="c-family",
    languages=frozenset({"c", "objective_c", "objective_cpp"}),
    metadata_keys=C_FAMILY_METADATA_KEYS,
    file_metadata=c_family_file_metadata,
)
