"""Portable Perl metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

PERL_PACKAGE_RE = re.compile(r"(?m)^\s*package\s+([A-Za-z_][A-Za-z0-9_:]*)\s*;")
PERL_USE_RE = re.compile(r"(?m)^\s*use\s+([A-Za-z_][A-Za-z0-9_:]*)")
PERL_REQUIRE_RE = re.compile(r"""(?m)^\s*require\s+(?:(?:["']([^"']+)["'])|([A-Za-z_][A-Za-z0-9_:]*))""")
PERL_SUB_RE = re.compile(r"(?m)^\s*sub\s+([A-Za-z_][A-Za-z0-9_]*)")

PERL_METADATA_KEYS = (
    "perl_packages",
    "perl_modules",
    "perl_requires",
    "perl_subroutines",
    "perl_uses_strict",
    "perl_uses_warnings",
)


def perl_file_metadata(_path: str, text: str) -> JsonObject:
    requires = [match.group(1) or match.group(2) for match in PERL_REQUIRE_RE.finditer(text)]
    return compact_metadata({
        "perl_packages": unique_limited(match.group(1) for match in PERL_PACKAGE_RE.finditer(text)),
        "perl_modules": unique_limited(match.group(1) for match in PERL_USE_RE.finditer(text)),
        "perl_requires": unique_limited(requires),
        "perl_subroutines": unique_limited(match.group(1) for match in PERL_SUB_RE.finditer(text)),
        "perl_uses_strict": bool(re.search(r"(?m)^\s*use\s+strict\b", text)),
        "perl_uses_warnings": bool(re.search(r"(?m)^\s*use\s+warnings\b", text)),
    })


PERL_PROFILE = LanguageProfile(
    name="perl",
    languages=frozenset({"perl"}),
    metadata_keys=PERL_METADATA_KEYS,
    file_metadata=perl_file_metadata,
)
