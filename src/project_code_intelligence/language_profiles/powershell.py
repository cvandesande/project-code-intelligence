"""Portable PowerShell metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

POWERSHELL_FUNCTION_RE = re.compile(r"(?im)^\s*function\s+([A-Za-z_][A-Za-z0-9_-]*)")
POWERSHELL_PARAM_RE = re.compile(r"(?m)\[Parameter(?:\([^)]*\))?\]\s*(?:\[[^\]]+\]\s*)?\$([A-Za-z_][A-Za-z0-9_]*)")
POWERSHELL_IMPORT_RE = re.compile(r"(?im)^\s*Import-Module\s+['\"]?([^'\"\s]+)")
POWERSHELL_CMDLET_RE = re.compile(r"\b([A-Z][A-Za-z]+-[A-Z][A-Za-z0-9]+)\b")

POWERSHELL_METADATA_KEYS = (
    "powershell_functions",
    "powershell_parameters",
    "powershell_imported_modules",
    "powershell_cmdlets",
)


def powershell_file_metadata(_path: str, text: str) -> JsonObject:
    functions = [match.group(1) for match in POWERSHELL_FUNCTION_RE.finditer(text)]
    return compact_metadata({
        "powershell_functions": unique_limited(functions),
        "powershell_parameters": unique_limited(match.group(1) for match in POWERSHELL_PARAM_RE.finditer(text)),
        "powershell_imported_modules": unique_limited(match.group(1) for match in POWERSHELL_IMPORT_RE.finditer(text)),
        "powershell_cmdlets": unique_limited(
            match.group(1) for match in POWERSHELL_CMDLET_RE.finditer(text) if match.group(1) not in functions
        ),
    })


POWERSHELL_PROFILE = LanguageProfile(
    name="powershell",
    languages=frozenset({"powershell"}),
    metadata_keys=POWERSHELL_METADATA_KEYS,
    file_metadata=powershell_file_metadata,
)
