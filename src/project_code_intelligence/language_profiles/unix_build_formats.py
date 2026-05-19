"""Metadata for common Unix build/config DSLs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

AUTOTOOLS_MACRO_RE = re.compile(r"\b(A[CM]_[A-Z0-9_]+|LT_[A-Z0-9_]+)\b")
M4_DEFINE_RE = re.compile(r"\b(?:m4_define|m4_defun|AC_DEFUN)\s*\(\s*\[?([A-Za-z_][A-Za-z0-9_]*)\]?")
LINKER_SECTION_RE = re.compile(r"(?m)^\s*\.([A-Za-z_][A-Za-z0-9_.]*)\s*:")
LINKER_PROVIDE_RE = re.compile(r"\bPROVIDE(?:_HIDDEN)?\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)")
LINKER_ENTRY_RE = re.compile(r"\bENTRY\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)")
BOOT_ASSIGN_RE = re.compile(r"(?m)^\s*(?:setenv\s+)?([A-Za-z_][A-Za-z0-9_]*)=")
BOOT_COMMAND_RE = re.compile(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\b")
AWK_FUNCTION_RE = re.compile(r"(?m)^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
LEX_STATE_RE = re.compile(r"(?m)^%[sx]\s+(.+)$")
YACC_TOKEN_RE = re.compile(r"(?m)^%token\s+(.+)$")
YACC_RULE_RE = re.compile(r"(?m)^([A-Za-z_][A-Za-z0-9_]*)\s*:")
BOOT_SKIP_COMMANDS = frozenset({"else", "fi", "if", "then"})

UNIX_BUILD_FORMAT_METADATA_KEYS = (
    "autotools_macros",
    "autotools_definitions",
    "linker_sections",
    "linker_provided_symbols",
    "linker_entry_symbols",
    "boot_script_commands",
    "boot_script_variables",
    "awk_functions",
    "lex_start_conditions",
    "yacc_tokens",
    "yacc_rules",
)


def autotools_metadata(text: str) -> JsonObject:
    return compact_metadata({
        "autotools_macros": unique_limited(match.group(1) for match in AUTOTOOLS_MACRO_RE.finditer(text)),
        "autotools_definitions": unique_limited(match.group(1) for match in M4_DEFINE_RE.finditer(text)),
    })


def linker_metadata(text: str) -> JsonObject:
    return compact_metadata({
        "linker_sections": unique_limited(match.group(1) for match in LINKER_SECTION_RE.finditer(text)),
        "linker_provided_symbols": unique_limited(match.group(1) for match in LINKER_PROVIDE_RE.finditer(text)),
        "linker_entry_symbols": unique_limited(match.group(1) for match in LINKER_ENTRY_RE.finditer(text)),
    })


def boot_script_metadata(text: str) -> JsonObject:
    commands = [match.group(1) for match in BOOT_COMMAND_RE.finditer(text) if match.group(1) not in BOOT_SKIP_COMMANDS]
    return compact_metadata({
        "boot_script_commands": unique_limited(commands),
        "boot_script_variables": unique_limited(match.group(1) for match in BOOT_ASSIGN_RE.finditer(text)),
    })


def unix_dsl_metadata(path: str, text: str) -> JsonObject:
    suffix = Path(path).suffix.lower()
    if suffix == ".awk":
        return compact_metadata({
            "awk_functions": unique_limited(match.group(1) for match in AWK_FUNCTION_RE.finditer(text))
        })
    if suffix == ".l":
        states: list[str] = []
        for match in LEX_STATE_RE.finditer(text):
            states.extend(match.group(1).split())
        return compact_metadata({"lex_start_conditions": unique_limited(states)})
    if suffix == ".y":
        tokens: list[str] = []
        for match in YACC_TOKEN_RE.finditer(text):
            tokens.extend(match.group(1).split())
        return compact_metadata({
            "yacc_tokens": unique_limited(tokens),
            "yacc_rules": unique_limited(match.group(1) for match in YACC_RULE_RE.finditer(text)),
        })
    return {}


def unix_build_format_metadata(path: str, text: str) -> JsonObject:
    suffix = Path(path).suffix.lower()
    if suffix in {".m4", ".am", ".ac"}:
        return autotools_metadata(text)
    if suffix in {".ld", ".lds"}:
        return linker_metadata(text)
    if suffix in {".bootscript", ".scr"}:
        return boot_script_metadata(text)
    return unix_dsl_metadata(path, text)


UNIX_BUILD_FORMAT_PROFILE = LanguageProfile(
    name="unix-build-formats",
    languages=frozenset({"autotools", "awk", "boot_script", "lex", "linker_script", "yacc"}),
    metadata_keys=UNIX_BUILD_FORMAT_METADATA_KEYS,
    file_metadata=unix_build_format_metadata,
)
