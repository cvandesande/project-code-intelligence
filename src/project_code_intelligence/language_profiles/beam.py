"""Metadata extraction for Elixir and Erlang files."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

ELIXIR_MODULE_RE = re.compile(r"(?m)^\s*defmodule\s+([A-Z][A-Za-z0-9_.]*)")
ELIXIR_IMPORT_RE = re.compile(r"(?m)^\s*(alias|import|require|use)\s+([A-Z][A-Za-z0-9_.]*)")
ELIXIR_FUNCTION_RE = re.compile(r"(?m)^\s*(def|defp|defmacro)\s+([a-z_][A-Za-z0-9_!?]*)")

ERLANG_MODULE_RE = re.compile(r"(?m)^\s*-module\(([a-zA-Z_][A-Za-z0-9_]*)\)\.")
ERLANG_EXPORT_RE = re.compile(r"(?ms)^\s*-export\(\[(.*?)\]\)\.")
ERLANG_RECORD_RE = re.compile(r"(?m)^\s*-record\(([a-zA-Z_][A-Za-z0-9_]*)\s*,")
ERLANG_BEHAVIOUR_RE = re.compile(r"(?m)^\s*-(?:behaviour|behavior)\(([a-zA-Z_][A-Za-z0-9_]*)\)\.")
ERLANG_FUNCTION_RE = re.compile(r"(?m)^([a-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*->")
ERLANG_EXPORT_ITEM_RE = re.compile(r"([a-z_][A-Za-z0-9_]*/\d+)")

BEAM_METADATA_KEYS = (
    "elixir_modules",
    "elixir_aliases",
    "elixir_imports",
    "elixir_requires",
    "elixir_uses",
    "elixir_functions",
    "elixir_private_functions",
    "elixir_macros",
    "erlang_module",
    "erlang_exports",
    "erlang_records",
    "erlang_behaviours",
    "erlang_functions",
)


def elixir_metadata(text: str) -> JsonObject:
    aliases: list[str] = []
    imports: list[str] = []
    requires: list[str] = []
    uses: list[str] = []
    functions: list[str] = []
    private_functions: list[str] = []
    macros: list[str] = []
    for match in ELIXIR_IMPORT_RE.finditer(text):
        kind = match.group(1)
        name = match.group(2)
        if kind == "alias":
            aliases.append(name)
        elif kind == "import":
            imports.append(name)
        elif kind == "require":
            requires.append(name)
        else:
            uses.append(name)
    for match in ELIXIR_FUNCTION_RE.finditer(text):
        kind = match.group(1)
        name = match.group(2)
        if kind == "defp":
            private_functions.append(name)
        elif kind == "defmacro":
            macros.append(name)
        else:
            functions.append(name)
    return compact_metadata({
        "elixir_modules": unique_limited(match.group(1) for match in ELIXIR_MODULE_RE.finditer(text)),
        "elixir_aliases": unique_limited(aliases),
        "elixir_imports": unique_limited(imports),
        "elixir_requires": unique_limited(requires),
        "elixir_uses": unique_limited(uses),
        "elixir_functions": unique_limited(functions),
        "elixir_private_functions": unique_limited(private_functions),
        "elixir_macros": unique_limited(macros),
    })


def erlang_metadata(text: str) -> JsonObject:
    module_match = ERLANG_MODULE_RE.search(text)
    exports: list[str] = []
    for match in ERLANG_EXPORT_RE.finditer(text):
        exports.extend(item.group(1) for item in ERLANG_EXPORT_ITEM_RE.finditer(match.group(1)))
    return compact_metadata({
        "erlang_module": module_match.group(1) if module_match else None,
        "erlang_exports": unique_limited(exports),
        "erlang_records": unique_limited(match.group(1) for match in ERLANG_RECORD_RE.finditer(text)),
        "erlang_behaviours": unique_limited(match.group(1) for match in ERLANG_BEHAVIOUR_RE.finditer(text)),
        "erlang_functions": unique_limited(match.group(1) for match in ERLANG_FUNCTION_RE.finditer(text)),
    })


def beam_file_metadata(path: str, text: str) -> JsonObject:
    if path.endswith((".ex", ".exs")):
        return elixir_metadata(text)
    return erlang_metadata(text)


BEAM_PROFILE = LanguageProfile(
    name="beam",
    languages=frozenset({"elixir", "erlang"}),
    metadata_keys=BEAM_METADATA_KEYS,
    file_metadata=beam_file_metadata,
)
