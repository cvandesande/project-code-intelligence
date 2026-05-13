"""Portable Lua metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

LUA_REQUIRE_RE = re.compile(r"""require\s*(?:\(\s*)?["']([^"']+)["']""")
LUA_FUNCTION_RE = re.compile(r"(?m)^\s*(?:local\s+)?function\s+([A-Za-z_][A-Za-z0-9_.:]*)\s*\(")
LUA_TABLE_FUNCTION_RE = re.compile(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_.:]*)\s*=\s*function\s*\(")
LUA_LOCAL_RE = re.compile(r"(?m)^\s*local\s+([A-Za-z_][A-Za-z0-9_]*)\s*=")
LUA_MODULE_RE = re.compile(r"""(?m)^\s*module\s*(?:\(\s*)?["']([^"']+)["']""")

LUA_METADATA_KEYS = (
    "lua_requires",
    "lua_modules",
    "lua_functions",
    "lua_locals",
)


def lua_file_metadata(_path: str, text: str) -> JsonObject:
    functions = [
        *(match.group(1) for match in LUA_FUNCTION_RE.finditer(text)),
        *(match.group(1) for match in LUA_TABLE_FUNCTION_RE.finditer(text)),
    ]
    return compact_metadata({
        "lua_requires": unique_limited(match.group(1) for match in LUA_REQUIRE_RE.finditer(text)),
        "lua_modules": unique_limited(match.group(1) for match in LUA_MODULE_RE.finditer(text)),
        "lua_functions": unique_limited(functions),
        "lua_locals": unique_limited(match.group(1) for match in LUA_LOCAL_RE.finditer(text)),
    })


LUA_PROFILE = LanguageProfile(
    name="lua",
    languages=frozenset({"lua"}),
    metadata_keys=LUA_METADATA_KEYS,
    file_metadata=lua_file_metadata,
)
