"""Portable Rust metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

RUST_USE_RE = re.compile(r"(?m)^\s*(?:pub\s+)?use\s+([^;]+);")
RUST_MOD_RE = re.compile(r"(?m)^\s*(?:pub\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*[;{]")
RUST_FN_RE = re.compile(
    r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+|unsafe\s+|const\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)"
)
RUST_TYPE_RE = re.compile(r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)")
RUST_IMPL_RE = re.compile(r"(?m)^\s*(?:unsafe\s+)?impl(?:\s*<[^>]+>)?\s+([A-Za-z_][A-Za-z0-9_:<>]*)")

RUST_METADATA_KEYS = (
    "rust_modules",
    "rust_uses",
    "rust_functions",
    "rust_structs",
    "rust_enums",
    "rust_traits",
    "rust_impls",
    "rust_uses_unsafe",
)


def rust_file_metadata(_path: str, text: str) -> JsonObject:
    structs: list[str] = []
    enums: list[str] = []
    traits: list[str] = []
    for match in RUST_TYPE_RE.finditer(text):
        kind = match.group(1)
        name = match.group(2)
        if kind == "struct":
            structs.append(name)
        elif kind == "enum":
            enums.append(name)
        else:
            traits.append(name)
    return compact_metadata({
        "rust_modules": unique_limited(match.group(1) for match in RUST_MOD_RE.finditer(text)),
        "rust_uses": unique_limited(match.group(1).strip() for match in RUST_USE_RE.finditer(text)),
        "rust_functions": unique_limited(match.group(1) for match in RUST_FN_RE.finditer(text)),
        "rust_structs": unique_limited(structs),
        "rust_enums": unique_limited(enums),
        "rust_traits": unique_limited(traits),
        "rust_impls": unique_limited(match.group(1) for match in RUST_IMPL_RE.finditer(text)),
        "rust_uses_unsafe": "unsafe" in text,
    })


RUST_PROFILE = LanguageProfile(
    name="rust",
    languages=frozenset({"rust"}),
    metadata_keys=RUST_METADATA_KEYS,
    file_metadata=rust_file_metadata,
)
