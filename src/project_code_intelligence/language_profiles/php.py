"""Portable PHP metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

PHP_NAMESPACE_RE = re.compile(r"(?m)^\s*namespace\s+([^;{]+)")
PHP_USE_RE = re.compile(r"(?m)^\s*use\s+(?:function\s+|const\s+)?([^;]+);")
PHP_TYPE_RE = re.compile(
    r"(?m)^\s*(?:abstract\s+|final\s+|readonly\s+)*(class|interface|trait|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
PHP_FUNCTION_RE = re.compile(
    r"(?m)^\s*(?:(?:public|private|protected|static|final|abstract)\s+)*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
PHP_ATTRIBUTE_RE = re.compile(r"(?m)^\s*#\[\s*([A-Za-z_\\][A-Za-z0-9_\\]*)")

PHP_METADATA_KEYS = (
    "php_namespaces",
    "php_uses",
    "php_classes",
    "php_interfaces",
    "php_traits",
    "php_enums",
    "php_functions",
    "php_attributes",
)


def split_php_uses(value: str) -> list[str]:
    return [item.strip().split(" as ", 1)[0].strip() for item in value.split(",")]


def php_file_metadata(_path: str, text: str) -> JsonObject:
    classes: list[str] = []
    interfaces: list[str] = []
    traits: list[str] = []
    enums: list[str] = []
    for match in PHP_TYPE_RE.finditer(text):
        kind = match.group(1)
        name = match.group(2)
        if kind == "class":
            classes.append(name)
        elif kind == "interface":
            interfaces.append(name)
        elif kind == "trait":
            traits.append(name)
        else:
            enums.append(name)
    uses: list[str] = []
    for match in PHP_USE_RE.finditer(text):
        uses.extend(split_php_uses(match.group(1)))
    return compact_metadata({
        "php_namespaces": unique_limited(match.group(1).strip() for match in PHP_NAMESPACE_RE.finditer(text)),
        "php_uses": unique_limited(uses),
        "php_classes": unique_limited(classes),
        "php_interfaces": unique_limited(interfaces),
        "php_traits": unique_limited(traits),
        "php_enums": unique_limited(enums),
        "php_functions": unique_limited(match.group(1) for match in PHP_FUNCTION_RE.finditer(text)),
        "php_attributes": unique_limited(match.group(1) for match in PHP_ATTRIBUTE_RE.finditer(text)),
    })


PHP_PROFILE = LanguageProfile(
    name="php",
    languages=frozenset({"php"}),
    metadata_keys=PHP_METADATA_KEYS,
    file_metadata=php_file_metadata,
)
