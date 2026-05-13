"""Portable Ruby metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

RUBY_REQUIRE_RE = re.compile(r"""(?m)^\s*require(?:_relative)?\s+["']([^"']+)["']""")
RUBY_MODULE_RE = re.compile(r"(?m)^\s*module\s+([A-Z][A-Za-z0-9_:]*)")
RUBY_CLASS_RE = re.compile(r"(?m)^\s*class\s+([A-Z][A-Za-z0-9_:]*)")
RUBY_METHOD_RE = re.compile(r"(?m)^\s*def\s+(?!self\.)([a-zA-Z_][A-Za-z0-9_!?=]*)")
RUBY_SINGLETON_METHOD_RE = re.compile(r"(?m)^\s*def\s+self\.([a-zA-Z_][A-Za-z0-9_!?=]*)")

RUBY_METADATA_KEYS = (
    "ruby_requires",
    "ruby_modules",
    "ruby_classes",
    "ruby_methods",
    "ruby_singleton_methods",
)


def ruby_file_metadata(_path: str, text: str) -> JsonObject:
    singleton_methods = [match.group(1) for match in RUBY_SINGLETON_METHOD_RE.finditer(text)]
    methods = [match.group(1) for match in RUBY_METHOD_RE.finditer(text)]
    return compact_metadata({
        "ruby_requires": unique_limited(match.group(1) for match in RUBY_REQUIRE_RE.finditer(text)),
        "ruby_modules": unique_limited(match.group(1) for match in RUBY_MODULE_RE.finditer(text)),
        "ruby_classes": unique_limited(match.group(1) for match in RUBY_CLASS_RE.finditer(text)),
        "ruby_methods": unique_limited(methods),
        "ruby_singleton_methods": unique_limited(singleton_methods),
    })


RUBY_PROFILE = LanguageProfile(
    name="ruby",
    languages=frozenset({"ruby"}),
    metadata_keys=RUBY_METADATA_KEYS,
    file_metadata=ruby_file_metadata,
)
