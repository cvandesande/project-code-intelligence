"""Portable Swift metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

SWIFT_IMPORT_RE = re.compile(r"(?m)^\s*import\s+([A-Za-z_][A-Za-z0-9_]*)")
SWIFT_TYPE_RE = re.compile(
    r"(?m)^\s*(?:@[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|private|fileprivate|internal|open|final)\s+)*"
    r"(class|struct|enum|protocol|actor)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
SWIFT_EXTENSION_RE = re.compile(
    r"(?m)^\s*(?:public|private|fileprivate|internal|open\s+)*extension\s+([A-Za-z_][A-Za-z0-9_.]*)"
)
SWIFT_FUNCTION_RE = re.compile(
    r"(?m)^\s*(?:@[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|private|fileprivate|internal|open|static|class|mutating|nonmutating|async)\s+)*"
    r"func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
SWIFT_PROPERTY_RE = re.compile(
    r"(?m)^\s*(?:(?:public|private|fileprivate|internal|open|static|class|lazy|weak)\s+)*"
    r"(?:let|var)\s+([A-Za-z_][A-Za-z0-9_]*)"
)

SWIFT_METADATA_KEYS = (
    "swift_imports",
    "swift_classes",
    "swift_structs",
    "swift_enums",
    "swift_protocols",
    "swift_actors",
    "swift_extensions",
    "swift_functions",
    "swift_properties",
    "swift_has_async",
)


def swift_file_metadata(_path: str, text: str) -> JsonObject:
    classes: list[str] = []
    structs: list[str] = []
    enums: list[str] = []
    protocols: list[str] = []
    actors: list[str] = []
    for match in SWIFT_TYPE_RE.finditer(text):
        kind = match.group(1)
        name = match.group(2)
        if kind == "class":
            classes.append(name)
        elif kind == "struct":
            structs.append(name)
        elif kind == "enum":
            enums.append(name)
        elif kind == "protocol":
            protocols.append(name)
        else:
            actors.append(name)
    return compact_metadata({
        "swift_imports": unique_limited(match.group(1) for match in SWIFT_IMPORT_RE.finditer(text)),
        "swift_classes": unique_limited(classes),
        "swift_structs": unique_limited(structs),
        "swift_enums": unique_limited(enums),
        "swift_protocols": unique_limited(protocols),
        "swift_actors": unique_limited(actors),
        "swift_extensions": unique_limited(match.group(1) for match in SWIFT_EXTENSION_RE.finditer(text)),
        "swift_functions": unique_limited(match.group(1) for match in SWIFT_FUNCTION_RE.finditer(text)),
        "swift_properties": unique_limited(match.group(1) for match in SWIFT_PROPERTY_RE.finditer(text)),
        "swift_has_async": " async " in text or re.search(r"\basync\s+throws\b", text) is not None,
    })


SWIFT_PROFILE = LanguageProfile(
    name="swift",
    languages=frozenset({"swift"}),
    metadata_keys=SWIFT_METADATA_KEYS,
    file_metadata=swift_file_metadata,
)
