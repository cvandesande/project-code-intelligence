"""Portable C# metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

CS_USING_RE = re.compile(r"(?m)^\s*using\s+(?:static\s+|[A-Za-z_][A-Za-z0-9_]*\s*=\s*)?([A-Za-z_][A-Za-z0-9_.]*)\s*;")
CS_NAMESPACE_RE = re.compile(r"(?m)^\s*namespace\s+([A-Za-z_][A-Za-z0-9_.]*)")
CS_TYPE_RE = re.compile(
    r"(?m)^\s*(?:\[[^\]]+\]\s*)*(?:(?:public|private|protected|internal|sealed|abstract|static|partial)\s+)*"
    r"(class|record|interface|enum|struct)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
CS_METHOD_RE = re.compile(
    r"(?m)^\s*(?:\[[^\]]+\]\s*)*(?:(?:public|private|protected|internal|static|async|virtual|override|sealed|partial)\s+)*"
    r"(?:[A-Za-z_][A-Za-z0-9_<>,.?[\]\s]*\s+)([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
CS_ATTRIBUTE_RE = re.compile(r"(?m)^\s*\[\s*([A-Za-z_][A-Za-z0-9_.]*)")
CS_CONTROL_NAMES = frozenset({"if", "for", "foreach", "switch", "while", "catch", "using", "lock"})

C_SHARP_METADATA_KEYS = (
    "csharp_usings",
    "csharp_namespaces",
    "csharp_classes",
    "csharp_records",
    "csharp_interfaces",
    "csharp_enums",
    "csharp_methods",
    "csharp_attributes",
    "csharp_has_async",
)


def csharp_file_metadata(_path: str, text: str) -> JsonObject:
    classes: list[str] = []
    records: list[str] = []
    interfaces: list[str] = []
    enums: list[str] = []
    for match in CS_TYPE_RE.finditer(text):
        kind = match.group(1)
        name = match.group(2)
        if kind in {"class", "struct"}:
            classes.append(name)
        elif kind == "record":
            records.append(name)
        elif kind == "interface":
            interfaces.append(name)
        else:
            enums.append(name)
    type_names = frozenset([*classes, *records, *interfaces, *enums])
    methods = [
        match.group(1)
        for match in CS_METHOD_RE.finditer(text)
        if match.group(1) not in CS_CONTROL_NAMES and match.group(1) not in type_names
    ]
    return compact_metadata({
        "csharp_usings": unique_limited(match.group(1) for match in CS_USING_RE.finditer(text)),
        "csharp_namespaces": unique_limited(match.group(1) for match in CS_NAMESPACE_RE.finditer(text)),
        "csharp_classes": unique_limited(classes),
        "csharp_records": unique_limited(records),
        "csharp_interfaces": unique_limited(interfaces),
        "csharp_enums": unique_limited(enums),
        "csharp_methods": unique_limited(methods),
        "csharp_attributes": unique_limited(match.group(1) for match in CS_ATTRIBUTE_RE.finditer(text)),
        "csharp_has_async": "async " in text or "await " in text,
    })


C_SHARP_PROFILE = LanguageProfile(
    name="csharp",
    languages=frozenset({"csharp"}),
    metadata_keys=C_SHARP_METADATA_KEYS,
    file_metadata=csharp_file_metadata,
)
