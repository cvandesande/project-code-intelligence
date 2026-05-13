"""Portable Scala and sbt metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

SCALA_PACKAGE_RE = re.compile(r"(?m)^\s*package\s+([A-Za-z_][A-Za-z0-9_.]*)")
SCALA_IMPORT_RE = re.compile(r"(?m)^\s*import\s+([A-Za-z_][A-Za-z0-9_.*{}=>, ]*)")
SCALA_TYPE_RE = re.compile(r"(?m)^\s*(?:case\s+)?(class|object|trait|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")
SCALA_DEF_RE = re.compile(r"(?m)^\s*(?:override\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\(:]")
SCALA_VAL_RE = re.compile(r"(?m)^\s*(?:lazy\s+)?(?:val|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[=:]")

SCALA_METADATA_KEYS = (
    "scala_package",
    "scala_imports",
    "scala_classes",
    "scala_objects",
    "scala_traits",
    "scala_enums",
    "scala_defs",
    "scala_values",
)


def scala_file_metadata(_path: str, text: str) -> JsonObject:
    classes: list[str] = []
    objects: list[str] = []
    traits: list[str] = []
    enums: list[str] = []
    for match in SCALA_TYPE_RE.finditer(text):
        kind = match.group(1)
        name = match.group(2)
        if kind == "class":
            classes.append(name)
        elif kind == "object":
            objects.append(name)
        elif kind == "trait":
            traits.append(name)
        else:
            enums.append(name)
    package_match = SCALA_PACKAGE_RE.search(text)
    return compact_metadata({
        "scala_package": package_match.group(1) if package_match else None,
        "scala_imports": unique_limited(match.group(1).strip() for match in SCALA_IMPORT_RE.finditer(text)),
        "scala_classes": unique_limited(classes),
        "scala_objects": unique_limited(objects),
        "scala_traits": unique_limited(traits),
        "scala_enums": unique_limited(enums),
        "scala_defs": unique_limited(match.group(1) for match in SCALA_DEF_RE.finditer(text)),
        "scala_values": unique_limited(match.group(1) for match in SCALA_VAL_RE.finditer(text)),
    })


SCALA_PROFILE = LanguageProfile(
    name="scala",
    languages=frozenset({"scala"}),
    metadata_keys=SCALA_METADATA_KEYS,
    file_metadata=scala_file_metadata,
)
