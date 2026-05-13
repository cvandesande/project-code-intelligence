"""Portable Groovy and Gradle metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

GROOVY_IMPORT_RE = re.compile(r"(?m)^\s*import\s+(?:static\s+)?([A-Za-z_][A-Za-z0-9_.*]*)")
GROOVY_TYPE_RE = re.compile(
    r"(?m)^\s*(?:@[A-Za-z_][A-Za-z0-9_.]*(?:\([^)]*\))?\s*)*(class|interface|trait|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
GROOVY_METHOD_RE = re.compile(r"(?m)^\s*(?:def|void|[A-Za-z_][A-Za-z0-9_<>,.?[\]\s]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
GROOVY_ANNOTATION_RE = re.compile(r"(?m)^\s*@([A-Za-z_][A-Za-z0-9_.]*)")
GRADLE_PLUGIN_RE = re.compile(r"""(?m)^\s*id\s+["']([^"']+)["']|^\s*id\(["']([^"']+)["']\)""")
GRADLE_DEP_RE = re.compile(
    r"""(?m)^\s*(?:api|implementation|compileOnly|runtimeOnly|testImplementation)\s+["']([^"']+)["']"""
)
GRADLE_TASK_RE = re.compile(r"""(?m)^\s*(?:task\s+([A-Za-z_][A-Za-z0-9_]*)|tasks\.register\(["']([^"']+)["']\))""")

GROOVY_METADATA_KEYS = (
    "groovy_imports",
    "groovy_classes",
    "groovy_interfaces",
    "groovy_traits",
    "groovy_enums",
    "groovy_methods",
    "groovy_annotations",
    "gradle_plugins",
    "gradle_dependencies",
    "gradle_tasks",
)


def first_present(*values: str | None) -> str | None:
    return next((value for value in values if value), None)


def groovy_file_metadata(_path: str, text: str) -> JsonObject:
    classes: list[str] = []
    interfaces: list[str] = []
    traits: list[str] = []
    enums: list[str] = []
    for match in GROOVY_TYPE_RE.finditer(text):
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
    return compact_metadata({
        "groovy_imports": unique_limited(match.group(1) for match in GROOVY_IMPORT_RE.finditer(text)),
        "groovy_classes": unique_limited(classes),
        "groovy_interfaces": unique_limited(interfaces),
        "groovy_traits": unique_limited(traits),
        "groovy_enums": unique_limited(enums),
        "groovy_methods": unique_limited(match.group(1) for match in GROOVY_METHOD_RE.finditer(text)),
        "groovy_annotations": unique_limited(match.group(1) for match in GROOVY_ANNOTATION_RE.finditer(text)),
        "gradle_plugins": unique_limited(
            first_present(match.group(1), match.group(2)) or "" for match in GRADLE_PLUGIN_RE.finditer(text)
        ),
        "gradle_dependencies": unique_limited(match.group(1) for match in GRADLE_DEP_RE.finditer(text)),
        "gradle_tasks": unique_limited(
            first_present(match.group(1), match.group(2)) or "" for match in GRADLE_TASK_RE.finditer(text)
        ),
    })


GROOVY_PROFILE = LanguageProfile(
    name="groovy",
    languages=frozenset({"groovy"}),
    metadata_keys=GROOVY_METADATA_KEYS,
    file_metadata=groovy_file_metadata,
)
