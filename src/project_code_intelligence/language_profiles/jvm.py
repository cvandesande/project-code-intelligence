"""Portable Java and Kotlin metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

JAVA_PACKAGE_RE = re.compile(r"(?m)^\s*package\s+([A-Za-z_][A-Za-z0-9_.]*)\s*;")
JAVA_IMPORT_RE = re.compile(r"(?m)^\s*import\s+(?:static\s+)?([A-Za-z_][A-Za-z0-9_.*]*)\s*;")
JAVA_TYPE_RE = re.compile(
    r"(?m)^\s*(?:@[A-Za-z_][A-Za-z0-9_.]*(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|private|protected|abstract|final|static|sealed|non-sealed|strictfp)\s+)*"
    r"(class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
JAVA_METHOD_RE = re.compile(
    r"(?m)^\s*(?:@[A-Za-z_][A-Za-z0-9_.]*(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|private|protected|abstract|final|static|synchronized|native|strictfp)\s+)*"
    r"(?:[A-Za-z_][A-Za-z0-9_<>,.?[\]\s]*\s+)([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
JAVA_ANNOTATION_RE = re.compile(r"(?m)^\s*@([A-Za-z_][A-Za-z0-9_.]*)")

KOTLIN_PACKAGE_RE = re.compile(r"(?m)^\s*package\s+([A-Za-z_][A-Za-z0-9_.]*)")
KOTLIN_IMPORT_RE = re.compile(r"(?m)^\s*import\s+([A-Za-z_][A-Za-z0-9_.*]*)")
KOTLIN_TYPE_RE = re.compile(
    r"(?m)^\s*(?:@[A-Za-z_][A-Za-z0-9_.]*(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|private|protected|internal|abstract|final|open|sealed|data|value|enum|annotation)\s+)*"
    r"(class|interface|object)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
KOTLIN_FUNCTION_RE = re.compile(
    r"(?m)^\s*(?:public|private|protected|internal|suspend|inline|operator|\s)*fun\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)

JVM_METADATA_KEYS = (
    "jvm_package",
    "jvm_imports",
    "java_classes",
    "java_interfaces",
    "java_enums",
    "java_records",
    "java_methods",
    "java_annotations",
    "kotlin_classes",
    "kotlin_interfaces",
    "kotlin_objects",
    "kotlin_functions",
    "kotlin_has_suspend",
)


def java_metadata(text: str) -> JsonObject:
    classes: list[str] = []
    interfaces: list[str] = []
    enums: list[str] = []
    records: list[str] = []
    for match in JAVA_TYPE_RE.finditer(text):
        kind = match.group(1)
        name = match.group(2)
        if kind == "class":
            classes.append(name)
        elif kind == "interface":
            interfaces.append(name)
        elif kind == "enum":
            enums.append(name)
        else:
            records.append(name)
    package_match = JAVA_PACKAGE_RE.search(text)
    type_names = frozenset([*classes, *interfaces, *enums, *records])
    return compact_metadata({
        "jvm_package": package_match.group(1) if package_match else None,
        "jvm_imports": unique_limited(match.group(1) for match in JAVA_IMPORT_RE.finditer(text)),
        "java_classes": unique_limited(classes),
        "java_interfaces": unique_limited(interfaces),
        "java_enums": unique_limited(enums),
        "java_records": unique_limited(records),
        "java_methods": unique_limited(
            match.group(1) for match in JAVA_METHOD_RE.finditer(text) if match.group(1) not in type_names
        ),
        "java_annotations": unique_limited(match.group(1) for match in JAVA_ANNOTATION_RE.finditer(text)),
    })


def kotlin_metadata(text: str) -> JsonObject:
    classes: list[str] = []
    interfaces: list[str] = []
    objects: list[str] = []
    for match in KOTLIN_TYPE_RE.finditer(text):
        kind = match.group(1)
        name = match.group(2)
        if kind == "class":
            classes.append(name)
        elif kind == "interface":
            interfaces.append(name)
        else:
            objects.append(name)
    package_match = KOTLIN_PACKAGE_RE.search(text)
    return compact_metadata({
        "jvm_package": package_match.group(1) if package_match else None,
        "jvm_imports": unique_limited(match.group(1) for match in KOTLIN_IMPORT_RE.finditer(text)),
        "kotlin_classes": unique_limited(classes),
        "kotlin_interfaces": unique_limited(interfaces),
        "kotlin_objects": unique_limited(objects),
        "kotlin_functions": unique_limited(match.group(1) for match in KOTLIN_FUNCTION_RE.finditer(text)),
        "kotlin_has_suspend": "suspend fun" in text,
    })


def jvm_file_metadata(_path: str, text: str) -> JsonObject:
    if "fun " in text or "val " in text or "object " in text:
        return kotlin_metadata(text)
    return java_metadata(text)


JVM_PROFILE = LanguageProfile(
    name="jvm",
    languages=frozenset({"java", "kotlin"}),
    metadata_keys=JVM_METADATA_KEYS,
    file_metadata=jvm_file_metadata,
)
